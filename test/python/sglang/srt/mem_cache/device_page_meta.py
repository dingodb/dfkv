from __future__ import annotations

"""Device-side page buffer meta for the HiCache L2-bypass (device-direct write)
prototype (SGLANG_HICACHE_L2_BYPASS=1).

Mirrors memory_pool_host.MHATokenToKVPoolHost.get_page_buffer_meta (the host
zero-copy arithmetic) but over the GPU KV pool. The GPU pool is *layer-first*:
each layer is a separate allocation (MHATokenToKVPool.k_buffer/v_buffer are
lists of per-layer tensors; MLATokenToKVPool.kv_buffer likewise). A page's KV is
therefore NOT contiguous across layers, so — unlike the host page-first pool,
whose page is one contiguous blob — a device page must be expressed as a
scatter-gather list of per-layer segments. Each sub-object (k, and v for MHA) of
a page becomes ``layer_num`` device segments; the dfkv backend hands them to the
scatter-gather C ABI (dfkv_batch_put_sg), which stores the payload as the
concatenation of the segments.

CORRECTNESS NOTE (layer-major vs page-first ordering): the segments here
concatenate in LAYER-major order (layer0[tok0..tokP], layer1[tok0..tokP], ...).
The stock host read path (batch_get_v1) scatters the stored blob into a
page-first host buffer, whose byte order is TOKEN-major (tok0[layer0..layerN],
...). The two orderings are transposes of each other. A page written
device-direct is therefore byte-coherent only with a matching device-direct
(layer-major SG) reader — increment 2 — not with the unchanged page-first host
read. Increment 1 wires the write path and its offload; enabling it in isolation
is a benchmark/prototype mode. See PATCH-MANIFEST.md.
"""

from typing import List, Tuple


def _mla_layer_bases(pool) -> Tuple[List[int], int]:
    """(per-layer base data_ptr, per-token byte stride) for an MLA GPU pool."""
    layers = pool.kv_buffer  # list[layer] of (size+page_size, 1, kv_cache_dim)
    itemsize = layers[0].element_size()
    token_stride = pool.kv_cache_dim * itemsize
    return [int(t.data_ptr()) for t in layers], token_stride


def _mha_layer_bases(pool) -> Tuple[List[int], int, List[int], int]:
    """(k bases, k token stride, v bases, v token stride) for an MHA GPU pool."""
    k_layers = pool.k_buffer  # list[layer] of (size+page_size, head_num, head_dim)
    v_layers = pool.v_buffer
    k_itemsize = k_layers[0].element_size()
    v_itemsize = v_layers[0].element_size()
    k_stride = pool.head_num * pool.head_dim * k_itemsize
    v_stride = pool.head_num * pool.v_head_dim * v_itemsize
    return (
        [int(t.data_ptr()) for t in k_layers],
        k_stride,
        [int(t.data_ptr()) for t in v_layers],
        v_stride,
    )


def _is_mla(pool) -> bool:
    return hasattr(pool, "kv_buffer") and hasattr(pool, "kv_cache_dim")


def _is_mha(pool) -> bool:
    return hasattr(pool, "k_buffer") and hasattr(pool, "v_buffer")


def supported(pool) -> bool:
    """True if get_device_page_buffer_meta can express this GPU pool.

    DSATokenToKVPool subclasses MLATokenToKVPool but its primary "kv" pool is a
    logical anchor with no device KV (real KV rides the v2 side pools, which stay
    on the host path in increment 1); use_dsa pools are declined here so the
    controller keeps them on the stock path.
    """
    if getattr(pool, "use_dsa", False):
        return False
    return _is_mla(pool) or _is_mha(pool)


def get_device_page_buffer_meta(pool, indices) -> Tuple[List[List[int]], List[List[int]]]:
    """Scatter-gather device page meta, parallel in shape to the host
    get_page_buffer_meta (one entry per page sub-object, k then v for MHA; k only
    for MLA), but each entry is a LIST of per-layer (ptr)/(size) segments.

    Returns (seg_ptrs, seg_sizes), each a list of length ``n_pages * sub`` whose
    element ``[p * sub + j]`` is the per-layer segment list of page ``p``'s
    sub-object ``j``.
    """
    page_size = pool.page_size
    idx = indices.tolist() if hasattr(indices, "tolist") else list(indices)
    assert len(idx) % page_size == 0, (
        f"device page meta needs page-aligned indices, got {len(idx)} "
        f"(page_size={page_size})"
    )
    n_pages = len(idx) // page_size

    seg_ptrs: List[List[int]] = []
    seg_sizes: List[List[int]] = []

    if _is_mla(pool):
        bases, token_stride = _mla_layer_bases(pool)
        seg_len = page_size * token_stride
        for p in range(n_pages):
            slot = idx[p * page_size]
            off = slot * token_stride
            seg_ptrs.append([b + off for b in bases])
            seg_sizes.append([seg_len] * len(bases))
    elif _is_mha(pool):
        k_bases, k_stride, v_bases, v_stride = _mha_layer_bases(pool)
        k_len = page_size * k_stride
        v_len = page_size * v_stride
        for p in range(n_pages):
            slot = idx[p * page_size]
            seg_ptrs.append([b + slot * k_stride for b in k_bases])
            seg_sizes.append([k_len] * len(k_bases))
            seg_ptrs.append([b + slot * v_stride for b in v_bases])
            seg_sizes.append([v_len] * len(v_bases))
    else:
        raise ValueError(
            f"get_device_page_buffer_meta: unsupported GPU pool {type(pool).__name__}"
        )

    return seg_ptrs, seg_sizes


def device_pool_regions(pool) -> List[Tuple[int, int]]:
    """(base, nbytes) of every per-layer device buffer, for RDMA registration."""
    regions: List[Tuple[int, int]] = []
    if _is_mla(pool):
        tensors = list(pool.kv_buffer)
    elif _is_mha(pool):
        tensors = list(pool.k_buffer) + list(pool.v_buffer)
    else:
        return regions
    for t in tensors:
        regions.append((int(t.data_ptr()), int(t.numel()) * int(t.element_size())))
    return regions


# --- DSA indexer sidecar (device-direct), task 4 ------------------------------
# Torch-free copy of the sidecar meta in the patched SGLang device_page_meta.py,
# so the dfkv sidecar-device roundtrip test can exercise _sidecar_device_set/get
# without a GPU. Kept in sync with the patched module (layer-first, page-indexed
# index_k_with_scale_buffer -> layer_num segments/page, same @sg chunking as the
# main latent).


def _is_indexer(pool) -> bool:
    layers = getattr(pool, "index_k_with_scale_buffer", None)
    return (
        isinstance(layers, (list, tuple))
        and len(layers) > 0
        and hasattr(pool, "page_size")
    )


def sidecar_supported(pool) -> bool:
    return _is_indexer(pool)


def sidecar_device_pool_regions(pool) -> List[Tuple[int, int]]:
    regions: List[Tuple[int, int]] = []
    for t in getattr(pool, "index_k_with_scale_buffer", None) or []:
        regions.append((int(t.data_ptr()), int(t.numel()) * int(t.element_size())))
    return regions


def get_device_sidecar_page_buffer_meta(
    pool, indices
) -> Tuple[List[List[int]], List[List[int]]]:
    page_size = pool.page_size
    idx = indices.tolist() if hasattr(indices, "tolist") else list(indices)
    assert len(idx) % page_size == 0, (
        f"sidecar device page meta needs page-aligned indices, got {len(idx)} "
        f"(page_size={page_size})"
    )
    n_pages = len(idx) // page_size

    layers = pool.index_k_with_scale_buffer
    itemsize = layers[0].element_size()
    row_bytes = int(layers[0].shape[1]) * itemsize
    row_stride = int(layers[0].stride(0)) * itemsize
    bases = [int(t.data_ptr()) for t in layers]

    seg_ptrs: List[List[int]] = []
    seg_sizes: List[List[int]] = []
    for p in range(n_pages):
        slot = idx[p * page_size]
        off = (slot // page_size) * row_stride
        seg_ptrs.append([b + off for b in bases])
        seg_sizes.append([row_bytes] * len(bases))
    return seg_ptrs, seg_sizes
