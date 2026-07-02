"""No-torch faithful subset of SGLang v0.5.13 HiCacheStorage ABC + config.

Used ONLY to test the DingoFS plugin on a machine without torch/GPU. Mirrors the
real abstract surface (get/batch_get/set/batch_set/exists abstract; v1 batch +
batch_exists + register_mem_pool_host non-abstract) so a plugin that runs here
runs unchanged against the real SGLang.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

STORAGE_BATCH_SIZE = 128


@dataclass
class HiCacheStorageConfig:
    tp_rank: int
    tp_size: int
    pp_rank: int = 0
    pp_size: int = 1
    attn_cp_rank: int = 0
    attn_cp_size: int = 1
    is_mla_model: bool = False
    enable_storage_metrics: bool = False
    is_page_first_layout: bool = False
    model_name: Optional[str] = None
    tp_lcm_size: Optional[int] = None
    should_split_heads: bool = False
    extra_config: Optional[dict] = None


@dataclass
class HiCacheStorageExtraInfo:
    prefix_keys: Optional[List[str]] = None
    extra_info: Optional[dict] = None


class PoolHitPolicy:
    ALL_PAGES = "all_pages"
    TRAILING_PAGES = "trailing_pages"


@dataclass
class PoolTransfer:
    name: str
    host_indices: Optional[List[int]] = None
    device_indices: Optional[List[int]] = None
    keys: Optional[List[str]] = None
    hit_policy: str = PoolHitPolicy.ALL_PAGES


@dataclass
class PoolTransferResult:
    kv_hit_pages: int
    extra_pool_hit_pages: dict


class HiCacheStorage(ABC):
    def register_mem_pool_host(self, mem_pool_host):
        self.mem_pool_host = mem_pool_host

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        if not hasattr(self, "registered_pools"):
            self.registered_pools = {}
        self.registered_pools[host_pool_name] = host_pool

    # v1 zero-copy batch interface (host_indices based) — default NotImplemented.
    def batch_get_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        raise NotImplementedError

    def batch_set_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        raise NotImplementedError

    # v2 pool-aware interface — default NotImplemented.
    def batch_get_v2(self, transfers, extra_info=None) -> dict:
        raise NotImplementedError

    def batch_set_v2(self, transfers, extra_info=None) -> dict:
        raise NotImplementedError

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        raise NotImplementedError

    @abstractmethod
    def get(self, key, target_location=None, target_sizes=None):
        ...

    @abstractmethod
    def batch_get(self, keys, target_locations=None, target_sizes=None):
        ...

    @abstractmethod
    def set(self, key, value=None, target_location=None, target_sizes=None) -> bool:
        ...

    @abstractmethod
    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None) -> bool:
        ...

    @abstractmethod
    def exists(self, key) -> bool:
        ...

    def batch_exists(self, keys, extra_info=None) -> int:
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None
