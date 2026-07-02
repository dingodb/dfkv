# dfkv LMCache connector — 设计

把 KV cache 块存到 **dfkv** 集群的 LMCache `RemoteConnector`。它通过 dfkv 的 C ABI
（`libdfkv.so`，Python ctypes 加载）对接 dfkv —— 与本仓库的 SGLang HiCache 插件
（[../hicache/DINGOFS_MODULE_README.md](../hicache/DINGOFS_MODULE_README.md)）走同一套底层 KV 客户端。

本 connector 移植自 dingofs 项目的 LMCache connector，但做了两处实质性改动，详见下文。

---

## 1. 整体架构

```
 vLLM (LMCacheConnectorV1, kv_both)
   └─ LMCache Engine
        └─ RemoteBackend ──创建──> DfkvConnectorAdapter (plugin://dfkv)
                                      └─ DfkvConnector(RemoteConnector)
                                           ├─ ExistsLRU            (本地存在性短路)
                                           └─ DfkvNativeClient     (ctypes + 线程池)
                                                └─ libdfkv.so (C ABI)
                                                     └─ KVClient ──一致性哈希──> dfkv cache nodes
                                                                                   (jg29 / jg31)
```

- LMCache 把每个 prompt 切成 `chunk_size`（token 数，默认 16/256）的 **chunk**，对每个 chunk
  做内容哈希得到 `CacheEngineKey`，再调用 `RemoteConnector` 的 get/put/exists。
- `DfkvConnector` 把 `CacheEngineKey` 序列化成 dfkv key 字符串，把 chunk 的字节
  （`MemoryObj.byte_array`，是 LMCache 固定 host arena 的一个切片）通过 `DfkvNativeClient`
  零拷贝交给 dfkv。
- dfkv `KVClient` 按 key 的一致性哈希路由到某个 cache node，走 RDMA（或 TCP）读写。

---

## 2. 与 HiCache connector 的异同

| 维度 | HiCache（SGLang） | LMCache（本 connector） |
|---|---|---|
| 上层框架 | SGLang HiCache（L3 external KV）| LMCache `RemoteConnector` 插件 |
| 接口 | `HiCacheStorage`（batch_set_v1/get_v1 …）| `RemoteConnector`（get/put/batched_* …）|
| 底层 | `libdfkv.so`（ctypes）| `libdfkv.so`（ctypes）—— **相同** |
| key | `{model}/{page_hash}_{tp}_{rank}_k/_v`（按页 K/V 拆分）| `{model}@{ws}@{wid}@{chunk_hash}`（按 chunk 单对象）|
| 块大小 | 固定页（page_size token）| **任意**（`full_chunk_size_bytes`，可变长）|
| 零拷贝 buffer | SGLang host KV pool | LMCache 的 `MixedMemoryAllocator` arena |

两者复用同一套 dfkv KV 客户端、value header、传输层，互不影响（key 命名空间不同）。

---

## 3. 两处实质性改动

### 3.1 支持任意块大小（含变长）

dingofs connector 把每个块**硬编码限制为 4 MiB**（dingofs cache node 用固定大小的
io_uring buffer）。dfkv 没有这个限制，能存任意大小的 value，所以：

- 删除了 4 MiB cap 相关逻辑（`max_chunk_mib` / `_check_chunk_size_within_cap`）。
- connector 直接使用 LMCache 计算出的 `full_chunk_size_bytes`（可以是几十 MiB）。

**关键难点：dfkv 的 GET 是精确大小匹配的。** 每个 value 在 dfkv 里存为
`[48B ValueHeader | payload]`，`KVClient::Get` 只有当 `payload_len == 调用方请求的 n`
且 geometry 头匹配时才算命中（见 [../../src/client/kv_client.cc](../../src/client/kv_client.cc) 的 `Get`/`BatchGet`）。
而 LMCache 会存**变长的不满（unfull）末块**：一个请求最后一个 chunk 的 token 数 < chunk_size，
字节数也更小。若用满块大小 `full_chunk_size_bytes` 去 GET 这种不满块，`payload_len != n`
会被判为 miss，导致不满块永远读不回。

**解法：给 dfkv C ABI 增加变长 get。** 我们包装了 dfkv 已有的 `KVClient::GetAuto`
（从存储头里读出真实 `payload_len`），新增：

- C++：`KVClient::GetAuto(key, out, cap, *out_len)` 与 `KVClient::BatchGetAuto(items, *out_lens)`
  （[../../src/client/kv_client.h](../../src/client/kv_client.h) / [.cc](../../src/client/kv_client.cc)）。
- C ABI：`dfkv_get_auto` / `dfkv_batch_get_auto`
  （[../../src/client/dfkv_c_api.h](../../src/client/dfkv_c_api.h) / [.cc](../../src/client/dfkv_c_api.cc)）。

`BatchGetAuto` 复用 `BatchGet` 的 **RDMA 零拷贝 `RangeInto` 路径**：把每个 buffer 容量当作
请求长度，server 返回真实 `[header|payload]`（payload 可能更短），`RangeInto` 把 payload 零拷贝
散射进调用方 buffer，并回传 value header；我们从头里读出真实 `payload_len`、判 `<= cap` 即命中。
这样**满块仍是零拷贝热路径**（与 HiCache 一致），不满块也能按真实长度读回。

connector 拿到真实长度后调用 `RemoteConnector.reshape_partial_chunk(memory_obj, bytes_read)`：
满块直接返回，不满块按 `bytes_read // single_token_size` 裁剪 shape。

### 3.2 后端切换：pybind11 → ctypes

dingofs connector 用 pybind11 原生模块 `_dingofs_native`（封装 dingofs `RemoteBlockCacheImpl`，
带 eventfd 完成队列）。dfkv 没有这个模块，改为**直接用 ctypes 调用 `libdfkv.so`**（与 HiCache
插件 `dfkv_hicache.py` 一致）。

dfkv 的 C ABI 是同步且内部线程安全的：`dfkv_batch_*` 阻塞，内部用线程池跨 owning node 并行
fan-out；成员 ring 由互斥锁保护，所以**一个 `dfkv_open` handle 可被多个工作线程共享**。
`ctypes.CDLL` 在外部函数调用期间释放 GIL，因此把阻塞调用派发到 `ThreadPoolExecutor`
即可获得真正的并发 —— 无需原生 demux 线程或跨线程 Future 桥接。

详见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。

---

## 4. geometry 映射（dfkv_open）

dfkv 每个 value 头里带 geometry（model_hash/page_size/dtype_tag/tp_*/layer/head…），GET 时校验
读端 geometry 与写端是否一致（`HeaderMatches`）。由于**一个 connector 实例 = 一次 `dfkv_open`、
固定 geometry**，同一部署内写/读 geometry 恒相等，校验恒通过 —— geometry 字段只是
**自一致性标记**。我们从 LMCache `metadata` 里**确定性地**推导（绝不用随机值，否则重启后旧缓存读不回）：

| dfkv_open 参数 | 取值 |
|---|---|
| `model_hash` | `blake2b(model_name)` 低 64 位（稳定）|
| `page_size` | `metadata.chunk_size` |
| `dtype_tag` | 0 |
| `flags` | `use_mla ? 1 : 0` |
| `tp_size` / `tp_rank` | `world_size` / `worker_id` |
| `layer_num`/`head_num`/`head_dim` | `kv_shape[0]/[3]/[4]` |

注意：改 model / chunk_size 会让旧缓存 key 失效（`chunk_hash` 本来也会变），这是预期行为。

---

## 5. key 方案

dfkv key 是不透明字符串、server 按整串哈希分片，所以任何稳定、无碰撞的渲染都行：

```
{model_name}@{world_size}@{worker_id}@{chunk_hash_hex}
```

与 dingofs 版的区别：用**完整 chunk_hash**（不截断 32 位）、world_size/worker_id 直接写明文
（不做位打包）。dtype 不编码（chunk_hash 已是内容哈希；geometry 头另做兜底校验）。

---

## 6. 成员 / 发现：两种模式

URL 形如 `dfkv://<endpoint>/<group>`，由 `extra_config` 的 `membership` 决定解释方式：

- **mds**（默认）：`<endpoint>` 是逗号分隔的 dfkv MDS `ip:port` 列表。`dfkv_open` 传空 members，
  后台 `dfkv_start_mds_discovery(endpoint, group)` 轮询 MDS、动态重建 ring。
- **static**：`<endpoint>` 是字面成员串 `name=ip:port,name2=ip:port2`，直接传给 `dfkv_open`；
  `<group>` 不用。

启动竞态：mds 模式下 open 后到首次轮询期间 ring 可能为空，早期操作会安全地 miss（LMCache 当作
cache miss 重算）。

---

## 7. 风险与边界

- **精确大小 GET vs 不满块** —— 由变长 get 解决（§3.1）。
- **geometry 稳定性** —— 所有字段确定性推导。
- **大 chunk 传输** —— dfkv 无 4 MiB cap，但部署前应用真实 `full_chunk_size_bytes` 冒烟一次，
  确认 wire frame 不会拒绝大 value。
- **L2 adapter 路径** —— ⚠️【2026-06-29 更新】初版只支持 `remote_storage_plugin`（in-process）路径，
  因 dingofs 的 L2 adapter 绑定原生 eventfd 模型。**现已补齐 MP-server L2 adapter**（`l2_adapter.py`,
  `DfkvL2Adapter`）：用后台 asyncio loop + 三 eventfd 把 dfkv 同步 ctypes 客户端桥接到 LMCache 的
  `L2AdapterInterface`，经内置 `plugin` L2 adapter 加载（`--l2-adapter`）。GLM-5.2 上实测 store→重启→
  从 dfkv 回载、prefill 跳过。这是 `LMCacheMPConnector`（多 KV-group 模型，如 GLM-5.x DSA）的唯一可用路径。
- **无 remove / 枚举** —— dfkv 无此 RPC，`list()` 返回 `[]`。
- **RDMA 可达性** —— a100↔jg29/jg31 需在同一 IB 网络，否则回退 TCP。
