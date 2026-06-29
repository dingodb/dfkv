# dfkv LMCache connector — 实现

源码：[../../integration/lmcache/](../../integration/lmcache/)（Python 包 `dfkv_connector`）
+ dfkv 核心的 C ABI 改动（[../../src/](../../src/)）。

---

## 1. 包结构

```
integration/lmcache/
├── pyproject.toml            # name="dfkv-connector"，纯 Python（py3-none-any wheel）
├── Makefile                  # make lib / wheel / install / test
├── README.md
└── src/dfkv_connector/
    ├── __init__.py           # 导出 DfkvConnector, DfkvConnectorAdapter
    ├── adapter.py            # DfkvConnectorAdapter —— dfkv:// URL + extra_config
    ├── config.py             # parse_dfkv_url -> DfkvEndpoint（mds | static）
    ├── remote_connector.py   # DfkvConnector(RemoteConnector)
    ├── native_client.py      # DfkvNativeClient —— ctypes + ThreadPoolExecutor
    ├── key_mapper.py         # CacheEngineKey -> dfkv key 字符串
    ├── access_log.py         # 逐操作访问日志（默认关，DFKV_ACCESS_LOG_*）
    └── exists_cache.py       # ExistsLRU —— 存在性短路
```

因为对接走 ctypes + `libdfkv.so`（运行时按路径加载），**没有要编译的 CPython 扩展**，wheel 是
纯 Python、平台无关的。

---

## 2. 各文件职责

| 文件 | 职责 | 相对 dingofs 版 |
|---|---|---|
| `exists_cache.py` | 进程内 LRU，短路「刚 put 又问存在」的远程往返 | 原样复制 |
| `access_log.py` | 逐操作一行日志，禁用时 ~100ns/调用 | 复制；`DINGOFS_*`→`DFKV_*`，logger `dfkv.access` |
| `key_mapper.py` | `CacheEngineKey` → dfkv key 串 | 重写：完整 hash、明文 ws/wid、去掉位打包 |
| `config.py` | 解析 `dfkv://<endpoint>/<group>`，两种成员模式 | 重写：mds/static 双模式 |
| `adapter.py` | `ConnectorAdapter`：识别 `dfkv://` / `plugin://dfkv`，读 extra_config，建 connector | 移植改名；knob 改为 url/membership/lib/mds_poll_ms；删 max_chunk_mib/conf |
| `native_client.py` | ctypes 加载 `libdfkv.so`，异步封装 batch_set/get/exists | **重写**（核心） |
| `remote_connector.py` | `RemoteConnector` 全部接口 | 移植改名；删 cap；接 reshape；改名环境变量 |

`l2_adapter.py`：⚠️【2026-06-29】初版丢弃（dingofs 版依赖原生 eventfd）；**现已重新实现** `DfkvL2Adapter`
——纯 Python，后台 asyncio loop 跑 `DfkvNativeClient` 的 batch_set/get/exists 协程，三个 `create_event_notifier`
（store/lookup/load）+ done-callback 桥接到 LMCache `L2AdapterInterface`；`ObjectKey`→`model_name@kv_rank@group@hash[@salt]`；
经 LMCache 内置 `plugin` L2 adapter（`--l2-adapter {"type":"plugin",...}`）加载，给 `LMCacheMPConnector`（多 KV-group 模型）用。
单测 `tests/test_l2_adapter.py`（fake client）+ GLM-5.2 真机 store/重启/回载已验证。

---

## 3. native_client.py —— ctypes + 线程池异步模型

替换 dingofs 的 pybind11 eventfd 完成队列。要点：

1. **加载库**：`load_lib(path)` 用 `ctypes.CDLL` 加载 `libdfkv.so`，声明全部 C ABI（含新增
   `dfkv_get_auto`/`dfkv_batch_get_auto`）的 `restype`/`argtypes`。库路径优先级：显式 `lib_path`
   → 环境 `DFKV_LIB` → `$DFKV_BUILD/libdfkv.so`。
2. **一个 handle/实例**：`dfkv_open(geometry…)`；mds 模式再 `dfkv_start_mds_discovery`；对
   `_collect_rdma_pools` 给出的每个区域 `dfkv_register_memory`（一次注册 host arena，RDMA
   读写其切片即免去逐操作 MR 注册）。
3. **专用线程池**：`ThreadPoolExecutor(max_workers=get_parallelism)`。异步方法用
   `loop.run_in_executor` 把**阻塞 ctypes 调用**派发到线程池；`ctypes.CDLL` 调用期间释放 GIL，
   故 N 个线程各自阻塞在 `dfkv_batch_*` 能获得真正并发。一个 handle 跨线程并发安全（dfkv 内部
   ring 有锁、其余为每调用局部量）。
4. **零拷贝指针**：`_mv_ptr(memoryview)` 用 `(c_char*nbytes).from_buffer(mv)` 取地址（可写连续
   buffer 直接别名，零拷贝）；只读 buffer 退回 `from_buffer_copy`（罕见，正确性优先）。keepalive
   对象在 C 调用返回前保持存活（RDMA 写入安全）。
5. **返回结构**：`batch_set`→`(ok, per_key)`；`batch_get`→`(ok, per_key, lengths)`（lengths 是
   每 key 真实字节数，给 reshape 用）；`batch_exists`→`per_key`。`exists_sync`/`ping_sync` 同步；
   `close()` 先 `executor.shutdown(wait=False)` 再 `dfkv_close`。

---

## 4. remote_connector.py 要点

- `__init__`：`super().__init__(config, metadata)` 计算 `meta_shapes/dtypes/fmt/full_chunk_size_bytes`；
  从 `metadata` 推导 geometry（`_geometry_from_metadata`）；建 `DfkvNativeClient`。**已删除** 4 MiB
  cap 全部逻辑。
- get/put：put 用 `len(byte_array)` 作为真实大小存（不满块也按真实大小写）；get 分配满块 buffer，
  `batch_get` 用变长 get 拿回 `(per_key, lengths)`，命中后 `_reshape_hit(obj, length)`：满块直接返回，
  不满块 `reshape_partial_chunk` 裁剪；非法长度（理论不会）当作安全 miss 丢弃。
- `batched_get` / `batched_get_non_blocking`：保持 LMCache 的「连续前缀」语义，逐命中做 reshape。
- 缓冲分配：`_allocate_chunk()` 走 `local_cpu_backend.allocate(meta_shapes[0], meta_dtypes[0], meta_fmt)`，
  是 host arena 的切片。
- `list()`→`[]`，`remove_sync`/`batched_contains`→`NotImplementedError`（保留可观测日志）。

---

## 5. C++ 侧改动（dfkv 核心）

为支持变长读，给 dfkv C ABI 增加变长 get（其余 dfkv 核心不动）：

- [../../src/kv_client.h](../../src/kv_client.h) / [.cc](../../src/kv_client.cc)：
  - `bool GetAuto(key, void* out, size_t cap, size_t* out_len)` —— 仿已有 string 版 `GetAuto`，
    `Range(0, kSize+cap)` → 解析头 → `HeaderMatches` → 校验 `payload_len<=cap` → memcpy + 回填真实长度。
  - `std::vector<bool> BatchGetAuto(items, out_lens)` —— TCP 走逐项 `GetAuto`；RDMA 仿 `BatchGet`
    的 `RangeInto` 零拷贝路径，按 (node,cap) 分组，接受任意 `payload_len<=cap` 并回填 `out_lens`。
- [../../src/dfkv_c_api.h](../../src/dfkv_c_api.h) / [.cc](../../src/dfkv_c_api.cc)：
  - `int dfkv_get_auto(c, key, ptr, cap, *out_len)` —— 1=命中/0=未命中。
  - `int dfkv_batch_get_auto(c, keys, ptrs, caps, n, out_hit, out_len)` —— 仿 `dfkv_batch_get` 的
    空指针保护，转调 `BatchGetAuto`。

`libdfkv.so` 由顶层 CMake 现有 `dfkv` 目标产出，无需改 CMake。

---

## 6. RDMA 内存注册

`_collect_rdma_pools` 从 LMCache `MixedMemoryAllocator.buffer`（一块连续 pinned tensor）取
`(addr, length)`。由于每个 Put/Get buffer 都是这块 arena 的切片，**一次注册覆盖全部流量**。
分页 / P2P allocator（`enable_p2p=true`，无单一 `buffer`）暂不支持，会打 warning 并退回逐操作
MR 注册（目标部署用默认 allocator，不触此路径）。

---

## 7. 环境变量

| 变量 | 默认 | 含义 |
|---|---|---|
| `DFKV_LIB` | `$DFKV_BUILD/libdfkv.so` | `libdfkv.so` 路径 |
| `DFKV_CONNECTOR_GET_PARALLELISM` | 1 | 并发 batched-get 组数 = 线程池 worker 数 |
| `DFKV_CONNECTOR_BATCH_MAX_KEYS` | 512 | 单次 native 批量的最大 key 数 |
| `DFKV_CONNECTOR_ASSUME_EXISTS` | 0 | 跳过远程 contains 检查（调试）|
| `DFKV_ACCESS_LOG_ENABLED` / `_PATH` / `_THRESHOLD_US` | 0 / stderr / 0 | 访问日志 |

---

## 8. 测试

- **C++ gtest**：[../../tests/get_auto_test.cc](../../tests/get_auto_test.cc) —— `GetAuto`/`BatchGetAuto`
  覆盖满块、不满块、cap 过小（miss）、geometry 不匹配（miss）、双节点批量混合大小。
- **Python 冒烟**：[../../tests/python/dfkv_lmcache_native_smoke.py](../../tests/python/dfkv_lmcache_native_smoke.py)
  —— `DfkvNativeClient` 对接本地 `dfkv_server`，put→exists→变长 get（满块+不满块）逐字节校验，
  无需 torch/lmcache。
