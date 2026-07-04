# dfkv 参数配置参考（服务端 + 客户端 + connector）

> 本文是 dfkv **所有可调参数的权威清单**：服务端 flag/env、客户端 `dfkv_open_v2`
> config struct、三条 connector 的 `extra_config` 键，按"作用域 × 渠道 × 默认 ×
> 优先级 × 冷热切换"矩阵组织。对接路径与部署见
> [CONNECTORS.md](CONNECTORS.md) / [DEPLOY.md](DEPLOY.md)。
>
> **三条参数渠道**：
> - **CLI flag**（服务端 `dfkv_server` / `dfkv_mds`）—— `--help` 自文档化
> - **`DFKV_*` 环境变量**（服务端 + 客户端 C 层）—— systemd `Environment=` 或进程 env
> - **`extra_config`**（Python connector）—— SGLang `--hicache-storage-backend-extra-config`
>   JSON、vLLM `kv_connector_extra_config`、LMCache yaml `extra_config`
>
> **优先级**：flag > env > 默认（服务端）；v2 config 字段 > env > 默认（客户端，非
> 0/NULL 字段才覆盖 env）；extra_config > env > 默认（connector，extra_config 填进
> v2 config 或直接设 env）。

---

## 1. 服务端参数（`dfkv_server`）

### 1.1 核心启动 flag（直接构造参数，非 env facade）

| Flag | 默认 | 语义 |
|---|---|---|
| `--dir <p1[,p2,...]>` | `/tmp/dfkv_node` | NVMe 路径（逗号分隔多盘；`--cap` 总量均摊） |
| `--cap <bytes>` | 1 GiB | 总容量（LRU 自限） |
| `--port <p>` | 0（ephemeral） | TCP bootstrap/数据端口 |
| `--rdma-port <p>` | -1（关） | RDMA QP-bootstrap 端口（开启 RDMA 数据路径） |
| `--rdma-dev <name>` | env `DFKV_RDMA_DEV` | RDMA 设备名（逗号列表=多轨）；**直接传给 RdmaServer，参数优先于 env** |
| `--mds <ip:port,...>` | 空 | MDS 端点（注册用，配合 `--group`/`--id`/`--advertise`） |
| `--group <g>` | `default` | 成员组名 |
| `--id <id>` | 空 | 节点 id |
| `--advertise <ip:port>` | 空 | 对外可达地址（须 host:port） |
| `--weight <n>` | 1 | 一致性哈希权重 |
| `--metrics-port <p>` | -1（关） | Prometheus `/metrics` 端口 |
| `--metrics-bind <addr>` | 空 | metrics 绑定地址 |

### 1.2 行为 flag（env facade：flag 非空 → `setenv(overwrite=1)`，覆盖 env）

每个 flag 有对应的 `DFKV_*` env twin；flag 留空时 env 生效。

| Flag | ENV | 默认 | 语义 / 冷热切换 |
|---|---|---|---|
| `--store-engine <e>` | `DFKV_STORE_ENGINE` | `file` | `file` \| `slab`；**冷启动**（改后数据 meta 不匹配会重初始化） |
| `--slab-write <m>` | `DFKV_SLAB_WRITE` | `direct` | `direct`（O_DIRECT，无 page cache） \| `buffered` |
| `--ram-tier <on\|off>` | `DFKV_RAM_TIER` | `off` | RAM 热层（write-through arena + RDMA 零拷贝 GET） |
| `--ram-tier-bytes <n>` | `DFKV_RAM_TIER_BYTES` | 4 GiB | RAM arena 字节（注册时 pin） |
| `--slab-granularity <n>` | `DFKV_SLAB_GRANULARITY` | 1 MiB | slab 槽量子；**冷启动**（改后 store 重初始化为空） |
| `--put-inflight-limit <n>` | `DFKV_PUT_INFLIGHT_LIMIT` | 0（关） | 并发 disk PUT 上限；超额 fast-fail `kCacheFull` |
| `--rdma-depth <n>` | `DFKV_RDMA_DEPTH` | 1 | RDMA 写流水深度（**须 ≤ 服务端**；客户端 deeper 会 RNR 降级） |
| `--rdma-numa <0\|1>` | `DFKV_RDMA_NUMA` | 0 | 每连接 NUMA-local 选轨 |
| `--rdma-idle-ms <n>` | `DFKV_RDMA_IDLE_MS` | — | 空闲连接回收间隔 ms |
| `--rdma-op-timeout-ms <n>` | `DFKV_RDMA_OP_TIMEOUT_MS` | 5000 | 单 op RDMA 超时 ms（`0` = 永久阻塞） |
| `--server-uring <0\|1>` | `DFKV_SERVER_URING` | 0 | io_uring 异步 GET 路径（须 `-DDFKV_WITH_URING`） |
| `--server-uring-depth <n>` | `DFKV_SERVER_URING_DEPTH` | — | io_uring SQ 深度 |
| `--ram-flush-threads <n>` | `DFKV_RAM_FLUSH_THREADS` | — | RAM 热层 flush 线程数 |
| `--ram-tier-numa <n>` | `DFKV_RAM_TIER_NUMA` | — | RAM 热层 NUMA 节点 |
| `--slab-table-sync-ms <n>` | `DFKV_SLAB_TABLE_SYNC_MS` | — | slab 表同步周期 ms |
| `--log <level>` | `DFKV_LOG` | `INFO` | 日志级别：`INFO\|DEBUG\|WARN\|ERROR` |

### 1.3 仅 env（无 flag，服务端进程级）

| ENV | 默认 | 语义 |
|---|---|---|
| `DFKV_PROBE_INTERVAL_MS` | 0（关） | 客户端主动 per-peer 延迟探针间隔（仅客户端用，但服务端进程也可设） |
| `DFKV_RDMA_MAX_BLOCK_BYTES` | 0（不声明） | DCP1 per-connection 容量声明（v1.11.0+，见 [CONNECTORS.md](CONNECTORS.md)） |
| `DFKV_RDMA_MAX_PAYLOAD_BYTES` / `DFKV_RDMA_MAX_MSG_BYTES` | 64 MiB | RDMA 最大 payload（clamp） |
| `DFKV_RDMA_CONNECT_MS` | 3000 | RDMA 连接超时 ms |
| `DFKV_RDMA_IO_MS` | 10000 | RDMA I/O 超时 ms |
| `DFKV_RDMA_POOL_MAX` | 256 | 空闲连接池上限 |

---

## 2. MDS 参数（`dfkv_mds`）

| Flag | ENV | 默认 | 语义 |
|---|---|---|---|
| `--listen <port>` | — | 0 | MDS 服务端口 |
| `--etcd <host:port>` | — | `127.0.0.1:2379` | etcd 端点（无 http:// scheme） |
| `--etcd-probe-ms <n>` | `DFKV_MDS_ETCD_PROBE_MS` | 30000 | etcd 可达性探测窗口 ms（超时退出） |
| `--metrics-port <p>` | — | -1（关） | Prometheus `/metrics` |
| `--metrics-bind <addr>` | — | 空 | metrics 绑定地址 |

---

## 3. 客户端参数（`libdfkv.so` / `dfkv_open_v2`）

### 3.1 `dfkv_client_config_t`（v2 config struct，PR#121）

字段顺序/类型镜像 `src/client/dfkv_c_api.h`。**0/NULL 字段 → env fallback**（v1 兼容）。

| 字段 | 类型 | 0/NULL 行为 | 语义 |
|---|---|---|---|
| `members` | `const char*` | 必填（或 `mds_endpoints` 路径） | `"name=ip:port,..."` 静态成员；`""` 走 MDS 发现 |
| `model_hash` | `uint64_t` | 0（裸奔，建议显式设） | value header 几何守卫 |
| `page_size` | `uint32_t` | 64 | 页大小 |
| `dtype_tag` | `uint32_t` | 0 | dtype 标签 |
| `flags` | `uint32_t` | 0 | bit0 = MLA |
| `tp_size` / `tp_rank` | `uint32_t` | 0 | TP 维度（MLA backup_skip 用 tp_rank） |
| `layer_num` / `head_num` / `head_dim` | `uint32_t` | 0 | 几何守卫 |
| `rdma_depth` | `uint32_t` | 0 → env `DFKV_RDMA_DEPTH` | RDMA 流水深度 |
| `rdma_numa` | `uint32_t` | 0 → env `DFKV_RDMA_NUMA` | NUMA 选轨 |
| `rdma_dev` | `const char*` | NULL → env `DFKV_RDMA_DEV` | RDMA 设备列表 |
| `require_rdma` | `uint32_t` | 0 → env `DFKV_REQUIRE_RDMA` | 强制 RDMA（无设备则失败） |
| `batch_concurrency` | `uint32_t` | 0 → 默认 8 | 每 client 批量并发上限 |

**v2 的 scoped env**：非 0/NULL 字段在 `dfkv_open_v2` 内部临时 setenv → 构造 KVClient → 恢复原 env。**不泄漏到进程 env**，多 connector 实例同进程不互相覆盖（v1 的 `os.environ[...]=...` 副作用已消除，PR#122）。

### 3.2 MDS 发现（`dfkv_start_mds_discovery`，独立于 v2 config）

`mds_endpoints` / `mds_group` / `mds_poll_ms` **不进 v2 struct**——它们是运行时动态发现参数，走独立 C API：

```c
// 阶段 1: dfkv_open_v2(members="")  →  空 ring client
// 阶段 2: dfkv_start_mds_discovery(c, mds_endpoints, group, poll_ms)
//         → 后台 MdsMemberPoller 周期性填 ring
```

两阶段设计：构造（几何/传输一次性配置）与发现（运行时后台线程）解耦。`members` 非空时走静态直连，不启动 poller。

### 3.3 仅 env（客户端 C 层，无 v2 字段）

| ENV | 默认 | 语义 |
|---|---|---|
| `DFKV_LIB` | — | libdfkv.so 路径（三 connector 共享 loader 解析） |
| `DFKV_BUILD` | `build` | libdfkv.so 父目录（`$DFKV_BUILD/libdfkv.so`） |
| `DFKV_PROBE_INTERVAL_MS` | 0（关） | 客户端 per-peer 延迟探针（KVClient 构造时读，env 路径） |
| `DFKV_ACCESS_LOG_ENABLED` | 0 | 访问日志开关（connector 侧） |
| `DFKV_ACCESS_LOG_PATH` | 空（stderr） | 访问日志路径 |
| `DFKV_ACCESS_LOG_THRESHOLD_US` | 0 | 仅记超此阈值的 op |
| `DFKV_METRICS_ENABLED` | 0 | OTLP fleet 指标推送 |
| `DFKV_CONNECTOR_CLIENT_RANKS` | unset | vLLM connector store 收敛（v1.11.0+，见 [CONNECTORS.md](CONNECTORS.md)） |
| `DFKV_CONNECTOR_BATCH_MAX_KEYS` | 512 | LMCache connector 批量键上限 |
| `DFKV_CONNECTOR_GET_PARALLELISM` | 1 | LMCache connector GET 并发 |
| `DFKV_CONNECTOR_ASSUME_EXISTS` | 0 | LMCache connector 假设存在优化 |

---

## 4. Connector `extra_config` 键

三 connector 共用的几何/连接键（填进 v2 config 或 start_mds_discovery）：

| 键 | SGLang | vLLM | LMCache | 进 v2 config? |
|---|---|---|---|---|
| `members` / `mds_endpoints` / `mds_group` / `mds_poll_ms` | ✅ | ✅ | ✅（URL 解析） | members ✅；其余 → start_mds_discovery |
| `model_hash` / `page_size` / `dtype_tag` / `flags` | ✅ | ✅（geometry） | ✅（geometry） | ✅ |
| `tp_size` / `tp_rank` / `layer_num` / `head_num` / `head_dim` | ✅ | ✅（geometry） | ✅（geometry） | ✅ |
| `lib` / `lib_path` | `lib_path` | `lib` | `lib_path` | 否（loader） |
| `rdma_depth` / `rdma_numa` / `rdma_dev` / `require_rdma` | ✅ | ✅（v1.13.1+） | — | ✅（v2 scoped env） |
| `batch_concurrency` | ✅ | ✅ | — | ✅（v2 直接设 KVClient） |

SGLang 专属：`probe_interval_ms`、`rail_affinity`（deprecated，no-op）、`interface_v1`（必填=1，零拷贝契约）。
vLLM 专属：见 `DFKV_CONNECTOR_CLIENT_RANKS`。
LMCache 专属：URL `dfkv://<endpoint>/<group>`；`membership` = `mds`（默认） \| `static`。

**unknown-key 警告**：当前 connector 不校验 unknown 键（拼错静默忽略）。计划在统一 `dfkv_config.py` 里加（B2 收尾，待 PR）。

---

## 5. 版本号

- 单一来源：`VERSION` 文件（CMake `file(READ)` 读，`project(dfkv VERSION ...)` 用）。
- `dfkv_server --version` / `dfkv_mds --version` / `dfkvctl --version` 报该值。
- bump 只改 `VERSION` 一个文件（v1.12.1 起，PR#120 修了 v1.12.0 的两处来源 drift）。

---

## 6. 参数治理历史

- **v1.12.0**：服务端 flag facade 从 6 个补到 19 个（PR#119）；`--help` 全自文档化。
- **v1.12.0**：版本号单一来源（PR#120）。
- **v1.12.1**：客户端 `dfkv_open_v2` config struct + scoped env（PR#121），消除 connector `os.environ[...]` 副作用（PR#122）。
