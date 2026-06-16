# dfkv 可观测性增强设计（P0/P1/P2）

> 2026-06-17 ｜ 目标版本 v1.4.0 ｜ 状态：已确认，待实现

## 1. 背景与目标

当前 dfkv 可观测性现状（v1.3.0）：

- **服务端**：`KvNodeServer::MetricsText()` 11 个计数，只能经私有 `kStats` wire op 取（`dfkvctl stat <ip:port>` 单点），**Prometheus 无法抓取**。
- **客户端**（`KVClient`）：**零指标**，仅 `PeerHealth` 内存表。
- **MDS**：零指标。
- **RDMA 层**：零导出计数。
- **集群/环**：**完全无视角**——无法看整个集群、整个环、聚合健康；MDS 里有权威成员表但无读工具。
- **插件**（Python）：已有 access log（逐 op + 耗时）+ 每 rank Prometheus 计数（set/get calls/pages/bytes/hits）——这层较完整。

目标：补齐"看整个集群 / 整个环 / 具体 server 节点健康 / I/O / 接口调用统计"四个视角，**硬约束：不影响性能**（不碰 RDMA 数据面热路径吞吐/延迟）。

## 2. 性能安全原则（最高约束）

整个设计的每一项都必须满足：

1. **热路径定义**：RDMA 数据面（`batch_get_v1`/`batch_set_v1` → C 客户端 → RDMA WRITE/READ）、TCP handler、`KVStore::Range/Cache`、`RdmaServer` 完成环。
2. **计数器**：热路径上一律 `std::atomic<uint64_t>` 的 relaxed `fetch_add`——与现有 `kv_node_server.cc` 完全同模式。零锁、零分配、零 syscall。
3. **延迟测量**：1/64 采样。一个 relaxed 原子序号计数器，`(seq & 63) == 0` 时才取 `clock_gettime(CLOCK_MONOTONIC)`（Linux vDSO，约 20ns，无 syscall），落无锁固定桶直方图。分支高度可预测，64 次只测 1 次 → 摊销近零。
4. **HTTP `/metrics`**：独立端口 + 独立 accept 线程；仅在 Prometheus scrape 时把原子读成文本。与数据面线程零交互（除 relaxed load）。
5. **文本序列化**：只读 relaxed load，发生在 scrape 线程，不在热路径。
6. **插件客户端指标**：后台轮询线程（睡眠态，每 ~10s 醒一次读 C ABI 快照），不在 per-batch/per-page 路径上。
7. **验证**：每个 PR 跑 RDMA Soft-RoCE loopback + ThreadSanitizer（CI 已有）；P2 完成后在 hd03/hd04 真机做 A/B 裸 bench（开/关 metrics-port），确认吞吐/延迟无回归。

## 3. 分层设计

### P0 — 集群/环可见性 + 服务端可抓取

**3.0.1 内嵌 HTTP `/metrics`**（新文件 `src/metrics_http.{h,cc}`）
- 极小 HTTP/1.0 响应器，复用 `net_util.h` 的 `ReadAll/WriteAll`。
- 单 accept 线程；读请求行，路由：`GET /metrics` → 200 + Prometheus 文本；`GET /healthz` → 200 `ok`；其余 → 404。
- 接口：`MetricsHttpServer(std::function<std::string()> render)`；`Start(int port)`；`Stop()`。`render` 回调由 `dfkv_server`/`dfkv_mds` 提供（分别返回各自的 MetricsText）。
- `dfkv_server` 加 `--metrics-port <p>`（0/缺省 = 关，opt-in，默认不开端口保证零行为变化）。
- `dfkv_mds` 同样加 `--metrics-port`。
- 无第三方依赖（不引入 brpc/civetweb），自包含 ~80 行。

**3.0.2 MetricsText 升级**（`kv_node_server.cc`）
- 加 `# HELP`/`# TYPE`（counter/gauge 正确标注）。
- 加节点身份 label：`{node="<id>",group="<g>"}`（id/group 经 `set_identity()` 由 main 传入；缺省空 label 省略）。
- 新增：`dfkv_build_info{version,transport}` gauge(=1)、`dfkv_uptime_seconds`、`dfkv_start_time_seconds`。

**3.0.3 集群/环读工具**（`dfkvctl_main.cc` 扩展）
- `dfkvctl ring --mds <eps> --group <g>`：经 `MdsMemberPoller`/一次性 ListMembers 查成员表，打印 id / addr / weight / epoch；用 `ConHash` 重建环，打印每节点 vnode 数与占比（路由倾斜可见）。
- `dfkvctl stat --all --mds <eps> --group <g>`：从 MDS 发现成员 → 逐节点 `Stats()`（TCP `kStats`）→ 解析关键字段 → 打印 per-node 表 + 集群聚合（总 used/cap、objects、命中率、bytes r/w）。
- 兼容旧用法：`dfkvctl stat <ip:port>` 保持不变。

**3.0.4 MDS 指标**（新增 MDS 侧 MetricsText）
- `dfkv_mds_members{group}` gauge、`dfkv_mds_lease_grants_total`、`dfkv_mds_keepalives_total`、`dfkv_mds_etcd_errors_total`、`dfkv_mds_list_requests_total`、`dfkv_mds_register_requests_total`。
- 全 relaxed atomic，在 `Upsert`/`ListMembers`/etcd 调用点自增。

### P1 — 单节点深度 + 客户端指标

**3.1.1 服务端深度**（`kv_node_server.{h,cc}` + `kv_store.{h,cc}`）
- 淘汰计数（`KVStore` 暴露）：`dfkv_evictions_total`、`dfkv_evicted_bytes_total`、`dfkv_cache_full_total`（在 `EvictLocked` / 容量拒绝处自增）。
- 错误分型：`dfkv_errors_total{op,status}`——对 `kIOError`/`kInvalid` 计数（`kNotFound` 已是 miss，不重复）。
- 采样延迟直方图：`dfkv_op_latency_seconds{op="get|put"}`，桶边界（秒）：`50µs,100µs,250µs,500µs,1ms,2.5ms,5ms,10ms,25ms,50ms,100ms,+Inf`（12 桶）。固定 `std::atomic<uint64_t>` 数组 + `_sum`/`_count`。1/64 采样。
- 连接 gauge：`dfkv_open_connections`（accept 处 +1，Handle 退出 -1，atomic）。
- per-disk：`dfkv_disk_used_bytes{disk}`、`dfkv_disk_objects{disk}`——`DiskCacheGroup` 暴露 per-disk 视图，仅 scrape 时读。

**3.1.2 客户端指标**（新 `src/client_metrics.h`，`KVClient` 持有）
- 全 relaxed atomic：`put/get/exist` calls；`get_hit/get_miss`；`errors{status}`；per-peer `routed_total{peer}`、`error_total{peer}`（小 map，路由时自增——注意：per-peer 用 `mutex` 保护的 map 仅在**首次见到新 peer**时加锁建项，热路径命中已存在项走原子，避免热路径锁；或固定从 ring 成员预建项）。peer 熔断切换在 `PeerHealth::MarkBad/MarkGood` 计数。
- C ABI：`int dfkv_stats_snapshot(dfkv_client_t c, char* buf, uint64_t cap)` → 写 Prometheus 文本（client-labeled），返回字节数（>cap 时返回所需长度，不写）。
- **插件侧**（`dfkv_hicache.py` + `dfkv_metrics.py`）：`DfkvHiCache` 启一个 daemon 后台线程，每 `client_stats_poll_s`（默认 10s，0=关）调 `dfkv_stats_snapshot`，解析、对上次快照 diff、`.inc(delta)` 到 prometheus Counter（兼容 multiproc）。线程睡眠态，进程退出时优雅停。

### P2 — RDMA 计数 + 插件直方图

**3.2.1 RDMA 计数**（`rdma_server.cc` / `rdma_verbs.cc` / `rdma_transport.cc`）
- Server（完成环里，非 per-byte）：`dfkv_rdma_completions_total`、`dfkv_rdma_completion_errors_total`、`dfkv_rdma_active_conns` gauge、`dfkv_rdma_idle_reclaims_total`。
- Client transport：`dfkv_rdma_mr_registrations_total`、per-rail `dfkv_rdma_rail_ops_total{dev}`（在 `PickRail`/rr 选轨处自增）。
- Server 计数并入 `dfkv_server` /metrics；client 计数并入 `dfkv_stats_snapshot`。

**3.2.2 插件直方图 + 错误 label**（`dfkv_metrics.py`）
- prometheus `Histogram`：`dfkv_client_set_seconds`/`dfkv_client_get_seconds{tp_rank}`——**复用 access log 已经用 `perf_counter` 算出的耗时**（per-batch 调用一次，非 per-page），`observe()` 零额外测量成本；access log 关闭时单独测一次 per-batch（可忽略）。
- `dfkv_client_errors_total{op,tp_rank}`：从各接口 FAIL 结果自增。

## 4. 交付方式与里程碑

按 dfkv 既有节奏：**独立 PR / 维护者合 CLEAN / TDD**。

| 阶段 | PR | 内容 | 主要测试 |
|---|---|---|---|
| M1 | PR-A | P0：HTTP `/metrics` + MetricsText 升级（server+mds） | `metrics_http_test`、`metrics_test` 扩展 |
| M2 | PR-B | P0：`dfkvctl ring`/`stat --all` + MDS 指标 | `dfkvctl` smoke、mds metrics 单测 |
| M3 | PR-C | P1：服务端深度（淘汰/错误/采样延迟/连接/per-disk） | `kv_store_test`、`kv_node_server` 指标单测 |
| M4 | PR-D | P1：客户端指标 + C ABI 快照 + 插件轮询 | `client_metrics_test`、`c_api_test`、python 插件单测 |
| M5 | PR-E | P2：RDMA 计数 + 插件直方图 | `rdma_loopback_test` 扩展、`dfkv_metrics` 单测 |
| M6 | — | bump v1.4.0 + GitHub release（挂 portable artifact） | 全 ctest + 真机 A/B 裸 bench |

每个 PR：从干净 main 开分支（`feat/obs-*`，避开 `perf` 前缀）→ TDD（先 RED 后 GREEN）→ push origin → PR 到 dingodb/dfkv → 等 `mergeStateStatus=CLEAN` → squash 合并 → 同步 main。全部合完后 M6 发版。

PR 顺序有依赖：A→B（B 用 A 的 MetricsText/HTTP），C 独立，D 依赖（无强依赖，可在 C 后），E 依赖 server/client metrics 基座（A、D）。实际按 A,B,C,D,E 串行，简单稳妥。

## 5. 兼容性

- 全部**新增**，无 wire 协议变更——v1.3.0 server/client 完全兼容。
- `--metrics-port` opt-in 默认关 → 不开端口时行为与 v1.3.0 完全一致。
- 客户端快照 C ABI 为新增符号，旧插件不调用即无影响。
- 插件轮询线程 `client_stats_poll_s=0` 可关。

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 采样延迟仍碰热路径 | 1/64 + vDSO clock；P2 后真机 A/B 裸 bench 实测无回归方可发版 |
| per-peer map 热路径加锁 | 命中已存在项走纯原子；仅新 peer 首见时加锁建项（ring 成员有限、稳态零加锁） |
| HTTP 端点被滥用/暴露 | opt-in；仅 GET /metrics、/healthz；按部署网络隔离（同 cache 端口口径，见 DEPLOY.md） |
| 插件 multiproc 下 Histogram | 复用 prometheus_client multiproc 支持的 Histogram 类型（文件聚合），不用自定义 collector |
| TSan 报数据竞争 | 全 relaxed atomic；CI 每 PR 跑 TSan |

## 7. 验收标准

- Prometheus 能直接抓 `dfkv_server`/`dfkv_mds` 的 `/metrics`。
- `dfkvctl ring` 一条命令看到整个环 + vnode 分布；`dfkvctl stat --all` 看到集群聚合。
- 单节点可见：淘汰、错误分型、采样延迟 p50/p99、连接数、per-disk、RDMA 完成/错误/per-rail。
- 客户端可见：per-op/per-peer 调用、命中、错误、熔断切换（经插件 /metrics）。
- 真机 A/B 裸 bench：开 metrics-port vs 关，吞吐/延迟差异在噪声内（fails=0）。
- 全 ctest + python 插件 + TSan + RDMA loopback 绿。
