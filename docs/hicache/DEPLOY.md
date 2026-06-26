# dfkv SGLang HiCache 对接 — 配置指南

> 前提：先按 [docs/DEPLOY.md](../DEPLOY.md) 部署好 dfkv 集群(server + MDS)。本文只讲 SGLang HiCache 如何对接 dfkv 作 L3。

把 `libdfkv.so` + `python/dfkv_hicache.py` 放到 pod 可访问路径，免 fork、`dynamic` 侧载。
前提：GLM-5.1 = MLA（每页 KV ≈ 2.74 MiB 单对象、跨 TP 复制、仅 tp_rank0 写）。

---

## 1. 环境注入

启动 `sglang serve` 前注入环境：
```bash
export PYTHONPATH=/userdata/dfkv:$PYTHONPATH
export DFKV_LIB=/userdata/dfkv/libdfkv.so
export DFKV_RDMA=1                       # 启用 RDMA 数据面（否则 TCP）
export DFKV_REQUIRE_RDMA=1               # 可选：禁止悄悄 TCP fallback
# 数据面设备；多轨用逗号列表（标准节点 8×400G）
export DFKV_RDMA_DEV=ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7
export DFKV_RDMA_NUMA=1                   # 可选：多 NUMA 大机按线程 NUMA 选本地轨（见下）
export DFKV_RDMA_MAX_PAYLOAD_BYTES=67108864  # 可选：单 chunk payload 上限，默认 64MiB
# 可选: DFKV_RDMA_DEPTH (单连接 pipeline, K 个请求在途; 仅网络延迟隐藏、对吞吐 flat——见下注; 默认 1 即可)
```
> ⚠️ **hd04 当前只有 `ib7s400p0,ib7s400p1` 两轨 up**，但标准训练计算网节点是 8×400G，应列全 8 轨。

> ⚠️ **depth 是延迟隐藏、不是吞吐旋钮(2026-06 实测更正)**：server 单连接 serve loop 串行处理,
> 实测 **PUT 与 GET 的吞吐对 depth 都是 flat**(1≈8≈16≈32)。早先"depth>1 抬高写带宽"的说法不成立。
> 写/读吞吐的真正杠杆是 **`batch_concurrency`(多连接跨节点 fan-out,客户端默认 8)** 与**少而大的 key**;
> depth 只在网络延迟受限的链路上有意义,**默认 1 即可**,见 [../datapath-perf-notes.md](../datapath-perf-notes.md)。
> (机制仍在:extra_config `"rdma_depth":K` 会在 dfkv_open 前自动设 DFKV_RDMA_DEPTH,且 client depth ≤ server depth;但一般无需设。)

> **多轨 NUMA 选轨**（v1.2.0 起）：设 extra_config `"rdma_numa":1`（或 env `DFKV_RDMA_NUMA=1`）+ 多轨 `DFKV_RDMA_DEV`，**C++ 客户端在建连时按调用线程的 NUMA 节点自动选本地轨**（无本地轨→轮转全轨）。SGLang/vLLM 两端通吃，无需各自改，仅在新建连接触发（热路径零开销）。
> - ⚠️ **旧 `"rail_affinity"` 已废弃为 no-op**（v1.2.0）：它按 `tp_rank` 收窄，但 DP-attention 下每 rank `tp_rank=0`→塌缩到单轨。配了只打 stderr 告警、不再生效；改用上面的 `rdma_numa`。

---

## 2. SGLang 启动 + 后端配置

**方案 A — MDS 动态发现（推荐）**：配置 `mds_endpoints` + `mds_group`；插件内部调用
`dfkv_start_mds_discovery` 自动轮询 MDS，epoch 变化时重建环，无需重启。
```bash
sglang serve ... \
  --enable-hierarchical-cache --hicache-write-policy write_through \
  --hicache-mem-layout page_first_direct --hicache-io-backend direct \
  --hicache-storage-prefetch-policy timeout \
  --hicache-size <字节> \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{
    "backend_name":"dfkv","module_path":"dfkv_hicache","class_name":"DfkvHiCache",
    "interface_v1":1,
    "mds_endpoints":"10.0.0.1:9400,10.0.0.2:9400",
    "mds_group":"default",
    "model_hash": 81, "page_size":64, "dtype_tag":1178092852,
    "layer_num":78, "head_num":1, "head_dim":576 }'
```

**方案 B — 静态成员表（遗留）**：无 MDS 时配置 `members` 字段，节点增减需重启 SGLang。
```bash
sglang serve ... \
  --enable-hierarchical-cache --hicache-write-policy write_through \
  --hicache-mem-layout page_first_direct --hicache-io-backend direct \
  --hicache-storage-prefetch-policy timeout \
  --hicache-size <字节> \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{
    "backend_name":"dfkv","module_path":"dfkv_hicache","class_name":"DfkvHiCache",
    "interface_v1":1,
    "members":"n57=192.168.1.57:28001,n58=192.168.1.58:28001",
    "model_hash": 81, "page_size":64, "dtype_tag":1178092852,
    "layer_num":78, "head_num":1, "head_dim":576 }'
```

### HiCache 关键 flag

| flag | 推荐 | 说明 |
|---|---|---|
| `--hicache-storage-backend dynamic` | 必填 | 侧载 dfkv 插件，免 fork SGLang |
| `--hicache-io-backend direct` | `direct` | O_DIRECT 零拷贝读写路径 |
| `--hicache-mem-layout page_first_direct` | `page_first_direct` | 配合 direct 后端的内存布局 |
| `--hicache-write-policy write_through` | `write_through` | 写穿，L3 与 L1/L2 同步 |
| `--hicache-storage-prefetch-policy timeout` | `timeout` | prefetch 用 timeout 策略（best_effort 之外更安全的杠杆） |
| `--hicache-size <字节>` | 按 L2-L1 容量 | L3 prefetch 容量；⚠️ 须满足 L2>L1 硬约束，prefetch 容量比建议 0.8 |

> **`interface_v1:1` 是必填项**，插件 `__init__` 强校验：缺失即 `raise ValueError` 启动失败。原因——对 `dynamic` 后端，SGLang 仅在 `interface_v1` 为真时才走零拷贝 `batch_set_v1/get_v1`；否则退回 generic `set/get` 路径，而 dfkv 的 generic `get/batch_get` 是未实现的桩，会导致**写成功、L3 读静默失败**（线上踩过：launch 脚本漏配此项，14GB 写入但 prefetch 全部 miss）。
> `interface_v1:1` 触发零拷贝 `batch_set_v1/get_v1` —— GET payload 经 RDMA 散射**直落 HiCache 宿主页**（client 端零拷贝），server 端 O_DIRECT 直读入已注册 direct buffer 并 scatter-send（server 端无 payload memcpy），两端零拷贝。
> MLA 下插件自动单对象、无 rank 后缀、`backup_skip`（仅 tp_rank0 写）。decode 共享前缀配同 members。
> 多池模型（Mamba/SWA/DeepSeek-V4）用 v2 PoolTransfer 接口（插件已实现）。
> 排查命中率/慢操作：可开启 access log（`access_log`/`access_log_path` 或 `DFKV_ACCESS_LOG_*`），见 [../access_log.md](../access_log.md)。
> **客户端指标（pull）**：插件自动在 SGLang 自带的 `/metrics` 上暴露 `dfkv_client_*{tp_rank}`（set/get 量、命中、IO 错误、peer 熔断切换、set/get 延迟直方图）。后台轮询线程读 C 客户端快照，间隔由 extra_config `client_stats_poll_s`（默认 10s，`DFKV_CLIENT_STATS_POLL_S` 兜底，`0`=关）控制。全指标见 [../METRICS.md](../METRICS.md) §3.3。
> **车队指标（push，opt-in）**：还可把本实例指标经 OTLP 主动推到中心 Collector→Grafana（命中率/吞吐/op 延迟 + 逐 peer 延迟）。HiCache 走 extra_config `"metrics":1,"otlp_endpoint":...` 或 `DFKV_METRICS_ENABLED=1`，默认 stdlib 零依赖。接法见 [../../deploy/observability/CONNECTOR-USAGE.md](../../deploy/observability/CONNECTOR-USAGE.md)、指标见 [../METRICS.md](../METRICS.md) §3.4。

---

## 3. 验证

在**一个受控 SGLang 副本**上切 `dynamic` 后端，发共享长前缀请求看命中上涨，确认后推广：
- SGLang `--enable-cache-report` 的 HiCache storage hit/miss、TTFT。
- server 侧 `dfkvctl stat --all --mds <eps> --group <g>` 或 `/metrics` 看 get 命中、写入量（见 [../DEPLOY.md](../DEPLOY.md) §监控）。
- 回滚：`--hicache-storage-backend` 改回原后端（mooncake 等）重启该副本，与 dfkv 解耦。

---

## 4. 引擎与特性边界（HiCache 侧）

dfkv 现在同时服务三种引擎：**SGLang HiCache**（本文）、**LMCache**（`docs/lmcache/`）、
**vLLM 直连**（`docs/vllm/DEPLOY.md`）。后续给 vLLM 连接器加的几个特性**并不适用于 HiCache 侧**，
不是漏配，是数据形状不匹配——HiCache 维持常规路径即可：

- **scatter-gather（SG，合并 key）— HiCache 不用。** SG 把"一个 chunk 的多个层段"合成一个多-SGE
  RDMA key，是为 vLLM 连接器的变长 chunk × 多层段做的。HiCache MLA **每页就是一个打包 latent 对象
  （~2.74 MiB）**，本来一页一 key、无碎段可合，SG 在此无收益。（仅 MHA 的 `_k`/`_v` 对或未来多池
  HiCache v2 才理论上有边际收益，且需改插件代码、非开关。）
- **io_uring async GET（`DFKV_SERVER_URING`）— HiCache 不要开。** 实测单盘上对吞吐 **flat/无收益**
  （瓶颈是盘不是读提交），默认关。HiCache 单盘场景开它纯属白费。
- **`DFKV_RDMA_DEPTH` — 保持默认 1**（见 §1：实测对 PUT/GET 吞吐都 flat）。
- HiCache 真正需要的硬化与可观测性（RDMA 空闲回收、CLOCK 持久指针、MDS 硬化、Prometheus 指标）
  **已在 v1.5.2 内**，无需额外动作。

### vLLM 实例与 HiCache 实例（即便同一模型）能否共用同一个池？

- **共用同一套 dfkv 集群 / 同一个哈希环：可以。** 环只负责把 key 路由到节点；两种引擎指同一 `members`
  即可共用存储池与容量（共享 LRU）。
- **但二者不会复用彼此的 KV（同模型也不行）。** SGLang HiCache 与 vLLM 连接器用**完全不同的 key 方案**
  （前缀/哈希/`@sg` 后缀/tp_rank 处理都不同）**和不同的 KV 字节布局**（MLA 单页打包对象 vs 连接器的
  逐组分段）。同一模型、同一段 token，两边算出的 **key 不同、字节也不兼容**——只是共存于同一池、
  keyspace 互不相交，谈不上跨引擎前缀命中。
- **推荐：给两者不同的命名空间隔离 keyspace**（vLLM 侧 `model_hash` / `served-model-name` 与 HiCache 侧
  取不同值）。这样即使理论上撞到同名 key，也因命名空间不同而不会发生"同 key 不同字节"的静默错读
  （dfkv 值头只守 `payload_len` 字节大小，大小相同而布局不同会读成脏数据）。结论：**共享节点与容量、
  隔离 keyspace**。

---

## 相关文档
- [../DEPLOY.md](../DEPLOY.md) — dfkv 集群(server + MDS)标准部署
- [../access_log.md](../access_log.md) — 逐操作访问日志
- [../METRICS.md](../METRICS.md) — Prometheus 观测
- [../datapath-perf-notes.md](../datapath-perf-notes.md) — RDMA 数据面性能笔记(depth-flat、coalescing 等)
