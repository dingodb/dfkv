# dfkv vLLM connector — 部署与配置指南

`DfkvStoreConnector` 是 vLLM `KVConnectorBase_V1` 直连连接器：把 KV cache 经
**GPUDirect RDMA** 直接读写到 dfkv 集群,**绕开 LMCache**,占据与
`MooncakeStoreConnector` 相同的 `--kv-transfer-config` 槽位。生产者和消费者读写同一
共享池,实现跨请求、跨实例、跨重启的前缀复用。

本文是端到端部署 + **各参数推荐配置**。快速参考见 `integration/vllm/README.md`。

---

## 0. 角色与前置条件

| 角色 | 要求 |
|---|---|
| **dfkv 存储节点** | NVMe SSD + 400G RDMA 网卡(IB/RoCE);跑 `dfkv_server`(可选 `dfkv_mds`+etcd 做动态成员) |
| **推理节点** | H100/A100 等 GPU + 同一 RDMA fabric;跑 vLLM(≥0.23.0) |
| **GPUDirect** | GPU 节点须加载 `nvidia-peermem`(`lsmod \| grep nvidia_peermem`),否则 `ibv_reg_mr` 拿不到 GPU MR |
| **KV 可再生** | dfkv 是纯 cache:节点丢失 = miss = 重算,无副本、无对象存储兜底 |

> dfkv 与 vLLM 可同机(GPU 节点既跑 server 又跑 vLLM,池化本机 NVMe),也可分离。

> **前提:先按 [docs/DEPLOY.md](../DEPLOY.md) 部署好 dfkv 集群(server + MDS);把 libdfkv.so
> 拷到推理节点;`pip install -e integration/vllm` 装 dfkv_vllm 包。**
> ⚠️ **`members` 端口必须是 server 的 `--rdma-port`(RDMA QP bootstrap),不是 `--port`。**

---

## 第 1 步:启动 vLLM

**推荐(生产):MDS 动态发现** —— 配 `mds_endpoints` + `mds_group`,连接器内部调
`dfkv_start_mds_discovery` 自动轮询 MDS、epoch 变化时重建环,节点增减无需重启 vLLM。
连接器要求 **`mds_endpoints` 或 `members` 二选一**,设了 `mds_endpoints` 即优先走 MDS。

```bash
PYTHONHASHSEED=0 \                       # ★ 必设,见下方说明,否则跨进程/重启不命中
DFKV_RDMA=1 \
DFKV_RDMA_DEV=ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7 \
DFKV_LIB=/opt/dfkv/libdfkv.so \          # 数据面 8×400G 多轨(逗号列表);单口节点列单个
vllm serve <model> \
  --tensor-parallel-size 2 --data-parallel-size 4 \
  --kv-transfer-config '{
    "kv_connector": "DfkvStoreConnector",
    "kv_connector_module_path": "dfkv_vllm.connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "mds_endpoints": "192.168.0.8:28150,192.168.0.9:28150,192.168.0.10:28150",
      "mds_group": "glm",
      "model_hash": "1234567890",
      "batch_concurrency": "8"
    }
  }'
```

> ⚠️ **`DFKV_RDMA_DEV` 默认列全 8 轨**:一台 8×400G 节点把 8 个口都列上(逗号分隔)。
> hd04 当前只有 `ib7s400p0,ib7s400p1` 两轨 up,但标准节点有 8 轨——按本机实际 up 的口列。

**备选(单节点 / 简单部署):静态成员表** —— 无 MDS 时改用 `members`,节点增减需重启 vLLM:

```bash
    "kv_connector_extra_config": {
      "members": "n1=192.168.1.1:28001,n2=192.168.1.2:28001",   # 端口=server --rdma-port
      "model_hash": "1234567890",
      "batch_concurrency": "8"
    }
```

> **`PYTHONHASHSEED=0` 是头号坑。** dfkv key 的 chunk_hash 源自 vLLM 的 block hash,
> 而 vLLM 0.23.0 的块哈希用 Python `hash()`——默认每进程随机化。DP 各 rank 是**独立进程**,
> 不固定 seed 则同样的 token 在不同 rank/重启后算出**不同的 key**,跨进程/跨重启复用静默掉到 ~0
> (写成功、读永不命中)。每个 rank 都要设,且全实例一致。

---

## 第 2 步:验证

1. **首轮(cold)**:发一个长 prompt,记 TTFT。
2. **重启 vLLM**(或换一个 DP 实例)后**发同一 prompt**:若连接器工作,vLLM 跳过 prefill
   (调度日志 `num_computed_tokens` 接近满、`WAITING_FOR_REMOTE_KVS`),TTFT 大幅下降,
   **输出与 cold 逐字一致**。
3. server 侧 `dfkvctl stat --all` 或 `/metrics` 看 get 命中、写入量。

不命中排查顺序:`PYTHONHASHSEED` → MDS 可达(或静态 `members` 端口是否 rdma-port) →
`nvidia-peermem` → `model_hash`/几何是否一致(见下)。

---

## 配置项参考

### A. 环境变量(每个 vLLM 引擎进程都要设)

| env | 默认 | 推荐 | 说明 |
|---|---|---|---|
| `DFKV_RDMA` | 未设=TCP | `1` | 选 RDMA 传输;未设则 TCP 回退 |
| `DFKV_RDMA_DEV` | — | 本机所有 400G 口名,逗号列表(标准节点 8 轨 `ib7s400p0,...,p7`) | RDMA 轨,逗号列表=多轨;`DFKV_RDMA=1` 时必填(hd04 当前仅 p0,p1 up,标准节点 8 轨) |
| **`PYTHONHASHSEED`** | 未设 | **`0`(全 rank/实例一致)** | 跨进程/跨重启 key 确定性,**不设=不命中** |
| `DFKV_RDMA_DEPTH` | `1` | **保持 1** | 每连接在途请求数;延迟隐藏、**非吞吐旋钮**(GET/PUT 都 depth-flat,server 单连接串行) |
| `DFKV_RDMA_NUMA` | `0` | 多 NUMA 大机可设 `1` | 绑 buffer/线程到轨的 NUMA 节点 + 选 NUMA-local 轨 |
| `DFKV_LIB` / `DFKV_BUILD` | — | `libdfkv.so` 路径 | 被 extra_config.lib 覆盖 |
| `DFKV_ACCESS_LOG_ENABLED` | `0` | 排查时设 `1` | 逐 op 访问日志(`batch_get_auto_sg`/`batch_put_sg`/`batch_exist`/`register_memory`);关=约 100ns 空操作,开=异步落盘、热路径约 µs |
| `DFKV_ACCESS_LOG_PATH` | stderr | `/var/log/dfkv/vllm-access.log` | 日志路径;空=写 stderr |
| `DFKV_ACCESS_LOG_THRESHOLD_US` | `0` | 查长尾设 `1000` | 仅记耗时 ≥ 该 µs 的 op,`0`=全记 |

> 访问日志与 dfkv HiCache / LMCache 连接器**同一套环境变量、同一行格式**:
> `<op>(<args>) : <result> <秒>`,如
> `batch_get_auto_sg(20 keys) : hits=20/20, 1310720 bytes <0.007234>`。
> 每个 vLLM 引擎进程各写各的,多 DP rank 建议用 `DFKV_ACCESS_LOG_PATH` 区分(如带 rank 后缀)或留 stderr 随引擎日志走。

> **车队指标（push，opt-in）**：除访问日志外，可把本实例指标经 OTLP 推到中心 Collector→Grafana（命中率/吞吐/op 延迟 + 逐 peer 延迟）。vLLM 连接器 telemetry **只认环境变量**（不读 `kv_connector_extra_config`）：`export DFKV_METRICS_ENABLED=1 OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector>:4317`，默认 stdlib 零依赖；想要空闲节点也出逐 peer 延迟再 `export DFKV_PROBE_INTERVAL_MS=5000`。接法见 [../../deploy/observability/CONNECTOR-USAGE.md](../../deploy/observability/CONNECTOR-USAGE.md)、指标见 [../METRICS.md](../METRICS.md) §3.4。

### B. `kv_connector_extra_config`

> 成员发现:连接器要求 **`mds_endpoints` 或 `members` 二选一**;设了 `mds_endpoints` 即优先走 MDS,生产推荐 MDS。

| key | 默认 | 推荐 | 说明 |
|---|---|---|---|
| `mds_endpoints` | — | `ip:port,...`(dfkv_mds 层,如 `192.168.0.8:28150,...`) | **生产首选**;设了即走 MDS 动态发现,省略 `members` |
| `mds_group` | `default` | `glm` 等 | MDS 成员组名,与 `dfkv_server --group` 一致 |
| `mds_poll_ms` | `3000` | 默认即可 | MDS 轮询间隔(ms),epoch 变化时重建环 |
| `members` | static fallback(用 mds_endpoints 时省略) | `n=ip:rdma-port,...` | **端口必须是 server 的 `--rdma-port`**,不是 `--port` |
| `model_hash` | `0` | 每模型一个固定 uint64 | key 命名空间;共享需几何一致(见下) |
| `lib` | env 兜底 | `libdfkv.so` 绝对路径 | |
| `batch_concurrency` | `8` | **大池可调高到≈节点数** | 跨节点 fan-out,**真正的吞吐杠杆**(depth 是平的) |
| `load_async` | `True` | **保持 True** | 异步 load,走 `WAITING_FOR_REMOTE_KVS`、不占关键路径 |
| `enable_cross_layers_blocks` | `False` | 默认 False | 仅当引擎分页布局层内交错时开 |
| `lookup_rpc_port` | ipc 自动 | 一般不设 | rank0 前缀查询 RPC,仅 socket 名冲突时设 |

> dfkv 集群侧 flag(`--dir`/`--port`/`--rdma-port`/`--mds`/`--group`/`--advertise` 等)见
> [docs/DEPLOY.md](../DEPLOY.md);**连接器侧只需记住 `members` 端口 = server 的 `--rdma-port`、
> `mds_group` = server 的 `--group`**。

---

## 按场景的推荐配置

**单实例 / 单 DP**:`PYTHONHASHSEED=0`(跨重启复用仍需)+ `DFKV_RDMA=1` + `batch_concurrency=8`
默认即可,depth 保持 1。

**多 DP / 多实例共享池**:除上面外,**所有 rank、所有实例的 `PYTHONHASHSEED` 必须同值**,
且 `model_hash` 相同 + 几何一致(见下)。这是跨 DP 复用能成立的前提。

**大集群 / 宽池**:把 `batch_concurrency` 提到接近 dfkv 节点数,让一批 KV 在更多节点并行;
depth 仍保持 1(无用)。吞吐杠杆永远是**多连接 fan-out + 少而大的 key**,不是 depth。

**长上下文(50k+)**:load 带宽随上下文线性增长,单盘会成为瓶颈;靠**分布式存储环**(多
server、多盘)摊带宽,而非调 depth。首请求 JIT 见下。

---

## 几何守卫(共享池务必确认)

实例 A 写的 KV,只有当张量几何与实例 B 一致时才能安全被 B 读回。共享 `model_hash` 前确认
全部相同:**`--kv-cache-dtype`、page/block size、KV 内存布局、`--max-model-len`**。dfkv 值
头只守 `payload_len`(字节大小)——**同大小不同布局会被静默读成脏数据**。要隔离就用不同的
`served-model-name` / `model_hash`。

---

## 实测结果(hd04 H100 + IB,DeepSeek-V4-Flash,参考)

- **功能**:5 个 kv_cache_group(MLA + 多组 SWA)全部正确 offload,跨重启 + 跨 DP 命中
  (present=1058/1058、failed=0),输出与 cold 逐字一致,vLLM 真跳 prefill。
- **首请求 JIT**:每个 DP rank 的**第一个**请求付一次性 ~2s Triton JIT(resumed-prefill +
  SWA-index kernel);暖后 12k 上下文 WARM≈2s < COLD 2.7s。在意首 token 延迟就在启动后给每个
  rank 打一个合成命中预热。
- **SG 合并**:每 chunk 一个 key(而非每层段一个),25392→1242 key(~20×),减少 per-key 磁盘读。
- **depth 平**:裸 GET 单连接 depth 1 = depth 32 ≈ 1.24 GB/s,完全一样;不要指望调 depth 提速。
- **传输层**:裸 GET 8 连接 5.2 GB/s、16 连接 6.2 GB/s(详见 `docs/datapath-perf-notes.md`)。

---

## 已知问题 / 排查

| 现象 | 原因 / 解 |
|---|---|
| 写成功但**读永不命中** | `PYTHONHASHSEED` 没设或各 rank 不一致(头号坑);或 `model_hash`/几何不一致 |
| 每个 RDMA `put` 失败 `rc=-1` | `members` 指了 `--port` 而非 `--rdma-port` |
| GPU buffer 上 `dfkv_get_auto` 段错误 | 单 get 在 CPU 上算 CRC;连接器只走 zero-copy 的 batch 路径(已内置) |
| `ibv_reg_mr` 失败 / 无 GPUDirect | GPU 节点没加载 `nvidia-peermem` |
| 首 token 偶发慢 ~2s | 每 DP rank 一次性 Triton JIT(非 bug);预热可消 |
| 异构 HCA(`max_sge<30`)某些 key 不缓存 | SG 段数客户端固定 29;超限的 key 只 fail 自己(降级重算,siblings 正常),非 corruption |

---

## 相关文档
- `integration/vllm/README.md` — 快速参考
- `docs/datapath-perf-notes.md` — RDMA 数据面性能笔记(depth-flat、coalescing 等)
- `docs/METRICS.md` — Prometheus 观测
- `docs/DEPLOY.md` — dfkv 集群(server + MDS)标准 rollout
- `docs/superpowers/specs/2026-06-18-dfkv-vllm-store-connector-design.md` — 设计文档
