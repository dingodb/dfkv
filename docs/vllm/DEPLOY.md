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

---

## 第 1 步:编译 dfkv(带 RDMA)

```bash
cmake -S . -B build -DDFKV_WITH_RDMA=ON          # 数据面 400G 必须开;可选 -DDFKV_WITH_URING=ON
cmake --build build -j
ldd build/dfkv_server | grep ibverbs              # 确认链接了 RDMA 库
# 产物:build/dfkv_server  build/dfkv_mds  build/libdfkv.so
```

`libdfkv.so` 是连接器唯一的原生依赖,拷到推理节点即可(或挂载共享盘)。

---

## 第 2 步:启动 dfkv 存储集群

每台缓存节点起一个 `dfkv_server`。**关键:`--rdma-port` 是 RDMA QP bootstrap 端口,
连接器的 `members` 必须指向它,而不是 `--port`。** `--advertise` 也用 rdma-port。

```bash
dfkv_server \
  --dir /mnt/disk1/dfkv,/mnt/disk2/dfkv,/mnt/disk3/dfkv \   # 多盘逗号分隔,节点内 Ketama
  --port 28000 --rdma-port 28001 --rdma-dev ib7s400p0 \      # port=TCP bootstrap; rdma-port=RDMA bootstrap; rdma-dev=数据面400G口
  --cap 6597069766656 \                                      # 总容量(字节),按盘均分,自带 LRU 自限
  --mds 10.0.0.1:9400,10.0.0.2:9400 --group glm \            # 可选:接 MDS 动态成员(否则连接器用静态 members)
  --id n1 --advertise 192.168.1.1:28001                      # advertise 端口 = rdma-port
```

- **容量隔离**:`--cap` 设保守值,确认 `现网用量 + dfkv cap + 预留 < 物理总量`。
- **MDS(可选)**:多副本无状态 `dfkv_mds --listen 9400 --etcd ...` + 节点 `--mds`;连接器目前用
  **静态 `members`**,MDS 主要服务 SGLang 侧,vLLM 侧直接列服务端 rdma 地址即可。
- 观测:加 `--metrics-port 28110` 开 Prometheus `/metrics`(off 时无监听,见 `docs/METRICS.md`)。

---

## 第 3 步:在推理节点安装 connector

```bash
pip install -e integration/vllm        # 提供 dfkv_vllm 包(纯 Python)
# libdfkv.so 路径通过 extra_config.lib 或 env DFKV_LIB 指定
```

---

## 第 4 步:启动 vLLM

```bash
PYTHONHASHSEED=0 \                       # ★ 必设,见下方说明,否则跨进程/重启不命中
DFKV_RDMA=1 DFKV_RDMA_DEV=ib7s400p0 \    # 选 RDMA 传输 + 数据面轨
DFKV_LIB=/opt/dfkv/libdfkv.so \
vllm serve <model> \
  --tensor-parallel-size 2 --data-parallel-size 4 \
  --kv-transfer-config '{
    "kv_connector": "DfkvStoreConnector",
    "kv_connector_module_path": "dfkv_vllm.connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "members": "n1=192.168.1.1:28001,n2=192.168.1.2:28001",
      "model_hash": "1234567890",
      "batch_concurrency": "8"
    }
  }'
```

> **`PYTHONHASHSEED=0` 是头号坑。** dfkv key 的 chunk_hash 源自 vLLM 的 block hash,
> 而 vLLM 0.23.0 的块哈希用 Python `hash()`——默认每进程随机化。DP 各 rank 是**独立进程**,
> 不固定 seed 则同样的 token 在不同 rank/重启后算出**不同的 key**,跨进程/跨重启复用静默掉到 ~0
> (写成功、读永不命中)。每个 rank 都要设,且全实例一致。

---

## 第 5 步:验证

1. **首轮(cold)**:发一个长 prompt,记 TTFT。
2. **重启 vLLM**(或换一个 DP 实例)后**发同一 prompt**:若连接器工作,vLLM 跳过 prefill
   (调度日志 `num_computed_tokens` 接近满、`WAITING_FOR_REMOTE_KVS`),TTFT 大幅下降,
   **输出与 cold 逐字一致**。
3. server 侧 `dfkvctl stat --all` 或 `/metrics` 看 get 命中、写入量。

不命中排查顺序:`PYTHONHASHSEED` → `members` 端口是否 rdma-port → `nvidia-peermem` →
`model_hash`/几何是否一致(见下)。

---

## 配置项参考

### A. 环境变量(每个 vLLM 引擎进程都要设)

| env | 默认 | 推荐 | 说明 |
|---|---|---|---|
| `DFKV_RDMA` | 未设=TCP | `1` | 选 RDMA 传输;未设则 TCP 回退 |
| `DFKV_RDMA_DEV` | — | 本机 400G 口名(`ib7s400p0`) | RDMA 轨,逗号列表=多轨;`DFKV_RDMA=1` 时必填 |
| **`PYTHONHASHSEED`** | 未设 | **`0`(全 rank/实例一致)** | 跨进程/跨重启 key 确定性,**不设=不命中** |
| `DFKV_RDMA_DEPTH` | `1` | **保持 1** | 每连接在途请求数;延迟隐藏、**非吞吐旋钮**(GET/PUT 都 depth-flat,server 单连接串行) |
| `DFKV_RDMA_NUMA` | `0` | 多 NUMA 大机可设 `1` | 绑 buffer/线程到轨的 NUMA 节点 + 选 NUMA-local 轨 |
| `DFKV_LIB` / `DFKV_BUILD` | — | `libdfkv.so` 路径 | 被 extra_config.lib 覆盖 |

### B. `kv_connector_extra_config`

| key | 默认 | 推荐 | 说明 |
|---|---|---|---|
| `members` | 必填 | `n=ip:rdma-port,...` | **端口必须是 server 的 `--rdma-port`**,不是 `--port` |
| `model_hash` | `0` | 每模型一个固定 uint64 | key 命名空间;共享需几何一致(见下) |
| `lib` | env 兜底 | `libdfkv.so` 绝对路径 | |
| `batch_concurrency` | `8` | **大池可调高到≈节点数** | 跨节点 fan-out,**真正的吞吐杠杆**(depth 是平的) |
| `load_async` | `True` | **保持 True** | 异步 load,走 `WAITING_FOR_REMOTE_KVS`、不占关键路径 |
| `enable_cross_layers_blocks` | `False` | 默认 False | 仅当引擎分页布局层内交错时开 |
| `lookup_rpc_port` | ipc 自动 | 一般不设 | rank0 前缀查询 RPC,仅 socket 名冲突时设 |

### C. dfkv_server 关键 flag

| flag | 推荐 | 说明 |
|---|---|---|
| `--dir` | 多块 NVMe 逗号分隔 | 节点内 Ketama 分散 |
| `--port` / `--rdma-port` | 错开两个端口 | TCP bootstrap / RDMA bootstrap |
| `--rdma-dev` | 数据面 400G 口名 | |
| `--cap` | 保守值(留现网余量) | LRU 自限 |
| `--mds`/`--group`/`--id`/`--advertise` | 接 MDS 时配 | advertise 端口=rdma-port |

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
