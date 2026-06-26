# dfkv LMCache connector — 使用指南

把 vLLM 的 KV cache 通过 LMCache 卸载到 **dfkv** 集群，实现跨请求/跨实例的前缀缓存复用。
本文按步骤说明：**装 connector → 改 lmcache.yaml → 起 vLLM → 验证**。

设计与实现细节见 [DESIGN.md](DESIGN.md) / [IMPLEMENTATION.md](IMPLEMENTATION.md)。

---

## 0. 角色与前置条件

> **前提：先按 [docs/DEPLOY.md](../DEPLOY.md) 部署好 dfkv 集群(server + MDS)。**

推理节点要求：

| 角色 | 跑什么 | 要求 |
|---|---|---|
| **推理节点** | vLLM + LMCache + dfkv connector | GPU；已装 vLLM、LMCache (≥0.4.5)；connector 是纯 Python 包(ctypes 调 `libdfkv.so`) |

网络要求：
- 推理节点必须能 TCP 连到每台缓存节点的 dfkv 端口。
- 若用 RDMA，推理节点与缓存节点要在**同一 IB/RoCE 网络**（否则用 TCP，见第 2 步）。

下文用占位符，请替换成你的实际值：
- `<CACHE1_IP>` `<CACHE2_IP>` … 缓存节点 IP
- `<DFKV_SRC>` dfkv 源码目录、`<LIBDFKV>` 部署好的 `libdfkv.so` 路径

---

## 第 1 步：在推理节点安装 connector

装到 vLLM 所在的 Python 环境（venv/conda）里：

```bash
source /path/to/your/vllm-venv/bin/activate
pip install <DFKV_SRC>/integration/lmcache       # 纯 Python wheel，不编译
export DFKV_LIB=<LIBDFKV>                          # 指向部署好的 libdfkv.so
```

`dfkv-connector` 是纯 Python 包（通过 ctypes 调用 `libdfkv.so`），无需匹配 ABI。

---

## 第 2 步：写 LMCache 配置 `lmcache.yaml`

新建一个 LMCache 配置文件，启用 dfkv 插件。**TCP 版**（最简单，先跑通用这个）：

```yaml
chunk_size: 16            # 每 chunk 的 token 数；dfkv 不限块大小，可调大
local_cpu: false
save_chunk_meta: false
remote_storage_plugins: ["dfkv"]
extra_config:
  remote_storage_plugin.dfkv.module_path: dfkv_connector.adapter
  remote_storage_plugin.dfkv.class_name:  DfkvConnectorAdapter
  # 静态成员：列出每台缓存节点的 name=ip:TCP端口，逗号分隔；末尾 /<group> 任意填
  remote_storage_plugin.dfkv.url:         dfkv://c1=<CACHE1_IP>:18800,c2=<CACHE2_IP>:18800/g1
  remote_storage_plugin.dfkv.membership:  static
  remote_storage_plugin.dfkv.lib:         <LIBDFKV>
```

**RDMA 版**：只改两点 —— URL 用 **RDMA 端口（18801）**，并在第 3 步给 vLLM 进程加 `export DFKV_RDMA=1`：

```yaml
  remote_storage_plugin.dfkv.url:         dfkv://c1=<CACHE1_IP>:18801,c2=<CACHE2_IP>:18801/g1
```

> 跨网卡（推理节点与缓存节点的 RDMA 设备名不同，如 a100 是 `ib6s200p0`、缓存节点是 `mlx5_0`）时，
> **不要**在推理节点设 `DFKV_RDMA_DEV`：客户端会用本机第一个设备、并让服务端用它自己的 `--rdma-dev`。
> 同轨同网卡且要多轨时，`DFKV_RDMA_DEV` 用逗号列表列全部口（标准节点 8×400G：
> `ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7`；hd04 当前仅 p0,p1 up）。

> **生产推荐 MDS 动态发现**（取代静态成员，节点增减自动生效）：把 `membership` 改为 `mds`、
> `url` 的 endpoint 改成 dfkv_mds 层的 `ip:port` 列表（组名走 URL 末尾 `/<group>`），
> 可选 `remote_storage_plugin.dfkv.mds_poll_ms`（默认 3000）。MDS 层如何部署见 [../DEPLOY.md](../DEPLOY.md) §4。
> ```yaml
>   remote_storage_plugin.dfkv.membership: mds
>   remote_storage_plugin.dfkv.url:        dfkv://192.168.0.8:28150,192.168.0.9:28150,192.168.0.10:28150/glm
> ```

---

## 第 3 步：启动 vLLM

让 vLLM 用 LMCache 连接器，并指向上面的 yaml：

```bash
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CONFIG_FILE=/path/to/lmcache.yaml
export DFKV_LIB=<LIBDFKV>
# 只有走 RDMA 才加这一行（TCP 不要加）：
export DFKV_RDMA=1

vllm serve <model_path> \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    --port 18200 --host 0.0.0.0 \
    # ... 你的其余 vllm 参数 ...
```

---

## 第 4 步：验证

1. **连接器已加载**（vLLM 启动日志里应有，且没有回退到 blackhole）：
   ```
   Discovered adapter: DfkvConnectorAdapter
   DfkvConnector ready: membership=static endpoint=c1=... full_chunk=... rdma_pools=1 ...
   ```
2. **缓存命中**：发两遍相同请求（或用同 `--seed` 的 bench 跑两遍）。第二遍 vLLM 日志的
   `External prefix cache hit rate` 应 > 0，TTFT 明显下降。也可设 `DFKV_ACCESS_LOG_ENABLED=1`
   看到 `batch_set`/`batch_get` 命中。
3. **数据确实写进了 dfkv**：在缓存节点 `du -sh <数据盘>/dfkv` 应看到 KV 数据增长。

跑 benchmark（示例）：

```bash
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len 16000 --random-output-len 100 \
  --model <model_path> --base-url http://127.0.0.1:18200 \
  --num-prompts 20 --max-concurrency 10 --seed 1109
```

---

## 配置项参考

`extra_config` 下的插件键（`remote_storage_plugin.dfkv.*`）：

| 键 | 必填 | 说明 |
|---|---|---|
| `module_path` / `class_name` | 是 | 固定为 `dfkv_connector.adapter` / `DfkvConnectorAdapter` |
| `url` | 是 | `dfkv://<endpoint>/<group>`。static 模式 endpoint=`name=ip:port,...`；mds 模式 endpoint=MDS `ip:port` 列表 |
| `membership` | 否 | `static`（默认）或 `mds` |
| `lib` | 否 | `libdfkv.so` 路径（覆盖 `DFKV_LIB`） |
| `mds_poll_ms` | 否 | mds 模式轮询间隔，默认 3000 |

环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DFKV_LIB` | — | `libdfkv.so` 路径（`lib` 键优先） |
| `DFKV_RDMA` | 关 | 设为 `1` 走 RDMA（需 RDMA 版库 + 可达 IB 网络），否则 TCP |
| `DFKV_REQUIRE_RDMA` | 关 | 设为 `1` 时禁止 TCP fallback；RDMA 不可用则 `dfkv_open` 失败 |
| `DFKV_RDMA_DEV` | 空 | 客户端 RDMA 设备名；跨网卡时**留空** |
| `DFKV_RDMA_MAX_PAYLOAD_BYTES` | 67108864 | RDMA 单 value payload 上限；普通控制 buffer 仍保持小尺寸 |
| `DFKV_CONNECTOR_GET_PARALLELISM` | 1 | 并行 batched-get 组数（提高可降 TTFT） |
| `DFKV_CONNECTOR_BATCH_MAX_KEYS` | 512 | 单次 native 批量最大 key 数 |
| `DFKV_ACCESS_LOG_ENABLED` / `_PATH` | 0 / stderr | 逐操作访问日志 |

> **车队指标（push，opt-in）**：可把本实例指标经 OTLP 推到中心 Collector→Grafana（命中率/吞吐/op 延迟 + 逐 peer 延迟），与 vLLM/SGLang 拉平。LMCache 连接器**只认环境变量**：`export DFKV_METRICS_ENABLED=1 OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector>:4317`，默认 stdlib 零依赖。接法见 [../../deploy/observability/CONNECTOR-USAGE.md](../../deploy/observability/CONNECTOR-USAGE.md)、指标见 [../METRICS.md](../METRICS.md) §3.4。

---

## 实测结果（参考）

环境：a100（vLLM 0.21 + LMCache 0.4.5）；dfkv 集群 = jg29 + jg30 各一个 `dfkv_server`
（static 成员，TCP 18800 / RDMA 18801）；模型 DeepSeek-R1-Distill-Qwen-32B（TP=1）；
`chunk_size=16`（每 chunk ≈4 MiB）；bench：random 16000-in / 100-out，20 prompts，并发 10，
同 `--seed`。冷遍写入约 **81 GB** KV（经一致性哈希分布到两节点）。

| 遍次 | 传输 | External 命中率 | 时长(s) | Mean TTFT(ms) | Median TTFT(ms) | TPOT(ms) | 吞吐(tok/s) | 失败 |
|---|---|---|---|---|---|---|---|---|
| 冷（写入）| TCP | 0% | 181.6 | 62494 | 82321 | 114.1 | 1774 | 0/20 |
| 热（满命中）| TCP | 100% | 126.5 | 42495 | 55981 | 86.5 | 2547 | 0/20 |
| 热（满命中）| **RDMA** | **100%** | **68.7** | **21914** | **28638** | **57.4** | **4688** | 0/20 |

- 缓存复用（RDMA 满命中 vs 冷遍）：Mean TTFT **−65%**，吞吐 **+164%**。
- RDMA vs TCP（同 100% 命中）：Mean TTFT **−48%**，吞吐 **+84%**（TCP 下 KV 加载带宽是瓶颈，
  换 RDMA 后命中收益才完全释放）。
- `chunk_size=16` 时每 chunk ≈4 MiB（旧 dingofs 上限），dfkv 正常存取——**任意块大小已验证**。

**纯传输对比（并发 1、output 1、100% 命中）**——去掉排队与 decode，单请求只测「从 dfkv 加载一个
16000-token prompt（≈4 GB KV）」的 TTFT：

| 传输 | Mean TTFT(ms) | Median TTFT(ms) | P99 TTFT(ms) | 时长(s) | 输入吞吐(tok/s) |
|---|---|---|---|---|---|
| TCP | 3842 | 4033 | 4059 | 76.9 | 4166 |
| **RDMA** | **1211** | **1259** | **1311** | **24.2** | **13212** |

即单请求 KV 加载 RDMA ≈1.2s vs TCP ≈3.8s，**TTFT −68%（约 3.2×）**，两者均 20/20 成功、100% 命中。

---

## 已知问题 / 排查

- **vLLM 在请求被中止时崩溃**：LMCache 0.4.5 + vLLM 0.21 在请求 `FINISHED_ABORTED` 时，
  scheduler 进程的 `vllm_v1_adapter.request_finished` 会 `assert self.lmcache_engine is not None`
  而崩溃。**与 dfkv 无关**（任何 remote backend 都会触发），正常完成的请求不走该路径；属上游已知问题。
- **启动日志没有 `DfkvConnector ready`**：检查 `lmcache.yaml` 的 `module_path/class_name/url`，
  以及 connector 是否装进了 vLLM 用的那个 Python 环境。
- **全部 miss / 连不上**：确认推理节点能连到缓存节点端口（TCP 用 `--port`，RDMA 用 `--rdma-port`）；
  RDMA 还要确认两端在同一 IB 网络、`DFKV_RDMA=1` 已设、跨网卡时 `DFKV_RDMA_DEV` 留空。
- **端口被占用**：换一组端口（`--port/--rdma-port` 与 yaml URL 同步改）。
</content>
</invoke>
