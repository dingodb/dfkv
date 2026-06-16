# dfkv LMCache connector — 使用指南

把 vLLM 的 KV cache 通过 LMCache 卸载到 **dfkv** 集群，实现跨请求/跨实例的前缀缓存复用。
本文按步骤说明怎么从零搭起来：**编译 → 起缓存集群 → 装 connector → 改 lmcache.yaml → 起 vLLM → 验证**。

设计与实现细节见 [DESIGN.md](DESIGN.md) / [IMPLEMENTATION.md](IMPLEMENTATION.md)。

---

## 0. 角色与前置条件

两类机器：

| 角色 | 跑什么 | 要求 |
|---|---|---|
| **推理节点** | vLLM + LMCache + dfkv connector | GPU；已装 vLLM、LMCache (≥0.4.5)；编译需 cmake≥3.20、gcc/g++ |
| **缓存节点**（≥1 台）| `dfkv_server` | 有数据盘（如 NVMe）；编译需 cmake、gcc/g++；走 RDMA 还需 `libibverbs-dev` |

网络要求：
- 推理节点必须能 TCP 连到每台缓存节点的 dfkv 端口。
- 若用 RDMA，推理节点与缓存节点要在**同一 IB/RoCE 网络**（否则用 TCP，见第 4 步）。

下文用占位符，请替换成你的实际值：
- `<CACHE1_IP>` `<CACHE2_IP>` … 缓存节点 IP
- `<DFKV_SRC>` dfkv 源码目录、`<LIBDFKV>` 编译出的 `libdfkv.so` 路径

---

## 第 1 步：编译 dfkv

推理节点和缓存节点都要编译（同一份源码，产物不同）。

```bash
git clone -b feat/lmcache_connector https://github.com/dingodb/dfkv.git
cd dfkv
# RDMA 版（推荐，自动带 TCP 回退）；只要 TCP 就把 ON 改成 OFF
cmake -S . -B build-rdma -DDFKV_WITH_RDMA=ON -DDFKV_BUILD_TESTS=OFF
cmake --build build-rdma -j"$(nproc)" --target dfkv dfkv_server
```

产物：
- `build-rdma/libdfkv.so` —— **推理节点** connector 通过 ctypes 加载的库。
- `build-rdma/dfkv_server` —— **缓存节点** 的缓存服务进程。

> 访问 github 需要代理时：`ALL_PROXY=socks5://<proxy_ip:port> git clone ...`。

---

## 第 2 步：在每台缓存节点启动 dfkv_server

每台缓存节点起一个 `dfkv_server`，指定数据盘目录、端口和容量：

```bash
# TCP + RDMA 都监听（RDMA 设备名用本机的，如 mlx5_0 / ib0；查 `ls /sys/class/infiniband`）
dfkv_server \
  --dir /data/disk0/dfkv,/data/disk1/dfkv \   # 逗号分隔多块盘
  --port 18800 \                              # TCP 端口
  --rdma-port 18801 \                         # RDMA 端口（不要 RDMA 就去掉这两行）
  --rdma-dev mlx5_0 \
  --cap 214748364800                          # 该节点总容量（字节），按盘量设
```

启动成功会打印 `PORT 18800` 并常驻。**记下每台节点的 `<IP>:<端口>`**，第 4 步要用：
- 走 TCP → 用 `--port`（如 `<CACHE1_IP>:18800`）
- 走 RDMA → 用 `--rdma-port`（如 `<CACHE1_IP>:18801`）

> 生产环境想让客户端自动发现节点增减，可改用 MDS 发现模式（`dfkv_mds` + etcd + `dfkv_server --mds ...`），
> 见根目录 [../DEPLOY.md](../DEPLOY.md)；本指南用更简单的**静态成员**模式。

---

## 第 3 步：在推理节点安装 connector

装到 vLLM 所在的 Python 环境（venv/conda）里：

```bash
source /path/to/your/vllm-venv/bin/activate
pip install <DFKV_SRC>/integration/lmcache       # 纯 Python wheel，不编译
export DFKV_LIB=<DFKV_SRC>/build-rdma/libdfkv.so  # 指向第 1 步编出的库
```

`dfkv-connector` 是纯 Python 包（通过 ctypes 调用 `libdfkv.so`），无需匹配 ABI。

---

## 第 4 步：写 LMCache 配置 `lmcache.yaml`

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
  remote_storage_plugin.dfkv.lib:         <DFKV_SRC>/build-rdma/libdfkv.so
```

**RDMA 版**：只改两点 —— URL 用 **RDMA 端口（18801）**，并在第 5 步给 vLLM 进程加 `export DFKV_RDMA=1`：

```yaml
  remote_storage_plugin.dfkv.url:         dfkv://c1=<CACHE1_IP>:18801,c2=<CACHE2_IP>:18801/g1
```

> 跨网卡（推理节点与缓存节点的 RDMA 设备名不同，如 a100 是 `ib6s200p0`、缓存节点是 `mlx5_0`）时，
> **不要**在推理节点设 `DFKV_RDMA_DEV`：客户端会用本机第一个设备、并让服务端用它自己的 `--rdma-dev`。

---

## 第 5 步：启动 vLLM

让 vLLM 用 LMCache 连接器，并指向上面的 yaml：

```bash
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CONFIG_FILE=/path/to/lmcache.yaml
export DFKV_LIB=<DFKV_SRC>/build-rdma/libdfkv.so
# 只有走 RDMA 才加这一行（TCP 不要加）：
export DFKV_RDMA=1

vllm serve <model_path> \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    --port 18200 --host 0.0.0.0 \
    # ... 你的其余 vllm 参数 ...
```

---

## 第 6 步：验证

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
| `DFKV_RDMA_DEV` | 空 | 客户端 RDMA 设备名；跨网卡时**留空** |
| `DFKV_CONNECTOR_GET_PARALLELISM` | 1 | 并行 batched-get 组数（提高可降 TTFT） |
| `DFKV_CONNECTOR_BATCH_MAX_KEYS` | 512 | 单次 native 批量最大 key 数 |
| `DFKV_ACCESS_LOG_ENABLED` / `_PATH` | 0 / stderr | 逐操作访问日志 |

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
