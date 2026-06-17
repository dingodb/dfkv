# dfkv 独立上线 runbook（与现网 dingo-cache 混部）

> 在 GPU 节点上**独立**起一套 dfkv KV 缓存集群供 SGLang HiCache 用，与现网
> `dingo-cache` 混部互不干扰。**对现网 dingofs/sglang 命名空间、dingo-cache 进程、
> MDS、对象存储一律不动（只读边界）**；dfkv 用独立端口/目录/进程。
> 前提：GLM-5.1 = MLA（每页 KV ≈ 2.74 MiB 单对象、跨 TP 复制、仅 tp_rank0 写）。
> 已在 400G InfiniBand 上端到端验证（两端零拷贝，单口 GET ~93% 线速）。

---

## 0. 网络模型（关键）

dfkv 把**控制面**与**数据面**解耦：

- **控制面 = TCP**：客户端用一份静态成员表 `name=ip:port` 连到各节点的 **bootstrap TCP 端口**，只交换 ~32B 的 QP 信息（LID/GID/QPN/PSN）。这条 IP 走任意两端可达的网（如 200G 存储网或管理网 bond0）。
- **数据面 = RDMA**：按**设备名**选 RDMA 口（`DFKV_RDMA_DEV=ib7s400p0`），数据走该 IB fabric，**无需该网有 IP**。因此 **400G 计算网（无 IP）与 200G 存储网（有 IP）即便是两张互不通的 IB 网也支持**：控制面走 200G、数据面走 400G。
- TCP 回退：未设 `DFKV_RDMA` 或无 RDMA 设备时自动用 TCP 传输（控制面那条连接直接当数据通道）。

发现：默认走 **MDS 动态发现**（etcd + dfkv_mds，见 §2b）；静态成员表仍作为遗留/单节点备用路径（见 §4-legacy）。无副本（一致性哈希单属主，节点挂 = 该分片 miss → 重算）。

---

## 1. 构建可移植产物（同架构构建机一次）

> **产物清单**（含 MDS 新增二进制）：`build/dfkv_server`、`build/dfkv_mds`、
> `build/libdfkv.so`、`build/dfkv_smoke`、`build/dfkvctl`、`build/dfkv_bench`。

```bash
git clone git@github.com:ketor/dfkv.git && cd dfkv
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DDFKV_WITH_RDMA=ON -DDFKV_STATIC_LIBSTDCXX=ON      # RDMA + 跨发行版可移植
cmake --build build -j
# 产物：build/dfkv_server  build/dfkv_mds  build/libdfkv.so
#        build/dfkv_smoke  build/dfkvctl  build/dfkv_bench
ldd build/dfkv_server | grep ibverbs                      # 确认链接了 RDMA 库
```
依赖：`libibverbs-dev`（构建期）+ 运行节点装 `rdma-core`。无 RDMA 也可去掉 `-DDFKV_WITH_RDMA` 构建纯 TCP 版。
> QP 信息走 TCP bootstrap 交换（非 librdmacm），所以只依赖 libibverbs，不需要 librdmacm。

> ⚠️ **glibc 下限 = 构建机的 glibc**。在新发行版（如 Ubuntu 24.04 / glibc 2.39）上构建的二进制，拿到老节点（glibc 2.35）会 `version GLIBC_2.3x not found` 起不来。
> **要部署到 glibc 2.35 的节点（如目标 GPU 节点），就在 glibc ≤ 2.35 的环境构建**（Ubuntu 22.04 / RHEL9）。仓库根的 `Dockerfile` 已固定 `ubuntu:22.04` + RDMA + 静态 libstdc++，**一次构建、glibc≥2.35 处处可跑**：
> ```bash
> docker build -t dfkv-build --target build . && id=$(docker create dfkv-build) && docker cp $id:/out ./dist && docker rm $id   # dist/bin/* dist/lib/libdfkv.so
> ```
> `DFKV_STATIC_LIBSTDCXX` 把 libstdc++/libgcc 静态进产物（这些不是 glibc）；libibverbs 无法静态（运行时 dlopen 驱动），运行节点仍需装 rdma-core。

## 2. 每节点：分发 + 缓存目录

```bash
install -m755 build/dfkv_server /usr/local/bin/dfkv_server
install -m755 build/libdfkv.so  /usr/local/lib/ && ldconfig
mkdir -p /mnt/disk1/dfkv /mnt/disk2/dfkv /mnt/disk3/dfkv   # 与现网 dingo-cache 子目录错开
```
容量隔离：`--cap`（总量，按盘均分）自带 LRU 自限；设保守值，确认 `现网用量 + dfkv cap + 预留 < 物理总量`。

## 2b. MDS 层：etcd + dfkv_mds

> **推荐路径**：用 MDS 动态发现时须先起 etcd 集群和 dfkv_mds 副本。

### 2b-1. etcd 集群

使用现有 etcd 或按官方文档起 1/3 节点集群（TTL 30 s 的 keepalive 流量极低）。
建议独立于 Kubernetes etcd，避免写放大干扰。

### 2b-2. dfkv_mds 副本（每个管理节点各一个）

```bash
install -m755 build/dfkv_mds /usr/local/bin/dfkv_mds
```

`/etc/systemd/system/dfkv_mds.service`：
```ini
[Unit]
Description=dfkv MDS (Membership Directory Service)
After=network-online.target
[Service]
Type=simple
# --listen: TCP 监听端口（与 dfkv_server 端口段错开）
# --etcd:   etcd 地址（默认 127.0.0.1:2379）
# --metrics-port: 可选，开 Prometheus /metrics（缺省=不开端口，见 §8 / docs/METRICS.md）
ExecStart=/usr/local/bin/dfkv_mds --listen 9400 --etcd 127.0.0.1:2379 --metrics-port 9410
Restart=on-failure
RestartSec=2
[Install]
WantedBy=multi-user.target
```
```bash
systemctl daemon-reload && systemctl enable --now dfkv_mds
journalctl -u dfkv_mds -n 5 --no-pager   # 应见 "dfkv_mds listening on 9400, etcd=..."
```

> **无状态**：dfkv_mds 自身不持久化任何数据，状态全在 etcd。重启/增减副本无需协调；
> 节点和客户端各持一份 MDS 端点列表，自动故障转移，无需负载均衡器。
> etcd member key 路径：`/dfkv/v1/groups/<group>/members/<id>`；epoch = etcd revision。

## 3. 每节点：systemd unit

`/etc/systemd/system/dfkv.service`：
```ini
[Unit]
Description=dfkv KV cache node (SGLang HiCache L3)
After=network-online.target
[Service]
Type=simple
# --port = TCP(bootstrap+TCP数据) 端口; --rdma-port = RDMA bootstrap 端口; --rdma-dev = 数据面 400G 口
# --metrics-port = 可选 Prometheus /metrics（缺省=不开端口；--id/--group 成为指标标签）
ExecStart=/usr/local/bin/dfkv_server \
  --dir /mnt/disk1/dfkv,/mnt/disk2/dfkv,/mnt/disk3/dfkv \
  --port 28000 --rdma-port 28001 --rdma-dev ib7s400p0 --cap 6597069766656 \
  --mds 10.0.0.1:9400,10.0.0.2:9400 --metrics-port 28010 \
  --group default --id n57 --advertise 192.168.1.57:28001
Restart=on-failure
RestartSec=2
CPUQuota=1600%
MemoryMax=64G
Nice=5
LimitNOFILE=1048576
LimitMEMLOCK=infinity        # RDMA 需要锁页内存
[Install]
WantedBy=multi-user.target
```
```bash
systemctl daemon-reload && systemctl enable --now dfkv
journalctl -u dfkv -n 10 --no-pager
# 应见 "PORT 28000" + "RDMA listening (TCP bootstrap) on port 28001, dev=ib7s400p0"
#      + "dfkv_server registered with MDS group=default id=n57 advertise=192.168.1.57:28001"
```
> server 的 bootstrap 监听 `0.0.0.0`，靠防火墙限制在内网。优雅关闭已修（`systemctl stop` 约 1s 退出）。
> **多轨**：让客户端在多张 400G 口间分散即可（见 §5 `DFKV_RDMA_DEV` 列表）；server 会按客户端请求的设备名在同轨开 QP，无需为多轨改 server 配置（保留 `--rdma-dev` 作默认）。
> ⚠️ 同机 8×400G 多轨受 NUMA 限制（NIC 跨双 socket，单内存域）；单口已近线速，多轨叠加需 NUMA 感知，暂不必配。

## 4. 集群成员管理

### 4a. MDS 动态发现（推荐）

节点通过 `--mds` + `--group` + `--id` + `--advertise` 向 MDS 自注册，客户端调用
`dfkv_start_mds_discovery(c, "mds1:9400,mds2:9400", "default", 3000)` 周期轮询
MDS（默认间隔 3000 ms）。epoch（etcd revision）变化时自动重建加权 Ketama 环。
增减节点只需启停 `dfkv_server`，无需改客户端配置。

两层离线检测：
- **层 2（权威）**：etcd lease 到期 → MDS 视图变更 → 客户端 epoch 推进 → 环重建（≤ 30 s）
- **层 1（快速）**：`PeerHealth` 传输 IO 失败即短路该节点为 miss 并进入 cooldown，不触发环重建

### 4b. 静态成员表（遗留/单节点备用）<a name="4-legacy"></a>

适用于无 etcd 的简单环境或单节点调试。成员 = 所有节点的
`name=<bootstrap-ip>:<rdma-port>`（RDMA）或 `:<port>`（TCP）：
```
n57=192.168.1.57:28001,n58=192.168.1.58:28001,...
```
增减节点 = 改成员字符串并重载 SGLang 端。`dfkv_server` 此时不带 `--mds` 启动。
建议 N≥4 降低单点 miss 影响。

## 5. SGLang 侧接入（免 fork，dynamic 侧载）

把 `libdfkv.so` + `python/dfkv_hicache.py` 放到 pod 可访问路径，启动 `sglang serve` 前注入环境：
```bash
export PYTHONPATH=/userdata/dfkv:$PYTHONPATH
export DFKV_LIB=/userdata/dfkv/libdfkv.so
export DFKV_RDMA=1                       # 启用 RDMA 数据面（否则 TCP）
export DFKV_REQUIRE_RDMA=1               # 可选：禁止悄悄 TCP fallback
export DFKV_RDMA_DEV=ib7s400p0           # 数据面设备；多轨用逗号列表 ib7s400p0,ib7s400p1,...
export DFKV_RDMA_MAX_PAYLOAD_BYTES=67108864  # 可选：单 chunk payload 上限，默认 64MiB
# 可选: DFKV_RDMA_DEPTH=16 (单连接 pipeline, K 个请求在途; 利于 PUT; client+server 两端都要设)
```
> 写流水也可写进 extra_config 的 `"rdma_depth":16`（插件在 dfkv_open 前自动设 DFKV_RDMA_DEPTH，免去手动 export）；**dfkv_server 仍须在自己的 env 里设同值或更大**（client depth ≤ server depth）。MLA 单 rank 写是延迟受限，depth>1 让多个 PUT 在途、隐藏单 op 延迟，实测抬高写带宽。

> **多轨 NUMA 选轨**（v1.2.0 起）：设 extra_config `"rdma_numa":1`（或 env `DFKV_RDMA_NUMA=1`）+ 多轨 `DFKV_RDMA_DEV`，**C++ 客户端在建连时按调用线程的 NUMA 节点自动选本地轨**（无本地轨→轮转全轨）。SGLang/vLLM 两端通吃，无需各自改，仅在新建连接触发（热路径零开销）。
> - ⚠️ **旧 `"rail_affinity"` 已废弃为 no-op**（v1.2.0）：它按 `tp_rank` 收窄，但 DP-attention 下每 rank `tp_rank=0`→塌缩到单轨。配了只打 stderr 告警、不再生效；改用上面的 `rdma_numa`。

**方案 A — MDS 动态发现（推荐）**：配置 `mds_endpoints` + `mds_group`；插件内部调用
`dfkv_start_mds_discovery` 自动轮询 MDS，epoch 变化时重建环，无需重启。
```bash
sglang serve ... \
  --enable-hierarchical-cache --hicache-write-policy write_through \
  --hicache-mem-layout page_first_direct --hicache-io-backend direct \
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
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{
    "backend_name":"dfkv","module_path":"dfkv_hicache","class_name":"DfkvHiCache",
    "interface_v1":1,
    "members":"n57=192.168.1.57:28001,n58=192.168.1.58:28001",
    "model_hash": 81, "page_size":64, "dtype_tag":1178092852,
    "layer_num":78, "head_num":1, "head_dim":576 }'
```
> **`interface_v1:1` 是必填项**，插件 `__init__` 强校验：缺失即 `raise ValueError` 启动失败。原因——对 `dynamic` 后端，SGLang 仅在 `interface_v1` 为真时才走零拷贝 `batch_set_v1/get_v1`；否则退回 generic `set/get` 路径，而 dfkv 的 generic `get/batch_get` 是未实现的桩，会导致**写成功、L3 读静默失败**（线上踩过：launch 脚本漏配此项，14GB 写入但 prefetch 全部 miss）。
> `interface_v1:1` 触发零拷贝 `batch_set_v1/get_v1` —— GET payload 经 RDMA 散射**直落 HiCache 宿主页**（client 端零拷贝），server 端 O_DIRECT 直读入已注册 direct buffer 并 scatter-send（server 端无 payload memcpy），两端零拷贝。
> MLA 下插件自动单对象、无 rank 后缀、`backup_skip`（仅 tp_rank0 写）。decode 共享前缀配同 members。
> 多池模型（Mamba/SWA/DeepSeek-V4）用 v2 PoolTransfer 接口（插件已实现）。
> 排查命中率/慢操作：可开启 access log（`access_log`/`access_log_path` 或 `DFKV_ACCESS_LOG_*`），见 [access_log.md](hicache/access_log.md)。
> **客户端指标**：插件自动在 SGLang 自带的 `/metrics` 上暴露 `dfkv_client_*{tp_rank}`（set/get 量、命中、IO 错误、peer 熔断切换、set/get 延迟直方图）。后台轮询线程读 C 客户端快照，间隔由 extra_config `client_stats_poll_s`（默认 10s，`DFKV_CLIENT_STATS_POLL_S` 兜底，`0`=关）控制。全指标见 [METRICS.md](METRICS.md) §3.3。

## 6. 上线顺序 + 冒烟（无需 GPU）

1. （MDS 路径）起 etcd，再起所有 `dfkv_mds` 副本（§2b），确认日志 "listening"。
2. 所有节点起 `dfkv_server`（§3），确认 PORT + "registered with MDS" 日志。
3. 冒烟（任一能访问内网的机器）：
   ```bash
   dfkv_smoke --members n57=192.168.1.57:28000 --size 2752512                          # TCP
   DFKV_RDMA=1 DFKV_RDMA_DEV=ib7s400p0 dfkv_smoke --members n57=192.168.1.57:28001 --size 2752512  # RDMA 400G
   ```
4. 端到端零拷贝校验（插件 → libdfkv → RDMA → server，验证 payload 直落缓冲）：
   ```bash
   DFKV_RDMA=1 DFKV_RDMA_DEV=ib7s400p0 DFKV_MEMBERS='n=192.168.1.57:28001' \
     python3 tests/python/rdma_e2e_validate.py    # 期望 RESULT: ZERO-COPY RDMA E2E OK
   ```
5. 压测（可选）：`DFKV_RDMA=1 DFKV_RDMA_DEV=ib7s400p0 dfkv_bench --members ... --size 2752512 --count 8000 --threads 64`。
6. 在**一个受控 SGLang 副本**上切 `dynamic` 后端，发共享长前缀请求看命中上涨，确认后推广。

## 7. 回滚（秒级）
- SGLang：`--hicache-storage-backend` 改回原后端（mooncake 等）重启该副本，与 dfkv 解耦。
- dfkv 节点：`systemctl stop dfkv`；缓存可丢（KV 可重算），彻底清理删 `/mnt/diskX/dfkv`。
- dfkv MDS：`systemctl stop dfkv_mds`；无状态，etcd 数据可保留也可清除（`etcdctl del /dfkv --prefix`）。
- 全程不影响 dingo-cache / dingofs / 生产 MDS / 对象存储。

## 8. 监控 / 边界
**完整指标/CLI 参考见 [METRICS.md](METRICS.md)。** 要点：
- **Prometheus 抓取**（opt-in）：`dfkv_server`/`dfkv_mds` 加 `--metrics-port <p>` → `GET /metrics`、`/healthz`。`--id/--group` 成为 `{node,group}` 标签（不设=无标签，向后兼容）。**缺省不开端口 → 行为与旧版一致、对数据面零影响**。Prometheus 直接抓每节点 `:<p>/metrics`。
  - 服务端含：put/hit/miss、bytes、淘汰、错误分型、`open_connections`、per-disk、**采样延迟直方图 `dfkv_op_latency_seconds{op}`**、RDMA 完成/错误/活跃连接。
  - MDS 含：register/keepalive/list/lease/etcd-error + members gauge。
  - 客户端（SGLang 插件 `/metrics`）：`dfkv_client_*{tp_rank}`，见 §5 / METRICS.md §3.3。
- **集群/环视图**（CLI，无需开端口）：
  - `dfkvctl ring --mds <eps> --group <g>` — 成员表 + 一致性哈希环每节点 vnode 占比。
  - `dfkvctl stat --all --mds <eps> --group <g>` — 逐节点指标 + 集群聚合（容量/对象/命中率）。
  - `dfkvctl stat <ip:port>` — 单节点原始 Prometheus 文本（旧用法不变）。
- SGLang `--enable-cache-report` 的 HiCache storage hit/miss、TTFT。
- 生产只读：不改现网组件；dfkv 端口（含 `/metrics`）仅内网开放、无鉴权勿暴露公网。
- 线协议带 1B 版本号，混版本部署会被 server 拒（不静默错读）；升级时整集群同版本。**指标均为新增、不改 wire，v1.3.0 节点与 v1.4.0 客户端互通。**
