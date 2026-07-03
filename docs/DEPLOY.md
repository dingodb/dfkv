# dfkv 独立上线 runbook（与现网 dingo-cache 混部）

> 在 GPU 节点上**独立**起一套 dfkv KV 缓存集群供 SGLang HiCache 用，与现网
> `dingo-cache` 混部互不干扰。**对现网 dingofs/sglang 命名空间、dingo-cache 进程、
> MDS、对象存储一律不动（只读边界）**；dfkv 用独立端口/目录/进程。
> 前提：GLM-5.1 = MLA（每页 KV ≈ 2.74 MiB 单对象、跨 TP 复制、仅 tp_rank0 写）。
> 已在 400G InfiniBand 上端到端验证（两端零拷贝，单口 GET ~93% 线速）。

> **本文只讲 dfkv 集群自身的部署;各推理引擎如何对接/配置 dfkv 见 [docs/hicache/DEPLOY.md](hicache/DEPLOY.md)、[docs/vllm/DEPLOY.md](vllm/DEPLOY.md)、[docs/lmcache/DEPLOY.md](lmcache/DEPLOY.md)。**

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

> **可选存储/加速开关（均默认关，向后兼容；见 [ARCHITECTURE.md](ARCHITECTURE.md) §5–7）**
> - `--store-engine=file|slab`（默认 `file`）：`slab` = extent 池 + slots.tbl 重启保温，消除"每块一文件"隐患。**切 slab 需清盘冷启**（旧 blocks/ 布局不复用），属独立迁移动作。
> - `Environment=DFKV_RAM_TIER=1`（默认关）：开写直通 RAM 热层；`DFKV_RAM_TIER_BYTES=<bytes>` 定 arena 大小（预注册即 pin 内存，须核 `MemoryMax` 有余量）。开后观测 `dfkv_ram_hit_total` / `dfkv_ram_put_bypass_total`（见 METRICS.md §3.1）。RDMA 构建下 RAM 命中走 arena MR 零拷贝直发。
> 滚动升级 v1.7.0 建议：**先全默认滚一遍**（等价现网）；RAM 热层、slab 引擎再各自单节点灰度评估。
```bash
systemctl daemon-reload && systemctl enable --now dfkv
journalctl -u dfkv -n 10 --no-pager
# 应见 "PORT 28000" + "RDMA listening (TCP bootstrap) on port 28001, dev=ib7s400p0"
#      + "dfkv_server registered with MDS group=default id=n57 advertise=192.168.1.57:28001"
```
> server 的 bootstrap 监听 `0.0.0.0`，靠防火墙限制在内网。优雅关闭已修（`systemctl stop` 约 1s 退出）。
> **多轨**：让客户端在多张 400G 口间分散即可（客户端 `DFKV_RDMA_DEV` 逗号列表，见各引擎对接文档如 [hicache/DEPLOY.md](hicache/DEPLOY.md)）；server 会按客户端请求的设备名在同轨开 QP，无需为多轨改 server 配置（保留 `--rdma-dev` 作默认）。
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

## 5. 上线顺序 + 冒烟（无需 GPU）

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
     python3 test/python/rdma_e2e_validate.py    # 期望 RESULT: ZERO-COPY RDMA E2E OK
   ```
5. 压测（可选）：`DFKV_RDMA=1 DFKV_RDMA_DEV=ib7s400p0 dfkv_bench --members ... --size 2752512 --count 8000 --threads 64`。
6. 在**一个受控 SGLang 副本**上切 `dynamic` 后端，发共享长前缀请求看命中上涨，确认后推广。

## 6. 回滚（秒级）
- SGLang：`--hicache-storage-backend` 改回原后端（mooncake 等）重启该副本，与 dfkv 解耦。
- dfkv 节点：`systemctl stop dfkv`；缓存可丢（KV 可重算），彻底清理删 `/mnt/diskX/dfkv`。
- dfkv MDS：`systemctl stop dfkv_mds`；无状态，etcd 数据可保留也可清除（`etcdctl del /dfkv --prefix`）。
- 全程不影响 dingo-cache / dingofs / 生产 MDS / 对象存储。

## 7. 监控 / 边界
**完整指标/CLI 参考见 [METRICS.md](METRICS.md)。** 要点：
- **Prometheus 抓取**（opt-in）：`dfkv_server`/`dfkv_mds` 加 `--metrics-port <p>` → `GET /metrics`、`/healthz`。`--id/--group` 成为 `{node,group}` 标签（不设=无标签，向后兼容）。**缺省不开端口 → 行为与旧版一致、对数据面零影响**。Prometheus 直接抓每节点 `:<p>/metrics`。
  - 服务端含：put/hit/miss、bytes、淘汰、错误分型、`open_connections`、per-disk、**采样延迟直方图 `dfkv_op_latency_seconds{op}`**、RDMA 完成/错误/活跃连接。
  - MDS 含：register/keepalive/list/lease/etcd-error + members gauge。
  - 客户端（SGLang 插件 `/metrics`）：`dfkv_client_*{tp_rank}`，见 [hicache/DEPLOY.md](hicache/DEPLOY.md) / METRICS.md §3.3。
- **集群/环视图**（CLI，无需开端口）：
  - `dfkvctl ring --mds <eps> --group <g>` — 成员表 + 一致性哈希环每节点 vnode 占比。
  - `dfkvctl stat --all --mds <eps> --group <g>` — 逐节点指标 + 集群聚合（容量/对象/命中率）。
  - `dfkvctl stat <ip:port>` — 单节点原始 Prometheus 文本（旧用法不变）。
- SGLang `--enable-cache-report` 的 HiCache storage hit/miss、TTFT。
- 生产只读：不改现网组件；dfkv 端口（含 `/metrics`）仅内网开放、无鉴权勿暴露公网。
- 线协议带 1B 版本号，混版本部署会被 server 拒（不静默错读）；升级时整集群同版本。**指标均为新增、不改 wire，v1.3.0 节点与 v1.4.0 客户端互通。**
