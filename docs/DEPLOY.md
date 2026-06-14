# dfkv 快速独立上线（与现网 dingo-cache 混部）— 可执行部署说明

> 目标：在 hd03 GPU 节点上**独立**起一套 dfkv KV 缓存集群，供 SGLang HiCache 用，
> 与现网 `dingo-cache`（dingofs-group）**混部互不干扰**。
> ⚠️ 本文是给运维按步骤执行的 runbook。**对现网 dingofs/sglang 命名空间、dingo-cache
> 进程、MDS、对象存储一律不动（只读边界）**；dfkv 用独立端口/目录/进程。
> 适用前提：GLM-5.1 = MLA（每页 KV ≈ 2.74 MiB 单对象，详见《容量与性能定量测算》）。

---

## 0. 拓扑与端口规划
```
每个 GPU 节点（hd03-gpu2-0057 / 0059 / …）：
  现网 dingo-cache : 192.168.1.<node>:11000  group=dingofs-group  /mnt/disk{1,2,3}/dingofs/dingo-cache   ← 不动
  新增 dfkv_server : 192.168.1.<node>:12000  --dir=/mnt/disk{1,2,3}/dfkv  --cap=<N>                       ← 本次新增
网络：走 200G 存储网 IPoIB（ib6s200p0，192.168.1.x/21），与 400G 计算/RDMA 网隔离
发现：dfkv 用【静态成员表】，不连 MDS；成员 = 所有节点的 :12000
```

## 1. 构建可移植产物（在一台同架构构建机上做一次）
dfkv 仅依赖 libc/libstdc++/pthread，产物小、可移植。为跨发行版稳妥，静态链接 libstdc++/libgcc：
```bash
git clone -b feat/kvcache-sglang git@github.com:ketor/dingofs.git
cd dingofs && git submodule update --init proto    # 仅需 proto（dfkv 内核不依赖重型子模块）
cmake -S /path/to/dfkv-harness -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
      -DCMAKE_SHARED_LINKER_FLAGS="-static-libstdc++ -static-libgcc"
cmake --build build -j --target dfkv_server dfkv
# 产物：build/dfkv_server  build/libdfkv.so
ldd build/dfkv_server   # 确认只剩 libc/libpthread/libm 等基础库
```
> dfkv-harness = 仓库里 `/home/ketor/dfkv-dev` 那个 CMakeLists（只编 `src/cache/kvclient` 的可移植核），不需要全量 dingofs/gcc-13 工具链。

## 2. 每节点：分发 + 目录 + 容量隔离
```bash
# 分发二进制
install -m755 dfkv_server /usr/local/bin/dfkv_server
install -m755 libdfkv.so  /usr/local/lib/libdfkv.so && ldconfig
# 独立缓存目录（与 dingofs-group 的 dingo-cache 子目录错开）
mkdir -p /mnt/disk1/dfkv /mnt/disk2/dfkv /mnt/disk3/dfkv
```
**容量隔离（非侵入式，首选）**：dfkv 的 `KVStore` 按 `--cap` 自带 LRU 自限，不会超过 `--cap`。
设 `--cap` 为保守值（例：3 盘共 **6 TiB**），并确认 `dingofs-group 实际用量 + dfkv cap + 预留 < 物理总量`
（hd03：单盘 3.84TB×3，dingofs-group 实际仅 ~150GB/盘，留足）。
**硬隔离（可选加固，需停机窗口）**：xfs project quota（注意 xfs 的 `prjquota` 必须在挂载时启用，
给 `/mnt/diskX` 加 quota 需 remount，会影响 dingofs-group，**非紧急不做**；先用 `--cap` 上线）。

## 3. 每节点：systemd unit（多盘 + 资源限额）
`/etc/systemd/system/dfkv.service`：
```ini
[Unit]
Description=dfkv KV cache node (SGLang HiCache)
After=network-online.target

[Service]
Type=simple
# <NODE_IPOIB> = 本机 200G 存储网 IPoIB 地址（ib6s200p0，如 192.168.1.57）
ExecStart=/usr/local/bin/dfkv_server \
  --dir /mnt/disk1/dfkv,/mnt/disk2/dfkv,/mnt/disk3/dfkv \
  --port 12000 --cap 6597069766656
Restart=on-failure
RestartSec=2
# 资源限额：别和推理抢核/内存（按节点规格调）
CPUQuota=800%
MemoryMax=32G
Nice=5
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
```
```bash
systemctl daemon-reload && systemctl enable --now dfkv
systemctl status dfkv ; journalctl -u dfkv -n 20 --no-pager   # 应看到 "PORT 12000"
```
> `dfkv_server` 监听 `0.0.0.0:12000`（如需绑定特定 IPoIB 网卡，后续可加 `--listen` 选项；当前监听全部地址，靠安全组/防火墙限制在存储网内访问）。

## 4. 集群成员表（静态）
dfkv 客户端（SGLang 插件）用一份静态成员表 = 所有节点的 `name=ipoib:12000`，例：
```
n57=192.168.1.57:12000,n59=192.168.1.59:12000
```
增减节点 = 改这份成员表并重启/重载 SGLang 端（无 MDS 动态发现；这是"快速独立"的取舍）。

## 5. SGLang 侧接入（免 fork，dynamic 侧载）
把插件与库放上 prefill/decode pod 能访问的 hostPath（例 `/alayanew-k8s/mengsz/dfkv/`）：
```
dfkv/libdfkv.so
dfkv/dingofs_hicache.py        # 来自 src/cache/kvclient/python/
```
在**启动 `sglang serve` 之前**注入环境并加后端参数（GLM-5.1 = MLA）：
```bash
export PYTHONPATH=/userdata/dfkv:$PYTHONPATH
export DFKV_LIB=/userdata/dfkv/libdfkv.so
sglang serve ... \
  --enable-hierarchical-cache --hicache-write-policy write_through \
  --hicache-mem-layout page_first_direct --hicache-io-backend direct \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{
    "backend_name":"dingofs","module_path":"dingofs_hicache","class_name":"DingoFSHiCache",
    "interface_v1":1,
    "members":"n57=192.168.1.57:12000,n59=192.168.1.59:12000",
    "model_hash": 81, "page_size":64, "dtype_tag":1178092852,
    "layer_num":78, "head_num":1, "head_dim":576 }'
```
> `interface_v1:1` 触发零拷贝 `batch_set_v1/get_v1`（已核 `cache_controller.py:530-534`）。
> MLA 下插件自动单对象、无 rank 后缀、`backup_skip`（仅 tp_rank0 写）。decode 想共享前缀也配同样后端+同 members。

## 6. 上线顺序 + 冒烟（KV 路径无需 GPU 即可验证）
1. 先在所有节点起 `dfkv_server`（第 3 步），`journalctl` 确认 PORT。
2. 冒烟（任一能访问存储网的机器，用 libdfkv + ctypes，或下面的 dfkv_smoke）：
   ```python
   import ctypes
   lib=ctypes.CDLL("/usr/local/lib/libdfkv.so")
   lib.dfkv_open.restype=ctypes.c_void_p
   lib.dfkv_open.argtypes=[ctypes.c_char_p]+[ctypes.c_uint64]+[ctypes.c_uint32]*8
   lib.dfkv_put.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_void_p,ctypes.c_uint64]
   lib.dfkv_get.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_void_p,ctypes.c_uint64]; lib.dfkv_get.restype=ctypes.c_int
   h=lib.dfkv_open(b"n57=192.168.1.57:12000,n59=192.168.1.59:12000",81,64,1178686260,1,8,0,78,1,576)
   v=b"x"*4096; assert lib.dfkv_put(h,b"smoke_k",v,len(v))==0
   o=ctypes.create_string_buffer(4096); assert lib.dfkv_get(h,b"smoke_k",o,4096)==1 and o.raw==v
   print("dfkv cluster smoke OK")
   ```
3. 在**一个受控 SGLang 副本**上切 `--hicache-storage-backend dynamic`（第 5 步），发两条共享长前缀请求，看 dfkv 命中（`--enable-cache-report` 的 storage hit 上涨）。确认无误再推广。

## 7. 回滚（秒级）
- SGLang 侧：把 `--hicache-storage-backend` 改回 `mooncake`（或去掉 dynamic 配置）重启该副本即可，**与 dfkv 解耦**。
- dfkv 侧：`systemctl stop dfkv`；缓存数据可丢（KV 可重算），需要彻底清理则删 `/mnt/diskX/dfkv`。
- 全程不影响 dingo-cache / dingofs / MDS / 对象存储。

## 8. 监控（最小）
- `dfkv_server` 进程存活（systemd）、端口可达、各 `/mnt/diskX/dfkv` 用量（`du`/`df`）。
- SGLang `--enable-cache-report` 的 HiCache storage hit/miss 率、TTFT。
- 告警：节点 down（成员表里少一台 = 该分片 miss→重算，非致命；N 越大单点影响越小，建议 N≥4）。

## 9. 安全/边界
- 生产只读：本方案不改 dingo-cache/MDS/对象存储/现网 sglang spec；dfkv 是新增独立进程。
- dfkv 端口仅在存储网内开放（防火墙/安全组限制）；无鉴权，勿暴露公网。
- 与 dingo-cache 共盘：靠 `--cap` 自限 + 容量红线；如需硬隔离再上 xfs quota（停机窗口）。
