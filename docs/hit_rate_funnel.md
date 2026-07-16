# dfkv L3 命中率漏斗（三层口径）

> "命中率不够"几乎从来不是一个数字。dfkv 的 L3 复用要拆成三层看，每层有各自的失败模式、各自的指标、各自的修法。一份 GLM-5.2 压测反馈"命中率不够"，实测第一层命中率 **99.8%**——问题在下面两层。本文给出每层的权威指标名与判读方法，让"感觉"变成仪表盘。

## 三层漏斗

```
请求 ──► ① 存在性命中 (exist) ──► ② 预取完成 (fetch) ──► ③ 转化收益 (net)
         "服务端有这页吗"          "拉回来了吗"           "比重算更快吗"
```

一页 KV 只有走完三层才真正省下算力。任何一层塌陷，"命中率"就"感觉不够"，但根因和修法完全不同。

---

## ① 存在性命中率 —— "服务端有这页吗"

**指标**（服务端 `:<metrics_port>/metrics`）：
- `dfkv_exist_hit_total` / `dfkv_exist_miss_total` → 命中率 = hit / (hit + miss)
- `dfkv_cache_hit_total`（按 key 的读命中）、`dfkv_objects`、`dfkv_used_bytes`

**健康态**：稳态热工作集下应 ≈ 100%（实测清环 99.8%）。

**失败模式与判读**：
- **环写满 → 命中率断崖归零**：`dfkv_used_bytes / cap > ~0.95` 且 `dfkv_evictions_total` 与写入量同步暴涨 = 满环自噬（刚写的热页被逐出）。**修**：容量水位（`DFKV_SLAB_EVICT_HIGH_PCT`，看 `dfkv_slab_watermark_evictions_total` 是否在动）+ 别让环写满。
- **keyspace 隔离/串味**：命中率异常低但环不满 → 查 `model_name`/`model_hash` 是否一致（dfkv key 前缀 = model_name）。

---

## ② 预取完成率 —— "拉回来了吗"

exist 命中 **≠** 数据真被拉回。SGLang HiCache 的 `prefetch-policy=timeout` 会在预取跑不赢时把它砍掉转重算。

**指标**：
- 客户端 access log（`access_log=1`）：`batch_get_auto_sg(N keys) hits=N/N`、`batch_exists prefix=X/N`
- 服务端读回量：`dfkv_bytes_read_total` 增量 vs 工作集应拉回的字节。远小于应拉回量 = 大量预取被 timeout 砍掉。
- 客户端会合（**本期新增遥测**）：`dfkv_connector_dedup_hits_total` / `_fetches_total` / `_wait_hits_total` / `_wait_timeouts_total`（GPU 目标另有 `dfkv_connector_gpu_dedup_*`）。

**判读**：
- `dedup_hits + wait_hits` 占比高 = 同机 TP rank 的锁步读被会合到 1×（好）；`fetches` 接近请求数 = 会合没生效，8× 读放大在白吃带宽 → 确认 `DFKV_CLIENT_NODE_DEDUP=1`（MLA+TP>1 现默认开）。
- `wait_timeouts` 高 = 会合等待超时回退直连 → 读回带宽跟不上，见第③层。

---

## ③ 转化收益 —— "比重算更快吗"

预取完成率高也可能**负收益**：读回速率必须 > 重算速率，L3 才划算。

**指标**：
- 冷/热对照（**必须分开报，禁止三轮取平均**）：同一批 prompt，R1 冷（写 L3）/ R2·R3 热（读 L3）。热轮吞吐 > 冷轮 = 正收益。
- 服务端读回速率：`dfkv_bytes_read_total` 增量 / 墙钟。对照单节点纯读上限（B200 单节点实测 ~17GB/s）。

**判读与修法**：
- **热轮 < 冷轮**：读回带宽 < 重算速率。单节点物理限制（一个节点的盘+NIC < B200 prefill）。**修**：多节点环（聚合读带宽随节点线性增长；五节点环 C10 实测 +77%），或改善推理侧 prefetch 与 decode 的 overlap（SGLang 侧，非 dfkv）。
- **重备份风暴**：热轮 `dfkv_bytes_written_total` 仍大 = 重算页被无谓重写（已由 hicache backup exist 门根治，`DFKV_BACKUP_EXIST_GATE=1` 默认）。
- **exist 队头阻塞**：热轮 `batch_exists` 尾延迟爆（access log），lookup 排在 1MB GET 后面 → `rdma_depth` 提到 32 + 控制 lane（已默认）。

---

## 一页速查

| 症状 | 层 | 首查指标 | 首选修法 |
|---|---|---|---|
| exist 命中率低、环快满 | ① | `used_bytes/cap`、`watermark_evictions` | 容量水位 + 别写满 |
| exist 命中率低、环不满 | ① | `model_name`/`model_hash` | keyspace 对齐 |
| exist 高但读回量小 | ② | `dedup_*`、`bytes_read` | 开 dedup、查 prefetch timeout |
| 热轮慢于冷轮 | ③ | 冷/热吞吐、读回 GB/s | 多节点、prefetch overlap |
| 热轮仍大量写入 | ③ | `bytes_written` 热轮增量 | backup exist 门 |
| lookup 尾延迟爆 | ②/③ | access log `batch_exists` p99 | rdma_depth=32、控制 lane |

**铁律**：报"命中率"必须指明是哪一层。三个数字，三种病，三种药。
