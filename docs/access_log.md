# dfkv HiCache access log（访问日志）

给 `DfkvHiCache` 插件的**每个继承接口**逐次操作打一行日志，用于排查命中率、慢操作、失败。
默认**关闭**，开启后近乎零热路径开销（异步落盘）。格式对齐 dingofs lmcache / vfs 的 access log：

```
<时间>.<毫秒> <op>(<参数>) : <结果> <耗时秒>
```

实测样例（一个 TP rank 的文件，含 `init` 与各数据面接口）：

```
2026-06-15 20:16:16.727 init(r0 glm-5.1 tp=0/8 mla=1) : ok static <0.000461>
2026-06-15 20:16:16.734 batch_set_v1(r0 3 keys) : ok 3/3 <0.005716>
2026-06-15 20:16:16.734 batch_get_v1(r0 3 keys) : hits=3/3 <0.000392>
2026-06-15 20:16:16.734 batch_exists(r0 3 keys) : prefix=2/3 <0.000110>
2026-06-15 20:16:16.735 set(r0 g0, 4.00KiB) : ok <0.000271>
2026-06-15 20:16:16.735 get(r0 g0, 4.00KiB) : hit <0.000153>
2026-06-15 20:16:16.735 get(r0 g_missing, 4.00KiB) : miss <0.000026>
2026-06-15 20:16:16.735 exists(r0 g0) : found <0.000036>
2026-06-15 20:16:16.735 exists(r0 g_missing) : not_found <0.000034>
```

每行 `<参数>` 都以 `r{tp_rank}` 开头，合并多 rank 文件查看时也能区分来源。

## 配置项

三个开关。**优先级：`extra_config` 键 > 环境变量 > 默认值**（按每个开关独立解析）。

| 作用 | `extra_config` 键 | 环境变量 | 默认 |
|---|---|---|---|
| 开 / 关 | `access_log` | `DFKV_ACCESS_LOG_ENABLED` | 关 |
| 日志路径 | `access_log_path` | `DFKV_ACCESS_LOG_PATH` | 空 → 写 stderr |
| 慢操作阈值（微秒） | `access_log_threshold_us` | `DFKV_ACCESS_LOG_THRESHOLD_US` | `0`（全记） |

- 开关取值接受 `1/true/yes/on`（大小写不敏感）或 JSON 里的 `1`/布尔。
- `access_log_threshold_us` 设成比如 `1000`，则只记录耗时 ≥ 1ms 的操作，排查长尾很方便。
- 配置在插件 `__init__` 里**运行时解析一次**（每进程首个实例生效，幂等）；不像 dingofs 在 import 时读 env。

## 使用方式

### 方式 A — SGLang 启动 JSON（推荐，贴合 dfkv 现有 `extra_config` 约定）

在 [DEPLOY.md](DEPLOY.md) 的 `--hicache-storage-backend-extra-config` JSON 里加 access_log 三个键：

```bash
sglang serve ... \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{
    "backend_name":"dfkv","module_path":"dfkv_hicache","class_name":"DfkvHiCache",
    "interface_v1":1,
    "mds_endpoints":"10.0.0.1:9400,10.0.0.2:9400","mds_group":"default",
    "model_hash":81, "page_size":64, "dtype_tag":1178092852,
    "layer_num":78, "head_num":1, "head_dim":576,
    "access_log":1,
    "access_log_path":"/var/log/dfkv/access.log",
    "access_log_threshold_us":0 }'
```

### 方式 B — 环境变量（不想改启动 JSON 时，运维临时开启）

```bash
DFKV_ACCESS_LOG_ENABLED=1 \
DFKV_ACCESS_LOG_PATH=/var/log/dfkv/access.log \
DFKV_ACCESS_LOG_THRESHOLD_US=0 \
sglang serve ...
```

> 路径目录需提前存在且可写；打开失败会打印一行告警并**回退到 stderr**，不影响服务。

## 路径与多 TP rank

SGLang 每个 TP rank 一个进程，会**同时写同一路径**。为避免互相覆盖/交错，路径按以下规则处理：

- 路径含 `{rank}` / `{model}` 占位符 → 直接替换。
  例：`/var/log/dfkv/access.{rank}.log` → `access.0.log`、`access.1.log` …
- 路径**不含**占位符 → 自动追加 `.r{rank}` 后缀。
  例：`access.log` → `access.log.r0`、`access.log.r1` …

所以上面方式 A/B 配 `access.log`，rank 0 的实际文件是 `access.log.r0`。

## 日志格式与各 op 的结果含义

`<op>(<参数>) : <结果> <耗时秒>`。覆盖的接口与对应 `<结果>`：

| op | 触发时机 | `<结果>` 取值 |
|---|---|---|
| `init` | 插件构造（打开连接 + MDS 发现） | `ok static` / `ok mds-discovery` |
| `batch_set_v1` | 零拷贝写（生产热路径） | `ok H/N`；MLA 非 0 号 rank 为 `backup_skip` |
| `batch_get_v1` | 零拷贝读（生产热路径） | `hits=H/N` |
| `batch_exists` | 前缀命中探测 | `prefix=N/total` |
| `batch_set_v2` / `batch_get_v2` | 多池（Mamba/SWA/DeepSeek-V4） | 逐池 `kv ok=3/3, extra ok=3/3` |
| `batch_exists_v2` | 多池前缀探测 | `kv=K/total <pool>=…` |
| `set` / `get` | generic 单页读写 | `ok`/`fail`/`fail none`；`hit`/`miss`，参数带字节数 |
| `batch_set` / `batch_get` | generic 批量（循环 set/get） | `ok H/N` / `hits=H/N` |
| `exists` | 单 key 存在性 | `found` / `not_found` |

> 异常会被记成 `... : FAIL <异常类型>: <信息>` 并**照常向上抛出**（不吞异常）。
> `init` 里若 `dfkv_open` / 发现启动失败即属此类。最早的 `interface_v1` 必填校验在日志配置之前就 `raise`，不记录。

## 性能

- **关闭时**：约几十纳秒/次。`access_log()` 返回一个冻结的 noop 单例，跳过计时与参数格式化。
- **开启时**：约几微秒/次。落盘走 `logging.handlers.QueueHandler` + 后台 `QueueListener` 线程，
  前台只入队、**不在热路径同步写/flush**。进程退出时 `atexit` 自动 flush。

## 验证 / 复跑

无需 GPU/torch，单测会拉起真实 `dfkv_server` 节点驱动插件：

```bash
DFKV_BUILD=$(pwd)/build python3 tests/python/test_dfkv_hicache.py   # 含 8 个 access-log 用例
```

## 实现位置

- [python/dfkv_access_log.py](../python/dfkv_access_log.py) — `configure()` / `access_log()` 上下文管理器、noop 单例、异步队列、格式化辅助。
- [python/dfkv_hicache.py](../python/dfkv_hicache.py) — `__init__` 里调 `configure()`，并用 `with access_log(...)` 包住各继承接口。
- [tests/python/test_dfkv_hicache.py](../tests/python/test_dfkv_hicache.py) — `DfkvAccessLogTest` 测试类。
