# dfkv Observability (P0/P1/P2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline, autonomous to v1.4.0 release). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make dfkv cluster/ring/node/IO/RPC observable (Prometheus-scrapeable server+MDS, cluster/ring CLI, client + RDMA metrics, plugin histograms) without touching datapath performance.

**Architecture:** Relaxed-atomic counters on the hot path (existing pattern); 1/64-sampled latency via vDSO clock into a lock-free histogram; a tiny self-contained HTTP/1.0 `/metrics` responder on a dedicated thread/port (opt-in); client metrics surfaced through a new C ABI snapshot consumed by a sleeping plugin poller thread.

**Tech Stack:** C++17 (atomics, std::thread, libibverbs), CMake/ctest/gtest, Python (prometheus_client, ctypes), GitHub PRs to dingodb/dfkv.

**Design:** `docs/superpowers/specs/2026-06-17-dfkv-observability-design.md`

**Delivery:** 5 PRs (A→E) each TDD'd, CI-green, squash-merged when `mergeStateStatus=CLEAN`; then bump v1.4.0 + GitHub release. Branch from clean main per PR, `feat/obs-*` naming (avoid `perf` prefix). Each PR: RED→GREEN→commit→push origin→`gh pr create --repo dingodb/dfkv`→wait CLEAN→`gh pr merge --squash --delete-branch`→sync main.

---

## PR-A — Embedded HTTP /metrics + MetricsText upgrade (branch: feat/obs-metrics-http)

**Files:**
- Create: `src/metrics_http.h`, `src/metrics_http.cc`
- Create: `tests/metrics_http_test.cc`
- Modify: `src/kv_node_server.h`, `src/kv_node_server.cc` (HELP/TYPE, labels, identity, uptime, build_info)
- Modify: `src/dfkv_server_main.cc` (`--metrics-port`, identity wiring)
- Modify: `CMakeLists.txt` (metrics_http in lib, new test)

### Task A1: MetricsHttpServer
- [ ] Test `tests/metrics_http_test.cc`: start `MetricsHttpServer([]{return "dfkv_x 1\n";})` on port 0; HTTP GET `/metrics` over a socket returns `200` + body contains `dfkv_x 1`; GET `/healthz` returns `200 ok`; GET `/bad` returns `404`.
- [ ] Run → FAIL (no class).
- [ ] Implement `metrics_http.{h,cc}`: ctor takes `std::function<std::string()> render`; `Start(int port)` binds INADDR_ANY, accept thread; per-conn read request line (until `\r\n`), parse method+path; route /metrics→200 text/plain + body, /healthz→200 "ok", else 404; `Stop()` shuts down + joins (mirror KvNodeServer Stop drain). `port()` accessor.
- [ ] Run → PASS. Commit.

### Task A2: MetricsText Prometheus upgrade + identity
- [ ] Test in `tests/metrics_test.cc` (extend): a KvNodeServer with `set_identity("n1","g1")` → MetricsText contains `# TYPE dfkv_cache_hit_total counter`, `dfkv_cache_hit_total{node="n1",group="g1"} 0`, `dfkv_build_info{`, `dfkv_uptime_seconds`.
- [ ] Run → FAIL.
- [ ] Implement: add `set_identity(id,group)`, `start_time_` (set in ctor); rewrite `MetricsText()` to emit HELP/TYPE, label suffix helper `lbl()` (empty id → no braces), gauges (used_bytes/objects/disks/open_connections) vs counters (`_total`), `dfkv_build_info{version="<VER>",transport="rdma|tcp"} 1`, `dfkv_uptime_seconds`. Version from a `DFKV_VERSION` compile def (add in CMake `target_compile_definitions`).
- [ ] Run → PASS. Commit.

### Task A3: Wire --metrics-port into dfkv_server
- [ ] Modify `dfkv_server_main.cc`: parse `--metrics-port`; if >0, construct `MetricsHttpServer([&srv]{return srv.MetricsText();})`, Start, log. Pass `--id`/`--group` to `srv.set_identity`. Stop on shutdown.
- [ ] Manual smoke (in plan exec): start server with `--metrics-port`, `curl :PORT/metrics`. Add a line to `tests/tool_smoke.sh` if feasible (curl optional → guard).
- [ ] Build all, run full ctest → green. Commit. Open PR-A (include the already-committed design doc commit on this branch).

---

## PR-B — Cluster/ring CLI + MDS metrics (branch: feat/obs-cluster-cli)

**Files:**
- Modify: `src/mds_server.h`, `src/mds_server.cc` (counters + MetricsText)
- Create: `tests/mds_metrics_test.cc`
- Modify: `src/dfkv_mds_main.cc` (`--metrics-port`)
- Modify: `src/dfkvctl_main.cc` (`ring`, `stat --all`)
- Modify: `src/tcp_transport.{h,cc}` if a ListMembers helper is needed for ctl
- Modify: `CMakeLists.txt`

### Task B1: MDS counters + MetricsText
- [ ] Test `mds_metrics_test.cc`: drive `MdsServer` against a fake/real etcd path is heavy — instead unit-test a `MdsMetrics` struct (atomics + `Render()`), assert counters increment and `Render()` emits `dfkv_mds_list_requests_total`. Keep MdsServer wiring covered by existing mds_server_test (extend to assert MetricsText nonempty after a ListMembers).
- [ ] Run → FAIL.
- [ ] Implement: add atomics to MdsServer (`list_requests_`, `register_requests_`, `keepalives_`, `lease_grants_`, `etcd_errors_`, `members_gauge_` per group is hard → expose total members last-listed); increment at Upsert (register vs keepalive by op), ListMembers, etcd error returns. Add `MetricsText()`.
- [ ] Run → PASS. Commit.

### Task B2: dfkv_mds --metrics-port
- [ ] Modify `dfkv_mds_main.cc`: `--metrics-port`; if >0 start MetricsHttpServer over `mds.MetricsText()`.
- [ ] Build, smoke. Commit.

### Task B3: dfkvctl ring
- [ ] Test: extend `tests/tool_smoke.sh` — bring up MDS + 2 nodes registering, run `dfkvctl ring --mds <ep> --group default`, assert output has both node ids and a `vnodes` column. (If smoke infra too heavy for ring, add a focused unit on the ring-render function.)
- [ ] Implement `ring` subcommand: query MDS ListMembers (reuse `MdsMemberPoller` one-shot or a direct `kListMembers` round-trip via TcpTransport), build `ConHash` with weights, print table: id, addr, weight; then per-node vnode count + share% from the ring.
- [ ] Run → PASS. Commit.

### Task B4: dfkvctl stat --all
- [ ] Implement `stat --all --mds <eps> --group <g>`: discover members, for each `TcpTransport::Stats(addr)`, parse `dfkv_used_bytes`/`dfkv_objects`/`dfkv_cache_hit_total`/`dfkv_cache_miss_total`/etc, print per-node rows + aggregate totals + cluster hit-rate.
- [ ] Smoke assert aggregate line present. Commit. Open PR-B.

---

## PR-C — Server-side depth metrics (branch: feat/obs-server-depth)

**Files:**
- Modify: `src/kv_store.h`, `src/kv_store.cc` (eviction/cache_full counters, per-disk surfaced via group)
- Modify: `src/disk_cache_group.h`, `src/disk_cache_group.cc` (per-disk used/objects accessors, evict counters passthrough)
- Create: `src/latency_hist.h` (lock-free sampled histogram)
- Create: `tests/latency_hist_test.cc`
- Modify: `src/kv_node_server.{h,cc}` (errors{op,status}, open_connections, latency sampling, per-disk in MetricsText, evict counters in MetricsText)
- Modify: `tests/kv_store_test.cc` (eviction counter assertions)
- Modify: `CMakeLists.txt`

### Task C1: LatencyHist (lock-free, sampled record API)
- [ ] Test `latency_hist_test.cc`: `LatencyHist h;` record 100 values across buckets via `h.Observe(seconds)`; assert `h.Count()==100`, `h.Sum()≈`, and `h.Render("dfkv_op_latency_seconds", "{op=\"get\"}")` emits `_bucket{le="..."}`, `_sum`, `_count` lines monotonic. Separately test the sampler: `Sampler s(64)` → `s.ShouldSample()` true exactly every 64th call.
- [ ] Run → FAIL.
- [ ] Implement `latency_hist.h`: fixed `static constexpr double kBounds[] = {50e-6,100e-6,250e-6,500e-6,1e-3,2.5e-3,5e-3,10e-3,25e-3,50e-3,100e-3}`; `std::atomic<uint64_t> buckets_[N+1]`, `_sum_us`, `_count`. `Observe(sec)`: linear/branch scan to bucket, relaxed inc. `Render(name,labels)`. `Sampler{atomic<uint64_t> seq; mask}` `ShouldSample(){return (seq.fetch_add(1,relaxed)&mask)==0;}`.
- [ ] Run → PASS. Commit.

### Task C2: KVStore eviction counters
- [ ] Test `kv_store_test.cc`: tiny-capacity store, write enough to evict; assert `store.Evictions()>0` and `store.EvictedBytes()>0`; on a write that can't fit assert `store.CacheFull()` increments (if applicable to CLOCK path).
- [ ] Run → FAIL.
- [ ] Implement: atomics in KVStore incremented inside `EvictLocked` (count + bytes per evicted entry); expose `Evictions()/EvictedBytes()/CacheFull()`. DiskCacheGroup aggregates across stores; expose `Evictions()` etc + per-disk `DiskUsedBytes(i)`/`DiskObjects(i)`/`DiskPath(i)`.
- [ ] Run → PASS. Commit.

### Task C3: KvNodeServer depth wiring
- [ ] Test (metrics_test extend): after a forced IO error path is hard; assert presence of new series in MetricsText: `dfkv_open_connections`, `dfkv_evictions_total`, `dfkv_op_latency_seconds_count`, `dfkv_disk_used_bytes{disk="..."}`. Assert open_connections increments under a live TCP conn (use existing client) then returns to 0 after close.
- [ ] Run → FAIL.
- [ ] Implement: `open_connections_` atomic (++ in AcceptLoop before handler, -- at handler end); `errors_` map keyed by (op,status) for kIOError/kInvalid (fixed small set → individual atomics: `put_io_err_`, `get_io_err_`, `invalid_`); latency: a `Sampler` + two `LatencyHist` (get/put), bracket in ProcessRequest/RangeInto/RangeDirect only when `sampler.ShouldSample()`; extend MetricsText to emit eviction, open_conns, latency, per-disk, errors.
- [ ] Run → PASS, full ctest green. Commit. Open PR-C.

---

## PR-D — Client metrics + C ABI snapshot + plugin poller (branch: feat/obs-client-metrics)

**Files:**
- Create: `src/client_metrics.h`
- Create: `tests/client_metrics_test.cc`
- Modify: `src/peer_health.h` (mark_bad/mark_good counters)
- Modify: `src/kv_client.{h,cc}` (hold ClientMetrics, increment per op/peer, Snapshot render)
- Modify: `src/dfkv_c_api.{h,cc}` (`dfkv_stats_snapshot`)
- Modify: `tests/c_api_test.cc` (snapshot smoke)
- Modify: `python/dfkv_metrics.py` (client-snapshot poller: parse + delta + Counters), `python/dfkv_hicache.py` (start/stop poller thread)
- Modify: `tests/python/test_dfkv_hicache.py` (poller parses snapshot, increments)
- Modify: `CMakeLists.txt`

### Task D1: ClientMetrics + PeerHealth counters
- [ ] Test `client_metrics_test.cc`: `ClientMetrics m; m.OnGet(true); m.OnGet(false); m.OnError(Status::kIOError,"1.2.3.4:1"); m.OnRoute("1.2.3.4:1");` → `m.Render()` text contains `dfkv_client_get_hit_total 1`, `dfkv_client_get_miss_total 1`, `dfkv_client_errors_total{status="io"} 1`, `dfkv_client_peer_routed_total{peer="1.2.3.4:1"} 1`. PeerHealth: MarkBad/MarkGood increment `bad_total/good_total`.
- [ ] Run → FAIL.
- [ ] Implement `client_metrics.h`: atomics for put/get/exist calls, get_hit/get_miss, errors by status (fixed enum), and a peer map (mutex only on first-seen insert; value is struct of atomics) for routed/error per peer; `Render()`. Add counters to PeerHealth.
- [ ] Run → PASS. Commit.

### Task D2: Wire into KVClient
- [ ] Test (kv_client_test extend): do a Put then Get(hit) then Get(miss) against a live node; `client.MetricsSnapshot()` contains get_hit_total ≥1 and peer_routed_total for the node addr.
- [ ] Run → FAIL.
- [ ] Implement: KVClient holds `ClientMetrics metrics_`; increment in Put/Get/GetAuto/Exist/Batch* (per-op + per-route + per-error); expose `std::string MetricsSnapshot() const`.
- [ ] Run → PASS. Commit.

### Task D3: C ABI dfkv_stats_snapshot
- [ ] Test `c_api_test.cc`: open client, do ops, `int n = dfkv_stats_snapshot(c, buf, sizeof buf); assert n>0 && strstr(buf,"dfkv_client_get")`. cap=0 returns required length.
- [ ] Run → FAIL.
- [ ] Implement `dfkv_stats_snapshot` in c_api: render snapshot to string, copy into buf (≤cap), return full length. Declare in header.
- [ ] Run → PASS. Commit.

### Task D4: Plugin poller
- [ ] Test (test_dfkv_hicache extend, module-level, no node needed where possible): feed `dfkv_metrics.ClientStatsPoller` a fake snapshot-provider returning two successive texts; assert it computes deltas and increments prometheus Counters / snapshot() reflects totals. Then an integration assert: DfkvHiCache with `client_stats_poll_s=0` starts no thread; >0 starts a daemon thread that stops on __del__.
- [ ] Run → FAIL.
- [ ] Implement: `dfkv_metrics.ClientStatsPoller(get_text_fn, interval_s)` thread: parse Prometheus lines → dict; diff vs last; `.inc(delta)` on mirrored Counters; `start()/stop()`. In DfkvHiCache.__init__ read `client_stats_poll_s` (extra_config/env, default 10), if >0 build poller with `lambda: _read_snapshot(self._lib,self._h)` and start; stop in __del__.
- [ ] Run → PASS, full python suite + ctest green. Commit. Open PR-D.

---

## PR-E — RDMA counters + plugin histograms (branch: feat/obs-rdma-metrics)

**Files:**
- Modify: `src/rdma_server.{h,cc}` (completions/errors/active_conns/idle_reclaims atomics + accessors)
- Modify: `src/rdma_transport.{h,cc}` (mr_registrations, per-rail rail_ops)
- Modify: `src/kv_node_server.{h,cc}` + `dfkv_server_main.cc` (fold RDMA server counters into MetricsText via a getter callback)
- Modify: `src/client_metrics.h` / kv_client to include transport rail/mr counters in snapshot
- Modify: `python/dfkv_metrics.py` (Histogram set/get seconds, errors{op})
- Modify: `python/dfkv_hicache.py` (observe duration into histogram in batch_set_v1/get_v1; count errors)
- Modify: tests: `tests/rdma_loopback_test.cc` (counter asserts), `tests/python/test_dfkv_hicache.py` (histogram observed)
- Modify: `CMakeLists.txt`

### Task E1: RDMA server counters
- [ ] Test `rdma_loopback_test.cc` (RDMA build only): after a loopback GET, `rserver.Completions()>0`, `rserver.ActiveConns()` tracked. Guard under DFKV_WITH_RDMA.
- [ ] Run → FAIL.
- [ ] Implement: atomics in RdmaServer incremented in Serve completion loop (`completions_`, on `wc.status!=SUCCESS` `completion_errors_`, active_conns ++/-- on Serve thread start/exit, `idle_reclaims_` on idle timeout path); accessors.
- [ ] Run → PASS. Commit.

### Task E2: RDMA client transport counters + fold into outputs
- [ ] Test: rdma_transport: after RegisterMemory + an op, `mr_registrations_>0`; rail_ops indexed by dev. (loopback test).
- [ ] Implement: counters in RdmaTransport (`mr_registrations_`, `rail_ops_[dev]`); expose via accessor; KVClient folds into snapshot; dfkv_server folds RdmaServer counters into MetricsText (pass a `std::function` getter to KvNodeServer or render in main and append).
- [ ] Run → PASS. Commit.

### Task E3: Plugin histograms + error counters
- [ ] Test (test_dfkv_hicache): after batch_set_v1/get_v1, `_Metrics` snapshot exposes histogram counts; an induced FAIL increments `errors`.
- [ ] Run → FAIL.
- [ ] Implement: dfkv_metrics add `Histogram` (multiproc-safe) for set/get seconds + `errors` counter{op}; DfkvHiCache passes the measured duration (perf_counter, reuse access-log timing where on) into `on_set/on_get`; count errors on FAIL.
- [ ] Run → PASS, full suite green. Commit. Open PR-E.

---

## M6 — Release v1.4.0

- [ ] After A–E all merged + main synced: branch `release-1.4.0`, bump `VERSION` + CMake `project()` to 1.4.0, commit, PR, wait CLEAN, merge.
- [ ] Annotated tag `v1.4.0` on merge commit, push upstream.
- [ ] Wait for main CI run → download `dfkv-linux-x86_64` artifact (`dfkv-1.4.0-linux-x86_64.tar.gz`), sha256.
- [ ] `gh release create v1.4.0 --repo dingodb/dfkv` with notes (P0/P1/P2 highlights, compatibility = no wire change / opt-in, validation incl. real-machine A/B bench) + artifact.
- [ ] Optional real-machine A/B bench on hd03/hd04 (metrics-port on vs off) to confirm no datapath regression; record in release notes.

## Perf-safety checklist (verify before E merge)
- [ ] All hot-path counters are relaxed atomics (grep for any mutex on Range/Cache/Process/Serve path).
- [ ] Latency only sampled (Sampler mask), only sampled ops call clock_gettime.
- [ ] No per-byte counters; RDMA counters in completion loop only.
- [ ] HTTP server on its own thread/port; render reads relaxed only.
- [ ] Plugin poller is a sleeping daemon, interval-gated, off batch path.
