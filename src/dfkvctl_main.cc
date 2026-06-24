/* dfkvctl — CLI for a dfkv cluster.
 *   dfkvctl --members "n=ip:port,..." put   <key> <value>
 *   dfkvctl --members "n=ip:port,..." get   <key>
 *   dfkvctl --members "n=ip:port,..." exist <key>
 *   dfkvctl stat <node-ip:port>           # fetch a node's Prometheus metrics
 *   dfkvctl ring --mds <ep,...> --group <g>          # show the cluster ring + vnode share
 *   dfkvctl stat --all --mds <ep,...> --group <g> [--stat-port <p>]  # per-node metrics + aggregate
 *     The MDS now hands back each node's TCP stat port (MemberInfo.tcp_port), so stat --all
 *     reaches nodes automatically even in RDMA deploys. --stat-port <p> stays as a manual
 *     override (e.g. for older servers that predate tcp_port registration).
 * Geometry flags (must match the writer for get to hit): --model_hash --page_size
 *   --dtype_tag --layer_num --head_num --head_dim --mla 0|1 --tp_size --tp_rank */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "con_hash.h"
#include "kv_client.h"
#include "mds_member_poller.h"
#include "membership.h"
#include "prom_parse.h"
#include "tcp_transport.h"
#include "value_header.h"
#include "version.h"

using namespace dfkv;  // NOLINT

static std::vector<std::string> SplitCsv(const std::string& s) {
  std::vector<std::string> out;
  for (size_t i = 0, j; i <= s.size(); i = j + 1) {
    j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    if (j > i) out.push_back(s.substr(i, j - i));
  }
  return out;
}

// One-shot MDS member query via the poller (PollOnce fires the callback on the
// first epoch). Returns true if the MDS answered.
static bool QueryMembers(const std::string& mds, const std::string& group,
                         std::vector<MemberInfo>* out) {
  auto eps = SplitCsv(mds);
  if (eps.empty()) return false;
  bool got = false;
  MdsMemberPoller poller(eps, group,
                         [&](const std::vector<MemberInfo>& ms) { *out = ms; got = true; },
                         1000, 2000);
  poller.PollOnce();
  return got;
}

static int CmdRing(const std::string& mds, const std::string& group) {
  if (mds.empty()) { std::fprintf(stderr, "ring needs --mds ip:port[,...]\n"); return 2; }
  std::vector<MemberInfo> ms;
  if (!QueryMembers(mds, group, &ms)) { std::fprintf(stderr, "ring: MDS query failed\n"); return 1; }
  ConHash ring;
  for (const auto& m : ms) ring.AddNode(m.id, m.weight);
  ring.Build();
  auto pc = ring.NodePointCounts();
  size_t total = ring.RingSize();
  std::printf("group=%s members=%zu ring_points=%zu\n", group.c_str(), ms.size(), total);
  std::printf("%-16s %-22s %6s %8s %7s\n", "ID", "ADDR", "WEIGHT", "VNODES", "SHARE");
  for (const auto& m : ms) {
    std::string addr = m.ip + ":" + std::to_string(m.port);
    size_t v = pc.count(m.id) ? pc[m.id] : 0;
    double share = total ? 100.0 * static_cast<double>(v) / static_cast<double>(total) : 0.0;
    std::printf("%-16s %-22s %6u %8zu %6.1f%%\n", m.id.c_str(), addr.c_str(), m.weight, v, share);
  }
  return 0;
}

static int CmdStatAll(const std::string& mds, const std::string& group,
                      uint32_t stat_port) {
  if (mds.empty()) { std::fprintf(stderr, "stat --all needs --mds ip:port[,...]\n"); return 2; }
  std::vector<MemberInfo> ms;
  if (!QueryMembers(mds, group, &ms)) { std::fprintf(stderr, "stat --all: MDS query failed\n"); return 1; }
  TcpTransport t; t.set_timeouts(2000, 3000);
  uint64_t T_used = 0, T_obj = 0, T_hit = 0, T_miss = 0, T_w = 0, T_r = 0;
  std::printf("%-14s %-20s %10s %9s %9s %9s %6s\n",
              "ID", "ADDR", "USED_MB", "OBJECTS", "HITS", "MISSES", "HIT%");
  for (const auto& m : ms) {
    // Port to reach kStats on: an explicit --stat-port wins; else the server-registered
    // TCP wire/stat port from the MDS (m.tcp_port); else fall back to the member's data
    // port (the rdma-port in an RDMA deploy, where it would print (unreachable)).
    uint32_t sp = stat_port ? stat_port : (m.tcp_port ? m.tcp_port : m.port);
    std::string addr = m.ip + ":" + std::to_string(sp), text;
    if (t.Stats(addr, &text) != Status::kOk) {
      std::printf("%-14s %-20s   (unreachable)\n", m.id.c_str(), addr.c_str());
      continue;
    }
    uint64_t used = PromMetricValue(text, "dfkv_used_bytes");
    uint64_t obj = PromMetricValue(text, "dfkv_objects");
    uint64_t hit = PromMetricValue(text, "dfkv_cache_hit_total");
    uint64_t miss = PromMetricValue(text, "dfkv_cache_miss_total");
    uint64_t w = PromMetricValue(text, "dfkv_bytes_written_total");
    uint64_t r = PromMetricValue(text, "dfkv_bytes_read_total");
    double hr = (hit + miss) ? 100.0 * static_cast<double>(hit) / static_cast<double>(hit + miss) : 0.0;
    std::printf("%-14s %-20s %10.1f %9llu %9llu %9llu %5.1f%%\n", m.id.c_str(), addr.c_str(),
                used / 1048576.0, (unsigned long long)obj, (unsigned long long)hit,
                (unsigned long long)miss, hr);
    T_used += used; T_obj += obj; T_hit += hit; T_miss += miss; T_w += w; T_r += r;
  }
  double thr = (T_hit + T_miss) ? 100.0 * static_cast<double>(T_hit) / static_cast<double>(T_hit + T_miss) : 0.0;
  std::printf("TOTAL used=%.1fMB objects=%llu hits=%llu misses=%llu hit%%=%.1f bytes_w=%llu bytes_r=%llu\n",
              T_used / 1048576.0, (unsigned long long)T_obj, (unsigned long long)T_hit,
              (unsigned long long)T_miss, thr, (unsigned long long)T_w, (unsigned long long)T_r);
  return 0;
}

static std::vector<std::pair<std::string, std::string>> ParseMembers(const std::string& s) {
  std::vector<std::pair<std::string, std::string>> out;
  size_t i = 0;
  while (i <= s.size()) {
    size_t c = s.find(',', i);
    if (c == std::string::npos) c = s.size();
    std::string tok = s.substr(i, c - i);
    size_t eq = tok.find('=');
    if (eq != std::string::npos) out.emplace_back(tok.substr(0, eq), tok.substr(eq + 1));
    if (c == s.size()) break;
    i = c + 1;
  }
  return out;
}

int main(int argc, char** argv) {
  if (WantsVersion(argc, argv)) { std::printf("dfkvctl %s\n", Version()); return 0; }
  std::string members;
  uint64_t model_hash = 0x51;
  uint32_t page = 64, dtype = 0x46384534u, layer = 78, head = 1, hd = 576, tps = 8, tpr = 0, mla = 1;
  std::string mds, group = "default";
  bool all = false;
  uint32_t stat_port = 0;  // override; 0 = use MDS-provided tcp_port, else the member's port
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nv = [&](uint64_t* d) { if (i + 1 < argc) *d = std::stoull(argv[++i]); };
    auto nv32 = [&](uint32_t* d) { if (i + 1 < argc) *d = (uint32_t)std::stoul(argv[++i]); };
    if (a == "--members" && i + 1 < argc) members = argv[++i];
    else if (a == "--mds" && i + 1 < argc) mds = argv[++i];
    else if (a == "--group" && i + 1 < argc) group = argv[++i];
    else if (a == "--all") all = true;
    else if (a == "--stat-port") nv32(&stat_port);
    else if (a == "--model_hash") nv(&model_hash);
    else if (a == "--page_size") nv32(&page);
    else if (a == "--dtype_tag") nv32(&dtype);
    else if (a == "--layer_num") nv32(&layer);
    else if (a == "--head_num") nv32(&head);
    else if (a == "--head_dim") nv32(&hd);
    else if (a == "--tp_size") nv32(&tps);
    else if (a == "--tp_rank") nv32(&tpr);
    else if (a == "--mla") nv32(&mla);
    else pos.push_back(a);
  }
  if (pos.empty()) { std::fprintf(stderr, "usage: dfkvctl [--members ...] put|get|exist|stat|ring ...\n"); return 2; }
  const std::string& cmd = pos[0];

  if (cmd == "ring") return CmdRing(mds, group);

  if (cmd == "stat") {
    if (all) return CmdStatAll(mds, group, stat_port);
    if (pos.size() < 2) { std::fprintf(stderr, "stat <node-ip:port>  |  stat --all --mds ... [--stat-port <p>]\n"); return 2; }
    TcpTransport t; std::string text;
    if (t.Stats(pos[1], &text) != Status::kOk) { std::fprintf(stderr, "stat failed\n"); return 1; }
    std::fputs(text.c_str(), stdout);
    return 0;
  }

  auto mem = ParseMembers(members);
  if (mem.empty()) { std::fprintf(stderr, "need --members name=ip:port,...\n"); return 2; }
  ValueHeader hdr = ValueHeader::Make(model_hash, page, dtype,
                                      mla ? ValueHeader::kFlagIsMla : 0,
                                      (uint16_t)tps, (uint16_t)tpr, (uint16_t)layer,
                                      (uint16_t)head, (uint16_t)hd);
  KVClient c(mem, hdr);

  if (cmd == "put" && pos.size() >= 3) {
    bool ok = c.Put(pos[1], pos[2].data(), pos[2].size());
    std::printf("%s\n", ok ? "OK" : "FAIL"); return ok ? 0 : 1;
  }
  if (cmd == "get" && pos.size() >= 2) {
    std::string out;
    if (!c.GetAuto(pos[1], &out)) { std::printf("(miss)\n"); return 1; }
    std::fwrite(out.data(), 1, out.size(), stdout); std::printf("\n"); return 0;
  }
  if (cmd == "exist" && pos.size() >= 2) {
    bool e = c.Exist(pos[1]); std::printf("%s\n", e ? "true" : "false"); return e ? 0 : 1;
  }
  std::fprintf(stderr, "unknown/!args: %s\n", cmd.c_str());
  return 2;
}
