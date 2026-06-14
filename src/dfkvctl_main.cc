/* dfkvctl — CLI for a dfkv cluster.
 *   dfkvctl --members "n=ip:port,..." put   <key> <value>
 *   dfkvctl --members "n=ip:port,..." get   <key>
 *   dfkvctl --members "n=ip:port,..." exist <key>
 *   dfkvctl stat <node-ip:port>           # fetch a node's Prometheus metrics
 * Geometry flags (must match the writer for get to hit): --model_hash --page_size
 *   --dtype_tag --layer_num --head_num --head_dim --mla 0|1 --tp_size --tp_rank */
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "kv_client.h"
#include "tcp_transport.h"
#include "value_header.h"

using namespace dfkv;  // NOLINT

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
  std::string members;
  uint64_t model_hash = 0x51;
  uint32_t page = 64, dtype = 0x46384534u, layer = 78, head = 1, hd = 576, tps = 8, tpr = 0, mla = 1;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nv = [&](uint64_t* d) { if (i + 1 < argc) *d = std::stoull(argv[++i]); };
    auto nv32 = [&](uint32_t* d) { if (i + 1 < argc) *d = (uint32_t)std::stoul(argv[++i]); };
    if (a == "--members" && i + 1 < argc) members = argv[++i];
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
  if (pos.empty()) { std::fprintf(stderr, "usage: dfkvctl [--members ...] put|get|exist|stat ...\n"); return 2; }
  const std::string& cmd = pos[0];

  if (cmd == "stat") {
    if (pos.size() < 2) { std::fprintf(stderr, "stat <node-ip:port>\n"); return 2; }
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
