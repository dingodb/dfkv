/* MdsMetrics — relaxed-atomic counters for the MDS, rendered as Prometheus text.
 * Held by MdsServer and incremented at the register/heartbeat/list/etcd call
 * sites. Kept in its own header so the counting + exposition format is unit-
 * testable without standing up etcd. */
#ifndef DFKV_MDS_METRICS_H_
#define DFKV_MDS_METRICS_H_

#include <atomic>
#include <cstdint>
#include <string>

#include "common/version.h"  // Version() for the build_info gauge

namespace dfkv {

struct MdsMetrics {
  std::atomic<uint64_t> register_requests{0};
  std::atomic<uint64_t> keepalives{0};
  std::atomic<uint64_t> list_requests{0};
  std::atomic<uint64_t> lease_grants{0};
  std::atomic<uint64_t> etcd_errors{0};
  std::atomic<uint64_t> members_last_list{0};  // members returned by the last ListMembers

  std::string Render() const {
    std::string s;
    // Version observability for the MDS tier (mirrors dfkv_server's build_info,
    // so the 3-replica MDS versions are visible via Prometheus, not just --version).
    s += "# HELP dfkv_build_info Build and version info (constant 1)\n";
    s += "# TYPE dfkv_build_info gauge\n";
    s += std::string("dfkv_build_info{version=\"") + Version() + "\",role=\"mds\"} 1\n";
    auto m = [&](const char* name, const char* type, const char* help, uint64_t v) {
      s += "# HELP "; s += name; s += " "; s += help; s += "\n";
      s += "# TYPE "; s += name; s += " "; s += type; s += "\n";
      s += name; s += " "; s += std::to_string(v); s += "\n";
    };
    auto ld = [](const std::atomic<uint64_t>& a) {
      return a.load(std::memory_order_relaxed);
    };
    m("dfkv_mds_register_requests_total", "counter", "Register ops received", ld(register_requests));
    m("dfkv_mds_keepalives_total", "counter", "Heartbeat ops received", ld(keepalives));
    m("dfkv_mds_list_requests_total", "counter", "ListMembers ops served", ld(list_requests));
    m("dfkv_mds_lease_grants_total", "counter", "etcd lease grants issued", ld(lease_grants));
    m("dfkv_mds_etcd_errors_total", "counter", "etcd I/O failures", ld(etcd_errors));
    m("dfkv_mds_members", "gauge", "Members returned by the last ListMembers", ld(members_last_list));
    return s;
  }
};

}  // namespace dfkv

#endif  // DFKV_MDS_METRICS_H_
