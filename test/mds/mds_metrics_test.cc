// TDD — MDS metrics counters + Prometheus exposition (no etcd needed).
#include "mds/mds_metrics.h"

#include <gtest/gtest.h>

#include <string>

using namespace dfkv;  // NOLINT

TEST(MdsMetrics, CountsAndRenders) {
  MdsMetrics m;
  m.register_requests.fetch_add(2);
  m.keepalives.fetch_add(5);
  m.list_requests.fetch_add(3);
  m.lease_grants.fetch_add(1);
  m.etcd_errors.fetch_add(0);
  m.members_last_list.store(4);

  std::string t = m.Render();
  EXPECT_NE(t.find("# TYPE dfkv_mds_list_requests_total counter"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_mds_register_requests_total 2"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_mds_keepalives_total 5"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_mds_list_requests_total 3"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_mds_lease_grants_total 1"), std::string::npos) << t;
  EXPECT_NE(t.find("# TYPE dfkv_mds_members gauge"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_mds_members 4"), std::string::npos) << t;
  // build_info gauge exposes the MDS version (mirrors dfkv_server).
  EXPECT_NE(t.find("# TYPE dfkv_build_info gauge"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_build_info{version=\""), std::string::npos) << t;
  EXPECT_NE(t.find("role=\"mds\"} 1"), std::string::npos) << t;
}
