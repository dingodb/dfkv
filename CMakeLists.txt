# dfkv — standalone distributed KV cache for SGLang HiCache (no brpc/MDS/S3 deps).
cmake_minimum_required(VERSION 3.20)
project(dfkv VERSION 1.6.5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

option(DFKV_BUILD_TESTS "Build dfkv tests" ON)
option(DFKV_STATIC_LIBSTDCXX "Statically link libstdc++/libgcc for portable artifacts" OFF)
option(DFKV_WITH_RDMA "Build the RDMA transport (needs libibverbs)" OFF)
# Additive, env-gated io_uring async-GET disk-read path in the RDMA server. When
# ON, links liburing and compiles the DFKV_WITH_URING code; the path is still
# inert unless the server is started with DFKV_SERVER_URING=1. When OFF (default)
# the io_uring code is #ifdef'd out and the env flag is a no-op. Requires RDMA.
option(DFKV_WITH_URING "Build the io_uring async GET path in the RDMA server (needs liburing)" OFF)
if(DFKV_STATIC_LIBSTDCXX)
  add_link_options(-static-libstdc++ -static-libgcc)
endif()
if(DFKV_WITH_URING AND NOT DFKV_WITH_RDMA)
  message(FATAL_ERROR "DFKV_WITH_URING requires DFKV_WITH_RDMA=ON (the io_uring path lives in the RDMA server)")
endif()
if(DFKV_WITH_URING)
  find_library(URING_LIB uring)
  if(NOT URING_LIB)
    message(FATAL_ERROR "DFKV_WITH_URING=ON but liburing not found (install liburing-dev)")
  endif()
endif()

# --- portable core library (everything in src/ except *_main.cc + gated rdma) ---
file(GLOB DFKV_SRCS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
list(REMOVE_ITEM DFKV_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dfkv_server_main.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dfkv_smoke_main.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dfkvctl_main.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dfkv_bench_main.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dfkv_mds_main.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dfkv_discover_smoke_main.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/rdma_verbs.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/rdma_transport.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/rdma_server.cc)
if(DFKV_WITH_RDMA)
  list(APPEND DFKV_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rdma_verbs.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rdma_transport.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rdma_server.cc)
endif()

add_library(dfkv_core STATIC ${DFKV_SRCS})
target_include_directories(dfkv_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(dfkv_core PRIVATE _GNU_SOURCE  # O_DIRECT/fallocate in kv_store.cc
  DFKV_VERSION="${PROJECT_VERSION}")
find_package(Threads REQUIRED)
target_link_libraries(dfkv_core PUBLIC Threads::Threads)
if(DFKV_WITH_RDMA)
  target_compile_definitions(dfkv_core PUBLIC DFKV_WITH_RDMA)
  target_link_libraries(dfkv_core PUBLIC ibverbs)
endif()
if(DFKV_WITH_URING)
  target_compile_definitions(dfkv_core PUBLIC DFKV_WITH_URING)
  target_link_libraries(dfkv_core PUBLIC ${URING_LIB})
endif()

# C ABI shared lib for the Python (ctypes) plugin
add_library(dfkv SHARED ${DFKV_SRCS})
target_include_directories(dfkv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(dfkv PRIVATE _GNU_SOURCE  # O_DIRECT/fallocate in kv_store.cc
  DFKV_VERSION="${PROJECT_VERSION}")
target_link_libraries(dfkv PUBLIC Threads::Threads)
if(DFKV_WITH_RDMA)
  target_compile_definitions(dfkv PUBLIC DFKV_WITH_RDMA)
  target_link_libraries(dfkv PUBLIC ibverbs)
endif()
if(DFKV_WITH_URING)
  target_compile_definitions(dfkv PUBLIC DFKV_WITH_URING)
  target_link_libraries(dfkv PUBLIC ${URING_LIB})
endif()

# standalone cache-node daemon + CLI tools
add_executable(dfkv_server ${CMAKE_CURRENT_SOURCE_DIR}/src/dfkv_server_main.cc)
target_link_libraries(dfkv_server PRIVATE dfkv_core)
add_executable(dfkv_mds ${CMAKE_CURRENT_SOURCE_DIR}/src/dfkv_mds_main.cc)
target_link_libraries(dfkv_mds PRIVATE dfkv_core)
add_executable(dfkv_smoke ${CMAKE_CURRENT_SOURCE_DIR}/src/dfkv_smoke_main.cc)
target_link_libraries(dfkv_smoke PRIVATE dfkv_core)
add_executable(dfkvctl ${CMAKE_CURRENT_SOURCE_DIR}/src/dfkvctl_main.cc)
target_link_libraries(dfkvctl PRIVATE dfkv_core)
add_executable(dfkv_bench ${CMAKE_CURRENT_SOURCE_DIR}/src/dfkv_bench_main.cc)
target_link_libraries(dfkv_bench PRIVATE dfkv_core)
add_executable(dfkv_discover_smoke ${CMAKE_CURRENT_SOURCE_DIR}/src/dfkv_discover_smoke_main.cc)
target_link_libraries(dfkv_discover_smoke PRIVATE dfkv_core)

include(GNUInstallDirs)
install(TARGETS dfkv dfkv_server dfkv_mds dfkv_smoke dfkvctl dfkv_bench dfkv_discover_smoke dfkv_core
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY src/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dfkv FILES_MATCHING PATTERN "*.h")
# dfkv_hicache.py imports dfkv_access_log and dfkv_metrics at module load, so all
# three must ship together or an installed plugin fails with ImportError.
install(FILES
  python/dfkv_hicache.py
  python/dfkv_access_log.py
  python/dfkv_metrics.py
  DESTINATION ${CMAKE_INSTALL_DATADIR}/dfkv/python)
# Shared telemetry package (push-metrics now; tracing later). Canonical source;
# the vLLM/LMCache pip connectors vendor a byte-identical copy (drift-guarded).
install(DIRECTORY python/dfkv_telemetry
  DESTINATION ${CMAKE_INSTALL_DATADIR}/dfkv/python
  FILES_MATCHING PATTERN "*.py")

# --- tests ---
if(DFKV_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)

  file(GLOB DFKV_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*_test.cc)
  # rdma_loopback_test needs the RDMA transport/server symbols -> RDMA build only.
  list(REMOVE_ITEM DFKV_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/rdma_loopback_test.cc)
  foreach(t ${DFKV_TESTS})
    get_filename_component(name ${t} NAME_WE)
    add_executable(${name} ${t})
    target_link_libraries(${name} PRIVATE dfkv_core GTest::gtest_main)
    gtest_discover_tests(${name})
  endforeach()
  if(DFKV_WITH_RDMA)  # exercised in CI under Soft-RoCE (rdma_rxe); skips if no device
    add_executable(rdma_loopback_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/rdma_loopback_test.cc)
    target_link_libraries(rdma_loopback_test PRIVATE dfkv_core GTest::gtest_main)
    gtest_discover_tests(rdma_loopback_test)
  endif()

  # Python plugin test (ctypes -> libdfkv.so -> dfkv_server). No GPU/torch.
  find_program(PYTHON3 python3)
  if(PYTHON3)
    add_test(NAME python_plugin
      COMMAND ${PYTHON3} -m unittest test_dfkv_hicache
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/python)
    set_tests_properties(python_plugin PROPERTIES
      ENVIRONMENT "DFKV_BUILD=${CMAKE_BINARY_DIR};DFKV_LIB=${CMAKE_BINARY_DIR}/libdfkv.so")
    # Shared telemetry layer: pure-python, no build/lib needed (OTel-push test
    # self-skips when the SDK is absent). Includes the vendored-copy drift guard.
    add_test(NAME python_telemetry
      COMMAND ${PYTHON3} -m unittest test_dfkv_telemetry test_telemetry_vendor_sync
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/python)
  endif()

  # CLI tools integration smoke (dfkv_server + dfkv_smoke + dfkvctl)
  add_test(NAME tool_smoke
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/tool_smoke.sh ${CMAKE_BINARY_DIR})
endif()

# --- packaging (deb/rpm via CPack) ---
set(CPACK_PACKAGE_NAME "dfkv")
set(CPACK_PACKAGE_VENDOR "ketor")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Distributed KV cache for SGLang HiCache")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_CONTACT "d.ketor@gmail.com")
set(CPACK_GENERATOR "TGZ")  # default; deb/rpm chosen explicitly via `cpack -G DEB|RPM`
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
include(CPack)
