# dfkv — standalone distributed KV cache for SGLang HiCache (no brpc/MDS/S3 deps).
cmake_minimum_required(VERSION 3.20)
project(dfkv CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

option(DFKV_BUILD_TESTS "Build dfkv tests" ON)
option(DFKV_STATIC_LIBSTDCXX "Statically link libstdc++/libgcc for portable artifacts" OFF)
if(DFKV_STATIC_LIBSTDCXX)
  add_link_options(-static-libstdc++ -static-libgcc)
endif()

# --- portable core library (everything in src/ except the server main) ---
file(GLOB DFKV_SRCS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
list(REMOVE_ITEM DFKV_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/dfkv_server_main.cc)

add_library(dfkv_core STATIC ${DFKV_SRCS})
target_include_directories(dfkv_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
find_package(Threads REQUIRED)
target_link_libraries(dfkv_core PUBLIC Threads::Threads)

# C ABI shared lib for the Python (ctypes) plugin
add_library(dfkv SHARED ${DFKV_SRCS})
target_include_directories(dfkv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dfkv PUBLIC Threads::Threads)

# standalone cache-node daemon
add_executable(dfkv_server ${CMAKE_CURRENT_SOURCE_DIR}/src/dfkv_server_main.cc)
target_link_libraries(dfkv_server PRIVATE dfkv_core)

include(GNUInstallDirs)
install(TARGETS dfkv dfkv_server dfkv_core
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY src/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dfkv FILES_MATCHING PATTERN "*.h")
install(FILES python/dingofs_hicache.py DESTINATION ${CMAKE_INSTALL_DATADIR}/dfkv/python)

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
  foreach(t ${DFKV_TESTS})
    get_filename_component(name ${t} NAME_WE)
    add_executable(${name} ${t})
    target_link_libraries(${name} PRIVATE dfkv_core GTest::gtest_main)
    gtest_discover_tests(${name})
  endforeach()

  # Python plugin test (ctypes -> libdfkv.so -> dfkv_server). No GPU/torch.
  find_program(PYTHON3 python3)
  if(PYTHON3)
    add_test(NAME python_plugin
      COMMAND ${PYTHON3} -m unittest test_dingofs_hicache
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/python)
    set_tests_properties(python_plugin PROPERTIES
      ENVIRONMENT "DFKV_BUILD=${CMAKE_BINARY_DIR};DFKV_LIB=${CMAKE_BINARY_DIR}/libdfkv.so")
  endif()
endif()
