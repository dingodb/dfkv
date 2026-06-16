# Build helpers for the dfkv LMCache connector.
#
# The connector itself is pure Python (ctypes -> libdfkv.so), so there is no
# CPython extension to compile. These targets (a) build libdfkv.so from the
# top-level dfkv CMake and (b) build/install the connector wheel.

# Repo root = two levels up from integration/lmcache.
DFKV_ROOT ?= $(abspath $(CURDIR)/../..)
BUILD_DIR ?= $(DFKV_ROOT)/build-rdma
RDMA      ?= ON
JOBS      ?= $(shell nproc)

.PHONY: help lib wheel install install-dev test clean

help:
	@echo "Targets:"
	@echo "  lib          build libdfkv.so (RDMA=$(RDMA)) into $(BUILD_DIR)"
	@echo "  wheel        build the pure-python connector wheel into dist/"
	@echo "  install      pip install . (the connector) into the active env"
	@echo "  install-dev  pip install -e . (editable)"
	@echo "  test         run the connector unit tests (pytest)"
	@echo "  clean        remove build/ dist/ *.egg-info"
	@echo ""
	@echo "After 'make lib', point the connector at the library:"
	@echo "  export DFKV_LIB=$(BUILD_DIR)/libdfkv.so"

lib:
	cmake -S $(DFKV_ROOT) -B $(BUILD_DIR) -DDFKV_WITH_RDMA=$(RDMA) -DDFKV_BUILD_TESTS=OFF
	cmake --build $(BUILD_DIR) --target dfkv -j$(JOBS)
	@echo "Built $(BUILD_DIR)/libdfkv.so"

wheel:
	pip wheel . --no-deps -w dist/

install:
	pip install .

install-dev:
	pip install -e .

test:
	DFKV_LIB=$(BUILD_DIR)/libdfkv.so pytest -q tests/

clean:
	rm -rf build dist *.egg-info src/*.egg-info
