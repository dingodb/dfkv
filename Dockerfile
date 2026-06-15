# Build a portable dfkv image (TCP transport). For RDMA add libibverbs-dev
# and -DDFKV_WITH_RDMA=ON.
FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build g++ git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DDFKV_BUILD_TESTS=OFF \
    -DDFKV_STATIC_LIBSTDCXX=ON && cmake --build build -j && cmake --install build --prefix /out

FROM ubuntu:24.04
COPY --from=build /out/ /usr/local/
RUN ldconfig
EXPOSE 12000
ENTRYPOINT ["dfkv_server"]
CMD ["--dir", "/data", "--port", "12000", "--cap", "6442450944"]
