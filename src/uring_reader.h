/* uring_reader — a thin, single-owner io_uring wrapper for the RDMA server's
 * async GET (kRange) disk-read path. ADDITIVE + env-gated: only used when the
 * server is built with -DDFKV_WITH_URING and started with DFKV_SERVER_URING=1.
 *
 * Design = Mooncake's batch_read pattern (uring_file.cpp) adapted to dfkv's
 * per-WaitComp completion batch:
 *  - One ring per connection serve loop (single owner => no locking).
 *  - BatchRead() submits up to QUEUE_DEPTH independent O_DIRECT preads at once
 *    (each its own fd/buffer/offset), then waits for the WHOLE batch to complete
 *    before returning. Concurrency comes from QD>1 reads in flight; ordering is
 *    irrelevant inside the ring because every read lands in its own buffer.
 *  - The serve loop only PostSendScatters AFTER BatchRead() returns (all reads
 *    done), iterating the descriptors in arrival order — so replies are strictly
 *    in request order with no reorder buffer.
 *
 * Header-only and #ifdef'd on DFKV_WITH_URING so the rest of the server compiles
 * unchanged when liburing is absent. */
#ifndef DFKV_URING_READER_H_
#define DFKV_URING_READER_H_

#ifdef DFKV_WITH_URING

#include <liburing.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace dfkv {

// Single-owner io_uring read ring. Not thread-safe by design (one serve loop
// owns one instance). QD bounds the number of reads in flight per batch.
class UringReader {
 public:
  // One independent read in a batch. `result` is filled by BatchRead (>=0 bytes
  // read, <0 = -errno) so the caller can validate each read after the batch.
  struct ReadDesc {
    int fd = -1;
    void* buf = nullptr;
    unsigned len = 0;
    uint64_t off = 0;
    long result = 0;  // out: cqe->res for this read
  };

  explicit UringReader(unsigned queue_depth) {
    // Default setup flags: portable across the 5.15 kernels in the fleet.
    int ret = io_uring_queue_init(queue_depth, &ring_, 0);
    ok_ = (ret == 0);
    depth_ = queue_depth;
  }

  ~UringReader() {
    if (ok_) io_uring_queue_exit(&ring_);
  }

  UringReader(const UringReader&) = delete;
  UringReader& operator=(const UringReader&) = delete;

  bool ok() const { return ok_; }
  unsigned depth() const { return depth_; }

  // Submit up to QUEUE_DEPTH reads at a time (each at its own fd/offset/buffer),
  // wait for the whole sub-batch, fill each desc.result, then repeat until all
  // `cnt` descs are done. Returns true if every read was submitted+reaped (a
  // per-read short/EOF/error is recorded in desc.result, NOT a hard failure — the
  // caller validates each). Returns false only on a submit/wait infrastructure
  // failure (the caller then falls back to the synchronous path for safety).
  bool BatchRead(ReadDesc* descs, int cnt) {
    if (!ok_ || cnt <= 0) return false;
    int idx = 0;
    while (idx < cnt) {
      const int batch =
          std::min(cnt - idx, static_cast<int>(depth_));
      // Use the descriptor's array index as user_data so we can map each CQE
      // back to its desc regardless of completion order.
      for (int i = 0; i < batch; ++i) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (!sqe) return false;  // SQ exhausted unexpectedly (batch <= depth_)
        ReadDesc& d = descs[idx + i];
        io_uring_prep_read(sqe, d.fd, d.buf, d.len, d.off);
        // Use the void* user_data API (not set_data64) so we build against
        // liburing >= 2.0 (the _data64 variants only exist in liburing >= 2.2).
        io_uring_sqe_set_data(
            sqe, reinterpret_cast<void*>(static_cast<uintptr_t>(idx + i)));
      }
      int submitted = io_uring_submit_and_wait(&ring_, batch);
      if (submitted < 0) return false;
      // Reap exactly `batch` completions, routing each to its desc by user_data.
      int reaped = 0;
      while (reaped < batch) {
        struct io_uring_cqe* cqe = nullptr;
        int w = io_uring_wait_cqe(&ring_, &cqe);
        if (w < 0 || cqe == nullptr) return false;
        const uint64_t di = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(io_uring_cqe_get_data(cqe)));
        if (di < static_cast<uint64_t>(cnt)) descs[di].result = cqe->res;
        io_uring_cqe_seen(&ring_, cqe);
        ++reaped;
      }
      idx += batch;
    }
    return true;
  }

 private:
  struct io_uring ring_{};
  bool ok_ = false;
  unsigned depth_ = 0;
};

}  // namespace dfkv

#endif  // DFKV_WITH_URING
#endif  // DFKV_URING_READER_H_
