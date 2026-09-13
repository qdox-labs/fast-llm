// Unified-memory bump arena. cudaMallocManaged under CUDA (one address space
// for both engines); aligned malloc otherwise. Allocation is build-time only,
// never on the token path.
#include "fastllm/fastllm.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(FASTLLM_CUDA)
#include <cuda_runtime.h>
#endif

namespace fastllm {

namespace {

class ArenaImpl final : public Arena {
public:
    ArenaImpl(void * base, size_t bytes, bool cuda)
        : base_((char *)base), cap_(bytes), cuda_(cuda) {}

    ~ArenaImpl() override {
#if defined(FASTLLM_CUDA)
        if (cuda_) { cudaFree(base_); return; }
#endif
        free(base_);
    }

    void * alloc(size_t bytes, size_t align) override {
        std::lock_guard<std::mutex> lk(mu_);
        size_t off = (used_ + align - 1) & ~(align - 1);
        if (off + bytes > cap_) {
            fprintf(stderr, "fastllm: arena exhausted (%zu + %zu > %zu)\n",
                    off, bytes, cap_);
            abort();
        }
        used_ = off + bytes;
        return base_ + off;
    }

    size_t used() const override { return used_; }

private:
    char * base_;
    size_t cap_;
    size_t used_ = 0;
    bool   cuda_;
    std::mutex mu_;
};

} // namespace

std::unique_ptr<Arena> Arena::create(size_t bytes) {
#if defined(FASTLLM_CUDA)
    void * p = nullptr;
    cudaError_t e = cudaMallocManaged(&p, bytes);
    if (e != cudaSuccess) {
        fprintf(stderr, "fastllm: cudaMallocManaged(%zu) failed: %s\n",
                bytes, cudaGetErrorString(e));
        abort();
    }
    memset(p, 0, bytes);
    return std::make_unique<ArenaImpl>(p, bytes, true);
#else
    void * p = nullptr;
    if (posix_memalign(&p, 4096, bytes) != 0 || !p) {
        fprintf(stderr, "fastllm: arena alloc(%zu) failed\n", bytes);
        abort();
    }
    memset(p, 0, bytes);
    return std::make_unique<ArenaImpl>(p, bytes, false);
#endif
}

} // namespace fastllm
