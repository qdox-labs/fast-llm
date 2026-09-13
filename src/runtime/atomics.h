// Cross-engine atomics for unified memory. One contract, two implementations:
//   host   : GCC __atomic builtins (Thor: hostNativeAtomicSupported=YES makes
//            host atomics coherent with the GPU over unified memory)
//   device : atomic*_system intrinsics; ordering via __threadfence_system()
//            (relaxed intrinsic + explicit fence = release/acquire pairing,
//            the pattern proven by the doorbell probe)
// All mutable cross-engine state MUST go through these helpers.
#pragma once
#include <cstdint>

#if defined(__CUDACC__)
#define FL_HD __host__ __device__ __forceinline__
#else
#define FL_HD inline
#endif

namespace fastllm {

FL_HD void fl_cpu_relax() {
#if defined(__CUDA_ARCH__)
    __nanosleep(64);
#elif defined(__x86_64__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    ;
#endif
}

// P8.3 scoped atomics. Compile-time so the A/B costs nothing at runtime
// (build a second tree with -DFASTLLM_SCOPED_ATOMICS=1, as P7.3 did for 110a).
//
// WHY. libcu++ emits `ld.acquire.sys.b32` as ONE instruction for
// cuda::atomic_ref<uint32_t, thread_scope_system>::load(memory_order_acquire)
// on sm_70+, and falls back to volatile-load + membar ONLY below sm_70
// (NV_DISPATCH_TARGET(NV_PROVIDES_SM_70,...) in CCCL cuda_ptx_generated.h).
// The legacy form here is therefore the PRE-VOLTA lowering, hand-written on
// sm_110. Worse, __threadfence_system() is documented-equivalent to
// cuda::atomic_thread_fence(memory_order_seq_cst, thread_scope_system) =
// fence.sc.sys - the STRONGEST fence in the ISA, ordering every prior and
// subsequent memory operation of the thread - paid per ring cell to observe a
// single 32-bit word. Measured on Thor (P8.1): 432.86 ns per system fence,
// 258.57 ns device, 10.64 ns block.
//
// Scope is unchanged (.sys), so this is NOT the correctness-relevant axis that
// P1.0 burned us on: system scope is required because a HOST thread is one of
// the communicating agents, and that is untouched here. What changes is the
// ORDERING STRENGTH (seq_cst -> acquire/release) and the instruction count
// (2 -> 1). Both are strictly weaker/fewer for the same guarantee.
// DEFAULT ON since P8.3. Measured, shipped-cell shape, 8 interleaved rounds:
// copy-spec (the anchor) 82.375 -> 84.416 = +2.48%, 8/8, ranges NON-OVERLAPPING,
// and sd 0.714 -> 0.159 (4.5x more reproducible); m=1 +0.03% (neutral); prose
// -1.31%, inside that cell's own documented ~3% noise floor (it produces 6
// distinct token streams in 6 identical runs - see P8.0d). Token-identical on
// the deterministic copy prompt at m=1 and m=8, all six unit suites green, and
// verified in the SASS: MEMBAR 46 -> 31.
//
// Build with -DFASTLLM_SCOPED_ATOMICS=0 to get the legacy lowering back for an
// A/B. Unlike queue fence-light, this changes only the INSTRUCTION LOWERING and
// not the claim algorithm, so it does not shift tiles between the CPU pool and
// the GPU - which is exactly why no cell regresses.
#ifndef FASTLLM_SCOPED_ATOMICS
#define FASTLLM_SCOPED_ATOMICS 1
#endif

FL_HD uint32_t fl_load_acquire_u32(const uint32_t * p) {
#if defined(__CUDA_ARCH__)
#if FASTLLM_SCOPED_ATOMICS
    uint32_t v;
    asm volatile("ld.acquire.sys.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
#else
    uint32_t v = *(volatile const uint32_t *)p;
    __threadfence_system();
    return v;
#endif
#else
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#endif
}

FL_HD void fl_store_release_u32(uint32_t * p, uint32_t v) {
#if defined(__CUDA_ARCH__)
#if FASTLLM_SCOPED_ATOMICS
    asm volatile("st.release.sys.b32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
#else
    __threadfence_system();
    *(volatile uint32_t *)p = v;
#endif
#else
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
#endif
}

FL_HD uint64_t fl_load_acquire_u64(const uint64_t * p) {
#if defined(__CUDA_ARCH__)
#if FASTLLM_SCOPED_ATOMICS
    uint64_t v;
    asm volatile("ld.acquire.sys.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
#else
    uint64_t v = *(volatile const uint64_t *)p;
    __threadfence_system();
    return v;
#endif
#else
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#endif
}

FL_HD void fl_store_release_u64(uint64_t * p, uint64_t v) {
#if defined(__CUDA_ARCH__)
#if FASTLLM_SCOPED_ATOMICS
    asm volatile("st.release.sys.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
#else
    __threadfence_system();
    *(volatile uint64_t *)p = v;
#endif
#else
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
#endif
}

// returns true and updates *expected like __atomic_compare_exchange
FL_HD bool fl_cas_u32(uint32_t * p, uint32_t * expected, uint32_t desired) {
#if defined(__CUDA_ARCH__)
    uint32_t old = atomicCAS_system((unsigned int *)p, *expected, desired);
    if (old == *expected) return true;
    *expected = old;
    return false;
#else
    return __atomic_compare_exchange_n(p, expected, desired, false,
                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
#endif
}

// fetch_sub with acq_rel: the completion-publish primitive (dep counters)
FL_HD int32_t fl_fetch_sub_i32(int32_t * p, int32_t v) {
#if defined(__CUDA_ARCH__)
    __threadfence_system();                       // release: y writes visible first
    return (int32_t)atomicSub_system((int *)p, v);
#else
    return __atomic_fetch_sub(p, v, __ATOMIC_ACQ_REL);
#endif
}

FL_HD uint32_t fl_fetch_add_u32(uint32_t * p, uint32_t v) {
#if defined(__CUDA_ARCH__)
    return atomicAdd_system((unsigned int *)p, v);
#else
    return __atomic_fetch_add(p, v, __ATOMIC_ACQ_REL);
#endif
}

FL_HD uint64_t fl_fetch_add_u64(uint64_t * p, uint64_t v) {
#if defined(__CUDA_ARCH__)
    return atomicAdd_system((unsigned long long *)p, (unsigned long long)v);
#else
    return __atomic_fetch_add(p, v, __ATOMIC_ACQ_REL);
#endif
}

// ---- P7.0: relaxed accessors + standalone fences -------------------------
// fl_load_acquire_u32 carries a __threadfence_system() on EVERY device load.
// That is correct but pays per-load for an ordering that is only needed once
// per synchronisation point. Queue::pop_bulk scans up to 16 cells per claim,
// so it issued ~17 system-scope fences where 1 suffices; measured, that costs
// the GPU up to 2.41x its claim throughput .
//
// These give the fence-based idiom instead: relaxed loads, then ONE fence.
//   acquire : relaxed_load(x)  ...  fl_fence_acquire()  ...  read(payload)
//   release : write(payload)   ...  fl_fence_release()  ...  relaxed_store(x)
// Both are the standard C++11 fence forms and are exactly as strong as the
// per-op versions, provided the fence sits between the relaxed access and the
// payload access ON THE SAME THREAD. Use only where that is checked.
FL_HD uint32_t fl_load_relaxed_u32(const uint32_t * p) {
#if defined(__CUDA_ARCH__)
    return *(volatile const uint32_t *)p;
#else
    return __atomic_load_n(p, __ATOMIC_RELAXED);
#endif
}

FL_HD void fl_store_relaxed_u32(uint32_t * p, uint32_t v) {
#if defined(__CUDA_ARCH__)
    *(volatile uint32_t *)p = v;
#else
    __atomic_store_n(p, v, __ATOMIC_RELAXED);
#endif
}

FL_HD void fl_fence_acquire() {
#if defined(__CUDA_ARCH__)
#if FASTLLM_SCOPED_ATOMICS
    asm volatile("fence.acq_rel.sys;" ::: "memory");   // not fence.sc.sys
#else
    __threadfence_system();
#endif
#else
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
#endif
}

FL_HD void fl_fence_release() {
#if defined(__CUDA_ARCH__)
#if FASTLLM_SCOPED_ATOMICS
    asm volatile("fence.acq_rel.sys;" ::: "memory");   // not fence.sc.sys
#else
    __threadfence_system();
#endif
#else
    __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
}

} // namespace fastllm
