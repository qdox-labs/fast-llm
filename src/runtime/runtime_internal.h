// Shared runtime state (host + device views). Lives in the arena so both
// engines address the same bytes. Single-writer discipline per field class;
// cross-engine mutation only via atomics.h helpers.
#pragma once
#include "fastllm/fastllm.h"
#include "queue.h"
#include "quants.h"
#include "decode_params.h"
#include <ctime>

namespace fastllm {

inline constexpr int      kMaxGpuBlocks   = 64;
inline constexpr uint32_t kMaxHandoff     = 8192;
inline constexpr uint32_t kQuitTask       = 0xFFFFFFFEu;
inline constexpr uint32_t kGpuMaxBatch    = 16;   // pop_bulk claim cap

// timestamp domain tag in bit 0 (ns granularity loss is irrelevant)
inline constexpr uint64_t kTsGpuBit = 1;

struct alignas(128) BlockMetrics {
    uint64_t tiles;
    uint64_t bytes;
    uint64_t busy_ns;
    uint64_t stolen;
    // P8.0 time decomposition, leader-stamped, gated on RtShared::prof_gpu.
    // The four ns buckets partition the persistent block's loop time:
    //   claim (pop_bulk scan) + idle (nanosleep on an empty claim)
    //   + exec (the task loop) + epi (consumer dedup, dep decrement, push)
    // Compare their sum against wall_ns x active blocks for the busy/idle
    // split. busy_ns is left as-is (exec+epi) so old numbers stay comparable.
    uint64_t claim_ns;
    uint64_t sleep_ns;
    uint64_t exec_ns;
    uint64_t epi_ns;
    uint64_t polls;      // loop iterations entered
    uint64_t empty;      // iterations that claimed nothing
    uint64_t retry;      // P8.2: pop_bulk passes BEYOND the first (lost race
                         // or lost CAS) - separates "claim is slow" from
                         // "claim is repeated" 
};
// 80 B used of the 128 B alignment quantum - no new cache lines, so the
// false-sharing profile between blocks is unchanged by this instrumentation.

struct alignas(128) RtShared {
    // queues: [0] GPU-affine, [1] CPU-affine (GPU may steal), [2] CPU-ONLY
    // op tasks (P2.2 depth-wall fix: leader-serialized op kinds ran 100x
    // slower when a starved GPU block stole them - measured 5.3 ms vs 50 us
    // per ATTN_HEAD at depth 200. The GPU never pops q[2]), [3] GPU-ONLY
    // codebook dots (P5.2, mirror of q[2]: IQ2/IQ3 have no CPU int8 path -
    // fl_dot_i8_any returns 0 for them - so a CPU steal would silently
    // corrupt output. The CPU never pops q[3]).
    Queue    q[4];

    // control (system-scope atomics)
    uint32_t run_flag;     // 0 -> GPU blocks exit, CPU workers exit
    uint32_t active;       // a run() is in flight (CPU workers spin, not park)
    uint32_t done_flag;    // terminal SIGNAL executed
    uint32_t n_done;       // completed tasks this run (quiesce guard)

    // frozen graph being executed
    Task *   tasks;
    uint32_t n_tasks;
    uint32_t terminal;

    // config mirror for device
    uint32_t gpu_enabled;
    uint32_t collect_handoff;
    uint32_t fast_kernels;
    uint32_t gpu_batch;      // tiles claimed per pop_bulk, [1, kGpuMaxBatch]
    uint32_t int8_dots;      // P1.1: quant tiles use q8_1-activation int dots
    uint32_t i8_gpu_mask;    // P1.2: per-fmt int8-path bits, GPU engine
    uint32_t i8_cpu_mask;    // P1.2: per-fmt int8-path bits, CPU engine
    uint32_t i8k_gpu_mask;   // P1.5: q8_K-appendix bits (K fmts), GPU engine
    uint32_t i8k_cpu_mask;   // P1.5: q8_K-appendix bits (K fmts), CPU engine
    uint32_t mc_legacy;      // P2.5: 1 = P2.4 per-column loops on nx>1 tiles
    // P8.5: 1 = STRICT static assignment. Each engine works only the queues
    // it owns and never steals the other's: GPU takes q[3] then q[0]; CPU
    // takes q[2] then q[1]. The task->queue map (fl_task_queue) is already a
    // pure function of kind/fmt/affinity and the graph is frozen once and
    // reused every token, so the assignment was ALWAYS precomputed and
    // deterministic - stealing is the only thing that made it a race.
    uint32_t no_steal;
    // P9.2 grid truth. gpu_start clamps `blocks` to the driver's residency cap
    // and may halve `threads`, both silently - so the host's gpu_blocks_ holds
    // a PRE-clamp value that can exceed the real grid. Nothing may index a
    // per-block structure from anything but this field, which gpu_start
    // release-stores with the ACTUAL launched count before the launch.
    // 0 = the executor is not running.
    uint32_t gpu_blocks;
    // P9.2 arrival. gpu_blocks says how many blocks were launched; this says
    // which ones have actually started executing. RunMetrics::gpu_blocks_active
    // could not answer that - it derives from `polls`, which only increments
    // under prof. Each block sets its slot once at kernel entry.
    uint32_t gpu_arrived[kMaxGpuBlocks];
    // P9.2c per-block GPU rings. Same Vyukov Queue type as q[0..3] - the point
    // is NOT a new lock-free algorithm, it is that each ring has ONE consumer
    // (its owning block) plus rare CPU thieves, instead of 20 GPU blocks and 12
    // CPU threads all CASing one tail. P8.2 measured 4.3 pop_bulk passes per
    // successful claim; a CAS that is never contended cannot fail.
    // Only gq[0 .. gpu_blocks) are initialised. 0 = feature off, use q[0].
    uint32_t perblock_q;
    // P9.2d: consecutive task ids per ring. 1 = pure round-robin (`id % nb`),
    // which scatters a batch's sibling tiles across every ring so pop_bulk can
    // only ever claim one at a time - the suspected cause of P9.2c's -1.48% on
    // the anchor, where wide batches are exactly what bulk claiming exists for.
    // Chunking keeps a run of ids together so one claim gets several.
    uint32_t perblock_chunk;
    Queue    gq[kMaxGpuBlocks];
    uint32_t prof_gpu;       // P8.0: 1 = leader-stamp the time decomposition
                             // and per-kind dot attribution (FASTLLM_PROF_GPU)

    // clock calibration: gpu_globaltimer - cpu_steady, ns (host-written)
    int64_t  gpu_clock_offset;

    // handoff samples (cpu-domain ns deltas), atomic append
    uint32_t n_samples;
    uint64_t samples[kMaxHandoff];

    // per-GPU-block metrics
    BlockMetrics gpu_m[kMaxGpuBlocks];

    // P2.2 depth-wall instrumentation: per-block per-kind attribution (the
    // CPU pool always had kind_ns; GPU-stolen op-tasks were invisible)
    uint64_t gpu_kind_ns[kMaxGpuBlocks][kNumTaskKinds];
    uint32_t gpu_kind_n [kMaxGpuBlocks][kNumTaskKinds];

    // calibration mailbox (probe idiom)
    uint32_t calib_ping;
    uint32_t calib_pong;
    uint64_t calib_gpu_ts;
};

// ---- shared task execution (host + device single-thread flavor) ----------
// STREAM_DOT: y[r] += dequant_dot(fmt, w_row_r, x), canonical ascending order
// MERGE     : y[i]  = sum_p x[p*rows + i], p in [0, cols)
// SIGNAL    : none
//
// HOST callers must use exec_task_scalar_host() (defined once, in graph.cpp):
// aarch64 gcc emits per-TU codegen variants of this header-inline body whose
// FP results differ (found on thor, P1.0) — bitwise canonical equality only
// holds when every host caller links the SAME instantiation. The FL_HD copy
// below exists for the DEVICE canonical path (gpu_exec.cu), paired with
// --fmad=false so device matches host.
void exec_task_scalar_host(Task & t);

// STREAM_DOT_IDX weight resolution (expert indirection); shared by every
// executor path so canonical == fast == device for the pointer math.
FL_HD const uint8_t * fl_idx_w(const Task & t) {
    const uint32_t id = *(const uint32_t *)t.aux;
    return (const uint8_t *)t.w + (size_t)id * t.aux2;
}

// P2.2: queue routing shared by host and device publishers. Op kinds
// (>= RMSNORM, leader-serialized on device) are CPU-only -> q[2]; everything
// else routes by affinity as before. Both engines PUSH to any queue; only
// CPU workers POP q[2].
// P5.2: codebook dots are GPU-ONLY -> q[3] whenever the GPU is available.
// With gpu_enabled == 0 they fall back to q[1]: the CPU's CANONICAL path
// (fl_dot_row_any) does implement IQ2/IQ3 at ~5.5 Gw/s, so cpu-only runs stay
// correct; the int8 path does not, which decode_main gates before it starts.
FL_HD int fl_task_queue(const Task & t, uint32_t gpu_enabled) {
    if (t.kind >= TaskKind::RMSNORM) return 2;
    if (fl_fmt_is_codebook(t.fmt)) return gpu_enabled ? 3 : 1;
    if (t.affinity == Engine::GPU) return 0;   // legacy: q[0] even gpu-off
    if (t.affinity == Engine::CPU) return 1;   //         (CPU steals it)
    return gpu_enabled ? 0 : 1;
}

// Timestamp in each engine's own domain; kTsGpuBit tags which. Host reads the
// monotonic clock, device reads %globaltimer (64 ns resolution, 78 ns per read
// measured on Thor in P8.0 - hence the collect_handoff guard at every caller).
FL_HD uint64_t fl_now_ts() {
#if defined(__CUDA_ARCH__)
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#else
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

// P9.2b: THE one publish path. Every enqueue in the engine funnels through
// here - host roots, the CPU worker's consumer loop, and the GPU epilogue - so
// a routing change is one edit instead of three that have to agree. Before
// this they had already drifted: the device hardcodes gpu_enabled=1 while the
// host passes sh->gpu_enabled, and the two stamp handoffs on opposite
// predicates.
//
// Behaviour is preserved EXACTLY, including two asymmetries that are
// deliberately NOT fixed here because they would change a measured metric:
//   - the host does not stamp a publish to q[3], although CPU -> GPU-only
//     codebook is a genuine cross-engine handoff (those samples are missing);
//   - the device DOES stamp a publish to q[3], which is GPU -> GPU, and it is
//     later recorded as though it were GPU -> CPU.
//
// Returns false only when the engine is tearing down (run_flag cleared) - the
// caller must not treat that as a successful enqueue.
FL_HD bool fl_publish(RtShared * sh, uint32_t id, bool from_gpu) {
    Task & c = sh->tasks[id];
    // The device only runs when the GPU is enabled, so hardcoding 1 there is
    // equivalent to reading the mirror - kept explicit so the two publishers
    // are visibly computing the same destination for the same task.
    const int cq = fl_task_queue(c, from_gpu ? 1u : sh->gpu_enabled);
    // P9.2c: GPU-affine work (cq == 0) goes to the owning block's private ring.
    // The index is a pure function of the FROZEN task id, so host and device
    // publishers always agree and the assignment stays reproducible run to run
    // - the determinism property P8.5 relies on. gpu_blocks is the REAL
    // launched grid (P9.2a); routing above it would orphan the task forever.
    const uint32_t nb = sh->perblock_q ? sh->gpu_blocks : 0u;
    if (cq == 0 && nb) {
        const uint32_t ch = sh->perblock_chunk ? sh->perblock_chunk : 1u;
        Queue & r = sh->gq[(id / ch) % nb];
        if (sh->collect_handoff && !from_gpu && sh->gpu_enabled)
            c.handoff_ts = fl_now_ts() & ~kTsGpuBit;
        while (!r.push(id)) {
            if (!fl_load_acquire_u32(&sh->run_flag)) return false;
            fl_cpu_relax();
        }
        return true;
    }
    if (sh->collect_handoff) {
        if (from_gpu) {
            if (cq != 0) c.handoff_ts = fl_now_ts() | kTsGpuBit;   // GPU -> CPU
        } else if (cq == 0 && sh->gpu_enabled) {
            c.handoff_ts = fl_now_ts() & ~kTsGpuBit;               // CPU -> GPU
        }
    }
    while (!sh->q[cq].push(id)) {
        // run_flag is the only teardown latch; a push spin that ignores it
        // wedges stop() (P9.2a).
        if (!fl_load_acquire_u32(&sh->run_flag)) return false;
        fl_cpu_relax();
    }
    return true;
}

FL_HD void exec_task_scalar(Task & t) {
    if (t.kind == TaskKind::STREAM_DOT || t.kind == TaskKind::STREAM_DOT_IDX) {
        const uint8_t * w = t.kind == TaskKind::STREAM_DOT_IDX
                          ? fl_idx_w(t) : (const uint8_t *)t.w;
        float * y = (float *)t.y;
        const uint64_t rb = fl_row_bytes(t.fmt, t.cols);
        // P2.4: nx activation columns share each weight-row read; per-(row,
        // column) accumulation order is unchanged, so canonical output equals
        // that of nx single-column tiles bitwise.
        const uint32_t nx = t.nx ? t.nx : 1;
        for (uint32_t r = 0; r < t.rows; ++r) {
            const uint8_t * row = w + (size_t)r * rb;
            for (uint32_t p = 0; p < nx; ++p) {
                const float * x = (const float *)t.x + (size_t)p * t.xstride;
                y[(size_t)p * t.ystride + r] +=
                    fl_dot_row_any(t.fmt, row, x, t.cols);
            }
        }
    } else if (t.kind == TaskKind::RMSNORM) {
        fl_op_rmsnorm(*(const RmsNormP *)t.aux);
    } else if (t.kind == TaskKind::ADD2) {
        fl_op_add2(*(const Add2P *)t.aux);
    } else if (t.kind == TaskKind::ROPE_DS2) {
        fl_op_rope_ds2(*(const RopeP *)t.aux);
    } else if (t.kind == TaskKind::KV_APPEND) {
        fl_op_kv_append(*(const KvAppendP *)t.aux);
    } else if (t.kind == TaskKind::ATTN_HEAD) {
        fl_op_attn_head(*(const AttnHeadP *)t.aux);
    } else if (t.kind == TaskKind::ROUTER_SEL) {
        fl_op_router_sel(*(const RouterP *)t.aux);
    } else if (t.kind == TaskKind::SILU_MUL) {
        fl_op_silu_mul(*(const SiluMulP *)t.aux);
    } else if (t.kind == TaskKind::WMERGE) {
        fl_op_wmerge(*(const WmergeP *)t.aux);
    } else if (t.kind == TaskKind::MERGE) {
        const float * x = (const float *)t.x;
        float * y = (float *)t.y;
        for (uint32_t i = 0; i < t.rows; ++i) {
            float acc = 0.f;
            for (uint32_t p = 0; p < t.cols; ++p) acc += x[(size_t)p * t.rows + i];
            y[i] = acc;
        }
    } else if (t.kind == TaskKind::ACT_Q8) {
        fl_exec_act_q8(t);
    }
    // SIGNAL: nothing
}

#if defined(FASTLLM_CUDA)
// implemented in gpu_exec.cu
void * gpu_make_stream();
// persist_bytes: extent of the contiguous [RtShared | q[0..3] cells] arena
// region, for the P6.3 L2 persisting-window probe (0 = disabled).
// Returns the ACTUAL number of blocks launched, which may be below `blocks`
// (residency clamp) - see RtShared::gpu_blocks. Also release-stores it there.
int gpu_start(RtShared * sh, int blocks, int threads, void * stream,
              size_t persist_bytes);
int  gpu_resident_blocks(int threads);
void gpu_stop_sync(void * stream);            // joins + destroys the stream
int64_t gpu_calibrate_clock(RtShared * sh, void * stream);
int  gpu_default_blocks();
#endif

} // namespace fastllm
