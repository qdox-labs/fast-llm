// Persistent GPU executor: launched once, all blocks loop pop -> exec ->
// publish over the same unified-memory protocol the CPU pool uses. No stream
// control-plane in steady state; parking = run_flag -> blocks exit, start()
// relaunches. Mailbox + clock-calibration idioms from the doorbell probe.
#include "runtime_internal.h"
#include "kernels_gpu.h"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

namespace fastllm {

#define FL_CUCHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "fastllm cuda %s:%d %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e)); abort(); } } while (0)

__device__ __forceinline__ uint64_t gpu_now() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Fast STREAM_DOT kernels (P0.9 f32 warp shape + P1.0 quant formats) live in
// kernels_gpu.h (included at the top, shared with tools/membench.cu). All
// warp-per-row, lane-coalesced, shfl-reduce; tolerance-validated vs canonical.

// Column-split STREAM_DOT for small-row tiles (fast mode only): all threads
// stream regardless of row count. Threads are grouped tpr-per-row (tpr = the
// largest power of two <= blockDim.x/rows); each thread accumulates a column
// stripe, then a per-group shared-memory tree reduces log2(tpr) steps with
// all rows reducing concurrently. Fixes the P0.6 "MLP death" (a 4-row tile
// left 4/256 threads active = 4.6 GB/s). Summation order differs from
// canonical -> tolerance-validated (fast path only).
__device__ static void stream_dot_colsplit(Task & t, float * red) {
    const float * w = (const float *)t.w;
    const float * x = (const float *)t.x;
    float * y = (float *)t.y;
    uint32_t tpr = blockDim.x / t.rows;
    tpr = 1u << (31 - __clz(tpr));            // pow2, >= 1
    const uint32_t g    = threadIdx.x / tpr;  // row group
    const uint32_t lane = threadIdx.x % tpr;
    float acc = 0.f;
    if (g < t.rows) {
        const float * row = w + (size_t)g * t.cols;
        for (uint32_t c = lane; c < t.cols; c += tpr)
            acc = fmaf(row[c], x[c], acc);
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t s = tpr >> 1; s > 0; s >>= 1) {
        if (lane < s && g < t.rows)
            red[g * tpr + lane] += red[g * tpr + lane + s];
        __syncthreads();
    }
    if (lane == 0 && g < t.rows) y[g] += red[g * tpr];
    __syncthreads();                          // red reused by next tile
}

// block-parallel task execution; every thread participates, leader publishes.
// Row-per-thread scalar keeps the canonical per-row order (bitwise vs serial);
// column-split is fast-mode-only for tiles whose rows would starve the block.
__device__ static void exec_task_block(Task & t, bool fast, uint32_t i8_mask,
                                       uint32_t i8k_mask, uint32_t mc_legacy,
                                       float * red) {
    if (t.kind == TaskKind::ACT_Q8) {           // canonical in every mode
        gact_q8(t);
        return;
    }
    // P2.0: expert indirection -> plain STREAM_DOT on a local copy (the id
    // slot is written by a ROUTER_SEL task ordered before this one by dep
    // edges; the publish fence makes the id visible system-wide).
    Task tt;                       // IDX: resolve once, no recursion (device
    Task & tr = t.kind == TaskKind::STREAM_DOT_IDX ? tt : t;   // stack budget)
    if (t.kind == TaskKind::STREAM_DOT_IDX) {
        tt = t;
        tt.kind = TaskKind::STREAM_DOT;
        tt.w    = fl_idx_w(t);
    }
    // P2.0 decode ops: canonical scalar by contract, leader-thread execution.
    // These are CPU-affine; the GPU sees them only via stealing. Correct but
    // serialized on one thread - a known lever, not a bug (docs/BENCH.md).
    if (t.kind >= TaskKind::RMSNORM) {
        __syncthreads();
        if (threadIdx.x == 0) exec_task_scalar(t);
        __syncthreads();
        return;
    }
    if (tr.kind == TaskKind::STREAM_DOT) {
        // P2.4 multi-column: run the same tile once per activation column.
        // A tile is ~rowsz x row_bytes (256 x 1152 B = 295 KB for 2048-col
        // q4_K), so columns 1..nx-1 hit L2 rather than DRAM: the weight is
        // fetched from memory once per pass instead of once per position.
        // Per-(row,column) order is untouched -> identical results to nx
        // single-column tiles. (Register-level reuse inside the warp loop
        // would also cut L2 traffic; DRAM is the binding constraint here.)
        const uint32_t ncol = tr.nx ? tr.nx : 1;
        // P2.5: unpack-sharing multi-column kernel for the GPU's i8k q4_K
        // path (scale decode + nibble split once per superblock per lane,
        // columns in register chunks of 4). Other formats keep the per-column
        // loop below; FASTLLM_MC_LEGACY=1 keeps it for q4_K too.
        if (ncol > 1 && fast && !mc_legacy && tr.fmt == Fmt::Q4_K
            && ((i8k_mask >> (int)Fmt::Q4_K) & 1u)) {
            gdot_i8k_q4_K_mc(tr);
            return;
        }
        Task tc = tr;                      // no recursion: device stack budget
        for (uint32_t p = 0; p < ncol; ++p) {
            tc.x = (const void *)((const float *)tr.x + (size_t)p * tr.xstride);
            tc.y = (void *)((float *)tr.y + (size_t)p * tr.ystride);
            if (fast && tc.fmt != Fmt::F32
                && (((i8_mask | i8k_mask) >> (int)tc.fmt) & 1u)) {
                gdot_i8(tc, i8k_mask);
            } else if (fast && tc.fmt == Fmt::F32 && tc.rows <= 64 &&
                       (blockDim.x / tc.rows) >= 2) {
                stream_dot_colsplit(tc, red);
            } else if (fast && (tc.fmt != Fmt::F32 || (tc.cols & 3u) == 0)) {
                gdot_fast(tc);
            } else {
                // canonical: row-per-thread, per-row order == run_serial
                const uint8_t * w = (const uint8_t *)tc.w;
                const float * x = (const float *)tc.x;
                float * y = (float *)tc.y;
                const uint64_t rb = fl_row_bytes(tc.fmt, tc.cols);
                for (uint32_t r = threadIdx.x; r < tc.rows; r += blockDim.x)
                    y[r] += fl_dot_row_any(tc.fmt, w + (size_t)r * rb, x,
                                           tc.cols);
            }
        }
    } else if (t.kind == TaskKind::MERGE) {
        const float * x = (const float *)t.x;
        float * y = (float *)t.y;
        for (uint32_t i = threadIdx.x; i < t.rows; i += blockDim.x) {
            float acc = 0.f;
            for (uint32_t p = 0; p < t.cols; ++p) acc += x[(size_t)p * t.rows + i];
            y[i] = acc;
        }
    }
}

// P0.7 executor: batch-claim + coalesced publish.
//  - pop_bulk claims up to gpu_batch tiles with ONE tail CAS (P0.6 put the
//    per-tile claim/publish system-RMW cost at ~45 us; this divides it).
//  - publish coalesces dep decrements per unique consumer (a batch of sibling
//    row-block tiles usually shares one MERGE consumer -> one RMW), with one
//    system fence and one n_done add per batch.
//  - empty polls back off exponentially (200 ns -> 51 us): idle blocks were
//    hammering the queue tail lines shared with CPU pops (the co-streaming
//    interference suspect from P0.5).
// P5.3: bound 1024, not 512. 128 regs x 512 threads was exactly the 64K
// register file, so one block per SM and no room to widen; ptxas fits 64 regs
// under this bound. The dedup scratch below moved to shared for the same
// reason: it is leader-only but was allocated per thread (576 B of stack each,
// dynamically indexed -> local memory), which is what the spills were made of.
__global__ void __launch_bounds__(1024, 1) executor(RtShared * sh) {
    __shared__ uint32_t s_ids[kGpuMaxBatch];
    __shared__ uint32_t s_n;
    __shared__ uint32_t s_stolen;
    __shared__ uint32_t s_ucid[kGpuMaxBatch * kMaxConsumers];
    __shared__ uint16_t s_ucnt[kGpuMaxBatch * kMaxConsumers];
    __shared__ float    s_red[1024]; // sized for max block width (P1.4)
    BlockMetrics & bm = sh->gpu_m[blockIdx.x];
    uint32_t idle_ns = 200;
    // P8.0: block-uniform, read once into a register. Every stamp below is
    // leader-only and inserts no barrier, so warp overlap inside a batch is
    // preserved exactly as it is with profiling off.
    const bool prof = sh->prof_gpu != 0;
    // P9.2: unconditional arrival stamp (NOT prof-gated - the whole point is
    // that the host can trust it in a shipped run). One store per block, once.
    if (threadIdx.x == 0) fl_store_release_u32(&sh->gpu_arrived[blockIdx.x], 1u);
    // P9.2c: rotating peer cursor for ring stealing (leader-only, persists
    // across iterations). Starts offset by blockIdx so idle blocks do not all
    // converge on ring 0.
    uint32_t peer_rr = blockIdx.x + 1;

    for (;;) {
        uint64_t tc0 = 0;
        if (prof && threadIdx.x == 0) tc0 = gpu_now();
        if (threadIdx.x == 0) {
            s_stolen = 0;
            if (!fl_load_acquire_u32(&sh->run_flag)) {
                s_n = kQuitTask;
            } else {
                uint32_t k = sh->gpu_batch;
                uint32_t rt = 0;   // P8.2: retries across all three queues
                // P5.2: q[3] (GPU-only codebook dots) first - nothing else can
                // run them, so leaving them queued while we work q[0] would
                // stall every consumer waiting on an expert.
                uint32_t n = sh->q[3].pop_bulk(s_ids, k, &rt);
                // P9.2c: this block's PRIVATE ring, when enabled. One consumer
                // instead of 32, so the tail CAS is effectively uncontended -
                // gpu_retry collapsing toward zero is the direct proof the
                // change did what it claims. q[3] still goes first: P5.2's
                // liveness argument (nothing else can run codebook dots, so
                // leaving them queued stalls every consumer) is unchanged.
                if (n == 0) {
                    n = sh->perblock_q ? sh->gq[blockIdx.x].pop_bulk(s_ids, k, &rt)
                                       : sh->q[0].pop_bulk(s_ids, k, &rt);
                }
                // P9.2c: peers BEFORE q[1], mirroring today's q[0]-then-q[1]
                // priority - GPU-affine work should stay on the GPU. This is
                // the slow path only: a block reaches it having found its own
                // ring empty, so the fast path stays uncontended while load
                // imbalance between rings still gets corrected. P8.5 showed
                // removing cross-engine balance costs -39% to -52%; removing
                // GPU-to-GPU balance is the same class of mistake.
                if (n == 0 && sh->perblock_q && !sh->no_steal) {
                    const uint32_t nb = sh->gpu_blocks;
                    for (uint32_t i = 1; i < nb && n == 0; ++i) {
                        const uint32_t b = (peer_rr + i) % nb;
                        n = sh->gq[b].pop_bulk(s_ids, k, &rt);
                        if (n) { peer_rr = b; s_stolen = n; }
                    }
                }
                if (n == 0 && !sh->no_steal) {   // P8.5: q[1] is the CPU's
                    n = sh->q[1].pop_bulk(s_ids, k, &rt);
                    s_stolen = n;
                }
                s_n = n;
                if (prof) bm.retry += rt;
            }
        }
        if (prof && threadIdx.x == 0) {
            bm.claim_ns += gpu_now() - tc0;
            bm.polls    += 1;
        }
        __syncthreads();
        const uint32_t n = s_n;
        if (n == kQuitTask) return;
        if (n == 0) {
            uint64_t ti0 = 0;
            if (prof && threadIdx.x == 0) ti0 = gpu_now();
            __nanosleep(idle_ns);
            idle_ns = min(idle_ns * 2u, 51200u);
            if (prof && threadIdx.x == 0) {
                bm.sleep_ns += gpu_now() - ti0;
                bm.empty   += 1;
            }
            continue;
        }
        idle_ns = 200;

        uint64_t t0 = 0;
        if (threadIdx.x == 0) {
            t0 = gpu_now();
            if (sh->collect_handoff) {
                for (uint32_t i = 0; i < n; ++i) {
                    Task & t = sh->tasks[s_ids[i]];
                    if (!t.handoff_ts) continue;
                    uint64_t en = t.handoff_ts;
                    int64_t  en_cpu = (en & kTsGpuBit)
                        ? (int64_t)(en & ~kTsGpuBit) - sh->gpu_clock_offset
                        : (int64_t)en;
                    int64_t claim_cpu = (int64_t)t0 - sh->gpu_clock_offset;
                    if (claim_cpu > en_cpu) {
                        uint32_t idx = fl_fetch_add_u32(&sh->n_samples, 1);
                        if (idx < kMaxHandoff)
                            sh->samples[idx] = (uint64_t)(claim_cpu - en_cpu);
                    }
                    t.handoff_ts = 0;
                }
            }
        }
        __syncthreads();
        uint64_t te0 = 0;
        if (prof && threadIdx.x == 0) te0 = gpu_now();
        for (uint32_t i = 0; i < n; ++i) {
            Task & tk = sh->tasks[s_ids[i]];
            // P2.2 stamped only the leader-serialized op class (kind >=
            // RMSNORM); that branch is internally synchronized, so its stamps
            // are exact. P8.0 extends stamping to dot/merge under prof. Those
            // DO overlap across a batch (no barrier between iterations), so a
            // dot's per-task figure is "leader time inside task i", not the
            // block's wall time for it: the SUM over the batch is faithful --
            // it is the leader's own timeline -- while a single entry is
            // approximate. Still no barrier here, so the overlap that made
            // P2.2 leave these alone is itself preserved.
            const bool stamp = (tk.kind >= TaskKind::RMSNORM) || prof;
            uint64_t k0 = 0;
            if (stamp && threadIdx.x == 0) k0 = gpu_now();
            exec_task_block(tk, sh->fast_kernels != 0,
                            sh->i8_gpu_mask, sh->i8k_gpu_mask,
                            sh->mc_legacy, s_red);
            if (stamp && threadIdx.x == 0) {
                sh->gpu_kind_ns[blockIdx.x][(int)tk.kind] += gpu_now() - k0;
                sh->gpu_kind_n [blockIdx.x][(int)tk.kind] += 1;
            }
        }
        __syncthreads();
        // after the barrier: covers stragglers, not just the leader's warp
        if (prof && threadIdx.x == 0) bm.exec_ns += gpu_now() - te0;

        if (threadIdx.x == 0) {
            uint64_t tep0 = prof ? gpu_now() : 0;
            uint64_t bytes = 0;
            uint32_t * ucid = s_ucid;   // leader-only scratch, see kernel head
            uint16_t * ucnt = s_ucnt;
            uint32_t um = 0;
            for (uint32_t i = 0; i < n; ++i) {
                Task & t = sh->tasks[s_ids[i]];
                if (t.kind == TaskKind::STREAM_DOT || t.kind == TaskKind::STREAM_DOT_IDX) bytes += t.w_bytes;
                for (uint16_t k = 0; k < t.n_consumers; ++k) {
                    uint32_t c = t.consumers[k];
                    uint32_t j = 0;
                    for (; j < um; ++j) if (ucid[j] == c) break;
                    if (j == um) { ucid[um] = c; ucnt[um] = 0; ++um; }
                    ucnt[j]++;
                }
            }
            bm.tiles   += n;
            bm.bytes   += bytes;
            bm.busy_ns += gpu_now() - t0;
            bm.stolen  += s_stolen;

            __threadfence_system();          // y + metrics visible before publish
            for (uint32_t i = 0; i < n; ++i) {
                if (s_ids[i] == sh->terminal) {
                    atomicExch_system((unsigned int *)&sh->done_flag, 1u);
                }
            }
            for (uint32_t j = 0; j < um; ++j) {
                Task & c = sh->tasks[ucid[j]];
                if (fl_fetch_sub_i32(&c.dep_count, (int32_t)ucnt[j])
                        == (int32_t)ucnt[j]) {
                    // fl_publish bails on teardown, which matters here: the
                    // leader spins with all 1023 other threads parked at the
                    // barrier below and cannot reach the run_flag check at the
                    // top of the loop, so a wedge here hangs stop() inside
                    // cudaStreamSynchronize (P9.2a).
                    fl_publish(sh, ucid[j], /*from_gpu=*/true);
                }
            }
            fl_fetch_add_u32(&sh->n_done, n);
            // Whole epilogue: dedup + threadfence_system + dep decrement +
            // pushes. Note busy_ns above stops after the dedup loop, so it
            // has never included the fence or the consumer publish - epi_ns
            // is the first metric that covers them. Every other thread in the
            // block is parked at the barrier below for this entire span.
            if (prof) bm.epi_ns += gpu_now() - tep0;
        }
        __syncthreads();
    }
}

// ---- clock calibration (probe idiom): GPU spins on ping, stamps its clock --
__global__ void calib_kernel(RtShared * sh, int iters) {
    for (int i = 1; i <= iters; ++i) {
        while ((int)fl_load_acquire_u32(&sh->calib_ping) < i) {
            if (!fl_load_acquire_u32(&sh->run_flag)) return;
        }
        sh->calib_gpu_ts = gpu_now();
        fl_store_release_u32(&sh->calib_pong, (uint32_t)i);
    }
}

static uint64_t host_now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int64_t gpu_calibrate_clock(RtShared * sh, void * stream) {
    const int iters = 512;
    sh->calib_ping = 0;
    sh->calib_pong = 0;
    calib_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(sh, iters);
    FL_CUCHECK(cudaGetLastError());
    std::vector<int64_t> off;
    off.reserve(iters);
    for (int i = 1; i <= iters; ++i) {
        uint64_t t0 = host_now_ns();
        fl_store_release_u32(&sh->calib_ping, (uint32_t)i);
        while ((int)fl_load_acquire_u32(&sh->calib_pong) < i) {}
        uint64_t t1 = host_now_ns();
        int64_t g = (int64_t)sh->calib_gpu_ts;
        off.push_back(g - (int64_t)((t0 + t1) / 2));
    }
    FL_CUCHECK(cudaStreamSynchronize((cudaStream_t)stream));
    std::nth_element(off.begin(), off.begin() + iters / 2, off.end());
    return off[iters / 2];
}

// One stream per Runtime instance: a static stream serializes a second
// Runtime's persistent grid behind the first's never-ending kernel, and a
// device-wide sync in stop() then deadlocks on it (found on thor, P0.5).
void * gpu_make_stream() {
    cudaStream_t st;
    FL_CUCHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    return (void *)st;
}

// P5.3: co-residency is the authority on how many blocks may usefully launch.
// Every executor block polls a queue forever, so a non-resident block does no
// work until the residents exit. We do NOT grid-sync (no cooperative launch),
// so exceeding the cap degrades instead of hanging -- unlike grid-barrier
// persistent kernels, where it hangs with no error. Ask the driver rather than
// computing 65536/(threads*regs) by hand or trusting the SM count.
int gpu_resident_blocks(int threads) {
    int per_sm = 0;
    FL_CUCHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, (const void *)executor, threads, 0));
    cudaDeviceProp p;
    FL_CUCHECK(cudaGetDeviceProperties(&p, 0));
    return per_sm * p.multiProcessorCount;
}

// P6.3: Thor has 32 MiB of L2 with a 24 MiB persisting window (Orin: 4 MiB
// total, no window) and this project has never used it. The motivated target
// is NOT weights - 22 GB of model against a 24 MiB window, read once per pass,
// has no reuse to exploit, and P3.7b established byte-side levers convert to
// ~0 because the pass is parallelism-bound. It is the QUEUE working set:
// P5.9's location analysis found LD.E concentrating 1951/2366 in non-dot loops
// (queue polling and task-descriptor decode), and that region is a few hundred
// KB polled continuously by every GPU block while ~8.5 GB/pass of weight
// streaming evicts it from L2. The hypothesis is protection-from-eviction - a
// latency argument, not a bandwidth one.
//
// Default OFF; same-binary A/B via FASTLLM_L2_PERSIST=1, per the standing rule
// that a flag flip on one binary is the only trustworthy comparison.
static void l2_persist_setup(RtShared * sh, void * stream, size_t persist_bytes) {
    const char * e = getenv("FASTLLM_L2_PERSIST");
    if (!e || !atoi(e) || !persist_bytes) return;

    int max_persist = 0, max_window = 0;
    FL_CUCHECK(cudaDeviceGetAttribute(&max_persist,
                   cudaDevAttrMaxPersistingL2CacheSize, 0));
    FL_CUCHECK(cudaDeviceGetAttribute(&max_window,
                   cudaDevAttrMaxAccessPolicyWindowSize, 0));
    if (max_persist <= 0 || max_window <= 0) {
        fprintf(stderr, "fastllm: L2 persist unavailable (persist=%d window=%d)\n",
                max_persist, max_window);
        return;
    }
    size_t win = persist_bytes;
    if (win > (size_t)max_window) win = (size_t)max_window;
    size_t reserve = win;
    if (reserve > (size_t)max_persist) reserve = (size_t)max_persist;

    FL_CUCHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, reserve));

    cudaStreamAttrValue av = {};
    av.accessPolicyWindow.base_ptr  = (void *)sh;
    av.accessPolicyWindow.num_bytes = win;
    av.accessPolicyWindow.hitRatio  = win ? (float)reserve / (float)win : 1.0f;
    av.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    av.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    FL_CUCHECK(cudaStreamSetAttribute((cudaStream_t)stream,
                   cudaStreamAttributeAccessPolicyWindow, &av));

    fprintf(stderr, "fastllm: L2 persist ON base=%p window=%zu B reserve=%zu B "
                    "hitRatio=%.3f (dev max persist=%d window=%d)\n",
            (void *)sh, win, reserve, (double)av.accessPolicyWindow.hitRatio,
            max_persist, max_window);
}

int gpu_start(RtShared * sh, int blocks, int threads, void * stream,
              size_t persist_bytes) {
    int cap = gpu_resident_blocks(threads);
    // P5.7: 1024 is the shipped default because it is worth +10.4% on thor, but
    // it is only launchable where a 64-reg 1024-thread block fits. A device with
    // a smaller register file (or a future build whose regs grow past 64) makes
    // the driver report zero resident blocks; launching anyway fails with
    // cudaErrorInvalidValue and takes the whole engine down for a config knob.
    // Halve until the driver admits one, floor 256 (kernel indexes s_red[1024]
    // by threadIdx.x, so narrower is always safe; wider is what needs proving).
    // UNTESTED off thor -- every device here reports cap>0 at 1024.
    while (cap <= 0 && threads > 256) {
        const int narrower = threads / 2;
        fprintf(stderr, "fastllm: %d threads/block not resident on this device"
                        " (driver reports 0 blocks); falling back to %d\n",
                threads, narrower);
        threads = narrower;
        cap = gpu_resident_blocks(threads);
    }
    if (getenv("FASTLLM_GPU_OCC_DEBUG")) {
        fprintf(stderr, "fastllm: threads=%d resident_cap=%d requested=%d\n",
                threads, cap, blocks);
    }
    if (cap > 0 && blocks > cap) blocks = cap;   // extra blocks never run
    // P2.0: the decode-op catch-all runs exec_task_scalar on device (leader
    // thread); its frames exceed the ~1KB default stack. 4KB x 512 x blocks
    // is ~40MB - cheap on this box.
    FL_CUCHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));
    l2_persist_setup(sh, stream, persist_bytes);   // P6.3, default OFF
    // P9.2: publish the REAL grid before the launch, so no publisher can ever
    // address a block index that was clamped away. Arrival slots are cleared
    // first for the same reason - a stale slot from a previous start() would
    // read as "alive".
    if (blocks > (int)kMaxGpuBlocks) blocks = (int)kMaxGpuBlocks;
    for (int b = 0; b < (int)kMaxGpuBlocks; ++b) sh->gpu_arrived[b] = 0;
    fl_store_release_u32(&sh->gpu_blocks, (uint32_t)blocks);
    executor<<<blocks, threads, 0, (cudaStream_t)stream>>>(sh);
    FL_CUCHECK(cudaGetLastError());
    return blocks;
}

void gpu_stop_sync(void * stream) {
    // caller already dropped run_flag; blocks exit on their next poll
    FL_CUCHECK(cudaStreamSynchronize((cudaStream_t)stream));
    FL_CUCHECK(cudaStreamDestroy((cudaStream_t)stream));
}

int gpu_default_blocks() {
    cudaDeviceProp p;
    FL_CUCHECK(cudaGetDeviceProperties(&p, 0));
    return p.multiProcessorCount;
}

} // namespace fastllm
