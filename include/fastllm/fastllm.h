// fast-llm public API — the contract between runtime and bench/model layers.
// C++17. POD task records in unified memory; system-scope atomics for
// cross-engine dataflow. CPU-only builds must compile this header unchanged.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace fastllm {

// ---------------------------------------------------------------- engines --
enum class Engine : uint8_t { GPU = 0, CPU = 1, ANY = 2 };

// Task kinds P0 needs. Interpreted by both executors; keep numeric values
// stable — the GPU kernel switches on them.
enum class TaskKind : uint8_t {
    STREAM_DOT = 0,  // y[0..rows) += dot(W[row], x): streams w_bytes from DRAM
    MERGE      = 1,  // y = sum of partial slots (small, latency-bound)
    SIGNAL     = 2,  // no compute; completion marker / dependency fan-in
    // P1.1: quantize the f32 activation x[0..cols) into per-32 q8_1 blocks at
    // y (canonically the appendix fl_xq_of(x, cols); the graph builder sets
    // y explicitly). Deterministic and engine-identical (roundf + RNE half),
    // idempotent (write-only, same bytes every run — no zeroing needed).
    // w = null, w_bytes = 0, rows = cols/32.
    ACT_Q8     = 3,
    // ---- P2.0 decode ops. Param blocks live in the arena and are pointed to
    // by Task::aux (POD structs in src/runtime/decode_params.h). All have a
    // single FL_HD canonical implementation (exec_task_scalar) shared by both
    // engines; they are small/latency-bound and default CPU-affine. ----
    STREAM_DOT_IDX = 4,  // STREAM_DOT with expert indirection: effective
                         // w = t.w + (*(const uint32_t*)t.aux) * t.aux2
                         // (aux -> router-written expert id, aux2 = expert
                         // stride in bytes; t.w may carry a row offset).
    RMSNORM    = 5,  // y = x * wnorm / sqrt(mean(x^2)+eps)     (aux RmsNormP)
    ADD2       = 6,  // y[i] = a[i] + b[i]                      (aux Add2P)
    ROPE_DS2   = 7,  // yarn NEOX rope in place, per head span  (aux RopeP)
    KV_APPEND  = 8,  // scatter K/V rows into the cache at *pos (aux KvAppendP)
    ATTN_HEAD  = 9,  // one head: scores+softmax+ctx over kv<=*pos (aux AttnHeadP)
    ROUTER_SEL = 10, // softmax(logits) top-k -> ids/gates      (aux RouterP)
    SILU_MUL   = 11, // h = silu(gate) * up                     (aux SiluMulP)
    WMERGE     = 12, // y = add1 (+add2) + scale*sum gates[p]*parts[p] (aux WmergeP)
};

// Weight storage format of a STREAM_DOT tile (P1.0). W is row-major in the
// format's block layout (ggml-compatible, see src/runtime/quants.h); cols must
// be a multiple of the block size (32 for Q8_0/Q5_0, 256 for Q4_K/Q6_K) and
// w_bytes == rows * row_bytes(fmt, cols). Values are stable: the GPU executor
// switches on them.
// P5.0: IQ2_XXS/IQ3_XXS (block 256) are codebook formats - blocks hold indices
// into a fixed 256-entry grid, not values. Thor's CPU decodes them at only
// 5.5 Gw/s (gather-dependency bound), so they are
// GPU-affine by construction; host paths are reference/testing only.
enum class Fmt : uint8_t { F32 = 0, Q8_0 = 1, Q5_0 = 2, Q4_K = 3, Q6_K = 4,
                           IQ2_XXS = 5, IQ3_XXS = 6 };

inline constexpr int kMaxConsumers = 6;
// P2.4: max activation columns per multi-column dot tile (verify-batch width)
inline constexpr uint32_t kMaxDotCols = 16;

// One schedulable tile. 128-byte aligned; exactly one writer per field class:
// immutable fields written at graph build, dep_count/epoch via atomics only.
struct alignas(128) Task {
    // --- immutable after Graph::freeze() ---
    uint32_t   id;
    TaskKind   kind;
    Engine     affinity;        // hint only; stealing may override
    Fmt        fmt;             // STREAM_DOT weight format (F32 for others)
    uint16_t   n_consumers;
    uint32_t   consumers[kMaxConsumers];
    const void * w;             // weight slice (unified memory)
    uint64_t   w_bytes;         // bytes streamed by this tile
    const void * x;             // input activation
    void *     y;               // output (engine-private partial slot ok)
    uint32_t   rows, cols;
    int32_t    dep_init;        // initial dependency count
    // --- mutable, system-scope atomic access only ---
    int32_t    dep_count;       // hits 0 -> ready; reset to dep_init per token
    uint32_t   epoch;           // speculation cancel tag (P2); 0 in P0
    uint64_t   handoff_ts;      // runtime-internal: cross-engine enable stamp
    // --- P2.0: immutable op parameters (see TaskKind docs) ---
    const void * aux;           // param POD in the arena (new kinds); id slot
                                // for STREAM_DOT_IDX
    uint64_t   aux2;            // STREAM_DOT_IDX: expert stride bytes
    // --- P2.4: multi-column dots (weight amortization across batch positions).
    // nx > 1 means this tile computes nx independent dots per weight row:
    //   y[p*ystride + r] += dot(W[r], x + p*xstride)   for p in [0, nx)
    // The weight row is read ONCE for all nx columns; per-(row,column)
    // accumulation order is unchanged, so results stay bitwise identical to
    // nx separate tiles. Quantized activation appendices are per column
    // (fl_xq_of/fl_xqk_of applied to each column base).
    uint32_t   nx = 1;          // number of activation columns (batch positions)
    uint32_t   xstride = 0;     // floats between consecutive x columns
    uint32_t   ystride = 0;     // floats between consecutive y columns
};
static_assert(sizeof(Task) <= 256, "keep tasks cache-friendly");

// ------------------------------------------------------------------ arena --
// Unified-memory allocator: cudaMallocManaged under CUDA, aligned malloc
// otherwise. All Tasks, queues, activations and weights live here.
class Arena {
public:
    static std::unique_ptr<Arena> create(size_t bytes);
    virtual ~Arena() = default;
    virtual void * alloc(size_t bytes, size_t align = 128) = 0;
    virtual size_t used() const = 0;
};

// ------------------------------------------------------------------ graph --
// Build once per (model, batch-shape); freeze() lays tasks contiguously in
// the arena and computes dep_init. Reused every token via Runtime::run().
class Graph {
public:
    static std::unique_ptr<Graph> create(Arena & arena, uint32_t max_tasks);
    virtual ~Graph() = default;
    virtual uint32_t add(const Task & t) = 0;           // returns id
    virtual void     edge(uint32_t producer, uint32_t consumer) = 0;
    virtual void     freeze() = 0;                       // no adds after this
    virtual Task *   tasks() = 0;                        // arena-resident array
    virtual uint32_t size() const = 0;
    virtual uint32_t terminal() const = 0;               // last SIGNAL task id
};

// ---------------------------------------------------------------- metrics --
struct EngineMetrics {
    uint64_t tiles = 0;
    uint64_t bytes = 0;          // weight bytes streamed
    uint64_t busy_ns = 0;
    uint64_t stolen = 0;         // tiles taken from the other queue
};
inline constexpr int kNumTaskKinds = 13;

struct RunMetrics {
    EngineMetrics gpu, cpu;
    uint64_t wall_ns = 0;
    uint64_t reset_ns = 0;       // P2.4: dep/handoff reset loop (before window)
    // handoff latency: producer-completion -> cross-engine consumer-start
    uint64_t handoff_ns_p50 = 0, handoff_ns_p95 = 0, handoff_ns_max = 0;
    uint64_t handoff_samples = 0;
    // P2.0 component attribution: per-kind busy ns/counts, CPU-executed tasks
    // only (GPU per-kind attribution is a known gap; run_serial fills both).
    uint64_t kind_ns[kNumTaskKinds] = {};
    uint64_t kind_n[kNumTaskKinds]  = {};
    // P2.2 closed this for op-tasks (kind >= RMSNORM) executed on GPU via
    // stealing. P8.0: dot/merge are covered too when FASTLLM_PROF_GPU=1;
    // without it they stay aggregate-only in gpu.busy_ns as before.
    uint64_t gpu_kind_ns[kNumTaskKinds] = {};
    uint64_t gpu_kind_n[kNumTaskKinds]  = {};
    // P8.0: GPU block time decomposition, summed over active blocks, only
    // populated under FASTLLM_PROF_GPU. claim+idle+exec+epi partition the
    // persistent blocks' loop time for the run window; the denominator to
    // compare against is wall_ns * gpu_blocks_active. This is what answers
    // "where does the GPU's time actually go" rather than inferring it.
    uint64_t gpu_claim_ns = 0, gpu_sleep_ns = 0;
    uint64_t gpu_exec_ns  = 0, gpu_epi_ns  = 0;
    uint64_t gpu_polls    = 0, gpu_empty   = 0;
    uint64_t gpu_retry    = 0;   // P8.2: extra pop_bulk passes
    uint32_t gpu_blocks_active = 0;
};

// ---------------------------------------------------------------- runtime --
struct RuntimeConfig {
    // P3.7 sweep on thor (14 cores, dual engine, shadow, spec m=8): t8 40.51,
    // t10 44.27, t12 47.53, t14 46.31 tok/s -> 12 is the optimum; t14 starves
    // the GPU-facing queue leader and the OS. The old default of 8 came from a
    // llama.cpp-era orchestrator-starvation result that no longer holds under
    // the queue runtime (and from a GR00T headroom reservation since lifted).
    int  cpu_threads      = 12;
    int  gpu_blocks       = 0;    // 0 = auto (all SMs); ignored if !gpu
    int  gpu_threads      = 1024; // block width, 256..1024 multiple of 32.
                                  // P1.4: quant int8 kernels are latency-bound
                                  // (long_scoreboard 74% at 2 warps/scheduler),
                                  // so width lifts every cell: 256 -> 512 gave
                                  // +16% int8 wf1 / +23% dequant wf4. P5.3 then
                                  // unblocked 1024 (a literal launch_bounds(512)
                                  // on top of 128 regs x 512 = the 64K register
                                  // file; the leader dedup scratch moved to
                                  // shared, and ptxas now fits 64 regs) for
                                  // +10.4% more: 74.56 -> 82.29 tok/s copy-spec.
                                  // P5.5: do NOT chase occupancy below 64 regs.
                                  // Thor caps at 1536 threads/SM, so 1024 is
                                  // 1 block/SM (67%) and the THREAD limit binds
                                  // before the register file; 100%-occupancy
                                  // configs measured worse (block width beats
                                  // block count). More BLOCKS is refuted too
                                  // (queue-leader cost). gpu_start() falls back
                                  // to 512 if the driver reports 1024 cannot be
                                  // resident on this device.
    bool gpu_enabled      = true; // false -> CPU executes GPU-affine work too
    int  spin_us_park     = 3000; // spin budget before CPU workers park
    bool collect_handoff  = true; // timestamped handoff histogram (small cost)
    // Vectorized per-engine kernels (NEON/AVX2 CPU, float4+shfl GPU).
    // false = canonical scalar order, bitwise-comparable to run_serial.
    // true  = fast; deterministic per engine but summation order differs, so
    // validate against canonical by relative tolerance (docs/BENCH.md).
    bool fast_kernels     = false;
    // Tiles claimed per GPU bulk pop (clamped to [1,16]). Amortizes the
    // system-scope queue RMW that dominated per-tile cost at P0.6 (P0.7).
    int  gpu_batch        = 1;    // P0.7 measured: batching starves blocks; default 1
    // P1.1: integer-dot fast path for quant STREAM_DOT tiles. Requires every
    // quant tile's x to carry a q8_1 appendix at fl_xq_of(x, cols), written by
    // an ACT_Q8 task in the same graph (builder contract; freeze() cannot see
    // it). Effective only with fast_kernels; canonical execution always runs
    // the f32 dequant-dot reference. Numerics: int8 dots differ from canonical
    // by activation-quantization error — tolerance-gated (docs/BENCH.md).
    // P1.5: ACT_Q8 additionally writes a q8_K appendix (per-256 superblock
    // scale + per-16 bsums) at fl_xqk_of(x, cols) for K-format consumers;
    // K tiles route to integer-scale-fold kernels per the FASTLLM_I8K_*_MASK
    // dispatch bits (both appendices are always produced, so a q8_1-vs-q8_K
    // A/B is a mask flip, not a graph change). fl_x_alloc_bytes covers both.
    bool int8_dots        = false;
};

// Contract: Runtime executes ONLY graphs built via Graph::create over arenas
// from Arena::create — the canonical implementations in libfastllm. The
// interfaces above are not extension points for third-party subclasses; the
// executors rely on freeze()-validated task semantics (see TaskKind) and on
// arena-resident task tables.
class Runtime {
public:
    static std::unique_ptr<Runtime> create(Arena & arena, const RuntimeConfig & cfg);
    virtual ~Runtime() = default;                        // stops executors
    virtual void start() = 0;                            // launch persistent executors
    // Execute one frozen graph instance: resets dep counts, publishes all
    // zero-dep tasks, blocks until terminal task completes. Reentrant per
    // Runtime; NOT thread-safe across concurrent run() calls.
    virtual RunMetrics run(Graph & g) = 0;
    virtual void stop() = 0;                             // park GPU grid, join CPU pool
};

// ---------------------------------------------------- serial reference ----
// Single-threaded topological executor for correctness baselines and bench
// validation. No queues, no atomics; same Task semantics.
RunMetrics run_serial(Graph & g);

} // namespace fastllm
