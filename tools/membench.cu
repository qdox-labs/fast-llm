// P0.6 diagnostic: GPU + CPU streaming read rate vs ALLOCATION TYPE, with the
// kernel shape held constant (the P0.5 scalar row-per-thread dot — fastest of
// the three measured variants). Answers: why does the executor read managed
// memory at ~32 GB/s when llama.cpp streams ~134 GB/s from cudaMalloc and the
// doorbell probe read pageable/ATS memory at 119 GB/s on this same part?
//
//   fastllm-membench [--mb 1536] [--reps 5] [--blocks 20] [--sweep]
//
// Allocation types:
//   managed        cudaMallocManaged (current arena policy)
//   managed+advise + SetPreferredLocation(device) + SetAccessedBy(cpu) + prefetch
//   device         cudaMalloc (GPU-only; CPU row skipped)
//   pinned         cudaHostAlloc mapped (device ptr via cudaHostGetDevicePointer)
//   pageable       posix_memalign, host-touched; GPU derefs via ATS
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <string>
#include <atomic>
#include <map>
namespace fastllm { struct Task; void exec_task_scalar(Task &); } // satisfy kernels.h
#include "../src/runtime/kernels.h"     // dot_row_fast + quant CPU paths
#include "../src/runtime/kernels_gpu.h" // gdot_* device kernels (P1.0)
#include "../src/model/gguf.h"          // P5.0 iq-check reads real tensors

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "cuda %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); } } while (0)

static uint64_t now_ns() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

// P0.5 executor inner loop, verbatim shape: row-per-thread, scalar loads.
__global__ void dot_rows(const float * w, const float * x, float * y,
                         uint32_t rows, uint32_t cols) {
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += stride) {
        const float * row = w + (size_t)r * cols;
        float acc = 0.f;
        for (uint32_t c = 0; c < cols; ++c) acc += row[c] * x[c];
        y[r] += acc;
    }
}

// Executor-structure emulation: same data, processed as tiles of tile_rows,
// claimed from a global counter. Modes add executor elements one at a time:
//   m2 all-thread atomic claim; m3 = m2 + threadfence_system per tile;
//   m4 leader-only claim + shared broadcast + 2x syncthreads (executor shape);
//   m5 = m4 + fence  (= executor minus queues/publish/metrics)
__global__ void dot_tiles(const float * w, const float * x, float * y,
                          uint32_t rows, uint32_t cols, uint32_t tile_rows,
                          uint32_t * counter, int mode) {
    const uint32_t n_tiles = (rows + tile_rows - 1) / tile_rows;
    __shared__ uint32_t s_tile;
    for (;;) {
        uint32_t tile;
        if (mode >= 4) {
            if (threadIdx.x == 0) s_tile = atomicAdd(counter, 1u);
            __syncthreads();
            tile = s_tile;
        } else {
            if (threadIdx.x == 0) s_tile = atomicAdd(counter, 1u);
            __syncthreads();
            tile = s_tile;
        }
        if (tile >= n_tiles) return;
        const uint32_t r0 = tile * tile_rows;
        const uint32_t r1 = min(rows, r0 + tile_rows);
        for (uint32_t r = r0 + threadIdx.x; r < r1; r += blockDim.x) {
            const float * row = w + (size_t)r * cols;
            float acc = 0.f;
            for (uint32_t c = 0; c < cols; ++c) acc += row[c] * x[c];
            y[r] += acc;
        }
        if (mode == 3 || mode == 5) __threadfence_system();
        if (mode >= 4) __syncthreads();
    }
}

// ---- P0.9 kernel-shape variants: name the ~5.2 GB/s/block cap ----
// v_f4: float4 loads, 2 independent accumulators (executor fast shape).
__global__ void dot_f4(const float * w, const float * x, float * y,
                       uint32_t rows, uint32_t cols) {
    uint32_t stride = gridDim.x * blockDim.x;
    const float4 * x4 = (const float4 *)x;
    uint32_t c4 = cols / 4;
    for (uint32_t r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += stride) {
        const float4 * row4 = (const float4 *)(w + (size_t)r * cols);
        float4 a0 = make_float4(0.f,0.f,0.f,0.f), a1 = a0;
        uint32_t c = 0;
        for (; c + 2 <= c4; c += 2) {
            float4 w0 = row4[c],   xv0 = x4[c];
            float4 w1 = row4[c+1], xv1 = x4[c+1];
            a0.x += w0.x*xv0.x; a0.y += w0.y*xv0.y; a0.z += w0.z*xv0.z; a0.w += w0.w*xv0.w;
            a1.x += w1.x*xv1.x; a1.y += w1.y*xv1.y; a1.z += w1.z*xv1.z; a1.w += w1.w*xv1.w;
        }
        y[r] += a0.x+a0.y+a0.z+a0.w + a1.x+a1.y+a1.z+a1.w;
    }
}
// v_msN / v_mfN: S interleaved row streams per thread (scalar / float4) —
// raises outstanding loads per thread; the MLP experiment.
template<int S, bool F4>
__global__ void dot_ms(const float * w, const float * x, float * y,
                       uint32_t rows, uint32_t cols) {
    uint32_t stride = gridDim.x * blockDim.x;
    uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const float4 * x4 = (const float4 *)x;
    uint32_t c4 = cols / 4;
    for (uint32_t r0 = tid; r0 < rows; r0 += stride * S) {
        const float * rp[S]; float acc[S];
        #pragma unroll
        for (int s = 0; s < S; ++s) {
            uint32_t r = r0 + (uint32_t)s * stride;
            rp[s]  = (r < rows) ? w + (size_t)r * cols : nullptr;
            acc[s] = 0.f;
        }
        if (F4) {
            for (uint32_t c = 0; c < c4; ++c) {
                float4 xv = x4[c];
                #pragma unroll
                for (int s = 0; s < S; ++s) if (rp[s]) {
                    float4 wv = ((const float4 *)rp[s])[c];
                    acc[s] += wv.x*xv.x + wv.y*xv.y + wv.z*xv.z + wv.w*xv.w;
                }
            }
        } else {
            for (uint32_t c = 0; c < cols; ++c) {
                float xv = x[c];
                #pragma unroll
                for (int s = 0; s < S; ++s) if (rp[s]) acc[s] += rp[s][c] * xv;
            }
        }
        #pragma unroll
        for (int s = 0; s < S; ++s) {
            uint32_t r = r0 + (uint32_t)s * stride;
            if (r < rows) y[r] += acc[s];
        }
    }
}
// v_warp: warp-per-row, lane-coalesced float4 (llama.cpp mmvq shape).
__global__ void dot_warp(const float * w, const float * x, float * y,
                         uint32_t rows, uint32_t cols) {
    uint32_t warp  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    uint32_t lane  = threadIdx.x & 31;
    uint32_t nwarp = (gridDim.x * blockDim.x) >> 5;
    const float4 * x4 = (const float4 *)x;
    uint32_t c4 = cols / 4;
    for (uint32_t r = warp; r < rows; r += nwarp) {
        const float4 * row4 = (const float4 *)(w + (size_t)r * cols);
        float acc = 0.f;
        for (uint32_t c = lane; c < c4; c += 32) {
            float4 a = row4[c], b = x4[c];
            acc += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
        }
        #pragma unroll
        for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) y[r] += acc;
    }
}

// ---- P1.0: per-format kernel rate probe (managed memory, both engines) ----
// GPU wrapper: split the row range across blocks, run the executor's own
// gdot_fast on each slice (Task by value; y private per row: correctness not
// timed here beyond a nan sink check).
__global__ void fmt_kernel(fastllm::Task t, bool int8, bool q4k_v1,
                           uint32_t i8k_mask) {
    using namespace fastllm;
    const uint32_t per = (t.rows + gridDim.x - 1) / gridDim.x;
    const uint32_t r0  = blockIdx.x * per;
    if (r0 >= t.rows) return;
    Task s = t;
    s.w    = (const uint8_t *)t.w + (size_t)r0 * fl_row_bytes(t.fmt, t.cols);
    s.y    = (float *)t.y + r0;
    s.rows = min(per, t.rows - r0);
    // int8 path indexes the q8_1 appendix from s.x, which must stay the
    // ORIGINAL x pointer (fl_xq_of convention) — only w/y/rows are sliced.
    if (int8 && s.fmt != Fmt::F32) {
        if (q4k_v1 && s.fmt == Fmt::Q4_K) gdot_i8_q4_K_v1(s);
        else gdot_i8(s, i8k_mask);
    }
    else                           gdot_fast(s);
}

static void fmt_probe(int reps, int blocks, int cpu_threads, size_t mb, bool int8, bool q4k_v1,
                      bool i8k, const std::string & fmt_only, bool no_gpu, bool no_cpu,
                      int gthreads) {
    using namespace fastllm;
    struct Spec { const char * name; Fmt f; uint32_t cols; };
    const Spec specs[] = {
        { "f32 ", Fmt::F32,  2048 },   // matches model widths per format
        { "q8_0", Fmt::Q8_0, 1408 },
        { "q5_0", Fmt::Q5_0, 1408 },
        { "q4_K", Fmt::Q4_K, 2048 },
        { "q6_K", Fmt::Q6_K, 2816 },
        // P5.0 codebook formats (V4 Flash): GPU-affine by construction, so the
        // CPU column is skipped rather than reported - Thor's CPU is
        // gather-bound on these (5.5 Gw/s) and its
        // only correct path here is the scalar reference.
        { "iq2x", Fmt::IQ2_XXS, 2048 },
        { "iq3x", Fmt::IQ3_XXS, 2048 },
    };
    printf("fmt probe%s%s%s: %zu MB/format, blocks %d x %d thr, cpu t%d (GB/s = quant bytes streamed)\n",
           int8 ? " [int8/q8_1 dots]" : "", q4k_v1 ? " [q4k=v1]" : "",
           (int8 && i8k) ? " [q8_K acts]" : "", mb, blocks, gthreads, cpu_threads);
    printf("  %-5s %10s %14s %14s\n", "fmt", "GPU", "GPU/blk", "CPU pool");
    for (const Spec & sp : specs) {
        if (!fmt_only.empty() && fmt_only != std::string(sp.name).substr(0, fmt_only.size()))
            continue;
        const uint64_t rb   = fl_row_bytes(sp.f, sp.cols);
        const uint32_t rows = (uint32_t)(mb * 1024ull * 1024ull / rb);
        const uint64_t bytes = (uint64_t)rows * rb;
        uint8_t * w; CK(cudaMallocManaged((void **)&w, bytes));
        // random payload + sane scales (any payload bytes are valid quants)
        uint64_t s = 0x9E3779B97F4A7C15ull;
        for (uint64_t i = 0; i < bytes; i += 8) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            memcpy(w + i, &s, i + 8 <= bytes ? 8 : bytes - i);
        }
        const uint16_t dv = 0x3400; // 0.25
        const uint32_t bb = fl_blk_bytes(sp.f);
        for (uint64_t o = 0; o + bb <= bytes; o += bb) {
            switch (sp.f) {
                case Fmt::Q8_0: case Fmt::Q5_0: memcpy(w + o, &dv, 2); break;
                case Fmt::Q4_K: memcpy(w + o, &dv, 2); memcpy(w + o + 2, &dv, 2); break;
                case Fmt::Q6_K: memcpy(w + o + 208, &dv, 2); break;
                case Fmt::IQ2_XXS: case Fmt::IQ3_XXS: memcpy(w + o, &dv, 2); break;
                default: break;
            }
        }
        float * x; CK(cudaMallocManaged((void **)&x, fl_x_alloc_bytes(sp.cols)));
        for (uint32_t i = 0; i < sp.cols; ++i) x[i] = 1.0f / (1 + i);
        if (int8) {                       // q8_1 appendix, host-quantized once
            Task aq{};
            aq.kind = TaskKind::ACT_Q8;
            aq.x = x; aq.y = (void *)fl_xq_of(x, sp.cols);
            aq.rows = sp.cols / 32; aq.cols = sp.cols;
            fl_exec_act_q8(aq);
        }
        float * y; CK(cudaMallocManaged((void **)&y, (size_t)rows * 4));
        memset(y, 0, (size_t)rows * 4);

        Task t{};
        t.kind = TaskKind::STREAM_DOT;
        t.fmt = sp.f;
        t.w = w; t.w_bytes = bytes;
        t.x = x; t.y = y;
        t.rows = rows; t.cols = sp.cols;

        double ggpu = 0;
        if (!no_gpu) {
            const uint32_t gkm = (int8 && i8k) ? kI8KMaskGpuDefault : 0;
            fmt_kernel<<<blocks, gthreads>>>(t, int8, q4k_v1, gkm); CK(cudaDeviceSynchronize()); // warm
            std::vector<double> g;
            for (int i = 0; i < reps; ++i) {
                uint64_t t0 = now_ns();
                fmt_kernel<<<blocks, gthreads>>>(t, int8, q4k_v1, gkm);
                CK(cudaDeviceSynchronize());
                g.push_back(bytes / (double)(now_ns() - t0));
            }
            std::nth_element(g.begin(), g.begin() + g.size()/2, g.end());
            ggpu = g[g.size()/2];
        }

        const bool cpu_affine = sp.f != Fmt::IQ2_XXS && sp.f != Fmt::IQ3_XXS;
        std::vector<double> c;
        if (no_cpu || !cpu_affine) c.push_back(0);
        else for (int i = 0; i < reps; ++i) {
            uint64_t t0 = now_ns();
            std::vector<std::thread> ts;
            for (int th = 0; th < cpu_threads; ++th)
                ts.emplace_back([&, th] {
                    Task lt = t;
                    const uint32_t lo = (uint32_t)((uint64_t)rows * th / cpu_threads);
                    const uint32_t hi = (uint32_t)((uint64_t)rows * (th + 1) / cpu_threads);
                    lt.w    = w + (size_t)lo * rb;
                    lt.y    = y + lo;
                    lt.rows = hi - lo;
                    lt.w_bytes = (uint64_t)(hi - lo) * rb;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
                    if (int8 && q4k_v1 && lt.fmt == Fmt::Q4_K) {
                        const blk_q8_1 * xq = fl_xq_of(lt.x, lt.cols);
                        const uint64_t lrb = fl_row_bytes(lt.fmt, lt.cols);
                        float * ly = (float *)lt.y;
                        for (uint32_t r = 0; r < lt.rows; ++r)
                            ly[r] += dot_i8_q4_K_neon_v1(
                                (const uint8_t *)lt.w + (size_t)r * lrb,
                                xq, lt.cols);
                    } else
#endif
                    exec_task_cpu(lt, true, int8 ? kI8MaskCpuDefault : 0,
                                  (int8 && i8k) ? kI8KMaskCpuDefault : 0);
                });
            for (auto & th : ts) th.join();
            c.push_back(bytes / (double)(now_ns() - t0));
        }
        std::nth_element(c.begin(), c.begin() + c.size()/2, c.end());
        printf("  %-5s %8.1f %11.2f %12.1f GB/s\n",
               sp.name, ggpu, ggpu / blocks, c[c.size()/2]);
        CK(cudaFree(w)); CK(cudaFree(x)); CK(cudaFree(y));
    }
}

// ---- P2.5: unpack-sharing multi-column kernel probe --------------------
// Same-binary A/B at m in {1,4,8,16}: legacy = per-column loop over the
// single-column kernel (the P2.4 shape); mc = the P2.5 unpack-sharing
// kernel. Reports BOTH work-GB/s (quant bytes x m / t, comparable with
// per-column kernel rates) and DRAM-GB/s (quant bytes / t, the honest
// traffic number - the P2.4 counter caveat).
__global__ void mc_kernel(fastllm::Task t, int mode) {
    using namespace fastllm;
    const uint32_t per = (t.rows + gridDim.x - 1) / gridDim.x;
    const uint32_t r0  = blockIdx.x * per;
    if (r0 >= t.rows) return;
    Task s = t;
    s.w    = (const uint8_t *)t.w + (size_t)r0 * fl_row_bytes(t.fmt, t.cols);
    s.y    = (float *)t.y + r0;
    s.rows = min(per, t.rows - r0);
    if (mode == 1 && s.nx > 1) { gdot_i8k_q4_K_mc(s); return; }
    Task c = s;                            // legacy: per-column single dots
    c.nx = 1;
    for (uint32_t p = 0; p < (s.nx ? s.nx : 1); ++p) {
        c.x = (const void *)((const float *)s.x + (size_t)p * s.xstride);
        c.y = (void *)((float *)s.y + (size_t)p * s.ystride);
        gdot_i8k_q4_K(c);
    }
}

static void mc_probe(int reps, int blocks, int cpu_threads, size_t mb) {
    using namespace fastllm;
    struct Spec { const char * name; Fmt f; uint32_t cols; bool gpu; };
    const Spec specs[] = {
        { "q4_K", Fmt::Q4_K, 2048, true  },
        { "q6_K", Fmt::Q6_K, 2816, false },
        { "q8_0", Fmt::Q8_0, 1408, false },
    };
    const int ms[] = {1, 4, 8, 16};
    printf("mc probe: %zu MB/format, blocks %d, cpu t%d\n", mb, blocks, cpu_threads);
    printf("  %-5s %-6s %3s %12s %12s %12s %12s\n", "fmt", "eng", "m",
           "legacy work", "mc work", "legacy DRAM", "mc DRAM");
    for (const Spec & sp : specs) {
        const uint64_t rb   = fl_row_bytes(sp.f, sp.cols);
        const uint32_t rows = (uint32_t)(mb * 1024ull * 1024ull / rb);
        const uint64_t bytes = (uint64_t)rows * rb;
        uint8_t * w; CK(cudaMallocManaged((void **)&w, bytes));
        uint64_t sd = 0x9E3779B97F4A7C15ull;
        for (uint64_t i = 0; i < bytes; i += 8) {
            sd = sd * 6364136223846793005ull + 1442695040888963407ull;
            memcpy(w + i, &sd, i + 8 <= bytes ? 8 : bytes - i);
        }
        const uint16_t dv = 0x3400;
        const uint32_t bb = fl_blk_bytes(sp.f);
        for (uint64_t o = 0; o + bb <= bytes; o += bb) {
            switch (sp.f) {
                case Fmt::Q8_0: memcpy(w + o, &dv, 2); break;
                case Fmt::Q4_K: memcpy(w + o, &dv, 2); memcpy(w + o + 2, &dv, 2); break;
                case Fmt::Q6_K: memcpy(w + o + 208, &dv, 2); break;
                default: break;
            }
        }
        const size_t xper = (fl_x_alloc_bytes(sp.cols) + 127) & ~(size_t)127;
        const uint32_t xs = (uint32_t)(xper / 4);
        float * x; CK(cudaMallocManaged((void **)&x, xper * 16));
        for (int p = 0; p < 16; ++p) {
            float * xc = x + (size_t)p * xs;
            uint64_t s2 = 0x243F6A8885A308D3ull + p;
            for (uint32_t i = 0; i < sp.cols; ++i) {
                s2 = s2 * 6364136223846793005ull + 1442695040888963407ull;
                xc[i] = (float)((s2 >> 40) & 0xFFFF) * (1.0f / 65536.0f) - 0.5f;
            }
            Task aq{};
            aq.kind = TaskKind::ACT_Q8;
            aq.x = xc; aq.y = (void *)fl_xq_of(xc, sp.cols);
            aq.rows = sp.cols / 32; aq.cols = sp.cols;
            fl_exec_act_q8(aq);
        }
        float * y; CK(cudaMallocManaged((void **)&y, (size_t)rows * 16 * 4));

        for (int m : ms) {
            Task t{};
            t.kind = TaskKind::STREAM_DOT;
            t.fmt = sp.f;
            t.w = w; t.w_bytes = bytes;
            t.x = x; t.y = y;
            t.rows = rows; t.cols = sp.cols;
            t.nx = (uint32_t)m; t.xstride = xs; t.ystride = rows;
            double gl[2] = {0, 0};   // work GB/s legacy/mc
            if (sp.gpu) {
                for (int mode = 0; mode < 2; ++mode) {
                    mc_kernel<<<blocks, 512>>>(t, mode); CK(cudaDeviceSynchronize());
                    std::vector<double> g;
                    for (int i = 0; i < reps; ++i) {
                        uint64_t t0 = now_ns();
                        mc_kernel<<<blocks, 512>>>(t, mode);
                        CK(cudaDeviceSynchronize());
                        g.push_back(bytes * (double)m / (double)(now_ns() - t0));
                    }
                    std::nth_element(g.begin(), g.begin() + g.size()/2, g.end());
                    gl[mode] = g[g.size()/2];
                }
                printf("  %-5s %-6s %3d %9.1f %12.1f %12.1f %12.1f GB/s\n",
                       sp.name, "GPU", m, gl[0], gl[1], gl[0] / m, gl[1] / m);
            }
            // CPU pool A/B
            double cl[2] = {0, 0};
            for (int mode = 0; mode < 2; ++mode) {
                std::vector<double> c;
                for (int i = 0; i < reps; ++i) {
                    uint64_t t0 = now_ns();
                    std::vector<std::thread> ts;
                    for (int th = 0; th < cpu_threads; ++th)
                        ts.emplace_back([&, th] {
                            const uint32_t lo = (uint32_t)((uint64_t)rows * th / cpu_threads);
                            const uint32_t hi = (uint32_t)((uint64_t)rows * (th + 1) / cpu_threads);
                            Task lt = t;
                            lt.w = w + (size_t)lo * rb;
                            lt.y = y + lo;
                            lt.rows = hi - lo;
                            if (mode == 0) {
                                Task c1 = lt; c1.nx = 1;
                                for (int p = 0; p < m; ++p) {
                                    c1.x = (const void *)((const float *)lt.x + (size_t)p * xs);
                                    c1.y = (void *)((float *)lt.y + (size_t)p * (size_t)rows);
                                    exec_task_cpu(c1, true, kI8MaskCpuDefault, kI8KMaskCpuDefault);
                                }
                            } else {
                                exec_task_cpu(lt, true, kI8MaskCpuDefault, kI8KMaskCpuDefault);
                            }
                        });
                    for (auto & th : ts) th.join();
                    c.push_back(bytes * (double)m / (double)(now_ns() - t0));
                }
                std::nth_element(c.begin(), c.begin() + c.size()/2, c.end());
                cl[mode] = c[c.size()/2];
            }
            printf("  %-5s %-6s %3d %9.1f %12.1f %12.1f %12.1f GB/s\n",
                   sp.name, "CPU", m, cl[0], cl[1], cl[0] / m, cl[1] / m);
        }
        CK(cudaFree(w)); CK(cudaFree(x)); CK(cudaFree(y));
    }
}

struct Buf { const char * name; float * host; float * dev; bool cpu_ok; };

static void fill(float * p, size_t n) {           // seeded dense fill
    uint64_t s = 0x243F6A8885A308D3ull;
    for (size_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        p[i] = (float)((s >> 40) & 0xFFFF) * (1.0f / 65536.0f) - 0.5f;
    }
}


// ---- P5.0: codebook correctness gate on REAL V4 tensors --------------------
// Block-layout misreads are the likeliest failure for a new quant format, and
// they are invisible to self-consistency tests (a wrong-but-consistent decoder
// agrees with itself). So this compares three independent decodings of the
// SAME real tensor rows: the host scalar reference (canonical order), the GPU
// f32-dequant kernel, and the GPU int8/dp4a kernel. tools/gen/check_iq_numpy.py
// adds an f64 anchor computed by gguf-py, outside our code entirely.
static int iq_check(const std::string & path, int nrows) {
    using namespace fastllm;
    Gguf g;
    if (!g.open(path)) { fprintf(stderr, "iq-check: cannot open %s\n", path.c_str()); return 2; }
    int bad = 0;
    {   // census first: a "not present" line must be distinguishable from a
        // reader that parsed nothing (split shards carry their own headers)
        std::map<uint32_t, int> hist;
        for (const auto & t : g.tensors()) hist[t.type]++;
        printf("  reader sees %zu tensors:", g.tensors().size());
        for (auto & kv : hist) printf(" %s x%d", Gguf::type_name(kv.first), kv.second);
        printf("\n");
    }
    // Q8_0/Q6_K are CONTROLS: kernels validated since P1.x, measured on the
    // identical metric so the codebook int8 error is judged against how this
    // engine's accepted paths behave here, not against a guessed constant.
    for (uint32_t want : { (uint32_t)GGML_TYPE_Q8_0, (uint32_t)GGML_TYPE_Q6_K,
                           (uint32_t)GGML_TYPE_IQ2_XXS, (uint32_t)GGML_TYPE_IQ3_XXS }) {
        const GgufTensor * sel = nullptr;
        // MoE expert weights are 3-D ([cols, rows, n_experts]); rows are
        // contiguous across the expert dimension, so flatten ne[1]*ne[2].
        // P5.2: prefer a 3-D (MoE expert) tensor when one exists. The 2-D-first
        // scan picked attn_q for IQ2_XXS, leaving the expert path - the one the
        // decode graph actually drives, through STREAM_DOT_IDX - ungated.
        for (const auto & t : g.tensors())
            if (t.type == want && t.n_dims >= 3 && t.ne[0] % 256 == 0) { sel = &t; break; }
        if (!sel)
            for (const auto & t : g.tensors())
                if (t.type == want && t.n_dims >= 2 && t.ne[0] % 256 == 0) { sel = &t; break; }
        if (!sel) { printf("  %-8s not present in this shard\n", Gguf::type_name(want)); continue; }
        Fmt f; Gguf::fmt_of(want, &f);
        const uint32_t cols = (uint32_t)sel->ne[0];
        const uint64_t all_rows = sel->ne[1] * sel->ne[2] * sel->ne[3];
        const uint32_t rows = (uint32_t)std::min<uint64_t>(nrows, all_rows);
        const uint64_t rb   = fl_row_bytes(f, cols);
        const uint8_t * src = g.data(*sel);

        uint8_t * w; CK(cudaMallocManaged((void **)&w, (size_t)rows * rb));
        memcpy(w, src, (size_t)rows * rb);
        float * x; CK(cudaMallocManaged((void **)&x, fl_x_alloc_bytes(cols)));
        uint64_t s2 = 0x243F6A8885A308D3ull;            // uniform noise, not a
        for (uint32_t i = 0; i < cols; ++i) {           // ramp (P3.5: ramps are
            s2 = s2 * 6364136223846793005ull + 1442695040888963407ull;
            x[i] = (float)((int32_t)(s2 >> 33) % 2001 - 1000) / 1000.0f;
        }                                               // adversarial for q8_1)
        Task aq{}; aq.kind = TaskKind::ACT_Q8; aq.x = x;
        aq.y = (void *)fl_xq_of(x, cols); aq.rows = cols / 32; aq.cols = cols;
        fl_exec_act_q8(aq);

        float * y; CK(cudaMallocManaged((void **)&y, (size_t)rows * 4));
        Task t{}; t.kind = TaskKind::STREAM_DOT; t.fmt = f;
        t.w = w; t.w_bytes = (uint64_t)rows * rb; t.x = x; t.y = y;
        t.rows = rows; t.cols = cols;

        std::vector<float> ref(rows);
        for (uint32_t r = 0; r < rows; ++r)
            ref[r] = fl_dot_row_any(f, w + (size_t)r * rb, x, cols);
        // RMS floor (P3.5 convention): a row whose dot lands near zero by
        // cancellation has no meaningful relative error - dividing by it
        // manufactures failures out of a correct kernel. Normalize by the
        // row's own magnitude OR the distribution's RMS, whichever is larger.
        double sq = 0;
        for (uint32_t r = 0; r < rows; ++r) sq += (double)ref[r] * ref[r];
        const double rms_floor = 0.1 * std::sqrt(sq / std::max(1u, rows));

        double worst_f32 = 0, worst_i8 = 0;
        std::vector<float> got[2];
        for (int pass = 0; pass < 2; ++pass) {
            memset(y, 0, (size_t)rows * 4);
            fmt_kernel<<<20, 512>>>(t, pass == 1, false, 0);
            CK(cudaDeviceSynchronize());
            got[pass].assign(y, y + rows);
            for (uint32_t r = 0; r < rows; ++r) {
                const double den = std::max(rms_floor, (double)std::fabs(ref[r]));
                const double rel = std::fabs(y[r] - ref[r]) / den;
                (pass ? worst_i8 : worst_f32) = std::max(pass ? worst_i8 : worst_f32, rel);
            }
        }
        // f32-dequant must match the reference to fp reassociation only. The
        // int8 threshold is calibrated from the controls in this same run
        // (printed below); codebook formats add NO weight error over the
        // controls - grid values are exact int8 - so they must land in the
        // same class, which is the real gate.
        const bool ok = worst_f32 < 1e-4 && worst_i8 < 2e-1;
        printf("  %-8s %s rows %4u cols %5u : gpu-f32 max_rel %.3e  gpu-int8 max_rel %.3e  %s\n",
               Gguf::type_name(want), sel->name.c_str(), rows, cols,
               worst_f32, worst_i8, ok ? "OK" : "FAIL");
        if (!ok) ++bad;
        // first rows printed so the gguf-py/numpy f64 anchor (tools/gen/
        // check_iq_numpy.py) can be diffed against our reference directly
        for (uint32_t r = 0; r < std::min(4u, rows); ++r)
            printf("      row %u ref %.9g  f32 %.9g  int8 %.9g\n", r,
                   (double)ref[r], (double)got[0][r], (double)got[1][r]);
        cudaFree(w); cudaFree(x); cudaFree(y);
    }
    return bad;
}

int main(int argc, char ** argv) {
    size_t mb = 1536; int reps = 5, blocks = 20; bool sweep = false;
    bool kernels = false; bool fmt = false; bool int8 = false; int cpu_threads = 8;
    bool q4k_v1 = false;
    bool i8k = false;
    std::string only, fmt_only;
    bool no_gpu = false, no_cpu = false;
    bool mc = false; int gthreads = 1024;  // P4.0/P5.7: match RuntimeConfig::
                                           // gpu_threads. A probe measured at a
                                           // width the engine does not use is
                                           // how P1.1-P3.9 understated engine
                                           // kernels ~2x (this was hardcoded 256
                                           // while the engine ran 512).
    std::string iq_check_path; int iq_rows = 64;      // P5.0 gate
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--mb") mb = atoi(argv[++i]);
        else if (a == "--reps") reps = atoi(argv[++i]);
        else if (a == "--blocks") blocks = atoi(argv[++i]);
        else if (a == "--sweep") sweep = true;
        else if (a == "--kernels") kernels = true;
        else if (a == "--fmt") fmt = true;
        else if (a == "--int8") int8 = true;
        else if (a == "--q4k") q4k_v1 = (std::string(argv[++i]) == "v1");
        else if (a == "--i8k") i8k = (std::string(argv[++i]) == "on");
        else if (a == "--cpu-threads") cpu_threads = atoi(argv[++i]);
        else if (a == "--only") { kernels = true; only = argv[++i]; }
        else if (a == "--gthreads") gthreads = atoi(argv[++i]);
        else if (a == "--mc") mc = true;                    // P2.5 mc probe
        else if (a == "--fmt-only") fmt_only = argv[++i];   // profile isolation
        else if (a == "--no-gpu") no_gpu = true;            // (P1.4 ncu/perf)
        else if (a == "--no-cpu") no_cpu = true;
        else if (a == "--iq-check") iq_check_path = argv[++i];  // P5.0
        else if (a == "--iq-rows") iq_rows = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (!iq_check_path.empty()) {
        printf("iq-check (real V4 tensors, 3-way: host ref / gpu f32 / gpu int8)\n");
        return iq_check(iq_check_path, iq_rows);
    }
    if (fmt) { fmt_probe(reps, blocks, cpu_threads, mb, int8, q4k_v1, i8k, fmt_only, no_gpu, no_cpu,
                        gthreads); return 0; }
    if (mc)  { mc_probe(reps, blocks, cpu_threads, mb); return 0; }
    const uint32_t cols = 2048;
    const uint32_t rows = (uint32_t)(mb * 1024ull * 1024ull / (cols * 4));
    const size_t   n    = (size_t)rows * cols;
    const size_t   bytes = n * 4;
    printf("membench: %u rows x %u cols = %.2f GB, reps %d, blocks %d\n",
           rows, cols, bytes / 1e9, reps, blocks);

    float * x_m; CK(cudaMallocManaged(&x_m, cols * 4));
    float * y_m; CK(cudaMallocManaged(&y_m, (size_t)rows * 4));
    fill(x_m, cols);

    std::vector<float> stage(n);           // one fill, copied everywhere
    fill(stage.data(), n);

    Buf bufs[5] = {};
    { float * p; CK(cudaMallocManaged(&p, bytes));
      memcpy(p, stage.data(), bytes);
      bufs[0] = {"managed       ", p, p, true}; }
    { float * p; CK(cudaMallocManaged(&p, bytes));
      memcpy(p, stage.data(), bytes);
      cudaMemLocation dev0{}; dev0.type = cudaMemLocationTypeDevice; dev0.id = 0;
      cudaMemLocation host{}; host.type = cudaMemLocationTypeHost;   host.id = 0;
      CK(cudaMemAdvise(p, bytes, cudaMemAdviseSetPreferredLocation, dev0));
      CK(cudaMemAdvise(p, bytes, cudaMemAdviseSetAccessedBy, host));
      cudaMemPrefetchAsync(p, bytes, dev0, 0, 0); cudaDeviceSynchronize(); // best effort
      bufs[1] = {"managed+advise", p, p, true}; }
    { float * p; CK(cudaMalloc(&p, bytes));
      CK(cudaMemcpy(p, stage.data(), bytes, cudaMemcpyDefault));
      bufs[2] = {"device        ", nullptr, p, false}; }
    { float * h; CK(cudaHostAlloc(&h, bytes, cudaHostAllocMapped));
      memcpy(h, stage.data(), bytes);
      float * d; CK(cudaHostGetDevicePointer(&d, h, 0));
      bufs[3] = {"pinned        ", h, d, true}; }
    { float * p = nullptr;
      if (posix_memalign((void **)&p, 4096, bytes)) { fprintf(stderr, "oom\n"); return 1; }
      memcpy(p, stage.data(), bytes);
      bufs[4] = {"pageable(ATS) ", p, p, true}; }

    auto gpu_pass = [&](const Buf & b, int blk) -> double {
        memset(y_m, 0, (size_t)rows * 4);
        dot_rows<<<blk, 256>>>(b.dev, x_m, y_m, rows, cols);  // warm
        CK(cudaDeviceSynchronize());
        std::vector<double> r;
        for (int i = 0; i < reps; ++i) {
            uint64_t t0 = now_ns();
            dot_rows<<<blk, 256>>>(b.dev, x_m, y_m, rows, cols);
            CK(cudaDeviceSynchronize());
            r.push_back(bytes / (double)(now_ns() - t0));
        }
        std::nth_element(r.begin(), r.begin() + r.size()/2, r.end());
        return r[r.size()/2];
    };
    auto cpu_pass = [&](const Buf & b, int threads) -> double {
        std::vector<double> r;
        for (int i = 0; i < reps; ++i) {
            uint64_t t0 = now_ns();
            std::vector<std::thread> ts;
            for (int t = 0; t < threads; ++t)
                ts.emplace_back([&, t] {
                    uint32_t lo = (uint32_t)((uint64_t)rows * t / threads);
                    uint32_t hi = (uint32_t)((uint64_t)rows * (t+1) / threads);
                    float acc = 0.f;
                    for (uint32_t rr = lo; rr < hi; ++rr)
                        acc += fastllm::dot_row_fast(b.host + (size_t)rr * cols, x_m, cols);
                    y_m[lo] = acc;   // sink
                });
            for (auto & th : ts) th.join();
            r.push_back(bytes / (double)(now_ns() - t0));
        }
        std::nth_element(r.begin(), r.begin() + r.size()/2, r.end());
        return r[r.size()/2];
    };

    if (kernels) {
        // P0.9: kernel-shape table on the managed buffer only.
        const float * W = bufs[0].dev;
        auto L = [&](const char * name, auto kern, int blk, int thr) -> double {
            if (!only.empty() && only != name) return -1;
            memset(y_m, 0, (size_t)rows * 4);
            kern<<<blk, thr>>>(W, x_m, y_m, rows, cols); CK(cudaDeviceSynchronize());
            std::vector<double> r;
            for (int i = 0; i < reps; ++i) {
                uint64_t t0 = now_ns();
                kern<<<blk, thr>>>(W, x_m, y_m, rows, cols);
                CK(cudaDeviceSynchronize());
                r.push_back(bytes / (double)(now_ns() - t0));
            }
            std::nth_element(r.begin(), r.begin() + r.size()/2, r.end());
            double g = r[r.size()/2];
            printf("  %-10s b%-3d t%-4d %8.1f GB/s (%.2f /block)\n", name, blk, thr, g, g / blk);
            return g;
        };
        printf("kernel-shape sweep (managed buffer):\n");
        for (int blk : {20, 40}) {
            L("scalar", dot_rows,        blk, 256);
            L("f4",     dot_f4,          blk, 256);
            L("ms2",    dot_ms<2,false>, blk, 256);
            L("ms4",    dot_ms<4,false>, blk, 256);
            L("ms8",    dot_ms<8,false>, blk, 256);
            L("mf2",    dot_ms<2,true>,  blk, 256);
            L("mf4",    dot_ms<4,true>,  blk, 256);
            L("warp",   dot_warp,        blk, 256);
        }
        L("scalar", dot_rows,        20, 512);
        L("mf2",    dot_ms<2,true>,  20, 512);
        L("warp",   dot_warp,        20, 512);
        L("warp",   dot_warp,        40, 512);
        return 0;
    }

    printf("%-15s %12s %12s\n", "alloc", "GPU GB/s", "CPU t8 GB/s");
    double best = 0, worst = 1e18; int bi = 0, wi = 0;
    for (int i = 0; i < 5; ++i) {
        double g = gpu_pass(bufs[i], blocks);
        double c = bufs[i].cpu_ok ? cpu_pass(bufs[i], 8) : 0;
        printf("%-15s %12.1f %12.1f\n", bufs[i].name, g, c);
        if (g > best)  { best = g;  bi = i; }
        if (g < worst) { worst = g; wi = i; }
    }
    if (sweep) {
        for (int i : {bi, wi}) {
            printf("sweep %-15s:", bufs[i].name);
            for (int blk : {10, 20, 40})
                printf("  b%d %.1f", blk, gpu_pass(bufs[i], blk));
            printf(" GB/s\n");
        }
    }

    // bus-contention control: GPU streams managed buf while CPU t8 streams the
    // disjoint pageable buf, simultaneously; compare to solo rates.
    {
        std::atomic<bool> go{false}, stop_cpu{false};
        double cpu_gbps = 0;
        std::thread cpu_thr([&] {
            while (!go.load()) {}
            uint64_t t0 = now_ns(); size_t passes = 0;
            while (!stop_cpu.load()) {
                std::vector<std::thread> ts;
                for (int t = 0; t < 8; ++t)
                    ts.emplace_back([&, t] {
                        uint32_t lo = (uint32_t)((uint64_t)rows * t / 8);
                        uint32_t hi = (uint32_t)((uint64_t)rows * (t+1) / 8);
                        float acc = 0.f;
                        for (uint32_t rr = lo; rr < hi && !stop_cpu.load(); ++rr)
                            acc += fastllm::dot_row_fast(bufs[4].host + (size_t)rr * cols, x_m, cols);
                        y_m[lo] = acc;
                    });
                for (auto & th : ts) th.join();
                passes++;
            }
            cpu_gbps = passes * (double)bytes / (now_ns() - t0);
        });
        go = true;
        std::vector<double> g;
        for (int i = 0; i <= reps; ++i) {
            uint64_t t0 = now_ns();
            dot_rows<<<blocks, 256>>>(bufs[0].dev, x_m, y_m, rows, cols);
            CK(cudaDeviceSynchronize());
            if (i) g.push_back(bytes / (double)(now_ns() - t0));
        }
        stop_cpu = true; cpu_thr.join();
        std::nth_element(g.begin(), g.begin() + g.size()/2, g.end());
        printf("\nconcurrent: GPU(managed) %.1f + CPU t8(pageable) %.1f = %.1f GB/s combined\n",
               g[g.size()/2], cpu_gbps, g[g.size()/2] + cpu_gbps);
    }

    // executor-structure emulation ladder on the managed buffer
    uint32_t * ctr; CK(cudaMallocManaged(&ctr, 4));
    auto tile_pass = [&](uint32_t tr, int mode) -> double {
        std::vector<double> r;
        for (int i = 0; i <= reps; ++i) {          // first is warm
            *ctr = 0;
            uint64_t t0 = now_ns();
            dot_tiles<<<blocks, 256>>>(bufs[0].dev, x_m, y_m, rows, cols, tr, ctr, mode);
            CK(cudaDeviceSynchronize());
            if (i) r.push_back(bytes / (double)(now_ns() - t0));
        }
        std::nth_element(r.begin(), r.begin() + r.size()/2, r.end());
        return r[r.size()/2];
    };
    printf("\nexecutor emulation (managed, %d blocks): baseline stride %.1f GB/s\n",
           blocks, gpu_pass(bufs[0], blocks));
    for (uint32_t tr : {2048u, 256u, 4u}) {
        printf("tile_rows %4u:", tr);
        printf("  m2(claim) %.1f", tile_pass(tr, 2));
        printf("  m3(+fence) %.1f", tile_pass(tr, 3));
        printf("  m4(+sync) %.1f", tile_pass(tr, 4));
        printf("  m5(all) %.1f GB/s\n", tile_pass(tr, 5));
    }
    return 0;
}
