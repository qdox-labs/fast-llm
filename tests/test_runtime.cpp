// Runtime unit tests (CPU-only; gpu_enabled=false everywhere). The parallel
// runtime must produce BITWISE-identical results to run_serial: both execute
// exec_task_scalar per task, each task exclusively owns its output range, and
// consumers read producer outputs only after the dependency edge fires.
#include "fastllm/fastllm.h"
#include "runtime/runtime_internal.h"
#include "runtime/kernels.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

using namespace fastllm;

static int g_checks = 0, g_fails = 0;
#define CHECK(cond, msg) do { g_checks++; if (!(cond)) { g_fails++; \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, msg); } } while (0)

namespace {

float * falloc(Arena & a, size_t n, uint32_t seed) {
    float * p = (float *)a.alloc(n * sizeof(float));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (size_t i = 0; i < n; ++i) p[i] = d(rng);
    return p;
}

Task dot_task(const float * w, const float * x, float * y,
              uint32_t rows, uint32_t cols, Engine aff) {
    Task t{};
    t.kind = TaskKind::STREAM_DOT;
    t.affinity = aff;
    t.w = w; t.x = x; t.y = y;
    t.rows = rows; t.cols = cols;
    t.w_bytes = (uint64_t)rows * cols * sizeof(float);
    return t;
}

Task signal_task() {
    Task t{};
    t.kind = TaskKind::SIGNAL;
    t.affinity = Engine::CPU;
    return t;
}

// diamond + chain: x -> {A,B} -> M(merge) -> C(dot reading M's y) -> S
struct DiamondBufs { float *ya, *yb, *part, *ym, *yc; uint32_t rows, cols; };

std::unique_ptr<Graph> build_diamond(Arena & a, DiamondBufs & b) {
    const uint32_t R = 96, C = 64;
    b.rows = R; b.cols = C;
    const float * x  = falloc(a, C, 11);
    const float * wa = falloc(a, (size_t)R * C, 12);
    const float * wb = falloc(a, (size_t)R * C, 13);
    const float * wc = falloc(a, (size_t)R * R, 14);
    b.part = (float *)a.alloc(2 * R * sizeof(float));
    b.ya = b.part; b.yb = b.part + R;
    b.ym = (float *)a.alloc(R * sizeof(float));
    b.yc = (float *)a.alloc(R * sizeof(float));

    auto g = Graph::create(a, 16);
    uint32_t A = g->add(dot_task(wa, x, b.ya, R, C, Engine::CPU));
    uint32_t B = g->add(dot_task(wb, x, b.yb, R, C, Engine::GPU)); // stolen
    Task m{};
    m.kind = TaskKind::MERGE;
    m.affinity = Engine::CPU;
    m.x = b.part; m.y = b.ym; m.rows = R; m.cols = 2;
    uint32_t M = g->add(m);
    uint32_t Cc = g->add(dot_task(wc, b.ym, b.yc, R, R, Engine::CPU));
    uint32_t S = g->add(signal_task());
    g->edge(A, M); g->edge(B, M); g->edge(M, Cc); g->edge(Cc, S);
    g->freeze();
    return g;
}

void zero_outputs(DiamondBufs & b) {
    memset(b.part, 0, 2 * b.rows * sizeof(float));
    memset(b.ym, 0, b.rows * sizeof(float));
    memset(b.yc, 0, b.rows * sizeof(float));
}

void test_diamond_and_repeat() {
    auto arena = Arena::create(64ull << 20);
    DiamondBufs b{};
    auto g = build_diamond(*arena, b);

    zero_outputs(b);
    run_serial(*g);
    std::vector<float> ref(b.yc, b.yc + b.rows);

    RuntimeConfig cfg;
    cfg.cpu_threads = 4;
    cfg.gpu_enabled = false;
    cfg.spin_us_park = 2000;
    auto rt = Runtime::create(*arena, cfg);
    rt->start();

    for (int rep = 0; rep < 3; ++rep) {
        zero_outputs(b);
        RunMetrics m = rt->run(*g);
        CHECK(memcmp(b.yc, ref.data(), b.rows * sizeof(float)) == 0,
              "parallel result must be bitwise-identical to serial");
        CHECK(m.cpu.tiles == g->size(), "all tasks executed on CPU");
        CHECK(m.gpu.tiles == 0, "no GPU in CPU-only build");
    }

    // stealing: task B was GPU-affine -> sits in q[0], CPU must steal it
    zero_outputs(b);
    RunMetrics m = rt->run(*g);
    CHECK(m.cpu.stolen >= 1, "GPU-affine work must be stolen by CPU pool");

    // park/wake: idle past the spin budget, then run again
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    zero_outputs(b);
    m = rt->run(*g);
    CHECK(memcmp(b.yc, ref.data(), b.rows * sizeof(float)) == 0,
          "correct after park/wake");
    rt->stop();
}

// randomized layered DAG vs serial, exact comparison
void test_fuzz(uint32_t seed, int n_tasks_target) {
    auto arena = Arena::create(256ull << 20);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> rrows(4, 16), rcols(8, 32),
        rlayer(2, 6);

    struct Node { uint32_t id; float * y; uint32_t rows; };
    std::vector<std::vector<Node>> layers;
    auto g = Graph::create(*arena, n_tasks_target + 8);
    const float * x0 = falloc(*arena, 64, seed * 7 + 1);
    std::vector<float *> outs; // for zeroing

    int made = 0;
    while (made < n_tasks_target) {
        layers.emplace_back();
        uint32_t width = rlayer(rng);
        for (uint32_t j = 0; j < width && made < n_tasks_target; ++j, ++made) {
            uint32_t rows = rrows(rng), cols;
            const float * x;
            uint32_t producer = UINT32_MAX;
            if (!layers.empty() && layers.size() > 1 && (rng() & 1)) {
                // consume a previous-layer output (true dataflow dependency)
                auto & prev = layers[layers.size() - 2];
                auto & p = prev[rng() % prev.size()];
                x = p.y; cols = p.rows; producer = p.id;
            } else {
                x = x0; cols = 64;
            }
            float * y = (float *)arena->alloc(rows * sizeof(float));
            outs.push_back(y);
            const float * w = falloc(*arena, (size_t)rows * cols, rng());
            Engine aff = (rng() % 3 == 0) ? Engine::GPU : Engine::CPU;
            uint32_t id = g->add(dot_task(w, x, y, rows, cols, aff));
            if (producer != UINT32_MAX && g->tasks()[producer].n_consumers < kMaxConsumers - 1) {
                g->edge(producer, id);
            }
            layers.back().push_back({id, y, rows});
        }
    }
    uint32_t S = g->add(signal_task());
    for (uint32_t i = 0; i < S; ++i) {
        if (g->tasks()[i].n_consumers == 0) g->edge(i, S);
    }
    g->freeze();

    auto zero_all = [&] {
        for (float * y : outs) {
            // rows unknown here; re-derive from task table
        }
        for (uint32_t i = 0; i < g->size(); ++i) {
            Task & t = g->tasks()[i];
            if (t.kind == TaskKind::STREAM_DOT) {
                memset(t.y, 0, t.rows * sizeof(float));
            }
        }
    };

    zero_all();
    run_serial(*g);
    std::vector<std::vector<float>> ref;
    for (uint32_t i = 0; i < g->size(); ++i) {
        Task & t = g->tasks()[i];
        if (t.kind == TaskKind::STREAM_DOT) {
            ref.emplace_back((float *)t.y, (float *)t.y + t.rows);
        } else {
            ref.emplace_back();
        }
    }

    RuntimeConfig cfg;
    cfg.cpu_threads = 8;
    cfg.gpu_enabled = false;
    cfg.spin_us_park = 1000;
    auto rt = Runtime::create(*arena, cfg);
    rt->start();
    zero_all();
    RunMetrics m = rt->run(*g);
    rt->stop();

    bool same = true;
    for (uint32_t i = 0; i < g->size(); ++i) {
        Task & t = g->tasks()[i];
        if (t.kind != TaskKind::STREAM_DOT) continue;
        if (memcmp(t.y, ref[i].data(), t.rows * sizeof(float)) != 0) same = false;
    }
    CHECK(same, "fuzz DAG: parallel bitwise == serial");
    CHECK(m.cpu.tiles == g->size(), "fuzz DAG: full execution");
}

// Integration: bench-shaped MoE graph (router -> expert tiles incl. a
// remainder-style cols<kCols tile -> range-MERGE -> next layer), accumulate
// (+=) semantics with zeroing between runs — the exact pattern fastllm-bench
// emits. Guards the bench<->runtime contract that the P0 crash violated.
void test_moe_shaped_integration() {
    auto arena = Arena::create(64ull << 20);
    Arena & a = *arena;
    const int L = 3, E = 4;
    const uint32_t cols = 64;
    const float * x = falloc(a, cols, 99);
    float * slots = (float *)a.alloc(512 * sizeof(float));
    float * final_out = (float *)a.alloc(L * sizeof(float));
    uint32_t n_slots = 0;

    auto g = Graph::create(a, 128);
    uint32_t prev_merge = UINT32_MAX;
    for (int l = 0; l < L; ++l) {
        uint32_t first = n_slots;
        // router
        const float * wr = falloc(a, (size_t)4 * cols, 500 + l);
        uint32_t router = g->add(dot_task(wr, x, slots + n_slots, 4, cols,
                                          Engine::GPU));
        n_slots += 4;
        if (prev_merge != UINT32_MAX) g->edge(prev_merge, router);
        // experts: rows vary; one remainder-style tile (rows=1, cols=17)
        std::vector<uint32_t> tiles;
        for (int e = 0; e < E; ++e) {
            uint32_t rows = 8 + 5 * e;
            const float * w = falloc(a, (size_t)rows * cols, 1000 + l * 16 + e);
            tiles.push_back(g->add(dot_task(w, x, slots + n_slots, rows, cols,
                                            e % 2 ? Engine::CPU : Engine::GPU)));
            n_slots += rows;
        }
        const float * wrem = falloc(a, 17, 2000 + l);
        tiles.push_back(g->add(dot_task(wrem, x, slots + n_slots, 1, 17,
                                        Engine::CPU)));
        n_slots += 1;
        // range-MERGE into final_out[l]
        Task m{};
        m.kind = TaskKind::MERGE;
        m.affinity = Engine::GPU;
        m.x = slots + first;
        m.rows = 1;
        m.cols = n_slots - first;
        m.y = final_out + l;
        uint32_t merge = g->add(m);
        for (uint32_t t : tiles) { g->edge(router, t); g->edge(t, merge); }
        g->edge(router, merge);
        prev_merge = merge;
    }
    uint32_t S = g->add(signal_task());
    g->edge(prev_merge, S);
    g->freeze();

    auto zero_all = [&] {
        memset(slots, 0, (size_t)n_slots * sizeof(float));
        memset(final_out, 0, (size_t)L * sizeof(float));
    };
    zero_all();
    run_serial(*g);
    std::vector<float> ref(final_out, final_out + L);
    CHECK(ref[0] != 0.f || ref[1] != 0.f, "moe-shaped: reference is non-trivial");

    RuntimeConfig cfg;
    cfg.cpu_threads = 4;
    cfg.gpu_enabled = false;
    auto rt = Runtime::create(a, cfg);
    rt->start();
    for (int rep = 0; rep < 2; ++rep) {
        zero_all();
        rt->run(*g);
        CHECK(memcmp(final_out, ref.data(), L * sizeof(float)) == 0,
              "moe-shaped: runtime bitwise == serial (rep)");
    }
    rt->stop();
}

} // namespace


// fast kernels: per-row vectorized dot vs canonical, tolerance-gated;
// covers 16-multiple, tail, and sub-16 col counts.
static void test_fast_kernels() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const uint32_t cases[][2] = {{4, 2048}, {3, 2050}, {5, 137}, {2, 8}, {1, 3}};
    for (auto & c : cases) {
        uint32_t rows = c[0], cols = c[1];
        std::vector<float> w((size_t)rows * cols), x(cols),
                           y0(rows, 0.f), y1(rows, 0.f);
        for (auto & v : w) v = d(rng);
        for (auto & v : x) v = d(rng);
        Task t{};
        t.kind = fastllm::TaskKind::STREAM_DOT;
        t.w = w.data(); t.x = x.data(); t.rows = rows; t.cols = cols;
        t.w_bytes = (uint64_t)rows * cols * 4;
        t.y = y0.data(); fastllm::exec_task_cpu(t, false, 0);
        t.y = y1.data(); fastllm::exec_task_cpu(t, true, 0);
        bool ok = true;
        for (uint32_t r = 0; r < rows; ++r) {
            double rel = fabs((double)y1[r] - y0[r]) /
                         std::max(1e-30, fabs((double)y0[r]));
            if (rel > 1e-5) ok = false;
        }
        CHECK(ok, "fast dot within 1e-5 of canonical");
    }
}

// P2.4 multi-column dots: ONE tile over nx activation columns must be BITWISE
// equal to nx single-column tiles in canonical and in every fast/int8 path
// (the weight row read is shared; per-(row,column) order is unchanged).
static void test_multicol_dot() {
    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const uint32_t rows = 37, cols = 2048;      // K-format friendly width
    for (int nx : {2, 5, 8, 16}) {   // 16 = kMaxDotCols (P2.5 mc kernels)
        const size_t per = (fastllm::fl_x_alloc_bytes(cols) + 127) & ~(size_t)127;
        const uint32_t xstride = (uint32_t)(per / 4);
        std::vector<float> xbuf((size_t)xstride * nx, 0.f);
        std::vector<float> w((size_t)rows * cols);
        for (auto & v : w) v = d(rng);
        for (int p = 0; p < nx; ++p)
            for (uint32_t c = 0; c < cols; ++c) xbuf[p * xstride + c] = d(rng);
        for (int p = 0; p < nx; ++p) {           // per-column q8 appendices
            Task q{};
            q.kind = fastllm::TaskKind::ACT_Q8;
            q.x = xbuf.data() + (size_t)p * xstride;
            q.y = (void *)fastllm::fl_xq_of(q.x, cols);
            q.rows = cols / 32; q.cols = cols;
            fastllm::exec_task_cpu(q, true, 0);
        }
        const struct { bool fast; uint32_t i8, i8k; } modes[] = {
            {false, 0, 0}, {true, 0, 0}, {true, 0xffu, 0}, {true, 0xffu, 0xffu},
        };
        auto mk = [&](float * y, const float * x) {
            Task t{};
            t.kind = fastllm::TaskKind::STREAM_DOT;
            t.fmt  = fastllm::Fmt::F32;
            t.w = w.data(); t.cols = cols; t.rows = rows;
            t.w_bytes = (uint64_t)rows * cols * 4;
            t.x = x; t.y = y;
            return t;
        };
        for (auto & md : modes) {
            std::vector<float> ys((size_t)rows * nx, 0.f), ym(ys.size(), 0.f);
            for (int p = 0; p < nx; ++p) {        // reference: nx single tiles
                Task t = mk(ys.data() + (size_t)p * rows,
                            xbuf.data() + (size_t)p * xstride);
                fastllm::exec_task_cpu(t, md.fast, md.i8, md.i8k);
            }
            Task t = mk(ym.data(), xbuf.data());  // one multi-column tile
            t.nx = (uint32_t)nx; t.xstride = xstride; t.ystride = rows;
            fastllm::exec_task_cpu(t, md.fast, md.i8, md.i8k);
            CHECK(memcmp(ys.data(), ym.data(), ys.size() * 4) == 0,
                  "multicol dot bitwise == per-column tiles (cpu)");
        }
        // canonical host path (run_serial / GPU-canonical share this loop)
        std::vector<float> yc((size_t)rows * nx, 0.f), yr(yc.size(), 0.f);
        Task tc = mk(yc.data(), xbuf.data());
        tc.nx = (uint32_t)nx; tc.xstride = xstride; tc.ystride = rows;
        fastllm::exec_task_scalar_host(tc);
        for (int p = 0; p < nx; ++p) {
            Task s = mk(yr.data() + (size_t)p * rows,
                        xbuf.data() + (size_t)p * xstride);
            fastllm::exec_task_scalar_host(s);
        }
        CHECK(memcmp(yc.data(), yr.data(), yc.size() * 4) == 0,
              "multicol dot bitwise == per-column tiles (canonical)");
    }
}

// pop_bulk (P0.7): single-consumer semantics, wrap-around, and multi-threaded
// drain equivalence against the single-pop protocol.
static void test_pop_bulk() {
    {   // basic order + partial batches + empty
        std::vector<QueueCell> cells(64);
        Queue q;
        Queue::init(&q, cells.data(), 64);
        for (uint32_t v = 0; v < 10; ++v) CHECK(q.push(v), "bulk: push");
        uint32_t out[16];
        CHECK(q.pop_bulk(out, 4) == 4, "bulk: first claim of 4");
        bool ord = out[0] == 0 && out[1] == 1 && out[2] == 2 && out[3] == 3;
        CHECK(ord, "bulk: FIFO order");
        CHECK(q.pop_bulk(out, 16) == 6, "bulk: partial claim drains rest");
        CHECK(out[5] == 9, "bulk: last value");
        CHECK(q.pop_bulk(out, 16) == 0, "bulk: empty returns 0");
    }
    {   // wrap-around across many cycles on a tiny queue
        std::vector<QueueCell> cells(8);
        Queue q;
        Queue::init(&q, cells.data(), 8);
        uint32_t next_push = 0, next_pop = 0;
        uint32_t out[8];
        std::mt19937 rng(7);
        for (int iter = 0; iter < 1000; ++iter) {
            uint32_t np = rng() % 5;
            for (uint32_t i = 0; i < np && q.push(next_push); ++i) next_push++;
            uint32_t n = q.pop_bulk(out, 1 + rng() % 8);
            for (uint32_t i = 0; i < n; ++i) {
                if (out[i] != next_pop) { CHECK(false, "bulk: wrap order"); break; }
                next_pop++;
            }
        }
        CHECK(next_pop > 400, "bulk: wrap test made progress");
    }
    {   // MT: bulk + single consumers vs producers; exactly-once delivery
        std::vector<QueueCell> cells(4096);
        Queue q;
        Queue::init(&q, cells.data(), 4096);
        const uint32_t kN = 20000; // per producer
        std::vector<std::atomic<uint8_t>> seen(2 * kN);
        for (auto & s : seen) s.store(0, std::memory_order_relaxed);
        std::atomic<uint32_t> consumed{0};
        auto consumer = [&](bool bulk, uint32_t seed) {
            uint32_t out[16];
            std::mt19937 rng(seed);
            while (consumed.load(std::memory_order_acquire) < 2 * kN) {
                uint32_t n = 0;
                if (bulk) n = q.pop_bulk(out, 1 + rng() % 16);
                else      n = q.pop(out) ? 1 : 0;
                if (!n) continue;
                for (uint32_t i = 0; i < n; ++i)
                    seen[out[i]].fetch_add(1, std::memory_order_relaxed);
                consumed.fetch_add(n, std::memory_order_release);
            }
        };
        std::vector<std::thread> ts;
        for (int p = 0; p < 2; ++p) ts.emplace_back([&, p] {
            for (uint32_t v = 0; v < kN; ++v)
                while (!q.push((uint32_t)p * kN + v)) {}
        });
        ts.emplace_back(consumer, true, 11u);
        ts.emplace_back(consumer, true, 12u);
        ts.emplace_back(consumer, false, 13u);
        ts.emplace_back(consumer, false, 14u);
        for (auto & t : ts) t.join();
        bool all = true;
        for (uint32_t v = 0; v < 2 * kN; ++v)
            if (seen[v].load(std::memory_order_relaxed) != 1) { all = false; break; }
        CHECK(all, "bulk MT: every value consumed exactly once");
    }
}

// ---- P1.1: q8_1 activation quantization + int8 dot paths -------------------
static uint32_t g_rs = 0x1234567u;
static float rndf() {           // [-1, 1)
    g_rs = g_rs * 1664525u + 1013904223u;
    return (float)(g_rs >> 8) * (1.0f / 8388608.0f) - 1.0f;
}
static void rnd_bytes(uint8_t * p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        g_rs = g_rs * 1664525u + 1013904223u;
        p[i] = (uint8_t)(g_rs >> 13);
    }
}
// sane random half in a small-positive range (random full halfs give inf/nan)
static uint16_t rnd_half() {
    g_rs = g_rs * 1664525u + 1013904223u;
    return (uint16_t)(0x2000u | ((g_rs >> 9) & 0x0BFFu));
}
// place sane scale halfs into a random block row (layouts per quants.h)
static void fix_scales(Fmt f, uint8_t * row, uint32_t nb, uint32_t blk_b) {
    for (uint32_t b = 0; b < nb; ++b) {
        uint8_t * bp = row + (size_t)b * blk_b;
        uint16_t h = rnd_half();
        if (f == Fmt::Q6_K) memcpy(bp + 208, &h, 2);
        else {
            memcpy(bp, &h, 2);
            if (f == Fmt::Q4_K) { h = rnd_half(); memcpy(bp + 2, &h, 2); }
        }
    }
}

// f64 dequant-dot reference: canonical dequant formulas, double summation.
// Also returns the dot's magnitude scale S = ||w.*x||_2 — the error metric
// divides by S, not |ref|, so sign cancellation cannot inflate the ratio.
static double dot_f64_ref(Fmt f, const uint8_t * row, const float * x,
                          uint32_t cols, double * scale) {
    std::vector<float> dq(cols);
    if (f == Fmt::Q8_0) {
        for (uint32_t b = 0; b < cols / 32; ++b) {
            const blk_q8_0 * B = (const blk_q8_0 *)(row + (size_t)b * 34);
            const float d = fl_half2float(B->d);
            for (int i = 0; i < 32; ++i) dq[b * 32 + i] = d * (float)B->qs[i];
        }
    } else {
        dequant_chunk(f, row, 0, cols, dq.data());
    }
    double acc = 0.0, s2 = 0.0;
    for (uint32_t i = 0; i < cols; ++i) {
        const double p = (double)dq[i] * (double)x[i];
        acc += p;
        s2  += p * p;
    }
    *scale = sqrt(s2);
    return acc;
}

static void quantize_x_appendix(float * xf, uint32_t cols) {
    Task aq{};
    aq.kind = TaskKind::ACT_Q8;
    aq.x = xf;
    aq.y = (void *)fl_xq_of(xf, cols);
    aq.rows = cols / 32;
    aq.cols = cols;
    fl_exec_act_q8(aq);
}

static void test_q81_quantize() {
    alignas(16) float x[64];
    for (int i = 0; i < 64; ++i) x[i] = rndf();
    blk_q8_1 a[2], b[2];
    fl_quantize_q8_1_block(x, &a[0]);      fl_quantize_q8_1_block(x + 32, &a[1]);
    fl_quantize_q8_1_block(x, &b[0]);      fl_quantize_q8_1_block(x + 32, &b[1]);
    CHECK(memcmp(a, b, sizeof a) == 0, "q8_1: quantization deterministic");
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, fabsf(x[i]));
    bool sane = true;
    const float d = fl_half2float(a[0].d);
    for (int i = 0; i < 32; ++i) {
        const float rec = d * a[0].qs[i];
        if (fabsf(rec - x[i]) > amax / 127.f + 1e-4f) sane = false;
    }
    CHECK(sane, "q8_1: reconstruction within one quantization step");
    CHECK(fl_float2half(1.0f) == 0x3C00, "f2h: 1.0");
    CHECK(fl_float2half(-2.0f) == 0xC000, "f2h: -2.0");
    CHECK(fl_float2half(65504.f) == 0x7BFF, "f2h: max half");
    CHECK(fl_float2half(1e30f) == 0x7C00, "f2h: overflow -> inf");
    CHECK(fl_float2half(5.96046448e-8f) == 0x0001, "f2h: min subnormal");
}

static void test_i8_budget() {
    const struct { Fmt f; uint32_t blk_b, blk_e; const char * n; } cases[] = {
        { Fmt::Q8_0,  34,  32, "q8_0" }, { Fmt::Q5_0,  22,  32, "q5_0" },
        { Fmt::Q4_K, 144, 256, "q4_K" }, { Fmt::Q6_K, 210, 256, "q6_K" },
    };
    const uint32_t cols = 512;
    for (const auto & cs : cases) {
        const uint32_t nb = cols / cs.blk_e;
        std::vector<uint8_t> row((size_t)nb * cs.blk_b);
        std::vector<float> x(cols);
        std::vector<uint8_t> xqb(fl_x_alloc_bytes(cols) + 16);
        double max_i8 = 0, sum_i8 = 0, max_cn = 0;
        const int trials = 200;
        for (int t = 0; t < trials; ++t) {
            rnd_bytes(row.data(), row.size());
            fix_scales(cs.f, row.data(), nb, cs.blk_b);
            float * xf = (float *)(((uintptr_t)xqb.data() + 15) & ~(uintptr_t)15);
            for (uint32_t i = 0; i < cols; ++i) { x[i] = rndf(); xf[i] = x[i]; }
            quantize_x_appendix(xf, cols);
            double S = 0;
            const double ref = dot_f64_ref(cs.f, row.data(), x.data(), cols, &S);
            const double den = std::max(1e-6, S);
            const double v8  = (double)fl_dot_i8_any(cs.f, row.data(),
                                   fl_xq_of(xf, cols), cols);
            const double vcn = (double)fl_dot_row_any(cs.f, row.data(),
                                   x.data(), cols);
            max_i8 = std::max(max_i8, fabs(v8 - ref) / den);
            sum_i8 += fabs(v8 - ref) / den;
            max_cn = std::max(max_cn, fabs(vcn - ref) / den);
        }
        printf("  i8-budget %s: max_rel %.3e mean_rel %.3e (canonical max %.3e)\n",
               cs.n, max_i8, sum_i8 / trials, max_cn);
        // measured (x86, 200 trials, cols=512, x ~ U(-1,1), S-normalized):
        //   q8_0 8.1e-3/3.0e-3  q5_0 1.29e-2/3.0e-3
        //   q4_K 1.11e-2/3.1e-3 q6_K 1.11e-2/2.9e-3   (max/mean)
        // = activation-quantization error class; gates set ~2x above.
        CHECK(max_i8 < 2.5e-2, "i8 budget: max rel under 2.5e-2");
        CHECK(sum_i8 / trials < 6e-3, "i8 budget: mean rel under 6e-3");
        CHECK(max_cn < 1e-5, "canonical vs f64 sanity");
    }
}

static void test_i8_simd_bitwise() {
    // dot_row_i8 must equal scalar fl_dot_i8_any bitwise (on ARM+DOTPROD this
    // exercises the SDOT kernels; elsewhere both sides share the scalar code).
    const struct { Fmt f; uint32_t blk_b, blk_e; } cases[] = {
        { Fmt::Q8_0, 34, 32 }, { Fmt::Q4_K, 144, 256 }, { Fmt::Q6_K, 210, 256 },
    };
    const uint32_t cols = 1024;
    for (const auto & cs : cases) {
        const uint32_t nb = cols / cs.blk_e;
        std::vector<uint8_t> row((size_t)nb * cs.blk_b);
        std::vector<uint8_t> xqb(fl_x_alloc_bytes(cols) + 16);
        bool same = true;
        for (int t = 0; t < 50 && same; ++t) {
            rnd_bytes(row.data(), row.size());
            fix_scales(cs.f, row.data(), nb, cs.blk_b);
            float * xf = (float *)(((uintptr_t)xqb.data() + 15) & ~(uintptr_t)15);
            for (uint32_t i = 0; i < cols; ++i) xf[i] = rndf();
            quantize_x_appendix(xf, cols);
            const float a = dot_row_i8(cs.f, row.data(), fl_xq_of(xf, cols), cols);
            const float b = fl_dot_i8_any(cs.f, row.data(), fl_xq_of(xf, cols), cols);
            if (memcmp(&a, &b, 4) != 0) same = false;
        }
        CHECK(same, "int8 SIMD == scalar reference bitwise");
    }
}

static void test_i8_q4k_v2_bitwise() {
    g_rs = 0xC0FFEEu;
    // P1.2: exec_task_cpu's q4_K v2 path (precomputed activation sums +
    // branchless scales) must stay bitwise-equal to the v1/reference order,
    // across odd row counts and both real tensor widths.
    for (uint32_t cols : { 1024u, 2048u, 2816u }) {
        const uint32_t nb = cols / 256, rows = 5;
        std::vector<uint8_t> w((size_t)rows * nb * 144);
        std::vector<uint8_t> xqb(fl_x_alloc_bytes(cols) + 16);
        bool same = true;
        for (int t = 0; t < 20 && same; ++t) {
            rnd_bytes(w.data(), w.size());
            fix_scales(Fmt::Q4_K, w.data(), rows * nb, 144);
            float * xf = (float *)(((uintptr_t)xqb.data() + 15) & ~(uintptr_t)15);
            for (uint32_t i = 0; i < cols; ++i) xf[i] = rndf();
            quantize_x_appendix(xf, cols);
            std::vector<float> y2(rows, 0.f), yr(rows, 0.f);
            Task tk{};
            tk.kind = fastllm::TaskKind::STREAM_DOT;
            tk.fmt = Fmt::Q4_K;
            tk.w = w.data(); tk.x = xf; tk.rows = rows; tk.cols = cols;
            tk.w_bytes = (uint64_t)rows * fl_row_bytes(Fmt::Q4_K, cols);
            tk.y = y2.data();
            fastllm::exec_task_cpu(tk, true, 1u << (int)Fmt::Q4_K);
            const blk_q8_1 * xq = fl_xq_of(xf, cols);
            for (uint32_t r = 0; r < rows; ++r)
                yr[r] += fl_dot_i8_q4_K(w.data() + (size_t)r * nb * 144, xq, cols);
            if (memcmp(y2.data(), yr.data(), rows * 4) != 0) same = false;
        }
        CHECK(same, "q4_K v2 tile path == int8 reference bitwise");
    }
}

// ---- P1.5 q8_K activation appendix ----------------------------------------
static void test_q8k_quantize() {
    g_rs = 0x5EEDBEEFu;
    alignas(16) float x[256];
    for (int i = 0; i < 256; ++i) x[i] = rndf();
    blk_q8_K a, b;
    fl_quantize_q8_K_block(x, &a);
    fl_quantize_q8_K_block(x, &b);
    CHECK(memcmp(&a, &b, sizeof a) == 0, "q8_K: quantization deterministic");
    bool bs_ok = true;
    for (int g = 0; g < 16; ++g) {
        int s = 0;
        for (int i = 0; i < 16; ++i) s += (int)a.qs[g * 16 + i];
        if (s != (int)a.bsums[g]) bs_ok = false;
    }
    CHECK(bs_ok, "q8_K: bsums equal per-16 sums of qs");
    float amax = 0.f;
    for (int i = 0; i < 256; ++i) amax = std::max(amax, fabsf(x[i]));
    bool sane = true;
    for (int i = 0; i < 256; ++i)
        if (fabsf(a.d * a.qs[i] - x[i]) > amax / 127.f + 1e-4f) sane = false;
    CHECK(sane, "q8_K: reconstruction within one quantization step");
    // appendix coexistence: both regions written by one ACT_Q8, disjoint
    const uint32_t cols = 512;
    std::vector<uint8_t> xqb(fl_x_alloc_bytes(cols) + 16);
    float * xf = (float *)(((uintptr_t)xqb.data() + 15) & ~(uintptr_t)15);
    for (uint32_t i = 0; i < cols; ++i) xf[i] = rndf();
    quantize_x_appendix(xf, cols);
    const blk_q8_1 * xq = fl_xq_of(xf, cols);
    const blk_q8_K * xk = fl_xqk_of(xf, cols);
    CHECK((const uint8_t *)xk >= (const uint8_t *)(xq + cols / 32),
          "q8_K region sits after the q8_1 region");
    CHECK(((uintptr_t)xk % 4) == 0, "q8_K region 4-aligned");
    blk_q8_K ref0;
    fl_quantize_q8_K_block(xf, &ref0);
    CHECK(memcmp(&ref0, &xk[0], sizeof ref0) == 0,
          "ACT_Q8 fills the q8_K appendix identically to direct quantization");
}

static void test_i8k_budget() {
    const struct { Fmt f; uint32_t blk_b; const char * n; } cases[] = {
        { Fmt::Q4_K, 144, "q4_K" }, { Fmt::Q6_K, 210, "q6_K" },
    };
    const uint32_t cols = 512;
    for (const auto & cs : cases) {
        const uint32_t nb = cols / 256;
        std::vector<uint8_t> row((size_t)nb * cs.blk_b);
        std::vector<float> x(cols);
        std::vector<uint8_t> xqb(fl_x_alloc_bytes(cols) + 16);
        double max_k = 0, sum_k = 0;
        const int trials = 200;
        for (int t = 0; t < trials; ++t) {
            rnd_bytes(row.data(), row.size());
            fix_scales(cs.f, row.data(), nb, cs.blk_b);
            float * xf = (float *)(((uintptr_t)xqb.data() + 15) & ~(uintptr_t)15);
            for (uint32_t i = 0; i < cols; ++i) { x[i] = rndf(); xf[i] = x[i]; }
            quantize_x_appendix(xf, cols);
            double S = 0;
            const double ref = dot_f64_ref(cs.f, row.data(), x.data(), cols, &S);
            const double den = std::max(1e-6, S);
            const double vk  = (double)fl_dot_i8k_any(cs.f, row.data(),
                                   fl_xqk_of(xf, cols), cols);
            max_k = std::max(max_k, fabs(vk - ref) / den);
            sum_k += fabs(vk - ref) / den;
        }
        printf("  i8k-budget %s: max_rel %.3e mean_rel %.3e\n",
               cs.n, max_k, sum_k / trials);
        // q8_K: per-256 activation scale (coarser than q8_1's per-32) but the
        // scale folding is exact-integer. Measured x86, 200 trials, cols=512:
        // q4_K 1.41e-2/2.93e-3, q6_K 1.24e-2/2.97e-3 (max/mean) - same class
        // as the q8_1 budget (mean parity, max ~25% wider); gates ~2x above
        // measured, matching test_i8_budget's convention.
        CHECK(max_k < 3e-2, "i8k budget: max rel under 3e-2");
        CHECK(sum_k / trials < 6e-3, "i8k budget: mean rel under 6e-3");
    }
}

static void test_i8k_simd_bitwise() {
    // dot_row_i8k must equal fl_dot_i8k_any bitwise (ARM+DOTPROD: NEON
    // integer-fold kernels; elsewhere both sides share the scalar code).
    const struct { Fmt f; uint32_t blk_b; } cases[] = {
        { Fmt::Q4_K, 144 }, { Fmt::Q6_K, 210 },
    };
    for (uint32_t cols : { 512u, 2048u, 2816u }) {
        for (const auto & cs : cases) {
            const uint32_t nb = cols / 256;
            std::vector<uint8_t> row((size_t)nb * cs.blk_b);
            std::vector<uint8_t> xqb(fl_x_alloc_bytes(cols) + 16);
            bool same = true;
            for (int t = 0; t < 30 && same; ++t) {
                rnd_bytes(row.data(), row.size());
                fix_scales(cs.f, row.data(), nb, cs.blk_b);
                float * xf = (float *)(((uintptr_t)xqb.data() + 15) & ~(uintptr_t)15);
                for (uint32_t i = 0; i < cols; ++i) xf[i] = rndf();
                quantize_x_appendix(xf, cols);
                const blk_q8_K * xk = fl_xqk_of(xf, cols);
                const float a = dot_row_i8k(cs.f, row.data(), xk, cols);
                const float b = fl_dot_i8k_any(cs.f, row.data(), xk, cols);
                if (memcmp(&a, &b, 4) != 0) same = false;
            }
            CHECK(same, "i8k SIMD == scalar reference bitwise");
        }
    }
}

static void test_i8k_exec_routing() {
    g_rs = 0xD15BA7Cu;
    // exec_task_cpu routes K tiles to the q8_K path iff the i8k mask bit is
    // set; with it clear the q8_1 path result must come back unchanged.
    const uint32_t cols = 1024, rows = 3, nb = cols / 256;
    std::vector<uint8_t> w((size_t)rows * nb * 144);
    std::vector<uint8_t> xqb(fl_x_alloc_bytes(cols) + 16);
    rnd_bytes(w.data(), w.size());
    fix_scales(Fmt::Q4_K, w.data(), rows * nb, 144);
    float * xf = (float *)(((uintptr_t)xqb.data() + 15) & ~(uintptr_t)15);
    for (uint32_t i = 0; i < cols; ++i) xf[i] = rndf();
    quantize_x_appendix(xf, cols);
    std::vector<float> yk(rows, 0.f), y1(rows, 0.f), yr(rows, 0.f);
    Task tk{};
    tk.kind = fastllm::TaskKind::STREAM_DOT;
    tk.fmt = Fmt::Q4_K;
    tk.w = w.data(); tk.x = xf; tk.rows = rows; tk.cols = cols;
    tk.w_bytes = (uint64_t)rows * fl_row_bytes(Fmt::Q4_K, cols);
    tk.y = yk.data();
    fastllm::exec_task_cpu(tk, true, 1u << (int)Fmt::Q4_K, 1u << (int)Fmt::Q4_K);
    tk.y = y1.data();
    fastllm::exec_task_cpu(tk, true, 1u << (int)Fmt::Q4_K, 0);
    const blk_q8_K * xk = fl_xqk_of(xf, cols);
    for (uint32_t r = 0; r < rows; ++r)
        yr[r] += fl_dot_i8k_q4_K(w.data() + (size_t)r * nb * 144, xk, cols);
    CHECK(memcmp(yk.data(), yr.data(), rows * 4) == 0,
          "i8k mask on: tile path == i8k reference bitwise");
    bool differs = memcmp(yk.data(), y1.data(), rows * 4) != 0;
    const blk_q8_1 * xq = fl_xq_of(xf, cols);
    std::vector<float> y1r(rows, 0.f);
    for (uint32_t r = 0; r < rows; ++r)
        y1r[r] += fl_dot_i8_q4_K(w.data() + (size_t)r * nb * 144, xq, cols);
    CHECK(memcmp(y1.data(), y1r.data(), rows * 4) == 0,
          "i8k mask off: q8_1 path intact");
    CHECK(differs, "q8_K and q8_1 paths are distinct schemes (sanity)");
}

static void test_i8_runtime_integration() {
    g_rs = 0x1B57A11u;
    // ACT_Q8 root -> {q8_0 tile, q4_K tile} -> merge -> signal; Runtime
    // cpu-only int8 vs serial canonical within the int8 tolerance.
    auto arena = Arena::create(16u << 20);
    const uint32_t cols = 512, rows = 8;
    const uint64_t rb8 = fl_row_bytes(Fmt::Q8_0, cols);
    const uint64_t rb4 = fl_row_bytes(Fmt::Q4_K, cols);
    uint8_t * w8 = (uint8_t *)arena->alloc(rows * rb8);
    uint8_t * w4 = (uint8_t *)arena->alloc(rows * rb4);
    rnd_bytes(w8, rows * rb8);
    rnd_bytes(w4, rows * rb4);
    for (uint32_t r = 0; r < rows; ++r) {
        fix_scales(Fmt::Q8_0, w8 + r * rb8, cols / 32, 34);
        fix_scales(Fmt::Q4_K, w4 + r * rb4, cols / 256, 144);
    }
    float * x = (float *)arena->alloc(fl_x_alloc_bytes(cols));
    for (uint32_t i = 0; i < cols; ++i) x[i] = rndf();
    float * slots = (float *)arena->alloc(2 * rows * 4);
    float * out   = (float *)arena->alloc(4);

    auto g = Graph::create(*arena, 16);
    Task aq{};
    aq.kind = TaskKind::ACT_Q8; aq.affinity = Engine::CPU;
    aq.x = x; aq.y = (void *)fl_xq_of(x, cols); aq.rows = cols / 32; aq.cols = cols;
    const uint32_t a0 = g->add(aq);
    Task t8{};
    t8.kind = TaskKind::STREAM_DOT; t8.fmt = Fmt::Q8_0; t8.affinity = Engine::CPU;
    t8.w = w8; t8.w_bytes = rows * rb8;
    t8.x = x; t8.y = slots; t8.rows = rows; t8.cols = cols;
    const uint32_t id8 = g->add(t8);
    Task t4{};
    t4.kind = TaskKind::STREAM_DOT; t4.fmt = Fmt::Q4_K; t4.affinity = Engine::CPU;
    t4.w = w4; t4.w_bytes = rows * rb4;
    t4.x = x; t4.y = slots + rows; t4.rows = rows; t4.cols = cols;
    const uint32_t id4 = g->add(t4);
    Task m{};
    m.kind = TaskKind::MERGE; m.affinity = Engine::CPU;
    m.x = slots; m.rows = 1; m.cols = 2 * rows; m.y = out;
    const uint32_t mid = g->add(m);
    Task sg{};
    sg.kind = TaskKind::SIGNAL; sg.affinity = Engine::CPU;
    const uint32_t sid = g->add(sg);
    g->edge(a0, id8); g->edge(a0, id4);
    g->edge(id8, mid); g->edge(id4, mid);
    g->edge(mid, sid);
    g->freeze();

    memset(slots, 0, 2 * rows * 4); out[0] = 0.f;
    run_serial(*g);
    const float ref = out[0];

    RuntimeConfig rc{};
    rc.cpu_threads = 4; rc.gpu_enabled = false;
    rc.fast_kernels = true; rc.int8_dots = true;
    auto rt = Runtime::create(*arena, rc);
    rt->start();
    float v1 = 0, v2 = 0;
    for (int rep = 0; rep < 2; ++rep) {
        memset(slots, 0, 2 * rows * 4); out[0] = 0.f;
        rt->run(*g);
        (rep ? v2 : v1) = out[0];
    }
    rt->stop();
    CHECK(memcmp(&v1, &v2, 4) == 0, "int8 runtime: repeat-run stable");
    // wiring test, not a numerics test (the budget test above is): loose gate
    // with a magnitude floor so sign cancellation in the merge cannot flake it
    const double rel = fabs((double)v1 - ref) / std::max(10.0, fabs((double)ref));
    CHECK(rel < 5e-2, "int8 runtime vs canonical serial within tolerance");
}

// P2.3: ATTN_HEAD striped canonical + NEON fast path.
// (a) striped canonical vs a local sequential (pre-P2.3-order) reference:
//     tolerance-only - the order change is a deliberate canonical revision;
// (b) odd dims take the legacy sequential path (bitwise vs the reference);
// (c) aarch64: NEON path bitwise-equal to the striped canonical.
static void test_attn_head_p23() {
    auto ref_seq = [](const AttnHeadP & p) {
        const uint32_t n = (uint32_t)*p.pos + 1, dk = p.dk, dv = p.dv;
        const float * q = p.q + (size_t)p.head * dk;
        std::vector<float> sc(n);
        float mx = -1e30f;
        for (uint32_t j = 0; j < n; ++j) {
            const float * k = p.kcache + ((size_t)j * p.n_head + p.head) * dk;
            float acc = 0.f;
            for (uint32_t d = 0; d < dk; ++d) acc += q[d] * k[d];
            acc *= p.scale;
            sc[j] = acc;
            if (acc > mx) mx = acc;
        }
        float sum = 0.f;
        for (uint32_t j = 0; j < n; ++j) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
        const float inv = 1.0f / sum;
        float * o = p.out + (size_t)p.head * dv;
        for (uint32_t d = 0; d < dv; ++d) o[d] = 0.f;
        for (uint32_t j = 0; j < n; ++j) {
            const float w = sc[j] * inv;
            const float * v = p.vcache + ((size_t)j * p.n_head + p.head) * dv;
            for (uint32_t d = 0; d < dv; ++d) o[d] += w * v[d];
        }
    };
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    for (uint32_t n : {1u, 7u, 400u}) {
        for (auto dims : {std::pair<uint32_t,uint32_t>{192, 128},
                          std::pair<uint32_t,uint32_t>{50, 33}}) {
            const uint32_t dk = dims.first, dv = dims.second, nh = 2, head = 1;
            std::vector<float> q(nh * dk), kc((size_t)n * nh * dk),
                vc((size_t)n * nh * dv), scr(n), out_a(nh * dv),
                out_b(nh * dv), out_c(nh * dv);
            for (auto & x : q)  x = U(rng);
            for (auto & x : kc) x = U(rng);
            for (auto & x : vc) x = U(rng);
            int32_t pos = (int32_t)n - 1;
            AttnHeadP p{};
            p.q = q.data(); p.kcache = kc.data(); p.vcache = vc.data();
            p.scratch = scr.data(); p.pos = &pos; p.scale = 0.135f;
            p.head = head; p.n_head = nh; p.dk = dk; p.dv = dv; p.max_kv = n;
            p.out = out_a.data();
            fl_op_attn_head(p);
            p.out = out_b.data();
            ref_seq(p);
            const bool striped = (dk % 8 == 0) && (dv % 4 == 0);
            double mrel = 0;
            for (uint32_t d = 0; d < dv; ++d) {
                const double r = fabs((double)out_a[head * dv + d]
                                    - out_b[head * dv + d])
                               / std::max(1e-3, fabs((double)out_b[head * dv + d]));
                mrel = std::max(mrel, r);
            }
            if (striped) {
                CHECK(mrel < 1e-4, "attn_head striped vs sequential tolerance");
            } else {
                CHECK(memcmp(out_a.data() + head * dv, out_b.data() + head * dv,
                             dv * 4) == 0, "attn_head odd-dims legacy bitwise");
            }
#if defined(__ARM_NEON)
            if (striped) {
                p.out = out_c.data();
                fl_op_attn_head_neon(p);
                CHECK(memcmp(out_c.data() + head * dv, out_a.data() + head * dv,
                             dv * 4) == 0, "attn_head NEON bitwise vs canonical");
            }
#endif
            (void)out_c;
        }
    }
}


// ---- P5.0: codebook format layout + reference sanity ----------------------
// The GPU kernels are the production path (Thor's CPU is gather-bound on these
// formats), so what a host-side test can prove is the LAYOUT and the reference
// semantics: block sizes, index/sign/scale decode, and the byte-negate
// invariant the GPU kernels rely on. The GPU-vs-reference numeric gate runs on
// thor via membench --iq-check (device code cannot run in this x86 suite).
static void test_iq_codebook_layout() {
    using namespace fastllm;
    CHECK(sizeof(blk_iq2_xxs) == 66, "iq2_xxs block is 66 B");
    CHECK(sizeof(blk_iq3_xxs) == 98, "iq3_xxs block is 98 B");
    CHECK(fl_blk_elems(Fmt::IQ2_XXS) == 256 && fl_blk_bytes(Fmt::IQ2_XXS) == 66,
          "iq2_xxs block geometry");
    CHECK(fl_blk_elems(Fmt::IQ3_XXS) == 256 && fl_blk_bytes(Fmt::IQ3_XXS) == 98,
          "iq3_xxs block geometry");
    CHECK(fl_row_bytes(Fmt::IQ2_XXS, 512) == 132, "iq2_xxs row bytes");

    // every grid byte >= 1: what makes (v ^ 0xFF) + 1 carry-free per byte
    int lo2 = 255, lo3 = 255;
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 8; ++j) {
            int b = (int)((fl_kIq2xxsGrid(i) >> (8 * j)) & 0xFF);
            if (b < lo2) lo2 = b;
            if (j < 4) { int c = (int)((fl_kIq3xxsGrid(i) >> (8 * j)) & 0xFF);
                         if (c < lo3) lo3 = c; }
        }
    CHECK(lo2 >= 1, "iq2 grid bytes are all >= 1 (negate is carry-free)");
    CHECK(lo3 >= 1, "iq3 grid bytes are all >= 1 (negate is carry-free)");

    // sign table: bit j of fl_kIqSigns(i) must equal the parity convention ggml
    // uses - index i carries its own 7 bits plus a parity bit in position 7
    int sign_ok = 1;
    for (int i = 0; i < 128; ++i) {
        int pc = 0;
        for (int b = 0; b < 7; ++b) pc += (i >> b) & 1;
        const int expect = i | ((pc & 1) << 7);
        if (fl_kIqSigns(i) != expect) sign_ok = 0;
    }
    CHECK(sign_ok, "ksigns table is index + parity bit");

    // hand-decode one synthetic block against fl_dot_row_iq2_xxs: pick grid
    // entry 0 for all four chunks, no signs, scale nibble 0 -> db = d*0.5*0.25
    uint8_t row[66] = {0};
    const uint16_t d_half = 0x3C00;                 // 1.0
    memcpy(row, &d_half, 2);
    std::vector<float> x(256, 0.f);
    x[3] = 2.0f;                                    // single live element
    const float got = fl_dot_row_iq2_xxs(row, x.data(), 256);
    const float gb  = (float)(uint8_t)(fl_kIq2xxsGrid(0) >> 24);   // element 3
    const float want = 1.0f * (0.5f + 0.f) * 0.25f * gb * 2.0f;
    CHECK(std::fabs(got - want) < 1e-6f, "iq2_xxs reference decodes one element");

    // dispatch reaches the codebook references rather than returning 0
    CHECK(fl_dot_row_any(Fmt::IQ2_XXS, row, x.data(), 256) == got,
          "fl_dot_row_any routes IQ2_XXS");
}

int main() {
    test_diamond_and_repeat();
    for (uint32_t s = 1; s <= 20; ++s) test_fuzz(s, 200);
    test_moe_shaped_integration();
    test_fast_kernels();
    test_multicol_dot();
    test_pop_bulk();
    test_q81_quantize();
    test_i8_budget();
    test_i8_simd_bitwise();
    test_i8_q4k_v2_bitwise();
    test_q8k_quantize();
    test_i8k_budget();
    test_i8k_simd_bitwise();
    test_i8k_exec_routing();
    test_i8_runtime_integration();
    test_attn_head_p23();
    test_iq_codebook_layout();
    printf("fastllm-tests: %d checks, %d failures\n", g_checks, g_fails);
    return g_fails ? 1 : 0;
}
