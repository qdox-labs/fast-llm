// P1.0 quant-format tests: block layouts, GGUF reader, runtime integration.
// The independent LAYOUT gate against a real GGUF (gguf-py + numpy f64 dot on
// thor) lives in the bench --probe-row flow; these tests catch reader bugs,
// dispatch bugs, and CPU fast-vs-canonical divergence with no model file.
#include "fastllm/fastllm.h"
#include "runtime/quants.h"
#include "runtime/kernels.h"
#include "model/gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace fastllm;

static int checks = 0, fails = 0;
#define CHECK(c) do { ++checks; if (!(c)) { ++fails; \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

// half encode (round-to-nearest not needed: test values are exact halves)
static uint16_t h(float f) {
    // exact for powers of two and small sums thereof used below
    uint32_t b; memcpy(&b, &f, 4);
    uint32_t s = (b >> 16) & 0x8000;
    int32_t  e = (int32_t)((b >> 23) & 0xFF) - 127 + 15;
    uint32_t m = (b >> 13) & 0x3FF;
    if (f == 0.f) return (uint16_t)s;
    return (uint16_t)(s | ((uint32_t)e << 10) | m);
}

// ---- independent per-element dequant (transcribed separately from the
// kernels; same ggml spec, different code path) -----------------------------
static float ref_elem_q8_0(const uint8_t * blk, int i) {
    uint16_t d; memcpy(&d, blk, 2);
    return fl_half2float(d) * (float)(int8_t)blk[2 + i];
}
static float ref_elem_q5_0(const uint8_t * blk, int i) {
    uint16_t d; memcpy(&d, blk, 2);
    uint32_t qh; memcpy(&qh, blk + 2, 4);
    int j = i & 15;
    int v;
    if (i < 16) v = (int)((blk[6 + j] & 0xF) | (((qh >> j) << 4) & 0x10)) - 16;
    else        v = (int)((blk[6 + j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16;
    return fl_half2float(d) * (float)v;
}
static float ref_elem_q4_K(const uint8_t * blk, int i) {
    uint16_t dh, mh; memcpy(&dh, blk, 2); memcpy(&mh, blk + 2, 2);
    const float d = fl_half2float(dh), dmin = fl_half2float(mh);
    const uint8_t * scales = blk + 4;
    const uint8_t * qs = blk + 16;
    int sub = i / 32;                 // 8 sub-blocks of 32
    uint8_t sc, mn;
    fl_q4k_scale_min(sub, scales, &sc, &mn);
    int chunk = i / 64, off = i % 64;
    uint8_t q = qs[chunk * 32 + (off % 32)];
    int v = off < 32 ? (q & 0xF) : (q >> 4);
    return d * sc * (float)v - dmin * mn;
}
static float ref_elem_q6_K(const uint8_t * blk, int i) {
    const uint8_t * ql = blk;
    const uint8_t * qh = blk + 128;
    const int8_t  * sc = (const int8_t *)(blk + 192);
    uint16_t dh; memcpy(&dh, blk + 208, 2);
    const float d = fl_half2float(dh);
    int half = i / 128, r = i % 128, grp = r / 32, l = r % 32;
    const uint8_t * qlh = ql + half * 64;
    const uint8_t * qhh = qh + half * 32;
    const int8_t  * sch = sc + half * 8;
    int q;
    switch (grp) {
        case 0: q = (qlh[l] & 0xF)      | (((qhh[l] >> 0) & 3) << 4); break;
        case 1: q = (qlh[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4); break;
        case 2: q = (qlh[l] >> 4)       | (((qhh[l] >> 4) & 3) << 4); break;
        default:q = (qlh[l + 32] >> 4)  | (((qhh[l] >> 6) & 3) << 4); break;
    }
    return d * (float)sch[l / 16 + grp * 2] * (float)(q - 32);
}

template <typename F>
static void check_row_dot(Fmt f, const std::vector<uint8_t> & row,
                          uint32_t cols, F ref_elem) {
    std::vector<float> x(cols);
    for (uint32_t i = 0; i < cols; ++i) x[i] = 1.0f / (1 + i);
    const uint32_t be = fl_blk_elems(f), bb = fl_blk_bytes(f);
    double ref = 0;
    for (uint32_t i = 0; i < cols; ++i)
        ref += (double)ref_elem(row.data() + (i / be) * bb, (int)(i % be))
             * x[i];
    const float got = fl_dot_row_any(f, row.data(), x.data(), cols);
    CHECK(fabs(got - ref) <= 1e-4 * (fabs(ref) + 1.0));
    // CPU fast path agrees within tolerance
    Task t{};
    t.kind = TaskKind::STREAM_DOT;
    t.fmt  = f;
    t.w    = row.data();
    t.cols = cols;
    t.rows = 1;
    t.w_bytes = fl_row_bytes(f, cols);
    float y_c = 0.f, y_f = 0.f;
    t.x = x.data();
    t.y = &y_c; exec_task_cpu(t, false, false);
    t.y = &y_f; exec_task_cpu(t, true, false);
    CHECK(fabs(y_f - y_c) <= 1e-5f * (fabs(y_c) + 1.0f));
}

static std::vector<uint8_t> random_row(Fmt f, uint32_t cols, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> row(fl_row_bytes(f, cols));
    for (auto & b : row) b = (uint8_t)rng();
    // overwrite scale halves with sane exact values so nothing is inf/nan
    const uint32_t bb = fl_blk_bytes(f);
    for (size_t o = 0; o + bb <= row.size(); o += bb) {
        uint16_t dv = h(0.25f), mv = h(0.0625f);
        switch (f) {
            case Fmt::Q8_0: case Fmt::Q5_0:
                memcpy(&row[o], &dv, 2); break;
            case Fmt::Q4_K:
                memcpy(&row[o], &dv, 2); memcpy(&row[o + 2], &mv, 2); break;
            case Fmt::Q6_K:
                memcpy(&row[o + 208], &dv, 2); break;
            default: break;
        }
    }
    return row;
}

// ---- fixture GGUF ---------------------------------------------------------
static void put_str(std::vector<uint8_t> & f, const std::string & s) {
    uint64_t n = s.size();
    f.insert(f.end(), (uint8_t *)&n, (uint8_t *)&n + 8);
    f.insert(f.end(), s.begin(), s.end());
}
template <typename T> static void put(std::vector<uint8_t> & f, T v) {
    f.insert(f.end(), (uint8_t *)&v, (uint8_t *)&v + sizeof(T));
}

static void test_gguf_reader() {
    struct Spec { const char * name; uint32_t type; Fmt fmt; uint32_t ne0, ne1; };
    const Spec specs[] = {
        { "t.q8",  GGML_TYPE_Q8_0, Fmt::Q8_0,  64, 3 },
        { "t.q50", GGML_TYPE_Q5_0, Fmt::Q5_0,  64, 2 },
        { "t.q4k", GGML_TYPE_Q4_K, Fmt::Q4_K, 256, 2 },
        { "t.q6k", GGML_TYPE_Q6_K, Fmt::Q6_K, 256, 1 },
        { "t.f32", GGML_TYPE_F32,  Fmt::F32,   16, 4 },
    };
    std::vector<std::vector<uint8_t>> payloads;
    for (auto & s : specs) {
        std::vector<uint8_t> p;
        for (uint32_t r = 0; r < s.ne1; ++r) {
            auto row = random_row(s.fmt, s.ne0, 77 + r + s.ne0);
            p.insert(p.end(), row.begin(), row.end());
        }
        payloads.push_back(std::move(p));
    }

    std::vector<uint8_t> f;
    put<uint32_t>(f, 0x46554747u);           // GGUF
    put<uint32_t>(f, 3);                     // version
    put<uint64_t>(f, 5);                     // n_tensors
    put<uint64_t>(f, 1);                     // n_kv
    put_str(f, "general.alignment");
    put<uint32_t>(f, 4);                     // U32
    put<uint32_t>(f, 32);
    uint64_t off = 0;
    for (size_t i = 0; i < 5; ++i) {
        put_str(f, specs[i].name);
        put<uint32_t>(f, 2);                 // n_dims
        put<uint64_t>(f, specs[i].ne0);
        put<uint64_t>(f, specs[i].ne1);
        put<uint32_t>(f, specs[i].type);
        off = (off + 31) & ~31ull;
        put<uint64_t>(f, off);
        off += payloads[i].size();
    }
    while (f.size() % 32) f.push_back(0);
    uint64_t data_start = f.size();
    off = 0;
    for (size_t i = 0; i < 5; ++i) {
        uint64_t aligned = (off + 31) & ~31ull;
        while (off < aligned) { f.push_back(0); ++off; }
        f.insert(f.end(), payloads[i].begin(), payloads[i].end());
        off += payloads[i].size();
    }
    (void)data_start;

    const char * path = "/tmp/fastllm-fixture.gguf";
    FILE * fp = fopen(path, "wb");
    fwrite(f.data(), 1, f.size(), fp);
    fclose(fp);

    Gguf gg;
    CHECK(gg.open(path));
    CHECK(gg.alignment() == 32);
    CHECK(gg.tensors().size() == 5);
    for (size_t i = 0; i < 5; ++i) {
        const GgufTensor * t = gg.find(specs[i].name);
        CHECK(t != nullptr);
        CHECK(t->type == specs[i].type);
        CHECK(t->ne[0] == specs[i].ne0 && t->ne[1] == specs[i].ne1);
        CHECK(t->nbytes == payloads[i].size());
        CHECK(memcmp(gg.data(*t), payloads[i].data(), t->nbytes) == 0);
    }
}

// ---- runtime integration: quant tiles through Runtime vs run_serial -------
static void test_runtime_quant() {
    auto arena = Arena::create(64ull << 20);
    auto graph = Graph::create(*arena, 64);
    const Fmt fmts[] = { Fmt::Q8_0, Fmt::Q5_0, Fmt::Q4_K, Fmt::Q6_K, Fmt::F32 };
    const uint32_t cols[] = { 1408, 1408, 2048, 2816, 2048 };
    const uint32_t rows = 8;
    float * x    = (float *)arena->alloc(4096 * 4);
    for (int i = 0; i < 4096; ++i) x[i] = 1.0f / (1 + i);
    float * yc   = (float *)arena->alloc(5 * rows * 4);
    float * fout = (float *)arena->alloc(4);

    std::vector<uint32_t> ids;
    for (int i = 0; i < 5; ++i) {
        auto rowb = random_row(fmts[i], cols[i], 1000 + i);
        uint8_t * w = (uint8_t *)arena->alloc(rowb.size() * rows);
        for (uint32_t r = 0; r < rows; ++r) {
            memcpy(w + r * rowb.size(), rowb.data(), rowb.size());
            // vary rows via a payload byte (offset 16 is inside qs/ql/mantissa
            // for every format, never a scale half -> no inf/nan risk)
            w[r * rowb.size() + 16] ^= (uint8_t)r;
        }
        Task t{};
        t.kind = TaskKind::STREAM_DOT;
        t.fmt  = fmts[i];
        t.affinity = Engine::CPU;
        t.w = w;
        t.w_bytes = rowb.size() * rows;
        t.rows = rows;
        t.cols = cols[i];
        t.x = x;
        t.y = yc + i * rows;
        ids.push_back(graph->add(t));
    }
    Task m{};
    m.kind = TaskKind::MERGE;
    m.affinity = Engine::CPU;
    m.x = yc; m.rows = 1; m.cols = 5 * rows; m.y = fout;
    uint32_t mid = graph->add(m);
    for (uint32_t id : ids) graph->edge(id, mid);
    Task sg{};
    sg.kind = TaskKind::SIGNAL;
    uint32_t tid = graph->add(sg);
    graph->edge(mid, tid);
    graph->freeze();

    auto zero = [&]() { memset(yc, 0, 5 * rows * 4); fout[0] = 0.f; };

    zero();
    run_serial(*graph);
    std::vector<float> ref(yc, yc + 5 * rows);
    const float fref = fout[0];

    RuntimeConfig rc{};
    rc.cpu_threads = 4;
    rc.gpu_enabled = false;
    rc.fast_kernels = false;
    {
        auto rt = Runtime::create(*arena, rc);
        rt->start();
        zero();
        rt->run(*graph);
        rt->stop();
    }
    CHECK(memcmp(ref.data(), yc, 5 * rows * 4) == 0);
    CHECK(memcmp(&fref, fout, 4) == 0);

    rc.fast_kernels = true;
    {
        auto rt = Runtime::create(*arena, rc);
        rt->start();
        zero();
        rt->run(*graph);
        rt->stop();
    }
    double max_rel = 0;
    for (uint32_t i = 0; i < 5 * rows; ++i) {
        double d = fabs((double)yc[i] - ref[i]);
        max_rel = std::max(max_rel, d / std::max(1e-30, fabs((double)ref[i])));
    }
    CHECK(max_rel <= 1e-5);
}

int main() {
    check_row_dot(Fmt::Q8_0, random_row(Fmt::Q8_0, 1408, 1), 1408, ref_elem_q8_0);
    check_row_dot(Fmt::Q5_0, random_row(Fmt::Q5_0, 1408, 2), 1408, ref_elem_q5_0);
    check_row_dot(Fmt::Q4_K, random_row(Fmt::Q4_K, 2048, 3), 2048, ref_elem_q4_K);
    check_row_dot(Fmt::Q6_K, random_row(Fmt::Q6_K, 2816, 4), 2816, ref_elem_q6_K);
    test_gguf_reader();
    test_runtime_quant();
    printf("fastllm-quant-tests: %d checks, %d failures\n", checks, fails);
    return fails ? 1 : 0;
}
