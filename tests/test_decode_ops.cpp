// P2.0 decode-op unit tests: each op vs a double-precision reference.
#include "runtime/runtime_internal.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace fastllm;

static int checks = 0, fails = 0;
#define CHECK(c, msg) do { checks++; if (!(c)) { fails++; \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, msg); } } while (0)

static uint32_t rng_s = 0x1234567u;
static float frand() {
    rng_s = rng_s * 1664525u + 1013904223u;
    return ((rng_s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
}

static void test_rmsnorm() {
    const uint32_t n = 512;
    std::vector<float> x(n), w(n), y(n);
    for (auto & v : x) v = frand() * 3.f;
    for (auto & v : w) v = 1.f + frand();
    RmsNormP p{w.data(), x.data(), y.data(), 1e-6f, n};
    Task t{}; t.kind = TaskKind::RMSNORM; t.aux = &p;
    exec_task_scalar_host(t);
    double ss = 0;
    for (uint32_t i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const double s = 1.0 / sqrt(ss / n + 1e-6);
    double err = 0;
    for (uint32_t i = 0; i < n; ++i)
        err = fmax(err, fabs(y[i] - x[i] * s * w[i]) / (fabs(y[i]) + 1e-9));
    CHECK(err < 1e-5, "rmsnorm vs f64 reference");
}

static void test_rope() {
    // identity at pos 0 with no yarn scaling
    const uint32_t nh = 2, hs = 8, nr = 4;
    std::vector<float> b(nh * hs), b0;
    for (auto & v : b) v = frand();
    b0 = b;
    int32_t pos = 0;
    RopeP p{b.data(), &pos, nh, hs, 2, nr, 10000.f, 1.f, 0.f, 1.f,
            32.f, 1.f, 4096};
    Task t{}; t.kind = TaskKind::ROPE_DS2; t.aux = &p;
    exec_task_scalar_host(t);
    bool same = true;
    for (size_t i = 0; i < b.size(); ++i)
        if (fabsf(b[i] - b0[i]) > 1e-7f) same = false;
    CHECK(same, "rope pos=0 no-yarn is identity");

    // norm preservation per NORM-style adjacent pair at any pos (mscale == 1)
    pos = 137;
    b = b0;
    exec_task_scalar_host(t);
    for (uint32_t h = 0; h < nh; ++h)
        for (uint32_t i = 0; i < nr / 2; ++i) {
            const uint32_t a = h * hs + 2 + 2 * i, c = a + 1;
            const double n0 = (double)b0[a] * b0[a] + (double)b0[c] * b0[c];
            const double n1 = (double)b[a] * b[a] + (double)b[c] * b[c];
            CHECK(fabs(n1 - n0) < 1e-5 * (n0 + 1e-9),
                  "rope preserves pair norm (mscale=1)");
        }
    // untouched dims stay untouched
    CHECK(b[0] == b0[0] && b[1] == b0[1] && b[6] == b0[6],
          "rope leaves non-rot dims alone");

    // yarn low-dim behavior: freq_scale != 1, ext_factor 1 -> first pair uses
    // extrapolated theta (ramp ~1 at dim 0 for small corr_lo) and mscale != 1
    pos = 5;
    b = b0;
    RopeP py = p;
    py.freq_scale = 0.025f; py.ext_factor = 1.f;
    py.attn_factor = 1.0f / (1.0f + 0.1f * logf(1.0f / 0.025f));
    Task ty{}; ty.kind = TaskKind::ROPE_DS2; ty.aux = &py;
    exec_task_scalar_host(ty);
    bool changed = false;
    for (size_t i = 0; i < b.size(); ++i)
        if (b[i] != b0[i]) changed = true;
    CHECK(changed, "yarn rope does something at pos>0");
}

static void test_router() {
    const uint32_t ne = 8, k = 3;
    float logits[ne] = {0.1f, 2.0f, -1.f, 2.0f, 0.5f, 1.9f, -3.f, 0.f};
    uint32_t ids[k]; float gates[k];
    RouterP p{logits, ids, gates, ne, k, 0, 1.0f};
    Task t{}; t.kind = TaskKind::ROUTER_SEL; t.aux = &p;
    exec_task_scalar_host(t);
    CHECK(ids[0] == 1 && ids[1] == 3 && ids[2] == 5,
          "router top-3 order with tie -> lower index first");
    double s = 0;
    for (uint32_t i = 0; i < ne; ++i) s += exp((double)logits[i] - 2.0);
    CHECK(fabs(gates[0] - exp(0.0) / s) < 1e-6, "router softmax weight");
    // norm flag renormalizes
    RouterP pn = p; pn.norm = 1;
    Task tn{}; tn.kind = TaskKind::ROUTER_SEL; tn.aux = &pn;
    exec_task_scalar_host(tn);
    CHECK(fabs(gates[0] + gates[1] + gates[2] - 1.0f) < 1e-6f,
          "router norm renormalizes top-k");
}

static void test_silu_wmerge_add() {
    const uint32_t n = 64;
    std::vector<float> g(n), u(n), h(n);
    for (auto & v : g) v = frand() * 4;
    for (auto & v : u) v = frand();
    SiluMulP ps{g.data(), u.data(), h.data(), n};
    Task t{}; t.kind = TaskKind::SILU_MUL; t.aux = &ps;
    exec_task_scalar_host(t);
    double err = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const double sg = (double)g[i] / (1.0 + exp(-(double)g[i]));
        err = fmax(err, fabs(h[i] - sg * u[i]));
    }
    CHECK(err < 1e-5, "silu_mul vs f64");

    std::vector<float> parts(3 * n), a1(n), a2(n), y(n);
    float gates[3] = {0.5f, 0.25f, -1.f};
    for (auto & v : parts) v = frand();
    for (auto & v : a1) v = frand();
    for (auto & v : a2) v = frand();
    WmergeP pw{parts.data(), gates, a1.data(), a2.data(), y.data(), 2.f, 3, n};
    Task tw{}; tw.kind = TaskKind::WMERGE; tw.aux = &pw;
    exec_task_scalar_host(tw);
    err = 0;
    for (uint32_t i = 0; i < n; ++i) {
        double acc = 0;
        for (int q = 0; q < 3; ++q) acc += (double)gates[q] * parts[q * n + i];
        acc = acc * 2.0 + a1[i] + a2[i];
        err = fmax(err, fabs(y[i] - acc));
    }
    CHECK(err < 1e-4, "wmerge vs f64");

    // P3.5b gate renormalisation. n_full == 0 (above) and n_full == n_parts
    // must both leave the merge BITWISE as it was - the exact path may not
    // move when the approximate mode is compiled in.
    std::vector<float> y_same(n);
    WmergeP pe{parts.data(), gates, a1.data(), a2.data(), y_same.data(),
               2.f, 3, n, 3};
    Task te{}; te.kind = TaskKind::WMERGE; te.aux = &pe;
    exec_task_scalar_host(te);
    bool bitwise = true;
    for (uint32_t i = 0; i < n; ++i)
        if (memcmp(&y[i], &y_same[i], sizeof(float))) bitwise = false;
    CHECK(bitwise, "wmerge n_full==n_parts is bitwise identical");

    // Truncated merge: summing only the first 2 of 3 gates but rescaling them
    // to carry all 3 gates' mass. Reference computed independently in f64.
    std::vector<float> y_tr(n);
    WmergeP pt{parts.data(), gates, a1.data(), a2.data(), y_tr.data(),
               2.f, 2, n, 3};
    Task tt{}; tt.kind = TaskKind::WMERGE; tt.aux = &pt;
    exec_task_scalar_host(tt);
    const double s2 = (double)gates[0] + gates[1];
    const double s3 = s2 + gates[2];
    err = 0;
    for (uint32_t i = 0; i < n; ++i) {
        double acc = 0;
        for (int q = 0; q < 2; ++q)
            acc += (s3 / s2) * (double)gates[q] * parts[q * n + i];
        acc = acc * 2.0 + a1[i] + a2[i];
        err = fmax(err, fabs(y_tr[i] - acc));
    }
    CHECK(err < 1e-4, "wmerge renormalised truncation vs f64");
    // and the rescale must actually change the result vs not renormalising
    std::vector<float> y_raw(n);
    WmergeP pr{parts.data(), gates, a1.data(), a2.data(), y_raw.data(),
               2.f, 2, n, 0};
    Task tr{}; tr.kind = TaskKind::WMERGE; tr.aux = &pr;
    exec_task_scalar_host(tr);
    bool differs = false;
    for (uint32_t i = 0; i < n; ++i)
        if (fabs(y_raw[i] - y_tr[i]) > 1e-6f) { differs = true; break; }
    CHECK(differs, "renormalisation changes the truncated merge");
}

static void test_attn_head_and_kv() {
    const uint32_t NH = 2, DN = 4, DR = 2, DV = 3, MK = 8;
    const uint32_t DK = DN + DR;
    std::vector<float> kvb(NH * (DN + DV)), kpe(DR);
    std::vector<float> kc(MK * NH * DK, -9.f), vc(MK * NH * DV, -9.f);
    for (auto & v : kvb) v = frand();
    for (auto & v : kpe) v = frand();
    int32_t pos = 3;
    KvAppendP pa{kvb.data(), kpe.data(), kc.data(), vc.data(), &pos,
                 NH, DN, DR, DV, MK};
    Task ta{}; ta.kind = TaskKind::KV_APPEND; ta.aux = &pa;
    exec_task_scalar_host(ta);
    CHECK(kc[(3 * NH + 1) * DK + 0] == kvb[1 * (DN + DV) + 0],
          "kv_append k_nope placement");
    CHECK(kc[(3 * NH + 0) * DK + DN + 1] == kpe[1],
          "kv_append shared k_pe placement");
    CHECK(vc[(3 * NH + 1) * DV + 2] == kvb[1 * (DN + DV) + DN + 2],
          "kv_append v placement");
    CHECK(kc[0] == -9.f, "kv_append does not touch other rows");

    // attention: fill rows 0..pos, compare vs f64 reference
    for (uint32_t j = 0; j <= (uint32_t)pos; ++j)
        for (uint32_t hh = 0; hh < NH; ++hh) {
            for (uint32_t d = 0; d < DK; ++d)
                kc[(j * NH + hh) * DK + d] = frand();
            for (uint32_t d = 0; d < DV; ++d)
                vc[(j * NH + hh) * DV + d] = frand();
        }
    std::vector<float> q(NH * DK), scr(MK), out(NH * DV);
    for (auto & v : q) v = frand();
    const float scale = 0.37f;
    AttnHeadP ph{q.data(), kc.data(), vc.data(), scr.data(), out.data(),
                 &pos, scale, 1, NH, DK, DV, MK};
    Task th{}; th.kind = TaskKind::ATTN_HEAD; th.aux = &ph;
    exec_task_scalar_host(th);
    // f64 reference for head 1
    double sc[8], mx = -1e30;
    for (uint32_t j = 0; j <= (uint32_t)pos; ++j) {
        double a = 0;
        for (uint32_t d = 0; d < DK; ++d)
            a += (double)q[DK + d] * kc[(j * NH + 1) * DK + d];
        sc[j] = a * scale;
        if (sc[j] > mx) mx = sc[j];
    }
    double sum = 0;
    for (uint32_t j = 0; j <= (uint32_t)pos; ++j) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
    for (uint32_t d = 0; d < DV; ++d) {
        double o = 0;
        for (uint32_t j = 0; j <= (uint32_t)pos; ++j)
            o += sc[j] / sum * vc[(j * NH + 1) * DV + d];
        CHECK(fabs(out[NH == 2 ? DV + d : d] - o) < 1e-5,
              "attn head vs f64 reference");
    }
}

static void test_idx_dot() {
    const uint32_t rows = 4, cols = 32, nexp = 3;
    std::vector<float> w(nexp * rows * cols), x(cols), y(rows, 0.f);
    for (auto & v : w) v = frand();
    for (auto & v : x) v = frand();
    uint32_t id = 2;
    Task t{};
    t.kind = TaskKind::STREAM_DOT_IDX;
    t.fmt = Fmt::F32;
    t.w = w.data();
    t.w_bytes = (uint64_t)rows * cols * 4;
    t.aux = &id;
    t.aux2 = (uint64_t)rows * cols * 4;
    t.x = x.data(); t.y = y.data(); t.rows = rows; t.cols = cols;
    exec_task_scalar_host(t);
    for (uint32_t r = 0; r < rows; ++r) {
        double ref = 0;
        for (uint32_t c = 0; c < cols; ++c)
            ref += (double)w[(2 * rows + r) * cols + c] * x[c];
        CHECK(fabs(y[r] - ref) < 1e-5, "idx dot picks expert 2");
    }
}

int main() {
    test_rmsnorm();
    test_rope();
    test_router();
    test_silu_wmerge_add();
    test_attn_head_and_kv();
    test_idx_dot();
    fprintf(stderr, "fastllm-decode-tests: %d checks, %d failures\n",
            checks, fails);
    return fails ? 1 : 0;
}
