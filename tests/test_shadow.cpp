// P3.2 shadow gates: the shadow paths must be BITWISE identical to the packed
// paths (integer work exact, same fmaf fold; see src/runtime/shadow.h).
// Comparisons are on float bit patterns (memcmp). NOTE (P4.4): that is only
// meaningful when the f16 scales are FINITE - see finite_half() below.
#include "fastllm/fastllm.h"
#include "runtime/quants.h"
#include "runtime/kernels.h"
#include "runtime/shadow.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace fastllm;

static int checks = 0, fails = 0;
#define CHECK(c) do { ++checks; if (!(c)) { ++fails; \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

static bool bits_eq(float a, float b) { return !memcmp(&a, &b, 4); }

// P4.4: quantized payloads may be arbitrary, but the f16 SCALES must be
// finite - a real GGUF never stores inf/NaN there. With fully random bytes
// the scales decode to inf/NaN, and then two arithmetically equivalent
// orders can produce differently-SIGNED NaNs (-nan vs nan): memcmp is not
// "immune to NaN semantics" as this file used to claim, so the bitwise gates
// reported spurious failures the moment NEON changed the instruction order.
static uint16_t finite_half(std::mt19937 & rng) {
    // sign + exponent in [1, 30] (no zero/subnormal-only, no inf/NaN)
    const uint16_t s = (uint16_t)((rng() & 1u) << 15);
    const uint16_t e = (uint16_t)(1 + (rng() % 30));
    return (uint16_t)(s | (e << 10) | (uint16_t)(rng() & 0x3FFu));
}
static std::vector<uint8_t> random_row(Fmt f, uint32_t cols, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> row(fl_row_bytes(f, cols));
    for (auto & b : row) b = (uint8_t)rng();
    if (f == Fmt::Q4_K) {                       // d, dmin at block offsets 0,2
        for (size_t o = 0; o + 144 <= row.size(); o += 144) {
            const uint16_t d = finite_half(rng), dm = finite_half(rng);
            memcpy(row.data() + o, &d, 2);
            memcpy(row.data() + o + 2, &dm, 2);
        }
    } else if (f == Fmt::Q5_0) {                // d at block offset 0
        for (size_t o = 0; o + 22 <= row.size(); o += 22) {
            const uint16_t d = finite_half(rng);
            memcpy(row.data() + o, &d, 2);
        }
    }
    return row;
}
// Same sanitation for the multi-row buffers the exec/registry gates build.
static void sanitize_scales(Fmt f, std::vector<uint8_t> & buf, uint32_t seed) {
    std::mt19937 rng(seed);
    const size_t bs = (f == Fmt::Q4_K) ? 144 : 22;
    for (size_t o = 0; o + bs <= buf.size(); o += bs) {
        const uint16_t d = finite_half(rng);
        memcpy(buf.data() + o, &d, 2);
        if (f == Fmt::Q4_K) {
            const uint16_t dm = finite_half(rng);
            memcpy(buf.data() + o + 2, &dm, 2);
        }
    }
}

// activation with q8_K and q8_1 appendices, quantized by the real quantizers
struct Act {
    std::vector<uint8_t> buf;
    const float * x() const { return (const float *)buf.data(); }
};
static Act random_act(uint32_t cols, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    Act a;
    a.buf.resize(fl_x_alloc_bytes(cols));
    float * x = (float *)a.buf.data();
    for (uint32_t i = 0; i < cols; ++i) x[i] = nd(rng);
    Task t{};
    t.kind = TaskKind::ACT_Q8;
    t.x = t.y = x;
    t.cols = cols;
    fl_exec_act_q8(t);
    return a;
}

static void test_q4k_bitwise() {
    for (uint32_t cols : {256u, 1408u, 2816u}) {
        for (uint32_t r = 0; r < 32; ++r) {
            auto row = random_row(Fmt::Q4_K, cols, 100 + r * 7 + cols);
            auto act = random_act(cols, 900 + r + cols);
            const blk_q8_K * xk = fl_xqk_of(act.x(), cols);
            std::vector<uint8_t> sh(fl_shadow_row_bytes(Fmt::Q4_K, cols));
            fl_shadow_build_rows_q4k(row.data(), 1, cols, sh.data());
            const float a = fl_dot_i8k_q4_K(row.data(), xk, cols);
            const float b = dot_i8k_q4_K_shadow(sh.data(), xk, cols);
            CHECK(bits_eq(a, b));
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
            const float c = dot_i8k_q4_K_neon(row.data(), xk, cols);
            CHECK(bits_eq(b, c));
#endif
        }
    }
}

static void test_q50_bitwise() {
    for (uint32_t cols : {32u, 1408u, 2048u}) {
        for (uint32_t r = 0; r < 32; ++r) {
            auto row = random_row(Fmt::Q5_0, cols, 300 + r * 11 + cols);
            auto act = random_act(cols, 700 + r + cols);
            const blk_q8_1 * xq = fl_xq_of(act.x(), cols);
            std::vector<uint8_t> sh(fl_shadow_row_bytes(Fmt::Q5_0, cols));
            fl_shadow_build_rows_q50(row.data(), 1, cols, sh.data());
            const float a = fl_dot_i8_q5_0(row.data(), xq, cols);
            const float b = dot_i8_q5_0_shadow(sh.data(), xq, cols);
            CHECK(bits_eq(a, b));
        }
    }
}

// exec-level: a multi-row STREAM_DOT through exec_task_cpu with the registry
// active must be bitwise equal to the packed run, including row addressing
// across a multi-row span (the registry row math under test).
static void test_exec_and_registry(Fmt f, uint32_t cols, uint32_t rows) {
    std::vector<uint8_t> big;
    const uint64_t prb = fl_row_bytes(f, cols);
    big.resize(prb * rows);
    std::mt19937 rng(42 + cols);
    for (auto & b : big) b = (uint8_t)rng();
    sanitize_scales(f, big, 42 + cols);
    auto act = random_act(cols, 4242);

    Task t{};
    t.kind = TaskKind::STREAM_DOT;
    t.fmt  = f;
    t.w    = big.data();
    t.w_bytes = prb * rows;
    t.x = act.x();
    t.cols = cols;
    t.rows = rows;

    std::vector<float> y0(rows, 0.f), y1(rows, 0.f);
    const uint32_t i8_mask  = 1u << (int)f;
    const uint32_t i8k_mask = (f == Fmt::Q4_K) ? (1u << (int)f) : 0;

    fl_shadow_clear_for_test();
    t.y = y0.data();
    exec_task_cpu(t, true, i8_mask, i8k_mask);

    std::vector<uint8_t> sh(fl_shadow_row_bytes(f, cols) * rows);
    if (f == Fmt::Q4_K) fl_shadow_build_rows_q4k(big.data(), rows, cols, sh.data());
    else                fl_shadow_build_rows_q50(big.data(), rows, cols, sh.data());
    ShadowSpan span;
    span.base = big.data();
    span.end  = big.data() + big.size();
    span.shadow = sh.data();
    span.packed_row_bytes = prb;
    span.shadow_row_bytes = fl_shadow_row_bytes(f, cols);
    span.fmt = f;
    fl_shadow_register_for_test(span);

    t.y = y1.data();
    exec_task_cpu(t, true, i8_mask, i8k_mask);
    fl_shadow_clear_for_test();

    CHECK(!memcmp(y0.data(), y1.data(), rows * 4));
    // offset tile inside the span (row addressing from a non-base pointer)
    if (rows >= 8) {
        std::vector<float> y2(4, 0.f), y3(4, 0.f);
        Task s = t;
        s.w = big.data() + prb * 3;
        s.rows = 4;
        s.w_bytes = prb * 4;
        s.y = y2.data();
        exec_task_cpu(s, true, i8_mask, i8k_mask);
        fl_shadow_register_for_test(span);
        s.y = y3.data();
        exec_task_cpu(s, true, i8_mask, i8k_mask);
        fl_shadow_clear_for_test();
        CHECK(!memcmp(y2.data(), y3.data(), 16));
    }
}

// P4.4: the SMMLA row-pair layout must produce BITWISE identical results to
// both the packed path and the plain shadow layout. It is a pure re-ordering
// of the same integers with the same per-row fold, so anything else is a bug.
static void test_q4k_smmla_layout(uint32_t cols, uint32_t rows) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    const uint64_t prb = fl_row_bytes(Fmt::Q4_K, cols);
    std::vector<uint8_t> big(prb * rows);
    std::mt19937 rng(4242 + cols + rows);
    for (auto & b : big) b = (uint8_t)rng();
    sanitize_scales(Fmt::Q4_K, big, 4242 + cols);
    auto act = random_act(cols, 77 + cols);

    Task t{};
    t.kind = TaskKind::STREAM_DOT;
    t.fmt  = Fmt::Q4_K;
    t.w    = big.data();
    t.w_bytes = prb * rows;
    t.x = act.x();
    t.cols = cols;
    t.rows = rows;
    const uint32_t i8_mask = 1u << (int)Fmt::Q4_K, i8k_mask = i8_mask;

    const uint64_t srb = fl_shadow_row_bytes(Fmt::Q4_K, cols);
    std::vector<uint8_t> plain(srb * rows), inter(srb * rows);
    fl_shadow_build_rows_q4k(big.data(), rows, cols, plain.data());
    fl_shadow_build_rows_q4k_smmla(big.data(), rows, cols, inter.data());

    ShadowSpan sp;
    sp.base = big.data();
    sp.end  = big.data() + big.size();
    sp.packed_row_bytes = prb;
    sp.shadow_row_bytes = srb;
    sp.fmt = Fmt::Q4_K;

    std::vector<float> yp(rows, 0.f), yi(rows, 0.f);
    sp.shadow = plain.data(); sp.smmla = false;
    fl_shadow_register_for_test(sp);
    t.y = yp.data();
    exec_task_cpu(t, true, i8_mask, i8k_mask);
    fl_shadow_clear_for_test();

    sp.shadow = inter.data(); sp.smmla = true;
    fl_shadow_register_for_test(sp);
    t.y = yi.data();
    exec_task_cpu(t, true, i8_mask, i8k_mask);
    fl_shadow_clear_for_test();
    CHECK(!memcmp(yp.data(), yi.data(), rows * 4));

    // direct pair kernel vs the plain single-row kernel
    const blk_q8_K * xk = fl_xqk_of(act.x(), cols);
    for (uint32_t rp = 0; rp * 2 + 1 < rows; ++rp) {
        float v0, v1;
        dot2_i8k_q4_K_shadow_smmla(inter.data() + (size_t)rp * srb * 2,
                                   xk, cols, &v0, &v1);
        CHECK(bits_eq(v0, dot_i8k_q4_K_shadow(
            plain.data() + (size_t)(rp * 2) * srb, xk, cols)));
        CHECK(bits_eq(v1, dot_i8k_q4_K_shadow(
            plain.data() + (size_t)(rp * 2 + 1) * srb, xk, cols)));
    }

    // odd tail AND an odd (misaligned) tile start must still be exact
    if (rows >= 6) {
        std::vector<float> a(3, 0.f), b(3, 0.f);
        Task s = t; s.w = big.data() + prb; s.rows = 3; s.w_bytes = prb * 3;
        sp.shadow = plain.data(); sp.smmla = false;
        fl_shadow_register_for_test(sp);
        s.y = a.data(); exec_task_cpu(s, true, i8_mask, i8k_mask);
        fl_shadow_clear_for_test();
        sp.shadow = inter.data(); sp.smmla = true;
        fl_shadow_register_for_test(sp);
        s.y = b.data(); exec_task_cpu(s, true, i8_mask, i8k_mask);
        fl_shadow_clear_for_test();
        CHECK(!memcmp(a.data(), b.data(), 12));
    }
#else
    (void)cols; (void)rows;
#endif
}

int main() {
    test_q4k_bitwise();
    test_q50_bitwise();
    test_exec_and_registry(Fmt::Q4_K, 2816, 16);
    test_exec_and_registry(Fmt::Q4_K, 1408, 8);
    test_exec_and_registry(Fmt::Q5_0, 1408, 16);
    test_q4k_smmla_layout(2816, 16);
    test_q4k_smmla_layout(1408, 8);
    test_q4k_smmla_layout(256, 6);
    printf("fastllm-shadow-tests: %d checks, %d failures\n", checks, fails);
    return fails ? 1 : 0;
}
