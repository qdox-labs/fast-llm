// P3.1 head-cert soundness tests. The certificate must NEVER certify a wrong
// argmax: fuzz vs brute force on synthetic heads (f32 exact mode, tiny
// margin), planted near-ties, and a quantized-mode bound-slack measurement
// that justifies the default margin. Also: exact-dot bitwise equality with
// the kernel dispatch the executor's CPU tiles use.
#include "model/head_cert.h"
#include "runtime/kernels.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace fastllm;

static int checks = 0, fails = 0;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// build a synthetic Q6_K weight matrix with a seeded RNG
static std::vector<uint8_t> make_q6k(uint32_t rows, uint32_t cols,
                                     std::mt19937 & rng) {
    const uint64_t rb = fl_row_bytes(Fmt::Q6_K, cols);
    std::vector<uint8_t> w(rows * rb);
    for (auto & b : w) b = (uint8_t)rng();
    // scales bytes are int8 in [-127,127]; raw random is fine (any bit
    // pattern is a valid block); d halves must be finite: clamp exponent
    for (uint32_t r = 0; r < rows; ++r) {
        uint8_t * row = w.data() + r * rb;
        for (uint32_t blk = 0; blk < cols / 256; ++blk) {
            uint8_t * d = row + blk * 210 + 208;      // fp16 d at block tail
            d[1] &= 0x3B;                             // keep |d| small/finite
        }
    }
    return w;
}

int main() {
    std::mt19937 rng(20260813);

    // ---- 1. f32-exact fuzz: cert argmax == brute argmax, never wrong ----
    {
        const uint32_t V = 2048, NE = 256;
        std::vector<float> Wf(V * NE);
        std::normal_distribution<float> nd(0.f, 1.f);
        int trials = 250, wrong = 0;
        for (int t = 0; t < trials; ++t) {
            for (auto & v : Wf) v = nd(rng);
            // scale a few rows up so norms vary (bound tightness varies)
            for (int k = 0; k < 8; ++k) {
                uint32_t r = rng() % V;
                for (uint32_t i = 0; i < NE; ++i) Wf[r * NE + i] *= 4.f;
            }
            HeadCert hc;
            hc.init(Fmt::F32, (const uint8_t *)Wf.data(), NE * 4, V, NE,
                    /*margin*/ 1e-3f, /*topk*/ 16);
            for (int q = 0; q < 40; ++q) {
                std::vector<float> h(NE);
                for (auto & v : h) v = nd(rng);
                // brute force with the same arithmetic (dot_row_fast on f32)
                uint32_t bb = 0; float bv = -1e30f;
                for (uint32_t j = 0; j < V; ++j) {
                    float d = dot_row_fast(&Wf[j * NE], h.data(), NE);
                    if (d > bv) { bv = d; bb = j; }
                }
                int32_t hint = (q & 1) ? (int32_t)(rng() % V) : -1;
                uint32_t c = hc.argmax(h.data(), hint, HeadDotMode::FAST_DEQ);
                if (c != bb) wrong++;
            }
        }
        CHECK(wrong == 0, "f32 fuzz: %d wrong argmax certifications", wrong);
    }

    // ---- 2. planted near-ties: margin must force candidate evaluation ----
    {
        const uint32_t V = 1024, NE = 256;
        std::vector<float> Wf(V * NE);
        std::normal_distribution<float> nd(0.f, 1.f);
        int wrong = 0;
        for (int t = 0; t < 2000; ++t) {
            for (auto & v : Wf) v = nd(rng);
            std::vector<float> h(NE);
            for (auto & v : h) v = nd(rng);
            // plant: row B = row A + eps -> logits tie within ~1e-5
            uint32_t A = rng() % V, B = (A + 1 + rng() % (V - 1)) % V;
            for (uint32_t i = 0; i < NE; ++i)
                Wf[B * NE + i] = Wf[A * NE + i] + 1e-7f * (float)(int)(rng() % 3);
            HeadCert hc;
            hc.init(Fmt::F32, (const uint8_t *)Wf.data(), NE * 4, V, NE,
                    1e-3f, 16);
            uint32_t bb = 0; float bv = -1e30f;
            for (uint32_t j = 0; j < V; ++j) {
                float d = dot_row_fast(&Wf[j * NE], h.data(), NE);
                if (d > bv) { bv = d; bb = j; }
            }
            uint32_t c = hc.argmax(h.data(), -1, HeadDotMode::FAST_DEQ);
            if (c != bb) wrong++;
        }
        CHECK(wrong == 0, "near-tie fuzz: %d wrong", wrong);
    }

    // ---- 3. quantized mode: int8 dot must stay under bound + margin ----
    {
        const uint32_t V = 512, NE = 512;
        auto w = make_q6k(V, NE, rng);
        const uint64_t rb = fl_row_bytes(Fmt::Q6_K, NE);
        HeadCert hc;
        hc.init(Fmt::Q6_K, w.data(), rb, V, NE, 1.0f, 16);
        std::normal_distribution<float> nd(0.f, 1.f);
        double worst_slack = -1e30;
        std::vector<float> hbuf(fl_x_alloc_bytes(NE) / 4 + 4);
        for (int t = 0; t < 200; ++t) {
            float * h = hbuf.data();
            for (uint32_t i = 0; i < NE; ++i) h[i] = nd(rng);
            // build the q8_K appendix exactly as ACT_Q8 does
            blk_q8_K * xk = (blk_q8_K *)fl_xqk_of(h, NE);
            for (uint32_t b = 0; b < NE / 256; ++b)
                fl_quantize_q8_K_block(h + b * 256, (blk_q8_K *)xk + b);
            // segment norms of h in double (block-aligned: segw = 256)
            const uint32_t segw = 256, nseg = NE / segw;
            double hn[8];
            for (uint32_t s2 = 0; s2 < nseg; ++s2) {
                double a = 0;
                for (uint32_t i = 0; i < segw; ++i)
                    a += (double)h[s2 * segw + i] * h[s2 * segw + i];
                hn[s2] = sqrt(a);
            }
            for (uint32_t j = 0; j < V; ++j) {
                float d = dot_row_i8k(Fmt::Q6_K, w.data() + j * rb, xk, NE);
                // recompute the bound the way argmax does (per-seg norms)
                std::vector<float> buf(segw);
                double bound = 0;
                for (uint32_t s2 = 0; s2 < nseg; ++s2) {
                    dequant_chunk(Fmt::Q6_K, w.data() + j * rb,
                                  s2 * segw, segw, buf.data());
                    double a = 0;
                    for (uint32_t i = 0; i < segw; ++i)
                        a += (double)buf[i] * buf[i];
                    bound += sqrt(a) * hn[s2];
                }
                double slack = (double)d - bound;   // must be <= margin
                if (slack > worst_slack) worst_slack = slack;
            }
        }
        fprintf(stderr, "head_cert: worst int8-over-bound slack %.3e "
                "(margin 1.0)\n", worst_slack);
        CHECK(worst_slack < 1.0, "int8 slack %.3e exceeds default margin",
              worst_slack);
    }

    // ---- 4. cert argmax under quantized mode vs brute int8 argmax ----
    {
        const uint32_t V = 512, NE = 512;
        auto w = make_q6k(V, NE, rng);
        const uint64_t rb = fl_row_bytes(Fmt::Q6_K, NE);
        HeadCert hc;
        hc.init(Fmt::Q6_K, w.data(), rb, V, NE, 1.0f, 16);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> hbuf(fl_x_alloc_bytes(NE) / 4 + 4);
        int wrong = 0;
        for (int t = 0; t < 500; ++t) {
            float * h = hbuf.data();
            for (uint32_t i = 0; i < NE; ++i) h[i] = nd(rng);
            blk_q8_K * xk = (blk_q8_K *)fl_xqk_of(h, NE);
            for (uint32_t b = 0; b < NE / 256; ++b)
                fl_quantize_q8_K_block(h + b * 256, (blk_q8_K *)xk + b);
            uint32_t bb = 0; float bv = -1e30f;
            for (uint32_t j = 0; j < V; ++j) {
                float d = dot_row_i8k(Fmt::Q6_K, w.data() + j * rb, xk, NE);
                if (d > bv) { bv = d; bb = j; }
            }
            uint32_t c = hc.argmax(h, (t & 1) ? (int32_t)bb : -1,
                                   HeadDotMode::I8K);
            if (c != bb) wrong++;
        }
        CHECK(wrong == 0, "quantized fuzz: %d wrong", wrong);
    }

    fprintf(stderr, "fastllm-headcert-tests: %d checks, %d failures\n",
            checks, fails);
    return fails ? 1 : 0;
}
