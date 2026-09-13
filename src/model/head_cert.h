// P3.1: certified head argmax - proof-pruned exact evaluation.
//
// For a verify position we only need argmax(logits). Instead of computing all
// V row dots, we compute a small anchor set exactly, then EXCLUDE every row
// whose Cauchy-Schwarz upper bound (segment-wise, precomputed f32-exact row
// norms x per-segment activation norms) cannot beat the anchor value T minus
// a sound margin. Remaining candidate rows are computed EXACTLY with the same
// kernel the executor's CPU tiles use (bitwise-identical arithmetic), so the
// argmax over {anchors u candidates} equals the full head's argmax whenever
// exclusion is sound. Soundness argument (documented in BENCH.md P3.1):
//   engine_logit(j) <= true_logit(j) + eps_q        (int8 quantization envelope)
//   true_logit(j)  <= sum_s ||W_js|| * ||h_s||      (Cauchy-Schwarz, per segment)
// so exclusion when bound(j) + margin < T is sound for margin >= eps_q, with
// bounds accumulated in double so f32 rounding cannot under-estimate. The
// margin default is validated by the audit gate on real weights (never-wrong
// assertion) and by tests/test_head_cert.cpp.
//
// Ties: candidates are evaluated in ascending row id with strict '>' so the
// lower id wins, matching the engine's host argmax convention. An excluded
// row cannot tie: exclusion is strict below T - margin < T.
#pragma once
#include "../runtime/kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace fastllm {

enum class HeadDotMode : uint8_t { I8K = 0, FAST_DEQ = 1, CANON = 2 };

struct HeadCertStats {
    uint64_t positions = 0;      // cert argmax calls
    uint64_t rows_eval = 0;      // exact row dots computed
    uint64_t rows_total = 0;     // V per position (denominator)
    uint64_t bytes_read = 0;     // weight bytes streamed by exact dots
    uint64_t worst_cand = 0;     // max candidate count seen in one call
};

class HeadCert {
public:
    static constexpr uint32_t SEGS = 8;

    void init(Fmt f, const uint8_t * w, uint64_t row_bytes,
              uint32_t V, uint32_t NE, float margin, uint32_t topk = 64) {
        f_ = f; w_ = w; rb_ = row_bytes; V_ = V; NE_ = NE; margin_ = margin;
        // segment width must be a multiple of the format block size
        // (dequant_chunk requires block-aligned calls)
        const uint32_t be = (f == Fmt::F32) ? 1 : fl_blk_elems(f);
        segw_ = NE / SEGS;
        if (segw_ < be) segw_ = be;
        segw_ = (segw_ / be) * be;
        while (NE % segw_) segw_ += be;           // NE must divide evenly
        nseg_ = NE / segw_;
        wn_.assign((size_t)V * nseg_, 0.f);
        std::vector<float> buf(segw_);
        std::vector<float> whole(V);
        for (uint32_t j = 0; j < V; ++j) {
            const uint8_t * row = w_ + (size_t)j * rb_;
            double tot = 0.0;
            for (uint32_t s = 0; s < nseg_; ++s) {
                seg_values(row, s * segw_, segw_, buf.data());
                double acc = 0.0;
                for (uint32_t i = 0; i < segw_; ++i)
                    acc += (double)buf[i] * (double)buf[i];
                // round the norm UP one ulp class so f32 storage cannot
                // under-estimate the true norm
                float n = std::nextafter((float)std::sqrt(acc), INFINITY);
                wn_[(size_t)j * nseg_ + s] = n;
                tot += acc;
            }
            whole[j] = (float)std::sqrt(tot);
        }
        topn_.resize(V);
        for (uint32_t j = 0; j < V; ++j) topn_[j] = j;
        if (topk < V) {
            std::partial_sort(topn_.begin(), topn_.begin() + topk, topn_.end(),
                [&](uint32_t a, uint32_t b) { return whole[a] > whole[b]; });
            topn_.resize(topk);
        }
        ready_ = true;
    }

    bool ready() const { return ready_; }

    // h: f32 activation (final-normed); its quantized appendices live at
    // fl_xq_of/fl_xqk_of(h, NE) per the activation layout contract.
    // hint: draft token id to anchor T (or -1). Returns argmax row id.
    uint32_t argmax(const float * h, int32_t hint, HeadDotMode mode) {
        // per-segment activation norms (double, exact side of the bound)
        double hn[SEGS];
        for (uint32_t s = 0; s < nseg_; ++s) {
            double acc = 0.0;
            const float * hs = h + s * segw_;
            for (uint32_t i = 0; i < segw_; ++i)
                acc += (double)hs[i] * (double)hs[i];
            hn[s] = std::sqrt(acc);
        }
        uint32_t best = UINT32_MAX;
        float bv = -INFINITY;
        const uint64_t eval0 = st_.rows_eval;
        auto eval = [&](uint32_t j) {
            float v = dot_exact(j, h, mode);
            st_.rows_eval++;
            st_.bytes_read += rb_;
            // ascending-id strict '>' preserves the lower-id tie rule as long
            // as anchors are evaluated before the ascending candidate sweep
            // only when their id ordering cannot regress; enforce explicitly:
            if (v > bv || (v == bv && j < best)) { bv = v; best = j; }
        };
        // anchors: hint first, then the fixed top-norm set
        if (hint >= 0 && (uint32_t)hint < V_) eval((uint32_t)hint);
        for (uint32_t j : topn_) if ((int32_t)j != hint) eval(j);
        const double T = (double)bv - (double)margin_;
        // bound scan + candidate evaluation, ascending id
        for (uint32_t j = 0; j < V_; ++j) {
            if ((int32_t)j == hint) continue;
            const float * wn = &wn_[(size_t)j * nseg_];
            double b = 0.0;
            for (uint32_t s = 0; s < nseg_; ++s) b += (double)wn[s] * hn[s];
            if (b < T) continue;                    // provably below the anchor
            bool is_anchor = false;
            for (uint32_t a : topn_) if (a == j) { is_anchor = true; break; }
            if (is_anchor) continue;                // already evaluated
            eval(j);
        }
        st_.positions++;
        st_.rows_total += V_;
        const uint64_t cand = st_.rows_eval - eval0;
        if (cand > st_.worst_cand) st_.worst_cand = cand;
        return best;
    }

    HeadCertStats & stats() { return st_; }

private:
    // exact row values for norm precompute (formats dequant_chunk lacks are
    // handled directly; norms need values only, order does not matter here)
    void seg_values(const uint8_t * row, uint32_t elem0, uint32_t n,
                    float * out) const {
        if (f_ == Fmt::F32) {
            memcpy(out, (const float *)row + elem0, n * 4);
        } else if (f_ == Fmt::Q8_0) {
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t e = elem0 + i;
                const uint8_t * b = row + (e / 32) * 34;
                out[i] = fl_half2float(*(const uint16_t *)b)
                       * (float)((const int8_t *)(b + 2))[e % 32];
            }
        } else {
            dequant_chunk(f_, row, elem0, n, out);
        }
    }

    float dot_exact(uint32_t j, const float * h, HeadDotMode mode) const {
        const uint8_t * row = w_ + (size_t)j * rb_;
        switch (mode) {
            case HeadDotMode::I8K:
                return dot_row_i8k(f_, row, fl_xqk_of(h, NE_), NE_);
            case HeadDotMode::FAST_DEQ:
                switch (f_) {
                    case Fmt::F32:
                        return dot_row_fast((const float *)row, h, NE_);
                    case Fmt::Q8_0:
                        return dot_row_fast_q8_0(row, h, NE_);
                    default:
                        return dot_row_fast_dequant(f_, row, h, NE_);
                }
            case HeadDotMode::CANON:
            default:
                switch (f_) {
                    case Fmt::Q6_K: return fl_dot_row_q6_K(row, h, NE_);
                    case Fmt::Q4_K: return fl_dot_row_q4_K(row, h, NE_);
                    case Fmt::Q8_0: return fl_dot_row_q8_0(row, h, NE_);
                    case Fmt::F32:
                        return dot_row_fast((const float *)row, h, NE_);
                    default:        return dot_row_fast_dequant(f_, row, h, NE_);
                }
        }
    }

    Fmt f_ = Fmt::F32;
    const uint8_t * w_ = nullptr;
    uint64_t rb_ = 0;
    uint32_t V_ = 0, NE_ = 0, segw_ = 0, nseg_ = 1;
    float margin_ = 1.0f;
    bool ready_ = false;
    std::vector<float> wn_;
    std::vector<uint32_t> topn_;
    HeadCertStats st_;
};

} // namespace fastllm
