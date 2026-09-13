// P2.0 decode-op parameter blocks + their single canonical implementations.
// Every struct is a POD living in the arena, pointed to by Task::aux. Every
// op has ONE FL_HD implementation used verbatim by both engines (host callers
// go through exec_task_scalar_host, device through exec_task_scalar): these
// ops are latency-bound and small, so there is no fast/canonical split -
// EXCEPT ATTN_HEAD since P2.3 (O(depth) inside the op): its NEON host fast
// path lives in kernels.h and matches the canonical bitwise lane-for-lane.
//
// Numerics: plain f32, sequential ascending accumulation everywhere; expf/
// logf/cosf/sinf from libm/device intrinsics (paired with -ffp-contract=off /
// --fmad=false like the rest of the canonical path).
//
// RoPE follows ggml_rope_ext NEOX + YaRN semantics exactly (ggml lineage);
// the deepseek2 kq_scale/mscale pre-scaling convention is the GRAPH BUILDER's
// job (src/model/decoder.cpp) - this file only rotates.
#pragma once
#include "fastllm/fastllm.h"
#include "quants.h"
#include <math.h>

namespace fastllm {

struct RmsNormP {
    const float * wnorm;   // n
    const float * x;       // n (Task::x may differ: sub-span support)
    float *       y;       // n
    float eps;
    uint32_t n;
};

struct Add2P {
    const float * a;
    const float * b;
    float *       y;
    uint32_t n;
};

struct RopeP {
    float * buf;           // in place; heads at head_stride, rotate span
                           // [rot_off, rot_off + n_rot) within each head
    const int32_t * pos;   // current position (host-written per token)
    uint32_t n_head, head_stride, rot_off, n_rot;
    float freq_base, freq_scale, ext_factor, attn_factor;
    float beta_fast, beta_slow;
    uint32_t n_ctx_orig;
};

struct KvAppendP {
    const float * kvb;     // n_head * (d_nope + d_v), per-head contiguous
    const float * kpe;     // d_rope, post-rope, shared across heads
    float * kcache;        // [max_kv][n_head][d_nope + d_rope]
    float * vcache;        // [max_kv][n_head][d_v]
    const int32_t * pos;
    uint32_t n_head, d_nope, d_rope, d_v, max_kv;
};

struct AttnHeadP {
    const float * q;       // n_head * dk, head-major (post-rope)
    const float * kcache;  // as KvAppendP
    const float * vcache;
    float * scratch;       // max_kv floats (this head's scores)
    float * out;           // n_head * dv, head-major
    const int32_t * pos;
    float scale;           // kq_scale (mscale^2 folded in by the builder)
    uint32_t head, n_head, dk, dv, max_kv;
};

struct RouterP {
    const float * logits;  // n_expert
    uint32_t * ids;        // top_k
    float *    gates;      // top_k
    uint32_t n_expert, top_k;
    uint32_t norm;         // renormalize top-k probs (expert_weights_norm)
    float scale;           // expert_weights_scale
};

struct SiluMulP {
    const float * gate;
    const float * up;
    float * h;
    uint32_t n;
};

struct WmergeP {
    const float * parts;   // n_parts * n, contiguous
    const float * gates;   // n_parts (nullptr -> all 1.0)
    const float * add1;    // nullptr ok
    const float * add2;    // nullptr ok
    float * y;
    float scale;
    uint32_t n_parts, n;
    // P3.5b: how many gates the router actually produced. When it exceeds
    // n_parts (k' truncation dropped the tail) the surviving gates are
    // rescaled to carry the dropped mass. 0 disables the rescale, which is
    // what every exact merge passes - the exact path stays bitwise as it was.
    uint32_t n_full;
};

// ---------------------------------------------------------------- impls ----

FL_HD void fl_op_rmsnorm(const RmsNormP & p) {
    float ss = 0.f;
    for (uint32_t i = 0; i < p.n; ++i) ss += p.x[i] * p.x[i];
    const float s = 1.0f / sqrtf(ss / (float)p.n + p.eps);
    for (uint32_t i = 0; i < p.n; ++i) p.y[i] = p.x[i] * s * p.wnorm[i];
}

FL_HD void fl_op_add2(const Add2P & p) {
    for (uint32_t i = 0; i < p.n; ++i) p.y[i] = p.a[i] + p.b[i];
}

// ggml rope_yarn machinery. Pairing is NORM style - ADJACENT pairs
// (i0, i0+1) - which is what llama.cpp uses for LLM_ARCH_DEEPSEEK2
// (llama_model_rope_type). P2.0 M1 gate history: the first build used NEOX
// half-split pairing; self-attention is basis-invariant so early tokens
// matched, but history keys misalign -> repeat-current degeneration. The
// teacher-forced argmax diff + rope-type switch in llama-model.cpp nailed it.
FL_HD float fl_rope_corr_dim(float n_dims, float n_ctx_orig, float beta,
                                    float base) {
    return n_dims * logf(n_ctx_orig / (beta * 2.0f * (float)M_PI))
         / (2.0f * logf(base));
}

FL_HD void fl_op_rope_ds2(const RopeP & p) {
    const float theta_scale = powf(p.freq_base, -2.0f / (float)p.n_rot);
    float corr_lo = fl_rope_corr_dim((float)p.n_rot, (float)p.n_ctx_orig,
                                     p.beta_fast, p.freq_base);
    float corr_hi = fl_rope_corr_dim((float)p.n_rot, (float)p.n_ctx_orig,
                                     p.beta_slow, p.freq_base);
    corr_lo = floorf(corr_lo); corr_hi = ceilf(corr_hi);
    if (corr_lo < 0.f) corr_lo = 0.f;
    if (corr_hi > (float)(p.n_rot - 1)) corr_hi = (float)(p.n_rot - 1);
    const float pos = (float)*p.pos;
    for (uint32_t h = 0; h < p.n_head; ++h) {
        float * v = p.buf + (size_t)h * p.head_stride + p.rot_off;
        float theta = pos;   // theta_base * theta_scale^(i0/2), i0 = pair*2
        for (uint32_t i0 = 0; i0 < p.n_rot; i0 += 2) {
            const float theta_extrap = theta;
            const float theta_interp = p.freq_scale * theta_extrap;
            float th = theta_interp;
            float mscale = p.attn_factor;
            if (p.ext_factor != 0.0f) {
                float ramp_y = ((float)(i0 / 2) - corr_lo)
                             / fmaxf(0.001f, corr_hi - corr_lo);
                ramp_y = 1.0f - fminf(1.0f, fmaxf(0.0f, ramp_y));
                const float ramp_mix = ramp_y * p.ext_factor;
                th = theta_interp * (1.0f - ramp_mix)
                   + theta_extrap * ramp_mix;
                mscale *= 1.0f + 0.1f * logf(1.0f / p.freq_scale);
            }
            const float c = cosf(th) * mscale;
            const float s = sinf(th) * mscale;
            const float x0 = v[i0];
            const float x1 = v[i0 + 1];
            v[i0]     = x0 * c - x1 * s;
            v[i0 + 1] = x0 * s + x1 * c;
            theta *= theta_scale;
        }
    }
}

FL_HD void fl_op_kv_append(const KvAppendP & p) {
    const uint32_t pos = (uint32_t)*p.pos;
    const uint32_t dk  = p.d_nope + p.d_rope;
    float * krow = p.kcache + (size_t)pos * p.n_head * dk;
    float * vrow = p.vcache + (size_t)pos * p.n_head * p.d_v;
    for (uint32_t h = 0; h < p.n_head; ++h) {
        const float * src = p.kvb + (size_t)h * (p.d_nope + p.d_v);
        for (uint32_t d = 0; d < p.d_nope; ++d) krow[h * dk + d] = src[d];
        for (uint32_t d = 0; d < p.d_rope; ++d)
            krow[h * dk + p.d_nope + d] = p.kpe[d];
        for (uint32_t d = 0; d < p.d_v; ++d)
            vrow[h * p.d_v + d] = src[p.d_nope + d];
    }
}

// P2.3: ATTN_HEAD is the one decode op with a fast path (NEON, kernels.h).
// The canonical order changed in lockstep to be lane-matchable: scores use a
// striped-8 fmaf accumulation folded left-associatively s0..s7, and the
// context accumulation uses fmaf per element (single rounding = NEON fmla).
// dk%8!=0 or dv%4!=0 falls back to the pre-P2.3 sequential order (never taken
// on V2-Lite: dk 192, dv 128).
FL_HD void fl_op_attn_head(const AttnHeadP & p) {
    const uint32_t n  = (uint32_t)*p.pos + 1;   // causal: keys 0..pos
    const uint32_t dk = p.dk, dv = p.dv, h = p.head;
    const float * q = p.q + (size_t)h * dk;
    const bool striped = (dk & 7u) == 0 && (dv & 3u) == 0;
    float mx = -1e30f;
    for (uint32_t j = 0; j < n; ++j) {
        const float * k = p.kcache + ((size_t)j * p.n_head + h) * dk;
        float acc;
        if (striped) {
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            float s4 = 0.f, s5 = 0.f, s6 = 0.f, s7 = 0.f;
            for (uint32_t d = 0; d < dk; d += 8) {
                s0 = fmaf(q[d + 0], k[d + 0], s0);
                s1 = fmaf(q[d + 1], k[d + 1], s1);
                s2 = fmaf(q[d + 2], k[d + 2], s2);
                s3 = fmaf(q[d + 3], k[d + 3], s3);
                s4 = fmaf(q[d + 4], k[d + 4], s4);
                s5 = fmaf(q[d + 5], k[d + 5], s5);
                s6 = fmaf(q[d + 6], k[d + 6], s6);
                s7 = fmaf(q[d + 7], k[d + 7], s7);
            }
            acc = ((((((s0 + s1) + s2) + s3) + s4) + s5) + s6) + s7;
        } else {
            acc = 0.f;
            for (uint32_t d = 0; d < dk; ++d) acc += q[d] * k[d];
        }
        acc *= p.scale;
        p.scratch[j] = acc;
        if (acc > mx) mx = acc;
    }
    float sum = 0.f;
    for (uint32_t j = 0; j < n; ++j) {
        const float e = expf(p.scratch[j] - mx);
        p.scratch[j] = e;
        sum += e;
    }
    const float inv = 1.0f / sum;
    float * o = p.out + (size_t)h * dv;
    for (uint32_t d = 0; d < dv; ++d) o[d] = 0.f;
    for (uint32_t j = 0; j < n; ++j) {
        const float w = p.scratch[j] * inv;
        const float * v = p.vcache + ((size_t)j * p.n_head + h) * dv;
        if (striped) {
            for (uint32_t d = 0; d < dv; ++d) o[d] = fmaf(w, v[d], o[d]);
        } else {
            for (uint32_t d = 0; d < dv; ++d) o[d] += w * v[d];
        }
    }
}

FL_HD void fl_op_router_sel(const RouterP & p) {
    // softmax over all logits (deepseek2 SOFTMAX gating), then top-k by
    // weight, ties -> lower index (matches ggml stable argsort semantics)
    float mx = -1e30f;
    for (uint32_t i = 0; i < p.n_expert; ++i)
        if (p.logits[i] > mx) mx = p.logits[i];
    float sum = 0.f;
    float w[256];   // n_expert cap; validated at graph build (device stack)
    for (uint32_t i = 0; i < p.n_expert; ++i) {
        w[i] = expf(p.logits[i] - mx);
        sum += w[i];
    }
    for (uint32_t i = 0; i < p.n_expert; ++i) w[i] /= sum;
    float tsum = 0.f;
    for (uint32_t k = 0; k < p.top_k; ++k) {
        uint32_t best = 0;
        float bw = -1.f;
        for (uint32_t i = 0; i < p.n_expert; ++i) {
            bool taken = false;
            for (uint32_t k2 = 0; k2 < k; ++k2)
                if (p.ids[k2] == i) { taken = true; break; }
            if (!taken && w[i] > bw) { bw = w[i]; best = i; }
        }
        p.ids[k]   = best;
        p.gates[k] = bw;
        tsum += bw;
    }
    for (uint32_t k = 0; k < p.top_k; ++k) {
        if (p.norm) p.gates[k] /= tsum;
        p.gates[k] *= p.scale;
    }
}

FL_HD void fl_op_silu_mul(const SiluMulP & p) {
    for (uint32_t i = 0; i < p.n; ++i) {
        const float g = p.gate[i];
        p.h[i] = (g / (1.0f + expf(-g))) * p.up[i];
    }
}

FL_HD void fl_op_wmerge(const WmergeP & p) {
    // P3.5b: rescale the surviving gates to carry the truncated tail's mass.
    // gsc stays exactly 1.0f whenever n_full is 0 or equal to n_parts, and
    // 1.0f * x is exact, so the exact merge is bitwise unchanged.
    float gsc = 1.0f;
    if (p.gates && p.n_full > p.n_parts) {
        float s = 0.f, sf = 0.f;
        for (uint32_t q = 0; q < p.n_parts; ++q) s  += p.gates[q];
        for (uint32_t q = 0; q < p.n_full;  ++q) sf += p.gates[q];
        if (s > 0.f) gsc = sf / s;
    }
    for (uint32_t i = 0; i < p.n; ++i) {
        float acc = 0.f;
        for (uint32_t q = 0; q < p.n_parts; ++q)
            acc += (p.gates ? gsc * p.gates[q] : 1.0f)
                   * p.parts[(size_t)q * p.n + i];
        acc *= p.scale;
        if (p.add1) acc += p.add1[i];
        if (p.add2) acc += p.add2[i];
        p.y[i] = acc;
    }
}

} // namespace fastllm
