// CPU STREAM_DOT kernels.
//
// Two modes:
//  - canonical: exec_task_scalar (runtime_internal.h) — plain sequential
//    accumulation, engine-independent, the bitwise reference for --check.
//  - fast: vectorized, one fixed "16-lane" accumulation order shared by the
//    NEON / AVX2 / scalar-fast variants below. Deterministic per build, but
//    NOT bitwise-equal to canonical (different summation order) — validated
//    against canonical by relative tolerance instead (docs/BENCH.md).
//
// 16-lane order spec, per row (cols floats):
//   n16 = cols - cols % 16
//   lane j in [0,16): L[j] = sum over k < n16, k % 16 == j of w[k]*x[k]
//                     (ascending k; FMA contraction permitted)
//   tree: t0[j] = L[j] + L[4+j]; t1[j] = L[8+j] + L[12+j]   (j in [0,4))
//         s[j]  = t0[j] + t1[j]
//         head  = (s[0] + s[1]) + (s[2] + s[3])
//   tail: for k in [n16, cols) ascending: tail += w[k]*x[k]
//   y[r] += head + tail
#pragma once
#include "fastllm/fastllm.h"
#include "quants.h"
#include "runtime_internal.h"   // exec_task_scalar (canonical fallback)
#include "shadow.h"             // P3.2 full-shadow experts (CPU fast paths)
#include <cstdlib>              // getenv (ATTN_HEAD A/B switch)

#if defined(__ARM_NEON)
  #include <arm_neon.h>
#elif defined(__AVX2__)
  #include <immintrin.h>
#endif

namespace fastllm {

// P2.5: FASTLLM_MC_LEGACY=1 keeps the P2.4 per-column loops on multi-column
// tiles (weight LOAD shared, unpack repeated) for same-binary attribution
// against the unpack-sharing kernels below.
inline bool fl_mc_legacy() {
    static const bool v = [] {
        const char * e = getenv("FASTLLM_MC_LEGACY");
        return e && e[0] == '1';
    }();
    return v;
}

// P4.2: FASTLLM_SMMLA=1 routes multi-column q4_K i8k tiles through the i8mm
// SMMLA kernel (2 weight rows x 2 activation columns per instruction, 32 MACs
// against SDOT's 16). Default OFF until cells arbitrate. Probe (P4.2,
// tools/smmla_probe.c) on the isolated primitive at nx=8: instructions/MAC
// 1.33 -> 0.72 (-46%), 15.1 -> 21.7 Gmac/s (+43.6%), bitwise identical.
inline bool fl_smmla_on() {
    static const bool v = [] {
        const char * e = getenv("FASTLLM_SMMLA");
        return e && e[0] == '1';
    }();
    return v;
}

inline float dot_row_fast(const float * row, const float * x, uint32_t cols) {
    const uint32_t n16 = cols & ~15u;
    float head;
#if defined(__ARM_NEON)
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = a0, a2 = a0, a3 = a0;
    for (uint32_t k = 0; k < n16; k += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(row + k),      vld1q_f32(x + k));
        a1 = vfmaq_f32(a1, vld1q_f32(row + k + 4),  vld1q_f32(x + k + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(row + k + 8),  vld1q_f32(x + k + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(row + k + 12), vld1q_f32(x + k + 12));
    }
    float32x4_t t0 = vaddq_f32(a0, a1), t1 = vaddq_f32(a2, a3);
    float32x4_t s  = vaddq_f32(t0, t1);
    head = (vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1))
         + (vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3));
#elif defined(__AVX2__)
    __m256 a01 = _mm256_setzero_ps(), a23 = a01;   // lanes 0-7 / 8-15
    for (uint32_t k = 0; k < n16; k += 16) {
        a01 = _mm256_fmadd_ps(_mm256_loadu_ps(row + k),
                              _mm256_loadu_ps(x + k), a01);
        a23 = _mm256_fmadd_ps(_mm256_loadu_ps(row + k + 8),
                              _mm256_loadu_ps(x + k + 8), a23);
    }
    __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(a01),
                           _mm256_extractf128_ps(a01, 1));
    __m128 t1 = _mm_add_ps(_mm256_castps256_ps128(a23),
                           _mm256_extractf128_ps(a23, 1));
    __m128 s  = _mm_add_ps(t0, t1);
    alignas(16) float sv[4];
    _mm_store_ps(sv, s);
    head = (sv[0] + sv[1]) + (sv[2] + sv[3]);
#else
    float L[16] = {0};
    for (uint32_t k = 0; k < n16; k += 16)
        for (int j = 0; j < 16; ++j) L[j] += row[k + j] * x[k + j];
    float t0[4], t1[4], s[4];
    for (int j = 0; j < 4; ++j) {
        t0[j] = L[j] + L[4 + j];
        t1[j] = L[8 + j] + L[12 + j];
        s[j]  = t0[j] + t1[j];
    }
    head = (s[0] + s[1]) + (s[2] + s[3]);
#endif
    float tail = 0.f;
    for (uint32_t k = n16; k < cols; ++k) tail += row[k] * x[k];
    return head + tail;
}

// ---- quant fast paths (P1.0) ----------------------------------------------
// Q8_0: fused dequant-dot. Deterministic order: per block (ascending), four
// f32x4 accumulators cycled over the 8 vector groups of 32 elems, same final
// tree as dot_row_fast. NEON on ARM; scalar-fused elsewhere (same order).
inline float dot_row_fast_q8_0(const uint8_t * row, const float * x, uint32_t cols) {
    const uint32_t nb = cols / 32;
#if defined(__ARM_NEON)
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = a0, a2 = a0, a3 = a0;
    for (uint32_t b = 0; b < nb; ++b) {
        const blk_q8_0 * B = (const blk_q8_0 *)(row + (size_t)b * 34);
        const float32x4_t dv = vdupq_n_f32(fl_half2float(B->d));
        const int8x16_t lo = vld1q_s8(B->qs);
        const int8x16_t hi = vld1q_s8(B->qs + 16);
        const int16x8_t s0 = vmovl_s8(vget_low_s8(lo));
        const int16x8_t s1 = vmovl_s8(vget_high_s8(lo));
        const int16x8_t s2 = vmovl_s8(vget_low_s8(hi));
        const int16x8_t s3 = vmovl_s8(vget_high_s8(hi));
        const float * xb = x + b * 32;
        a0 = vfmaq_f32(a0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16 (s0))), dv), vld1q_f32(xb +  0));
        a1 = vfmaq_f32(a1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0))), dv), vld1q_f32(xb +  4));
        a2 = vfmaq_f32(a2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16 (s1))), dv), vld1q_f32(xb +  8));
        a3 = vfmaq_f32(a3, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1))), dv), vld1q_f32(xb + 12));
        a0 = vfmaq_f32(a0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16 (s2))), dv), vld1q_f32(xb + 16));
        a1 = vfmaq_f32(a1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s2))), dv), vld1q_f32(xb + 20));
        a2 = vfmaq_f32(a2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16 (s3))), dv), vld1q_f32(xb + 24));
        a3 = vfmaq_f32(a3, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s3))), dv), vld1q_f32(xb + 28));
    }
    const float32x4_t t0 = vaddq_f32(a0, a1), t1 = vaddq_f32(a2, a3);
    const float32x4_t s  = vaddq_f32(t0, t1);
    return (vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1))
         + (vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3));
#else
    float a[16] = {0};
    for (uint32_t b = 0; b < nb; ++b) {
        const blk_q8_0 * B = (const blk_q8_0 *)(row + (size_t)b * 34);
        const float d = fl_half2float(B->d);
        const float * xb = x + b * 32;
        for (int g = 0; g < 8; ++g)
            for (int j = 0; j < 4; ++j)
                a[(g & 3) * 4 + j] += d * (float)B->qs[g * 4 + j] * xb[g * 4 + j];
    }
    float t0[4], t1[4], s[4];
    for (int j = 0; j < 4; ++j) {
        t0[j] = a[j] + a[4 + j];
        t1[j] = a[8 + j] + a[12 + j];
        s[j]  = t0[j] + t1[j];
    }
    return (s[0] + s[1]) + (s[2] + s[3]);
#endif
}

// Q5_0 / Q4_K / Q6_K: chunked dequant-to-buffer + vector dot. Deterministic
// order: per 256-elem chunk, scalar-dequant into a stack buffer then
// dot_row_fast over the chunk, chunks ascending. Correct on every arch; the
// dequant is scalar (a NEON nibble path is a known lever, not P1.0 scope).
inline void dequant_chunk(Fmt f, const uint8_t * row, uint32_t elem0,
                          uint32_t n, float * out) {
    // dequant elems [elem0, elem0+n) of the row into out; block-aligned calls
    const uint32_t be = fl_blk_elems(f);
    const uint8_t * bp = row + (size_t)(elem0 / be) * fl_blk_bytes(f);
    // reuse the canonical dot with basis vectors would be O(n^2); instead do a
    // direct scalar dequant per format, same value formulas as quants.h.
    if (f == Fmt::Q5_0) {
        const uint32_t nb = n / 32;
        for (uint32_t b = 0; b < nb; ++b) {
            const blk_q5_0 * B = (const blk_q5_0 *)(bp + (size_t)b * 22);
            const float d = fl_half2float(B->d);
            uint32_t qh; memcpy(&qh, B->qh, 4);
            for (int j = 0; j < 16; ++j) {
                const uint8_t xh0 = (uint8_t)(((qh >> j) << 4) & 0x10);
                const uint8_t xh1 = (uint8_t)((qh >> (j + 12)) & 0x10);
                out[b * 32 + j]      = d * (float)((int)((B->qs[j] & 0x0F) | xh0) - 16);
                out[b * 32 + 16 + j] = d * (float)((int)((B->qs[j] >> 4)   | xh1) - 16);
            }
        }
    } else if (f == Fmt::Q4_K) {
        const uint32_t nb = n / 256;
        for (uint32_t sb = 0; sb < nb; ++sb) {
            const blk_q4_K * B = (const blk_q4_K *)(bp + (size_t)sb * 144);
            const float d = fl_half2float(B->d), dmin = fl_half2float(B->dmin);
            const uint8_t * q = B->qs;
            float * o = out + sb * 256;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc, mn;
                fl_q4k_scale_min(is + 0, B->scales, &sc, &mn);
                const float d1 = d * sc, m1 = dmin * mn;
                fl_q4k_scale_min(is + 1, B->scales, &sc, &mn);
                const float d2 = d * sc, m2 = dmin * mn;
                for (int l = 0; l < 32; ++l) o[j + l]      = d1 * (float)(q[l] & 0xF) - m1;
                for (int l = 0; l < 32; ++l) o[j + 32 + l] = d2 * (float)(q[l] >> 4)  - m2;
                q += 32; is += 2;
            }
        }
    } else if (f == Fmt::F32) {
        memcpy(out, (const float *)row + elem0, (size_t)n * sizeof(float));
    } else if (f == Fmt::Q8_0) {
        // P5.2: was missing - the trailing else assumed Q6_K, so a Q8_0 row
        // (e.g. token_embd in an IQ2_XXS-recipe file) was reinterpreted as
        // Q6_K blocks: values came out as small ints times a bogus scale and
        // the residual stream reached 1e8 before the first layer. Silent
        // because every model tried before had a K-quant embedding.
        const uint32_t nb = n / 32;
        for (uint32_t b = 0; b < nb; ++b) {
            const blk_q8_0 * B = (const blk_q8_0 *)(bp + (size_t)b * 34);
            const float d = fl_half2float(B->d);
            for (int j = 0; j < 32; ++j) out[b * 32 + j] = d * (float)B->qs[j];
        }
    } else if (f == Fmt::IQ2_XXS || f == Fmt::IQ3_XXS) {
        // Codebook rows are never dequantized in bulk today (experts and the
        // head go through the dot kernels). Zero-fill would be silent, so make
        // the gap loud if a future graph routes one here.
        fprintf(stderr, "dequant_chunk: fmt %d unimplemented\n", (int)f);
        memset(out, 0, (size_t)n * sizeof(float));
    } else { // Q6_K
        const uint32_t nb = n / 256;
        for (uint32_t sb = 0; sb < nb; ++sb) {
            const blk_q6_K * B = (const blk_q6_K *)(bp + (size_t)sb * 210);
            const float d = fl_half2float(B->d);
            const uint8_t * ql = B->ql;
            const uint8_t * qh = B->qh;
            const int8_t  * sc = B->scales;
            float * o = out + sb * 256;
            for (int nn = 0; nn < 256; nn += 128) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    o[nn + l]      = d * (float)sc[is + 0] * (float)((int)((ql[l] & 0xF)      | (((qh[l] >> 0) & 3) << 4)) - 32);
                    o[nn + l + 32] = d * (float)sc[is + 2] * (float)((int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                    o[nn + l + 64] = d * (float)sc[is + 4] * (float)((int)((ql[l] >> 4)       | (((qh[l] >> 4) & 3) << 4)) - 32);
                    o[nn + l + 96] = d * (float)sc[is + 6] * (float)((int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32);
                }
                ql += 64; qh += 32; sc += 8;
            }
        }
    }
}

inline float dot_row_fast_dequant(Fmt f, const uint8_t * row, const float * x,
                                  uint32_t cols) {
    float buf[256];
    float acc = 0.f;
    for (uint32_t e = 0; e < cols; e += 256) {
        const uint32_t n = (cols - e < 256) ? cols - e : 256;
        dequant_chunk(f, row, e, n, buf);
        acc += dot_row_fast(buf, x + e, n);
    }
    return acc;
}

// ---- int8 (q8_1 activation) fast paths (P1.1) -----------------------------
// SDOT kernels reproduce the fl_dot_i8_* scalar order BITWISE: integer block
// sums are exact, and the per-block float folds use the same explicit fmaf
// sequence. q5_0 keeps the scalar path (qh bit-scatter compose; a known
// lever, and format-aware affinity routes q5_0 to the GPU anyway).
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
// P1.3: all int8 NEON kernels reproduce the 4-accumulator canonical order
// (quants.h): term t -> a4[t & 3], final fold (a0+a1)+(a2+a3). Bitwise vs
// the fl_dot_i8_* references, as before — only the order contract moved.
inline float dot_i8_q8_0_neon(const uint8_t * row, const blk_q8_1 * xq, uint32_t cols) {
    const uint32_t nb = cols / 32;
    float a4[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t b = 0; b < nb; ++b) {
        const blk_q8_0 * W = (const blk_q8_0 *)(row + (size_t)b * 34);
        const int8x16_t w0 = vld1q_s8(W->qs), w1 = vld1q_s8(W->qs + 16);
        const int8x16_t x0 = vld1q_s8(xq[b].qs), x1 = vld1q_s8(xq[b].qs + 16);
        const int32x4_t s  = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0), w1, x1);
        a4[b & 3] = fmaf(fl_half2float(W->d) * fl_half2float(xq[b].d),
                         (float)vaddvq_s32(s), a4[b & 3]);
    }
    return fl_fold4(a4);
}

// v1 (P1.1): per-group branchy scale decode; recomputes the activation block
// sum with two ones-vdots per group per row. Kept for same-session A/B in
// membench (--q4k v1); the executor uses v2.
inline float dot_i8_q4_K_neon_v1(const uint8_t * row, const blk_q8_1 * xq, uint32_t cols) {
    const uint32_t nsb = cols / 256;
    const int8x16_t ones = vdupq_n_s8(1);
    const uint8x16_t m4  = vdupq_n_u8(0x0F);
    float a4[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        const blk_q4_K * W = (const blk_q4_K *)(row + (size_t)sb * 144);
        const float d = fl_half2float(W->d), dmin = fl_half2float(W->dmin);
        for (int g = 0; g < 8; ++g) {
            uint8_t sc, mn;
            fl_q4k_scale_min(g, W->scales, &sc, &mn);
            const uint8_t * q = W->qs + (g / 2) * 32;
            const uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
            const int8x16_t w0 = vreinterpretq_s8_u8(
                (g & 1) ? vshrq_n_u8(q0, 4) : vandq_u8(q0, m4));
            const int8x16_t w1 = vreinterpretq_s8_u8(
                (g & 1) ? vshrq_n_u8(q1, 4) : vandq_u8(q1, m4));
            const int8x16_t x0 = vld1q_s8(xq[sb * 8 + g].qs);
            const int8x16_t x1 = vld1q_s8(xq[sb * 8 + g].qs + 16);
            const int idot = vaddvq_s32(
                vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0), w1, x1));
            const int isum = vaddvq_s32(
                vdotq_s32(vdotq_s32(vdupq_n_s32(0), ones, x0), ones, x1));
            const float dx = fl_half2float(xq[sb * 8 + g].d);
            a4[g & 3] = fmaf(d * sc * dx,       (float)idot, a4[g & 3]);
            a4[g & 3] = fmaf(-(dmin * mn) * dx, (float)isum, a4[g & 3]);
        }
    }
    return fl_fold4(a4);
}

// v2 (P1.2): the activation side of the fold depends only on the q8_1 blocks,
// never on the weight row, so the caller precomputes per-block integer sums
// (xsum) and float scales (xd) ONCE per tile. Integer sums are exact and the
// per-row fmaf fold order is unchanged, so v2 is bitwise-identical to the
// fl_dot_i8_q4_K reference (and to v1). Scale pairs decode branchlessly per
// superblock instead of per group. Removes the two ones-vdots per group per
// row: half the SDOT work of v1.
// NOTE i8mm SMMLA was evaluated and deliberately NOT used: for batch-1 GEMV
// the 2x2 MMLA tile computes 50% garbage products, landing at exact SDOT
// useful-MAC parity on the V3AE pipes. llama.cpp's i8mm win is an nrc==2
// (two-output-column) effect that does not exist for single-vector dots.
inline float dot_i8_q4_K_neon_v2(const uint8_t * row, const blk_q8_1 * xq,
                                 const int32_t * xsum, const float * xd,
                                 uint32_t cols) {
    const uint32_t nsb = cols / 256;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    float a4[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        const blk_q4_K * W = (const blk_q4_K *)(row + (size_t)sb * 144);
        const float d = fl_half2float(W->d), dmin = fl_half2float(W->dmin);
        uint8_t sc8[8], mn8[8];
        fl_q4k_scales_all(W->scales, sc8, mn8);
        const uint8_t * q = W->qs;
        for (int c = 0; c < 4; ++c) {                 // 64-elem chunks
            const uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
            const uint32_t b0 = sb * 8 + 2 * c, b1 = b0 + 1;
            const int8x16_t xa0 = vld1q_s8(xq[b0].qs);
            const int8x16_t xa1 = vld1q_s8(xq[b0].qs + 16);
            const int8x16_t xb0 = vld1q_s8(xq[b1].qs);
            const int8x16_t xb1 = vld1q_s8(xq[b1].qs + 16);
            const int8x16_t wl0 = vreinterpretq_s8_u8(vandq_u8(q0, m4));
            const int8x16_t wl1 = vreinterpretq_s8_u8(vandq_u8(q1, m4));
            const int8x16_t wh0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
            const int8x16_t wh1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
            const int ilo = vaddvq_s32(
                vdotq_s32(vdotq_s32(vdupq_n_s32(0), wl0, xa0), wl1, xa1));
            const int ihi = vaddvq_s32(
                vdotq_s32(vdotq_s32(vdupq_n_s32(0), wh0, xb0), wh1, xb1));
            // groups 2c (lo terms) and 2c+1 (hi terms), canonical rotation
            const int glo = (2 * c) & 3, ghi = (2 * c + 1) & 3;
            a4[glo] = fmaf(d * sc8[2 * c] * xd[b0],           (float)ilo,      a4[glo]);
            a4[glo] = fmaf(-(dmin * mn8[2 * c]) * xd[b0],     (float)xsum[b0], a4[glo]);
            a4[ghi] = fmaf(d * sc8[2 * c + 1] * xd[b1],       (float)ihi,      a4[ghi]);
            a4[ghi] = fmaf(-(dmin * mn8[2 * c + 1]) * xd[b1], (float)xsum[b1], a4[ghi]);
            q += 32;
        }
    }
    return fl_fold4(a4);
}

inline float dot_i8_q6_K_neon(const uint8_t * row, const blk_q8_1 * xq, uint32_t cols) {
    const uint32_t nsb = cols / 256;
    const int8x16_t ones = vdupq_n_s8(1);
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m3 = vdupq_n_u8(0x03);
    float a4[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        const blk_q6_K * W = (const blk_q6_K *)(row + (size_t)sb * 210);
        const float d = fl_half2float(W->d);
        for (int h = 0; h < 2; ++h) {
            const uint8_t * ql = W->ql + h * 64;
            const uint8_t * qh = W->qh + h * 32;
            const int8_t  * sc = W->scales + h * 8;
            for (int s = 0; s < 4; ++s) {
                const int8_t * xg = xq[sb * 8 + h * 4 + s].qs;
                const float dx = fl_half2float(xq[sb * 8 + h * 4 + s].d);
                for (int k16 = 0; k16 < 2; ++k16) {
                    const uint8x16_t lb = vld1q_u8(ql + (s & 1) * 32 + k16 * 16);
                    const uint8x16_t hv = vld1q_u8(qh + k16 * 16);
                    const uint8x16_t nib = (s < 2) ? vandq_u8(lb, m4)
                                                   : vshrq_n_u8(lb, 4);
                    uint8x16_t hb;
                    switch (s) {
                        case 0: hb = vandq_u8(hv, m3); break;
                        case 1: hb = vandq_u8(vshrq_n_u8(hv, 2), m3); break;
                        case 2: hb = vandq_u8(vshrq_n_u8(hv, 4), m3); break;
                        default: hb = vshrq_n_u8(hv, 6); break;
                    }
                    const int8x16_t wq = vreinterpretq_s8_u8(
                        vorrq_u8(nib, vshlq_n_u8(hb, 4)));
                    const int8x16_t xv = vld1q_s8(xg + k16 * 16);
                    const int idot = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), wq, xv));
                    const int isum = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), ones, xv));
                    const int t = (int)sc[2 * s + k16] * (idot - 32 * isum);
                    const int ti = h * 8 + s * 2 + k16;   // canonical term idx
                    a4[ti & 3] = fmaf(d * dx, (float)t, a4[ti & 3]);
                }
            }
        }
    }
    return fl_fold4(a4);
}
// ---- q8_K (P1.5) NEON paths: integer-domain scale folding -----------------
// Reproduce the fl_dot_i8k_* order bitwise: all integer work exact, the two
// (q4_K) / one (q6_K) per-superblock fmaf folds identical to the reference.
// This is the P1.4-named CPU fix: the q8_1 appendix forced 16 fmaf+cvt per
// superblock (instruction-count-bound at IPC 3.2); here scales fold as ints.
inline float dot_i8k_q4_K_neon(const uint8_t * row, const blk_q8_K * xk, uint32_t cols) {
    const uint32_t nsb = cols / 256;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    float a4[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        const blk_q4_K * W = (const blk_q4_K *)(row + (size_t)sb * 144);
        const float d = fl_half2float(W->d), dmin = fl_half2float(W->dmin);
        const float dk = xk[sb].d;
        uint8_t sc8[8], mn8[8];
        fl_q4k_scales_all(W->scales, sc8, mn8);
        const uint8_t * q = W->qs;
        const int8_t  * xs = xk[sb].qs;
        int32_t iacc = 0, imins = 0;
        for (int c = 0; c < 4; ++c) {                 // 64-elem chunks
            const uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
            const int8x16_t wl0 = vreinterpretq_s8_u8(vandq_u8(q0, m4));
            const int8x16_t wl1 = vreinterpretq_s8_u8(vandq_u8(q1, m4));
            const int8x16_t wh0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
            const int8x16_t wh1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
            const int8x16_t xa0 = vld1q_s8(xs + c * 64);
            const int8x16_t xa1 = vld1q_s8(xs + c * 64 + 16);
            const int8x16_t xb0 = vld1q_s8(xs + c * 64 + 32);
            const int8x16_t xb1 = vld1q_s8(xs + c * 64 + 48);
            const int ilo = vaddvq_s32(
                vdotq_s32(vdotq_s32(vdupq_n_s32(0), wl0, xa0), wl1, xa1));
            const int ihi = vaddvq_s32(
                vdotq_s32(vdotq_s32(vdupq_n_s32(0), wh0, xb0), wh1, xb1));
            iacc  += (int)sc8[2 * c] * ilo + (int)sc8[2 * c + 1] * ihi;
            imins += (int)mn8[2 * c]
                       * ((int)xk[sb].bsums[4 * c]     + (int)xk[sb].bsums[4 * c + 1])
                   + (int)mn8[2 * c + 1]
                       * ((int)xk[sb].bsums[4 * c + 2] + (int)xk[sb].bsums[4 * c + 3]);
            q += 32;
        }
        a4[sb & 3] = fmaf(d * dk,       (float)iacc,  a4[sb & 3]);
        a4[sb & 3] = fmaf(-(dmin * dk), (float)imins, a4[sb & 3]);
    }
    return fl_fold4(a4);
}

inline float dot_i8k_q6_K_neon(const uint8_t * row, const blk_q8_K * xk, uint32_t cols) {
    const uint32_t nsb = cols / 256;
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m3 = vdupq_n_u8(0x03);
    float a4[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        const blk_q6_K * W = (const blk_q6_K *)(row + (size_t)sb * 210);
        const float d = fl_half2float(W->d);
        const float dk = xk[sb].d;
        int32_t iacc = 0;
        for (int h = 0; h < 2; ++h) {
            const uint8_t * ql = W->ql + h * 64;
            const uint8_t * qh = W->qh + h * 32;
            const int8_t  * sc = W->scales + h * 8;
            for (int s = 0; s < 4; ++s) {
                const int8_t * xg = xk[sb].qs + (h * 4 + s) * 32;
                for (int k16 = 0; k16 < 2; ++k16) {
                    const uint8x16_t lb = vld1q_u8(ql + (s & 1) * 32 + k16 * 16);
                    const uint8x16_t hv = vld1q_u8(qh + k16 * 16);
                    const uint8x16_t nib = (s < 2) ? vandq_u8(lb, m4)
                                                   : vshrq_n_u8(lb, 4);
                    uint8x16_t hb;
                    switch (s) {
                        case 0: hb = vandq_u8(hv, m3); break;
                        case 1: hb = vandq_u8(vshrq_n_u8(hv, 2), m3); break;
                        case 2: hb = vandq_u8(vshrq_n_u8(hv, 4), m3); break;
                        default: hb = vshrq_n_u8(hv, 6); break;
                    }
                    const int8x16_t wq = vreinterpretq_s8_u8(
                        vorrq_u8(nib, vshlq_n_u8(hb, 4)));
                    const int8x16_t xv = vld1q_s8(xg + k16 * 16);
                    const int idot = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), wq, xv));
                    const int bs = (int)xk[sb].bsums[2 * (h * 4 + s) + k16];
                    iacc += (int)sc[2 * s + k16] * (idot - 32 * bs);
                }
            }
        }
        a4[sb & 3] = fmaf(d * dk, (float)iacc, a4[sb & 3]);
    }
    return fl_fold4(a4);
}
#endif // __ARM_FEATURE_DOTPROD

// ---- P2.5 unpack-sharing multi-column kernels ------------------------------
// One weight row is unpacked ONCE, then every activation column's dot runs
// against the unpacked registers. Per-column arithmetic order is IDENTICAL to
// the single-column kernels above (integer work exact; the same fmaf fold
// sequence per column), so results are bitwise equal to nx single-column
// tiles -- the P2.4 multicol gate covers these paths unchanged. This attacks
// the ALU term P1.4 named (unpack was 2.1x q8_0's instructions/byte): at
// nx=8 the nibble/scale unpack amortizes 8x and only the SDOT work scales.
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
inline void dot_i8k_q4_K_neon_mc(const uint8_t * row,
                                 const blk_q8_K * const * xk, uint32_t nx,
                                 uint32_t cols, float * out) {
    const uint32_t nsb = cols / 256;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    float a4[kMaxDotCols][4];
    for (uint32_t p = 0; p < nx; ++p)
        a4[p][0] = a4[p][1] = a4[p][2] = a4[p][3] = 0.f;
    int32_t iacc[kMaxDotCols], imins[kMaxDotCols];
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        const blk_q4_K * W = (const blk_q4_K *)(row + (size_t)sb * 144);
        const float d = fl_half2float(W->d), dmin = fl_half2float(W->dmin);
        uint8_t sc8[8], mn8[8];
        fl_q4k_scales_all(W->scales, sc8, mn8);
        for (uint32_t p = 0; p < nx; ++p) { iacc[p] = 0; imins[p] = 0; }
        const uint8_t * q = W->qs;
        for (int c = 0; c < 4; ++c) {                 // 64-elem chunks
            const uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
            const int8x16_t wl0 = vreinterpretq_s8_u8(vandq_u8(q0, m4));
            const int8x16_t wl1 = vreinterpretq_s8_u8(vandq_u8(q1, m4));
            const int8x16_t wh0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
            const int8x16_t wh1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
            for (uint32_t p = 0; p < nx; ++p) {
                const int8_t * xs = xk[p][sb].qs;
                const int8x16_t xa0 = vld1q_s8(xs + c * 64);
                const int8x16_t xa1 = vld1q_s8(xs + c * 64 + 16);
                const int8x16_t xb0 = vld1q_s8(xs + c * 64 + 32);
                const int8x16_t xb1 = vld1q_s8(xs + c * 64 + 48);
                const int ilo = vaddvq_s32(
                    vdotq_s32(vdotq_s32(vdupq_n_s32(0), wl0, xa0), wl1, xa1));
                const int ihi = vaddvq_s32(
                    vdotq_s32(vdotq_s32(vdupq_n_s32(0), wh0, xb0), wh1, xb1));
                iacc[p]  += (int)sc8[2 * c] * ilo + (int)sc8[2 * c + 1] * ihi;
                imins[p] += (int)mn8[2 * c]
                              * ((int)xk[p][sb].bsums[4 * c]
                               + (int)xk[p][sb].bsums[4 * c + 1])
                          + (int)mn8[2 * c + 1]
                              * ((int)xk[p][sb].bsums[4 * c + 2]
                               + (int)xk[p][sb].bsums[4 * c + 3]);
            }
            q += 32;
        }
        for (uint32_t p = 0; p < nx; ++p) {
            const float dk = xk[p][sb].d;
            a4[p][sb & 3] = fmaf(d * dk,       (float)iacc[p],  a4[p][sb & 3]);
            a4[p][sb & 3] = fmaf(-(dmin * dk), (float)imins[p], a4[p][sb & 3]);
        }
    }
    for (uint32_t p = 0; p < nx; ++p) out[p] = fl_fold4(a4[p]);
}

// P4.2: i8mm SMMLA variant of the q4_K multi-column kernel. Processes a PAIR
// of weight rows against pairs of activation columns: vmmlaq_s32 consumes
// 2x8 * 8x2 int8 and yields the 2x2 int32 tile {r0c0, r0c1, r1c0, r1c1} in one
// instruction (32 MACs vs SDOT's 16).
//
// Two operand-layout facts decide the design (measured, tools/smmla_probe.c):
//   * activations must already be in [c0 8B | c1 8B] order, so they are
//     interleaved ONCE per tile into scratch by the caller and reused across
//     all t.rows/2 row pairs -- interleaving them per row-pair is a wash;
//   * weights are combined here at runtime, which amortizes over nx columns.
// Per-column arithmetic order is unchanged (same per-sub-block integer dot,
// same fmaf fold), so results stay BITWISE equal to dot_i8k_q4_K_neon_mc.
//
// xi: nx/2 col-pair planes, each cols bytes, plane p holding the interleaved
// 8-byte groups of columns 2p and 2p+1. Scales/bsums still come from xk.
#if defined(__ARM_FEATURE_MATMUL_INT8)
inline void dot_i8k_q4_K_neon_mc_smmla(const uint8_t * row0, const uint8_t * row1,
                                       const blk_q8_K * const * xk,
                                       const int8_t * xi, uint32_t nx,
                                       uint32_t cols, float * out0, float * out1) {
    const uint32_t nsb = cols / 256;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    float a4[2][kMaxDotCols][4];
    for (uint32_t h = 0; h < 2; ++h)
        for (uint32_t p = 0; p < nx; ++p)
            a4[h][p][0] = a4[h][p][1] = a4[h][p][2] = a4[h][p][3] = 0.f;
    int32_t iacc[2][kMaxDotCols], imins[2][kMaxDotCols];
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        const blk_q4_K * W0 = (const blk_q4_K *)(row0 + (size_t)sb * 144);
        const blk_q4_K * W1 = (const blk_q4_K *)(row1 + (size_t)sb * 144);
        const float d0 = fl_half2float(W0->d), dmin0 = fl_half2float(W0->dmin);
        const float d1 = fl_half2float(W1->d), dmin1 = fl_half2float(W1->dmin);
        uint8_t sc0[8], mn0[8], sc1[8], mn1[8];
        fl_q4k_scales_all(W0->scales, sc0, mn0);
        fl_q4k_scales_all(W1->scales, sc1, mn1);
        for (uint32_t h = 0; h < 2; ++h)
            for (uint32_t p = 0; p < nx; ++p) { iacc[h][p] = 0; imins[h][p] = 0; }
        const uint8_t * q0 = W0->qs;
        const uint8_t * q1 = W1->qs;
        for (int c = 0; c < 4; ++c) {                 // 64-elem chunks
            // unpack both rows: lo nibbles = sub-block 2c, hi = sub-block 2c+1
            const uint8x16_t p00 = vld1q_u8(q0), p01 = vld1q_u8(q0 + 16);
            const uint8x16_t p10 = vld1q_u8(q1), p11 = vld1q_u8(q1 + 16);
            const int8x16_t l00 = vreinterpretq_s8_u8(vandq_u8(p00, m4));
            const int8x16_t l01 = vreinterpretq_s8_u8(vandq_u8(p01, m4));
            const int8x16_t l10 = vreinterpretq_s8_u8(vandq_u8(p10, m4));
            const int8x16_t l11 = vreinterpretq_s8_u8(vandq_u8(p11, m4));
            const int8x16_t h00 = vreinterpretq_s8_u8(vshrq_n_u8(p00, 4));
            const int8x16_t h01 = vreinterpretq_s8_u8(vshrq_n_u8(p01, 4));
            const int8x16_t h10 = vreinterpretq_s8_u8(vshrq_n_u8(p10, 4));
            const int8x16_t h11 = vreinterpretq_s8_u8(vshrq_n_u8(p11, 4));
            // A operands: [row0 8B | row1 8B] per 8-element chunk
            const int8x16_t AL0 = vcombine_s8(vget_low_s8(l00),  vget_low_s8(l10));
            const int8x16_t AL1 = vcombine_s8(vget_high_s8(l00), vget_high_s8(l10));
            const int8x16_t AL2 = vcombine_s8(vget_low_s8(l01),  vget_low_s8(l11));
            const int8x16_t AL3 = vcombine_s8(vget_high_s8(l01), vget_high_s8(l11));
            const int8x16_t AH0 = vcombine_s8(vget_low_s8(h00),  vget_low_s8(h10));
            const int8x16_t AH1 = vcombine_s8(vget_high_s8(h00), vget_high_s8(h10));
            const int8x16_t AH2 = vcombine_s8(vget_low_s8(h01),  vget_low_s8(h11));
            const int8x16_t AH3 = vcombine_s8(vget_high_s8(h01), vget_high_s8(h11));
            for (uint32_t cp = 0; cp * 2 < nx; ++cp) {
                // interleaved activations: plane cp holds 2 columns, so its
                // stride is 2*cols and each 64-elem chunk occupies 128 bytes
                // (8 groups of [colA 8B | colB 8B]).
                const int8_t * b = xi + (size_t)cp * cols * 2
                                 + (size_t)sb * 512 + (size_t)c * 128;
                int32x4_t lo = vdupq_n_s32(0), hi = vdupq_n_s32(0);
                lo = vmmlaq_s32(lo, AL0, vld1q_s8(b));
                lo = vmmlaq_s32(lo, AL1, vld1q_s8(b + 16));
                lo = vmmlaq_s32(lo, AL2, vld1q_s8(b + 32));
                lo = vmmlaq_s32(lo, AL3, vld1q_s8(b + 48));
                hi = vmmlaq_s32(hi, AH0, vld1q_s8(b + 64));
                hi = vmmlaq_s32(hi, AH1, vld1q_s8(b + 80));
                hi = vmmlaq_s32(hi, AH2, vld1q_s8(b + 96));
                hi = vmmlaq_s32(hi, AH3, vld1q_s8(b + 112));
                // lanes: {r0c0, r0c1, r1c0, r1c1}
                const int ilo00 = vgetq_lane_s32(lo, 0), ilo01 = vgetq_lane_s32(lo, 1);
                const int ilo10 = vgetq_lane_s32(lo, 2), ilo11 = vgetq_lane_s32(lo, 3);
                const int ihi00 = vgetq_lane_s32(hi, 0), ihi01 = vgetq_lane_s32(hi, 1);
                const int ihi10 = vgetq_lane_s32(hi, 2), ihi11 = vgetq_lane_s32(hi, 3);
                const uint32_t pa = 2 * cp, pb = 2 * cp + 1;
                iacc[0][pa] += (int)sc0[2*c] * ilo00 + (int)sc0[2*c+1] * ihi00;
                iacc[0][pb] += (int)sc0[2*c] * ilo01 + (int)sc0[2*c+1] * ihi01;
                iacc[1][pa] += (int)sc1[2*c] * ilo10 + (int)sc1[2*c+1] * ihi10;
                iacc[1][pb] += (int)sc1[2*c] * ilo11 + (int)sc1[2*c+1] * ihi11;
                const int bsa = (int)xk[pa][sb].bsums[4*c]   + (int)xk[pa][sb].bsums[4*c+1];
                const int bsa2= (int)xk[pa][sb].bsums[4*c+2] + (int)xk[pa][sb].bsums[4*c+3];
                const int bsb = (int)xk[pb][sb].bsums[4*c]   + (int)xk[pb][sb].bsums[4*c+1];
                const int bsb2= (int)xk[pb][sb].bsums[4*c+2] + (int)xk[pb][sb].bsums[4*c+3];
                imins[0][pa] += (int)mn0[2*c] * bsa + (int)mn0[2*c+1] * bsa2;
                imins[0][pb] += (int)mn0[2*c] * bsb + (int)mn0[2*c+1] * bsb2;
                imins[1][pa] += (int)mn1[2*c] * bsa + (int)mn1[2*c+1] * bsa2;
                imins[1][pb] += (int)mn1[2*c] * bsb + (int)mn1[2*c+1] * bsb2;
            }
            q0 += 32; q1 += 32;
        }
        for (uint32_t p = 0; p < nx; ++p) {
            const float dk = xk[p][sb].d;
            a4[0][p][sb & 3] = fmaf(d0 * dk,       (float)iacc[0][p],  a4[0][p][sb & 3]);
            a4[0][p][sb & 3] = fmaf(-(dmin0 * dk), (float)imins[0][p], a4[0][p][sb & 3]);
            a4[1][p][sb & 3] = fmaf(d1 * dk,       (float)iacc[1][p],  a4[1][p][sb & 3]);
            a4[1][p][sb & 3] = fmaf(-(dmin1 * dk), (float)imins[1][p], a4[1][p][sb & 3]);
        }
    }
    for (uint32_t p = 0; p < nx; ++p) {
        out0[p] = fl_fold4(a4[0][p]);
        out1[p] = fl_fold4(a4[1][p]);
    }
}

// Interleave nx activation column planes into col-pair order, ONCE per tile
// (amortized over t.rows/2 row pairs). Plane cp holds columns 2cp and 2cp+1 as
// alternating 8-byte groups -- the SMMLA B-operand layout. Scratch must be
// nx * cols bytes (nx/2 planes, each 2*cols).
inline void fl_interleave_act_q8k(const blk_q8_K * const * xk, uint32_t nx,
                                  uint32_t cols, int8_t * xi) {
    const uint32_t nsb = cols / 256;
    for (uint32_t cp = 0; cp * 2 < nx; ++cp) {
        int8_t * dst = xi + (size_t)cp * cols * 2;
        const blk_q8_K * A = xk[2 * cp];
        const blk_q8_K * B = xk[2 * cp + 1];
        for (uint32_t sb = 0; sb < nsb; ++sb) {
            const int8_t * a = A[sb].qs;
            const int8_t * b = B[sb].qs;
            int8_t * d = dst + (size_t)sb * 512;
            for (uint32_t g = 0; g < 32; ++g) {       // 32 groups of 8 bytes
                memcpy(d + (size_t)g * 16,     a + (size_t)g * 8, 8);
                memcpy(d + (size_t)g * 16 + 8, b + (size_t)g * 8, 8);
            }
        }
    }
}
#endif // __ARM_FEATURE_MATMUL_INT8

inline void dot_i8k_q6_K_neon_mc(const uint8_t * row,
                                 const blk_q8_K * const * xk, uint32_t nx,
                                 uint32_t cols, float * out) {
    const uint32_t nsb = cols / 256;
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m3 = vdupq_n_u8(0x03);
    float a4[kMaxDotCols][4];
    for (uint32_t p = 0; p < nx; ++p)
        a4[p][0] = a4[p][1] = a4[p][2] = a4[p][3] = 0.f;
    int32_t iacc[kMaxDotCols];
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        const blk_q6_K * W = (const blk_q6_K *)(row + (size_t)sb * 210);
        const float d = fl_half2float(W->d);
        for (uint32_t p = 0; p < nx; ++p) iacc[p] = 0;
        for (int h = 0; h < 2; ++h) {
            const uint8_t * ql = W->ql + h * 64;
            const uint8_t * qh = W->qh + h * 32;
            const int8_t  * sc = W->scales + h * 8;
            for (int s = 0; s < 4; ++s) {
                for (int k16 = 0; k16 < 2; ++k16) {
                    const uint8x16_t lb = vld1q_u8(ql + (s & 1) * 32 + k16 * 16);
                    const uint8x16_t hv = vld1q_u8(qh + k16 * 16);
                    const uint8x16_t nib = (s < 2) ? vandq_u8(lb, m4)
                                                   : vshrq_n_u8(lb, 4);
                    uint8x16_t hb;
                    switch (s) {
                        case 0: hb = vandq_u8(hv, m3); break;
                        case 1: hb = vandq_u8(vshrq_n_u8(hv, 2), m3); break;
                        case 2: hb = vandq_u8(vshrq_n_u8(hv, 4), m3); break;
                        default: hb = vshrq_n_u8(hv, 6); break;
                    }
                    const int8x16_t wq = vreinterpretq_s8_u8(
                        vorrq_u8(nib, vshlq_n_u8(hb, 4)));
                    for (uint32_t p = 0; p < nx; ++p) {
                        const int8_t * xg = xk[p][sb].qs + (h * 4 + s) * 32;
                        const int8x16_t xv = vld1q_s8(xg + k16 * 16);
                        const int idot = vaddvq_s32(
                            vdotq_s32(vdupq_n_s32(0), wq, xv));
                        const int bs =
                            (int)xk[p][sb].bsums[2 * (h * 4 + s) + k16];
                        iacc[p] += (int)sc[2 * s + k16] * (idot - 32 * bs);
                    }
                }
            }
        }
        for (uint32_t p = 0; p < nx; ++p)
            a4[p][sb & 3] = fmaf(d * xk[p][sb].d, (float)iacc[p],
                                 a4[p][sb & 3]);
    }
    for (uint32_t p = 0; p < nx; ++p) out[p] = fl_fold4(a4[p]);
}

inline void dot_i8_q8_0_neon_mc(const uint8_t * row,
                                const blk_q8_1 * const * xq, uint32_t nx,
                                uint32_t cols, float * out) {
    const uint32_t nb = cols / 32;
    float a4[kMaxDotCols][4];
    for (uint32_t p = 0; p < nx; ++p)
        a4[p][0] = a4[p][1] = a4[p][2] = a4[p][3] = 0.f;
    for (uint32_t b = 0; b < nb; ++b) {
        const blk_q8_0 * W = (const blk_q8_0 *)(row + (size_t)b * 34);
        const int8x16_t w0 = vld1q_s8(W->qs), w1 = vld1q_s8(W->qs + 16);
        const float dw = fl_half2float(W->d);
        for (uint32_t p = 0; p < nx; ++p) {
            const int8x16_t x0 = vld1q_s8(xq[p][b].qs);
            const int8x16_t x1 = vld1q_s8(xq[p][b].qs + 16);
            const int32x4_t s  = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0),
                                           w1, x1);
            a4[p][b & 3] = fmaf(dw * fl_half2float(xq[p][b].d),
                                (float)vaddvq_s32(s), a4[p][b & 3]);
        }
    }
    for (uint32_t p = 0; p < nx; ++p) out[p] = fl_fold4(a4[p]);
}
#endif // __ARM_FEATURE_DOTPROD

inline float dot_row_i8k(Fmt f, const uint8_t * row, const blk_q8_K * xk,
                         uint32_t cols) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    switch (f) {
        case Fmt::Q4_K: return dot_i8k_q4_K_neon(row, xk, cols);
        case Fmt::Q6_K: return dot_i8k_q6_K_neon(row, xk, cols);
        default: break;
    }
#endif
    return fl_dot_i8k_any(f, row, xk, cols);
}

// ---- P3.2 shadow kernels ---------------------------------------------------
// Read the pre-unpacked int8 shadows (shadow.h). Integer work is exact and
// the per-superblock/per-block fmaf fold sequence is identical to the packed
// kernels, so results are BITWISE equal to the packed paths on every arch
// (gated by tests/test_shadow.cpp). Scalar variants mirror fl_dot_i8k_q4_K /
// fl_dot_i8_q5_0; NEON variants replace unpack logicals with plain loads.
inline float dot_i8k_q4_K_shadow(const uint8_t * srow, const blk_q8_K * xk,
                                 uint32_t cols) {
    const uint32_t nsb = cols / 256;
    float a4[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        const uint8_t * o = srow + (size_t)sb * kShadowQ4KSbBytes;
        float d, dmin;
        memcpy(&d, o, 4); memcpy(&dmin, o + 4, 4);
        const uint8_t * sc8 = o + 8;
        const uint8_t * mn8 = o + 16;
        const int8_t  * qs  = (const int8_t *)(o + 24);
        int32_t iacc = 0, imins = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        const int8_t * xs = xk[sb].qs;
        for (int c = 0; c < 4; ++c) {
            const int8x16_t wl0 = vld1q_s8(qs + c * 64);
            const int8x16_t wl1 = vld1q_s8(qs + c * 64 + 16);
            const int8x16_t wh0 = vld1q_s8(qs + c * 64 + 32);
            const int8x16_t wh1 = vld1q_s8(qs + c * 64 + 48);
            const int8x16_t xa0 = vld1q_s8(xs + c * 64);
            const int8x16_t xa1 = vld1q_s8(xs + c * 64 + 16);
            const int8x16_t xb0 = vld1q_s8(xs + c * 64 + 32);
            const int8x16_t xb1 = vld1q_s8(xs + c * 64 + 48);
            const int ilo = vaddvq_s32(
                vdotq_s32(vdotq_s32(vdupq_n_s32(0), wl0, xa0), wl1, xa1));
            const int ihi = vaddvq_s32(
                vdotq_s32(vdotq_s32(vdupq_n_s32(0), wh0, xb0), wh1, xb1));
            iacc  += (int)sc8[2 * c] * ilo + (int)sc8[2 * c + 1] * ihi;
            imins += (int)mn8[2 * c]
                       * ((int)xk[sb].bsums[4 * c]     + (int)xk[sb].bsums[4 * c + 1])
                   + (int)mn8[2 * c + 1]
                       * ((int)xk[sb].bsums[4 * c + 2] + (int)xk[sb].bsums[4 * c + 3]);
        }
#else
        for (int g = 0; g < 8; ++g) {
            const int8_t * w = qs + (g / 2) * 64 + (g & 1) * 32;
            const int8_t * xg = xk[sb].qs + g * 32;
            int idot = 0;
            for (int i = 0; i < 32; ++i) idot += (int)w[i] * (int)xg[i];
            iacc  += (int)sc8[g] * idot;
            imins += (int)mn8[g]
                   * ((int)xk[sb].bsums[2 * g] + (int)xk[sb].bsums[2 * g + 1]);
        }
#endif
        const float dk = xk[sb].d;
        a4[sb & 3] = fmaf(d * dk,       (float)iacc,  a4[sb & 3]);
        a4[sb & 3] = fmaf(-(dmin * dk), (float)imins, a4[sb & 3]);
    }
    return fl_fold4(a4);
}

// P4.4: two rows at once from the SMMLA row-pair-interleaved shadow.
// A operand = [row0 8B ; row1 8B] straight out of memory - no runtime repack,
// which is what destroyed P4.2's win. B = the single activation column
// duplicated into both halves, so MMLA lanes 1,3 are discarded by design;
// lanes 0,2 hold exactly the integers the SDOT path computes, and each row
// keeps its own fmaf fold sequence, so results are BITWISE identical to
// dot_i8k_q4_K_shadow (P4.4 probe: max_rel 0.000e+00). Measured +24.5% at
// nx=1 - the shape expert tiles actually have (P4.2/P4.3).
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
inline void dot2_i8k_q4_K_shadow_smmla(const uint8_t * pbase,
                                       const blk_q8_K * xk, uint32_t cols,
                                       float * o0, float * o1) {
    const uint32_t nsb = cols / 256;
    float a0[4] = {0.f, 0.f, 0.f, 0.f}, a1[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        const uint8_t * o = pbase + (size_t)sb * kShadowQ4KPairSbBytes;
        float d0, dm0, d1, dm1;
        memcpy(&d0, o, 4);       memcpy(&dm0, o + 4, 4);
        memcpy(&d1, o + 24, 4);  memcpy(&dm1, o + 28, 4);
        const uint8_t * sc0 = o + 8,  * mn0 = o + 16;
        const uint8_t * sc1 = o + 32, * mn1 = o + 40;
        const int8_t  * qi  = (const int8_t *)(o + 48);   // 32 x (r0 8B, r1 8B)
        const int8_t  * xs  = xk[sb].qs;
        int32_t iacc0 = 0, iacc1 = 0, imins0 = 0, imins1 = 0;
        for (int g = 0; g < 8; ++g) {                     // 32 elements each
            int32x4_t acc = vdupq_n_s32(0);
            for (int c8 = 0; c8 < 4; ++c8) {              // 8 elements each
                const int8x16_t A = vld1q_s8(qi + (size_t)(g * 4 + c8) * 16);
                const int8x8_t  x = vld1_s8(xs + g * 32 + c8 * 8);
                acc = vmmlaq_s32(acc, A, vcombine_s8(x, x));
            }
            iacc0 += (int)sc0[g] * vgetq_lane_s32(acc, 0);
            iacc1 += (int)sc1[g] * vgetq_lane_s32(acc, 2);
            const int bs = (int)xk[sb].bsums[2 * g] + (int)xk[sb].bsums[2 * g + 1];
            imins0 += (int)mn0[g] * bs;
            imins1 += (int)mn1[g] * bs;
        }
        const float dk = xk[sb].d;
        a0[sb & 3] = fmaf(d0 * dk,     (float)iacc0,  a0[sb & 3]);
        a0[sb & 3] = fmaf(-(dm0 * dk), (float)imins0, a0[sb & 3]);
        a1[sb & 3] = fmaf(d1 * dk,     (float)iacc1,  a1[sb & 3]);
        a1[sb & 3] = fmaf(-(dm1 * dk), (float)imins1, a1[sb & 3]);
    }
    *o0 = fl_fold4(a0);
    *o1 = fl_fold4(a1);
}
#endif

inline float dot_i8_q5_0_shadow(const uint8_t * srow, const blk_q8_1 * xq,
                                uint32_t cols) {
    const uint32_t nb = cols / 32;
    float a4[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t * o = srow + (size_t)b * kShadowQ50BlkBytes;
        float d; memcpy(&d, o, 4);
        const int8_t * w = (const int8_t *)(o + 4);
        int idot = 0, isum = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        const int8x16_t ones = vdupq_n_s8(1);
        const int8x16_t w0 = vld1q_s8(w), w1 = vld1q_s8(w + 16);
        const int8x16_t x0 = vld1q_s8(xq[b].qs), x1 = vld1q_s8(xq[b].qs + 16);
        idot = vaddvq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0), w1, x1));
        isum = vaddvq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), ones, x0), ones, x1));
#else
        for (int i = 0; i < 32; ++i) {
            idot += (int)w[i] * (int)xq[b].qs[i];
            isum += (int)xq[b].qs[i];
        }
#endif
        a4[b & 3] = fmaf(d * fl_half2float(xq[b].d),
                         (float)(idot - 16 * isum), a4[b & 3]);
    }
    return fl_fold4(a4);
}

inline float dot_row_i8(Fmt f, const uint8_t * row, const blk_q8_1 * xq,
                        uint32_t cols) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    switch (f) {
        case Fmt::Q8_0: return dot_i8_q8_0_neon(row, xq, cols);
        case Fmt::Q4_K: return dot_i8_q4_K_neon_v1(row, xq, cols);
        case Fmt::Q6_K: return dot_i8_q6_K_neon(row, xq, cols);
        default: break;
    }
#endif
    return fl_dot_i8_any(f, row, xq, cols);
}

// P1.2 per-(fmt,engine) path masks: bit (1 << (int)fmt) set = int8 path.
// Defaults from measured rates : CPU
// int8 wins every quant format; GPU int8 wins q8_0/q5_0/q4_K, loses q6_K.
inline constexpr uint32_t kI8MaskCpuDefault =
    (1u << (int)Fmt::Q8_0) | (1u << (int)Fmt::Q5_0) |
    (1u << (int)Fmt::Q4_K) | (1u << (int)Fmt::Q6_K);
// P5.2: codebook formats take the GPU int8 path by default - that is what
// P5.0 measured (IQ2_XXS 175 Gw/s, IQ3_XXS 211, against q4_K's 188); without
// the bits they would silently fall to the slower dequant kernels. They stay
// out of the CPU mask above: fl_dot_i8_any has no codebook case (runtime.cpp
// clears the bits defensively, fl_task_queue keeps the tiles off the pool).
inline constexpr uint32_t kI8MaskGpuDefault =
    (1u << (int)Fmt::Q8_0) | (1u << (int)Fmt::Q5_0) | (1u << (int)Fmt::Q4_K) |
    (1u << (int)Fmt::IQ2_XXS) | (1u << (int)Fmt::IQ3_XXS);
// P1.5 q8_K-appendix path masks (K formats only; take precedence over the
// q8_1 bits above when set). Arbitrated on thor :
// CPU both K formats (probe q4_K 14.7->25.9, q6_K 15.0->21.4 GB/s t8);
// GPU q4_K only - the q6_K i8k kernel probes at 40.9 vs its dequant 50.7,
// and executor cells cannot distinguish the two GPU masks (39.13 +/- 0.99
// vs 38.81 +/- 0.85), so the probe evidence keeps q6_K-GPU on dequant.
inline constexpr uint32_t kI8KMaskCpuDefault =
    (1u << (int)Fmt::Q4_K) | (1u << (int)Fmt::Q6_K);
inline constexpr uint32_t kI8KMaskGpuDefault =
    (1u << (int)Fmt::Q4_K);

inline constexpr uint32_t kMaxActBlocks = 128;   // 4096 cols; V2-Lite max 2816
// P4.2 SMMLA activation scratch: nx * cols bytes (nx/2 planes of 2*cols).
// 8 columns x 4096 cols = 32 KiB, per worker thread.
inline constexpr size_t kSmmlaScratchBytes = 32u * 1024u;

#if defined(__ARM_NEON)
// P2.3 ATTN_HEAD fast path. Bitwise-equal to the striped-8 canonical in
// decode_params.h by construction: two f32x4 fmla accumulators are exactly
// stripes s0..s3 / s4..s7 (vfmaq lane = fmaf), lanes folded in the same
// left-associative s0..s7 order; the context loop's vfmaq per element equals
// the canonical's fmaf per element (same j-major chain per output dim).
// Softmax lines are copied verbatim from the canonical. Caller guarantees
// dk % 8 == 0 and dv % 4 == 0 (checked at dispatch; V2-Lite: 192/128).
inline void fl_op_attn_head_neon(const AttnHeadP & p) {
    const uint32_t n  = (uint32_t)*p.pos + 1;
    const uint32_t dk = p.dk, dv = p.dv, h = p.head;
    const float * q = p.q + (size_t)h * dk;
    float mx = -1e30f;
    for (uint32_t j = 0; j < n; ++j) {
        const float * k = p.kcache + ((size_t)j * p.n_head + h) * dk;
        float32x4_t a = vdupq_n_f32(0.f);   // stripes 0..3
        float32x4_t b = vdupq_n_f32(0.f);   // stripes 4..7
        for (uint32_t d = 0; d < dk; d += 8) {
            a = vfmaq_f32(a, vld1q_f32(q + d),     vld1q_f32(k + d));
            b = vfmaq_f32(b, vld1q_f32(q + d + 4), vld1q_f32(k + d + 4));
        }
        const float s0 = vgetq_lane_f32(a, 0), s1 = vgetq_lane_f32(a, 1);
        const float s2 = vgetq_lane_f32(a, 2), s3 = vgetq_lane_f32(a, 3);
        const float s4 = vgetq_lane_f32(b, 0), s5 = vgetq_lane_f32(b, 1);
        const float s6 = vgetq_lane_f32(b, 2), s7 = vgetq_lane_f32(b, 3);
        float acc = ((((((s0 + s1) + s2) + s3) + s4) + s5) + s6) + s7;
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
    for (uint32_t d = 0; d < dv; d += 4) vst1q_f32(o + d, vdupq_n_f32(0.f));
    for (uint32_t j = 0; j < n; ++j) {
        const float w = p.scratch[j] * inv;
        const float32x4_t wv = vdupq_n_f32(w);
        const float * v = p.vcache + ((size_t)j * p.n_head + h) * dv;
        for (uint32_t d = 0; d < dv; d += 4) {
            float32x4_t ov = vld1q_f32(o + d);
            ov = vfmaq_f32(ov, wv, vld1q_f32(v + d));
            vst1q_f32(o + d, ov);
        }
    }
}
#endif // __ARM_NEON

// CPU-side task execution with kernel-mode switch. MERGE/SIGNAL are small and
// stay canonical in both modes; ACT_Q8 is canonical by definition.
// i8_mask: per-format int8-path selection (0 disables the int8 path).
inline void exec_task_cpu(Task & t, bool fast, uint32_t i8_mask,
                          uint32_t i8k_mask = 0) {
    if (t.kind == TaskKind::ACT_Q8) {
        fl_exec_act_q8(t);
        return;
    }
    // P2.0: expert indirection resolves to a plain STREAM_DOT on a local copy
    // (same fast paths); the other decode ops are canonical-only by contract.
    if (t.kind == TaskKind::STREAM_DOT_IDX) {
        Task tt = t;
        tt.kind = TaskKind::STREAM_DOT;
        tt.w    = fl_idx_w(t);
        exec_task_cpu(tt, fast, i8_mask, i8k_mask);
        return;
    }
    // P2.3: ATTN_HEAD takes the NEON fast path when lane-matchable (bitwise
    // equal to canonical, see above). FASTLLM_ATTN_NEON=0 forces canonical
    // for same-binary A/Bs.
    if (t.kind == TaskKind::ATTN_HEAD) {
#if defined(__ARM_NEON)
        static const bool attn_neon = [] {
            const char * e = getenv("FASTLLM_ATTN_NEON");
            return !(e && e[0] == '0');
        }();
        const AttnHeadP & ap = *(const AttnHeadP *)t.aux;
        if (attn_neon && (ap.dk & 7u) == 0 && (ap.dv & 3u) == 0) {
            fl_op_attn_head_neon(ap);
            return;
        }
#endif
        exec_task_scalar_host(t);
        return;
    }
    if (t.kind >= TaskKind::RMSNORM) {
        exec_task_scalar_host(t);
        return;
    }
    if (fast && t.kind == TaskKind::STREAM_DOT) {
        const uint8_t * w = (const uint8_t *)t.w;
        float * y = (float *)t.y;
        const uint64_t rb = fl_row_bytes(t.fmt, t.cols);
        // P2.4: nx columns share every weight-row read. Per-(row,column)
        // accumulation order is untouched, so output is bitwise identical to
        // nx single-column tiles; only the DRAM traffic changes (1/nx).
        const uint32_t nx = t.nx ? t.nx : 1;
        const size_t   xs = t.xstride, ys = t.ystride;
        // P1.5: q8_K appendix path for K formats takes precedence
        if ((t.fmt == Fmt::Q4_K || t.fmt == Fmt::Q6_K)
            && ((i8k_mask >> (int)t.fmt) & 1u)) {
            const blk_q8_K * xk[kMaxDotCols];
            for (uint32_t p = 0; p < nx; ++p)
                xk[p] = fl_xqk_of((const float *)t.x + p * xs, t.cols);
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
            // P2.5: unpack once per row, all columns dot the shared registers
            // (bitwise == per-column loop; FASTLLM_MC_LEGACY=1 keeps the P2.4
            // load-sharing loop for same-binary attribution)
            if (nx > 1 && !fl_mc_legacy()) {
                float vals[kMaxDotCols];
#if defined(__ARM_FEATURE_MATMUL_INT8)
                // P4.2: i8mm SMMLA over row PAIRS. The activation interleave
                // is done once here and reused by every pair, which is what
                // makes SMMLA pay (per-pair interleaving measured as a wash).
                if (t.fmt == Fmt::Q4_K && fl_smmla_on() && nx >= 2 && !(nx & 1)
                    && t.rows >= 2 && t.cols % 256 == 0
                    && (size_t)nx * t.cols <= kSmmlaScratchBytes) {
                    static thread_local int8_t xi[kSmmlaScratchBytes];
                    fl_interleave_act_q8k(xk, nx, t.cols, xi);
                    float v0[kMaxDotCols], v1[kMaxDotCols];
                    uint32_t r = 0;
                    for (; r + 1 < t.rows; r += 2) {
                        dot_i8k_q4_K_neon_mc_smmla(w + (size_t)r * rb,
                                                   w + (size_t)(r + 1) * rb,
                                                   xk, xi, nx, t.cols, v0, v1);
                        for (uint32_t p = 0; p < nx; ++p) {
                            y[p * ys + r]     += v0[p];
                            y[p * ys + r + 1] += v1[p];
                        }
                    }
                    for (; r < t.rows; ++r) {          // odd tail
                        dot_i8k_q4_K_neon_mc(w + (size_t)r * rb, xk, nx,
                                             t.cols, vals);
                        for (uint32_t p = 0; p < nx; ++p) y[p * ys + r] += vals[p];
                    }
                    return;
                }
#endif
                for (uint32_t r = 0; r < t.rows; ++r) {
                    const uint8_t * row = w + (size_t)r * rb;
                    if (t.fmt == Fmt::Q4_K)
                        dot_i8k_q4_K_neon_mc(row, xk, nx, t.cols, vals);
                    else
                        dot_i8k_q6_K_neon_mc(row, xk, nx, t.cols, vals);
                    for (uint32_t p = 0; p < nx; ++p) y[p * ys + r] += vals[p];
                }
                return;
            }
#endif
            // P3.2: full-shadow path (q4_K only on the i8k side). Bitwise
            // equal to the packed path; skips nibble+scale unpack. Multicol
            // unpack-sharing kernels already amortize the unpack, so the
            // shadow applies to the per-column loop only.
            if (t.fmt == Fmt::Q4_K && fl_shadow_enabled()) {
                if (const ShadowSpan * sp = fl_shadow_find(w)) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
                    // P4.4: row-pair SMMLA over the interleaved layout. Tiles
                    // start on even rows (rowsz and rows-per-expert are both
                    // even), so pairs align; the guard falls back rather than
                    // trusting that, and an odd tail row uses the plain path
                    // reconstructed from its half of the pair.
                    const uint64_t r0i = fl_shadow_row_index(sp, w);
                    if (sp->smmla && !(r0i & 1)) {
                        const uint64_t prb2 = sp->shadow_row_bytes * 2;
                        uint32_t r = 0;
                        for (; r + 1 < t.rows; r += 2) {
                            const uint8_t * pb =
                                sp->shadow + ((r0i + r) >> 1) * prb2;
                            for (uint32_t p = 0; p < nx; ++p) {
                                float v0, v1;
                                dot2_i8k_q4_K_shadow_smmla(pb, xk[p], t.cols,
                                                           &v0, &v1);
                                y[p * ys + r]     += v0;
                                y[p * ys + r + 1] += v1;
                            }
                        }
                        for (; r < t.rows; ++r) {   // odd tail: packed path
                            const uint8_t * row = w + (size_t)r * rb;
                            for (uint32_t p = 0; p < nx; ++p)
                                y[p * ys + r] +=
                                    dot_row_i8k(t.fmt, row, xk[p], t.cols);
                        }
                        return;
                    }
                    if (sp->smmla) {                // misaligned start: packed
                        for (uint32_t r = 0; r < t.rows; ++r) {
                            const uint8_t * row = w + (size_t)r * rb;
                            for (uint32_t p = 0; p < nx; ++p)
                                y[p * ys + r] +=
                                    dot_row_i8k(t.fmt, row, xk[p], t.cols);
                        }
                        return;
                    }
#endif
                    const uint8_t * s0 = fl_shadow_row(sp, w);
                    for (uint32_t r = 0; r < t.rows; ++r) {
                        const uint8_t * srow = s0 + (size_t)r * sp->shadow_row_bytes;
                        for (uint32_t p = 0; p < nx; ++p)
                            y[p * ys + r] += dot_i8k_q4_K_shadow(srow, xk[p], t.cols);
                    }
                    return;
                }
            }
            for (uint32_t r = 0; r < t.rows; ++r) {
                const uint8_t * row = w + (size_t)r * rb;
                for (uint32_t p = 0; p < nx; ++p)
                    y[p * ys + r] += dot_row_i8k(t.fmt, row, xk[p], t.cols);
            }
            return;
        }
        if (t.fmt != Fmt::F32 && ((i8_mask >> (int)t.fmt) & 1u)) {
            const blk_q8_1 * xq[kMaxDotCols];
            for (uint32_t p = 0; p < nx; ++p)
                xq[p] = fl_xq_of((const float *)t.x + p * xs, t.cols);
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
            if (t.fmt == Fmt::Q4_K && t.cols / 32 <= kMaxActBlocks) {
                // v2: hoist activation block sums/scales out of the row loop
                // (bitwise-identical to v1/reference; see dot_i8_q4_K_neon_v2)
                int32_t xsum[kMaxDotCols][kMaxActBlocks];
                float   xd[kMaxDotCols][kMaxActBlocks];
                const uint32_t nb = t.cols / 32;
                for (uint32_t p = 0; p < nx; ++p)
                    for (uint32_t b = 0; b < nb; ++b) {
                        const int8x16_t x0 = vld1q_s8(xq[p][b].qs);
                        const int8x16_t x1 = vld1q_s8(xq[p][b].qs + 16);
                        const int8x16_t ones = vdupq_n_s8(1);
                        xsum[p][b] = vaddvq_s32(
                            vdotq_s32(vdotq_s32(vdupq_n_s32(0), ones, x0),
                                      ones, x1));
                        xd[p][b] = fl_half2float(xq[p][b].d);
                    }
                for (uint32_t r = 0; r < t.rows; ++r) {
                    const uint8_t * row = w + (size_t)r * rb;
                    for (uint32_t p = 0; p < nx; ++p)
                        y[p * ys + r] += dot_i8_q4_K_neon_v2(
                            row, xq[p], xsum[p], xd[p], t.cols);
                }
                return;
            }
#endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
            if (nx > 1 && !fl_mc_legacy() && t.fmt == Fmt::Q8_0) {
                float vals[kMaxDotCols];
                for (uint32_t r = 0; r < t.rows; ++r) {
                    const uint8_t * row = w + (size_t)r * rb;
                    dot_i8_q8_0_neon_mc(row, xq, nx, t.cols, vals);
                    for (uint32_t p = 0; p < nx; ++p) y[p * ys + r] += vals[p];
                }
                return;
            }
#endif
            // P3.2: full-shadow path (q5_0 on the q8_1 side): replaces the
            // scalar qh bit-scatter compose with straight int8 loads + SDOT.
            // Bitwise equal to fl_dot_i8_q5_0 (integer work exact, same fold).
            if (t.fmt == Fmt::Q5_0 && fl_shadow_enabled()) {
                if (const ShadowSpan * sp = fl_shadow_find(w)) {
                    const uint8_t * s0 = fl_shadow_row(sp, w);
                    for (uint32_t r = 0; r < t.rows; ++r) {
                        const uint8_t * srow = s0 + (size_t)r * sp->shadow_row_bytes;
                        for (uint32_t p = 0; p < nx; ++p)
                            y[p * ys + r] += dot_i8_q5_0_shadow(srow, xq[p], t.cols);
                    }
                    return;
                }
            }
            for (uint32_t r = 0; r < t.rows; ++r) {
                const uint8_t * row = w + (size_t)r * rb;
                for (uint32_t p = 0; p < nx; ++p)
                    y[p * ys + r] += dot_row_i8(t.fmt, row, xq[p], t.cols);
            }
            return;
        }
        // P2.5: dequant fallback shares the per-chunk dequant buffer across
        // columns (chunks ascending, per-column acc += per chunk -- identical
        // order to dot_row_fast_dequant per column).
        if (nx > 1 && !fl_mc_legacy() && t.fmt != Fmt::F32
            && t.fmt != Fmt::Q8_0) {
            for (uint32_t r = 0; r < t.rows; ++r) {
                const uint8_t * row = w + (size_t)r * rb;
                float buf[256];
                float acc[kMaxDotCols];
                for (uint32_t p = 0; p < nx; ++p) acc[p] = 0.f;
                for (uint32_t e = 0; e < t.cols; e += 256) {
                    const uint32_t n = (t.cols - e < 256) ? t.cols - e : 256;
                    dequant_chunk(t.fmt, row, e, n, buf);
                    for (uint32_t p = 0; p < nx; ++p)
                        acc[p] += dot_row_fast(
                            buf, (const float *)t.x + p * xs + e, n);
                }
                for (uint32_t p = 0; p < nx; ++p) y[p * ys + r] += acc[p];
            }
            return;
        }
        for (uint32_t r = 0; r < t.rows; ++r) {
            const uint8_t * row = w + (size_t)r * rb;
            for (uint32_t p = 0; p < nx; ++p) {
                const float * x = (const float *)t.x + p * xs;
                float * yp = y + p * ys;
                switch (t.fmt) {
                    case Fmt::F32:  yp[r] += dot_row_fast((const float *)row, x, t.cols); break;
                    case Fmt::Q8_0: yp[r] += dot_row_fast_q8_0(row, x, t.cols); break;
                    default:        yp[r] += dot_row_fast_dequant(t.fmt, row, x, t.cols); break;
                }
            }
        }
        return;
    }
    exec_task_scalar_host(t);
}

} // namespace fastllm
