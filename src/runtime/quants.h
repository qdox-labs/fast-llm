// GGUF/ggml quantized weight block layouts + scalar dequant-dot references.
// Layouts ported from ggml (ggml-common.h / ggml-quants.c, MIT); byte-for-byte
// identical structs so tensors mmap'd from a GGUF are consumed in place.
//
// Canonical order contract: fl_dot_row_any() accumulates ELEMENTWISE ASCENDING
// per row (identical FP order to "dequantize row, then dot ascending"), on
// host and device alike. It is THE bitwise reference for quant tiles, exactly
// as the f32 loop in exec_task_scalar is for F32 tiles. Fast engine kernels
// (kernels.h CPU, kernels_gpu.h GPU) may reorder and are tolerance-validated.
#pragma once
#include "fastllm/fastllm.h"
#include "atomics.h"
#include "model/iq_tables.h"
#include <cmath>
#include <cstring>
#if defined(FASTLLM_CUDA) && defined(__CUDACC__)
#include <cuda_fp16.h>
#endif

namespace fastllm {

// ---- block layouts (packed by construction: members need <= 2-byte align) --
struct blk_q8_0 { uint16_t d; int8_t qs[32]; };                      // 34 B/32
struct blk_q5_0 { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; };      // 22 B/32
struct blk_q4_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; }; // 144 B/256
struct blk_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; }; // 210 B/256
// P5.0 codebook formats. IQ2_XXS: per 32-weight chunk, 4 u16 = two u32 words -
// word0's 4 bytes are grid indices (8 weights each), word1 packs 4x 7-bit sign
// indices plus a 4-bit scale in its top nibble. IQ3_XXS: 64 grid indices of 4
// weights each, then 8 u32 of the same sign+scale packing.
struct blk_iq2_xxs { uint16_t d; uint16_t qs[32]; };                  // 66 B/256
struct blk_iq3_xxs { uint16_t d; uint8_t qs[96]; };                   // 98 B/256
static_assert(sizeof(blk_q8_0) ==  34, "q8_0 layout");
static_assert(sizeof(blk_q5_0) ==  22, "q5_0 layout");
static_assert(sizeof(blk_q4_K) == 144, "q4_K layout");
static_assert(sizeof(blk_q6_K) == 210, "q6_K layout");
static_assert(sizeof(blk_iq2_xxs) == 66, "iq2_xxs layout");
static_assert(sizeof(blk_iq3_xxs) == 98, "iq3_xxs layout");

FL_HD float fl_half2float(uint16_t h) {
#if defined(__CUDA_ARCH__)
    __half_raw r; r.x = h;
    return __half2float(r);
#else
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t em   = h & 0x7FFFu;
    uint32_t f;
    if (em >= 0x7C00u) {                    // inf / nan
        f = sign | 0x7F800000u | ((uint32_t)(em & 0x03FFu) << 13);
    } else if (em >= 0x0400u) {             // normal
        f = sign | ((em + 0x1C000u) << 13);
    } else if (em == 0) {
        f = sign;
    } else {                                // subnormal: renormalize
        uint32_t m = em;
        int e = -1;
        do { m <<= 1; ++e; } while (!(m & 0x0400u));
        f = sign | ((uint32_t)(112 - e) << 23) | ((m & 0x03FFu) << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
#endif
}

// P5.2: codebook formats decode through a 2 KiB grid table. The GPU stages it
// in shared memory once per block (175-211 Gw/s, P5.0); the CPU would gather
// it per 8 weights (5.5 Gw/s regardless of effort - SVE is 128-bit here and
// the grid exceeds every register table). There is
// deliberately no CPU int8 path, so these tiles must never reach the pool
// while the GPU is available (fl_task_queue -> q[3]).
FL_HD bool fl_fmt_is_codebook(Fmt f) {
    return f == Fmt::IQ2_XXS || f == Fmt::IQ3_XXS;
}

FL_HD uint32_t fl_blk_elems(Fmt f) {
    switch (f) {
        case Fmt::F32:  return 1;
        case Fmt::Q8_0: case Fmt::Q5_0: return 32;
        case Fmt::Q4_K: case Fmt::Q6_K:
        case Fmt::IQ2_XXS: case Fmt::IQ3_XXS: return 256;
    }
    return 1;
}
FL_HD uint32_t fl_blk_bytes(Fmt f) {
    switch (f) {
        case Fmt::F32:  return 4;
        case Fmt::Q8_0: return 34;
        case Fmt::Q5_0: return 22;
        case Fmt::Q4_K: return 144;
        case Fmt::Q6_K: return 210;
        case Fmt::IQ2_XXS: return 66;    // d + 32 u16 indices (2.0625 bpw)
        case Fmt::IQ3_XXS: return 98;    // d + 96 B (64 idx + 32 scale/sign)
    }
    return 4;
}
FL_HD uint64_t fl_row_bytes(Fmt f, uint32_t cols) {
    return (uint64_t)(cols / fl_blk_elems(f)) * fl_blk_bytes(f);
}

// ggml's 6-bit scale/min unpack for q4_K (get_scale_min_k4)
// fl_q4k_scales_all decodes all 8 pairs branchlessly (unrolled); values are
// identical to per-j fl_q4k_scale_min calls.
FL_HD void fl_q4k_scales_all(const uint8_t * s, uint8_t * d8, uint8_t * m8) {
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int j = 0; j < 4; ++j) {
        d8[j] = s[j] & 63;
        m8[j] = s[j + 4] & 63;
    }
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int j = 4; j < 8; ++j) {
        d8[j] = (uint8_t)((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4));
        m8[j] = (uint8_t)((s[j + 4] >>  4) | ((s[j]     >> 6) << 4));
    }
}

FL_HD void fl_q4k_scale_min(int j, const uint8_t * s, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = s[j] & 63;
        *m = s[j + 4] & 63;
    } else {
        *d = (uint8_t)((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4));
        *m = (uint8_t)((s[j + 4] >>  4) | ((s[j - 0] >> 6) << 4));
    }
}

// ---- canonical scalar dequant-dot per row (elementwise ascending) ---------
FL_HD float fl_dot_row_q8_0(const uint8_t * row, const float * x, uint32_t cols) {
    float acc = 0.f;
    const uint32_t nb = cols / 32;
    for (uint32_t b = 0; b < nb; ++b) {
        const blk_q8_0 * B = (const blk_q8_0 *)(row + (size_t)b * 34);
        const float d = fl_half2float(B->d);
        for (int i = 0; i < 32; ++i)
            acc += d * (float)B->qs[i] * x[b * 32 + i];
    }
    return acc;
}

FL_HD float fl_dot_row_q5_0(const uint8_t * row, const float * x, uint32_t cols) {
    float acc = 0.f;
    const uint32_t nb = cols / 32;
    for (uint32_t b = 0; b < nb; ++b) {
        const blk_q5_0 * B = (const blk_q5_0 *)(row + (size_t)b * 22);
        const float d = fl_half2float(B->d);
        uint32_t qh;
        memcpy(&qh, B->qh, 4);
        // ascending element order: elems 0..15 (low nibbles), 16..31 (high)
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh = (uint8_t)(((qh >> j) << 4) & 0x10);
            const int v = (int)((B->qs[j] & 0x0F) | xh) - 16;
            acc += d * (float)v * x[b * 32 + j];
        }
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh = (uint8_t)((qh >> (j + 12)) & 0x10);
            const int v = (int)((B->qs[j] >> 4) | xh) - 16;
            acc += d * (float)v * x[b * 32 + 16 + j];
        }
    }
    return acc;
}

FL_HD float fl_dot_row_q4_K(const uint8_t * row, const float * x, uint32_t cols) {
    float acc = 0.f;
    const uint32_t nb = cols / 256;
    for (uint32_t sb = 0; sb < nb; ++sb) {
        const blk_q4_K * B = (const blk_q4_K *)(row + (size_t)sb * 144);
        const float d    = fl_half2float(B->d);
        const float dmin = fl_half2float(B->dmin);
        const uint8_t * q = B->qs;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, mn;
            fl_q4k_scale_min(is + 0, B->scales, &sc, &mn);
            const float d1 = d * sc, m1 = dmin * mn;
            fl_q4k_scale_min(is + 1, B->scales, &sc, &mn);
            const float d2 = d * sc, m2 = dmin * mn;
            const float * xb = x + sb * 256 + j;
            for (int l = 0; l < 32; ++l)
                acc += (d1 * (float)(q[l] & 0xF) - m1) * xb[l];
            for (int l = 0; l < 32; ++l)
                acc += (d2 * (float)(q[l] >> 4) - m2) * xb[32 + l];
            q += 32;
            is += 2;
        }
    }
    return acc;
}

FL_HD float fl_dot_row_q6_K(const uint8_t * row, const float * x, uint32_t cols) {
    float acc = 0.f;
    const uint32_t nb = cols / 256;
    for (uint32_t sb = 0; sb < nb; ++sb) {
        const blk_q6_K * B = (const blk_q6_K *)(row + (size_t)sb * 210);
        const float d = fl_half2float(B->d);
        const uint8_t * ql = B->ql;
        const uint8_t * qh = B->qh;
        const int8_t  * sc = B->scales;
        const float   * xb = x + sb * 256;
        for (int n = 0; n < 256; n += 128) {
            // ascending element order within the half: l, then l+32, ...
            // matches dequantize_row_q6_K's write order y[l], y[l+32], ...
            // followed by an ascending dot => do columns in ascending order:
            for (int l = 0; l < 32; ++l)
                acc += d * (float)sc[l / 16 + 0]
                     * (float)((int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     * xb[n + l];
            for (int l = 0; l < 32; ++l)
                acc += d * (float)sc[l / 16 + 2]
                     * (float)((int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     * xb[n + l + 32];
            for (int l = 0; l < 32; ++l)
                acc += d * (float)sc[l / 16 + 4]
                     * (float)((int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     * xb[n + l + 64];
            for (int l = 0; l < 32; ++l)
                acc += d * (float)sc[l / 16 + 6]
                     * (float)((int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32)
                     * xb[n + l + 96];
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return acc;
}

// P5.0 codebook references. Element order is already ascending in ggml's own
// dequant (grid entries emit 8 consecutive weights), so a straight ascending
// loop IS the canonical order. Host-side only for reference/testing: Thor's
// CPU is gather-bound on these formats (5.5 Gw/s) and never runs them in the
// engine.
FL_HD float fl_dot_row_iq2_xxs(const uint8_t * row, const float * x, uint32_t cols) {
    float acc = 0.f;
    const uint32_t nb = cols / 256;
    for (uint32_t sb = 0; sb < nb; ++sb) {
        const blk_iq2_xxs * B = (const blk_iq2_xxs *)(row + (size_t)sb * 66);
        const float d = fl_half2float(B->d);
        const float * xb = x + (size_t)sb * 256;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint16_t * q = B->qs + 4 * ib32;
            const uint32_t w0 = (uint32_t)q[0] | ((uint32_t)q[1] << 16);
            const uint32_t w1 = (uint32_t)q[2] | ((uint32_t)q[3] << 16);
            const float db = d * (0.5f + (float)(w1 >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint64_t g = fl_kIq2xxsGrid((w0 >> (8 * l)) & 0xFF);
                const uint8_t  s = fl_kIqSigns((w1 >> (7 * l)) & 127);
                for (int j = 0; j < 8; ++j) {
                    const float v = (float)(uint8_t)(g >> (8 * j));
                    acc += db * (s & fl_kIqSignMask(j) ? -v : v)
                         * xb[ib32 * 32 + l * 8 + j];
                }
            }
        }
    }
    return acc;
}

FL_HD float fl_dot_row_iq3_xxs(const uint8_t * row, const float * x, uint32_t cols) {
    float acc = 0.f;
    const uint32_t nb = cols / 256;
    for (uint32_t sb = 0; sb < nb; ++sb) {
        const blk_iq3_xxs * B = (const blk_iq3_xxs *)(row + (size_t)sb * 98);
        const float d = fl_half2float(B->d);
        const uint8_t * qs = B->qs;              // 64 grid indices
        const uint8_t * ss = B->qs + 64;         // 8 x u32 scale/sign words
        const float * xb = x + (size_t)sb * 256;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t w1 = 0;
            for (int b = 0; b < 4; ++b) w1 |= (uint32_t)ss[4 * ib32 + b] << (8 * b);
            const float db = d * (0.5f + (float)(w1 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint32_t g1 = fl_kIq3xxsGrid(qs[8 * ib32 + 2 * l + 0]);
                const uint32_t g2 = fl_kIq3xxsGrid(qs[8 * ib32 + 2 * l + 1]);
                const uint8_t  s  = fl_kIqSigns((w1 >> (7 * l)) & 127);
                for (int j = 0; j < 4; ++j) {
                    const float v1 = (float)(uint8_t)(g1 >> (8 * j));
                    const float v2 = (float)(uint8_t)(g2 >> (8 * j));
                    acc += db * (s & fl_kIqSignMask(j) ? -v1 : v1)
                         * xb[ib32 * 32 + l * 8 + j];
                    acc += db * (s & fl_kIqSignMask(j + 4) ? -v2 : v2)
                         * xb[ib32 * 32 + l * 8 + j + 4];
                }
            }
        }
    }
    return acc;
}

// ---- q8_1 activation blocks (P1.1 int8-dot path) --------------------------
// Activation vectors carry a quantized appendix at fl_xq_of(x, cols) when
// RuntimeConfig::int8_dots is on: [f32 x cols][pad to 16][blk_q8_1 x cols/32],
// written by an ACT_Q8 task. s = d * sum(qs) is stored for ggml parity; the
// engine's dots recompute integer sums from qs (exact) instead of reading it.
struct blk_q8_1 { uint16_t d; uint16_t s; int8_t qs[32]; };           // 36 B/32
static_assert(sizeof(blk_q8_1) == 36, "q8_1 layout");

FL_HD const blk_q8_1 * fl_xq_of(const void * x, uint32_t cols) {
    return (const blk_q8_1 *)((const uint8_t *)x + (((size_t)cols * 4 + 15) & ~(size_t)15));
}

// ---- q8_K activation superblocks (P1.5, K-format int8 path) ---------------
// Per-256 activation quantization: ONE scale + per-16 integer block sums.
// K-format weight scales fold in the INTEGER domain against these, cutting
// the per-superblock float work from 16 fmaf+cvt (q8_1 appendix) to 1-2 fmaf.
// The appendix layout when int8_dots is on becomes:
//   [f32 x cols][pad16][blk_q8_1 x cols/32][pad16][blk_q8_K x cols/256]
// ACT_Q8 writes BOTH appendices (a same-binary A/B between the q8_1 and q8_K
// paths then needs only a dispatch-mask flip, no graph change). Widths that
// are not 256-multiples never have K-format consumers (K blocks are 256-wide
// by layout), so their truncated q8_K region is dead space, never read.
struct blk_q8_K { float d; int16_t bsums[16]; int8_t qs[256]; };      // 292 B/256
static_assert(sizeof(blk_q8_K) == 292, "q8_K layout");

FL_HD size_t fl_xq_bytes(uint32_t cols) { return ((size_t)cols / 32) * 36; }

FL_HD const blk_q8_K * fl_xqk_of(const void * x, uint32_t cols) {
    const size_t f32   = ((size_t)cols * 4 + 15) & ~(size_t)15;
    const size_t q81   = (fl_xq_bytes(cols) + 15) & ~(size_t)15;
    return (const blk_q8_K *)((const uint8_t *)x + f32 + q81);
}
FL_HD size_t fl_x_alloc_bytes(uint32_t cols) {
    const size_t f32 = ((size_t)cols * 4 + 15) & ~(size_t)15;
    const size_t q81 = (fl_xq_bytes(cols) + 15) & ~(size_t)15;
    return f32 + q81 + ((size_t)cols / 256) * sizeof(blk_q8_K);
}

// round-to-nearest-even float -> half, bit-exact vs CUDA __float2half
FL_HD uint16_t fl_float2half(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t em = u & 0x7FFFFFFFu;
    if (em >= 0x47800000u) {                       // overflow / inf / nan
        return (uint16_t)(sign | (em > 0x7F800000u ? 0x7E00u : 0x7C00u));
    }
    if (em < 0x38800000u) {                        // subnormal / zero
        // half = h * 2^-24 -> h = m * 2^(E-127-23+24) = m >> (126 - E)
        const uint32_t shift = 126u - (em >> 23);
        if (shift > 24) return (uint16_t)sign;
        const uint32_t m = (em & 0x7FFFFFu) | 0x800000u;
        uint32_t h = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((em >> 23) - 112u) << 10 | ((em >> 13) & 0x3FFu);
    const uint32_t rem = em & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return (uint16_t)(sign | h);
}

// canonical q8_1 quantizer: per-32 block, deterministic on host and device
// (roundf + RNE half conversion; both engines produce identical bytes).
FL_HD void fl_quantize_q8_1_block(const float * x, blk_q8_1 * o) {
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) {
        const float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    const float d  = amax / 127.f;
    const float id = amax > 0.f ? 127.f / amax : 0.f;
    int sum = 0;
    for (int i = 0; i < 32; ++i) {
        float q = roundf(x[i] * id);
        if (q >  127.f) q =  127.f;
        if (q < -127.f) q = -127.f;
        o->qs[i] = (int8_t)q;
        sum += (int)q;
    }
    o->d = fl_float2half(d);
    o->s = fl_float2half(d * (float)sum);
}

// canonical q8_K quantizer: per-256 superblock, deterministic on host and
// device (roundf; d stored as f32 so no half roundoff on the scale).
FL_HD void fl_quantize_q8_K_block(const float * x, blk_q8_K * o) {
    float amax = 0.f;
    for (int i = 0; i < 256; ++i) {
        const float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    const float d  = amax / 127.f;
    const float id = amax > 0.f ? 127.f / amax : 0.f;
    for (int g = 0; g < 16; ++g) {
        int bs = 0;
        for (int i = 0; i < 16; ++i) {
            float q = roundf(x[g * 16 + i] * id);
            if (q >  127.f) q =  127.f;
            if (q < -127.f) q = -127.f;
            o->qs[g * 16 + i] = (int8_t)q;
            bs += (int)q;
        }
        o->bsums[g] = (int16_t)bs;
    }
    o->d = d;
}

FL_HD void fl_exec_act_q8(const Task & t) {
    const float * x = (const float *)t.x;
    blk_q8_1 * o = (blk_q8_1 *)t.y;
    const uint32_t nb = t.cols / 32;
    for (uint32_t b = 0; b < nb; ++b)
        fl_quantize_q8_1_block(x + b * 32, o + b);
    // q8_K appendix (P1.5): only complete 256-superblocks; widths that are
    // not 256-multiples have no K-format consumers (see layout comment).
    blk_q8_K * ok = (blk_q8_K *)fl_xqk_of(t.x, t.cols);
    const uint32_t nsb = t.cols / 256;
    for (uint32_t sb = 0; sb < nsb; ++sb)
        fl_quantize_q8_K_block(x + sb * 256, ok + sb);
}

// ---- int8-order scalar reference dots -------------------------------------
// THE accumulation-order contract for the int8 path on CPU (P1.3 revision):
// FOUR independent float accumulators per row, term index t (blocks/groups
// ascending, exactly as before) folds into acc[t & 3] via explicit fmaf,
// final fold (acc0 + acc1) + (acc2 + acc3). Integer partial sums unchanged
// (exact). Rationale: the P1.2 fingerprint (halving SDOT work at constant
// fold chain -> exact parity) proved the serially-dependent single-acc fmaf
// chain was the limiter; 4 chains expose ILP without changing any product
// term. The NEON SDOT kernels in kernels.h reproduce this order bitwise.
// GPU int8 kernels partition blocks across lanes -> engine-level tolerance.
inline int fl_idot32(const int8_t * a, const int8_t * b) {
    int s = 0;
    for (int i = 0; i < 32; ++i) s += (int)a[i] * (int)b[i];
    return s;
}

inline float fl_fold4(const float * a) {
    return (a[0] + a[1]) + (a[2] + a[3]);
}

inline float fl_dot_i8_q8_0(const uint8_t * row, const blk_q8_1 * xq, uint32_t cols) {
    const uint32_t nb = cols / 32;
    float a4[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t b = 0; b < nb; ++b) {
        const blk_q8_0 * W = (const blk_q8_0 *)(row + (size_t)b * 34);
        a4[b & 3] = fmaf(fl_half2float(W->d) * fl_half2float(xq[b].d),
                         (float)fl_idot32(W->qs, xq[b].qs), a4[b & 3]);
    }
    return fl_fold4(a4);
}

inline float fl_dot_i8_q5_0(const uint8_t * row, const blk_q8_1 * xq, uint32_t cols) {
    const uint32_t nb = cols / 32;
    float a4[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t b = 0; b < nb; ++b) {
        const blk_q5_0 * W = (const blk_q5_0 *)(row + (size_t)b * 22);
        uint32_t qh;
        memcpy(&qh, W->qh, 4);
        int idot = 0, isum = 0;
        for (int j = 0; j < 16; ++j) {
            const int lo = (int)((W->qs[j] & 0x0F) | (((qh >> j) << 4) & 0x10));
            const int hi = (int)((W->qs[j] >> 4)   | ((qh >> (j + 12)) & 0x10));
            idot += lo * xq[b].qs[j] + hi * xq[b].qs[16 + j];
        }
        for (int i = 0; i < 32; ++i) isum += xq[b].qs[i];
        a4[b & 3] = fmaf(fl_half2float(W->d) * fl_half2float(xq[b].d),
                         (float)(idot - 16 * isum), a4[b & 3]);
    }
    return fl_fold4(a4);
}

inline float fl_dot_i8_q4_K(const uint8_t * row, const blk_q8_1 * xq, uint32_t cols) {
    const uint32_t nsb = cols / 256;
    float a4[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        const blk_q4_K * W = (const blk_q4_K *)(row + (size_t)sb * 144);
        const float d = fl_half2float(W->d), dmin = fl_half2float(W->dmin);
        for (int g = 0; g < 8; ++g) {                 // 32-elem groups
            uint8_t sc, mn;
            fl_q4k_scale_min(g, W->scales, &sc, &mn);
            const uint8_t * q = W->qs + (g / 2) * 32; // 64-elem chunk base
            const int8_t * xg = xq[sb * 8 + g].qs;
            int idot = 0, isum = 0;
            for (int i = 0; i < 32; ++i) {
                const int w4 = (g & 1) ? (q[i] >> 4) : (q[i] & 0x0F);
                idot += w4 * xg[i];
                isum += xg[i];
            }
            const float dx = fl_half2float(xq[sb * 8 + g].d);
            // both terms of group g chain into acc[g & 3]
            a4[g & 3] = fmaf(d * sc * dx,       (float)idot, a4[g & 3]);
            a4[g & 3] = fmaf(-(dmin * mn) * dx, (float)isum, a4[g & 3]);
        }
    }
    return fl_fold4(a4);
}

inline float fl_dot_i8_q6_K(const uint8_t * row, const blk_q8_1 * xq, uint32_t cols) {
    const uint32_t nsb = cols / 256;
    float a4[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        const blk_q6_K * W = (const blk_q6_K *)(row + (size_t)sb * 210);
        const float d = fl_half2float(W->d);
        for (int h = 0; h < 2; ++h) {                 // 128-elem halves
            const uint8_t * ql = W->ql + h * 64;
            const uint8_t * qh = W->qh + h * 32;
            const int8_t  * sc = W->scales + h * 8;
            for (int s = 0; s < 4; ++s) {             // 32-elem spans
                const int8_t * xg = xq[sb * 8 + h * 4 + s].qs;
                const float dx = fl_half2float(xq[sb * 8 + h * 4 + s].d);
                for (int k16 = 0; k16 < 2; ++k16) {   // 16-elem scale groups
                    int idot = 0, isum = 0;
                    for (int i = 0; i < 16; ++i) {
                        const int e = k16 * 16 + i;   // elem within span
                        const uint8_t lb = ql[(s & 1) * 32 + e];
                        const int q6 = (s < 2 ? (lb & 0x0F) : (lb >> 4))
                                     | (((qh[e] >> (2 * s)) & 3) << 4);
                        idot += q6 * xg[e];
                        isum += xg[e];
                    }
                    const int t = (int)sc[2 * s + k16] * (idot - 32 * isum);
                    const int ti = h * 8 + s * 2 + k16;   // term idx in sb
                    a4[ti & 3] = fmaf(d * dx, (float)t, a4[ti & 3]);
                }
            }
        }
    }
    return fl_fold4(a4);
}

// ---- q8_K-order scalar reference dots (P1.5, K formats) -------------------
// THE accumulation-order contract for the i8k path: per superblock all
// integer work is EXACT (int32); the only float terms are per-superblock
// folds into acc[sb & 3] via explicit fmaf, final fold fl_fold4:
//   q4_K: acc[sb&3] = fmaf(d*dk,  iacc_sb, .); acc[sb&3] = fmaf(-dmin*dk, imins_sb, .)
//   q6_K: acc[sb&3] = fmaf(d*dk,  iacc_sb, .)
// Overflow audit (worst case): q4_K iacc <= 8*63*(32*15*127) ~ 3.1e7,
// imins <= 8*63*8128 ~ 4.1e6; q6_K per-sb <= 16*127*(16*63*127 + 32*2032)
// ~ 3.9e8 -- all inside int32. int->float conversion at ~1e7-1e8 rounds to
// 24-bit mantissa identically on both engines (same value -> bitwise safe).
inline float fl_dot_i8k_q4_K(const uint8_t * row, const blk_q8_K * xk, uint32_t cols) {
    const uint32_t nsb = cols / 256;
    float a4[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        const blk_q4_K * W = (const blk_q4_K *)(row + (size_t)sb * 144);
        const float d = fl_half2float(W->d), dmin = fl_half2float(W->dmin);
        const float dk = xk[sb].d;
        int32_t iacc = 0, imins = 0;
        for (int g = 0; g < 8; ++g) {
            uint8_t sc, mn;
            fl_q4k_scale_min(g, W->scales, &sc, &mn);
            const uint8_t * q = W->qs + (g / 2) * 32;
            const int8_t * xg = xk[sb].qs + g * 32;
            int idot = 0;
            for (int i = 0; i < 32; ++i)
                idot += ((g & 1) ? (q[i] >> 4) : (q[i] & 0x0F)) * xg[i];
            iacc  += (int)sc * idot;
            imins += (int)mn * ((int)xk[sb].bsums[2 * g] + (int)xk[sb].bsums[2 * g + 1]);
        }
        a4[sb & 3] = fmaf(d * dk,     (float)iacc,  a4[sb & 3]);
        a4[sb & 3] = fmaf(-(dmin * dk), (float)imins, a4[sb & 3]);
    }
    return fl_fold4(a4);
}

inline float fl_dot_i8k_q6_K(const uint8_t * row, const blk_q8_K * xk, uint32_t cols) {
    const uint32_t nsb = cols / 256;
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
                    int idot = 0;
                    for (int i = 0; i < 16; ++i) {
                        const int e = k16 * 16 + i;
                        const uint8_t lb = ql[(s & 1) * 32 + e];
                        const int q6 = (s < 2 ? (lb & 0x0F) : (lb >> 4))
                                     | (((qh[e] >> (2 * s)) & 3) << 4);
                        idot += q6 * xg[e];
                    }
                    const int bs = (int)xk[sb].bsums[2 * (h * 4 + s) + k16];
                    iacc += (int)sc[2 * s + k16] * (idot - 32 * bs);
                }
            }
        }
        a4[sb & 3] = fmaf(d * dk, (float)iacc, a4[sb & 3]);
    }
    return fl_fold4(a4);
}

inline float fl_dot_i8k_any(Fmt f, const void * wrow, const blk_q8_K * xk, uint32_t cols) {
    const uint8_t * r = (const uint8_t *)wrow;
    switch (f) {
        case Fmt::Q4_K: return fl_dot_i8k_q4_K(r, xk, cols);
        case Fmt::Q6_K: return fl_dot_i8k_q6_K(r, xk, cols);
        default: break;                        // i8k covers K formats only
    }
    return 0.f;
}

inline float fl_dot_i8_any(Fmt f, const void * wrow, const blk_q8_1 * xq, uint32_t cols) {
    const uint8_t * r = (const uint8_t *)wrow;
    switch (f) {
        case Fmt::Q8_0: return fl_dot_i8_q8_0(r, xq, cols);
        case Fmt::Q5_0: return fl_dot_i8_q5_0(r, xq, cols);
        case Fmt::Q4_K: return fl_dot_i8_q4_K(r, xq, cols);
        case Fmt::Q6_K: return fl_dot_i8_q6_K(r, xq, cols);
        // P5.0: codebook formats have NO CPU int8 path by design (gather-bound
        // at 5.5 Gw/s) and no int8 fallback is
        // possible here - this entry point only receives q8_1 blocks, not the
        // floats the canonical reference needs. Contract: codebook tiles are
        // GPU-affine and must never be routed to the CPU pool; the graph
        // builder that emits them (P5.1) owns enforcing that.
        case Fmt::IQ2_XXS: case Fmt::IQ3_XXS: break;
        case Fmt::F32:  break;                        // no int8 path for f32
    }
    return 0.f;
}

FL_HD float fl_dot_row_any(Fmt f, const void * wrow, const float * x, uint32_t cols) {
    const uint8_t * r = (const uint8_t *)wrow;
    switch (f) {
        case Fmt::F32: {
            const float * row = (const float *)wrow;
            float acc = 0.f;
            for (uint32_t c = 0; c < cols; ++c) acc += row[c] * x[c];
            return acc;
        }
        case Fmt::Q8_0: return fl_dot_row_q8_0(r, x, cols);
        case Fmt::Q5_0: return fl_dot_row_q5_0(r, x, cols);
        case Fmt::Q4_K: return fl_dot_row_q4_K(r, x, cols);
        case Fmt::Q6_K: return fl_dot_row_q6_K(r, x, cols);
        case Fmt::IQ2_XXS: return fl_dot_row_iq2_xxs(r, x, cols);
        case Fmt::IQ3_XXS: return fl_dot_row_iq3_xxs(r, x, cols);
    }
    return 0.f;
}

} // namespace fastllm
