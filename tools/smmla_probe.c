// P4.2 probe: does i8mm SMMLA reduce instructions/weight in the multi-column
// CPU dot, where SDOT currently runs at IPC 3.67 (P4.1: instruction-bound)?
//
// SMMLA does 32 int8 MACs per instruction (2x8 * 8x2 -> 2x2) against SDOT's 16,
// but demands a 2x8 interleaved operand layout. The question is whether that 2x
// survives the interleave cost. Three variants isolate the two effects:
//
//   sdot_blk    baseline: today's kernel shape, blk_q8_0 packing, 1 row x nx col
//   smmla_rt    SMMLA with vcombine at runtime (no layout change)
//   smmla_pre   SMMLA on pre-interleaved weights AND activations (repack at load)
//   sdot_pln    SDOT on the planar layout, to separate layout from instruction
//
// All four compute the same logical result: 2*R rows x nx cols x K weights,
// per-32-block scales. Outputs are compared so a variant cannot win by doing
// less work. Report is instructions/(row*col*weight) under perf stat.
//
// Build: gcc -O3 -mcpu=native -o smmla_probe smmla_probe.c -lm
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define K       2048            // weights per row (V2-Lite hidden)
#define NBLK    (K / 32)        // 32-element scale blocks
#define RPAIR   384             // row pairs -> 768 rows, the adopted rowsz
#define MAXNX   8

typedef struct { uint16_t d; int8_t qs[32]; } blk_q8_0_t;   // 34 B, as shipped
typedef struct { uint16_t d; uint16_t s; int8_t qs[32]; } blk_q8_1_t;

static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15), e = (uint32_t)((h >> 10) & 0x1F);
    uint32_t m = (uint32_t)(h & 0x3FF), b;
    if (e == 0) { if (!m) { b = s << 31; } else {
            e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; }
            m &= 0x3FF; b = (s << 31) | (e << 23) | (m << 13); } }
    else if (e == 31) b = (s << 31) | 0x7F800000u | (m << 13);
    else b = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &b, 4); return f;
}
static uint16_t f2h(float f) {   // round-to-nearest-even, probe-grade
    uint32_t b; memcpy(&b, &f, 4);
    uint32_t s = (b >> 16) & 0x8000; int32_t e = (int32_t)((b >> 23) & 0xFF) - 112;
    uint32_t m = b & 0x7FFFFF;
    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7C00);
    uint32_t h = s | ((uint32_t)e << 10) | (m >> 13);
    if ((m & 0x1FFF) > 0x1000) h++;
    return (uint16_t)h;
}
static inline float fold4(const float a[4]) {
    return (a[0] + a[1]) + (a[2] + a[3]);   // matches fl_fold4
}

// ---- buffers ---------------------------------------------------------------
static blk_q8_0_t *Wblk;                 // [2*RPAIR][NBLK]  packed weights
static blk_q8_1_t *Xblk;                 // [MAXNX][NBLK]    packed activations
static int8_t     *Wpre;                 // pre-interleaved weights, row pairs
static int8_t     *Xpre;                 // pre-interleaved activations, col pairs
static float      *Wsc, *Xsc;            // planar scales
static float       out_ref[2 * RPAIR * MAXNX];
static float       out_got[2 * RPAIR * MAXNX];

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static int8_t rnd8(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (int8_t)((int)(rng >> 24) % 96 - 48);
}

static void build(void) {
    Wblk = aligned_alloc(64, (size_t)2 * RPAIR * NBLK * sizeof(blk_q8_0_t));
    Xblk = aligned_alloc(64, (size_t)MAXNX * NBLK * sizeof(blk_q8_1_t));
    Wpre = aligned_alloc(64, (size_t)RPAIR * K * 2);
    Xpre = aligned_alloc(64, (size_t)(MAXNX / 2) * K * 2);
    Wsc  = aligned_alloc(64, (size_t)2 * RPAIR * NBLK * sizeof(float));
    Xsc  = aligned_alloc(64, (size_t)MAXNX * NBLK * sizeof(float));
    for (int r = 0; r < 2 * RPAIR; ++r)
        for (int b = 0; b < NBLK; ++b) {
            blk_q8_0_t *W = &Wblk[(size_t)r * NBLK + b];
            W->d = f2h(0.01f + 0.0001f * (float)((r + b) % 17));
            for (int i = 0; i < 32; ++i) W->qs[i] = rnd8();
            Wsc[(size_t)r * NBLK + b] = h2f(W->d);
        }
    for (int p = 0; p < MAXNX; ++p)
        for (int b = 0; b < NBLK; ++b) {
            blk_q8_1_t *X = &Xblk[(size_t)p * NBLK + b];
            X->d = f2h(0.02f + 0.0001f * (float)((p * 3 + b) % 13));
            X->s = 0;
            for (int i = 0; i < 32; ++i) X->qs[i] = rnd8();
            Xsc[(size_t)p * NBLK + b] = h2f(X->d);
        }
    // Pre-interleave: [r0[0..7], r1[0..7], r0[8..15], r1[8..15], ...]
    for (int rp = 0; rp < RPAIR; ++rp) {
        int8_t *dst = Wpre + (size_t)rp * K * 2;
        for (int c8 = 0; c8 < K / 8; ++c8)
            for (int h = 0; h < 2; ++h) {
                const blk_q8_0_t *W = &Wblk[(size_t)(2 * rp + h) * NBLK
                                            + (c8 * 8) / 32];
                memcpy(dst + (size_t)c8 * 16 + h * 8,
                       W->qs + ((c8 * 8) % 32), 8);
            }
    }
    for (int cp = 0; cp < MAXNX / 2; ++cp) {
        int8_t *dst = Xpre + (size_t)cp * K * 2;
        for (int c8 = 0; c8 < K / 8; ++c8)
            for (int h = 0; h < 2; ++h) {
                const blk_q8_1_t *X = &Xblk[(size_t)(2 * cp + h) * NBLK
                                            + (c8 * 8) / 32];
                memcpy(dst + (size_t)c8 * 16 + h * 8,
                       X->qs + ((c8 * 8) % 32), 8);
            }
    }
}

// ---- variant A: today's shape (1 row x nx cols, blk packing, SDOT) ---------
static void run_sdot_blk(int nx, float *out) {
    for (int r = 0; r < 2 * RPAIR; ++r) {
        const blk_q8_0_t *Wr = &Wblk[(size_t)r * NBLK];
        float a4[MAXNX][4];
        for (int p = 0; p < nx; ++p) a4[p][0] = a4[p][1] = a4[p][2] = a4[p][3] = 0.f;
        for (int b = 0; b < NBLK; ++b) {
            const int8x16_t w0 = vld1q_s8(Wr[b].qs), w1 = vld1q_s8(Wr[b].qs + 16);
            const float dw = h2f(Wr[b].d);
            for (int p = 0; p < nx; ++p) {
                const blk_q8_1_t *X = &Xblk[(size_t)p * NBLK + b];
                const int8x16_t x0 = vld1q_s8(X->qs), x1 = vld1q_s8(X->qs + 16);
                const int32x4_t s = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0),
                                              w1, x1);
                a4[p][b & 3] = fmaf(dw * h2f(X->d), (float)vaddvq_s32(s),
                                    a4[p][b & 3]);
            }
        }
        for (int p = 0; p < nx; ++p) out[(size_t)r * MAXNX + p] = fold4(a4[p]);
    }
}

// ---- variant B: SDOT on planar layout (isolates layout from instruction) ---
static void run_sdot_pln(int nx, float *out) {
    for (int rp = 0; rp < RPAIR; ++rp)
        for (int h = 0; h < 2; ++h) {
            const int r = 2 * rp + h;
            const int8_t *Wr = Wpre + (size_t)rp * K * 2;   // interleaved src
            float a4[MAXNX][4];
            for (int p = 0; p < nx; ++p)
                a4[p][0] = a4[p][1] = a4[p][2] = a4[p][3] = 0.f;
            for (int b = 0; b < NBLK; ++b) {
                // gather this row's 32 bytes out of the interleaved pairs
                int8_t wtmp[32];
                for (int c8 = 0; c8 < 4; ++c8)
                    memcpy(wtmp + c8 * 8,
                           Wr + (size_t)(b * 4 + c8) * 16 + h * 8, 8);
                const int8x16_t w0 = vld1q_s8(wtmp), w1 = vld1q_s8(wtmp + 16);
                const float dw = Wsc[(size_t)r * NBLK + b];
                for (int p = 0; p < nx; ++p) {
                    const blk_q8_1_t *X = &Xblk[(size_t)p * NBLK + b];
                    const int8x16_t x0 = vld1q_s8(X->qs), x1 = vld1q_s8(X->qs + 16);
                    const int32x4_t s = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0),
                                                  w1, x1);
                    a4[p][b & 3] = fmaf(dw * Xsc[(size_t)p * NBLK + b],
                                        (float)vaddvq_s32(s), a4[p][b & 3]);
                }
            }
            for (int p = 0; p < nx; ++p) out[(size_t)r * MAXNX + p] = fold4(a4[p]);
        }
}

// ---- variant C: SMMLA, runtime interleave (no layout change) ---------------
static void run_smmla_rt(int nx, float *out) {
    for (int rp = 0; rp < RPAIR; ++rp) {
        const blk_q8_0_t *W0 = &Wblk[(size_t)(2 * rp) * NBLK];
        const blk_q8_0_t *W1 = &Wblk[(size_t)(2 * rp + 1) * NBLK];
        float a4[2][MAXNX][4];
        for (int h = 0; h < 2; ++h)
            for (int p = 0; p < nx; ++p)
                a4[h][p][0] = a4[h][p][1] = a4[h][p][2] = a4[h][p][3] = 0.f;
        for (int b = 0; b < NBLK; ++b) {
            const int8x16_t wa = vld1q_s8(W0[b].qs), wb = vld1q_s8(W0[b].qs + 16);
            const int8x16_t wc = vld1q_s8(W1[b].qs), wd = vld1q_s8(W1[b].qs + 16);
            // A operands: [r0 8B ; r1 8B] for each of the 4 8-element chunks
            const int8x16_t A0 = vcombine_s8(vget_low_s8(wa),  vget_low_s8(wc));
            const int8x16_t A1 = vcombine_s8(vget_high_s8(wa), vget_high_s8(wc));
            const int8x16_t A2 = vcombine_s8(vget_low_s8(wb),  vget_low_s8(wd));
            const int8x16_t A3 = vcombine_s8(vget_high_s8(wb), vget_high_s8(wd));
            const float dw0 = h2f(W0[b].d), dw1 = h2f(W1[b].d);
            for (int cp = 0; cp * 2 < nx; ++cp) {
                const blk_q8_1_t *Xa = &Xblk[(size_t)(2 * cp) * NBLK + b];
                const blk_q8_1_t *Xb = &Xblk[(size_t)(2 * cp + 1) * NBLK + b];
                const int8x16_t xa = vld1q_s8(Xa->qs), xb = vld1q_s8(Xa->qs + 16);
                const int8x16_t xc = vld1q_s8(Xb->qs), xd = vld1q_s8(Xb->qs + 16);
                const int8x16_t B0 = vcombine_s8(vget_low_s8(xa),  vget_low_s8(xc));
                const int8x16_t B1 = vcombine_s8(vget_high_s8(xa), vget_high_s8(xc));
                const int8x16_t B2 = vcombine_s8(vget_low_s8(xb),  vget_low_s8(xd));
                const int8x16_t B3 = vcombine_s8(vget_high_s8(xb), vget_high_s8(xd));
                int32x4_t acc = vdupq_n_s32(0);
                acc = vmmlaq_s32(acc, A0, B0);
                acc = vmmlaq_s32(acc, A1, B1);
                acc = vmmlaq_s32(acc, A2, B2);
                acc = vmmlaq_s32(acc, A3, B3);
                // acc = {r0c0, r0c1, r1c0, r1c1}
                const float dxa = h2f(Xa->d), dxb = h2f(Xb->d);
                a4[0][2*cp  ][b&3] = fmaf(dw0*dxa, (float)vgetq_lane_s32(acc,0),
                                          a4[0][2*cp  ][b&3]);
                a4[0][2*cp+1][b&3] = fmaf(dw0*dxb, (float)vgetq_lane_s32(acc,1),
                                          a4[0][2*cp+1][b&3]);
                a4[1][2*cp  ][b&3] = fmaf(dw1*dxa, (float)vgetq_lane_s32(acc,2),
                                          a4[1][2*cp  ][b&3]);
                a4[1][2*cp+1][b&3] = fmaf(dw1*dxb, (float)vgetq_lane_s32(acc,3),
                                          a4[1][2*cp+1][b&3]);
            }
        }
        for (int h = 0; h < 2; ++h)
            for (int p = 0; p < nx; ++p)
                out[(size_t)(2 * rp + h) * MAXNX + p] = fold4(a4[h][p]);
    }
}

// ---- variant D: SMMLA on pre-interleaved weights AND activations ----------
static void run_smmla_pre(int nx, float *out) {
    for (int rp = 0; rp < RPAIR; ++rp) {
        const int8_t *Wr = Wpre + (size_t)rp * K * 2;
        float a4[2][MAXNX][4];
        for (int h = 0; h < 2; ++h)
            for (int p = 0; p < nx; ++p)
                a4[h][p][0] = a4[h][p][1] = a4[h][p][2] = a4[h][p][3] = 0.f;
        for (int b = 0; b < NBLK; ++b) {
            const int8x16_t A0 = vld1q_s8(Wr + (size_t)(b * 4 + 0) * 16);
            const int8x16_t A1 = vld1q_s8(Wr + (size_t)(b * 4 + 1) * 16);
            const int8x16_t A2 = vld1q_s8(Wr + (size_t)(b * 4 + 2) * 16);
            const int8x16_t A3 = vld1q_s8(Wr + (size_t)(b * 4 + 3) * 16);
            const float dw0 = Wsc[(size_t)(2 * rp)     * NBLK + b];
            const float dw1 = Wsc[(size_t)(2 * rp + 1) * NBLK + b];
            for (int cp = 0; cp * 2 < nx; ++cp) {
                const int8_t *Xc = Xpre + (size_t)cp * K * 2;
                int32x4_t acc = vdupq_n_s32(0);
                acc = vmmlaq_s32(acc, A0, vld1q_s8(Xc + (size_t)(b*4+0) * 16));
                acc = vmmlaq_s32(acc, A1, vld1q_s8(Xc + (size_t)(b*4+1) * 16));
                acc = vmmlaq_s32(acc, A2, vld1q_s8(Xc + (size_t)(b*4+2) * 16));
                acc = vmmlaq_s32(acc, A3, vld1q_s8(Xc + (size_t)(b*4+3) * 16));
                const float dxa = Xsc[(size_t)(2*cp)   * NBLK + b];
                const float dxb = Xsc[(size_t)(2*cp+1) * NBLK + b];
                a4[0][2*cp  ][b&3] = fmaf(dw0*dxa, (float)vgetq_lane_s32(acc,0),
                                          a4[0][2*cp  ][b&3]);
                a4[0][2*cp+1][b&3] = fmaf(dw0*dxb, (float)vgetq_lane_s32(acc,1),
                                          a4[0][2*cp+1][b&3]);
                a4[1][2*cp  ][b&3] = fmaf(dw1*dxa, (float)vgetq_lane_s32(acc,2),
                                          a4[1][2*cp  ][b&3]);
                a4[1][2*cp+1][b&3] = fmaf(dw1*dxb, (float)vgetq_lane_s32(acc,3),
                                          a4[1][2*cp+1][b&3]);
            }
        }
        for (int h = 0; h < 2; ++h)
            for (int p = 0; p < nx; ++p)
                out[(size_t)(2 * rp + h) * MAXNX + p] = fold4(a4[h][p]);
    }
}

// ---- variant E: activations interleaved ONCE per tile into scratch, weights
// combined at runtime. This is the design that needs NO persistent layout
// change: the B-interleave is amortized over t.rows (768), and the A-combine
// is amortized over nx columns. If it lands near smmla_pre, the implementation
// is a scratch buffer + kernel rather than a shadow/appendix re-layout.
static void run_smmla_scratch(int nx, float *out) {
    for (int rp = 0; rp < RPAIR; ++rp) {
        const blk_q8_0_t *W0 = &Wblk[(size_t)(2 * rp) * NBLK];
        const blk_q8_0_t *W1 = &Wblk[(size_t)(2 * rp + 1) * NBLK];
        float a4[2][MAXNX][4];
        for (int h = 0; h < 2; ++h)
            for (int p = 0; p < nx; ++p)
                a4[h][p][0] = a4[h][p][1] = a4[h][p][2] = a4[h][p][3] = 0.f;
        for (int b = 0; b < NBLK; ++b) {
            const int8x16_t wa = vld1q_s8(W0[b].qs), wb = vld1q_s8(W0[b].qs + 16);
            const int8x16_t wc = vld1q_s8(W1[b].qs), wd = vld1q_s8(W1[b].qs + 16);
            const int8x16_t A0 = vcombine_s8(vget_low_s8(wa),  vget_low_s8(wc));
            const int8x16_t A1 = vcombine_s8(vget_high_s8(wa), vget_high_s8(wc));
            const int8x16_t A2 = vcombine_s8(vget_low_s8(wb),  vget_low_s8(wd));
            const int8x16_t A3 = vcombine_s8(vget_high_s8(wb), vget_high_s8(wd));
            const float dw0 = h2f(W0[b].d), dw1 = h2f(W1[b].d);
            for (int cp = 0; cp * 2 < nx; ++cp) {
                const int8_t *Xc = Xpre + (size_t)cp * K * 2;   // scratch, prebuilt
                int32x4_t acc = vdupq_n_s32(0);
                acc = vmmlaq_s32(acc, A0, vld1q_s8(Xc + (size_t)(b*4+0) * 16));
                acc = vmmlaq_s32(acc, A1, vld1q_s8(Xc + (size_t)(b*4+1) * 16));
                acc = vmmlaq_s32(acc, A2, vld1q_s8(Xc + (size_t)(b*4+2) * 16));
                acc = vmmlaq_s32(acc, A3, vld1q_s8(Xc + (size_t)(b*4+3) * 16));
                const float dxa = Xsc[(size_t)(2*cp)   * NBLK + b];
                const float dxb = Xsc[(size_t)(2*cp+1) * NBLK + b];
                a4[0][2*cp  ][b&3] = fmaf(dw0*dxa, (float)vgetq_lane_s32(acc,0),
                                          a4[0][2*cp  ][b&3]);
                a4[0][2*cp+1][b&3] = fmaf(dw0*dxb, (float)vgetq_lane_s32(acc,1),
                                          a4[0][2*cp+1][b&3]);
                a4[1][2*cp  ][b&3] = fmaf(dw1*dxa, (float)vgetq_lane_s32(acc,2),
                                          a4[1][2*cp  ][b&3]);
                a4[1][2*cp+1][b&3] = fmaf(dw1*dxb, (float)vgetq_lane_s32(acc,3),
                                          a4[1][2*cp+1][b&3]);
            }
        }
        for (int h = 0; h < 2; ++h)
            for (int p = 0; p < nx; ++p)
                out[(size_t)(2 * rp + h) * MAXNX + p] = fold4(a4[h][p]);
    }
}

// ---- variant F: SMMLA on pre-interleaved weights at nx=1 (P4.4 gate) ------
// The engine's expert tiles are SINGLE-COLUMN today (P4.2/P4.3: grouping was
// rejected), so this is the shape that actually occurs for 41.8% of expert
// work. MMLA pairs weight ROWS, which the tile supplies at any nx - but its
// B operand holds TWO activation columns, and with only one column available
// the second half must be a duplicate, so useful MACs are 2 rows x 1 col:
// half the instruction's capacity. This measures whether what remains (no
// vaddvq horizontal reduce, one A-load pair per two rows, one fold pair)
// still beats SDOT at the width the engine really sees.
static void run_smmla_pre1(int nx, float *out) {
    (void)nx;                                    // nx == 1 by construction
    for (int rp = 0; rp < RPAIR; ++rp) {
        const int8_t *Wr = Wpre + (size_t)rp * K * 2;
        float a4[2][4];
        for (int h = 0; h < 2; ++h)
            a4[h][0] = a4[h][1] = a4[h][2] = a4[h][3] = 0.f;
        for (int b = 0; b < NBLK; ++b) {
            const int8x16_t A0 = vld1q_s8(Wr + (size_t)(b * 4 + 0) * 16);
            const int8x16_t A1 = vld1q_s8(Wr + (size_t)(b * 4 + 1) * 16);
            const int8x16_t A2 = vld1q_s8(Wr + (size_t)(b * 4 + 2) * 16);
            const int8x16_t A3 = vld1q_s8(Wr + (size_t)(b * 4 + 3) * 16);
            const float dw0 = Wsc[(size_t)(2 * rp)     * NBLK + b];
            const float dw1 = Wsc[(size_t)(2 * rp + 1) * NBLK + b];
            // B halves are the SAME column: lanes 1,3 are wasted by design.
            const blk_q8_1_t *X = &Xblk[b];
            const int8x8_t x0 = vld1_s8(X->qs),      x1 = vld1_s8(X->qs + 8);
            const int8x8_t x2 = vld1_s8(X->qs + 16), x3 = vld1_s8(X->qs + 24);
            int32x4_t acc = vdupq_n_s32(0);
            acc = vmmlaq_s32(acc, A0, vcombine_s8(x0, x0));
            acc = vmmlaq_s32(acc, A1, vcombine_s8(x1, x1));
            acc = vmmlaq_s32(acc, A2, vcombine_s8(x2, x2));
            acc = vmmlaq_s32(acc, A3, vcombine_s8(x3, x3));
            const float dx = Xsc[b];
            a4[0][b & 3] = fmaf(dw0 * dx, (float)vgetq_lane_s32(acc, 0),
                                a4[0][b & 3]);
            a4[1][b & 3] = fmaf(dw1 * dx, (float)vgetq_lane_s32(acc, 2),
                                a4[1][b & 3]);
        }
        for (int h = 0; h < 2; ++h)
            out[(size_t)(2 * rp + h) * MAXNX] = fold4(a4[h]);
    }
}

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "all";
    const int nx     = argc > 2 ? atoi(argv[2]) : 8;
    const int iters  = argc > 3 ? atoi(argv[3]) : 20;
    // nx=1 is the shape the engine actually emits for expert tiles (P4.4).
    // The cp-based SMMLA variants cannot express it (their B operand needs two
    // distinct columns), so only sdot_* and smmla_pre1 run there.
    if (nx < 1 || nx > MAXNX || (nx > 1 && (nx & 1))) {
        fprintf(stderr, "nx must be 1 or even 2..8\n"); return 2;
    }
    build();

    // Reference + correctness: every variant must match sdot_blk.
    run_sdot_blk(nx, out_ref);
    struct { const char *name; void (*fn)(int, float *); } V[] = {
        { "sdot_blk",  run_sdot_blk  },
        { "sdot_pln",  run_sdot_pln  },
        { "smmla_rt",  run_smmla_rt  },
        { "smmla_pre", run_smmla_pre },
        { "smmla_scr", run_smmla_scratch },
        { "smmla_pre1", run_smmla_pre1 },
    };
    const int NV = 6;
    const int v_lo = (nx == 1) ? 0 : 0, v_hi = (nx == 1) ? NV : NV - 1;
    for (int v = v_lo; v < v_hi; ++v) {
        if (nx == 1 && (v == 2 || v == 3 || v == 4)) continue;   // need nx>=2
        memset(out_got, 0, sizeof(out_got));
        V[v].fn(nx, out_got);
        double worst = 0;
        for (int r = 0; r < 2 * RPAIR; ++r)
            for (int p = 0; p < nx; ++p) {
                const size_t i = (size_t)r * MAXNX + p;
                const double den = fabs((double)out_ref[i]) + 1e-3;
                const double e = fabs((double)out_got[i] - (double)out_ref[i]) / den;
                if (e > worst) worst = e;
            }
        if (strcmp(mode, "all") == 0 || strcmp(mode, "check") == 0)
            printf("check %-10s max_rel %.3e\n", V[v].name, worst);
        if (worst > 1e-5) { fprintf(stderr, "MISMATCH %s\n", V[v].name); return 3; }
    }
    if (strcmp(mode, "check") == 0) return 0;

    const double macs = (double)2 * RPAIR * nx * K * iters;
    for (int v = v_lo; v < v_hi; ++v) {
        if (nx == 1 && (v == 2 || v == 3 || v == 4)) continue;   // need nx>=2
        if (strcmp(mode, "all") && strcmp(mode, V[v].name)) continue;
        V[v].fn(nx, out_got);                       // warm
        const double t0 = now_s();
        for (int i = 0; i < iters; ++i) V[v].fn(nx, out_got);
        const double dt = now_s() - t0;
        printf("%-10s nx=%d %8.3f ms  %7.2f Gmac/s  (weights %.3e)\n",
               V[v].name, nx, dt * 1e3, macs / dt * 1e-9, macs);
    }
    return 0;
}
