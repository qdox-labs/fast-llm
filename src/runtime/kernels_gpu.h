// Device STREAM_DOT kernels, shared by the persistent executor (gpu_exec.cu)
// and the standalone membench tool. CUDA-only header.
//
// Shapes (P0.9/P1.0): every fast path is warp-per-row with a shfl-tree reduce
// - lane-coalesced loads keep DRAM sectors near-perfect (the P0.9 fix; the
// row-per-thread shape over-fetched ~8x). Quant paths are lane-per-element
// within a block (block32 formats) or lane-per-byte within a superblock
// (k-quants): all 32 lanes read a contiguous 16-32 B window per step.
// Deterministic per-engine order; NOT bitwise vs canonical (tolerance gate).
#pragma once
#include "quants.h"

namespace fastllm {

// P8.9: superblock unroll depth for the q8_K-appendix q4_K kernel (the
// shipped bs=1 path). 1 = the original one-superblock-in-flight loop; >1
// hoists U superblocks of payload loads before any consumer. Compile-time so
// the A/B costs nothing at runtime and the shipped path is exactly
// recoverable with -DFASTLLM_Q4K_UNROLL=1.
#ifndef FASTLLM_Q4K_UNROLL
#define FASTLLM_Q4K_UNROLL 1
#endif

__device__ __forceinline__ float warp_reduce_add(float acc) {
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc;
}

// Shared byte/word helpers (used by both the dequant and int8 paths).
__device__ __forceinline__ int ld_i32_u16al(const uint8_t * p) {
    const uint16_t * q = (const uint16_t *)p;
    return (int)((uint32_t)q[0] | ((uint32_t)q[1] << 16));
}
// spread low 4 bits to the LSB of each byte, then to bit 4 (0x10 flags)
__device__ __forceinline__ int spread4_hi(uint32_t v) {
    return (int)((((v & 0xFu) * 0x00204081u) & 0x01010101u) << 4);
}

// F32: lane-coalesced float4 (P0.9).
__device__ inline void gdot_f32(Task & t) {
    const float  * w  = (const float *)t.w;
    const float4 * x4 = (const float4 *)t.x;
    float * y = (float *)t.y;
    const uint32_t n4   = t.cols >> 2;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const float4 * row4 = (const float4 *)(w + (size_t)r * t.cols);
        float acc = 0.f;
        for (uint32_t c = lane; c < n4; c += 32) {
            float4 a = row4[c], b = x4[c];
            acc = fmaf(a.x, b.x, acc); acc = fmaf(a.y, b.y, acc);
            acc = fmaf(a.z, b.z, acc); acc = fmaf(a.w, b.w, acc);
        }
        acc = warp_reduce_add(acc);
        if (lane == 0) y[r] += acc;
    }
}

// Q8_0: block-serial, lane-per-element. Lane loads qs[lane] (1 B of a 32 B
// window); d broadcast from lane 0.
__device__ inline void gdot_q8_0(Task & t) {
    const uint8_t * w = (const uint8_t *)t.w;
    const float   * x = (const float *)t.x;
    float * y = (float *)t.y;
    const uint32_t nb   = t.cols >> 5;
    const uint64_t rb   = (uint64_t)nb * 34;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const uint8_t * row = w + (size_t)r * rb;
        float acc = 0.f;
        for (uint32_t b = 0; b < nb; ++b) {
            const uint8_t * bp = row + (size_t)b * 34;
            float d = 0.f;
            if (lane == 0) d = fl_half2float(*(const uint16_t *)bp);
            d = __shfl_sync(0xffffffffu, d, 0);
            const int q = (int)(int8_t)bp[2 + lane];
            acc = fmaf(d * (float)q, x[(b << 5) + lane], acc);
        }
        acc = warp_reduce_add(acc);
        if (lane == 0) y[r] += acc;
    }
}

// Q5_0: block-serial; lanes 0-15 low nibbles (elems j), 16-31 high (elems
// 16+j). d and the 32-bit qh broadcast from lane 0 (2-byte-aligned loads).
__device__ inline void gdot_q5_0(Task & t) {
    const uint8_t * w = (const uint8_t *)t.w;
    const float   * x = (const float *)t.x;
    float * y = (float *)t.y;
    const uint32_t nb   = t.cols >> 5;
    const uint64_t rb   = (uint64_t)nb * 22;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const uint8_t * row = w + (size_t)r * rb;
        float acc = 0.f;
        for (uint32_t b = 0; b < nb; ++b) {
            const uint8_t * bp = row + (size_t)b * 22;
            float d = 0.f; uint32_t qh = 0;
            if (lane == 0) {
                d = fl_half2float(*(const uint16_t *)bp);
                qh = (uint32_t)*(const uint16_t *)(bp + 2)
                   | ((uint32_t)*(const uint16_t *)(bp + 4) << 16);
            }
            d  = __shfl_sync(0xffffffffu, d, 0);
            qh = __shfl_sync(0xffffffffu, qh, 0);
            const uint32_t j = lane & 15u;
            const uint8_t qb = bp[6 + j];
            int v; uint32_t xi;
            if (lane < 16) {
                v  = (int)((qb & 0x0F) | (((qh >> j) << 4) & 0x10)) - 16;
                xi = (b << 5) + j;
            } else {
                v  = (int)((qb >> 4) | ((qh >> (j + 12)) & 0x10)) - 16;
                xi = (b << 5) + 16 + j;
            }
            acc = fmaf(d * (float)v, x[xi], acc);
        }
        acc = warp_reduce_add(acc);
        if (lane == 0) y[r] += acc;
    }
}

// Q4_K: superblock-serial; per 64-elem chunk, lane l covers elems l (low
// nibble) and 32+l (high). Scales recomputed per lane from the 12-byte table
// (L1-broadcast reads, cheap vs shfl choreography).
__device__ inline void gdot_q4_K(Task & t) {
    const uint8_t * w = (const uint8_t *)t.w;
    const float   * x = (const float *)t.x;
    float * y = (float *)t.y;
    const uint32_t nb   = t.cols >> 8;
    const uint64_t rb   = (uint64_t)nb * 144;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const uint8_t * row = w + (size_t)r * rb;
        float acc = 0.f;
        for (uint32_t sb = 0; sb < nb; ++sb) {
            const uint8_t * bp = row + (size_t)sb * 144;
            const float d    = fl_half2float(*(const uint16_t *)bp);
            const float dmin = fl_half2float(*(const uint16_t *)(bp + 2));
            const uint8_t * scales = bp + 4;
            const uint8_t * qs     = bp + 16;
            const float   * xb     = x + (sb << 8);
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                uint8_t sc, mn;
                fl_q4k_scale_min(2 * c + 0, scales, &sc, &mn);
                const float d1 = d * sc, m1 = dmin * mn;
                fl_q4k_scale_min(2 * c + 1, scales, &sc, &mn);
                const float d2 = d * sc, m2 = dmin * mn;
                const uint8_t q = qs[c * 32 + lane];
                acc = fmaf(d1 * (float)(q & 0xF) - m1, xb[c * 64 + lane], acc);
                acc = fmaf(d2 * (float)(q >> 4)  - m2, xb[c * 64 + 32 + lane], acc);
            }
        }
        acc = warp_reduce_add(acc);
        if (lane == 0) y[r] += acc;
    }
}

// Q6_K: superblock-serial; per 128-elem half, lane l covers elems l, l+32,
// l+64, l+96 (the reference dequant's own lane structure).
__device__ inline void gdot_q6_K(Task & t) {
    const uint8_t * w = (const uint8_t *)t.w;
    const float   * x = (const float *)t.x;
    float * y = (float *)t.y;
    const uint32_t nb   = t.cols >> 8;
    const uint64_t rb   = (uint64_t)nb * 210;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const uint8_t * row = w + (size_t)r * rb;
        float acc = 0.f;
        for (uint32_t sb = 0; sb < nb; ++sb) {
            const uint8_t * bp = row + (size_t)sb * 210;
            const float d = fl_half2float(*(const uint16_t *)(bp + 208));
            const float * xb = x + (sb << 8);
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint8_t * ql = bp       + h * 64;
                const uint8_t * qh = bp + 128 + h * 32;
                const int8_t  * sc = (const int8_t *)(bp + 192) + h * 8;
                const int is = lane >> 4;
                const uint8_t l0 = ql[lane], l32 = ql[lane + 32], hh = qh[lane];
                const float * xh = xb + h * 128;
                acc = fmaf(d * (float)sc[is + 0] * (float)((int)((l0 & 0xF)  | (((hh >> 0) & 3) << 4)) - 32), xh[lane],      acc);
                acc = fmaf(d * (float)sc[is + 2] * (float)((int)((l32 & 0xF) | (((hh >> 2) & 3) << 4)) - 32), xh[lane + 32], acc);
                acc = fmaf(d * (float)sc[is + 4] * (float)((int)((l0 >> 4)   | (((hh >> 4) & 3) << 4)) - 32), xh[lane + 64], acc);
                acc = fmaf(d * (float)sc[is + 6] * (float)((int)((l32 >> 4)  | (((hh >> 6) & 3) << 4)) - 32), xh[lane + 96], acc);
            }
        }
        acc = warp_reduce_add(acc);
        if (lane == 0) y[r] += acc;
    }
}

// P5.0 codebook f32-dequant paths. The engine always runs these formats
// through the int8 kernels below; these exist so the --int8 off path is
// correct rather than a silent zero, and as the dequant-side A/B reference.
// Lane-per-weight within a 32-wide chunk (simpler than the int8 quad walk).
__device__ inline void gdot_iq2_xxs(Task & t) {
    __shared__ uint64_t g2[256];
    __shared__ uint8_t  sgn[128];
    for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) g2[i] = fl_kIq2xxsGrid(i);
    for (uint32_t i = threadIdx.x; i < 128; i += blockDim.x) sgn[i] = fl_kIqSigns(i);
    __syncthreads();
    const uint8_t * w = (const uint8_t *)t.w;
    const float   * x = (const float *)t.x;
    float * y = (float *)t.y;
    const uint32_t nsb  = t.cols >> 8;
    const uint64_t rb   = (uint64_t)nsb * 66;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    const uint32_t l = lane >> 3, jj = lane & 7;
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const uint8_t * row = w + (size_t)r * rb;
        float acc = 0.f;
        for (uint32_t sb = 0; sb < nsb; ++sb) {
            const uint8_t * bp = row + (size_t)sb * 66;
            const float d = fl_half2float(*(const uint16_t *)bp);
            for (uint32_t ib32 = 0; ib32 < 8; ++ib32) {
                const uint8_t * q = bp + 2 + 8 * ib32;
                const uint32_t w0 = (uint32_t)ld_i32_u16al(q);
                const uint32_t w1 = (uint32_t)ld_i32_u16al(q + 4);
                const uint64_t g = g2[(w0 >> (8 * l)) & 0xFFu];
                const uint32_t s = sgn[(w1 >> (7 * l)) & 127u];
                const float v = (float)(uint8_t)(g >> (8 * jj));
                const float db = d * (0.5f + (float)(w1 >> 28)) * 0.25f;
                acc = fmaf(db * ((s >> jj) & 1u ? -v : v),
                           x[(sb << 8) + ib32 * 32 + lane], acc);
            }
        }
        acc = warp_reduce_add(acc);
        if (lane == 0) y[r] += acc;
    }
    __syncthreads();
}

__device__ inline void gdot_iq3_xxs(Task & t) {
    __shared__ uint32_t g3[256];
    __shared__ uint8_t  sgn[128];
    for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) g3[i] = fl_kIq3xxsGrid(i);
    for (uint32_t i = threadIdx.x; i < 128; i += blockDim.x) sgn[i] = fl_kIqSigns(i);
    __syncthreads();
    const uint8_t * w = (const uint8_t *)t.w;
    const float   * x = (const float *)t.x;
    float * y = (float *)t.y;
    const uint32_t nsb  = t.cols >> 8;
    const uint64_t rb   = (uint64_t)nsb * 98;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    const uint32_t quad = lane >> 2, jj = lane & 3;   // 8 grid entries x 4 vals
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const uint8_t * row = w + (size_t)r * rb;
        float acc = 0.f;
        for (uint32_t sb = 0; sb < nsb; ++sb) {
            const uint8_t * bp = row + (size_t)sb * 98;
            const float d = fl_half2float(*(const uint16_t *)bp);
            for (uint32_t ib32 = 0; ib32 < 8; ++ib32) {
                const uint32_t w1 = (uint32_t)ld_i32_u16al(bp + 2 + 64 + 4 * ib32);
                const uint32_t g = g3[bp[2 + 8 * ib32 + quad]];
                const uint32_t s = sgn[(w1 >> (7 * (quad >> 1))) & 127u];
                const uint32_t bit = 4 * (quad & 1) + jj;
                const float v = (float)(uint8_t)(g >> (8 * jj));
                const float db = d * (0.5f + (float)(w1 >> 28)) * 0.5f;
                acc = fmaf(db * ((s >> bit) & 1u ? -v : v),
                           x[(sb << 8) + ib32 * 32 + lane], acc);
            }
        }
        acc = warp_reduce_add(acc);
        if (lane == 0) y[r] += acc;
    }
    __syncthreads();
}

__device__ inline void gdot_fast(Task & t) {
    switch (t.fmt) {
        case Fmt::F32:  gdot_f32(t);  break;
        case Fmt::Q8_0: gdot_q8_0(t); break;
        case Fmt::Q5_0: gdot_q5_0(t); break;
        case Fmt::Q4_K: gdot_q4_K(t); break;
        case Fmt::Q6_K: gdot_q6_K(t); break;
        case Fmt::IQ2_XXS: gdot_iq2_xxs(t); break;
        case Fmt::IQ3_XXS: gdot_iq3_xxs(t); break;
    }
}

// ---- int8 (q8_1 activation) dp4a kernels (P1.1) ---------------------------
// Warp-per-row, block-serial like the f32-dequant shapes above, but each lane
// integer-dots a 4-byte quad via __dp4a. Weight quad loads use 2-byte-aligned
// composition where the block layout only guarantees u16 alignment (q8_0 34 B,
// q5_0 22 B, q6_K 210 B); q4_K (144 B) rows and all q8_1 appendices are
// >=16-aligned, so those load ints directly. Lane float folds reorder vs the
// CPU int8 order -> engine-level tolerance, same as every fast path.


__device__ inline void gdot_i8_q8_0(Task & t) {
    const uint8_t  * w  = (const uint8_t *)t.w;
    const blk_q8_1 * xq = fl_xq_of(t.x, t.cols);
    float * y = (float *)t.y;
    const uint32_t nb   = t.cols >> 5;
    const uint64_t rb   = (uint64_t)nb * 34;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    const uint32_t bg   = lane >> 3, k = lane & 7;      // 4 blocks x 8 quads
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const uint8_t * row = w + (size_t)r * rb;
        // P1.3: 4 independent per-lane accumulators break the serial fmaf
        // chain (engine-tolerance path; order is free)
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        uint32_t it = 0;
        for (uint32_t b0 = 0; b0 < nb; b0 += 4, ++it) {
            const uint32_t b = b0 + bg;
            if (b < nb) {
                const uint8_t * bp = row + (size_t)b * 34;
                const int wq = ld_i32_u16al(bp + 2 + 4 * k);
                const int xv = ((const int *)xq[b].qs)[k];
                const float dw = fl_half2float(*(const uint16_t *)bp);
                const float dx = fl_half2float(xq[b].d);
                const float p = dw * dx, v = (float)__dp4a(wq, xv, 0);
                switch (it & 3) {
                    case 0: a0 = fmaf(p, v, a0); break;
                    case 1: a1 = fmaf(p, v, a1); break;
                    case 2: a2 = fmaf(p, v, a2); break;
                    default: a3 = fmaf(p, v, a3); break;
                }
            }
        }
        float acc = warp_reduce_add((a0 + a1) + (a2 + a3));
        if (lane == 0) y[r] += acc;
    }
}

__device__ inline void gdot_i8_q5_0(Task & t) {
    const uint8_t  * w  = (const uint8_t *)t.w;
    const blk_q8_1 * xq = fl_xq_of(t.x, t.cols);
    float * y = (float *)t.y;
    const uint32_t nb   = t.cols >> 5;
    const uint64_t rb   = (uint64_t)nb * 22;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    const uint32_t bg   = lane >> 2, k = lane & 3;      // 8 blocks x 4 quads
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const uint8_t * row = w + (size_t)r * rb;
        float a0 = 0.f, a1 = 0.f;                       // P1.3 ILP split
        uint32_t it = 0;
        for (uint32_t b0 = 0; b0 < nb; b0 += 8, ++it) {
            const uint32_t b = b0 + bg;
            if (b < nb) {
                const uint8_t * bp = row + (size_t)b * 22;
                const int qsv = ld_i32_u16al(bp + 6 + 4 * k);
                const uint32_t qh = (uint32_t)ld_i32_u16al(bp + 2);
                const int wlo = (qsv & 0x0F0F0F0F) | spread4_hi(qh >> (4 * k));
                const int whi = ((qsv >> 4) & 0x0F0F0F0F)
                              | spread4_hi(qh >> (16 + 4 * k));
                const int xlo = ((const int *)xq[b].qs)[k];
                const int xhi = ((const int *)xq[b].qs)[4 + k];
                const int idot = __dp4a(wlo, xlo, __dp4a(whi, xhi, 0));
                const int isum = __dp4a(0x01010101, xlo,
                                        __dp4a(0x01010101, xhi, 0));
                const float dw = fl_half2float(*(const uint16_t *)bp);
                const float dx = fl_half2float(xq[b].d);
                const float p = dw * dx, v = (float)(idot - 16 * isum);
                if (it & 1) a1 = fmaf(p, v, a1); else a0 = fmaf(p, v, a0);
            }
        }
        float acc = warp_reduce_add(a0 + a1);
        if (lane == 0) y[r] += acc;
    }
}

// v1 (P1.1): per-lane branchy scale decode (64 redundant unpacks per warp per
// superblock) + per-lane isum dp4a pairs. Kept for membench A/B (--q4k v1).
__device__ inline void gdot_i8_q4_K_v1(Task & t) {
    const uint8_t  * w  = (const uint8_t *)t.w;
    const blk_q8_1 * xq = fl_xq_of(t.x, t.cols);
    float * y = (float *)t.y;
    const uint32_t nsb  = t.cols >> 8;
    const uint64_t rb   = (uint64_t)nsb * 144;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    const uint32_t c    = lane >> 3, k = lane & 7;      // 4 chunks x 8 quads
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const uint8_t * row = w + (size_t)r * rb;       // 16-aligned (144|16)
        // P1.3: the 4 per-superblock terms spread across 4 independent
        // accumulators (was one serial 4-fmaf chain per sb per lane)
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (uint32_t sb = 0; sb < nsb; ++sb) {
            const uint8_t * bp = row + (size_t)sb * 144;
            const float d    = fl_half2float(*(const uint16_t *)bp);
            const float dmin = fl_half2float(*(const uint16_t *)(bp + 2));
            uint8_t sc1, mn1, sc2, mn2;
            fl_q4k_scale_min(2 * c + 0, bp + 4, &sc1, &mn1);
            fl_q4k_scale_min(2 * c + 1, bp + 4, &sc2, &mn2);
            const int wq  = ((const int *)(bp + 16))[c * 8 + k];
            const int lo  = wq & 0x0F0F0F0F;
            const int hi  = (wq >> 4) & 0x0F0F0F0F;
            const blk_q8_1 & xlo_b = xq[sb * 8 + 2 * c];
            const blk_q8_1 & xhi_b = xq[sb * 8 + 2 * c + 1];
            const int xlo = ((const int *)xlo_b.qs)[k];
            const int xhi = ((const int *)xhi_b.qs)[k];
            const float dxl = fl_half2float(xlo_b.d);
            const float dxh = fl_half2float(xhi_b.d);
            a0 = fmaf(d * sc1 * dxl,       (float)__dp4a(lo, xlo, 0), a0);
            a1 = fmaf(-(dmin * mn1) * dxl, (float)__dp4a(0x01010101, xlo, 0), a1);
            a2 = fmaf(d * sc2 * dxh,       (float)__dp4a(hi, xhi, 0), a2);
            a3 = fmaf(-(dmin * mn2) * dxh, (float)__dp4a(0x01010101, xhi, 0), a3);
        }
        float acc = warp_reduce_add((a0 + a1) + (a2 + a3));
        if (lane == 0) y[r] += acc;
    }
}

// v2 (P1.2): the 8-lane chunk-group leader (k==0) decodes the group's two
// scale pairs once and shfl-broadcasts them (width 8); the min terms use the
// q8_1 block's precomputed s = d*sum (the ggml mmvq trick), folded once per
// group at the leader instead of per-lane isum dp4a pairs. Halves the dp4a
// count and removes 62/64 of the scale unpacks. s is a half-rounded product,
// so v2's min terms differ from v1 in low bits -> engine tolerance gate
// (re-measured this phase), same contract as every fast GPU path.
__device__ inline void gdot_i8_q4_K(Task & t) {
    const uint8_t  * w  = (const uint8_t *)t.w;
    const blk_q8_1 * xq = fl_xq_of(t.x, t.cols);
    float * y = (float *)t.y;
    const uint32_t nsb  = t.cols >> 8;
    const uint64_t rb   = (uint64_t)nsb * 144;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    const uint32_t c    = lane >> 3, k = lane & 7;      // 4 chunks x 8 quads
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const uint8_t * row = w + (size_t)r * rb;       // 16-aligned (144|16)
        float acc = 0.f;
        for (uint32_t sb = 0; sb < nsb; ++sb) {
            const uint8_t * bp = row + (size_t)sb * 144;
            const float d    = fl_half2float(*(const uint16_t *)bp);
            const float dmin = fl_half2float(*(const uint16_t *)(bp + 2));
            uint32_t packed = 0;
            if (k == 0) {
                uint8_t sc1, mn1, sc2, mn2;
                fl_q4k_scale_min(2 * c + 0, bp + 4, &sc1, &mn1);
                fl_q4k_scale_min(2 * c + 1, bp + 4, &sc2, &mn2);
                packed = (uint32_t)sc1 | ((uint32_t)mn1 << 8)
                       | ((uint32_t)sc2 << 16) | ((uint32_t)mn2 << 24);
            }
            packed = __shfl_sync(0xffffffffu, packed, 0, 8);
            const float sc1 = (float)(packed & 0xFFu);
            const float mn1 = (float)((packed >>  8) & 0xFFu);
            const float sc2 = (float)((packed >> 16) & 0xFFu);
            const float mn2 = (float)(packed >> 24);
            const int wq  = ((const int *)(bp + 16))[c * 8 + k];
            const int lo  = wq & 0x0F0F0F0F;
            const int hi  = (wq >> 4) & 0x0F0F0F0F;
            const blk_q8_1 & xlo_b = xq[sb * 8 + 2 * c];
            const blk_q8_1 & xhi_b = xq[sb * 8 + 2 * c + 1];
            const int xlo = ((const int *)xlo_b.qs)[k];
            const int xhi = ((const int *)xhi_b.qs)[k];
            acc = fmaf(d * sc1 * fl_half2float(xlo_b.d),
                       (float)__dp4a(lo, xlo, 0), acc);
            acc = fmaf(d * sc2 * fl_half2float(xhi_b.d),
                       (float)__dp4a(hi, xhi, 0), acc);
            if (k == 0) {
                acc = fmaf(-(dmin * mn1), fl_half2float(xlo_b.s), acc);
                acc = fmaf(-(dmin * mn2), fl_half2float(xhi_b.s), acc);
            }
        }
        acc = warp_reduce_add(acc);
        if (lane == 0) y[r] += acc;
    }
}

__device__ inline void gdot_i8_q6_K(Task & t) {
    const uint8_t  * w  = (const uint8_t *)t.w;
    const blk_q8_1 * xq = fl_xq_of(t.x, t.cols);
    float * y = (float *)t.y;
    const uint32_t nsb  = t.cols >> 8;
    const uint64_t rb   = (uint64_t)nsb * 210;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    const uint32_t s    = lane >> 3, k = lane & 7;      // 4 spans x 8 quads
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const uint8_t * row = w + (size_t)r * rb;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;   // P1.3 ILP split
        for (uint32_t sb = 0; sb < nsb; ++sb) {
            const uint8_t * bp = row + (size_t)sb * 210;
            const float dw = fl_half2float(*(const uint16_t *)(bp + 208));
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int qlv = ld_i32_u16al(bp + h * 64 + (s & 1) * 32 + 4 * k);
                const int qhv = ld_i32_u16al(bp + 128 + h * 32 + 4 * k);
                const int nib = (s < 2) ? (qlv & 0x0F0F0F0F)
                                        : ((qlv >> 4) & 0x0F0F0F0F);
                const int hb  = (qhv >> (2 * s)) & 0x03030303;
                const int wq  = nib | (hb << 4);
                const blk_q8_1 & xb = xq[sb * 8 + h * 4 + s];
                const int xv = ((const int *)xb.qs)[k];
                const int idot = __dp4a(wq, xv, 0);
                const int isum = __dp4a(0x01010101, xv, 0);
                const int sc = ((const int8_t *)(bp + 192))[h * 8 + 2 * s + (k >= 4)];
                const float p = dw * fl_half2float(xb.d);
                const float v = (float)(sc * (idot - 32 * isum));
                switch (((sb << 1) | (uint32_t)h) & 3) {
                    case 0: a0 = fmaf(p, v, a0); break;
                    case 1: a1 = fmaf(p, v, a1); break;
                    case 2: a2 = fmaf(p, v, a2); break;
                    default: a3 = fmaf(p, v, a3); break;
                }
            }
        }
        float acc = warp_reduce_add((a0 + a1) + (a2 + a3));
        if (lane == 0) y[r] += acc;
    }
}

// ---- q8_K appendix kernels (P1.5): integer scale folding ------------------
// Same warp-per-row lane geometry as the v1 shapes (redundant per-lane scale
// decode via L1 broadcasts - P1.2 proved that beats shfl choreography). The
// per-lane superblock partial folds through ONE int accumulator, then a
// single fmaf(d*dk, ...) per superblock (q4_K adds the exact bsums min term
// on the k==0 lane). Removes the isum dp4a pairs and most float folds.
__device__ inline void gdot_i8k_q4_K(Task & t) {
    const uint8_t  * w  = (const uint8_t *)t.w;
    const blk_q8_K * xk = fl_xqk_of(t.x, t.cols);
    float * y = (float *)t.y;
    const uint32_t nsb  = t.cols >> 8;
    const uint64_t rb   = (uint64_t)nsb * 144;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    const uint32_t c    = lane >> 3, k = lane & 7;      // 4 chunks x 8 quads
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const uint8_t * row = w + (size_t)r * rb;
        float a0 = 0.f, a1 = 0.f;                       // ILP pair
        uint32_t sb = 0;
#if FASTLLM_Q4K_UNROLL > 1
        // P8.9 MLP unroll. The loop below carried exactly ONE superblock in
        // flight per warp: iteration sb+1's loads cannot issue until sb's
        // __dp4a has consumed wq/xlo/xhi. ncu named it - long_scoreboard 12.03
        // of 16.28 cyc/issued-inst (74%), no-eligible 87.7% (docs/BENCH.md).
        //
        // Little's Law, fitted against our OWN P1.4 block sweep (8->16 warps/SM
        // gave 1.98x for exactly 2x in-flight bytes, so L ~ 956 ns): sustaining
        // 262 GB/s needs ~391 B in flight per warp; one superblock is 144 B.
        // Hence ~2.7x short, and why bigger tiles never helped - rowsz changes
        // rows per warp, not superblocks in flight.
        //
        // Only the PAYLOAD loads are hoisted. The header (d/dmin/scales) is
        // warp-uniform and L1-resident (88.6% hit), so it costs one broadcast
        // sector, not DRAM traffic - hoisting it too would buy nothing and
        // would need either a 16 B aligned load (unsafe on mmap'd GGUF) or
        // dynamic register indexing (spills to local memory).
        //
        // BIT-IDENTICAL BY CONSTRUCTION: the fold phase walks u in ascending
        // sb with the arithmetic unchanged, so a0/a1 see the same fmaf
        // sequence in the same order as the scalar loop. Costs 3 ints x U
        // registers (12 at U=4) against the 64-register launch bound.
        for (; sb + FASTLLM_Q4K_UNROLL <= nsb; sb += FASTLLM_Q4K_UNROLL) {
            int wq[FASTLLM_Q4K_UNROLL];
            int xlo[FASTLLM_Q4K_UNROLL], xhi[FASTLLM_Q4K_UNROLL];
            #pragma unroll
            for (int u = 0; u < FASTLLM_Q4K_UNROLL; ++u) {   // loads, no consumer
                const uint8_t * bp = row + (size_t)(sb + u) * 144;
                wq[u]  = ((const int *)(bp + 16))[c * 8 + k];
                xlo[u] = ((const int *)(xk[sb + u].qs + 2 * c * 32))[k];
                xhi[u] = ((const int *)(xk[sb + u].qs + (2 * c + 1) * 32))[k];
            }
            #pragma unroll
            for (int u = 0; u < FASTLLM_Q4K_UNROLL; ++u) {   // fold, ascending
                const uint8_t * bp = row + (size_t)(sb + u) * 144;
                const float d    = fl_half2float(*(const uint16_t *)bp);
                const float dmin = fl_half2float(*(const uint16_t *)(bp + 2));
                const float dk   = xk[sb + u].d;
                uint8_t sc1, mn1, sc2, mn2;
                fl_q4k_scale_min(2 * c + 0, bp + 4, &sc1, &mn1);
                fl_q4k_scale_min(2 * c + 1, bp + 4, &sc2, &mn2);
                const int lo  = wq[u] & 0x0F0F0F0F;
                const int hi  = (wq[u] >> 4) & 0x0F0F0F0F;
                const int iacc = (int)sc1 * __dp4a(lo, xlo[u], 0)
                               + (int)sc2 * __dp4a(hi, xhi[u], 0);
                a0 = fmaf(d * dk, (float)iacc, a0);
                if (k == 0) {
                    const int imins = (int)mn1 * ((int)xk[sb + u].bsums[4 * c]
                                                + (int)xk[sb + u].bsums[4 * c + 1])
                                    + (int)mn2 * ((int)xk[sb + u].bsums[4 * c + 2]
                                                + (int)xk[sb + u].bsums[4 * c + 3]);
                    a1 = fmaf(-(dmin * dk), (float)imins, a1);
                }
            }
        }
#endif
        for (; sb < nsb; ++sb) {
            const uint8_t * bp = row + (size_t)sb * 144;
            const float d    = fl_half2float(*(const uint16_t *)bp);
            const float dmin = fl_half2float(*(const uint16_t *)(bp + 2));
            const float dk   = xk[sb].d;
            uint8_t sc1, mn1, sc2, mn2;
            fl_q4k_scale_min(2 * c + 0, bp + 4, &sc1, &mn1);
            fl_q4k_scale_min(2 * c + 1, bp + 4, &sc2, &mn2);
            const int wq  = ((const int *)(bp + 16))[c * 8 + k];
            const int lo  = wq & 0x0F0F0F0F;
            const int hi  = (wq >> 4) & 0x0F0F0F0F;
            const int xlo = ((const int *)(xk[sb].qs + 2 * c * 32))[k];
            const int xhi = ((const int *)(xk[sb].qs + (2 * c + 1) * 32))[k];
            const int iacc = (int)sc1 * __dp4a(lo, xlo, 0)
                           + (int)sc2 * __dp4a(hi, xhi, 0);
            a0 = fmaf(d * dk, (float)iacc, a0);
            if (k == 0) {
                const int imins = (int)mn1 * ((int)xk[sb].bsums[4 * c]
                                            + (int)xk[sb].bsums[4 * c + 1])
                                + (int)mn2 * ((int)xk[sb].bsums[4 * c + 2]
                                            + (int)xk[sb].bsums[4 * c + 3]);
                a1 = fmaf(-(dmin * dk), (float)imins, a1);
            }
        }
        float acc = warp_reduce_add(a0 + a1);
        if (lane == 0) y[r] += acc;
    }
}

__device__ inline void gdot_i8k_q6_K(Task & t) {
    const uint8_t  * w  = (const uint8_t *)t.w;
    const blk_q8_K * xk = fl_xqk_of(t.x, t.cols);
    float * y = (float *)t.y;
    const uint32_t nsb  = t.cols >> 8;
    const uint64_t rb   = (uint64_t)nsb * 210;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    const uint32_t s    = lane >> 3, k = lane & 7;      // 4 spans x 8 quads
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const uint8_t * row = w + (size_t)r * rb;
        float acc = 0.f;
        for (uint32_t sb = 0; sb < nsb; ++sb) {
            const uint8_t * bp = row + (size_t)sb * 210;
            const float dw = fl_half2float(*(const uint16_t *)(bp + 208));
            const float dk = xk[sb].d;
            int isb = 0;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int qlv = ld_i32_u16al(bp + h * 64 + (s & 1) * 32 + 4 * k);
                const int qhv = ld_i32_u16al(bp + 128 + h * 32 + 4 * k);
                const int nib = (s < 2) ? (qlv & 0x0F0F0F0F)
                                        : ((qlv >> 4) & 0x0F0F0F0F);
                const int hb  = (qhv >> (2 * s)) & 0x03030303;
                const int wq  = nib | (hb << 4);
                const int g16 = (int)(k >= 4);          // 16-elem scale group
                const int sc  = ((const int8_t *)(bp + 192))[h * 8 + 2 * s + g16];
                const int xv  = ((const int *)(xk[sb].qs + (h * 4 + s) * 32))[k];
                int tpart = sc * __dp4a(wq, xv, 0);
                if ((k & 3) == 0) {                     // one lane per 16-group
                    const int bs = (int)xk[sb].bsums[2 * (h * 4 + s) + g16];
                    tpart -= sc * 32 * bs;
                }
                isb += tpart;
            }
            acc = fmaf(dw * dk, (float)isb, acc);
        }
        acc = warp_reduce_add(acc);
        if (lane == 0) y[r] += acc;
    }
}

// P2.5 unpack-sharing multi-column q4_K/i8k: per superblock the weight int,
// nibble split and scale decode happen ONCE per lane; columns run in register
// chunks of 4 (a0/a1 per column). Per-column op order is identical to
// gdot_i8k_q4_K, so each column is bitwise equal to its single-column tile.
// Weights re-walk once per 4-column chunk (L2-resident on the re-walk).
__device__ inline void gdot_i8k_q4_K_mc(Task & t) {
    const uint8_t * w = (const uint8_t *)t.w;
    float * ybase = (float *)t.y;
    const uint32_t nx  = t.nx;
    const uint32_t nsb = t.cols >> 8;
    const uint64_t rb  = (uint64_t)nsb * 144;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    const uint32_t c    = lane >> 3, k = lane & 7;
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const uint8_t * row = w + (size_t)r * rb;
        for (uint32_t p0 = 0; p0 < nx; p0 += 4) {
            const uint32_t pc = (nx - p0 < 4u) ? nx - p0 : 4u;
            float a0[4] = {0.f, 0.f, 0.f, 0.f};
            float a1[4] = {0.f, 0.f, 0.f, 0.f};
            for (uint32_t sb = 0; sb < nsb; ++sb) {
                const uint8_t * bp = row + (size_t)sb * 144;
                const float d    = fl_half2float(*(const uint16_t *)bp);
                const float dmin = fl_half2float(*(const uint16_t *)(bp + 2));
                uint8_t sc1, mn1, sc2, mn2;
                fl_q4k_scale_min(2 * c + 0, bp + 4, &sc1, &mn1);
                fl_q4k_scale_min(2 * c + 1, bp + 4, &sc2, &mn2);
                const int wq = ((const int *)(bp + 16))[c * 8 + k];
                const int lo = wq & 0x0F0F0F0F;
                const int hi = (wq >> 4) & 0x0F0F0F0F;
                #pragma unroll
                for (uint32_t pp = 0; pp < 4; ++pp) {
                    if (pp >= pc) break;
                    const blk_q8_K * xkp = fl_xqk_of(
                        (const float *)t.x + (size_t)(p0 + pp) * t.xstride,
                        t.cols);
                    const float dk = xkp[sb].d;
                    const int xlo = ((const int *)(xkp[sb].qs + 2 * c * 32))[k];
                    const int xhi = ((const int *)(xkp[sb].qs
                                                   + (2 * c + 1) * 32))[k];
                    const int iacc = (int)sc1 * __dp4a(lo, xlo, 0)
                                   + (int)sc2 * __dp4a(hi, xhi, 0);
                    a0[pp] = fmaf(d * dk, (float)iacc, a0[pp]);
                    if (k == 0) {
                        const int imins =
                            (int)mn1 * ((int)xkp[sb].bsums[4 * c]
                                      + (int)xkp[sb].bsums[4 * c + 1])
                          + (int)mn2 * ((int)xkp[sb].bsums[4 * c + 2]
                                      + (int)xkp[sb].bsums[4 * c + 3]);
                        a1[pp] = fmaf(-(dmin * dk), (float)imins, a1[pp]);
                    }
                }
            }
            #pragma unroll
            for (uint32_t pp = 0; pp < 4; ++pp) {
                if (pp >= pc) break;
                const float acc = warp_reduce_add(a0[pp] + a1[pp]);
                if (lane == 0)
                    ybase[(size_t)(p0 + pp) * t.ystride + r] += acc;
            }
        }
    }
}

// ---- P5.0: codebook (grid) formats -----------------------------------------
// The property the CPU lacks and the GPU has: the 2 KiB grid is staged into
// shared memory once per block, so a lookup is an SMEM read instead of the
// DRAM gather that pins Thor's CPU at 5.5 Gw/s .
// Shape is the usual warp-per-row lane-coalesced quad walk: each lane owns one
// 4-weight quad, dequantizes it into an int8 quad in registers, and dp4a's it
// against the q8_1 activation appendix.
//
// Sign application is byte-wise: v ^ neg_mask, then + (neg & 0x01010101). No
// inter-byte carry is possible because every grid byte is >= 1 (asserted at
// table generation), so ~v + 1 <= 255 per lane.
// expand the 4 sign bits that apply to a quad into a 0xFF-per-byte mask
__device__ __forceinline__ int sign_mask4(uint32_t signs, int shift) {
    const uint32_t b = (signs >> shift) & 0xFu;      // one bit per byte
    return (int)(((b * 0x00204081u) & 0x01010101u) * 0xFFu);
}

__device__ inline void gdot_i8_iq2_xxs(Task & t) {
    // 2 KiB grid + 128 B sign table, staged once per task by the whole block.
    // The trailing sync keeps a following task's stage from racing readers.
    __shared__ uint64_t g2[256];
    __shared__ uint8_t  sgn[128];
    for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) g2[i] = fl_kIq2xxsGrid(i);
    for (uint32_t i = threadIdx.x; i < 128; i += blockDim.x) sgn[i] = fl_kIqSigns(i);
    __syncthreads();
    const uint8_t  * w  = (const uint8_t *)t.w;
    const blk_q8_1 * xq = fl_xq_of(t.x, t.cols);
    float * y = (float *)t.y;
    const uint32_t nsb  = t.cols >> 8;                   // 256-weight blocks
    const uint64_t rb   = (uint64_t)nsb * 66;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    const uint32_t ib   = lane >> 3, k = lane & 7;       // 4 chunks x 8 quads
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const uint8_t * row = w + (size_t)r * rb;
        float a0 = 0.f, a1 = 0.f;
        for (uint32_t sb = 0; sb < nsb; ++sb) {
            const uint8_t * bp = row + (size_t)sb * 66;
            const float d = fl_half2float(*(const uint16_t *)bp);
            #pragma unroll
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t ib32 = half * 4 + ib;     // 8 chunks / superblock
                const uint8_t * q = bp + 2 + 8 * ib32;
                const uint32_t w0 = (uint32_t)ld_i32_u16al(q);
                const uint32_t w1 = (uint32_t)ld_i32_u16al(q + 4);
                const uint32_t l = k >> 1, sel = k & 1;  // grid entry, its half
                const uint64_t g = g2[(w0 >> (8 * l)) & 0xFFu];
                const int vq = (int)(uint32_t)(sel ? (g >> 32) : g);
                const uint32_t sg = sgn[(w1 >> (7 * l)) & 127u];
                const int neg = sign_mask4(sg, 4 * sel);
                const int wq = (vq ^ neg) + (neg & 0x01010101);
                const uint32_t b = (sb << 3) + ib32;     // 32-wide q8_1 block
                const int xv = ((const int *)xq[b].qs)[k];
                const float db = d * (0.5f + (float)(w1 >> 28)) * 0.25f;
                const float p  = db * fl_half2float(xq[b].d);
                const float v  = (float)__dp4a(wq, xv, 0);
                if (half) a1 = fmaf(p, v, a1); else a0 = fmaf(p, v, a0);
            }
        }
        float acc = warp_reduce_add(a0 + a1);
        if (lane == 0) y[r] += acc;
    }
    __syncthreads();
}

__device__ inline void gdot_i8_iq3_xxs(Task & t) {
    __shared__ uint32_t g3[256];
    __shared__ uint8_t  sgn[128];
    for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) g3[i] = fl_kIq3xxsGrid(i);
    for (uint32_t i = threadIdx.x; i < 128; i += blockDim.x) sgn[i] = fl_kIqSigns(i);
    __syncthreads();
    const uint8_t  * w  = (const uint8_t *)t.w;
    const blk_q8_1 * xq = fl_xq_of(t.x, t.cols);
    float * y = (float *)t.y;
    const uint32_t nsb  = t.cols >> 8;
    const uint64_t rb   = (uint64_t)nsb * 98;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nw   = blockDim.x >> 5;
    const uint32_t ib   = lane >> 3, k = lane & 7;
    for (uint32_t r = warp; r < t.rows; r += nw) {
        const uint8_t * row = w + (size_t)r * rb;
        float a0 = 0.f, a1 = 0.f;
        for (uint32_t sb = 0; sb < nsb; ++sb) {
            const uint8_t * bp = row + (size_t)sb * 98;
            const float d = fl_half2float(*(const uint16_t *)bp);
            const uint8_t * qs = bp + 2;
            const uint8_t * ss = bp + 2 + 64;
            #pragma unroll
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t ib32 = half * 4 + ib;
                const uint32_t w1 = (uint32_t)ld_i32_u16al(ss + 4 * ib32);
                // each grid entry is 4 weights, so a quad IS one entry
                const int vq = (int)g3[qs[8 * ib32 + k]];
                const uint32_t sg = sgn[(w1 >> (7 * (k >> 1))) & 127u];
                const int neg = sign_mask4(sg, 4 * (k & 1));
                const int wq = (vq ^ neg) + (neg & 0x01010101);
                const uint32_t b = (sb << 3) + ib32;
                const int xv = ((const int *)xq[b].qs)[k];
                const float db = d * (0.5f + (float)(w1 >> 28)) * 0.5f;
                const float p  = db * fl_half2float(xq[b].d);
                const float v  = (float)__dp4a(wq, xv, 0);
                if (half) a1 = fmaf(p, v, a1); else a0 = fmaf(p, v, a0);
            }
        }
        float acc = warp_reduce_add(a0 + a1);
        if (lane == 0) y[r] += acc;
    }
    __syncthreads();
}

__device__ inline void gdot_i8(Task & t, uint32_t i8k_mask) {
    // P1.5: q8_K appendix path for K formats when its mask bit is set
    if ((t.fmt == Fmt::Q4_K || t.fmt == Fmt::Q6_K)
        && ((i8k_mask >> (int)t.fmt) & 1u)) {
        if (t.fmt == Fmt::Q4_K) gdot_i8k_q4_K(t);
        else                    gdot_i8k_q6_K(t);
        return;
    }
    switch (t.fmt) {
        case Fmt::Q8_0: gdot_i8_q8_0(t); break;
        case Fmt::Q5_0: gdot_i8_q5_0(t); break;
        // q4_K dispatches to v1: the P1.2 probe measured v2 (leader scale
        // decode + s-trick) at 21.1 vs v1's 23.2 GB/s - the shfl choreography
        // and k==0 divergence cost more than the work they remove. v2 stays
        // for the record; the profiling output records this.
        case Fmt::Q4_K: gdot_i8_q4_K_v1(t); break;
        case Fmt::Q6_K: gdot_i8_q6_K(t); break;
        case Fmt::IQ2_XXS: gdot_i8_iq2_xxs(t); break;
        case Fmt::IQ3_XXS: gdot_i8_iq3_xxs(t); break;
        case Fmt::F32:  gdot_f32(t);     break;   // never routed here
    }
}

// ACT_Q8: thread-per-32-block; fl_quantize_q8_1_block is pure bit arithmetic
// (roundf + software RNE half), so device output is byte-identical to host.
// P1.5: also fills the q8_K appendix (thread-per-256-superblock), same
// determinism argument (roundf + f32 scale, no half conversion).
__device__ inline void gact_q8(Task & t) {
    const float * x = (const float *)t.x;
    blk_q8_1 * o = (blk_q8_1 *)t.y;
    const uint32_t nb = t.cols / 32;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x)
        fl_quantize_q8_1_block(x + b * 32, o + b);
    blk_q8_K * ok = (blk_q8_K *)fl_xqk_of(t.x, t.cols);
    const uint32_t nsb = t.cols / 256;
    for (uint32_t sb = threadIdx.x; sb < nsb; sb += blockDim.x)
        fl_quantize_q8_K_block(x + sb * 256, ok + sb);
}

} // namespace fastllm
