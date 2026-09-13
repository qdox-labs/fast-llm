// P3.2 full-shadow experts (task #10, revised by the P3.0 measurement).
//
// There is no small hot set (16-token windows touch 39-45 of 64 experts), so
// selection/eviction logic is pointless: ALL routed-expert weights are
// pre-unpacked to int8 at load into unified memory (~19 GB of ~100 GB free).
// CPU STREAM_DOT tiles read the shadow and skip the nibble/scale unpack (the
// ALU term P1.4 named: IPC 3.20, instruction-count-bound). The GPU keeps
// reading packed weights (it is bytes-bound; the shadow ~doubles bytes).
//
// Bitwise contract: a shadow stores the EXACT integers and f32 scales the
// packed kernel derives during unpack, and the shadow kernels replicate the
// packed kernels' accumulation order (integer work is exact; the fmaf fold
// sequence is unchanged), so shadow results are bitwise-identical to packed
// results on every arch. Gated by tests/test_shadow.cpp.
//
// Formats:
//   Q4_K (i8k path)  per 256-elem superblock, stride 280:
//     f32 d; f32 dmin; u8 sc[8]; u8 mn[8]; int8 qs[256]
//     qs = raw nibble values 0..15 in element order (chunk c: 32 lo, 32 hi).
//   Q5_0 (i8 path)   per 32-elem block, stride 40:
//     f32 d; int8 qs[32]; u8 pad[4]
//     qs = composed 5-bit values 0..31 (the -16 offset stays algebraic via
//     the activation sum, exactly as fl_dot_i8_q5_0 does).
//   Q8_0: NO shadow - its weights are already int8; unpack is pure loads.
//
// Modes (FASTLLM_SHADOW): off (default) | build | cache. "cache" mmaps
// ~/.fastllm-shadow.bin (or FASTLLM_SHADOW_FILE) when valid - a
// file-backed, page-cache-evictable mapping (the OOM-safe pattern) - and
// builds+writes it on first use.
#pragma once
#include "fastllm/fastllm.h"
#include "quants.h"
#include <cstdint>
#include <string>
#include <vector>

namespace fastllm {

inline constexpr uint32_t kShadowQ4KSbBytes = 280;  // per 256 elems
inline constexpr uint32_t kShadowQ50BlkBytes = 40;  // per 32 elems

// P4.4 SMMLA layout (Q4_K only). Same bytes, row-PAIR interleaved so i8mm
// SMMLA can consume the A operand with no runtime repack: per superblock,
//   [row0 hdr 24B][row1 hdr 24B][ 32 x ( row0 8B, row1 8B ) = 512B ] = 560B
// (hdr = f32 d; f32 dmin; u8 sc[8]; u8 mn[8]). Element order inside a row is
// unchanged, so the integer products - and therefore the per-row fmaf fold -
// are identical to the plain layout: results stay BITWISE equal (P4.4 probe
// measured max_rel 0.000e+00 at nx=1). Measured +24.5% at the engine's nx=1
// expert-tile shape, +67% at nx=2.
inline constexpr uint32_t kShadowQ4KPairSbBytes = 560;   // per 2 rows

struct ShadowSpan {
    const uint8_t * base = nullptr;   // packed weights (mmap), span start
    const uint8_t * end  = nullptr;   // packed span end (all experts)
    const uint8_t * shadow = nullptr; // shadow payload for the span
    uint64_t packed_row_bytes = 0;
    uint64_t shadow_row_bytes = 0;    // logical bytes per row (both layouts)
    Fmt      fmt = Fmt::F32;
    bool     smmla = false;           // row-pair interleaved (Q4_K only)
};

// Row index of a resolved packed row pointer inside its span.
inline uint64_t fl_shadow_row_index(const ShadowSpan * s,
                                    const uint8_t * packed_row) {
    return (uint64_t)(packed_row - s->base) / s->packed_row_bytes;
}
// Base of the row PAIR containing row r (SMMLA layout only).
inline const uint8_t * fl_shadow_pair(const ShadowSpan * s, uint64_t r) {
    return s->shadow + (r >> 1) * (s->shadow_row_bytes * 2);
}

// Registry: sorted by base; lookup is one branch when empty.
bool           fl_shadow_enabled();
const ShadowSpan * fl_shadow_find(const void * w);
// Row pointer inside a span for a resolved packed row pointer.
inline const uint8_t * fl_shadow_row(const ShadowSpan * s,
                                     const uint8_t * packed_row) {
    const uint64_t r = (uint64_t)(packed_row - s->base) / s->packed_row_bytes;
    return s->shadow + r * s->shadow_row_bytes;
}

struct ShadowTensorDesc {                // one packed tensor to shadow
    std::string     name;                // cache-file identity
    const uint8_t * p = nullptr;
    Fmt      fmt = Fmt::F32;
    uint32_t cols = 0;
    uint64_t rows_total = 0;             // rows * n_expert (contiguous)
};

// Build (or cache-load) shadows for the given tensors per FASTLLM_SHADOW.
// Returns seconds spent building (0 on cache hit / off). Idempotent.
double fl_shadow_init(const std::vector<ShadowTensorDesc> & tensors);

// Exposed for tests.
bool fl_shadow_layout_smmla();           // FASTLLM_SHADOW_LAYOUT=smmla
void fl_shadow_build_rows_q4k(const uint8_t * packed, uint64_t rows,
                              uint32_t cols, uint8_t * out);
// P4.4: same content as fl_shadow_build_rows_q4k, row-pair interleaved.
// `rows` must be even (callers chunk on even boundaries; rows_total is
// rows_per_expert * n_expert and both are even for every model we load).
void fl_shadow_build_rows_q4k_smmla(const uint8_t * packed, uint64_t rows,
                                    uint32_t cols, uint8_t * out);
void fl_shadow_build_rows_q50(const uint8_t * packed, uint64_t rows,
                              uint32_t cols, uint8_t * out);
void fl_shadow_register_for_test(const ShadowSpan & s);   // tests only
void fl_shadow_clear_for_test();                          // tests only
uint64_t fl_shadow_row_bytes(Fmt f, uint32_t cols);

} // namespace fastllm
