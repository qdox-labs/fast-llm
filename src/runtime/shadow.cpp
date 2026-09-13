// P3.2 shadow build / registry / cache. See shadow.h for the contract.
#include "shadow.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fastllm {

namespace {
std::vector<ShadowSpan> g_spans;         // sorted by base
std::atomic<bool>       g_enabled{false};

enum class Mode { OFF, BUILD, CACHE };
Mode mode_from_env() {
    const char * e = getenv("FASTLLM_SHADOW");
    if (!e || !e[0] || !strcmp(e, "off") || !strcmp(e, "0")) return Mode::OFF;
    if (!strcmp(e, "cache")) return Mode::CACHE;
    return Mode::BUILD;                  // "build", "1", anything else
}
std::string cache_path() {
    const char * e = getenv("FASTLLM_SHADOW_FILE");
    if (e && e[0]) return e;
    const char * h = getenv("HOME");
    return std::string(h ? h : ".") + "/.fastllm-shadow.bin";
}

struct CacheHead {                       // followed by n entries + payload
    char     magic[4];                   // "FLSH"
    uint32_t version;                    // 2 (P4.4 added `layout`)
    uint32_t n;
    uint32_t layout;                     // 0 = plain, 1 = smmla row-pair
};
inline constexpr uint32_t kCacheVersion = 2;
struct CacheEnt {
    char     name[64];
    uint32_t fmt;
    uint32_t cols;
    uint64_t rows_total;
    uint64_t shadow_bytes;               // payload bytes for this tensor
};
} // namespace

bool fl_shadow_enabled() { return g_enabled.load(std::memory_order_acquire); }

bool fl_shadow_layout_smmla() {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    const char * e = getenv("FASTLLM_SHADOW_LAYOUT");
    return e && !strcmp(e, "smmla");
#else
    return false;                        // no i8mm: layout would be pointless
#endif
}

const ShadowSpan * fl_shadow_find(const void * w) {
    // spans are few (78 for V2-Lite); sorted upper_bound
    const uint8_t * p = (const uint8_t *)w;
    auto it = std::upper_bound(g_spans.begin(), g_spans.end(), p,
        [](const uint8_t * v, const ShadowSpan & s) { return v < s.base; });
    if (it == g_spans.begin()) return nullptr;
    --it;
    return (p >= it->base && p < it->end) ? &*it : nullptr;
}

uint64_t fl_shadow_row_bytes(Fmt f, uint32_t cols) {
    if (f == Fmt::Q4_K) return (uint64_t)(cols / 256) * kShadowQ4KSbBytes;
    if (f == Fmt::Q5_0) return (uint64_t)(cols / 32) * kShadowQ50BlkBytes;
    return 0;
}

void fl_shadow_build_rows_q4k(const uint8_t * packed, uint64_t rows,
                              uint32_t cols, uint8_t * out) {
    const uint32_t nsb = cols / 256;
    const uint64_t prb = (uint64_t)nsb * 144, srb = (uint64_t)nsb * 280;
    for (uint64_t r = 0; r < rows; ++r) {
        const uint8_t * prow = packed + r * prb;
        uint8_t * srow = out + r * srb;
        for (uint32_t sb = 0; sb < nsb; ++sb) {
            const blk_q4_K * W = (const blk_q4_K *)(prow + (size_t)sb * 144);
            uint8_t * o = srow + (size_t)sb * kShadowQ4KSbBytes;
            const float d = fl_half2float(W->d), dmin = fl_half2float(W->dmin);
            memcpy(o, &d, 4);
            memcpy(o + 4, &dmin, 4);
            fl_q4k_scales_all(W->scales, o + 8, o + 16);
            int8_t * qs = (int8_t *)(o + 24);
            const uint8_t * q = W->qs;
            for (int c = 0; c < 4; ++c) {           // chunk: 32 lo then 32 hi
                for (int i = 0; i < 32; ++i) {
                    qs[c * 64 + i]      = (int8_t)(q[i] & 0x0F);
                    qs[c * 64 + 32 + i] = (int8_t)(q[i] >> 4);
                }
                q += 32;
            }
        }
    }
}

// P4.4: identical content to fl_shadow_build_rows_q4k, row-pair interleaved.
// Per superblock: [r0 hdr 24B][r1 hdr 24B][32 x (r0 8B, r1 8B)].
void fl_shadow_build_rows_q4k_smmla(const uint8_t * packed, uint64_t rows,
                                    uint32_t cols, uint8_t * out) {
    const uint32_t nsb = cols / 256;
    const uint64_t prb = (uint64_t)nsb * 144;
    const uint64_t pairb = (uint64_t)nsb * kShadowQ4KPairSbBytes;
    int8_t tmp[2][256];
    for (uint64_t rp = 0; rp * 2 < rows; ++rp) {
        uint8_t * pbase = out + rp * pairb;
        for (uint32_t sb = 0; sb < nsb; ++sb) {
            uint8_t * o = pbase + (size_t)sb * kShadowQ4KPairSbBytes;
            for (int h = 0; h < 2; ++h) {
                const uint8_t * prow = packed + (rp * 2 + h) * prb;
                const blk_q4_K * W = (const blk_q4_K *)(prow + (size_t)sb * 144);
                uint8_t * hdr = o + h * 24;
                const float d = fl_half2float(W->d), dmin = fl_half2float(W->dmin);
                memcpy(hdr, &d, 4);
                memcpy(hdr + 4, &dmin, 4);
                fl_q4k_scales_all(W->scales, hdr + 8, hdr + 16);
                const uint8_t * q = W->qs;
                for (int c = 0; c < 4; ++c) {       // chunk: 32 lo then 32 hi
                    for (int i = 0; i < 32; ++i) {
                        tmp[h][c * 64 + i]      = (int8_t)(q[i] & 0x0F);
                        tmp[h][c * 64 + 32 + i] = (int8_t)(q[i] >> 4);
                    }
                    q += 32;
                }
            }
            int8_t * qi = (int8_t *)(o + 48);       // 32 x (r0 8B, r1 8B)
            for (int c8 = 0; c8 < 32; ++c8) {
                memcpy(qi + c8 * 16,     tmp[0] + c8 * 8, 8);
                memcpy(qi + c8 * 16 + 8, tmp[1] + c8 * 8, 8);
            }
        }
    }
}

void fl_shadow_build_rows_q50(const uint8_t * packed, uint64_t rows,
                              uint32_t cols, uint8_t * out) {
    const uint32_t nb = cols / 32;
    const uint64_t prb = (uint64_t)nb * 22, srb = (uint64_t)nb * 40;
    for (uint64_t r = 0; r < rows; ++r) {
        const uint8_t * prow = packed + r * prb;
        uint8_t * srow = out + r * srb;
        for (uint32_t b = 0; b < nb; ++b) {
            const blk_q5_0 * W = (const blk_q5_0 *)(prow + (size_t)b * 22);
            uint8_t * o = srow + (size_t)b * kShadowQ50BlkBytes;
            const float d = fl_half2float(W->d);
            memcpy(o, &d, 4);
            int8_t * qs = (int8_t *)(o + 4);
            uint32_t qh; memcpy(&qh, W->qh, 4);
            for (int j = 0; j < 16; ++j) {          // 0..31, offset stays algebraic
                qs[j]      = (int8_t)((W->qs[j] & 0x0F) | (((qh >> j) << 4) & 0x10));
                qs[16 + j] = (int8_t)((W->qs[j] >> 4)   | ((qh >> (j + 12)) & 0x10));
            }
            memset(o + 36, 0, 4);
        }
    }
}

void fl_shadow_register_for_test(const ShadowSpan & s) {
    g_spans.push_back(s);
    std::sort(g_spans.begin(), g_spans.end(),
              [](const ShadowSpan & a, const ShadowSpan & b) {
                  return a.base < b.base;
              });
    g_enabled.store(true, std::memory_order_release);
}
void fl_shadow_clear_for_test() {
    g_spans.clear();
    g_enabled.store(false, std::memory_order_release);
}

static void build_parallel(const ShadowTensorDesc & t, uint8_t * out,
                           bool smmla) {
    const unsigned nt = std::max(2u, std::thread::hardware_concurrency()) - 1;
    uint64_t chunk = (t.rows_total + nt - 1) / nt;
    chunk += (chunk & 1);                // even: SMMLA pairs must not straddle
    const uint64_t prb = fl_row_bytes(t.fmt, t.cols);
    const uint64_t srb = fl_shadow_row_bytes(t.fmt, t.cols);
    std::vector<std::thread> th;
    for (unsigned i = 0; i < nt; ++i) {
        const uint64_t r0 = (uint64_t)i * chunk;
        if (r0 >= t.rows_total) break;
        const uint64_t n = std::min(chunk, t.rows_total - r0);
        th.emplace_back([&, r0, n] {
            const uint8_t * p = t.p + r0 * prb;
            uint8_t * o = out + r0 * srb;
            if (t.fmt == Fmt::Q4_K) {
                // Chunks start on even rows (see the assert in fl_shadow_init),
                // so a pair never straddles two threads.
                if (smmla)
                    fl_shadow_build_rows_q4k_smmla(p, n, t.cols, o);
                else
                    fl_shadow_build_rows_q4k(p, n, t.cols, o);
            } else {
                fl_shadow_build_rows_q50(p, n, t.cols, o);
            }
        });
    }
    for (auto & x : th) x.join();
}

double fl_shadow_init(const std::vector<ShadowTensorDesc> & tensors) {
    if (fl_shadow_enabled()) return 0.0;             // idempotent
    const Mode m = mode_from_env();
    if (m == Mode::OFF) return 0.0;

    bool smmla = fl_shadow_layout_smmla();

    std::vector<ShadowTensorDesc> want;
    uint64_t total = 0;
    for (const auto & t : tensors) {
        if (t.fmt != Fmt::Q4_K && t.fmt != Fmt::Q5_0) continue;  // q8_0: no win
        // SMMLA pairs rows; an odd row count would leave a half-pair at the
        // span end that the pair kernel cannot address. Fall back wholesale
        // rather than build a layout the kernels must special-case.
        if (smmla && t.fmt == Fmt::Q4_K && (t.rows_total & 1)) {
            fprintf(stderr, "shadow: %s has odd rows_total %llu - "
                            "SMMLA layout disabled\n",
                    t.name.c_str(), (unsigned long long)t.rows_total);
            smmla = false;
        }
        want.push_back(t);
        total += t.rows_total * fl_shadow_row_bytes(t.fmt, t.cols);
    }
    if (want.empty()) return 0.0;

    uint8_t * payload = nullptr;
    double build_s = 0.0;
    bool from_cache = false;

    if (m == Mode::CACHE) {
        const std::string path = cache_path();
        int fd = open(path.c_str(), O_RDONLY);
        if (fd >= 0) {
            struct stat st{};
            fstat(fd, &st);
            const uint64_t need = sizeof(CacheHead)
                                + want.size() * sizeof(CacheEnt) + total;
            if ((uint64_t)st.st_size == need) {
                void * mp = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
                if (mp != MAP_FAILED) {
                    const CacheHead * h = (const CacheHead *)mp;
                    const CacheEnt * e = (const CacheEnt *)(h + 1);
                    // P4.4: layout is part of cache identity. A plain cache
                    // read as SMMLA (or vice versa) is silent corruption, so
                    // a mismatch invalidates and rebuilds - never reinterprets.
                    bool ok = !memcmp(h->magic, "FLSH", 4)
                              && h->version == kCacheVersion
                              && h->n == want.size()
                              && h->layout == (uint32_t)(smmla ? 1 : 0);
                    for (uint32_t i = 0; ok && i < h->n; ++i)
                        ok = !strncmp(e[i].name, want[i].name.c_str(), 63)
                          && e[i].fmt == (uint32_t)want[i].fmt
                          && e[i].cols == want[i].cols
                          && e[i].rows_total == want[i].rows_total;
                    if (ok) {
                        payload = (uint8_t *)mp + sizeof(CacheHead)
                                + h->n * sizeof(CacheEnt);
                        from_cache = true;
                    } else {
                        munmap(mp, st.st_size);
                    }
                }
            }
            close(fd);
        }
    }

    if (!payload) {
        const auto t0 = std::chrono::steady_clock::now();
        payload = (uint8_t *)malloc(total);
        if (!payload) {
            fprintf(stderr, "shadow: alloc %.2f GB failed - disabled\n",
                    total / 1e9);
            return 0.0;
        }
        uint64_t off = 0;
        for (const auto & t : want) {
            build_parallel(t, payload + off, smmla);
            off += t.rows_total * fl_shadow_row_bytes(t.fmt, t.cols);
        }
        build_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        if (m == Mode::CACHE) {
            const std::string path = cache_path(), tmp = path + ".tmp";
            FILE * f = fopen(tmp.c_str(), "wb");
            if (f) {
                CacheHead h{}; memcpy(h.magic, "FLSH", 4);
                h.version = kCacheVersion; h.n = (uint32_t)want.size();
                h.layout = smmla ? 1u : 0u;
                fwrite(&h, sizeof h, 1, f);
                for (const auto & t : want) {
                    CacheEnt e{}; strncpy(e.name, t.name.c_str(), 63);
                    e.fmt = (uint32_t)t.fmt; e.cols = t.cols;
                    e.rows_total = t.rows_total;
                    e.shadow_bytes = t.rows_total
                                   * fl_shadow_row_bytes(t.fmt, t.cols);
                    fwrite(&e, sizeof e, 1, f);
                }
                fwrite(payload, 1, total, f);
                fclose(f);
                rename(tmp.c_str(), path.c_str());
            }
        }
    }

    uint64_t off = 0;
    for (const auto & t : want) {
        ShadowSpan s;
        const uint64_t prb = fl_row_bytes(t.fmt, t.cols);
        s.base = t.p;
        s.end  = t.p + t.rows_total * prb;
        s.shadow = payload + off;
        s.packed_row_bytes = prb;
        s.shadow_row_bytes = fl_shadow_row_bytes(t.fmt, t.cols);
        s.fmt = t.fmt;
        s.smmla = smmla && t.fmt == Fmt::Q4_K;
        g_spans.push_back(s);
        off += t.rows_total * s.shadow_row_bytes;
    }
    std::sort(g_spans.begin(), g_spans.end(),
              [](const ShadowSpan & a, const ShadowSpan & b) {
                  return a.base < b.base;
              });
    g_enabled.store(true, std::memory_order_release);
    fprintf(stderr, "shadow: %zu spans, %.2f GB %s layout=%s (%.2f s)\n",
            g_spans.size(), total / 1e9,
            from_cache ? "mmap-cache" : "built",
            smmla ? "smmla" : "plain", build_s);
    return build_s;
}

} // namespace fastllm
