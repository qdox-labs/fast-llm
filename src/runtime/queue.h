// Bounded MPMC ready queue (Vyukov), arena-resident, usable from host and
// device. Values are task ids (uint32). Capacity is a power of two and is
// sized >= n_tasks at Runtime creation, so push never blocks in steady state
// (push still spin-retries on transient full for safety).
#pragma once
#include "atomics.h"

namespace fastllm {

inline constexpr uint32_t kQueueEmpty = 0xFFFFFFFFu;

struct QueueCell {
    uint32_t seq;
    uint32_t val;
};

struct alignas(128) Queue {
    QueueCell * cells;    // arena
    uint32_t    mask;     // capacity-1
    uint32_t    fence_light;  // P7.0: 1 = fence-per-claim pop_bulk (see below)
    alignas(128) uint32_t head; // enqueue cursor
    alignas(128) uint32_t tail; // dequeue cursor

    // host-side init (build time, single-threaded)
    static void init(Queue * q, QueueCell * cells, uint32_t capacity_pow2) {
        q->cells = cells;
        q->mask  = capacity_pow2 - 1;
        q->fence_light = 0;
        q->head  = 0;
        q->tail  = 0;
        for (uint32_t i = 0; i < capacity_pow2; ++i) {
            cells[i].seq = i;
            cells[i].val = kQueueEmpty;
        }
    }

    FL_HD bool push(uint32_t v) {
        uint32_t pos = fl_load_acquire_u32(&head);
        for (;;) {
            QueueCell * c = &cells[pos & mask];
            uint32_t seq  = fl_load_acquire_u32(&c->seq);
            int32_t  dif  = (int32_t)(seq - pos);
            if (dif == 0) {
                if (fl_cas_u32(&head, &pos, pos + 1)) {
                    c->val = v;
                    fl_store_release_u32(&c->seq, pos + 1);
                    return true;
                }
                // pos updated by CAS failure; retry
            } else if (dif < 0) {
                return false; // full (caller spin-retries)
            } else {
                pos = fl_load_acquire_u32(&head);
            }
        }
    }

    // Bulk pop: claim up to maxn contiguous ready cells with ONE tail CAS.
    // Amortizes the system-scope RMW that dominates GPU-side claim cost
    // (P0.6: ~45 us/tile claim+publish at matrix tiles). Returns count (0 if
    // empty). Values are written to out[] in queue order. Correctness: cells
    // are verified ready (seq == pos+i+1) BEFORE the CAS; a competing
    // consumer advancing tail fails our CAS and we rescan. After the CAS the
    // claimed cells are exclusively ours; producers cannot reuse a cell until
    // we release its seq.
    // P7.0 fence-light variant. Same algorithm; the difference is purely where
    // ordering is enforced.
    //
    // WHY: fl_load_acquire_u32 carries a __threadfence_system() per call, and
    // the scan below calls it once per cell examined, so a 16-wide claim issued
    // ~17 system-scope fences. Measured cost: the GPU claims 0.069 Mtask/s
    // against a single CPU thread's 5.2 Mtask/s, and loses every race for work
    // . Fence-light is worth 1.84x at
    // batch 4 and 2.41x at batch 16 in the isolated probe.
    //
    // CORRECTNESS. The contract is: a consumer must observe the producer's
    // `c->val = v` before reading it. The producer publishes with
    // `val = v; store_release(seq)`. So:
    //  - the SCAN only needs to *detect* readiness, and a relaxed load suffices
    //    for that -- it reads either the pre- or post-release seq, and a stale
    //    read merely means we claim fewer cells this round;
    //  - the acquire that matters is the one ordering the eventual `val` read
    //    against that release store. A relaxed load that reads a release store,
    //    followed by an acquire FENCE on the same thread before the payload
    //    read, synchronizes-with the release -- the standard C++11 fence idiom,
    //    exactly as strong as a per-load acquire;
    //  - the tail CAS is the mutual-exclusion point, so after it succeeds the
    //    range [pos, pos+m) is ours exclusively;
    //  - recycling the cells must not be visible before we have read `val`,
    //    so a release FENCE precedes the relaxed seq stores.
    // On the host path fl_cas_u32 is already __ATOMIC_ACQ_REL, so the acquire
    // fence there is redundant-but-harmless.
    FL_HD uint32_t pop_bulk_light(uint32_t * out, uint32_t maxn,
                                  uint32_t * nretry = nullptr) {
        for (;;) {
            uint32_t pos = fl_load_relaxed_u32(&tail);
            uint32_t m = 0;
            while (m < maxn) {
                QueueCell * c = &cells[(pos + m) & mask];
                uint32_t seq  = fl_load_relaxed_u32(&c->seq);
                if ((int32_t)(seq - (pos + m + 1)) != 0) break;
                ++m;
            }
            if (m == 0) {
                QueueCell * c = &cells[pos & mask];
                uint32_t seq  = fl_load_relaxed_u32(&c->seq);
                if ((int32_t)(seq - (pos + 1)) < 0) return 0; // truly empty
                if (nretry) ++*nretry;
                continue;                                     // lost a race
            }
            uint32_t expect = pos;
            if (fl_cas_u32(&tail, &expect, pos + m)) {
                fl_fence_acquire();          // pairs with producer store_release(seq)
                for (uint32_t i = 0; i < m; ++i)
                    out[i] = cells[(pos + i) & mask].val;
                fl_fence_release();          // val reads happen-before recycling
                for (uint32_t i = 0; i < m; ++i)
                    fl_store_relaxed_u32(&cells[(pos + i) & mask].seq,
                                         pos + i + mask + 1);
                return m;
            }
            if (nretry) ++*nretry;                            // CAS lost
        }
    }

    // P8.2: `nretry` (optional) counts loop iterations BEYOND the first. Both
    // the lost-race `continue` and a failed CAS restart the whole loop with no
    // bound, so a consumer that keeps losing does not pay one claim - it pays
    // N. That is the difference between "the claim is slow" and "the claim is
    // repeated", and P8.0 could not tell them apart: it measured 47.6 us per
    // claim without knowing how many passes were inside it.
    FL_HD uint32_t pop_bulk(uint32_t * out, uint32_t maxn,
                            uint32_t * nretry = nullptr) {
        if (fence_light) return pop_bulk_light(out, maxn, nretry);
        for (;;) {
            uint32_t pos = fl_load_acquire_u32(&tail);
            uint32_t m = 0;
            while (m < maxn) {
                QueueCell * c = &cells[(pos + m) & mask];
                uint32_t seq  = fl_load_acquire_u32(&c->seq);
                if ((int32_t)(seq - (pos + m + 1)) != 0) break;
                ++m;
            }
            if (m == 0) {
                QueueCell * c = &cells[pos & mask];
                uint32_t seq  = fl_load_acquire_u32(&c->seq);
                if ((int32_t)(seq - (pos + 1)) < 0) return 0; // truly empty
                if (nretry) ++*nretry;
                continue;                                     // lost a race
            }
            uint32_t expect = pos;
            if (fl_cas_u32(&tail, &expect, pos + m)) {
                for (uint32_t i = 0; i < m; ++i) {
                    QueueCell * c = &cells[(pos + i) & mask];
                    out[i] = c->val;
                    fl_store_release_u32(&c->seq, pos + i + mask + 1);
                }
                return m;
            }
            if (nretry) ++*nretry;                            // CAS lost
        }
    }

    FL_HD bool pop(uint32_t * out) {
        uint32_t pos = fl_load_acquire_u32(&tail);
        for (;;) {
            QueueCell * c = &cells[pos & mask];
            uint32_t seq  = fl_load_acquire_u32(&c->seq);
            int32_t  dif  = (int32_t)(seq - (pos + 1));
            if (dif == 0) {
                if (fl_cas_u32(&tail, &pos, pos + 1)) {
                    *out = c->val;
                    fl_store_release_u32(&c->seq, pos + mask + 1);
                    return true;
                }
            } else if (dif < 0) {
                return false; // empty
            } else {
                pos = fl_load_acquire_u32(&tail);
            }
        }
    }
};

} // namespace fastllm
