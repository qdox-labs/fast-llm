// Runtime: persistent CPU pool + (optionally) persistent GPU grid, one shared
// dataflow protocol over unified memory. No launches, syncs, or copies on the
// token path: run() resets counters, publishes roots, and waits on a flag.
#include "runtime_internal.h"
#include "kernels.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace fastllm {

namespace {

inline uint64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

inline constexpr uint32_t kQueueCapacity = 65536; // cells per queue, pow2

struct alignas(128) WorkerMetrics {
    EngineMetrics m;
    uint64_t kind_ns[kNumTaskKinds] = {};
    uint64_t kind_n[kNumTaskKinds]  = {};
};

class RuntimeImpl final : public Runtime {
public:
    RuntimeImpl(Arena & arena, const RuntimeConfig & cfg)
        : cfg_(cfg), arena_(&arena) {
#if !defined(FASTLLM_CUDA)
        cfg_.gpu_enabled = false;
#endif
        // P6.3: sh_ and the four queue-cell arrays are consecutive bump
        // allocations, so they form ONE contiguous range - the exact working
        // set every GPU block polls. Record its extent for the L2
        // persisting-window probe (arena.used() delta, so it stays correct if
        // the allocations change).
        const size_t arena_before = arena.used();
        sh_ = (RtShared *)arena.alloc(sizeof(RtShared), 128);
        memset((void *)sh_, 0, sizeof(RtShared));
        // P7.0: fence-light pop_bulk. Default OFF - this is a memory-model
        // change and the class has bitten us before (P1.0 silent corruption).
        uint32_t fl = 0;
        if (const char * e = getenv("FASTLLM_QUEUE_FENCE_LIGHT")) fl = atoi(e) ? 1u : 0u;
        for (int i = 0; i < 4; ++i) {
            auto * cells = (QueueCell *)arena.alloc(sizeof(QueueCell) * kQueueCapacity, 128);
            Queue::init(&sh_->q[i], cells, kQueueCapacity);
            sh_->q[i].fence_light = fl;
        }
        if (fl) fprintf(stderr, "fastllm: queue fence-light ON\n");
        persist_bytes_ = arena.used() - arena_before;
        sh_->gpu_enabled     = cfg_.gpu_enabled ? 1 : 0;
        sh_->collect_handoff = cfg_.collect_handoff ? 1 : 0;
        sh_->fast_kernels    = cfg_.fast_kernels ? 1 : 0;
        sh_->int8_dots       = cfg_.int8_dots ? 1 : 0;
        uint32_t gm = kI8MaskGpuDefault, cm = kI8MaskCpuDefault;
        if (const char * e = getenv("FASTLLM_I8_GPU_MASK")) gm = (uint32_t)strtoul(e, nullptr, 0);
        if (const char * e = getenv("FASTLLM_I8_CPU_MASK")) cm = (uint32_t)strtoul(e, nullptr, 0);
        // P5.2: fl_dot_i8_any has no codebook case and returns 0 for one, so a
        // codebook bit in the CPU int8 mask would silently zero those tiles.
        // Routing keeps them off the pool (q[3]), but a cpu-only run or an env
        // override must not be able to reach that path: clear the bits here.
        const uint32_t cb_bits = (1u << (int)Fmt::IQ2_XXS) | (1u << (int)Fmt::IQ3_XXS);
        cm &= ~cb_bits;
        sh_->i8_gpu_mask     = cfg_.int8_dots ? gm : 0;
        sh_->i8_cpu_mask     = cfg_.int8_dots ? cm : 0;
        uint32_t gk = kI8KMaskGpuDefault, ck = kI8KMaskCpuDefault;
        if (const char * e = getenv("FASTLLM_I8K_GPU_MASK")) gk = (uint32_t)strtoul(e, nullptr, 0);
        if (const char * e = getenv("FASTLLM_I8K_CPU_MASK")) ck = (uint32_t)strtoul(e, nullptr, 0);
        sh_->i8k_gpu_mask    = cfg_.int8_dots ? gk : 0;
        sh_->i8k_cpu_mask    = cfg_.int8_dots ? ck : 0;
        {   // P8.5: strict static assignment (no cross-engine stealing).
            uint32_t ns = 0;
            if (const char * e = getenv("FASTLLM_NO_STEAL")) ns = atoi(e) ? 1u : 0u;
            sh_->no_steal = ns;
            if (ns) fprintf(stderr, "fastllm: STRICT assignment - no stealing\n");
        }
        {   // P9.2c: per-block GPU rings. Default OFF - this changes who
            // consumes what, and an orphaned ring is a hang, not a wrong number.
            uint32_t pb = 0;
            if (const char * e = getenv("FASTLLM_PERBLOCK_Q")) pb = atoi(e) ? 1u : 0u;
            sh_->perblock_q = (pb && cfg_.gpu_enabled) ? 1u : 0u;
            uint32_t ch = 1;
            if (const char * e = getenv("FASTLLM_PERBLOCK_CHUNK")) {
                const long v = strtol(e, nullptr, 10);
                if (v > 0) ch = (uint32_t)v;
            }
            sh_->perblock_chunk = ch;
            if (pb && !cfg_.gpu_enabled)
                fprintf(stderr, "fastllm: FASTLLM_PERBLOCK_Q ignored (gpu off)\n");
        }
        {   // P8.0: GPU time decomposition + per-kind dot attribution. Off by
            // default so the shipped kernel path is unchanged; the cost of
            // turning it on is itself an A/B (see docs/BENCH.md P8.0).
            uint32_t pg = 0;
            if (const char * e = getenv("FASTLLM_PROF_GPU")) pg = atoi(e) ? 1u : 0u;
            sh_->prof_gpu = pg;
            if (pg) fprintf(stderr, "fastllm: GPU profiling ON "
                                    "(time split + per-kind dot)\n");
        }
        {   // P2.5: same-binary attribution switch for multi-column kernels
            const char * e = getenv("FASTLLM_MC_LEGACY");
            sh_->mc_legacy = (e && e[0] == '1') ? 1u : 0u;
        }
        sh_->gpu_batch       = (uint32_t)std::max(1, std::min(cfg_.gpu_batch,
                                                              (int)kGpuMaxBatch));
        wm_ = new WorkerMetrics[cfg_.cpu_threads]();
    }

    ~RuntimeImpl() override {
        stop();
        delete[] wm_;
    }

    void start() override {
        if (started_) return;
        fl_store_release_u32(&sh_->run_flag, 1);
#if defined(FASTLLM_CUDA)
        if (cfg_.gpu_enabled) {
            gpu_stream_ = gpu_make_stream();
            sh_->gpu_clock_offset = gpu_calibrate_clock(sh_, gpu_stream_);
            int blocks = cfg_.gpu_blocks > 0 ? cfg_.gpu_blocks : gpu_default_blocks();
            // P4.0: occupancy is the GPU limiter (ncu: 16.6% achieved, 8 warps
            // /SM at 20x256; the fmt probe lifts 3.5x at 20x1024). Env
            // overrides exist so geometry can be A/B'd in one binary.
            if (const char * e = getenv("FASTLLM_GPU_BLOCKS")) { int v = atoi(e); if (v > 0) blocks = v; }
            blocks = std::min(blocks, (int)kMaxGpuBlocks);
            int threads = cfg_.gpu_threads;
            if (const char * e = getenv("FASTLLM_GPU_THREADS")) { int v = atoi(e); if (v > 0) threads = v; }
            if (threads < 256) threads = 256;
            if (threads > 1024) threads = 1024;
            threads &= ~31;                 // warp multiple (P1.4 occupancy fix)
            // P9.2c: rings MUST be initialised before the launch. gpu_start
            // starts the persistent kernel, and its blocks read
            // gq[blockIdx.x] on their very first poll - against a memset-zero
            // Queue that would be cells == nullptr, mask == 0, i.e. a null
            // deref inside pop_bulk that kills the kernel silently (the CPU
            // then absorbs 100% of the work and output still looks correct,
            // which is exactly how this was missed until the soak).
            // `blocks` here is the PRE-clamp count and gpu_start only ever
            // clamps DOWN, so every block it launches is guaranteed a ring.
            // Capacity stays at kQueueCapacity so push can never block
            // (queue.h's invariant is capacity >= n_tasks; m=8 is 35,527).
            // These land AFTER persist_bytes_ was computed, so the P6.3 L2
            // window keeps covering only [RtShared | q[0..3]] (2.08 MB)
            // rather than ballooning past the device's 24 MiB limit.
            if (sh_->perblock_q) {
                for (int b = 0; b < blocks; ++b) {
                    auto * cells = (QueueCell *)arena_->alloc(
                        sizeof(QueueCell) * kQueueCapacity, 128);
                    Queue::init(&sh_->gq[b], cells, kQueueCapacity);
                    sh_->gq[b].fence_light = sh_->q[0].fence_light;
                }
            }
            // P9.2: gpu_start clamps to the driver's residency cap, so the
            // launched grid is what it RETURNS, not what we asked for.
            gpu_blocks_ = gpu_start(sh_, blocks, threads, gpu_stream_,
                                    persist_bytes_);
            if (sh_->perblock_q)
                fprintf(stderr, "fastllm: per-block queues ON (%d rings, "
                                "%d launched)\n", blocks, gpu_blocks_);
        }
#endif
        for (int i = 0; i < cfg_.cpu_threads; ++i) {
            threads_.emplace_back([this, i] { worker(i); });
        }
        started_ = true;
    }

    RunMetrics run(Graph & g) override {
        Task * tasks = g.tasks();
        const uint32_t n = g.size();

        // -- reset (quiescent: previous run fully drained via n_done) --
        sh_->tasks    = tasks;
        sh_->n_tasks  = n;
        sh_->terminal = g.terminal();
        sh_->done_flag = 0;
        sh_->n_done    = 0;
        sh_->n_samples = 0;
        // (P8.0: the gpu_m / gpu_kind_* memsets that used to sit here raced the
        // still-running executor and have been replaced by the snapshot below.)
        const uint64_t tr0 = now_ns();
        for (uint32_t i = 0; i < n; ++i) {
            tasks[i].dep_count  = tasks[i].dep_init;
            tasks[i].handoff_ts = 0;
        }
        const uint64_t reset_ns = now_ns() - tr0;
        std::vector<WorkerMetrics> snap(cfg_.cpu_threads);
        for (int i = 0; i < cfg_.cpu_threads; ++i) snap[i] = wm_[i];
        // P8.0: same idiom for the GPU side, taken as late as possible so the
        // counted span lines up with wall_ns. 8-byte aligned loads of
        // monotonically increasing counters, so a concurrent device increment
        // can only be missed, never torn - and being missed just moves that
        // increment into this window, where it belongs.
        memcpy((void *)gsnap_,     (const void *)sh_->gpu_m,       sizeof(gsnap_));
        memcpy((void *)gkns_snap_, (const void *)sh_->gpu_kind_ns, sizeof(gkns_snap_));
        memcpy((void *)gkn_snap_,  (const void *)sh_->gpu_kind_n,  sizeof(gkn_snap_));

        // -- publish roots, open the active window, wake the pool --
        uint64_t t0 = now_ns();
        fl_store_release_u32(&sh_->active, 1);
        {
            std::lock_guard<std::mutex> lk(mu_);
            wake_gen_++;
        }
        cv_.notify_all();
        for (uint32_t i = 0; i < n; ++i) {
            if (tasks[i].dep_init == 0) publish(i);
        }

        // -- wait: terminal fired AND every publish loop finished --
        // P4.1: the deadline is 60 s, so reading the clock every spin is pure
        // waste - perf attributed 7.5% of task-clock to __kernel_clock_gettime
        // from this loop alone (the host burns a core that the 12 workers +
        // GPU leader are contending for on 14 cores). Check once per
        // kSpinCheck spins instead; stall detection is unchanged in effect.
        // FASTLLM_SPIN_CHECK=1 restores per-iteration checking for A/Bs.
        const uint64_t deadline = t0 + 60ull * 1000000000ull;
        static const uint32_t spin_check = [] {
            const char * e = getenv("FASTLLM_SPIN_CHECK");
            const uint32_t v = (e && e[0]) ? (uint32_t)strtoul(e, nullptr, 10) : 4096u;
            return v ? v : 1u;   // 0 would check every spin (== legacy); clamp
        }();
        uint32_t spins = 0;
        while (!(fl_load_acquire_u32(&sh_->done_flag) &&
                 fl_load_acquire_u32(&sh_->n_done) == n)) {
            fl_cpu_relax();
            if (++spins >= spin_check) {
                spins = 0;
                if (now_ns() > deadline) dump_stall_and_abort(tasks, n);
            }
        }
        uint64_t t1 = now_ns();
        fl_store_release_u32(&sh_->active, 0);

        // -- collect --
        RunMetrics m{};
        m.wall_ns  = t1 - t0;
        m.reset_ns = reset_ns;
        for (int i = 0; i < cfg_.cpu_threads; ++i) {
            m.cpu.tiles   += wm_[i].m.tiles   - snap[i].m.tiles;
            m.cpu.bytes   += wm_[i].m.bytes   - snap[i].m.bytes;
            m.cpu.busy_ns += wm_[i].m.busy_ns - snap[i].m.busy_ns;
            m.cpu.stolen  += wm_[i].m.stolen  - snap[i].m.stolen;
            for (int k = 0; k < kNumTaskKinds; ++k) {
                m.kind_ns[k] += wm_[i].kind_ns[k] - snap[i].kind_ns[k];
                m.kind_n[k]  += wm_[i].kind_n[k]  - snap[i].kind_n[k];
            }
        }
        for (int b = 0; b < kMaxGpuBlocks; ++b) {
            const BlockMetrics & g = sh_->gpu_m[b];   // now, vs gsnap_ = then
            m.gpu.tiles   += g.tiles   - gsnap_[b].tiles;
            m.gpu.bytes   += g.bytes   - gsnap_[b].bytes;
            m.gpu.busy_ns += g.busy_ns - gsnap_[b].busy_ns;
            m.gpu.stolen  += g.stolen  - gsnap_[b].stolen;
            // P8.0 (all zero unless prof_gpu). polls counts loop iterations, so
            // it is the honest test for "did this block turn over at all" - a
            // block can spin the whole window without ever claiming a tile.
            const uint64_t dp = g.polls - gsnap_[b].polls;
            m.gpu_claim_ns += g.claim_ns - gsnap_[b].claim_ns;
            m.gpu_sleep_ns += g.sleep_ns - gsnap_[b].sleep_ns;
            m.gpu_exec_ns  += g.exec_ns  - gsnap_[b].exec_ns;
            m.gpu_epi_ns   += g.epi_ns   - gsnap_[b].epi_ns;
            m.gpu_polls    += dp;
            m.gpu_empty    += g.empty - gsnap_[b].empty;
            m.gpu_retry    += g.retry - gsnap_[b].retry;
            if (dp) m.gpu_blocks_active++;
            for (int k = 0; k < kNumTaskKinds; ++k) {
                m.gpu_kind_ns[k] += sh_->gpu_kind_ns[b][k] - gkns_snap_[b][k];
                m.gpu_kind_n[k]  += sh_->gpu_kind_n[b][k]  - gkn_snap_[b][k];
            }
        }
        uint32_t ns = std::min(sh_->n_samples, kMaxHandoff);
        if (ns) {
            std::vector<uint64_t> s(sh_->samples, sh_->samples + ns);
            std::sort(s.begin(), s.end());
            m.handoff_samples = ns;
            m.handoff_ns_p50  = s[ns / 2];
            m.handoff_ns_p95  = s[(size_t)(ns * 0.95)];
            m.handoff_ns_max  = s.back();
        }
        return m;
    }

    void stop() override {
        if (!started_) return;
        fl_store_release_u32(&sh_->run_flag, 0);
        {
            std::lock_guard<std::mutex> lk(mu_);
            wake_gen_++;
        }
        cv_.notify_all();
        for (auto & t : threads_) t.join();
        threads_.clear();
#if defined(FASTLLM_CUDA)
        if (cfg_.gpu_enabled && gpu_stream_) {
            gpu_stop_sync(gpu_stream_);
            gpu_stream_ = nullptr;
        }
#endif
        started_ = false;
    }

private:
    // roots from the host thread; stamped as a CPU producer inside fl_publish
    void publish(uint32_t id) { fl_publish(sh_, id, /*from_gpu=*/false); }

    uint64_t to_cpu_domain(uint64_t ts) const {
        if (ts & kTsGpuBit) return (uint64_t)((int64_t)(ts & ~kTsGpuBit) - sh_->gpu_clock_offset);
        return ts;
    }

    void record_handoff(uint64_t enable_ts, uint64_t claim_ts) {
        uint64_t ec = to_cpu_domain(enable_ts);
        if (claim_ts <= ec) return; // clock skew guard
        uint32_t idx = fl_fetch_add_u32(&sh_->n_samples, 1);
        if (idx < kMaxHandoff) sh_->samples[idx] = claim_ts - ec;
    }

    void exec_and_publish(int wid, uint32_t id, bool stolen) {
        Task & t = sh_->tasks[id];
        uint64_t claim = now_ns();
        if (sh_->collect_handoff && t.handoff_ts) {
            record_handoff(t.handoff_ts, claim);
            t.handoff_ts = 0;
        }
        exec_task_cpu(t, sh_->fast_kernels != 0, sh_->i8_cpu_mask,
                      sh_->i8k_cpu_mask);
        uint64_t end = now_ns();
        auto & lm = wm_[wid].m;
        lm.tiles++;
        lm.busy_ns += end - claim;
        if (t.kind == TaskKind::STREAM_DOT
            || t.kind == TaskKind::STREAM_DOT_IDX) {
            uint64_t wb = t.w_bytes;
            // P3.2 honest accounting: CPU tiles served from the int8 shadow
            // stream shadow bytes, not packed bytes (~1.9x q4_K / ~1.8x
            // q5_0). The span ratio is expert-independent, so t.w (tensor
            // base for IDX, row ptr for DOT) resolves the span either way.
            if (fl_shadow_enabled()
                && (t.fmt == Fmt::Q4_K || t.fmt == Fmt::Q5_0)) {
                if (const ShadowSpan * sp = fl_shadow_find(t.w))
                    wb = wb * sp->shadow_row_bytes / sp->packed_row_bytes;
            }
            lm.bytes += wb;
        }
        wm_[wid].kind_ns[(int)t.kind] += end - claim;
        wm_[wid].kind_n[(int)t.kind]++;
        if (stolen) lm.stolen++;

        if (id == sh_->terminal) {
            fl_store_release_u32(&sh_->done_flag, 1);
        }
        for (uint16_t k = 0; k < t.n_consumers; ++k) {
            uint32_t cid = t.consumers[k];
            Task & c = sh_->tasks[cid];
            if (fl_fetch_sub_i32(&c.dep_count, 1) == 1) {
                if (!fl_publish(sh_, cid, /*from_gpu=*/false)) return; // tearing down
            }
        }
        fl_fetch_add_u32(&sh_->n_done, 1);
    }

    void worker(int wid) {
        // P9.2c: per-worker start offset so 12 thieves do not all probe ring 0
        uint32_t rr = (uint32_t)wid;
        while (fl_load_acquire_u32(&sh_->run_flag)) {
            if (fl_load_acquire_u32(&sh_->active)) {
                uint32_t id;
                if (sh_->q[2].pop(&id))      { exec_and_publish(wid, id, false); continue; }
                if (sh_->q[1].pop(&id))      { exec_and_publish(wid, id, false); continue; }
                // P8.5: q[0] is the GPU's queue. Stealing from it is what
                // turns a precomputed assignment into a contested one - but
                // P8.5 also measured that REMOVING the steal costs -39% to
                // -52%, so it stays. With per-block rings the steal has to
                // sweep every ring or one straggler stalls the whole run.
                if (!sh_->no_steal) {
                    if (sh_->perblock_q) {
                        const uint32_t nb = sh_->gpu_blocks;
                        bool got = false;
                        for (uint32_t k = 0; k < nb && !got; ++k) {
                            const uint32_t b = (rr + k) % nb;
                            if (sh_->gq[b].pop(&id)) { rr = b + 1; got = true; }
                        }
                        if (got) { exec_and_publish(wid, id, true); continue; }
                    } else if (sh_->q[0].pop(&id)) {
                        exec_and_publish(wid, id, true); continue;
                    }
                }
                fl_cpu_relax();
                continue;
            }
            // idle window: spin briefly, then park (llama.cpp parking lesson)
            uint64_t spin_until = now_ns() + (uint64_t)cfg_.spin_us_park * 1000;
            bool went_active = false;
            while (now_ns() < spin_until) {
                if (fl_load_acquire_u32(&sh_->active) ||
                    !fl_load_acquire_u32(&sh_->run_flag)) { went_active = true; break; }
                fl_cpu_relax();
            }
            if (went_active) continue;
            std::unique_lock<std::mutex> lk(mu_);
            uint64_t gen = wake_gen_;
            cv_.wait(lk, [&] {
                // P9.2: `active` MUST be in the predicate. Without it a worker
                // that reads active==0, then loses the race to run()'s
                // active=1 + wake_gen_++ before taking mu_, snapshots the NEW
                // generation and sleeps through the entire run. Benign while
                // every queue has 12 other consumers; a guaranteed stall once
                // any consumer owns a private queue.
                return wake_gen_ != gen
                    || fl_load_acquire_u32(&sh_->active)
                    || !fl_load_acquire_u32(&sh_->run_flag);
            });
        }
    }

    [[noreturn]] void dump_stall_and_abort(Task * tasks, uint32_t n) {
        fprintf(stderr, "fastllm: run() stalled. n_done=%u/%u done_flag=%u\n",
                sh_->n_done, n, sh_->done_flag);
        uint32_t blocked = 0;
        for (uint32_t i = 0; i < n; ++i) {
            int32_t d = __atomic_load_n(&tasks[i].dep_count, __ATOMIC_ACQUIRE);
            if (d > 0) { fprintf(stderr, "  task %u dep_count=%d\n", i, d); ++blocked; }
        }
        // P9.2: the loop above only sees tasks still WAITING on a dependency.
        // The failure mode a per-consumer queue introduces is the opposite - a
        // task that became ready, was pushed to a queue nobody pops, and sits
        // there with dep_count == 0. It prints nothing, so a stall would show
        // an empty task list and look like a mystery. Dump the queues too:
        // a non-empty queue with n_done < n names the orphan's location, and
        // head==tail everywhere means the work was never enqueued at all.
        if (!blocked) fprintf(stderr, "  (no task is dependency-blocked: the "
                                      "stalled work is enqueued but unclaimed)\n");
        for (int i = 0; i < 4; ++i) {
            const uint32_t h = __atomic_load_n(&sh_->q[i].head, __ATOMIC_ACQUIRE);
            const uint32_t t = __atomic_load_n(&sh_->q[i].tail, __ATOMIC_ACQUIRE);
            fprintf(stderr, "  q[%d] head=%u tail=%u depth=%u\n", i, h, t, h - t);
        }
        fprintf(stderr, "  gpu_blocks=%u (launched) perblock_q=%u\n",
                sh_->gpu_blocks, sh_->perblock_q);
        for (uint32_t b = 0; sh_->perblock_q && b < sh_->gpu_blocks
                             && b < kMaxGpuBlocks; ++b) {
            const uint32_t h = __atomic_load_n(&sh_->gq[b].head, __ATOMIC_ACQUIRE);
            const uint32_t t = __atomic_load_n(&sh_->gq[b].tail, __ATOMIC_ACQUIRE);
            if (h != t) fprintf(stderr, "  gq[%u] head=%u tail=%u DEPTH=%u"
                                        " (orphaned work)\n", b, h, t, h - t);
        }
        for (uint32_t b = 0; b < sh_->gpu_blocks && b < kMaxGpuBlocks; ++b) {
            fprintf(stderr, "    block %u arrived=%u polls=%llu tiles=%llu\n", b,
                    sh_->gpu_arrived[b],
                    (unsigned long long)sh_->gpu_m[b].polls,
                    (unsigned long long)sh_->gpu_m[b].tiles);
        }
        abort();
    }

    RuntimeConfig cfg_;
    Arena * arena_ = nullptr;      // P9.2c: rings are sized in start()
    void * gpu_stream_ = nullptr;
    RtShared * sh_ = nullptr;
    size_t persist_bytes_ = 0;      // P6.3: [RtShared | q cells] extent
    WorkerMetrics * wm_ = nullptr;
    std::vector<std::thread> threads_;
    std::mutex mu_;
    std::condition_variable cv_;
    uint64_t wake_gen_ = 0;
    int  gpu_blocks_ = 0;
    bool started_ = false;
    // P8.0: pre-window snapshot of the GPU-side counters. The persistent
    // executor never stops - it spins claim->sleep between runs - so the old
    // host-side memset of gpu_m raced thread 0 of every block. Harmless while
    // only claim-time fields existed (a block writes those only when it claims
    // work, which does not happen between runs), fatal once per-poll counters
    // were added: the memset was losing the race and pre-window spin time
    // leaked into the window. Snapshot-and-difference instead, exactly as the
    // CPU pool has always done with WorkerMetrics.
    BlockMetrics gsnap_[kMaxGpuBlocks];
    uint64_t gkns_snap_[kMaxGpuBlocks][kNumTaskKinds];
    uint32_t gkn_snap_[kMaxGpuBlocks][kNumTaskKinds];
};

} // namespace

std::unique_ptr<Runtime> Runtime::create(Arena & arena, const RuntimeConfig & cfg) {
    return std::make_unique<RuntimeImpl>(arena, cfg);
}

} // namespace fastllm
