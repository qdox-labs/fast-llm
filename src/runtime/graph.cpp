// Graph: arena-resident task table built once, validated at freeze(), reused
#include <chrono>
// every run. Validation is host-side scratch; nothing here is on the token path.
#include "runtime_internal.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace fastllm {

// THE single host instantiation of the canonical executor (see
// runtime_internal.h): all host callers link this symbol.
__attribute__((noinline)) void exec_task_scalar_host(Task & t) {
    exec_task_scalar(t);
}

namespace {

class GraphImpl final : public Graph {
public:
    GraphImpl(Arena & a, uint32_t max_tasks) : max_(max_tasks) {
        tasks_ = (Task *)a.alloc(sizeof(Task) * max_tasks, alignof(Task));
    }

    uint32_t add(const Task & t) override {
        if (frozen_) die("add() after freeze()");
        if (n_ >= max_) die("graph capacity exceeded");
        Task & d = tasks_[n_];
        d = t;
        d.id = n_;
        d.n_consumers = 0;
        d.dep_init = 0;
        d.dep_count = 0;
        d.epoch = 0;
        d.handoff_ts = 0;
        return n_++;
    }

    void edge(uint32_t p, uint32_t c) override {
        if (frozen_) die("edge() after freeze()");
        if (p >= n_ || c >= n_ || p == c) die("bad edge");
        Task & tp = tasks_[p];
        if (tp.n_consumers >= kMaxConsumers) die("consumer fan-out exceeded");
        tp.consumers[tp.n_consumers++] = c;
        tasks_[c].dep_init++;
    }

    void freeze() override {
        if (frozen_) return;
        // Kahn: cycle check + find the unique sink, which must be SIGNAL
        std::vector<int32_t> deps(n_);
        std::vector<uint32_t> ready;
        for (uint32_t i = 0; i < n_; ++i) {
            deps[i] = tasks_[i].dep_init;
            if (deps[i] == 0) ready.push_back(i);
        }
        uint32_t seen = 0;
        int sinks = 0;
        while (!ready.empty()) {
            uint32_t id = ready.back();
            ready.pop_back();
            seen++;
            Task & t = tasks_[id];
            if (t.n_consumers == 0) {
                sinks++;
                terminal_ = id;
            }
            for (uint16_t k = 0; k < t.n_consumers; ++k) {
                if (--deps[t.consumers[k]] == 0) ready.push_back(t.consumers[k]);
            }
        }
        if (seen != n_) die("cycle in graph");
        if (sinks != 1) die("graph must have exactly one sink (terminal SIGNAL)");
        if (tasks_[terminal_].kind != TaskKind::SIGNAL) die("terminal must be SIGNAL");
        for (uint32_t i = 0; i < n_; ++i) {
            Task & t = tasks_[i];
            if (t.kind != TaskKind::STREAM_DOT
                && t.kind != TaskKind::STREAM_DOT_IDX) {
                if ((int)t.kind >= kNumTaskKinds) die("unknown task kind");
                if (t.kind >= TaskKind::RMSNORM && t.aux == nullptr)
                    die("decode op without aux param block");
                continue;
            }
            if (t.kind == TaskKind::STREAM_DOT_IDX
                && (t.aux == nullptr || t.aux2 == 0))
                die("STREAM_DOT_IDX without id slot / stride");
            if (t.cols % fl_blk_elems(t.fmt) != 0)
                die("STREAM_DOT cols not a multiple of the format block size");
            if (t.w_bytes != (uint64_t)t.rows * fl_row_bytes(t.fmt, t.cols))
                die("STREAM_DOT w_bytes mismatch");
        }
        frozen_ = true;
    }

    Task *   tasks() override        { return tasks_; }
    uint32_t size() const override   { return n_; }
    uint32_t terminal() const override { return terminal_; }

private:
    [[noreturn]] static void die(const char * msg) {
        fprintf(stderr, "fastllm graph: %s\n", msg);
        abort();
    }
    Task *   tasks_ = nullptr;
    uint32_t max_ = 0, n_ = 0, terminal_ = 0;
    bool     frozen_ = false;
};

} // namespace

std::unique_ptr<Graph> Graph::create(Arena & a, uint32_t max_tasks) {
    return std::make_unique<GraphImpl>(a, max_tasks);
}

// ---- serial reference executor -------------------------------------------
RunMetrics run_serial(Graph & g) {
    RunMetrics m{};
    Task * tasks = g.tasks();
    const uint32_t n = g.size();
    std::vector<int32_t> deps(n);
    std::vector<uint32_t> ready;
    for (uint32_t i = 0; i < n; ++i) {
        deps[i] = tasks[i].dep_init;
        if (deps[i] == 0) ready.push_back(i);
    }
    auto t0 = std::chrono::steady_clock::now();
    while (!ready.empty()) {
        uint32_t id = ready.back();
        ready.pop_back();
        Task & t = tasks[id];
        auto k0 = std::chrono::steady_clock::now();
        exec_task_scalar_host(t);
        auto k1 = std::chrono::steady_clock::now();
        m.cpu.tiles++;
        m.cpu.bytes += (t.kind == TaskKind::STREAM_DOT
                     || t.kind == TaskKind::STREAM_DOT_IDX) ? t.w_bytes : 0;
        m.kind_ns[(int)t.kind] += (uint64_t)std::chrono::duration_cast<
            std::chrono::nanoseconds>(k1 - k0).count();
        m.kind_n[(int)t.kind]++;
        for (uint16_t k = 0; k < t.n_consumers; ++k) {
            if (--deps[t.consumers[k]] == 0) ready.push_back(t.consumers[k]);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    m.wall_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    m.cpu.busy_ns = m.wall_ns;
    return m;
}

} // namespace fastllm
