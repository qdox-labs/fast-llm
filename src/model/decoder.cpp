// P2.0 decoder implementation. Wiring follows llama.cpp's llm_build_deepseek2
// (non-MLA branch: MHA with per-head K = [k_nope | k_pe], V = v_states) and
// its YaRN/kq_scale convention; see docs/BENCH.md P2.0 for the derivation.
#include "model/decoder.h"
#include "model/gguf.h"
#include "runtime/runtime_internal.h"   // exec helpers, decode_params
#include "runtime/kernels.h"            // dequant_chunk (embedding rows)
#include "model/head_cert.h"            // P3.1 certified head argmax
#include "runtime/shadow.h"             // P3.2 expert shadows

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>

namespace fastllm {
namespace {

struct HParams {
    uint32_t n_embd, n_layer, n_head, n_vocab;
    uint32_t n_dense_lead, n_ff_dense, n_ff_exp, n_expert, n_used;
    uint32_t kv_lora, dk, dv, d_rope, d_nope;
    float    rms_eps, expert_scale;
    uint32_t expert_norm;
    // rope
    float freq_base, freq_scale, ext_factor, attn_factor;
    float beta_fast, beta_slow, yarn_log_mul;
    uint32_t n_ctx_orig;
    float kq_scale;
};

struct WTensor {                 // one weight matrix, mmap-resident
    const uint8_t * p = nullptr;
    Fmt fmt = Fmt::F32;
    uint32_t rows = 0, cols = 0;
    uint64_t row_bytes = 0, estride = 0;
    uint32_t nexp = 1;
};

class DecoderImpl final : public Decoder {
public:
    bool load(const DecodeOpts & o);

    ~DecoderImpl() { if (rlog_) fclose(rlog_); }

    int32_t step(int32_t token) override;
    void    reset() override { n_past_ = 0; }
    int32_t n_past() const override { return n_past_; }
    const float * logits() const override { return logits_; }
    const DecodeStats & stats() const override { return stats_; }
    std::string detok(const std::vector<int32_t> & ids) const override;
    int32_t bos_id() const override { return bos_; }
    uint32_t n_vocab() const override { return hp_.n_vocab; }

    void step_batch(const int32_t * toks, int m, int32_t * outs) override;
    const float * last_margins() const override { return margins_.data(); }
    void rewind_to(int32_t pos) override {
        if (pos < 0 || pos > n_past_) {
            fprintf(stderr, "decoder: bad rewind %d (n_past %d)\n", pos, n_past_);
            abort();
        }
        n_past_ = pos;
    }
    int max_batch() const override { return (int)gsets_.size(); }

private:
    // ---- graph building ----
    // one frozen graph per batch size m = 1..max_batch; positions are columns
    // of one sequence (causal within the pass via the cumulative kv-append
    // signal chain)
    struct GraphSet {
        std::unique_ptr<Graph> g;
        std::vector<float *> x_cur;     // per-position embedding row
        std::vector<float *> logits;    // per-position logits (slot region)
        size_t slot_off = 0, slot_len = 0;
        // P3.1: final-norm activation columns (cert argmax input)
        float * xf_base = nullptr; uint32_t xf_stride = 0;
        // P3.0 router taps: build-time pointers to each (layer, position)
        // selected-expert id array; read post-run when FASTLLM_ROUTER_LOG set.
        struct RouterTap { uint16_t layer, pos; const uint32_t * ids; };
        std::vector<RouterTap> rtaps;
    };
    void build_graphs();
    void build_graph_m(int m, GraphSet & gs);
    uint32_t add_dot(const WTensor & w, const float * x, float * y,
                     const std::vector<uint32_t> & deps, uint32_t rowsz,
                     std::vector<uint32_t> * tiles_out = nullptr,
                     const uint32_t * id_slot = nullptr);
    uint32_t add_op(TaskKind k, const void * aux,
                    const std::vector<uint32_t> & deps, Engine aff = Engine::CPU);
    void dep_edge(uint32_t p, uint32_t c);   // ported: relay trees keep every
                                             // producer under kMaxConsumers
    uint32_t fan_in(const std::vector<uint32_t> & producers);
    struct Port { uint32_t trunk; int trunk_used; uint32_t leaf; int leaf_used; };
    std::map<uint32_t, Port> ports_;
    uint32_t raw_signal();
    Engine dot_aff(Fmt f) const;

    float * slot_alloc(uint32_t n);          // zeroed-per-run region
    float * buf_alloc(uint32_t n);           // "=" written buffers (+appendix)
    // P2.4: m activation columns laid out contiguously with a fixed float
    // stride, so one multi-column dot tile can address them all. Returns the
    // base; *stride is the float distance between columns (appendix included).
    float * buf_alloc_cols(uint32_t n, int m, uint32_t * stride);
    // P2.5 fix: dot outputs ACCUMULATE (y[r] +=) and must live in the
    // zeroed-per-run slots region; a cols variant for stage-major buffers.
    float * slot_alloc_cols(uint32_t n, int m, uint32_t * stride);
    // one dot over m columns: weight read once, m results. Falls back to the
    // single-column path when m == 1.
    uint32_t add_dot_m(const WTensor & w, const float * xbase, uint32_t xstride,
                       float * ybase, uint32_t ystride, int m,
                       const std::vector<uint32_t> & deps, uint32_t rowsz);
    // P2.5: one shared-weight stage over m per-position columns. With
    // multicol OFF this emits exactly the per-position graph (per-column tile,
    // per-position dep) the P2.0-P2.4 builders produced -- the same-binary
    // baseline; with multicol ON it emits one multi-column tile set whose
    // deps are the union of all positions' producers (a per-stage barrier,
    // priced by the cells). sig[p] receives the id consumers of position p
    // must depend on.
    void stage_dot(const WTensor & w, const float * xbase, uint32_t xstride,
                   float * ybase, uint32_t ystride, int m,
                   const std::vector<uint32_t> & pdeps, uint32_t rowsz,
                   uint32_t * sig);
    template <typename T> T * aux_alloc() {
        return (T *)arena_->alloc(sizeof(T), 64);
    }

    const WTensor & W(const std::string & name) const {
        auto it = weights_.find(name);
        if (it == weights_.end()) {
            fprintf(stderr, "decoder: missing tensor %s\n", name.c_str());
            abort();
        }
        return it->second;
    }
    bool load_tensor(const std::string & name, bool required = true);
    void load_hparams();
    void build_detok();
    void dump_tap(const char * name, const float * v, uint32_t n) const;

    DecodeOpts o_;
    Gguf gguf_;
    HParams hp_{};
    std::map<std::string, WTensor> weights_;

    std::unique_ptr<Arena>   arena_;
    std::vector<GraphSet>    gsets_;     // index m-1
    Graph *                  graph_ = nullptr;   // graph under construction
    std::unique_ptr<Runtime> rt_;

    // host-side per-token state
    int32_t   n_past_ = 0;
    // P3.0: router-selection log (FASTLLM_ROUTER_LOG=path). Lines:
    // "pass pos tok layer id0..id5". Post-run dump only; no hot-path cost
    // when unset.
    FILE *    rlog_ = nullptr;
    uint64_t  rlog_pass_ = 0;
    int32_t * pos_ = nullptr;            // int32[max_batch]; pos_[p] = n_past+p
    float *   logits_ = nullptr;         // = gsets_[0].logits[0] (API compat)
    float *   slots_ = nullptr;          // zero-per-run region (per-graph slices)
    size_t    slots_floats_ = 0, slots_used_ = 0;
    float *   kcache_ = nullptr, * vcache_ = nullptr;   // all layers
    size_t    kc_layer_stride_ = 0, vc_layer_stride_ = 0;
    DecodeStats stats_{};
    int32_t   bos_ = 0;

    // P3.1 certified head argmax (see head_cert.h). hc_mode_: 0/1/2 per
    // DecodeOpts::head_cert after env override; audit keeps head tiles.
    HeadCert    hc_;
    int         hc_mode_ = 0;
    HeadDotMode hc_dot_ = HeadDotMode::CANON;
    uint64_t    hc_bytes_prev_ = 0;
    std::vector<float> margins_;   // P3.5: per-position top1-top2, last pass

    // checkpoint taps (M1 gate): name -> {ptr, n}
    std::vector<std::pair<std::string, std::pair<const float *, uint32_t>>> taps_;

    // detok
    std::vector<std::string> tok_piece_;   // decoded byte strings per id
};

// ------------------------------------------------------------- loading ----

bool DecoderImpl::load_tensor(const std::string & name, bool required) {
    const GgufTensor * gt = gguf_.find(name);
    if (!gt) {
        if (required) fprintf(stderr, "decoder: missing %s\n", name.c_str());
        return false;
    }
    WTensor w;
    if (!Gguf::fmt_of(gt->type, &w.fmt)) {
        fprintf(stderr, "decoder: %s unsupported type %s\n",
                name.c_str(), Gguf::type_name(gt->type));
        return false;
    }
    w.cols      = (uint32_t)gt->ne[0];
    w.rows      = (uint32_t)gt->ne[1];
    w.nexp      = (uint32_t)(gt->n_dims >= 3 ? gt->ne[2] : 1);
    w.row_bytes = fl_row_bytes(w.fmt, w.cols);
    w.estride   = w.nexp > 1 ? w.row_bytes * w.rows : 0;
    w.p         = gguf_.data(*gt);       // mmap-in-place (P1.1 lesson)
    weights_[name] = w;
    return true;
}

void DecoderImpl::load_hparams() {
    auto I = [&](const char * k, int64_t d) {
        return (int64_t)gguf_.kv_i(std::string("deepseek2.") + k, d);
    };
    auto F = [&](const char * k, double d) {
        return (float)gguf_.kv_f(std::string("deepseek2.") + k, d);
    };
    hp_.n_embd  = (uint32_t)I("embedding_length", 2048);
    hp_.n_layer = (uint32_t)I("block_count", 27);
    hp_.n_head  = (uint32_t)I("attention.head_count", 16);
    hp_.rms_eps = F("attention.layer_norm_rms_epsilon", 1e-6);
    hp_.kv_lora = (uint32_t)I("attention.kv_lora_rank", 512);
    hp_.dk      = (uint32_t)I("attention.key_length", 192);
    hp_.dv      = (uint32_t)I("attention.value_length", 128);
    hp_.d_rope  = (uint32_t)I("rope.dimension_count", 64);
    hp_.d_nope  = hp_.dk - hp_.d_rope;
    hp_.n_dense_lead = (uint32_t)I("leading_dense_block_count", 1);
    hp_.n_ff_dense   = (uint32_t)I("feed_forward_length", 10944);
    hp_.n_ff_exp     = (uint32_t)I("expert_feed_forward_length", 1408);
    hp_.n_expert     = (uint32_t)I("expert_count", 64);
    if (hp_.n_expert > 256) {
        fprintf(stderr, "decoder: n_expert %u exceeds router device cap\n",
                hp_.n_expert);
        abort();
    }
    hp_.n_used       = (uint32_t)I("expert_used_count", 6);
    hp_.expert_scale = F("expert_weights_scale", 1.0);
    hp_.expert_norm  = (uint32_t)I("expert_weights_norm", 0);

    hp_.freq_base  = F("rope.freq_base", 10000.0);
    const std::string sty = gguf_.kv_s("deepseek2.rope.scaling.type", "none");
    const float factor = F("rope.scaling.factor", 1.0);
    hp_.n_ctx_orig = (uint32_t)I("rope.scaling.original_context_length", 4096);
    if (sty == "yarn") {
        hp_.freq_scale  = 1.0f / factor;
        hp_.ext_factor  = 1.0f;
        // llama.cpp yarn_attn_factor_adjust: rope gets the adjusted factor so
        // ggml's internal *= (1+0.1 ln(1/fs)) restores 1.0
        hp_.attn_factor = 1.0f / (1.0f + 0.1f * logf(1.0f / hp_.freq_scale));
    } else {
        hp_.freq_scale = 1.0f; hp_.ext_factor = 0.0f; hp_.attn_factor = 1.0f;
    }
    hp_.beta_fast = 32.0f; hp_.beta_slow = 1.0f;    // llama.cpp cparam defaults
    // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX] convert script stores 0.1*mscale_all_dim
    hp_.yarn_log_mul =
        F("rope.scaling.yarn_log_multiplier", 0.0) / 0.1f;

    // deepseek2 kq_scale pre-scaling (llama.cpp deepseek2.cpp):
    //   attn_factor_org = attn_factor_adj * (1 + 0.1 ln(1/fs)) == 1.0
    //   mscale   = attn_factor_org * (1 + 0.1 * yarn_log_mul * ln(1/fs))
    //   kq_scale = mscale^2 / sqrt(dk)
    const float mscale = 1.0f
        + 0.1f * hp_.yarn_log_mul * logf(1.0f / hp_.freq_scale);
    hp_.kq_scale = mscale * mscale / sqrtf((float)hp_.dk);
    if (const char * e = getenv("FASTLLM_KQ_SCALE")) hp_.kq_scale = atof(e);

    hp_.n_vocab = 0;
    if (const GgufTensor * t = gguf_.find("token_embd.weight"))
        hp_.n_vocab = (uint32_t)t->ne[1];
    bos_ = (int32_t)gguf_.kv_i("tokenizer.ggml.bos_token_id", 100000);

    fprintf(stderr, "decoder: deepseek2 n_embd=%u n_layer=%u n_head=%u "
            "dk=%u dv=%u d_rope=%u kv_lora=%u vocab=%u | yarn fs=%.6f "
            "log_mul=%.3f kq_scale=%.6f | moe %u/%u ff=%u scale=%.2f norm=%u\n",
            hp_.n_embd, hp_.n_layer, hp_.n_head, hp_.dk, hp_.dv, hp_.d_rope,
            hp_.kv_lora, hp_.n_vocab, hp_.freq_scale, hp_.yarn_log_mul,
            hp_.kq_scale, hp_.n_used, hp_.n_expert, hp_.n_ff_exp,
            hp_.expert_scale, hp_.expert_norm);
}

bool DecoderImpl::load(const DecodeOpts & o) {
    o_ = o;
    if (!gguf_.open(o.model_path)) {
        fprintf(stderr, "decoder: cannot open %s\n", o.model_path.c_str());
        return false;
    }
    const std::string arch = gguf_.kv_s("general.architecture");
    if (arch != "deepseek2") {
        fprintf(stderr, "decoder: arch '%s' unsupported (deepseek2 only)\n",
                arch.c_str());
        return false;
    }
    load_hparams();

    bool ok = load_tensor("token_embd.weight")
           && load_tensor("output_norm.weight")
           && load_tensor("output.weight");
    char nm[128];
    for (uint32_t l = 0; ok && l < hp_.n_layer; ++l) {
        static const char * common[] = {
            "attn_norm", "attn_q", "attn_kv_a_mqa", "attn_kv_a_norm",
            "attn_kv_b", "attn_output", "ffn_norm",
        };
        for (const char * c : common) {
            snprintf(nm, sizeof nm, "blk.%u.%s.weight", l, c);
            ok = ok && load_tensor(nm);
        }
        if (l < hp_.n_dense_lead) {
            static const char * dn[] = {"ffn_gate", "ffn_up", "ffn_down"};
            for (const char * c : dn) {
                snprintf(nm, sizeof nm, "blk.%u.%s.weight", l, c);
                ok = ok && load_tensor(nm);
            }
        } else {
            static const char * mo[] = {
                "ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps",
                "ffn_down_exps", "ffn_gate_shexp", "ffn_up_shexp",
                "ffn_down_shexp",
            };
            for (const char * c : mo) {
                snprintf(nm, sizeof nm, "blk.%u.%s.weight", l, c);
                ok = ok && load_tensor(nm);
            }
        }
    }
    if (!ok) return false;

    // P3.2 full-shadow experts: pre-unpack ALL routed experts to int8 when
    // FASTLLM_SHADOW is set (see src/runtime/shadow.h; P3.0 refuted hot-set
    // selection - the working set is 2/3 of all experts in 16 tokens).
    {
        std::vector<ShadowTensorDesc> sd;
        for (uint32_t l = hp_.n_dense_lead; l < hp_.n_layer; ++l) {
            static const char * ex[] = {
                "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
            };
            for (const char * c : ex) {
                snprintf(nm, sizeof nm, "blk.%u.%s.weight", l, c);
                const WTensor & w = weights_[nm];
                sd.push_back({nm, w.p, w.fmt, w.cols,
                              (uint64_t)w.rows * w.nexp});
            }
        }
        const double bs = fl_shadow_init(sd);
        if (bs > 0.0)
            fprintf(stderr, "decoder: expert shadow build %.2f s\n", bs);
    }

    // sanity: q rows must equal n_head*dk; kv_b rows n_head*(d_nope+dv)
    if (W("blk.0.attn_q.weight").rows != hp_.n_head * hp_.dk
        || W("blk.0.attn_kv_b.weight").rows != hp_.n_head * (hp_.d_nope + hp_.dv)) {
        fprintf(stderr, "decoder: attention tensor shape mismatch\n");
        return false;
    }

    build_detok();
    hc_mode_ = o_.head_cert;
    if (const char * e = getenv("FASTLLM_HEAD_CERT")) hc_mode_ = atoi(e);
    if (const char * e = getenv("FASTLLM_HEAD_CERT_MARGIN"))
        o_.head_cert_margin = (float)atof(e);
    if (hc_mode_) {
        const WTensor & hw = W("output.weight");
        if (o_.rt.int8_dots
            && hw.fmt != Fmt::Q4_K && hw.fmt != Fmt::Q6_K) {
            fprintf(stderr, "decoder: head_cert disabled (head fmt %d has no "
                    "i8k path)\n", (int)hw.fmt);
            hc_mode_ = 0;
        } else {
            hc_dot_ = o_.rt.int8_dots    ? HeadDotMode::I8K
                    : o_.rt.fast_kernels ? HeadDotMode::FAST_DEQ
                                         : HeadDotMode::CANON;
            auto t0 = std::chrono::steady_clock::now();
            hc_.init(hw.fmt, hw.p, hw.row_bytes, hp_.n_vocab, hp_.n_embd,
                     o_.head_cert_margin);
            fprintf(stderr, "decoder: head_cert mode=%d dot=%d margin=%.3f "
                    "norms %.2f s\n", hc_mode_, (int)hc_dot_,
                    o_.head_cert_margin,
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count());
        }
    }
    build_graphs();
    if (const char * e = getenv("FASTLLM_ROUTER_LOG")) {
        rlog_ = fopen(e, "w");
        if (!rlog_) fprintf(stderr, "decoder: cannot open router log %s\n", e);
        else setvbuf(rlog_, nullptr, _IOFBF, 1 << 20);
    }
    return true;
}

// -------------------------------------------------------- graph helpers ----

Engine DecoderImpl::dot_aff(Fmt f) const {
    switch (o_.affinity) {
        case 0: return Engine::GPU;
        case 1: return Engine::CPU;
        default: return f == Fmt::Q8_0 ? Engine::CPU : Engine::GPU;
    }
}

float * DecoderImpl::slot_alloc(uint32_t n) {
    if (slots_used_ + n > slots_floats_) {
        fprintf(stderr, "decoder: slots region overflow\n");
        abort();
    }
    float * p = slots_ + slots_used_;
    slots_used_ += n;
    return p;
}

float * DecoderImpl::buf_alloc(uint32_t n) {
    return (float *)arena_->alloc(fl_x_alloc_bytes(n), 128);
}

float * DecoderImpl::buf_alloc_cols(uint32_t n, int m, uint32_t * stride) {
    const size_t per = (fl_x_alloc_bytes(n) + 127) & ~(size_t)127;   // 128 B
    *stride = (uint32_t)(per / 4);
    return (float *)arena_->alloc(per * (size_t)m, 128);
}

float * DecoderImpl::slot_alloc_cols(uint32_t n, int m, uint32_t * stride) {
    const uint32_t per = (uint32_t)((((size_t)n * 4 + 127) & ~(size_t)127) / 4);
    *stride = per;
    return slot_alloc(per * (uint32_t)m);
}

// FASTLLM_MULTICOL mode: 0 = off (single-column tiles, the P2.0-P2.4 graph),
// 1 = multi-column tiles, 'c'/2 = multi-column pinned CPU-affine.
static int fl_mc_mode() {
    static const int v = [] {
        const char * e = getenv("FASTLLM_MULTICOL");
        if (!e) return 0;
        if (e[0] == '0') return 0;
        if (e[0] == 'c') return 2;
        return 1;
    }();
    return v;
}

// P3.7 probe: FASTLLM_FFN_BARRIER=1 gates every routed-expert dot of a layer
// behind ALL m routers of that layer - the enqueue point expert-grouped
// batching would require (one tile set per distinct expert can only be formed
// once every position's routing is known). Bytes and arithmetic are unchanged,
// so a cell A/B prices the wavefront cost of grouping in isolation.
static bool fl_ffn_barrier() {
    static const bool v = [] {
        const char * e = getenv("FASTLLM_FFN_BARRIER");
        return e && e[0] == '1';
    }();
    return v;
}

// P3.5 ceiling gate: evaluate INTERMEDIATE verify positions with only the k'
// highest-weight routed experts. The router emits top-k in descending weight
// order, so truncation drops the least-weighted; the last position of a pass
// always keeps all n_used because it produces the correction token. This
// measures what the work-removal can buy BEFORE any margin/backstop machinery
// is built.
static uint32_t fl_approx_k() {
    static const uint32_t v = [] {
        const char * e = getenv("FASTLLM_APPROX_K");
        return e ? (uint32_t)atoi(e) : 0u;
    }();
    return v;
}

// P3.5b: truncating to k' drops the discarded experts' gate mass, biasing the
// merged FFN output low. Renormalising the survivors to carry that mass is the
// obvious fix and was MEASURED WORSE for the only thing this mode cares about,
// argmax agreement: on 6000 teacher-forced positions the flip rate rose 2.38 ->
// 2.83% at k'=5 and 3.28 -> 4.25% at k'=4. Rescaling restores the magnitude but
// overweights the surviving experts, moving the argmax more often than the
// uniform low bias does. Default OFF; set 1 to reproduce the A/B.
static bool fl_approx_renorm() {
    static const bool v = [] {
        const char * e = getenv("FASTLLM_APPROX_RENORM");
        return e && e[0] == '1';
    }();
    return v;
}

// P2.4: one tile set over m activation columns. Bytes streamed per pass drop
// by ~m for this weight; results are bitwise identical to m separate dots
// (same per-(row,column) accumulation order, see fastllm.h Task::nx).
uint32_t DecoderImpl::add_dot_m(const WTensor & w, const float * xbase,
                                uint32_t xstride, float * ybase,
                                uint32_t ystride, int m,
                                const std::vector<uint32_t> & deps,
                                uint32_t rowsz) {
    // FASTLLM_MULTICOL=0 emits m single-column tiles instead (same-binary A/B;
    // P1.3 rule: cross-build probe comparisons are confounded by whole-TU
    // register pressure). FASTLLM_MULTICOL=cpu additionally pins multi-column
    // tiles to the CPU pool, where the interleaved row loop reuses L1.
    // Default OFF: P2.4 measured multi-column at parity-to--2% on both engines
    // (same-binary A/B). Sharing the weight READ buys nothing because neither
    // engine is DRAM-bound at m>=8 - the CPU dot is ALU-bound (P1.4: IPC 3.20,
    // instruction-count-bound) and it still re-unpacks the row per column,
    // while GPU tiles (20 blocks x 0.3-1.7 MB) exceed L2. The machinery stays
    // for P2.5, whose kernels must share the UNPACK, not just the load.
    const int mc_mode = fl_mc_mode();
    if (mc_mode == 0) {
        std::vector<uint32_t> sigs;
        for (int p = 0; p < m; ++p)
            sigs.push_back(add_dot(w, xbase + (size_t)p * xstride,
                                   ybase + (size_t)p * ystride, deps, rowsz));
        return fan_in(sigs);
    }
    if (m == 1) return add_dot(w, xbase, ybase, deps, rowsz);
    std::vector<uint32_t> tiles;
    for (uint32_t r0 = 0; r0 < w.rows; r0 += rowsz) {
        const uint32_t r = w.rows - r0 < rowsz ? w.rows - r0 : rowsz;
        Task t{};
        t.kind = TaskKind::STREAM_DOT;
        t.fmt  = w.fmt;
        t.affinity = mc_mode == 2 ? Engine::CPU : dot_aff(w.fmt);
        t.w    = w.p + (size_t)r0 * w.row_bytes;
        t.w_bytes = (uint64_t)r * w.row_bytes;   // counted ONCE per pass
        t.x    = xbase;
        t.y    = ybase + r0;
        t.rows = r;
        t.cols = w.cols;
        t.nx      = (uint32_t)m;
        t.xstride = xstride;
        t.ystride = ystride;
        uint32_t id = graph_->add(t);
        for (uint32_t d : deps) dep_edge(d, id);
        tiles.push_back(id);
    }
    if (tiles.size() == 1) return tiles[0];
    return fan_in(tiles);
}

void DecoderImpl::stage_dot(const WTensor & w, const float * xbase,
                            uint32_t xstride, float * ybase, uint32_t ystride,
                            int m, const std::vector<uint32_t> & pdeps,
                            uint32_t rowsz, uint32_t * sig) {
    if (fl_mc_mode() != 0 && m > 1) {
        const uint32_t id = add_dot_m(w, xbase, xstride, ybase, ystride, m,
                                      pdeps, rowsz);
        for (int p = 0; p < m; ++p) sig[p] = id;
        return;
    }
    for (int p = 0; p < m; ++p)
        sig[p] = add_dot(w, xbase + (size_t)p * xstride,
                         ybase + (size_t)p * ystride, {pdeps[p]}, rowsz);
}

uint32_t DecoderImpl::raw_signal() {
    Task s{};
    s.kind = TaskKind::SIGNAL;
    s.affinity = Engine::CPU;
    return graph_->add(s);
}

// Ported edges: the first kMaxConsumers-1 consumers attach to the producer
// directly; after that a relay tree grows lazily (trunk chain, <=5 leaves per
// trunk, <=6 consumers per leaf). Keeps fan-out legal at ~2 extra SIGNAL hops
// for ~100 consumers.
void DecoderImpl::dep_edge(uint32_t p, uint32_t c) {
    Task * ts = graph_->tasks();
    if (ts[p].n_consumers < kMaxConsumers - 1) {   // keep one slot for a relay
        graph_->edge(p, c);
        return;
    }
    auto it = ports_.find(p);
    if (it == ports_.end()) {
        uint32_t trunk = raw_signal();
        graph_->edge(p, trunk);                    // uses the reserved slot
        uint32_t leaf = raw_signal();
        graph_->edge(trunk, leaf);
        it = ports_.emplace(p, Port{trunk, 1, leaf, 0}).first;
    }
    Port & po = it->second;
    if (po.leaf_used >= (int)kMaxConsumers) {
        if (po.trunk_used >= (int)kMaxConsumers - 1) {
            uint32_t nt = raw_signal();            // chain a new trunk
            graph_->edge(po.trunk, nt);
            po.trunk = nt;
            po.trunk_used = 0;
        }
        po.leaf = raw_signal();
        graph_->edge(po.trunk, po.leaf);
        po.trunk_used++;
        po.leaf_used = 0;
    }
    graph_->edge(po.leaf, c);
    po.leaf_used++;
}

// fan a producer set into ONE dependency-carrier id (SIGNAL trees keep every
// node under kMaxConsumers on the consumer side; producers with spare slots
// connect directly)
uint32_t DecoderImpl::fan_in(const std::vector<uint32_t> & producers) {
    uint32_t sig = raw_signal();
    for (uint32_t p : producers) dep_edge(p, sig);
    return sig;
}

uint32_t DecoderImpl::add_op(TaskKind k, const void * aux,
                             const std::vector<uint32_t> & deps, Engine aff) {
    Task t{};
    t.kind = k;
    t.affinity = aff;
    t.aux = aux;
    uint32_t id = graph_->add(t);
    for (uint32_t d : deps) dep_edge(d, id);
    return id;
}

// STREAM_DOT(_IDX) row-tiled; returns a fan-in id covering all tiles.
// deps: every tile depends on each listed id (use a fan-in signal upstream
// when deps would exceed producer fan-out).
uint32_t DecoderImpl::add_dot(const WTensor & w, const float * x, float * y,
                              const std::vector<uint32_t> & deps, uint32_t rowsz,
                              std::vector<uint32_t> * tiles_out,
                              const uint32_t * id_slot) {
    std::vector<uint32_t> tiles;
    for (uint32_t r0 = 0; r0 < w.rows; r0 += rowsz) {
        const uint32_t r = w.rows - r0 < rowsz ? w.rows - r0 : rowsz;
        Task t{};
        t.kind = id_slot ? TaskKind::STREAM_DOT_IDX : TaskKind::STREAM_DOT;
        t.fmt  = w.fmt;
        t.affinity = dot_aff(w.fmt);
        t.w    = w.p + (size_t)r0 * w.row_bytes;
        t.w_bytes = (uint64_t)r * w.row_bytes;
        t.x    = x;
        t.y    = y + r0;
        t.rows = r;
        t.cols = w.cols;
        if (id_slot) { t.aux = id_slot; t.aux2 = w.estride; }
        uint32_t id = graph_->add(t);
        for (uint32_t d : deps) dep_edge(d, id);
        tiles.push_back(id);
    }
    if (tiles_out) *tiles_out = tiles;
    if (tiles.size() == 1) return tiles[0];
    return fan_in(tiles);
}

// ---------------------------------------------------------- graph build ----

void DecoderImpl::build_graphs() {
    const HParams & h = hp_;
    const uint32_t NH = h.n_head;
    const int MB = o_.max_batch < 1 ? 1 : (o_.max_batch > 16 ? 16 : o_.max_batch);
    o_.max_batch = MB;

    // budget arena: per-graph task tables + slots + buffers + shared caches.
    // Graph m costs ~m x the single-position graph; sum over m = MB(MB+1)/2.
    const uint64_t sum_m = (uint64_t)MB * (MB + 1) / 2;
    const uint64_t kc_bytes = (uint64_t)h.n_layer * o_.max_kv * NH * h.dk * 4;
    const uint64_t vc_bytes = (uint64_t)h.n_layer * o_.max_kv * NH * h.dv * 4;
    const uint64_t slot_budget =
        ((uint64_t)h.n_layer * 80000 + h.n_vocab + (1u << 20)) * sum_m;
    arena_ = Arena::create(sum_m * 18000ull * sizeof(Task)
                           + slot_budget * 4 + kc_bytes + vc_bytes
                           + (256ull << 20) + sum_m * (48ull << 20));

    slots_floats_ = slot_budget;
    slots_ = (float *)arena_->alloc(slots_floats_ * 4, 128);
    kcache_ = (float *)arena_->alloc(kc_bytes, 128);
    vcache_ = (float *)arena_->alloc(vc_bytes, 128);
    kc_layer_stride_ = (size_t)o_.max_kv * NH * h.dk;
    vc_layer_stride_ = (size_t)o_.max_kv * NH * h.dv;
    pos_ = (int32_t *)arena_->alloc(64 * ((MB + 15) / 16), 64);

    gsets_.resize(MB);
    for (int m = 1; m <= MB; ++m) build_graph_m(m, gsets_[m - 1]);
    logits_ = gsets_[0].logits[0];

    if (o_.use_runtime) {
        rt_ = Runtime::create(*arena_, o_.rt);
        rt_->start();
    }
}

void DecoderImpl::build_graph_m(int m, GraphSet & gs) {
    const HParams & h = hp_;
    const uint32_t NE = h.n_embd, NH = h.n_head;
    // Tile granularity is m-DEPENDENT (P3.8). P3.7b named the binding
    // constraint: the pass is parallelism-bound (32 execution contexts = 12
    // CPU workers + 20 GPU blocks), and tiles per pass scale with m, so the
    // rowsz that keeps those contexts fed without drowning them in per-tile
    // protocol scales with m too. Measured on thor (t12, shadow on, copy):
    //   m=1: 128->23.25  192->25.42  [256->26.05]  384->24.51  512->22.12  768->18.41
    //   m=8: 128->34.54  [256->47.35] 512->52.79  [768->53.94] 1024->53.36  2048->46.76
    // A single global value costs ~30% at whichever end it is wrong for.
    // P3.9 measured every m instead of interpolating (the adaptive controller
    // picks m=3..7 on medium-alpha workloads, so the interpolated middle was
    // the common real case). tok/s by rowsz, copy prompt, alpha=1, [best]:
    //   m=2: [256->38.69] 384->36.83  512->35.23  768->31.79
    //   m=3: 256->44.49  [384->45.48] 512->43.29  768->40.27
    //   m=4: 256->46.12  [384->48.53] 512->47.62  768->45.47
    //   m=5: 256->46.34  384->49.14  [512->49.77] 768->48.82
    //   m=6: 512->50.51  640->50.33  768->50.56   (flat within noise)
    //   m=7: 512->53.41  640->53.56  [768->54.42]
    //   m=8: 512->52.79  640->53.41  [768->53.94] 896->53.58
    // The old interpolation (512 at m=3,4) cost 5.1% / 1.9% there. No closed
    // form fits (work per tile is not conserved across m); this is the
    // measured step table. Env override wins for sweeps.
    //
    // P4.6 RE-SWEEP under NEON: the table above was fitted against SCALAR
    // kernels (the aarch64 build shipped no dotprod/i8mm until P4.4), and NEON
    // cut per-tile compute 2.80x. Re-measured, shadow on, t12, copy prompt:
    //   m=8: 384->68.77 512->71.16 [768->72.59] 1024->72.03 1536->69.91
    //   m=5: [384->65.69] 512->65.61 768->65.42 1024->61.53   (384~512, flat)
    //   m=3: 256->57.23 [384->58.20] 512->57.37 768->55.07
    //   m=2: 160->44.94 192->47.27 [256->48.48] 320->48.72     (>=256 wanted)
    //   m=1: 128->31.83 [192->36.20] 256->33.81 384->33.14 512->33.03
    // Only m=1 moved: 256 -> 192, ABAB-confirmed +7.0% (35.86/36.06 vs
    // 33.35/33.74). m>=2 is unchanged, so the old m<3 branch had to SPLIT --
    // 192 is 2.5% WORSE at m=2. Batch-1 decode issues the fewest tiles per
    // pass, so it starves the 32 contexts first and wants the finest tiles;
    // faster kernels moved that break-even point down. head_rowsz stays 1024
    // (512->71.71, 2048->72.91, both within noise of 72.80).
    uint32_t rowsz = m >= 7 ? 768u
                   : (m >= 5 ? 512u
                   : (m >= 3 ? 384u
                   : (m >= 2 ? 256u : 192u)));
    uint32_t head_rowsz = 1024;
    if (const char * e = getenv("FASTLLM_ROWSZ"))      rowsz = (uint32_t)atoi(e);
    if (const char * e = getenv("FASTLLM_HEAD_ROWSZ")) head_rowsz = (uint32_t)atoi(e);
    if (rowsz == 0) rowsz = 256;
    if (head_rowsz == 0) head_rowsz = 1024;

    ports_.clear();                     // relay state is per-graph
    gs.g = Graph::create(*arena_, 18000u * m);
    graph_ = gs.g.get();
    gs.slot_off = slots_used_;
    gs.x_cur.resize(m);
    gs.logits.resize(m);
    for (int p = 0; p < m; ++p) gs.x_cur[p] = buf_alloc(NE);

    char nm[128];
    auto Wl = [&](uint32_t l, const char * c) -> const WTensor & {
        snprintf(nm, sizeof nm, "blk.%u.%s.weight", l, c);
        return W(nm);
    };
    auto tapf = [&](uint32_t l, const char * what, const float * p, uint32_t n) {
        if (m == 1 && (l <= 1 || l == h.n_layer - 1)) {
            char t[64];
            snprintf(t, sizeof t, "l%u.%s", l, what);
            taps_.emplace_back(t, std::make_pair(p, n));
        }
    };

    // per-position residual streams; positions couple ONLY through the KV
    // cache (cumulative append signal chain per layer), giving the causal
    // in-pass wavefront: position p at layer l waits for appends 0..p.
    std::vector<const float *> x_res(m);
    std::vector<uint32_t> x_res_dep(m, UINT32_MAX);
    for (int p = 0; p < m; ++p) x_res[p] = gs.x_cur[p];

    for (uint32_t l = 0; l < h.n_layer; ++l) {
        // ---- P2.5 stage-major layer build ----
        // Per-position ops stay per position; every shared-weight dot goes
        // through stage_dot (multicol OFF -> exactly the per-position graph
        // of P2.0-P2.4; ON -> one multi-column tile per weight, whose barrier
        // cost the cells arbitrate). Routed experts stay per position: their
        // ids differ per position (expert-grouped batching is P2.5 lever 3,
        // out of scope here).
        const uint32_t KVA = h.kv_lora + h.d_rope;
        uint32_t xnS, qbS, kvaS, ckvS, kvbS, ctxS, aoS, x2S, xfS;
        float * xnB  = buf_alloc_cols(NE, m, &xnS);
        float * qbB  = slot_alloc_cols(NH * h.dk, m, &qbS);
        float * kvaB = slot_alloc_cols(KVA, m, &kvaS);
        float * ckvB = buf_alloc_cols(h.kv_lora, m, &ckvS);
        float * kvbB = slot_alloc_cols(NH * (h.d_nope + h.dv), m, &kvbS);
        float * ctxB = buf_alloc_cols(NH * h.dv, m, &ctxS);
        float * aoB  = slot_alloc_cols(NE, m, &aoS);
        float * x2B  = buf_alloc_cols(NE, m, &x2S);
        float * xfB  = buf_alloc_cols(NE, m, &xfS);

        std::vector<uint32_t> aq(m), kq8(m), cq8(m), fq8(m), fn(m), tx2(m);
        std::vector<uint32_t> sq(m), skva(m), skvb(m), so(m);

        // Stage A (per p): attn norm + activation quant
        for (int p = 0; p < m; ++p) {
            float * xn = xnB + (size_t)p * xnS;
            std::vector<uint32_t> nd;
            if (x_res_dep[p] != UINT32_MAX) nd.push_back(x_res_dep[p]);
            auto * pn = aux_alloc<RmsNormP>();
            *pn = { (const float *)Wl(l, "attn_norm").p, x_res[p], xn,
                    h.rms_eps, NE };
            uint32_t t_an = add_op(TaskKind::RMSNORM, pn, nd);
            aq[p] = add_op(TaskKind::ACT_Q8, nullptr, {t_an});
            Task & t = graph_->tasks()[aq[p]];
            t.x = xn; t.y = (void *)fl_xq_of(xn, NE);
            t.rows = NE / 32; t.cols = NE;
        }

        // Stage B: q / kv_a projections (shared weights)
        stage_dot(Wl(l, "attn_q"), xnB, xnS, qbB, qbS, m, aq, rowsz,
                  sq.data());
        stage_dot(Wl(l, "attn_kv_a_mqa"), xnB, xnS, kvaB, kvaS, m, aq, rowsz,
                  skva.data());

        // Stage C (per p): kv_a norm + quant
        for (int p = 0; p < m; ++p) {
            float * kva = kvaB + (size_t)p * kvaS;
            float * ckv = ckvB + (size_t)p * ckvS;
            auto * pkn = aux_alloc<RmsNormP>();
            *pkn = { (const float *)Wl(l, "attn_kv_a_norm").p, kva, ckv,
                     h.rms_eps, h.kv_lora };
            uint32_t t_kn = add_op(TaskKind::RMSNORM, pkn, {skva[p]});
            kq8[p] = add_op(TaskKind::ACT_Q8, nullptr, {t_kn});
            Task & t = graph_->tasks()[kq8[p]];
            t.x = ckv; t.y = (void *)fl_xq_of(ckv, h.kv_lora);
            t.rows = h.kv_lora / 32; t.cols = h.kv_lora;
        }

        // Stage D: kv_b expansion (shared weight)
        stage_dot(Wl(l, "attn_kv_b"), ckvB, ckvS, kvbB, kvbS, m, kq8, rowsz,
                  skvb.data());

        // Stage E (per p, in position order): rope, causal KV append chain,
        // attention heads, context quant
        uint32_t app_cum = UINT32_MAX;   // covers kv appends of positions <= p
        for (int p = 0; p < m; ++p) {
            float * qb  = qbB  + (size_t)p * qbS;
            float * kva = kvaB + (size_t)p * kvaS;
            float * kvb = kvbB + (size_t)p * kvbS;
            float * ctx = ctxB + (size_t)p * ctxS;
            auto * prq = aux_alloc<RopeP>();
            *prq = { qb, pos_ + p, NH, h.dk, h.d_nope, h.d_rope,
                     h.freq_base, h.freq_scale, h.ext_factor, h.attn_factor,
                     h.beta_fast, h.beta_slow, h.n_ctx_orig };
            uint32_t t_rq = add_op(TaskKind::ROPE_DS2, prq, {sq[p]});
            auto * prk = aux_alloc<RopeP>();
            *prk = { kva + h.kv_lora, pos_ + p, 1, h.d_rope, 0, h.d_rope,
                     h.freq_base, h.freq_scale, h.ext_factor, h.attn_factor,
                     h.beta_fast, h.beta_slow, h.n_ctx_orig };
            uint32_t t_rk = add_op(TaskKind::ROPE_DS2, prk, {skva[p]});

            float * kc = kcache_ + (size_t)l * kc_layer_stride_;
            float * vc = vcache_ + (size_t)l * vc_layer_stride_;
            auto * pap = aux_alloc<KvAppendP>();
            *pap = { kvb, kva + h.kv_lora, kc, vc, pos_ + p,
                     NH, h.d_nope, h.d_rope, h.dv, (uint32_t)o_.max_kv };
            uint32_t t_app = add_op(TaskKind::KV_APPEND, pap, {skvb[p], t_rk});
            app_cum = (app_cum == UINT32_MAX)
                ? t_app
                : fan_in({app_cum, t_app});

            float * scr = (float *)arena_->alloc((size_t)NH * o_.max_kv * 4, 128);
            std::vector<uint32_t> heads;
            for (uint32_t hh = 0; hh < NH; ++hh) {
                auto * ph = aux_alloc<AttnHeadP>();
                *ph = { qb, kc, vc, scr + (size_t)hh * o_.max_kv, ctx, pos_ + p,
                        h.kq_scale, hh, NH, h.dk, h.dv, (uint32_t)o_.max_kv };
                heads.push_back(add_op(TaskKind::ATTN_HEAD, ph, {t_rq, app_cum}));
            }
            uint32_t t_heads = fan_in(heads);
            cq8[p] = add_op(TaskKind::ACT_Q8, nullptr, {t_heads});
            Task & t = graph_->tasks()[cq8[p]];
            t.x = ctx; t.y = (void *)fl_xq_of(ctx, NH * h.dv);
            t.rows = NH * h.dv / 32; t.cols = NH * h.dv;
        }

        // Stage F: attention output projection (shared weight)
        stage_dot(Wl(l, "attn_output"), ctxB, ctxS, aoB, aoS, m, cq8, rowsz,
                  so.data());

        // Stage G (per p): residual add, ffn norm + quant
        for (int p = 0; p < m; ++p) {
            float * attn_out = aoB + (size_t)p * aoS;
            float * x2 = x2B + (size_t)p * x2S;
            float * xf = xfB + (size_t)p * xfS;
            tapf(l, "attn_out", attn_out, NE);
            auto * pa2 = aux_alloc<Add2P>();
            *pa2 = { x_res[p], attn_out, x2, NE };
            std::vector<uint32_t> a2d = {so[p]};
            if (x_res_dep[p] != UINT32_MAX) a2d.push_back(x_res_dep[p]);
            tx2[p] = add_op(TaskKind::ADD2, pa2, a2d);
            tapf(l, "ffn_inp", x2, NE);
            auto * pfn = aux_alloc<RmsNormP>();
            *pfn = { (const float *)Wl(l, "ffn_norm").p, x2, xf, h.rms_eps, NE };
            fn[p] = add_op(TaskKind::RMSNORM, pfn, {tx2[p]});
            fq8[p] = add_op(TaskKind::ACT_Q8, nullptr, {fn[p]});
            Task & t = graph_->tasks()[fq8[p]];
            t.x = xf; t.y = (void *)fl_xq_of(xf, NE);
            t.rows = NE / 32; t.cols = NE;
        }

        // Stage H: FFN
        uint32_t xoS;
        float * xoB = buf_alloc_cols(NE, m, &xoS);
        if (l < h.n_dense_lead) {
            const uint32_t FF = h.n_ff_dense;
            uint32_t ggS, uuS, hhS, ddS;
            float * ggB = slot_alloc_cols(FF, m, &ggS);
            float * uuB = slot_alloc_cols(FF, m, &uuS);
            float * hhB = buf_alloc_cols(FF, m, &hhS);
            float * ddB = slot_alloc_cols(NE, m, &ddS);
            std::vector<uint32_t> sg(m), su(m), hq8(m), sd(m);
            stage_dot(Wl(l, "ffn_gate"), xfB, xfS, ggB, ggS, m, fq8, rowsz,
                      sg.data());
            stage_dot(Wl(l, "ffn_up"), xfB, xfS, uuB, uuS, m, fq8, rowsz,
                      su.data());
            for (int p = 0; p < m; ++p) {
                auto * ps = aux_alloc<SiluMulP>();
                *ps = { ggB + (size_t)p * ggS, uuB + (size_t)p * uuS,
                        hhB + (size_t)p * hhS, FF };
                uint32_t t_s = add_op(TaskKind::SILU_MUL, ps, {sg[p], su[p]});
                hq8[p] = add_op(TaskKind::ACT_Q8, nullptr, {t_s});
                Task & t = graph_->tasks()[hq8[p]];
                t.x = hhB + (size_t)p * hhS;
                t.y = (void *)fl_xq_of(hhB + (size_t)p * hhS, FF);
                t.rows = FF / 32; t.cols = FF;
            }
            stage_dot(Wl(l, "ffn_down"), hhB, hhS, ddB, ddS, m, hq8, rowsz,
                      sd.data());
            for (int p = 0; p < m; ++p) {
                auto * pm = aux_alloc<WmergeP>();
                *pm = { ddB + (size_t)p * ddS, nullptr, x2B + (size_t)p * x2S,
                        nullptr, xoB + (size_t)p * xoS, 1.0f, 1, NE };
                uint32_t t_lo = add_op(TaskKind::WMERGE, pm, {sd[p], tx2[p]});
                tapf(l, "l_out", xoB + (size_t)p * xoS, NE);
                x_res[p] = xoB + (size_t)p * xoS;
                x_res_dep[p] = t_lo;
            }
        } else {
            // routed experts: per position (ids differ per position)
            const WTensor & wge = Wl(l, "ffn_gate_exps");
            const WTensor & wue = Wl(l, "ffn_up_exps");
            const WTensor & wde = Wl(l, "ffn_down_exps");
            std::vector<uint32_t> rdown_sig(m);
            std::vector<float *> rde(m);
            std::vector<uint32_t *> rgate(m);
            // P3.7 probe: expert-grouped batching would emit one tile set per
            // DISTINCT expert serving every position that routed to it, so the
            // expert dots cannot be enqueued until ALL m routers of this layer
            // have run. FASTLLM_FFN_BARRIER=1 reproduces exactly that
            // dependency structure (routers of all positions fan in ahead of
            // every expert dot) while leaving bytes and arithmetic untouched,
            // isolating the wavefront cost of grouping from its byte saving.
            const bool grp_barrier = fl_ffn_barrier();
            const uint32_t approx_k = fl_approx_k();
            std::vector<uint32_t> ku_p(m, h.n_used);
            std::vector<uint32_t> rt_all;
            std::vector<uint32_t> rt_ids(m), rl_ids(m);
            std::vector<uint32_t *> rid_p(m);
            std::vector<float *> rgt_p(m);
            for (int p = 0; p < m; ++p) {
                float * xf = xfB + (size_t)p * xfS;
                float * rlog = slot_alloc(h.n_expert);
                uint32_t t_rl = add_dot(Wl(l, "ffn_gate_inp"), xf, rlog,
                                        {fn[p]}, rowsz);
                uint32_t * ids  = (uint32_t *)arena_->alloc(h.n_used * 4, 64);
                float    * gate = (float *)arena_->alloc(h.n_used * 4, 64);
                gs.rtaps.push_back({ (uint16_t)l, (uint16_t)p, ids });
                auto * prt = aux_alloc<RouterP>();
                *prt = { rlog, ids, gate, h.n_expert, h.n_used,
                         h.expert_norm, h.expert_scale };
                uint32_t t_rt = add_op(TaskKind::ROUTER_SEL, prt, {t_rl});
                rt_ids[p] = t_rt;
                rl_ids[p] = t_rl;
                rid_p[p] = ids;
                rgt_p[p] = gate;
                if (grp_barrier) rt_all.push_back(t_rt);
            }
            // With the barrier probe every expert dot waits on all m routers
            // (the enqueue point grouping would force); without it each
            // position's experts wait only on its own router, as before.
            const uint32_t rt_gate = grp_barrier ? fan_in(rt_all) : UINT32_MAX;
            for (int p = 0; p < m; ++p) {
                float * xf = xfB + (size_t)p * xfS;
                uint32_t * ids = rid_p[p];
                const uint32_t t_rt = grp_barrier ? rt_gate : rt_ids[p];
                float * de = slot_alloc(h.n_used * NE);
                std::vector<uint32_t> downs;
                // P3.5: positions 1..m-2 may run k' < n_used. Position 0 and
                // the last stay exact - the last produces the correction token,
                // and position 0 guarantees forward progress when the margin
                // gate truncates acceptance (a truncated position is simply
                // recomputed as position 0 of the next pass, which is exact).
                const uint32_t ku = (approx_k && m > 2 && p >= 1 && p < m - 1 &&
                                     approx_k < h.n_used) ? approx_k : h.n_used;
                ku_p[p] = ku;
                for (uint32_t s = 0; s < ku; ++s) {
                    float * ge = slot_alloc(h.n_ff_exp);
                    float * ue = slot_alloc(h.n_ff_exp);
                    uint32_t t_ge = add_dot(wge, xf, ge, {fq8[p], t_rt}, rowsz,
                                            nullptr, ids + s);
                    uint32_t t_ue = add_dot(wue, xf, ue, {fq8[p], t_rt}, rowsz,
                                            nullptr, ids + s);
                    float * he = buf_alloc(h.n_ff_exp);
                    auto * ps = aux_alloc<SiluMulP>();
                    *ps = { ge, ue, he, h.n_ff_exp };
                    uint32_t t_s = add_op(TaskKind::SILU_MUL, ps, {t_ge, t_ue});
                    uint32_t t_hq8 = add_op(TaskKind::ACT_Q8, nullptr, {t_s});
                    {
                        Task & t = graph_->tasks()[t_hq8];
                        t.x = he; t.y = (void *)fl_xq_of(he, h.n_ff_exp);
                        t.rows = h.n_ff_exp / 32; t.cols = h.n_ff_exp;
                    }
                    downs.push_back(add_dot(wde, he, de + (size_t)s * NE,
                                            {t_hq8}, rowsz, nullptr, ids + s));
                }
                rdown_sig[p] = fan_in(downs);
                rde[p] = de;
                rgate[p] = (uint32_t *)rgt_p[p];
            }

            // shared expert chain (shared weights): gate/up -> silu -> down
            const uint32_t FS = Wl(l, "ffn_gate_shexp").rows;
            uint32_t gsS, usS, hsS, dsS;
            float * gsB = slot_alloc_cols(FS, m, &gsS);
            float * usB = slot_alloc_cols(FS, m, &usS);
            float * hsB = buf_alloc_cols(FS, m, &hsS);
            float * dsB = slot_alloc_cols(NE, m, &dsS);
            std::vector<uint32_t> sgs(m), sus(m), sq8(m), sds(m);
            stage_dot(Wl(l, "ffn_gate_shexp"), xfB, xfS, gsB, gsS, m, fq8,
                      rowsz, sgs.data());
            stage_dot(Wl(l, "ffn_up_shexp"), xfB, xfS, usB, usS, m, fq8,
                      rowsz, sus.data());
            for (int p = 0; p < m; ++p) {
                auto * pss = aux_alloc<SiluMulP>();
                *pss = { gsB + (size_t)p * gsS, usB + (size_t)p * usS,
                         hsB + (size_t)p * hsS, FS };
                uint32_t t_ss = add_op(TaskKind::SILU_MUL, pss, {sgs[p], sus[p]});
                sq8[p] = add_op(TaskKind::ACT_Q8, nullptr, {t_ss});
                Task & t = graph_->tasks()[sq8[p]];
                t.x = hsB + (size_t)p * hsS;
                t.y = (void *)fl_xq_of(hsB + (size_t)p * hsS, FS);
                t.rows = FS / 32; t.cols = FS;
            }
            stage_dot(Wl(l, "ffn_down_shexp"), hsB, hsS, dsB, dsS, m, sq8,
                      rowsz, sds.data());

            for (int p = 0; p < m; ++p) {
                auto * pm = aux_alloc<WmergeP>();
                // n_full > n_parts only when this position was truncated to
                // k' AND renorm is enabled; every exact merge passes 0.
                const uint32_t nfull = (fl_approx_renorm() &&
                                        ku_p[p] < h.n_used) ? h.n_used : 0u;
                *pm = { rde[p], (float *)rgate[p], x2B + (size_t)p * x2S,
                        dsB + (size_t)p * dsS, xoB + (size_t)p * xoS,
                        1.0f, ku_p[p], NE, nfull };
                uint32_t t_lo = add_op(TaskKind::WMERGE, pm,
                                       {rdown_sig[p], sds[p], tx2[p]});
                tapf(l, "l_out", xoB + (size_t)p * xoS, NE);
                x_res[p] = xoB + (size_t)p * xoS;
                x_res_dep[p] = t_lo;
            }
        }
    }

    // final norm + head. The norm/quant stay per position; the head is ONE
    // multi-column dot (P2.4): output.weight is the single largest weight in
    // the pass and is shared by every position, so it streams once instead of
    // m times.
    std::vector<uint32_t> head_sigs;
    uint32_t xF_stride = 0;
    float * xF_base = buf_alloc_cols(NE, m, &xF_stride);
    gs.xf_base = xF_base; gs.xf_stride = xF_stride;
    std::vector<uint32_t> head_deps;
    for (int p = 0; p < m; ++p) {
        float * xF = xF_base + (size_t)p * xF_stride;
        auto * pfn = aux_alloc<RmsNormP>();
        *pfn = { (const float *)W("output_norm.weight").p, x_res[p], xF,
                 hp_.rms_eps, NE };
        uint32_t t_fn = add_op(TaskKind::RMSNORM, pfn, {x_res_dep[p]});
        uint32_t t_fq8 = add_op(TaskKind::ACT_Q8, nullptr, {t_fn});
        {
            Task & t = graph_->tasks()[t_fq8];
            t.x = xF; t.y = (void *)fl_xq_of(xF, NE);
            t.rows = NE / 32; t.cols = NE;
        }
        if (m == 1)
            taps_.emplace_back("result_norm",
                               std::make_pair((const float *)xF, NE));
        head_deps.push_back(t_fq8);
    }
    if (hc_mode_ == 1) {
        // P3.1: no head tiles. The certified argmax reads xF columns on the
        // host post-run; logits are never materialized (logits() -> null).
        for (int p = 0; p < m; ++p) gs.logits[p] = nullptr;
    } else {
        // logits: m contiguous columns so one tile set writes them all
        const uint32_t V = hp_.n_vocab;
        float * lg = slot_alloc(V * (uint32_t)m);
        for (int p = 0; p < m; ++p) gs.logits[p] = lg + (size_t)p * V;
        head_sigs.push_back(add_dot_m(W("output.weight"), xF_base, xF_stride,
                                      lg, V, m, head_deps, head_rowsz));
    }

    Task done{};
    done.kind = TaskKind::SIGNAL;
    done.affinity = Engine::CPU;
    uint32_t term = graph_->add(done);
    // head_sigs are fresh fan-in signals (0 consumers): direct edges cannot
    // spawn relay SIGNALs after the terminal, keeping term the last SIGNAL.
    // Under cert-on the ACT_Q8 tasks have zero consumers (their only consumer
    // was the head), so the same direct-edge argument applies to them.
    if (hc_mode_ == 1)
        for (uint32_t d : head_deps) graph_->edge(d, term);
    else
        for (uint32_t hs : head_sigs) graph_->edge(hs, term);

    graph_->freeze();
    gs.slot_len = slots_used_ - gs.slot_off;
    fprintf(stderr, "decoder: graph[m=%d] %u tasks, slots %.2f MB, "
            "arena %.2f GB\n", m, graph_->size(),
            gs.slot_len * 4 / 1e6, arena_->used() / 1e9);
    // P8.6 (FASTLLM_GRAPH_STATS=1): the shape of the DAG itself, so the
    // question "could we prepare the next layer while this one runs?" is
    // answered from structure instead of intuition. Walks the frozen graph
    // once on the host - no device work, no effect on the token path.
    //
    // Reports an ASAP level schedule (level = longest dependency chain to a
    // node), the width at each level, and a critical path weighted by the
    // MEASURED per-kind CPU cost (P8.0, depth 128). The two numbers that
    // matter: the critical path is the floor no amount of parallelism or
    // prefetch can go below, and the count of levels narrower than the worker
    // pool is where workers must idle no matter how cheap claiming becomes.
    if (getenv("FASTLLM_GRAPH_STATS")) {
        const uint32_t n = graph_->size();
        const Task * T = graph_->tasks();
        // measured us/tile, CPU engine, P8.0 SWEEPK depth 128
        static const double kcost[kNumTaskKinds] = {
            37.36, 0.10, 0.133, 9.98, 49.04, 2.59, 1.19,
            5.80, 2.89, 20.17, 1.77, 8.30, 13.26 };
        std::vector<uint32_t> lvl(n, 0);
        std::vector<double>   start(n, 0.0);
        double work = 0.0;
        uint32_t back = 0;                 // consumers with id <= producer
        for (uint32_t i = 0; i < n; ++i) work += kcost[(int)T[i].kind];
        // Task ids are NOT guaranteed topological (measured: 1477 edges run
        // backwards at m=1), so a single forward sweep would silently drop
        // those edges and UNDER-report both depth and the critical path -
        // exactly the number this whole analysis rests on. Relax to a fixpoint
        // instead; the pass count is reported so a non-converged run is visible.
        uint32_t iter = 0;
        for (bool changed = true; changed && iter < 4096; ++iter) {
            changed = false;
            for (uint32_t i = 0; i < n; ++i) {
                const double fin = start[i] + kcost[(int)T[i].kind];
                for (uint16_t k = 0; k < T[i].n_consumers; ++k) {
                    const uint32_t c = T[i].consumers[k];
                    if (c >= n) continue;
                    if (iter == 0 && c <= i) ++back;
                    if (lvl[i] + 1 > lvl[c]) { lvl[c] = lvl[i] + 1; changed = true; }
                    if (fin > start[c] + 1e-9) { start[c] = fin; changed = true; }
                }
            }
        }
        uint32_t depth = 0;
        double cp = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            if (lvl[i] > depth) depth = lvl[i];
            const double f = start[i] + kcost[(int)T[i].kind];
            if (f > cp) cp = f;
        }
        std::vector<uint32_t> w(depth + 1, 0);
        for (uint32_t i = 0; i < n; ++i) w[lvl[i]]++;
        uint32_t narrow = 0, wmax = 0;
        uint64_t wsum = 0;
        for (uint32_t l = 0; l <= depth; ++l) {
            if (w[l] < 32) ++narrow;        // 12 CPU threads + 20 GPU blocks
            if (w[l] > wmax) wmax = w[l];
            wsum += w[l];
        }
        fprintf(stderr,
            "GRAPHSTATS m=%d tasks=%u levels=%u backedges=%u relax_passes=%u\n"
            "GRAPHSTATS   total_work=%.2f ms   perfect_pack_32w=%.2f ms   "
            "critical_path=%.2f ms\n"
            "GRAPHSTATS   width: max=%u mean=%.1f   levels_narrower_than_32=%u"
            " (%.1f%% of levels)\n",
            m, n, depth + 1, back, iter,
            work / 1000.0, work / 32000.0, cp / 1000.0,
            wmax, (double)wsum / (depth + 1), narrow,
            100.0 * narrow / (depth + 1));
    }
}

// ------------------------------------------------------------- stepping ----

void DecoderImpl::step_batch(const int32_t * toks, int m, int32_t * outs) {
    auto t0 = std::chrono::steady_clock::now();
    if (m < 1 || m > (int)gsets_.size()) {
        fprintf(stderr, "decoder: bad batch %d (max %zu)\n", m, gsets_.size());
        abort();
    }
    if (n_past_ + m > (int32_t)o_.max_kv) {
        fprintf(stderr, "decoder: kv full (%d + %d > %d)\n",
                n_past_, m, o_.max_kv);
        abort();
    }
    GraphSet & gs = gsets_[m - 1];
    const WTensor & emb = W("token_embd.weight");
    auto ns_since = [](std::chrono::steady_clock::time_point a) {
        return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - a).count();
    };
    auto te0 = std::chrono::steady_clock::now();
    for (int p = 0; p < m; ++p) {
        if (emb.fmt == Fmt::F32) {
            memcpy(gs.x_cur[p], emb.p + (size_t)toks[p] * emb.row_bytes,
                   hp_.n_embd * 4);
        } else {
            dequant_chunk(emb.fmt, emb.p + (size_t)toks[p] * emb.row_bytes,
                          0, hp_.n_embd, gs.x_cur[p]);
        }
        pos_[p] = n_past_ + p;
    }
    stats_.ph_embed_ns += ns_since(te0);

    auto tz0 = std::chrono::steady_clock::now();
    memset(slots_ + gs.slot_off, 0, gs.slot_len * 4);
    stats_.ph_zero_ns += ns_since(tz0);

    auto tr0 = std::chrono::steady_clock::now();
    if (rt_) stats_.last = rt_->run(*gs.g);
    else     stats_.last = run_serial(*gs.g);
    stats_.ph_run_ns   += ns_since(tr0);
    stats_.ph_reset_ns += stats_.last.reset_ns;
    stats_.ph_bytes    += stats_.last.cpu.bytes + stats_.last.gpu.bytes;
    stats_.ph_passes   += 1;
    stats_.ph_positions += (uint64_t)m;

    if (rlog_) {
        for (const auto & rt : gs.rtaps) {
            fprintf(rlog_, "%llu %d %d %u", (unsigned long long)rlog_pass_,
                    pos_[rt.pos], toks[rt.pos], (unsigned)rt.layer);
            for (uint32_t s = 0; s < hp_.n_used; ++s)
                fprintf(rlog_, " %u", rt.ids[s]);
            fputc('\n', rlog_);
        }
        rlog_pass_++;
    }

    n_past_ += m;
    auto ta0 = std::chrono::steady_clock::now();
    float bv_last = 0.f;
    if (hc_mode_ == 1) {
        // P3.1: certified argmax; the draft hint for position p is the next
        // fed token (verify semantics: outs[p] is compared against toks[p+1])
        for (int p = 0; p < m; ++p) {
            int32_t hint = (p < m - 1) ? toks[p + 1] : -1;
            outs[p] = (int32_t)hc_.argmax(
                gs.xf_base + (size_t)p * gs.xf_stride, hint, hc_dot_);
        }
        stats_.ph_cert_ns += ns_since(ta0);
        const HeadCertStats & cs = hc_.stats();
        stats_.cert_rows_eval  = cs.rows_eval;
        stats_.cert_rows_total = cs.rows_total;
        stats_.cert_bytes      = cs.bytes_read;
        stats_.cert_worst      = cs.worst_cand;
        stats_.ph_bytes       += cs.bytes_read - hc_bytes_prev_;  // gamma honesty
        hc_bytes_prev_ = cs.bytes_read;
    } else {
        // greedy argmax per position (host; ties -> lower id, llama.cpp conv.)
        // P3.5: the same scan yields the runner-up, so the decision margin
        // (top1 - top2) costs one extra comparison per vocab entry.
        margins_.assign(m, 0.f);
        for (int p = 0; p < m; ++p) {
            const float * lg = gs.logits[p];
            uint32_t best = 0;
            float bv = lg[0], sv = -1e30f;
            for (uint32_t i = 1; i < hp_.n_vocab; ++i)
                if (lg[i] > bv) { sv = bv; bv = lg[i]; best = i; }
                else if (lg[i] > sv) { sv = lg[i]; }
            outs[p] = (int32_t)best;
            margins_[p] = bv - sv;
            bv_last = bv;
        }
        stats_.ph_argmax_ns += ns_since(ta0);
        if (hc_mode_ == 2) {   // audit: cert must reproduce the engine argmax
            auto tc0 = std::chrono::steady_clock::now();
            for (int p = 0; p < m; ++p) {
                int32_t hint = (p < m - 1) ? toks[p + 1] : -1;
                uint32_t c = hc_.argmax(
                    gs.xf_base + (size_t)p * gs.xf_stride, hint, hc_dot_);
                if ((int32_t)c != outs[p]) {
                    fprintf(stderr, "HEAD_CERT AUDIT MISMATCH pos=%d "
                            "engine=%d cert=%u\n", pos_[p], outs[p], c);
                    if (!getenv("FASTLLM_CERT_AUDIT_WARN")) abort();
                }
            }
            stats_.ph_cert_ns += ns_since(tc0);
            const HeadCertStats & cs = hc_.stats();
            stats_.cert_rows_eval  = cs.rows_eval;
            stats_.cert_rows_total = cs.rows_total;
            stats_.cert_bytes      = cs.bytes_read;
            stats_.cert_worst      = cs.worst_cand;
        }
    }

    if (o_.dump_taps && m == 1) {
        for (auto & t : taps_) dump_tap(t.first.c_str(), t.second.first,
                                        t.second.second);
        fprintf(stderr, "TAP pos=%d argmax=%d logit=%.6f\n",
                n_past_ - 1, outs[m - 1], bv_last);
    }
    auto t1 = std::chrono::steady_clock::now();
    stats_.step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int32_t DecoderImpl::step(int32_t token) {
    int32_t out = 0;
    step_batch(&token, 1, &out);
    return out;
}

void DecoderImpl::dump_tap(const char * name, const float * v, uint32_t n) const {
    double sum = 0, amax = 0;
    for (uint32_t i = 0; i < n; ++i) {
        sum += v[i];
        const double a = fabs(v[i]);
        if (a > amax) amax = a;
    }
    fprintf(stderr, "TAP %-16s n=%-6u first8=[% .5e % .5e % .5e % .5e "
            "% .5e % .5e % .5e % .5e] sum=% .6e absmax=% .6e\n",
            name, n, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
            sum, amax);
}

// ---------------------------------------------------------------- detok ----

void DecoderImpl::build_detok() {
    const std::vector<std::string> * toks =
        gguf_.kv_sarr("tokenizer.ggml.tokens");
    if (!toks) {
        fprintf(stderr, "decoder: no tokenizer table (detok disabled)\n");
        return;
    }
    // gpt2 bytes_to_unicode inverse: codepoint -> byte
    int cp_of_byte[256];
    {
        bool direct[256] = {false};
        auto rng = [&](int a, int b) { for (int i = a; i <= b; ++i) direct[i] = true; };
        rng(33, 126); rng(161, 172); rng(174, 255);
        int next = 256;
        for (int b = 0; b < 256; ++b)
            cp_of_byte[b] = direct[b] ? b : next++;
    }
    std::map<int, uint8_t> byte_of_cp;
    for (int b = 0; b < 256; ++b) byte_of_cp[cp_of_byte[b]] = (uint8_t)b;

    tok_piece_.resize(toks->size());
    for (size_t i = 0; i < toks->size(); ++i) {
        const std::string & s = (*toks)[i];
        std::string out;
        for (size_t j = 0; j < s.size();) {
            // decode one UTF-8 codepoint
            int cp = 0, len = 1;
            const uint8_t c0 = (uint8_t)s[j];
            if (c0 < 0x80) { cp = c0; }
            else if ((c0 >> 5) == 0x6 && j + 1 < s.size()) {
                cp = ((c0 & 0x1F) << 6) | ((uint8_t)s[j+1] & 0x3F); len = 2;
            } else if ((c0 >> 4) == 0xE && j + 2 < s.size()) {
                cp = ((c0 & 0x0F) << 12) | (((uint8_t)s[j+1] & 0x3F) << 6)
                   | ((uint8_t)s[j+2] & 0x3F); len = 3;
            } else if ((c0 >> 3) == 0x1E && j + 3 < s.size()) {
                cp = ((c0 & 0x07) << 18) | (((uint8_t)s[j+1] & 0x3F) << 12)
                   | (((uint8_t)s[j+2] & 0x3F) << 6) | ((uint8_t)s[j+3] & 0x3F);
                len = 4;
            }
            auto it = byte_of_cp.find(cp);
            if (it != byte_of_cp.end()) out.push_back((char)it->second);
            else out.append(s, j, len);   // passthrough (special tokens)
            j += len;
        }
        tok_piece_[i] = out;
    }
}

std::string DecoderImpl::detok(const std::vector<int32_t> & ids) const {
    std::string out;
    for (int32_t id : ids)
        if (id >= 0 && (size_t)id < tok_piece_.size()) out += tok_piece_[id];
    return out;
}

} // namespace

std::unique_ptr<Decoder> Decoder::create(const DecodeOpts & o) {
    auto d = std::make_unique<DecoderImpl>();
    if (!d->load(o)) return nullptr;
    return d;
}

} // namespace fastllm
