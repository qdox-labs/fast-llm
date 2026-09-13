// P2.0: end-to-end greedy decoder for DeepSeek-V2-Lite-family GGUFs
// (deepseek2 arch, non-MLA layout: unsplit attn_kv_b, direct attn_q).
// One frozen task graph per model; per token the host writes the embedding
// row + position, zeroes the accumulation slots, runs the graph, and argmaxes
// the logits. Correctness contract: canonical mode is run_serial-bitwise;
// fast/int8 modes are tolerance-gated like every other fast path (BENCH.md).
#pragma once
#include "fastllm/fastllm.h"
#include <memory>
#include <string>
#include <vector>

namespace fastllm {

struct DecodeOpts {
    std::string model_path;
    int   max_kv      = 512;
    int   affinity    = 3;      // 0 gpu, 1 cpu, 3 format-aware (bench conv.)
    bool  use_runtime = true;   // false -> run_serial
    RuntimeConfig rt;           // engine config (fast/int8/threads/gpu)
    bool  dump_taps   = false;  // print layer-0/1/final checkpoints per step
    int   max_batch   = 1;      // P2.1: verify-batch graphs for m = 1..max_batch
    // P3.1 certified head argmax: 0 off, 1 on (head tiles removed; a proof-
    // pruned exact evaluation computes each position's argmax), 2 audit (head
    // tiles kept AND cert runs; any disagreement aborts). Soundness argument
    // in src/model/head_cert.h; margin validated by the audit gate.
    int   head_cert   = 0;
    float head_cert_margin = 1.0f;
};

struct DecodeStats {
    RunMetrics last;            // metrics of the last graph run
    double     step_ms = 0.0;   // wall of the last step incl. host work
    // P2.4 pass decomposition, accumulated over all steps (ns). run_ns is the
    // runtime window (== sum of RunMetrics::wall_ns); reset_ns is inside run().
    uint64_t ph_embed_ns = 0;   // embedding row fetch/dequant
    uint64_t ph_zero_ns  = 0;   // slot region memset
    uint64_t ph_run_ns   = 0;   // execute window (publish -> quiesce)
    uint64_t ph_reset_ns = 0;   // dep-count reset loop (inside run, pre-window)
    uint64_t ph_argmax_ns = 0;  // host greedy argmax over n_vocab x m
    uint64_t ph_passes    = 0;  // number of step_batch calls
    uint64_t ph_positions  = 0; // sum of m over passes
    uint64_t ph_bytes      = 0; // weight bytes streamed (cpu+gpu; includes
                                // cert host-side row reads for gamma honesty)
    // P3.1 certified head (zeros when head_cert == 0)
    uint64_t ph_cert_ns    = 0; // host cert time (norms scan + candidate dots)
    uint64_t cert_rows_eval = 0, cert_rows_total = 0;
    uint64_t cert_bytes    = 0; // weight bytes read by cert exact dots
    uint64_t cert_worst    = 0; // max exact rows in one argmax call
};

class Decoder {
public:
    static std::unique_ptr<Decoder> create(const DecodeOpts & o);
    virtual ~Decoder() = default;

    // feed one token at position n_past (prompt or generated); returns argmax
    // token id of the resulting logits.
    virtual int32_t step(int32_t token) = 0;
    virtual void    reset() = 0;               // n_past = 0 (KV discarded)
    virtual int32_t n_past() const = 0;
    virtual const float * logits() const = 0;  // n_vocab, after step()
    virtual const DecodeStats & stats() const = 0;

    // P2.1 verify batch: feed m tokens at positions n_past..n_past+m-1 in ONE
    // graph pass (m <= max_batch); writes per-position greedy argmax to outs;
    // n_past advances by m. KV commit/discard is rewind_to(): stale KV rows
    // beyond n_past are overwritten by later passes before any attention can
    // read them (overwrite-before-read invariant, see BENCH.md P2.1).
    virtual void step_batch(const int32_t * toks, int m, int32_t * outs) = 0;
    // P3.5: decision margin (top1 - top2 logit) per position of the last
    // step_batch - free, the argmax scan already visits every logit.
    virtual const float * last_margins() const = 0;
    virtual void rewind_to(int32_t pos) = 0;   // discard KV tail (pos <= n_past)
    virtual int  max_batch() const = 0;

    // detokenize via the GGUF's byte-level BPE table (gpt2 lineage)
    virtual std::string detok(const std::vector<int32_t> & ids) const = 0;
    virtual int32_t bos_id() const = 0;
    virtual uint32_t n_vocab() const = 0;
};

} // namespace fastllm
