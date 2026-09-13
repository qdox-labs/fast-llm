// fastllm-decode: end-to-end greedy decode driver (P2.0) + speculative
// decoding via suffix-match drafting and verify-batch wavefronts (P2.1).
// Token ids in (from llama-tokenize for parity), ids + text out.
// Losslessness contract: --spec on output is token-identical to --spec off.
#include "model/decoder.h"
#include "spec/suffix_index.h"
#include "spec/spec_controller.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

using namespace fastllm;

static std::vector<int32_t> parse_ids(const char * s) {
    std::vector<int32_t> v;
    while (*s) {
        char * e;
        long x = strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t)x);
        s = *e ? e + 1 : e;
    }
    return v;
}

struct SpecStats {
    int passes = 0, drafted = 0, accepted = 0;
    int trunc = 0;              // P3.5: margin-gate truncations (backstop fires)
    int n_hist[17] = {0};              // chosen draft length distribution
    // P3.3 chaining gate: the ceiling on pass chaining is the time the engines
    // are not executing this pass - host work (embed/zero/argmax inside
    // step_batch, plus draft/accept/rewind in the driver loop) and whatever
    // idle sits inside the run window (busy/wall vs worker count).
    double host_ms = 0.0;              // step_batch wall minus runtime window
    double run_ms  = 0.0;              // sum of runtime windows (wall)
    double cpu_busy_ms = 0.0, gpu_busy_ms = 0.0;
    // P9.3: weight bytes moved, so the bench path can report BUS OCCUPANCY.
    // tok/s = achieved GB/s / GB-per-token, and until now only the m=1
    // depth-sweep reported either term - the anchor cell (m=8), our best
    // number, had never had its occupancy measured at all.
    double cpu_bytes = 0.0, gpu_bytes = 0.0;
};

int main(int argc, char ** argv) {
    DecodeOpts o;
    std::vector<int32_t> ids;
    int n_gen = 64, reps = 1;
    bool json = false, bench = false;
    std::vector<int32_t> sweep_depths;   // P2.2 depth-wall instrumentation
    int sweep_steps = 6;
    bool spec = false;
    int spec_max = 7, spec_fixed = -1;
    uint32_t spec_min = 3, spec_chain = 64;
    // P3.5 calibration: teacher-force the continuation so an exact run and an
    // approximate run visit IDENTICAL positions - a flip would otherwise
    // desynchronise the two token streams and make them incomparable.
    std::string force_path, cal_path;
    // P3.5 margin gate: accept an approximated position only if its decision
    // margin exceeds tau; otherwise truncate acceptance there so the position
    // is recomputed exactly as position 0 of the next pass. tau <= 0 disables
    // (the default) - approximation is opt-in and NOT lossless, see BENCH.md.
    float approx_tau = 0.f;
    o.rt.fast_kernels = true;
    o.rt.int8_dots = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if      (a == "--model")   o.model_path = next();
        else if (a == "--ids")     ids = parse_ids(next());
        else if (a == "-n")        n_gen = atoi(next());
        else if (a == "--kv-max")  o.max_kv = atoi(next());
        else if (a == "--engine")  o.use_runtime = std::string(next()) == "runtime";
        else if (a == "--cpu-threads") o.rt.cpu_threads = atoi(next());
        else if (a == "--gpu")     o.rt.gpu_enabled = std::string(next()) == "on";
        else if (a == "--fast")    o.rt.fast_kernels = std::string(next()) == "on";
        else if (a == "--int8")    o.rt.int8_dots = std::string(next()) == "on";
        else if (a == "--spec")    spec = std::string(next()) == "on";
        else if (a == "--spec-max")   spec_max = atoi(next());
        else if (a == "--spec-fixed") spec_fixed = atoi(next());
        else if (a == "--force-ids") force_path = next();   // P3.5 calibration
        else if (a == "--cal-dump")  cal_path  = next();    // P3.5 calibration
        else if (a == "--approx-verify-tau") approx_tau = (float)atof(next());
        else if (a == "--spec-min-match") spec_min = (uint32_t)atoi(next());
        else if (a == "--spec-chain")     spec_chain = (uint32_t)atoi(next());
        else if (a == "--affinity") {
            std::string v = next();
            o.affinity = v == "gpu" ? 0 : v == "cpu" ? 1 : 3;
        }
        else if (a == "--head-cert") {
            std::string v = next();
            o.head_cert = v == "on" ? 1 : v == "audit" ? 2 : 0;
        }
        else if (a == "--head-cert-margin") o.head_cert_margin = atof(next());
        else if (a == "--dump")    o.dump_taps = true;
        else if (a == "--bench")   bench = true;
        else if (a == "--reps")    reps = atoi(next());
        else if (a == "--json")    json = true;
        else if (a == "--depth-sweep") sweep_depths = parse_ids(next());
        else if (a == "--sweep-steps") sweep_steps = atoi(next());
        else {
            fprintf(stderr, "usage: fastllm-decode --model M --ids \"i,i,...\" "
                    "[-n N] [--engine runtime|serial] [--cpu-threads N] "
                    "[--gpu on|off] [--fast on|off] [--int8 on|off] "
                    "[--affinity fmt|cpu|gpu] [--kv-max N] [--dump] "
                    "[--spec on|off] [--spec-max N] [--spec-fixed N] "
                    "[--spec-min-match N] [--spec-chain N] "
                    "[--bench --reps R] [--json]\n");
            return 2;
        }
    }
    if (o.model_path.empty() || ids.empty()) {
        fprintf(stderr, "need --model and --ids\n");
        return 2;
    }
    if (spec_fixed >= 0) { spec = true; if (spec_fixed > spec_max) spec_max = spec_fixed; }
    if (spec_max < 1) spec_max = 1;
    if (spec_max > 15) spec_max = 15;
    o.max_batch = spec ? spec_max + 1 : 1;
    if (!sweep_depths.empty()) {
        o.max_batch = 8;                 // batched prefill to reach depth fast
        for (int32_t d : sweep_depths)
            if (d >= o.max_kv) { fprintf(stderr, "sweep depth %d >= kv-max %d\n",
                                         d, o.max_kv); return 2; }
    }

    auto dec = Decoder::create(o);
    if (!dec) return 1;

    // ---- P2.2 depth sweep: measure wall/token and per-kind attribution at
    // fixed KV depths; the depth-linear term names the wall. Non-spec m=1
    // steps; prefill feeds --ids cyclically in max-batch chunks. ----
    if (!sweep_depths.empty()) {
        static const char * kn[] = {"dot","merge","signal","actq8","dot_idx",
            "rmsnorm","add2","rope","kv_append","attn_head","router",
            "silu_mul","wmerge"};
        printf("SWEEPHDR,depth,ms_tok,run_ms,cpu_busy_ms,gpu_busy_ms\n");
        for (int32_t d : sweep_depths) {
            dec->reset();
            const int MB = dec->max_batch();
            std::vector<int32_t> toks(MB), outs(MB);
            int fed = 0;
            while (fed < d) {
                const int m = (d - fed) < MB ? (d - fed) : MB;
                for (int k = 0; k < m; ++k)
                    toks[k] = ids[(size_t)(fed + k) % ids.size()];
                dec->step_batch(toks.data(), m, outs.data());
                fed += m;
            }
            double w_ms = 0, run_ms = 0, cpu_ms = 0, gpu_ms = 0;
            uint64_t ckn[kNumTaskKinds] = {}, ckc[kNumTaskKinds] = {};
            uint64_t gkn[kNumTaskKinds] = {}, gkc[kNumTaskKinds] = {};
            // P8.0 time decomposition (zero unless FASTLLM_PROF_GPU=1)
            double gcl = 0, gid = 0, gex = 0, gep = 0;
            double gpolls = 0, gempty = 0, gretry = 0, gbytes = 0;
            double gpubytes = 0, cpubytes = 0;
            uint32_t gblk = 0;
            int32_t cur = outs[0];
            for (int s = 0; s < sweep_steps; ++s) {
                cur = dec->step(cur);
                const DecodeStats & ds = dec->stats();
                const RunMetrics & m2 = ds.last;
                w_ms   += ds.step_ms;
                run_ms += m2.wall_ns / 1e6;
                cpu_ms += m2.cpu.busy_ns / 1e6;
                gpu_ms += m2.gpu.busy_ns / 1e6;
                gcl += m2.gpu_claim_ns / 1e6; gid += m2.gpu_sleep_ns / 1e6;
                gex += m2.gpu_exec_ns  / 1e6; gep += m2.gpu_epi_ns  / 1e6;
                gpolls += (double)m2.gpu_polls; gempty += (double)m2.gpu_empty;
                gretry += (double)m2.gpu_retry;
                if (m2.gpu_blocks_active > gblk) gblk = m2.gpu_blocks_active;
                gbytes  += (double)(m2.gpu.bytes + m2.cpu.bytes);
                gpubytes += (double)m2.gpu.bytes;
                cpubytes += (double)m2.cpu.bytes;
                for (int k = 0; k < kNumTaskKinds; ++k) {
                    ckn[k] += m2.kind_ns[k];     ckc[k] += m2.kind_n[k];
                    gkn[k] += m2.gpu_kind_ns[k]; gkc[k] += m2.gpu_kind_n[k];
                }
                dec->rewind_to(dec->n_past() - 1);   // hold depth constant
            }
            const double inv = 1.0 / sweep_steps;
            printf("SWEEP,%d,%.2f,%.2f,%.2f,%.2f\n", d, w_ms * inv,
                   run_ms * inv, cpu_ms * inv, gpu_ms * inv);
            // P8.7: weight bytes per token and the resulting bus rate. Settles
            // whether m=1 is bandwidth-bound at all - if occupancy is low, the
            // wall is DAG width and per-task overhead, not the memory system,
            // and redundant execution would have had bandwidth headroom to
            // burn even though the arithmetic rules it out.
            {
                const double sec = run_ms * inv / 1e3;
                // Per-engine achieved bandwidth. This is the number that
                // answers "is our CUDA any good": llama.cpp does this whole
                // model GPU-only at ~130 GB/s of weight traffic on the same
                // silicon, so our GPU's GB/s here is a direct comparison and
                // NOT confounded by how the work is split.
                printf("SWEEPB,%d,GB_per_tok,%.4f,bus_GBs,%.1f,"
                       "gpu_GB,%.4f,gpu_GBs,%.1f,cpu_GB,%.4f,cpu_GBs,%.1f,"
                       "gpu_share,%.1f%%\n",
                       d, gbytes * inv / 1e9, (gbytes * inv / 1e9) / sec,
                       gpubytes * inv / 1e9, (gpubytes * inv / 1e9) / sec,
                       cpubytes * inv / 1e9, (cpubytes * inv / 1e9) / sec,
                       100.0 * gpubytes / (gbytes > 0 ? gbytes : 1));
            }
            // Block-summed, per step. The comparison denominator is
            // wall_ms * blocks -- these four buckets partition that product,
            // and whatever is left over is loop/barrier residue.
            if (gblk) {
                printf("SWEEPG,%d,blocks,%u,wall_ms,%.4f,claim_ms,%.4f,"
                       "sleep_ms,%.4f,exec_ms,%.4f,epi_ms,%.4f,"
                       "polls,%.1f,empty,%.1f,retry,%.1f\n",
                       d, gblk, run_ms * inv, gcl * inv, gid * inv,
                       gex * inv, gep * inv, gpolls * inv, gempty * inv,
                       gretry * inv);
                // Self-audit. Two invariants: the buckets must fit inside
                // wall x blocks, and empty must be a subset of polls. P8.0's
                // first run violated both (empty 24850 > polls 7689; buckets
                // at 155% of the window) and that is what exposed the reset
                // race. Print the check every time so a broken instrument
                // announces itself instead of producing a plausible number.
                const double den = run_ms * inv * gblk;
                const double sum = (gcl + gid + gex + gep) * inv;
                const char * ok = (gempty <= gpolls && sum <= den * 1.05)
                                  ? "OK" : "VIOLATION";
                printf("SWEEPGX,%d,denom_ms,%.4f,sum_ms,%.4f,residue_ms,%.4f,"
                       "claim_pct,%.1f,sleep_pct,%.1f,exec_pct,%.1f,"
                       "epi_pct,%.1f,check,%s\n",
                       d, den, sum, den - sum,
                       100.0 * gcl * inv / den, 100.0 * gid * inv / den,
                       100.0 * gex * inv / den, 100.0 * gep * inv / den, ok);
            }
            for (int k = 0; k < kNumTaskKinds; ++k) {
                if (!ckc[k] && !gkc[k]) continue;
                printf("SWEEPK,%d,%s,cpu_ms,%.3f,cpu_n,%.0f,gpu_ms,%.3f,gpu_n,%.0f\n",
                       d, kn[k], ckn[k] / 1e6 * inv, ckc[k] * inv,
                       gkn[k] / 1e6 * inv, gkc[k] * inv);
            }
            fflush(stdout);
        }
        return 0;
    }
    const bool dbg = getenv("FASTLLM_SPEC_DEBUG") != nullptr;

    // one generation run; returns wall ms of the generation loop only
    auto run_once = [&](bool print, SpecStats * st) -> double {
        dec->reset();
        SuffixIndex idx(spec_chain, spec_min);
        SpecController ctrl(SpecController::cfg_from_env());

        int32_t next_tok = 0;
        if (!spec) {
            for (size_t i = 0; i < ids.size(); ++i) next_tok = dec->step(ids[i]);
        } else {
            // prompt prefill in max-batch chunks (argmax of last position only)
            const int MB = dec->max_batch();
            std::vector<int32_t> outs(MB);
            size_t i = 0;
            while (i < ids.size()) {
                const int m = (int)((ids.size() - i) < (size_t)MB
                                    ? (ids.size() - i) : (size_t)MB);
                dec->step_batch(ids.data() + i, m, outs.data());
                next_tok = outs[m - 1];
                i += m;
            }
            for (int32_t t : ids) idx.push(t);
        }

        std::vector<int32_t> out;
        auto d0 = std::chrono::steady_clock::now();
        if (!force_path.empty()) {
            // P3.5 calibration run: the continuation is FORCED from a file, so
            // every pass covers the same absolute positions in both the exact
            // and the approximate build. Nothing is accepted or rejected on
            // argmax - we only record (position, is_intermediate, argmax,
            // margin) for the offline P(flip | margin) join. Not a decode mode.
            std::vector<int32_t> forced;
            { FILE * f = fopen(force_path.c_str(), "r");
              if (!f) { fprintf(stderr, "cannot open %s\n", force_path.c_str()); return 1; }
              int32_t t; while (fscanf(f, "%d", &t) == 1) forced.push_back(t);
              fclose(f); }
            FILE * cf = cal_path.empty() ? stdout : fopen(cal_path.c_str(), "w");
            if (!cf) { fprintf(stderr, "cannot write %s\n", cal_path.c_str()); return 1; }
            fprintf(cf, "pos,inter,argmax,margin\n");
            const int mw = spec_fixed >= 0 ? spec_fixed + 1 : spec_max + 1;
            std::vector<int32_t> toks(mw), outs(mw);
            size_t fi = 0;
            int32_t cur = next_tok;
            while (fi < forced.size()) {
                const int m = (int)std::min((size_t)mw, forced.size() - fi + 1);
                toks[0] = cur;
                for (int k = 1; k < m; ++k) toks[k] = forced[fi + k - 1];
                const int32_t base = dec->n_past();
                dec->step_batch(toks.data(), m, outs.data());
                const float * mg = dec->last_margins();
                for (int p = 0; p < m; ++p)
                    fprintf(cf, "%d,%d,%d,%.6f\n", base + p, p < m - 1 ? 1 : 0,
                            outs[p], mg[p]);
                // advance along the FORCED stream regardless of argmax
                const int adv = m - 1;
                if (adv <= 0) break;
                cur = forced[fi + adv - 1];
                fi += adv;
                dec->rewind_to(base + m);
                out.push_back(cur);
            }
            if (cf != stdout) fclose(cf);
            fprintf(stderr, "cal: %zu positions dumped to %s\n", forced.size(),
                    cal_path.empty() ? "stdout" : cal_path.c_str());
            return 0;
        }
        if (!spec) {
            for (int i = 0; i < n_gen; ++i) {
                out.push_back(next_tok);
                next_tok = dec->step(next_tok);
                if (st) {
                    const RunMetrics & lm = dec->stats().last;
                    st->run_ms      += lm.wall_ns / 1e6;
                    st->cpu_busy_ms += lm.cpu.busy_ns / 1e6;
                    st->gpu_busy_ms += lm.gpu.busy_ns / 1e6;
                    st->cpu_bytes   += (double)lm.cpu.bytes;
                    st->gpu_bytes   += (double)lm.gpu.bytes;
                }
            }
        } else {
            // speculative loop: t_prev = next_tok is emitted but not yet fed
            std::vector<int32_t> toks(spec_max + 1), outs(spec_max + 1);
            std::vector<int32_t> draft(spec_max);
            while ((int)out.size() < n_gen) {
                out.push_back(next_tok);            // emit t_prev
                if ((int)out.size() >= n_gen) break;
                idx.push(next_tok);

                int want = spec_fixed >= 0 ? spec_fixed : ctrl.choose();
                if (want > spec_max) want = spec_max;
                SuffixIndex::Diag dg;
                int nd = want > 0 ? idx.draft(want, draft.data(), &dg) : 0;

                const int m = nd + 1;
                toks[0] = next_tok;
                for (int k = 0; k < nd; ++k) toks[k + 1] = draft[k];
                const int32_t base = dec->n_past();
                auto p0 = std::chrono::steady_clock::now();
                dec->step_batch(toks.data(), m, outs.data());
                auto p1 = std::chrono::steady_clock::now();
                const double pass_ms =
                    std::chrono::duration<double, std::milli>(p1 - p0).count();

                int a = 0;
                while (a < nd && outs[a] == draft[a]) a++;
                // P3.5 margin gate: a matched-but-low-margin approximated
                // position may not be trusted as a correction source, so
                // acceptance truncates there. Positions 0 and m-1 are always
                // exact, so a >= 1 whenever this fires and progress is made.
                bool trunc = false;
                if (approx_tau > 0.f && a > 0) {
                    const float * mg = dec->last_margins();
                    for (int p = 1; p < a; ++p)
                        if (mg[p] <= approx_tau) { a = p; trunc = true; break; }
                }
                // emit accepted drafts; correction becomes the new t_prev
                for (int k = 0; k < a && (int)out.size() < n_gen; ++k) {
                    out.push_back(draft[k]);
                    idx.push(draft[k]);
                }
                if (trunc) {
                    // do not emit outs[a] (approximate, near-tie): re-decide it
                    // exactly next pass, whose position 0 is never approximated
                    next_tok = draft[a - 1];
                    dec->rewind_to(base + a);
                    if (st) st->trunc++;
                } else {
                    next_tok = outs[a];
                    dec->rewind_to(base + a + 1);
                }
                if (spec_fixed < 0) ctrl.observe(nd, a, pass_ms);
                if (st) {
                    st->passes++;
                    st->drafted += nd;
                    st->accepted += a;
                    st->n_hist[nd < 0 ? 0 : (nd > 16 ? 16 : nd)]++;
                    const RunMetrics & lm = dec->stats().last;
                    st->run_ms      += lm.wall_ns / 1e6;
                    st->cpu_busy_ms += lm.cpu.busy_ns / 1e6;
                    st->gpu_busy_ms += lm.gpu.busy_ns / 1e6;
                    st->host_ms     += pass_ms - lm.wall_ns / 1e6;
                    st->cpu_bytes   += (double)lm.cpu.bytes;
                    st->gpu_bytes   += (double)lm.gpu.bytes;
                }
                if (dbg)
                    fprintf(stderr,
                            "SPEC,%d,%d,%d,%d,%.3f,%.3f,%d,%d,%d,%d,%d\n",
                            (int)out.size(), nd, a, dec->n_past(), pass_ms,
                            ctrl.alpha(), want, dg.gram_seen, dg.cands,
                            dg.best_len, dg.walked);
            }
        }
        auto d1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(d1 - d0).count();
        if (print) {
            printf("ids:");
            for (int32_t t : out) printf(" %d", t);
            printf("\ntext: %s\n", dec->detok(out).c_str());
            fflush(stdout);
        }
        return ms;
    };

    if (!bench) {
        SpecStats st;
        double ms = run_once(true, &st);
        fprintf(stderr, "decode: %d tokens in %.1f ms = %.2f tok/s\n",
                n_gen, ms, n_gen * 1000.0 / ms);
        if (spec && st.passes)
            fprintf(stderr, "spec: %d passes, %.2f tok/pass, alpha %.3f\n",
                    st.passes, (double)n_gen / st.passes,
                    st.drafted ? (double)st.accepted / st.drafted : 0.0);
    } else {
        run_once(true, nullptr);   // warmup + output sanity
        double best = 1e30, sum = 0, sum2 = 0;
        SpecStats st;
        for (int r = 0; r < reps; ++r) {
            SpecStats sr;
            double ms = run_once(false, &sr);
            sum += ms; sum2 += ms * ms;
            if (ms < best) best = ms;
            st = sr;   // keep last rep's spec stats
        }
        const double mean = sum / reps;
        const double sd = reps > 1 ? sqrt((sum2 - sum * sum / reps) / (reps - 1)) : 0;
        const double tps  = n_gen * 1000.0 / mean;
        const double tsd  = tps * sd / mean;
        if (json) {
            printf("{\"tokens\":%d,\"reps\":%d,\"ms_mean\":%.3f,\"ms_sd\":%.3f,"
                   "\"tok_s\":%.3f,\"tok_s_sd\":%.3f,\"spec\":%d",
                   n_gen, reps, mean, sd, tps, tsd, spec ? 1 : 0);
            // P2.2 BENCH rule: every cell logs its full input id list (the
            // P2.1 W1 "copy" cell is unreproducible because it did not).
            printf(",\"input_ids\":[");
            for (size_t i = 0; i < ids.size(); ++i)
                printf("%s%d", i ? "," : "", ids[i]);
            printf("]");
            {   // P9.3 bus occupancy. Thor's measured read ceiling is 262.8
                // GB/s (P7.1). Bytes are for ONE rep (st is the last rep's
                // stats), so pair them with that rep's own window, not the
                // mean over reps.
                const double gb  = (st.cpu_bytes + st.gpu_bytes) / 1e9;
                const double sec = st.run_ms / 1e3;
                printf(",\"gb_per_tok\":%.4f,\"bus_gbs\":%.1f,"
                       "\"gpu_gbs\":%.1f,\"cpu_gbs\":%.1f,\"occupancy_pct\":%.1f",
                       n_gen ? gb / n_gen : 0.0,
                       sec > 0 ? gb / sec : 0.0,
                       sec > 0 ? st.gpu_bytes / 1e9 / sec : 0.0,
                       sec > 0 ? st.cpu_bytes / 1e9 / sec : 0.0,
                       sec > 0 ? 100.0 * (gb / sec) / 262.8 : 0.0);
            }
            if (spec) {
                printf(",\"passes\":%d,\"drafted\":%d,\"accepted\":%d,"
                       "\"trunc\":%d,"
                       "\"alpha\":%.4f,\"tok_per_pass\":%.3f,\"n_hist\":[",
                       st.passes, st.drafted, st.accepted, st.trunc,
                       st.drafted ? (double)st.accepted / st.drafted : 0.0,
                       st.passes ? (double)n_gen / st.passes : 0.0);
                for (int k = 0; k <= 16; ++k)
                    printf("%s%d", k ? "," : "", st.n_hist[k]);
                printf("]");
                // P3.3 chaining gate (last rep): engine-window share of wall,
                // and average worker parallelism inside the window. The
                // chaining ceiling is 1 - run_ms/wall (host serialization);
                // sub-saturation parallelism is the intra-window idle that
                // pass k+1's independent work could also fill.
                printf(",\"chain\":{\"run_ms\":%.3f,\"host_in_step_ms\":%.3f,"
                       "\"wall_ms\":%.3f,\"cpu_busy_ms\":%.3f,"
                       "\"gpu_busy_ms\":%.3f,\"cpu_par\":%.3f,\"gpu_par\":%.3f,"
                       "\"engine_share\":%.4f}",
                       st.run_ms, st.host_ms, mean, st.cpu_busy_ms,
                       st.gpu_busy_ms,
                       st.run_ms > 0 ? st.cpu_busy_ms / st.run_ms : 0.0,
                       st.run_ms > 0 ? st.gpu_busy_ms / st.run_ms : 0.0,
                       mean > 0 ? st.run_ms / mean : 0.0);
            }
            const DecodeStats & dst = dec->stats();
            if (dst.cert_rows_total) {
                printf(",\"cert_skip\":%.5f,\"cert_rows\":%llu,"
                       "\"cert_bytes\":%llu,\"cert_worst\":%llu,"
                       "\"cert_ms\":%.2f",
                       1.0 - (double)dst.cert_rows_eval / dst.cert_rows_total,
                       (unsigned long long)dst.cert_rows_eval,
                       (unsigned long long)dst.cert_bytes,
                       (unsigned long long)dst.cert_worst,
                       dst.ph_cert_ns / 1e6);
            }
            const RunMetrics & m = dec->stats().last;
            printf(",\"kinds\":{");
            static const char * kn[] = {"dot","merge","signal","actq8","dot_idx",
                "rmsnorm","add2","rope","kv_append","attn_head","router",
                "silu_mul","wmerge"};
            bool first = true;
            for (int k = 0; k < kNumTaskKinds; ++k) {
                if (!m.kind_n[k]) continue;
                printf("%s\"%s\":{\"n\":%llu,\"us\":%.1f}", first ? "" : ",",
                       kn[k], (unsigned long long)m.kind_n[k],
                       m.kind_ns[k] / 1e3);
                first = false;
            }
            printf("}");
            // P2.4 pass decomposition, per-pass means over the whole run
            {
                const DecodeStats & ds = dec->stats();
                const double np = ds.ph_passes ? (double)ds.ph_passes : 1.0;
                printf(",\"pass\":{\"passes\":%llu,\"positions\":%llu,"
                       "\"embed_ms\":%.3f,\"zero_ms\":%.3f,\"reset_ms\":%.3f,"
                       "\"run_ms\":%.3f,\"argmax_ms\":%.3f,"
                       "\"bytes_per_pass\":%.0f}",
                       (unsigned long long)ds.ph_passes,
                       (unsigned long long)ds.ph_positions,
                       ds.ph_embed_ns / 1e6 / np, ds.ph_zero_ns / 1e6 / np,
                       ds.ph_reset_ns / 1e6 / np, ds.ph_run_ns / 1e6 / np,
                       ds.ph_argmax_ns / 1e6 / np,
                       ds.ph_passes ? (double)ds.ph_bytes / np : 0.0);
            }
            printf("}\n");
        } else {
            {   // P2.4 pass decomposition (human-readable)
                const DecodeStats & ds = dec->stats();
                const double np = ds.ph_passes ? (double)ds.ph_passes : 1.0;
                fprintf(stderr, "pass: %llu passes, %.2f pos/pass | per pass: "
                        "embed %.2f zero %.2f reset %.2f run %.2f argmax %.2f ms"
                        " | %.1f MB/pass\n",
                        (unsigned long long)ds.ph_passes,
                        (double)ds.ph_positions / np,
                        ds.ph_embed_ns / 1e6 / np, ds.ph_zero_ns / 1e6 / np,
                        ds.ph_reset_ns / 1e6 / np, ds.ph_run_ns / 1e6 / np,
                        ds.ph_argmax_ns / 1e6 / np,
                        (double)ds.ph_bytes / np / 1e6);
            }
            printf("decode: %.2f +/- %.2f tok/s (%d tokens x %d reps, "
                   "best %.1f ms)", tps, tsd, n_gen, reps, best);
            if (spec && st.passes)
                printf(" | spec %.2f tok/pass alpha %.3f",
                       (double)n_gen / st.passes,
                       st.drafted ? (double)st.accepted / st.drafted : 0.0);
            printf("\n");
        }
    }
    return 0;
}
