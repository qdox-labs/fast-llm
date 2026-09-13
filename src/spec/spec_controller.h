// Adaptive draft-length controller. Per round choose
//   n* = argmax over n in {0..n_max} of E[emitted](alpha_hat, n) / latency(n)
// with E[emitted] = (1 - alpha^(n+1)) / (1 - alpha)  (accepted geometric
// prefix + the correction token). alpha_hat is an EMA of per-round accept
// fractions; latency(n) is a self-calibrating EMA per batch bucket, seeded
// from a mildly superlinear cost model off the measured n=0 bucket. n*=0
// means a plain single-token pass. Forced probe rounds keep alpha_hat from
// locking stale after a run of n*=0. Bandit flavor: exploit the argmax,
// probe for alpha drift. Self-contained; ASCII only.
#pragma once
#include <cstdint>
#include <cstdlib>

namespace fastllm {

class SpecController {
public:
    struct Cfg {
        int    n_max        = 7;      // drafts; batch m = n+1
        double alpha0       = 0.75;
        double alpha_decay  = 0.90;
        double lat_decay    = 0.80;
        int    probe_iv     = 16;     // draftless rounds before a probe
        int    probe_n      = 2;
        double seed_slope   = 0.15;   // latency seed: lat1 * (1 + slope*n)
    };

    SpecController() { init(); }
    explicit SpecController(const Cfg & c) : c_(c) { init(); }

    // choose the draft length for this round (0 => no speculation)
    int choose() {
        if (lat_ms_[0] < 0.0) return c_.n_max;   // cold start: try full draft
        int best_n = 0;
        double best_rate = 1.0 / lat_est(0);
        for (int n = 1; n <= c_.n_max; ++n) {
            const double r = expected_emitted(n) / lat_est(n);
            if (r > best_rate) { best_rate = r; best_n = n; }
        }
        if (best_n == 0) {
            if (++zero_streak_ >= c_.probe_iv) {
                zero_streak_ = 0;
                return c_.probe_n;                 // forced probe
            }
        } else {
            zero_streak_ = 0;
        }
        return best_n;
    }

    // n_drafted may be less than requested (drafter found a shorter match);
    // n_accepted <= n_drafted; pass_ms = wall of the whole verify pass
    void observe(int n_drafted, int n_accepted, double pass_ms) {
        const int b = bucket(n_drafted);
        lat_ms_[b] = lat_ms_[b] < 0.0
            ? pass_ms
            : c_.lat_decay * lat_ms_[b] + (1.0 - c_.lat_decay) * pass_ms;
        if (n_drafted > 0) {
            const double frac = (double)n_accepted / (double)n_drafted;
            alpha_ = c_.alpha_decay * alpha_ + (1.0 - c_.alpha_decay) * frac;
            if (alpha_ < 0.02) alpha_ = 0.02;
            if (alpha_ > 0.98) alpha_ = 0.98;
        }
    }

    double alpha() const { return alpha_; }
    double lat_est(int n) const {
        const int b = bucket(n);
        if (lat_ms_[b] >= 0.0) return lat_ms_[b];
        // seed unseen buckets from the nearest measured one, superlinear-ish
        for (int d = 1; d <= kMaxN; ++d) {
            if (b - d >= 0 && lat_ms_[b - d] >= 0.0)
                return lat_ms_[b - d] * (1.0 + c_.seed_slope * d);
            if (b + d <= kMaxN && lat_ms_[b + d] >= 0.0)
                return lat_ms_[b + d] / (1.0 + c_.seed_slope * d);
        }
        return 1.0;   // nothing measured yet
    }
    double expected_emitted(int n) const {
        // 1 + alpha + alpha^2 + ... + alpha^n  (accepted prefix + correction)
        double e = 0.0, p = 1.0;
        for (int i = 0; i <= n; ++i) { e += p; p *= alpha_; }
        return e;
    }

    static Cfg cfg_from_env() {
        Cfg c;
        if (const char * e = getenv("FASTLLM_SPEC_ALPHA0"))   c.alpha0 = atof(e);
        if (const char * e = getenv("FASTLLM_SPEC_ADECAY"))   c.alpha_decay = atof(e);
        if (const char * e = getenv("FASTLLM_SPEC_LDECAY"))   c.lat_decay = atof(e);
        if (const char * e = getenv("FASTLLM_SPEC_PROBE_IV")) c.probe_iv = atoi(e);
        return c;
    }

private:
    static constexpr int kMaxN = 15;
    void init() {
        alpha_ = c_.alpha0;
        for (int i = 0; i <= kMaxN; ++i) lat_ms_[i] = -1.0;
    }
    int bucket(int n) const { return n < 0 ? 0 : (n > kMaxN ? kMaxN : n); }

    Cfg    c_;
    double alpha_ = 0.75;
    double lat_ms_[kMaxN + 1];
    int    zero_streak_ = 0;
};

} // namespace fastllm
