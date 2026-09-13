// Longest-suffix-match draft source over the live token stream.
// LZ77/deflate-lineage hash chains: every position is indexed
// by a hash of its trailing H=3 tokens; a draft walks the chain of prior
// occurrences of the current tail gram, keeps the LONGEST full suffix match
// (ties -> most recent), and proposes the continuation after that match.
// O(1) amortized append; draft cost bounded by chain_cap * match walk.
// Self-contained, no llama/fastllm deps; ASCII only.
#pragma once
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace fastllm {

class SuffixIndex {
public:
    explicit SuffixIndex(uint32_t chain_cap = 64, uint32_t min_match = 3)
        : chain_cap_(chain_cap), min_match_(min_match) {}

    void reset() {
        toks_.clear();
        prev_.clear();
        head_.clear();
    }

    // append one token; indexes the gram ending at this position
    void push(int32_t t) {
        toks_.push_back(t);
        prev_.push_back(-1);
        const size_t n = toks_.size();
        if (n < H) return;
        const uint64_t h = gram_hash(n - H);
        auto it = head_.find(h);
        if (it != head_.end()) prev_[n - 1] = it->second;
        head_[h] = (int64_t)(n - 1);   // chain stores END position of gram
    }

    size_t size() const { return toks_.size(); }

    // P4.7 miss attribution: a draft returning 0 is ambiguous (no prior gram
    // at all, vs a gram whose match was shorter than min_match, vs a match
    // with no room to continue). Those want different fixes, so the caller
    // can ask. Optional and default-null: unused it costs nothing.
    struct Diag {
        int gram_seen = 0;    // the tail gram had a chain (coverage)
        int cands = 0;        // chain entries that survived the collision check
        int best_len = 0;     // longest full suffix match found (0 = none)
        int walked = 0;       // chain steps taken (cap pressure if == chain_cap)
    };

    // longest-suffix-match draft: up to n_max continuation tokens after the
    // best match of the current tail. Returns count written to out (0 if no
    // match >= min_match). Never proposes past the end of known stream.
    int draft(int n_max, int32_t * out, Diag * dg = nullptr) const {
        const size_t n = toks_.size();
        if (n < H || n_max <= 0) return 0;
        const uint64_t h = gram_hash(n - H);
        auto it = head_.find(h);
        if (it == head_.end()) return 0;
        if (dg) dg->gram_seen = 1;

        int64_t best_end = -1;
        size_t  best_len = 0;
        uint32_t walked = 0;
        for (int64_t e = it->second; e >= 0 && walked < chain_cap_;
             e = prev_[e], ++walked) {
            // e is the end position of a candidate gram occurrence; the
            // candidate must be a PRIOR occurrence with room to continue
            if ((size_t)e == n - 1) continue;          // the tail itself
            if (!gram_eq((size_t)e + 1 - H, n - H)) continue;  // hash collision
            // extend the match backward beyond the gram
            size_t len = H;
            while (len < (size_t)e + 1 && len < n - 1
                   && toks_[e - len] == toks_[n - 1 - len])
                ++len;
            if (dg) dg->cands++;
            if (len > best_len) {          // strict: ties keep the more
                best_len = len;            // recent (chain is recency-ordered,
                best_end = e;              // first-found wins ties)
            }
        }
        if (dg) { dg->best_len = (int)best_len; dg->walked = (int)walked; }
        if (best_end < 0 || best_len < min_match_) return 0;

        int cnt = 0;
        for (size_t p = (size_t)best_end + 1; p < n && cnt < n_max; ++p)
            out[cnt++] = toks_[p];
        return cnt;
    }

private:
    static constexpr size_t H = 3;

    uint64_t gram_hash(size_t start) const {
        // splitmix-style mix of the 3 tokens
        uint64_t x = 0x9e3779b97f4a7c15ull;
        for (size_t i = 0; i < H; ++i) {
            x ^= (uint64_t)(uint32_t)toks_[start + i] + 0x9e3779b97f4a7c15ull
                 + (x << 6) + (x >> 2);
        }
        return x;
    }
    bool gram_eq(size_t a, size_t b) const {
        for (size_t i = 0; i < H; ++i)
            if (toks_[a + i] != toks_[b + i]) return false;
        return true;
    }

    uint32_t chain_cap_, min_match_;
    std::vector<int32_t> toks_;
    std::vector<int64_t> prev_;                    // per END position
    std::unordered_map<uint64_t, int64_t> head_;   // gram hash -> latest end
};

} // namespace fastllm
