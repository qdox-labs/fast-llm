// P2.1 spec component tests: suffix index vs brute-force reference,
// controller policy behaviors. CPU-only, no model needed.
#include "spec/suffix_index.h"
#include "spec/spec_controller.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace fastllm;

static int g_checks = 0, g_fails = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fails; \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); } } while (0)

// brute-force longest-suffix-match reference: same definition as the index
// (candidate end positions e < n-1, match extends backward, ties -> most
// recent), continuation after the match end.
static int ref_draft(const std::vector<int32_t> & t, int n_max,
                     uint32_t min_match, int32_t * out) {
    const size_t n = t.size();
    if (n < 3 || n_max <= 0) return 0;
    size_t best_len = 0;
    int64_t best_end = -1;
    for (int64_t e = (int64_t)n - 2; e >= 0; --e) {
        size_t len = 0;
        while (len < (size_t)e + 1 && len < n - 1
               && t[e - len] == t[n - 1 - len])
            ++len;
        if (len > best_len) { best_len = len; best_end = e; }
    }
    if (best_end < 0 || best_len < min_match) return 0;
    int cnt = 0;
    for (size_t p = (size_t)best_end + 1; p < n && cnt < n_max; ++p)
        out[cnt++] = t[p];
    return cnt;
}

static void test_suffix_vs_ref(uint32_t seed, int n_toks, int vocab) {
    std::mt19937 rng(seed);
    SuffixIndex idx(1u << 20, 3);
    std::vector<int32_t> stream;
    int32_t a[64], b[64];
    for (int i = 0; i < n_toks; ++i) {
        // mix: random tokens with planted repeats of earlier spans
        if (i > 20 && rng() % 4 == 0) {
            const int start = (int)(rng() % (stream.size() - 10));
            const int len = 3 + (int)(rng() % 8);
            for (int k = 0; k < len && (int)stream.size() < n_toks; ++k) {
                const int32_t t = stream[start + k];
                stream.push_back(t);
                idx.push(t);
            }
        } else {
            const int32_t t = (int32_t)(rng() % vocab);
            stream.push_back(t);
            idx.push(t);
        }
        if (stream.size() > 8 && stream.size() % 7 == 0) {
            const int na = idx.draft(16, a);
            const int nb = ref_draft(stream, 16, 3, b);
            CHECK(na == nb);
            if (na == nb)
                for (int k = 0; k < na; ++k) CHECK(a[k] == b[k]);
        }
        i = (int)stream.size() - 1;
    }
}

static void test_suffix_basics() {
    SuffixIndex idx(64, 3);
    int32_t out[16];
    // no match on short/distinct streams
    for (int32_t t : {1, 2, 3, 4, 5}) idx.push(t);
    CHECK(idx.draft(8, out) == 0);
    // planted repeat: ... 10 11 12 13 14 ... 10 11 12 -> continuation starts
    // 13 14 and keeps proposing the stream after the match (up to n_max;
    // acceptance decides how much survives)
    SuffixIndex i2(64, 3);
    for (int32_t t : {10, 11, 12, 13, 14, 7, 8, 9, 10, 11, 12}) i2.push(t);
    const int n = i2.draft(8, out);
    CHECK(n == 8);
    if (n >= 2) { CHECK(out[0] == 13); CHECK(out[1] == 14); }
    // all-same-token stream: longest match wins (most recent end), which
    // leaves 1 continuation token - the documented degenerate-case floor
    SuffixIndex i3(64, 3);
    for (int i = 0; i < 50; ++i) i3.push(42);
    CHECK(i3.draft(4, out) == 1);
    CHECK(out[0] == 42);
}

static void test_controller() {
    // cold start: full draft
    SpecController c0;
    CHECK(c0.choose() == SpecController::Cfg{}.n_max);
    // high acceptance -> large n
    SpecController c1;
    for (int i = 0; i < 30; ++i) c1.observe(4, 4, 10.0 + 0.3 * 4);
    CHECK(c1.alpha() > 0.9);
    CHECK(c1.choose() >= 4);
    // sustained rejection -> n = 0 (until probe)
    SpecController c2;
    for (int i = 0; i < 40; ++i) c2.observe(4, 0, 12.0);
    c2.observe(0, 0, 10.0);   // seed the n=0 latency bucket
    CHECK(c2.alpha() < 0.1);
    int zeros = 0, probes = 0;
    for (int i = 0; i < 40; ++i) {
        const int n = c2.choose();
        if (n == 0) zeros++;
        else { probes++; c2.observe(n, 0, 11.0); }
        if (n == 0) c2.observe(0, 0, 10.0);
    }
    CHECK(zeros > 30);
    CHECK(probes >= 1);       // probe fired despite zero alpha
    // steep latency growth -> smaller n than flat latency at same alpha
    SpecController c3, c4;
    for (int i = 0; i < 20; ++i) {
        for (int n = 0; n <= 7; ++n) {
            c3.observe(n, (int)(0.8 * n + 0.5), 10.0 * (1.0 + 0.05 * n));
            c4.observe(n, (int)(0.8 * n + 0.5), 10.0 * (1.0 + 1.50 * n));
        }
    }
    CHECK(c3.choose() >= c4.choose());
}

int main() {
    test_suffix_basics();
    for (uint32_t s = 1; s <= 8; ++s) test_suffix_vs_ref(s, 600, 50);
    for (uint32_t s = 9; s <= 12; ++s) test_suffix_vs_ref(s, 400, 5);
    test_controller();
    printf("fastllm-spec-tests: %d checks, %d failures\n", g_checks, g_fails);
    return g_fails ? 1 : 0;
}
