// Minimal GGUF v2/v3 reader: mmap the file, expose tensor metadata + raw
// quant-block payload pointers. Host-only, read-only, no dequantization here.
#pragma once
#include "fastllm/fastllm.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace fastllm {

// ggml_type ids as stored in GGUF tensor infos (subset we understand)
enum : uint32_t {
    GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1,
    GGML_TYPE_Q5_0 = 6, GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q4_K = 12, GGML_TYPE_Q6_K = 14,
    GGML_TYPE_IQ2_XXS = 16, GGML_TYPE_IQ3_XXS = 18,
};

struct GgufTensor {
    std::string name;
    uint32_t    type   = 0;      // ggml_type
    int         n_dims = 0;
    uint64_t    ne[4]  = {1, 1, 1, 1};
    uint64_t    offset = 0;      // relative to ITS shard's data section, aligned
    uint64_t    nbytes = 0;      // full payload size
    int         shard  = 0;      // index into the mapped shard list
};

class Gguf {
public:
    // Opens a GGUF. Split models ("...-00001-of-00003.gguf") are detected from
    // the filename and every shard is mapped: metadata comes from the shard
    // given (shard 1 of a split carries all KV and often zero tensors), while
    // tensors are merged across shards, each keeping its own data section.
    // Aborts with a message on malformed files; missing file returns false.
    bool open(const std::string & path);
    ~Gguf();

    const std::vector<GgufTensor> & tensors() const { return tensors_; }
    const GgufTensor * find(const std::string & name) const;
    const uint8_t * data(const GgufTensor & t) const {
        const Shard & s = shards_[t.shard];
        return s.base + s.data_off + t.offset;
    }
    uint32_t alignment() const { return align_; }
    int      n_shards() const { return (int)shards_.size(); }

    // P2.0 metadata access (captured at open()): scalars normalized to
    // int64/double; string arrays kept verbatim (tokenizer.ggml.tokens).
    bool   has(const std::string & k) const;
    int64_t     kv_i(const std::string & k, int64_t dflt) const;
    double      kv_f(const std::string & k, double dflt) const;
    std::string kv_s(const std::string & k, const std::string & dflt = "") const;
    const std::vector<std::string> * kv_sarr(const std::string & k) const;

    // Fmt mapping; returns false for types the engine has no kernels for
    static bool fmt_of(uint32_t ggml_type, Fmt * out);
    static const char * type_name(uint32_t ggml_type);

private:
    struct Shard {
        const uint8_t * base     = nullptr;
        size_t          size     = 0;
        uint64_t        data_off = 0;
        int             fd       = -1;
    };
    // maps one file, appending its tensors with shard index `idx`; keep_kv
    // selects whose metadata wins (the shard the caller named)
    bool map_one(const std::string & path, int idx, bool keep_kv);

    std::map<std::string, int64_t>     kv_i_;
    std::map<std::string, double>      kv_f_;
    std::map<std::string, std::string> kv_s_;
    std::map<std::string, std::vector<std::string>> kv_sa_;
    std::vector<GgufTensor> tensors_;
    std::vector<Shard>      shards_;
    uint32_t align_ = 32;
};

} // namespace fastllm
