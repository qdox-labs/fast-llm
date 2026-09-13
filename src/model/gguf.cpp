#include "model/gguf.h"
#include "runtime/quants.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fastllm {

namespace {

struct Cursor {
    const uint8_t * p;
    const uint8_t * end;
    void need(size_t n) {
        if ((size_t)(end - p) < n) {
            fprintf(stderr, "gguf: truncated file\n");
            abort();
        }
    }
    template <typename T> T get() {
        need(sizeof(T));
        T v;
        memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        uint64_t n = get<uint64_t>();
        need(n);
        std::string s((const char *)p, n);
        p += n;
        return s;
    }
    void skip(size_t n) { need(n); p += n; }
};

// GGUF value type ids
enum : uint32_t {
    GV_U8 = 0, GV_I8, GV_U16, GV_I16, GV_U32, GV_I32, GV_F32, GV_BOOL,
    GV_STR, GV_ARR, GV_U64, GV_I64, GV_F64,
};

uint64_t scalar_size(uint32_t t) {
    switch (t) {
        case GV_U8: case GV_I8: case GV_BOOL: return 1;
        case GV_U16: case GV_I16: return 2;
        case GV_U32: case GV_I32: case GV_F32: return 4;
        case GV_U64: case GV_I64: case GV_F64: return 8;
    }
    return 0;
}

// skip a KV value, returning it as u64 when it is an unsigned scalar (for
// general.alignment); 0 otherwise
uint64_t skip_value(Cursor & c, uint32_t t) {
    if (t == GV_STR) { c.str(); return 0; }
    if (t == GV_ARR) {
        uint32_t et = c.get<uint32_t>();
        uint64_t n  = c.get<uint64_t>();
        if (et == GV_STR) { for (uint64_t i = 0; i < n; ++i) c.str(); }
        else if (et == GV_ARR) {
            for (uint64_t i = 0; i < n; ++i) skip_value(c, GV_ARR);
        } else c.skip(n * scalar_size(et));
        return 0;
    }
    uint64_t sz = scalar_size(t);
    if (!sz) { fprintf(stderr, "gguf: bad kv type %u\n", t); abort(); }
    uint64_t v = 0;
    c.need(sz);
    memcpy(&v, c.p, sz < 8 ? sz : 8);
    c.p += sz;
    return (t == GV_U8 || t == GV_U16 || t == GV_U32 || t == GV_U64) ? v : 0;
}

} // namespace

bool Gguf::fmt_of(uint32_t t, Fmt * out) {
    switch (t) {
        case GGML_TYPE_F32:  *out = Fmt::F32;  return true;
        case GGML_TYPE_Q8_0: *out = Fmt::Q8_0; return true;
        case GGML_TYPE_Q5_0: *out = Fmt::Q5_0; return true;
        case GGML_TYPE_Q4_K: *out = Fmt::Q4_K; return true;
        case GGML_TYPE_Q6_K: *out = Fmt::Q6_K; return true;
        case GGML_TYPE_IQ2_XXS: *out = Fmt::IQ2_XXS; return true;
        case GGML_TYPE_IQ3_XXS: *out = Fmt::IQ3_XXS; return true;
    }
    return false;
}

const char * Gguf::type_name(uint32_t t) {
    switch (t) {
        case GGML_TYPE_F32: return "F32";
        case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
        case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
    }
    static char buf[16];
    snprintf(buf, sizeof buf, "type%u", t);
    return buf;
}

// A split model is named "<stem>-00001-of-000NN.gguf". Returns NN and writes a
// printf pattern for the sibling shards; returns 0 when the name is not split.
static int split_parts(const std::string & path, std::string * pattern) {
    const std::string suf = ".gguf";
    if (path.size() < suf.size() + 14) return 0;
    if (path.compare(path.size() - suf.size(), suf.size(), suf) != 0) return 0;
    const size_t tail = path.size() - suf.size();   // index of ".gguf"
    if (tail < 14) return 0;
    const size_t of = tail - 9;                     // expected "-of-" position
    if (path.compare(of, 4, "-of-") != 0) return 0;
    for (size_t i = of + 4; i < tail; ++i) if (!isdigit((unsigned char)path[i])) return 0;
    // shard index is "-NNNNN" (hyphen + 5 digits) immediately before "-of-"
    if (of < 6) return 0;
    const size_t nb = of - 6;
    if (path[nb] != '-') return 0;
    for (size_t i = nb + 1; i < of; ++i) if (!isdigit((unsigned char)path[i])) return 0;
    const int total = atoi(path.c_str() + of + 4);
    if (total < 1 || total > 999) return 0;
    *pattern = path.substr(0, nb) + "-%05d-of-" + path.substr(of + 4, tail - (of + 4)) + suf;
    return total;
}

bool Gguf::open(const std::string & path) {
    std::string pattern;
    const int total = split_parts(path, &pattern);
    if (total <= 1) return map_one(path, 0, true);

    // Split model: map every shard. Metadata comes from the file the caller
    // named - shard 1 in practice, which carries all KV and may hold NO
    // tensors at all (V4 Flash keeps its 1328 tensors in shards 2 and 3).
    char buf[1024];
    for (int i = 1; i <= total; ++i) {
        snprintf(buf, sizeof buf, pattern.c_str(), i);
        const bool keep_kv = (path == buf);
        if (!map_one(buf, (int)shards_.size(), keep_kv)) {
            fprintf(stderr, "gguf: missing shard %s\n", buf);
            return false;
        }
    }
    const int64_t want = kv_i("split.tensors.count", 0);
    if (want > 0 && (int64_t)tensors_.size() != want) {
        fprintf(stderr, "gguf: split.tensors.count=%lld but mapped %zu\n",
                (long long)want, tensors_.size());
        abort();
    }
    return true;
}

bool Gguf::map_one(const std::string & path, int idx, bool keep_kv) {
    Shard sh;
    sh.fd = ::open(path.c_str(), O_RDONLY);
    if (sh.fd < 0) return false;
    struct stat st{};
    fstat(sh.fd, &st);
    sh.size = (size_t)st.st_size;
    void * m = mmap(nullptr, sh.size, PROT_READ, MAP_PRIVATE, sh.fd, 0);
    if (m == MAP_FAILED) { fprintf(stderr, "gguf: mmap failed\n"); abort(); }
    sh.base = (const uint8_t *)m;
    const uint8_t * const base_ = sh.base;
    const size_t          size_ = sh.size;

    Cursor c{base_, base_ + size_};
    uint32_t magic = c.get<uint32_t>();
    if (magic != 0x46554747u) { fprintf(stderr, "gguf: bad magic\n"); abort(); }
    uint32_t version = c.get<uint32_t>();
    if (version < 2 || version > 3) {
        fprintf(stderr, "gguf: unsupported version %u\n", version);
        abort();
    }
    uint64_t n_tensors = c.get<uint64_t>();
    uint64_t n_kv      = c.get<uint64_t>();

    // Only the shard the caller named contributes metadata; the others are
    // parsed into scratch so the cursor advances but their (partial, and for
    // split.no deliberately different) KV cannot shadow it.
    std::map<std::string, int64_t>     scratch_i;
    std::map<std::string, double>      scratch_f;
    std::map<std::string, std::string> scratch_s;
    std::map<std::string, std::vector<std::string>> scratch_sa;
    auto & kvi  = keep_kv ? kv_i_  : scratch_i;
    auto & kvf  = keep_kv ? kv_f_  : scratch_f;
    auto & kvs  = keep_kv ? kv_s_  : scratch_s;
    auto & kvsa = keep_kv ? kv_sa_ : scratch_sa;

    uint32_t align_local = 32;   // GGUF alignment is a per-file property
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = c.str();
        uint32_t t = c.get<uint32_t>();
        // P2.0: capture scalars + string arrays instead of skipping
        switch (t) {
            case GV_U8:   kvi[key] = c.get<uint8_t>();  break;
            case GV_I8:   kvi[key] = c.get<int8_t>();   break;
            case GV_BOOL: kvi[key] = c.get<uint8_t>() != 0; break;
            case GV_U16:  kvi[key] = c.get<uint16_t>(); break;
            case GV_I16:  kvi[key] = c.get<int16_t>();  break;
            case GV_U32:  kvi[key] = c.get<uint32_t>(); break;
            case GV_I32:  kvi[key] = c.get<int32_t>();  break;
            case GV_U64:  kvi[key] = (int64_t)c.get<uint64_t>(); break;
            case GV_I64:  kvi[key] = c.get<int64_t>();  break;
            case GV_F32:  kvf[key] = c.get<float>();    break;
            case GV_F64:  kvf[key] = c.get<double>();   break;
            case GV_STR:  kvs[key] = c.str();           break;
            case GV_ARR: {
                uint32_t et = c.get<uint32_t>();
                uint64_t n  = c.get<uint64_t>();
                if (et == GV_STR) {
                    auto & v = kvsa[key];
                    v.reserve(n);
                    for (uint64_t j = 0; j < n; ++j) v.push_back(c.str());
                } else if (et == GV_ARR) {
                    for (uint64_t j = 0; j < n; ++j) skip_value(c, GV_ARR);
                } else {
                    c.skip(n * scalar_size(et));
                }
                break;
            }
            default:
                fprintf(stderr, "gguf: bad kv type %u\n", t);
                abort();
        }
        if (key == "general.alignment") {
            auto it = kvi.find(key);
            if (it != kvi.end() && it->second > 0) align_local = (uint32_t)it->second;
        }
    }
    if (keep_kv) align_ = align_local;

    tensors_.reserve(tensors_.size() + n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensor t;
        t.name   = c.str();
        t.n_dims = (int)c.get<uint32_t>();
        if (t.n_dims < 1 || t.n_dims > 4) {
            fprintf(stderr, "gguf: bad n_dims\n");
            abort();
        }
        for (int d = 0; d < t.n_dims; ++d) t.ne[d] = c.get<uint64_t>();
        t.type   = c.get<uint32_t>();
        t.offset = c.get<uint64_t>();
        Fmt f;
        if (fmt_of(t.type, &f)) {
            t.nbytes = fl_row_bytes(f, (uint32_t)t.ne[0])
                     * t.ne[1] * t.ne[2] * t.ne[3];
        } else if (t.type == GGML_TYPE_F16) {
            t.nbytes = 2 * t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
        } else {
            t.nbytes = 0;   // unknown type: metadata visible, payload unusable
        }
        t.shard = idx;      // offsets are relative to THIS shard's data section
        tensors_.push_back(std::move(t));
    }
    const uint64_t off = (uint64_t)(c.p - base_);
    sh.data_off = (off + align_local - 1) & ~(uint64_t)(align_local - 1);
    shards_.push_back(sh);
    return true;
}

bool Gguf::has(const std::string & k) const {
    return kv_i_.count(k) || kv_f_.count(k) || kv_s_.count(k) || kv_sa_.count(k);
}
int64_t Gguf::kv_i(const std::string & k, int64_t dflt) const {
    auto it = kv_i_.find(k);
    if (it != kv_i_.end()) return it->second;
    auto itf = kv_f_.find(k);
    return itf != kv_f_.end() ? (int64_t)itf->second : dflt;
}
double Gguf::kv_f(const std::string & k, double dflt) const {
    auto it = kv_f_.find(k);
    if (it != kv_f_.end()) return it->second;
    auto iti = kv_i_.find(k);
    return iti != kv_i_.end() ? (double)iti->second : dflt;
}
std::string Gguf::kv_s(const std::string & k, const std::string & dflt) const {
    auto it = kv_s_.find(k);
    return it != kv_s_.end() ? it->second : dflt;
}
const std::vector<std::string> * Gguf::kv_sarr(const std::string & k) const {
    auto it = kv_sa_.find(k);
    return it != kv_sa_.end() ? &it->second : nullptr;
}

const GgufTensor * Gguf::find(const std::string & name) const {
    for (const auto & t : tensors_)
        if (t.name == name) return &t;
    return nullptr;
}

Gguf::~Gguf() {
    for (Shard & s : shards_) {
        if (s.base) munmap((void *)s.base, s.size);
        if (s.fd >= 0) close(s.fd);
    }
}

} // namespace fastllm
