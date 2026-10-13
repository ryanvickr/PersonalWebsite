#include "tlm/model.hpp"

#include <cstring>

namespace tlm {
namespace {

constexpr char kMagic[4] = {'T', 'L', 'M', '1'};
constexpr uint32_t kVersion = 1;
constexpr size_t kAlign = 64;

// Walks the file in the same order tools/export.py writes it.
class Reader {
public:
    explicit Reader(std::span<const uint8_t> file) : file_(file), pos_(kAlign) {}

    template <typename T>
    const T* take(size_t count) {
        pos_ = (pos_ + kAlign - 1) / kAlign * kAlign;
        size_t bytes = count * sizeof(T);
        if (pos_ + bytes > file_.size()) {
            ok_ = false;
            return nullptr;
        }
        const T* p = reinterpret_cast<const T*>(file_.data() + pos_);
        pos_ += bytes;
        return p;
    }

    QTensor q8(size_t count, int gs) {
        QTensor t;
        t.q = take<int8_t>(count);
        t.s = take<float>(count / gs);
        return t;
    }

    bool ok() const { return ok_; }
    size_t pos() const { return pos_; }

private:
    std::span<const uint8_t> file_;
    size_t pos_;
    bool ok_ = true;
};

}  // namespace

std::string ModelFile::parse(std::span<const uint8_t> file) {
    if (file.size() < kAlign || std::memcmp(file.data(), kMagic, 4) != 0) return "not a .tlm file";
    if (reinterpret_cast<uintptr_t>(file.data()) % alignof(float) != 0) return "buffer is not 4-byte aligned";

    uint32_t version;
    int32_t h[11];
    std::memcpy(&version, file.data() + 4, 4);
    std::memcpy(h, file.data() + 8, sizeof(h));
    if (version != kVersion) return "unsupported .tlm version " + std::to_string(version);

    Config& c = config;
    c.dim = h[0];
    c.hidden_dim = h[1];
    c.n_layers = h[2];
    c.n_heads = h[3];
    c.n_kv_heads = h[4];
    c.vocab_size = h[5];
    c.seq_len = h[6];
    c.group_size = h[7];
    c.shared_classifier = h[8] != 0;
    const size_t tok_offset = static_cast<uint32_t>(h[9]);
    const size_t tok_size = static_cast<uint32_t>(h[10]);

    if (c.dim <= 0 || c.n_layers <= 0 || c.n_heads <= 0 || c.n_kv_heads <= 0 || c.vocab_size <= 0 ||
        c.seq_len <= 0 || c.group_size <= 0 || c.dim % c.n_heads != 0 || c.n_heads % c.n_kv_heads != 0 ||
        c.dim % c.group_size != 0 || c.hidden_dim % c.group_size != 0 || c.kv_dim() % c.group_size != 0)
        return "invalid model config";

    const size_t dim = c.dim, hidden = c.hidden_dim, kv = c.kv_dim(), L = c.n_layers, gs = c.group_size;
    Reader r(file);
    const float* rms_att = r.take<float>(L * dim);
    const float* rms_ffn = r.take<float>(L * dim);
    weights.rms_final = r.take<float>(dim);
    weights.tok_embedding = r.q8(c.vocab_size * dim, gs);
    weights.layers.resize(L);
    for (size_t l = 0; l < L; ++l) {
        LayerWeights& lw = weights.layers[l];
        lw.rms_att = rms_att + l * dim;
        lw.rms_ffn = rms_ffn + l * dim;
        lw.wq = r.q8(dim * dim, gs);
        lw.wk = r.q8(kv * dim, gs);
        lw.wv = r.q8(kv * dim, gs);
        lw.wo = r.q8(dim * dim, gs);
        lw.w1 = r.q8(hidden * dim, gs);
        lw.w2 = r.q8(dim * hidden, gs);
        lw.w3 = r.q8(hidden * dim, gs);
    }
    weights.wcls = c.shared_classifier ? weights.tok_embedding : r.q8(c.vocab_size * dim, gs);
    if (!r.ok()) return "file truncated";
    if (tok_offset < r.pos() || tok_offset + tok_size > file.size()) return "bad tokenizer offset";
    tokenizer = file.subspan(tok_offset, tok_size);
    return {};
}

}  // namespace tlm
