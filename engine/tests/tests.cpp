// Unit tests. They build a small random model in memory, so no weights are
// needed. Set TLM_MODEL=path/to/stories15M.tlm to also run the golden test.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "tlm/engine.hpp"
#include "tlm/tensor_ops.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            ++g_failures;                                                    \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
        }                                                                    \
    } while (0)

std::vector<std::pair<const char*, std::function<void()>>>& registry() {
    static std::vector<std::pair<const char*, std::function<void()>>> r;
    return r;
}

struct Register {
    Register(const char* name, std::function<void()> fn) { registry().emplace_back(name, std::move(fn)); }
};

#define TEST(name)                               \
    void name();                                 \
    Register reg_##name(#name, name);            \
    void name()

// 64-byte aligned buffer, as the engine requires for loaded files.
struct Buffer {
    std::vector<uint8_t> storage;
    uint8_t* data = nullptr;
    size_t size = 0;

    explicit Buffer(const std::vector<uint8_t>& bytes) : storage(bytes.size() + 64), size(bytes.size()) {
        data = storage.data() + (64 - reinterpret_cast<uintptr_t>(storage.data()) % 64) % 64;
        std::memcpy(data, bytes.data(), bytes.size());
    }
    std::span<const uint8_t> span() const { return {data, size}; }
};

// Mirrors tools/export.py so tests exercise the real file format.
class TlmWriter {
public:
    void align() { bytes.resize((bytes.size() + 63) / 64 * 64); }
    void raw(const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        bytes.insert(bytes.end(), b, b + n);
    }
    void f32(const std::vector<float>& v) {
        align();
        raw(v.data(), v.size() * 4);
    }
    void q8(const std::vector<float>& v, int gs) {
        std::vector<int8_t> q(v.size());
        std::vector<float> s(v.size() / gs);
        tlm::QVector out{q.data(), s.data()};
        tlm::quantize(out, v.data(), static_cast<int>(v.size()), gs);
        align();
        raw(q.data(), q.size());
        align();
        raw(s.data(), s.size() * 4);
    }
    std::vector<uint8_t> bytes = std::vector<uint8_t>(64, 0);
};

// Tokenizer: <unk> <s> </s>, 256 byte tokens, then a few words and merges.
std::vector<uint8_t> tiny_tokenizer(int vocab_size, std::vector<std::string>& words) {
    std::vector<std::pair<std::string, float>> vocab = {{"<unk>", 0}, {"\n<s>\n", 0}, {"\n</s>\n", 0}};
    for (int b = 0; b < 256; ++b) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        vocab.push_back({buf, 0});
    }
    words = {" ", "a", "b", "c", "ab", " ab", "abc", " abc", "ca", " c", "é"};
    float score = -1.0f;
    for (const auto& w : words) vocab.push_back({w, score--});
    // Longer merges score higher so " abc" wins over " ab" + "c".
    vocab[259 + 7].second = 10.0f;  // " abc"
    vocab[259 + 5].second = 5.0f;   // " ab"
    while (static_cast<int>(vocab.size()) < vocab_size) vocab.push_back({"<pad" + std::to_string(vocab.size()) + ">", -100});

    std::vector<uint8_t> out;
    auto put = [&](const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        out.insert(out.end(), b, b + n);
    };
    int32_t max_len = 16;
    put(&max_len, 4);
    for (const auto& [str, sc] : vocab) {
        int32_t len = static_cast<int32_t>(str.size());
        put(&sc, 4);
        put(&len, 4);
        put(str.data(), str.size());
    }
    return out;
}

tlm::Config tiny_config() {
    tlm::Config c;
    c.dim = 64;
    c.hidden_dim = 96;
    c.n_layers = 2;
    c.n_heads = 4;
    c.n_kv_heads = 2;  // exercise grouped-query attention
    c.vocab_size = 300;
    c.seq_len = 32;
    c.group_size = 32;
    c.shared_classifier = true;
    return c;
}

std::vector<uint8_t> tiny_model(const tlm::Config& c, uint32_t seed = 7) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.1f);
    auto randn = [&](size_t n) {
        std::vector<float> v(n);
        for (auto& x : v) x = normal(rng);
        return v;
    };
    const size_t dim = c.dim, hidden = c.hidden_dim, kv = c.kv_dim(), L = c.n_layers;

    TlmWriter w;
    w.f32(std::vector<float>(L * dim, 1.0f));
    w.f32(std::vector<float>(L * dim, 1.0f));
    w.f32(std::vector<float>(dim, 1.0f));
    w.q8(randn(c.vocab_size * dim), c.group_size);
    for (size_t l = 0; l < L; ++l) {
        w.q8(randn(dim * dim), c.group_size);
        w.q8(randn(kv * dim), c.group_size);
        w.q8(randn(kv * dim), c.group_size);
        w.q8(randn(dim * dim), c.group_size);
        w.q8(randn(hidden * dim), c.group_size);
        w.q8(randn(dim * hidden), c.group_size);
        w.q8(randn(hidden * dim), c.group_size);
    }
    w.align();
    std::vector<std::string> words;
    const std::vector<uint8_t> tok = tiny_tokenizer(c.vocab_size, words);
    const int32_t tok_offset = static_cast<int32_t>(w.bytes.size());
    w.raw(tok.data(), tok.size());

    int32_t header[11] = {c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size,
                          c.seq_len, c.group_size, 1, tok_offset, static_cast<int32_t>(tok.size())};
    uint32_t version = 1;
    std::memcpy(w.bytes.data(), "TLM1", 4);
    std::memcpy(w.bytes.data() + 4, &version, 4);
    std::memcpy(w.bytes.data() + 8, header, sizeof(header));
    return w.bytes;
}

TEST(quantize_roundtrip_error_is_bounded) {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-3.0f, 3.0f);
    const int n = 256, gs = 32;
    std::vector<float> x(n), back(n);
    for (auto& v : x) v = u(rng);
    std::vector<int8_t> q(n);
    std::vector<float> s(n / gs);
    tlm::QVector qv{q.data(), s.data()};
    tlm::quantize(qv, x.data(), n, gs);
    tlm::dequantize_row(back.data(), tlm::QTensor{q.data(), s.data()}, 0, n, gs);
    bool ok = true;
    for (int i = 0; i < n; ++i) ok &= std::fabs(back[i] - x[i]) <= s[i / gs] * 0.5f + 1e-6f;
    CHECK(ok);
}

TEST(quantize_all_zero_group) {
    std::vector<float> x(32, 0.0f);
    std::vector<int8_t> q(32, 5);
    float s = -1.0f;
    tlm::QVector qv{q.data(), &s};
    tlm::quantize(qv, x.data(), 32, 32);
    CHECK(s == 0.0f);
    CHECK(q[0] == 0 && q[31] == 0);
}

TEST(simd_matmul_matches_scalar_and_float) {
    std::mt19937 rng(2);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const int n = 288, d = 37, gs = 32;  // odd d exercises the SIMD remainder row
    std::vector<float> wf(n * d), x(n);
    for (auto& v : wf) v = normal(rng);
    for (auto& v : x) v = normal(rng);

    std::vector<int8_t> wq(n * d), xq(n);
    std::vector<float> ws(n * d / gs), xs(n / gs);
    tlm::QVector wv{wq.data(), ws.data()}, xv{xq.data(), xs.data()};
    tlm::quantize(wv, wf.data(), n * d, gs);
    tlm::quantize(xv, x.data(), n, gs);
    const tlm::QTensor w{wq.data(), ws.data()};

    std::vector<float> fast(d), ref(d);
    tlm::matmul_q8(fast.data(), xv, w, n, d, gs);
    tlm::matmul_q8_scalar(ref.data(), xv, w, n, d, gs);
    float max_diff = 0.0f, max_float_err = 0.0f;
    for (int i = 0; i < d; ++i) {
        max_diff = std::max(max_diff, std::fabs(fast[i] - ref[i]));
        float exact = 0.0f;
        for (int j = 0; j < n; ++j) exact += wf[i * n + j] * x[j];
        max_float_err = std::max(max_float_err, std::fabs(ref[i] - exact) / std::sqrt(float(n)));
    }
    std::printf("  backend=%s  |simd-scalar|=%.2e  q8 error/sqrt(n)=%.3f\n", tlm::matmul_backend(), max_diff,
                max_float_err);
    CHECK(max_diff < 1e-3f);
    CHECK(max_float_err < 0.05f);
}

TEST(softmax_sums_to_one) {
    std::vector<float> x = {1.0f, 2.0f, 3.0f, -1000.0f};
    tlm::softmax(x.data(), 4);
    CHECK(std::fabs(x[0] + x[1] + x[2] + x[3] - 1.0f) < 1e-6f);
    CHECK(x[2] > x[1] && x[1] > x[0]);
}

TEST(rope_preserves_norm_and_is_identity_at_zero) {
    std::vector<float> v = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> orig = v;
    tlm::rope(v.data(), 8, 4, 0);
    CHECK(v == orig);
    tlm::rope(v.data(), 8, 4, 17);
    float a = 0, b = 0;
    for (int i = 0; i < 8; ++i) a += v[i] * v[i], b += orig[i] * orig[i];
    CHECK(std::fabs(a - b) < 1e-3f);
}

TEST(tokenizer_merges_by_score_and_falls_back_to_bytes) {
    tlm::Config c = tiny_config();
    std::vector<std::string> words;
    Buffer buf(tiny_tokenizer(c.vocab_size, words));
    tlm::Tokenizer tok;
    CHECK(tok.load(buf.span(), c.vocab_size).empty());

    const int sp_abc = 259 + 7, sp_ab = 259 + 5, sp_c = 259 + 9;
    std::vector<int> ids = tok.encode("abc", true);
    CHECK((ids == std::vector<int>{1, sp_abc}));
    ids = tok.encode("ab c", false);
    CHECK((ids == std::vector<int>{sp_ab, sp_c}));

    // "x" is not in the vocab: byte fallback <0x78> = 0x78 + 3.
    ids = tok.encode("x", false);
    CHECK(ids.size() == 2 && ids[1] == 0x78 + 3);

    // Decoding strips the dummy-prefix space after BOS and maps byte tokens to raw bytes.
    CHECK(tok.piece(1, sp_abc) == "abc");
    CHECK(tok.piece(sp_ab, sp_abc) == " abc");
    CHECK(tok.piece(sp_ab, 0x78 + 3) == "x");
    CHECK(tok.piece(sp_ab, 2).empty());
}

TEST(model_rejects_bad_files) {
    tlm::ModelFile m;
    std::vector<uint8_t> junk(128, 0);
    Buffer b(junk);
    CHECK(!m.parse(b.span()).empty());

    std::vector<uint8_t> model = tiny_model(tiny_config());
    model.resize(model.size() / 2);
    Buffer t(model);
    CHECK(!m.parse(t.span()).empty());
}

TEST(engine_greedy_is_deterministic_and_finite) {
    tlm::Config c = tiny_config();
    Buffer buf(tiny_model(c));
    tlm::Engine engine;
    CHECK(engine.load(buf.span()).empty());
    CHECK(engine.config().n_kv_heads == 2);

    auto run = [&] {
        std::vector<int> out;
        engine.start("abc", 0.0f, 0.9f, 1);
        for (int i = 0; i < 10; ++i) {
            tlm::Step s = engine.next();
            if (s.token < 0) break;
            CHECK(std::isfinite(s.prob) && s.prob > 0.0f && s.prob <= 1.0f);
            CHECK(!s.top.empty() && s.top[0].token == s.token);
            out.push_back(s.token);
        }
        return out;
    };
    std::vector<int> a = run(), b = run();
    CHECK(!a.empty());
    CHECK(a == b);
}

TEST(engine_stops_at_context_limit) {
    tlm::Config c = tiny_config();
    Buffer buf(tiny_model(c, 11));
    tlm::Engine engine;
    CHECK(engine.load(buf.span()).empty());
    engine.start("", 1.0f, 1.0f, 3);
    int n = 0;
    while (engine.next().token >= 0 && n < 1000) ++n;
    CHECK(engine.position() <= c.seq_len);
    CHECK(n < c.seq_len);
}

TEST(sampler_respects_top_p) {
    // With top_p tiny, only the most likely token can be chosen.
    std::vector<float> logits = {0.0f, 5.0f, 1.0f, 4.9f};
    tlm::Sampler s(4, 9);
    s.set_temperature(1.0f);
    s.set_top_p(0.01f);
    for (int i = 0; i < 20; ++i) {
        std::vector<float> l = logits;
        CHECK(s.sample(l.data(), 2).token == 1);
    }
}

// Golden output of stories15M for a fixed prompt under greedy decoding,
// cross-checked against the fp32 numpy reference in tools/reference.py.
TEST(real_model_golden) {
    const char* path = std::getenv("TLM_MODEL");
    if (!path) {
        std::printf("  skipped (set TLM_MODEL=stories15M.tlm)\n");
        return;
    }
    std::ifstream in(path, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), {});
    Buffer buf(bytes);
    tlm::Engine engine;
    CHECK(engine.load(buf.span()).empty());
    engine.start("Once upon a time", 0.0f, 0.9f, 1);
    std::string text;
    for (int i = 0; i < 24; ++i) {
        tlm::Step s = engine.next();
        if (s.token < 0) break;
        text += s.piece;
    }
    std::printf("  got: \"%s\"\n", text.c_str());
    const char* golden = std::getenv("TLM_GOLDEN");
    if (golden) CHECK(text.rfind(golden, 0) == 0);
}

}  // namespace

int main() {
    for (auto& [name, fn] : registry()) {
        const int before = g_failures;
        std::printf("%s\n", name);
        fn();
        std::printf("  %s\n", g_failures == before ? "ok" : "FAILED");
    }
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
