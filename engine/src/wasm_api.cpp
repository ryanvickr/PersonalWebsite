// Plain C exports for the browser. JS copies the .tlm file into WASM memory,
// then calls tlm_step() in a loop and reads results out of a fixed struct,
// so no strings or objects cross the JS/WASM boundary per token.
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "tlm/engine.hpp"
#include "tlm/tensor_ops.hpp"

namespace {

constexpr int kTopK = 5;

// Layout mirrored in site/src/scripts/llm-worker.ts.
struct StepOut {
    int32_t token;
    float prob;
    int32_t piece_ptr;
    int32_t piece_len;
    int32_t n_top;
    int32_t top_token[kTopK];
    float top_prob[kTopK];
};

tlm::Engine g_engine;
StepOut g_step;
std::string g_error;

}  // namespace

extern "C" {

void* tlm_alloc(size_t n) { return std::aligned_alloc(64, (n + 63) / 64 * 64); }
void tlm_free(void* p) { std::free(p); }

// Returns 0 on success, otherwise a pointer to a NUL-terminated error message.
const char* tlm_load(const uint8_t* data, size_t n) {
    g_error = g_engine.load({data, n});
    return g_error.empty() ? nullptr : g_error.c_str();
}

// Returns the number of prompt tokens.
int tlm_start(const char* prompt, size_t len, float temperature, float top_p, uint32_t seed) {
    return g_engine.start({prompt, len}, temperature, top_p, seed);
}

// Returns a pointer to the StepOut for the next token; token is -1 when finished.
const StepOut* tlm_step() {
    tlm::Step s = g_engine.next(kTopK);
    g_step.token = s.token;
    g_step.prob = s.prob;
    g_step.piece_ptr = static_cast<int32_t>(reinterpret_cast<uintptr_t>(s.piece.data()));
    g_step.piece_len = static_cast<int32_t>(s.piece.size());
    g_step.n_top = static_cast<int32_t>(s.top.size());
    for (size_t i = 0; i < s.top.size(); ++i) {
        g_step.top_token[i] = s.top[i].token;
        g_step.top_prob[i] = s.top[i].prob;
    }
    return &g_step;
}

// Text of a token as it would appear mid-sentence (for the alternatives list).
// Writes the length to *len and returns a pointer to the bytes.
const char* tlm_piece(int token, int32_t* len) {
    std::string_view p = g_engine.tokenizer().piece(/*prev=*/-1, token);
    *len = static_cast<int32_t>(p.size());
    return p.data();
}

int tlm_position() { return g_engine.position(); }
int tlm_seq_len() { return g_engine.config().seq_len; }
int tlm_n_params() {
    const tlm::Config& c = g_engine.config();
    const long long per_layer = 2LL * c.dim * c.dim + 2LL * c.dim * c.kv_dim() + 3LL * c.dim * c.hidden_dim;
    long long n = per_layer * c.n_layers + 1LL * c.vocab_size * c.dim;
    if (!c.shared_classifier) n += 1LL * c.vocab_size * c.dim;
    return static_cast<int>(n);
}
const char* tlm_backend() { return tlm::matmul_backend(); }

}  // extern "C"
