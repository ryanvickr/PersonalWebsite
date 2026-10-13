// Ties model, tokenizer and sampler into a streaming text generator.
// This is the surface both the native CLI and the WASM exports use.
#pragma once

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tlm/model.hpp"
#include "tlm/sampler.hpp"
#include "tlm/tokenizer.hpp"
#include "tlm/transformer.hpp"

namespace tlm {

struct Step {
    int token = -1;              // -1 when generation has finished
    float prob = 0.0f;
    std::string_view piece;      // raw UTF-8 bytes (may split a codepoint)
    std::vector<Candidate> top;  // top-k alternatives at this step
};

class Engine {
public:
    // `file` must stay alive and unmodified for the Engine's lifetime.
    std::string load(std::span<const uint8_t> file);

    // Resets the KV cache and feeds the prompt. Returns the prompt length in tokens.
    int start(std::string_view prompt, float temperature, float top_p, uint64_t seed);

    // Produces the next token. Returns token = -1 at EOS or the context limit.
    Step next(int top_k = 5);

    const Config& config() const { return model_.config; }
    const Tokenizer& tokenizer() const { return tokenizer_; }
    int position() const { return pos_; }

private:
    ModelFile model_;
    Tokenizer tokenizer_;
    std::unique_ptr<Transformer> transformer_;
    std::unique_ptr<Sampler> sampler_;
    float* logits_ = nullptr;  // logits for predicting position pos_
    int pos_ = 0;
    int prev_ = Tokenizer::kBos;
};

}  // namespace tlm
