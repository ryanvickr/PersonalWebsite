// Llama-2 forward pass with a KV cache, one token at a time.
#pragma once

#include <vector>

#include "tlm/model.hpp"
#include "tlm/tensor_ops.hpp"

namespace tlm {

class Transformer {
public:
    explicit Transformer(const ModelFile& model);

    // Runs `token` at position `pos` and returns logits[vocab_size].
    // Positions must be fed in order starting from 0.
    float* forward(int token, int pos);

    const Config& config() const { return cfg_; }

private:
    void matmul(float* out, const float* x, const QTensor& w, int n, int d);

    const Config& cfg_;
    const Weights& w_;

    // Activations, reused every step.
    std::vector<float> x_, xb_, xb2_, hb_, hb2_, q_, att_, logits_;
    std::vector<int8_t> xq_q_;
    std::vector<float> xq_s_;
    // KV cache: [layer][pos][kv_dim]
    std::vector<float> key_cache_, value_cache_;
};

}  // namespace tlm
