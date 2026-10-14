// Model config and a zero-copy view of the weights inside a loaded .tlm file.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace tlm {

struct Config {
    int dim = 0;         // residual stream width
    int hidden_dim = 0;  // FFN width (padded to a multiple of group_size)
    int n_layers = 0;
    int n_heads = 0;
    int n_kv_heads = 0;  // < n_heads means grouped-query attention
    int vocab_size = 0;
    int seq_len = 0;
    int group_size = 0;  // int8 quantization group
    bool shared_classifier = false;

    int head_size() const { return dim / n_heads; }
    int kv_dim() const { return n_kv_heads * head_size(); }
};

// int8 tensor with one fp32 scale per `group_size` consecutive values.
struct QTensor {
    const int8_t* q = nullptr;
    const float* s = nullptr;
};

struct LayerWeights {
    const float* rms_att = nullptr;
    const float* rms_ffn = nullptr;
    QTensor wq, wk, wv, wo;  // attention projections, [out, in]
    QTensor w1, w2, w3;      // SwiGLU FFN: w2(silu(w1 x) * w3 x)
};

struct Weights {
    QTensor tok_embedding;  // [vocab, dim]
    QTensor wcls;           // [vocab, dim], aliases tok_embedding when shared
    const float* rms_final = nullptr;
    std::vector<LayerWeights> layers;
};

// Parses a .tlm image. The weights point into `file`, which must outlive the
// ModelFile and be at least 4-byte aligned. Returns an empty error on success.
struct ModelFile {
    Config config;
    Weights weights;
    std::span<const uint8_t> tokenizer;

    std::string parse(std::span<const uint8_t> file);
};

}  // namespace tlm
