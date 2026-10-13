#include "tlm/transformer.hpp"

#include <algorithm>
#include <cmath>

namespace tlm {

Transformer::Transformer(const ModelFile& model) : cfg_(model.config), w_(model.weights) {
    const size_t dim = cfg_.dim, hidden = cfg_.hidden_dim, kv = cfg_.kv_dim();
    const size_t widest = std::max(dim, hidden);
    x_.resize(dim);
    xb_.resize(dim);
    xb2_.resize(dim);
    hb_.resize(hidden);
    hb2_.resize(hidden);
    q_.resize(dim);
    att_.resize(static_cast<size_t>(cfg_.n_heads) * cfg_.seq_len);
    logits_.resize(cfg_.vocab_size);
    xq_q_.resize(widest);
    xq_s_.resize(widest / cfg_.group_size);
    key_cache_.resize(static_cast<size_t>(cfg_.n_layers) * cfg_.seq_len * kv);
    value_cache_.resize(key_cache_.size());
}

void Transformer::matmul(float* out, const float* x, const QTensor& w, int n, int d) {
    QVector xq{xq_q_.data(), xq_s_.data()};
    quantize(xq, x, n, cfg_.group_size);
    matmul_q8(out, xq, w, n, d, cfg_.group_size);
}

float* Transformer::forward(int token, int pos) {
    const int dim = cfg_.dim, hidden = cfg_.hidden_dim, kv_dim = cfg_.kv_dim();
    const int head_size = cfg_.head_size(), kv_mul = cfg_.n_heads / cfg_.n_kv_heads;
    const float att_scale = 1.0f / std::sqrt(float(head_size));

    dequantize_row(x_.data(), w_.tok_embedding, token, dim, cfg_.group_size);

    for (int l = 0; l < cfg_.n_layers; ++l) {
        const LayerWeights& lw = w_.layers[l];
        const size_t layer_off = static_cast<size_t>(l) * cfg_.seq_len * kv_dim;
        float* k = key_cache_.data() + layer_off + static_cast<size_t>(pos) * kv_dim;
        float* v = value_cache_.data() + layer_off + static_cast<size_t>(pos) * kv_dim;

        // Attention: q, k, v projections write k and v straight into the cache.
        rmsnorm(xb_.data(), x_.data(), lw.rms_att, dim);
        QVector xq{xq_q_.data(), xq_s_.data()};
        quantize(xq, xb_.data(), dim, cfg_.group_size);
        matmul_q8(q_.data(), xq, lw.wq, dim, dim, cfg_.group_size);
        matmul_q8(k, xq, lw.wk, dim, kv_dim, cfg_.group_size);
        matmul_q8(v, xq, lw.wv, dim, kv_dim, cfg_.group_size);
        rope(q_.data(), dim, head_size, pos);
        rope(k, kv_dim, head_size, pos);

        for (int h = 0; h < cfg_.n_heads; ++h) {
            const float* qh = q_.data() + h * head_size;
            float* att = att_.data() + static_cast<size_t>(h) * cfg_.seq_len;
            const int kv_off = (h / kv_mul) * head_size;
            for (int t = 0; t <= pos; ++t) {
                const float* kt = key_cache_.data() + layer_off + static_cast<size_t>(t) * kv_dim + kv_off;
                float score = 0.0f;
                for (int i = 0; i < head_size; ++i) score += qh[i] * kt[i];
                att[t] = score * att_scale;
            }
            softmax(att, pos + 1);
            float* out = xb_.data() + h * head_size;
            std::fill(out, out + head_size, 0.0f);
            for (int t = 0; t <= pos; ++t) {
                const float* vt = value_cache_.data() + layer_off + static_cast<size_t>(t) * kv_dim + kv_off;
                const float a = att[t];
                for (int i = 0; i < head_size; ++i) out[i] += a * vt[i];
            }
        }
        matmul(xb2_.data(), xb_.data(), lw.wo, dim, dim);
        for (int i = 0; i < dim; ++i) x_[i] += xb2_[i];

        // SwiGLU feed-forward.
        rmsnorm(xb_.data(), x_.data(), lw.rms_ffn, dim);
        quantize(xq, xb_.data(), dim, cfg_.group_size);
        matmul_q8(hb_.data(), xq, lw.w1, dim, hidden, cfg_.group_size);
        matmul_q8(hb2_.data(), xq, lw.w3, dim, hidden, cfg_.group_size);
        for (int i = 0; i < hidden; ++i) {
            const float g = hb_[i];
            hb_[i] = g / (1.0f + std::exp(-g)) * hb2_[i];
        }
        matmul(xb_.data(), hb_.data(), lw.w2, hidden, dim);
        for (int i = 0; i < dim; ++i) x_[i] += xb_[i];
    }

    rmsnorm(x_.data(), x_.data(), w_.rms_final, dim);
    matmul(logits_.data(), x_.data(), w_.wcls, dim, cfg_.vocab_size);
    return logits_.data();
}

}  // namespace tlm
