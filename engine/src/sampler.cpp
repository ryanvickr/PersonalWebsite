#include "tlm/sampler.hpp"

#include <algorithm>

#include "tlm/tensor_ops.hpp"

namespace tlm {

Sampler::Sampler(int vocab_size, uint64_t seed) : vocab_size_(vocab_size), rng_(seed ? seed : 1) {
    scratch_.reserve(vocab_size);
}

float Sampler::next_float() {
    // xorshift64*: tiny, fast, and deterministic across native and WASM builds.
    rng_ ^= rng_ >> 12;
    rng_ ^= rng_ << 25;
    rng_ ^= rng_ >> 27;
    return float((rng_ * 0x2545F4914F6CDD1Dull) >> 40) / float(1u << 24);
}

SampleResult Sampler::sample(float* logits, int top_k) {
    const bool greedy = temperature_ <= 0.0f;
    if (!greedy) {
        const float inv_t = 1.0f / temperature_;
        for (int i = 0; i < vocab_size_; ++i) logits[i] *= inv_t;
    }
    softmax(logits, vocab_size_);

    // Only tokens above a small floor can matter for top-k or top-p; filtering
    // first avoids sorting all 32k entries every step.
    const float floor = greedy ? 0.0f : (1.0f - top_p_) / float(vocab_size_ - 1);
    scratch_.clear();
    for (int i = 0; i < vocab_size_; ++i)
        if (logits[i] >= floor) scratch_.push_back({i, logits[i]});
    const auto by_prob = [](const Candidate& a, const Candidate& b) { return a.prob > b.prob; };

    SampleResult r;
    const size_t k = std::min<size_t>(top_k, scratch_.size());
    if (greedy) {
        std::partial_sort(scratch_.begin(), scratch_.begin() + k, scratch_.end(), by_prob);
        r.top.assign(scratch_.begin(), scratch_.begin() + k);
        r.token = scratch_[0].token;
        r.prob = scratch_[0].prob;
        return r;
    }

    std::sort(scratch_.begin(), scratch_.end(), by_prob);
    r.top.assign(scratch_.begin(), scratch_.begin() + k);

    // Smallest prefix whose mass reaches top_p, then sample within it.
    float cum = 0.0f;
    size_t last = scratch_.size() - 1;
    for (size_t i = 0; i < scratch_.size(); ++i) {
        cum += scratch_[i].prob;
        if (cum > top_p_) {
            last = i;
            break;
        }
    }
    const float target = next_float() * cum;
    float acc = 0.0f;
    for (size_t i = 0; i <= last; ++i) {
        acc += scratch_[i].prob;
        if (target < acc || i == last) {
            r.token = scratch_[i].token;
            r.prob = scratch_[i].prob;
            break;
        }
    }
    return r;
}

}  // namespace tlm
