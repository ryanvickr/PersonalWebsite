// Turns logits into a token: temperature, nucleus (top-p) sampling, and the
// top-k probabilities the UI shows for every step.
#pragma once

#include <cstdint>
#include <vector>

namespace tlm {

struct Candidate {
    int token;
    float prob;
};

struct SampleResult {
    int token = 0;
    float prob = 0.0f;                // probability of the chosen token
    std::vector<Candidate> top;       // most likely tokens, descending
};

class Sampler {
public:
    explicit Sampler(int vocab_size, uint64_t seed = 42);

    void set_temperature(float t) { temperature_ = t; }
    void set_top_p(float p) { top_p_ = p; }
    void set_seed(uint64_t seed) { rng_ = seed ? seed : 1; }

    // Overwrites logits with probabilities. temperature 0 means greedy.
    SampleResult sample(float* logits, int top_k);

private:
    float next_float();

    int vocab_size_;
    float temperature_ = 0.8f;
    float top_p_ = 0.9f;
    uint64_t rng_;
    std::vector<Candidate> scratch_;
};

}  // namespace tlm
