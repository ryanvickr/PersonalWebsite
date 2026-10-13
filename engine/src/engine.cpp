#include "tlm/engine.hpp"

namespace tlm {

std::string Engine::load(std::span<const uint8_t> file) {
    transformer_.reset();
    sampler_.reset();
    if (std::string err = model_.parse(file); !err.empty()) return err;
    if (std::string err = tokenizer_.load(model_.tokenizer, model_.config.vocab_size); !err.empty()) return err;
    transformer_ = std::make_unique<Transformer>(model_);
    sampler_ = std::make_unique<Sampler>(model_.config.vocab_size);
    return {};
}

int Engine::start(std::string_view prompt, float temperature, float top_p, uint64_t seed) {
    sampler_->set_temperature(temperature);
    sampler_->set_top_p(top_p);
    sampler_->set_seed(seed);

    std::vector<int> tokens = tokenizer_.encode(prompt, /*bos=*/true);
    const int limit = config().seq_len - 1;
    if (static_cast<int>(tokens.size()) > limit) tokens.resize(limit);
    pos_ = 0;
    for (int t : tokens) logits_ = transformer_->forward(t, pos_++);
    prev_ = tokens.back();
    return static_cast<int>(tokens.size()) - 1;  // excluding BOS
}

Step Engine::next(int top_k) {
    Step step;
    if (!logits_ || pos_ >= config().seq_len) return step;

    SampleResult r = sampler_->sample(logits_, top_k);
    logits_ = nullptr;
    step.prob = r.prob;
    step.top = std::move(r.top);
    if (r.token == Tokenizer::kEos || r.token == Tokenizer::kBos) return step;

    step.token = r.token;
    step.piece = tokenizer_.piece(prev_, r.token);
    prev_ = r.token;
    if (pos_ < config().seq_len) logits_ = transformer_->forward(r.token, pos_++);
    return step;
}

}  // namespace tlm
