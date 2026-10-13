// Llama-2 SentencePiece BPE tokenizer over the llama2.c tokenizer.bin format.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tlm {

class Tokenizer {
public:
    static constexpr int kBos = 1;
    static constexpr int kEos = 2;

    // Parses tokenizer.bin bytes. Returns an empty error on success.
    std::string load(std::span<const uint8_t> bytes, int vocab_size);

    // Greedy highest-score-merge BPE, matching SentencePiece for Llama-2.
    std::vector<int> encode(std::string_view text, bool bos) const;

    // Raw UTF-8 bytes this token contributes after `prev`. May be a partial
    // multi-byte sequence (byte-fallback tokens like <0xE2>), so callers
    // stream-decode rather than treat each piece as a full string.
    std::string_view piece(int prev, int token) const;

    int vocab_size() const { return static_cast<int>(vocab_.size()); }

private:
    int lookup(std::string_view s) const;

    std::vector<std::string> vocab_;
    std::vector<float> scores_;
    std::unordered_map<std::string_view, int> index_;
    std::string byte_pieces_[256];
};

}  // namespace tlm
