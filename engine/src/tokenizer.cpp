#include "tlm/tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace tlm {

std::string Tokenizer::load(std::span<const uint8_t> bytes, int vocab_size) {
    size_t pos = 0;
    auto read = [&](void* dst, size_t n) {
        if (pos + n > bytes.size()) return false;
        std::memcpy(dst, bytes.data() + pos, n);
        pos += n;
        return true;
    };

    int32_t max_token_length;
    if (!read(&max_token_length, 4)) return "tokenizer truncated";
    vocab_.resize(vocab_size);
    scores_.resize(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        int32_t len;
        if (!read(&scores_[i], 4) || !read(&len, 4) || len < 0 || len > max_token_length) return "tokenizer truncated";
        vocab_[i].resize(len);
        if (!read(vocab_[i].data(), len)) return "tokenizer truncated";
    }
    index_.clear();
    index_.reserve(vocab_size);
    // Later duplicates never win: keep the first id for each string.
    for (int i = 0; i < vocab_size; ++i) index_.emplace(vocab_[i], i);
    for (int b = 0; b < 256; ++b) byte_pieces_[b] = std::string(1, static_cast<char>(b));
    return {};
}

int Tokenizer::lookup(std::string_view s) const {
    auto it = index_.find(s);
    return it == index_.end() ? -1 : it->second;
}

std::vector<int> Tokenizer::encode(std::string_view text, bool bos) const {
    std::vector<int> tokens;
    if (bos) tokens.push_back(kBos);
    if (text.empty()) return tokens;

    // SentencePiece prepends a space ("dummy prefix") to the input.
    if (int sp = lookup(" "); sp >= 0) tokens.push_back(sp);

    // Split into UTF-8 codepoints; unknown ones fall back to <0xXX> byte tokens (ids 3..258).
    for (size_t i = 0; i < text.size();) {
        size_t len = 1;
        const auto c = static_cast<unsigned char>(text[i]);
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        len = std::min(len, text.size() - i);
        if (int id = lookup(text.substr(i, len)); id >= 0) {
            tokens.push_back(id);
        } else {
            for (size_t k = 0; k < len; ++k) tokens.push_back(static_cast<unsigned char>(text[i + k]) + 3);
        }
        i += len;
    }

    // Repeatedly merge the adjacent pair whose merged token scores highest.
    std::string merged;
    const size_t first = bos ? 1 : 0;
    while (true) {
        float best_score = -1e10f;
        int best_id = -1;
        size_t best_idx = 0;
        for (size_t i = first; i + 1 < tokens.size(); ++i) {
            merged = vocab_[tokens[i]];
            merged += vocab_[tokens[i + 1]];
            const int id = lookup(merged);
            if (id >= 0 && scores_[id] > best_score) {
                best_score = scores_[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_id < 0) break;
        tokens[best_idx] = best_id;
        tokens.erase(tokens.begin() + best_idx + 1);
    }
    return tokens;
}

std::string_view Tokenizer::piece(int prev, int token) const {
    if (token < 0 || token >= vocab_size() || token == kBos || token == kEos) return {};
    std::string_view p = vocab_[token];
    // After BOS, SentencePiece strips the leading space of the first piece.
    if (prev == kBos && !p.empty() && p[0] == ' ') p.remove_prefix(1);
    unsigned int byte;
    if (p.size() == 6 && std::sscanf(std::string(p).c_str(), "<0x%02X>", &byte) == 1) return byte_pieces_[byte];
    return p;
}

}  // namespace tlm
