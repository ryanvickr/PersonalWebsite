// Native CLI: tlm <model.tlm> [-p prompt] [-n tokens] [-t temperature] [-P top_p] [-s seed]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

#include "tlm/engine.hpp"
#include "tlm/tensor_ops.hpp"

namespace {

struct AlignedFree {
    void operator()(uint8_t* p) const { std::free(p); }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s model.tlm [-p prompt] [-n tokens] [-t temp] [-P top_p] [-s seed]\n", argv[0]);
        return 1;
    }
    std::string prompt = "Once upon a time";
    int steps = 200;
    float temperature = 0.0f, top_p = 0.9f;
    unsigned seed = 42;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "-p")) prompt = argv[i + 1];
        else if (!std::strcmp(argv[i], "-n")) steps = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "-t")) temperature = std::atof(argv[i + 1]);
        else if (!std::strcmp(argv[i], "-P")) top_p = std::atof(argv[i + 1]);
        else if (!std::strcmp(argv[i], "-s")) seed = std::atoi(argv[i + 1]);
    }

    std::ifstream in(argv[1], std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    const size_t size = in.tellg();
    std::unique_ptr<uint8_t, AlignedFree> buf(static_cast<uint8_t*>(std::aligned_alloc(64, (size + 63) / 64 * 64)));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(buf.get()), size);

    tlm::Engine engine;
    if (std::string err = engine.load({buf.get(), size}); !err.empty()) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }

    const auto t0 = std::chrono::steady_clock::now();
    engine.start(prompt, temperature, top_p, seed);
    std::printf("%s", prompt.c_str());
    int generated = 0;
    for (; generated < steps; ++generated) {
        tlm::Step s = engine.next();
        if (s.token < 0) break;
        std::fwrite(s.piece.data(), 1, s.piece.size(), stdout);
        std::fflush(stdout);
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "\n[%s] %d tokens, %.1f tok/s\n", tlm::matmul_backend(), generated, generated / secs);
    return 0;
}
