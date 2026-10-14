// Benchmarks the int8 matmul kernel (SIMD vs scalar) on the classifier shape
// of stories15M, then end-to-end tok/s if a model path is given.
//   tlm-bench [model.tlm]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <vector>

#include "tlm/engine.hpp"
#include "tlm/tensor_ops.hpp"

namespace {

template <typename F>
double time_ms(F&& f, int reps) {
    f();  // warm up
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) f();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}

}  // namespace

int main(int argc, char** argv) {
    const int n = 288, d = 32000, gs = 32;
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> u(-127, 127);
    std::vector<int8_t> wq(size_t(n) * d), xq(n);
    std::vector<float> ws(size_t(n) * d / gs, 0.01f), xs(n / gs, 0.01f), out(d);
    for (auto& v : wq) v = static_cast<int8_t>(u(rng));
    for (auto& v : xq) v = static_cast<int8_t>(u(rng));
    const tlm::QTensor w{wq.data(), ws.data()};
    const tlm::QVector x{xq.data(), xs.data()};

    const double scalar = time_ms([&] { tlm::matmul_q8_scalar(out.data(), x, w, n, d, gs); }, 20);
    const double fast = time_ms([&] { tlm::matmul_q8(out.data(), x, w, n, d, gs); }, 20);
    const double gmacs = double(n) * d / 1e6;
    std::printf("matmul_q8 %dx%d  scalar %.3f ms (%.2f GMAC/s)   %s %.3f ms (%.2f GMAC/s)   speedup %.1fx\n", d, n,
                scalar, gmacs / scalar, tlm::matmul_backend(), fast, gmacs / fast, scalar / fast);

    if (argc < 2) return 0;
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), {});
    tlm::Engine engine;
    if (std::string err = engine.load(bytes); !err.empty()) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    engine.start("Once upon a time", 0.0f, 0.9f, 1);
    const auto t0 = std::chrono::steady_clock::now();
    int tokens = 0;
    while (engine.next().token >= 0) ++tokens;
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("generate  %d tokens  %.1f tok/s  %.2f ms/token\n", tokens, tokens / secs, 1000 * secs / tokens);
    return 0;
}
