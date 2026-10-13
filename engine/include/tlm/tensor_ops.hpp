// Numeric kernels. The int8 matmul is the hot path; it has a scalar reference
// and SIMD variants (WASM simd128, x86 AVX2) chosen at compile time.
#pragma once

#include <cstdint>

#include "tlm/model.hpp"

namespace tlm {

// Activation vector quantized the same way as the weights.
struct QVector {
    int8_t* q = nullptr;
    float* s = nullptr;
};

// out[i] = sum_j W[i, j] * x[j] for W of shape [d, n]. n must be a multiple of gs.
void matmul_q8(float* out, const QVector& x, const QTensor& w, int n, int d, int gs);
void matmul_q8_scalar(float* out, const QVector& x, const QTensor& w, int n, int d, int gs);

// Name of the kernel matmul_q8 dispatches to ("wasm-simd128", "avx2", "scalar").
const char* matmul_backend();

void quantize(QVector& out, const float* x, int n, int gs);
void dequantize_row(float* out, const QTensor& w, int row, int n, int gs);

void rmsnorm(float* out, const float* x, const float* weight, int n);
void softmax(float* x, int n);
// Rotary position embedding on interleaved (real, imag) pairs, in place.
void rope(float* vec, int n, int head_size, int pos);

}  // namespace tlm
