#include "tlm/tensor_ops.hpp"

#include <algorithm>
#include <cmath>

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace tlm {

void quantize(QVector& out, const float* x, int n, int gs) {
    for (int g = 0; g < n / gs; ++g) {
        const float* xg = x + g * gs;
        float amax = 0.0f;
        for (int i = 0; i < gs; ++i) amax = std::max(amax, std::fabs(xg[i]));
        const float scale = amax / 127.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        out.s[g] = scale;
        for (int i = 0; i < gs; ++i) out.q[g * gs + i] = static_cast<int8_t>(std::lround(xg[i] * inv));
    }
}

void dequantize_row(float* out, const QTensor& w, int row, int n, int gs) {
    const size_t base = static_cast<size_t>(row) * n;
    for (int i = 0; i < n; ++i) out[i] = w.q[base + i] * w.s[(base + i) / gs];
}

void matmul_q8_scalar(float* out, const QVector& x, const QTensor& w, int n, int d, int gs) {
    for (int i = 0; i < d; ++i) {
        const int8_t* wr = w.q + static_cast<size_t>(i) * n;
        const float* ws = w.s + static_cast<size_t>(i) * n / gs;
        float acc = 0.0f;
        for (int g = 0; g < n / gs; ++g) {
            int32_t dot = 0;
            for (int k = 0; k < gs; ++k) dot += int32_t(x.q[g * gs + k]) * int32_t(wr[g * gs + k]);
            acc += float(dot) * ws[g] * x.s[g];
        }
        out[i] = acc;
    }
}

#if defined(__wasm_simd128__)

// Integer dot product of 16 int8 pairs, accumulated into four int32 lanes.
// WASM has no int8 dot instruction, so widen to i16 and use i32x4.dot_i16x8.
static inline v128_t dot16(v128_t acc, const int8_t* a, const int8_t* b) {
    const v128_t va = wasm_v128_load(a);
    const v128_t vb = wasm_v128_load(b);
    acc = wasm_i32x4_add(acc, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(va), wasm_i16x8_extend_low_i8x16(vb)));
    acc = wasm_i32x4_add(acc, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(va), wasm_i16x8_extend_high_i8x16(vb)));
    return acc;
}

static inline int32_t hsum(v128_t v) {
    return wasm_i32x4_extract_lane(v, 0) + wasm_i32x4_extract_lane(v, 1) + wasm_i32x4_extract_lane(v, 2) +
           wasm_i32x4_extract_lane(v, 3);
}

void matmul_q8(float* out, const QVector& x, const QTensor& w, int n, int d, int gs) {
    if (gs % 16 != 0) return matmul_q8_scalar(out, x, w, n, d, gs);
    const int groups = n / gs;
    int i = 0;
    // Two output rows per iteration: each activation load feeds two weight rows.
    for (; i + 1 < d; i += 2) {
        const int8_t* w0 = w.q + static_cast<size_t>(i) * n;
        const int8_t* w1 = w0 + n;
        const float* s0 = w.s + static_cast<size_t>(i) * groups;
        const float* s1 = s0 + groups;
        v128_t facc0 = wasm_f32x4_splat(0.0f), facc1 = wasm_f32x4_splat(0.0f);
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int g = 0; g < groups; ++g) {
            v128_t a0 = wasm_i32x4_splat(0), a1 = wasm_i32x4_splat(0);
            for (int k = g * gs; k < (g + 1) * gs; k += 16) {
                const v128_t vx = wasm_v128_load(x.q + k);
                const v128_t xl = wasm_i16x8_extend_low_i8x16(vx), xh = wasm_i16x8_extend_high_i8x16(vx);
                const v128_t v0 = wasm_v128_load(w0 + k), v1 = wasm_v128_load(w1 + k);
                a0 = wasm_i32x4_add(a0, wasm_i32x4_dot_i16x8(xl, wasm_i16x8_extend_low_i8x16(v0)));
                a0 = wasm_i32x4_add(a0, wasm_i32x4_dot_i16x8(xh, wasm_i16x8_extend_high_i8x16(v0)));
                a1 = wasm_i32x4_add(a1, wasm_i32x4_dot_i16x8(xl, wasm_i16x8_extend_low_i8x16(v1)));
                a1 = wasm_i32x4_add(a1, wasm_i32x4_dot_i16x8(xh, wasm_i16x8_extend_high_i8x16(v1)));
            }
            // Scale in float lanes and defer the horizontal sum to the end of the row.
            facc0 = wasm_f32x4_add(facc0, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(a0), wasm_f32x4_splat(s0[g] * x.s[g])));
            facc1 = wasm_f32x4_add(facc1, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(a1), wasm_f32x4_splat(s1[g] * x.s[g])));
        }
        acc0 = wasm_f32x4_extract_lane(facc0, 0) + wasm_f32x4_extract_lane(facc0, 1) +
               wasm_f32x4_extract_lane(facc0, 2) + wasm_f32x4_extract_lane(facc0, 3);
        acc1 = wasm_f32x4_extract_lane(facc1, 0) + wasm_f32x4_extract_lane(facc1, 1) +
               wasm_f32x4_extract_lane(facc1, 2) + wasm_f32x4_extract_lane(facc1, 3);
        out[i] = acc0;
        out[i + 1] = acc1;
    }
    for (; i < d; ++i) {
        const int8_t* wr = w.q + static_cast<size_t>(i) * n;
        const float* ws = w.s + static_cast<size_t>(i) * groups;
        float acc = 0.0f;
        for (int g = 0; g < groups; ++g) {
            v128_t a = wasm_i32x4_splat(0);
            for (int k = g * gs; k < (g + 1) * gs; k += 16) a = dot16(a, x.q + k, wr + k);
            acc += float(hsum(a)) * ws[g] * x.s[g];
        }
        out[i] = acc;
    }
}

const char* matmul_backend() { return "wasm-simd128"; }

#elif defined(__AVX2__)

void matmul_q8(float* out, const QVector& x, const QTensor& w, int n, int d, int gs) {
    if (gs % 32 != 0) return matmul_q8_scalar(out, x, w, n, d, gs);
    const int groups = n / gs;
    const __m256i ones = _mm256_set1_epi16(1);
    for (int i = 0; i < d; ++i) {
        const int8_t* wr = w.q + static_cast<size_t>(i) * n;
        const float* ws = w.s + static_cast<size_t>(i) * groups;
        __m256 facc = _mm256_setzero_ps();
        for (int g = 0; g < groups; ++g) {
            __m256i acc = _mm256_setzero_si256();
            for (int k = g * gs; k < (g + 1) * gs; k += 32) {
                const __m256i vx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.q + k));
                const __m256i vw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wr + k));
                // maddubs needs one unsigned operand: move x's sign onto w.
                const __m256i ax = _mm256_sign_epi8(vx, vx);
                const __m256i sw = _mm256_sign_epi8(vw, vx);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sw), ones));
            }
            facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_set1_ps(ws[g] * x.s[g]), facc);
        }
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(facc), _mm256_extractf128_ps(facc, 1));
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        out[i] = _mm_cvtss_f32(s);
    }
}

const char* matmul_backend() { return "avx2"; }

#else

void matmul_q8(float* out, const QVector& x, const QTensor& w, int n, int d, int gs) {
    matmul_q8_scalar(out, x, w, n, d, gs);
}

const char* matmul_backend() { return "scalar"; }

#endif

void rmsnorm(float* out, const float* x, const float* weight, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    const float inv = 1.0f / std::sqrt(ss / n + 1e-5f);
    for (int i = 0; i < n; ++i) out[i] = weight[i] * (x[i] * inv);
}

void softmax(float* x, int n) {
    const float max = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max);
        sum += x[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

void rope(float* vec, int n, int head_size, int pos) {
    for (int i = 0; i < n; i += 2) {
        const int hd = i % head_size;
        const float freq = 1.0f / std::pow(10000.0f, float(hd) / float(head_size));
        const float angle = pos * freq;
        const float c = std::cos(angle), s = std::sin(angle);
        const float v0 = vec[i], v1 = vec[i + 1];
        vec[i] = v0 * c - v1 * s;
        vec[i + 1] = v0 * s + v1 * c;
    }
}

}  // namespace tlm
