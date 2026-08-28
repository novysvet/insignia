#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace insignia {

// e4m3 -> fp32, exact for every finite code incl. subnormals:
// place the 7 magnitude bits at fp16 bits 7..13 (mantissa lsb + exponent), keep the
// sign at bit 15, then multiply by 2^8 — normal and subnormal paths share the factor.
// Two lanes per u32 in three logic ops + one F2F.H2.F32 pair convert.
__device__ __forceinline__ float2 e4m3x2(uint32_t u) {
    const uint32_t h2 = ((u & 0x007f007fu) << 7) | ((u & 0x00800080u) << 8);
    const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&h2));
    return make_float2(f.x * 256.f, f.y * 256.f);
}
__device__ __forceinline__ float e4m1(uint32_t b) {
    const unsigned short hb = (unsigned short)(((b & 0x7f) << 7) | ((b & 0x80) << 8));
    return __half2float(*reinterpret_cast<const __half *>(&hb)) * 256.f;
}

// FP8 e4m3 weights + BF16 block scales [rows/128][cols/128] (Qwen official FP8 layout).
void fp8_gemv(const uint8_t *weights, const uint16_t *scales, const float *x, float *y, int rows, int cols, cudaStream_t stream = nullptr);
void fp8_gemv2(const uint8_t *weights, const uint16_t *scales, const float *x /*[2,cols]*/, float *y /*[2,rows]*/, int rows, int cols, cudaStream_t stream = nullptr);
void fp8_gemm(const uint8_t *weights, const uint16_t *scales, const void *x16 /*bf16 [64,cols]*/, float *y /*[64,rows] padded*/, int rows, int cols, int T, cudaStream_t stream = nullptr);
void bf16_get_row(const uint16_t *w, float *out, const int *row_dev, int cols, cudaStream_t stream = nullptr);

}
