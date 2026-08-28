#include "insignia_glm53_q8.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace insignia::glm53 {
namespace {

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, offset);
    return value;
}

__global__ __launch_bounds__(256, 1) void quantize_q8_x64_kernel(
    const float *__restrict__ x,
    uint32_t *__restrict__ quantized,
    float *__restrict__ scales,
    int groups) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int group = thread >> 2;
    if (group >= groups) return;
    const int quarter = thread & 3;
    const float4 *source = reinterpret_cast<const float4 *>(
        x + group * kQ8GroupSize + quarter * 16);
    float values[16];
    float maximum = 0.0f;
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        const float4 value = __ldg(source + word);
        values[word * 4 + 0] = value.x;
        values[word * 4 + 1] = value.y;
        values[word * 4 + 2] = value.z;
        values[word * 4 + 3] = value.w;
        maximum = fmaxf(maximum, fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                                       fmaxf(fabsf(value.z), fabsf(value.w))));
    }
    const unsigned active = __activemask();
    maximum = fmaxf(maximum, __shfl_xor_sync(active, maximum, 1));
    maximum = fmaxf(maximum, __shfl_xor_sync(active, maximum, 2));
    const float inverse = maximum > 0.0f ? 127.0f / maximum : 0.0f;
    if (!quarter) scales[group] = maximum * (1.0f / 127.0f);
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        uint32_t packed = 0;
#pragma unroll
        for (int byte = 0; byte < 4; ++byte)
            packed |= uint32_t(uint8_t(__float2int_rn(values[word * 4 + byte] * inverse))) <<
                      (byte * 8);
        quantized[group * 16 + quarter * 4 + word] = packed;
    }
}

__global__ __launch_bounds__(256) void q8_dp4a_kernel(
    const uint32_t *__restrict__ weights,
    const __half *__restrict__ weight_scales,
    const uint32_t *__restrict__ x,
    const float *__restrict__ x_scales,
    float *__restrict__ y,
    int rows,
    int groups) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_weights = weights + size_t(row) * groups * 16;
    const __half *row_scales = weight_scales + size_t(row) * groups;
    float sum = 0.0f;
#pragma unroll 2
    for (int group = lane; group < groups; group += 32) {
        const uint4 *packed = reinterpret_cast<const uint4 *>(row_weights + group * 16);
        const uint32_t *activation = x + group * 16;
        int dot = 0;
#pragma unroll
        for (int vector = 0; vector < 4; ++vector) {
            const uint4 weight = __ldcs(packed + vector);
            dot = __dp4a(int(weight.x), int(activation[vector * 4 + 0]), dot);
            dot = __dp4a(int(weight.y), int(activation[vector * 4 + 1]), dot);
            dot = __dp4a(int(weight.z), int(activation[vector * 4 + 2]), dot);
            dot = __dp4a(int(weight.w), int(activation[vector * 4 + 3]), dot);
        }
        sum = fmaf(float(dot), __half2float(row_scales[group]) * x_scales[group], sum);
    }
    sum = warp_sum(sum);
    if (!lane) y[row] = sum;
}

}  // namespace

size_t q8_workspace_bytes(int cols) {
    if (cols <= 0 || (cols & (kQ8GroupSize - 1))) return 0;
    const size_t quantized = size_t(cols);
    const size_t aligned = (quantized + 255) & ~size_t(255);
    return aligned + size_t(cols / kQ8GroupSize) * sizeof(float);
}

cudaError_t q8_gemv(
    const uint32_t *weights,
    const uint16_t *scales,
    const float *x,
    float *y,
    int rows,
    int cols,
    void *workspace,
    cudaStream_t stream) {
    if (!weights || !scales || !x || !y || !workspace || rows <= 0 || cols <= 0 ||
        (cols & (kQ8GroupSize - 1)) || cols / kQ8GroupSize > 256)
        return cudaErrorInvalidValue;
    auto *bytes = static_cast<uint8_t *>(workspace);
    auto *quantized = reinterpret_cast<uint32_t *>(bytes);
    const size_t aligned = (size_t(cols) + 255) & ~size_t(255);
    auto *activation_scales = reinterpret_cast<float *>(bytes + aligned);
    const int groups = cols / kQ8GroupSize;
    quantize_q8_x64_kernel<<<(groups + 63) / 64, 256, 0, stream>>>(
        x, quantized, activation_scales, groups);
    q8_dp4a_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(
        weights, reinterpret_cast<const __half *>(scales), quantized,
        activation_scales, y, rows, groups);
    return cudaGetLastError();
}

}  // namespace insignia::glm53
