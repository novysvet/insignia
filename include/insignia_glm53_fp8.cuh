#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace insignia::glm53 {

constexpr int kFp8GroupSize = 64;

size_t fp8_workspace_bytes(int cols);
size_t fp8_batch_workspace_bytes(int cols, int tokens);

// Native Ada FP8 Tensor Core GEMV. Eight identical activation columns are
// presented to m16n8k32 MMA; only column zero is retained. That deliberate 8x
// arithmetic waste buys coalesced 16-row weight tiles at batch one.
cudaError_t fp8_tc_gemv(
    const uint8_t *weights,
    const uint16_t *scales,
    const float *x,
    float *y,
    int rows,
    int cols,
    void *workspace,
    cudaStream_t stream = nullptr);

// Paired FP8 GEMV for equal-row weight pairs fed one activation (MLP gate +
// up): quantizes x once, then computes y_a = W_a x and y_b = W_b x in a
// single launch. blockIdx.y selects the matrix, so per-row accumulation
// order matches fp8_tc_gemv exactly (bitwise parity per output).
cudaError_t fp8_tc_gemv2(
    const uint8_t *weights_a,
    const uint16_t *scales_a,
    const uint8_t *weights_b,
    const uint16_t *scales_b,
    const float *x,
    float *y_a,
    float *y_b,
    int rows,
    int cols,
    void *workspace,
    cudaStream_t stream = nullptr);

// Token-major prefill path. A weight tile is loaded once and reused across up
// to eight activation rows per kernel launch; output_stride is in floats.
cudaError_t fp8_tc_gemv_batch(
    const uint8_t *weights,
    const uint16_t *scales,
    const float *x,
    float *y,
    int tokens,
    int rows,
    int cols,
    int output_stride,
    void *workspace,
    cudaStream_t stream = nullptr);

// Paired token-major path. Quantizes the activation rows once, then applies
// two equal-geometry matrices. This is the batched counterpart of
// fp8_tc_gemv2; callers with tokens > 1 must use this API rather than treating
// fp8_tc_gemv2's single activation as a batch.
cudaError_t fp8_tc_gemv2_batch(
    const uint8_t *weights_a,
    const uint16_t *scales_a,
    const uint8_t *weights_b,
    const uint16_t *scales_b,
    const float *x,
    float *y_a,
    float *y_b,
    int tokens,
    int rows,
    int cols,
    int output_stride,
    void *workspace,
    cudaStream_t stream = nullptr);

}  // namespace insignia::glm53
