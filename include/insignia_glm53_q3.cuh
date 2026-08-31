#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace insignia::glm53 {

// GGML Q3_K is a 256-weight super-block: 32 high-mask bytes, 64 packed
// low-bit bytes, sixteen signed 6-bit sub-scales packed into 12 bytes, and one
// FP16 super-scale.  The resulting 110-byte block is 3.4375 bpw.
constexpr int kQ3KBlockWeights = 256;
constexpr int kQ3KBlockBytes = 110;
constexpr int kQ3KSubgroup = 16;
constexpr int kQ3KMaxRows = 8;

// Each activation row is Q8-per-16 (one byte/value plus one FP32 scale/group).
// This deliberately differs from llama.cpp's Q8_1-per-32 layout: Q3_K already
// has a distinct weight scale for every 16 values, so matching that boundary
// removes cross-scale correction work and improves activation fidelity.
size_t q3k_workspace_rows_bytes(int cols, int count);

cudaError_t q3k_quantize_activation_rows(
    const float *x,
    int cols,
    const int *row_ids,
    int count,
    void *workspace,
    cudaStream_t stream = nullptr);

cudaError_t q3k_quantize_swiglu_rows(
    const float *gate,
    const float *up,
    int cols,
    const int *row_ids,
    int count,
    void *workspace,
    cudaStream_t stream = nullptr);

// Independent FP32-activation oracle for one row.  It decodes Q3_K directly
// and is used to measure the error introduced only by Q8 activation packing.
cudaError_t q3k_gemv_f32(
    const uint8_t *weights,
    const float *x,
    float *y,
    int rows,
    int cols,
    cudaStream_t stream = nullptr);

// One Q3_K weight read serves count=1..8 token rows.  Output row r is written
// at y[y_ids[r] * rows].  GLM's 2048x4096 and 4096x2048 expert geometries are
// compiled separately; no generic hot-path division remains.
cudaError_t q3k_gemv_quantized_rows(
    const uint8_t *weights,
    const void *workspace,
    int count,
    float *y,
    const int *y_ids,
    int rows,
    int cols,
    cudaStream_t stream = nullptr);

// Fused gate+up projection.  Activation codes/scales and launch overhead are
// shared while the two independent Q3_K streams retain their own scale DAG.
cudaError_t q3k_gemv2_quantized_rows(
    const uint8_t *weights_a,
    const uint8_t *weights_b,
    const void *workspace,
    int count,
    float *y_a,
    float *y_b,
    const int *y_ids,
    int rows,
    int cols,
    cudaStream_t stream = nullptr);

// Down-projection epilogue: y = fmaf(dot, combine[r], y), preserving expert
// replay order at the call site.
cudaError_t q3k_gemv_acc_quantized_rows(
    const uint8_t *weights,
    const void *workspace,
    int count,
    float *y,
    const int *y_ids,
    const float *combine,
    int rows,
    int cols,
    cudaStream_t stream = nullptr);

}  // namespace insignia::glm53
