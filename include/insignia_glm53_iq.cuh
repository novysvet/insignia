#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace insignia::glm53 {

constexpr int kIQBlockWeights = 256;
constexpr int kIQ3XXSBlockBytes = 98;
constexpr int kIQ4XSBlockBytes = 136;
constexpr int kIQActivationGroup = 32;
constexpr int kIQMaxRows = 8;

// Q8-per-32 is the native activation companion for both live UD-Q3_K_XL
// formats.  One packed activation row is followed by one FP32 scale per 32
// values; count is specialized from one through eight at dispatch.
size_t iq_workspace_rows_bytes(int cols, int count);

cudaError_t iq_quantize_activation_rows(
    const float *x,
    int cols,
    const int *row_ids,
    int count,
    void *workspace,
    cudaStream_t stream = nullptr);

cudaError_t iq_quantize_swiglu_rows(
    const float *gate,
    const float *up,
    int cols,
    const int *row_ids,
    int count,
    void *workspace,
    cudaStream_t stream = nullptr);

// IQ3_XXS is a 256-entry vector codebook plus parity-compressed signs.  A
// warp owns an output row; each eight-lane cohort consumes one 98-byte block,
// and the decoded signed-byte vectors are reused by one through eight tokens.
cudaError_t iq3_xxs_gemv_rows(
    const uint8_t *weights,
    const void *workspace,
    int count,
    float *y,
    const int *y_ids,
    int rows,
    int cols,
    cudaStream_t stream = nullptr);

// IQ4_XS uses a nonlinear 16-entry nibble codebook and eight signed scales per
// 256 values.  The same warp/block map is used so exception layers can be
// dispatched without a generic matrix engine.
cudaError_t iq4_xs_gemv_rows(
    const uint8_t *weights,
    const void *workspace,
    int count,
    float *y,
    const int *y_ids,
    int rows,
    int cols,
    cudaStream_t stream = nullptr);

cudaError_t iq4_xs_gemv_acc_rows(
    const uint8_t *weights,
    const void *workspace,
    int count,
    float *y,
    const int *y_ids,
    const float *combine,
    int rows,
    int cols,
    cudaStream_t stream = nullptr);

// Independent scalar decoders used by the GGUF fixture and repacker gates.
void iq3_xxs_dequantize_row_cpu(const uint8_t *weights, float *output, int cols);
void iq4_xs_dequantize_row_cpu(const uint8_t *weights, float *output, int cols);

}  // namespace insignia::glm53
