#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace insignia::glm53 {

constexpr int kQ8GroupSize = 64;

// Q8 weights are signed bytes with one FP16 scale per 64 contiguous values.
// Activations are quantized to the same geometry in scratch and consumed with
// Ada's signed INT8 DP4A instruction.
size_t q8_workspace_bytes(int cols);

cudaError_t q8_gemv(
    const uint32_t *weights,
    const uint16_t *scales,
    const float *x,
    float *y,
    int rows,
    int cols,
    void *workspace,
    cudaStream_t stream = nullptr);

}  // namespace insignia::glm53
