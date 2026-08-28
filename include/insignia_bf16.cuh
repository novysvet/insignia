#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace insignia {
// bf16 dense linear kernels for the 27B FP8 checkpoint's un-quantized tensors:
// embed table [248320,5120], lm_head [248320,5120], in_proj_a/b [48,5120], mtp.fc [5120,10240].
// All weight pointers must be 16B aligned (tiered storage rebases at load; launchers assert).
void bf16_gemv_v2(const uint32_t *w, const float *x, float *y, int rows, int cols, cudaStream_t stream = nullptr);        // T=1
void bf16_gemv2_v2(const uint32_t *w, const float *x, float *y, int rows, int cols, cudaStream_t stream = nullptr);       // T=2 pair, y[0..rows) + y[rows..2*rows)
void bf16_gemm(const uint32_t *w, const void *x16, float *y, int rows, int cols, int T, cudaStream_t stream = nullptr);   // T<=64, x16 bf16 zero-padded to 64 rows; y must be 64-row padded
void embed_gather_bf16(const uint16_t *w, const int *tokens_dev, float *out, int cols, int T, cudaStream_t stream = nullptr); // w may be host-pinned (UVA); out [T,cols] f32
void bf16_gemv_ab2_pair(const uint32_t *wa, const uint32_t *wb, const float *x0, const float *x1,
                        float *a0, float *a1, float *b0, float *b1, int cols, int heads,
                        cudaStream_t stream = nullptr);  // a/b [heads,cols] both tokens, one launch
void bf16_gemv_ab_rows(const uint32_t *wa, const uint32_t *wb, const void *x16, float *a, float *b,
                       int cols, int heads, int T, cudaStream_t stream = nullptr);  // a/b [T,heads] from bf16 x rows
}
