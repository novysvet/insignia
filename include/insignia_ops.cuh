#pragma once
#include <cuda_runtime.h>
namespace insignia {
void rmsnorm_zero_centered(const float *x,const float *weight,float *y,int rows,int cols,float eps=1e-6f,cudaStream_t stream=nullptr);
void rmsnorm_gated_silu(const float *x,const float *weight,const float *gate,float *y,int rows,int cols,float eps=1e-6f,cudaStream_t stream=nullptr);
void silu_mul(const float *gate,const float *up,float *y,int count,cudaStream_t stream=nullptr);
void residual_add(float *x,const float *delta,int count,cudaStream_t stream=nullptr);
void qwen35_qk_norm_rope_gate(float *q,float *k,const uint16_t *q_weight,const uint16_t *k_weight,float *gate,const int *pos_dev,int offset,cudaStream_t stream=nullptr);
}
