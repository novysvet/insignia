#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace insignia {
void rmsnorm_bf16(const float*x,const uint16_t*w,float*y,int rows,int cols,bool zero_centered,cudaStream_t stream=nullptr);
void gated_rmsnorm_bf16(const float*x,const uint16_t*w,const float*gate,float*y,int rows,int cols,cudaStream_t stream=nullptr);
void causal_conv4_silu(float*x,float*state,const uint16_t*weight,int channels,cudaStream_t stream=nullptr);
void deltanet_parameters(float*a,float*b,const float*A_log,const uint16_t*dt_bias,int heads,cudaStream_t stream=nullptr);
void expand_gate_heads(const float*head_gate,float*gate256,cudaStream_t stream=nullptr);
void sigmoid_mul(float*x,const float*gate,int n,cudaStream_t stream=nullptr);
void split_q_gate(const float*interleaved,float*q,float*gate,cudaStream_t stream=nullptr);
void store_kv(const float*k,const float*v,float*kc,float*vc,int position,cudaStream_t stream=nullptr);
void bf16_gemv(const uint16_t*weight,const float*x,float*y,int rows,int cols,cudaStream_t stream=nullptr);
void concat(const float*a,const float*b,float*out,int n,cudaStream_t stream=nullptr);
void argmax_logits(const float*logits,int n,int*device_token,cudaStream_t stream=nullptr);
}