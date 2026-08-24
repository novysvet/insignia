#pragma once
#include <cuda_runtime.h>
namespace insignia {
void gqa_decode(const float *q,const float *k_cache,const float *v_cache,float *out,int tokens,cudaStream_t stream=nullptr);
}
