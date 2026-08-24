#pragma once
#include <cuda_runtime.h>
namespace insignia {
// base selects the cache slot relative to pos_dev[0]: 0 = store/attend through the current
// token position (main layers and the MTP layer both run at pos_dev[0]).
void gqa_decode(const float *q,const float *k_cache,const float *v_cache,float *out,const int *pos_dev,int base,int max_context,cudaStream_t stream=nullptr);
}
