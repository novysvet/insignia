#pragma once
#include <cuda_runtime.h>
namespace insignia {
constexpr int DELTA_HEADS=32, DELTA_K=128, DELTA_V=128;
void deltanet_decode(float *state, const float *q16, const float *k16, const float *v32, const float *g32, const float *beta32, float *out32, cudaStream_t stream=nullptr);
}
