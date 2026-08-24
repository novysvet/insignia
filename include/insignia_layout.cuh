#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace insignia {

constexpr int MXFP4_BLOCK_VALUES = 32;
constexpr int MXFP4_BLOCK_BYTES = 17;

#pragma pack(push, 1)
struct MxFp4Block {
    uint8_t scale;
    uint8_t q[16];
};
#pragma pack(pop)

static_assert(sizeof(MxFp4Block) == MXFP4_BLOCK_BYTES);
static_assert(alignof(MxFp4Block) == 1);

__host__ __device__ __forceinline__ float fp4_e2m1(uint8_t x) {
    constexpr float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return lut[x & 15];
}

__host__ __device__ __forceinline__ float e8m0(uint8_t x) {
    const uint32_t bits = x == 0xff ? 0x7fffffffU : uint32_t(x) << 23;
    float value;
    static_assert(sizeof(value) == sizeof(bits));
    memcpy(&value, &bits, sizeof(value));
    return value;
}

__host__ __device__ __forceinline__ uint8_t mxfp4_code(const MxFp4Block &block, int lane) {
    const uint8_t packed = block.q[lane & 15];
    return lane < 16 ? packed & 15 : packed >> 4;
}

__host__ __device__ __forceinline__ float mxfp4_value(const MxFp4Block &block, int lane) {
    return e8m0(block.scale) * fp4_e2m1(mxfp4_code(block, lane));
}

void mxfp4_gemv(const MxFp4Block *weights, const float *x, float *y, int rows, int cols, cudaStream_t stream = nullptr);

// MLX checkpoint layout: eight E2M1 nibbles per U32 and one E8M0 byte per group of 32.
void mxfp4_gemv_mlx(const uint32_t *weights, const uint8_t *scales, const float *x, float *y, int rows, int cols, int warps_per_row = 4, cudaStream_t stream = nullptr);
void mxfp4_gemv_v2(const uint32_t *weights, const uint8_t *scales, const float *x, float *y, int rows, int cols, cudaStream_t stream = nullptr);
void mxfp4_get_row_mlx(const uint32_t *weights, const uint8_t *scales, float *out, const int *row_dev, int cols, cudaStream_t stream = nullptr);
void quantize_q8_groups(const float *x, int8_t *qx, float *qscale, int cols, cudaStream_t stream = nullptr);
void mxfp4_gemv_dp4a(const uint32_t *weights, const uint8_t *scales, const int8_t *qx, const float *qscale, float *y, int rows, int cols, int warps_per_row = 4, cudaStream_t stream = nullptr);

}
