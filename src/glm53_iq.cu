#include "insignia_glm53_iq.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

using insignia::glm53::kIQ3XXSBlockBytes;
using insignia::glm53::kIQ4XSBlockBytes;
using insignia::glm53::kIQActivationGroup;
using insignia::glm53::kIQBlockWeights;
using insignia::glm53::kIQMaxRows;
using insignia::glm53::kQ6KBlockBytes;

struct alignas(2) IQ3XXSBlock {
    __half d;
    uint8_t qs[96];
};
static_assert(sizeof(IQ3XXSBlock) == kIQ3XXSBlockBytes);

struct alignas(2) IQ4XSBlock {
    __half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
static_assert(sizeof(IQ4XSBlock) == kIQ4XSBlockBytes);

struct alignas(2) Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    __half d;
};
static_assert(sizeof(Q6KBlock) == kQ6KBlockBytes);

struct IQRowIds {
    int ids[kIQMaxRows]{};
};

struct IQRowOut {
    int ids[kIQMaxRows]{};
    float weights[kIQMaxRows]{};
};

constexpr uint32_t kIQ3GridHost[256] = {
#include "insignia_iq3xxs_grid.inc"
};

__device__ __align__(128) uint32_t kIQ3GridDevice[256] = {
#include "insignia_iq3xxs_grid.inc"
};

__device__ __forceinline__ uint32_t load_u32_any(const uint8_t *pointer) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    const auto *aligned = reinterpret_cast<const uint32_t *>(address & ~uintptr_t(3));
    const uint32_t lo = __ldcs(aligned);
    const unsigned shift = unsigned(address & 3u) * 8u;
    if (!shift) return lo;
    const uint32_t hi = __ldcs(aligned + 1);
    return __funnelshift_r(lo, hi, shift);
}

__host__ __device__ __forceinline__ uint32_t unpack_iq_sign8(uint32_t sign7) {
#ifdef __CUDA_ARCH__
    const uint32_t parity = __popc(sign7) & 1u;
#else
    const uint32_t parity = std::popcount(sign7) & 1u;
#endif
    return sign7 ^ (parity << 7);
}

__device__ __forceinline__ uint32_t apply_iq3_signs(
    uint32_t values, uint32_t signs, uint32_t selectors) {
    const uint32_t negative = __vcmpne4(signs & selectors, 0);
    return __vsub4(values ^ negative, negative);
}

__device__ __forceinline__ void decode_iq3_pair(
    uint32_t pair_indices, uint32_t auxiliary, int pair,
    uint32_t &decoded0, uint32_t &decoded1) {
    const int shift = 8 * (2 * (pair & 1));
    const uint32_t code0 = (pair_indices >> shift) & 255u;
    const uint32_t code1 = (pair_indices >> (shift + 8)) & 255u;
    const uint32_t sign8 =
        unpack_iq_sign8((auxiliary >> (7 * pair)) & 127u);
    const uint32_t grid0 = __ldg(kIQ3GridDevice + code0);
    const uint32_t grid1 = __ldg(kIQ3GridDevice + code1);
    const uint32_t signs = sign8 * 0x01010101u;
    decoded0 = apply_iq3_signs(grid0, signs, 0x08040201u);
    decoded1 = apply_iq3_signs(grid1, signs, 0x80402010u);
}

__device__ __forceinline__ int2 iq4_lookup8(uint32_t q4) {
    constexpr uint32_t table0 = 0xbfad9881u;
    constexpr uint32_t table1 = 0xf6eaddcfu;
    constexpr uint32_t table2 = 0x26190d01u;
    constexpr uint32_t table3 = 0x71594535u;
    const uint32_t select = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    uint32_t result[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int shift = 16 * half;
        const uint32_t low = __byte_perm(table0, table1, q4 >> shift);
        const uint32_t high = __byte_perm(table2, table3, q4 >> shift);
        result[half] = __byte_perm(low, high, select >> shift);
    }
    return make_int2(__byte_perm(result[0], result[1], 0x6420),
                     __byte_perm(result[0], result[1], 0x7531));
}

__device__ __forceinline__ uint32_t decode_q6_word(
    const Q6KBlock &block, int half, int quadrant, int word) {
    const uint8_t *ql = block.ql + 64 * half + 32 * (quadrant & 1);
    const uint8_t *qh = block.qh + 32 * half;
    const uint32_t low =
        (load_u32_any(ql + 4 * word) >> (4 * (quadrant >> 1))) &
        0x0f0f0f0fu;
    const uint32_t high =
        (load_u32_any(qh + 4 * word) >> (2 * quadrant)) & 0x03030303u;
    return __vsub4(low | (high << 4), 0x20202020u);
}

template <int R>
__global__ __launch_bounds__(128, 4) void iq_quantize_x32_rows_kernel(
    const float *__restrict__ x,
    uint32_t *__restrict__ xq,
    float *__restrict__ xscale,
    int groups,
    int words_per_row,
    IQRowIds rows) {
    const int group = threadIdx.x;
    if (group >= groups) return;
    const float4 *source = reinterpret_cast<const float4 *>(
        x + static_cast<size_t>(rows.ids[blockIdx.x]) * groups * 32 + group * 32);
    uint32_t *row_q = xq + static_cast<size_t>(blockIdx.x) * words_per_row;
    float *row_scale = xscale + static_cast<size_t>(blockIdx.x) * groups;
    float values[32];
    float maximum = 0.0f;
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        const float4 value = __ldg(source + word);
        values[4 * word + 0] = value.x;
        values[4 * word + 1] = value.y;
        values[4 * word + 2] = value.z;
        values[4 * word + 3] = value.w;
        maximum = fmaxf(maximum, fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                                       fmaxf(fabsf(value.z), fabsf(value.w))));
    }
    const float inverse = maximum > 0.0f ? 127.0f / maximum : 0.0f;
    row_scale[group] = maximum * (1.0f / 127.0f);
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        uint32_t packed = 0;
#pragma unroll
        for (int byte = 0; byte < 4; ++byte)
            packed |= uint32_t(uint8_t(__float2int_rn(values[4 * word + byte] * inverse)))
                      << (8 * byte);
        row_q[group * 8 + word] = packed;
    }
}

template <int R>
__global__ __launch_bounds__(128, 4) void iq_quantize_swiglu_x32_rows_kernel(
    const float *__restrict__ gate,
    const float *__restrict__ up,
    uint32_t *__restrict__ xq,
    float *__restrict__ xscale,
    int groups,
    int words_per_row,
    IQRowIds rows) {
    const int group = threadIdx.x;
    if (group >= groups) return;
    const size_t base = static_cast<size_t>(rows.ids[blockIdx.x]) * groups * 32;
    const float4 *gate4 = reinterpret_cast<const float4 *>(gate + base + group * 32);
    const float4 *up4 = reinterpret_cast<const float4 *>(up + base + group * 32);
    uint32_t *row_q = xq + static_cast<size_t>(blockIdx.x) * words_per_row;
    float *row_scale = xscale + static_cast<size_t>(blockIdx.x) * groups;
    float values[32];
    float maximum = 0.0f;
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        const float4 g = __ldg(gate4 + word);
        const float4 u = __ldg(up4 + word);
        const float gv[4] = {g.x, g.y, g.z, g.w};
        const float uv[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float gc = fminf(gv[i], 10.0f);
            const float uc = fminf(fmaxf(uv[i], -10.0f), 10.0f);
            const float value = (gc / (1.0f + __expf(-gc))) * uc;
            values[4 * word + i] = value;
            maximum = fmaxf(maximum, fabsf(value));
        }
    }
    const float inverse = maximum > 0.0f ? 127.0f / maximum : 0.0f;
    row_scale[group] = maximum * (1.0f / 127.0f);
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        uint32_t packed = 0;
#pragma unroll
        for (int byte = 0; byte < 4; ++byte)
            packed |= uint32_t(uint8_t(__float2int_rn(values[4 * word + byte] * inverse)))
                      << (8 * byte);
        row_q[group * 8 + word] = packed;
    }
}

template <int R, int BLOCKS, bool REPACKED, int WARPS>
__global__ __launch_bounds__(WARPS * 32, 16 / WARPS) void iq3_xxs_rows_kernel(
    const uint8_t *__restrict__ weights,
    const uint32_t *__restrict__ xq,
    const float *__restrict__ xscale,
    int words_per_row,
    float *__restrict__ y,
    IQRowOut out) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int cohort = lane >> 3;
    const int subgroup = lane & 7;
    const int row = blockIdx.x * WARPS + warp;
    constexpr int row_bytes = BLOCKS * kIQ3XXSBlockBytes;
    const uint8_t *row_weights = weights + static_cast<size_t>(row) * row_bytes;
    float sums[R] = {};
#pragma unroll
    for (int wave = 0; wave < BLOCKS / 4; ++wave) {
        const int block_id = 4 * wave + cohort;
        const uint8_t *indices;
        const uint8_t *auxiliary;
        float d;
        if constexpr (REPACKED) {
            const auto *scales = reinterpret_cast<const __half *>(row_weights);
            indices = row_weights + 2 * BLOCKS + block_id * 64 + subgroup * 8;
            auxiliary = row_weights + 66 * BLOCKS + block_id * 32 + subgroup * 4;
            d = __half2float(scales[block_id]);
        } else {
            const IQ3XXSBlock &block =
                reinterpret_cast<const IQ3XXSBlock *>(row_weights)[block_id];
            indices = block.qs + 8 * subgroup;
            auxiliary = block.qs + 64 + 4 * subgroup;
            d = __half2float(block.d);
        }
        const uint32_t indices0 = REPACKED
            ? __ldcs(reinterpret_cast<const uint32_t *>(indices + 0))
            : load_u32_any(indices + 0);
        const uint32_t indices1 = REPACKED
            ? __ldcs(reinterpret_cast<const uint32_t *>(indices + 4))
            : load_u32_any(indices + 4);
        const uint32_t aux = REPACKED
            ? __ldcs(reinterpret_cast<const uint32_t *>(auxiliary))
            : load_u32_any(auxiliary);
        const int activation_group = block_id * 8 + subgroup;
        const float weight_scale = d * (0.25f + 0.5f * float(aux >> 28));
        if constexpr (R == 2) {
            // The two-row decode path is occupancy-sensitive on Ada.
            // Consume each signed codebook pair immediately so the compiler
            // keeps two decoded words instead of the whole eight-word block.
            // DP4A order within every row is unchanged.
            int dots[R] = {};
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const uint32_t pair_indices = pair < 2 ? indices0 : indices1;
                uint32_t decoded0, decoded1;
                decode_iq3_pair(pair_indices, aux, pair, decoded0, decoded1);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t *activation =
                        xq + static_cast<size_t>(r) * words_per_row +
                        activation_group * 8 + 2 * pair;
                    dots[r] = __dp4a(int(decoded0), int(__ldg(activation + 0)),
                                     dots[r]);
                    dots[r] = __dp4a(int(decoded1), int(__ldg(activation + 1)),
                                     dots[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float scale = xscale[static_cast<size_t>(r) * BLOCKS * 8 +
                                           activation_group];
                sums[r] = fmaf(float(dots[r]), weight_scale * scale, sums[r]);
            }
        } else {
            uint32_t decoded[8];
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const uint32_t pair_indices = pair < 2 ? indices0 : indices1;
                decode_iq3_pair(pair_indices, aux, pair,
                                decoded[2 * pair + 0], decoded[2 * pair + 1]);
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t *activation =
                    xq + static_cast<size_t>(r) * words_per_row +
                    activation_group * 8;
                int dot = 0;
#pragma unroll
                for (int word = 0; word < 8; ++word)
                    dot = __dp4a(int(decoded[word]), int(__ldg(activation + word)),
                                 dot);
                const float scale = xscale[static_cast<size_t>(r) * BLOCKS * 8 +
                                           activation_group];
                sums[r] = fmaf(float(dot), weight_scale * scale, sums[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            sums[r] += __shfl_down_sync(0xffffffffu, sums[r], offset);
        if (!lane)
            y[static_cast<size_t>(out.ids[r]) * gridDim.x * WARPS + row] = sums[r];
    }
}

template <int R, int BLOCKS, bool ACCUMULATE>
__global__ __launch_bounds__(256, 2) void iq4_xs_rows_kernel(
    const IQ4XSBlock *__restrict__ weights,
    const uint32_t *__restrict__ xq,
    const float *__restrict__ xscale,
    int words_per_row,
    float *__restrict__ y,
    IQRowOut out) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int cohort = lane >> 3;
    const int subgroup = lane & 7;
    const int row = blockIdx.x * 8 + warp;
    const IQ4XSBlock *row_weights = weights + static_cast<size_t>(row) * BLOCKS;
    float sums[R] = {};
#pragma unroll
    for (int wave = 0; wave < BLOCKS / 4; ++wave) {
        const int block_id = 4 * wave + cohort;
        const IQ4XSBlock &block = row_weights[block_id];
        const uint32_t *packed = reinterpret_cast<const uint32_t *>(
            block.qs + 16 * subgroup);
        int decoded[8];
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            const int2 values = iq4_lookup8(__ldcs(packed + word));
            decoded[word] = values.x;
            decoded[word + 4] = values.y;
        }
        const int low = (block.scales_l[subgroup >> 1] >> (4 * (subgroup & 1))) & 15;
        const int high = (block.scales_h >> (2 * subgroup)) & 3;
        const float weight_scale = __half2float(block.d) * float((low | (high << 4)) - 32);
        const int activation_group = block_id * 8 + subgroup;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint32_t *activation = xq + static_cast<size_t>(r) * words_per_row +
                                         activation_group * 8;
            int dot = 0;
#pragma unroll
            for (int word = 0; word < 8; ++word)
                dot = __dp4a(decoded[word], int(__ldg(activation + word)), dot);
            const float scale = xscale[static_cast<size_t>(r) * BLOCKS * 8 +
                                       activation_group];
            sums[r] = fmaf(float(dot), weight_scale * scale, sums[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            sums[r] += __shfl_down_sync(0xffffffffu, sums[r], offset);
        if (!lane) {
            float *destination = y + static_cast<size_t>(out.ids[r]) * gridDim.x * 8 + row;
            if constexpr (ACCUMULATE)
                *destination = fmaf(sums[r], out.weights[r], *destination);
            else
                *destination = sums[r];
        }
    }
}

template <int R, int BLOCKS, bool ACCUMULATE>
__global__ __launch_bounds__(256, 2) void q6_k_rows_kernel(
    const Q6KBlock *__restrict__ weights,
    const uint32_t *__restrict__ xq,
    const float *__restrict__ xscale,
    int words_per_row,
    float *__restrict__ y,
    IQRowOut out) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int cohort = lane >> 3;
    const int subgroup = lane & 7;
    const int row = blockIdx.x * 8 + warp;
    const Q6KBlock *row_weights = weights + static_cast<size_t>(row) * BLOCKS;
    float sums[R] = {};
#pragma unroll
    for (int wave = 0; wave < BLOCKS / 4; ++wave) {
        const int block_id = 4 * wave + cohort;
        const Q6KBlock &block = row_weights[block_id];
        const int half = subgroup >> 2;
        const int quadrant = subgroup & 3;
        const int scale_index = 8 * half + 2 * quadrant;
        const float d = __half2float(block.d);
        const float weight_scale0 = d * float(block.scales[scale_index + 0]);
        const float weight_scale1 = d * float(block.scales[scale_index + 1]);
        const int activation_group = block_id * 8 + subgroup;
        if constexpr (R <= 2) {
            int dot0[R] = {}, dot1[R] = {};
#pragma unroll
            for (int word = 0; word < 8; ++word) {
                const uint32_t decoded =
                    decode_q6_word(block, half, quadrant, word);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t *activation =
                        xq + static_cast<size_t>(r) * words_per_row +
                        activation_group * 8;
                    if (word < 4)
                        dot0[r] = __dp4a(int(decoded), int(__ldg(activation + word)),
                                         dot0[r]);
                    else
                        dot1[r] = __dp4a(int(decoded), int(__ldg(activation + word)),
                                         dot1[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float scale = xscale[static_cast<size_t>(r) * BLOCKS * 8 +
                                           activation_group];
                sums[r] = fmaf(float(dot0[r]), weight_scale0 * scale, sums[r]);
                sums[r] = fmaf(float(dot1[r]), weight_scale1 * scale, sums[r]);
            }
        } else {
            uint32_t decoded[8];
#pragma unroll
            for (int word = 0; word < 8; ++word)
                decoded[word] = decode_q6_word(block, half, quadrant, word);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t *activation =
                    xq + static_cast<size_t>(r) * words_per_row +
                    activation_group * 8;
                int dot0 = 0, dot1 = 0;
#pragma unroll
                for (int word = 0; word < 4; ++word)
                    dot0 = __dp4a(int(decoded[word]), int(__ldg(activation + word)),
                                  dot0);
#pragma unroll
                for (int word = 4; word < 8; ++word)
                    dot1 = __dp4a(int(decoded[word]), int(__ldg(activation + word)),
                                  dot1);
                const float scale = xscale[static_cast<size_t>(r) * BLOCKS * 8 +
                                           activation_group];
                sums[r] = fmaf(float(dot0), weight_scale0 * scale, sums[r]);
                sums[r] = fmaf(float(dot1), weight_scale1 * scale, sums[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            sums[r] += __shfl_down_sync(0xffffffffu, sums[r], offset);
        if (!lane) {
            float *destination =
                y + static_cast<size_t>(out.ids[r]) * gridDim.x * 8 + row;
            if constexpr (ACCUMULATE)
                *destination = fmaf(sums[r], out.weights[r], *destination);
            else
                *destination = sums[r];
        }
    }
}

// Prefill is a different machine from decode.  Each CTA owns sixteen output
// rows and thirty-two routed tokens.  It expands one 16x16 weight tile to FP16
// once, then two warps reuse it through HMMA.  The expert payload is therefore
// read once per 32 tokens instead of once per eight-token GEMV launch.
template <int BLOCKS>
__global__ __launch_bounds__(128, 4) void iq3_xxs_wmma32_kernel(
    const uint8_t *__restrict__ weights,
    const float *__restrict__ x,
    float *__restrict__ y,
    int rows,
    int cols) {
    using namespace nvcuda;
    __shared__ __align__(128) __half shared_a[16 * 32];
    __shared__ __align__(128) __half shared_b[2][16 * 32];
    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int row_start = blockIdx.x * 16;
    const int token_start = blockIdx.y * 32;
    constexpr int row_bytes = BLOCKS * kIQ3XXSBlockBytes;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    for (int k0 = 0; k0 < cols; k0 += 32) {
        // Decode the whole 32-value scale/sign group once.  The 128 threads
        // each own one codeword, covering the entire 16x32 tile in one pass.
        const int block_id = k0 >> 8;
        const int subgroup = (k0 >> 5) & 7;
#pragma unroll
        for (int slot = thread; slot < 16 * 8; slot += 128) {
            const int local_row = slot >> 3;
            const int local_code = slot & 7;
            const auto &block = reinterpret_cast<const IQ3XXSBlock *>(
                weights + static_cast<size_t>(row_start + local_row) * row_bytes)[block_id];
            const uint32_t auxiliary =
                load_u32_any(block.qs + 64 + 4 * subgroup);
            const int pair = local_code >> 1;
            const int side = local_code & 1;
            const uint32_t code = block.qs[8 * subgroup + 2 * pair + side];
            const uint32_t sign8 =
                unpack_iq_sign8((auxiliary >> (7 * pair)) & 127u);
            const uint32_t signs = sign8 * 0x01010101u;
            const uint32_t decoded = apply_iq3_signs(
                __ldg(kIQ3GridDevice + code), signs,
                side ? 0x80402010u : 0x08040201u);
            const float scale = __half2float(block.d) *
                                (0.25f + 0.5f * float(auxiliary >> 28));
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const int8_t value = int8_t(decoded >> (8 * byte));
                shared_a[local_row * 32 + 4 * local_code + byte] =
                    __float2half_rn(float(value) * scale);
            }
        }

        // The input tile is stored exactly in the col-major layout expected by
        // matrix_b: [K, token].  Each thread converts eight FP32 activations.
#pragma unroll
        for (int item = thread; item < 32 * 32; item += 128) {
            const int local_token = item >> 5;
            const int local_k = item & 31;
            shared_b[local_token >> 4][local_k + (local_token & 15) * 32] =
                __float2half_rn(x[static_cast<size_t>(token_start + local_token) * cols +
                                   k0 + local_k]);
        }
        __syncthreads();
        if (warp < 2) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                               wmma::row_major> a_fragment;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                               wmma::col_major> b_fragment;
                wmma::load_matrix_sync(a_fragment, shared_a + 16 * half, 32);
                wmma::load_matrix_sync(b_fragment, shared_b[warp] + 16 * half, 32);
                wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
            }
        }
        __syncthreads();
    }
    if (warp < 2)
        wmma::store_matrix_sync(
            y + static_cast<size_t>(token_start + warp * 16) * rows + row_start,
            accumulator, rows, wmma::mem_col_major);
}

template <int BLOCKS>
__global__ __launch_bounds__(64, 8) void iq4_xs_wmma32_kernel(
    const IQ4XSBlock *__restrict__ weights,
    const float *__restrict__ x,
    float *__restrict__ y,
    int rows,
    int cols) {
    using namespace nvcuda;
    __shared__ __align__(128) __half shared_a[16 * 32];
    __shared__ __align__(128) __half shared_b[2][16 * 32];
    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int row_start = blockIdx.x * 16;
    const int token_start = blockIdx.y * 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    for (int k0 = 0; k0 < cols; k0 += 32) {
        const int local_row = thread >> 2;
        const int local_word = thread & 3;
        const int block_id = k0 >> 8;
        const int subgroup = (k0 >> 5) & 7;
        const IQ4XSBlock &block =
            weights[static_cast<size_t>(row_start + local_row) * BLOCKS + block_id];
        const int low =
            (block.scales_l[subgroup >> 1] >> (4 * (subgroup & 1))) & 15;
        const int high = (block.scales_h >> (2 * subgroup)) & 3;
        const float scale = __half2float(block.d) *
                            float((low | (high << 4)) - 32);
        const int2 decoded8 = iq4_lookup8(__ldcs(
            reinterpret_cast<const uint32_t *>(block.qs + 16 * subgroup) +
            local_word));
#pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
            const int8_t low_value = int8_t(uint32_t(decoded8.x) >> (8 * byte));
            const int8_t high_value = int8_t(uint32_t(decoded8.y) >> (8 * byte));
            shared_a[local_row * 32 + 4 * local_word + byte] =
                __float2half_rn(float(low_value) * scale);
            shared_a[local_row * 32 + 16 + 4 * local_word + byte] =
                __float2half_rn(float(high_value) * scale);
        }
#pragma unroll
        for (int item = thread; item < 32 * 32; item += 64) {
            const int local_token = item >> 5;
            const int local_k = item & 31;
            shared_b[local_token >> 4][local_k + (local_token & 15) * 32] =
                __float2half_rn(x[static_cast<size_t>(token_start + local_token) * cols +
                                   k0 + local_k]);
        }
        __syncthreads();
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                           wmma::row_major> a_fragment;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                           wmma::col_major> b_fragment;
            wmma::load_matrix_sync(a_fragment, shared_a + 16 * half, 32);
            wmma::load_matrix_sync(b_fragment, shared_b[warp] + 16 * half, 32);
            wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(
        y + static_cast<size_t>(token_start + warp * 16) * rows + row_start,
        accumulator, rows, wmma::mem_col_major);
}

template <int R, int BLOCKS, bool REPACKED, int WARPS = 8>
cudaError_t launch_iq3(const uint8_t *weights, const uint32_t *xq,
                       const float *xscale, int words_per_row, float *y,
                       IQRowOut out, int rows, cudaStream_t stream) {
    iq3_xxs_rows_kernel<R, BLOCKS, REPACKED, WARPS>
        <<<rows / WARPS, WARPS * 32, 0, stream>>>(weights, xq, xscale,
                                                  words_per_row, y, out);
    return cudaPeekAtLastError();
}

template <int R, int BLOCKS, bool ACCUMULATE>
cudaError_t launch_iq4(const uint8_t *weights, const uint32_t *xq,
                       const float *xscale, int words_per_row, float *y,
                       IQRowOut out, int rows, cudaStream_t stream) {
    iq4_xs_rows_kernel<R, BLOCKS, ACCUMULATE><<<rows / 8, 256, 0, stream>>>(
        reinterpret_cast<const IQ4XSBlock *>(weights), xq, xscale,
        words_per_row, y, out);
    return cudaPeekAtLastError();
}

template <int R, int BLOCKS, bool ACCUMULATE>
cudaError_t launch_q6(const uint8_t *weights, const uint32_t *xq,
                      const float *xscale, int words_per_row, float *y,
                      IQRowOut out, int rows, cudaStream_t stream) {
    q6_k_rows_kernel<R, BLOCKS, ACCUMULATE><<<rows / 8, 256, 0, stream>>>(
        reinterpret_cast<const Q6KBlock *>(weights), xq, xscale,
        words_per_row, y, out);
    return cudaPeekAtLastError();
}

bool valid_geometry(int rows, int cols, int count) {
    return rows > 0 && (rows & 7) == 0 && (cols == 2048 || cols == 4096) &&
           count > 0 && count <= kIQMaxRows;
}

template <bool REPACKED>
cudaError_t dispatch_iq3(const uint8_t *weights, const void *workspace,
                         int count, float *y, const int *y_ids,
                         int rows, int cols, cudaStream_t stream) {
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    const auto *xq = reinterpret_cast<const uint32_t *>(workspace);
    const auto *xscale = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(workspace) + count * aligned);
    IQRowOut out{};
    for (int r = 0; r < count; ++r) out.ids[r] = y_ids[r];
#define INSIGNIA_IQ3_CASE(R)                                                       \
    case R:                                                                        \
        return cols == 4096                                                        \
            ? launch_iq3<R, 16, REPACKED>(                                         \
                  weights, xq, xscale, int(aligned / 4), y, out, rows, stream)     \
            : launch_iq3<R, 8, REPACKED>(                                          \
                  weights, xq, xscale, int(aligned / 4), y, out, rows, stream)
    switch (count) {
        case 1:
            return cols == 4096
                ? launch_iq3<1, 16, REPACKED, 2>(
                      weights, xq, xscale, int(aligned / 4), y, out,
                      rows, stream)
                : launch_iq3<1, 8, REPACKED, 2>(
                      weights, xq, xscale, int(aligned / 4), y, out,
                      rows, stream);
        INSIGNIA_IQ3_CASE(2);
        INSIGNIA_IQ3_CASE(3); INSIGNIA_IQ3_CASE(4);
        INSIGNIA_IQ3_CASE(5); INSIGNIA_IQ3_CASE(6);
        INSIGNIA_IQ3_CASE(7); INSIGNIA_IQ3_CASE(8);
        default: return cudaErrorInvalidValue;
    }
#undef INSIGNIA_IQ3_CASE
}

template <bool ACCUMULATE>
cudaError_t dispatch_iq4(const uint8_t *weights, const void *workspace,
                         int count, float *y, const int *y_ids,
                         const float *combine, int rows, int cols,
                         cudaStream_t stream) {
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    const auto *xq = reinterpret_cast<const uint32_t *>(workspace);
    const auto *xscale = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(workspace) + count * aligned);
    IQRowOut out{};
    for (int r = 0; r < count; ++r) {
        out.ids[r] = y_ids[r];
        if constexpr (ACCUMULATE) out.weights[r] = combine[r];
    }
#define INSIGNIA_IQ4_CASE(R)                                                       \
    case R:                                                                        \
        return cols == 4096                                                        \
            ? launch_iq4<R, 16, ACCUMULATE>(weights, xq, xscale, int(aligned / 4), \
                                              y, out, rows, stream)                 \
            : launch_iq4<R, 8, ACCUMULATE>(weights, xq, xscale, int(aligned / 4),  \
                                             y, out, rows, stream)
    switch (count) {
        INSIGNIA_IQ4_CASE(1); INSIGNIA_IQ4_CASE(2);
        INSIGNIA_IQ4_CASE(3); INSIGNIA_IQ4_CASE(4);
        INSIGNIA_IQ4_CASE(5); INSIGNIA_IQ4_CASE(6);
        INSIGNIA_IQ4_CASE(7); INSIGNIA_IQ4_CASE(8);
        default: return cudaErrorInvalidValue;
    }
#undef INSIGNIA_IQ4_CASE
}

template <bool ACCUMULATE>
cudaError_t dispatch_q6(const uint8_t *weights, const void *workspace,
                        int count, float *y, const int *y_ids,
                        const float *combine, int rows, int cols,
                        cudaStream_t stream) {
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    const auto *xq = reinterpret_cast<const uint32_t *>(workspace);
    const auto *xscale = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(workspace) + count * aligned);
    IQRowOut out{};
    for (int r = 0; r < count; ++r) {
        out.ids[r] = y_ids[r];
        if constexpr (ACCUMULATE) out.weights[r] = combine[r];
    }
#define INSIGNIA_Q6_CASE(R)                                                        \
    case R:                                                                        \
        return cols == 4096                                                        \
            ? launch_q6<R, 16, ACCUMULATE>(weights, xq, xscale,                   \
                                            int(aligned / 4), y, out, rows, stream)\
            : launch_q6<R, 8, ACCUMULATE>(weights, xq, xscale,                    \
                                           int(aligned / 4), y, out, rows, stream)
    switch (count) {
        INSIGNIA_Q6_CASE(1); INSIGNIA_Q6_CASE(2);
        INSIGNIA_Q6_CASE(3); INSIGNIA_Q6_CASE(4);
        INSIGNIA_Q6_CASE(5); INSIGNIA_Q6_CASE(6);
        INSIGNIA_Q6_CASE(7); INSIGNIA_Q6_CASE(8);
        default: return cudaErrorInvalidValue;
    }
#undef INSIGNIA_Q6_CASE
}

}  // namespace

namespace insignia::glm53 {

size_t iq_workspace_rows_bytes(int cols, int count) {
    if ((cols != 2048 && cols != 4096) || count <= 0 || count > kIQMaxRows)
        return 0;
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    return static_cast<size_t>(count) *
           (aligned + static_cast<size_t>(cols / kIQActivationGroup) * sizeof(float));
}

cudaError_t iq3_xxs_gemm_prefill32(
    const uint8_t *weights, const float *x, int tokens, float *y,
    int rows, int cols, cudaStream_t stream) {
    if (!weights || !x || !y || rows <= 0 || (rows & 15) ||
        tokens <= 0 || (tokens & 31) || (cols != 2048 && cols != 4096))
        return cudaErrorInvalidValue;
    if (cols == 4096)
        iq3_xxs_wmma32_kernel<16><<<dim3(rows / 16, tokens / 32), 128, 0, stream>>>(
            weights, x, y, rows, cols);
    else
        iq3_xxs_wmma32_kernel<8><<<dim3(rows / 16, tokens / 32), 128, 0, stream>>>(
            weights, x, y, rows, cols);
    return cudaPeekAtLastError();
}

cudaError_t iq4_xs_gemm_prefill32(
    const uint8_t *weights, const float *x, int tokens, float *y,
    int rows, int cols, cudaStream_t stream) {
    if (!weights || !x || !y || rows <= 0 || (rows & 15) ||
        tokens <= 0 || (tokens & 31) || (cols != 2048 && cols != 4096))
        return cudaErrorInvalidValue;
    if (cols == 4096)
        iq4_xs_wmma32_kernel<16><<<dim3(rows / 16, tokens / 32), 64, 0, stream>>>(
            reinterpret_cast<const IQ4XSBlock *>(weights), x, y, rows, cols);
    else
        iq4_xs_wmma32_kernel<8><<<dim3(rows / 16, tokens / 32), 64, 0, stream>>>(
            reinterpret_cast<const IQ4XSBlock *>(weights), x, y, rows, cols);
    return cudaPeekAtLastError();
}

cudaError_t iq_quantize_activation_rows(
    const float *x, int cols, const int *row_ids, int count, void *workspace,
    cudaStream_t stream) {
    if (!x || !row_ids || !workspace || !valid_geometry(8, cols, count))
        return cudaErrorInvalidValue;
    const int groups = cols / kIQActivationGroup;
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    auto *xq = reinterpret_cast<uint32_t *>(workspace);
    auto *xscale = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(workspace) +
                                             count * aligned);
    IQRowIds rows{};
    for (int r = 0; r < count; ++r) rows.ids[r] = row_ids[r];
#define INSIGNIA_IQ_QUANT_CASE(R)                                                  \
    case R:                                                                        \
        iq_quantize_x32_rows_kernel<R><<<R, 128, 0, stream>>>(                    \
            x, xq, xscale, groups, int(aligned / 4), rows);                        \
        return cudaPeekAtLastError()
    switch (count) {
        INSIGNIA_IQ_QUANT_CASE(1); INSIGNIA_IQ_QUANT_CASE(2);
        INSIGNIA_IQ_QUANT_CASE(3); INSIGNIA_IQ_QUANT_CASE(4);
        INSIGNIA_IQ_QUANT_CASE(5); INSIGNIA_IQ_QUANT_CASE(6);
        INSIGNIA_IQ_QUANT_CASE(7); INSIGNIA_IQ_QUANT_CASE(8);
        default: return cudaErrorInvalidValue;
    }
#undef INSIGNIA_IQ_QUANT_CASE
}

cudaError_t iq_quantize_swiglu_rows(
    const float *gate, const float *up, int cols, const int *row_ids, int count,
    void *workspace, cudaStream_t stream) {
    if (!gate || !up || !row_ids || !workspace || !valid_geometry(8, cols, count))
        return cudaErrorInvalidValue;
    const int groups = cols / kIQActivationGroup;
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    auto *xq = reinterpret_cast<uint32_t *>(workspace);
    auto *xscale = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(workspace) +
                                             count * aligned);
    IQRowIds rows{};
    for (int r = 0; r < count; ++r) rows.ids[r] = row_ids[r];
#define INSIGNIA_IQ_SWIGLU_CASE(R)                                                 \
    case R:                                                                        \
        iq_quantize_swiglu_x32_rows_kernel<R><<<R, 128, 0, stream>>>(             \
            gate, up, xq, xscale, groups, int(aligned / 4), rows);                 \
        return cudaPeekAtLastError()
    switch (count) {
        INSIGNIA_IQ_SWIGLU_CASE(1); INSIGNIA_IQ_SWIGLU_CASE(2);
        INSIGNIA_IQ_SWIGLU_CASE(3); INSIGNIA_IQ_SWIGLU_CASE(4);
        INSIGNIA_IQ_SWIGLU_CASE(5); INSIGNIA_IQ_SWIGLU_CASE(6);
        INSIGNIA_IQ_SWIGLU_CASE(7); INSIGNIA_IQ_SWIGLU_CASE(8);
        default: return cudaErrorInvalidValue;
    }
#undef INSIGNIA_IQ_SWIGLU_CASE
}

cudaError_t iq3_xxs_gemv_rows(
    const uint8_t *weights, const void *workspace, int count, float *y,
    const int *y_ids, int rows, int cols, cudaStream_t stream) {
    if (!weights || !workspace || !y || !y_ids || !valid_geometry(rows, cols, count))
        return cudaErrorInvalidValue;
    return dispatch_iq3<false>(weights, workspace, count, y, y_ids,
                               rows, cols, stream);
}

void iq3_xxs_repack_cpu(const uint8_t *source, uint8_t *destination,
                        int rows, int cols) {
    const int blocks = cols / kIQBlockWeights;
    const size_t row_bytes = static_cast<size_t>(blocks) * kIQ3XXSBlockBytes;
    for (int row = 0; row < rows; ++row) {
        const auto *input = reinterpret_cast<const IQ3XXSBlock *>(
            source + static_cast<size_t>(row) * row_bytes);
        uint8_t *output = destination + static_cast<size_t>(row) * row_bytes;
        for (int block = 0; block < blocks; ++block) {
            std::memcpy(output + 2 * block, &input[block].d, 2);
            std::memcpy(output + 2 * blocks + 64 * block,
                        input[block].qs, 64);
            std::memcpy(output + 66 * blocks + 32 * block,
                        input[block].qs + 64, 32);
        }
    }
}

cudaError_t iq3_xxs_gemv_repacked_rows(
    const uint8_t *weights, const void *workspace, int count, float *y,
    const int *y_ids, int rows, int cols, cudaStream_t stream) {
    if (!weights || !workspace || !y || !y_ids || !valid_geometry(rows, cols, count))
        return cudaErrorInvalidValue;
    return dispatch_iq3<true>(weights, workspace, count, y, y_ids,
                              rows, cols, stream);
}

cudaError_t iq4_xs_gemv_rows(
    const uint8_t *weights, const void *workspace, int count, float *y,
    const int *y_ids, int rows, int cols, cudaStream_t stream) {
    if (!weights || !workspace || !y || !y_ids || !valid_geometry(rows, cols, count))
        return cudaErrorInvalidValue;
    return dispatch_iq4<false>(weights, workspace, count, y, y_ids, nullptr,
                               rows, cols, stream);
}

cudaError_t iq4_xs_gemv_acc_rows(
    const uint8_t *weights, const void *workspace, int count, float *y,
    const int *y_ids, const float *combine, int rows, int cols,
    cudaStream_t stream) {
    if (!weights || !workspace || !y || !y_ids || !combine ||
        !valid_geometry(rows, cols, count))
        return cudaErrorInvalidValue;
    return dispatch_iq4<true>(weights, workspace, count, y, y_ids, combine,
                              rows, cols, stream);
}

cudaError_t q6_k_gemv_rows(
    const uint8_t *weights, const void *workspace, int count, float *y,
    const int *y_ids, int rows, int cols, cudaStream_t stream) {
    if (!weights || !workspace || !y || !y_ids ||
        !valid_geometry(rows, cols, count))
        return cudaErrorInvalidValue;
    return dispatch_q6<false>(weights, workspace, count, y, y_ids, nullptr,
                               rows, cols, stream);
}

cudaError_t q6_k_gemv_acc_rows(
    const uint8_t *weights, const void *workspace, int count, float *y,
    const int *y_ids, const float *combine, int rows, int cols,
    cudaStream_t stream) {
    if (!weights || !workspace || !y || !y_ids || !combine ||
        !valid_geometry(rows, cols, count))
        return cudaErrorInvalidValue;
    return dispatch_q6<true>(weights, workspace, count, y, y_ids, combine,
                              rows, cols, stream);
}

void iq3_xxs_dequantize_row_cpu(const uint8_t *weights, float *output, int cols) {
    const int blocks = cols / kIQBlockWeights;
    const auto *row = reinterpret_cast<const IQ3XXSBlock *>(weights);
    for (int block_id = 0; block_id < blocks; ++block_id) {
        const IQ3XXSBlock &block = row[block_id];
        const float d = __half2float(block.d);
        for (int subgroup = 0; subgroup < 8; ++subgroup) {
            uint32_t aux;
            std::memcpy(&aux, block.qs + 64 + 4 * subgroup, sizeof(aux));
            const float scale = d * (0.5f + float(aux >> 28)) * 0.5f;
            const uint8_t *indices = block.qs + 8 * subgroup;
            for (int pair = 0; pair < 4; ++pair) {
                const uint32_t sign7 = (aux >> (7 * pair)) & 127u;
                const uint32_t signs = sign7 ^ ((std::popcount(sign7) & 1u) << 7);
                for (int side = 0; side < 2; ++side) {
                    const uint32_t grid = kIQ3GridHost[indices[2 * pair + side]];
                    for (int byte = 0; byte < 4; ++byte) {
                        const int element = 8 * pair + 4 * side + byte;
                        const float sign = (signs >> (4 * side + byte)) & 1u ? -1.0f : 1.0f;
                        output[block_id * 256 + subgroup * 32 + element] =
                            scale * float((grid >> (8 * byte)) & 255u) * sign;
                    }
                }
            }
        }
    }
}

void iq4_xs_dequantize_row_cpu(const uint8_t *weights, float *output, int cols) {
    constexpr int8_t table[16] =
        {-127, -104, -83, -65, -49, -35, -22, -10,
          1,   13,   25,  38,  53,  69,  89, 113};
    const int blocks = cols / kIQBlockWeights;
    const auto *row = reinterpret_cast<const IQ4XSBlock *>(weights);
    for (int block_id = 0; block_id < blocks; ++block_id) {
        const IQ4XSBlock &block = row[block_id];
        const float d = __half2float(block.d);
        for (int subgroup = 0; subgroup < 8; ++subgroup) {
            const int low = (block.scales_l[subgroup >> 1] >> (4 * (subgroup & 1))) & 15;
            const int high = (block.scales_h >> (2 * subgroup)) & 3;
            const float scale = d * float((low | (high << 4)) - 32);
            const uint8_t *packed = block.qs + 16 * subgroup;
            for (int element = 0; element < 16; ++element) {
                output[block_id * 256 + subgroup * 32 + element] =
                    scale * table[packed[element] & 15];
                output[block_id * 256 + subgroup * 32 + element + 16] =
                    scale * table[packed[element] >> 4];
            }
        }
    }
}

void q6_k_dequantize_row_cpu(const uint8_t *weights, float *output, int cols) {
    const int blocks = cols / kIQBlockWeights;
    const auto *row = reinterpret_cast<const Q6KBlock *>(weights);
    for (int block_id = 0; block_id < blocks; ++block_id) {
        const Q6KBlock &block = row[block_id];
        const float d = __half2float(block.d);
        for (int half = 0; half < 2; ++half) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t ql0 = block.ql[64 * half + l];
                const uint8_t ql1 = block.ql[64 * half + 32 + l];
                const uint8_t qh = block.qh[32 * half + l];
                const int values[4] = {
                    int((ql0 & 15) | (((qh >> 0) & 3) << 4)) - 32,
                    int((ql1 & 15) | (((qh >> 2) & 3) << 4)) - 32,
                    int((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32,
                    int((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32,
                };
                for (int quadrant = 0; quadrant < 4; ++quadrant) {
                    const int subgroup = 4 * half + quadrant;
                    const int scale_index = 8 * half + 2 * quadrant + l / 16;
                    output[block_id * 256 + subgroup * 32 + l] =
                        d * float(block.scales[scale_index]) * float(values[quadrant]);
                }
            }
        }
    }
}

}  // namespace insignia::glm53
