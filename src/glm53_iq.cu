#include "insignia_glm53_iq.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

__device__ __forceinline__ uint32_t unpack_iq_signs(uint32_t sign7) {
    const uint32_t parity = __popc(sign7) & 1u;
    return (sign7 ^ (parity << 7)) * 0x01010101u;
}

__device__ __forceinline__ uint32_t apply_iq3_signs(
    uint32_t values, uint32_t signs, uint32_t selectors) {
    const uint32_t negative = __vcmpne4(signs & selectors, 0);
    return __vsub4(values ^ negative, negative);
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

template <int R, int BLOCKS, bool SHARED_GRID, bool REPACKED>
__global__ __launch_bounds__(256, 2) void iq3_xxs_rows_kernel(
    const uint8_t *__restrict__ weights,
    const uint32_t *__restrict__ xq,
    const float *__restrict__ xscale,
    int words_per_row,
    float *__restrict__ y,
    IQRowOut out) {
    __shared__ uint32_t shared_grid[SHARED_GRID ? 256 : 1];
    if constexpr (SHARED_GRID) {
        shared_grid[threadIdx.x] = kIQ3GridDevice[threadIdx.x];
        __syncthreads();
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int cohort = lane >> 3;
    const int subgroup = lane & 7;
    const int row = blockIdx.x * 8 + warp;
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
        if constexpr (R <= 2) {
            // The scalar/small-block decode path is occupancy-sensitive on Ada.
            // Consume each signed codebook pair immediately so the compiler
            // keeps two decoded words instead of the whole eight-word block.
            // DP4A order within every row is unchanged.
            int dots[R] = {};
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const uint32_t pair_indices = pair < 2 ? indices0 : indices1;
                const int shift = 8 * (2 * (pair & 1));
                const uint32_t code0 = (pair_indices >> shift) & 255u;
                const uint32_t code1 = (pair_indices >> (shift + 8)) & 255u;
                const uint32_t grid0 = SHARED_GRID ? shared_grid[code0]
                                                   : __ldg(kIQ3GridDevice + code0);
                const uint32_t grid1 = SHARED_GRID ? shared_grid[code1]
                                                   : __ldg(kIQ3GridDevice + code1);
                const uint32_t signs = unpack_iq_signs((aux >> (7 * pair)) & 127u);
                const uint32_t decoded0 =
                    apply_iq3_signs(grid0, signs, 0x08040201u);
                const uint32_t decoded1 =
                    apply_iq3_signs(grid1, signs, 0x80402010u);
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
                const int shift = 8 * (2 * (pair & 1));
                const uint32_t code0 = (pair_indices >> shift) & 255u;
                const uint32_t code1 = (pair_indices >> (shift + 8)) & 255u;
                const uint32_t grid0 = SHARED_GRID ? shared_grid[code0]
                                                   : __ldg(kIQ3GridDevice + code0);
                const uint32_t grid1 = SHARED_GRID ? shared_grid[code1]
                                                   : __ldg(kIQ3GridDevice + code1);
                const uint32_t signs = unpack_iq_signs((aux >> (7 * pair)) & 127u);
                decoded[2 * pair + 0] =
                    apply_iq3_signs(grid0, signs, 0x08040201u);
                decoded[2 * pair + 1] =
                    apply_iq3_signs(grid1, signs, 0x80402010u);
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
            y[static_cast<size_t>(out.ids[r]) * gridDim.x * 8 + row] = sums[r];
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

template <int R, int BLOCKS, bool SHARED_GRID, bool REPACKED>
cudaError_t launch_iq3(const uint8_t *weights, const uint32_t *xq,
                       const float *xscale, int words_per_row, float *y,
                       IQRowOut out, int rows, cudaStream_t stream) {
    iq3_xxs_rows_kernel<R, BLOCKS, SHARED_GRID, REPACKED>
        <<<rows / 8, 256, 0, stream>>>(weights, xq, xscale,
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

bool valid_geometry(int rows, int cols, int count) {
    return rows > 0 && (rows & 7) == 0 && (cols == 2048 || cols == 4096) &&
           count > 0 && count <= kIQMaxRows;
}

bool iq3_shared_grid_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("INSIGNIA_GLM53_IQ3_SHARED_GRID");
        return value && value[0] != '0';
    }();
    return enabled;
}

template <bool SHARED_GRID, bool REPACKED>
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
            ? launch_iq3<R, 16, SHARED_GRID, REPACKED>(                            \
                  weights, xq, xscale, int(aligned / 4), y, out, rows, stream)     \
            : launch_iq3<R, 8, SHARED_GRID, REPACKED>(                             \
                  weights, xq, xscale, int(aligned / 4), y, out, rows, stream)
    switch (count) {
        INSIGNIA_IQ3_CASE(1); INSIGNIA_IQ3_CASE(2);
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

}  // namespace

namespace insignia::glm53 {

size_t iq_workspace_rows_bytes(int cols, int count) {
    if ((cols != 2048 && cols != 4096) || count <= 0 || count > kIQMaxRows)
        return 0;
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    return static_cast<size_t>(count) *
           (aligned + static_cast<size_t>(cols / kIQActivationGroup) * sizeof(float));
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
    if (iq3_shared_grid_enabled())
        return dispatch_iq3<true, false>(weights, workspace, count, y, y_ids,
                                         rows, cols, stream);
    return dispatch_iq3<false, false>(weights, workspace, count, y, y_ids,
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
    if (iq3_shared_grid_enabled())
        return dispatch_iq3<true, true>(weights, workspace, count, y, y_ids,
                                        rows, cols, stream);
    return dispatch_iq3<false, true>(weights, workspace, count, y, y_ids,
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

}  // namespace insignia::glm53
