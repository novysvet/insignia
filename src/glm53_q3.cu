#include "insignia_glm53_q3.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

using insignia::glm53::kQ3KBlockBytes;
using insignia::glm53::kQ3KBlockWeights;
using insignia::glm53::kQ3KMaxRows;

struct alignas(2) Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    __half d;
};
static_assert(sizeof(Q3KBlock) == kQ3KBlockBytes);

struct Q3KRowIds {
    int ids[kQ3KMaxRows]{};
};

struct Q3KRowOut {
    int ids[kQ3KMaxRows]{};
    float weights[kQ3KMaxRows]{};
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

__device__ __forceinline__ int q3k_scale(const Q3KBlock &block, int group) {
    const int low = (block.scales[group & 7] >> (4 * (group >> 3))) & 15;
    const int high =
        (block.scales[8 + (group & 3)] >> (2 * (group >> 2))) & 3;
    return (low | (high << 4)) - 32;
}

__device__ __forceinline__ uint32_t q3k_values4(
    const Q3KBlock &block, int group, int word) {
    const int q_offset = (group >= 8 ? 32 : 0) + ((group & 1) ? 16 : 0);
    const int h_offset = (group & 1) ? 16 : 0;
    const int shift = 2 * ((group >> 1) & 3);
    const int bit = group >> 1;
    const uint32_t q = load_u32_any(block.qs + q_offset + 4 * word);
    const uint32_t h = load_u32_any(block.hmask + h_offset + 4 * word);
    const uint32_t low = (q >> shift) & 0x03030303u;
    const uint32_t subtract = ((~h >> bit) << 2) & 0x04040404u;
    return __vsubss4(low, subtract);
}

template <int R>
__global__ __launch_bounds__(256, 2) void q3k_quantize_x16_rows_kernel(
    const float *__restrict__ x,
    uint32_t *__restrict__ xq,
    float *__restrict__ xscale,
    int groups,
    int words_per_row,
    Q3KRowIds rows) {
    const int group = threadIdx.x;
    if (group >= groups) return;
    const float4 *source = reinterpret_cast<const float4 *>(
        x + static_cast<size_t>(rows.ids[blockIdx.x]) * groups * 16 + group * 16);
    uint32_t *row_q = xq + static_cast<size_t>(blockIdx.x) * words_per_row;
    float *row_scale = xscale + static_cast<size_t>(blockIdx.x) * groups;
    float values[16];
    float maximum = 0.0f;
#pragma unroll
    for (int word = 0; word < 4; ++word) {
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
    for (int word = 0; word < 4; ++word) {
        uint32_t packed = 0;
#pragma unroll
        for (int byte = 0; byte < 4; ++byte)
            packed |= uint32_t(uint8_t(__float2int_rn(values[4 * word + byte] * inverse)))
                      << (8 * byte);
        row_q[group * 4 + word] = packed;
    }
}

template <int R>
__global__ __launch_bounds__(256, 2) void q3k_quantize_swiglu_x16_rows_kernel(
    const float *__restrict__ gate,
    const float *__restrict__ up,
    uint32_t *__restrict__ xq,
    float *__restrict__ xscale,
    int groups,
    int words_per_row,
    Q3KRowIds rows) {
    const int group = threadIdx.x;
    if (group >= groups) return;
    const size_t base = static_cast<size_t>(rows.ids[blockIdx.x]) * groups * 16;
    const float4 *gate4 = reinterpret_cast<const float4 *>(gate + base + group * 16);
    const float4 *up4 = reinterpret_cast<const float4 *>(up + base + group * 16);
    uint32_t *row_q = xq + static_cast<size_t>(blockIdx.x) * words_per_row;
    float *row_scale = xscale + static_cast<size_t>(blockIdx.x) * groups;
    float values[16];
    float maximum = 0.0f;
#pragma unroll
    for (int word = 0; word < 4; ++word) {
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
    for (int word = 0; word < 4; ++word) {
        uint32_t packed = 0;
#pragma unroll
        for (int byte = 0; byte < 4; ++byte)
            packed |= uint32_t(uint8_t(__float2int_rn(values[4 * word + byte] * inverse)))
                      << (8 * byte);
        row_q[group * 4 + word] = packed;
    }
}

template <int BLOCKS>
__global__ __launch_bounds__(256, 2) void q3k_f32_kernel(
    const Q3KBlock *__restrict__ weights,
    const float *__restrict__ x,
    float *__restrict__ y) {
    const int lane = threadIdx.x & 31;
    const int group = lane & 15;
    const int half = lane >> 4;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 16 + warp * 2 + half;
    const Q3KBlock *row_weights = weights + static_cast<size_t>(row) * BLOCKS;
    float sum = 0.0f;
#pragma unroll
    for (int block_id = 0; block_id < BLOCKS; ++block_id) {
        const Q3KBlock &block = row_weights[block_id];
        const float base = __half2float(block.d) * float(q3k_scale(block, group));
        const float *xg = x + block_id * kQ3KBlockWeights + group * 16;
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            const uint32_t values = q3k_values4(block, group, word);
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const int value = int(int8_t(values >> (8 * byte)));
                sum = fmaf(base * float(value), __ldg(xg + 4 * word + byte), sum);
            }
        }
    }
#pragma unroll
    for (int offset = 8; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset, 16);
    if (!group) y[row] = sum;
}

template <int R, int BLOCKS, bool ACCUMULATE>
__global__ __launch_bounds__(256, 2) void q3k_dp4a_rows_kernel(
    const Q3KBlock *__restrict__ weights,
    const uint32_t *__restrict__ xq,
    const float *__restrict__ xscale,
    int words_per_row,
    float *__restrict__ y,
    Q3KRowOut out) {
    const int lane = threadIdx.x & 31;
    const int group = lane & 15;
    const int half = lane >> 4;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 16 + warp * 2 + half;
    const Q3KBlock *row_weights = weights + static_cast<size_t>(row) * BLOCKS;
    float sums[R] = {};
#pragma unroll
    for (int block_id = 0; block_id < BLOCKS; ++block_id) {
        const Q3KBlock &block = row_weights[block_id];
        const float weight_scale =
            __half2float(block.d) * float(q3k_scale(block, group));
        uint32_t values[4];
#pragma unroll
        for (int word = 0; word < 4; ++word)
            values[word] = q3k_values4(block, group, word);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int activation_group = block_id * 16 + group;
            const uint32_t *xg = xq + static_cast<size_t>(r) * words_per_row +
                                 activation_group * 4;
            int dot = __dp4a(int(values[0]), int(__ldg(xg + 0)), 0);
            dot = __dp4a(int(values[1]), int(__ldg(xg + 1)), dot);
            dot = __dp4a(int(values[2]), int(__ldg(xg + 2)), dot);
            dot = __dp4a(int(values[3]), int(__ldg(xg + 3)), dot);
            sums[r] = fmaf(float(dot),
                           weight_scale * xscale[static_cast<size_t>(r) * BLOCKS * 16 +
                                                 activation_group],
                           sums[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int offset = 8; offset; offset >>= 1)
            sums[r] += __shfl_down_sync(0xffffffffu, sums[r], offset, 16);
        if (!group) {
            float *destination = y + static_cast<size_t>(out.ids[r]) * gridDim.x * 16 + row;
            if constexpr (ACCUMULATE)
                *destination = fmaf(sums[r], out.weights[r], *destination);
            else
                *destination = sums[r];
        }
    }
}

template <int R, int BLOCKS>
__global__ __launch_bounds__(256, 1) void q3k_dp4a_pair_rows_kernel(
    const Q3KBlock *__restrict__ weights_a,
    const Q3KBlock *__restrict__ weights_b,
    const uint32_t *__restrict__ xq,
    const float *__restrict__ xscale,
    int words_per_row,
    float *__restrict__ y_a,
    float *__restrict__ y_b,
    Q3KRowIds out) {
    const int lane = threadIdx.x & 31;
    const int group = lane & 15;
    const int half = lane >> 4;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 16 + warp * 2 + half;
    const Q3KBlock *row_a = weights_a + static_cast<size_t>(row) * BLOCKS;
    const Q3KBlock *row_b = weights_b + static_cast<size_t>(row) * BLOCKS;
    float sums_a[R] = {};
    float sums_b[R] = {};
#pragma unroll
    for (int block_id = 0; block_id < BLOCKS; ++block_id) {
        const Q3KBlock &block_a = row_a[block_id];
        const Q3KBlock &block_b = row_b[block_id];
        const float scale_a =
            __half2float(block_a.d) * float(q3k_scale(block_a, group));
        const float scale_b =
            __half2float(block_b.d) * float(q3k_scale(block_b, group));
        uint32_t values_a[4], values_b[4];
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            values_a[word] = q3k_values4(block_a, group, word);
            values_b[word] = q3k_values4(block_b, group, word);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int activation_group = block_id * 16 + group;
            const uint32_t *xg = xq + static_cast<size_t>(r) * words_per_row +
                                 activation_group * 4;
            const uint32_t x0 = __ldg(xg + 0);
            const uint32_t x1 = __ldg(xg + 1);
            const uint32_t x2 = __ldg(xg + 2);
            const uint32_t x3 = __ldg(xg + 3);
            int dot_a = __dp4a(int(values_a[0]), int(x0), 0);
            dot_a = __dp4a(int(values_a[1]), int(x1), dot_a);
            dot_a = __dp4a(int(values_a[2]), int(x2), dot_a);
            dot_a = __dp4a(int(values_a[3]), int(x3), dot_a);
            int dot_b = __dp4a(int(values_b[0]), int(x0), 0);
            dot_b = __dp4a(int(values_b[1]), int(x1), dot_b);
            dot_b = __dp4a(int(values_b[2]), int(x2), dot_b);
            dot_b = __dp4a(int(values_b[3]), int(x3), dot_b);
            const float activation_scale =
                xscale[static_cast<size_t>(r) * BLOCKS * 16 + activation_group];
            sums_a[r] = fmaf(float(dot_a), scale_a * activation_scale, sums_a[r]);
            sums_b[r] = fmaf(float(dot_b), scale_b * activation_scale, sums_b[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int offset = 8; offset; offset >>= 1) {
            sums_a[r] += __shfl_down_sync(0xffffffffu, sums_a[r], offset, 16);
            sums_b[r] += __shfl_down_sync(0xffffffffu, sums_b[r], offset, 16);
        }
        if (!group) {
            const size_t index = static_cast<size_t>(out.ids[r]) * gridDim.x * 16 + row;
            y_a[index] = sums_a[r];
            y_b[index] = sums_b[r];
        }
    }
}

template <int R, int BLOCKS, bool ACCUMULATE>
cudaError_t launch_q3k(
    const uint8_t *weights,
    const uint32_t *xq,
    const float *xscale,
    int words_per_row,
    float *y,
    Q3KRowOut out,
    int rows,
    cudaStream_t stream) {
    q3k_dp4a_rows_kernel<R, BLOCKS, ACCUMULATE><<<rows / 16, 256, 0, stream>>>(
        reinterpret_cast<const Q3KBlock *>(weights), xq, xscale,
        words_per_row, y, out);
    return cudaPeekAtLastError();
}

template <int R, int BLOCKS>
cudaError_t launch_q3k_pair(
    const uint8_t *weights_a,
    const uint8_t *weights_b,
    const uint32_t *xq,
    const float *xscale,
    int words_per_row,
    float *y_a,
    float *y_b,
    Q3KRowIds out,
    int rows,
    cudaStream_t stream) {
    q3k_dp4a_pair_rows_kernel<R, BLOCKS><<<rows / 16, 256, 0, stream>>>(
        reinterpret_cast<const Q3KBlock *>(weights_a),
        reinterpret_cast<const Q3KBlock *>(weights_b),
        xq, xscale, words_per_row, y_a, y_b, out);
    return cudaPeekAtLastError();
}

template <bool ACCUMULATE>
cudaError_t dispatch_q3k(
    const uint8_t *weights,
    const void *workspace,
    int count,
    float *y,
    const int *y_ids,
    const float *combine,
    int rows,
    int cols,
    cudaStream_t stream) {
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    const auto *xq = reinterpret_cast<const uint32_t *>(workspace);
    const auto *xscale = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(workspace) + count * aligned);
    Q3KRowOut out{};
    for (int r = 0; r < count; ++r) {
        out.ids[r] = y_ids[r];
        if constexpr (ACCUMULATE) out.weights[r] = combine[r];
    }
#define INSIGNIA_Q3_CASE(R)                                                        \
    case R:                                                                        \
        return cols == 4096                                                        \
            ? launch_q3k<R, 16, ACCUMULATE>(weights, xq, xscale, int(aligned / 4), \
                                             y, out, rows, stream)                  \
            : launch_q3k<R, 8, ACCUMULATE>(weights, xq, xscale, int(aligned / 4),  \
                                            y, out, rows, stream)
    switch (count) {
        INSIGNIA_Q3_CASE(1);
        INSIGNIA_Q3_CASE(2);
        INSIGNIA_Q3_CASE(3);
        INSIGNIA_Q3_CASE(4);
        INSIGNIA_Q3_CASE(5);
        INSIGNIA_Q3_CASE(6);
        INSIGNIA_Q3_CASE(7);
        INSIGNIA_Q3_CASE(8);
        default: return cudaErrorInvalidValue;
    }
#undef INSIGNIA_Q3_CASE
}

cudaError_t dispatch_q3k_pair(
    const uint8_t *weights_a,
    const uint8_t *weights_b,
    const void *workspace,
    int count,
    float *y_a,
    float *y_b,
    const int *y_ids,
    int rows,
    int cols,
    cudaStream_t stream) {
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    const auto *xq = reinterpret_cast<const uint32_t *>(workspace);
    const auto *xscale = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(workspace) + count * aligned);
    Q3KRowIds out{};
    for (int r = 0; r < count; ++r) out.ids[r] = y_ids[r];
#define INSIGNIA_Q3_PAIR_CASE(R)                                                   \
    case R:                                                                        \
        return cols == 4096                                                        \
            ? launch_q3k_pair<R, 16>(weights_a, weights_b, xq, xscale,             \
                                      int(aligned / 4), y_a, y_b, out, rows, stream)\
            : launch_q3k_pair<R, 8>(weights_a, weights_b, xq, xscale,              \
                                     int(aligned / 4), y_a, y_b, out, rows, stream)
    switch (count) {
        INSIGNIA_Q3_PAIR_CASE(1);
        INSIGNIA_Q3_PAIR_CASE(2);
        INSIGNIA_Q3_PAIR_CASE(3);
        INSIGNIA_Q3_PAIR_CASE(4);
        INSIGNIA_Q3_PAIR_CASE(5);
        INSIGNIA_Q3_PAIR_CASE(6);
        INSIGNIA_Q3_PAIR_CASE(7);
        INSIGNIA_Q3_PAIR_CASE(8);
        default: return cudaErrorInvalidValue;
    }
#undef INSIGNIA_Q3_PAIR_CASE
}

bool valid_geometry(int rows, int cols, int count) {
    return rows > 0 && (rows & 15) == 0 && (cols == 2048 || cols == 4096) &&
           count > 0 && count <= kQ3KMaxRows;
}

}  // namespace

namespace insignia::glm53 {

size_t q3k_workspace_rows_bytes(int cols, int count) {
    if ((cols != 2048 && cols != 4096) || count <= 0 || count > kQ3KMaxRows)
        return 0;
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    return static_cast<size_t>(count) *
           (aligned + static_cast<size_t>(cols / kQ3KSubgroup) * sizeof(float));
}

cudaError_t q3k_quantize_activation_rows(
    const float *x, int cols, const int *row_ids, int count, void *workspace,
    cudaStream_t stream) {
    if (!x || !row_ids || !workspace || !valid_geometry(16, cols, count))
        return cudaErrorInvalidValue;
    const int groups = cols / kQ3KSubgroup;
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    auto *xq = reinterpret_cast<uint32_t *>(workspace);
    auto *xscale = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(workspace) +
                                             count * aligned);
    Q3KRowIds rows{};
    for (int r = 0; r < count; ++r) rows.ids[r] = row_ids[r];
#define INSIGNIA_Q3_QUANT_CASE(R)                                                  \
    case R:                                                                        \
        q3k_quantize_x16_rows_kernel<R><<<R, 256, 0, stream>>>(                    \
            x, xq, xscale, groups, int(aligned / 4), rows);                        \
        return cudaPeekAtLastError()
    switch (count) {
        INSIGNIA_Q3_QUANT_CASE(1);
        INSIGNIA_Q3_QUANT_CASE(2);
        INSIGNIA_Q3_QUANT_CASE(3);
        INSIGNIA_Q3_QUANT_CASE(4);
        INSIGNIA_Q3_QUANT_CASE(5);
        INSIGNIA_Q3_QUANT_CASE(6);
        INSIGNIA_Q3_QUANT_CASE(7);
        INSIGNIA_Q3_QUANT_CASE(8);
        default: return cudaErrorInvalidValue;
    }
#undef INSIGNIA_Q3_QUANT_CASE
}

cudaError_t q3k_quantize_swiglu_rows(
    const float *gate, const float *up, int cols, const int *row_ids, int count,
    void *workspace, cudaStream_t stream) {
    if (!gate || !up || !row_ids || !workspace || !valid_geometry(16, cols, count))
        return cudaErrorInvalidValue;
    const int groups = cols / kQ3KSubgroup;
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    auto *xq = reinterpret_cast<uint32_t *>(workspace);
    auto *xscale = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(workspace) +
                                             count * aligned);
    Q3KRowIds rows{};
    for (int r = 0; r < count; ++r) rows.ids[r] = row_ids[r];
#define INSIGNIA_Q3_SWIGLU_CASE(R)                                                 \
    case R:                                                                        \
        q3k_quantize_swiglu_x16_rows_kernel<R><<<R, 256, 0, stream>>>(             \
            gate, up, xq, xscale, groups, int(aligned / 4), rows);                 \
        return cudaPeekAtLastError()
    switch (count) {
        INSIGNIA_Q3_SWIGLU_CASE(1);
        INSIGNIA_Q3_SWIGLU_CASE(2);
        INSIGNIA_Q3_SWIGLU_CASE(3);
        INSIGNIA_Q3_SWIGLU_CASE(4);
        INSIGNIA_Q3_SWIGLU_CASE(5);
        INSIGNIA_Q3_SWIGLU_CASE(6);
        INSIGNIA_Q3_SWIGLU_CASE(7);
        INSIGNIA_Q3_SWIGLU_CASE(8);
        default: return cudaErrorInvalidValue;
    }
#undef INSIGNIA_Q3_SWIGLU_CASE
}

cudaError_t q3k_gemv_f32(
    const uint8_t *weights, const float *x, float *y, int rows, int cols,
    cudaStream_t stream) {
    if (!weights || !x || !y || !valid_geometry(rows, cols, 1))
        return cudaErrorInvalidValue;
    if (cols == 4096)
        q3k_f32_kernel<16><<<rows / 16, 256, 0, stream>>>(
            reinterpret_cast<const Q3KBlock *>(weights), x, y);
    else
        q3k_f32_kernel<8><<<rows / 16, 256, 0, stream>>>(
            reinterpret_cast<const Q3KBlock *>(weights), x, y);
    return cudaPeekAtLastError();
}

cudaError_t q3k_gemv_quantized_rows(
    const uint8_t *weights, const void *workspace, int count, float *y,
    const int *y_ids, int rows, int cols, cudaStream_t stream) {
    if (!weights || !workspace || !y || !y_ids || !valid_geometry(rows, cols, count))
        return cudaErrorInvalidValue;
    return dispatch_q3k<false>(weights, workspace, count, y, y_ids, nullptr,
                               rows, cols, stream);
}

cudaError_t q3k_gemv2_quantized_rows(
    const uint8_t *weights_a, const uint8_t *weights_b, const void *workspace,
    int count, float *y_a, float *y_b, const int *y_ids, int rows, int cols,
    cudaStream_t stream) {
    if (!weights_a || !weights_b || !workspace || !y_a || !y_b || !y_ids ||
        !valid_geometry(rows, cols, count))
        return cudaErrorInvalidValue;
    return dispatch_q3k_pair(weights_a, weights_b, workspace, count, y_a, y_b,
                             y_ids, rows, cols, stream);
}

cudaError_t q3k_gemv_acc_quantized_rows(
    const uint8_t *weights, const void *workspace, int count, float *y,
    const int *y_ids, const float *combine, int rows, int cols,
    cudaStream_t stream) {
    if (!weights || !workspace || !y || !y_ids || !combine ||
        !valid_geometry(rows, cols, count))
        return cudaErrorInvalidValue;
    return dispatch_q3k<true>(weights, workspace, count, y, y_ids, combine,
                              rows, cols, stream);
}

}  // namespace insignia::glm53
