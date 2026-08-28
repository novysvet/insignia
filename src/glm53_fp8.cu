#include "insignia_glm53_fp8.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace insignia::glm53 {
namespace {

__global__ __launch_bounds__(256, 1) void quantize_fp8_x64_kernel(
    const float *__restrict__ x,
    uint8_t *__restrict__ quantized,
    float *__restrict__ scales,
    int groups,
    int tokens,
    int quantized_stride) {
    const int token = blockIdx.y;
    if (token >= tokens) return;
    x += size_t(token) * groups * kFp8GroupSize;
    quantized += size_t(token) * quantized_stride;
    scales += size_t(token) * groups;
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int group = thread >> 2;
    if (group >= groups) return;
    const int quarter = thread & 3;
    const float4 *source = reinterpret_cast<const float4 *>(
        x + group * kFp8GroupSize + quarter * 16);
    float values[16];
    float maximum = 0.0f;
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        const float4 value = __ldg(source + word);
        values[word * 4 + 0] = value.x;
        values[word * 4 + 1] = value.y;
        values[word * 4 + 2] = value.z;
        values[word * 4 + 3] = value.w;
        maximum = fmaxf(maximum, fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                                       fmaxf(fabsf(value.z), fabsf(value.w))));
    }
    const unsigned active = __activemask();
    maximum = fmaxf(maximum, __shfl_xor_sync(active, maximum, 1));
    maximum = fmaxf(maximum, __shfl_xor_sync(active, maximum, 2));
    const float inverse = maximum > 0.0f ? 448.0f / maximum : 0.0f;
    if (!quarter) scales[group] = maximum * (1.0f / 448.0f);
#pragma unroll
    for (int element = 0; element < 16; ++element)
        quantized[group * kFp8GroupSize + quarter * 16 + element] =
            __nv_fp8_e4m3(values[element] * inverse).__x;
}

__device__ __forceinline__ void mma_e4m3(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int kSplits>
__global__ __launch_bounds__(32 * kSplits, 1) void fp8_tc_gemv_kernel(
    const uint8_t *__restrict__ weights,
    const __half *__restrict__ weight_scales,
    const uint8_t *__restrict__ x,
    const float *__restrict__ x_scales,
    float *__restrict__ y,
    int rows,
    int cols,
    int groups) {
    extern __shared__ __align__(16) uint8_t shared[];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int group_id = lane >> 2;
    const int thread_in_group = lane & 3;
    const int first_row = blockIdx.x * 16;
    uint8_t *tile = shared + warp * 16 * kFp8GroupSize;
    float result0 = 0.0f, result1 = 0.0f;

    for (int group = warp; group < groups; group += kSplits) {
        // 64 vector loads cover a 16x64 tile. Four consecutive lanes form one
        // aligned 64-byte transaction from a checkpoint row.
        for (int chunk = lane; chunk < 64; chunk += 32) {
            const int local_row = chunk >> 2;
            const int segment = chunk & 3;
            uint4 value{};
            if (first_row + local_row < rows)
                value = __ldcs(reinterpret_cast<const uint4 *>(
                    weights + size_t(first_row + local_row) * cols +
                    group * kFp8GroupSize + segment * 16));
            reinterpret_cast<uint4 *>(tile)[chunk] = value;
        }
        __syncwarp();

        const int local_row0 = group_id;
        const int local_row1 = group_id + 8;
        const int k0 = thread_in_group * 4;
        const uint8_t *row0 = tile + local_row0 * kFp8GroupSize;
        const uint8_t *row1 = tile + local_row1 * kFp8GroupSize;
        const uint8_t *activation = x + group * kFp8GroupSize;
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        mma_e4m3(d0, d1, d2, d3,
                 *reinterpret_cast<const uint32_t *>(row0 + k0),
                 *reinterpret_cast<const uint32_t *>(row1 + k0),
                 *reinterpret_cast<const uint32_t *>(row0 + k0 + 16),
                 *reinterpret_cast<const uint32_t *>(row1 + k0 + 16),
                 *reinterpret_cast<const uint32_t *>(activation + k0),
                 *reinterpret_cast<const uint32_t *>(activation + k0 + 16));
        mma_e4m3(d0, d1, d2, d3,
                 *reinterpret_cast<const uint32_t *>(row0 + k0 + 32),
                 *reinterpret_cast<const uint32_t *>(row1 + k0 + 32),
                 *reinterpret_cast<const uint32_t *>(row0 + k0 + 48),
                 *reinterpret_cast<const uint32_t *>(row1 + k0 + 48),
                 *reinterpret_cast<const uint32_t *>(activation + k0 + 32),
                 *reinterpret_cast<const uint32_t *>(activation + k0 + 48));
        if (!thread_in_group) {
            const float activation_scale = x_scales[group];
            const int global_row0 = first_row + local_row0;
            const int global_row1 = first_row + local_row1;
            if (global_row0 < rows)
                result0 = fmaf(d0, __half2float(weight_scales[
                    size_t(global_row0) * groups + group]) * activation_scale, result0);
            if (global_row1 < rows)
                result1 = fmaf(d2, __half2float(weight_scales[
                    size_t(global_row1) * groups + group]) * activation_scale, result1);
        }
        __syncwarp();
    }
    if constexpr (kSplits == 1) {
        if (!thread_in_group) {
            const int row0 = first_row + group_id;
            const int row1 = row0 + 8;
            if (row0 < rows) y[row0] = result0;
            if (row1 < rows) y[row1] = result1;
        }
    } else {
        float *partials = reinterpret_cast<float *>(
            shared + kSplits * 16 * kFp8GroupSize);
        if (!thread_in_group) {
            partials[warp * 16 + group_id] = result0;
            partials[warp * 16 + group_id + 8] = result1;
        }
        __syncthreads();
        if (!warp && lane < 16 && first_row + lane < rows) {
            float sum = 0.0f;
#pragma unroll
            for (int split = 0; split < kSplits; ++split)
                sum += partials[split * 16 + lane];
            y[first_row + lane] = sum;
        }
    }
}

// Paired GEMV: blockIdx.y selects W_a (0) or W_b (1). The pointer select is
// a one-time register move before the tile loop; the body below is the
// single-matrix kernel verbatim, so each row's accumulation order is
// unchanged and outputs stay bitwise-identical to two separate calls.
template <int kSplits>
__global__ __launch_bounds__(32 * kSplits, 1) void fp8_tc_gemv2_kernel(
    const uint8_t *__restrict__ weights_a,
    const __half *__restrict__ weight_scales_a,
    const uint8_t *__restrict__ weights_b,
    const __half *__restrict__ weight_scales_b,
    const uint8_t *__restrict__ x,
    const float *__restrict__ x_scales,
    float *__restrict__ y_a,
    float *__restrict__ y_b,
    int rows,
    int cols,
    int groups) {
    extern __shared__ __align__(16) uint8_t shared[];
    const uint8_t *__restrict__ weights = blockIdx.y ? weights_b : weights_a;
    const __half *__restrict__ weight_scales =
        blockIdx.y ? weight_scales_b : weight_scales_a;
    float *__restrict__ y = blockIdx.y ? y_b : y_a;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int group_id = lane >> 2;
    const int thread_in_group = lane & 3;
    const int first_row = blockIdx.x * 16;
    uint8_t *tile = shared + warp * 16 * kFp8GroupSize;
    float result0 = 0.0f, result1 = 0.0f;

    for (int group = warp; group < groups; group += kSplits) {
        for (int chunk = lane; chunk < 64; chunk += 32) {
            const int local_row = chunk >> 2;
            const int segment = chunk & 3;
            uint4 value{};
            if (first_row + local_row < rows)
                value = __ldcs(reinterpret_cast<const uint4 *>(
                    weights + size_t(first_row + local_row) * cols +
                    group * kFp8GroupSize + segment * 16));
            reinterpret_cast<uint4 *>(tile)[chunk] = value;
        }
        __syncwarp();

        const int local_row0 = group_id;
        const int local_row1 = group_id + 8;
        const int k0 = thread_in_group * 4;
        const uint8_t *row0 = tile + local_row0 * kFp8GroupSize;
        const uint8_t *row1 = tile + local_row1 * kFp8GroupSize;
        const uint8_t *activation = x + group * kFp8GroupSize;
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        mma_e4m3(d0, d1, d2, d3,
                 *reinterpret_cast<const uint32_t *>(row0 + k0),
                 *reinterpret_cast<const uint32_t *>(row1 + k0),
                 *reinterpret_cast<const uint32_t *>(row0 + k0 + 16),
                 *reinterpret_cast<const uint32_t *>(row1 + k0 + 16),
                 *reinterpret_cast<const uint32_t *>(activation + k0),
                 *reinterpret_cast<const uint32_t *>(activation + k0 + 16));
        mma_e4m3(d0, d1, d2, d3,
                 *reinterpret_cast<const uint32_t *>(row0 + k0 + 32),
                 *reinterpret_cast<const uint32_t *>(row1 + k0 + 32),
                 *reinterpret_cast<const uint32_t *>(row0 + k0 + 48),
                 *reinterpret_cast<const uint32_t *>(row1 + k0 + 48),
                 *reinterpret_cast<const uint32_t *>(activation + k0 + 32),
                 *reinterpret_cast<const uint32_t *>(activation + k0 + 48));
        if (!thread_in_group) {
            const float activation_scale = x_scales[group];
            const int global_row0 = first_row + local_row0;
            const int global_row1 = first_row + local_row1;
            if (global_row0 < rows)
                result0 = fmaf(d0, __half2float(weight_scales[
                    size_t(global_row0) * groups + group]) * activation_scale, result0);
            if (global_row1 < rows)
                result1 = fmaf(d2, __half2float(weight_scales[
                    size_t(global_row1) * groups + group]) * activation_scale, result1);
        }
        __syncwarp();
    }
    if constexpr (kSplits == 1) {
        if (!thread_in_group) {
            const int row0 = first_row + group_id;
            const int row1 = row0 + 8;
            if (row0 < rows) y[row0] = result0;
            if (row1 < rows) y[row1] = result1;
        }
    } else {
        float *partials = reinterpret_cast<float *>(
            shared + kSplits * 16 * kFp8GroupSize);
        if (!thread_in_group) {
            partials[warp * 16 + group_id] = result0;
            partials[warp * 16 + group_id + 8] = result1;
        }
        __syncthreads();
        if (!warp && lane < 16 && first_row + lane < rows) {
            float sum = 0.0f;
#pragma unroll
            for (int split = 0; split < kSplits; ++split)
                sum += partials[split * 16 + lane];
            y[first_row + lane] = sum;
        }
    }
}

template <int kSplits, int kTokens>
__global__ __launch_bounds__(32 * kSplits, 1) void fp8_tc_gemv_batch_kernel(
    const uint8_t *__restrict__ weights,
    const __half *__restrict__ weight_scales,
    const uint8_t *__restrict__ x,
    const float *__restrict__ x_scales,
    float *__restrict__ y,
    int tokens,
    int rows,
    int cols,
    int groups,
    int x_stride,
    int y_stride) {
    extern __shared__ __align__(16) uint8_t shared[];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int group_id = lane >> 2;
    const int thread_in_group = lane & 3;
    const int first_row = blockIdx.x * 16;
    uint8_t *tile = shared + warp * 16 * kFp8GroupSize;
    float result0[kTokens] = {};
    float result1[kTokens] = {};

    for (int group = warp; group < groups; group += kSplits) {
        for (int chunk = lane; chunk < 64; chunk += 32) {
            const int local_row = chunk >> 2;
            const int segment = chunk & 3;
            uint4 value{};
            if (first_row + local_row < rows)
                value = __ldcs(reinterpret_cast<const uint4 *>(
                    weights + size_t(first_row + local_row) * cols +
                    group * kFp8GroupSize + segment * 16));
            reinterpret_cast<uint4 *>(tile)[chunk] = value;
        }
        __syncwarp();

        const int local_row0 = group_id;
        const int local_row1 = group_id + 8;
        const int k0 = thread_in_group * 4;
        const uint8_t *row0 = tile + local_row0 * kFp8GroupSize;
        const uint8_t *row1 = tile + local_row1 * kFp8GroupSize;
#pragma unroll
        for (int token = 0; token < kTokens; ++token) {
            if (token >= tokens) continue;
            const uint8_t *activation = x + size_t(token) * x_stride + group * kFp8GroupSize;
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            mma_e4m3(d0, d1, d2, d3,
                     *reinterpret_cast<const uint32_t *>(row0 + k0),
                     *reinterpret_cast<const uint32_t *>(row1 + k0),
                     *reinterpret_cast<const uint32_t *>(row0 + k0 + 16),
                     *reinterpret_cast<const uint32_t *>(row1 + k0 + 16),
                     *reinterpret_cast<const uint32_t *>(activation + k0),
                     *reinterpret_cast<const uint32_t *>(activation + k0 + 16));
            mma_e4m3(d0, d1, d2, d3,
                     *reinterpret_cast<const uint32_t *>(row0 + k0 + 32),
                     *reinterpret_cast<const uint32_t *>(row1 + k0 + 32),
                     *reinterpret_cast<const uint32_t *>(row0 + k0 + 48),
                     *reinterpret_cast<const uint32_t *>(row1 + k0 + 48),
                     *reinterpret_cast<const uint32_t *>(activation + k0 + 32),
                     *reinterpret_cast<const uint32_t *>(activation + k0 + 48));
            if (!thread_in_group) {
                const float activation_scale = x_scales[size_t(token) * groups + group];
                const int global_row0 = first_row + local_row0;
                const int global_row1 = first_row + local_row1;
                if (global_row0 < rows)
                    result0[token] = fmaf(d0, __half2float(weight_scales[
                        size_t(global_row0) * groups + group]) * activation_scale,
                        result0[token]);
                if (global_row1 < rows)
                    result1[token] = fmaf(d2, __half2float(weight_scales[
                        size_t(global_row1) * groups + group]) * activation_scale,
                        result1[token]);
            }
        }
        __syncwarp();
    }
    if constexpr (kSplits == 1) {
        if (!thread_in_group) {
            const int row0 = first_row + group_id;
            const int row1 = row0 + 8;
#pragma unroll
            for (int token = 0; token < kTokens; ++token) {
                if (token >= tokens) continue;
                if (row0 < rows) y[size_t(token) * y_stride + row0] = result0[token];
                if (row1 < rows) y[size_t(token) * y_stride + row1] = result1[token];
            }
        }
    } else {
        float *partials = reinterpret_cast<float *>(
            shared + kSplits * 16 * kFp8GroupSize);
        if (!thread_in_group) {
#pragma unroll
            for (int token = 0; token < kTokens; ++token) {
                if (token >= tokens) continue;
                partials[(warp * kTokens + token) * 16 + group_id] = result0[token];
                partials[(warp * kTokens + token) * 16 + group_id + 8] = result1[token];
            }
        }
        __syncthreads();
        if (!warp && lane < 16 && first_row + lane < rows) {
#pragma unroll
            for (int token = 0; token < kTokens; ++token) {
                if (token >= tokens) continue;
                float sum = 0.0f;
#pragma unroll
                for (int split = 0; split < kSplits; ++split)
                    sum += partials[(split * kTokens + token) * 16 + lane];
                y[size_t(token) * y_stride + first_row + lane] = sum;
            }
        }
    }
}

template <int kTokens>
void launch_fp8_batch(
    const uint8_t *weights,
    const __half *weight_scales,
    const uint8_t *x,
    const float *x_scales,
    float *y,
    int tokens,
    int rows,
    int cols,
    int groups,
    int x_stride,
    int y_stride,
    cudaStream_t stream) {
    if (groups > 64) {
        constexpr int splits = 8;
        constexpr int shared = splits * 16 * kFp8GroupSize +
                               splits * kTokens * 16 * sizeof(float);
        fp8_tc_gemv_batch_kernel<splits, kTokens>
            <<<(rows + 15) / 16, 32 * splits, shared, stream>>>(
                weights, weight_scales, x, x_scales, y, tokens, rows, cols,
                groups, x_stride, y_stride);
    } else {
        fp8_tc_gemv_batch_kernel<1, kTokens>
            <<<(rows + 15) / 16, 32, 16 * kFp8GroupSize, stream>>>(
                weights, weight_scales, x, x_scales, y, tokens, rows, cols,
                groups, x_stride, y_stride);
    }
}

template <int kTokens>
void launch_fp8_batch_pair(
    const uint8_t *weights_a,
    const __half *weight_scales_a,
    const uint8_t *weights_b,
    const __half *weight_scales_b,
    const uint8_t *x,
    const float *x_scales,
    float *y_a,
    float *y_b,
    int tokens,
    int rows,
    int cols,
    int groups,
    int x_stride,
    int y_stride,
    cudaStream_t stream) {
    launch_fp8_batch<kTokens>(weights_a, weight_scales_a, x, x_scales, y_a,
                              tokens, rows, cols, groups, x_stride, y_stride, stream);
    launch_fp8_batch<kTokens>(weights_b, weight_scales_b, x, x_scales, y_b,
                              tokens, rows, cols, groups, x_stride, y_stride, stream);
}

}  // namespace

size_t fp8_workspace_bytes(int cols) {
    return fp8_batch_workspace_bytes(cols, 1);
}

size_t fp8_batch_workspace_bytes(int cols, int tokens) {
    if (cols <= 0 || tokens <= 0 || (cols & (kFp8GroupSize - 1))) return 0;
    const size_t aligned = (size_t(cols) + 255) & ~size_t(255);
    return size_t(tokens) *
           (aligned + size_t(cols / kFp8GroupSize) * sizeof(float));
}

cudaError_t fp8_tc_gemv(
    const uint8_t *weights,
    const uint16_t *scales,
    const float *x,
    float *y,
    int rows,
    int cols,
    void *workspace,
    cudaStream_t stream) {
    if (!weights || !scales || !x || !y || !workspace || rows <= 0 || cols <= 0 ||
        (cols & (kFp8GroupSize - 1)) || cols / kFp8GroupSize > 256)
        return cudaErrorInvalidValue;
    auto *bytes = static_cast<uint8_t *>(workspace);
    const size_t aligned = (size_t(cols) + 255) & ~size_t(255);
    auto *activation_scales = reinterpret_cast<float *>(bytes + aligned);
    const int groups = cols / kFp8GroupSize;
    quantize_fp8_x64_kernel<<<dim3((groups + 63) / 64, 1), 256, 0, stream>>>(
        x, bytes, activation_scales, groups, 1, int(aligned));
    if (groups > 64) {
        constexpr int splits = 8;
        constexpr int shared = splits * 16 * kFp8GroupSize + splits * 16 * sizeof(float);
        fp8_tc_gemv_kernel<splits><<<(rows + 15) / 16, 32 * splits, shared, stream>>>(
            weights, reinterpret_cast<const __half *>(scales), bytes,
            activation_scales, y, rows, cols, groups);
    } else {
        fp8_tc_gemv_kernel<1><<<(rows + 15) / 16, 32, 16 * kFp8GroupSize, stream>>>(
            weights, reinterpret_cast<const __half *>(scales), bytes,
            activation_scales, y, rows, cols, groups);
    }
    return cudaGetLastError();
}

cudaError_t fp8_tc_gemv2(
    const uint8_t *weights_a,
    const uint16_t *scales_a,
    const uint8_t *weights_b,
    const uint16_t *scales_b,
    const float *x,
    float *y_a,
    float *y_b,
    int rows,
    int cols,
    void *workspace,
    cudaStream_t stream) {
    if (!weights_a || !scales_a || !weights_b || !scales_b || !x || !y_a || !y_b ||
        !workspace || y_a == y_b || rows <= 0 || cols <= 0 ||
        (cols & (kFp8GroupSize - 1)) || cols / kFp8GroupSize > 256)
        return cudaErrorInvalidValue;
    auto *bytes = static_cast<uint8_t *>(workspace);
    const size_t aligned = (size_t(cols) + 255) & ~size_t(255);
    auto *activation_scales = reinterpret_cast<float *>(bytes + aligned);
    const int groups = cols / kFp8GroupSize;
    quantize_fp8_x64_kernel<<<dim3((groups + 63) / 64, 1), 256, 0, stream>>>(
        x, bytes, activation_scales, groups, 1, int(aligned));
    if (groups > 64) {
        constexpr int splits = 8;
        constexpr int shared = splits * 16 * kFp8GroupSize + splits * 16 * sizeof(float);
        fp8_tc_gemv2_kernel<splits><<<dim3((rows + 15) / 16, 2), 32 * splits, shared, stream>>>(
            weights_a, reinterpret_cast<const __half *>(scales_a),
            weights_b, reinterpret_cast<const __half *>(scales_b),
            bytes, activation_scales, y_a, y_b, rows, cols, groups);
    } else {
        fp8_tc_gemv2_kernel<1><<<dim3((rows + 15) / 16, 2), 32, 16 * kFp8GroupSize, stream>>>(
            weights_a, reinterpret_cast<const __half *>(scales_a),
            weights_b, reinterpret_cast<const __half *>(scales_b),
            bytes, activation_scales, y_a, y_b, rows, cols, groups);
    }
    return cudaGetLastError();
}

cudaError_t fp8_tc_gemv_batch(
    const uint8_t *weights,
    const uint16_t *scales,
    const float *x,
    float *y,
    int tokens,
    int rows,
    int cols,
    int output_stride,
    void *workspace,
    cudaStream_t stream) {
    if (!weights || !scales || !x || !y || !workspace || tokens <= 0 || tokens > 64 ||
        rows <= 0 || cols <= 0 || output_stride < rows ||
        (cols & (kFp8GroupSize - 1)) || cols / kFp8GroupSize > 256)
        return cudaErrorInvalidValue;
    auto *bytes = static_cast<uint8_t *>(workspace);
    const size_t aligned = (size_t(cols) + 255) & ~size_t(255);
    auto *activation_scales = reinterpret_cast<float *>(bytes + size_t(tokens) * aligned);
    const int groups = cols / kFp8GroupSize;
    quantize_fp8_x64_kernel<<<dim3((groups + 63) / 64, tokens), 256, 0, stream>>>(
        x, bytes, activation_scales, groups, tokens, int(aligned));
    for (int base = 0; base < tokens; base += 8) {
        const int count = min(8, tokens - base);
        const uint8_t *batch_x = bytes + size_t(base) * aligned;
        const float *batch_scales = activation_scales + size_t(base) * groups;
        float *batch_y = y + size_t(base) * output_stride;
        if (count == 1)
            launch_fp8_batch<1>(weights, reinterpret_cast<const __half *>(scales),
                batch_x, batch_scales, batch_y, count, rows, cols, groups,
                int(aligned), output_stride, stream);
        else if (count == 2)
            launch_fp8_batch<2>(weights, reinterpret_cast<const __half *>(scales),
                batch_x, batch_scales, batch_y, count, rows, cols, groups,
                int(aligned), output_stride, stream);
        else if (count <= 4)
            launch_fp8_batch<4>(weights, reinterpret_cast<const __half *>(scales),
                batch_x, batch_scales, batch_y, count, rows, cols, groups,
                int(aligned), output_stride, stream);
        else
            launch_fp8_batch<8>(weights, reinterpret_cast<const __half *>(scales),
                batch_x, batch_scales, batch_y, count, rows, cols, groups,
                int(aligned), output_stride, stream);
    }
    return cudaGetLastError();
}

cudaError_t fp8_tc_gemv2_batch(
    const uint8_t *weights_a,
    const uint16_t *scales_a,
    const uint8_t *weights_b,
    const uint16_t *scales_b,
    const float *x,
    float *y_a,
    float *y_b,
    int tokens,
    int rows,
    int cols,
    int output_stride,
    void *workspace,
    cudaStream_t stream) {
    if (!weights_a || !scales_a || !weights_b || !scales_b || !x || !y_a || !y_b ||
        !workspace || tokens <= 0 || tokens > 64 || rows <= 0 || cols <= 0 ||
        output_stride < rows || (cols & (kFp8GroupSize - 1)) ||
        cols / kFp8GroupSize > 256)
        return cudaErrorInvalidValue;
    auto *bytes = static_cast<uint8_t *>(workspace);
    const size_t aligned = (size_t(cols) + 255) & ~size_t(255);
    auto *activation_scales = reinterpret_cast<float *>(bytes + size_t(tokens) * aligned);
    const int groups = cols / kFp8GroupSize;
    quantize_fp8_x64_kernel<<<dim3((groups + 63) / 64, tokens), 256, 0, stream>>>(
        x, bytes, activation_scales, groups, tokens, int(aligned));
    for (int base = 0; base < tokens; base += 8) {
        const int count = min(8, tokens - base);
        const uint8_t *batch_x = bytes + size_t(base) * aligned;
        const float *batch_scales = activation_scales + size_t(base) * groups;
        float *batch_y_a = y_a + size_t(base) * output_stride;
        float *batch_y_b = y_b + size_t(base) * output_stride;
        const auto *half_scales_a = reinterpret_cast<const __half *>(scales_a);
        const auto *half_scales_b = reinterpret_cast<const __half *>(scales_b);
        if (count == 1)
            launch_fp8_batch_pair<1>(weights_a, half_scales_a, weights_b, half_scales_b,
                batch_x, batch_scales, batch_y_a, batch_y_b, count, rows, cols, groups,
                int(aligned), output_stride, stream);
        else if (count == 2)
            launch_fp8_batch_pair<2>(weights_a, half_scales_a, weights_b, half_scales_b,
                batch_x, batch_scales, batch_y_a, batch_y_b, count, rows, cols, groups,
                int(aligned), output_stride, stream);
        else if (count <= 4)
            launch_fp8_batch_pair<4>(weights_a, half_scales_a, weights_b, half_scales_b,
                batch_x, batch_scales, batch_y_a, batch_y_b, count, rows, cols, groups,
                int(aligned), output_stride, stream);
        else
            launch_fp8_batch_pair<8>(weights_a, half_scales_a, weights_b, half_scales_b,
                batch_x, batch_scales, batch_y_a, batch_y_b, count, rows, cols, groups,
                int(aligned), output_stride, stream);
    }
    return cudaGetLastError();
}

}  // namespace insignia::glm53
