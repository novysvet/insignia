#include "insignia_glm53.cuh"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace insignia::glm53 {
namespace {

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, offset);
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
    return value;
}

__device__ __forceinline__ float fp8_to_float(uint8_t value) {
    return __half2float(__nv_cvt_fp8_to_halfraw(value, __NV_E4M3));
}

__device__ __forceinline__ uint8_t float_to_fp8(float value) {
    return uint8_t(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float bf16_to_float(uint16_t value) {
    return __uint_as_float(uint32_t(value) << 16);
}

// Eight blocks per row expose 192 independent chunks to Ada's 56 SMs.  The
// extra activation traffic is cheaper than leaving 48 SMs idle with a fused
// three-row block.  Each chunk accumulates x^2 while x is already in a register,
// eliminating a separate RMS launch and its global synchronization.
__global__ __launch_bounds__(256) void mhc_fn_kernel(
    const uint16_t *__restrict__ fn,
    const float *__restrict__ streams,
    float *__restrict__ partials,
    int width) {
    const int row = blockIdx.x;
    const int part = blockIdx.y;
    float sum = 0.0f, sum2 = 0.0f;
    const uint16_t *weights = fn + size_t(row) * width;
    for (int i = part * blockDim.x + threadIdx.x; i < width; i += 8 * blockDim.x) {
        const float x = streams[i];
        sum = fmaf(bf16_to_float(weights[i]), x, sum);
        sum2 = fmaf(x, x, sum2);
    }
    sum = warp_sum(sum);
    sum2 = warp_sum(sum2);
    __shared__ float partial[2][8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (!lane) {
        partial[0][warp] = sum;
        partial[1][warp] = sum2;
    }
    __syncthreads();
    if (!warp) {
        sum = lane < 8 ? partial[0][lane] : 0.0f;
        sum2 = lane < 8 ? partial[1][lane] : 0.0f;
        sum = warp_sum(sum);
        sum2 = warp_sum(sum2);
        if (!lane) {
            const int output = row * 8 + part;
            partials[output] = sum;
            partials[192 + output] = sum2;
        }
    }
}

__global__ __launch_bounds__(256) void mhc_finalize_kernel(
    const float *__restrict__ partials,
    const float *__restrict__ base,
    const float *__restrict__ scale,
    const float *__restrict__ streams,
            float *__restrict__ post,
            float *__restrict__ comb,
            float *__restrict__ collapsed,
            int width) {
    __shared__ float pre[4];
    __shared__ float params[24];
    if (threadIdx.x < 24) {
        const int offset = threadIdx.x * 8;
        const float dot = ((partials[offset] + partials[offset + 1]) +
                           (partials[offset + 2] + partials[offset + 3])) +
                          ((partials[offset + 4] + partials[offset + 5]) +
                           (partials[offset + 6] + partials[offset + 7]));
        const int square = 192 + offset;
        const float sum2 = ((partials[square] + partials[square + 1]) +
                            (partials[square + 2] + partials[square + 3])) +
                           ((partials[square + 4] + partials[square + 5]) +
                            (partials[square + 6] + partials[square + 7]));
        params[threadIdx.x] = dot * rsqrtf(sum2 * (1.0f / float(width)) + 1.0e-5f);
    }
    __syncthreads();
    if (!threadIdx.x) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float pre_logit = fmaf(params[i], scale[0], base[i]);
            const float post_logit = fmaf(params[4 + i], scale[1], base[4 + i]);
            pre[i] = 1.0f / (1.0f + expf(-pre_logit)) + 1.0e-6f;
            post[i] = 2.0f / (1.0f + expf(-post_logit));
        }
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            float maximum = -3.402823466e38F;
#pragma unroll
            for (int col = 0; col < 4; ++col) {
                const int index = row * 4 + col;
                comb[index] = fmaf(params[8 + index], scale[2], base[8 + index]);
                maximum = fmaxf(maximum, comb[index]);
            }
            float denominator = 0.0f;
#pragma unroll
            for (int col = 0; col < 4; ++col) {
                const int index = row * 4 + col;
                comb[index] = expf(comb[index] - maximum);
                denominator += comb[index];
            }
#pragma unroll
            for (int col = 0; col < 4; ++col)
                comb[row * 4 + col] = comb[row * 4 + col] / denominator + 1.0e-6f;
        }
        // The model initializes with a column normalization and then performs
        // 19 row/column Sinkhorn rounds.
#pragma unroll
        for (int col = 0; col < 4; ++col) {
            float denominator = 1.0e-6f;
#pragma unroll
            for (int row = 0; row < 4; ++row) denominator += comb[row * 4 + col];
#pragma unroll
            for (int row = 0; row < 4; ++row) comb[row * 4 + col] /= denominator;
        }
#pragma unroll 1
        for (int iteration = 1; iteration < 20; ++iteration) {
#pragma unroll
            for (int row = 0; row < 4; ++row) {
                float denominator = 1.0e-6f;
#pragma unroll
                for (int col = 0; col < 4; ++col) denominator += comb[row * 4 + col];
#pragma unroll
                for (int col = 0; col < 4; ++col) comb[row * 4 + col] /= denominator;
            }
#pragma unroll
            for (int col = 0; col < 4; ++col) {
                float denominator = 1.0e-6f;
#pragma unroll
                for (int row = 0; row < 4; ++row) denominator += comb[row * 4 + col];
#pragma unroll
                for (int row = 0; row < 4; ++row) comb[row * 4 + col] /= denominator;
            }
        }
    }
    __syncthreads();
    const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
    for (int dimension = threadIdx.x; dimension < width / kHyperStreams; dimension += blockDim.x) {
        collapsed[dimension] = fmaf(p0, streams[dimension],
            fmaf(p1, streams[width / kHyperStreams + dimension],
            fmaf(p2, streams[2 * (width / kHyperStreams) + dimension], p3 * streams[3 * (width / kHyperStreams) + dimension])));
    }
}

// mhc_finalize with the decode-side RMSNorm folded in: the collapse loop and
// the normalization run in one launch. The variance reduction below is a
// verbatim transplant of rms_bf16_kernel's tree (same per-thread index
// sequence, same warp_sum shape), and the collapsed value is recomputed with
// the identical fmaf chain instead of being stored and reloaded, so the
// normalized output stays bit-identical to the two-launch pipeline.
__global__ __launch_bounds__(256) void mhc_finalize_rms_kernel(
    const float *__restrict__ partials,
    const float *__restrict__ base,
    const float *__restrict__ scale,
    const float *__restrict__ streams,
    const uint16_t *__restrict__ rms_weight,
    float *__restrict__ post,
    float *__restrict__ comb,
    float *__restrict__ normalized,
    int width) {
    __shared__ float pre[4];
    __shared__ float params[24];
    if (threadIdx.x < 24) {
        const int offset = threadIdx.x * 8;
        const float dot = ((partials[offset] + partials[offset + 1]) +
                           (partials[offset + 2] + partials[offset + 3])) +
                          ((partials[offset + 4] + partials[offset + 5]) +
                           (partials[offset + 6] + partials[offset + 7]));
        const int square = 192 + offset;
        const float sum2 = ((partials[square] + partials[square + 1]) +
                            (partials[square + 2] + partials[square + 3])) +
                           ((partials[square + 4] + partials[square + 5]) +
                            (partials[square + 6] + partials[square + 7]));
        params[threadIdx.x] = dot * rsqrtf(sum2 * (1.0f / float(width)) + 1.0e-5f);
    }
    __syncthreads();
    if (!threadIdx.x) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float pre_logit = fmaf(params[i], scale[0], base[i]);
            const float post_logit = fmaf(params[4 + i], scale[1], base[4 + i]);
            pre[i] = 1.0f / (1.0f + expf(-pre_logit)) + 1.0e-6f;
            post[i] = 2.0f / (1.0f + expf(-post_logit));
        }
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            float maximum = -3.402823466e38F;
#pragma unroll
            for (int col = 0; col < 4; ++col) {
                const int index = row * 4 + col;
                comb[index] = fmaf(params[8 + index], scale[2], base[8 + index]);
                maximum = fmaxf(maximum, comb[index]);
            }
            float denominator = 0.0f;
#pragma unroll
            for (int col = 0; col < 4; ++col) {
                const int index = row * 4 + col;
                comb[index] = expf(comb[index] - maximum);
                denominator += comb[index];
            }
#pragma unroll
            for (int col = 0; col < 4; ++col)
                comb[row * 4 + col] = comb[row * 4 + col] / denominator + 1.0e-6f;
        }
#pragma unroll
        for (int col = 0; col < 4; ++col) {
            float denominator = 1.0e-6f;
#pragma unroll
            for (int row = 0; row < 4; ++row) denominator += comb[row * 4 + col];
#pragma unroll
            for (int row = 0; row < 4; ++row) comb[row * 4 + col] /= denominator;
        }
#pragma unroll 1
        for (int iteration = 1; iteration < 20; ++iteration) {
#pragma unroll
            for (int row = 0; row < 4; ++row) {
                float denominator = 1.0e-6f;
#pragma unroll
                for (int col = 0; col < 4; ++col) denominator += comb[row * 4 + col];
#pragma unroll
                for (int col = 0; col < 4; ++col) comb[row * 4 + col] /= denominator;
            }
#pragma unroll
            for (int col = 0; col < 4; ++col) {
                float denominator = 1.0e-6f;
#pragma unroll
                for (int row = 0; row < 4; ++row) denominator += comb[row * 4 + col];
#pragma unroll
                for (int row = 0; row < 4; ++row) comb[row * 4 + col] /= denominator;
            }
        }
    }
    __syncthreads();
    const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
    const int cols = width / kHyperStreams;
    float square = 0.0f;
    for (int dimension = threadIdx.x; dimension < cols; dimension += blockDim.x) {
        const float value = fmaf(p0, streams[dimension],
            fmaf(p1, streams[cols + dimension],
            fmaf(p2, streams[2 * cols + dimension], p3 * streams[3 * cols + dimension])));
        square = fmaf(value, value, square);
    }
    square = warp_sum(square);
    // params[] is dead past the Sinkhorn barrier; its first 32 bytes double
    // as the RMS partials so shared usage (and occupancy) are unchanged.
    float *const partial = params;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (!lane) partial[warp] = square;
    __syncthreads();
    if (!warp) {
        square = lane < 8 ? partial[lane] : 0.0f;
        square = warp_sum(square);
        if (!lane) partial[0] = rsqrtf(square / cols + 1.0e-5f);
    }
    __syncthreads();
    const float inverse = partial[0];
    for (int dimension = threadIdx.x; dimension < cols; dimension += blockDim.x) {
        const float value = fmaf(p0, streams[dimension],
            fmaf(p1, streams[cols + dimension],
            fmaf(p2, streams[2 * cols + dimension], p3 * streams[3 * cols + dimension])));
        normalized[dimension] = value * inverse * bf16_to_float(rms_weight[dimension]);
    }
}

__global__ __launch_bounds__(256) void mhc_mix_kernel(
    const float *__restrict__ streams,
    const float *__restrict__ sublayer,
    const float *__restrict__ post,
            const float *__restrict__ comb,
            float *__restrict__ output,
            int hidden) {
    const int dimension = blockIdx.x * blockDim.x + threadIdx.x;
    if (dimension >= hidden) return;
    const float x0 = streams[dimension];
    const float x1 = streams[hidden + dimension];
    const float x2 = streams[2 * hidden + dimension];
    const float x3 = streams[3 * hidden + dimension];
    const float update = sublayer[dimension];
#pragma unroll
        for (int stream = 0; stream < 4; ++stream) {
            // NumPy reference: residual = comb.T @ streams.
            float residual = comb[stream] * x0;
            residual = fmaf(comb[4 + stream], x1, residual);
            residual = fmaf(comb[8 + stream], x2, residual);
            residual = fmaf(comb[12 + stream], x3, residual);
            output[stream * hidden + dimension] = fmaf(post[stream], update, residual);
        }
}

// The 128-wide head loop must stay a compile-time bound: a runtime head_dim
// costs half the state-loop unroll (measured 2x on the 34-layer ring), so the
// recurrence is specialized per supported head width instead.
template <int HEAD_DIM>
__global__ __launch_bounds__(HEAD_DIM, 4) void kda_decode_kernel(
    float *__restrict__ state,
    const float *__restrict__ q_input,
    const float *__restrict__ k_input,
    const float *__restrict__ v,
    const float *__restrict__ g,
    const float *__restrict__ beta,
    float *__restrict__ output) {
    static_assert(HEAD_DIM == 32 || HEAD_DIM == 64 || HEAD_DIM == 128, "unsupported KDA head width");
    constexpr int WARPS = HEAD_DIM / 32;
    const int head = blockIdx.x;
    const int element = threadIdx.x;
    const int offset = head * HEAD_DIM;
    float q = q_input[offset + element];
    float k = k_input[offset + element];
    float qn = q * q;
    float kn = k * k;
    qn = warp_sum(qn);
    kn = warp_sum(kn);
    __shared__ float norms[2][WARPS];
    const int lane = element & 31;
    const int warp = element >> 5;
    if (!lane) {
        norms[0][warp] = qn;
        norms[1][warp] = kn;
    }
    __syncthreads();
    if (!element) {
        float qsum = 0.0f, ksum = 0.0f;
#pragma unroll
        for (int index = 0; index < WARPS; ++index) {
            qsum += norms[0][index];
            ksum += norms[1][index];
        }
        norms[0][0] = rsqrtf(qsum + 1.0e-6f) * rsqrtf((float)HEAD_DIM);
        norms[0][1] = rsqrtf(ksum + 1.0e-6f);
    }
    __syncthreads();
    const float q_scale = norms[0][0], k_scale = norms[0][1];
    __shared__ float sq[HEAD_DIM], sk[HEAD_DIM], decay[HEAD_DIM], delta[HEAD_DIM];
    sq[element] = q * q_scale;
    sk[element] = k * k_scale;
    decay[element] = expf(g[offset + element]);
    __syncthreads();

    float *head_state = state + size_t(head) * HEAD_DIM * HEAD_DIM;
    float memory = 0.0f;
#pragma unroll
    for (int key = 0; key < HEAD_DIM; ++key) {
        const float cell = head_state[key * HEAD_DIM + element];
        memory = fmaf(cell * decay[key], sk[key], memory);
    }
    delta[element] = (v[offset + element] - memory) * beta[head];
    __syncthreads();

    float result = 0.0f;
#pragma unroll
    for (int key = 0; key < HEAD_DIM; ++key) {
        float &cell = head_state[key * HEAD_DIM + element];
        cell = fmaf(sk[key], delta[element], cell * decay[key]);
        result = fmaf(cell, sq[key], result);
    }
    output[offset + element] = result;
}

template <bool WEIGHTS_FP32>
__global__ __launch_bounds__(256) void kda_conv_silu_kernel(
    float *__restrict__ projection,
    const void *__restrict__ conv,
    float *__restrict__ history,
    int position,
    int count) {
    const uint16_t *conv16 = static_cast<const uint16_t *>(conv);
    const float *conv32 = static_cast<const float *>(conv);
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * gridDim.x) {
        const float current = projection[index];
        const float taps[4] = {
            WEIGHTS_FP32 ? conv32[index * 4 + 0] : bf16_to_float(conv16[index * 4 + 0]),
            WEIGHTS_FP32 ? conv32[index * 4 + 1] : bf16_to_float(conv16[index * 4 + 1]),
            WEIGHTS_FP32 ? conv32[index * 4 + 2] : bf16_to_float(conv16[index * 4 + 2]),
            WEIGHTS_FP32 ? conv32[index * 4 + 3] : bf16_to_float(conv16[index * 4 + 3]),
        };
        float value = current * taps[3];
#pragma unroll
        for (int lag = 1; lag <= 3; ++lag) {
            if (position >= lag) {
                const int slot = (position - lag) % 3;
                value = fmaf(history[slot * count + index], taps[3 - lag], value);
            }
        }
        history[(position % 3) * count + index] = current;
        projection[index] = value / (1.0f + expf(-value));
    }
}

__global__ __launch_bounds__(256) void kda_conv_silu3_kernel(
    float *__restrict__ q,
    float *__restrict__ k,
    float *__restrict__ v,
    const void *__restrict__ conv_q,
    const void *__restrict__ conv_k,
    const void *__restrict__ conv_v,
    float *__restrict__ history,
    int position,
    int count,
    int weights_fp32) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < 3 * count;
         index += blockDim.x * gridDim.x) {
        const int segment = index / count;
        const int within = index - segment * count;
        float *projection = segment == 0 ? q : (segment == 1 ? k : v);
        const void *conv = segment == 0 ? conv_q : (segment == 1 ? conv_k : conv_v);
        const uint16_t *conv16 = static_cast<const uint16_t *>(conv);
        const float *conv32 = static_cast<const float *>(conv);
        float *segment_history = history + segment * 3 * count;
        const float current = projection[within];
        const float taps[4] = {
            weights_fp32 ? conv32[within * 4 + 0] : bf16_to_float(conv16[within * 4 + 0]),
            weights_fp32 ? conv32[within * 4 + 1] : bf16_to_float(conv16[within * 4 + 1]),
            weights_fp32 ? conv32[within * 4 + 2] : bf16_to_float(conv16[within * 4 + 2]),
            weights_fp32 ? conv32[within * 4 + 3] : bf16_to_float(conv16[within * 4 + 3]),
        };
        float value = current * taps[3];
#pragma unroll
        for (int lag = 1; lag <= 3; ++lag) {
            if (position >= lag) {
                const int slot = (position - lag) % 3;
                value = fmaf(segment_history[slot * count + within], taps[3 - lag], value);
            }
        }
        segment_history[(position % 3) * count + within] = current;
        projection[within] = value / (1.0f + expf(-value));
    }
}

constexpr int kMlaExactContext = 256;

__global__ __launch_bounds__(256) void mla_decode_kernel(
    const float *__restrict__ query,
    const float *__restrict__ kv,
    float *__restrict__ key_cache,
    float *__restrict__ value_cache,
    float *__restrict__ output,
    int position,
    int cache_stride) {
    const int head = blockIdx.x;
    const int element = threadIdx.x;
    const int head_dim = blockDim.x;
    const int warps = head_dim >> 5;
    const int vector_offset = head * head_dim + element;
    key_cache[position * cache_stride + vector_offset] = kv[head * (2 * head_dim) + element];
    value_cache[position * cache_stride + vector_offset] = kv[head * (2 * head_dim) + head_dim + element];
    __syncthreads();

    __shared__ float logits[kMlaExactContext];
    __shared__ float partial[8];
    const int lane = element & 31;
    const int warp = element >> 5;
    const float q = query[vector_offset];
    const float scale = rsqrtf((float)head_dim);
    for (int token = 0; token <= position; ++token) {
        float dot = warp_sum(q * key_cache[token * cache_stride + vector_offset]);
        if (!lane) partial[warp] = dot;
        __syncthreads();
        if (!element) {
            float total = 0.0f;
#pragma unroll 4
            for (int index = 0; index < warps; ++index) total += partial[index];
            logits[token] = total * scale;
        }
        __syncthreads();
    }
    if (!element) {
        float maximum = -3.402823466e38F;
        for (int token = 0; token <= position; ++token) maximum = fmaxf(maximum, logits[token]);
        float denominator = 0.0f;
        for (int token = 0; token <= position; ++token) {
            logits[token] = expf(logits[token] - maximum);
            denominator += logits[token];
        }
        const float inverse = 1.0f / denominator;
        for (int token = 0; token <= position; ++token) logits[token] *= inverse;
    }
    __syncthreads();
    float result = 0.0f;
    for (int token = 0; token <= position; ++token)
        result = fmaf(logits[token], value_cache[token * cache_stride + vector_offset], result);
    output[vector_offset] = result;
}

__global__ __launch_bounds__(256) void mla_store_kv_batch_kernel(
    const float *__restrict__ kv,
    float *__restrict__ key_cache,
    float *__restrict__ value_cache,
    int position_base,
    int heads,
    int head_dim) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int element = threadIdx.x;
    const int width = heads * head_dim;
    const float *source = kv + size_t(token) * 2 * width + head * (2 * head_dim);
    const size_t destination = size_t(position_base + token) * width +
                               head * head_dim + element;
    key_cache[destination] = source[element];
    value_cache[destination] = source[head_dim + element];
}

// Ada-specialized FA2 forward tile.  One CTA owns eight consecutive causal
// queries from one head, and every cache K/V element is fetched once and reused
// by all eight.  Scores stay in SRAM.  The capped 256-token exact-attention
// window makes a two-pass softmax cheap and, crucially, preserves the scalar
// kernel's operation order so tiny rounding changes cannot flip an MoE route.
template <int kQueries>
__global__ __launch_bounds__(256, 2) void mla_flash2_prefill_kernel(
    const float *__restrict__ query,
    const float *__restrict__ key_cache,
    const float *__restrict__ value_cache,
    float *__restrict__ output,
    int tokens,
    int position_base,
    int heads,
    int head_dim) {
    const int head = blockIdx.x;
    const int query_base = blockIdx.y * kQueries;
    const int query_count = min(kQueries, tokens - query_base);
    const int element = threadIdx.x;
    const int lane = element & 31;
    const int warp = element >> 5;
    const int warps = head_dim >> 5;
    const int width = heads * head_dim;
    const int head_offset = head * head_dim + element;
    const float scale = rsqrtf(float(head_dim));

    float q[kQueries], accumulator[kQueries];
#pragma unroll
    for (int slot = 0; slot < kQueries; ++slot) {
        q[slot] = slot < query_count
            ? query[size_t(query_base + slot) * width + head_offset]
            : 0.0f;
        accumulator[slot] = 0.0f;
    }

    __shared__ float partial[kQueries][8];
    __shared__ float logits[kQueries][kMlaExactContext];
    const int last_key = position_base + query_base + query_count - 1;

    for (int key = 0; key <= last_key; ++key) {
        const size_t cache_index = size_t(key) * width + head_offset;
        const float key_value = key_cache[cache_index];
#pragma unroll
        for (int slot = 0; slot < kQueries; ++slot) {
            const float dot = warp_sum(q[slot] * key_value);
            if (!lane) partial[slot][warp] = dot;
        }
        __syncthreads();

        if (element < kQueries) {
            const int slot = element;
            if (slot < query_count) {
                float dot = 0.0f;
#pragma unroll 4
                for (int part = 0; part < warps; ++part) dot += partial[slot][part];
                logits[slot][key] = dot * scale;
            }
        }
        __syncthreads();
    }

    if (element < query_count) {
        const int slot = element;
        const int query_last_key = position_base + query_base + slot;
        float maximum = -3.402823466e38F;
        for (int key = 0; key <= query_last_key; ++key)
            maximum = fmaxf(maximum, logits[slot][key]);
        float denominator = 0.0f;
        for (int key = 0; key <= query_last_key; ++key) {
            logits[slot][key] = expf(logits[slot][key] - maximum);
            denominator += logits[slot][key];
        }
        const float inverse = 1.0f / denominator;
        for (int key = 0; key <= query_last_key; ++key) logits[slot][key] *= inverse;
    }
    __syncthreads();

    for (int key = 0; key <= last_key; ++key) {
        const float value = value_cache[size_t(key) * width + head_offset];
#pragma unroll
        for (int slot = 0; slot < kQueries; ++slot)
            if (slot < query_count && key <= position_base + query_base + slot)
                accumulator[slot] = fmaf(logits[slot][key], value, accumulator[slot]);
    }
#pragma unroll
    for (int slot = 0; slot < kQueries; ++slot)
        if (slot < query_count)
            output[size_t(query_base + slot) * width + head_offset] = accumulator[slot];
}

}  // namespace

size_t mhc_workspace_bytes() {
    // Dot and x^2 partials for 24 rows x 8 chunks, rounded to 32 cache lines.
    return 512 * sizeof(float);
}

cudaError_t mhc_analyze(
    const uint16_t *fn,
    const float *base,
    const float *scale,
    const float *streams,
    const uint16_t *rms_weight,
    float *post,
    float *comb,
    float *collapsed,
    void *workspace,
    int width,
    cudaStream_t stream) {
    float *scratch = static_cast<float *>(workspace);
    mhc_fn_kernel<<<dim3(24, 8), 256, 0, stream>>>(fn, streams, scratch, width);
    if (rms_weight)
        mhc_finalize_rms_kernel<<<1, 256, 0, stream>>>(
            scratch, base, scale, streams, rms_weight, post, comb, collapsed, width);
    else
        mhc_finalize_kernel<<<1, 256, 0, stream>>>(scratch, base, scale, streams, post, comb, collapsed, width);
    return cudaGetLastError();
}

cudaError_t mhc_mix(
    const float *streams,
    const float *sublayer,
    const float *post,
    const float *comb,
    float *out_streams,
    int hidden,
    cudaStream_t stream) {
    mhc_mix_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(streams, sublayer, post, comb, out_streams, hidden);
    return cudaGetLastError();
}

cudaError_t kda_decode(
    float *state,
    const float *q,
    const float *k,
    const float *v,
    const float *g,
    const float *beta,
    float *output,
    int heads,
    int head_dim,
    cudaStream_t stream) {
    switch (head_dim) {
    case 32:
        kda_decode_kernel<32><<<heads, 32, 0, stream>>>(state, q, k, v, g, beta, output);
        break;
    case 64:
        kda_decode_kernel<64><<<heads, 64, 0, stream>>>(state, q, k, v, g, beta, output);
        break;
    case 128:
        kda_decode_kernel<128><<<heads, 128, 0, stream>>>(state, q, k, v, g, beta, output);
        break;
    default:
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

cudaError_t kda_conv_silu(
    float *projection,
    const uint16_t *conv,
    float *history,
    int position,
    int count,
    bool weights_fp32,
    cudaStream_t stream) {
    if (position < 0) return cudaErrorInvalidValue;
    if (weights_fp32)
        kda_conv_silu_kernel<true><<<32, 256, 0, stream>>>(
            projection, conv, history, position, count);
    else
        kda_conv_silu_kernel<false><<<32, 256, 0, stream>>>(
            projection, conv, history, position, count);
    return cudaGetLastError();
}

cudaError_t kda_conv_silu3(
    float *q,
    float *k,
    float *v,
    const void *conv_q,
    const void *conv_k,
    const void *conv_v,
    float *history,
    int position,
    int count,
    bool weights_fp32,
    cudaStream_t stream) {
    if (position < 0) return cudaErrorInvalidValue;
    kda_conv_silu3_kernel<<<96, 256, 0, stream>>>(
        q, k, v, conv_q, conv_k, conv_v, history, position, count, weights_fp32 ? 1 : 0);
    return cudaGetLastError();
}

cudaError_t mla_decode(
    const float *query,
    const float *kv,
    float *key_cache,
    float *value_cache,
    float *output,
    int position,
    int heads,
    int head_dim,
    cudaStream_t stream) {
    if (position < 0 || position >= kMlaExactContext) return cudaErrorInvalidValue;
    if (head_dim < 32 || head_dim > 256 || head_dim % 32) return cudaErrorInvalidValue;
    mla_decode_kernel<<<heads, head_dim, 0, stream>>>(
        query, kv, key_cache, value_cache, output, position, heads * head_dim);
    return cudaGetLastError();
}

cudaError_t mla_flash2_prefill(
    const float *query,
    const float *kv,
    float *key_cache,
    float *value_cache,
    float *output,
    int tokens,
    int position_base,
    int heads,
    int head_dim,
    cudaStream_t stream) {
    if (!query || !kv || !key_cache || !value_cache || !output ||
        tokens <= 0 || tokens > 64 || position_base < 0 ||
        position_base + tokens > kMlaExactContext || heads <= 0 ||
        head_dim < 32 || head_dim > 256 || head_dim % 32)
        return cudaErrorInvalidValue;
    mla_store_kv_batch_kernel<<<dim3(heads, tokens), head_dim, 0, stream>>>(
        kv, key_cache, value_cache, position_base, heads, head_dim);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    constexpr int queries = 8;
    mla_flash2_prefill_kernel<queries>
        <<<dim3(heads, (tokens + queries - 1) / queries), head_dim, 0, stream>>>(
            query, key_cache, value_cache, output, tokens, position_base, heads, head_dim);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Latent-cache MLA (absorbed).  The cache stores the compressed
// post-kv_a-layernorm latent (512 wide) per position instead of the expanded
// per-head K/V (16384 wide), so an 8192-token cache costs ~50 MiB in FP8.
// Attention runs algebraically identically: scores are (q_head @ W_uk) . c_kv
// and the output is W_uv @ (weighted latent sum).
// ---------------------------------------------------------------------------

constexpr int kMlaDecodeTile = 512;

__global__ __launch_bounds__(512) void mla_store_latent_kernel(
    const float *__restrict__ latent,
    uint8_t *__restrict__ cache,
    float *__restrict__ scales,
    float *__restrict__ cache_f32,
    int position_base,
    int latent_dim) {
    const int token = blockIdx.x;
    const int dim = threadIdx.x;
    const float value = latent[size_t(token) * latent_dim + dim];
    __shared__ float partial[16];
    __shared__ float scale_shared[kMlaLatentGroups];
    const int lane = dim & 31;
    const int warp = dim >> 5;
    float magnitude = warp_max(fabsf(value));
    if (!lane) partial[warp] = magnitude;
    __syncthreads();
    if (dim < kMlaLatentGroups) {
        const float peak = fmaxf(partial[2 * dim], partial[2 * dim + 1]);
        scale_shared[dim] = fmaxf(peak / 448.0f, 1e-30f);
        if (scales)
            scales[size_t(position_base + token) * kMlaLatentGroups + dim] =
                scale_shared[dim];
    }
    __syncthreads();
    if (cache_f32) {
        cache_f32[size_t(position_base + token) * latent_dim + dim] = value;
    } else {
        cache[size_t(position_base + token) * latent_dim + dim] =
            float_to_fp8(value / scale_shared[dim / kMlaLatentGroupSize]);
    }
}

__global__ __launch_bounds__(256) void mla_store_value_batch_kernel(
    const float *__restrict__ kv,
    float *__restrict__ value_cache,
    int position_base,
    int heads,
    int head_dim) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int element = threadIdx.x;
    const int width = heads * head_dim;
    const float *source = kv + size_t(token) * 2 * width +
                          head * (2 * head_dim) + head_dim;
    value_cache[size_t(position_base + token) * width +
                head * head_dim + element] = source[element];
}

// Prefix oracle: retain absorbed latent-space score computation, but consume
// the exact expanded FP32 values produced by kv_b_proj.  A two-pass softmax
// preserves the baseline accumulation order and isolates value-side
// reassociation without paying for an expanded 8192-token K/V cache.
__global__ __launch_bounds__(256) void mla_decode_latent_exact_value_kernel(
    const float *__restrict__ query,
    const uint8_t *__restrict__ cache,
    const float *__restrict__ scales,
    const float *__restrict__ cache_f32,
    const float *__restrict__ exact_value_cache,
    const float *__restrict__ w_uk,
    float *__restrict__ output,
    int position,
    int latent_dim) {
    const int head = blockIdx.x;
    const int element = threadIdx.x;
    const int lane = element & 31;
    const int warp = element >> 5;
    const int width = gridDim.x * 256;
    __shared__ float q_shared[256];
    __shared__ float partial_sum[8];
    __shared__ float logits[kMlaExactContext];
    __shared__ float scale_shared[kMlaLatentGroups];

    q_shared[element] = query[head * 256 + element];
    __syncthreads();

    float qe0 = 0.0f, qe1 = 0.0f;
    for (int j = 0; j < 256; ++j) {
        const float qj = q_shared[j];
        const float *row = w_uk + (size_t(head) * 256 + j) * latent_dim;
        qe0 = fmaf(qj, row[element], qe0);
        qe1 = fmaf(qj, row[element + 256], qe1);
    }

    const float score_scale = rsqrtf(256.0f);
    for (int key = 0; key <= position; ++key) {
        float k0, k1;
        if (cache_f32) {
            k0 = cache_f32[size_t(key) * latent_dim + element];
            k1 = cache_f32[size_t(key) * latent_dim + element + 256];
        } else {
            if (element < kMlaLatentGroups)
                scale_shared[element] = scales[size_t(key) * kMlaLatentGroups + element];
            __syncthreads();
            k0 = fp8_to_float(cache[size_t(key) * latent_dim + element]) *
                 scale_shared[element / kMlaLatentGroupSize];
            k1 = fp8_to_float(cache[size_t(key) * latent_dim + element + 256]) *
                 scale_shared[(element + 256) / kMlaLatentGroupSize];
        }
        const float dot = warp_sum(qe0 * k0 + qe1 * k1);
        if (!lane) partial_sum[warp] = dot;
        __syncthreads();
        if (!element) {
            float total = 0.0f;
#pragma unroll
            for (int index = 0; index < 8; ++index) total += partial_sum[index];
            logits[key] = total * score_scale;
        }
        __syncthreads();
    }

    if (!element) {
        float maximum = -3.402823466e38F;
        for (int key = 0; key <= position; ++key) maximum = fmaxf(maximum, logits[key]);
        float denominator = 0.0f;
        for (int key = 0; key <= position; ++key) {
            logits[key] = expf(logits[key] - maximum);
            denominator += logits[key];
        }
        const float inverse = 1.0f / denominator;
        for (int key = 0; key <= position; ++key) logits[key] *= inverse;
    }
    __syncthreads();

    const int value_offset = head * 256 + element;
    float result = 0.0f;
    for (int key = 0; key <= position; ++key)
        result = fmaf(logits[key], exact_value_cache[size_t(key) * width + value_offset], result);
    output[value_offset] = result;
}

__global__ __launch_bounds__(256) void mla_prefill_latent_exact_value_kernel(
    const float *__restrict__ query,
    const uint8_t *__restrict__ cache,
    const float *__restrict__ scales,
    const float *__restrict__ cache_f32,
    const float *__restrict__ exact_value_cache,
    const float *__restrict__ w_uk,
    float *__restrict__ output,
    int tokens,
    int position_base,
    int latent_dim) {
    constexpr int kQueries = 8;
    const int head = blockIdx.x;
    const int query_base = blockIdx.y * kQueries;
    const int query_count = min(kQueries, tokens - query_base);
    const int element = threadIdx.x;
    const int lane = element & 31;
    const int warp = element >> 5;
    const int width = gridDim.x * 256;
    __shared__ float q_shared[256];
    __shared__ float partial_sum[kQueries][8];
    __shared__ float logits[kQueries][kMlaExactContext];
    __shared__ float scale_shared[kMlaLatentGroups];

    float qe[kQueries][2];
#pragma unroll
    for (int slot = 0; slot < kQueries; ++slot) qe[slot][0] = qe[slot][1] = 0.0f;
    for (int slot = 0; slot < query_count; ++slot) {
        const float *qrow = query + size_t(query_base + slot) * width + head * 256;
        q_shared[element] = qrow[element];
        __syncthreads();
        float qe0 = 0.0f, qe1 = 0.0f;
        for (int j = 0; j < 256; ++j) {
            const float qj = q_shared[j];
            const float *row = w_uk + (size_t(head) * 256 + j) * latent_dim;
            qe0 = fmaf(qj, row[element], qe0);
            qe1 = fmaf(qj, row[element + 256], qe1);
        }
        qe[slot][0] = qe0;
        qe[slot][1] = qe1;
        __syncthreads();
    }

    const int last_key = position_base + query_base + query_count - 1;
    const float score_scale = rsqrtf(256.0f);
    for (int key = 0; key <= last_key; ++key) {
        float k0, k1;
        if (cache_f32) {
            k0 = cache_f32[size_t(key) * latent_dim + element];
            k1 = cache_f32[size_t(key) * latent_dim + element + 256];
        } else {
            if (element < kMlaLatentGroups)
                scale_shared[element] = scales[size_t(key) * kMlaLatentGroups + element];
            __syncthreads();
            k0 = fp8_to_float(cache[size_t(key) * latent_dim + element]) *
                 scale_shared[element / kMlaLatentGroupSize];
            k1 = fp8_to_float(cache[size_t(key) * latent_dim + element + 256]) *
                 scale_shared[(element + 256) / kMlaLatentGroupSize];
        }
#pragma unroll
        for (int slot = 0; slot < kQueries; ++slot) {
            const float dot = warp_sum(qe[slot][0] * k0 + qe[slot][1] * k1);
            if (!lane) partial_sum[slot][warp] = dot;
        }
        __syncthreads();
        if (element < query_count) {
            float total = 0.0f;
#pragma unroll
            for (int index = 0; index < 8; ++index) total += partial_sum[element][index];
            logits[element][key] = total * score_scale;
        }
        __syncthreads();
    }

    if (element < query_count) {
        const int slot = element;
        const int query_last_key = position_base + query_base + slot;
        float maximum = -3.402823466e38F;
        for (int key = 0; key <= query_last_key; ++key)
            maximum = fmaxf(maximum, logits[slot][key]);
        float denominator = 0.0f;
        for (int key = 0; key <= query_last_key; ++key) {
            logits[slot][key] = expf(logits[slot][key] - maximum);
            denominator += logits[slot][key];
        }
        const float inverse = 1.0f / denominator;
        for (int key = 0; key <= query_last_key; ++key) logits[slot][key] *= inverse;
    }
    __syncthreads();

    const int value_offset = head * 256 + element;
    float accumulator[kQueries];
#pragma unroll
    for (int slot = 0; slot < kQueries; ++slot) accumulator[slot] = 0.0f;
    for (int key = 0; key <= last_key; ++key) {
        const float value = exact_value_cache[size_t(key) * width + value_offset];
#pragma unroll
        for (int slot = 0; slot < kQueries; ++slot)
            if (slot < query_count && key <= position_base + query_base + slot)
                accumulator[slot] = fmaf(logits[slot][key], value, accumulator[slot]);
    }
#pragma unroll
    for (int slot = 0; slot < kQueries; ++slot)
        if (slot < query_count)
            output[size_t(query_base + slot) * width + value_offset] = accumulator[slot];
}

// Stage 1: one block per (head, tile of keys).  Absorbs q into the latent
// space (q_eff = q @ W_uk), runs the online softmax over its tile, and
// writes (max, denominator, weighted latent sum) partials.
__global__ __launch_bounds__(256) void mla_decode_latent_partial_kernel(
    const float *__restrict__ query,
    const uint8_t *__restrict__ cache,
    const float *__restrict__ scales,
    const float *__restrict__ cache_f32,
    const float *__restrict__ w_uk,
    float *__restrict__ partial,
    int position,
    int tiles,
    int latent_dim) {
    const int head = blockIdx.x;
    const int tile = blockIdx.y;
    const int first = tile * kMlaDecodeTile;
    const int last = min(position, first + kMlaDecodeTile - 1);
    const int element = threadIdx.x;  // owns latent dims element and element+256
    const int lane = element & 31;
    const int warp = element >> 5;
    __shared__ float q_shared[256];
    __shared__ float partial_sum[8];
    __shared__ float broadcast;
    __shared__ float scale_shared[kMlaLatentGroups];

    q_shared[element] = query[head * 256 + element];
    __syncthreads();

    float qe0 = 0.0f, qe1 = 0.0f;
    for (int j = 0; j < 256; ++j) {
        const float qj = q_shared[j];
        const float *row = w_uk + (size_t(head) * 256 + j) * latent_dim;
        qe0 = fmaf(qj, row[element], qe0);
        qe1 = fmaf(qj, row[element + 256], qe1);
    }

    float maximum = -3.402823466e38F;
    float denominator = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;
    const float scale = rsqrtf(256.0f);
    for (int key = first; key <= last; ++key) {
        float k0, k1;
        if (cache_f32) {
            k0 = cache_f32[size_t(key) * latent_dim + element];
            k1 = cache_f32[size_t(key) * latent_dim + element + 256];
        } else {
            if (element < kMlaLatentGroups)
                scale_shared[element] = scales[size_t(key) * kMlaLatentGroups + element];
            __syncthreads();
            k0 = fp8_to_float(cache[size_t(key) * latent_dim + element]) *
                 scale_shared[element / kMlaLatentGroupSize];
            k1 = fp8_to_float(cache[size_t(key) * latent_dim + element + 256]) *
                 scale_shared[(element + 256) / kMlaLatentGroupSize];
        }
        const float dot = warp_sum(qe0 * k0 + qe1 * k1);
        if (!lane) partial_sum[warp] = dot;
        __syncthreads();
        if (!element) {
            float total = 0.0f;
#pragma unroll
            for (int index = 0; index < 8; ++index) total += partial_sum[index];
            broadcast = total * scale;
        }
        __syncthreads();
        const float score = broadcast;
        const float maximum_new = fmaxf(maximum, score);
        const float correction = expf(maximum - maximum_new);
        const float weight = expf(score - maximum_new);
        maximum = maximum_new;
        denominator = denominator * correction + weight;
        acc0 = fmaf(weight, k0, acc0 * correction);
        acc1 = fmaf(weight, k1, acc1 * correction);
    }

    float *dst = partial + (size_t(head) * tiles + tile) * (latent_dim + 2);
    if (!element) {
        dst[0] = maximum;
        dst[1] = denominator;
    }
    dst[2 + element] = acc0;
    dst[2 + element + 256] = acc1;
}

// Stage 2: one block per head.  Merges tile partials, then projects the
// normalized weighted latent sum through W_uv (rows transposed to
// [head_dim, latent_dim] so each output dimension streams one row).
__global__ __launch_bounds__(256) void mla_decode_latent_merge_kernel(
    const float *__restrict__ w_uv,
    const float *__restrict__ partial,
    float *__restrict__ output,
    int tiles,
    int latent_dim) {
    const int head = blockIdx.x;
    const int element = threadIdx.x;  // owns latent dims element and element+256
    const float *base = partial + size_t(head) * tiles * (latent_dim + 2);
    float maximum = -3.402823466e38F;
    float denominator = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int tile = 0; tile < tiles; ++tile) {
        const float *src = base + size_t(tile) * (latent_dim + 2);
        const float tile_max = src[0];
        const float tile_denominator = src[1];
        const float maximum_new = fmaxf(maximum, tile_max);
        const float correction = expf(maximum - maximum_new);
        const float tile_correction = expf(tile_max - maximum_new);
        maximum = maximum_new;
        denominator = denominator * correction + tile_denominator * tile_correction;
        acc0 = fmaf(src[2 + element], tile_correction, acc0 * correction);
        acc1 = fmaf(src[2 + element + 256], tile_correction, acc1 * correction);
    }
    const float inverse = 1.0f / denominator;
    __shared__ float acc[512];
    acc[element] = acc0 * inverse;
    acc[element + 256] = acc1 * inverse;
    __syncthreads();
    const float *row = w_uv + (size_t(head) * 256 + element) * latent_dim;
    float total = 0.0f;
    for (int c = 0; c < latent_dim; ++c) total = fmaf(acc[c], row[c], total);
    output[head * 256 + element] = total;
}

// Absorbed prefill: eight queries per block, latent cache keys, online
// softmax with the same causal masking as the expanded flash2 kernel.
__global__ __launch_bounds__(256) void mla_prefill_latent_kernel(
    const float *__restrict__ query,
    const uint8_t *__restrict__ cache,
    const float *__restrict__ scales,
    const float *__restrict__ cache_f32,
    const float *__restrict__ w_uk,
    const float *__restrict__ w_uv,
    float *__restrict__ output,
    int tokens,
    int position_base,
    int latent_dim) {
    const int head = blockIdx.x;
    const int query_base = blockIdx.y * 8;
    const int query_count = min(8, tokens - query_base);
    const int element = threadIdx.x;  // owns latent dims element and element+256
    const int lane = element & 31;
    const int warp = element >> 5;
    const int width = gridDim.x * 256;
    __shared__ float q_shared[256];
    __shared__ float partial_sum[8][8];
    __shared__ float scores[8];
    __shared__ float acc_shared[512];
    __shared__ float scale_shared[kMlaLatentGroups];

    float qe[8][2];
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) qe[slot][0] = qe[slot][1] = 0.0f;
    const float score_scale = rsqrtf(256.0f);
    for (int slot = 0; slot < query_count; ++slot) {
        const float *qrow = query + size_t(query_base + slot) * width + head * 256;
        q_shared[element] = qrow[element];
        __syncthreads();
        float e0 = 0.0f, e1 = 0.0f;
        for (int j = 0; j < 256; ++j) {
            const float qj = q_shared[j];
            const float *row = w_uk + (size_t(head) * 256 + j) * latent_dim;
            e0 = fmaf(qj, row[element], e0);
            e1 = fmaf(qj, row[element + 256], e1);
        }
        qe[slot][0] = e0;
        qe[slot][1] = e1;
        __syncthreads();
    }

    float maximum[8], denominator[8], acc0[8], acc1[8];
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        maximum[slot] = -3.402823466e38F;
        denominator[slot] = 0.0f;
        acc0[slot] = acc1[slot] = 0.0f;
    }
    const int last_key = position_base + query_base + query_count - 1;
    for (int key = 0; key <= last_key; ++key) {
        float k0, k1;
        if (cache_f32) {
            k0 = cache_f32[size_t(key) * latent_dim + element];
            k1 = cache_f32[size_t(key) * latent_dim + element + 256];
        } else {
            if (element < kMlaLatentGroups)
                scale_shared[element] = scales[size_t(key) * kMlaLatentGroups + element];
            __syncthreads();
            k0 = fp8_to_float(cache[size_t(key) * latent_dim + element]) *
                 scale_shared[element / kMlaLatentGroupSize];
            k1 = fp8_to_float(cache[size_t(key) * latent_dim + element + 256]) *
                 scale_shared[(element + 256) / kMlaLatentGroupSize];
        }
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const float dot = warp_sum(qe[slot][0] * k0 + qe[slot][1] * k1);
            if (!lane) partial_sum[slot][warp] = dot;
        }
        __syncthreads();
        if (element < 8) {
            float total = 0.0f;
#pragma unroll
            for (int index = 0; index < 8; ++index) total += partial_sum[element][index];
            scores[element] = total * score_scale;
        }
        __syncthreads();
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            if (slot < query_count && key <= position_base + query_base + slot) {
                const float score = scores[slot];
                const float maximum_new = fmaxf(maximum[slot], score);
                const float correction = expf(maximum[slot] - maximum_new);
                const float weight = expf(score - maximum_new);
                maximum[slot] = maximum_new;
                denominator[slot] = denominator[slot] * correction + weight;
                acc0[slot] = fmaf(weight, k0, acc0[slot] * correction);
                acc1[slot] = fmaf(weight, k1, acc1[slot] * correction);
            }
        }
        __syncthreads();
    }

    for (int slot = 0; slot < query_count; ++slot) {
        const float inverse = 1.0f / denominator[slot];
        acc_shared[element] = acc0[slot] * inverse;
        acc_shared[element + 256] = acc1[slot] * inverse;
        __syncthreads();
        const float *row = w_uv + (size_t(head) * 256 + element) * latent_dim;
        float total = 0.0f;
        for (int c = 0; c < latent_dim; ++c) total = fmaf(acc_shared[c], row[c], total);
        output[size_t(query_base + slot) * width + head * 256 + element] = total;
        __syncthreads();
    }
}

cudaError_t mla_store_latent(
    const float *latent,
    uint8_t *cache,
    float *scales,
    float *cache_f32,
    int tokens,
    int position_base,
    int latent_dim,
    cudaStream_t stream) {
    if (!latent || tokens <= 0 || position_base < 0 ||
        position_base + tokens > kMlaMaxContext || latent_dim != kMlaLatentDim ||
        (!cache && !cache_f32))
        return cudaErrorInvalidValue;
    mla_store_latent_kernel<<<tokens, latent_dim, 0, stream>>>(
        latent, cache, scales, cache_f32, position_base, latent_dim);
    return cudaGetLastError();
}

cudaError_t mla_decode_latent(
    const float *query,
    const float *latent,
    const float *expanded_kv,
    uint8_t *cache,
    float *scales,
    float *cache_f32,
    float *exact_value_cache,
    const float *w_uk,
    const float *w_uv,
    float *partial,
    float *output,
    int position,
    int heads,
    int head_dim,
    int latent_dim,
    cudaStream_t stream) {
    if (position < 0 || position >= kMlaMaxContext) return cudaErrorInvalidValue;
    if (heads != kKdaHeads || head_dim != kMlaHeadDim || latent_dim != kMlaLatentDim ||
        !query || !latent || !w_uk || !w_uv || !partial || !output ||
        (!!expanded_kv != !!exact_value_cache) ||
        (expanded_kv && position >= kMlaExactContext))
        return cudaErrorInvalidValue;
    mla_store_latent_kernel<<<1, latent_dim, 0, stream>>>(
        latent, cache, scales, cache_f32, position, latent_dim);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    if (expanded_kv) {
        mla_store_value_batch_kernel<<<dim3(heads, 1), head_dim, 0, stream>>>(
            expanded_kv, exact_value_cache, position, heads, head_dim);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        mla_decode_latent_exact_value_kernel<<<heads, head_dim, 0, stream>>>(
            query, cache, scales, cache_f32, exact_value_cache, w_uk, output,
            position, latent_dim);
        return cudaGetLastError();
    }
    const int tiles = (position + kMlaDecodeTile) / kMlaDecodeTile;
    mla_decode_latent_partial_kernel<<<dim3(heads, tiles), 256, 0, stream>>>(
        query, cache, scales, cache_f32, w_uk, partial, position, tiles, latent_dim);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    mla_decode_latent_merge_kernel<<<heads, 256, 0, stream>>>(
        w_uv, partial, output, tiles, latent_dim);
    return cudaGetLastError();
}

cudaError_t mla_prefill_latent(
    const float *query,
    const float *latents,
    const float *expanded_kv,
    uint8_t *cache,
    float *scales,
    float *cache_f32,
    float *exact_value_cache,
    const float *w_uk,
    const float *w_uv,
    float *output,
    int tokens,
    int position_base,
    int heads,
    int head_dim,
    int latent_dim,
    cudaStream_t stream) {
    if (tokens <= 0 || tokens > 64 || position_base < 0 ||
        position_base + tokens > kMlaMaxContext || heads != kKdaHeads ||
        head_dim != kMlaHeadDim || latent_dim != kMlaLatentDim ||
        !query || !latents || !w_uk || !w_uv || !output ||
        (!!expanded_kv != !!exact_value_cache) ||
        (expanded_kv && position_base + tokens > kMlaExactContext))
        return cudaErrorInvalidValue;
    mla_store_latent_kernel<<<tokens, latent_dim, 0, stream>>>(
        latents, cache, scales, cache_f32, position_base, latent_dim);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    if (expanded_kv) {
        mla_store_value_batch_kernel<<<dim3(heads, tokens), head_dim, 0, stream>>>(
            expanded_kv, exact_value_cache, position_base, heads, head_dim);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        mla_prefill_latent_exact_value_kernel
            <<<dim3(heads, (tokens + 7) / 8), head_dim, 0, stream>>>(
                query, cache, scales, cache_f32, exact_value_cache, w_uk,
                output, tokens, position_base, latent_dim);
        return cudaGetLastError();
    }
    mla_prefill_latent_kernel<<<dim3(heads, (tokens + 7) / 8), 256, 0, stream>>>(
        query, cache, scales, cache_f32, w_uk, w_uv, output, tokens, position_base, latent_dim);
    return cudaGetLastError();
}

}  // namespace insignia::glm53
