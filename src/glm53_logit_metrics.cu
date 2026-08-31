#include "insignia_glm53_logit_metrics.cuh"

#include <algorithm>
#include <climits>
#include <cmath>

#include <math_constants.h>

namespace insignia::glm53 {
namespace {

constexpr int kThreads = 256;
constexpr int kMaximumBlocks = 256;
constexpr int kRowMaximumBlocks = 64;

struct alignas(64) Moments {
    float left_max;
    float right_max;
    float left_min;
    float right_min;
    int32_t left_argmax;
    int32_t right_argmax;
    double left_sum;
    double right_sum;
    double left_square;
    double right_square;
    double cross;
    double difference_square;
};
static_assert(sizeof(Moments) == 128);

struct ProbabilitySums {
    double left;
    double right;
};

struct DivergenceSums {
    double kl_left_right;
    double kl_right_left;
    double js;
    double left_entropy;
    double right_entropy;
};

union alignas(64) Partial {
    Moments moments;
    ProbabilitySums probabilities;
    DivergenceSums divergences;
};
static_assert(sizeof(Partial) == 128);

struct alignas(64) State {
    Moments moments;
    double left_sum_exp;
    double right_sum_exp;
    double left_logsumexp;
    double right_logsumexp;
};

__device__ __forceinline__ Moments empty_moments() {
    Moments value{};
    value.left_max = -CUDART_INF_F;
    value.right_max = -CUDART_INF_F;
    value.left_min = CUDART_INF_F;
    value.right_min = CUDART_INF_F;
    value.left_argmax = INT_MAX;
    value.right_argmax = INT_MAX;
    return value;
}

__device__ __forceinline__ Moments combine(Moments left, const Moments &right) {
    if (right.left_max > left.left_max ||
        (right.left_max == left.left_max && right.left_argmax < left.left_argmax)) {
        left.left_max = right.left_max;
        left.left_argmax = right.left_argmax;
    }
    if (right.right_max > left.right_max ||
        (right.right_max == left.right_max && right.right_argmax < left.right_argmax)) {
        left.right_max = right.right_max;
        left.right_argmax = right.right_argmax;
    }
    left.left_min = fminf(left.left_min, right.left_min);
    left.right_min = fminf(left.right_min, right.right_min);
    left.left_sum += right.left_sum;
    left.right_sum += right.right_sum;
    left.left_square += right.left_square;
    left.right_square += right.right_square;
    left.cross += right.cross;
    left.difference_square += right.difference_square;
    return left;
}

__global__ void moments_kernel(const float *__restrict__ left,
                               const float *__restrict__ right, int count,
                               Partial *__restrict__ partial) {
    __shared__ Moments shared[kThreads];
    Moments local = empty_moments();
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * gridDim.x) {
        const float a = left[index];
        const float b = right[index];
        if (a > local.left_max || (a == local.left_max && index < local.left_argmax)) {
            local.left_max = a;
            local.left_argmax = index;
        }
        if (b > local.right_max || (b == local.right_max && index < local.right_argmax)) {
            local.right_max = b;
            local.right_argmax = index;
        }
        local.left_min = fminf(local.left_min, a);
        local.right_min = fminf(local.right_min, b);
        const double da = static_cast<double>(a);
        const double db = static_cast<double>(b);
        const double difference = da - db;
        local.left_sum += da;
        local.right_sum += db;
        local.left_square = fma(da, da, local.left_square);
        local.right_square = fma(db, db, local.right_square);
        local.cross = fma(da, db, local.cross);
        local.difference_square = fma(difference, difference, local.difference_square);
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int offset = kThreads / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset)
            shared[threadIdx.x] = combine(shared[threadIdx.x], shared[threadIdx.x + offset]);
        __syncthreads();
    }
    if (!threadIdx.x) partial[blockIdx.x].moments = shared[0];
}

__global__ void reduce_moments_kernel(const Partial *__restrict__ partial,
                                      int blocks, State *__restrict__ state) {
    __shared__ Moments shared[kThreads];
    Moments local = empty_moments();
    for (int index = threadIdx.x; index < blocks; index += kThreads)
        local = combine(local, partial[index].moments);
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int offset = kThreads / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset)
            shared[threadIdx.x] = combine(shared[threadIdx.x], shared[threadIdx.x + offset]);
        __syncthreads();
    }
    if (!threadIdx.x) state->moments = shared[0];
}

__global__ void probability_kernel(const float *__restrict__ left,
                                   const float *__restrict__ right, int count,
                                   const State *__restrict__ state,
                                   Partial *__restrict__ partial) {
    __shared__ ProbabilitySums shared[kThreads];
    ProbabilitySums local{};
    const double left_max = static_cast<double>(state->moments.left_max);
    const double right_max = static_cast<double>(state->moments.right_max);
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * gridDim.x) {
        local.left += exp(static_cast<double>(left[index]) - left_max);
        local.right += exp(static_cast<double>(right[index]) - right_max);
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int offset = kThreads / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x].left += shared[threadIdx.x + offset].left;
            shared[threadIdx.x].right += shared[threadIdx.x + offset].right;
        }
        __syncthreads();
    }
    if (!threadIdx.x) partial[blockIdx.x].probabilities = shared[0];
}

__global__ void reduce_probability_kernel(const Partial *__restrict__ partial,
                                          int blocks, State *__restrict__ state) {
    __shared__ ProbabilitySums shared[kThreads];
    ProbabilitySums local{};
    for (int index = threadIdx.x; index < blocks; index += kThreads) {
        local.left += partial[index].probabilities.left;
        local.right += partial[index].probabilities.right;
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int offset = kThreads / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x].left += shared[threadIdx.x + offset].left;
            shared[threadIdx.x].right += shared[threadIdx.x + offset].right;
        }
        __syncthreads();
    }
    if (!threadIdx.x) {
        state->left_sum_exp = shared[0].left;
        state->right_sum_exp = shared[0].right;
        state->left_logsumexp = static_cast<double>(state->moments.left_max) + log(shared[0].left);
        state->right_logsumexp = static_cast<double>(state->moments.right_max) + log(shared[0].right);
    }
}

__device__ __forceinline__ DivergenceSums add(DivergenceSums left,
                                               const DivergenceSums &right) {
    left.kl_left_right += right.kl_left_right;
    left.kl_right_left += right.kl_right_left;
    left.js += right.js;
    left.left_entropy += right.left_entropy;
    left.right_entropy += right.right_entropy;
    return left;
}

__global__ void divergence_kernel(const float *__restrict__ left,
                                  const float *__restrict__ right, int count,
                                  const State *__restrict__ state,
                                  Partial *__restrict__ partial) {
    __shared__ DivergenceSums shared[kThreads];
    DivergenceSums local{};
    const double left_log_z = state->left_logsumexp;
    const double right_log_z = state->right_logsumexp;
    constexpr double kLn2 = 0.693147180559945309417232121458176568;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * gridDim.x) {
        const double log_p = static_cast<double>(left[index]) - left_log_z;
        const double log_q = static_cast<double>(right[index]) - right_log_z;
        const double p = exp(log_p);
        const double q = exp(log_q);
        const double maximum = fmax(log_p, log_q);
        const double log_mixture = maximum +
            log(exp(log_p - maximum) + exp(log_q - maximum)) - kLn2;
        local.kl_left_right += p * (log_p - log_q);
        local.kl_right_left += q * (log_q - log_p);
        local.js += 0.5 * (p * (log_p - log_mixture) +
                           q * (log_q - log_mixture));
        local.left_entropy -= p * log_p;
        local.right_entropy -= q * log_q;
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int offset = kThreads / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset)
            shared[threadIdx.x] = add(shared[threadIdx.x], shared[threadIdx.x + offset]);
        __syncthreads();
    }
    if (!threadIdx.x) partial[blockIdx.x].divergences = shared[0];
}

__global__ void finish_kernel(const Partial *__restrict__ partial, int blocks,
                              int count, const State *__restrict__ state,
                              LogitMetrics *__restrict__ result) {
    __shared__ DivergenceSums shared[kThreads];
    DivergenceSums local{};
    for (int index = threadIdx.x; index < blocks; index += kThreads)
        local = add(local, partial[index].divergences);
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int offset = kThreads / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset)
            shared[threadIdx.x] = add(shared[threadIdx.x], shared[threadIdx.x + offset]);
        __syncthreads();
    }
    if (threadIdx.x) return;

    const Moments &moments = state->moments;
    const double inverse_count = 1.0 / static_cast<double>(count);
    const double left_mean = moments.left_sum * inverse_count;
    const double right_mean = moments.right_sum * inverse_count;
    const double left_centered_square = moments.left_min == moments.left_max
        ? 0.0
        : fmax(0.0, moments.left_square -
                       moments.left_sum * moments.left_sum * inverse_count);
    const double right_centered_square = moments.right_min == moments.right_max
        ? 0.0
        : fmax(0.0, moments.right_square -
                       moments.right_sum * moments.right_sum * inverse_count);
    const double centered_cross = moments.cross -
        moments.left_sum * moments.right_sum * inverse_count;
    const double mse = moments.difference_square * inverse_count;
    const double mean_difference = left_mean - right_mean;

    LogitMetrics output{};
    output.left_max = moments.left_max;
    output.right_max = moments.right_max;
    output.left_logsumexp = state->left_logsumexp;
    output.right_logsumexp = state->right_logsumexp;
    output.left_mean = left_mean;
    output.right_mean = right_mean;
    output.left_centered_rms = sqrt(left_centered_square * inverse_count);
    output.right_centered_rms = sqrt(right_centered_square * inverse_count);
    output.mse = mse;
    output.centered_mse = fmax(0.0, mse - mean_difference * mean_difference);
    const double raw_norm = sqrt(moments.left_square * moments.right_square);
    output.raw_cosine = raw_norm > 0.0
        ? moments.cross / raw_norm
        : (moments.left_square == 0.0 && moments.right_square == 0.0 ? 1.0 : 0.0);
    const double centered_norm = sqrt(left_centered_square * right_centered_square);
    output.centered_cosine = centered_norm > 0.0
        ? centered_cross / centered_norm
        : (left_centered_square == 0.0 && right_centered_square == 0.0 ? 1.0 : 0.0);
    // These are non-negative analytically; clamp only reduction roundoff.
    output.kl_left_right = fmax(0.0, shared[0].kl_left_right);
    output.kl_right_left = fmax(0.0, shared[0].kl_right_left);
    output.js = fmax(0.0, shared[0].js);
    output.left_entropy = shared[0].left_entropy;
    output.right_entropy = shared[0].right_entropy;
    output.left_top1_probability = exp(output.left_max - output.left_logsumexp);
    output.right_top1_probability = exp(output.right_max - output.right_logsumexp);
    output.left_argmax = moments.left_argmax;
    output.right_argmax = moments.right_argmax;
    *result = output;
}

int block_count(int count) noexcept {
    return std::min(kMaximumBlocks, (count + kThreads - 1) / kThreads);
}

struct alignas(32) LogSumExpState {
    double maximum;
    double scaled_sum;
    int32_t argmax;
};
static_assert(sizeof(LogSumExpState) == 32);

__device__ __forceinline__ LogSumExpState empty_logsumexp() {
    LogSumExpState value{};
    value.maximum = -CUDART_INF;
    value.argmax = INT_MAX;
    return value;
}

__device__ __forceinline__ LogSumExpState combine(
    LogSumExpState left, const LogSumExpState &right) {
    if (right.scaled_sum == 0.0) return left;
    if (left.scaled_sum == 0.0) return right;
    if (right.maximum > left.maximum) {
        left.scaled_sum = right.scaled_sum +
                          left.scaled_sum * exp(left.maximum - right.maximum);
        left.maximum = right.maximum;
        left.argmax = right.argmax;
    } else {
        left.scaled_sum += right.scaled_sum * exp(right.maximum - left.maximum);
        if (right.maximum == left.maximum && right.argmax < left.argmax)
            left.argmax = right.argmax;
    }
    return left;
}

__global__ void row_logsumexp_kernel(const float *__restrict__ logits, int rows,
                                     int count, int blocks_per_row,
                                     LogSumExpState *__restrict__ partial) {
    __shared__ LogSumExpState shared[kThreads];
    const int row = blockIdx.y;
    LogSumExpState local = empty_logsumexp();
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * blocks_per_row) {
        LogSumExpState item{};
        item.maximum = static_cast<double>(logits[static_cast<std::size_t>(row) * count + index]);
        item.scaled_sum = 1.0;
        item.argmax = index;
        local = combine(local, item);
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int offset = kThreads / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset)
            shared[threadIdx.x] = combine(shared[threadIdx.x], shared[threadIdx.x + offset]);
        __syncthreads();
    }
    if (!threadIdx.x)
        partial[static_cast<std::size_t>(row) * blocks_per_row + blockIdx.x] = shared[0];
}

__global__ void finish_row_logsumexp_kernel(
    const LogSumExpState *__restrict__ partial, int rows, int blocks_per_row,
    LogitRowStats *__restrict__ result) {
    __shared__ LogSumExpState shared[kThreads];
    const int row = blockIdx.x;
    LogSumExpState local = empty_logsumexp();
    for (int index = threadIdx.x; index < blocks_per_row; index += kThreads)
        local = combine(local, partial[static_cast<std::size_t>(row) * blocks_per_row + index]);
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int offset = kThreads / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset)
            shared[threadIdx.x] = combine(shared[threadIdx.x], shared[threadIdx.x + offset]);
        __syncthreads();
    }
    if (!threadIdx.x) {
        LogitRowStats output{};
        output.maximum = shared[0].maximum;
        output.logsumexp = shared[0].maximum + log(shared[0].scaled_sum);
        output.top1_probability = 1.0 / shared[0].scaled_sum;
        output.argmax = shared[0].argmax;
        result[row] = output;
    }
}

int row_block_count(int count) noexcept {
    return std::min(kRowMaximumBlocks, (count + kThreads - 1) / kThreads);
}

}  // namespace

std::size_t logit_metrics_workspace_bytes(int count) noexcept {
    return count > 0 ? static_cast<std::size_t>(block_count(count)) * sizeof(Partial) +
                           sizeof(State)
                     : 0;
}

cudaError_t logit_metrics_async(const float *left, const float *right, int count,
                                void *workspace, LogitMetrics *result,
                                cudaStream_t stream) noexcept {
    if (!left || !right || count <= 0 || !workspace || !result)
        return cudaErrorInvalidValue;
    const int blocks = block_count(count);
    auto *partial = static_cast<Partial *>(workspace);
    auto *state = reinterpret_cast<State *>(partial + blocks);
    moments_kernel<<<blocks, kThreads, 0, stream>>>(left, right, count, partial);
    reduce_moments_kernel<<<1, kThreads, 0, stream>>>(partial, blocks, state);
    probability_kernel<<<blocks, kThreads, 0, stream>>>(left, right, count, state, partial);
    reduce_probability_kernel<<<1, kThreads, 0, stream>>>(partial, blocks, state);
    divergence_kernel<<<blocks, kThreads, 0, stream>>>(left, right, count, state, partial);
    finish_kernel<<<1, kThreads, 0, stream>>>(partial, blocks, count, state, result);
    return cudaGetLastError();
}

std::size_t logit_row_stats_workspace_bytes(int rows, int count) noexcept {
    return rows > 0 && count > 0
        ? static_cast<std::size_t>(rows) * row_block_count(count) * sizeof(LogSumExpState)
        : 0;
}

cudaError_t logit_row_stats_async(const float *logits, int rows, int count,
                                  void *workspace, LogitRowStats *result,
                                  cudaStream_t stream) noexcept {
    if (!logits || rows <= 0 || count <= 0 || !workspace || !result)
        return cudaErrorInvalidValue;
    const int blocks = row_block_count(count);
    auto *partial = static_cast<LogSumExpState *>(workspace);
    row_logsumexp_kernel<<<dim3(blocks, rows), kThreads, 0, stream>>>(
        logits, rows, count, blocks, partial);
    finish_row_logsumexp_kernel<<<rows, kThreads, 0, stream>>>(
        partial, rows, blocks, result);
    return cudaGetLastError();
}

}  // namespace insignia::glm53
