// End-to-end synthetic execution ceiling for the native INT8 Falsifier-MoE
// sidecar.  Unlike benchmark_falsifier_vnni.cpp, this includes dynamic
// activation quantization/dequantization, nonlinearities, routing, mHC
// Sinkhorn mixing, block-depth attention, absorbed causal MLA, and a
// layer-synchronous persistent worker schedule.  Weights and features are
// deterministic synthetic values; this is a runtime ceiling, not a quality
// result or a trained-controller integration test.

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

constexpr int kWidth = 192;
constexpr int kHeads = 4;
constexpr int kHeadDim = 32;
constexpr int kLatent = 64;
constexpr int kMaxHistory = 336;
constexpr int kScratchWidth = 768;

#if defined(INSIGNIA_FALSIFIER_PROFILE)
#define INSIGNIA_PROFILE_NOINLINE __attribute__((noinline))
#else
#define INSIGNIA_PROFILE_NOINLINE
#endif

struct Matrix {
    int rows;
    int logical_cols;
    int cols;
    std::vector<int8_t> weight;
    std::vector<int32_t> correction;
    std::vector<float> scale;

    Matrix(int output, int input, std::mt19937 &random)
        : rows(output), logical_cols(input), cols((input + 31) & ~31),
          weight(static_cast<size_t>(output) * cols, 0), correction(output),
          scale(output) {
        std::uniform_int_distribution<int> quantized(-127, 127);
        std::uniform_real_distribution<float> scales(0.0015f, 0.0035f);
        for (int row = 0; row < rows; ++row) {
            int32_t sum = 0;
            for (int column = 0; column < logical_cols; ++column) {
                const int8_t value = static_cast<int8_t>(quantized(random));
                weight[static_cast<size_t>(row) * cols + column] = value;
                sum += value;
            }
            correction[row] = 128 * sum;
            scale[row] = scales(random);
        }
    }

    uint64_t macs() const { return static_cast<uint64_t>(rows) * cols; }
    uint64_t logical_macs() const {
        return static_cast<uint64_t>(rows) * logical_cols;
    }
};

__attribute__((noinline)) static void vnni_rows(
    const int8_t *__restrict weight, const int32_t *__restrict correction,
    int rows, int cols, const int8_t *__restrict input,
    int32_t *__restrict output) {
    const __m256i flip = _mm256_set1_epi8(static_cast<char>(0x80));
    const auto horizontal_sum = [](__m256i value) {
        __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value),
                                   _mm256_extracti128_si256(value, 1));
        sum = _mm_hadd_epi32(sum, sum);
        sum = _mm_hadd_epi32(sum, sum);
        return _mm_cvtsi128_si32(sum);
    };
    int row = 0;
    for (; row + 4 <= rows; row += 4) {
        __m256i accumulator0 = _mm256_setzero_si256();
        __m256i accumulator1 = _mm256_setzero_si256();
        __m256i accumulator2 = _mm256_setzero_si256();
        __m256i accumulator3 = _mm256_setzero_si256();
        const int8_t *row0 = weight + static_cast<size_t>(row) * cols;
        const int8_t *row1 = row0 + cols;
        const int8_t *row2 = row1 + cols;
        const int8_t *row3 = row2 + cols;
        for (int column = 0; column < cols; column += 32) {
            const __m256i signed_input = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(input + column));
            const __m256i unsigned_input = _mm256_xor_si256(signed_input, flip);
            accumulator0 = _mm256_dpbusd_epi32(accumulator0, unsigned_input,
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row0 + column)));
            accumulator1 = _mm256_dpbusd_epi32(accumulator1, unsigned_input,
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row1 + column)));
            accumulator2 = _mm256_dpbusd_epi32(accumulator2, unsigned_input,
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row2 + column)));
            accumulator3 = _mm256_dpbusd_epi32(accumulator3, unsigned_input,
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row3 + column)));
        }
        output[row] = horizontal_sum(accumulator0) - correction[row];
        output[row + 1] = horizontal_sum(accumulator1) - correction[row + 1];
        output[row + 2] = horizontal_sum(accumulator2) - correction[row + 2];
        output[row + 3] = horizontal_sum(accumulator3) - correction[row + 3];
    }
    for (; row < rows; ++row) {
        __m256i accumulator = _mm256_setzero_si256();
        const int8_t *row_weight = weight + static_cast<size_t>(row) * cols;
        for (int column = 0; column < cols; column += 32) {
            const __m256i signed_input = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(input + column));
            const __m256i unsigned_input = _mm256_xor_si256(signed_input, flip);
            accumulator = _mm256_dpbusd_epi32(accumulator, unsigned_input,
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row_weight + column)));
        }
        output[row] = horizontal_sum(accumulator) - correction[row];
    }
}

__attribute__((noinline)) static float quantize_vector(
    const float *__restrict input, int logical, int padded,
    int8_t *__restrict output) {
    if ((logical & 7) != 0 || padded < logical || (padded & 31) != 0)
        throw std::runtime_error("invalid dynamic INT8 width");
    __m256 maximum = _mm256_setzero_ps();
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (int column = 0; column < logical; column += 8) {
        const __m256 value = _mm256_loadu_ps(input + column);
        maximum = _mm256_max_ps(maximum, _mm256_andnot_ps(sign, value));
    }
    alignas(32) float maxima[8];
    _mm256_store_ps(maxima, maximum);
    float absmax = maxima[0];
    for (int lane = 1; lane < 8; ++lane) absmax = std::max(absmax, maxima[lane]);
    const float scale = absmax > 1.0e-12f ? absmax / 127.0f : 1.0f;
    const __m256 inverse = _mm256_set1_ps(1.0f / scale);
    alignas(32) int32_t integers[8];
    for (int column = 0; column < logical; column += 8) {
        const __m256 value = _mm256_loadu_ps(input + column);
        const __m256i rounded = _mm256_cvtps_epi32(_mm256_mul_ps(value, inverse));
        _mm256_store_si256(reinterpret_cast<__m256i *>(integers), rounded);
        for (int lane = 0; lane < 8; ++lane)
            output[column + lane] = static_cast<int8_t>(
                std::clamp(integers[lane], -127, 127));
    }
    std::memset(output + logical, 0, static_cast<size_t>(padded - logical));
    return scale;
}

static void linear_quantized(const Matrix &matrix, const int8_t *input,
                             float input_scale, int32_t *accumulator,
                             float *output) {
    vnni_rows(matrix.weight.data(), matrix.correction.data(), matrix.rows,
              matrix.cols, input, accumulator);
    for (int row = 0; row < matrix.rows; ++row)
        output[row] = static_cast<float>(accumulator[row])
            * input_scale * matrix.scale[row];
}

static void expert_linear_quantized(
    const Matrix &pool, int expert, int expert_rows, const int8_t *input,
    float input_scale, int32_t *accumulator, float *output) {
    const size_t row = static_cast<size_t>(expert) * expert_rows;
    vnni_rows(pool.weight.data() + row * pool.cols,
              pool.correction.data() + row, expert_rows, pool.cols, input,
              accumulator);
    for (int output_row = 0; output_row < expert_rows; ++output_row)
        output[output_row] = static_cast<float>(accumulator[output_row])
            * input_scale * pool.scale[row + output_row];
}

struct Weights {
    Matrix logit{192, 208, random};
    Matrix hidden{192, 64, random};
    Matrix cache{192, 144, random};
    Matrix router_tail{192, 32, random};
    Matrix candidate_a{64, 32, random};
    Matrix candidate_b{192, 64, random};

    Matrix depth_q{128, 192, random};
    Matrix depth_k{128, 192, random};
    Matrix depth_v{128, 192, random};
    Matrix depth_out{192, 128, random};
    Matrix mhc_dynamic{16, 192, random};

    Matrix mla_q{128, 192, random};
    Matrix mla_kv_down{64, 192, random};
    Matrix mla_kv_up{256, 64, random};
    Matrix mla_gate{128, 192, random};
    Matrix mla_out{192, 128, random};

    Matrix moe_router{256, 192, random};
    Matrix latent_down{96, 192, random};
    Matrix latent_up{192, 96, random};
    Matrix shared_gate_up{768, 192, random};
    Matrix shared_down{192, 384, random};
    Matrix expert_gate_up{256 * 256, 96, random};
    Matrix expert_down{256 * 96, 128, random};
    Matrix heads{107, 192, random};

    uint64_t event_macs(bool logical) const {
        const auto count = [logical](const Matrix &matrix) {
            return logical ? matrix.logical_macs() : matrix.macs();
        };
        const uint64_t encoder = count(logit) + count(hidden) + count(cache)
            + count(router_tail) + 32 * (count(candidate_a) + count(candidate_b));
        const uint64_t mla = count(mla_q) + count(mla_kv_down)
            + count(mla_kv_up) + count(mla_gate) + count(mla_out);
        const uint64_t moe = count(moe_router) + count(latent_down)
            + count(latent_up) + count(shared_gate_up) + count(shared_down)
            + 2 * (static_cast<uint64_t>(256) * 96
                   + static_cast<uint64_t>(96) * 128);
        uint64_t cells = 0;
        for (uint64_t sources = 1; sources <= 3; ++sources)
            cells += count(depth_q) + sources * (count(depth_k) + count(depth_v))
                + count(depth_out) + 2 * count(mhc_dynamic) + mla + moe;
        const uint64_t final_depth = count(depth_q)
            + 4 * (count(depth_k) + count(depth_v)) + count(depth_out);
        return encoder + cells + final_depth + count(heads);
    }

private:
    inline static std::mt19937 random{0x53f17a2u};
};

using Wide = std::array<float, kScratchWidth>;
using QWide = std::array<int8_t, kScratchWidth>;
using AccWide = std::array<int32_t, kScratchWidth>;

struct Scratch {
    QWide quantized{};
    AccWide accumulator{};
    Wide input{};
    Wide temp0{};
    Wide temp1{};
    Wide temp2{};
    Wide temp3{};
    Wide temp4{};
    std::array<std::array<float, kWidth>, 4> streams{};
    std::array<std::array<float, kWidth>, 4> block_history{};
    std::array<std::array<float, 128>, 4> depth_keys{};
    std::array<std::array<float, 128>, 4> depth_values{};
    std::array<std::array<float, kLatent>, kHeads> q_absorbed{};
    std::array<std::array<float, kLatent>, kHeads> latent_sum{};
    std::array<std::array<float, kMaxHistory>, kHeads> attention_weights{};
    std::array<std::array<std::array<float, kLatent>, kMaxHistory>, 3> latents{};
    std::array<float, 16> mix_logits{};
    std::array<float, 16> mix_matrix{};
    std::array<std::array<float, kWidth>, 4> mixed{};
    std::array<std::array<float, 256>, 2> expert_gate{};
    std::array<std::array<float, 128>, 2> expert_activation{};
    std::array<std::array<float, 96>, 2> expert_output{};

    explicit Scratch(int seed) {
        std::mt19937 random(seed);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        for (float &value : input) value = distribution(random);
        for (auto &repeat : latents)
            for (auto &token : repeat)
                for (float &value : token) value = distribution(random);
    }
};

static void linear(const Matrix &matrix, const float *input, float *output,
                   Scratch &scratch) {
    const float scale = quantize_vector(input, matrix.logical_cols, matrix.cols,
                                        scratch.quantized.data());
    linear_quantized(matrix, scratch.quantized.data(), scale,
                     scratch.accumulator.data(), output);
}

static void rms_norm(float *value, int width) {
    float square = 0.0f;
    for (int i = 0; i < width; ++i) square = std::fma(value[i], value[i], square);
    const float scale = 1.0f / std::sqrt(square / width + 1.0e-6f);
    for (int i = 0; i < width; ++i) value[i] *= scale;
}

static float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

static void situ_glu(const float *gate_up, float *output, int hidden) {
    for (int i = 0; i < hidden; ++i) {
        const float gate = gate_up[i];
        const float up = gate_up[i + hidden];
        output[i] = (4.0f * std::tanh(gate * 0.25f) * sigmoid(gate))
            * (25.0f * std::tanh(up * 0.04f));
    }
}

static void silu(float *value, int width) {
    for (int i = 0; i < width; ++i) value[i] *= sigmoid(value[i]);
}

static void collapse_streams(const Scratch &scratch, float *output) {
    for (int d = 0; d < kWidth; ++d)
        output[d] = 0.25f * (scratch.streams[0][d] + scratch.streams[1][d]
                            + scratch.streams[2][d] + scratch.streams[3][d]);
}

INSIGNIA_PROFILE_NOINLINE static void mix_streams(
    const Weights &weights, Scratch &scratch, const float *context) {
    linear(weights.mhc_dynamic, context, scratch.mix_logits.data(), scratch);
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column) {
            const float diagonal = row == column ? 4.0f : 0.0f;
            scratch.mix_matrix[row * 4 + column] = std::exp(std::clamp(
                scratch.mix_logits[row * 4 + column] + diagonal, -12.0f, 12.0f));
        }
    for (int iteration = 0; iteration < 6; ++iteration) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int column = 0; column < 4; ++column)
                sum += scratch.mix_matrix[row * 4 + column];
            for (int column = 0; column < 4; ++column)
                scratch.mix_matrix[row * 4 + column] /= std::max(sum, 1.0e-12f);
        }
        for (int column = 0; column < 4; ++column) {
            float sum = 0.0f;
            for (int row = 0; row < 4; ++row)
                sum += scratch.mix_matrix[row * 4 + column];
            for (int row = 0; row < 4; ++row)
                scratch.mix_matrix[row * 4 + column] /= std::max(sum, 1.0e-12f);
        }
    }
    for (int row = 0; row < 4; ++row)
        for (int d = 0; d < kWidth; ++d) {
            float sum = 0.0f;
            for (int column = 0; column < 4; ++column)
                sum = std::fma(scratch.mix_matrix[row * 4 + column],
                               scratch.streams[column][d], sum);
            scratch.mixed[row][d] = sum;
        }
    scratch.streams = scratch.mixed;
}

INSIGNIA_PROFILE_NOINLINE static void depth_attention(
    const Weights &weights, Scratch &scratch, int source_count, float *output) {
    const float *current = scratch.block_history[source_count - 1].data();
    linear(weights.depth_q, current, scratch.temp0.data(), scratch);
    for (int source = 0; source < source_count; ++source) {
        linear(weights.depth_k, scratch.block_history[source].data(),
               scratch.depth_keys[source].data(), scratch);
        linear(weights.depth_v, scratch.block_history[source].data(),
               scratch.depth_values[source].data(), scratch);
    }
    constexpr float inverse_root = 0.1767766952966369f;
    for (int head = 0; head < kHeads; ++head) {
        float score[4];
        float maximum = -std::numeric_limits<float>::infinity();
        for (int source = 0; source < source_count; ++source) {
            float sum = 0.0f;
            for (int d = 0; d < kHeadDim; ++d)
                sum = std::fma(scratch.temp0[head * kHeadDim + d],
                               scratch.depth_keys[source][head * kHeadDim + d], sum);
            score[source] = sum * inverse_root;
            maximum = std::max(maximum, score[source]);
        }
        float denominator = 0.0f;
        for (int source = 0; source < source_count; ++source) {
            score[source] = std::exp(score[source] - maximum);
            denominator += score[source];
        }
        for (int d = 0; d < kHeadDim; ++d) {
            float sum = 0.0f;
            for (int source = 0; source < source_count; ++source)
                sum = std::fma(score[source] / denominator,
                               scratch.depth_values[source][head * kHeadDim + d], sum);
            scratch.temp1[head * kHeadDim + d] = sum;
        }
    }
    linear(weights.depth_out, scratch.temp1.data(), output, scratch);
}

INSIGNIA_PROFILE_NOINLINE static void causal_mla(
    const Weights &weights, Scratch &scratch, int repeat, int history_count,
    const float *input, float *output) {
    std::copy_n(input, kWidth, scratch.temp4.data());
    rms_norm(scratch.temp4.data(), kWidth);
    const float input_scale = quantize_vector(
        scratch.temp4.data(), kWidth, weights.mla_q.cols, scratch.quantized.data());
    linear_quantized(weights.mla_q, scratch.quantized.data(), input_scale,
                     scratch.accumulator.data(), scratch.temp0.data());
    linear_quantized(weights.mla_kv_down, scratch.quantized.data(), input_scale,
                     scratch.accumulator.data(), scratch.temp1.data());
    linear_quantized(weights.mla_gate, scratch.quantized.data(), input_scale,
                     scratch.accumulator.data(), scratch.temp2.data());
    rms_norm(scratch.temp1.data(), kLatent);
    std::copy_n(scratch.temp1.data(), kLatent,
                scratch.latents[repeat][history_count - 1].data());

    // Absorb the K projection into each query: q_h W_k,h -> 64-wide query.
    for (int head = 0; head < kHeads; ++head)
        for (int latent = 0; latent < kLatent; ++latent) {
            float sum = 0.0f;
            for (int d = 0; d < kHeadDim; ++d) {
                const int row = head * kHeadDim + d;
                const float weight = static_cast<float>(weights.mla_kv_up.weight[
                    static_cast<size_t>(row) * weights.mla_kv_up.cols + latent])
                    * weights.mla_kv_up.scale[row];
                sum = std::fma(scratch.temp0[row], weight, sum);
            }
            scratch.q_absorbed[head][latent] = sum;
        }

    constexpr float inverse_root = 0.1767766952966369f;
    for (int head = 0; head < kHeads; ++head) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int token = 0; token < history_count; ++token) {
            float score = 0.0f;
            for (int latent = 0; latent < kLatent; ++latent)
                score = std::fma(scratch.q_absorbed[head][latent],
                                 scratch.latents[repeat][token][latent], score);
            score *= inverse_root;
            scratch.attention_weights[head][token] = score;
            maximum = std::max(maximum, score);
        }
        float denominator = 0.0f;
        for (int token = 0; token < history_count; ++token) {
            const float value = std::exp(
                scratch.attention_weights[head][token] - maximum);
            scratch.attention_weights[head][token] = value;
            denominator += value;
        }
        for (int latent = 0; latent < kLatent; ++latent) {
            float sum = 0.0f;
            for (int token = 0; token < history_count; ++token)
                sum = std::fma(scratch.attention_weights[head][token] / denominator,
                               scratch.latents[repeat][token][latent], sum);
            scratch.latent_sum[head][latent] = sum;
        }
        for (int d = 0; d < kHeadDim; ++d) {
            const int row = 128 + head * kHeadDim + d;
            float sum = 0.0f;
            for (int latent = 0; latent < kLatent; ++latent) {
                const float weight = static_cast<float>(weights.mla_kv_up.weight[
                    static_cast<size_t>(row) * weights.mla_kv_up.cols + latent])
                    * weights.mla_kv_up.scale[row];
                sum = std::fma(weight, scratch.latent_sum[head][latent], sum);
            }
            scratch.temp3[head * kHeadDim + d] = sum
                * sigmoid(scratch.temp2[head * kHeadDim + d]);
        }
    }
    linear(weights.mla_out, scratch.temp3.data(), output, scratch);
}

INSIGNIA_PROFILE_NOINLINE static void stable_moe(
    const Weights &weights, Scratch &scratch, const float *input, float *output) {
    std::copy_n(input, kWidth, scratch.temp4.data());
    rms_norm(scratch.temp4.data(), kWidth);
    const float input_scale = quantize_vector(
        scratch.temp4.data(), kWidth, weights.moe_router.cols,
        scratch.quantized.data());
    linear_quantized(weights.moe_router, scratch.quantized.data(), input_scale,
                     scratch.accumulator.data(), scratch.temp0.data());
    linear_quantized(weights.latent_down, scratch.quantized.data(), input_scale,
                     scratch.accumulator.data(), scratch.temp1.data());
    linear_quantized(weights.shared_gate_up, scratch.quantized.data(), input_scale,
                     scratch.accumulator.data(), scratch.temp2.data());

    int selected[3] = {-1, -1, -1};
    float selected_score[3] = {-1.0f, -1.0f, -1.0f};
    for (int expert = 0; expert < 256; ++expert)
        scratch.temp0[expert] = sigmoid(scratch.temp0[expert]);
    for (int expert = 0; expert < 256; ++expert) {
        const float score = scratch.temp0[expert];
        for (int rank = 0; rank < 3; ++rank)
            if (score > selected_score[rank]) {
                for (int move = 2; move > rank; --move) {
                    selected_score[move] = selected_score[move - 1];
                    selected[move] = selected[move - 1];
                }
                selected_score[rank] = score;
                selected[rank] = expert;
                break;
            }
    }
    const float expert_weight0 = selected_score[0]
        / std::max(selected_score[0] + selected_score[1], 1.0e-12f);
    const float expert_weight1 = 1.0f - expert_weight0;

    const float latent_scale = quantize_vector(
        scratch.temp1.data(), 96, weights.expert_gate_up.cols,
        scratch.quantized.data());
    for (int route = 0; route < 2; ++route)
        expert_linear_quantized(weights.expert_gate_up, selected[route], 256,
                                scratch.quantized.data(), latent_scale,
                                scratch.accumulator.data(),
                                scratch.expert_gate[route].data());
    for (int route = 0; route < 2; ++route) {
        situ_glu(scratch.expert_gate[route].data(),
                 scratch.expert_activation[route].data(), 128);
        const float activation_scale = quantize_vector(
            scratch.expert_activation[route].data(), 128,
            weights.expert_down.cols, scratch.quantized.data());
        expert_linear_quantized(weights.expert_down, selected[route], 96,
                                scratch.quantized.data(), activation_scale,
                                scratch.accumulator.data(),
                                scratch.expert_output[route].data());
    }
    for (int d = 0; d < 96; ++d)
        scratch.temp3[d] = expert_weight0 * scratch.expert_output[0][d]
            + expert_weight1 * scratch.expert_output[1][d];
    rms_norm(scratch.temp3.data(), 96);
    linear(weights.latent_up, scratch.temp3.data(), scratch.temp0.data(), scratch);

    situ_glu(scratch.temp2.data(), scratch.temp3.data(), 384);
    linear(weights.shared_down, scratch.temp3.data(), scratch.temp1.data(), scratch);
    for (int d = 0; d < kWidth; ++d)
        output[d] = scratch.temp0[d] + scratch.temp1[d];
}

INSIGNIA_PROFILE_NOINLINE static void encode(
    const Weights &weights, Scratch &scratch, int layer, int row) {
    linear(weights.logit, scratch.input.data(), scratch.streams[0].data(), scratch);
    rms_norm(scratch.streams[0].data(), kWidth);

    std::fill(scratch.streams[1].begin(), scratch.streams[1].end(), 0.0f);
    for (int candidate = 0; candidate < 32; ++candidate) {
        for (int d = 0; d < 32; ++d)
            scratch.temp4[d] = scratch.input[(d + candidate * 7 + layer + row) % 224];
        linear(weights.candidate_a, scratch.temp4.data(), scratch.temp0.data(), scratch);
        silu(scratch.temp0.data(), 64);
        linear(weights.candidate_b, scratch.temp0.data(), scratch.temp1.data(), scratch);
        rms_norm(scratch.temp1.data(), kWidth);
        const float pool = 1.0f / 32.0f;
        for (int d = 0; d < kWidth; ++d)
            scratch.streams[1][d] += pool * scratch.temp1[d];
    }
    linear(weights.router_tail, scratch.input.data(), scratch.temp0.data(), scratch);
    rms_norm(scratch.temp0.data(), kWidth);
    for (int d = 0; d < kWidth; ++d) scratch.streams[1][d] += scratch.temp0[d];

    linear(weights.hidden, scratch.input.data(), scratch.streams[2].data(), scratch);
    rms_norm(scratch.streams[2].data(), kWidth);
    for (int d = 0; d < kWidth; ++d)
        scratch.streams[2][d] += 0.001f * float((layer * 13 + row * 7 + d) & 31);

    linear(weights.cache, scratch.input.data(), scratch.streams[3].data(), scratch);
    rms_norm(scratch.streams[3].data(), kWidth);
}

INSIGNIA_PROFILE_NOINLINE static uint64_t run_event(
    const Weights &weights, Scratch &scratch, int layer, int row,
    int verify_rows) {
    encode(weights, scratch, layer, row);
    collapse_streams(scratch, scratch.block_history[0].data());
    const int history_count = std::min(kMaxHistory, layer * verify_rows + row + 1);
    for (int repeat = 0; repeat < 3; ++repeat) {
        depth_attention(weights, scratch, repeat + 1, scratch.temp0.data());
        mix_streams(weights, scratch, scratch.temp0.data());
        causal_mla(weights, scratch, repeat, history_count,
                   scratch.temp0.data(), scratch.temp1.data());
        for (int stream = 0; stream < 4; ++stream)
            for (int d = 0; d < kWidth; ++d)
                scratch.streams[stream][d] += 0.1f * scratch.temp1[d];
        collapse_streams(scratch, scratch.temp2.data());
        stable_moe(weights, scratch, scratch.temp2.data(), scratch.temp3.data());
        for (int stream = 0; stream < 4; ++stream)
            for (int d = 0; d < kWidth; ++d)
                scratch.streams[stream][d] += 0.1f * scratch.temp3[d];
        collapse_streams(scratch, scratch.temp2.data());
        mix_streams(weights, scratch, scratch.temp2.data());
        collapse_streams(scratch, scratch.block_history[repeat + 1].data());
    }
    depth_attention(weights, scratch, 4, scratch.temp0.data());
    rms_norm(scratch.temp0.data(), kWidth);
    linear(weights.heads, scratch.temp0.data(), scratch.temp1.data(), scratch);
    uint64_t checksum = 0;
    for (int i = 0; i < 107; i += 7) {
        uint32_t bits;
        std::memcpy(&bits, &scratch.temp1[i], sizeof(bits));
        checksum = (checksum << 5) | (checksum >> 59);
        checksum += bits;
    }
    if (!std::isfinite(scratch.temp1[layer % 107]))
        throw std::runtime_error("pipeline produced non-finite output");
    return checksum;
}

static int32_t scalar_dot(const int8_t *weight, const int8_t *input, int width) {
    int32_t result = 0;
    for (int i = 0; i < width; ++i) result += int32_t(weight[i]) * int32_t(input[i]);
    return result;
}

static void verify_kernel(const Weights &weights) {
    Scratch scratch(19);
    const float scale = quantize_vector(scratch.input.data(), weights.logit.logical_cols,
                                        weights.logit.cols, scratch.quantized.data());
    (void)scale;
    vnni_rows(weights.logit.weight.data(), weights.logit.correction.data(),
              weights.logit.rows, weights.logit.cols, scratch.quantized.data(),
              scratch.accumulator.data());
    for (int row = 0; row < weights.logit.rows; ++row) {
        const int32_t expected = scalar_dot(
            weights.logit.weight.data() + static_cast<size_t>(row) * weights.logit.cols,
            scratch.quantized.data(), weights.logit.cols);
        if (scratch.accumulator[row] != expected)
            throw std::runtime_error("VPDPBUSD exactness failure");
    }
}

static void pin_thread(int worker) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET((worker * 2) % std::thread::hardware_concurrency(), &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

struct Measurement {
    double milliseconds;
    uint64_t checksum;
};

static Measurement measure(const Weights &weights, int threads, int iterations) {
    std::barrier start(threads + 1);
    std::barrier layer_barrier(threads);
    std::atomic<uint64_t> checksum{0};
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (int worker = 0; worker < threads; ++worker) {
        workers.emplace_back([&, worker] {
            pin_thread(worker);
            Scratch scratch(1000 + worker);
            uint64_t local = 0;
            start.arrive_and_wait();
            for (int iteration = 0; iteration < iterations; ++iteration)
                for (int layer = 0; layer < 42; ++layer) {
                    local += run_event(weights, scratch, layer, worker, threads);
                    layer_barrier.arrive_and_wait();
                }
            checksum.fetch_add(local, std::memory_order_relaxed);
        });
    }
    start.arrive_and_wait();
    const auto begin = std::chrono::steady_clock::now();
    for (std::thread &worker : workers) worker.join();
    const auto end = std::chrono::steady_clock::now();
    return {std::chrono::duration<double, std::milli>(end - begin).count(),
            checksum.load(std::memory_order_relaxed)};
}

int main(int argc, char **argv) {
    const int threads = argc > 1 ? std::stoi(argv[1]) : 4;
    const int iterations = argc > 2 ? std::stoi(argv[2]) : 20;
    if (threads < 1 || threads > 8 || iterations < 1)
        throw std::runtime_error(
            "usage: benchmark_falsifier_vnni_pipeline [verify_rows 1..8] [iterations]");
    if (!__builtin_cpu_supports("avxvnni"))
        throw std::runtime_error("Raptor Lake AVX-VNNI is unavailable");
    Weights weights;
    verify_kernel(weights);
    (void)measure(weights, threads, 1);
    const Measurement measured = measure(weights, threads, iterations);
    const double round_ms = measured.milliseconds / iterations;
    const uint64_t physical_macs = weights.event_macs(false)
        * static_cast<uint64_t>(threads) * 42;
    std::cout
        << "{\n"
        << "  \"schema\": \"insignia-falsifier-vnni-pipeline-v1\",\n"
        << "  \"synthetic_weights\": true,\n"
        << "  \"verify_rows\": " << threads << ",\n"
        << "  \"iterations\": " << iterations << ",\n"
        << "  \"logical_matrix_macs_per_event\": " << weights.event_macs(true) << ",\n"
        << "  \"physical_matrix_macs_per_round\": " << physical_macs << ",\n"
        << "  \"round_ms\": " << round_ms << ",\n"
        << "  \"layer_group_ms\": " << round_ms / 42.0 << ",\n"
        << "  \"checksum\": " << measured.checksum << ",\n"
        << "  \"kernel_exact\": true,\n"
        << "  \"includes_dynamic_quantization\": true,\n"
        << "  \"includes_layer_barriers\": true,\n"
        << "  \"includes_absorbed_mla\": true\n"
        << "}\n";
    return 0;
}
