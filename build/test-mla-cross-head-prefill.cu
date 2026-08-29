#include "insignia_glm53.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

void check(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

template <typename T>
T *device(size_t count) {
    T *pointer = nullptr;
    check(cudaMalloc(&pointer, count * sizeof(T)), "cudaMalloc");
    return pointer;
}

float sample(uint32_t index, float scale = 1.0f) {
    const int value = int((index * 2654435761u) >> 22) - 512;
    return float(value) * (scale / 512.0f);
}

void run_position(
    int position_base, int tokens,
    const float *queries, const float *latents,
    uint8_t *cache_scalar, float *scales_scalar,
    uint8_t *cache_cross, float *scales_cross,
    const uint8_t *weights, const uint16_t *weight_scales,
    uint8_t *qeff, float *qeff_scales,
    float *out_scalar, float *out_cross) {
    constexpr int heads = 64, head_dim = 256, latent_dim = 512;
    for (int base = 0; base < position_base; base += 128) {
        const int count = min(128, position_base - base);
        check(insignia::glm53::mla_store_latent(
                  latents + size_t(base) * latent_dim, cache_scalar, scales_scalar,
                  nullptr, count, base, latent_dim), "seed scalar cache");
        check(insignia::glm53::mla_store_latent(
                  latents + size_t(base) * latent_dim, cache_cross, scales_cross,
                  nullptr, count, base, latent_dim), "seed cross cache");
    }
    const float *new_latents = latents + size_t(position_base) * latent_dim;
    check(insignia::glm53::mla_prefill_latent_fp8_absorb(
              queries, new_latents, cache_scalar, scales_scalar, nullptr,
              weights, weight_scales, out_scalar, tokens, position_base,
              heads, head_dim, latent_dim), "scalar prefill");
    check(insignia::glm53::mla_prefill_latent_cross_head_fp8_absorb(
              queries, new_latents, cache_cross, scales_cross,
              weights, weight_scales, qeff, qeff_scales, out_cross,
              tokens, position_base, heads, head_dim, latent_dim),
          "cross-head prefill");
    check(cudaDeviceSynchronize(), "comparison synchronize");

    const size_t output_count = size_t(tokens) * heads * head_dim;
    std::vector<float> scalar(output_count), cross(output_count);
    check(cudaMemcpy(scalar.data(), out_scalar, output_count * sizeof(float),
                     cudaMemcpyDeviceToHost), "download scalar output");
    check(cudaMemcpy(cross.data(), out_cross, output_count * sizeof(float),
                     cudaMemcpyDeviceToHost), "download cross output");
    double error2 = 0.0, reference2 = 0.0, dot = 0.0, cross2 = 0.0;
    double maximum_error = 0.0;
    for (size_t index = 0; index < output_count; ++index) {
        if (!std::isfinite(scalar[index]) || !std::isfinite(cross[index])) {
            std::fprintf(stderr, "FAIL non-finite output at %zu: %.9g %.9g\n",
                         index, scalar[index], cross[index]);
            std::exit(2);
        }
        const double error = double(cross[index]) - scalar[index];
        error2 += error * error;
        reference2 += double(scalar[index]) * scalar[index];
        cross2 += double(cross[index]) * cross[index];
        dot += double(scalar[index]) * cross[index];
        maximum_error = fmax(maximum_error, fabs(error));
    }
    const double relative = std::sqrt(error2 / reference2);
    const double cosine = dot / std::sqrt(reference2 * cross2);
    std::printf("base %d tokens %d quality rel_l2=%.7f cosine=%.9f max=%.7g\n",
                position_base, tokens, relative, cosine, maximum_error);
    if (relative > 0.08 || cosine < 0.995) {
        std::fprintf(stderr, "FAIL cross-head numerical gate\n");
        std::exit(2);
    }

    constexpr int repeats = 10;
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin), "create begin event");
    check(cudaEventCreate(&end), "create end event");
    for (int iteration = 0; iteration < 2; ++iteration)
        check(insignia::glm53::mla_prefill_latent_fp8_absorb(
                  queries, new_latents, cache_scalar, scales_scalar, nullptr,
                  weights, weight_scales, out_scalar, tokens, position_base,
                  heads, head_dim, latent_dim), "warm scalar");
    check(cudaEventRecord(begin), "record scalar begin");
    for (int iteration = 0; iteration < repeats; ++iteration)
        check(insignia::glm53::mla_prefill_latent_fp8_absorb(
                  queries, new_latents, cache_scalar, scales_scalar, nullptr,
                  weights, weight_scales, out_scalar, tokens, position_base,
                  heads, head_dim, latent_dim), "time scalar");
    check(cudaEventRecord(end), "record scalar end");
    check(cudaEventSynchronize(end), "sync scalar timing");
    float scalar_ms = 0.0f;
    check(cudaEventElapsedTime(&scalar_ms, begin, end), "scalar elapsed");

    for (int iteration = 0; iteration < 2; ++iteration)
        check(insignia::glm53::mla_prefill_latent_cross_head_fp8_absorb(
                  queries, new_latents, cache_cross, scales_cross,
                  weights, weight_scales, qeff, qeff_scales, out_cross,
                  tokens, position_base, heads, head_dim, latent_dim),
              "warm cross");
    check(cudaEventRecord(begin), "record cross begin");
    for (int iteration = 0; iteration < repeats; ++iteration)
        check(insignia::glm53::mla_prefill_latent_cross_head_fp8_absorb(
                  queries, new_latents, cache_cross, scales_cross,
                  weights, weight_scales, qeff, qeff_scales, out_cross,
                  tokens, position_base, heads, head_dim, latent_dim),
              "time cross");
    check(cudaEventRecord(end), "record cross end");
    check(cudaEventSynchronize(end), "sync cross timing");
    float cross_ms = 0.0f;
    check(cudaEventElapsedTime(&cross_ms, begin, end), "cross elapsed");
    scalar_ms /= repeats;
    cross_ms /= repeats;
    std::printf("base %d tokens %d scalar=%.4f ms cross=%.4f ms speedup=%.2fx\n",
                position_base, tokens, scalar_ms, cross_ms, scalar_ms / cross_ms);
    check(cudaEventDestroy(begin), "destroy begin event");
    check(cudaEventDestroy(end), "destroy end event");
}

}  // namespace

int main() {
    constexpr int heads = 64, head_dim = 256, latent_dim = 512;
    constexpr int tokens = 128, max_context = 8192;
    constexpr int weight_rows = heads * 2 * head_dim;
    constexpr int groups = latent_dim / 64;
    std::vector<float> queries(size_t(tokens) * heads * head_dim);
    std::vector<float> latents(size_t(max_context) * latent_dim);
    std::vector<uint8_t> weights(size_t(weight_rows) * latent_dim);
    std::vector<__half> weight_scales(size_t(weight_rows) * groups);
    for (size_t index = 0; index < queries.size(); ++index)
        queries[index] = sample(index + 1, 0.2f);
    for (size_t index = 0; index < latents.size(); ++index)
        latents[index] = sample(index + 7, 0.5f);
    for (size_t index = 0; index < weights.size(); ++index) {
        const uint8_t magnitude = uint8_t((index * 17 + 11) % 0x77);
        weights[index] = magnitude | ((index & 16) ? 0x80 : 0);
    }
    for (size_t index = 0; index < weight_scales.size(); ++index)
        weight_scales[index] = __float2half(0.0002f * float(1 + index % 7));

    float *d_queries = device<float>(queries.size());
    float *d_latents = device<float>(latents.size());
    uint8_t *d_weights = device<uint8_t>(weights.size());
    uint16_t *d_weight_scales = device<uint16_t>(weight_scales.size());
    uint8_t *d_cache_scalar = device<uint8_t>(latents.size());
    uint8_t *d_cache_cross = device<uint8_t>(latents.size());
    float *d_scales_scalar = device<float>(size_t(max_context) * groups);
    float *d_scales_cross = device<float>(size_t(max_context) * groups);
    uint8_t *d_qeff = device<uint8_t>(size_t(tokens) * heads * latent_dim);
    float *d_qeff_scales = device<float>(size_t(tokens) * heads * groups);
    float *d_out_scalar = device<float>(queries.size());
    float *d_out_cross = device<float>(queries.size());
    check(cudaMemcpy(d_queries, queries.data(), queries.size() * sizeof(float),
                     cudaMemcpyHostToDevice), "upload queries");
    check(cudaMemcpy(d_latents, latents.data(), latents.size() * sizeof(float),
                     cudaMemcpyHostToDevice), "upload latents");
    check(cudaMemcpy(d_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice), "upload weights");
    check(cudaMemcpy(d_weight_scales, weight_scales.data(),
                     weight_scales.size() * sizeof(__half), cudaMemcpyHostToDevice),
          "upload weight scales");
    for (const auto [position_base, token_count] : {
             std::pair{256, 1}, std::pair{4096, 1}, std::pair{8064, 1},
             std::pair{256, 7}, std::pair{256, 8}, std::pair{1024, 8},
             std::pair{4096, 8}, std::pair{8064, 8},
             std::pair{512, 16}, std::pair{1024, 32}, std::pair{1024, 64},
             std::pair{256, 128}, std::pair{512, 128}, std::pair{1024, 128},
             std::pair{2048, 128}, std::pair{4096, 128}, std::pair{8064, 128}})
        run_position(position_base, token_count, d_queries, d_latents,
                     d_cache_scalar, d_scales_scalar, d_cache_cross, d_scales_cross,
                     d_weights, d_weight_scales, d_qeff, d_qeff_scales,
                     d_out_scalar, d_out_cross);
    return 0;
}
