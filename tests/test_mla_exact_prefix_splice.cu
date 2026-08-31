#include "insignia_glm53.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

constexpr int kHeads = 64;
constexpr int kHeadDim = 256;
constexpr int kLatent = 512;
constexpr int kGroups = 8;
constexpr int kPrefix = 256;
constexpr int kTokens = 16;
constexpr int kMaxTokens = 96;
constexpr int kWidth = kHeads * kHeadDim;
constexpr int kMaxContext = 8192;

void check(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

template <class T>
T *device(size_t count) {
    T *pointer = nullptr;
    check(cudaMalloc(&pointer, count * sizeof(T)), "cudaMalloc");
    check(cudaMemset(pointer, 0, count * sizeof(T)), "cudaMemset");
    return pointer;
}

template <class T>
T *upload(const std::vector<T> &host) {
    T *pointer = device<T>(host.size());
    check(cudaMemcpy(pointer, host.data(), host.size() * sizeof(T),
                     cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    return pointer;
}

struct Metrics {
    double mse = 0.0;
    double rel_l2 = 0.0;
    double cosine = 0.0;
    double kl = 0.0;
    double js = 0.0;
    double ppl_reference = 0.0;
    double ppl_candidate = 0.0;
    int top1_mismatches = 0;
};

Metrics metrics(const float *reference_device, const float *candidate_device,
                int rows) {
    const size_t count = size_t(rows) * kHeadDim;
    std::vector<float> reference(count), candidate(count);
    check(cudaMemcpy(reference.data(), reference_device, count * sizeof(float),
                     cudaMemcpyDeviceToHost), "download reference");
    check(cudaMemcpy(candidate.data(), candidate_device, count * sizeof(float),
                     cudaMemcpyDeviceToHost), "download candidate");
    double error2 = 0.0, reference2 = 0.0, candidate2 = 0.0, dot = 0.0;
    double kl = 0.0, js = 0.0;
    double reference_nll = 0.0, candidate_nll = 0.0;
    int top1_mismatches = 0;
    for (int row = 0; row < rows; ++row) {
        const float *a = reference.data() + size_t(row) * kHeadDim;
        const float *b = candidate.data() + size_t(row) * kHeadDim;
        float maximum_a = -std::numeric_limits<float>::infinity();
        float maximum_b = -std::numeric_limits<float>::infinity();
        int target = 0, candidate_target = 0;
        for (int column = 0; column < kHeadDim; ++column) {
            if (!std::isfinite(a[column]) || !std::isfinite(b[column])) {
                std::fprintf(stderr, "non-finite output at row %d column %d\n",
                             row, column);
                std::exit(2);
            }
            if (a[column] > a[target]) target = column;
            if (b[column] > b[candidate_target]) candidate_target = column;
            maximum_a = std::max(maximum_a, a[column]);
            maximum_b = std::max(maximum_b, b[column]);
            const double error = double(b[column]) - a[column];
            error2 += error * error;
            reference2 += double(a[column]) * a[column];
            candidate2 += double(b[column]) * b[column];
            dot += double(a[column]) * b[column];
        }
        double denominator_a = 0.0, denominator_b = 0.0;
        for (int column = 0; column < kHeadDim; ++column) {
            denominator_a += std::exp(double(a[column] - maximum_a));
            denominator_b += std::exp(double(b[column] - maximum_b));
        }
        const double log_z_a = double(maximum_a) + std::log(denominator_a);
        const double log_z_b = double(maximum_b) + std::log(denominator_b);
        reference_nll += log_z_a - a[target];
        candidate_nll += log_z_b - b[target];
        top1_mismatches += target != candidate_target;
        for (int column = 0; column < kHeadDim; ++column) {
            const double p = std::exp(double(a[column]) - log_z_a);
            const double q = std::exp(double(b[column]) - log_z_b);
            const double log_p = double(a[column]) - log_z_a;
            const double log_q = double(b[column]) - log_z_b;
            const double mixture = 0.5 * (p + q);
            kl += p * (log_p - log_q);
            js += 0.5 * (p * (log_p - std::log(mixture)) +
                         q * (log_q - std::log(mixture)));
        }
    }
    Metrics result;
    result.mse = error2 / count;
    result.rel_l2 = std::sqrt(error2 / reference2);
    result.cosine = dot / std::sqrt(reference2 * candidate2);
    result.kl = std::max(0.0, kl / rows);
    result.js = std::max(0.0, js / rows);
    result.ppl_reference = std::exp(reference_nll / rows);
    result.ppl_candidate = std::exp(candidate_nll / rows);
    result.top1_mismatches = top1_mismatches;
    return result;
}

void print_metrics(const char *name, const Metrics &value) {
    std::printf(
        "%s mse=%.9e rel_l2=%.9e cosine=%.12f kl=%.9e js=%.9e "
        "ppl=%.9f->%.9f delta=%+.5f%% top1_mismatch=%d\n",
        name, value.mse, value.rel_l2, value.cosine, value.kl, value.js,
        value.ppl_reference, value.ppl_candidate,
        100.0 * (value.ppl_candidate / value.ppl_reference - 1.0),
        value.top1_mismatches);
}

bool quality_gate(const Metrics &value, double mse_limit, double rel_l2_limit,
                  double cosine_floor, double kl_limit, double js_limit) {
    return value.mse < mse_limit && value.rel_l2 < rel_l2_limit &&
        value.cosine > cosine_floor && value.kl < kl_limit &&
        value.js < js_limit &&
        value.ppl_candidate / value.ppl_reference < 1.035 &&
        !value.top1_mismatches;
}

template <class F>
float time_ms(F &&call, int repetitions) {
    cudaEvent_t begin = nullptr, end = nullptr;
    check(cudaEventCreate(&begin), "create timing begin");
    check(cudaEventCreate(&end), "create timing end");
    for (int iteration = 0; iteration < 3; ++iteration)
        check(call(), "timing warmup");
    check(cudaEventRecord(begin), "record timing begin");
    for (int iteration = 0; iteration < repetitions; ++iteration)
        check(call(), "timed call");
    check(cudaEventRecord(end), "record timing end");
    check(cudaEventSynchronize(end), "synchronize timing");
    float elapsed = 0.0f;
    check(cudaEventElapsedTime(&elapsed, begin, end), "timing elapsed");
    check(cudaEventDestroy(begin), "destroy timing begin");
    check(cudaEventDestroy(end), "destroy timing end");
    return elapsed / repetitions;
}

template <class F>
float median_ms(F &&call, int repetitions) {
    std::array<float, 7> samples{};
    for (float &sample : samples) sample = time_ms(call, repetitions);
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace

int main() {
    static_assert(insignia::glm53::mla_exact_prefix_overlap_tokens(96, 192) == 64);
    static_assert(insignia::glm53::mla_exact_prefix_overlap_tokens(96, 0) == 96);
    static_assert(insignia::glm53::mla_exact_prefix_overlap_tokens(8, 256) == 0);
    static_assert(!insignia::glm53::mla_cross_head_use_fused_prefill(8, 256));
    static_assert(insignia::glm53::mla_cross_head_use_fused_prefill(16, 256));
    static_assert(!insignia::glm53::mla_cross_head_use_fused_prefill(8, 4096));
    constexpr int weight_rows = kHeads * 2 * kHeadDim;
    std::vector<float> queries(size_t(kMaxTokens) * kWidth);
    std::vector<float> latents(size_t(kMaxContext) * kLatent);
    std::vector<uint8_t> weights(size_t(weight_rows) * kLatent, uint8_t(0));
    std::vector<__half> weight_scales(size_t(weight_rows) * kGroups,
                                      __float2half(1.0f));

    for (int token = 0; token < kMaxTokens; ++token)
        for (int head = 0; head < kHeads; ++head)
            for (int column = 0; column < kHeadDim; ++column)
                queries[(size_t(token) * kHeads + head) * kHeadDim + column] =
                    0.14f * std::sin(
                        float((token * kHeadDim + column) * 17 + head * 3 + 3) *
                        0.013f) +
                    0.05f * std::cos(
                        float(token * 11 + column * 5 + head * 7 + 1) * 0.017f);
    // Every suffix row is non-zero. Distributed aligned rows on both sides of
    // the seam keep prefix and suffix attention competitive without a single
    // key-zero sentinel dominating the output.
    for (int key = 0; key < kMaxContext; ++key)
        for (int column = 0; column < kLatent; ++column)
            latents[size_t(key) * kLatent + column] =
                0.24f * std::sin(float(key * 521 + column * 29 + 11) * 0.007f) +
                0.11f * std::cos(float(key * 31 + column * 7 + 5) * 0.019f);
    for (int key = 7; key < kMaxContext; key += 127)
        for (int column = 0; column < kHeadDim; ++column)
            latents[size_t(key) * kLatent + column] +=
                1.7f * queries[column];
    for (int key = 0; key < kMaxContext; ++key) {
        latents[size_t(key) * kLatent] += 0.50f;
        latents[size_t(key) * kLatent + 1] += 0.10f;
    }

    // Compact W_uk/W_uv identity on the first 256 latent dimensions. E4M3
    // code 0x38 with FP16 scale 1 is exactly 1.0.
    for (int head = 0; head < kHeads; ++head)
        for (int element = 0; element < kHeadDim; ++element) {
            const int key_row = head * 512 + element;
            const int value_row = key_row + 256;
            weights[size_t(key_row) * kLatent + element] = 0x38;
            weights[size_t(value_row) * kLatent + element] = 0x38;
        }

    float *d_queries = upload(queries);
    float *d_latents = upload(latents);
    uint8_t *d_weights = upload(weights);
    uint16_t *d_weight_scales = reinterpret_cast<uint16_t *>(upload(weight_scales));
    uint8_t *d_cache_cross = device<uint8_t>(size_t(kMaxContext) * kLatent);
    float *d_scale_cross = device<float>(size_t(kMaxContext) * kGroups);
    float *d_cache_reference = device<float>(size_t(kMaxContext) * kLatent);
    float *d_scale_reference = device<float>(size_t(kMaxContext) * kGroups);
    constexpr int partial_tiles = kMaxContext / 512;
    float *d_partial_cross = device<float>(
        size_t(kHeads) * partial_tiles * (kLatent + 2));
    float *d_partial_reference = device<float>(
        size_t(kHeads) * partial_tiles * (kLatent + 2));
    uint8_t *d_qeff = device<uint8_t>(size_t(kMaxTokens) * kHeads * kLatent);
    float *d_qeff_scales = device<float>(size_t(kMaxTokens) * kHeads * kGroups);
    float *d_qeff_f32 = device<float>(size_t(kMaxTokens) * kHeads * kLatent);
    float *d_prefix_partial = device<float>(
        size_t(kHeads) * (kPrefix / 16) * (kLatent + 2));
    float *d_crossing_prefix = device<float>(size_t(kPrefix) * kLatent);
    float *d_output_cross = device<float>(size_t(kMaxTokens) * kWidth);
    float *d_output_scalar = device<float>(size_t(kMaxTokens) * kWidth);
    float *d_output_no_splice = device<float>(size_t(kMaxTokens) * kWidth);
    float *d_output_reference = device<float>(size_t(kMaxTokens) * kWidth);

    const bool invalid_pair_gate =
        insignia::glm53::mla_decode_latent_cross_head_fp8_absorb(
            d_queries, d_latents + size_t(kPrefix) * kLatent,
            d_cache_cross, d_scale_cross, d_weights, d_weight_scales,
            d_qeff, d_qeff_scales, d_partial_cross, d_output_cross,
            kPrefix, kHeads, kHeadDim, kLatent, d_latents, nullptr) ==
            cudaErrorInvalidValue &&
        insignia::glm53::mla_decode_latent_cross_head_fp8_absorb(
            d_queries, d_latents + size_t(kPrefix) * kLatent,
            d_cache_cross, d_scale_cross, d_weights, d_weight_scales,
            d_qeff, d_qeff_scales, d_partial_cross, d_output_cross,
            kPrefix, kHeads, kHeadDim, kLatent, d_latents, d_qeff_f32,
            d_prefix_partial, false) == cudaErrorInvalidValue &&
        insignia::glm53::mla_decode_latent_cross_head_fp8_absorb(
            d_queries, d_latents + size_t(kPrefix) * kLatent,
            d_cache_cross, d_scale_cross, d_weights, d_weight_scales,
            d_qeff, d_qeff_scales, d_partial_cross, d_output_cross,
            kPrefix, kHeads, kHeadDim, kLatent, d_latents, d_qeff_f32,
            nullptr, true) == cudaErrorInvalidValue &&
        insignia::glm53::mla_prefill_latent_cross_head_fp8_absorb(
            d_queries, d_latents + size_t(kPrefix) * kLatent,
            d_cache_cross, d_scale_cross, d_weights, d_weight_scales,
            d_qeff, d_qeff_scales, d_output_cross, kTokens, kPrefix,
            kHeads, kHeadDim, kLatent, d_latents, nullptr) ==
            cudaErrorInvalidValue;
    std::printf("pointer-pairing fail-closed=%s\n",
                invalid_pair_gate ? "PASS" : "FAIL");

    check(insignia::glm53::mla_store_latent(
              d_latents, nullptr, d_scale_reference, d_cache_reference,
              kPrefix, 0), "seed exact prefix reference");
    check(insignia::glm53::mla_store_latent(
              d_latents, d_cache_cross, d_scale_cross, nullptr,
              kPrefix, 0), "seed FP8 prefix shadow");
    check(insignia::glm53::mla_decode_latent_fp8_absorb(
              d_queries, d_latents + size_t(kPrefix) * kLatent,
              nullptr, d_scale_reference, d_cache_reference,
              d_weights, d_weight_scales, d_partial_reference,
              d_output_reference, kPrefix), "exact-prefix decode reference");
    check(insignia::glm53::mla_decode_latent_cross_head_fp8_absorb(
              d_queries, d_latents + size_t(kPrefix) * kLatent,
              d_cache_cross, d_scale_cross, d_weights, d_weight_scales,
              d_qeff, d_qeff_scales, d_partial_cross, d_output_scalar,
              kPrefix, kHeads, kHeadDim, kLatent, d_latents, d_qeff_f32),
          "scalar exact-prefix decode");
    check(insignia::glm53::mla_decode_latent_cross_head_fp8_absorb(
              d_queries, d_latents + size_t(kPrefix) * kLatent,
              d_cache_cross, d_scale_cross, d_weights, d_weight_scales,
              d_qeff, d_qeff_scales, d_partial_cross, d_output_cross,
              kPrefix, kHeads, kHeadDim, kLatent, d_latents, d_qeff_f32,
              d_prefix_partial, true),
          "parallel exact-prefix decode");
    check(cudaDeviceSynchronize(), "decode synchronize");
    const Metrics decode_scalar = metrics(
        d_output_reference, d_output_scalar, kHeads);
    const Metrics decode_parallel = metrics(
        d_output_reference, d_output_cross, kHeads);
    const Metrics decode_fast_vs_scalar = metrics(
        d_output_scalar, d_output_cross, kHeads);
    print_metrics("decode-scalar-diagnostic", decode_scalar);
    print_metrics("decode-parallel", decode_parallel);
    print_metrics("decode-fast-vs-scalar-diagnostic", decode_fast_vs_scalar);

    check(cudaMemset(d_cache_cross, 0, size_t(kMaxContext) * kLatent),
          "clear cross cache");
    check(cudaMemset(d_cache_reference, 0,
                     size_t(kMaxContext) * kLatent * sizeof(float)),
          "clear reference cache");
    check(insignia::glm53::mla_store_latent(
              d_latents, nullptr, d_scale_reference, d_cache_reference,
              kPrefix, 0), "seed exact prefill reference");
    check(insignia::glm53::mla_store_latent(
              d_latents, d_cache_cross, d_scale_cross, nullptr,
              kPrefix, 0), "seed FP8 prefill shadow");
    check(insignia::glm53::mla_prefill_latent_fp8_absorb(
              d_queries, d_latents + size_t(kPrefix) * kLatent,
              nullptr, d_scale_reference, d_cache_reference,
              d_weights, d_weight_scales, d_output_reference,
              kTokens, kPrefix), "exact-prefix prefill reference");
    check(insignia::glm53::mla_prefill_latent_cross_head_fp8_absorb(
              d_queries, d_latents + size_t(kPrefix) * kLatent,
              d_cache_cross, d_scale_cross, d_weights, d_weight_scales,
              d_qeff, d_qeff_scales, d_output_cross, kTokens, kPrefix,
              kHeads, kHeadDim, kLatent, d_latents, d_qeff_f32),
          "cross-head exact-prefix prefill");
    check(cudaDeviceSynchronize(), "prefill synchronize");
    const Metrics prefill = metrics(
        d_output_reference, d_output_cross, kTokens * kHeads);
    print_metrics("prefill", prefill);

    const auto prefill_no_splice = [&] {
        return insignia::glm53::mla_prefill_latent_cross_head_fp8_absorb(
            d_queries, d_latents + size_t(kPrefix) * kLatent,
            d_cache_cross, d_scale_cross, d_weights, d_weight_scales,
            d_qeff, d_qeff_scales, d_output_no_splice, kTokens, kPrefix);
    };
    const auto prefill_splice = [&] {
        return insignia::glm53::mla_prefill_latent_cross_head_fp8_absorb(
            d_queries, d_latents + size_t(kPrefix) * kLatent,
            d_cache_cross, d_scale_cross, d_weights, d_weight_scales,
            d_qeff, d_qeff_scales, d_output_cross, kTokens, kPrefix,
            kHeads, kHeadDim, kLatent, d_latents, d_qeff_f32);
    };
    const float prefill_no_splice_ms = median_ms(prefill_no_splice, 20);
    const float prefill_splice_ms = median_ms(prefill_splice, 20);
    std::printf("prefill16@256 no_splice=%.4f ms splice=%.4f ms overhead=%+.2f%%\n",
                prefill_no_splice_ms, prefill_splice_ms,
                100.0f * (prefill_splice_ms / prefill_no_splice_ms - 1.0f));

    check(cudaMemset(d_cache_cross, 0, size_t(kMaxContext) * kLatent),
          "clear long cross cache");
    check(cudaMemset(d_cache_reference, 0,
                     size_t(kMaxContext) * kLatent * sizeof(float)),
          "clear long reference cache");
    check(insignia::glm53::mla_store_latent(
              d_latents, nullptr, d_scale_reference, d_cache_reference,
              kMaxContext - 1, 0), "seed long exact cache");
    check(insignia::glm53::mla_store_latent(
              d_latents, d_cache_cross, d_scale_cross, nullptr,
              kMaxContext - 1, 0), "seed long FP8 cache");
    Metrics long_no_splice, long_scalar, long_parallel;
    bool scalar_context_gate = true, parallel_context_gate = true;
    bool fast_vs_scalar_gate = true;
    constexpr std::array<int, 6> contexts{257, 512, 1024, 2048, 4096, 8192};
    for (int context : contexts) {
        const int position = context - 1;
        check(insignia::glm53::mla_decode_latent_fp8_absorb(
                  d_queries, d_latents + size_t(position) * kLatent,
                  nullptr, d_scale_reference, d_cache_reference,
                  d_weights, d_weight_scales, d_partial_reference,
                  d_output_reference, position), "long exact decode reference");
        check(insignia::glm53::mla_decode_latent_cross_head_fp8_absorb(
                  d_queries, d_latents + size_t(position) * kLatent,
                  d_cache_cross, d_scale_cross, d_weights, d_weight_scales,
                  d_qeff, d_qeff_scales, d_partial_cross, d_output_no_splice,
                  position), "long decode without splice");
        check(insignia::glm53::mla_decode_latent_cross_head_fp8_absorb(
                  d_queries, d_latents + size_t(position) * kLatent,
                  d_cache_cross, d_scale_cross, d_weights, d_weight_scales,
                  d_qeff, d_qeff_scales, d_partial_cross, d_output_scalar,
                  position, kHeads, kHeadDim, kLatent,
                  d_latents, d_qeff_f32),
              "long scalar exact-prefix decode");
        check(insignia::glm53::mla_decode_latent_cross_head_fp8_absorb(
                  d_queries, d_latents + size_t(position) * kLatent,
                  d_cache_cross, d_scale_cross, d_weights, d_weight_scales,
                  d_qeff, d_qeff_scales, d_partial_cross, d_output_cross,
                  position, kHeads, kHeadDim, kLatent,
                  d_latents, d_qeff_f32, d_prefix_partial, true),
              "long parallel exact-prefix decode");
        check(cudaDeviceSynchronize(), "long decode synchronize");
        const Metrics no_splice = metrics(
            d_output_reference, d_output_no_splice, kHeads);
        const Metrics scalar = metrics(
            d_output_reference, d_output_scalar, kHeads);
        const Metrics parallel = metrics(
            d_output_reference, d_output_cross, kHeads);
        const Metrics fast_vs_scalar = metrics(
            d_output_scalar, d_output_cross, kHeads);
        scalar_context_gate = scalar_context_gate &&
            quality_gate(scalar, 1.0e-7, 2.5e-2, 0.9997, 1.0e-7, 3.0e-8);
        parallel_context_gate = parallel_context_gate &&
            quality_gate(parallel, 1.0e-7, 2.5e-2, 0.9997, 1.0e-7, 3.0e-8);
        fast_vs_scalar_gate = fast_vs_scalar_gate &&
            quality_gate(fast_vs_scalar, 1.0e-10, 1.0e-4, 0.99999999,
                         1.0e-9, 3.0e-10);
        char name[64];
        std::snprintf(name, sizeof(name), "decode%d-no-splice", context);
        print_metrics(name, no_splice);
        std::snprintf(name, sizeof(name),
                      "decode%d-scalar-diagnostic", context);
        print_metrics(name, scalar);
        std::snprintf(name, sizeof(name), "decode%d-parallel", context);
        print_metrics(name, parallel);
        std::snprintf(name, sizeof(name),
                      "decode%d-fast-vs-scalar-diagnostic", context);
        print_metrics(name, fast_vs_scalar);

        const auto decode_no_splice = [&] {
            return insignia::glm53::mla_decode_latent_cross_head_fp8_absorb(
                d_queries, d_latents + size_t(position) * kLatent,
                d_cache_cross, d_scale_cross, d_weights, d_weight_scales,
                d_qeff, d_qeff_scales, d_partial_cross, d_output_no_splice,
                position);
        };
        const auto decode_scalar = [&] {
            return insignia::glm53::mla_decode_latent_cross_head_fp8_absorb(
                d_queries, d_latents + size_t(position) * kLatent,
                d_cache_cross, d_scale_cross, d_weights, d_weight_scales,
                d_qeff, d_qeff_scales, d_partial_cross, d_output_scalar,
                position, kHeads, kHeadDim, kLatent,
                d_latents, d_qeff_f32);
        };
        const auto decode_parallel = [&] {
            return insignia::glm53::mla_decode_latent_cross_head_fp8_absorb(
                d_queries, d_latents + size_t(position) * kLatent,
                d_cache_cross, d_scale_cross, d_weights, d_weight_scales,
                d_qeff, d_qeff_scales, d_partial_cross, d_output_cross,
                position, kHeads, kHeadDim, kLatent,
                d_latents, d_qeff_f32, d_prefix_partial, true);
        };
        const float no_splice_ms = median_ms(decode_no_splice, 30);
        const float scalar_ms = median_ms(decode_scalar, 30);
        const float parallel_ms = median_ms(decode_parallel, 30);
        std::printf(
            "decode%d no_splice=%.4f ms scalar_diagnostic=%.4f ms "
            "parallel=%.4f ms parallel_vs_diagnostic=%+.2f%% "
            "parallel_vs_no_splice=%+.2f%%\n",
            context, no_splice_ms, scalar_ms, parallel_ms,
            100.0f * (parallel_ms / scalar_ms - 1.0f),
            100.0f * (parallel_ms / no_splice_ms - 1.0f));
        if (context == kMaxContext) {
            long_no_splice = no_splice;
            long_scalar = scalar;
            long_parallel = parallel;
        }
    }

    // Exercise the exact production policy for DFlash-sized verify chunks.
    // Both candidates are correct splices; timings decide whether the shared
    // dispatch predicate selects repeated decode or the persistent prefill CTA.
    bool verify_dispatch_gate = true;
    constexpr std::array<int, 7> verify_contexts{
        257, 512, 1024, 2048, 4096, 8185, 8192};
    constexpr std::array<int, 2> verify_rows{1, 8};
    for (int context : verify_contexts) {
        const int base = context - 1;
        for (int rows : verify_rows) {
            if (base + rows > kMaxContext) continue;
            check(insignia::glm53::mla_prefill_latent_fp8_absorb(
                      d_queries, d_latents + size_t(base) * kLatent,
                      nullptr, d_scale_reference, d_cache_reference,
                      d_weights, d_weight_scales, d_output_reference,
                      rows, base), "verify exact prefill oracle");
            for (int row = 0; row < rows; ++row)
                check(insignia::glm53::mla_decode_latent_cross_head_fp8_absorb(
                          d_queries + size_t(row) * kWidth,
                          d_latents + size_t(base + row) * kLatent,
                          d_cache_cross, d_scale_cross,
                          d_weights, d_weight_scales,
                          d_qeff, d_qeff_scales, d_partial_cross,
                          d_output_cross + size_t(row) * kWidth,
                          base + row, kHeads, kHeadDim, kLatent,
                          d_latents, d_qeff_f32, d_prefix_partial, true),
                      "verify repeated exact-prefix decode");
            check(insignia::glm53::mla_prefill_latent_cross_head_fp8_absorb(
                      d_queries, d_latents + size_t(base) * kLatent,
                      d_cache_cross, d_scale_cross,
                      d_weights, d_weight_scales,
                      d_qeff, d_qeff_scales, d_output_scalar,
                      rows, base, kHeads, kHeadDim, kLatent,
                      d_latents, d_qeff_f32),
                  "verify fused exact-prefix prefill");
            check(cudaDeviceSynchronize(), "verify dispatch synchronize");
            const Metrics repeated = metrics(
                d_output_reference, d_output_cross, rows * kHeads);
            const Metrics fused = metrics(
                d_output_reference, d_output_scalar, rows * kHeads);
            const Metrics repeated_vs_fused = metrics(
                d_output_scalar, d_output_cross, rows * kHeads);
            char name[80];
            std::snprintf(name, sizeof(name),
                          "verify%d-rows%d-repeated", context, rows);
            print_metrics(name, repeated);
            std::snprintf(name, sizeof(name),
                          "verify%d-rows%d-fused", context, rows);
            print_metrics(name, fused);
            std::snprintf(name, sizeof(name),
                          "verify%d-rows%d-repeated-vs-fused", context, rows);
            print_metrics(name, repeated_vs_fused);
            verify_dispatch_gate = verify_dispatch_gate &&
                quality_gate(repeated, 1.0e-7, 2.5e-2, 0.9997,
                             1.0e-7, 3.0e-8) &&
                quality_gate(fused, 1.0e-7, 2.5e-2, 0.9997,
                             1.0e-7, 3.0e-8) &&
                quality_gate(repeated_vs_fused, 1.0e-10, 1.0e-4,
                             0.99999999, 1.0e-9, 3.0e-10);

            const auto repeated_call = [&] {
                cudaError_t status = cudaSuccess;
                for (int row = 0; row < rows && status == cudaSuccess; ++row)
                    status = insignia::glm53::mla_decode_latent_cross_head_fp8_absorb(
                        d_queries + size_t(row) * kWidth,
                        d_latents + size_t(base + row) * kLatent,
                        d_cache_cross, d_scale_cross,
                        d_weights, d_weight_scales,
                        d_qeff, d_qeff_scales, d_partial_cross,
                        d_output_cross + size_t(row) * kWidth,
                        base + row, kHeads, kHeadDim, kLatent,
                        d_latents, d_qeff_f32, d_prefix_partial, true);
                return status;
            };
            const auto fused_call = [&] {
                return insignia::glm53::mla_prefill_latent_cross_head_fp8_absorb(
                    d_queries, d_latents + size_t(base) * kLatent,
                    d_cache_cross, d_scale_cross,
                    d_weights, d_weight_scales,
                    d_qeff, d_qeff_scales, d_output_scalar,
                    rows, base, kHeads, kHeadDim, kLatent,
                    d_latents, d_qeff_f32);
            };
            const float repeated_ms = median_ms(repeated_call, 10);
            const float fused_ms = median_ms(fused_call, 10);
            const bool production_fused =
                insignia::glm53::mla_cross_head_use_fused_prefill(rows, base);
            const float chosen_ms = production_fused ? fused_ms : repeated_ms;
            const float best_ms = std::min(repeated_ms, fused_ms);
            verify_dispatch_gate = verify_dispatch_gate &&
                chosen_ms <= best_ms * 1.10f;
            std::printf(
                "verify%d rows%d repeated=%.4f ms fused=%.4f ms "
                "production=%s chosen_vs_best=%+.2f%%\n",
                context, rows, repeated_ms, fused_ms,
                production_fused ? "fused" : "repeated",
                100.0f * (chosen_ms / best_ms - 1.0f));
        }
    }

    // Reproduce PREFILL_CHUNK=96 crossing 192..287. Seed 0..191 as a prior
    // chunk, copy exactly the 64-row overlap, then run only the 256..287 suffix.
    constexpr int crossing_base = 192;
    constexpr int crossing_tokens = 96;
    constexpr int crossing_prefix =
        insignia::glm53::mla_exact_prefix_overlap_tokens(
            crossing_tokens, crossing_base);
    constexpr int crossing_long = crossing_tokens - crossing_prefix;
    check(cudaMemset(d_crossing_prefix, 0,
                     size_t(kPrefix) * kLatent * sizeof(float)),
          "clear crossing exact sidecar");
    check(cudaMemcpy(d_crossing_prefix, d_latents,
                     size_t(crossing_base) * kLatent * sizeof(float),
                     cudaMemcpyDeviceToDevice), "seed prior exact sidecar rows");
    check(cudaMemcpy(d_crossing_prefix + size_t(crossing_base) * kLatent,
                     d_latents + size_t(crossing_base) * kLatent,
                     size_t(crossing_prefix) * kLatent * sizeof(float),
                     cudaMemcpyDeviceToDevice), "copy clipped crossing overlap");
    check(insignia::glm53::mla_prefill_latent_fp8_absorb(
              d_queries, d_latents + size_t(crossing_base) * kLatent,
              nullptr, d_scale_reference, d_cache_reference,
              d_weights, d_weight_scales, d_output_reference,
              crossing_tokens, crossing_base), "crossing exact oracle");
    check(insignia::glm53::mla_prefill_latent_cross_head_fp8_absorb(
              d_queries + size_t(crossing_prefix) * kWidth,
              d_latents + size_t(kPrefix) * kLatent,
              d_cache_cross, d_scale_cross,
              d_weights, d_weight_scales,
              d_qeff, d_qeff_scales,
              d_output_cross + size_t(crossing_prefix) * kWidth,
              crossing_long, kPrefix, kHeads, kHeadDim, kLatent,
              d_crossing_prefix, d_qeff_f32),
          "crossing exact-prefix suffix prefill");
    check(cudaDeviceSynchronize(), "crossing synchronize");
    const Metrics crossing = metrics(
        d_output_reference + size_t(crossing_prefix) * kWidth,
        d_output_cross + size_t(crossing_prefix) * kWidth,
        crossing_long * kHeads);
    print_metrics("crossing192+96-suffix", crossing);
    const bool crossing_gate = quality_gate(
        crossing, 1.0e-7, 2.5e-2, 0.9997, 1.0e-7, 3.0e-8);

    const bool pass =
        quality_gate(decode_scalar, 1.0e-7, 5.0e-3, 0.99999,
                     1.0e-8, 3.0e-9) &&
        quality_gate(decode_parallel, 1.0e-7, 5.0e-3, 0.99999,
                     1.0e-8, 3.0e-9) &&
        quality_gate(prefill, 1.0e-7, 1.2e-2, 0.9999,
                     1.0e-8, 3.0e-9) &&
        quality_gate(decode_fast_vs_scalar, 1.0e-10, 1.0e-4,
                     0.99999999, 1.0e-9, 3.0e-10) &&
        invalid_pair_gate && scalar_context_gate && parallel_context_gate &&
        fast_vs_scalar_gate &&
        verify_dispatch_gate && crossing_gate &&
        long_parallel.rel_l2 <= long_no_splice.rel_l2 &&
        quality_gate(long_scalar, 1.0e-7, 2.5e-2, 0.9997,
                     1.0e-7, 3.0e-8) &&
        quality_gate(long_parallel, 1.0e-7, 2.5e-2, 0.9997,
                     1.0e-7, 3.0e-8);
    if (!pass) {
        std::fprintf(stderr, "FAIL exact-prefix rows were not spliced\n");
        return 2;
    }
    std::puts("PASS exact-prefix decode and prefill splice");
    return 0;
}
