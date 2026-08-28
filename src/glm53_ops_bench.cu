#include "insignia_glm53.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void check(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(2);
    }
}

template <typename T>
T *device_alloc(size_t count) {
    T *pointer = nullptr;
    check(cudaMalloc(&pointer, count * sizeof(T)), "cudaMalloc");
    return pointer;
}

template <typename T>
T *device_copy(const std::vector<T> &source) {
    T *pointer = device_alloc<T>(source.size());
    check(cudaMemcpy(pointer, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    return pointer;
}

uint16_t to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return uint16_t(bits >> 16);
}

float from_bf16(uint16_t value) {
    const uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

struct Metrics {
    double relative;
    double cosine;
    double maximum;
};

Metrics compare(const std::vector<float> &actual, const std::vector<float> &reference) {
    double error2 = 0.0, actual2 = 0.0, reference2 = 0.0, dot = 0.0, maximum = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double error = double(actual[i]) - reference[i];
        error2 += error * error;
        actual2 += double(actual[i]) * actual[i];
        reference2 += double(reference[i]) * reference[i];
        dot += double(actual[i]) * reference[i];
        maximum = std::max(maximum, std::abs(error));
    }
    return {std::sqrt(error2 / std::max(reference2, 1.0e-300)),
            dot / std::sqrt(std::max(actual2 * reference2, 1.0e-300)), maximum};
}

float benchmark_ms(void (*launch)(void *), void *context, int iterations) {
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin), "cudaEventCreate");
    check(cudaEventCreate(&end), "cudaEventCreate");
    // Ada takes tens of milliseconds to leave P8.  A tiny warmup benchmarks
    // the boost governor instead of the kernel.
    for (int i = 0; i < 10000; ++i) launch(context);
    check(cudaDeviceSynchronize(), "benchmark warmup");
    check(cudaEventRecord(begin), "cudaEventRecord begin");
    for (int i = 0; i < iterations; ++i) launch(context);
    check(cudaEventRecord(end), "cudaEventRecord end");
    check(cudaEventSynchronize(end), "cudaEventSynchronize");
    float elapsed = 0.0f;
    check(cudaEventElapsedTime(&elapsed, begin, end), "cudaEventElapsedTime");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return elapsed / iterations;
}

struct MhcDevice {
    uint16_t *fn;
    float *base;
    float *scale;
    float *streams;
    float *post;
    float *comb;
    float *collapsed;
    float *sublayer;
    float *mixed;
    void *workspace;
};

void launch_mhc(void *opaque) {
    auto &device = *static_cast<MhcDevice *>(opaque);
    check(insignia::glm53::mhc_analyze(device.fn, device.base, device.scale, device.streams,
        nullptr,
        device.post, device.comb, device.collapsed, device.workspace), "mhc_analyze");
}

void launch_mhc_mix(void *opaque) {
    auto &device = *static_cast<MhcDevice *>(opaque);
    check(insignia::glm53::mhc_mix(device.streams, device.sublayer, device.post,
        device.comb, device.mixed), "mhc_mix");
}

void test_mhc(std::mt19937 &rng) {
    using namespace insignia::glm53;
    constexpr int flat = kHyperStreams * kHidden;
    std::normal_distribution<float> normal(0.0f, 0.08f);
    std::vector<float> streams(flat), base(24), scale{0.6f, 0.7f, 0.5f}, sublayer(kHidden);
    std::vector<uint16_t> fn(size_t(24) * flat);
    for (float &value : streams) value = normal(rng);
    for (float &value : sublayer) value = normal(rng);
    for (float &value : base) value = normal(rng);
    for (uint16_t &value : fn) value = to_bf16(normal(rng));

    double sum2 = 0.0;
    for (float value : streams) sum2 += double(value) * value;
    const float inverse_rms = 1.0f / std::sqrt(float(sum2 / flat) + 1.0e-5f);
    std::vector<float> params(24, 0.0f);
    for (int row = 0; row < 24; ++row) {
        double sum = 0.0;
        for (int i = 0; i < flat; ++i) sum += double(from_bf16(fn[size_t(row) * flat + i])) * streams[i];
        params[row] = float(sum) * inverse_rms;
    }
    std::vector<float> pre(4), post(4), comb(16), collapsed(kHidden), mixed(flat);
    for (int i = 0; i < 4; ++i) {
        pre[i] = sigmoid(params[i] * scale[0] + base[i]) + 1.0e-6f;
        post[i] = 2.0f * sigmoid(params[4 + i] * scale[1] + base[4 + i]);
    }
    for (int row = 0; row < 4; ++row) {
        float maximum = -INFINITY;
        for (int col = 0; col < 4; ++col) {
            comb[row * 4 + col] = params[8 + row * 4 + col] * scale[2] + base[8 + row * 4 + col];
            maximum = std::max(maximum, comb[row * 4 + col]);
        }
        float denominator = 0.0f;
        for (int col = 0; col < 4; ++col) denominator += std::exp(comb[row * 4 + col] - maximum);
        for (int col = 0; col < 4; ++col) comb[row * 4 + col] = std::exp(comb[row * 4 + col] - maximum) / denominator + 1.0e-6f;
    }
    for (int col = 0; col < 4; ++col) {
        float denominator = 1.0e-6f;
        for (int row = 0; row < 4; ++row) denominator += comb[row * 4 + col];
        for (int row = 0; row < 4; ++row) comb[row * 4 + col] /= denominator;
    }
    for (int iteration = 1; iteration < 20; ++iteration) {
        for (int row = 0; row < 4; ++row) {
            float denominator = 1.0e-6f;
            for (int col = 0; col < 4; ++col) denominator += comb[row * 4 + col];
            for (int col = 0; col < 4; ++col) comb[row * 4 + col] /= denominator;
        }
        for (int col = 0; col < 4; ++col) {
            float denominator = 1.0e-6f;
            for (int row = 0; row < 4; ++row) denominator += comb[row * 4 + col];
            for (int row = 0; row < 4; ++row) comb[row * 4 + col] /= denominator;
        }
    }
    for (int dimension = 0; dimension < kHidden; ++dimension) {
        collapsed[dimension] = 0.0f;
        for (int stream = 0; stream < 4; ++stream)
            collapsed[dimension] += pre[stream] * streams[stream * kHidden + dimension];
        for (int stream = 0; stream < 4; ++stream) {
            float residual = 0.0f;
            for (int source = 0; source < 4; ++source)
                residual += comb[source * 4 + stream] * streams[source * kHidden + dimension];
            mixed[stream * kHidden + dimension] = post[stream] * sublayer[dimension] + residual;
        }
    }

    MhcDevice device{device_copy(fn), device_copy(base), device_copy(scale), device_copy(streams),
        device_alloc<float>(4), device_alloc<float>(16), device_alloc<float>(kHidden),
        device_copy(sublayer), device_alloc<float>(flat), nullptr};
    check(cudaMalloc(&device.workspace, mhc_workspace_bytes()), "cudaMalloc mHC workspace");
    launch_mhc(&device);
    launch_mhc_mix(&device);
    check(cudaDeviceSynchronize(), "mHC synchronize");
    std::vector<float> gpu_post(4), gpu_comb(16), gpu_collapsed(kHidden), gpu_mixed(flat);
    check(cudaMemcpy(gpu_post.data(), device.post, sizeof(float) * 4, cudaMemcpyDeviceToHost), "copy mHC post");
    check(cudaMemcpy(gpu_comb.data(), device.comb, sizeof(float) * 16, cudaMemcpyDeviceToHost), "copy mHC comb");
    check(cudaMemcpy(gpu_collapsed.data(), device.collapsed, sizeof(float) * kHidden, cudaMemcpyDeviceToHost), "copy mHC collapsed");
    check(cudaMemcpy(gpu_mixed.data(), device.mixed, sizeof(float) * flat, cudaMemcpyDeviceToHost), "copy mHC mixed");
    const Metrics post_error = compare(gpu_post, post);
    const Metrics comb_error = compare(gpu_comb, comb);
    const Metrics collapse_error = compare(gpu_collapsed, collapsed);
    const Metrics mix_error = compare(gpu_mixed, mixed);
    const float analyze_ms = benchmark_ms(launch_mhc, &device, 1000);
    const float mix_ms = benchmark_ms(launch_mhc_mix, &device, 2000);
    std::printf("mHC analyze %.3f us, mix %.3f us | post rel %.3g, comb rel %.3g, collapse rel %.3g, mix rel %.3g\n",
        analyze_ms * 1000.0f, mix_ms * 1000.0f, post_error.relative, comb_error.relative,
        collapse_error.relative, mix_error.relative);
    if (collapse_error.relative > 2.0e-5 || mix_error.relative > 2.0e-5) std::exit(3);
    cudaFree(device.fn); cudaFree(device.base); cudaFree(device.scale); cudaFree(device.streams);
    cudaFree(device.post); cudaFree(device.comb); cudaFree(device.collapsed); cudaFree(device.sublayer);
    cudaFree(device.mixed); cudaFree(device.workspace);
}

struct KdaDevice {
    float *state;
    float *q;
    float *k;
    float *v;
    float *g;
    float *beta;
    float *output;
};

struct KdaRing {
    KdaDevice *device;
    float *states;
    int layer;
};

void launch_kda(void *opaque) {
    auto &device = *static_cast<KdaDevice *>(opaque);
    check(insignia::glm53::kda_decode(device.state, device.q, device.k, device.v,
        device.g, device.beta, device.output), "kda_decode");
}

void launch_kda_ring(void *opaque) {
    auto &ring = *static_cast<KdaRing *>(opaque);
    constexpr size_t state_size = size_t(insignia::glm53::kKdaHeads) *
                                  insignia::glm53::kKdaHeadDim * insignia::glm53::kKdaHeadDim;
    check(insignia::glm53::kda_decode(ring.states + size_t(ring.layer) * state_size,
        ring.device->q, ring.device->k, ring.device->v, ring.device->g,
        ring.device->beta, ring.device->output), "kda_decode ring");
    ring.layer = ring.layer == 33 ? 0 : ring.layer + 1;
}

void test_kda(std::mt19937 &rng) {
    using namespace insignia::glm53;
    constexpr int vector_size = kKdaHeads * kKdaHeadDim;
    constexpr int state_size = kKdaHeads * kKdaHeadDim * kKdaHeadDim;
    std::normal_distribution<float> normal(0.0f, 0.08f);
    std::uniform_real_distribution<float> decay_gate(-4.0f, -0.05f);
    std::uniform_real_distribution<float> beta_distribution(0.1f, 0.9f);
    std::vector<float> state(state_size), q(vector_size), k(vector_size), v(vector_size),
        g(vector_size), beta(kKdaHeads), reference_output(vector_size);
    for (float &value : state) value = normal(rng) * 0.1f;
    for (float &value : q) value = normal(rng);
    for (float &value : k) value = normal(rng);
    for (float &value : v) value = normal(rng);
    for (float &value : g) value = decay_gate(rng);
    for (float &value : beta) value = beta_distribution(rng);
    std::vector<float> reference_state = state;
    for (int head = 0; head < kKdaHeads; ++head) {
        const int offset = head * kKdaHeadDim;
        double qsum = 1.0e-6, ksum = 1.0e-6;
        for (int i = 0; i < kKdaHeadDim; ++i) {
            qsum += double(q[offset + i]) * q[offset + i];
            ksum += double(k[offset + i]) * k[offset + i];
        }
        const float qscale = float(1.0 / std::sqrt(qsum)) / std::sqrt(float(kKdaHeadDim));
        const float kscale = float(1.0 / std::sqrt(ksum));
        for (int value = 0; value < kKdaHeadDim; ++value) {
            double memory = 0.0;
            for (int key = 0; key < kKdaHeadDim; ++key) {
                const size_t index = (size_t(head) * kKdaHeadDim + key) * kKdaHeadDim + value;
                memory += double(reference_state[index]) * std::exp(g[offset + key]) * (k[offset + key] * kscale);
            }
            const float delta = (v[offset + value] - float(memory)) * beta[head];
            double output = 0.0;
            for (int key = 0; key < kKdaHeadDim; ++key) {
                const size_t index = (size_t(head) * kKdaHeadDim + key) * kKdaHeadDim + value;
                reference_state[index] = reference_state[index] * std::exp(g[offset + key]) + (k[offset + key] * kscale) * delta;
                output += double(reference_state[index]) * (q[offset + key] * qscale);
            }
            reference_output[offset + value] = float(output);
        }
    }
    KdaDevice device{device_copy(state), device_copy(q), device_copy(k), device_copy(v),
        device_copy(g), device_copy(beta), device_alloc<float>(vector_size)};
    launch_kda(&device);
    check(cudaDeviceSynchronize(), "KDA synchronize");
    std::vector<float> gpu_state(state_size), gpu_output(vector_size);
    check(cudaMemcpy(gpu_state.data(), device.state, sizeof(float) * state_size, cudaMemcpyDeviceToHost), "copy KDA state");
    check(cudaMemcpy(gpu_output.data(), device.output, sizeof(float) * vector_size, cudaMemcpyDeviceToHost), "copy KDA output");
    const Metrics state_error = compare(gpu_state, reference_state);
    const Metrics output_error = compare(gpu_output, reference_output);
    const float hot_ms = benchmark_ms(launch_kda, &device, 1000);
    float *state_ring = device_alloc<float>(size_t(state_size) * 34);
    for (int layer = 0; layer < 34; ++layer)
        check(cudaMemcpy(state_ring + size_t(layer) * state_size, device.state,
                         sizeof(float) * state_size, cudaMemcpyDeviceToDevice), "initialize KDA state ring");
    KdaRing ring{&device, state_ring, 0};
    const float ring_ms = benchmark_ms(launch_kda_ring, &ring, 680);
    const double traffic_gib = double(3) * state_size * sizeof(float) / (double(1ull << 30));
    std::printf("KDA decode hot %.3f us (%.1f GiB/s), 34-layer ring %.3f us (%.1f GiB/s) | state rel %.3g cos %.9f, output rel %.3g cos %.9f max %.3g\n",
        hot_ms * 1000.0f, traffic_gib / (hot_ms * 1.0e-3), ring_ms * 1000.0f,
        traffic_gib / (ring_ms * 1.0e-3), state_error.relative,
        state_error.cosine, output_error.relative, output_error.cosine, output_error.maximum);
    if (state_error.relative > 2.0e-5 || output_error.relative > 2.0e-5) std::exit(4);
    cudaFree(device.state); cudaFree(device.q); cudaFree(device.k); cudaFree(device.v);
    cudaFree(device.g); cudaFree(device.beta); cudaFree(device.output);
    cudaFree(state_ring);
}

struct ConvDevice {
    float *projection;
    uint16_t *conv;
    float *history;
    int position;
};

void launch_conv(void *opaque) {
    auto &device = *static_cast<ConvDevice *>(opaque);
    check(insignia::glm53::kda_conv_silu(device.projection, device.conv,
        device.history, device.position++), "kda_conv_silu");
}

void test_conv(std::mt19937 &rng) {
    using namespace insignia::glm53;
    constexpr int count = kKdaHeads * kKdaHeadDim;
    constexpr int position = 5;
    std::normal_distribution<float> normal(0.0f, 0.08f);
    std::vector<float> projection(count), history(size_t(3) * count);
    std::vector<uint16_t> conv(size_t(count) * 4);
    for (float &value : projection) value = normal(rng);
    for (float &value : history) value = normal(rng);
    for (uint16_t &value : conv) value = to_bf16(normal(rng));
    std::vector<float> reference_projection = projection;
    std::vector<float> reference_history = history;
    for (int index = 0; index < count; ++index) {
        const float current = projection[index];
        float value = current * from_bf16(conv[size_t(index) * 4 + 3]);
        for (int lag = 1; lag <= 3; ++lag) {
            const int slot = (position - lag) % 3;
            value = std::fma(reference_history[size_t(slot) * count + index],
                             from_bf16(conv[size_t(index) * 4 + 3 - lag]), value);
        }
        reference_history[size_t(position % 3) * count + index] = current;
        reference_projection[index] = value / (1.0f + std::exp(-value));
    }

    ConvDevice device{device_copy(projection), device_copy(conv), device_copy(history), position};
    launch_conv(&device);
    check(cudaDeviceSynchronize(), "convolution synchronize");
    std::vector<float> gpu_projection(count), gpu_history(size_t(3) * count);
    check(cudaMemcpy(gpu_projection.data(), device.projection, sizeof(float) * count,
                     cudaMemcpyDeviceToHost), "copy convolution output");
    check(cudaMemcpy(gpu_history.data(), device.history, sizeof(float) * gpu_history.size(),
                     cudaMemcpyDeviceToHost), "copy convolution history");
    const Metrics projection_error = compare(gpu_projection, reference_projection);
    const Metrics history_error = compare(gpu_history, reference_history);
    const float elapsed_ms = benchmark_ms(launch_conv, &device, 2000);
    std::printf("KDA conv+SiLU %.3f us | output rel %.3g cos %.9f, history rel %.3g\n",
        elapsed_ms * 1000.0f, projection_error.relative, projection_error.cosine,
        history_error.relative);
    if (projection_error.relative > 2.0e-6 || history_error.relative > 2.0e-6) std::exit(5);
    cudaFree(device.projection); cudaFree(device.conv); cudaFree(device.history);
}

struct MlaDevice {
    float *query;
    float *kv;
    float *key_cache;
    float *value_cache;
    float *output;
    int position;
};

void launch_mla(void *opaque) {
    auto &device = *static_cast<MlaDevice *>(opaque);
    check(insignia::glm53::mla_decode(device.query, device.kv, device.key_cache,
        device.value_cache, device.output, device.position), "mla_decode");
}

struct MlaPrefillDevice {
    float *query;
    float *kv;
    float *scalar_keys;
    float *scalar_values;
    float *scalar_output;
    float *flash_keys;
    float *flash_values;
    float *flash_output;
    int tokens;
};

void launch_mla_scalar_prefill(void *opaque) {
    auto &device = *static_cast<MlaPrefillDevice *>(opaque);
    constexpr int width = insignia::glm53::kKdaHeads * insignia::glm53::kMlaHeadDim;
    for (int token = 0; token < device.tokens; ++token)
        check(insignia::glm53::mla_decode(
                  device.query + size_t(token) * width,
                  device.kv + size_t(token) * 2 * width,
                  device.scalar_keys, device.scalar_values,
                  device.scalar_output + size_t(token) * width, token),
              "scalar MLA prefill");
}

void launch_mla_flash2_prefill(void *opaque) {
    auto &device = *static_cast<MlaPrefillDevice *>(opaque);
    check(insignia::glm53::mla_flash2_prefill(
              device.query, device.kv, device.flash_keys, device.flash_values,
              device.flash_output, device.tokens, 0),
          "FlashAttention-2 MLA prefill");
}

void test_mla(std::mt19937 &rng) {
    using namespace insignia::glm53;
    constexpr int vector_size = kKdaHeads * kMlaHeadDim;
    constexpr int position = 7;
    constexpr size_t cache_size = size_t(kMlaMaxContext) * vector_size;
    std::normal_distribution<float> normal(0.0f, 0.08f);
    std::vector<float> query(vector_size), kv(size_t(kKdaHeads) * 2 * kMlaHeadDim),
        key_cache(cache_size, 0.0f), value_cache(cache_size, 0.0f),
        reference_output(vector_size);
    for (float &value : query) value = normal(rng);
    for (float &value : kv) value = normal(rng);
    for (int token = 0; token < position; ++token) {
        for (int index = 0; index < vector_size; ++index) {
            key_cache[size_t(token) * vector_size + index] = normal(rng);
            value_cache[size_t(token) * vector_size + index] = normal(rng);
        }
    }
    std::vector<float> reference_keys = key_cache;
    std::vector<float> reference_values = value_cache;
    for (int head = 0; head < kKdaHeads; ++head) {
        for (int element = 0; element < kMlaHeadDim; ++element) {
            const int vector_index = head * kMlaHeadDim + element;
            reference_keys[size_t(position) * vector_size + vector_index] =
                kv[size_t(head) * 2 * kMlaHeadDim + element];
            reference_values[size_t(position) * vector_size + vector_index] =
                kv[size_t(head) * 2 * kMlaHeadDim + kMlaHeadDim + element];
        }
        float logits[position + 1];
        float maximum = -INFINITY;
        for (int token = 0; token <= position; ++token) {
            double dot = 0.0;
            for (int element = 0; element < kMlaHeadDim; ++element) {
                const int vector_index = head * kMlaHeadDim + element;
                dot += double(query[vector_index]) *
                       reference_keys[size_t(token) * vector_size + vector_index];
            }
            logits[token] = float(dot) * (1.0f / 16.0f);
            maximum = std::max(maximum, logits[token]);
        }
        float denominator = 0.0f;
        for (int token = 0; token <= position; ++token) {
            logits[token] = std::exp(logits[token] - maximum);
            denominator += logits[token];
        }
        for (int element = 0; element < kMlaHeadDim; ++element) {
            const int vector_index = head * kMlaHeadDim + element;
            double result = 0.0;
            for (int token = 0; token <= position; ++token)
                result += double(logits[token] / denominator) *
                          reference_values[size_t(token) * vector_size + vector_index];
            reference_output[vector_index] = float(result);
        }
    }

    MlaDevice device{device_copy(query), device_copy(kv), device_copy(key_cache),
        device_copy(value_cache), device_alloc<float>(vector_size), position};
    launch_mla(&device);
    check(cudaDeviceSynchronize(), "MLA synchronize");
    std::vector<float> gpu_output(vector_size);
    check(cudaMemcpy(gpu_output.data(), device.output, sizeof(float) * vector_size,
                     cudaMemcpyDeviceToHost), "copy MLA output");
    const Metrics output_error = compare(gpu_output, reference_output);
    const float elapsed_ms = benchmark_ms(launch_mla, &device, 2000);
    std::printf("MLA decode (context 8) %.3f us | output rel %.3g cos %.9f max %.3g\n",
        elapsed_ms * 1000.0f, output_error.relative, output_error.cosine, output_error.maximum);
    if (output_error.relative > 2.0e-5) std::exit(6);
    cudaFree(device.query); cudaFree(device.kv); cudaFree(device.key_cache);
    cudaFree(device.value_cache); cudaFree(device.output);
}

void test_mla_prefill(std::mt19937 &rng) {
    using namespace insignia::glm53;
    constexpr int tokens = 16;
    constexpr int width = kKdaHeads * kMlaHeadDim;
    constexpr size_t cache_size = size_t(kMlaMaxContext) * width;
    std::normal_distribution<float> normal(0.0f, 0.08f);
    std::vector<float> query(size_t(tokens) * width), kv(size_t(tokens) * 2 * width),
        empty_cache(cache_size, 0.0f);
    for (float &value : query) value = normal(rng);
    for (float &value : kv) value = normal(rng);
    MlaPrefillDevice device{device_copy(query), device_copy(kv),
        device_copy(empty_cache), device_copy(empty_cache), device_alloc<float>(query.size()),
        device_copy(empty_cache), device_copy(empty_cache), device_alloc<float>(query.size()),
        tokens};
    launch_mla_scalar_prefill(&device);
    launch_mla_flash2_prefill(&device);
    check(cudaDeviceSynchronize(), "MLA prefill synchronize");
    std::vector<float> scalar_output(query.size()), flash_output(query.size());
    check(cudaMemcpy(scalar_output.data(), device.scalar_output,
                     scalar_output.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "copy scalar MLA prefill");
    check(cudaMemcpy(flash_output.data(), device.flash_output,
                     flash_output.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "copy FlashAttention-2 MLA prefill");
    const Metrics error = compare(flash_output, scalar_output);
    const float scalar_ms = benchmark_ms(launch_mla_scalar_prefill, &device, 2000);
    const float flash_ms = benchmark_ms(launch_mla_flash2_prefill, &device, 2000);
    std::printf("MLA FA2 prefill (%d tokens) scalar %.3f us, flash %.3f us, %.2fx faster | "
                "rel %.3g cos %.9f max %.3g\n",
                tokens, scalar_ms * 1000.0f, flash_ms * 1000.0f,
                scalar_ms / flash_ms, error.relative, error.cosine, error.maximum);
    if (error.relative > 2.0e-5) std::exit(7);
    cudaFree(device.query); cudaFree(device.kv);
    cudaFree(device.scalar_keys); cudaFree(device.scalar_values); cudaFree(device.scalar_output);
    cudaFree(device.flash_keys); cudaFree(device.flash_values); cudaFree(device.flash_output);
}

}  // namespace

int main() {
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    if (properties.major != 8 || properties.minor != 9) {
        std::fprintf(stderr, "GLM-5.3 ops require sm_89, got sm_%d%d\n", properties.major, properties.minor);
        return 1;
    }
    std::mt19937 rng(0x53f1a5u);
    test_mhc(rng);
    test_kda(rng);
    test_conv(rng);
    test_mla(rng);
    test_mla_prefill(rng);
    return 0;
}
