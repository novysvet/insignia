#include "insignia_glm53.cuh"
#include "insignia_glm53_index.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using insignia::glm53::ShardedIndex;
using insignia::glm53::TensorLocation;

void check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(2);
    }
}

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

__global__ void swiglu_kernel(const float *__restrict__ gate, const float *__restrict__ up,
                              float *__restrict__ output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float g = fminf(gate[index], 10.0f);
        const float u = fminf(fmaxf(up[index], -10.0f), 10.0f);
        output[index] = (g / (1.0f + __expf(-g))) * u;
    }
}

struct Metrics {
    double relative;
    double cosine;
    double maximum;
};

Metrics compare(const std::vector<float> &actual, const std::vector<float> &reference) {
    double dot = 0.0, aa = 0.0, rr = 0.0, error = 0.0, maximum = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const double a = actual[index], r = reference[index], delta = a - r;
        dot += a * r;
        aa += a * a;
        rr += r * r;
        error += delta * delta;
        maximum = std::max(maximum, std::abs(delta));
    }
    return {std::sqrt(error / rr), dot / std::sqrt(aa * rr), maximum};
}

template <typename Launch>
float gpu_time(Launch launch, int iterations) {
    for (int index = 0; index < 30; ++index) launch();
    check(cudaDeviceSynchronize(), "warmup");
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin), "event create");
    check(cudaEventCreate(&end), "event create");
    check(cudaEventRecord(begin), "event record");
    for (int index = 0; index < iterations; ++index) launch();
    check(cudaEventRecord(end), "event record");
    check(cudaEventSynchronize(end), "event sync");
    float milliseconds = 0.0f;
    check(cudaEventElapsedTime(&milliseconds, begin, end), "event elapsed");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return milliseconds / iterations;
}

struct Expert {
    const TensorLocation *down_weight;
    const TensorLocation *gate_weight;
    const TensorLocation *up_weight;
    const TensorLocation *down_scale;
    const TensorLocation *gate_scale;
    const TensorLocation *up_scale;
    const TensorLocation *down_global;
    const TensorLocation *gate_global;
    const TensorLocation *up_global;
};

Expert locate(ShardedIndex &model, int layer, int expert) {
    const std::string stem = "model.language_model.layers." + std::to_string(layer) +
                             ".mlp.experts." + std::to_string(expert) + ".";
    return {
        &model.tensor(stem + "down_proj.weight"),
        &model.tensor(stem + "gate_proj.weight"),
        &model.tensor(stem + "up_proj.weight"),
        &model.tensor(stem + "down_proj.weight_scale"),
        &model.tensor(stem + "gate_proj.weight_scale"),
        &model.tensor(stem + "up_proj.weight_scale"),
        &model.tensor(stem + "down_proj.weight_scale_2"),
        &model.tensor(stem + "gate_proj.weight_scale_2"),
        &model.tensor(stem + "up_proj.weight_scale_2"),
    };
}

void read_triplet(ShardedIndex &model, const TensorLocation *first, const TensorLocation *second,
                  const TensorLocation *third, void *destination, bool direct) {
    auto *output = static_cast<uint8_t *>(destination);
    if (first->shard == second->shard && second->shard == third->shard &&
        first->offset + first->bytes == second->offset && second->offset + second->bytes == third->offset) {
        if (direct)
            model.read_span_direct(first->shard, first->offset, first->bytes + second->bytes + third->bytes, output);
        else
            model.read_span(first->shard, first->offset, first->bytes + second->bytes + third->bytes, output);
        return;
    }
    const TensorLocation *parts[3] = {first, second, third};
    size_t offset = 0;
    for (const TensorLocation *part : parts) {
        if (direct) model.read_span_direct(part->shard, part->offset, part->bytes, output + offset);
        else model.read(*part, output + offset);
        offset += size_t(part->bytes);
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr, "usage: %s MODEL_ROOT MODEL.index [LAYER=3] [EXPERT=0]\n", argv[0]);
        return 64;
    }
    const int layer = argc > 3 ? std::atoi(argv[3]) : 3;
    const int expert_id = argc > 4 ? std::atoi(argv[4]) : 0;
    try {
        ShardedIndex model(argv[2], argv[1]);
        Expert expert = locate(model, layer, expert_id);
        require(expert.down_weight->bytes == 4ull << 20 && expert.gate_weight->bytes == 4ull << 20 &&
                expert.up_weight->bytes == 4ull << 20, "unexpected GLM-5.3 expert weight geometry");
        require(expert.down_scale->bytes == 512ull << 10 && expert.gate_scale->bytes == 512ull << 10 &&
                expert.up_scale->bytes == 512ull << 10, "unexpected GLM-5.3 expert scale geometry");

        constexpr size_t body_bytes = 12ull << 20;
        constexpr size_t scale_bytes = 1536ull << 10;
        uint8_t *host_body = nullptr, *host_scale = nullptr;
        check(cudaHostAlloc(&host_body, body_bytes, cudaHostAllocDefault), "pinned body allocation");
        check(cudaHostAlloc(&host_scale, scale_bytes, cudaHostAllocDefault), "pinned scale allocation");
        float global[3];

        const auto io_begin = std::chrono::steady_clock::now();
        read_triplet(model, expert.down_weight, expert.gate_weight, expert.up_weight, host_body, true);
        read_triplet(model, expert.down_scale, expert.gate_scale, expert.up_scale, host_scale, true);
        model.read(*expert.down_global, global + 0);
        model.read(*expert.gate_global, global + 1);
        model.read(*expert.up_global, global + 2);
        const auto io_end = std::chrono::steady_clock::now();
        const double io_seconds = std::chrono::duration<double>(io_end - io_begin).count();

        uint8_t *device_body = nullptr, *device_scale = nullptr;
        float *x = nullptr, *gate = nullptr, *up = nullptr, *activation = nullptr, *output = nullptr;
        void *workspace_4096 = nullptr, *workspace_2048 = nullptr;
        check(cudaMalloc(&device_body, body_bytes), "device body allocation");
        check(cudaMalloc(&device_scale, scale_bytes), "device scale allocation");
        check(cudaMalloc(&x, 4096 * sizeof(float)), "x allocation");
        check(cudaMalloc(&gate, 2048 * sizeof(float)), "gate allocation");
        check(cudaMalloc(&up, 2048 * sizeof(float)), "up allocation");
        check(cudaMalloc(&activation, 2048 * sizeof(float)), "activation allocation");
        check(cudaMalloc(&output, 4096 * sizeof(float)), "output allocation");
        check(cudaMalloc(&workspace_4096, insignia::glm53::nvfp4_workspace_bytes(4096)), "workspace 4096");
        check(cudaMalloc(&workspace_2048, insignia::glm53::nvfp4_workspace_bytes(2048)), "workspace 2048");
        check(insignia::glm53::initialize_nvfp4(), "NVFP4 initialization");

        std::vector<float> host_x(4096);
        uint32_t random = 0x53f1a5u;
        double square_sum = 0.0;
        for (float &value : host_x) {
            random = random * 1664525u + 1013904223u;
            value = (float(int32_t(random)) / 2147483648.0f);
            square_sum += double(value) * value;
        }
        const float inverse_rms = float(std::sqrt(host_x.size() / square_sum));
        for (float &value : host_x) value *= inverse_rms;
        check(cudaMemcpy(x, host_x.data(), host_x.size() * sizeof(float), cudaMemcpyHostToDevice), "x upload");

        cudaEvent_t upload_begin, upload_end;
        check(cudaEventCreate(&upload_begin), "upload event");
        check(cudaEventCreate(&upload_end), "upload event");
        check(cudaEventRecord(upload_begin), "upload record");
        check(cudaMemcpyAsync(device_body, host_body, body_bytes, cudaMemcpyHostToDevice), "body upload");
        check(cudaMemcpyAsync(device_scale, host_scale, scale_bytes, cudaMemcpyHostToDevice), "scale upload");
        check(cudaEventRecord(upload_end), "upload record");
        check(cudaEventSynchronize(upload_end), "upload sync");
        float upload_ms = 0.0f;
        check(cudaEventElapsedTime(&upload_ms, upload_begin, upload_end), "upload elapsed");

        const uint8_t *down_w = device_body;
        const uint8_t *gate_w = device_body + (4ull << 20);
        const uint8_t *up_w = device_body + (8ull << 20);
        const uint8_t *down_s = device_scale;
        const uint8_t *gate_s = device_scale + (512ull << 10);
        const uint8_t *up_s = device_scale + (1024ull << 10);

        auto launch_float = [&] {
            check(insignia::glm53::nvfp4_gemv_f32(gate_w, gate_s, global[1], x, gate, 2048, 4096), "float gate");
            check(insignia::glm53::nvfp4_gemv_f32(up_w, up_s, global[2], x, up, 2048, 4096), "float up");
            swiglu_kernel<<<8, 256>>>(gate, up, activation, 2048);
            check(insignia::glm53::nvfp4_gemv_f32(down_w, down_s, global[0], activation, output, 4096, 2048), "float down");
        };
        const float float_ms = gpu_time(launch_float, 500);
        launch_float();
        check(cudaDeviceSynchronize(), "float completion");
        std::vector<float> reference(4096);
        check(cudaMemcpy(reference.data(), output, reference.size() * sizeof(float), cudaMemcpyDeviceToHost), "float result");

        auto launch_dp4a = [&] {
            check(insignia::glm53::nvfp4_quantize_activation(x, 4096, workspace_4096), "Q8 input");
            check(insignia::glm53::nvfp4_gemv2_dp4a_quantized(
                gate_w, gate_s, global[1], up_w, up_s, global[2], workspace_4096,
                gate, up, 2048, 4096), "DP4A gate/up");
            check(insignia::glm53::quantize_swiglu_activation(gate, up, 2048, workspace_2048), "SwiGLU Q8");
            check(insignia::glm53::nvfp4_gemv_dp4a_quantized(
                down_w, down_s, global[0], workspace_2048, output, 4096, 2048), "DP4A down");
        };
        const float dp4a_ms = gpu_time(launch_dp4a, 1000);
        launch_dp4a();
        check(cudaDeviceSynchronize(), "DP4A completion");
        std::vector<float> actual(4096);
        check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "DP4A result");
        const Metrics metrics = compare(actual, reference);

        std::printf("GLM-5.3 layer %d expert %d: %.2f MiB direct read %.3f ms (%.2f GB/s)\n",
                    layer, expert_id, (body_bytes + scale_bytes) / double(1 << 20), io_seconds * 1000.0,
                    (body_bytes + scale_bytes) / io_seconds / 1e9);
        std::printf("pinned H2D %.3f ms (%.2f GB/s)\n", upload_ms,
                    (body_bytes + scale_bytes) / (upload_ms / 1000.0) / 1e9);
        std::printf("expert float %.3f us; DP4A fused %.3f us (%.2fx) rel=%.6f cos=%.9f max=%g\n",
                    float_ms * 1000.0f, dp4a_ms * 1000.0f, float_ms / dp4a_ms,
                    metrics.relative, metrics.cosine, metrics.maximum);

        cudaEventDestroy(upload_begin); cudaEventDestroy(upload_end);
        cudaFreeHost(host_body); cudaFreeHost(host_scale);
        cudaFree(device_body); cudaFree(device_scale); cudaFree(x); cudaFree(gate); cudaFree(up);
        cudaFree(activation); cudaFree(output); cudaFree(workspace_4096); cudaFree(workspace_2048);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "glm53-stream-bench: %s\n", error.what());
        return 1;
    }
    return 0;
}
