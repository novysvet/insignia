#include "insignia_glm53_q3.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kRows = 2048;
constexpr int kCols = 4096;
constexpr int kTokens = insignia::glm53::kQ3KMaxRows;
constexpr size_t kExpertBytes =
    size_t(kRows) * (kCols / insignia::glm53::kQ3KBlockWeights) *
    insignia::glm53::kQ3KBlockBytes;

[[noreturn]] void die(const char *message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

void check(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(2);
    }
}

std::vector<uint8_t> read_slice(const char *path, uint64_t offset, size_t bytes) {
    std::FILE *file = std::fopen(path, "rb");
    if (!file) die("cannot open GGUF shard");
    if (fseeko(file, static_cast<off_t>(offset), SEEK_SET))
        die("cannot seek to GGUF tensor");
    std::vector<uint8_t> result(bytes);
    if (std::fread(result.data(), 1, bytes, file) != bytes)
        die("short GGUF tensor read");
    std::fclose(file);
    return result;
}

template <typename T>
T *device_copy(const std::vector<T> &source) {
    T *device = nullptr;
    check(cudaMalloc(&device, source.size() * sizeof(T)), "cudaMalloc");
    check(cudaMemcpy(device, source.data(), source.size() * sizeof(T),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy H2D");
    return device;
}

template <typename T>
T *device_alloc(size_t count) {
    T *device = nullptr;
    check(cudaMalloc(&device, count * sizeof(T)), "cudaMalloc");
    return device;
}

float fp16_to_float(const uint8_t *pointer) {
    __half value;
    std::memcpy(&value, pointer, sizeof(value));
    return __half2float(value);
}

int q3_scale(const uint8_t *block, int group) {
    const uint8_t *scales = block + 96;
    const int low = (scales[group & 7] >> (4 * (group >> 3))) & 15;
    const int high = (scales[8 + (group & 3)] >> (2 * (group >> 2))) & 3;
    return (low | (high << 4)) - 32;
}

int q3_value(const uint8_t *block, int group, int element) {
    const uint8_t *hmask = block;
    const uint8_t *qs = block + 32;
    const int q_offset = (group >= 8 ? 32 : 0) + ((group & 1) ? 16 : 0);
    const int h_offset = (group & 1) ? 16 : 0;
    const int shift = 2 * ((group >> 1) & 3);
    const int bit = group >> 1;
    const int low = (qs[q_offset + element] >> shift) & 3;
    return low - (((hmask[h_offset + element] >> bit) & 1) ? 0 : 4);
}

std::vector<float> cpu_reference(const std::vector<uint8_t> &weights,
                                 const std::vector<float> &x) {
    std::vector<float> output(kRows);
    constexpr int blocks = kCols / insignia::glm53::kQ3KBlockWeights;
    for (int row = 0; row < kRows; ++row) {
        double sum = 0.0;
        for (int block_id = 0; block_id < blocks; ++block_id) {
            const uint8_t *block = weights.data() +
                (size_t(row) * blocks + block_id) *
                insignia::glm53::kQ3KBlockBytes;
            const double d = fp16_to_float(block + 108);
            for (int group = 0; group < 16; ++group) {
                const double scale = d * q3_scale(block, group);
                for (int element = 0; element < 16; ++element) {
                    const int col = block_id * 256 + group * 16 + element;
                    sum += scale * q3_value(block, group, element) * x[col];
                }
            }
        }
        output[row] = float(sum);
    }
    return output;
}

struct Metrics {
    double mse;
    double relative;
    double cosine;
    double maximum;
};

Metrics compare(const std::vector<float> &actual,
                const std::vector<float> &reference) {
    if (actual.size() != reference.size()) die("metric vector length mismatch");
    double error2 = 0.0, actual2 = 0.0, reference2 = 0.0, dot = 0.0;
    double maximum = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const double a = actual[index];
        const double r = reference[index];
        const double error = a - r;
        error2 += error * error;
        actual2 += a * a;
        reference2 += r * r;
        dot += a * r;
        maximum = std::max(maximum, std::abs(error));
    }
    return {error2 / actual.size(), std::sqrt(error2 / reference2),
            dot / std::sqrt(actual2 * reference2), maximum};
}

void print_metrics(const char *name, const Metrics &metrics) {
    std::printf("%-20s mse %.7g rel %.7g cos %.10f max %.7g\n", name,
                metrics.mse, metrics.relative, metrics.cosine, metrics.maximum);
}

template <typename Launch>
float benchmark_us(Launch launch, int warmup = 200, int iterations = 2000) {
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin), "cudaEventCreate begin");
    check(cudaEventCreate(&end), "cudaEventCreate end");
    for (int i = 0; i < warmup; ++i) launch();
    check(cudaDeviceSynchronize(), "benchmark warmup");
    check(cudaEventRecord(begin), "cudaEventRecord begin");
    for (int i = 0; i < iterations; ++i) launch();
    check(cudaEventRecord(end), "cudaEventRecord end");
    check(cudaEventSynchronize(end), "cudaEventSynchronize");
    float elapsed_ms = 0.0f;
    check(cudaEventElapsedTime(&elapsed_ms, begin, end), "cudaEventElapsedTime");
    check(cudaEventDestroy(begin), "cudaEventDestroy begin");
    check(cudaEventDestroy(end), "cudaEventDestroy end");
    return elapsed_ms * 1000.0f / iterations;
}

std::vector<float> gather(const std::vector<float> &storage,
                          const int *ids, int count) {
    std::vector<float> result(size_t(count) * kRows);
    for (int token = 0; token < count; ++token)
        std::copy_n(storage.data() + size_t(ids[token]) * kRows, kRows,
                    result.data() + size_t(token) * kRows);
    return result;
}

std::vector<float> concatenate(const std::vector<std::vector<float>> &rows,
                               int count) {
    std::vector<float> result(size_t(count) * kRows);
    for (int token = 0; token < count; ++token)
        std::copy(rows[token].begin(), rows[token].end(),
                  result.begin() + size_t(token) * kRows);
    return result;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s SHARD.gguf GATE_OFFSET UP_OFFSET [--bench]\n", argv[0]);
        return 64;
    }
    const char *path = argv[1];
    const uint64_t gate_offset = std::strtoull(argv[2], nullptr, 0);
    const uint64_t up_offset = std::strtoull(argv[3], nullptr, 0);
    const bool run_benchmark = argc > 4 && std::string(argv[4]) == "--bench";
    std::printf("reading two Q3_K experts: %.3f MiB each\n",
                double(kExpertBytes) / (1024.0 * 1024.0));
    const std::vector<uint8_t> gate = read_slice(path, gate_offset, kExpertBytes);
    const std::vector<uint8_t> up = read_slice(path, up_offset, kExpertBytes);

    std::mt19937 rng(0x40703u);
    std::normal_distribution<float> normal(0.0f, 0.19f);
    std::vector<float> activations(size_t(kTokens) * kCols);
    for (int token = 0; token < kTokens; ++token) {
        for (int col = 0; col < kCols; ++col) {
            float value = normal(rng);
            if ((col + token * 137) % 521 == 0) value *= 9.0f;
            activations[size_t(token) * kCols + col] = value;
        }
    }
    std::vector<std::vector<float>> gate_reference(kTokens), up_reference(kTokens);
    for (int token = 0; token < kTokens; ++token) {
        std::vector<float> x(activations.begin() + size_t(token) * kCols,
                             activations.begin() + size_t(token + 1) * kCols);
        gate_reference[token] = cpu_reference(gate, x);
        up_reference[token] = cpu_reference(up, x);
    }

    auto *gate_device = device_copy(gate);
    auto *up_device = device_copy(up);
    auto *x_device = device_copy(activations);
    auto *output_device = device_alloc<float>(size_t(kTokens) * kRows);
    auto *pair_a_device = device_alloc<float>(size_t(kTokens) * kRows);
    auto *pair_b_device = device_alloc<float>(size_t(kTokens) * kRows);
    auto *f32_device = device_alloc<float>(kRows);
    void *workspace = nullptr;
    check(cudaMalloc(&workspace,
                     insignia::glm53::q3k_workspace_rows_bytes(kCols, kTokens)),
          "cudaMalloc Q3 workspace");
    const int row_ids[kTokens] = {7, 0, 5, 1, 6, 2, 4, 3};
    const int out_ids[kTokens] = {3, 7, 1, 6, 0, 5, 2, 4};
    check(insignia::glm53::q3k_gemv_f32(gate_device, x_device, f32_device,
                                         kRows, kCols),
          "q3k_gemv_f32");
    check(cudaDeviceSynchronize(), "f32 oracle synchronize");
    std::vector<float> f32_output(kRows);
    check(cudaMemcpy(f32_output.data(), f32_device, kRows * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "copy f32 oracle");
    print_metrics("FP32 activation", compare(f32_output, gate_reference[0]));

    bool failed = compare(f32_output, gate_reference[0]).relative > 2.0e-5;
    for (int count = 1; count <= kTokens; ++count) {
        check(cudaMemset(output_device, 0x7f,
                         size_t(kTokens) * kRows * sizeof(float)),
              "poison output");
        check(insignia::glm53::q3k_quantize_activation_rows(
                  x_device, kCols, row_ids, count, workspace),
              "q3k_quantize_activation_rows");
        check(insignia::glm53::q3k_gemv_quantized_rows(
                  gate_device, workspace, count, output_device, out_ids,
                  kRows, kCols),
              "q3k_gemv_quantized_rows");
        check(cudaDeviceSynchronize(), "Q3 multi-row synchronize");
        std::vector<float> storage(size_t(kTokens) * kRows);
        check(cudaMemcpy(storage.data(), output_device,
                         storage.size() * sizeof(float), cudaMemcpyDeviceToHost),
              "copy Q3 multi-row output");
        std::vector<std::vector<float>> selected(count);
        for (int r = 0; r < count; ++r) selected[r] = gate_reference[row_ids[r]];
        const Metrics metrics = compare(gather(storage, out_ids, count),
                                        concatenate(selected, count));
        char label[32];
        std::snprintf(label, sizeof(label), "Q8x16 reuse x%d", count);
        print_metrics(label, metrics);
        failed |= metrics.relative > 1.5e-2 || metrics.cosine < 0.99985;
    }

    check(insignia::glm53::q3k_quantize_activation_rows(
              x_device, kCols, row_ids, kTokens, workspace),
          "q3k_quantize_activation_rows pair");
    check(insignia::glm53::q3k_gemv2_quantized_rows(
              gate_device, up_device, workspace, kTokens,
              pair_a_device, pair_b_device, out_ids, kRows, kCols),
          "q3k_gemv2_quantized_rows");
    check(cudaDeviceSynchronize(), "Q3 pair synchronize");
    std::vector<float> pair_a(size_t(kTokens) * kRows),
                       pair_b(size_t(kTokens) * kRows);
    check(cudaMemcpy(pair_a.data(), pair_a_device, pair_a.size() * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy pair A");
    check(cudaMemcpy(pair_b.data(), pair_b_device, pair_b.size() * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy pair B");
    std::vector<std::vector<float>> gate_selected(kTokens), up_selected(kTokens);
    for (int r = 0; r < kTokens; ++r) {
        gate_selected[r] = gate_reference[row_ids[r]];
        up_selected[r] = up_reference[row_ids[r]];
    }
    const Metrics pair_a_metrics = compare(gather(pair_a, out_ids, kTokens),
                                            concatenate(gate_selected, kTokens));
    const Metrics pair_b_metrics = compare(gather(pair_b, out_ids, kTokens),
                                            concatenate(up_selected, kTokens));
    print_metrics("fused pair gate", pair_a_metrics);
    print_metrics("fused pair up", pair_b_metrics);
    failed |= pair_a_metrics.relative > 1.5e-2 || pair_b_metrics.relative > 1.5e-2;

    if (run_benchmark) {
        std::puts("serialized CUDA-event timings (weights resident in VRAM):");
        for (int count : {1, 2, 4, 8}) {
            const auto launch = [&] {
                check(insignia::glm53::q3k_gemv_quantized_rows(
                          gate_device, workspace, count, output_device, out_ids,
                          kRows, kCols), "timed q3k_gemv_quantized_rows");
            };
            const float us = benchmark_us(launch);
            const double gbs = double(kExpertBytes) / (us * 1000.0);
            std::printf("Q3_K x%d %8.3f us %7.1f GB/s %8.1f output rows/us\n",
                        count, us, gbs, double(count) * kRows / us);
        }
        const auto launch_separate_pair = [&] {
            check(insignia::glm53::q3k_gemv_quantized_rows(
                      gate_device, workspace, kTokens, pair_a_device, out_ids,
                      kRows, kCols), "timed separate gate");
            check(insignia::glm53::q3k_gemv_quantized_rows(
                      up_device, workspace, kTokens, pair_b_device, out_ids,
                      kRows, kCols), "timed separate up");
        };
        const auto launch_fused_pair = [&] {
            check(insignia::glm53::q3k_gemv2_quantized_rows(
                      gate_device, up_device, workspace, kTokens,
                      pair_a_device, pair_b_device, out_ids, kRows, kCols),
                  "timed fused pair");
        };
        const float separate_us = benchmark_us(launch_separate_pair);
        const float fused_us = benchmark_us(launch_fused_pair);
        std::printf("Q3_K gate+up x8 separate %8.3f us, fused %8.3f us, %.3fx\n",
                    separate_us, fused_us, separate_us / fused_us);
    } else {
        std::puts("timing skipped; pass --bench only on an uncontended glm-box run");
    }

    cudaFree(gate_device); cudaFree(up_device); cudaFree(x_device);
    cudaFree(output_device); cudaFree(pair_a_device); cudaFree(pair_b_device);
    cudaFree(f32_device); cudaFree(workspace);
    return failed ? 3 : 0;
}
