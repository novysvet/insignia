#include "insignia_glm53_iq.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kRows = 4096;
constexpr int kCols = 2048;
constexpr int kTokens = insignia::glm53::kIQMaxRows;
constexpr size_t kExpertBytes =
    size_t(kRows) * (kCols / insignia::glm53::kIQBlockWeights) *
    insignia::glm53::kQ6KBlockBytes;

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

std::vector<uint8_t> read_slice(const char *path, uint64_t offset) {
    std::FILE *file = std::fopen(path, "rb");
    if (!file) die("cannot open GGUF shard");
    if (fseeko(file, static_cast<off_t>(offset), SEEK_SET))
        die("cannot seek to Q6_K tensor");
    std::vector<uint8_t> result(kExpertBytes);
    if (std::fread(result.data(), 1, result.size(), file) != result.size())
        die("short Q6_K expert read");
    std::fclose(file);
    return result;
}

template <typename T>
T *device_copy(const std::vector<T> &source) {
    T *device = nullptr;
    check(cudaMalloc(&device, source.size() * sizeof(T)), "cudaMalloc copy");
    check(cudaMemcpy(device, source.data(), source.size() * sizeof(T),
                     cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    return device;
}

template <typename T>
T *device_alloc(size_t count) {
    T *device = nullptr;
    check(cudaMalloc(&device, count * sizeof(T)), "cudaMalloc output");
    return device;
}

std::vector<float> cpu_reference(const std::vector<uint8_t> &weights,
                                 const std::vector<float> &x) {
    constexpr size_t row_bytes =
        size_t(kCols / insignia::glm53::kIQBlockWeights) *
        insignia::glm53::kQ6KBlockBytes;
    std::vector<float> result(size_t(kTokens) * kRows);
    std::vector<float> row(kCols);
    for (int output = 0; output < kRows; ++output) {
        insignia::glm53::q6_k_dequantize_row_cpu(
            weights.data() + size_t(output) * row_bytes, row.data(), kCols);
        for (int token = 0; token < kTokens; ++token) {
            double sum = 0.0;
            for (int col = 0; col < kCols; ++col)
                sum += double(row[col]) * x[size_t(token) * kCols + col];
            result[size_t(token) * kRows + output] = float(sum);
        }
    }
    return result;
}

struct Metrics {
    double mse, relative, cosine, maximum;
};

Metrics compare(const std::vector<float> &actual,
                const std::vector<float> &reference) {
    if (actual.size() != reference.size()) die("metric size mismatch");
    double error2 = 0.0, actual2 = 0.0, reference2 = 0.0, dot = 0.0;
    double maximum = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const double a = actual[index], r = reference[index], error = a - r;
        error2 += error * error;
        actual2 += a * a;
        reference2 += r * r;
        dot += a * r;
        maximum = std::max(maximum, std::abs(error));
    }
    return {error2 / actual.size(), std::sqrt(error2 / reference2),
            dot / std::sqrt(actual2 * reference2), maximum};
}

template <typename Launch>
float benchmark_us(Launch launch, int warmup = 200, int iterations = 2000) {
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin), "create begin event");
    check(cudaEventCreate(&end), "create end event");
    for (int i = 0; i < warmup; ++i) launch();
    check(cudaDeviceSynchronize(), "benchmark warmup");
    check(cudaEventRecord(begin), "record begin");
    for (int i = 0; i < iterations; ++i) launch();
    check(cudaEventRecord(end), "record end");
    check(cudaEventSynchronize(end), "wait end");
    float milliseconds = 0.0f;
    check(cudaEventElapsedTime(&milliseconds, begin, end), "event elapsed");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return milliseconds * 1000.0f / iterations;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s SHARD.gguf Q6_EXPERT_OFFSET [--bench]\n", argv[0]);
        return 64;
    }
    const uint64_t offset = std::strtoull(argv[2], nullptr, 0);
    const bool run_benchmark = argc > 3 && std::string(argv[3]) == "--bench";
    std::vector<uint8_t> weights = read_slice(argv[1], offset);
    std::vector<float> activations(size_t(kTokens) * kCols);
    std::mt19937 rng(0x6a4070u);
    std::normal_distribution<float> normal(0.0f, 0.19f);
    for (size_t index = 0; index < activations.size(); ++index) {
        float value = normal(rng);
        if (index % 521 == 0) value *= 9.0f;
        activations[index] = value;
    }
    const std::vector<float> reference = cpu_reference(weights, activations);
    auto *weights_device = device_copy(weights);
    auto *x_device = device_copy(activations);
    auto *y_device = device_alloc<float>(size_t(kTokens) * kRows);
    void *workspace = nullptr;
    check(cudaMalloc(&workspace,
                     insignia::glm53::iq_workspace_rows_bytes(kCols, kTokens)),
          "cudaMalloc Q6 workspace");
    const int row_ids[kTokens] = {7, 0, 5, 1, 6, 2, 4, 3};
    const int out_ids[kTokens] = {3, 7, 1, 6, 0, 5, 2, 4};
    bool failed = false;
    for (int count = 1; count <= kTokens; ++count) {
        check(insignia::glm53::iq_quantize_activation_rows(
                  x_device, kCols, row_ids, count, workspace),
              "quantize Q6 activations");
        check(insignia::glm53::q6_k_gemv_rows(
                  weights_device, workspace, count, y_device, out_ids,
                  kRows, kCols), "Q6_K GEMV");
        check(cudaDeviceSynchronize(), "Q6 correctness sync");
        std::vector<float> storage(size_t(kTokens) * kRows);
        check(cudaMemcpy(storage.data(), y_device, storage.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), "copy Q6 output");
        std::vector<float> actual(size_t(count) * kRows);
        std::vector<float> expected(size_t(count) * kRows);
        for (int token = 0; token < count; ++token) {
            std::copy_n(storage.data() + size_t(out_ids[token]) * kRows, kRows,
                        actual.data() + size_t(token) * kRows);
            std::copy_n(reference.data() + size_t(row_ids[token]) * kRows, kRows,
                        expected.data() + size_t(token) * kRows);
        }
        const Metrics metrics = compare(actual, expected);
        std::printf("Q6_K down x%d mse %.7g rel %.7g cos %.10f max %.7g\n",
                    count, metrics.mse, metrics.relative,
                    metrics.cosine, metrics.maximum);
        failed |= metrics.relative > 2.0e-2 || metrics.cosine < 0.99980;
    }
    if (run_benchmark) {
        check(insignia::glm53::iq_quantize_activation_rows(
                  x_device, kCols, row_ids, kTokens, workspace),
              "prepare Q6 benchmark");
        for (int count : {1, 2, 4, 8}) {
            const auto launch = [&] {
                check(insignia::glm53::q6_k_gemv_rows(
                          weights_device, workspace, count, y_device,
                          out_ids, kRows, kCols), "timed Q6_K GEMV");
            };
            const float microseconds = benchmark_us(launch);
            std::printf("Q6_K down x%d %8.3f us %7.1f GB/s\n", count,
                        microseconds, double(kExpertBytes) / (microseconds * 1000.0));
        }
    } else {
        std::puts("timing skipped; pass --bench only on an uncontended glm-box run");
    }
    cudaFree(weights_device);
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(workspace);
    return failed ? 3 : 0;
}

