#include "insignia_bf16.cuh"
#include "insignia_glm53_fp8.cuh"
#include "insignia_glm53_q8.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
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

uint16_t to_f16(float value) {
    const __half converted = __float2half(value);
    uint16_t result;
    std::memcpy(&result, &converted, sizeof(result));
    return result;
}

template <typename T>
T *device_copy(const std::vector<T> &source) {
    T *pointer = nullptr;
    check(cudaMalloc(&pointer, source.size() * sizeof(T)), "cudaMalloc");
    check(cudaMemcpy(pointer, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice),
          "cudaMemcpy H2D");
    return pointer;
}

struct Device {
    uint16_t *bf16;
    uint32_t *q8;
    uint16_t *q8_scales;
    uint8_t *fp8;
    uint16_t *fp8_scales;
    float *x;
    float *bf16_output;
    float *q8_output;
    float *fp8_output;
    void *q8_workspace;
    void *fp8_workspace;
};

void launch_bf16(Device &device, int rows, int cols) {
    insignia::bf16_gemv_v2(reinterpret_cast<const uint32_t *>(device.bf16),
                           device.x, device.bf16_output, rows, cols);
}

void launch_q8(Device &device, int rows, int cols) {
    check(insignia::glm53::q8_gemv(device.q8, device.q8_scales, device.x,
          device.q8_output, rows, cols, device.q8_workspace), "q8_gemv");
}

void launch_fp8(Device &device, int rows, int cols) {
    check(insignia::glm53::fp8_tc_gemv(device.fp8, device.fp8_scales, device.x,
          device.fp8_output, rows, cols, device.fp8_workspace), "fp8_tc_gemv");
}

template <typename Launch>
float benchmark_ms(Launch launch, Device &device, int rows, int cols) {
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin), "cudaEventCreate");
    check(cudaEventCreate(&end), "cudaEventCreate");
    for (int iteration = 0; iteration < 5000; ++iteration) launch(device, rows, cols);
    check(cudaDeviceSynchronize(), "benchmark warmup");
    check(cudaEventRecord(begin), "cudaEventRecord begin");
    for (int iteration = 0; iteration < 1000; ++iteration) launch(device, rows, cols);
    check(cudaEventRecord(end), "cudaEventRecord end");
    check(cudaEventSynchronize(end), "cudaEventSynchronize");
    float elapsed = 0.0f;
    check(cudaEventElapsedTime(&elapsed, begin, end), "cudaEventElapsedTime");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return elapsed / 1000.0f;
}

template <typename Launch>
float benchmark_batch_ms(Launch launch) {
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin), "cudaEventCreate");
    check(cudaEventCreate(&end), "cudaEventCreate");
    for (int iteration = 0; iteration < 1000; ++iteration) launch();
    check(cudaDeviceSynchronize(), "batch benchmark warmup");
    check(cudaEventRecord(begin), "cudaEventRecord batch begin");
    for (int iteration = 0; iteration < 1000; ++iteration) launch();
    check(cudaEventRecord(end), "cudaEventRecord batch end");
    check(cudaEventSynchronize(end), "cudaEventSynchronize batch");
    float elapsed = 0.0f;
    check(cudaEventElapsedTime(&elapsed, begin, end), "cudaEventElapsedTime batch");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return elapsed / 1000.0f;
}

struct Metrics {
    double relative;
    double cosine;
    double maximum;
};

Metrics compare(const std::vector<float> &actual, const std::vector<float> &reference) {
    double error2 = 0.0, actual2 = 0.0, reference2 = 0.0, dot = 0.0, maximum = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const double error = double(actual[index]) - reference[index];
        error2 += error * error;
        actual2 += double(actual[index]) * actual[index];
        reference2 += double(reference[index]) * reference[index];
        dot += double(actual[index]) * reference[index];
        maximum = std::max(maximum, std::abs(error));
    }
    return {std::sqrt(error2 / reference2), dot / std::sqrt(actual2 * reference2), maximum};
}

}  // namespace

int main(int argc, char **argv) {
    constexpr int batch_tokens = 8;
    const int rows = argc > 1 ? std::atoi(argv[1]) : 2048;
    // The 8,192-wide attention output catches activation-quantizer grid bugs
    // that a single 256-thread block at 4,096 columns cannot expose.
    const int cols = argc > 2 ? std::atoi(argv[2]) : 8192;
    if (rows <= 0 || cols <= 0 || cols % insignia::glm53::kQ8GroupSize) return 64;
    const int groups = cols / insignia::glm53::kQ8GroupSize;
    std::mt19937 rng(0x4089a53u);
    std::normal_distribution<float> normal(0.0f, 0.08f);
    std::vector<uint16_t> bf16(size_t(rows) * cols), q8_scales(size_t(rows) * groups),
        fp8_scales(size_t(rows) * groups);
    std::vector<int8_t> q8(size_t(rows) * cols);
    std::vector<uint8_t> fp8(size_t(rows) * cols);
    std::vector<float> x(cols), reference(rows, 0.0f), batch_x(size_t(batch_tokens) * cols);
    for (float &value : x) value = normal(rng);
    for (float &value : batch_x) value = normal(rng);
    for (uint16_t &value : bf16) value = to_bf16(normal(rng));
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < groups; ++group) {
            float maximum = 0.0f;
            for (int element = 0; element < insignia::glm53::kQ8GroupSize; ++element) {
                const size_t index = (size_t(row) * groups + group) *
                                     insignia::glm53::kQ8GroupSize + element;
                maximum = std::max(maximum, std::abs(from_bf16(bf16[index])));
            }
            const float q8_scale = maximum / 127.0f;
            const float fp8_scale = maximum / 448.0f;
            q8_scales[size_t(row) * groups + group] = to_f16(q8_scale);
            fp8_scales[size_t(row) * groups + group] = to_f16(fp8_scale);
            for (int element = 0; element < insignia::glm53::kQ8GroupSize; ++element) {
                const size_t index = (size_t(row) * groups + group) *
                                     insignia::glm53::kQ8GroupSize + element;
                q8[index] = int8_t(std::lrint(from_bf16(bf16[index]) / q8_scale));
                fp8[index] = __nv_fp8_e4m3(from_bf16(bf16[index]) / fp8_scale).__x;
            }
        }
        double sum = 0.0;
        for (int col = 0; col < cols; ++col)
            sum += double(from_bf16(bf16[size_t(row) * cols + col])) * x[col];
        reference[row] = float(sum);
    }

    Device device{device_copy(bf16),
                  reinterpret_cast<uint32_t *>(device_copy(q8)),
                  device_copy(q8_scales), device_copy(fp8), device_copy(fp8_scales),
                  device_copy(x), nullptr, nullptr, nullptr, nullptr, nullptr};
    check(cudaMalloc(&device.bf16_output, rows * sizeof(float)), "cudaMalloc BF16 output");
    check(cudaMalloc(&device.q8_output, rows * sizeof(float)), "cudaMalloc Q8 output");
    check(cudaMalloc(&device.fp8_output, rows * sizeof(float)), "cudaMalloc FP8 output");
    check(cudaMalloc(&device.q8_workspace, insignia::glm53::q8_workspace_bytes(cols)),
          "cudaMalloc Q8 workspace");
    check(cudaMalloc(&device.fp8_workspace, insignia::glm53::fp8_workspace_bytes(cols)),
          "cudaMalloc FP8 workspace");
    float *batch_x_device = device_copy(batch_x);
    float *batch_scalar_device = nullptr;
    float *batch_output_device = nullptr;
    float *batch_pair_a_device = nullptr;
    float *batch_pair_b_device = nullptr;
    void *batch_workspace = nullptr;
    check(cudaMalloc(&batch_scalar_device, size_t(batch_tokens) * rows * sizeof(float)),
          "cudaMalloc scalar batch output");
    check(cudaMalloc(&batch_output_device, size_t(batch_tokens) * rows * sizeof(float)),
          "cudaMalloc batch output");
    check(cudaMalloc(&batch_pair_a_device, size_t(batch_tokens) * rows * sizeof(float)),
          "cudaMalloc batch pair A output");
    check(cudaMalloc(&batch_pair_b_device, size_t(batch_tokens) * rows * sizeof(float)),
          "cudaMalloc batch pair B output");
    check(cudaMalloc(&batch_workspace,
                     insignia::glm53::fp8_batch_workspace_bytes(cols, batch_tokens)),
          "cudaMalloc batch workspace");
    launch_bf16(device, rows, cols);
    launch_q8(device, rows, cols);
    launch_fp8(device, rows, cols);
    const auto launch_fp8_scalar_batch = [&] {
        for (int token = 0; token < batch_tokens; ++token)
            check(insignia::glm53::fp8_tc_gemv(
                      device.fp8, device.fp8_scales, batch_x_device + size_t(token) * cols,
                      batch_scalar_device + size_t(token) * rows, rows, cols,
                      device.fp8_workspace),
                  "fp8_tc_gemv scalar batch");
    };
    const auto launch_fp8_batch = [&] {
        check(insignia::glm53::fp8_tc_gemv_batch(
                  device.fp8, device.fp8_scales, batch_x_device, batch_output_device,
                  batch_tokens, rows, cols, rows, batch_workspace),
              "fp8_tc_gemv_batch");
    };
    const auto launch_fp8_batch_pair = [&] {
        check(insignia::glm53::fp8_tc_gemv2_batch(
                  device.fp8, device.fp8_scales, device.fp8, device.fp8_scales,
                  batch_x_device, batch_pair_a_device, batch_pair_b_device,
                  batch_tokens, rows, cols, rows, batch_workspace),
              "fp8_tc_gemv2_batch");
    };
    launch_fp8_scalar_batch();
    launch_fp8_batch();
    launch_fp8_batch_pair();
    check(cudaDeviceSynchronize(), "GEMV synchronize");
    std::vector<float> bf16_output(rows), q8_output(rows), fp8_output(rows);
    std::vector<float> batch_scalar(size_t(batch_tokens) * rows),
        batch_output(size_t(batch_tokens) * rows),
        batch_pair_a(size_t(batch_tokens) * rows),
        batch_pair_b(size_t(batch_tokens) * rows);
    check(cudaMemcpy(bf16_output.data(), device.bf16_output, rows * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy BF16 output");
    check(cudaMemcpy(q8_output.data(), device.q8_output, rows * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy Q8 output");
    check(cudaMemcpy(fp8_output.data(), device.fp8_output, rows * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy FP8 output");
    check(cudaMemcpy(batch_scalar.data(), batch_scalar_device,
                     batch_scalar.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "copy scalar batch output");
    check(cudaMemcpy(batch_output.data(), batch_output_device,
                     batch_output.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "copy batch output");
    check(cudaMemcpy(batch_pair_a.data(), batch_pair_a_device,
                     batch_pair_a.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "copy batch pair A output");
    check(cudaMemcpy(batch_pair_b.data(), batch_pair_b_device,
                     batch_pair_b.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "copy batch pair B output");
    const Metrics bf16_error = compare(bf16_output, reference);
    const Metrics q8_error = compare(q8_output, reference);
    const Metrics fp8_error = compare(fp8_output, reference);
    const Metrics batch_error = compare(batch_output, batch_scalar);
    const Metrics batch_pair_a_error = compare(batch_pair_a, batch_scalar);
    const Metrics batch_pair_b_error = compare(batch_pair_b, batch_scalar);
    const float bf16_ms = benchmark_ms(launch_bf16, device, rows, cols);
    const float q8_ms = benchmark_ms(launch_q8, device, rows, cols);
    const float fp8_ms = benchmark_ms(launch_fp8, device, rows, cols);
    const float fp8_scalar_batch_ms = benchmark_batch_ms(launch_fp8_scalar_batch);
    const float fp8_batch_ms = benchmark_batch_ms(launch_fp8_batch);
    const float fp8_batch_pair_ms = benchmark_batch_ms(launch_fp8_batch_pair);
    const double bf16_gbs = double(rows) * cols * 2 / (bf16_ms * 1.0e6);
    const double q8_gbs = double(rows) * (cols + groups * 2) / (q8_ms * 1.0e6);
    const double fp8_gbs = double(rows) * (cols + groups * 2) / (fp8_ms * 1.0e6);
    std::printf("BF16 %8.3f us %6.1f GB/s rel %.3g cos %.9f\n",
                bf16_ms * 1000.0f, bf16_gbs, bf16_error.relative, bf16_error.cosine);
    std::printf("Q8-g64 %7.3f us %6.1f GB/s rel %.3g cos %.9f max %.3g, %.2fx faster\n",
                q8_ms * 1000.0f, q8_gbs, q8_error.relative, q8_error.cosine,
                q8_error.maximum, bf16_ms / q8_ms);
    std::printf("FP8 TC  %7.3f us %6.1f GB/s rel %.3g cos %.9f max %.3g, %.2fx faster\n",
                fp8_ms * 1000.0f, fp8_gbs, fp8_error.relative, fp8_error.cosine,
                fp8_error.maximum, bf16_ms / fp8_ms);
    std::printf("FP8 TC x%d scalar %7.3f us, batch %7.3f us, %.2fx faster, "
                "rel %.3g cos %.9f max %.3g\n",
                batch_tokens, fp8_scalar_batch_ms * 1000.0f, fp8_batch_ms * 1000.0f,
                fp8_scalar_batch_ms / fp8_batch_ms, batch_error.relative,
                batch_error.cosine, batch_error.maximum);
    std::printf("FP8 TC x%d pair   %7.3f us, rel A %.3g B %.3g, cos A %.9f B %.9f\n",
                batch_tokens, fp8_batch_pair_ms * 1000.0f,
                batch_pair_a_error.relative, batch_pair_b_error.relative,
                batch_pair_a_error.cosine, batch_pair_b_error.cosine);
    if (bf16_error.relative > 2.0e-5 || q8_error.relative > 2.0e-2 ||
        fp8_error.relative > 5.0e-2 || batch_error.relative > 1.0e-6 ||
        batch_pair_a_error.relative > 1.0e-6 || batch_pair_b_error.relative > 1.0e-6)
        return 3;
    cudaFree(device.bf16); cudaFree(device.q8); cudaFree(device.q8_scales);
    cudaFree(device.fp8); cudaFree(device.fp8_scales); cudaFree(device.x);
    cudaFree(device.bf16_output); cudaFree(device.q8_output); cudaFree(device.fp8_output);
    cudaFree(device.q8_workspace); cudaFree(device.fp8_workspace);
    cudaFree(batch_x_device); cudaFree(batch_scalar_device); cudaFree(batch_output_device);
    cudaFree(batch_pair_a_device); cudaFree(batch_pair_b_device);
    cudaFree(batch_workspace);
    return 0;
}
