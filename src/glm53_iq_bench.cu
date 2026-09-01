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

constexpr int kTokens = insignia::glm53::kIQMaxRows;
constexpr int kPrefillTokens = 32;

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
                     cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    return device;
}

template <typename T>
T *device_alloc(size_t count) {
    T *device = nullptr;
    check(cudaMalloc(&device, count * sizeof(T)), "cudaMalloc");
    return device;
}

using Decoder = void (*)(const uint8_t *, float *, int);

std::vector<float> cpu_reference(const std::vector<uint8_t> &weights,
                                 int rows, int cols, int block_bytes,
                                 Decoder decoder,
                                 const std::vector<float> &activations,
                                 int tokens) {
    const size_t row_bytes = size_t(cols / insignia::glm53::kIQBlockWeights) *
                             block_bytes;
    std::vector<float> output(size_t(tokens) * rows);
    std::vector<float> dequantized(cols);
    for (int row = 0; row < rows; ++row) {
        decoder(weights.data() + size_t(row) * row_bytes,
                dequantized.data(), cols);
        for (int token = 0; token < tokens; ++token) {
            double sum = 0.0;
            const float *x = activations.data() + size_t(token) * cols;
            for (int col = 0; col < cols; ++col)
                sum += double(dequantized[col]) * x[col];
            output[size_t(token) * rows + row] = float(sum);
        }
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
    std::printf("%-22s mse %.7g rel %.7g cos %.10f max %.7g\n", name,
                metrics.mse, metrics.relative, metrics.cosine, metrics.maximum);
}

std::vector<float> gather(const std::vector<float> &storage,
                          const int *ids, int count, int rows) {
    std::vector<float> result(size_t(count) * rows);
    for (int token = 0; token < count; ++token)
        std::copy_n(storage.data() + size_t(ids[token]) * rows, rows,
                    result.data() + size_t(token) * rows);
    return result;
}

std::vector<float> select_reference(const std::vector<float> &reference,
                                    const int *ids, int count, int rows) {
    return gather(reference, ids, count, rows);
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
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return elapsed_ms * 1000.0f / iterations;
}

struct MatrixFixture {
    const char *name;
    int rows;
    int cols;
    int block_bytes;
    size_t expert_bytes;
    std::vector<uint8_t> weights;
    std::vector<float> activations;
    std::vector<float> reference;
    uint8_t *weights_device{};
    float *x_device{};
    float *output_device{};
    void *workspace{};
};

}  // namespace

int main(int argc, char **argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s SHARD.gguf IQ3_GATE_OFFSET IQ3_UP_OFFSET IQ4_DOWN_OFFSET [--bench]\n",
            argv[0]);
        return 64;
    }
    const char *path = argv[1];
    const uint64_t gate_offset = std::strtoull(argv[2], nullptr, 0);
    const uint64_t up_offset = std::strtoull(argv[3], nullptr, 0);
    const uint64_t down_offset = std::strtoull(argv[4], nullptr, 0);
    const bool run_benchmark = argc > 5 && std::string(argv[5]) == "--bench";
    constexpr int gate_rows = 2048, gate_cols = 4096;
    constexpr int down_rows = 4096, down_cols = 2048;
    constexpr size_t iq3_bytes = size_t(gate_rows) * (gate_cols / 256) *
                                 insignia::glm53::kIQ3XXSBlockBytes;
    constexpr size_t iq4_bytes = size_t(down_rows) * (down_cols / 256) *
                                 insignia::glm53::kIQ4XSBlockBytes;
    std::printf("real expert slices: IQ3_XXS %.4f MiB, IQ4_XS %.4f MiB\n",
                double(iq3_bytes) / (1024.0 * 1024.0),
                double(iq4_bytes) / (1024.0 * 1024.0));

    MatrixFixture gate{"IQ3_XXS gate", gate_rows, gate_cols,
                       insignia::glm53::kIQ3XXSBlockBytes, iq3_bytes,
                       read_slice(path, gate_offset, iq3_bytes)};
    const std::vector<uint8_t> up_weights = read_slice(path, up_offset, iq3_bytes);
    std::vector<uint8_t> gate_repacked(iq3_bytes), up_repacked(iq3_bytes);
    std::vector<uint8_t> gate_wim32(iq3_bytes), up_wim32(iq3_bytes);
    insignia::glm53::iq3_xxs_repack_cpu(gate.weights.data(), gate_repacked.data(),
                                         gate_rows, gate_cols);
    insignia::glm53::iq3_xxs_repack_cpu(up_weights.data(), up_repacked.data(),
                                         gate_rows, gate_cols);
    insignia::glm53::iq3_xxs_repack_wim32_cpu(
        gate.weights.data(), gate_wim32.data(), gate_rows, gate_cols);
    insignia::glm53::iq3_xxs_repack_wim32_cpu(
        up_weights.data(), up_wim32.data(), gate_rows, gate_cols);
    MatrixFixture down{"IQ4_XS down", down_rows, down_cols,
                       insignia::glm53::kIQ4XSBlockBytes, iq4_bytes,
                       read_slice(path, down_offset, iq4_bytes)};

    std::mt19937 rng(0x40703u);
    std::normal_distribution<float> normal(0.0f, 0.19f);
    for (MatrixFixture *fixture : {&gate, &down}) {
        fixture->activations.resize(size_t(kTokens) * fixture->cols);
        for (int token = 0; token < kTokens; ++token) {
            for (int col = 0; col < fixture->cols; ++col) {
                float value = normal(rng);
                if ((col + token * 137) % 521 == 0) value *= 9.0f;
                fixture->activations[size_t(token) * fixture->cols + col] = value;
            }
        }
    }
    gate.reference = cpu_reference(gate.weights, gate.rows, gate.cols,
                                   gate.block_bytes,
                                   insignia::glm53::iq3_xxs_dequantize_row_cpu,
                                   gate.activations, kTokens);
    const std::vector<float> up_reference = cpu_reference(
        up_weights, gate.rows, gate.cols, gate.block_bytes,
        insignia::glm53::iq3_xxs_dequantize_row_cpu,
        gate.activations, kTokens);
    down.reference = cpu_reference(down.weights, down.rows, down.cols,
                                   down.block_bytes,
                                   insignia::glm53::iq4_xs_dequantize_row_cpu,
                                   down.activations, kTokens);

    std::vector<float> gate_prefill(size_t(kPrefillTokens) * gate.cols);
    std::vector<float> down_prefill(size_t(kPrefillTokens) * down.cols);
    for (std::vector<float> *values : {&gate_prefill, &down_prefill}) {
        for (size_t index = 0; index < values->size(); ++index) {
            float value = normal(rng);
            if (index % 521 == 0) value *= 9.0f;
            (*values)[index] = value;
        }
    }
    const std::vector<float> gate_prefill_reference = cpu_reference(
        gate.weights, gate.rows, gate.cols, gate.block_bytes,
        insignia::glm53::iq3_xxs_dequantize_row_cpu, gate_prefill,
        kPrefillTokens);
    const std::vector<float> down_prefill_reference = cpu_reference(
        down.weights, down.rows, down.cols, down.block_bytes,
        insignia::glm53::iq4_xs_dequantize_row_cpu, down_prefill,
        kPrefillTokens);

    for (MatrixFixture *fixture : {&gate, &down}) {
        fixture->weights_device = device_copy(fixture->weights);
        fixture->x_device = device_copy(fixture->activations);
        fixture->output_device = device_alloc<float>(size_t(kTokens) * fixture->rows);
        check(cudaMalloc(&fixture->workspace,
                         insignia::glm53::iq_workspace_rows_bytes(
                             fixture->cols, kTokens)),
              "cudaMalloc IQ workspace");
    }
    auto *up_device = device_copy(up_weights);
    auto *gate_repacked_device = device_copy(gate_repacked);
    auto *up_repacked_device = device_copy(up_repacked);
    auto *gate_wim32_device = device_copy(gate_wim32);
    auto *up_wim32_device = device_copy(up_wim32);
    auto *up_output_device = device_alloc<float>(size_t(kTokens) * gate_rows);
    auto *gate_prefill_device = device_copy(gate_prefill);
    auto *down_prefill_device = device_copy(down_prefill);
    auto *gate_prefill_output_device =
        device_alloc<float>(size_t(kPrefillTokens) * gate.rows);
    auto *down_prefill_output_device =
        device_alloc<float>(size_t(kPrefillTokens) * down.rows);
    const int row_ids[kTokens] = {7, 0, 5, 1, 6, 2, 4, 3};
    const int out_ids[kTokens] = {3, 7, 1, 6, 0, 5, 2, 4};
    const int linear_ids[kTokens] = {0, 1, 2, 3, 4, 5, 6, 7};
    bool failed = false;

    for (MatrixFixture *fixture : {&gate, &down}) {
        for (int count = 1; count <= kTokens; ++count) {
            check(cudaMemset(fixture->output_device, 0x7f,
                             size_t(kTokens) * fixture->rows * sizeof(float)),
                  "poison IQ output");
            check(insignia::glm53::iq_quantize_activation_rows(
                      fixture->x_device, fixture->cols, row_ids, count,
                      fixture->workspace), "iq_quantize_activation_rows");
            if (fixture == &gate)
                check(insignia::glm53::iq3_xxs_gemv_rows(
                          fixture->weights_device, fixture->workspace, count,
                          fixture->output_device, out_ids, fixture->rows,
                          fixture->cols), "iq3_xxs_gemv_rows");
            else
                check(insignia::glm53::iq4_xs_gemv_rows(
                          fixture->weights_device, fixture->workspace, count,
                          fixture->output_device, out_ids, fixture->rows,
                          fixture->cols), "iq4_xs_gemv_rows");
            check(cudaDeviceSynchronize(), "IQ correctness synchronize");
            std::vector<float> storage(size_t(kTokens) * fixture->rows);
            check(cudaMemcpy(storage.data(), fixture->output_device,
                             storage.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  "copy IQ output");
            const Metrics metrics = compare(
                gather(storage, out_ids, count, fixture->rows),
                select_reference(fixture->reference, row_ids, count, fixture->rows));
            char label[48];
            std::snprintf(label, sizeof(label), "%s x%d", fixture->name, count);
            print_metrics(label, metrics);
            failed |= metrics.relative > 2.0e-2 || metrics.cosine < 0.99980;
        }
    }

    check(insignia::glm53::iq_quantize_activation_rows(
              gate.x_device, gate.cols, row_ids, kTokens, gate.workspace),
          "prepare repacked IQ3 correctness");
    check(insignia::glm53::iq3_xxs_gemv_repacked_rows(
              gate_repacked_device, gate.workspace, kTokens, gate.output_device,
              out_ids, gate.rows, gate.cols), "iq3_xxs_gemv_repacked_rows");
    check(cudaDeviceSynchronize(), "repacked IQ3 synchronize");
    std::vector<float> repacked_output(size_t(kTokens) * gate.rows);
    check(cudaMemcpy(repacked_output.data(), gate.output_device,
                     repacked_output.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "copy repacked IQ3 output");
    const Metrics repacked_metrics = compare(
        gather(repacked_output, out_ids, kTokens, gate.rows),
        select_reference(gate.reference, row_ids, kTokens, gate.rows));
    print_metrics("IQ3 repacked x8", repacked_metrics);
    failed |= repacked_metrics.relative > 2.0e-2 || repacked_metrics.cosine < 0.99980;

    for (int count : {1, kTokens}) {
        check(insignia::glm53::iq_quantize_activation_rows(
                  gate.x_device, gate.cols, row_ids, count, gate.workspace),
              "prepare paired IQ3 correctness");
        for (int layout = 0; layout < 3; ++layout) {
            check(layout == 2
                      ? insignia::glm53::iq3_xxs_gemv2_wim32_rows(
                            gate_wim32_device, up_wim32_device,
                            gate.workspace, count, gate.output_device,
                            up_output_device, out_ids, gate.rows, gate.cols)
                      : layout == 1
                      ? insignia::glm53::iq3_xxs_gemv2_repacked_rows(
                            gate_repacked_device, up_repacked_device,
                            gate.workspace, count, gate.output_device,
                            up_output_device, out_ids, gate.rows, gate.cols)
                      : insignia::glm53::iq3_xxs_gemv2_rows(
                            gate.weights_device, up_device, gate.workspace,
                            count, gate.output_device, up_output_device,
                            out_ids, gate.rows, gate.cols),
                  "paired IQ3 correctness");
            check(cudaDeviceSynchronize(), "paired IQ3 synchronize");
            std::vector<float> pair_gate(size_t(kTokens) * gate.rows);
            std::vector<float> pair_up(size_t(kTokens) * gate.rows);
            check(cudaMemcpy(pair_gate.data(), gate.output_device,
                             pair_gate.size() * sizeof(float),
                             cudaMemcpyDeviceToHost), "copy paired IQ3 gate");
            check(cudaMemcpy(pair_up.data(), up_output_device,
                             pair_up.size() * sizeof(float),
                             cudaMemcpyDeviceToHost), "copy paired IQ3 up");
            const Metrics pair_gate_metrics = compare(
                gather(pair_gate, out_ids, count, gate.rows),
                select_reference(gate.reference, row_ids, count, gate.rows));
            const Metrics pair_up_metrics = compare(
                gather(pair_up, out_ids, count, gate.rows),
                select_reference(up_reference, row_ids, count, gate.rows));
            char gate_label[64], up_label[64];
            std::snprintf(gate_label, sizeof(gate_label),
                          "IQ3 pair%s gate x%d",
                          layout == 2 ? " wim" : layout == 1 ? " rep" : "", count);
            std::snprintf(up_label, sizeof(up_label),
                          "IQ3 pair%s up x%d",
                          layout == 2 ? " wim" : layout == 1 ? " rep" : "", count);
            print_metrics(gate_label, pair_gate_metrics);
            print_metrics(up_label, pair_up_metrics);
            failed |= pair_gate_metrics.relative > 2.0e-2 ||
                      pair_gate_metrics.cosine < 0.99980 ||
                      pair_up_metrics.relative > 2.0e-2 ||
                      pair_up_metrics.cosine < 0.99980;
        }
    }

    check(insignia::glm53::iq3_xxs_gemm_prefill32(
              gate.weights_device, gate_prefill_device, kPrefillTokens,
              gate_prefill_output_device, gate.rows, gate.cols),
          "iq3_xxs_gemm_prefill32");
    check(insignia::glm53::iq4_xs_gemm_prefill32(
              down.weights_device, down_prefill_device, kPrefillTokens,
              down_prefill_output_device, down.rows, down.cols),
          "iq4_xs_gemm_prefill32");
    check(cudaDeviceSynchronize(), "WMMA32 correctness synchronize");
    std::vector<float> gate_prefill_output(size_t(kPrefillTokens) * gate.rows);
    std::vector<float> down_prefill_output(size_t(kPrefillTokens) * down.rows);
    check(cudaMemcpy(gate_prefill_output.data(), gate_prefill_output_device,
                     gate_prefill_output.size() * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy IQ3 WMMA32 output");
    check(cudaMemcpy(down_prefill_output.data(), down_prefill_output_device,
                     down_prefill_output.size() * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy IQ4 WMMA32 output");
    const Metrics gate_wmma_metrics =
        compare(gate_prefill_output, gate_prefill_reference);
    const Metrics down_wmma_metrics =
        compare(down_prefill_output, down_prefill_reference);
    print_metrics("IQ3 WMMA prefill32", gate_wmma_metrics);
    print_metrics("IQ4 WMMA prefill32", down_wmma_metrics);
    failed |= gate_wmma_metrics.relative > 2.0e-2 ||
              gate_wmma_metrics.cosine < 0.99980 ||
              down_wmma_metrics.relative > 2.0e-2 ||
              down_wmma_metrics.cosine < 0.99980;

    if (run_benchmark) {
        std::puts("serialized CUDA-event timings (weights resident in VRAM):");
        constexpr int kBatchExperts = 8;
        uint8_t *batch_gate[kBatchExperts]{};
        uint8_t *batch_up[kBatchExperts]{};
        uint8_t *batch_down[kBatchExperts]{};
        batch_gate[0] = gate_wim32_device;
        batch_up[0] = up_wim32_device;
        batch_down[0] = down.weights_device;
        for (int expert = 1; expert < kBatchExperts; ++expert) {
            const std::vector<uint8_t> expert_gate = read_slice(
                path, gate_offset + uint64_t(expert) * iq3_bytes, iq3_bytes);
            const std::vector<uint8_t> expert_up = read_slice(
                path, up_offset + uint64_t(expert) * iq3_bytes, iq3_bytes);
            std::vector<uint8_t> expert_gate_wim32(iq3_bytes);
            std::vector<uint8_t> expert_up_wim32(iq3_bytes);
            insignia::glm53::iq3_xxs_repack_wim32_cpu(
                expert_gate.data(), expert_gate_wim32.data(),
                gate_rows, gate_cols);
            insignia::glm53::iq3_xxs_repack_wim32_cpu(
                expert_up.data(), expert_up_wim32.data(),
                gate_rows, gate_cols);
            batch_gate[expert] = device_copy(expert_gate_wim32);
            batch_up[expert] = device_copy(expert_up_wim32);
            batch_down[expert] = device_copy(read_slice(
                path, down_offset + uint64_t(expert) * iq4_bytes, iq4_bytes));
        }
        const std::vector<uint8_t *> batch_gate_host(
            batch_gate, batch_gate + kBatchExperts);
        const std::vector<uint8_t *> batch_up_host(
            batch_up, batch_up + kBatchExperts);
        const std::vector<uint8_t *> batch_down_host(
            batch_down, batch_down + kBatchExperts);
        auto **batch_gate_table = device_copy(batch_gate_host);
        auto **batch_up_table = device_copy(batch_up_host);
        auto **batch_down_table = device_copy(batch_down_host);
        auto *batch_gate_output =
            device_alloc<float>(size_t(kBatchExperts) * gate.rows);
        auto *batch_up_output =
            device_alloc<float>(size_t(kBatchExperts) * gate.rows);
        const std::vector<float> batch_combine_host = {
            0.22f, 0.18f, 0.16f, 0.14f, 0.11f, 0.08f, 0.06f, 0.05f};
        auto *batch_combine = device_copy(batch_combine_host);
        check(insignia::glm53::iq_quantize_activation_rows(
                  gate.x_device, gate.cols, row_ids, kTokens, gate.workspace),
              "prepare gate benchmark");
        check(insignia::glm53::iq_quantize_activation_rows(
                  down.x_device, down.cols, row_ids, kTokens, down.workspace),
              "prepare down benchmark");

        check(insignia::glm53::iq_quantize_activation_rows(
                  gate.x_device, gate.cols, row_ids, 1, gate.workspace),
              "prepare exact top8 hidden Q8");
        for (int expert = 0; expert < kBatchExperts; ++expert) {
            const int output_id = expert;
            check(insignia::glm53::iq3_xxs_gemv2_wim32_rows(
                      batch_gate[expert], batch_up[expert], gate.workspace, 1,
                      gate.output_device, up_output_device, &output_id,
                      gate.rows, gate.cols),
                  "sequential exact top8 gate/up");
        }
        check(cudaDeviceSynchronize(),
              "sequential exact top8 gate/up synchronize");
        std::vector<float> top8_gate_reference(
            size_t(kBatchExperts) * gate.rows);
        std::vector<float> top8_up_reference(
            size_t(kBatchExperts) * gate.rows);
        check(cudaMemcpy(top8_gate_reference.data(), gate.output_device,
                         top8_gate_reference.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "copy sequential exact top8 gate");
        check(cudaMemcpy(top8_up_reference.data(), up_output_device,
                         top8_up_reference.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "copy sequential exact top8 up");
        check(insignia::glm53::iq3_xxs_gemv2_wim32_topk_x1(
                  batch_gate_table, batch_up_table, gate.workspace,
                  kBatchExperts, batch_gate_output, batch_up_output,
                  gate.rows, gate.cols),
              "batched exact top8 gate/up");
        check(cudaDeviceSynchronize(),
              "batched exact top8 gate/up synchronize");
        std::vector<float> top8_gate_output(size_t(kBatchExperts) * gate.rows);
        std::vector<float> top8_up_output(size_t(kBatchExperts) * gate.rows);
        check(cudaMemcpy(top8_gate_output.data(), batch_gate_output,
                         top8_gate_output.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "copy batched exact top8 gate");
        check(cudaMemcpy(top8_up_output.data(), batch_up_output,
                         top8_up_output.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "copy batched exact top8 up");
        const Metrics top8_gate_metrics =
            compare(top8_gate_output, top8_gate_reference);
        const Metrics top8_up_metrics =
            compare(top8_up_output, top8_up_reference);
        print_metrics("IQ3 exact top8 gate", top8_gate_metrics);
        print_metrics("IQ3 exact top8 up", top8_up_metrics);
        failed |= top8_gate_metrics.maximum != 0.0 ||
                  top8_up_metrics.maximum != 0.0;

        check(cudaMemset(down.output_device, 0,
                         size_t(down.rows) * sizeof(float)),
              "clear sequential exact top8 down");
        for (int expert = 0; expert < kBatchExperts; ++expert) {
            check(insignia::glm53::iq4_xs_swiglu_gemv_acc_fused_x1(
                      batch_down[expert], batch_gate_output, batch_up_output,
                      expert, down.output_device, 0,
                      batch_combine_host[expert], down.rows, down.cols),
                  "sequential exact top8 down");
        }
        check(cudaDeviceSynchronize(),
              "sequential exact top8 down synchronize");
        std::vector<float> top8_down_reference(down.rows);
        check(cudaMemcpy(top8_down_reference.data(), down.output_device,
                         top8_down_reference.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "copy sequential exact top8 down");
        check(cudaMemset(down.output_device, 0,
                         size_t(down.rows) * sizeof(float)),
              "clear batched exact top8 down");
        check(insignia::glm53::iq4_xs_swiglu_gemv_acc_topk_x1(
                  batch_down_table, batch_gate_output, batch_up_output,
                  batch_combine, kBatchExperts, down.output_device,
                  down.rows, down.cols),
              "batched exact top8 down");
        check(cudaDeviceSynchronize(),
              "batched exact top8 down synchronize");
        std::vector<float> top8_down_output(down.rows);
        check(cudaMemcpy(top8_down_output.data(), down.output_device,
                         top8_down_output.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "copy batched exact top8 down");
        const Metrics top8_down_metrics =
            compare(top8_down_output, top8_down_reference);
        print_metrics("IQ4 exact top8 down", top8_down_metrics);
        failed |= top8_down_metrics.maximum != 0.0;
        check(cudaMemset(down.output_device, 0,
                         size_t(down.rows) * sizeof(float)),
              "clear double-fused exact top8 down");
        for (int expert = 0; expert < kBatchExperts; ++expert) {
            check(insignia::glm53::iq3_xxs_gemv2_wim32_fused_quant_x1(
                      batch_gate[expert], batch_up[expert], gate.x_device,
                      row_ids[0], gate.output_device, up_output_device, 0,
                      gate.rows, gate.cols, 8),
                  "double-fused exact top8 gate/up");
            check(insignia::glm53::iq4_xs_swiglu_gemv_acc_fused_x1(
                      batch_down[expert], gate.output_device,
                      up_output_device, 0, down.output_device, 0,
                      batch_combine_host[expert], down.rows, down.cols),
                  "double-fused exact top8 down");
        }
        check(cudaDeviceSynchronize(),
              "double-fused exact top8 synchronize");
        std::vector<float> top8_double_fused_output(down.rows);
        check(cudaMemcpy(top8_double_fused_output.data(), down.output_device,
                         top8_double_fused_output.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "copy double-fused exact top8 down");
        const Metrics top8_double_fused_metrics =
            compare(top8_double_fused_output, top8_down_reference);
        print_metrics("IQ4 exact top8 double", top8_double_fused_metrics);
        failed |= top8_double_fused_metrics.maximum != 0.0;
        for (int count : {1, 2, 4, 8}) {
            const auto launch_gate = [&] {
                check(insignia::glm53::iq3_xxs_gemv_rows(
                          gate.weights_device, gate.workspace, count,
                          gate.output_device, out_ids, gate.rows, gate.cols),
                      "timed IQ3 gate");
            };
            const auto launch_down = [&] {
                check(insignia::glm53::iq4_xs_gemv_rows(
                          down.weights_device, down.workspace, count,
                          down.output_device, out_ids, down.rows, down.cols),
                      "timed IQ4 down");
            };
            const auto launch_gate_repacked = [&] {
                check(insignia::glm53::iq3_xxs_gemv_repacked_rows(
                          gate_repacked_device, gate.workspace, count,
                          gate.output_device, out_ids, gate.rows, gate.cols),
                      "timed repacked IQ3 gate");
            };
            const auto launch_gate_wim32 = [&] {
                check(insignia::glm53::iq3_xxs_gemv_wim32_rows(
                          gate_wim32_device, gate.workspace, count,
                          gate.output_device, out_ids, gate.rows, gate.cols),
                      "timed WIM32 IQ3 gate");
            };
            const float gate_us = benchmark_us(launch_gate);
            const float gate_repacked_us = benchmark_us(launch_gate_repacked);
            const float gate_wim32_us = benchmark_us(launch_gate_wim32);
            const float down_us = benchmark_us(launch_down);
            std::printf("IQ3 raw/repacked/WIM32 x%d %8.3f/%8.3f/%8.3f us "
                        "%.3fx/%.3fx | "
                        "IQ4_XS x%d %8.3f us %7.1f GB/s\n",
                        count, gate_us, gate_repacked_us, gate_wim32_us,
                        gate_us / gate_repacked_us, gate_us / gate_wim32_us,
                        count, down_us, double(iq4_bytes) / (down_us * 1000.0));
        }
        const auto launch_gate_up = [&] {
            check(insignia::glm53::iq3_xxs_gemv_rows(
                      gate.weights_device, gate.workspace, kTokens,
                      gate.output_device, out_ids, gate.rows, gate.cols),
                  "timed IQ3 gate");
            check(insignia::glm53::iq3_xxs_gemv_rows(
                      up_device, gate.workspace, kTokens,
                      up_output_device, out_ids, gate.rows, gate.cols),
                  "timed IQ3 up");
        };
        const auto launch_gate_up_repacked = [&] {
            check(insignia::glm53::iq3_xxs_gemv_repacked_rows(
                      gate_repacked_device, gate.workspace, kTokens,
                      gate.output_device, out_ids, gate.rows, gate.cols),
                  "timed repacked IQ3 gate");
            check(insignia::glm53::iq3_xxs_gemv_repacked_rows(
                      up_repacked_device, gate.workspace, kTokens,
                      up_output_device, out_ids, gate.rows, gate.cols),
                      "timed repacked IQ3 up");
        };
        const auto launch_gate_up_pair = [&] {
            check(insignia::glm53::iq3_xxs_gemv2_rows(
                      gate.weights_device, up_device, gate.workspace, kTokens,
                      gate.output_device, up_output_device, out_ids,
                      gate.rows, gate.cols), "timed paired IQ3 gate/up");
        };
        const auto launch_gate_up_pair_repacked = [&] {
            check(insignia::glm53::iq3_xxs_gemv2_repacked_rows(
                      gate_repacked_device, up_repacked_device, gate.workspace,
                      kTokens, gate.output_device, up_output_device, out_ids,
                      gate.rows, gate.cols),
                  "timed paired repacked IQ3 gate/up");
        };
        const auto launch_gate_up_wim32 = [&] {
            check(insignia::glm53::iq3_xxs_gemv_wim32_rows(
                      gate_wim32_device, gate.workspace, kTokens,
                      gate.output_device, out_ids, gate.rows, gate.cols),
                  "timed WIM32 IQ3 gate");
            check(insignia::glm53::iq3_xxs_gemv_wim32_rows(
                      up_wim32_device, gate.workspace, kTokens,
                      up_output_device, out_ids, gate.rows, gate.cols),
                  "timed WIM32 IQ3 up");
        };
        const auto launch_gate_up_pair_wim32 = [&] {
            check(insignia::glm53::iq3_xxs_gemv2_wim32_rows(
                      gate_wim32_device, up_wim32_device, gate.workspace,
                      kTokens, gate.output_device, up_output_device, out_ids,
                      gate.rows, gate.cols), "timed paired WIM32 IQ3 gate/up");
        };
        const float gate_up_us = benchmark_us(launch_gate_up);
        const float gate_up_repacked_us = benchmark_us(launch_gate_up_repacked);
        const float gate_up_pair_us = benchmark_us(launch_gate_up_pair);
        const float gate_up_pair_repacked_us =
            benchmark_us(launch_gate_up_pair_repacked);
        const float gate_up_wim32_us = benchmark_us(launch_gate_up_wim32);
        const float gate_up_pair_wim32_us =
            benchmark_us(launch_gate_up_pair_wim32);
        std::printf("IQ3 gate+up x8 raw/repacked %8.3f/%8.3f us %.3fx\n",
                    gate_up_us, gate_up_repacked_us,
                    gate_up_us / gate_up_repacked_us);
        std::printf("IQ3 gate+up x8 sequential/pair raw %8.3f/%8.3f us %.3fx | "
                    "repacked %8.3f/%8.3f us %.3fx\n",
                    gate_up_us, gate_up_pair_us, gate_up_us / gate_up_pair_us,
                    gate_up_repacked_us, gate_up_pair_repacked_us,
                    gate_up_repacked_us / gate_up_pair_repacked_us);
        std::printf("IQ3 gate+up x8 sequential/pair WIM32 %8.3f/%8.3f us %.3fx "
                    "(vs repacked pair %.3fx)\n",
                    gate_up_wim32_us, gate_up_pair_wim32_us,
                    gate_up_wim32_us / gate_up_pair_wim32_us,
                    gate_up_pair_repacked_us / gate_up_pair_wim32_us);

        check(insignia::glm53::iq_quantize_activation_rows(
                  gate.x_device, gate.cols, row_ids, 1, gate.workspace),
              "prepare x1 paired benchmark");
        const auto launch_gate_up_x1 = [&] {
            check(insignia::glm53::iq3_xxs_gemv_repacked_rows(
                      gate_repacked_device, gate.workspace, 1,
                      gate.output_device, out_ids, gate.rows, gate.cols),
                  "timed x1 repacked IQ3 gate");
            check(insignia::glm53::iq3_xxs_gemv_repacked_rows(
                      up_repacked_device, gate.workspace, 1,
                      up_output_device, out_ids, gate.rows, gate.cols),
                  "timed x1 repacked IQ3 up");
        };
        const auto launch_gate_up_pair_x1 = [&] {
            check(insignia::glm53::iq3_xxs_gemv2_repacked_rows(
                      gate_repacked_device, up_repacked_device, gate.workspace,
                      1, gate.output_device, up_output_device, out_ids,
                      gate.rows, gate.cols), "timed x1 paired IQ3 gate/up");
        };
        const auto launch_gate_up_pair_wim32_x1 = [&] {
            check(insignia::glm53::iq3_xxs_gemv2_wim32_rows(
                      gate_wim32_device, up_wim32_device, gate.workspace,
                      1, gate.output_device, up_output_device, out_ids,
                      gate.rows, gate.cols), "timed x1 paired WIM32 IQ3 gate/up");
        };
        const float gate_up_x1_us = benchmark_us(launch_gate_up_x1);
        const float gate_up_pair_x1_us = benchmark_us(launch_gate_up_pair_x1);
        const float gate_up_pair_wim32_x1_us =
            benchmark_us(launch_gate_up_pair_wim32_x1);
        std::printf("IQ3 gate+up x1 sequential/pair repacked %8.3f/%8.3f us %.3fx\n",
                    gate_up_x1_us, gate_up_pair_x1_us,
                    gate_up_x1_us / gate_up_pair_x1_us);
        std::printf("IQ3 gate+up x1 pair repacked/WIM32 %8.3f/%8.3f us %.3fx\n",
                    gate_up_pair_x1_us, gate_up_pair_wim32_x1_us,
                    gate_up_pair_x1_us / gate_up_pair_wim32_x1_us);

        launch_gate_up_pair_wim32_x1();
        check(cudaDeviceSynchronize(),
              "fused hidden-quant gate/up baseline synchronize");
        std::vector<float> fused_quant_gate_baseline(
            size_t(kTokens) * gate.rows);
        std::vector<float> fused_quant_up_baseline(
            size_t(kTokens) * gate.rows);
        check(cudaMemcpy(fused_quant_gate_baseline.data(), gate.output_device,
                         fused_quant_gate_baseline.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "copy fused hidden-quant gate baseline");
        check(cudaMemcpy(fused_quant_up_baseline.data(), up_output_device,
                         fused_quant_up_baseline.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "copy fused hidden-quant up baseline");
        float fused_hidden_quant_us[4]{};
        int fused_hidden_index = 0;
        for (int rows_per_matrix : {2, 4, 8, 16}) {
            check(insignia::glm53::iq3_xxs_gemv2_wim32_fused_quant_x1(
                      gate_wim32_device, up_wim32_device, gate.x_device,
                      row_ids[0], gate.output_device, up_output_device,
                      out_ids[0], gate.rows, gate.cols, rows_per_matrix),
                  "fused hidden-quant gate/up correctness");
            check(cudaDeviceSynchronize(),
                  "fused hidden-quant gate/up synchronize");
            std::vector<float> fused_quant_gate(size_t(kTokens) * gate.rows);
            std::vector<float> fused_quant_up(size_t(kTokens) * gate.rows);
            check(cudaMemcpy(fused_quant_gate.data(), gate.output_device,
                             fused_quant_gate.size() * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "copy fused hidden-quant gate");
            check(cudaMemcpy(fused_quant_up.data(), up_output_device,
                             fused_quant_up.size() * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "copy fused hidden-quant up");
            const Metrics fused_gate_metrics = compare(
                gather(fused_quant_gate, out_ids, 1, gate.rows),
                gather(fused_quant_gate_baseline, out_ids, 1, gate.rows));
            const Metrics fused_up_metrics = compare(
                gather(fused_quant_up, out_ids, 1, gate.rows),
                gather(fused_quant_up_baseline, out_ids, 1, gate.rows));
            char gate_label[64], up_label[64];
            std::snprintf(gate_label, sizeof(gate_label),
                          "IQ3 fused hiddenQ r%d gate", rows_per_matrix);
            std::snprintf(up_label, sizeof(up_label),
                          "IQ3 fused hiddenQ r%d up", rows_per_matrix);
            print_metrics(gate_label, fused_gate_metrics);
            print_metrics(up_label, fused_up_metrics);
            failed |= fused_gate_metrics.maximum != 0.0 ||
                      fused_up_metrics.maximum != 0.0;
            const auto launch_fused_hidden_quant = [&] {
                check(insignia::glm53::iq3_xxs_gemv2_wim32_fused_quant_x1(
                          gate_wim32_device, up_wim32_device, gate.x_device,
                          row_ids[0], gate.output_device, up_output_device,
                          out_ids[0], gate.rows, gate.cols, rows_per_matrix),
                      "timed fused hidden-quant gate/up");
            };
            fused_hidden_quant_us[fused_hidden_index++] =
                benchmark_us(launch_fused_hidden_quant);
        }
        const auto launch_separate_hidden_quant_pair = [&] {
            check(insignia::glm53::iq_quantize_activation_rows(
                      gate.x_device, gate.cols, row_ids, 1, gate.workspace),
                  "timed separate hidden quantize");
            launch_gate_up_pair_wim32_x1();
        };
        const float separate_hidden_quant_pair_us =
            benchmark_us(launch_separate_hidden_quant_pair);
        std::printf("IQ3 hiddenQ+gate/up x1 separate/fused-r2/r4/r8/r16 "
                    "%8.3f/%8.3f/%8.3f/%8.3f/%8.3f us speedups "
                    "%.3fx/%.3fx/%.3fx/%.3fx\n",
                    separate_hidden_quant_pair_us,
                    fused_hidden_quant_us[0], fused_hidden_quant_us[1],
                    fused_hidden_quant_us[2], fused_hidden_quant_us[3],
                    separate_hidden_quant_pair_us / fused_hidden_quant_us[0],
                    separate_hidden_quant_pair_us / fused_hidden_quant_us[1],
                    separate_hidden_quant_pair_us / fused_hidden_quant_us[2],
                    separate_hidden_quant_pair_us / fused_hidden_quant_us[3]);
        check(insignia::glm53::iq_quantize_activation_rows(
                  gate.x_device, gate.cols, row_ids, kTokens, gate.workspace),
              "restore x8 gate benchmark workspace");

        for (int count : {1, kTokens}) {
            check(insignia::glm53::iq_quantize_activation_rows(
                      gate.x_device, gate.cols, row_ids, count, gate.workspace),
                  "prepare resident expert gate workspace");
            check(insignia::glm53::iq3_xxs_gemv2_repacked_rows(
                      gate_repacked_device, up_repacked_device, gate.workspace,
                      count, gate.output_device, up_output_device, out_ids,
                      gate.rows, gate.cols), "prepare resident expert gate/up");
            check(insignia::glm53::iq_quantize_swiglu_rows(
                      gate.output_device, up_output_device, down.cols,
                      out_ids, count, down.workspace),
                  "prepare resident expert SwiGLU workspace");
            check(cudaDeviceSynchronize(), "prepare resident expert timing");
            const auto launch_hidden_quant = [&] {
                check(insignia::glm53::iq_quantize_activation_rows(
                          gate.x_device, gate.cols, row_ids, count, gate.workspace),
                      "timed resident hidden quantize");
            };
            const auto launch_pair = [&] {
                check(insignia::glm53::iq3_xxs_gemv2_repacked_rows(
                          gate_repacked_device, up_repacked_device, gate.workspace,
                          count, gate.output_device, up_output_device, out_ids,
                          gate.rows, gate.cols), "timed resident pair");
            };
            const auto launch_swiglu_quant = [&] {
                check(insignia::glm53::iq_quantize_swiglu_rows(
                          gate.output_device, up_output_device, down.cols,
                          out_ids, count, down.workspace),
                      "timed resident SwiGLU quantize");
            };
            const auto launch_down = [&] {
                check(insignia::glm53::iq4_xs_gemv_rows(
                          down.weights_device, down.workspace, count,
                          down.output_device, out_ids, down.rows, down.cols),
                      "timed resident IQ4 down");
            };
            const auto launch_pipeline = [&] {
                launch_hidden_quant();
                launch_pair();
                launch_swiglu_quant();
                launch_down();
            };
            const float hidden_quant_us = benchmark_us(launch_hidden_quant);
            const float pair_us = benchmark_us(launch_pair);
            const float swiglu_quant_us = benchmark_us(launch_swiglu_quant);
            const float resident_down_us = benchmark_us(launch_down);
            const float pipeline_us = benchmark_us(launch_pipeline);
            std::printf("IQ3/IQ4 resident expert x%d hiddenQ/pair/swigluQ/down "
                        "%7.3f/%7.3f/%7.3f/%7.3f us sum %7.3f pipe %7.3f\n",
                        count, hidden_quant_us, pair_us, swiglu_quant_us,
                        resident_down_us,
                        hidden_quant_us + pair_us + swiglu_quant_us + resident_down_us,
                        pipeline_us);

            if (count == 1) {
                launch_pipeline();
                check(cudaDeviceSynchronize(),
                      "resident expert baseline synchronize");
                std::vector<float> resident_baseline(
                    size_t(kTokens) * down.rows);
                check(cudaMemcpy(resident_baseline.data(), down.output_device,
                                 resident_baseline.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      "copy resident expert baseline");

                const auto launch_optimized_pipeline = [&] {
                    check(insignia::glm53::iq_quantize_activation_rows(
                              gate.x_device, gate.cols, row_ids, 1,
                              gate.workspace),
                          "timed optimized resident hidden quantize");
                    check(insignia::glm53::iq3_xxs_gemv2_wim32_rows(
                              gate_wim32_device, up_wim32_device,
                              gate.workspace, 1, gate.output_device,
                              up_output_device, out_ids, gate.rows, gate.cols),
                          "timed optimized resident WIM32 pair");
                    check(insignia::glm53::iq4_xs_swiglu_gemv_fused_x1(
                              down.weights_device, gate.output_device,
                              up_output_device, out_ids[0], down.output_device,
                              out_ids[0], down.rows, down.cols, 32),
                          "timed optimized resident fused down");
                };
                const auto launch_double_fused_pipeline = [&] {
                    check(insignia::glm53::
                              iq3_xxs_gemv2_wim32_fused_quant_x1(
                                  gate_wim32_device, up_wim32_device,
                                  gate.x_device, row_ids[0],
                                  gate.output_device, up_output_device,
                                  out_ids[0], gate.rows, gate.cols, 8),
                          "timed double-fused resident gate/up");
                    check(insignia::glm53::iq4_xs_swiglu_gemv_fused_x1(
                              down.weights_device, gate.output_device,
                              up_output_device, out_ids[0], down.output_device,
                              out_ids[0], down.rows, down.cols, 32),
                          "timed double-fused resident down");
                };
                launch_optimized_pipeline();
                check(cudaDeviceSynchronize(),
                      "optimized resident expert synchronize");
                std::vector<float> resident_optimized(
                    size_t(kTokens) * down.rows);
                check(cudaMemcpy(resident_optimized.data(), down.output_device,
                                 resident_optimized.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      "copy optimized resident expert");
                const Metrics resident_metrics = compare(
                    gather(resident_optimized, out_ids, 1, down.rows),
                    gather(resident_baseline, out_ids, 1, down.rows));
                print_metrics("IQ3/IQ4 optimized resident x1",
                              resident_metrics);
                failed |= resident_metrics.maximum != 0.0;

                launch_double_fused_pipeline();
                check(cudaDeviceSynchronize(),
                      "double-fused resident expert synchronize");
                check(cudaMemcpy(resident_optimized.data(),
                                 down.output_device,
                                 resident_optimized.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      "copy double-fused resident expert");
                const Metrics double_fused_metrics = compare(
                    gather(resident_optimized, out_ids, 1, down.rows),
                    gather(resident_baseline, out_ids, 1, down.rows));
                print_metrics("IQ3/IQ4 double-fused x1",
                              double_fused_metrics);
                failed |= double_fused_metrics.maximum != 0.0;

                const float optimized_pipeline_us =
                    benchmark_us(launch_optimized_pipeline);
                const float double_fused_pipeline_us =
                    benchmark_us(launch_double_fused_pipeline);
                std::printf("IQ3/IQ4 resident expert x1 "
                            "old/WIM32/double-fused "
                            "%8.3f/%8.3f/%8.3f us %.3fx/%.3fx\n",
                            pipeline_us, optimized_pipeline_us,
                            double_fused_pipeline_us,
                            pipeline_us / optimized_pipeline_us,
                            pipeline_us / double_fused_pipeline_us);
            }
        }

        for (int expert_count : {1, 2, 4, 8}) {
            const auto launch_shared_quant = [&] {
                check(insignia::glm53::iq_quantize_activation_rows(
                          gate.x_device, gate.cols, row_ids, 1,
                          gate.workspace),
                      "timed shared hidden quantize");
                for (int expert = 0; expert < expert_count; ++expert) {
                    check(insignia::glm53::iq3_xxs_gemv2_wim32_rows(
                              batch_gate[expert], batch_up[expert],
                              gate.workspace, 1, gate.output_device,
                              up_output_device, out_ids, gate.rows,
                              gate.cols),
                          "timed shared-Q gate/up");
                    check(insignia::glm53::iq4_xs_swiglu_gemv_fused_x1(
                              batch_down[expert], gate.output_device,
                              up_output_device, out_ids[0],
                              down.output_device, out_ids[0], down.rows,
                              down.cols, 32),
                          "timed shared-Q down");
                }
            };
            const auto launch_double_fused = [&] {
                for (int expert = 0; expert < expert_count; ++expert) {
                    check(insignia::glm53::
                              iq3_xxs_gemv2_wim32_fused_quant_x1(
                                  batch_gate[expert], batch_up[expert],
                                  gate.x_device, row_ids[0],
                                  gate.output_device, up_output_device,
                                  out_ids[0], gate.rows, gate.cols, 8),
                          "timed double-fused gate/up");
                    check(insignia::glm53::iq4_xs_swiglu_gemv_fused_x1(
                              batch_down[expert], gate.output_device,
                              up_output_device, out_ids[0],
                              down.output_device, out_ids[0], down.rows,
                              down.cols, 32),
                          "timed double-fused down");
                }
            };
            const float shared_quant_us =
                benchmark_us(launch_shared_quant, 40, 400);
            const float double_fused_us =
                benchmark_us(launch_double_fused, 40, 400);
            std::printf("IQ3/IQ4 resident top%d sharedQ/double-fused "
                        "%8.3f/%8.3f us %.3fx\n",
                        expert_count, shared_quant_us, double_fused_us,
                        shared_quant_us / double_fused_us);
        }

        for (int expert_count : {1, 2, 4, 8}) {
            const auto launch_exact_serial = [&] {
                check(cudaMemsetAsync(down.output_device, 0,
                                      size_t(down.rows) * sizeof(float)),
                      "timed clear serial top-k");
                check(insignia::glm53::iq_quantize_activation_rows(
                          gate.x_device, gate.cols, row_ids, 1, gate.workspace),
                      "timed serial top-k hidden quantize");
                for (int expert = 0; expert < expert_count; ++expert) {
                    const int output_id = expert;
                    check(insignia::glm53::iq3_xxs_gemv2_wim32_rows(
                              batch_gate[expert], batch_up[expert],
                              gate.workspace, 1, gate.output_device,
                              up_output_device, &output_id, gate.rows,
                              gate.cols),
                          "timed serial top-k gate/up");
                }
                for (int expert = 0; expert < expert_count; ++expert) {
                    check(insignia::glm53::iq4_xs_swiglu_gemv_acc_fused_x1(
                              batch_down[expert], gate.output_device,
                              up_output_device, expert, down.output_device, 0,
                              batch_combine_host[expert], down.rows, down.cols),
                          "timed serial top-k down");
                }
            };
            const auto launch_exact_batched = [&] {
                check(cudaMemsetAsync(down.output_device, 0,
                                      size_t(down.rows) * sizeof(float)),
                      "timed clear batched top-k");
                check(insignia::glm53::iq_quantize_activation_rows(
                          gate.x_device, gate.cols, row_ids, 1, gate.workspace),
                      "timed batched top-k hidden quantize");
                check(insignia::glm53::iq3_xxs_gemv2_wim32_topk_x1(
                          batch_gate_table, batch_up_table, gate.workspace,
                          expert_count, batch_gate_output, batch_up_output,
                          gate.rows, gate.cols),
                      "timed batched top-k gate/up");
                check(insignia::glm53::iq4_xs_swiglu_gemv_acc_topk_x1(
                          batch_down_table, batch_gate_output, batch_up_output,
                          batch_combine, expert_count, down.output_device,
                          down.rows, down.cols),
                      "timed batched top-k down");
            };
            const auto launch_exact_double_fused = [&] {
                check(cudaMemsetAsync(down.output_device, 0,
                                      size_t(down.rows) * sizeof(float)),
                      "timed clear double-fused top-k");
                for (int expert = 0; expert < expert_count; ++expert) {
                    check(insignia::glm53::
                              iq3_xxs_gemv2_wim32_fused_quant_x1(
                                  batch_gate[expert], batch_up[expert],
                                  gate.x_device, row_ids[0],
                                  gate.output_device, up_output_device, 0,
                                  gate.rows, gate.cols, 8),
                          "timed exact double-fused top-k gate/up");
                    check(insignia::glm53::iq4_xs_swiglu_gemv_acc_fused_x1(
                              batch_down[expert], gate.output_device,
                              up_output_device, 0, down.output_device, 0,
                              batch_combine_host[expert], down.rows, down.cols),
                          "timed exact double-fused top-k down");
                }
            };
            const float exact_serial_us =
                benchmark_us(launch_exact_serial, 40, 400);
            const float exact_batched_us =
                benchmark_us(launch_exact_batched, 40, 400);
            const float exact_double_fused_us =
                benchmark_us(launch_exact_double_fused, 40, 400);
            std::printf("IQ3/IQ4 exact top%d serial/batched/double-fused "
                        "%8.3f/%8.3f/%8.3f us %.3fx/%.3fx\n",
                        expert_count, exact_serial_us, exact_batched_us,
                        exact_double_fused_us,
                        exact_serial_us / exact_batched_us,
                        exact_serial_us / exact_double_fused_us);
        }

        check(insignia::glm53::iq_quantize_activation_rows(
                  gate.x_device, gate.cols, row_ids, 1, gate.workspace),
              "prepare fused SwiGLU/down input");
        check(insignia::glm53::iq3_xxs_gemv2_repacked_rows(
                  gate_repacked_device, up_repacked_device, gate.workspace, 1,
                  gate.output_device, up_output_device, out_ids,
                  gate.rows, gate.cols), "prepare fused SwiGLU/down gate/up");
        check(insignia::glm53::iq_quantize_swiglu_rows(
                  gate.output_device, up_output_device, down.cols,
                  out_ids, 1, down.workspace),
              "prepare fused SwiGLU/down baseline activation");
        check(insignia::glm53::iq4_xs_gemv_rows(
                  down.weights_device, down.workspace, 1,
                  down.output_device, out_ids, down.rows, down.cols),
              "prepare fused SwiGLU/down baseline output");
        check(cudaDeviceSynchronize(), "prepare fused SwiGLU/down baseline");
        std::vector<float> fused_baseline(size_t(kTokens) * down.rows);
        check(cudaMemcpy(fused_baseline.data(), down.output_device,
                         fused_baseline.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "copy fused SwiGLU/down baseline");
        for (int rows_per_cta : {16, 32, 64}) {
            check(insignia::glm53::iq4_xs_swiglu_gemv_fused_x1(
                      down.weights_device, gate.output_device, up_output_device,
                      out_ids[0], down.output_device, out_ids[0],
                      down.rows, down.cols, rows_per_cta),
                  "fused SwiGLU/down correctness");
            check(cudaDeviceSynchronize(), "fused SwiGLU/down synchronize");
            std::vector<float> fused_output(size_t(kTokens) * down.rows);
            check(cudaMemcpy(fused_output.data(), down.output_device,
                             fused_output.size() * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "copy fused SwiGLU/down output");
            const Metrics fused_metrics = compare(
                gather(fused_output, out_ids, 1, down.rows),
                gather(fused_baseline, out_ids, 1, down.rows));
            char label[64];
            std::snprintf(label, sizeof(label),
                          "IQ4 fused SwiGLU r%d", rows_per_cta);
            print_metrics(label, fused_metrics);
            failed |= fused_metrics.maximum != 0.0;
        }
        constexpr float kFusedCombine = 0.37109375f;
        check(cudaMemset(down.output_device, 0,
                         size_t(kTokens) * down.rows * sizeof(float)),
              "clear IQ4 accumulation baseline");
        check(insignia::glm53::iq_quantize_swiglu_rows(
                  gate.output_device, up_output_device, down.cols,
                  out_ids, 1, down.workspace),
              "prepare IQ4 accumulation baseline activation");
        check(insignia::glm53::iq4_xs_gemv_acc_rows(
                  down.weights_device, down.workspace, 1,
                  down.output_device, out_ids, &kFusedCombine,
                  down.rows, down.cols), "IQ4 accumulation baseline");
        check(cudaDeviceSynchronize(), "IQ4 accumulation baseline synchronize");
        std::vector<float> accumulation_baseline(size_t(kTokens) * down.rows);
        check(cudaMemcpy(accumulation_baseline.data(), down.output_device,
                         accumulation_baseline.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "copy IQ4 accumulation baseline");
        check(cudaMemset(down.output_device, 0,
                         size_t(kTokens) * down.rows * sizeof(float)),
              "clear fused IQ4 accumulation");
        check(insignia::glm53::iq4_xs_swiglu_gemv_acc_fused_x1(
                  down.weights_device, gate.output_device, up_output_device,
                  out_ids[0], down.output_device, out_ids[0], kFusedCombine,
                  down.rows, down.cols), "fused IQ4 accumulation");
        check(cudaDeviceSynchronize(), "fused IQ4 accumulation synchronize");
        std::vector<float> accumulation_output(size_t(kTokens) * down.rows);
        check(cudaMemcpy(accumulation_output.data(), down.output_device,
                         accumulation_output.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "copy fused IQ4 accumulation");
        const Metrics accumulation_metrics = compare(
            gather(accumulation_output, out_ids, 1, down.rows),
            gather(accumulation_baseline, out_ids, 1, down.rows));
        print_metrics("IQ4 fused accumulate", accumulation_metrics);
        failed |= accumulation_metrics.maximum != 0.0;
        const auto launch_swiglu_down_x1 = [&] {
            check(insignia::glm53::iq_quantize_swiglu_rows(
                      gate.output_device, up_output_device, down.cols,
                      out_ids, 1, down.workspace), "timed baseline SwiGLU quantize");
            check(insignia::glm53::iq4_xs_gemv_rows(
                      down.weights_device, down.workspace, 1,
                      down.output_device, out_ids, down.rows, down.cols),
                  "timed baseline IQ4 down");
        };
        const float swiglu_down_x1_us = benchmark_us(launch_swiglu_down_x1);
        float fused_swiglu_down_us[3]{};
        int fused_index = 0;
        for (int rows_per_cta : {16, 32, 64}) {
            const auto launch_fused = [&] {
                check(insignia::glm53::iq4_xs_swiglu_gemv_fused_x1(
                          down.weights_device, gate.output_device, up_output_device,
                          out_ids[0], down.output_device, out_ids[0],
                          down.rows, down.cols, rows_per_cta),
                      "timed fused SwiGLU/down");
            };
            fused_swiglu_down_us[fused_index++] = benchmark_us(launch_fused);
        }
        std::printf("IQ4 SwiGLU+down x1 separate/r16/r32/r64 "
                    "%8.3f/%8.3f/%8.3f/%8.3f us speedups %.3fx/%.3fx/%.3fx\n",
                    swiglu_down_x1_us, fused_swiglu_down_us[0],
                    fused_swiglu_down_us[1], fused_swiglu_down_us[2],
                    swiglu_down_x1_us / fused_swiglu_down_us[0],
                    swiglu_down_x1_us / fused_swiglu_down_us[1],
                    swiglu_down_x1_us / fused_swiglu_down_us[2]);

        void *gate_prefill_workspace[4]{};
        void *down_prefill_workspace[4]{};
        for (int batch = 0; batch < 4; ++batch) {
            check(cudaMalloc(&gate_prefill_workspace[batch],
                             insignia::glm53::iq_workspace_rows_bytes(
                                 gate.cols, kTokens)),
                  "cudaMalloc gate prefill workspace");
            check(cudaMalloc(&down_prefill_workspace[batch],
                             insignia::glm53::iq_workspace_rows_bytes(
                                 down.cols, kTokens)),
                  "cudaMalloc down prefill workspace");
            check(insignia::glm53::iq_quantize_activation_rows(
                      gate_prefill_device + size_t(batch * kTokens) * gate.cols,
                      gate.cols, linear_ids, kTokens,
                      gate_prefill_workspace[batch]),
                  "prepare gate prefill Q8");
            check(insignia::glm53::iq_quantize_activation_rows(
                      down_prefill_device + size_t(batch * kTokens) * down.cols,
                      down.cols, linear_ids, kTokens,
                      down_prefill_workspace[batch]),
                  "prepare down prefill Q8");
        }
        check(cudaDeviceSynchronize(), "prepare prefill benchmark");
        const auto launch_gate_q8_compute32 = [&] {
            for (int batch = 0; batch < 4; ++batch)
                check(insignia::glm53::iq3_xxs_gemv_rows(
                          gate.weights_device, gate_prefill_workspace[batch],
                          kTokens,
                          gate_prefill_output_device +
                              size_t(batch * kTokens) * gate.rows,
                          linear_ids, gate.rows, gate.cols),
                      "timed gate Q8 compute32");
        };
        const auto launch_down_q8_compute32 = [&] {
            for (int batch = 0; batch < 4; ++batch)
                check(insignia::glm53::iq4_xs_gemv_rows(
                          down.weights_device, down_prefill_workspace[batch],
                          kTokens,
                          down_prefill_output_device +
                              size_t(batch * kTokens) * down.rows,
                          linear_ids, down.rows, down.cols),
                      "timed down Q8 compute32");
        };
        const auto launch_gate_q8_pipeline32 = [&] {
            for (int batch = 0; batch < 4; ++batch) {
                check(insignia::glm53::iq_quantize_activation_rows(
                          gate_prefill_device +
                              size_t(batch * kTokens) * gate.cols,
                          gate.cols, linear_ids, kTokens,
                          gate_prefill_workspace[batch]),
                      "timed gate Q8 quantize32");
                check(insignia::glm53::iq3_xxs_gemv_rows(
                          gate.weights_device, gate_prefill_workspace[batch],
                          kTokens,
                          gate_prefill_output_device +
                              size_t(batch * kTokens) * gate.rows,
                          linear_ids, gate.rows, gate.cols),
                      "timed gate Q8 pipeline32");
            }
        };
        const auto launch_down_q8_pipeline32 = [&] {
            for (int batch = 0; batch < 4; ++batch) {
                check(insignia::glm53::iq_quantize_activation_rows(
                          down_prefill_device +
                              size_t(batch * kTokens) * down.cols,
                          down.cols, linear_ids, kTokens,
                          down_prefill_workspace[batch]),
                      "timed down Q8 quantize32");
                check(insignia::glm53::iq4_xs_gemv_rows(
                          down.weights_device, down_prefill_workspace[batch],
                          kTokens,
                          down_prefill_output_device +
                              size_t(batch * kTokens) * down.rows,
                          linear_ids, down.rows, down.cols),
                      "timed down Q8 pipeline32");
            }
        };
        const auto launch_gate_wmma32 = [&] {
            check(insignia::glm53::iq3_xxs_gemm_prefill32(
                      gate.weights_device, gate_prefill_device,
                      kPrefillTokens, gate_prefill_output_device,
                      gate.rows, gate.cols), "timed gate WMMA32");
        };
        const auto launch_down_wmma32 = [&] {
            check(insignia::glm53::iq4_xs_gemm_prefill32(
                      down.weights_device, down_prefill_device,
                      kPrefillTokens, down_prefill_output_device,
                      down.rows, down.cols), "timed down WMMA32");
        };
        const float gate_q8_compute32_us = benchmark_us(launch_gate_q8_compute32);
        const float gate_q8_pipeline32_us = benchmark_us(launch_gate_q8_pipeline32);
        const float gate_wmma32_us = benchmark_us(launch_gate_wmma32);
        const float down_q8_compute32_us = benchmark_us(launch_down_q8_compute32);
        const float down_q8_pipeline32_us = benchmark_us(launch_down_q8_pipeline32);
        const float down_wmma32_us = benchmark_us(launch_down_wmma32);
        std::printf("IQ3 prefill32 Q8compute/Q8pipe/WMMA %8.3f/%8.3f/%8.3f us "
                    "%.3fx pipe speedup\n",
                    gate_q8_compute32_us, gate_q8_pipeline32_us, gate_wmma32_us,
                    gate_q8_pipeline32_us / gate_wmma32_us);
        std::printf("IQ4 prefill32 Q8compute/Q8pipe/WMMA %8.3f/%8.3f/%8.3f us "
                    "%.3fx pipe speedup\n",
                    down_q8_compute32_us, down_q8_pipeline32_us, down_wmma32_us,
                    down_q8_pipeline32_us / down_wmma32_us);
        for (int batch = 0; batch < 4; ++batch) {
            cudaFree(gate_prefill_workspace[batch]);
            cudaFree(down_prefill_workspace[batch]);
        }
        for (int expert = 1; expert < kBatchExperts; ++expert) {
            cudaFree(batch_gate[expert]);
            cudaFree(batch_up[expert]);
            cudaFree(batch_down[expert]);
        }
        cudaFree(batch_gate_table);
        cudaFree(batch_up_table);
        cudaFree(batch_down_table);
        cudaFree(batch_gate_output);
        cudaFree(batch_up_output);
        cudaFree(batch_combine);
    } else {
        std::puts("timing skipped; pass --bench only on an uncontended glm-box run");
    }

    for (MatrixFixture *fixture : {&gate, &down}) {
        cudaFree(fixture->weights_device); cudaFree(fixture->x_device);
        cudaFree(fixture->output_device); cudaFree(fixture->workspace);
    }
    cudaFree(up_device); cudaFree(up_output_device);
    cudaFree(gate_repacked_device); cudaFree(up_repacked_device);
    cudaFree(gate_wim32_device); cudaFree(up_wim32_device);
    cudaFree(gate_prefill_device); cudaFree(down_prefill_device);
    cudaFree(gate_prefill_output_device); cudaFree(down_prefill_output_device);
    return failed ? 3 : 0;
}
