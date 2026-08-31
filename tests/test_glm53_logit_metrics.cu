#include "insignia_glm53_logit_metrics.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using insignia::glm53::LogitMetrics;
using insignia::glm53::LogitRowStats;

void check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

LogitMetrics cpu_reference(const std::vector<float> &left,
                           const std::vector<float> &right) {
    if (left.empty() || left.size() != right.size()) throw std::runtime_error("bad CPU input");
    const int count = static_cast<int>(left.size());
    LogitMetrics result{};
    result.left_max = -std::numeric_limits<double>::infinity();
    result.right_max = result.left_max;
    result.left_argmax = result.right_argmax = -1;
    double left_sum = 0.0, right_sum = 0.0;
    double left_square = 0.0, right_square = 0.0, cross = 0.0, difference_square = 0.0;
    for (int index = 0; index < count; ++index) {
        const double a = left[index];
        const double b = right[index];
        if (a > result.left_max) result.left_max = a, result.left_argmax = index;
        if (b > result.right_max) result.right_max = b, result.right_argmax = index;
        left_sum += a;
        right_sum += b;
        left_square = std::fma(a, a, left_square);
        right_square = std::fma(b, b, right_square);
        cross = std::fma(a, b, cross);
        difference_square = std::fma(a - b, a - b, difference_square);
    }
    double left_sum_exp = 0.0, right_sum_exp = 0.0;
    for (int index = 0; index < count; ++index) {
        left_sum_exp += std::exp(static_cast<double>(left[index]) - result.left_max);
        right_sum_exp += std::exp(static_cast<double>(right[index]) - result.right_max);
    }
    result.left_logsumexp = result.left_max + std::log(left_sum_exp);
    result.right_logsumexp = result.right_max + std::log(right_sum_exp);
    constexpr double ln2 = 0.693147180559945309417232121458176568;
    for (int index = 0; index < count; ++index) {
        const double log_p = static_cast<double>(left[index]) - result.left_logsumexp;
        const double log_q = static_cast<double>(right[index]) - result.right_logsumexp;
        const double p = std::exp(log_p);
        const double q = std::exp(log_q);
        const double maximum = std::max(log_p, log_q);
        const double log_mixture = maximum +
            std::log(std::exp(log_p - maximum) + std::exp(log_q - maximum)) - ln2;
        result.kl_left_right += p * (log_p - log_q);
        result.kl_right_left += q * (log_q - log_p);
        result.js += 0.5 *
            (p * (log_p - log_mixture) + q * (log_q - log_mixture));
        result.left_entropy -= p * log_p;
        result.right_entropy -= q * log_q;
    }
    const double inverse_count = 1.0 / count;
    result.left_mean = left_sum * inverse_count;
    result.right_mean = right_sum * inverse_count;
    const double left_centered_square = std::max(
        0.0, left_square - left_sum * left_sum * inverse_count);
    const double right_centered_square = std::max(
        0.0, right_square - right_sum * right_sum * inverse_count);
    result.left_centered_rms = std::sqrt(left_centered_square * inverse_count);
    result.right_centered_rms = std::sqrt(right_centered_square * inverse_count);
    result.mse = difference_square * inverse_count;
    const double mean_difference = result.left_mean - result.right_mean;
    result.centered_mse = std::max(0.0, result.mse - mean_difference * mean_difference);
    const double raw_norm = std::sqrt(left_square * right_square);
    result.raw_cosine = raw_norm > 0.0
        ? cross / raw_norm
        : (left_square == 0.0 && right_square == 0.0 ? 1.0 : 0.0);
    const double centered_norm = std::sqrt(left_centered_square * right_centered_square);
    result.centered_cosine = centered_norm > 0.0
        ? (cross - left_sum * right_sum * inverse_count) / centered_norm
        : (left_centered_square == 0.0 && right_centered_square == 0.0 ? 1.0 : 0.0);
    result.left_top1_probability = std::exp(result.left_max - result.left_logsumexp);
    result.right_top1_probability = std::exp(result.right_max - result.right_logsumexp);
    return result;
}

bool close(double actual, double expected, double absolute = 2.0e-10,
           double relative = 2.0e-10) {
    return std::abs(actual - expected) <=
        absolute + relative * std::max(std::abs(actual), std::abs(expected));
}

void compare(const char *name, const LogitMetrics &gpu, const LogitMetrics &cpu) {
    const std::array<std::pair<const char *, std::pair<double, double>>, 19> values{{
        {"left_max", {gpu.left_max, cpu.left_max}},
        {"right_max", {gpu.right_max, cpu.right_max}},
        {"left_logsumexp", {gpu.left_logsumexp, cpu.left_logsumexp}},
        {"right_logsumexp", {gpu.right_logsumexp, cpu.right_logsumexp}},
        {"left_mean", {gpu.left_mean, cpu.left_mean}},
        {"right_mean", {gpu.right_mean, cpu.right_mean}},
        {"left_centered_rms", {gpu.left_centered_rms, cpu.left_centered_rms}},
        {"right_centered_rms", {gpu.right_centered_rms, cpu.right_centered_rms}},
        {"mse", {gpu.mse, cpu.mse}},
        {"centered_mse", {gpu.centered_mse, cpu.centered_mse}},
        {"raw_cosine", {gpu.raw_cosine, cpu.raw_cosine}},
        {"centered_cosine", {gpu.centered_cosine, cpu.centered_cosine}},
        {"kl_left_right", {gpu.kl_left_right, cpu.kl_left_right}},
        {"kl_right_left", {gpu.kl_right_left, cpu.kl_right_left}},
        {"js", {gpu.js, cpu.js}},
        {"left_entropy", {gpu.left_entropy, cpu.left_entropy}},
        {"right_entropy", {gpu.right_entropy, cpu.right_entropy}},
        {"left_top1_probability", {gpu.left_top1_probability, cpu.left_top1_probability}},
        {"right_top1_probability", {gpu.right_top1_probability, cpu.right_top1_probability}},
    }};
    double maximum_error = 0.0;
    for (const auto &[field, pair] : values) {
        maximum_error = std::max(maximum_error, std::abs(pair.first - pair.second));
        if (!close(pair.first, pair.second))
            throw std::runtime_error(std::string(name) + " " + field + " mismatch: " +
                                     std::to_string(pair.first) + " vs " +
                                     std::to_string(pair.second));
    }
    if (gpu.left_argmax != cpu.left_argmax || gpu.right_argmax != cpu.right_argmax)
        throw std::runtime_error(std::string(name) + " argmax mismatch");
    std::printf("case %-10s JS %.12g KL %.12g max_abs_cpu_delta %.3e\n",
                name, gpu.js, gpu.kl_left_right, maximum_error);
}

std::pair<std::vector<float>, std::vector<float>> random_pair(int count) {
    std::mt19937 random(1701);
    std::normal_distribution<float> normal(0.0f, 2.2f);
    std::normal_distribution<float> delta(0.0f, 0.35f);
    std::vector<float> left(count), right(count);
    for (int index = 0; index < count; ++index) {
        left[index] = normal(random);
        right[index] = left[index] + delta(random);
    }
    return {std::move(left), std::move(right)};
}

std::pair<std::vector<float>, std::vector<float>> heavy_tail_pair(int count) {
    std::mt19937 random(2309);
    std::vector<int> permutation(count);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::shuffle(permutation.begin(), permutation.end(), random);
    std::vector<float> left(count), right(count);
    for (int rank = 0; rank < count; ++rank)
        left[permutation[rank]] = static_cast<float>(-1.12 * std::log(rank + 1.0));
    right = left;
    for (int index = 0; index < 16; ++index) right[permutation[index]] += 0.55f;
    for (int index = 1024; index < 1088; ++index) right[permutation[index]] += 1.1f;
    return {std::move(left), std::move(right)};
}

uint32_t public_hash(uint32_t token) {
    uint32_t value = token + 1;
    value *= UINT32_C(0x9E3779B1);
    value ^= value >> 16;
    value *= UINT32_C(0x85EBCA6B);
    value ^= value >> 13;
    return value;
}

std::pair<std::vector<float>, std::vector<float>> collision_pair(int count) {
    std::array<std::vector<int>, 128> cells;
    for (int token = 32; token < count; ++token) {
        const uint32_t hash = public_hash(static_cast<uint32_t>(token));
        const int bucket = static_cast<int>(hash & 63);
        const int sign = (hash & 64) ? 1 : 0;
        cells[sign * 64 + bucket].push_back(token);
    }
    std::vector<float> left(count, -20.0f), right(count, -20.0f);
    for (int token = 0; token < 32; ++token)
        left[token] = right[token] = 11.0f - 0.03f * token;
    for (const auto &cell : cells)
        for (std::size_t index = 0; index + 1 < cell.size(); index += 2) {
            left[cell[index]] = 9.0f;
            right[cell[index + 1]] = 9.0f;
        }
    return {std::move(left), std::move(right)};
}

template <typename T>
double percentile(std::vector<T> values, double fraction) {
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(fraction * (values.size() - 1));
    return values[index];
}

LogitRowStats cpu_row_stats(const float *logits, int count) {
    LogitRowStats result{};
    result.maximum = -std::numeric_limits<double>::infinity();
    result.argmax = -1;
    for (int token = 0; token < count; ++token)
        if (logits[token] > result.maximum)
            result.maximum = logits[token], result.argmax = token;
    double sum = 0.0;
    for (int token = 0; token < count; ++token)
        sum += std::exp(static_cast<double>(logits[token]) - result.maximum);
    result.logsumexp = result.maximum + std::log(sum);
    result.top1_probability = 1.0 / sum;
    return result;
}

double cpu_js_only(const float *left, const float *right, int count) {
    double left_max = -std::numeric_limits<double>::infinity();
    double right_max = left_max;
    for (int token = 0; token < count; ++token) {
        left_max = std::max(left_max, static_cast<double>(left[token]));
        right_max = std::max(right_max, static_cast<double>(right[token]));
    }
    double left_sum = 0.0, right_sum = 0.0;
    for (int token = 0; token < count; ++token) {
        left_sum += std::exp(static_cast<double>(left[token]) - left_max);
        right_sum += std::exp(static_cast<double>(right[token]) - right_max);
    }
    const double left_log_z = left_max + std::log(left_sum);
    const double right_log_z = right_max + std::log(right_sum);
    constexpr double ln2 = 0.693147180559945309417232121458176568;
    double js = 0.0;
    for (int token = 0; token < count; ++token) {
        const double log_p = static_cast<double>(left[token]) - left_log_z;
        const double log_q = static_cast<double>(right[token]) - right_log_z;
        const double p = std::exp(log_p);
        const double q = std::exp(log_q);
        const double maximum = std::max(log_p, log_q);
        const double log_mixture = maximum +
            std::log(std::exp(log_p - maximum) + std::exp(log_q - maximum)) - ln2;
        js += 0.5 * (p * (log_p - log_mixture) + q * (log_q - log_mixture));
    }
    return std::max(0.0, js);
}

}  // namespace

int main() try {
    constexpr int count = 154880;
    int device = -1;
    check(cudaGetDevice(&device), "get CUDA device");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "get CUDA properties");
    std::printf("device %s sm_%d%d count %d result_bytes %zu workspace_bytes %zu\n",
                properties.name, properties.major, properties.minor, count,
                sizeof(LogitMetrics), insignia::glm53::logit_metrics_workspace_bytes(count));

    float *device_left = nullptr, *device_right = nullptr, *device_retained = nullptr;
    void *workspace = nullptr;
    LogitMetrics *device_result = nullptr;
    constexpr int draft_rows = 7;
    float *device_rows = nullptr;
    void *row_workspace = nullptr;
    LogitRowStats *device_row_stats = nullptr;
    LogitMetrics *host_result_pinned = nullptr;
    LogitRowStats *host_row_stats_pinned = nullptr;
    check(cudaMalloc(&device_left, count * sizeof(float)), "allocate left");
    check(cudaMalloc(&device_right, count * sizeof(float)), "allocate right");
    check(cudaMalloc(&device_retained, count * sizeof(float)), "allocate retained");
    check(cudaMalloc(&workspace, insignia::glm53::logit_metrics_workspace_bytes(count)),
          "allocate workspace");
    check(cudaMalloc(&device_result, sizeof(LogitMetrics)), "allocate result");
    check(cudaMalloc(&device_rows, static_cast<std::size_t>(draft_rows) * count * sizeof(float)),
          "allocate rows");
    check(cudaMalloc(&row_workspace,
                     insignia::glm53::logit_row_stats_workspace_bytes(draft_rows, count)),
          "allocate row workspace");
    check(cudaMalloc(&device_row_stats, draft_rows * sizeof(LogitRowStats)),
          "allocate row stats");
    check(cudaMallocHost(&host_result_pinned, sizeof(LogitMetrics)),
          "allocate pinned metric result");
    check(cudaMallocHost(&host_row_stats_pinned, draft_rows * sizeof(LogitRowStats)),
          "allocate pinned row stats");
    if (insignia::glm53::logit_metrics_workspace_bytes(0) != 0 ||
        insignia::glm53::logit_row_stats_workspace_bytes(0, count) != 0 ||
        insignia::glm53::logit_metrics_async(
            nullptr, device_right, count, workspace, device_result) != cudaErrorInvalidValue ||
        insignia::glm53::logit_row_stats_async(
            device_rows, 0, count, row_workspace, device_row_stats) != cudaErrorInvalidValue)
        throw std::runtime_error("invalid-argument contract mismatch");
    std::puts("case invalid arguments: PASS");

    auto run_case = [&](const char *name, const std::vector<float> &left,
                        const std::vector<float> &right) {
        check(cudaMemcpy(device_left, left.data(), count * sizeof(float), cudaMemcpyHostToDevice),
              "upload left");
        check(cudaMemcpy(device_right, right.data(), count * sizeof(float), cudaMemcpyHostToDevice),
              "upload right");
        check(insignia::glm53::logit_metrics_async(
                  device_left, device_right, count, workspace, device_result),
              "launch metrics");
        LogitMetrics gpu{};
        check(cudaMemcpy(&gpu, device_result, sizeof(gpu), cudaMemcpyDeviceToHost),
              "download metrics");
        compare(name, gpu, cpu_reference(left, right));
    };

    auto random = random_pair(count);
    run_case("random", random.first, random.second);
    auto heavy = heavy_tail_pair(count);
    run_case("heavy", heavy.first, heavy.second);
    std::vector<float> shifted = random.first;
    for (float &value : shifted) value += 37.0f;
    run_case("shift", random.first, shifted);
    auto collision = collision_pair(count);
    run_case("collision", collision.first, collision.second);
    std::vector<float> zero(count, 0.0f);
    run_case("zero-random", zero, random.first);
    run_case("zero-zero", zero, zero);
    std::vector<float> constant_left(count, 5.0f), constant_right(count, -2.0f);
    run_case("constants", constant_left, constant_right);
    run_case("const-random", constant_left, random.first);

    std::vector<float> rows(static_cast<std::size_t>(draft_rows) * count);
    for (int row = 0; row < draft_rows; ++row)
        for (int token = 0; token < count; ++token)
            rows[static_cast<std::size_t>(row) * count + token] =
                random.first[token] + 0.03f * row + 0.01f * std::sin(0.001f * token * (row + 1));
    check(cudaMemcpy(device_rows, rows.data(), rows.size() * sizeof(float), cudaMemcpyHostToDevice),
          "upload rows");
    check(insignia::glm53::logit_row_stats_async(
              device_rows, draft_rows, count, row_workspace, device_row_stats),
          "launch row stats");
    std::array<LogitRowStats, draft_rows> row_stats{};
    check(cudaMemcpy(row_stats.data(), device_row_stats, sizeof(row_stats), cudaMemcpyDeviceToHost),
          "download row stats");
    for (int row = 0; row < draft_rows; ++row) {
        const LogitRowStats expected = cpu_row_stats(rows.data() +
            static_cast<std::size_t>(row) * count, count);
        if (row_stats[row].argmax != expected.argmax ||
            !close(row_stats[row].maximum, expected.maximum) ||
            !close(row_stats[row].logsumexp, expected.logsumexp) ||
            !close(row_stats[row].top1_probability, expected.top1_probability))
            throw std::runtime_error("batched row-stat mismatch at row " + std::to_string(row));
    }
    std::puts("case row_stats 7x max/logsumexp/top1: exact within tolerance");

    check(cudaMemcpy(device_left, random.first.data(), count * sizeof(float), cudaMemcpyHostToDevice),
          "upload timing left");
    check(cudaMemcpy(device_right, random.second.data(), count * sizeof(float), cudaMemcpyHostToDevice),
          "upload timing right");
    cudaStream_t integration_stream{};
    check(cudaStreamCreateWithFlags(&integration_stream, cudaStreamNonBlocking),
          "create integration stream");
    check(cudaMemcpyAsync(device_retained, device_left, count * sizeof(float),
                          cudaMemcpyDeviceToDevice, integration_stream),
          "retain prior on integration stream");
    check(insignia::glm53::logit_metrics_async(
              device_retained, device_right, count, workspace, device_result,
              integration_stream),
          "launch integration metrics");
    check(insignia::glm53::logit_row_stats_async(
              device_rows, draft_rows, count, row_workspace, device_row_stats,
              integration_stream),
          "launch integration row stats");
    check(cudaMemcpyAsync(host_result_pinned, device_result, sizeof(LogitMetrics),
                          cudaMemcpyDeviceToHost, integration_stream),
          "download integration metrics");
    check(cudaMemcpyAsync(host_row_stats_pinned, device_row_stats,
                          draft_rows * sizeof(LogitRowStats), cudaMemcpyDeviceToHost,
                          integration_stream),
          "download integration row stats");
    check(cudaStreamSynchronize(integration_stream), "synchronize integration stream");
    compare("stream-prior", *host_result_pinned, cpu_reference(random.first, random.second));
    for (int row = 0; row < draft_rows; ++row)
        if (host_row_stats_pinned[row].argmax != row_stats[row].argmax ||
            host_row_stats_pinned[row].maximum != row_stats[row].maximum ||
            host_row_stats_pinned[row].logsumexp != row_stats[row].logsumexp ||
            host_row_stats_pinned[row].top1_probability != row_stats[row].top1_probability)
            throw std::runtime_error("integration-stream row-stat mismatch");
    check(cudaMemcpyAsync(device_retained, device_right, count * sizeof(float),
                          cudaMemcpyDeviceToDevice, integration_stream),
          "adopt prior on integration stream");
    check(insignia::glm53::logit_metrics_async(
              device_retained, device_left, count, workspace, device_result,
              integration_stream),
          "launch adopted-prior metrics");
    check(cudaMemcpyAsync(host_result_pinned, device_result, sizeof(LogitMetrics),
                          cudaMemcpyDeviceToHost, integration_stream),
          "download adopted-prior metrics");
    check(cudaStreamSynchronize(integration_stream), "synchronize adopted prior");
    compare("stream-adopt", *host_result_pinned, cpu_reference(random.second, random.first));
    check(cudaStreamDestroy(integration_stream), "destroy integration stream");
    std::puts("case stream ordering + persistent prior adoption: PASS");
    for (int iteration = 0; iteration < 20; ++iteration)
        check(insignia::glm53::logit_metrics_async(
                  device_left, device_right, count, workspace, device_result),
              "warm metrics");
    check(cudaDeviceSynchronize(), "warm synchronization");

    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "create start event");
    check(cudaEventCreate(&stop), "create stop event");
    std::vector<float> metrics_ms, retained_ms, row_stats_ms;
    std::vector<double> decision_path_ms, row_stats_copy_ms;
    for (int iteration = 0; iteration < 101; ++iteration) {
        check(cudaEventRecord(start), "record metrics start");
        check(insignia::glm53::logit_metrics_async(
                  device_left, device_right, count, workspace, device_result),
              "timed metrics");
        check(cudaEventRecord(stop), "record metrics stop");
        check(cudaEventSynchronize(stop), "wait metrics stop");
        float elapsed = 0.0f;
        check(cudaEventElapsedTime(&elapsed, start, stop), "measure metrics");
        metrics_ms.push_back(elapsed);

        check(cudaEventRecord(start), "record retained start");
        check(cudaMemcpyAsync(device_retained, device_left, count * sizeof(float),
                              cudaMemcpyDeviceToDevice),
              "copy retained logits");
        check(insignia::glm53::logit_metrics_async(
                  device_retained, device_right, count, workspace, device_result),
              "timed retained metrics");
        check(cudaEventRecord(stop), "record retained stop");
        check(cudaEventSynchronize(stop), "wait retained stop");
        check(cudaEventElapsedTime(&elapsed, start, stop), "measure retained metrics");
        retained_ms.push_back(elapsed);

        check(cudaEventRecord(start), "record row stats start");
        check(insignia::glm53::logit_row_stats_async(
                  device_rows, draft_rows, count, row_workspace, device_row_stats),
              "timed row stats");
        check(cudaEventRecord(stop), "record row stats stop");
        check(cudaEventSynchronize(stop), "wait row stats stop");
        check(cudaEventElapsedTime(&elapsed, start, stop), "measure row stats");
        row_stats_ms.push_back(elapsed);

        const auto decision_begin = std::chrono::steady_clock::now();
        check(cudaMemcpyAsync(device_retained, device_left, count * sizeof(float),
                              cudaMemcpyDeviceToDevice),
              "decision retained copy");
        check(insignia::glm53::logit_metrics_async(
                  device_retained, device_right, count, workspace, device_result),
              "decision metrics");
        check(cudaMemcpyAsync(host_result_pinned, device_result, sizeof(LogitMetrics),
                              cudaMemcpyDeviceToHost),
              "decision scalar download");
        check(cudaStreamSynchronize(nullptr), "decision synchronization");
        decision_path_ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - decision_begin).count());

        const auto rows_begin = std::chrono::steady_clock::now();
        check(insignia::glm53::logit_row_stats_async(
                  device_rows, draft_rows, count, row_workspace, device_row_stats),
              "decision row stats");
        check(cudaMemcpyAsync(host_row_stats_pinned, device_row_stats,
                              draft_rows * sizeof(LogitRowStats), cudaMemcpyDeviceToHost),
              "row-stat scalar download");
        check(cudaStreamSynchronize(nullptr), "row-stat synchronization");
        row_stats_copy_ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - rows_begin).count());
    }
    std::printf("timing metrics_only median %.6f ms p95 %.6f ms\n",
                percentile(metrics_ms, 0.50), percentile(metrics_ms, 0.95));
    std::printf("timing d2d_619520B_plus_metrics median %.6f ms p95 %.6f ms\n",
                percentile(retained_ms, 0.50), percentile(retained_ms, 0.95));
    std::printf("timing row_stats_7x154880 median %.6f ms p95 %.6f ms\n",
                percentile(row_stats_ms, 0.50), percentile(row_stats_ms, 0.95));
    std::printf("timing d2d_metrics_scalar_d2h exposed median %.6f ms p95 %.6f ms\n",
                percentile(decision_path_ms, 0.50), percentile(decision_path_ms, 0.95));
    std::printf("timing row_stats_scalar_d2h exposed median %.6f ms p95 %.6f ms\n",
                percentile(row_stats_copy_ms, 0.50), percentile(row_stats_copy_ms, 0.95));

    std::vector<float> downloaded_prior(count);
    std::vector<double> current_host_path_ms, current_row_scan_ms;
    double benchmark_sink = 0.0;
    for (int iteration = 0; iteration < 31; ++iteration) {
        const auto host_begin = std::chrono::steady_clock::now();
        check(cudaMemcpy(downloaded_prior.data(), device_left, count * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "current prior-logit download");
        benchmark_sink += cpu_js_only(downloaded_prior.data(), random.second.data(), count);
        current_host_path_ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - host_begin).count());

        const auto scan_begin = std::chrono::steady_clock::now();
        for (int row = 0; row < draft_rows; ++row)
            benchmark_sink += cpu_row_stats(
                rows.data() + static_cast<std::size_t>(row) * count, count).top1_probability;
        current_row_scan_ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - scan_begin).count());
    }
    std::printf("timing current_pageable_d2h_plus_cpu_js median %.6f ms p95 %.6f ms\n",
                percentile(current_host_path_ms, 0.50), percentile(current_host_path_ms, 0.95));
    std::printf("timing current_cpu_row_stats_7x median %.6f ms p95 %.6f ms sink %.6f\n",
                percentile(current_row_scan_ms, 0.50), percentile(current_row_scan_ms, 0.95),
                benchmark_sink);

    check(cudaEventDestroy(start), "destroy start event");
    check(cudaEventDestroy(stop), "destroy stop event");
    check(cudaFreeHost(host_row_stats_pinned), "free pinned row stats");
    check(cudaFreeHost(host_result_pinned), "free pinned metric result");
    check(cudaFree(device_row_stats), "free row stats");
    check(cudaFree(row_workspace), "free row workspace");
    check(cudaFree(device_rows), "free rows");
    check(cudaFree(device_result), "free result");
    check(cudaFree(workspace), "free workspace");
    check(cudaFree(device_retained), "free retained");
    check(cudaFree(device_right), "free right");
    check(cudaFree(device_left), "free left");
    std::puts("logit metrics CUDA test: PASS");
    return 0;
} catch (const std::exception &error) {
    std::fprintf(stderr, "logit metrics CUDA test: FAIL: %s\n", error.what());
    return 1;
}
