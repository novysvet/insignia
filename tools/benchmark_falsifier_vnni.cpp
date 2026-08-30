// Raptor Lake AVX-VNNI compute ceiling for the 10M Falsifier-MoE controller.
// Build only for glm-box; portability and CPUID fallback are intentionally out
// of scope under the project's exact-hardware contract.

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct Matrix {
    int rows;
    int logical_cols;
    int cols;
    std::vector<int8_t> weight;
    std::vector<int32_t> correction;

    Matrix(int output, int input, std::mt19937 &random)
        : rows(output), logical_cols(input), cols((input + 31) & ~31),
          weight(static_cast<size_t>(output) * cols, 0),
          correction(output) {
        std::uniform_int_distribution<int> distribution(-127, 127);
        for (int row = 0; row < rows; ++row) {
            int32_t sum = 0;
            for (int column = 0; column < logical_cols; ++column) {
                const int8_t value = static_cast<int8_t>(distribution(random));
                weight[static_cast<size_t>(row) * cols + column] = value;
                sum += value;
            }
            correction[row] = 128 * sum;
        }
    }

    uint64_t macs() const { return static_cast<uint64_t>(rows) * cols; }
    uint64_t logical_macs() const {
        return static_cast<uint64_t>(rows) * logical_cols;
    }
};

__attribute__((noinline)) static void vnni_gemv_rows(
    const int8_t *__restrict weight,
    const int32_t *__restrict correction,
    int rows,
    int cols,
    const int8_t *__restrict input,
    int32_t *__restrict output) {
    const __m256i sign_flip = _mm256_set1_epi8(static_cast<char>(0x80));
    alignas(32) int32_t lanes[8];
    for (int row = 0; row < rows; ++row) {
        __m256i accumulator = _mm256_setzero_si256();
        const int8_t *row_weight = weight + static_cast<size_t>(row) * cols;
        for (int column = 0; column < cols; column += 32) {
            const __m256i signed_input = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(input + column));
            const __m256i unsigned_input = _mm256_xor_si256(signed_input, sign_flip);
            const __m256i signed_weight = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(row_weight + column));
            accumulator = _mm256_dpbusd_epi32(accumulator, unsigned_input, signed_weight);
        }
        _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), accumulator);
        int32_t sum = lanes[0] + lanes[1] + lanes[2] + lanes[3]
                    + lanes[4] + lanes[5] + lanes[6] + lanes[7];
        output[row] = sum - correction[row];
    }
}

static void gemv(const Matrix &matrix, const std::vector<int8_t> &input,
                 std::vector<int32_t> &output) {
    vnni_gemv_rows(matrix.weight.data(), matrix.correction.data(), matrix.rows,
                   matrix.cols, input.data(), output.data());
}

static void expert_gemv(const Matrix &pool, int expert, int expert_rows,
                        const std::vector<int8_t> &input,
                        std::vector<int32_t> &output) {
    const size_t row = static_cast<size_t>(expert) * expert_rows;
    vnni_gemv_rows(pool.weight.data() + row * pool.cols,
                   pool.correction.data() + row, expert_rows, pool.cols,
                   input.data(), output.data());
}

struct ControllerWeights {
    // Four modality encoders and the per-candidate two-layer encoder.
    Matrix logit{192, 208, random};
    Matrix hidden{192, 64, random};
    Matrix cache{192, 144, random};
    Matrix router_tail{192, 32, random};
    Matrix candidate_a{64, 32, random};
    Matrix candidate_b{192, 64, random};

    // One weight-tied cell pass.
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

    // 256 independent latent experts: [96 -> 256 -> 96].
    Matrix expert_gate_up{256 * 256, 96, random};
    Matrix expert_down{256 * 96, 128, random};

    // Runtime consumes every controller head.  Combining them does not change
    // the MAC ledger: 5 + 6 + 3 + 2 + 32 + 8 + 24 + 18 + 9 = 107 rows.
    Matrix heads{107, 192, random};

    ControllerWeights() = default;

    uint64_t event_macs() const {
        const uint64_t encoder = logit.macs() + hidden.macs() + cache.macs()
            + router_tail.macs() + 32 * (candidate_a.macs() + candidate_b.macs());
        const uint64_t mla = mla_q.macs() + mla_kv_down.macs() + mla_kv_up.macs()
            + mla_gate.macs() + mla_out.macs();
        const uint64_t moe = moe_router.macs() + latent_down.macs() + latent_up.macs()
            + shared_gate_up.macs() + shared_down.macs()
            + 2 * (static_cast<uint64_t>(256) * 96 + static_cast<uint64_t>(96) * 128);
        uint64_t cells = 0;
        for (uint64_t sources = 1; sources <= 3; ++sources)
            cells += depth_q.macs() + sources * (depth_k.macs() + depth_v.macs())
                + depth_out.macs() + 2 * mhc_dynamic.macs() + mla + moe;
        const uint64_t final_depth = depth_q.macs()
            + 4 * (depth_k.macs() + depth_v.macs()) + depth_out.macs();
        return encoder + cells + final_depth + heads.macs();
    }

    uint64_t event_logical_macs() const {
        const uint64_t encoder = logit.logical_macs() + hidden.logical_macs()
            + cache.logical_macs() + router_tail.logical_macs()
            + 32 * (candidate_a.logical_macs() + candidate_b.logical_macs());
        const uint64_t mla = mla_q.logical_macs() + mla_kv_down.logical_macs()
            + mla_kv_up.logical_macs() + mla_gate.logical_macs()
            + mla_out.logical_macs();
        const uint64_t moe = moe_router.logical_macs() + latent_down.logical_macs()
            + latent_up.logical_macs() + shared_gate_up.logical_macs()
            + shared_down.logical_macs()
            + 2 * (static_cast<uint64_t>(256) * 96
                   + static_cast<uint64_t>(96) * 128);
        uint64_t cells = 0;
        for (uint64_t sources = 1; sources <= 3; ++sources)
            cells += depth_q.logical_macs()
                + sources * (depth_k.logical_macs() + depth_v.logical_macs())
                + depth_out.logical_macs() + 2 * mhc_dynamic.logical_macs()
                + mla + moe;
        const uint64_t final_depth = depth_q.logical_macs()
            + 4 * (depth_k.logical_macs() + depth_v.logical_macs())
            + depth_out.logical_macs();
        return encoder + cells + final_depth + heads.logical_macs();
    }

private:
    // Declared last in source but initialized first by C++ member order is not
    // possible, so use an inline static deterministic generator instead.
    inline static std::mt19937 random{0x53f17a2u};
};

struct Scratch {
    std::vector<int8_t> x32 = std::vector<int8_t>(32);
    std::vector<int8_t> x64 = std::vector<int8_t>(64);
    std::vector<int8_t> x96 = std::vector<int8_t>(96);
    std::vector<int8_t> x128 = std::vector<int8_t>(128);
    std::vector<int8_t> x160 = std::vector<int8_t>(160);
    std::vector<int8_t> x192 = std::vector<int8_t>(192);
    std::vector<int8_t> x224 = std::vector<int8_t>(224);
    std::vector<int8_t> x384 = std::vector<int8_t>(384);
    std::vector<int32_t> output = std::vector<int32_t>(768);

    explicit Scratch(int seed) {
        std::mt19937 random(seed);
        std::uniform_int_distribution<int> distribution(-127, 127);
        for (auto *buffer : {&x32, &x64, &x96, &x128, &x160, &x192, &x224, &x384})
            for (int8_t &value : *buffer) value = static_cast<int8_t>(distribution(random));
    }
};

static int32_t scalar_dot(const int8_t *weight, const int8_t *input, int width) {
    int32_t result = 0;
    for (int i = 0; i < width; ++i) result += int32_t(weight[i]) * int32_t(input[i]);
    return result;
}

static void verify_kernel(const ControllerWeights &weights) {
    Scratch scratch(19);
    std::vector<int32_t> output(weights.logit.rows);
    gemv(weights.logit, scratch.x224, output);
    for (int row = 0; row < weights.logit.rows; ++row) {
        const int32_t expected = scalar_dot(
            weights.logit.weight.data() + static_cast<size_t>(row) * weights.logit.cols,
            scratch.x224.data(), weights.logit.cols);
        if (output[row] != expected)
            throw std::runtime_error("VPDPBUSD correction mismatch at row " + std::to_string(row));
    }
}

static uint64_t run_event(const ControllerWeights &w, Scratch &s, int layer, int pass_seed) {
    uint64_t checksum = 0;
    gemv(w.logit, s.x224, s.output);
    gemv(w.hidden, s.x64, s.output);
    gemv(w.cache, s.x160, s.output);
    gemv(w.router_tail, s.x32, s.output);
    for (int candidate = 0; candidate < 32; ++candidate) {
        gemv(w.candidate_a, s.x32, s.output);
        gemv(w.candidate_b, s.x64, s.output);
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
        gemv(w.depth_q, s.x192, s.output);
        for (int source = 0; source <= repeat; ++source) {
            gemv(w.depth_k, s.x192, s.output);
            gemv(w.depth_v, s.x192, s.output);
        }
        gemv(w.depth_out, s.x128, s.output);
        gemv(w.mhc_dynamic, s.x192, s.output);
        gemv(w.mla_q, s.x192, s.output);
        gemv(w.mla_kv_down, s.x192, s.output);
        gemv(w.mla_kv_up, s.x64, s.output);
        gemv(w.mla_gate, s.x192, s.output);
        gemv(w.mla_out, s.x128, s.output);
        gemv(w.moe_router, s.x192, s.output);
        gemv(w.latent_down, s.x192, s.output);
        gemv(w.latent_up, s.x96, s.output);
        gemv(w.shared_gate_up, s.x192, s.output);
        gemv(w.shared_down, s.x384, s.output);
        const int expert0 = (layer * 17 + repeat * 31 + pass_seed * 13) & 255;
        const int expert1 = (expert0 + 97) & 255;
        expert_gemv(w.expert_gate_up, expert0, 256, s.x96, s.output);
        expert_gemv(w.expert_down, expert0, 96, s.x128, s.output);
        expert_gemv(w.expert_gate_up, expert1, 256, s.x96, s.output);
        expert_gemv(w.expert_down, expert1, 96, s.x128, s.output);
        gemv(w.mhc_dynamic, s.x192, s.output);
        checksum = (checksum << 7) | (checksum >> 57);
        checksum += static_cast<uint32_t>(s.output[(layer + repeat) % 96]);
    }
    gemv(w.depth_q, s.x192, s.output);
    for (int source = 0; source < 4; ++source) {
        gemv(w.depth_k, s.x192, s.output);
        gemv(w.depth_v, s.x192, s.output);
    }
    gemv(w.depth_out, s.x128, s.output);
    gemv(w.heads, s.x192, s.output);
    checksum += static_cast<uint32_t>(s.output[layer % 107]);
    return checksum;
}

static void pin_thread(int worker) {
    cpu_set_t set;
    CPU_ZERO(&set);
    // WSL exposes adjacent SMT siblings for each virtual core. Use one logical
    // CPU per core before consuming siblings.
    CPU_SET((worker * 2) % std::thread::hardware_concurrency(), &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

struct Measurement {
    double milliseconds;
    uint64_t checksum;
};

static Measurement measure(const ControllerWeights &weights, int threads, int iterations) {
    std::barrier start(threads + 1);
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
                for (int layer = 0; layer < 42; ++layer)
                    local += run_event(weights, scratch, layer, iteration + worker * 11);
            checksum.fetch_add(local, std::memory_order_relaxed);
        });
    }
    start.arrive_and_wait();
    const auto begin = std::chrono::steady_clock::now();
    for (std::thread &worker : workers) worker.join();
    const auto end = std::chrono::steady_clock::now();
    return {
        std::chrono::duration<double, std::milli>(end - begin).count(),
        checksum.load(std::memory_order_relaxed),
    };
}

int main(int argc, char **argv) {
    const int threads = argc > 1 ? std::stoi(argv[1]) : 4;
    const int iterations = argc > 2 ? std::stoi(argv[2]) : 20;
    if (threads < 1 || threads > 14 || iterations < 1)
        throw std::runtime_error("usage: benchmark_falsifier_vnni [threads 1..14] [iterations]");
    if (!__builtin_cpu_supports("avxvnni"))
        throw std::runtime_error("Raptor Lake AVX-VNNI is unavailable");
    ControllerWeights weights;
    verify_kernel(weights);
    (void)measure(weights, threads, 1);
    const Measurement measured = measure(weights, threads, iterations);
    const uint64_t macs_per_event = weights.event_macs();
    const uint64_t logical_macs_per_event = weights.event_logical_macs();
    const uint64_t events_per_round = static_cast<uint64_t>(threads) * 42;
    const double round_ms = measured.milliseconds / iterations;
    const double total_macs = double(macs_per_event) * events_per_round * iterations;
    const double gmacs_second = total_macs / (measured.milliseconds * 1.0e6);
    std::cout
        << "{\n"
        << "  \"schema\": \"insignia-falsifier-vnni-ceiling-v1\",\n"
        << "  \"threads\": " << threads << ",\n"
        << "  \"verify_rows\": " << threads << ",\n"
        << "  \"iterations\": " << iterations << ",\n"
        << "  \"logical_macs_per_event\": " << logical_macs_per_event << ",\n"
        << "  \"macs_per_event\": " << macs_per_event << ",\n"
        << "  \"padding_overhead_percent\": "
        << (100.0 * (double(macs_per_event) / logical_macs_per_event - 1.0)) << ",\n"
        << "  \"macs_per_round\": " << macs_per_event * events_per_round << ",\n"
        << "  \"round_ms\": " << round_ms << ",\n"
        << "  \"layer_group_ms\": " << round_ms / 42.0 << ",\n"
        << "  \"gmacs_per_second\": " << gmacs_second << ",\n"
        << "  \"checksum\": " << measured.checksum << ",\n"
        << "  \"kernel_exact\": true\n"
        << "}\n";
    return 0;
}
