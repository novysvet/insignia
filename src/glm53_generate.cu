#include "insignia_bf16.cuh"
#include "insignia_glm53.cuh"
#include "insignia_glm53_dflash2.cuh"
#include "insignia_glm53_fp8.cuh"
#include "insignia_glm53_index.hpp"
#include "insignia_glm53_iq.cuh"
#include "insignia_glm53_logit_metrics.cuh"
#include "insignia_glm53_q8.cuh"
#include "insignia_glm53_q8_index.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <immintrin.h>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using insignia::glm53::AlternateShardPolicy;
using insignia::glm53::ShardedIndex;
using insignia::glm53::TensorLocation;
using insignia::glm53::TensorType;
using insignia::glm53::Q8Index;
using insignia::glm53::Q8TensorLocation;
using insignia::glm53::Cache8Format;
constexpr int kStreams = insignia::glm53::kHyperStreams;
static int kMaxContext() { static const int limit = [] { const char *v = std::getenv("INSIGNIA_GLM53_CONTEXT"); return std::clamp(v ? std::atoi(v) : 8192, 512, 262144); }(); return limit; }
constexpr int kLegacyMlaContext = 256;
// Twenty-one serialized medians on the local sm_89 4070 SUPER.  Down-store
// and fused weighted-accumulate have distinct occupancy optima; sharing one
// table left 1--5% on the floor.  Gate/up pair is a different 2048x4096 shape
// and independently prefers eight warps for every B=1..8.
constexpr std::array<int, 9> kNvfp4DownStoreCtaWarps{0, 4, 4, 8, 4, 4, 4, 4, 4};
constexpr std::array<int, 9> kNvfp4PackedDownStoreCtaWarps{0, 4, 4, 8, 4, 4, 4, 4, 4};
constexpr std::array<int, 9> kNvfp4PackedDownAccCtaWarps{0, 4, 4, 8, 8, 4, 4, 4, 4};
// glm-box (4070 Ti SUPER) serialized medians: 8 warps wins B=1..4/6/7;
// 4 warps wins B=5 by ~18% and B=8 by ~6-8%.  This branch targets that box.
constexpr std::array<int, 9> kNvfp4PackedPairCtaWarps{0, 8, 8, 8, 8, 4, 8, 8, 4};

struct ExpertMask288 {
    std::array<uint64_t, 5> word{};
    inline void add(int expert) {
        word[size_t(expert) >> 6] |= uint64_t(1) << (expert & 63);
    }
    inline void merge(const ExpertMask288 &other) {
#pragma unroll
        for (int index = 0; index < 5; ++index)
            word[size_t(index)] |= other.word[size_t(index)];
    }
};

inline int expert_mask_count(const ExpertMask288 &mask) {
    return int(_mm_popcnt_u64(mask.word[0])) + int(_mm_popcnt_u64(mask.word[1])) +
           int(_mm_popcnt_u64(mask.word[2])) + int(_mm_popcnt_u64(mask.word[3])) +
           int(_mm_popcnt_u64(mask.word[4]));
}

inline int expert_mask_intersection_count(const ExpertMask288 &left,
                                          const ExpertMask288 &right) {
    return int(_mm_popcnt_u64(left.word[0] & right.word[0])) +
           int(_mm_popcnt_u64(left.word[1] & right.word[1])) +
           int(_mm_popcnt_u64(left.word[2] & right.word[2])) +
           int(_mm_popcnt_u64(left.word[3] & right.word[3])) +
           int(_mm_popcnt_u64(left.word[4] & right.word[4]));
}

void check(cudaError_t status, const char *what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

void debug_linear_output(std::string_view name, const float *device, int count) {
    if (!std::getenv("INSIGNIA_GLM53_CHECK_LINEAR")) return;
    std::vector<float> host(count);
    check(cudaMemcpy(host.data(), device, size_t(count) * sizeof(float), cudaMemcpyDeviceToHost),
          "download linear debug output");
    float maximum = 0.0f;
    for (int index = 0; index < count; ++index) {
        if (!std::isfinite(host[index]))
            throw std::runtime_error("non-finite linear output at " + std::string(name) +
                                     "[" + std::to_string(index) + "]");
        maximum = std::max(maximum, std::fabs(host[index]));
    }
    std::fprintf(stderr, "linear %-78.*s max=%g\n", int(name.size()), name.data(), maximum);
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count = 0) { reset(count); }
    ~DeviceBuffer() { if (pointer_) cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    void reset(size_t count) {
        if (pointer_) check(cudaFree(pointer_), "cudaFree");
        pointer_ = nullptr;
        count_ = count;
        if (count) check(cudaMalloc(&pointer_, count * sizeof(T)), "cudaMalloc");
    }
    T *get() { return pointer_; }
    const T *get() const { return pointer_; }
    size_t size() const { return count_; }
    operator T *() { return pointer_; }
    operator const T *() const { return pointer_; }
private:
    T *pointer_ = nullptr;
    size_t count_ = 0;
};

class TensorStager {
public:
    static constexpr size_t kCapacity = 128ull << 20;

    explicit TensorStager(ShardedIndex &model)
        : model_(model), buffered_(std::getenv("INSIGNIA_GLM53_BUFFERED_BF16") != nullptr) {
        check(cudaHostAlloc(&host_, kCapacity, cudaHostAllocDefault), "cudaHostAlloc tensor stage");
        check(cudaMalloc(&device_, kCapacity), "cudaMalloc tensor stage");
    }
    ~TensorStager() {
        if (host_) cudaFreeHost(host_);
        if (device_) cudaFree(device_);
        for (auto &[key, pointer] : resident_)
            if (pointer) cudaFree(const_cast<uint8_t *>(pointer));
    }
    TensorStager(const TensorStager &) = delete;
    TensorStager &operator=(const TensorStager &) = delete;

    // Whole-tensor VRAM residency for models that fit. The FP8 Flash path uses
    // 128 MiB for its repeatedly-read BF16 metadata; the tiny oracle can pin
    // its entire checkpoint with a larger override.
    void set_resident_budget(uint64_t bytes) { resident_budget_ = bytes; }
    uint64_t resident_bytes() const { return resident_used_; }
    bool is_resident(const TensorLocation &tensor) const { return lookup(tensor) != nullptr; }

    uint8_t *load(const TensorLocation &tensor, uint64_t relative_offset = 0, uint64_t bytes = 0) {
        if (!bytes) bytes = tensor.bytes - relative_offset;
        require(relative_offset <= tensor.bytes && bytes <= tensor.bytes - relative_offset,
                "tensor stage range exceeds source tensor");
        require(bytes <= kCapacity, "tensor exceeds 128 MiB streaming slot");
        if (resident_budget_) {
            if (const uint8_t *cached = lookup(tensor))
                return const_cast<uint8_t *>(cached) + relative_offset;
            if (tensor.bytes <= kCapacity && resident_used_ + tensor.bytes <= resident_budget_) {
                uint8_t *persistent = nullptr;
                check(cudaMalloc(&persistent, size_t(tensor.bytes)), "cudaMalloc resident tensor");
                stage_new(tensor, 0, tensor.bytes, persistent);
                resident_[resident_key(tensor)] = persistent;
                resident_used_ += tensor.bytes;
                return persistent + relative_offset;
            }
        }
        stage_new(tensor, relative_offset, bytes, device_);
        return device_;
    }

    uint8_t *load(std::string_view name) { return load(model_.tensor(name)); }
    double io_seconds() const { return io_seconds_; }
    uint64_t io_bytes() const { return io_bytes_; }

private:
    void stage_new(const TensorLocation &tensor, uint64_t relative_offset, uint64_t bytes,
                   uint8_t *destination) {
        const auto read_begin = std::chrono::steady_clock::now();
        // The BF16 working set is larger than this machine's safe page-cache
        // budget. A cyclic 17 GiB scan otherwise misses every token and forces
        // Windows to page underneath WSL. Buffered I/O remains opt-in for
        // one-layer experiments where reuse actually fits.
        if (buffered_)
            model_.read_span(tensor.shard, tensor.offset + relative_offset, bytes, host_);
        else
            model_.read_span_direct(tensor.shard, tensor.offset + relative_offset, bytes, host_);
        io_seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - read_begin).count();
        io_bytes_ += bytes;
        check(cudaMemcpy(destination, host_, size_t(bytes), cudaMemcpyHostToDevice), "stage tensor H2D");
    }

    struct ResidentKey {
        uint16_t shard;
        uint64_t offset;
        bool operator==(const ResidentKey &other) const {
            return shard == other.shard && offset == other.offset;
        }
    };
    static ResidentKey resident_key(const TensorLocation &tensor) {
        return ResidentKey{tensor.shard, tensor.offset};
    }
    const uint8_t *lookup(const TensorLocation &tensor) const {
        const auto found = resident_.find(resident_key(tensor));
        return found == resident_.end() ? nullptr : found->second;
    }

    ShardedIndex &model_;
    uint8_t *host_ = nullptr;
    uint8_t *device_ = nullptr;
    bool buffered_ = false;
    uint64_t resident_budget_ = 0;
    uint64_t resident_used_ = 0;
    struct ResidentHash {
        size_t operator()(const ResidentKey &key) const {
            return size_t(key.shard) * 1099511628211ull ^ key.offset;
        }
    };
    std::unordered_map<ResidentKey, const uint8_t *, ResidentHash> resident_;
    double io_seconds_ = 0.0;
    uint64_t io_bytes_ = 0;
};

template <typename T>
std::vector<T> read_host(ShardedIndex &model, std::string_view name, TensorType expected) {
    const TensorLocation &tensor = model.tensor(name);
    require(tensor.type == expected, "wrong dtype for " + std::string(name));
    require(!(tensor.bytes % sizeof(T)), "unaligned byte count for " + std::string(name));
    std::vector<T> result(size_t(tensor.bytes) / sizeof(T));
    model.read(tensor, result.data());
    return result;
}

std::string layer_stem(int layer) {
    return "model.language_model.layers." + std::to_string(layer) + ".";
}

// The compact index cannot express per-layer dispatch or MLA head counts, so
// the runner scans the upstream config.json for the few keys it needs.  The
// scouted keys appear exactly once, inside text_config, in every GLM-5.3
// checkpoint seen so far.
size_t config_value_at(const std::string &text, const std::string &key) {
    const size_t at = text.find("\"" + key + "\"");
    require(at != std::string::npos, "config.json is missing key " + key);
    const size_t colon = text.find(':', at);
    require(colon != std::string::npos, "config.json key " + key + " has no value");
    return colon + 1;
}

int config_int(const std::string &text, const std::string &key) {
    return std::atoi(text.c_str() + config_value_at(text, key));
}

std::vector<std::string> config_string_array(const std::string &text, const std::string &key) {
    const size_t at = config_value_at(text, key);
    const size_t open = text.find('[', at);
    const size_t close = text.find(']', open);
    require(open != std::string::npos && close != std::string::npos,
            "config.json key " + key + " is not an array");
    std::vector<std::string> items;
    std::istringstream stream(text.substr(open + 1, close - open - 1));
    std::string item;
    while (std::getline(stream, item, ',')) {
        const size_t begin = item.find_first_not_of(" \t\r\n\"");
        const size_t end = item.find_last_not_of(" \t\r\n\"");
        if (begin != std::string::npos) items.push_back(item.substr(begin, end - begin + 1));
    }
    return items;
}

std::string read_config(const char *root) {
    std::ifstream file(std::filesystem::path(root) / "config.json", std::ios::binary);
    require(file.good(), "cannot open config.json under the model root");
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

__device__ __forceinline__ float bf16_value(uint16_t value) {
    return __uint_as_float(uint32_t(value) << 16);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, offset);
    return value;
}

__global__ __launch_bounds__(256) void embed_repeat_kernel(
    const uint16_t *__restrict__ row,
    float *__restrict__ streams,
    int hidden) {
    for (int dimension = threadIdx.x + blockIdx.x * blockDim.x;
         dimension < hidden; dimension += blockDim.x * gridDim.x) {
        const float value = bf16_value(row[dimension]);
#pragma unroll
        for (int stream = 0; stream < kStreams; ++stream)
            streams[stream * hidden + dimension] = value;
    }
}

// RMSNorm of a raw BF16 row (the MTP embed path never widens the row first).
__global__ __launch_bounds__(256) void bf16_rms_kernel(
    const uint16_t *__restrict__ input,
    const uint16_t *__restrict__ weight,
    float *__restrict__ output,
    int cols) {
    float square = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = bf16_value(input[col]);
        square = fmaf(value, value, square);
    }
    square = warp_sum(square);
    __shared__ float partial[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (!lane) partial[warp] = square;
    __syncthreads();
    if (!warp) {
        square = lane < 8 ? partial[lane] : 0.0f;
        square = warp_sum(square);
        if (!lane) partial[0] = rsqrtf(square / cols + 1.0e-5f);
    }
    __syncthreads();
    const float inverse = partial[0];
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        output[col] = bf16_value(input[col]) * inverse * bf16_value(weight[col]);
}

__global__ __launch_bounds__(256) void rms_bf16_kernel(
    const float *__restrict__ input,
    const uint16_t *__restrict__ weight,
    float *__restrict__ output,
    int rows,
    int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float *source = input + size_t(row) * cols;
    float *destination = output + size_t(row) * cols;
    float square = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        square = fmaf(source[col], source[col], square);
    square = warp_sum(square);
    __shared__ float partial[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (!lane) partial[warp] = square;
    __syncthreads();
    if (!warp) {
        square = lane < 8 ? partial[lane] : 0.0f;
        square = warp_sum(square);
        if (!lane) partial[0] = rsqrtf(square / cols + 1.0e-5f);
    }
    __syncthreads();
    const float inverse = partial[0];
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        destination[col] = source[col] * inverse * bf16_value(weight[col]);
}

__global__ __launch_bounds__(256) void kda_gate_kernel(
    float *__restrict__ forget,
    const float *__restrict__ dt_bias,
    const float *__restrict__ a_log,
    float *__restrict__ beta,
    int heads,
    int width) {
    const int head_dim = width / heads;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < width;
         index += blockDim.x * gridDim.x) {
        const int head = index / head_dim;
        const float argument = __expf(a_log[head]) * (forget[index] + dt_bias[index]);
        forget[index] = -5.0f / (1.0f + __expf(-argument));
    }
    for (int head = blockIdx.x * blockDim.x + threadIdx.x; head < heads;
         head += blockDim.x * gridDim.x)
        beta[head] = 1.0f / (1.0f + __expf(-beta[head]));
}

__global__ __launch_bounds__(128) void kda_output_kernel(
    const float *__restrict__ core,
    const float *__restrict__ gate,
    const uint16_t *__restrict__ norm_weight,
    float *__restrict__ output) {
    const int head = blockIdx.x;
    const int element = threadIdx.x;
    const int head_dim = blockDim.x;
    const int warps = head_dim >> 5;
    const int offset = head * head_dim;
    const float value = core[offset + element];
    float square = warp_sum(value * value);
    __shared__ float partial[8];
    const int lane = element & 31;
    const int warp = element >> 5;
    if (!lane) partial[warp] = square;
    __syncthreads();
    if (!warp) {
        square = lane < warps ? partial[lane] : 0.0f;
        square = warp_sum(square);
        if (!lane) partial[0] = rsqrtf(square * (1.0f / float(head_dim)) + 1.0e-5f);
    }
    __syncthreads();
    const float gate_value = gate[offset + element];
    output[offset + element] = value * partial[0] * bf16_value(norm_weight[element]) /
                               (1.0f + __expf(-gate_value));
}

__global__ __launch_bounds__(256) void clamped_swiglu_kernel(
    const float *__restrict__ gate,
    const float *__restrict__ up,
    float *__restrict__ output,
    int count) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * gridDim.x) {
        const float g = fminf(gate[index], 10.0f);
        const float u = fminf(fmaxf(up[index], -10.0f), 10.0f);
        output[index] = (g / (1.0f + __expf(-g))) * u;
    }
}

__global__ __launch_bounds__(256) void scale_add_kernel(
    float *__restrict__ destination,
    const float *__restrict__ source,
    float scale,
    int count) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * gridDim.x)
        destination[index] = fmaf(source[index], scale, destination[index]);
}

__global__ __launch_bounds__(256) void add_kernel(
    float *__restrict__ destination,
    const float *__restrict__ source,
    int count) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * gridDim.x)
        destination[index] += source[index];
}

__global__ __launch_bounds__(256) void average_streams_kernel(
    const float *__restrict__ streams,
    float *__restrict__ output,
    int hidden) {
    for (int dimension = blockIdx.x * blockDim.x + threadIdx.x; dimension < hidden;
         dimension += blockDim.x * gridDim.x)
        output[dimension] = 0.25f * (streams[dimension] + streams[hidden + dimension] +
                                    streams[2 * hidden + dimension] + streams[3 * hidden + dimension]);
}

// DFlash2 drafter consumes the mean-contracted completed outputs of these
// target layers (SGLang capture points 6/15/25/34/43).
constexpr int kDfCaptureLayers[5] = {5, 14, 24, 33, 42};

__global__ __launch_bounds__(256) void finite_kernel(
    const float *__restrict__ values,
    int count,
    int *__restrict__ valid) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * gridDim.x)
        if (!isfinite(values[index])) atomicExch(valid, 0);
}

// One block per logits row; monotonic float->uint32 mapping makes the running
// maximum a plain integer max so the warp reduce needs no branch.
__global__ __launch_bounds__(1024) void rows_argmax_kernel(
    const float *__restrict__ logits,
    int *__restrict__ out,
    int rows,
    int cols) {
    const int row = blockIdx.y;
    const float *line = logits + size_t(row) * cols;
    unsigned best = 0u;
    int best_index = 0;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols;
         col += blockDim.x * gridDim.x) {
        const unsigned bits = __float_as_uint(line[col]);
        const unsigned key = bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
        if (key > best) {
            best = key;
            best_index = col;
        }
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        const unsigned other_key = __shfl_xor_sync(0xffffffff, best, offset);
        const int other_index = __shfl_xor_sync(0xffffffff, best_index, offset);
        if (other_key > best) {
            best = other_key;
            best_index = other_index;
        }
    }
    __shared__ unsigned keys[32];
    __shared__ int indexes[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (!lane) {
        keys[warp] = best;
        indexes[warp] = best_index;
    }
    __syncthreads();
    if (!warp && !lane) {
        unsigned winner = keys[0];
        int winner_index = indexes[0];
        for (int scan = 1; scan < blockDim.x >> 5; ++scan)
            if (keys[scan] > winner) {
                winner = keys[scan];
                winner_index = indexes[scan];
            }
        out[row] = winner_index;
    }
}

void launch_rms(const float *input, const uint16_t *weight, float *output, int rows, int cols) {
    rms_bf16_kernel<<<rows, 256>>>(input, weight, output, rows, cols);
    check(cudaGetLastError(), "RMSNorm launch");
}

void launch_clamped_swiglu(const float *gate, const float *up, float *output, int count) {
    clamped_swiglu_kernel<<<std::min((count + 255) / 256, 256), 256>>>(gate, up, output, count);
    check(cudaGetLastError(), "clamped SwiGLU launch");
}

struct ExpertLocations {
    std::array<const TensorLocation *, 3> body;
    std::array<const TensorLocation *, 3> scales;
    std::array<const TensorLocation *, 3> globals;
};

ExpertLocations locate_expert(ShardedIndex &model, int layer, int expert) {
    const std::string stem = layer_stem(layer) + "mlp.experts." + std::to_string(expert) + ".";
    return {{&model.tensor(stem + "down_proj.weight"), &model.tensor(stem + "gate_proj.weight"),
             &model.tensor(stem + "up_proj.weight")},
            {&model.tensor(stem + "down_proj.weight_scale"), &model.tensor(stem + "gate_proj.weight_scale"),
             &model.tensor(stem + "up_proj.weight_scale")},
            {&model.tensor(stem + "down_proj.weight_scale_2"), &model.tensor(stem + "gate_proj.weight_scale_2"),
             &model.tensor(stem + "up_proj.weight_scale_2")}};
}

struct Q3ExpertLocations {
    std::array<const TensorLocation *, 3> body;
};

Q3ExpertLocations locate_q3_expert(ShardedIndex &model, int layer) {
    const std::string stem = layer_stem(layer) + "mlp.q3_experts.";
    return {{&model.tensor(stem + "down_proj.weight"),
             &model.tensor(stem + "gate_proj.weight"),
             &model.tensor(stem + "up_proj.weight")}};
}

struct PackedExpertFileHeader {
    char magic[8];
    uint32_t version, layers, experts, records;
    uint64_t index_offset, data_offset, file_bytes, source_bytes, stored_bytes;
};
static_assert(sizeof(PackedExpertFileHeader) == 64);

struct PackedExpertIndexEntry {
    uint64_t offset;
    uint32_t stored_bytes, padded_bytes;
};
static_assert(sizeof(PackedExpertIndexEntry) == 16);

struct PackedExpertRecordHeader {
    char magic[4];
    uint16_t layer, expert;
    uint32_t escapes[3];
    float globals[3];
    uint8_t codebooks[3][16];
    uint8_t reserved[48];
};
static_assert(sizeof(PackedExpertRecordHeader) == 128);

// Trace for the learned DFlash falsifier. The fixed record is fourteen cache
// lines. Exact-teacher traces include the expert-contribution Gram label;
// feature-only traces run on the approximate trajectory and stop before expert
// execution. Full target/DFlash logits remain in their existing raw dumps and
// are joined offline by (epoch,row).
struct alignas(64) DfFalsifierTraceHeader {
    char magic[8];
    uint16_t version, header_bytes;
    uint32_t record_bytes;
    uint16_t layer_count, expert_count, topk, candidate_k, hidden_sketch, reserved16;
    uint32_t hidden, flags;
    uint8_t reserved[28];
};
static_assert(sizeof(DfFalsifierTraceHeader) == 64);

struct alignas(64) DfFalsifierEventV2 {
    uint32_t epoch;
    uint16_t layer;
    uint8_t row, tokens;
    uint8_t verify_row, exec_k;
    uint16_t flags;  // bit 0: MLA, bit 1: KDA archive, bit 2: feature-only
    // Four 32-bit candidate masks: host-ready, host-in-flight,
    // device-resident, pinned.
    std::array<uint32_t, 4> candidate_residency;
    std::array<uint16_t, 8> expert;
    std::array<float, 8> weight;
    std::array<uint16_t, 32> candidate_expert;
    std::array<float, 32> candidate_logit;
    std::array<float, 32> candidate_choice;
    // raw mean/std/max/second; all-sigmoid sum; selected/all sigmoid mass;
    // biased top1-top2 gap; entropy of the normalized selected weights.
    std::array<float, 8> router_summary;
    std::array<float, 64> hidden_countsketch;
    // Upper triangle, row-major, of G_ij=dot(w_i e_i,w_j e_j)/hidden.
    std::array<float, 36> contribution_gram;
    // exact norm2/hidden, cancellation ratio, exact replay max error,
    // normalized-input norm2/hidden.
    std::array<float, 4> tail;
    uint8_t reserved[52];
};
static_assert(sizeof(DfFalsifierEventV2) == 896);

// One routed-expert record: 3 projections x (4 MiB nibbles + 512 KiB scales)
// + 12 B globals, streamed with O_DIRECT into a 4096-aligned pinned window.
// Windows double as a host-RAM LRU: completed records stay resident so a
// repeat (layer, expert) skips the NVMe round trip entirely -- one decode
// token touches 42x8 = 336 records, so the tier must hold more than that
// (default 5 GiB pinned ~= 370 records) or it can never hit. A fixed pool of
// reader threads keeps queue depth on the disk; demand records always jump
// ahead of speculative ones in the pool queues.
class ExpertStager {
public:
    static constexpr size_t kBodyBytes = 12ull << 20;
    static constexpr size_t kScaleBytes = 1536ull << 10;
    static constexpr size_t kProjectionBodyBytes = 4ull << 20;
    static constexpr size_t kProjectionScaleBytes = 512ull << 10;
    static constexpr size_t kPackedScaleBytes = 256ull << 10;
    static constexpr size_t kScalePrefixEntries = kPackedScaleBytes / 256 + 1;
    static constexpr size_t kPackedDeviceCapacity = kScaleBytes + (64ull << 10);
    static constexpr size_t kPayloadCapacity = kBodyBytes + kScaleBytes + 64;
    static constexpr size_t kAlignment = 4096;
    // v2 sidecar: per-projection scale "region" = packed nibbles + escapes +
    // codebook + align pad + prefix table, padded to 4 KiB. The reader scratch
    // only needs the header page plus one such region (CPU-expand path).
    static constexpr size_t kV2RegionOverhead =
        kPackedScaleBytes + 16 + alignof(uint32_t) +
        kScalePrefixEntries * sizeof(uint32_t) + kAlignment;
    static constexpr size_t kV2MaxEscapes =
        (kPayloadCapacity - kBodyBytes - 3 * kV2RegionOverhead) / 3;
    static constexpr size_t kV2ReaderScratchBytes =
        kAlignment + ((kPackedScaleBytes + kV2MaxEscapes + 16 + alignof(uint32_t) +
                       kScalePrefixEntries * sizeof(uint32_t) + kAlignment - 1) &
                      ~(kAlignment - 1));
    static constexpr size_t kWindowBytes =
        (kPayloadCapacity + 2 * kAlignment - 2) & ~(kAlignment - 1);
    // Block 11 is the largest live Q3 record: IQ4 gate + IQ4 up + Q6 down =
    // 15.0625 MiB. Each component is read through its own page-aligned
    // O_DIRECT window, so reserve one exact 16 MiB size class for now.
    static constexpr size_t kQ3PayloadCapacity = 16ull << 20;
    static constexpr size_t kQ3WindowBytes =
        (kQ3PayloadCapacity + 2 * kAlignment - 2) & ~(kAlignment - 1);
    static constexpr uint32_t kNoKey = 0xffffffffu;

    explicit ExpertStager(ShardedIndex &model, ShardedIndex *stripe_model,
                          uint64_t host_cache_bytes, bool q3_experts = false)
        : model_(model), stripe_model_(stripe_model), q3_experts_(q3_experts) {
        window_bytes_ = q3_experts_ ? kQ3WindowBytes : kWindowBytes;
        record_capacity_ = q3_experts_ ? kQ3PayloadCapacity : kPayloadCapacity;
        // A full decode token needs 336 records; default the tier just above
        // that and let the environment shrink it on smaller hosts.
        window_count_ = int(std::clamp<uint64_t>(host_cache_bytes / window_bytes_, 64, 4096));
        // Write-combined host arena (ioaudit #5): windows are pread-written
        // and H2D-read, never CPU-read except the tiny v2 prefix tripwire, so
        // WC pages are safe and may reclaim 5-15% H2D efficiency. A/B only.
        const unsigned arena_flags = std::getenv("INSIGNIA_GLM53_TIER_WC")
                                         ? cudaHostAllocWriteCombined
                                         : cudaHostAllocDefault;
        size_t attempt = size_t(window_count_);
        while (attempt >= 64) {
            void *block = nullptr;
            const cudaError_t status = cudaHostAlloc(&block, attempt * window_bytes_ + kAlignment - 1,
                                                     arena_flags);
            if (status == cudaSuccess) {
                host_raw_ = static_cast<uint8_t *>(block);
                window_count_ = int(attempt);
                break;
            }
            cudaGetLastError();  // clear the sticky error and retry smaller
            attempt /= 2;
        }
        require(host_raw_, "cudaHostAlloc expert host cache");
        host_ = reinterpret_cast<uint8_t *>(
            (reinterpret_cast<uintptr_t>(host_raw_) + kAlignment - 1) & ~(uintptr_t(kAlignment) - 1));
        windows_.resize(size_t(window_count_));
        for (WindowState &state : windows_)
            check(cudaEventCreateWithFlags(&state.copy_done, cudaEventDisableTiming),
                  "cudaEventCreate expert copy");
        check(cudaStreamCreate(&copy_stream_), "cudaStreamCreate expert copies");
        overlap_reads_ = std::getenv("INSIGNIA_GLM53_EAGER_EXPERT_JOIN") == nullptr;
        l2_mode_ = std::getenv("INSIGNIA_GLM53_PAGECACHE_L2") != nullptr;
        if (const char *filter = std::getenv("INSIGNIA_GLM53_ADMIT_N"))
            admit_threshold_ = std::max(1, std::atoi(filter));
        else if (const char *filter = std::getenv("INSIGNIA_GLM53_ADMIT"))
            admit_threshold_ = std::atoi(filter) != 0 ? 2 : 1;
        if (const char *budget = std::getenv("INSIGNIA_GLM53_EXPERT_VRAM_MB"))
            vram_budget_mb_ = std::max(0, std::atoi(budget));
        if (const char *value = std::getenv("INSIGNIA_GLM53_DEVICE_PACKED_SCALES"))
            device_packed_scales_ = std::atoi(value) != 0;
        const char *packed_direct_setting =
            std::getenv("INSIGNIA_GLM53_PACKED_DIRECT");
        const bool packed_direct_explicit = packed_direct_setting != nullptr;
        if (packed_direct_setting)
            packed_direct_ = std::atoi(packed_direct_setting) != 0;
        const char *packed_tablefree_setting =
            std::getenv("INSIGNIA_GLM53_NVFP4_TABLEFREE");
        const bool packed_tablefree_explicit = packed_tablefree_setting != nullptr;
        packed_tablefree_ = !packed_tablefree_setting ||
                            std::atoi(packed_tablefree_setting) != 0;
        // The model has exactly 42 sparse layers (3..44).  The legacy
        // layer-id mapping carved 46 segments and stranded four complete
        // slices.  Keep the corrected dense mapping A/B-gated until the
        // real-prompt campaign confirms the trace-replay gain.
        if (const char *compact =
                std::getenv("INSIGNIA_GLM53_VRAM_COMPACT_SEGMENTS"))
            compact_device_segments_ = std::atoi(compact) != 0;
        else
            compact_device_segments_ = q3_experts_;
        // A front-of-batch miss must not blindly evict an expert required a
        // few canonical slots later.  Spend a handful of integer compares
        // to retain whole 12.8 MiB records: prefer an entry absent from the
        // rest of this batch, otherwise evict its farthest-future member.
        if (const char *batch = std::getenv("INSIGNIA_GLM53_VRAM_BATCH_VICTIM"))
            batch_aware_device_victim_ = std::atoi(batch) != 0;
        else
            batch_aware_device_victim_ = q3_experts_;
        // F3 residency ordering: consult the VRAM expert tier before starting
        // an NVMe read. The legacy order only noticed device residency at
        // upload time, after the bytes had already been re-read from disk.
        if (const char *f3 = std::getenv("INSIGNIA_GLM53_F3"))
            f3_device_consult_ = std::atoi(f3) != 0;
        else
            f3_device_consult_ = true;
        // O(1) intrusive LRU for the host tier: the ioaudit measured 10-12
        // ms/verify-round in the O(2425) victim scans. The list order mirrors
        // the admission-stamp ordering the scanner selected, so eviction
        // choice is unchanged outside stamp-0 (never-admitted) ties.
        if (const char *o1 = std::getenv("INSIGNIA_GLM53_TIER_O1"))
            tier_o1_ = std::atoi(o1) != 0;
        else
            tier_o1_ = true;
        tier_slru_ = std::getenv("INSIGNIA_GLM53_TIER_SLRU") != nullptr;
        stripe_required_ = std::getenv("INSIGNIA_GLM53_STRIPE_REQUIRED") != nullptr;
        lru_prev_.assign(size_t(window_count_), -1);
        lru_next_.assign(size_t(window_count_), -1);
        if (const char *path = std::getenv("INSIGNIA_GLM53_PACKED_EXPERTS")) {
            require(!stripe_model_,
                    "INSIGNIA_GLM53_PACKED_EXPERTS and STRIPE_INDEX cannot be combined yet");
            const char *gpu = std::getenv("INSIGNIA_GLM53_PACKED_GPU");
            packed_gpu_scales_ = !gpu || std::atoi(gpu) != 0;
            if (const char *merge = std::getenv("INSIGNIA_GLM53_PACKED_V2"))
                packed_merge_h2d_ = std::atoi(merge) != 0;
            if (const char *kernel = std::getenv("INSIGNIA_GLM53_PACKED_KERNEL"))
                packed_kernel_v2_ = std::atoi(kernel) == 2;
            open_packed_experts(path);
            // XPR1-v2 carries the exact random-access prefix directory needed
            // by the direct kernels.  Three matched local model runs made the
            // direct path 2.34% faster end-to-end with byte-identical logits,
            // so v2 promotes it by default.  PACKED_DIRECT=0 remains the
            // explicit expanded-scale rollback arm; v1 stays expanded.
            if (!packed_direct_explicit && packed_version_ >= 2)
                packed_direct_ = true;
            if (packed_direct_) {
                // Direct views consume the v2 blobs in-place and require the
                // packed device-slot layout.  Keep the single switch complete.
                device_packed_scales_ = true;
                packed_gpu_scales_ = true;
                packed_kernel_v2_ = true;
            }
            if (device_packed_scales_)
                require(packed_version_ >= 2 && packed_gpu_scales_ && packed_kernel_v2_,
                        "packed device slots require XPR1-v2, GPU scales, and PACKED_KERNEL=2");
            if (packed_gpu_scales_)
                check(cudaMalloc(&packed_scale_device_, kPackedDeviceCapacity),
                      "cudaMalloc packed scale transport/execution scratch");
            std::printf("packed scale execution: %s%s\n",
                        packed_direct_ ? "direct XPR1-v2" : "expanded E4M3",
                        packed_direct_explicit ? " (explicit)" : " (default)");
            if (packed_direct_)
                std::printf("packed E2M1 decode: %s%s\n",
                            packed_tablefree_ ? "table-free arithmetic" : "shared LUT",
                            packed_tablefree_explicit ? " (explicit)" : " (default)");
        }
        if (device_packed_scales_)
            require(packed_fd_ >= 0,
                    "packed device slots require INSIGNIA_GLM53_PACKED_EXPERTS");
        for (int window = 0; window < window_count_; ++window) free_windows_.push_back(window);
        start_pool();
        load_pin_list();
    }
    ~ExpertStager() {
        stop_pool();
        if (copy_stream_) {
            cudaStreamSynchronize(copy_stream_);
            cudaStreamDestroy(copy_stream_);
        }
        cudaDeviceSynchronize();
        for (WindowState &state : windows_)
            if (state.copy_done) cudaEventDestroy(state.copy_done);
        if (host_raw_) cudaFreeHost(host_raw_);
        if (device_arena_) cudaFree(device_arena_);
        for (cudaEvent_t &event : device_slot_reads_)
            if (event) cudaEventDestroy(event);
        if (packed_scale_device_) cudaFree(packed_scale_device_);
        if (device_) cudaFree(device_);
        if (packed_direct_fd_ >= 0) ::close(packed_direct_fd_);
        if (packed_fd_ >= 0) ::close(packed_fd_);
    }
    ExpertStager(const ExpertStager &) = delete;
    ExpertStager &operator=(const ExpertStager &) = delete;

    // Verify-round epoch for acceptance-prefix demotion (Runner calls this at
    // the head of every verify round; demote_round rejects foreign epochs).
    void set_epoch(uint32_t epoch) { round_epoch_ = epoch; }
    static uint32_t route_key_public(int layer, int expert) { return route_key(layer, expert); }
    // Acceptance-prefix demote (insert-then-demote): keys staged at `epoch`
    // whose rows were all rejected are pushed to the cold end of
    // probationary so the protected half survives verify bursts (P7 fix).
    uint64_t demote_round(const std::unordered_set<uint32_t> &rejected_keys, uint32_t epoch) {
        if (!tier_slru_) return 0;
        uint64_t moved = 0;
        for (const uint32_t key : rejected_keys) {
            const auto found = flight_index_.find(key);
            if (found == flight_index_.end()) continue;
            WindowState &state = windows_[size_t(found->second)];
            if (!state.done || state.claimed || state.releasing || state.pinned) continue;
            if (state.round_epoch != epoch) continue;  // not this round's residue
            slru_demote_cold(found->second);
            ++moved;
        }
        demoted_cold_ += moved;
        return moved;
    }
    uint64_t demoted_cold() const { return demoted_cold_; }
    // Speculative read-ahead keyed on the previous token's routing. Cheap to
    // call: resident or in-flight experts are skipped, and 8 windows stay
    // unreserved so a demand batch always finds free slots immediately.
    int prefetch(int layer, const int *experts, int count) {
        if (!overlap_reads_) return 0;
        // Pass-through H2D copies complete between layers, but only demand's
        // take_window() used to reclaim them. Without this nonblocking reap,
        // the speculative path sees an empty free list and silently stops.
        reap_released();
        int started = 0;
        for (int index = 0; index < count; ++index) {
            if (int(free_windows_.size()) <= 8) break;
            if (experts[index] < 0) continue;  // routing unknown (first token)
            const uint32_t key = route_key(layer, experts[index]);
            if (flight_index_.count(key)) continue;
            const int window = free_windows_.back();
            free_windows_.pop_back();
            start_read(window, key, layer, experts[index], false);
            ++prefetch_started_;
            ++started;
        }
        return started;
    }

    // Static hot-set pinning: a trace-derived list of the hottest experts
    // per sparse layer is loaded into dedicated windows (and, later, VRAM
    // slots) that eviction can never reclaim. Real-text routing entropy is
    // ~5 bits (not the 8.17 maximum), so the top-8 per layer alone covers
    // ~40% of accesses — capacity the recency-only LRU never captures.
    void load_pin_list() {
        const char *path = std::getenv("INSIGNIA_GLM53_PIN_LIST");
        if (!path || !*path) return;
        int per_layer = 8, per_layer_device = 2;
        if (const char *value = std::getenv("INSIGNIA_GLM53_PIN_HOST"))
            per_layer = std::max(0, std::atoi(value));
        if (const char *value = std::getenv("INSIGNIA_GLM53_PIN_DEV"))
            per_layer_device = std::max(0, std::atoi(value));
        std::FILE *file = std::fopen(path, "r");
        if (!file) {
            std::printf("pin list: cannot open %s, ignoring\n", path);
            return;
        }
        std::vector<int> staged;
        int layer = -1, taken = 0, taken_device = 0, pinned_count = 0;
        int parsed_layer = 0, parsed_expert = 0, parsed_hits = 0;
        while (std::fscanf(file, "%d %d %d", &parsed_layer, &parsed_expert, &parsed_hits) == 3) {
            if (parsed_layer != layer) {
                layer = parsed_layer;
                taken = 0;
                taken_device = 0;
            }
            if (taken >= per_layer) continue;
            if (int(free_windows_.size()) <= 16) break;
            const uint32_t key = route_key(layer, parsed_expert);
            if (flight_index_.count(key)) {
                ++taken;
                ++taken_device;
                continue;
            }
            const int window = free_windows_.back();
            free_windows_.pop_back();
            start_read(window, key, layer, parsed_expert, false);
            windows_[size_t(window)].pinned = true;
            lru_unlink(window);
            staged.push_back(window);
            if (taken < per_layer_device) pinned_device_keys_.insert(key);
            ++taken;
            ++taken_device;
            ++pinned_count;
        }
        std::fclose(file);
        for (int window : staged) wait_and_consume_error(window);
        for (int window : staged)
            if (windows_[size_t(window)].error) {
                windows_[size_t(window)].pinned = false;
                seg_push_back(window, 1);
            }
        if (pinned_count)
            std::printf("pin list: %d hot records pinned in host tier (%zu VRAM keys)\n",
                        pinned_count, pinned_device_keys_.size());
    }

    void load_batch(int layer, const std::array<int, 8> &experts, int count = 8,
                    bool populate_cache = true, uint8_t populate_mask = 0xffu) {
        require(count >= 1 && count <= 8, "expert batch must contain 1..8 records");
        batch_layer_ = layer;
        for (int slot = 0; slot < count; ++slot) batch_experts_[slot] = experts[slot];
        batch_cached_.fill(false);
        batch_admit_.fill(true);
        batch_populate_.fill(false);
        batch_window_.fill(-1);
        batch_device_.fill(false);
        batch_count_ = count;
        batch_read_begin_ = std::chrono::steady_clock::now();
        batch_read_ends_.clear();
        for (int slot = 0; slot < count; ++slot)
            batch_populate_[size_t(slot)] = populate_cache && ((populate_mask >> slot) & 1u);
        // Claim every resident member before any miss chooses a victim, so
        // staging cannot evict a later slot in the same batch.
        for (int slot = 0; slot < count; ++slot) {
            const auto found = flight_index_.find(route_key(layer, experts[slot]));
            if (found != flight_index_.end()) windows_[size_t(found->second)].claimed = true;
        }
        int hits = 0, adopted = 0, f3_rescued = 0;
        uint64_t started_bytes = 0;
        for (int slot = 0; slot < count; ++slot) {
            const int expert = experts[slot];
            const uint32_t key = route_key(layer, expert);
            unsigned sightings = 1;
            const auto sight = sight_count_.find(key);
            if (sight != sight_count_.end()) sightings = ++sight->second;
            else {
                sight_count_.emplace(key, 1u);
                sight_order_.push_back(key);
                if (sight_order_.size() > 8192) {
                    sight_count_.erase(sight_order_.front());
                    sight_order_.pop_front();
                }
            }
            const auto resident = flight_index_.find(key);
            if (resident != flight_index_.end()) {
                WindowState &state = windows_[size_t(resident->second)];
                state.claimed = true;
                state.round_epoch = round_epoch_;
                batch_window_[slot] = resident->second;
                if (window_done(resident->second)) {
                    // Host-tier hit: the record is already pinned in RAM and
                    // only owes the 13.5 MiB H2D copy.
                    batch_cached_[slot] = true;
                    ++state.hits;
                    ++hits;
                    if (!state.demand) ++prefetch_useful_;
                    if (tier_slru_) slru_hit(resident->second);
                } else {
                    // A prefetch for exactly this (layer, expert) is in
                    // flight; promote it instead of reading bytes twice.
                    const bool speculative = !state.demand;
                    promote_read(resident->second);
                    if (speculative) {
                        ++adopted;
                        ++prefetch_useful_;
                    }
                }
                continue;
            }
            // F3 residency ordering: the record may sit in the VRAM tier from
            // an earlier token even though its host window was evicted. The
            // legacy flow re-read 13.5 MiB from NVMe and only noticed the
            // device hit inside upload(); adopting the slot here skips both
            // the disk read and the window churn. If the slot is recycled in
            // the narrow window before upload (this batch's own misses are
            // the only possible evictor), upload falls back to a demand read.
            if (f3_device_consult_ && device_arena_) {
                const auto device_resident = device_index_.find(key);
                if (device_resident != device_index_.end()) {
                    batch_device_[size_t(slot)] = true;
                    batch_cached_[size_t(slot)] = true;
                    ++f3_rescued;
                    continue;
                }
            }
            const int window = take_window();
            start_read(window, key, layer, expert, true);
            batch_window_[slot] = window;
            started_bytes += windows_[size_t(window)].source_bytes;
            // TinyLFU-style door: a record may enter the tier only once its
            // lifetime demand count beats the hit count of the coldest
            // resident (threshold 1 disables the door entirely).
            if (admit_threshold_ <= 1) {
                batch_admit_[slot] = true;
            } else {
                int victim = -1;
                for (int scan = 0; scan < window_count_; ++scan) {
                    const WindowState &state = windows_[size_t(scan)];
                    if (state.key == kNoKey || !state.done || state.claimed || state.releasing)
                        continue;
                    if (victim < 0 || state.hits < windows_[size_t(victim)].hits ||
                        (state.hits == windows_[size_t(victim)].hits &&
                         state.stamp < windows_[size_t(victim)].stamp))
                        victim = scan;
                }
                batch_admit_[slot] = victim < 0 || sightings >= unsigned(admit_threshold_) &&
                    sightings > windows_[size_t(victim)].hits;
            }
        }
        cache_hits_ += hits;
        cache_lookups_ += count;
        f3_rescued_ += uint64_t(f3_rescued);
        // Include F3 candidates in the expected completion count. upload()
        // contributes a zero-I/O marker for a surviving device hit or the
        // actual read end if an earlier slot recycles it before use.
        batch_demand_count_ = count - hits;
        batch_read_ends_.clear();
        batch_read_ends_.reserve(8);
        if (!overlap_reads_)
            for (int slot = 0; slot < count; ++slot)
                if (batch_window_[slot] >= 0) wait_and_consume_error(batch_window_[slot]);
        io_bytes_ += started_bytes;
        (void)adopted;
    }
    // Whole-layer demand read-ahead: puts every still-missing record of the
    // layer's deduplicated union into the reader pool immediately at demand
    // priority, so all readers stream while the GPU works through the first
    // 8-record batches. Records already resident or in flight are skipped.
    void stage_layer(int layer, const int *experts, int count) {
        if (!overlap_reads_) return;
        for (int index = 0; index < count; ++index) {
            if (experts[index] < 0) continue;
            const uint32_t key = route_key(layer, experts[index]);
            const auto resident = flight_index_.find(key);
            if (resident != flight_index_.end()) {
                windows_[size_t(resident->second)].claimed = true;
                windows_[size_t(resident->second)].round_epoch = round_epoch_;
                if (!window_done(resident->second)) promote_read(resident->second);
                continue;
            }
            // F3: a device-resident record needs no host read-ahead; the
            // batch consult adopts the VRAM slot directly at upload time.
            if (f3_device_consult_ && device_arena_ && device_index_.count(key))
                continue;
            const int window = take_window();
            start_read(window, key, layer, experts[index], true);
            windows_[size_t(window)].claimed = true;
            io_bytes_ += windows_[size_t(window)].source_bytes;
        }
    }
    void upload(int slot) {
        if (batch_device_[size_t(slot)]) {
            // F3 device-resident adoption (consulted in load_batch). Serve
            // straight from the VRAM arena: no host window was ever read.
            const uint32_t key = route_key(batch_layer_, batch_experts_[size_t(slot)]);
            const auto found = device_index_.find(key);
            if (found != device_index_.end()) {
                // Fence the previous active slot exactly like the regular
                // path, then adopt this one.
                if (active_device_slot_ >= 0)
                    check(cudaEventRecord(device_slot_reads_[size_t(active_device_slot_)], nullptr),
                          "fence active expert slot");
                ++device_lookups_;
                ++device_hits_;
                const int device_slot = found->second;
                device_slot_stamps_[size_t(device_slot)] = ++device_stamp_;
                active_device_ = device_arena_ + size_t(device_slot) * device_stride_;
                active_device_slot_ = device_slot;
                active_ = device_slot_layouts_[size_t(device_slot)];
                active_globals_ = device_slot_globals_[size_t(device_slot)];
                if (!packed_direct_) expand_active_packed_slot_scales();
                // Keep one completion marker per non-host-hit slot. Device
                // hits use the batch start as a zero-I/O marker; a later
                // same-batch recycle instead falls through and records its
                // real demand-read completion below. This prevents an F3
                // fallback from finalizing the batch timer too early.
                record_batch_read_end(batch_read_begin_);
                batch_device_[size_t(slot)] = false;
                return;
            }
            // The slot was recycled between the batch consult and this
            // upload (only this batch's own miss uploads can evict). Stage
            // the demand read now and fall through to the regular path.
            const int window = take_window();
            start_read(window, key, batch_layer_, batch_experts_[size_t(slot)], true);
            io_bytes_ += windows_[size_t(window)].source_bytes;
            batch_window_[size_t(slot)] = window;
            batch_device_[size_t(slot)] = false;
            batch_cached_[size_t(slot)] = false;
        }
        const int window = batch_window_[slot];
        require(window >= 0, "expert slot has no record in flight");
        WindowState &state = windows_[size_t(window)];
        if (!window_done(window)) {
            const auto wait_begin = std::chrono::steady_clock::now();
            wait_window(window);
            read_wait_seconds_ +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_begin).count();
        }
        if (!batch_cached_[slot]) record_batch_read_end(state.end);
        if (state.error) {
            std::exception_ptr error = state.error;
            release_window(window);
            std::rethrow_exception(error);
        }
        ensure_device_arena();
        if (device_arena_) {
            // Slot-owning arena path. The previously active device slot is
            // fenced here: by now the default stream holds every GEMV launch
            // that reads it, so the event provably covers its last consumer
            // before any future recycle overwrites the slot.
            if (active_device_slot_ >= 0)
                check(cudaEventRecord(device_slot_reads_[size_t(active_device_slot_)], nullptr),
                      "fence active expert slot");
            ++device_lookups_;
            const uint32_t key = state.key;
            const auto found = device_index_.find(key);
            int device_slot;
            if (found != device_index_.end()) {
                device_slot = found->second;
                ++device_hits_;
            } else {
                device_slot = take_device_slot(state.layer, slot);
                // The recycle waits on the victim's read fence: the copy
                // stream never overwrites bytes a default-stream GEMV may
                // still be reading. First fill of a fresh slot waits on an
                // unrecorded event, which is immediately signalled.
                check(cudaStreamWaitEvent(copy_stream_, device_slot_reads_[size_t(device_slot)], 0),
                      "order expert slot recycle");
                enqueue_record_copy(state,
                                    device_arena_ + size_t(device_slot) * device_stride_);
                check(cudaEventRecord(state.copy_done, copy_stream_), "record expert copy");
                check(cudaStreamWaitEvent(nullptr, state.copy_done, 0), "order expert copy");
                state.copy_issued = true;
                device_index_.emplace(key, device_slot);
                device_slot_keys_[size_t(device_slot)] = key;
                device_slot_pinned_[size_t(device_slot)] =
                    pinned_device_keys_.count(key) ? 1 : 0;
                // F3: remember how this slot's image is laid out (and its
                // host-side globals) so a host-evicted record can be served
                // by upload() without an NVMe re-read.
                device_slot_layouts_[size_t(device_slot)] = state.layout;
                device_slot_globals_[size_t(device_slot)] = state.globals;
            }
            device_slot_stamps_[size_t(device_slot)] = ++device_stamp_;
            active_device_ = device_arena_ + size_t(device_slot) * device_stride_;
            active_device_slot_ = device_slot;
        } else {
            ensure_device_scratch();
            // Legacy single-scratch path: async H2D on the copy stream lets
            // the copy engine overlap the SMs' previous expert GEMVs instead
            // of stalling the CPU per record. The default-stream GEMVs below
            // wait on copy_done, so ordering holds. copy_stream_ MUST stay a
            // legacy-synchronizing stream (created with cudaStreamCreate):
            // the device_ scratch is reused across slots and only the
            // legacy-sync semantics keep slot N+1's copy behind slot N's
            // default-stream GEMVs.
            enqueue_record_copy(state, device_);
            check(cudaEventRecord(state.copy_done, copy_stream_), "record expert copy");
            check(cudaStreamWaitEvent(nullptr, state.copy_done, 0), "order expert copy");
            state.copy_issued = true;
            active_device_ = device_;
            active_device_slot_ = -1;
        }
        if (device_packed_scales_ && active_device_slot_ >= 0) {
            active_ = device_slot_layouts_[size_t(active_device_slot_)];
            active_globals_ = device_slot_globals_[size_t(active_device_slot_)];
            if (!packed_direct_) expand_active_packed_slot_scales();
        } else {
            active_ = state.layout;
            active_globals_ = state.globals;
        }
        if (state.pinned || (batch_populate_[size_t(slot)] && batch_admit_[size_t(slot)])) {            // Admitted: the window stays resident in the host LRU; eviction
            // re-checks copy_done before the slot can be refilled. The pinned
            // window now owns the bytes, so drop the page-cache shadow.
            if (l2_mode_ && state.l2_shard >= 0) {
                ShardedIndex &source =
                    state.drive == 1 && stripe_model_ ? *stripe_model_ : model_;
                source.evict_span_cache(uint16_t(state.l2_shard),
                                        state.l2_offset, state.l2_bytes);
            }
            state.claimed = false;
            state.stamp = ++stamp_;
            seg_move_front(window, 1);
        } else {
            // Passed through: drop it from the resident index immediately so
            // later lookups re-read, and free the window once the async copy
            // has drained (reap_released polls the event).
            if (state.key != kNoKey) flight_index_.erase(state.key);
            state.key = kNoKey;
            state.releasing = true;
            releasing_.push_back(window);
            lru_unlink(window);
            if (!state.demand) ++prefetch_wasted_;
        }
    }
    bool active_slot_is_packed() const {
        return device_packed_scales_ && active_device_slot_ >= 0;
    }
    bool q3_experts() const { return q3_experts_; }
    size_t window_bytes() const { return window_bytes_; }
    size_t record_capacity() const { return record_capacity_; }
    TensorType projection_type(int projection) const {
        require(q3_experts_ && projection >= 0 && projection < 3,
                "Q3 projection type requested outside native Q3 execution");
        return active_.types[size_t(projection)];
    }
    bool packed_direct_active() const {
        return packed_direct_ && active_slot_is_packed();
    }
    bool packed_tablefree() const { return packed_tablefree_; }
    insignia::glm53::Nvfp4PackedScaleView packed_scale_view(int projection) const {
        require(packed_direct_active() && projection >= 0 && projection < 3,
                "packed scale view requested outside direct packed execution");
        const uint8_t *blob = active_device_ + active_.packed_blob[size_t(projection)];
        const size_t escapes = active_.packed_escapes[size_t(projection)];
        const size_t codebook = active_.packed_codebook[size_t(projection)];
        const size_t prefix = active_.packed_prefix[size_t(projection)];
        require(codebook >= escapes && codebook - escapes <= kProjectionScaleBytes,
                "packed scale escape span is invalid");
        return {
            blob,
            blob + escapes,
            blob + codebook,
            reinterpret_cast<const uint32_t *>(blob + prefix),
            uint32_t(kProjectionScaleBytes),
            uint32_t(codebook - escapes),
            uint32_t(kScalePrefixEntries),
            256u,
            15u,
        };
    }
    insignia::glm53::Nvfp4PackedScaleView down_packed_scale() const {
        return packed_scale_view(0);
    }
    insignia::glm53::Nvfp4PackedScaleView gate_packed_scale() const {
        return packed_scale_view(1);
    }
    insignia::glm53::Nvfp4PackedScaleView up_packed_scale() const {
        return packed_scale_view(2);
    }
    const uint8_t *down_weight() const {
        return active_device_ + (active_slot_is_packed() ? active_.packed_body[0] : active_.body[0]);
    }
    const uint8_t *gate_weight() const {
        return active_device_ + (active_slot_is_packed() ? active_.packed_body[1] : active_.body[1]);
    }
    const uint8_t *up_weight() const {
        return active_device_ + (active_slot_is_packed() ? active_.packed_body[2] : active_.body[2]);
    }
    const uint8_t *down_scale() const {
        return active_slot_is_packed() ? packed_scale_device_
                                       : active_device_ + active_.scales[0];
    }
    const uint8_t *gate_scale() const {
        return active_slot_is_packed() ? packed_scale_device_ + kProjectionScaleBytes
                                       : active_device_ + active_.scales[1];
    }
    const uint8_t *up_scale() const {
        return active_slot_is_packed() ? packed_scale_device_ + 2 * kProjectionScaleBytes
                                       : active_device_ + active_.scales[2];
    }
    float down_global(int) const { return active_globals_[0]; }
    float gate_global(int) const { return active_globals_[1]; }
    float up_global(int) const { return active_globals_[2]; }
    double io_seconds() const { return io_seconds_; }
    uint64_t io_bytes() const { return io_bytes_; }
    uint64_t prefetch_bytes() const { return prefetch_bytes_; }
    uint64_t cache_hits() const { return cache_hits_; }
    // Demand NVMe record reads started (load_batch misses, F3 fallbacks,
    // stage_layer unions) - the U3 adaptive-k cost estimator's denominator.
    uint64_t records_read() const { return records_read_; }
    uint64_t drive_records(int drive) const {
        return drive >= 0 && drive < 2 ? drive_records_[size_t(drive)].load() : 0;
    }
    uint64_t drive_bytes(int drive) const {
        return drive >= 0 && drive < 2 ? drive_bytes_[size_t(drive)].load() : 0;
    }
    uint64_t stripe_fallbacks() const { return stripe_fallbacks_.load(); }
    uint64_t cache_lookups() const { return cache_lookups_; }
    uint64_t prefetch_started() const { return prefetch_started_; }
    uint64_t prefetch_useful() const { return prefetch_useful_; }
    uint64_t prefetch_wasted_observable() const { return prefetch_wasted_; }
    int cache_slots() const { return window_count_; }
    double read_wait_seconds() const { return read_wait_seconds_; }
    uint64_t device_hits() const { return device_hits_; }
    uint64_t f3_rescued() const { return f3_rescued_; }
    uint64_t device_lookups() const { return device_lookups_; }
    int device_slots() const { return device_slot_count_; }
    bool device_topk_capable() const {
        const int segments = compact_device_segments_ ? 42 : 46;
        return device_arena_ && device_slot_count_ / segments >= 8;
    }
    int active_device_slot() const { return active_device_slot_; }
    void fence_device_slots(const int *slots, int count) {
        require(device_arena_ && slots && count >= 1 && count <= 8,
                "invalid expert device batch fence");
        for (int index = 0; index < count; ++index) {
            require(slots[index] >= 0 && slots[index] < device_slot_count_,
                    "expert device batch lost a slot");
            for (int earlier = 0; earlier < index; ++earlier)
                require(slots[index] != slots[earlier],
                        "expert device batch recycled a live slot");
            check(cudaEventRecord(device_slot_reads_[size_t(slots[index])], nullptr),
                  "fence batched expert slots");
        }
        active_device_slot_ = -1;
        active_device_ = nullptr;
    }
    // Whole-layer prompt staging owns large transient sidecars.  Freeze the
    // permanent expert tier first so those temporaries cannot silently reduce
    // the decode-time slot count chosen from free VRAM.
    void prime_device_arena() { ensure_device_arena(); }
    bool packed_experts() const { return packed_fd_ >= 0; }
    bool packed_gpu_scales() const { return packed_gpu_scales_; }
    uint64_t packed_h2d_bytes() const { return packed_h2d_bytes_.load(); }
    uint64_t packed_h2d_records() const { return packed_h2d_records_.load(); }
    uint64_t packed_expanded_bytes() const { return packed_expanded_bytes_.load(); }
    double packed_expand_seconds() const {
        return packed_expand_nanoseconds_.load() * 1.0e-9;
    }
    std::array<uint32_t, 4> residency_masks(int layer, const uint16_t *experts,
                                            int count) {
        require(count >= 1 && count <= 32, "residency probe count must be 1..32");
        std::array<uint32_t, 4> result{};
        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            for (int slot = 0; slot < count; ++slot) {
                const auto found = flight_index_.find(route_key(layer, experts[size_t(slot)]));
                if (found == flight_index_.end()) continue;
                const WindowState &state = windows_[size_t(found->second)];
                result[size_t(state.done ? 0 : 1)] |= uint32_t(1u) << slot;
                if (state.pinned) result[3] |= uint32_t(1u) << slot;
            }
        }
        for (int slot = 0; slot < count; ++slot)
            if (device_index_.count(route_key(layer, experts[size_t(slot)])))
                result[2] |= uint32_t(1u) << slot;
        return result;
    }
private:
    struct Layout {
        std::array<size_t, 3> body{};
        std::array<size_t, 3> scales{};
        std::array<TensorType, 3> types{};
        std::array<size_t, 3> packed_body{}, packed_blob{}, packed_device{};
        std::array<size_t, 3> packed_escapes{}, packed_codebook{}, packed_prefix{};
        std::array<size_t, 3> packed_blob_bytes{};
        size_t packed_blob_span = 0;
        size_t bytes = 0;
        bool packed_scales = false;
    };
    struct WindowState {
        uint32_t key = kNoKey;
        int layer = -1, expert = -1;
        uint8_t drive = 0;
        bool demand = false, done = true, claimed = false, copy_issued = false;
        bool releasing = false, pinned = false;
        uint8_t segment = 0;    // 0=unlisted, 1=probationary, 2=protected
        uint32_t round_epoch = 0;
        uint64_t stamp = 0;
        uint64_t source_bytes = 0;
        unsigned hits = 0;
        Layout layout{};
        std::array<float, 3> globals{};
        uint8_t *payload = nullptr;
        cudaEvent_t copy_done = nullptr;
        std::exception_ptr error;
        std::chrono::steady_clock::time_point end{};
        // Page-cache L2 span bookkeeping (record packed reads only).
        int l2_shard = -1;
        uint64_t l2_offset = 0, l2_bytes = 0;
    };

    static uint32_t route_key(int layer, int expert) {
        return uint32_t(layer) * 4096u + uint32_t(expert);
    }
    static void pread_exact(int fd, uint64_t offset, void *destination, size_t bytes,
                            const char *what) {
        auto *output = static_cast<uint8_t *>(destination);
        size_t done = 0;
        while (done < bytes) {
            const ssize_t count = ::pread(fd, output + done, bytes - done, off_t(offset + done));
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0)
                throw std::system_error(count < 0 ? errno : EIO, std::generic_category(), what);
            done += size_t(count);
        }
    }
    void open_packed_experts(const std::filesystem::path &path) {
        packed_fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (packed_fd_ < 0)
            throw std::system_error(errno, std::generic_category(),
                                    "open packed expert sidecar " + path.string());
        PackedExpertFileHeader header{};
        pread_exact(packed_fd_, 0, &header, sizeof(header), "read packed expert header");
        require(std::memcmp(header.magic, "IG53XPK1", 8) == 0 &&
                (header.version == 1 || header.version == 2),
                "bad packed expert sidecar header");
        packed_version_ = header.version;
        require(header.layers == model_.layers() && header.experts == model_.experts(),
                "packed expert sidecar geometry does not match model");
        std::error_code error;
        const uint64_t actual_bytes = std::filesystem::file_size(path, error);
        require(!error && actual_bytes == header.file_bytes,
                "packed expert sidecar size does not match header");
        packed_entries_.resize(size_t(header.layers) * header.experts);
        pread_exact(packed_fd_, header.index_offset, packed_entries_.data(),
                    packed_entries_.size() * sizeof(PackedExpertIndexEntry),
                    "read packed expert index");
        size_t populated = 0;
        size_t max_v2_payload = 0;
        for (const PackedExpertIndexEntry &entry : packed_entries_) {
            if (!entry.offset) {
                require(entry.stored_bytes == 0 && entry.padded_bytes == 0,
                        "malformed empty packed expert index entry");
                continue;
            }
            require((entry.offset & (kAlignment - 1)) == 0 &&
                    (entry.padded_bytes & (kAlignment - 1)) == 0 &&
                    entry.stored_bytes <= entry.padded_bytes &&
                    entry.offset <= header.file_bytes &&
                    entry.padded_bytes <= header.file_bytes - entry.offset,
                    "malformed packed expert index entry");
            if (packed_version_ == 1) {
                packed_scratch_bytes_ = std::max(packed_scratch_bytes_, size_t(entry.padded_bytes));
            } else {
                require(entry.stored_bytes >= kAlignment + kBodyBytes,
                        "v2 packed expert record is shorter than header plus bodies");
                max_v2_payload = std::max(max_v2_payload,
                                          size_t(entry.stored_bytes) - kAlignment);
            }
            ++populated;
        }
        require(populated == header.records, "packed expert index record count mismatch");
        if (packed_version_ >= 2) {
            // v2 stages straight into the window; the reader scratch only
            // ever holds the 4 KiB header page (GPU path) or one scale
            // region (CPU-expand path). No 12.8 MiB scratch records.
            packed_scratch_bytes_ = kV2ReaderScratchBytes;
            packed_device_stride_ =
                (max_v2_payload + kAlignment - 1) & ~(kAlignment - 1);
            require(packed_device_stride_ >= kBodyBytes,
                    "v2 packed device stride was not derived from the full index");
        }
        require(packed_scratch_bytes_, "packed expert index record count mismatch");
#ifdef O_DIRECT
        packed_direct_fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
#endif
        const double fraction = header.source_bytes
            ? double(header.stored_bytes) / double(header.source_bytes) : 1.0;
        std::printf("packed experts: %zu records, %.3f GiB logical, %.2f%% smaller, %s + %s (format v%d)\n",
                    populated, header.stored_bytes / double(1ull << 30),
                    100.0 * (1.0 - fraction),
                    packed_direct_fd_ >= 0 ? "O_DIRECT" : "buffered I/O",
                    packed_gpu_scales_ ? "GPU packed-scale transport"
                                       : "AVX2 expanded-scale transport",
                    packed_version_);
    }
    const PackedExpertIndexEntry &packed_entry(int layer, int expert) const {
        const size_t index = size_t(layer) * model_.experts() + size_t(expert);
        require(index < packed_entries_.size() && packed_entries_[index].offset,
                "expert is missing from packed sidecar");
        return packed_entries_[index];
    }
    uint64_t source_bytes(int layer, int expert) const {
        if (q3_experts_) {
            (void)expert;
            const Q3ExpertLocations tensors = locate_q3_expert(model_, layer);
            uint64_t bytes = 0;
            for (const TensorLocation *tensor : tensors.body) {
                require(tensor->shape.size() == 3 &&
                            tensor->shape[0] == model_.experts() &&
                            tensor->bytes % model_.experts() == 0,
                        "malformed aggregate Q3 expert tensor");
                bytes += tensor->bytes / model_.experts();
            }
            return bytes;
        }
        return packed_fd_ >= 0 ? packed_entry(layer, expert).padded_bytes
                               : kBodyBytes + kScaleBytes + 3 * sizeof(float);
    }
    static uint32_t count_escape_nibbles_256(const uint8_t *packed) {
        const __m256i nibble_mask = _mm256_set1_epi8(15);
        const __m256i escape_code = _mm256_set1_epi8(15);
        uint32_t count = 0;
        for (size_t offset = 0; offset < 256; offset += 32) {
            const __m256i bytes =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(packed + offset));
            const __m256i low = _mm256_and_si256(bytes, nibble_mask);
            const __m256i high = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), nibble_mask);
            count += uint32_t(__builtin_popcount(uint32_t(_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(low, escape_code)))));
            count += uint32_t(__builtin_popcount(uint32_t(_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(high, escape_code)))));
        }
        return count;
    }
    static void build_escape_prefix(const uint8_t *packed, uint32_t escape_count,
                                    uint32_t *prefix) {
        uint32_t count = 0;
        for (size_t block = 0; block + 1 < kScalePrefixEntries; ++block) {
            prefix[block] = count;
            count += count_escape_nibbles_256(packed + block * 256);
        }
        prefix[kScalePrefixEntries - 1] = count;
        require(count == escape_count, "packed expert scale escape-count mismatch");
    }
    static void expand_scale_nibbles(const uint8_t *packed, const uint8_t *escapes,
                                     uint32_t escape_count, const uint8_t *codebook,
                                     uint8_t *output, size_t bytes) {
        const __m128i lookup = _mm_loadu_si128(reinterpret_cast<const __m128i *>(codebook));
        const __m128i nibble_mask = _mm_set1_epi8(15);
        const __m128i escape_code = _mm_set1_epi8(15);
        uint32_t escape_at = 0;
        for (size_t index = 0; index < bytes; index += 32) {
            const __m128i bytes =
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(packed + index / 2));
            const __m128i low = _mm_and_si128(bytes, nibble_mask);
            const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask);
            const __m128i first = _mm_unpacklo_epi8(low, high);
            const __m128i second = _mm_unpackhi_epi8(low, high);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output + index),
                             _mm_shuffle_epi8(lookup, first));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output + index + 16),
                             _mm_shuffle_epi8(lookup, second));
            uint32_t masks[2] = {
                uint32_t(_mm_movemask_epi8(_mm_cmpeq_epi8(first, escape_code))),
                uint32_t(_mm_movemask_epi8(_mm_cmpeq_epi8(second, escape_code))),
            };
            for (int half = 0; half < 2; ++half) {
                uint32_t mask = masks[half];
                while (mask) {
                    if (escape_at >= escape_count)
                        throw std::runtime_error("packed expert scale escape underflow");
                    const unsigned bit = unsigned(__builtin_ctz(mask));
                    output[index + size_t(half) * 16 + bit] = escapes[escape_at++];
                    mask &= mask - 1;
                }
            }
        }
        require(escape_at == escape_count, "packed expert scale escape overflow");
    }
    void stage_packed_gpu(Layout &layout, std::array<float, 3> &globals, uint8_t *&payload,
                          uint8_t *window, int layer, int expert, void *scratch) {
        const PackedExpertIndexEntry &entry = packed_entry(layer, expert);
        require(scratch && entry.padded_bytes <= packed_scratch_bytes_,
                "packed expert reader scratch is unavailable");
        pread_exact(packed_direct_fd_ >= 0 ? packed_direct_fd_ : packed_fd_,
                    entry.offset, scratch, entry.padded_bytes,
                    "read packed expert record");
        const auto *header = static_cast<const PackedExpertRecordHeader *>(scratch);
        require(std::memcmp(header->magic, "XPR1", 4) == 0 &&
                header->layer == layer && header->expert == expert,
                "packed expert record key mismatch");
        // A corrupt-but-plausible sidecar can claim huge escape tails; bound
        // them before any window write so a bad record fails loudly instead
        // of spilling past this window into a neighbouring live record.
        constexpr size_t kBlobOverhead = kPackedScaleBytes + 16 +
                                         kScalePrefixEntries * sizeof(uint32_t) +
                                         alignof(uint32_t);
        constexpr size_t kMaxProjectionEscapes =
            (kPayloadCapacity - kBodyBytes - 3 * kBlobOverhead) / 3;
        for (int projection = 0; projection < 3; ++projection)
            require(header->escapes[projection] <= kMaxProjectionEscapes,
                    "packed expert scale escape tail exceeds staging capacity");
        const uint8_t *input = static_cast<const uint8_t *>(scratch) + sizeof(*header);
        const uint8_t *const input_end = static_cast<const uint8_t *>(scratch) + entry.stored_bytes;
        std::array<const uint8_t *, 3> bodies{}, packed_scales{}, escape_values{};
        for (int projection = 0; projection < 3; ++projection) {
            require(size_t(input_end - input) >= kProjectionBodyBytes + kPackedScaleBytes +
                                                header->escapes[projection],
                    "truncated packed expert projection");
            bodies[projection] = input;
            packed_scales[projection] = input + kProjectionBodyBytes;
            escape_values[projection] = packed_scales[projection] + kPackedScaleBytes;
            input = escape_values[projection] + header->escapes[projection];
        }
        require(input == input_end, "packed expert record has trailing bytes");

        layout = {};
        for (int projection = 0; projection < 3; ++projection) {
            layout.packed_body[projection] = size_t(projection) * kProjectionBodyBytes;
            std::memcpy(window + layout.packed_body[projection], bodies[projection],
                        kProjectionBodyBytes);
            layout.body[projection] = size_t(projection) *
                                      (kProjectionBodyBytes + kProjectionScaleBytes);
            layout.scales[projection] = layout.body[projection] + kProjectionBodyBytes;
            globals[projection] = header->globals[projection];
        }
        size_t cursor = kBodyBytes;
        size_t device_cursor = 0;
        for (int projection = 0; projection < 3; ++projection) {
            const size_t blob = cursor;
            layout.packed_blob[projection] = blob;
            layout.packed_device[projection] = device_cursor;
            std::memcpy(window + cursor, packed_scales[projection], kPackedScaleBytes);
            cursor += kPackedScaleBytes;
            layout.packed_escapes[projection] = cursor - blob;
            std::memcpy(window + cursor, escape_values[projection], header->escapes[projection]);
            cursor += header->escapes[projection];
            layout.packed_codebook[projection] = cursor - blob;
            std::memcpy(window + cursor, header->codebooks[projection], 16);
            cursor += 16;
            cursor = (cursor + alignof(uint32_t) - 1) & ~(alignof(uint32_t) - 1);
            layout.packed_prefix[projection] = cursor - blob;
            std::array<uint32_t, kScalePrefixEntries> prefix{};
            build_escape_prefix(packed_scales[projection], header->escapes[projection],
                                prefix.data());
            std::memcpy(window + cursor, prefix.data(), sizeof(prefix));
            cursor += sizeof(prefix);
            layout.packed_blob_bytes[projection] = cursor - blob;
            device_cursor += layout.packed_blob_bytes[projection];
        }
        require(cursor <= kPayloadCapacity && device_cursor <= kPackedDeviceCapacity,
                "packed GPU scale transport exceeds staging capacity");
        layout.bytes = cursor;
        layout.packed_blob_span = device_cursor;
        layout.packed_scales = true;
        payload = window;
    }
    void stage_packed_cpu(Layout &layout, std::array<float, 3> &globals, uint8_t *&payload,
                          uint8_t *window, int layer, int expert, void *scratch) {
        const PackedExpertIndexEntry &entry = packed_entry(layer, expert);
        require(scratch && entry.padded_bytes <= packed_scratch_bytes_,
                "packed expert reader scratch is unavailable");
        pread_exact(packed_direct_fd_ >= 0 ? packed_direct_fd_ : packed_fd_,
                    entry.offset, scratch, entry.padded_bytes,
                    "read packed expert record");
        const auto expand_begin = std::chrono::steady_clock::now();
        const auto *header = static_cast<const PackedExpertRecordHeader *>(scratch);
        require(std::memcmp(header->magic, "XPR1", 4) == 0 &&
                header->layer == layer && header->expert == expert,
                "packed expert record key mismatch");
        const uint8_t *input = static_cast<const uint8_t *>(scratch) + sizeof(*header);
        const uint8_t *const input_end = static_cast<const uint8_t *>(scratch) + entry.stored_bytes;
        size_t cursor = 0;
        layout = {};
        for (int projection = 0; projection < 3; ++projection) {
            require(size_t(input_end - input) >= (4ull << 20) + (256ull << 10) +
                                                header->escapes[projection],
                    "truncated packed expert projection");
            layout.body[projection] = cursor;
            std::memcpy(window + cursor, input, 4ull << 20);
            input += 4ull << 20;
            cursor += 4ull << 20;
            layout.scales[projection] = cursor;
            const uint8_t *packed_scales = input;
            const uint8_t *escape_values = input + (256ull << 10);
            expand_scale_nibbles(packed_scales, escape_values,
                                 header->escapes[projection], header->codebooks[projection],
                                 window + cursor, kScaleBytes / 3);
            input = escape_values + header->escapes[projection];
            cursor += 512ull << 10;
            globals[projection] = header->globals[projection];
        }
        require(input == input_end && cursor == kBodyBytes + kScaleBytes,
                "packed expert record has trailing bytes");
        layout.bytes = cursor;
        payload = window;
        packed_expanded_bytes_.fetch_add(kBodyBytes + kScaleBytes, std::memory_order_relaxed);
        packed_expand_nanoseconds_.fetch_add(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - expand_begin).count()), std::memory_order_relaxed);
    }
    // v2 direct-read staging: header page into scratch, then ONE payload
    // pread straight into the window (bodies contiguous, then the three
    // pre-assembled scale regions at 4 KiB strides). Zero reader-thread CPU
    // work beyond bounds checks: no memcpy, no prefix build, no expand.
    void stage_packed_v2_gpu(Layout &layout, std::array<float, 3> &globals, uint8_t *&payload,
                             uint8_t *window, int layer, int expert, void *scratch) {
        const PackedExpertIndexEntry &entry = packed_entry(layer, expert);
        require(scratch && entry.stored_bytes >= kAlignment &&
                entry.stored_bytes - kAlignment <= kPayloadCapacity,
                "packed expert v2 record exceeds staging capacity");
        const int fd = packed_direct_fd_ >= 0 ? packed_direct_fd_ : packed_fd_;
        pread_exact(fd, entry.offset, scratch, kAlignment, "read packed v2 header");
        const auto *header = static_cast<const PackedExpertRecordHeader *>(scratch);
        require(std::memcmp(header->magic, "XPR1", 4) == 0 &&
                header->layer == layer && header->expert == expert,
                "packed expert record key mismatch");
        for (int projection = 0; projection < 3; ++projection)
            require(header->escapes[projection] <= kV2MaxEscapes,
                    "packed expert v2 escape tail exceeds staging capacity");
        layout = {};
        size_t cursor = kBodyBytes;
        size_t device_cursor = 0;
        size_t seg = kAlignment + kBodyBytes;  // record-relative region cursor
        for (int projection = 0; projection < 3; ++projection) {
            layout.packed_body[projection] = size_t(projection) * kProjectionBodyBytes;
            layout.body[projection] = size_t(projection) *
                                      (kProjectionBodyBytes + kProjectionScaleBytes);
            layout.scales[projection] = layout.body[projection] + kProjectionBodyBytes;
            globals[projection] = header->globals[projection];
            layout.packed_blob[projection] = cursor;
            layout.packed_device[projection] = device_cursor;
            layout.packed_escapes[projection] = kPackedScaleBytes;
            layout.packed_codebook[projection] =
                kPackedScaleBytes + header->escapes[projection];
            layout.packed_prefix[projection] =
                (layout.packed_codebook[projection] + 16 + alignof(uint32_t) - 1) &
                ~(alignof(uint32_t) - 1);
            layout.packed_blob_bytes[projection] =
                layout.packed_prefix[projection] + kScalePrefixEntries * sizeof(uint32_t);
            const size_t region_disk =
                (layout.packed_blob_bytes[projection] + kAlignment - 1) & ~(kAlignment - 1);
            seg += region_disk;
            cursor += region_disk;
            device_cursor += region_disk;
        }
        require(seg == entry.stored_bytes,
                "packed v2 record span does not match its index entry");
        require(cursor <= kPayloadCapacity && device_cursor <= kPackedDeviceCapacity,
                "packed v2 GPU scale transport exceeds staging capacity");
        layout.bytes = cursor;
        layout.packed_blob_span = device_cursor;
        layout.packed_scales = true;
        payload = window;
        pread_exact(fd, entry.offset + kAlignment, window, entry.stored_bytes - kAlignment,
                    "read packed v2 record");
        // Direct execution trusts this directory inside the hot kernel, so
        // validate every block once as its record enters the host tier. The
        // expanded path keeps the cheaper endpoint tripwire.
        for (int projection = 0; projection < 3; ++projection) {
            const uint8_t *packed =
                window + layout.packed_blob[projection];
            const uint32_t *prefix = reinterpret_cast<const uint32_t *>(
                window + layout.packed_blob[projection] + layout.packed_prefix[projection]);
            require(prefix[0] == 0u, "packed v2 prefix must start at zero");
            if (packed_direct_)
                for (size_t block = 0; block + 1 < kScalePrefixEntries; ++block) {
                    const uint32_t actual = count_escape_nibbles_256(packed + block * 256u);
                    require(prefix[block + 1] >= prefix[block] &&
                            prefix[block + 1] - prefix[block] == actual,
                            "packed v2 prefix block mismatch");
                }
            require(prefix[kScalePrefixEntries - 1] == header->escapes[projection],
                     "packed v2 prefix table mismatch");
        }
    }
    void stage_packed_v2_cpu(Layout &layout, std::array<float, 3> &globals, uint8_t *&payload,
                             uint8_t *window, int layer, int expert, void *scratch) {
        const PackedExpertIndexEntry &entry = packed_entry(layer, expert);
        require(scratch && entry.stored_bytes >= kAlignment &&
                entry.stored_bytes - kAlignment <= kPayloadCapacity,
                "packed expert v2 record exceeds staging capacity");
        const int fd = packed_direct_fd_ >= 0 ? packed_direct_fd_ : packed_fd_;
        pread_exact(fd, entry.offset, scratch, kAlignment, "read packed v2 header");
        const auto *header = static_cast<const PackedExpertRecordHeader *>(scratch);
        require(std::memcmp(header->magic, "XPR1", 4) == 0 &&
                header->layer == layer && header->expert == expert,
                "packed expert record key mismatch");
        for (int projection = 0; projection < 3; ++projection)
            require(header->escapes[projection] <= kV2MaxEscapes,
                    "packed expert v2 escape tail exceeds staging capacity");
        const auto expand_begin = std::chrono::steady_clock::now();
        layout = {};
        size_t cursor = 0;
        size_t seg = kAlignment;
        // The header page lives at scratch[0, 4 KiB) for the whole function;
        // region reads go after it (kV2ReaderScratchBytes budgeted for both).
        uint8_t *region = static_cast<uint8_t *>(scratch) + kAlignment;
        for (int projection = 0; projection < 3; ++projection) {
            layout.body[projection] = cursor;
            pread_exact(fd, entry.offset + seg, window + cursor, kProjectionBodyBytes,
                        "read packed v2 body");
            seg += kProjectionBodyBytes;
            cursor += kProjectionBodyBytes;
            layout.scales[projection] = cursor;
            const size_t prefix_at =
                (kPackedScaleBytes + header->escapes[projection] + 16 +
                 alignof(uint32_t) - 1) & ~(alignof(uint32_t) - 1);
            const size_t region_disk =
                (prefix_at + kScalePrefixEntries * sizeof(uint32_t) + kAlignment - 1) &
                ~(kAlignment - 1);
            require(region_disk + kAlignment <= packed_scratch_bytes_,
                    "packed expert v2 region exceeds reader scratch");
            pread_exact(fd, entry.offset + seg, region, region_disk,
                        "read packed v2 scale region");
            seg += region_disk;
            expand_scale_nibbles(region, region + kPackedScaleBytes,
                                 header->escapes[projection], region + kPackedScaleBytes +
                                     header->escapes[projection],
                                 window + cursor, kScaleBytes / 3);
            cursor += kProjectionScaleBytes;
            globals[projection] = header->globals[projection];
        }
        require(seg == entry.stored_bytes && cursor == kBodyBytes + kScaleBytes,
                "packed v2 record span does not match its index entry");
        layout.bytes = cursor;
        payload = window;
        packed_expanded_bytes_.fetch_add(kBodyBytes + kScaleBytes, std::memory_order_relaxed);
        packed_expand_nanoseconds_.fetch_add(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - expand_begin).count()), std::memory_order_relaxed);
    }
    void stage_packed(Layout &layout, std::array<float, 3> &globals, uint8_t *&payload,
                      uint8_t *window, int layer, int expert, void *scratch) {
        if (packed_version_ >= 2) {
            if (packed_gpu_scales_)
                stage_packed_v2_gpu(layout, globals, payload, window, layer, expert, scratch);
            else
                stage_packed_v2_cpu(layout, globals, payload, window, layer, expert, scratch);
            return;
        }
        if (packed_gpu_scales_)
            stage_packed_gpu(layout, globals, payload, window, layer, expert, scratch);
        else
            stage_packed_cpu(layout, globals, payload, window, layer, expert, scratch);
    }
    void enqueue_record_copy(WindowState &state, uint8_t *destination) {
        if (device_packed_scales_ && device_arena_) {
            require(packed_version_ >= 2 && state.layout.packed_scales &&
                    state.layout.bytes <= device_stride_,
                    "packed device slot received a non-v2 or oversized payload");
            check(cudaMemcpyAsync(destination, state.payload, state.layout.bytes,
                                  cudaMemcpyHostToDevice, copy_stream_),
                  "packed device-slot H2D");
            packed_h2d_bytes_.fetch_add(state.layout.bytes, std::memory_order_relaxed);
            packed_h2d_records_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (!state.layout.packed_scales) {
            check(cudaMemcpyAsync(destination, state.payload, state.layout.bytes,
                                  cudaMemcpyHostToDevice, copy_stream_),
                  "expert record H2D");
            return;
        }
        require(packed_scale_device_, "packed GPU scale scratch is unavailable");
        uint64_t transported = 0;
        constexpr size_t kSlotPitch = kProjectionBodyBytes + kProjectionScaleBytes;
        if (packed_merge_h2d_) {
            // v2 transport: the three 4 MiB bodies are contiguous in the
            // window and land at kSlotPitch strides in the destination slot,
            // so one pitched 2D copy replaces three; the three scale blobs
            // are contiguous on both sides by construction (device_cursor
            // accumulation), so one linear copy replaces three; and the
            // per-projection expansions fuse into a single launch. Six
            // memcpy calls plus three launches become two plus one.
            check(cudaMemcpy2DAsync(destination + state.layout.body[0], kSlotPitch,
                                    state.payload + state.layout.packed_body[0],
                                    kProjectionBodyBytes, kProjectionBodyBytes, 3,
                                    cudaMemcpyHostToDevice, copy_stream_),
                  "packed expert bodies H2D (2D)");
            transported += kBodyBytes;
            check(cudaMemcpyAsync(packed_scale_device_,
                                  state.payload + state.layout.packed_blob[0],
                                  state.layout.packed_blob_span,
                                  cudaMemcpyHostToDevice, copy_stream_),
                  "packed expert scales H2D (merged)");
            transported += state.layout.packed_blob_span;
            std::array<size_t, 3> packed{}, escapes{}, codebooks{}, prefixes{};
            for (int projection = 0; projection < 3; ++projection) {
                packed[projection] = state.layout.packed_device[projection];
                escapes[projection] = state.layout.packed_device[projection] +
                                      state.layout.packed_escapes[projection];
                codebooks[projection] = state.layout.packed_device[projection] +
                                        state.layout.packed_codebook[projection];
                prefixes[projection] = state.layout.packed_device[projection] +
                                       state.layout.packed_prefix[projection];
            }
            check(packed_kernel_v2_
                      ? insignia::glm53::expand_nvfp4_scale_nibbles3_v2(
                            packed_scale_device_, packed.data(), escapes.data(),
                            codebooks.data(), prefixes.data(), destination,
                            kSlotPitch, kProjectionBodyBytes, kProjectionScaleBytes,
                            copy_stream_)
                      : insignia::glm53::expand_nvfp4_scale_nibbles3(
                            packed_scale_device_, packed.data(), escapes.data(),
                            codebooks.data(), prefixes.data(), destination,
                            kSlotPitch, kProjectionBodyBytes, kProjectionScaleBytes,
                            copy_stream_),
                  "expand packed expert scales on GPU (fused)");
        } else {
            for (int projection = 0; projection < 3; ++projection) {
                check(cudaMemcpyAsync(destination + state.layout.body[projection],
                                      state.payload + state.layout.packed_body[projection],
                                      kProjectionBodyBytes, cudaMemcpyHostToDevice, copy_stream_),
                      "packed expert body H2D");
                transported += kProjectionBodyBytes;
                uint8_t *blob = packed_scale_device_ + state.layout.packed_device[projection];
                check(cudaMemcpyAsync(blob,
                                      state.payload + state.layout.packed_blob[projection],
                                      state.layout.packed_blob_bytes[projection],
                                      cudaMemcpyHostToDevice, copy_stream_),
                      "packed expert scales H2D");
                transported += state.layout.packed_blob_bytes[projection];
            }
            for (int projection = 0; projection < 3; ++projection) {
                const uint8_t *blob =
                    packed_scale_device_ + state.layout.packed_device[projection];
                check(packed_kernel_v2_
                          ? insignia::glm53::expand_nvfp4_scale_nibbles_v2(
                                blob,
                                blob + state.layout.packed_escapes[projection],
                                blob + state.layout.packed_codebook[projection],
                                reinterpret_cast<const uint32_t *>(
                                    blob + state.layout.packed_prefix[projection]),
                                destination + state.layout.scales[projection],
                                kProjectionScaleBytes, copy_stream_)
                          : insignia::glm53::expand_nvfp4_scale_nibbles(
                                blob,
                                blob + state.layout.packed_escapes[projection],
                                blob + state.layout.packed_codebook[projection],
                                reinterpret_cast<const uint32_t *>(
                                    blob + state.layout.packed_prefix[projection]),
                                destination + state.layout.scales[projection],
                                kProjectionScaleBytes, copy_stream_),
                      "expand packed expert scales on GPU");
            }
        }
        packed_h2d_bytes_.fetch_add(transported, std::memory_order_relaxed);
        packed_h2d_records_.fetch_add(1, std::memory_order_relaxed);
    }
    void expand_active_packed_slot_scales() {
        if (!device_packed_scales_ || active_device_slot_ < 0) return;
        require(active_device_ && packed_scale_device_ && active_.packed_scales,
                "packed device slot has no execution-scale scratch");
        std::array<size_t, 3> packed{}, escapes{}, codebooks{}, prefixes{};
        for (int projection = 0; projection < 3; ++projection) {
            packed[projection] = active_.packed_blob[projection];
            escapes[projection] = active_.packed_blob[projection] +
                                  active_.packed_escapes[projection];
            codebooks[projection] = active_.packed_blob[projection] +
                                    active_.packed_codebook[projection];
            prefixes[projection] = active_.packed_blob[projection] +
                                   active_.packed_prefix[projection];
        }
        check(insignia::glm53::expand_nvfp4_scale_nibbles3_v2(
                  active_device_, packed.data(), escapes.data(), codebooks.data(),
                  prefixes.data(), packed_scale_device_, kProjectionScaleBytes,
                  0, kProjectionScaleBytes, nullptr),
              "expand packed device-slot scales into execution scratch");
    }
    void ensure_device_scratch() {
        if (!device_)
            check(cudaMalloc(&device_, record_capacity_), "cudaMalloc expert record");
    }

    // Sizes and allocates the VRAM tier at first expert use, when every
    // startup allocation (dense FP8 residency, drafter, state buffers) has
    // already claimed its VRAM and cudaMemGetInfo reflects steady state.
    void ensure_device_arena() {
        if (device_arena_ready_) return;
        device_arena_ready_ = true;
        size_t budget = 0;
        if (vram_budget_mb_ >= 0) {
            budget = size_t(vram_budget_mb_) << 20;
        } else {
            size_t free_bytes = 0, total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
                cudaGetLastError();
                return;  // stay on the legacy single-scratch path
            }
            // Leave headroom for context growth and activation spikes.
            budget = free_bytes > (768ull << 20) ? free_bytes - (768ull << 20) : 0;
        }
        const size_t expanded_stride =
            (record_capacity_ + kAlignment - 1) & ~(kAlignment - 1);
        const size_t stride = device_packed_scales_ ? packed_device_stride_ : expanded_stride;
        require(stride && (!device_packed_scales_ || stride < expanded_stride),
                "packed device-slot stride is unavailable or not smaller");
        size_t attempt = budget / stride;
        while (attempt > 1) {
            if (cudaMalloc(&device_arena_, attempt * stride) == cudaSuccess) break;
            cudaGetLastError();  // clear the sticky error and retry smaller
            attempt /= 2;
        }
        const size_t segment_count = compact_device_segments_ ? 42u : 46u;
        if (attempt < 2 * segment_count) {
            // Every fixed layer segment needs one active slot plus one legal
            // recycle victim. A smaller global arena leaves some one-slot
            // segments and would fail on their second miss; use the exact
            // single-scratch fallback instead.
            if (device_arena_) {
                cudaFree(device_arena_);
                device_arena_ = nullptr;
            }
            return;
        }
        device_stride_ = stride;
        device_slot_count_ = int(attempt);
        if (device_packed_scales_)
            std::printf("expert VRAM packed slots: stride=%zu, slots=%d, scratch=%zu bytes\n",
                        device_stride_, device_slot_count_, kScaleBytes);
        device_slot_keys_.assign(attempt, kNoKey);
        device_slot_stamps_.assign(attempt, 0);
        device_slot_pinned_.assign(attempt, 0);
        device_slot_layouts_.assign(attempt, Layout{});
        device_slot_globals_.assign(attempt, std::array<float, 3>{});
        device_slot_reads_.resize(attempt, nullptr);
        for (cudaEvent_t &event : device_slot_reads_)
            check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                  "cudaEventCreate expert slot fence");
        // Slots are individually owned now, so copies for different records
        // no longer alias: swap the legacy-synchronizing copy stream for a
        // real async one and let ordering live entirely in the events.
        cudaStreamSynchronize(copy_stream_);
        cudaStreamDestroy(copy_stream_);
        check(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking),
              "expert copy stream (async)");
    }
    // One fixed slot segment per sparse layer: a global LRU thrashes at
    // round scale (~1,300 distinct records/round >> slots), but the slice
    // for layer L retains its own most recent records, so the measured ~27%
    // adjacent-token routing overlap converts directly into PCIe-free hits.
    int take_device_slot(int layer, int upload_slot) {
        const int segments = std::max(
            1, std::min(compact_device_segments_ ? 42 : 46, device_slot_count_));
        const size_t segment = compact_device_segments_
                                   ? size_t(layer - 3) % size_t(segments)
                                   : size_t(layer) % size_t(segments);
        const int begin = segment * device_slot_count_ / segments;
        const int end = segment + 1 == size_t(segments)
                            ? device_slot_count_
                            : (segment + 1) * device_slot_count_ / segments;
        int victim = -1;
        int cold_victim = -1;
        int future_victim = -1;
        int farthest_future = -1;
        for (int index = begin; index < end; ++index) {
            if (index == active_device_slot_) continue;  // fence not recorded yet
            if (device_slot_pinned_[size_t(index)]) continue;
            if (device_slot_keys_[size_t(index)] == kNoKey) return index;
            if (!batch_aware_device_victim_) {
                if (victim < 0 ||
                    device_slot_stamps_[size_t(index)] < device_slot_stamps_[size_t(victim)])
                    victim = index;
                continue;
            }
            int future = -1;
            for (int slot = upload_slot + 1; slot < batch_count_; ++slot)
                if (device_slot_keys_[size_t(index)] ==
                    route_key(layer, batch_experts_[size_t(slot)])) {
                    future = slot;
                    break;
                }
            if (future < 0) {
                if (cold_victim < 0 ||
                    device_slot_stamps_[size_t(index)] <
                        device_slot_stamps_[size_t(cold_victim)])
                    cold_victim = index;
            } else if (future > farthest_future) {
                farthest_future = future;
                future_victim = index;
            }
        }
        if (batch_aware_device_victim_)
            victim = cold_victim >= 0 ? cold_victim : future_victim;
        require(victim >= 0, "expert VRAM segment needs a recyclable slot");
        device_index_.erase(device_slot_keys_[size_t(victim)]);
        device_slot_keys_[size_t(victim)] = kNoKey;
        return victim;
    }
    void start_read(int window, uint32_t key, int layer, int expert, bool demand) {
        WindowState &state = windows_[size_t(window)];
        state.key = key;
        state.layer = layer;
        state.expert = expert;
        state.demand = demand;
        state.done = false;
        state.claimed = false;
        state.copy_issued = false;
        state.pinned = false;
        state.stamp = 0;
        state.source_bytes = source_bytes(layer, expert);
        state.hits = 0;
        state.error = nullptr;
        state.payload = nullptr;
        state.layout = Layout{};
        state.globals = {};
        state.l2_shard = -1;
        state.round_epoch = round_epoch_;
        flight_index_.emplace(key, window);
        seg_push_back(window, 1);
        if (demand) ++records_read_;
        submit_window(window, demand);
    }
    // Relabeling an adopted prefetch is insufficient: if it has not started,
    // it still sits in the speculative FIFO behind unrelated hints. Move that
    // exact window to the demand queue so the advertised priority is real.
    void promote_read(int window) {
        bool moved = false;
        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            windows_[size_t(window)].demand = true;
            for (int drive = 0; drive < 2 && !moved; ++drive) {
                auto &queue = prefetch_queue_[drive];
                const auto found = std::find(queue.begin(), queue.end(), window);
                if (found == queue.end()) continue;  // already being read
                queue.erase(found);
                demand_queue_[drive].push_back(window);
                moved = true;
            }
        }
        if (moved) pool_cv_.notify_all();
    }
    // Blocks until the window's read completed; used by the demand path.
    void wait_window(int window) {
        std::unique_lock<std::mutex> lock(pool_mutex_);
        pool_done_.wait(lock, [this, window] { return windows_[size_t(window)].done; });
    }
    // Acquire-ordered peek so the fast path (already complete) still forms a
    // proper happens-before with the reader thread's payload writes.
    bool window_done(int window) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        return windows_[size_t(window)].done;
    }
    void wait_and_consume_error(int window) {
        wait_window(window);
        if (windows_[size_t(window)].error) std::rethrow_exception(windows_[size_t(window)].error);
    }
    void record_batch_read_end(std::chrono::steady_clock::time_point end) {
        batch_read_ends_.push_back(end);
        if (batch_read_ends_.size() < size_t(batch_demand_count_)) return;
        auto latest = batch_read_begin_;
        for (const auto &stamp : batch_read_ends_) latest = std::max(latest, stamp);
        io_seconds_ += std::chrono::duration<double>(latest - batch_read_begin_).count();
    }
    // Intrusive (segmented) LRU over evictable windows. Two segments share
    // the link arrays; WindowState::segment records membership:
    //   1 = probationary: new reads (never-admitted) arrive at the back
    //       (oldest candidates), admission moves to the front.
    //   2 = protected:    probationary hits promote here (soft cap 50%);
    //       protected hits move to the front; overflow demotes the tail
    //       back to probationary-front.
    // Eviction/pass-through unlinks; pinned windows leave the lists entirely
    // so the tail walk never crosses a pin prefix. With INSIGNIA_GLM53_TIER_O1
    // only (no SLRU), the protected segment stays empty and the probationary
    // order mirrors the old min-stamp scan exactly.
    int &seg_head(int segment) { return segment == 2 ? pt_head_ : pb_head_; }
    int &seg_tail(int segment) { return segment == 2 ? pt_tail_ : pb_tail_; }
    void lru_unlink(int window) {
        WindowState &state = windows_[size_t(window)];
        const int segment = state.segment;
        if (segment == 0) return;
        const int p = lru_prev_[size_t(window)], n = lru_next_[size_t(window)];
        if (p >= 0) lru_next_[size_t(p)] = n;
        else if (seg_head(segment) == window) seg_head(segment) = n;
        if (n >= 0) lru_prev_[size_t(n)] = p;
        else if (seg_tail(segment) == window) seg_tail(segment) = p;
        lru_prev_[size_t(window)] = lru_next_[size_t(window)] = -1;
        state.segment = 0;
        if (segment == 2) --protected_count_;
    }
    void seg_push_back(int window, int segment) {
        WindowState &state = windows_[size_t(window)];
        lru_prev_[size_t(window)] = seg_tail(segment);
        lru_next_[size_t(window)] = -1;
        if (seg_tail(segment) >= 0) lru_next_[size_t(seg_tail(segment))] = window;
        else seg_head(segment) = window;
        seg_tail(segment) = window;
        state.segment = uint8_t(segment);
        if (segment == 2) ++protected_count_;
    }
    void seg_move_front(int window, int segment) {
        if (seg_head(segment) == window) return;
        lru_unlink(window);
        WindowState &state = windows_[size_t(window)];
        lru_prev_[size_t(window)] = -1;
        lru_next_[size_t(window)] = seg_head(segment);
        if (seg_head(segment) >= 0) lru_prev_[size_t(seg_head(segment))] = window;
        else seg_tail(segment) = window;
        seg_head(segment) = window;
        state.segment = uint8_t(segment);
        if (segment == 2) ++protected_count_;
    }
    // SLRU soft cap: demote the protected tail to the probationary front
    // until the protected segment holds at most half the tier.
    void slru_enqueue_capacity() {
        while (protected_count_ > window_count_ / 2) {
            const int window = pt_tail_;
            require(window >= 0, "protected expert segment lost its tail");
            lru_unlink(window);
            seg_move_front(window, 1);
        }
    }
    // Host-tier hit: probationary members promote to the protected front;
    // protected members refresh to their front.
    void slru_hit(int window) {
        WindowState &state = windows_[size_t(window)];
        if (state.segment == 2) {
            seg_move_front(window, 2);
            return;
        }
        if (state.segment == 1) {
            lru_unlink(window);
            seg_move_front(window, 2);
            slru_enqueue_capacity();
        }
    }
    // Acceptance-prefix demote (insert-then-demote): a record staged for the
    // verify union that only served rejected rows gets pushed to the cold
    // (tail) end of probationary instead of riding the protected segment.
    void slru_demote_cold(int window) {
        lru_unlink(window);         // leaves whichever segment it rode
        seg_push_back(window, 1);   // re-enter at the cold end of probationary
    }
    // Returns windows whose async H2D finished to the free list.
    void reap_released() {
        size_t out = 0;
        for (size_t index = 0; index < releasing_.size(); ++index) {
            const int window = releasing_[index];
            WindowState &state = windows_[size_t(window)];
            if (state.key != kNoKey || !state.releasing) continue;  // already freed
            if (cudaEventQuery(state.copy_done) != cudaSuccess) {
                releasing_[out++] = window;
                continue;
            }
            state.releasing = false;
            state.copy_issued = false;
            free_windows_.push_back(window);
        }
        releasing_.resize(out);
    }
    int take_window() {
        reap_released();
        if (!free_windows_.empty()) {
            const int window = free_windows_.back();
            free_windows_.pop_back();
            return window;
        }
        if (tier_o1_ || tier_slru_) {
            // O(1) victim: first eligible window from the probationary tail,
            // then the protected tail (SLRU). The probationary order mirrors
            // the admission-stamp recency the legacy scan below used; the
            // protected fallback only exists when SLRU promoted records.
            const auto evict = [&](int walk) {
                WindowState &state = windows_[size_t(walk)];
                if (state.copy_issued) {
                    check(cudaEventSynchronize(state.copy_done), "drain evicted expert copy");
                    state.copy_issued = false;
                }
                if (!state.demand) ++prefetch_wasted_;
                release_window(walk);
            };
            for (int walk = pb_tail_; walk >= 0; walk = lru_prev_[size_t(walk)]) {
                const WindowState &state = windows_[size_t(walk)];
                if (state.key == kNoKey || !state.done || state.claimed || state.releasing ||
                    state.pinned)
                    continue;
                evict(walk);
                return take_window();
            }
            for (int walk = pt_tail_; walk >= 0; walk = lru_prev_[size_t(walk)]) {
                const WindowState &state = windows_[size_t(walk)];
                if (state.key == kNoKey || !state.done || state.claimed || state.releasing ||
                    state.pinned)
                    continue;
                evict(walk);
                return take_window();
            }
        }
        // Evict the least-recently-used completed, unclaimed record. A batch
        // claims at most 8 windows and releases them at upload, so with a
        // tier larger than one token's 336 records a victim always exists.
        // (Measured: frequency-based eviction churns newcomers and halves
        // the hit rate on this near-uniform routing; plain LRU wins.)
        int victim = -1;
        for (int window = 0; window < window_count_; ++window) {
            const WindowState &state = windows_[size_t(window)];
            if (state.key == kNoKey || !state.done || state.claimed || state.releasing ||
                state.pinned)
                continue;
            if (victim < 0 || state.stamp < windows_[size_t(victim)].stamp) victim = window;
        }
        if (victim < 0) {
            // Everything is claimed or draining: wait out the oldest copy.
            for (int window = 0; window < window_count_; ++window) {
                WindowState &state = windows_[size_t(window)];
                if (!state.releasing) continue;
                check(cudaEventSynchronize(state.copy_done), "drain released expert copy");
                state.releasing = false;
                state.copy_issued = false;
                free_windows_.push_back(window);
                const int taken = free_windows_.back();
                free_windows_.pop_back();
                return taken;
            }
        }
        require(victim >= 0, "no expert window available for a demand read");
        WindowState &victim_state = windows_[size_t(victim)];
        // A pending async H2D may still be draining this window's bytes.
        if (victim_state.copy_issued) {
            check(cudaEventSynchronize(victim_state.copy_done), "drain evicted expert copy");
            victim_state.copy_issued = false;
        }
        if (!victim_state.demand) ++prefetch_wasted_;
        release_window(victim);
        return take_window();
    }
    void release_window(int window) {
        WindowState &state = windows_[size_t(window)];
        flight_index_.erase(state.key);
        state.key = kNoKey;
        state.demand = false;
        lru_unlink(window);
        free_windows_.push_back(window);
    }
    // Drive of a routing target: 0 = main store, 1 = ALT_SHARD_DIR (second
    // physical drive). Cached per (layer,expert); the lookup walks 9 index
    // entries once per distinct key.
    int drive_of(uint32_t key, int layer, int expert) {
        if (q3_experts_) return 0;
        if (packed_fd_ >= 0) return 0;
        if (stripe_failed_.load(std::memory_order_relaxed)) return 0;
        std::lock_guard<std::mutex> lock(pool_mutex_);
        const auto found = drive_cache_.find(key);
        if (found != drive_cache_.end()) return found->second;
        ShardedIndex &routing_model = stripe_model_ ? *stripe_model_ : model_;
        const ExpertLocations tensors = locate_expert(routing_model, layer, expert);
        const int drive = routing_model.shard_is_alt(tensors.body[0]->shard) ? 1 : 0;
        drive_cache_.emplace(key, drive);
        return drive;
    }
    void start_pool() {
        // Per-drive pools: a shared FIFO gates every batch on the slower
        // drive, so each physical drive gets its own demand/prefetch queues.
        // The virtio-blk sweet spot per drive measured with fio is 4-8
        // outstanding multi-MiB reads on the main disk (~5.9 GB/s) and 2-4 on
        // the DRAM-less second disk (~2.6 GB/s).
        int readers = 4, readers_e = 2;
        if (const char *workers = std::getenv("INSIGNIA_GLM53_READERS"))
            readers = std::max(1, std::atoi(workers));
        if (const char *workers = std::getenv("INSIGNIA_GLM53_READERS_E"))
            readers_e = std::max(1, std::atoi(workers));
        stop_ = false;
        const int drive_count =
            (stripe_model_ ? stripe_model_->alt_shard_count() : model_.alt_shard_count()) ? 2 : 1;
        for (int drive = 0; drive < drive_count; ++drive) {
            const int workers = std::min<int>(drive ? readers_e : readers, window_count_);
            for (int index = 0; index < workers; ++index)
                pool_.emplace_back([this, drive] {
                    void *packed_scratch = nullptr;
                    if (drive == 0 && packed_fd_ >= 0 &&
                        ::posix_memalign(&packed_scratch, kAlignment, packed_scratch_bytes_))
                        packed_scratch = nullptr;
                    for (;;) {
                        int window = -1;
                        {
                            std::unique_lock<std::mutex> lock(pool_mutex_);
                            pool_cv_.wait(lock, [this, drive] {
                                return stop_ || !demand_queue_[drive].empty() ||
                                       !prefetch_queue_[drive].empty();
                            });
                            if (stop_) {
                                std::free(packed_scratch);
                                return;
                            }
                            // Demand records always jump ahead of speculative ones:
                            // a prefetch that delays the current layer's reads is a
                            // net loss no matter how good its prediction is.
                            if (!demand_queue_[drive].empty()) {
                                window = demand_queue_[drive].front();
                                demand_queue_[drive].pop_front();
                            } else {
                                window = prefetch_queue_[drive].front();
                                prefetch_queue_[drive].pop_front();
                            }
                        }
                        read_window(window, packed_scratch);
                        {
                            std::lock_guard<std::mutex> lock(pool_mutex_);
                            windows_[size_t(window)].done = true;
                            windows_[size_t(window)].end = std::chrono::steady_clock::now();
                        }
                        pool_done_.notify_all();
                    }
                });
        }
    }
    void stop_pool() {
        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            stop_ = true;
        }
        pool_cv_.notify_all();
        for (std::thread &worker : pool_) if (worker.joinable()) worker.join();
        pool_.clear();
    }
    void submit_window(int window, bool demand) {
        WindowState &state = windows_[size_t(window)];
        const int drive = drive_of(state.key, state.layer, state.expert);
        state.drive = uint8_t(drive);
        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            (demand ? demand_queue_[drive] : prefetch_queue_[drive]).push_back(window);
        }
        // notify_one can land on a wrong-drive worker whose predicate is
        // false; with partitioned queues it must be notify_all or the last
        // record in a drive's queue waits forever.
        pool_cv_.notify_all();
    }
    void stage_q3(Layout &layout, std::array<float, 3> &globals,
                  uint8_t *&payload, uint8_t *window, int layer, int expert) {
        require(q3_experts_ && expert >= 0 && expert < int(model_.experts()),
                "invalid native Q3 expert request");
        const Q3ExpertLocations tensors = locate_q3_expert(model_, layer);
        layout = {};
        globals = {};
        size_t cursor = 0;
        for (int projection = 0; projection < 3; ++projection) {
            const TensorLocation &tensor = *tensors.body[size_t(projection)];
            require(tensor.shape.size() == 3 && tensor.shape[0] == model_.experts() &&
                        tensor.bytes % model_.experts() == 0 &&
                        (tensor.type == TensorType::iq3_xxs ||
                         tensor.type == TensorType::iq4_xs ||
                         tensor.type == TensorType::q6_k),
                    "malformed native Q3 expert component");
            const uint64_t bytes = tensor.bytes / model_.experts();
            const uint64_t offset = tensor.offset + uint64_t(expert) * bytes;
            const uint64_t aligned_offset = offset & ~(uint64_t(kAlignment) - 1);
            const size_t delta = size_t(offset - aligned_offset);
            const size_t request =
                (delta + size_t(bytes) + kAlignment - 1) & ~(kAlignment - 1);
            require(cursor + request <= record_capacity_,
                    "native Q3 expert exceeds staging size class");
            const uint64_t actual_delta = model_.read_span_direct_window(
                tensor.shard, offset, bytes, window + cursor, request);
            require(actual_delta == delta, "native Q3 direct-read delta changed");
            layout.body[size_t(projection)] = cursor + delta;
            layout.types[size_t(projection)] = tensor.type;
            cursor += request;
        }
        layout.bytes = cursor;
        payload = window;
    }
    void read_window(int window, void *packed_scratch) {
        WindowState &state = windows_[size_t(window)];
        try {
            if (q3_experts_) {
                stage_q3(state.layout, state.globals, state.payload,
                         host_ + size_t(window) * window_bytes_,
                         state.layer, state.expert);
                ++drive_records_[0];
                drive_bytes_[0].fetch_add(state.source_bytes,
                                          std::memory_order_relaxed);
            } else if (packed_fd_ >= 0) {
                stage_packed(state.layout, state.globals, state.payload,
                             host_ + size_t(window) * window_bytes_,
                             state.layer, state.expert, packed_scratch);
            } else {
                auto stage_from = [&](ShardedIndex &source) {
                    const ExpertLocations tensors =
                        locate_expert(source, state.layer, state.expert);
                    stage(source, state.layout, state.globals, state.payload, tensors,
                          host_ + size_t(window) * window_bytes_, state);
                };
                if (state.drive == 1 && stripe_model_ &&
                    stripe_failed_.load(std::memory_order_relaxed)) {
                    if (stripe_required_)
                        throw std::runtime_error("required expert stripe was disabled after an I/O error");
                    state.drive = 0;
                    ++stripe_fallbacks_;
                    stage_from(model_);
                } else if (state.drive == 1 && stripe_model_) {
                    try {
                        stage_from(*stripe_model_);
                    } catch (...) {
                        if (stripe_required_) throw;
                        stripe_failed_.store(true, std::memory_order_relaxed);
                        ++stripe_fallbacks_;
                        state.drive = 0;
                        state.l2_shard = -1;
                        state.l2_offset = state.l2_bytes = 0;
                        stage_from(model_);
                    }
                } else {
                    stage_from(model_);
                }
                ++drive_records_[state.drive];
                drive_bytes_[state.drive].fetch_add(state.layout.bytes,
                                                     std::memory_order_relaxed);
            }
            if (!state.demand) prefetch_bytes_ += state.source_bytes;
        } catch (...) {
            state.error = std::current_exception();
        }
    }
    void stage(ShardedIndex &source, Layout &layout,
               std::array<float, 3> &globals, uint8_t *&payload,
               const ExpertLocations &tensors, uint8_t *window, WindowState &wstate) {
        layout = {};
        const TensorLocation &first = *tensors.body[0];
        const TensorLocation &last = *tensors.globals[2];
        bool packed = true;
        for (int projection = 0; projection < 3; ++projection) {
            const TensorLocation &body = *tensors.body[projection];
            const TensorLocation &scales = *tensors.scales[projection];
            const TensorLocation &global = *tensors.globals[projection];
            require(body.bytes == 4ull << 20 && scales.bytes == 512ull << 10 && global.bytes == 4 &&
                    body.shard == scales.shard && scales.shard == global.shard &&
                    body.offset + body.bytes == scales.offset && scales.offset + scales.bytes == global.offset,
                    "expert record is not compact weight/scale/global storage");
            packed &= body.shard == first.shard;
            if (projection < 2) {
                const TensorLocation &next = *tensors.body[projection + 1];
                packed &= global.offset + global.bytes <= next.offset &&
                    next.offset - (global.offset + global.bytes) <= 64;
            }
        }
        if (packed && last.offset >= first.offset &&
            last.offset + last.bytes - first.offset <= kPayloadCapacity) {
            layout.bytes = size_t(last.offset + last.bytes - first.offset);
            uint64_t delta = 0;
            if (l2_mode_) {
                // Page-cache L2: read through cache; the upload path later
                // evicts the pages iff the record is admitted to the pinned
                // tier, so transients (verify-union residue) stay RAM-served.
                delta = source.read_span_cached_window(
                    first.shard, first.offset, layout.bytes, window, kWindowBytes);
                wstate.l2_shard = first.shard;
                wstate.l2_offset = first.offset;
                wstate.l2_bytes = layout.bytes;
            } else {
                delta = source.read_span_direct_window(
                    first.shard, first.offset, layout.bytes, window, kWindowBytes);
            }
            payload = window + delta;
            for (int projection = 0; projection < 3; ++projection) {
                layout.body[projection] = size_t(tensors.body[projection]->offset - first.offset);
                layout.scales[projection] = size_t(tensors.scales[projection]->offset - first.offset);
                const size_t offset = size_t(tensors.globals[projection]->offset - first.offset);
                std::memcpy(globals.data() + projection, payload + offset, sizeof(float));
            }
            return;
        }

        // A few compact-store records straddle the original tensor ordering.
        // Pack just those three projection records into the same device ABI.
        payload = window;
        size_t cursor = 0;
        for (int projection = 0; projection < 3; ++projection) {
            const TensorLocation &body = *tensors.body[projection];
            const TensorLocation &scales = *tensors.scales[projection];
            layout.body[projection] = cursor;
            layout.scales[projection] = cursor + size_t(body.bytes);
            source.read_span_direct(body.shard, body.offset, body.bytes + scales.bytes,
                                    window + cursor);
            cursor += size_t(body.bytes + scales.bytes);
            source.read(*tensors.globals[projection], globals.data() + projection);
        }
        layout.bytes = cursor;
    }

    ShardedIndex &model_;
    ShardedIndex *stripe_model_ = nullptr;
    bool q3_experts_ = false;
    size_t window_bytes_ = kWindowBytes;
    size_t record_capacity_ = kPayloadCapacity;
    int packed_fd_ = -1, packed_direct_fd_ = -1;
    int packed_version_ = 0;
    std::vector<PackedExpertIndexEntry> packed_entries_;
    size_t packed_scratch_bytes_ = 0, packed_device_stride_ = 0;
    std::atomic<uint64_t> packed_expanded_bytes_{0}, packed_expand_nanoseconds_{0};
    std::atomic<uint64_t> packed_h2d_bytes_{0}, packed_h2d_records_{0};
    uint8_t *host_raw_ = nullptr, *host_ = nullptr, *device_ = nullptr;
    uint8_t *packed_scale_device_ = nullptr;
    uint8_t *active_device_ = nullptr;
    Layout active_{};
    std::array<float, 3> active_globals_{};
    std::vector<WindowState> windows_;
    int window_count_ = 0;
    std::unordered_map<uint32_t, int> flight_index_;
    std::vector<int> free_windows_;
    std::vector<std::thread> pool_;
    std::mutex pool_mutex_;
    std::condition_variable pool_cv_, pool_done_;
    std::deque<int> demand_queue_[2], prefetch_queue_[2];
    std::unordered_map<uint32_t, int> drive_cache_;
    std::array<std::atomic<uint64_t>, 2> drive_records_{};
    std::array<std::atomic<uint64_t>, 2> drive_bytes_{};
    std::atomic<uint64_t> stripe_fallbacks_{0};
    std::atomic<bool> stripe_failed_{false};
    cudaStream_t copy_stream_ = nullptr;
    bool stop_ = false, packed_gpu_scales_ = false, stripe_required_ = false;
    bool packed_merge_h2d_ = false, packed_kernel_v2_ = false;
    bool device_packed_scales_ = false, packed_direct_ = false, packed_tablefree_ = true;
    std::array<int, 8> batch_experts_{};
    std::array<bool, 8> batch_cached_{};
    std::array<bool, 8> batch_admit_{};
    std::array<bool, 8> batch_populate_{};
    std::array<int, 8> batch_window_{};
    std::deque<uint32_t> sight_order_;
    std::unordered_map<uint32_t, unsigned> sight_count_;
    std::vector<int> releasing_;
    int admit_threshold_ = 1;
    bool l2_mode_ = false;
    std::vector<std::chrono::steady_clock::time_point> batch_read_ends_;
    std::chrono::steady_clock::time_point batch_read_begin_{};
    int batch_layer_ = -1, batch_count_ = 0, batch_demand_count_ = 0;
    uint64_t stamp_ = 0;
    bool overlap_reads_ = true, batch_io_recorded_ = true;
    uint64_t cache_hits_ = 0, cache_lookups_ = 0;
    uint64_t prefetch_started_ = 0, prefetch_useful_ = 0, prefetch_wasted_ = 0, prefetch_bytes_ = 0;
    double io_seconds_ = 0.0;
    uint64_t io_bytes_ = 0;
    uint64_t records_read_ = 0;
    double read_wait_seconds_ = 0.0;
    // VRAM tier: an arena of per-record device slots acts as an LRU above the
    // pinned host windows. A device hit skips both the NVMe read and the PCIe
    // H2D entirely; a miss streams into its own slot, so copies for different
    // records pipeline instead of serializing through one shared scratch.
    uint8_t *device_arena_ = nullptr;
    size_t device_stride_ = 0;
    int device_slot_count_ = 0;
    int active_device_slot_ = -1;
    bool compact_device_segments_ = false;
    bool batch_aware_device_victim_ = false;
    bool device_arena_ready_ = false;
    int vram_budget_mb_ = -1;  // -1 = size from free VRAM at first use
    uint64_t device_stamp_ = 0;
    std::vector<uint32_t> device_slot_keys_;
    std::vector<uint64_t> device_slot_stamps_;
    std::vector<uint8_t> device_slot_pinned_;
    std::vector<Layout> device_slot_layouts_;
    std::vector<std::array<float, 3>> device_slot_globals_;
    std::unordered_set<uint32_t> pinned_device_keys_;
    std::vector<cudaEvent_t> device_slot_reads_;
    std::unordered_map<uint32_t, int> device_index_;
    uint64_t device_hits_ = 0, device_lookups_ = 0;
    // F3 device-consult state (INSIGNIA_GLM53_F3).
    bool f3_device_consult_ = false;
    std::array<bool, 8> batch_device_{};
    uint64_t f3_rescued_ = 0;
    // O(1) intrusive LRU state (INSIGNIA_GLM53_TIER_O1) plus the segmented
    // variant (INSIGNIA_GLM53_TIER_SLRU): probationary + protected lists,
    // hit-promotion, soft 50% protected cap, verify-rejection demotions.
    bool tier_o1_ = false;
    bool tier_slru_ = false;
    std::vector<int> lru_prev_, lru_next_;
    int pb_head_ = -1, pb_tail_ = -1, pt_head_ = -1, pt_tail_ = -1;
    int protected_count_ = 0;
    uint64_t demoted_cold_ = 0;
    uint32_t round_epoch_ = 0;
};

class Q8Stager {
public:
    static constexpr size_t kWeightCapacity = 128ull << 20;
    static constexpr size_t kScaleCapacity = 4ull << 20;
    struct ResidentView {
        const uint8_t *weights = nullptr;
        const uint16_t *scales = nullptr;
    };

    explicit Q8Stager(Q8Index &index) : index_(index) {
        check(cudaHostAlloc(&host_weights_, kWeightCapacity, cudaHostAllocDefault),
              "cudaHostAlloc Q8 weights");
        check(cudaHostAlloc(&host_scales_, kScaleCapacity, cudaHostAllocDefault),
              "cudaHostAlloc Q8 scales");
    }
    ~Q8Stager() {
        if (host_weights_) cudaFreeHost(host_weights_);
        if (host_scales_) cudaFreeHost(host_scales_);
        if (stream_weights_) cudaFree(stream_weights_);
        if (stream_scales_) cudaFree(stream_scales_);
        for (auto &[name, entry] : resident_) {
            (void)name;
            if (entry.weights) cudaFree(entry.weights);
            if (entry.scales) cudaFree(entry.scales);
        }
    }
    Q8Stager(const Q8Stager &) = delete;
    Q8Stager &operator=(const Q8Stager &) = delete;

    // Whole-tensor VRAM residency: the 8.7 GiB 8-bit cache fits on the card,
    // so after the first pass every dense GEMV reads VRAM instead of NVMe.
    void load(std::string_view name, const Q8TensorLocation &tensor, uint32_t row, uint32_t rows) {
        const uint64_t weight_bytes = uint64_t(rows) * tensor.cols;
        const uint64_t scale_bytes = uint64_t(rows) *
            (tensor.cols / insignia::glm53::kQ8GroupSize) * 2;
        if (resident_budget_ && row == 0 && rows == tensor.rows) {
            const auto found = resident_.find(std::string(name));
            if (found != resident_.end()) {
                device_weights_ = found->second.weights;
                device_scales_ = found->second.scales;
                return;
            }
            if (tensor.weight_bytes + tensor.scale_bytes <= kWeightCapacity + kScaleCapacity &&
                resident_used_ + tensor.weight_bytes + tensor.scale_bytes <= resident_budget_) {
                Resident entry{};
                check(cudaMalloc(&entry.weights, size_t(tensor.weight_bytes)), "cudaMalloc resident Q8");
                check(cudaMalloc(&entry.scales, size_t(tensor.scale_bytes)), "cudaMalloc resident Q8 scales");
                const auto begin = std::chrono::steady_clock::now();
                index_.read_rows(tensor, 0, tensor.rows, host_weights_, host_scales_);
                const auto end = std::chrono::steady_clock::now();
                io_seconds_ += std::chrono::duration<double>(end - begin).count();
                io_bytes_ += tensor.weight_bytes + tensor.scale_bytes;
                check(cudaMemcpy(entry.weights, host_weights_, size_t(tensor.weight_bytes),
                                 cudaMemcpyHostToDevice), "upload resident Q8");
                check(cudaMemcpy(entry.scales, host_scales_, size_t(tensor.scale_bytes),
                                 cudaMemcpyHostToDevice), "upload resident Q8 scales");
                resident_used_ += tensor.weight_bytes + tensor.scale_bytes;
                device_weights_ = entry.weights;
                device_scales_ = entry.scales;
                resident_.emplace(std::string(name), entry);
                return;
            }
        }
        require(weight_bytes <= kWeightCapacity && scale_bytes <= kScaleCapacity,
                "Q8 tensor slice exceeds streaming slot");
        ensure_stream_slots();
        const auto begin = std::chrono::steady_clock::now();
        index_.read_rows(tensor, row, rows, host_weights_, host_scales_);
        const auto end = std::chrono::steady_clock::now();
        io_seconds_ += std::chrono::duration<double>(end - begin).count();
        io_bytes_ += weight_bytes + scale_bytes;
        check(cudaMemcpy(stream_weights_, host_weights_, size_t(weight_bytes), cudaMemcpyHostToDevice),
              "stage Q8 weights H2D");
        check(cudaMemcpy(stream_scales_, host_scales_, size_t(scale_bytes), cudaMemcpyHostToDevice),
              "stage Q8 scales H2D");
        device_weights_ = stream_weights_;
        device_scales_ = stream_scales_;
    }
    void set_resident_budget(uint64_t bytes) { resident_budget_ = bytes; }
    uint64_t resident_budget() const { return resident_budget_; }
    bool is_resident(std::string_view name) const {
        return resident_.find(std::string(name)) != resident_.end();
    }
    bool resident_view(std::string_view name, ResidentView *out) const {
        const auto found = resident_.find(std::string(name));
        if (found == resident_.end() || !out) return false;
        out->weights = found->second.weights;
        out->scales = reinterpret_cast<const uint16_t *>(found->second.scales);
        return true;
    }
    // Pins the whole 8-bit cache up front, reading the data file in on-disk
    // order with O_DIRECT. Lazy per-tensor pinning interleaves buffered
    // preads with expert O_DIRECT streaming, and that mix collapses to
    // ~0.2 GB/s on the WSL virtio-blk stack (39 s for the 7.8 GiB cache
    // during a 16-token prefill); the sequential upfront pass takes ~2 s.
    uint64_t pin_all() {
        if (!resident_budget_ || pinned_all_) return resident_used_;
        pinned_all_ = true;
        uint64_t largest = 0;
        index_.for_each_by_offset([&](const std::string &, const Q8TensorLocation &tensor) {
            largest = std::max(largest, tensor.weight_bytes + tensor.scale_bytes);
        });
        uint8_t *scratch_raw = nullptr;
        check(cudaHostAlloc(&scratch_raw, size_t(largest) + 16384, cudaHostAllocDefault),
              "cudaHostAlloc pin scratch");
        const auto align_up = [](uint8_t *pointer) {
            return reinterpret_cast<uint8_t *>(
                (reinterpret_cast<uintptr_t>(pointer) + 4095) & ~uintptr_t(4095));
        };
        uint8_t *const weights_dst = align_up(scratch_raw + 4096);
        const auto begin = std::chrono::steady_clock::now();
        index_.for_each_by_offset([&](const std::string &name, const Q8TensorLocation &tensor) {
            if (resident_.count(name)) return;
            if (resident_used_ + tensor.weight_bytes + tensor.scale_bytes > resident_budget_)
                return;
            uint8_t *const scales_dst =
                align_up(weights_dst + tensor.weight_bytes + 4096);
            Resident entry{};
            check(cudaMalloc(&entry.weights, size_t(tensor.weight_bytes)), "cudaMalloc resident Q8");
            check(cudaMalloc(&entry.scales, size_t(tensor.scale_bytes)), "cudaMalloc resident Q8 scales");
            index_.read_rows_direct(tensor, weights_dst, scales_dst);
            check(cudaMemcpy(entry.weights, weights_dst, size_t(tensor.weight_bytes),
                             cudaMemcpyHostToDevice), "upload resident Q8");
            check(cudaMemcpy(entry.scales, scales_dst,
                             size_t(tensor.scale_bytes), cudaMemcpyHostToDevice),
                  "upload resident Q8 scales");
            resident_used_ += tensor.weight_bytes + tensor.scale_bytes;
            io_bytes_ += tensor.weight_bytes + tensor.scale_bytes;
            resident_.emplace(name, entry);
        });
        io_seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        cudaFreeHost(scratch_raw);
        return resident_used_;
    }
    // Pins tensors that dwarf the streaming slot (lm_head is ~620 MB in 8-bit)
    // through a temporary host buffer; false means the budget ran out.
    bool try_pin(std::string_view name, const Q8TensorLocation &tensor) {
        if (!resident_budget_ || is_resident(name))
            return is_resident(name);
        if (resident_used_ + tensor.weight_bytes + tensor.scale_bytes > resident_budget_)
            return false;
        Resident entry{};
        check(cudaMalloc(&entry.weights, size_t(tensor.weight_bytes)), "cudaMalloc resident Q8");
        check(cudaMalloc(&entry.scales, size_t(tensor.scale_bytes)), "cudaMalloc resident Q8 scales");
        std::vector<uint8_t> host(size_t(tensor.weight_bytes) + size_t(tensor.scale_bytes));
        const auto begin = std::chrono::steady_clock::now();
        index_.read_rows(tensor, 0, tensor.rows, host.data(), host.data() + tensor.weight_bytes);
        const auto end = std::chrono::steady_clock::now();
        io_seconds_ += std::chrono::duration<double>(end - begin).count();
        io_bytes_ += tensor.weight_bytes + tensor.scale_bytes;
        check(cudaMemcpy(entry.weights, host.data(), size_t(tensor.weight_bytes),
                         cudaMemcpyHostToDevice), "upload resident Q8");
        check(cudaMemcpy(entry.scales, host.data() + tensor.weight_bytes,
                         size_t(tensor.scale_bytes), cudaMemcpyHostToDevice),
              "upload resident Q8 scales");
        resident_used_ += tensor.weight_bytes + tensor.scale_bytes;
        resident_.emplace(std::string(name), entry);
        return true;
    }
    const uint32_t *weights() const { return reinterpret_cast<const uint32_t *>(device_weights_); }
    const uint8_t *weight_bytes() const { return device_weights_; }
    const uint16_t *scales() const { return reinterpret_cast<const uint16_t *>(device_scales_); }
    double io_seconds() const { return io_seconds_; }
    uint64_t io_bytes() const { return io_bytes_; }
    uint64_t resident_bytes() const { return resident_used_; }

private:
    void ensure_stream_slots() {
        if (!stream_weights_)
            check(cudaMalloc(&stream_weights_, kWeightCapacity), "cudaMalloc Q8 weights");
        if (!stream_scales_)
            check(cudaMalloc(&stream_scales_, kScaleCapacity), "cudaMalloc Q8 scales");
    }

    struct Resident {
        uint8_t *weights = nullptr;
        uint8_t *scales = nullptr;
    };
    Q8Index &index_;
    uint8_t *host_weights_ = nullptr, *host_scales_ = nullptr;
    uint8_t *stream_weights_ = nullptr, *stream_scales_ = nullptr;
    uint8_t *device_weights_ = nullptr, *device_scales_ = nullptr;
    std::unordered_map<std::string, Resident> resident_;
    uint64_t resident_used_ = 0;
    uint64_t resident_budget_ = 0;
    bool pinned_all_ = false;
    double io_seconds_ = 0.0;
    uint64_t io_bytes_ = 0;
};

std::unique_ptr<Q8Index> open_q8(const char *prefix) {
    return prefix && *prefix ? std::make_unique<Q8Index>(prefix) : nullptr;
}

float from_bf16_host(uint16_t value) {
    const uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float from_f16_host(uint16_t value) {
    const uint32_t sign = uint32_t(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = sign;
    if (!exponent) {
        if (mantissa) {
            int unbiased = -14;
            while (!(mantissa & 0x0400u)) {
                mantissa <<= 1;
                --unbiased;
            }
            mantissa &= 0x03ffu;
            bits |= uint32_t(unbiased + 127) << 23;
            bits |= mantissa << 13;
        }
    } else if (exponent == 0x1fu) {
        bits |= 0x7f800000u | (mantissa << 13);
    } else {
        bits |= (exponent + 112u) << 23;
        bits |= mantissa << 13;
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float from_fp8_e4m3_host(uint8_t value) {
    const int exponent = (value >> 3) & 0x0f;
    const int mantissa = value & 0x07;
    if (exponent == 0x0f && mantissa == 0x07)
        return std::numeric_limits<float>::quiet_NaN();
    const float magnitude = exponent
        ? std::ldexp(float(8 + mantissa), exponent - 10)
        : std::ldexp(float(mantissa), -9);
    return value & 0x80 ? -magnitude : magnitude;
}

// Flash stores the mHC metadata as FP32, the tiny oracle as BF16; both widen
// to the same FP32 device upload.
std::vector<float> read_widened(ShardedIndex &model, std::string_view name) {
    const TensorLocation &tensor = model.tensor(name);
    if (tensor.type == TensorType::f32) return read_host<float>(model, name, TensorType::f32);
    require(tensor.type == TensorType::bf16 && !(tensor.bytes % 2),
            "wrong dtype for " + std::string(name));
    std::vector<uint16_t> raw(size_t(tensor.bytes) / 2);
    model.read(tensor, raw.data());
    std::vector<float> result(raw.size());
    for (size_t index = 0; index < raw.size(); ++index)
        result[index] = from_bf16_host(raw[index]);
    return result;
}

std::vector<float> read_widened(ShardedIndex &model, std::string_view name);

class Runner {
public:
    Runner(const char *root, const char *index, const char *q8_prefix)
        : model_(index, root,
                 std::getenv("INSIGNIA_GLM53_STRIPE_INDEX")
                     ? AlternateShardPolicy::disabled
                     : AlternateShardPolicy::environment),
          stager_(model_),
          q8_index_(open_q8(q8_prefix)),
          q8_stager_(q8_index_ ? std::make_unique<Q8Stager>(*q8_index_) : nullptr),
          logits_(0), finite_(1) {
        if (const char *stripe_index = std::getenv("INSIGNIA_GLM53_STRIPE_INDEX")) {
            try {
                stripe_model_ = std::make_unique<ShardedIndex>(
                    stripe_index, root, AlternateShardPolicy::strict_overlay);
                require(stripe_model_->hidden_size() == model_.hidden_size() &&
                            stripe_model_->layers() == model_.layers() &&
                            stripe_model_->experts() == model_.experts() &&
                            stripe_model_->active_experts() == model_.active_experts() &&
                            stripe_model_->tensor_count() == model_.tensor_count(),
                        "stripe overlay geometry does not match the primary index");
                std::printf("expert stripe overlay: %zu shards, %.3f GiB from alternate device\n",
                            stripe_model_->alt_shard_count(),
                            stripe_model_->alt_shard_bytes() / double(1ull << 30));
            } catch (const std::exception &error) {
                stripe_model_.reset();
                if (std::getenv("INSIGNIA_GLM53_STRIPE_REQUIRED")) throw;
                std::fprintf(stderr,
                             "expert stripe unavailable; using exact primary store: %s\n",
                             error.what());
            }
        }
        if (const char *budget = std::getenv("INSIGNIA_GLM53_VRAM_BUDGET_MB"))
            stager_.set_resident_budget(uint64_t(std::max(0, std::atoi(budget))) << 20);
        else if (q8_stager_)
            stager_.set_resident_budget(128ull << 20);
        if (q8_stager_) {
            if (const char *budget = std::getenv("INSIGNIA_GLM53_Q8_BUDGET_MB"))
                q8_stager_->set_resident_budget(uint64_t(std::max(0, std::atoi(budget))) << 20);
            else
                q8_stager_->set_resident_budget(9300ull << 20);
        }
        // Pin the dense 8-bit cache before anything else touches the disk so
        // its reads never interleave with expert O_DIRECT streaming.
        if (q8_stager_ && q8_stager_->resident_budget() &&
            !std::getenv("INSIGNIA_GLM53_NO_PREPIN"))
            q8_stager_->pin_all();
        const std::string config = read_config(root);
        require(model_.hc_mult() == 4, "only 4 hyper-connection streams are supported");
        const uint32_t layer_count = model_.layers();
        layer_types_ = config_string_array(config, "layer_types");
        mlp_types_ = config_string_array(config, "mlp_layer_types");
        require(layer_types_.size() == layer_count && mlp_types_.size() == layer_count,
                "config layer arrays disagree with the indexed layer count");
        hidden_ = int(model_.hidden_size());

        // KDA geometry hangs off the width-0 tensors of layer 0; every other
        // KDA layer shares it (GLM never mixes linear-attention shapes).
        const TensorLocation &q_proj = model_.tensor(layer_stem(0) + "self_attn.q_proj.weight");
        kda_width_ = int(q_proj.shape[0]);
        kda_heads_ = int(model_.tensor(layer_stem(0) + "self_attn.A_log").shape[0]);
        kda_head_dim_ = kda_width_ / kda_heads_;
        f_a_rows_ = int(model_.tensor(layer_stem(0) + "self_attn.f_a_proj.weight").shape[0]);
        require(kda_width_ % kda_heads_ == 0 &&
                (kda_head_dim_ == 32 || kda_head_dim_ == 64 || kda_head_dim_ == 128),
                "unsupported KDA head geometry (widths 32/64/128 are compiled in)");
        require(model_.tensor(layer_stem(0) + "self_attn.b_proj.weight").shape[0] ==
                uint32_t(kda_heads_), "b_proj width disagrees with the KDA head count");

        for (uint32_t layer = 0; layer < layer_count; ++layer) {
            const bool mla = layer_types_[layer] != "linear_attention";
            is_mla_.push_back(mla);
            is_sparse_.push_back(mlp_types_[layer] == "sparse");
            if (mla) mla_slot_.push_back(int(layer));
        }
        mla_layers_ = int(mla_slot_.size());
        kda_layers_ = int(layer_count) - mla_layers_;
        // MTP speculative decoding: layer 45 is a full MLA+MoE decoder with its
        // own router, eh_proj fusion of [enorm(embed)|hnorm(hidden)], and a
        // shared_head norm feeding the tied lm_head. It gets the extra MLA KV
        // slot (mla_layers_ becomes 12); the main loop never iterates to it.
        if (const char *mtp = std::getenv("INSIGNIA_GLM53_MTP")) {
            mtp_draft_total_ = std::max(0, std::atoi(mtp));
            if (mtp_draft_total_) {
                require(mtp_draft_total_ >= 2 && mtp_draft_total_ <= 8,
                        "INSIGNIA_GLM53_MTP must be 2..8 draft tokens (or 0)");
                require(model_.has(layer_stem(45) + "eh_proj.weight") &&
                        model_.has(layer_stem(45) + "shared_head.norm.weight"),
                        "MTP requested but the checkpoint lacks layer 45");
                if (const char *variant = std::getenv("INSIGNIA_GLM53_MTP_VARIANT"))
                    mtp_variant_ = std::atoi(variant);
                mtp_bf16_ = std::getenv("INSIGNIA_GLM53_MTP_BF16") != nullptr;
                mla_slot_.push_back(45);
                ++mla_layers_;
                // eh_proj + the layer-45 MLA projections are ~270 MiB of BF16
                // that the 128 MiB default resident budget cannot hold; they
                // pin after the first round and never stream again.
                if (!std::getenv("INSIGNIA_GLM53_VRAM_BUDGET_MB"))
                    stager_.set_resident_budget(448ull << 20);
            }
        }
        // DFlash2 block-diffusion drafter (z-lab/dflash): one 8-position
        // block forward per round, target-hidden KV injection at layers
        // 5/14/24/33/42, greedy top-16 selector. Shares the target embedding
        // + lm_head; the verify machinery below is the MTP one with K=7.
        if (const char *df = std::getenv("INSIGNIA_GLM53_DFLASH2"); df && std::atoi(df)) {
            const char *index = std::getenv("INSIGNIA_GLM53_DFLASH2_INDEX");
            const std::string index_path =
                index ? index : "/var/lib/insignia/glm53-dflash2.index";
            const std::string root = std::filesystem::path(index_path).parent_path().string();
            const char *fp8 = std::getenv("INSIGNIA_GLM53_DFLASH2_FP8");
            // The superseded default cache (glm53-dflash2-fp8, FC layout bug)
            // no longer exists on disk; -fixed is the only valid target.
            df_ = std::make_unique<insignia::glm53::DFlash2Drafter>(
                index_path, root,
                fp8 ? fp8 : "/var/lib/insignia/glm53-dflash2-fp8-fixed",
                int(model_.vocab_size()));
            dflash2_on_ = true;
            mtp_draft_total_ = insignia::glm53::DFlash2Drafter::kDrafts;
            check(cudaHostAlloc(&df_logits_host_,
                                size_t(insignia::glm53::DFlash2Drafter::kDrafts) * model_.vocab_size() *
                                    sizeof(float),
                                cudaHostAllocDefault),
                  "pin DFlash2 logits buffer");
            check(cudaHostAlloc(&df_hp_host_,
                                size_t(insignia::glm53::DFlash2Drafter::kDrafts) *
                                    insignia::glm53::DFlash2Drafter::kRank * sizeof(float),
                                cudaHostAllocDefault),
                  "pin DFlash2 hp buffer");
            require(kMaxChunk() <= insignia::glm53::DFlash2Drafter::kMaxTokens,
                    "prefill chunk exceeds the DFlash2 capture batch");
        }
        if (const char *topm = std::getenv("INSIGNIA_GLM53_DF_APPROX_TOPM")) {
            const int requested = std::atoi(topm);
            require(requested == 0 || (requested >= 1 && requested <= 7),
                    "INSIGNIA_GLM53_DF_APPROX_TOPM must be 0 or 1..7");
            require(requested == 0 || dflash2_on_,
                    "INSIGNIA_GLM53_DF_APPROX_TOPM requires DFlash2");
            df_approx_topm_ = requested;
        }
        if (const char *mass = std::getenv("INSIGNIA_GLM53_DF_APPROX_MASS")) {
            const float requested = std::strtof(mass, nullptr);
            require(requested > 0.0f && requested <= 1.0f,
                    "INSIGNIA_GLM53_DF_APPROX_MASS must be in (0, 1]");
            require(dflash2_on_, "INSIGNIA_GLM53_DF_APPROX_MASS requires DFlash2");
            require(!df_approx_topm_,
                    "choose fixed INSIGNIA_GLM53_DF_APPROX_TOPM or adaptive MASS, not both");
            df_approx_mass_ = requested;
            if (const char *min_k = std::getenv("INSIGNIA_GLM53_DF_APPROX_MIN_K"))
                df_approx_min_k_ = std::atoi(min_k);
            if (const char *max_k = std::getenv("INSIGNIA_GLM53_DF_APPROX_MAX_K"))
                df_approx_max_k_ = std::atoi(max_k);
            require(df_approx_min_k_ >= 1 && df_approx_min_k_ <= df_approx_max_k_ &&
                        df_approx_max_k_ <= 8,
                    "adaptive DFlash k requires 1 <= MIN_K <= MAX_K <= 8");
        }
        df_approx_renorm_ = df_approx_topm_ &&
            std::getenv("INSIGNIA_GLM53_DF_APPROX_RENORM") &&
            std::atoi(std::getenv("INSIGNIA_GLM53_DF_APPROX_RENORM")) != 0;
        if (df_approx_topm_)
            std::printf("DFlash2 approximate verify: top-%d, retained weights %s\n",
                        df_approx_topm_, df_approx_renorm_ ? "renormalized" : "unchanged");
        if (df_approx_mass_ > 0.0f)
            std::printf("DFlash2 adaptive verify: mass %.3f, k=%d..%d, retained weights unchanged\n",
                        df_approx_mass_, df_approx_min_k_, df_approx_max_k_);
        if (const char *candidate_k = std::getenv("INSIGNIA_GLM53_DF_CACHE_ROUTE_K")) {
            df_cache_route_k_ = std::atoi(candidate_k);
            require(df_cache_route_k_ >= 8 && df_cache_route_k_ <= 32,
                    "INSIGNIA_GLM53_DF_CACHE_ROUTE_K must be 8..32");
            require(dflash2_on_, "cache-aware DFlash routing requires DFlash2");
            require(df_approx_mass_ == 0.0f,
                    "cache-aware routing does not compose with adaptive-mass pruning");
            if (const char *retain = std::getenv("INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN"))
                df_cache_route_retain_ = std::atoi(retain);
            else if (df_approx_topm_)
                df_cache_route_retain_ = df_approx_topm_ - 1;
            if (const char *regret = std::getenv("INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET"))
                df_cache_route_regret_ = std::strtof(regret, nullptr);
            if (df_approx_topm_) {
                require(df_approx_topm_ >= 2 &&
                            df_cache_route_retain_ == df_approx_topm_ - 1,
                        "pruned cache routing keeps Top-M-1 and selects one cache-aware tail");
                require(!df_approx_renorm_,
                        "pruned cache routing currently preserves original Top-8 weights");
            } else {
                require(df_cache_route_retain_ == 6 || df_cache_route_retain_ == 7,
                        "exact-width cache-aware routing retain must be 6 or 7");
            }
            require(df_cache_route_regret_ >= 0.0f && df_cache_route_regret_ <= 0.05f,
                    "cache-aware DFlash routing regret must be in [0,0.05]");
            if (const char *joint =
                    std::getenv("INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS"))
                df_cache_joint_options_ = std::atoi(joint);
            if (const char *mask =
                    std::getenv("INSIGNIA_GLM53_DF_CACHE_MASK_SEARCH"))
                df_cache_mask_search_ = std::atoi(mask) != 0;
            if (const char *verify =
                    std::getenv("INSIGNIA_GLM53_DF_CACHE_MASK_VERIFY"))
                df_cache_mask_verify_ = std::atoi(verify) != 0;
            require(!df_cache_mask_verify_ || df_cache_mask_search_,
                    "cache mask verification requires cache mask search");
            if (const char *guard_retain =
                    std::getenv("INSIGNIA_GLM53_DF_CACHE_GUARD_RETAIN"))
                df_cache_guard_retain_ = std::atoi(guard_retain);
            require(!df_cache_joint_options_ ||
                        (df_cache_joint_options_ >= 2 && df_cache_joint_options_ <= 8),
                    "joint cache routing options must be 0 or 2..8");
            require(df_cache_guard_retain_ >= df_cache_route_retain_ &&
                        df_cache_guard_retain_ <= 8,
                    "cache guard retain must tighten the base policy and be at most 8");
            std::printf("DFlash2 cache-aware route: top-%d, retain %d, regret %.4f\n",
                        df_cache_route_k_, df_cache_route_retain_, df_cache_route_regret_);
            if (df_approx_topm_)
                std::printf("DFlash2 pruned cache route: keep %d strongest, choose slot %d "
                            "from top-%d\n",
                            df_cache_route_retain_, df_approx_topm_, df_cache_route_k_);
            if (df_cache_joint_options_)
                std::printf("DFlash2 joint cache route: %d actions/row, exact union search "
                            "up to k4 (%s)\n", df_cache_joint_options_,
                            df_cache_mask_search_ ? "adaptive Raptor Lake 288-bit POPCNT" :
                                                   "byte-array rollback");
            if (df_cache_guard_retain_ != 8)
                std::printf("DFlash2 cache guard fallback: retain %d\n",
                            df_cache_guard_retain_);
        }
        if (const char *prefill_approx =
                std::getenv("INSIGNIA_GLM53_PREFILL_APPROX_MOE")) {
            prefill_approx_moe_ = std::atoi(prefill_approx) != 0;
            if (const char *first =
                    std::getenv("INSIGNIA_GLM53_PREFILL_APPROX_FIRST_LAYER"))
                prefill_approx_first_layer_ = std::atoi(first);
            require(!prefill_approx_moe_ || df_approx_topm_ || df_approx_mass_ > 0.0f,
                    "approximate prefill needs DF_APPROX_TOPM or DF_APPROX_MASS");
            require(prefill_approx_first_layer_ >= 0 &&
                        prefill_approx_first_layer_ < int(model_.layers()),
                    "PREFILL_APPROX_FIRST_LAYER is outside the model");
            if (prefill_approx_moe_)
                std::printf("prefill approximate MoE: layers %d..%u reuse DFlash %s policy%s\n",
                            prefill_approx_first_layer_, model_.layers() - 1,
                            df_approx_topm_ ? "fixed-k" : "adaptive-mass",
                            df_cache_route_k_ ? " + cache-aware tail" : "");
        }
        if (const char *whole =
                std::getenv("INSIGNIA_GLM53_PREFILL_WHOLE_LAYER_MOE"))
            prefill_whole_layer_moe_ = std::atoi(whole) != 0;
        if (const char *fixed = std::getenv("INSIGNIA_GLM53_NVFP4_FIXED_ROWS"))
            nvfp4_fixed_rows_ = std::atoi(fixed) != 0;
        if (const char *margin = std::getenv("INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN")) {
            df_logit_guard_margin_ = std::strtof(margin, nullptr);
            require(df_logit_guard_margin_ > 0.0f,
                    "INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN must be positive");
            require(dflash2_on_, "DFlash logit guard requires DFlash2");
            if (const char *prefix = std::getenv("INSIGNIA_GLM53_DF_LOGIT_GUARD_PREFIX"))
                df_logit_guard_prefix_ = std::atoi(prefix) != 0;
            std::printf("DFlash2 logit guard: disable %s approximation when draft margin < %.3f\n",
                        df_logit_guard_prefix_ ? "causal-prefix" : "row-only",
                        df_logit_guard_margin_);
        }
        if (const char *threshold =
                std::getenv("INSIGNIA_GLM53_DF_CALIBRATION_GUARD_JS")) {
            df_calibration_guard_js_ = std::strtof(threshold, nullptr);
            require(df_calibration_guard_js_ > 0.0f &&
                        df_calibration_guard_js_ <= 0.693147181f,
                    "DFlash calibration JS guard must be in (0, ln(2)]");
            require(dflash2_on_, "DFlash calibration JS guard requires DFlash2");
            require(df_approx_topm_ || df_approx_mass_ > 0.0f || df_cache_route_k_,
                    "DFlash calibration JS guard requires an approximate verification policy");
            std::printf("DFlash2 calibration guard: exact block when target/draft JS > %.4f\n",
                        df_calibration_guard_js_);
        }
        if (const char *threshold =
                std::getenv("INSIGNIA_GLM53_DF_UNCERTAINTY_TOP1_P"))
            df_uncertainty_top1_p_ = std::strtof(threshold, nullptr);
        if (const char *threshold =
                std::getenv("INSIGNIA_GLM53_DF_UNCERTAINTY_TOP1_DROP"))
            df_uncertainty_top1_drop_ = std::strtof(threshold, nullptr);
        if (df_uncertainty_top1_p_ > 0.0f || df_uncertainty_top1_drop_ > 0.0f) {
            require(dflash2_on_, "DFlash uncertainty guard requires DFlash2");
            require(df_approx_topm_ > 0,
                    "DFlash uncertainty guard currently requires fixed Top-M verification");
            require(df_uncertainty_top1_p_ >= 0.0f &&
                        df_uncertainty_top1_p_ < 1.0f &&
                        df_uncertainty_top1_drop_ >= 0.0f &&
                        df_uncertainty_top1_drop_ < 1.0f,
                    "DFlash uncertainty probabilities must be in [0,1)");
            if (const char *guard_k =
                    std::getenv("INSIGNIA_GLM53_DF_UNCERTAINTY_GUARD_K"))
                df_uncertainty_guard_k_ = std::atoi(guard_k);
            if (const char *hold =
                    std::getenv("INSIGNIA_GLM53_DF_UNCERTAINTY_HOLD_ROUNDS"))
                df_uncertainty_hold_rounds_ = std::atoi(hold);
            require(df_uncertainty_guard_k_ > df_approx_topm_ &&
                        df_uncertainty_guard_k_ <= 8,
                    "DFlash uncertainty guard k must exceed Top-M and be at most 8");
            require(df_uncertainty_hold_rounds_ >= 0 &&
                        df_uncertainty_hold_rounds_ <= 8,
                    "DFlash uncertainty hold must be 0..8 rounds");
            std::printf("DFlash2 distribution guard: p<=%.6f or adjacent p drop>=%.6f; "
                        "causal-prefix k%d, hold %d rounds\n",
                        df_uncertainty_top1_p_, df_uncertainty_top1_drop_,
                        df_uncertainty_guard_k_, df_uncertainty_hold_rounds_);
        }
        if (const char *encoded = std::getenv("INSIGNIA_GLM53_DF_EXACT_ROUNDS")) {
            require(dflash2_on_ && df_approx_topm_ > 0,
                    "DFlash diagnostic exact rounds require approximate DFlash2 verification");
            const char *cursor = encoded;
            while (*cursor) {
                char *end = nullptr;
                const long round = std::strtol(cursor, &end, 10);
                require(end != cursor && round >= 0 && round <= 4096,
                        "DFlash exact-round list contains an invalid round");
                df_diagnostic_exact_rounds_.push_back(int(round));
                if (!*end) break;
                require(*end == ',' && end[1],
                        "DFlash exact-round list must be comma-separated integers");
                cursor = end + 1;
            }
            std::sort(df_diagnostic_exact_rounds_.begin(),
                      df_diagnostic_exact_rounds_.end());
            df_diagnostic_exact_rounds_.erase(
                std::unique(df_diagnostic_exact_rounds_.begin(),
                            df_diagnostic_exact_rounds_.end()),
                df_diagnostic_exact_rounds_.end());
            std::printf("DFlash2 diagnostic exact rounds:");
            for (int round : df_diagnostic_exact_rounds_)
                std::printf(" %d", round);
            std::printf("\n");
        }
        if (const char *threshold =
                std::getenv("INSIGNIA_GLM53_DF_RETRY_TOP1_DROP")) {
            df_retry_top1_drop_ = std::strtof(threshold, nullptr);
            require(dflash2_on_ && df_approx_topm_ > 0,
                    "DFlash post-verify retry requires fixed Top-M DFlash2 verification");
            require(df_retry_top1_drop_ > 0.0f && df_retry_top1_drop_ < 1.0f,
                    "DFlash retry target-probability drop must be in (0,1)");
            require(std::getenv("INSIGNIA_GLM53_DF_BATCH_VERIFY") &&
                        !std::getenv("INSIGNIA_GLM53_DF_SEQ_VERIFY"),
                    "DFlash post-verify retry requires forced batch verification");
            check(cudaHostAlloc(&df_retry_logits_host_,
                                size_t(kMaxVerify) * model_.vocab_size() * sizeof(float),
                                cudaHostAllocDefault),
                  "pin DFlash2 retry target-logit buffer");
            std::printf("DFlash2 post-verify exact retry: target top-1 probability drop "
                        ">= %.6f\n", df_retry_top1_drop_);
        }
        if (const char *path = std::getenv("INSIGNIA_GLM53_DF_MOE_METRICS"))
            moe_metrics_path_ = path;
        const char *exact_falsifier_trace =
            std::getenv("INSIGNIA_GLM53_DF_FALSIFIER_TRACE");
        const char *feature_falsifier_trace =
            std::getenv("INSIGNIA_GLM53_DF_FALSIFIER_FEATURE_TRACE");
        require(!exact_falsifier_trace || !feature_falsifier_trace,
                "choose exact or feature-only DFlash falsifier trace, not both");
        if (exact_falsifier_trace) {
            falsifier_trace_path_ = exact_falsifier_trace;
        } else if (feature_falsifier_trace) {
            falsifier_trace_path_ = feature_falsifier_trace;
            falsifier_feature_only_ = true;
        }
        forced_sequential_verify_ =
            dflash2_on_ && std::getenv("INSIGNIA_GLM53_DF_SEQ_VERIFY") != nullptr;
        // Dense layer -> KDA archive row (recurrent-state replay indexing).
        kda_row_.assign(size_t(layer_count) + 1, -1);
        {
            int row = 0;
            for (uint32_t layer = 0; layer < layer_count; ++layer)
                if (!is_mla_[layer]) kda_row_[layer] = row++;
        }
        prev_routing_.assign(size_t(layer_count),
                             {-1, -1, -1, -1, -1, -1, -1, -1});
        row_routing_.assign(size_t(layer_count), {});
        row_vein_.assign(size_t(layer_count), {});
        early_routing_.assign(size_t(layer_count),
                              {-1, -1, -1, -1, -1, -1, -1, -1});
        if (const char *prefetch = std::getenv("INSIGNIA_GLM53_PREFETCH"))
            prefetch_on_ = std::atoi(prefetch) != 0;
        early_route_on_ = std::getenv("INSIGNIA_GLM53_EARLY_ROUTE") != nullptr;
        early_route_prefetch_ = std::getenv("INSIGNIA_GLM53_EARLY_PREFETCH") != nullptr;
        early_route_on_ = early_route_on_ || early_route_prefetch_;
        if (const char *count = std::getenv("INSIGNIA_GLM53_EARLY_PREFETCH_N"))
            early_route_prefetch_n_ = std::clamp(std::atoi(count), 1, 8);
        early_multi_route_on_ = std::getenv("INSIGNIA_GLM53_EARLY_MULTI_ROUTE") != nullptr;
        early_multi_prefetch_ = std::getenv("INSIGNIA_GLM53_EARLY_MULTI_PREFETCH") != nullptr;
        early_multi_route_on_ = early_multi_route_on_ || early_multi_prefetch_;
        if (const char *count = std::getenv("INSIGNIA_GLM53_EARLY_MULTI_N"))
            early_multi_n_ = std::clamp(std::atoi(count), 1, 8);
        if (const char *count = std::getenv("INSIGNIA_GLM53_EARLY_MULTI_MAX"))
            early_multi_max_ = std::max(1, std::atoi(count));
        if (const char *path = std::getenv("INSIGNIA_GLM53_EARLY_ROUTE_TRACE")) {
            early_route_trace_ = std::fopen(path, "w");
            early_route_on_ = true;
        }
        if (const char *path = std::getenv("INSIGNIA_GLM53_EARLY_MULTI_TRACE")) {
            early_multi_trace_ = std::fopen(path, "w");
            early_multi_route_on_ = true;
        }
        early_multi_rows_.resize(size_t(layer_count));
        if (early_multi_route_on_)
            std::printf("early multi route: prefetch %s, top-%d, cap %d\n",
                        early_multi_prefetch_ ? "on" : "off", early_multi_n_, early_multi_max_);
        deep_checks_ = std::getenv("INSIGNIA_GLM53_FINITE_EVERY_LAYER") != nullptr;
        trace_layers_ = std::getenv("INSIGNIA_GLM53_PROFILE") != nullptr;
        if (mla_layers_) {
            const std::string stem = layer_stem(mla_slot_.front()) + "self_attn.";
            q_a_rows_ = int(model_.tensor(stem + "q_a_proj.weight").shape[0]);
            q_b_rows_ = int(model_.tensor(stem + "q_b_proj.weight").shape[0]);
            kv_a_rows_ = int(model_.tensor(stem + "kv_a_proj_with_mqa.weight").shape[0]);
            kv_b_rows_ = int(model_.tensor(stem + "kv_b_proj.weight").shape[0]);
            mla_heads_ = config_int(config, "num_attention_heads");
            mla_head_dim_ = q_b_rows_ / mla_heads_;
            require(q_b_rows_ % mla_heads_ == 0 && kv_b_rows_ == 2 * q_b_rows_ &&
                    mla_head_dim_ >= 32 && mla_head_dim_ % 32 == 0,
                    "unsupported MLA geometry (key and value dims must be equal)");
        }

        moe_experts_ = int(model_.experts());
        moe_topk_ = int(model_.active_experts());
        moe_intermediate_ = int(model_.moe_intermediate());
        dense_intermediate_ = int(model_.tensor(layer_stem(0) + "mlp.gate_proj.weight").shape[0]);
        size_t first_sparse = mlp_types_.size();
        for (size_t layer = 0; layer < mlp_types_.size(); ++layer)
            if (is_sparse_[layer]) { first_sparse = layer; break; }
        if (first_sparse != mlp_types_.size()) {
            const std::string stem = layer_stem(int(first_sparse)) + "mlp.";
            shared_intermediate_ = int(model_.tensor(stem + "shared_experts.gate_proj.weight").shape[0]);
            nvfp4_experts_ = model_.has(stem + "experts.0.down_proj.weight_scale");
            q3_experts_ = model_.has(stem + "q3_experts.down_proj.weight");
            require(!(nvfp4_experts_ && q3_experts_),
                    "checkpoint exposes both NVFP4 and native Q3 expert schemas");
            if (nvfp4_experts_ || q3_experts_) {
                // Host-RAM LRU over whole expert records: one decode token
                // touches 42 x 8 = 336 records, so the tier must exceed that
                // to hit at all. Default 5 GiB pinned (~370 records) inside
                // the 14 GiB WSL VM; VRAM keeps only the 13.5 MiB H2D target.
                uint64_t host_cache = 32768ull << 20;
                if (const char *budget = std::getenv("INSIGNIA_GLM53_EXPERT_CACHE_MB"))
                    host_cache = uint64_t(std::max(0, std::atoi(budget))) << 20;
                expert_stager_ =
                    std::make_unique<ExpertStager>(model_, stripe_model_.get(), host_cache,
                                                   q3_experts_);
            }
        }
        if (!moe_metrics_path_.empty() ||
            (!falsifier_trace_path_.empty() && !falsifier_feature_only_)) {
            require(dflash2_on_, "DFlash MoE diagnostics require DFlash2");
            require(!df_approx_topm_ && df_approx_mass_ == 0.0f && !df_cache_route_k_,
                    "MoE diagnostics must run on the exact top-8 verification path");
            require(nvfp4_experts_, "MoE diagnostics require retained NVFP4 expert outputs");
        }
        if (!falsifier_trace_path_.empty()) {
            require(dflash2_on_, "DFlash falsifier traces require DFlash2");
            require(nvfp4_experts_, "DFlash falsifier traces require NVFP4 expert residency");
        }
        if (!moe_metrics_path_.empty()) {
            moe_metrics_ = std::fopen(moe_metrics_path_.c_str(), "w");
            require(moe_metrics_, "cannot open DFlash MoE metrics output");
            std::fprintf(moe_metrics_,
                "epoch,layer,row,topm,semantics,mse,rel_l2,cosine,max_abs,"
                "norm_ratio,retained_mass,exact_cancel,approx_cancel,replay_max_abs\n");
        }
        if (!falsifier_trace_path_.empty()) {
            falsifier_trace_ = std::fopen(falsifier_trace_path_.c_str(), "wb");
            require(falsifier_trace_, "cannot open DFlash falsifier trace output");
            DfFalsifierTraceHeader header{};
            std::memcpy(header.magic, "INSFAL1", 7);
            header.version = 2;
            header.header_bytes = sizeof(header);
            header.record_bytes = sizeof(DfFalsifierEventV2);
            header.layer_count = uint16_t(layer_count);
            header.expert_count = uint16_t(moe_experts_);
            header.topk = uint16_t(moe_topk_);
            header.candidate_k = 32;
            header.hidden_sketch = 64;
            header.hidden = uint32_t(hidden_);
            // bit 0: normalized Gram labels are present; bit 1: signed
            // CountSketch is scaled by sqrt(64/hidden).
            header.flags = falsifier_feature_only_ ? 2u : 3u;
            require(std::fwrite(&header, sizeof(header), 1, falsifier_trace_) == 1,
                    "write DFlash falsifier trace header");
            std::printf("DFlash falsifier %s trace: %s (%zu-byte records)\n",
                        falsifier_feature_only_ ? "on-policy feature-only" : "exact-teacher",
                        falsifier_trace_path_.c_str(), sizeof(DfFalsifierEventV2));
        }

        streams_a_.reset(size_t(kStreams) * hidden_);
        streams_b_.reset(size_t(kStreams) * hidden_);
        post_.reset(4);
        comb_.reset(16);
        mhc_workspace_.reset(insignia::glm53::mhc_workspace_bytes());
        collapsed_.reset(hidden_);
        normalized_.reset(hidden_);
        attention_.reset(hidden_);
        ffn_.reset(hidden_);
        routed_.reset(hidden_);
        expert_scratch_.reset(hidden_);
        q_.reset(kda_width_);
        k_.reset(kda_width_);
        v_.reset(kda_width_);
        gate_8192_.reset(kda_width_);
        core_.reset(kda_width_);
        projected_8192_.reset(kda_width_);
        small_a_.reset(std::max({f_a_rows_, q_a_rows_, kv_a_rows_}));
        small_b_.reset(std::max({q_a_rows_, kv_a_rows_}));
        kv_.reset(kv_b_rows_);
        mla_query_.reset(q_b_rows_);
        mla_output_.reset(q_b_rows_);
        gate_.reset(std::max({dense_intermediate_, moe_intermediate_, shared_intermediate_}));
        up_.reset(gate_.size());
        activation_.reset(gate_.size());
        router_.reset(moe_experts_);
        beta_.reset(kda_heads_);
        logits_.reset(model_.vocab_size());
        kda_states_.reset(size_t(kda_layers_) * kda_width_ * kda_head_dim_);
        conv_history_.reset(size_t(kda_layers_) * 9 * kda_width_);
        mla_legacy_ = std::getenv("INSIGNIA_GLM53_MLA_LEGACY") != nullptr;
        kv_fp8_ = !std::getenv("INSIGNIA_GLM53_KV_FP8") ||
                  std::atoi(std::getenv("INSIGNIA_GLM53_KV_FP8")) != 0;
        if (mla_legacy_) {
            const size_t expanded_stride = size_t(kLegacyMlaContext) * q_b_rows_;
            mla_keys_.reset(size_t(mla_layers_) * expanded_stride);
            mla_values_.reset(size_t(mla_layers_) * expanded_stride);
        } else {
            const size_t latent_stride = size_t(kMaxContext()) * kv_a_rows_;
            mla_latent_u8_.reset(kv_fp8_ ? size_t(mla_layers_) * latent_stride : 0);
            mla_latent_f32_.reset(kv_fp8_ ? 0 : size_t(mla_layers_) * latent_stride);
            mla_latent_scale_.reset(size_t(mla_layers_) * kMaxContext() *
                                    insignia::glm53::kMlaLatentGroups);
            mla_partial_.reset(size_t(mla_heads_) * ((kMaxContext() + 511) / 512) *
                               (size_t(kv_a_rows_) + 2));
            absorb_per_layer_ = size_t(mla_heads_) * mla_head_dim_ * kv_a_rows_;
            const char *compact_absorb = std::getenv("INSIGNIA_GLM53_MLA_FP8_ABSORB");
            // Exact on-consumption reconstruction is the normal full-FP8
            // path: it trades idle Ada ALU for 704 MiB of expert residency.
            // Explicit zero retains the materialized FP32 A/B/oracle path.
            mla_fp8_absorb_ = (!compact_absorb || std::atoi(compact_absorb) != 0) &&
                              bind_mla_absorb_fp8();
            if (!mla_fp8_absorb_) {
                w_uk_.reset(absorb_per_layer_ * mla_layers_);
                w_uv_.reset(absorb_per_layer_ * mla_layers_);
                extract_mla_absorb();
            } else {
                const double removed_mib =
                    2.0 * absorb_per_layer_ * mla_layers_ * sizeof(float) / double(1 << 20);
                std::printf("MLA absorb: exact on-consumption resident FP8 "
                            "(%.0f MiB FP32 duplicate removed)\n", removed_mib);
            }
            mla_cross_head_fp8_ = mla_fp8_absorb_ && kv_fp8_ &&
                std::getenv("INSIGNIA_GLM53_MLA_CROSS_HEAD_FP8") &&
                std::atoi(std::getenv("INSIGNIA_GLM53_MLA_CROSS_HEAD_FP8")) != 0;
            if (mla_cross_head_fp8_) {
                const char *parallel_prefix =
                    std::getenv("INSIGNIA_GLM53_MLA_PREFIX_PARALLEL");
                mla_prefix_parallel_ = !parallel_prefix ||
                    std::atoi(parallel_prefix) != 0;
                mla_qeff_u8_.reset(size_t(kMaxChunk()) * mla_heads_ * kv_a_rows_);
                mla_qeff_scale_.reset(size_t(kMaxChunk()) * mla_heads_ *
                                       insignia::glm53::kMlaLatentGroups);
                mla_qeff_f32_.reset(size_t(kMaxChunk()) * mla_heads_ * kv_a_rows_);
                if (mla_prefix_parallel_)
                    mla_prefix_partial_.reset(
                        size_t(mla_heads_) * (kLegacyMlaContext / 16) *
                        (kv_a_rows_ + 2));
                std::printf("MLA long path: exact 256-row prefix + approximate "
                            "H8 decode / H4xQ8 fused prefill FP8 suffix\n");
                std::printf("MLA decode prefix: %s\n",
                            mla_prefix_parallel_
                                ? "ordered-partial FP32 default (reassociated)"
                                : "scalar FP32 diagnostic (explicit opt-out)");
            }
            const char *reconstruct = std::getenv("INSIGNIA_GLM53_MLA_RECON_PREFIX");
            mla_prefix_reconstruct_ = reconstruct && std::atoi(reconstruct) != 0 &&
                                      mla_fp8_absorb_;
            const size_t prefix_latent_stride =
                size_t(kLegacyMlaContext) * kv_a_rows_;
            if (mla_prefix_reconstruct_) {
                mla_prefix_latent_.reset(size_t(mla_layers_) * prefix_latent_stride);
                mla_prefix_kv_.reset(size_t(kLegacyMlaContext) * kv_b_rows_);
                const double old_mib = 2.0 * mla_layers_ * kLegacyMlaContext *
                                       q_b_rows_ * sizeof(float) / double(1 << 20);
                const double new_mib = (mla_prefix_latent_.size() + mla_prefix_kv_.size()) *
                                       sizeof(float) / double(1 << 20);
                std::printf("MLA exact prefix: FP32 latent reconstruction "
                            "(%.1f MiB -> %.1f MiB, %.1f MiB reclaimed)\n",
                            old_mib, new_mib, old_mib - new_mib);
            } else {
                if (mla_cross_head_fp8_)
                    mla_prefix_latent_.reset(
                        size_t(mla_layers_) * prefix_latent_stride);
                const size_t expanded_stride = size_t(kLegacyMlaContext) * q_b_rows_;
                mla_keys_.reset(size_t(mla_layers_) * expanded_stride);
                mla_values_.reset(size_t(mla_layers_) * expanded_stride);
                if (reconstruct && std::atoi(reconstruct) != 0)
                    std::printf("MLA exact prefix reconstruction unavailable: "
                                "resident compact FP8 absorb is required\n");
            }
        }
        c_stream_a_.reset(size_t(kMaxChunk()) * kStreams * hidden_);
        c_stream_b_.reset(c_stream_a_.size());
        c_collapsed_.reset(size_t(kMaxChunk()) * hidden_);
        c_normalized_.reset(size_t(kMaxChunk()) * hidden_);
        c_attn_.reset(size_t(kMaxChunk()) * hidden_);
        c_ffn_.reset(size_t(kMaxChunk()) * hidden_);
        c_routed_.reset(size_t(kMaxChunk()) * hidden_);
        // Verification stores one down-projection per pick so it can replay
        // scalar router order.  Chunks above the parity-proven 64 rows need
        // the same scratch for every row: expert weights are still read once
        // for the 128-row union, then routed accumulation is replayed in the
        // two legacy 64-row union orders.
        const int expert_out_rows = kMaxChunk() > 64 ? kMaxChunk() : kMaxVerify;
        c_expert_out_.reset(size_t(expert_out_rows) * moe_topk_ * hidden_);
        c_q_.reset(size_t(kMaxChunk()) * kda_width_);
        c_k_.reset(size_t(kMaxChunk()) * kda_width_);
        c_v_.reset(size_t(kMaxChunk()) * kda_width_);
        c_gate_.reset(size_t(kMaxChunk()) * kda_width_);
        c_core_.reset(size_t(kMaxChunk()) * kda_width_);
        c_proj_.reset(size_t(kMaxChunk()) * kda_width_);
        c_small_.reset(size_t(kMaxChunk()) * std::max({f_a_rows_, q_a_rows_, kv_a_rows_}));
        c_kv_.reset(size_t(kMaxChunk()) * kv_b_rows_);
        c_mlaq_.reset(size_t(kMaxChunk()) * q_b_rows_);
        c_mlao_.reset(size_t(kMaxChunk()) * q_b_rows_);
        c_gateu_.reset(size_t(kMaxChunk()) * std::max({dense_intermediate_, moe_intermediate_,
                                                    shared_intermediate_}));
        c_up_.reset(c_gateu_.size());
        c_act_.reset(c_gateu_.size());
        c_router_.reset(size_t(kMaxChunk()) * moe_experts_);
        c_post_.reset(size_t(kMaxChunk()) * 4);
        c_comb_.reset(size_t(kMaxChunk()) * 16);
        c_beta_.reset(size_t(kMaxChunk()) * kda_heads_);
        if (nvfp4_experts_) {
            nv_workspace_4096_.reset(insignia::glm53::nvfp4_workspace_rows_bytes(hidden_, kMaxVerify));
            nv_workspace_2048_.reset(
                insignia::glm53::nvfp4_workspace_rows_bytes(moe_intermediate_, kMaxVerify));
        }
        if (q3_experts_) {
            iq_workspace_4096_.reset(
                insignia::glm53::iq_workspace_rows_bytes(hidden_, kMaxVerify));
            iq_workspace_2048_.reset(
                insignia::glm53::iq_workspace_rows_bytes(moe_intermediate_, kMaxVerify));
        }
        if (q8_index_)
            q8_workspace_.reset(q8_index_->format() == Cache8Format::fp8_e4m3 ?
                insignia::glm53::fp8_batch_workspace_bytes(16384, kMaxChunk()) :
                insignia::glm53::q8_workspace_bytes(16384));
        last_avg_.reset(hidden_);
        last_normed_.reset(hidden_);
        if (mtp_draft_total_) {
            mtp_eh_in_.reset(2 * size_t(hidden_));
            mtp_hidden_.reset(hidden_);
            mtp_attn_.reset(hidden_);
            mtp_moe_.reset(hidden_);
            mtp_recycle_.reset(hidden_);
            mtp_logits_.reset(model_.vocab_size());
            verify_means_.reset(size_t(kMaxVerify) * hidden_);
            verify_normed_.reset(size_t(kMaxVerify) * hidden_);
            verify_logits_.reset(size_t(kMaxVerify) * model_.vocab_size());
            verify_arg_.reset(kMaxVerify);
            if (!forced_sequential_verify_) {
                kda_snap_.reset(kda_states_.size());
                conv_snap_.reset(conv_history_.size());
            } else {
                const double removed_mib =
                    (kda_states_.size() + conv_history_.size()) * sizeof(float) /
                    double(1 << 20);
                std::printf("DFlash sequential verify: %.1f MiB rollback snapshots removed\n",
                            removed_mib);
            }
            // Per (kda layer, token): pre-conv q,k,v + raw gate + raw beta, the
            // exact inputs the recurrence replay needs after a rejected draft.
            kda_arch_.reset(size_t(kda_layers_) * kMaxVerify * (4 * size_t(kda_width_) + kda_heads_));
        }
        if (df_calibration_guard_js_ > 0.0f) {
            const int vocab = int(model_.vocab_size());
            df_prior_logits_device_.reset(size_t(vocab));
            df_logit_metrics_workspace_.reset(
                insignia::glm53::logit_metrics_workspace_bytes(vocab));
            df_logit_metrics_device_.reset(1);
            check(cudaHostAlloc(&df_logit_metrics_host_,
                                sizeof(insignia::glm53::LogitMetrics),
                                cudaHostAllocDefault),
                  "pin DFlash2 calibration metrics");
        }
        if (df_uncertainty_top1_p_ > 0.0f || df_uncertainty_top1_drop_ > 0.0f) {
            constexpr int rows = insignia::glm53::DFlash2Drafter::kDrafts;
            const int vocab = int(model_.vocab_size());
            df_logit_row_stats_workspace_.reset(
                insignia::glm53::logit_row_stats_workspace_bytes(rows, vocab));
            df_logit_row_stats_device_.reset(rows);
            check(cudaHostAlloc(&df_logit_row_stats_host_,
                                size_t(rows) * sizeof(insignia::glm53::LogitRowStats),
                                cudaHostAllocDefault),
                  "pin DFlash2 row statistics");
        }
        check(cudaMemset(kda_states_, 0, kda_states_.size() * sizeof(float)), "clear KDA states");
        check(cudaMemset(conv_history_, 0, conv_history_.size() * sizeof(float)), "clear convolution history");
        if (nvfp4_experts_)
            check(insignia::glm53::initialize_nvfp4(), "initialize NVFP4 tables");
        if (q8_index_)
            std::printf("%s cache: %zu matrices, %.3f GiB\n",
                        q8_index_->format() == Cache8Format::q8 ? "Q8" : "FP8",
                        q8_index_->tensor_count(),
                        q8_index_->data_bytes() / double(1ull << 30));
        std::printf("geometry: hidden %d, %u layers (%d KDA %d MLA), experts %dx%d@%d, "
                    "KDA %dx%d, MLA %dx%d%s\n",
                    hidden_, layer_count, kda_layers_, mla_layers_, moe_experts_, moe_topk_,
                    moe_intermediate_, kda_heads_, kda_head_dim_, mla_heads_, mla_head_dim_,
                    q3_experts_ ? ", native Q3 experts" :
                    (nvfp4_experts_ ? ", NVFP4 experts" : ", BF16 experts"));
        if (expert_stager_ && expert_stager_->cache_slots())
            std::printf("expert cache: %d pinned host-RAM %s records (%.1f MiB)\n",
                        expert_stager_->cache_slots(),
                        q3_experts_ ? "Q3" : "NVFP4",
                        expert_stager_->cache_slots() * expert_stager_->window_bytes() /
                            double(1 << 20));
        else if (expert_stager_)
            std::printf("expert cache: disabled (no pinned host records)\n");
        load_cct();
        if (stager_.resident_bytes() || (q8_stager_ && q8_stager_->resident_bytes()))
            std::printf("resident: %llu MiB BF16 + %llu MiB %s pinned in VRAM\n",
                        (unsigned long long)(stager_.resident_bytes() >> 20),
                        (unsigned long long)(q8_stager_ ? q8_stager_->resident_bytes() : 0) >> 20,
                        q8_index_->format() == Cache8Format::q8 ? "Q8" : "FP8");
    }

    ~Runner() {
        if (df_logit_row_stats_host_) cudaFreeHost(df_logit_row_stats_host_);
        if (df_logit_metrics_host_) cudaFreeHost(df_logit_metrics_host_);
        if (df_retry_logits_host_) cudaFreeHost(df_retry_logits_host_);
        if (df_hp_host_) cudaFreeHost(df_hp_host_);
        if (df_logits_host_) cudaFreeHost(df_logits_host_);
    }

    int layer_count() const { return int(model_.layers()); }
    static constexpr int kMaxChunkCap = 128;
    static int kMaxChunk() {
        static const int chunk = [] {
            const char *value = std::getenv("INSIGNIA_GLM53_PREFILL_CHUNK");
            return std::clamp(value ? std::atoi(value) : kMaxChunkCap, 8, kMaxChunkCap);
        }();
        return chunk;
    }
    // Verify passes are hard-capped by the drafter (DFlash2 kDrafts=7, MTP
    // clamp 8); sizing their scratch by kMaxChunk would burn ~240 MiB per
    // chunk-size doubling for nothing.
    static constexpr int kMaxVerify = 8;
    bool dflash2_on() const { return dflash2_on_; }

    std::vector<std::pair<int, float>> step(int token, int position, int layer_limit, bool produce_logits);
    // Initial prompt only: preserve the proven <=128-row kernels and their
    // arithmetic order, but visit every chunk of a layer before advancing to
    // the next layer so the complete 288-record expert layer stays hot.
    void prefill_prompt_full_layer_major(const std::vector<int> &tokens);
    void prefill(const std::vector<int> &tokens, int position_base, bool capture = false);
    void force_logits(const std::vector<int> &tokens, int position_base, int anchor,
                      const char *dump_path);
    // MTP speculative decoding surface.
    int mtp_k() const { return mtp_draft_total_; }
    void report_cache_selector() const {
        if (!df_cache_selector_calls_) return;
        std::printf("  DFlash cache selector %.3f ms total / %.3f us per layer group "
                    "across %llu calls (%llu POPCNT / %llu byte%s)\n",
                    1.0e3 * df_cache_selector_seconds_,
                    1.0e6 * df_cache_selector_seconds_ / df_cache_selector_calls_,
                    (unsigned long long)df_cache_selector_calls_,
                    (unsigned long long)df_cache_selector_mask_calls_,
                    (unsigned long long)(df_cache_selector_calls_ -
                                         df_cache_selector_mask_calls_),
                    df_cache_mask_verify_ ? ", shadow-verified" : "");
    }
    const float *last_avg() const { return last_avg_.get(); }
    const float *chain_root_hidden() const {
        return (mtp_variant_ == 1 ? last_normed_ : last_avg_).get();
    }
    const float *verify_row_hidden(int row) const {
        return (mtp_variant_ == 1 ? verify_normed_ : verify_means_).get() + size_t(row) * hidden_;
    }
    const float *mtp_recycle() const { return mtp_recycle_.get(); }
    const float *verify_mean(int row) const { return verify_means_.get() + size_t(row) * hidden_; }
    // DFlash2 surface: draft 7 candidates for the anchor at `position`, and
    // append `rows` verify-captured tokens to the drafter K/V cache.
    std::vector<int> df_draft(int anchor, int position);
    void df_commit(int rows, int pos0) { df_->commit(rows, pos0); }
    void adopt_df_prior_logits(int row) {
        if (df_calibration_guard_js_ <= 0.0f && df_retry_top1_drop_ <= 0.0f) return;
        require(row >= 0 && row < kMaxVerify, "DFlash prior-logit row is out of range");
        const size_t vocab = model_.vocab_size();
        retain_df_prior_logits(verify_logits_.get() + size_t(row) * vocab, nullptr);
    }
    bool dflash_retry_needed(int rows);
    void begin_dflash_exact_retry() {
        for (int row = 0; row < kMaxVerify; ++row)
            guard_dflash_row(row, 8);
        df_retry_replay_ = true;
    }
    void end_dflash_exact_retry() { df_retry_replay_ = false; }
    // Re-key prev_routing_ on the accepted anchor row of the last multi-row
    // verify: moe_multi stored the chunk's LAST row (a rejected draft
    // position after partial acceptance), which then keyed every
    // prev-routing prefetch for the next round.
    void adopt_anchor_routing(int row) {
        if (row < 0) return;
        for (size_t layer = 0; layer < prev_routing_.size(); ++layer)
            for (int slot = 0; slot < moe_topk_; ++slot)
                prev_routing_[layer][size_t(slot)] =
                    row_routing_[layer][size_t(row * moe_topk_ + slot)];
    }
    // Marks the start of a verify round: the stager round-epoch advances so
    // acceptance-prefix demotion can tell this round's residue from residue
    // that earned earlier-term residency (INSIGNIA_GLM53_TIER_SLRU).
    void begin_verify_epoch() {
        if (expert_stager_) expert_stager_->set_epoch(++verify_epoch_);
    }
    // Acceptance-prefix demote after a verified round: experts whose verify
    // rows were entirely in the rejected tail (and never in an accepted row
    // this round) get pushed to the cold end of the host tier's probationary
    // segment, turning verify-burst insert pressure into early eviction
    // candidates instead of protected-segment residents (P7 fix).
    void demote_rejected_routing(int matched, int drafted) {
        if (!expert_stager_ || matched >= drafted) return;
        std::unordered_set<uint32_t> accepted, rejected;
        for (size_t layer = 0; layer < row_routing_.size(); ++layer) {
            if (row_routing_[layer].empty() || !is_sparse_[layer]) continue;
            for (int row = 0; row < drafted && row < 8; ++row) {
                // Vein guard: rows not routed by THIS round's verify chunks
                // (stale leftovers, seq-verify's never-forwarded tail) are
                // out of scope entirely.
                if (row_vein_[layer][size_t(row)] != verify_epoch_) continue;
                for (int slot = 0; slot < moe_topk_; ++slot) {
                    const int expert = row_routing_[layer][size_t(row * moe_topk_ + slot)];
                    if (expert < 0) continue;
                    const uint32_t key =
                        ExpertStager::route_key_public(int(layer), expert);
                    (row < matched ? accepted : rejected).insert(key);
                }
            }
        }
        for (const uint32_t key : accepted) rejected.erase(key);
        slru_demoted_total_ += expert_stager_->demote_round(rejected, verify_epoch_);
    }
    // Verifies candidate tokens at position_base+1.. against the target and
    // returns (accepted_count, argmax rows) — see the caller in main().
    int verify_token(int token, int position);
    std::pair<int, std::vector<int>> verify_round(const std::vector<int> &candidates,
                                                  int position_base);
    int mtp_forward(int token, const float *hidden_in, int position);
    void mtp_moe(const float *input, float *output);
    void rollback_kda(int accepted, int position_base);
    // Row-sequential verify controls: snapshot suppression + drafter capture slot.
    bool forced_sequential_verify_ = false;
    bool verify_may_rollback_ = true;
    int capture_offset_ = 0;
    void set_last_avg(const float *device_row) {
        check(cudaMemcpyAsync(last_avg_.get(), device_row, size_t(hidden_) * sizeof(float),
                              cudaMemcpyDeviceToDevice), "copy pending hidden");
    }
    // U3 adaptive-k cost estimation: demand NVMe record reads so far.
    uint64_t expert_records_read() const {
        return expert_stager_ ? expert_stager_->records_read() : 0;
    }

private:
    void linear(std::string_view name, const float *input, float *output, int rows, int cols);
    void linear_pair(std::string_view name_a, std::string_view name_b, const float *input,
                     float *out_a, float *out_b, int rows, int cols);
    void linear_multi(std::string_view name, const float *inputs, float *outputs,
                      int tokens, int rows, int cols);
    void linear_rows(std::string_view name, const TensorLocation &weight, uint64_t row, int rows,
                     const float *input, float *output, int cols);
    void rms(std::string_view weight, const float *input, float *output, int rows, int cols);
    void mhc(std::string_view stem, std::string_view norm,
             const float *streams, float *normalized);
    void kda(int layer, const float *input, float *output, int position);
    void mla(int layer, const float *input, float *output, int position);
    bool bind_mla_absorb_fp8();
    void extract_mla_absorb();
    void reconstruct_mla_prefix(int slot, int positions, int dirty_base);
    void compute_mlp(std::string_view stem, const float *input, float *output, int intermediate);
    void dense_mlp(std::string_view stem, const float *input, float *output, int intermediate);
    void sparse_moe(int layer, const float *input, float *output);
    void mhc_multi(std::string_view stem, const float *streams, float *collapsed, int tokens);
    void kda_multi(int layer, const float *input, float *output, int tokens, int position_base);
    void mla_multi(int layer, const float *input, float *output, int tokens, int position_base);
    void mlp_multi(std::string_view stem, const float *input, float *output, int tokens, int intermediate);
    void moe_multi(int layer, const float *input, float *output, int tokens);
    void report_moe_metrics(
        int layer, const std::vector<std::vector<std::pair<int, float>>> &selection,
        const std::vector<std::array<uint16_t, 32>> &candidate_experts,
        const std::vector<std::array<float, 32>> &candidate_logits,
        const std::vector<std::array<float, 32>> &candidate_choice,
        const std::vector<std::array<float, 8>> &router_summary,
        const std::vector<std::array<uint32_t, 4>> &candidate_residency,
        const float *input, int tokens);
    void report_falsifier_features(
        int layer, const std::vector<std::vector<std::pair<int, float>>> &selection,
        const std::vector<int> &exec_count,
        const std::vector<std::array<uint16_t, 32>> &candidate_experts,
        const std::vector<std::array<float, 32>> &candidate_logits,
        const std::vector<std::array<float, 32>> &candidate_choice,
        const std::vector<std::array<float, 8>> &router_summary,
        const std::vector<std::array<uint32_t, 4>> &candidate_residency,
        const float *input, int tokens);
    void prefill_layer_chunk_exact(int layer, float *in_place, float *scratch,
                                   int count, int position_base, float *dflash_capture);
    const float *device_f32(std::string_view name);
    const std::vector<float> &host_f32(std::string_view name);
    void archive_kda_rows(int layer, const float *src, int rows, int slot);
    double expert_io_seconds() const { return expert_stager_ ? expert_stager_->io_seconds() : 0.0; }
    uint64_t expert_io_bytes() const { return expert_stager_ ? expert_stager_->io_bytes() : 0; }

    ShardedIndex model_;
    TensorStager stager_;
    std::unique_ptr<ShardedIndex> stripe_model_;
    std::unique_ptr<ExpertStager> expert_stager_;
    std::vector<std::array<int, 8>> prev_routing_;
    std::vector<std::array<int, 64>> row_routing_;  // [layer][row*8+slot], last multi-row chunk
    // Per-row freshness vein for acceptance-prefix demotion: moe_multi stamps
    // rows it routed with the current verify epoch; demote only trusts rows
    // carrying the live epoch (seq-verify tails and stale chunks are shut out).
    std::vector<std::array<uint32_t, 8>> row_vein_;
    uint32_t verify_epoch_ = 0;
    uint64_t slru_demoted_total_ = 0;
    std::vector<std::array<int, 8>> early_routing_;
    DeviceBuffer<float> expert_scratch_;
    FILE *route_trace_ = nullptr;
    FILE *early_route_trace_ = nullptr;
    FILE *early_multi_trace_ = nullptr;
    bool route_trace_probed_ = false;
    long token_index_ = 0;
    // Parity instrumentation: INSIGNIA_GLM53_LAYER_DUMP=<path> appends one
    // binary record per decode layer: i32[3] {token_index_, layer, hidden_}
    // followed by hidden_ f32 values, the mean of the 4 mHC streams after
    // that layer's FFN mix.
    FILE *layer_dump_ = nullptr;
    bool layer_dump_probed_ = false;
    std::vector<float> layer_dump_host_;
    // Sub-operation seam dump for parity bisection: INSIGNIA_GLM53_SEAM_DUMP
    // names the file, INSIGNIA_GLM53_SEAM_LAYER the layer to trace (default 0,
    // -1 traces every layer).
    // Record: i32[4] {token_index, layer, tag, count} + count f32 values.
    // 1 attn-norm  2 attn-out  3 streams after attn mix  4 ffn-norm
    // 5 ffn-out    6 streams after ffn mix.
    FILE *seam_dump_ = nullptr;
    bool seam_dump_probed_ = false;
    int seam_layer_ = 0;
    std::vector<float> seam_host_;
    bool prefetch_on_ = true, deep_checks_ = false, trace_layers_ = false;
    bool full_layer_major_active_ = false;
    bool prefill_approx_moe_ = false;
    bool prefill_whole_layer_moe_ = false;
    bool nvfp4_fixed_rows_ = true;
    int prefill_approx_first_layer_ = 0;
    bool early_route_on_ = false, early_route_prefetch_ = false;
    int early_route_prefetch_n_ = 8;
    uint64_t early_route_hits_ = 0, early_route_total_ = 0;
    bool early_multi_route_on_ = false, early_multi_prefetch_ = false;
    int early_multi_n_ = 4, early_multi_max_ = 64;
    uint64_t early_multi_batch_ = 0, early_multi_hits_ = 0;
    uint64_t early_multi_predicted_ = 0, early_multi_actual_ = 0;
    uint64_t early_multi_hints_ = 0, early_multi_started_ = 0;
    std::vector<std::vector<std::array<int, 8>>> early_multi_rows_;

    struct WholeMoeRouteSink {
        std::array<int, 8> *experts;
        std::array<float, 8> *weights;
        int row_base;
    };
    WholeMoeRouteSink *whole_moe_route_sink_ = nullptr;

    void route_trace(int layer, const std::vector<int> &selected, const std::vector<float> &scores);
    void early_route(int layer, const float *input);
    void early_route_multi(int layer, const float *input, int tokens);
    void load_cct();
    void cct_prefetch(int layer);
    void seam(int layer, int tag, const float *device, int count) {
        if (!seam_dump_probed_) {
            seam_dump_probed_ = true;
            if (const char *path = std::getenv("INSIGNIA_GLM53_SEAM_DUMP")) {
                seam_dump_ = std::fopen(path, "wb");
                if (const char *layer_arg = std::getenv("INSIGNIA_GLM53_SEAM_LAYER"))
                    seam_layer_ = std::atoi(layer_arg);
            }
        }
        if (!seam_dump_ || (seam_layer_ >= 0 && layer != seam_layer_)) return;
        seam_host_.assign(size_t(count), 0.0f);
        check(cudaMemcpy(seam_host_.data(), device, size_t(count) * sizeof(float),
                         cudaMemcpyDeviceToHost), "download seam");
        const int32_t header[4] = {int32_t(token_index_), layer, tag, count};
        std::fwrite(header, sizeof(header), 1, seam_dump_);
        std::fwrite(seam_host_.data(), sizeof(float), size_t(count), seam_dump_);
        std::fflush(seam_dump_);
    }
    std::unique_ptr<insignia::glm53::DFlash2Drafter> df_;
    bool dflash2_on_ = false;
    int df_approx_topm_ = 0;
    float df_approx_mass_ = 0.0f;
    int df_approx_min_k_ = 3, df_approx_max_k_ = 8;
    int df_cache_route_k_ = 0, df_cache_route_retain_ = 7;
    int df_cache_guard_retain_ = 8;
    int df_cache_joint_options_ = 0;
    bool df_cache_mask_search_ = true, df_cache_mask_verify_ = false;
    float df_cache_route_regret_ = 0.0025f;
    uint64_t df_cache_route_rows_ = 0, df_cache_route_changed_ = 0;
    uint64_t df_cache_route_substitutions_ = 0;
    double df_cache_route_regret_sum_ = 0.0, df_cache_route_regret_max_ = 0.0;
    int64_t df_cache_route_disk_saved_ = 0, df_cache_route_h2d_saved_ = 0;
    uint64_t df_cache_joint_groups_ = 0, df_cache_joint_baseline_union_ = 0;
    uint64_t df_cache_joint_selected_union_ = 0;
    int64_t df_cache_joint_disk_saved_ = 0, df_cache_joint_h2d_saved_ = 0;
    uint64_t df_cache_selector_calls_ = 0;
    uint64_t df_cache_selector_mask_calls_ = 0;
    double df_cache_selector_seconds_ = 0.0;
    float df_logit_guard_margin_ = 0.0f;
    bool df_logit_guard_prefix_ = true;
    float df_calibration_guard_js_ = 0.0f;
    float df_uncertainty_top1_p_ = 0.0f, df_uncertainty_top1_drop_ = 0.0f;
    int df_uncertainty_guard_k_ = 8, df_uncertainty_hold_rounds_ = 0;
    int df_uncertainty_hold_left_ = 0;
    float df_retry_top1_drop_ = 0.0f;
    bool df_retry_replay_ = false;
    uint64_t df_retry_rounds_ = 0, df_retry_triggered_rounds_ = 0;
    double df_retry_drop_sum_ = 0.0, df_retry_drop_max_ = 0.0;
    uint64_t df_draft_round_ = 0;
    std::vector<int> df_diagnostic_exact_rounds_;
    std::array<uint8_t, kMaxVerify> df_logit_guard_exact_{};
    std::array<uint8_t, kMaxVerify> df_logit_guard_k_{};
    uint64_t df_logit_guard_rows_ = 0, df_logit_guarded_rows_ = 0;
    uint64_t df_calibration_guard_rounds_ = 0, df_calibration_guarded_rounds_ = 0;
    double df_calibration_js_sum_ = 0.0, df_calibration_js_max_ = 0.0;
    std::vector<float> df_prior_logits_host_;
    bool df_approx_renorm_ = false;
    uint64_t df_approx_rows_ = 0, df_approx_slots_ = 0;
    uint64_t df_approx_union_ = 0, df_approx_exact_union_ = 0;
    std::array<uint64_t, 9> df_approx_k_hist_{};
    std::string moe_metrics_path_;
    FILE *moe_metrics_ = nullptr;
    std::string falsifier_trace_path_;
    FILE *falsifier_trace_ = nullptr;
    bool falsifier_feature_only_ = false;
    std::vector<float> moe_metrics_expert_, moe_metrics_exact_, moe_metrics_input_;
    float *df_logits_host_ = nullptr, *df_hp_host_ = nullptr;
    float *df_retry_logits_host_ = nullptr;
    std::unique_ptr<Q8Index> q8_index_;
    std::unique_ptr<Q8Stager> q8_stager_;
    int hidden_ = 0, kda_width_ = 0, kda_heads_ = 0, kda_head_dim_ = 0, f_a_rows_ = 0;
    int q_a_rows_ = 0, q_b_rows_ = 0, kv_a_rows_ = 0, kv_b_rows_ = 0;
    int mla_heads_ = 0, mla_head_dim_ = 0, mla_layers_ = 0, kda_layers_ = 0;
    int moe_experts_ = 0, moe_topk_ = 0, moe_intermediate_ = 0;
    int dense_intermediate_ = 0, shared_intermediate_ = 0;
    bool nvfp4_experts_ = false, q3_experts_ = false;
    bool df_logit_guard_on() const {
        return df_logit_guard_margin_ > 0.0f || df_calibration_guard_js_ > 0.0f ||
            df_uncertainty_top1_p_ > 0.0f || df_uncertainty_top1_drop_ > 0.0f ||
            df_retry_top1_drop_ > 0.0f || !df_diagnostic_exact_rounds_.empty();
    }
    void guard_dflash_row(int row, int k) {
        require(row >= 0 && row < kMaxVerify && k >= 1 && k <= 8,
                "DFlash guard row/k is out of range");
        df_logit_guard_exact_[size_t(row)] = 1;
        df_logit_guard_k_[size_t(row)] = uint8_t(std::max<int>(
            df_logit_guard_k_[size_t(row)], k));
    }
    void retain_df_prior_logits(const float *device_logits, const float *host_logits = nullptr);
    std::vector<std::string> layer_types_, mlp_types_;
    std::unordered_map<std::string, std::vector<float>> host_cache_;
    std::unordered_map<std::string, std::unique_ptr<DeviceBuffer<float>>> f32_cache_;
    std::vector<uint8_t> is_mla_, is_sparse_;
    std::vector<int> mla_slot_;
    DeviceBuffer<float> streams_a_, streams_b_, collapsed_, normalized_, attention_, ffn_;
    DeviceBuffer<float> post_, comb_, routed_;
    DeviceBuffer<uint8_t> mhc_workspace_;
    DeviceBuffer<float> q_, k_, v_, gate_8192_, core_, projected_8192_;
    DeviceBuffer<float> small_a_, small_b_, kv_, mla_query_, mla_output_;
    DeviceBuffer<float> gate_, up_, activation_, router_, beta_;
    DeviceBuffer<float> logits_;
    DeviceBuffer<int> finite_;
    DeviceBuffer<float> kda_states_, conv_history_;
    // Latent-cache MLA: compressed kv_lora-wide latents (FP8 e4m3 with
    // per-token scales, or FP32 for the INSIGNIA_GLM53_KV_FP8=0 A/B path),
    // per-head absorb weights projected out of kv_b_proj, and flash-decode
    // merge scratch [heads, tiles, latent+2].
    DeviceBuffer<uint8_t> mla_latent_u8_, mla_qeff_u8_;
    DeviceBuffer<float> mla_latent_f32_, mla_latent_scale_, mla_partial_, w_uk_, w_uv_;
    DeviceBuffer<float> mla_qeff_scale_, mla_qeff_f32_, mla_prefix_partial_;
    DeviceBuffer<float> mla_keys_, mla_values_, mla_prefix_latent_, mla_prefix_kv_;
    std::vector<Q8Stager::ResidentView> mla_absorb_fp8_views_;
    bool kv_fp8_ = true, mla_legacy_ = false, mla_fp8_absorb_ = false;
    bool mla_prefix_reconstruct_ = false;
    bool mla_cross_head_fp8_ = false;
    bool mla_prefix_parallel_ = false;
    int mla_prefix_kv_slot_ = -1, mla_prefix_kv_positions_ = 0;
    size_t absorb_per_layer_ = 0;
    const uint32_t *chunk_bf16_weights_ = nullptr;
    DeviceBuffer<float> c_stream_a_, c_stream_b_, c_collapsed_, c_normalized_, c_attn_, c_ffn_;
    DeviceBuffer<float> c_routed_, c_expert_out_, c_q_, c_k_, c_v_, c_gate_, c_core_, c_proj_, c_small_;
    DeviceBuffer<float> c_kv_, c_mlaq_, c_mlao_, c_gateu_, c_up_, c_act_, c_router_;
    DeviceBuffer<float> c_post_, c_comb_, c_beta_;
    DeviceBuffer<uint8_t> nv_workspace_4096_, nv_workspace_2048_, q8_workspace_;
    DeviceBuffer<uint8_t> iq_workspace_4096_, iq_workspace_2048_;
    // MTP speculative decoding state (INSIGNIA_GLM53_MTP=K, 0 disables).
    int mtp_draft_total_ = 0;
    int mtp_variant_ = 0;  // 0: hnorm(mean-of-streams) 1: hnorm(final-normed) 2: swapped eh concat
    bool mtp_bf16_ = false;
    bool kda_archive_ = false, verify_populate_ = false;
    DeviceBuffer<float> last_avg_, last_normed_;
    DeviceBuffer<float> mtp_eh_in_, mtp_hidden_, mtp_attn_, mtp_moe_, mtp_recycle_, mtp_logits_;
    DeviceBuffer<float> verify_means_, verify_normed_, verify_logits_;
    DeviceBuffer<int> verify_arg_;
    DeviceBuffer<float> kda_snap_, conv_snap_, kda_arch_;
    DeviceBuffer<float> df_prior_logits_device_;
    bool df_prior_logits_ready_ = false;
    DeviceBuffer<uint8_t> df_logit_metrics_workspace_, df_logit_row_stats_workspace_;
    DeviceBuffer<insignia::glm53::LogitMetrics> df_logit_metrics_device_;
    DeviceBuffer<insignia::glm53::LogitRowStats> df_logit_row_stats_device_;
    insignia::glm53::LogitMetrics *df_logit_metrics_host_ = nullptr;
    insignia::glm53::LogitRowStats *df_logit_row_stats_host_ = nullptr;
    std::vector<int> kda_row_;
    // CCT: per layer, the byte offset of its (layer, layer+1) table (or -1),
    // plus the flat top-8-per-expert id table and its header constants.
    std::vector<int64_t> cct_offset_;
    std::vector<uint16_t> cct_;
    int cct_experts_ = 0, cct_topk_ = 0;
};

// The FP32 sidecars are a few kilobytes each but are re-read from disk at
// every layer of every token without these caches; keeping them host- and
// device-resident removes several hundred pread calls per token.
const std::vector<float> &Runner::host_f32(std::string_view name) {
    const std::string key(name);
    const auto found = host_cache_.find(key);
    if (found != host_cache_.end()) return found->second;
    return host_cache_.emplace(key, read_widened(model_, name)).first->second;
}

const float *Runner::device_f32(std::string_view name) {
    const std::string key(name);
    const auto found = f32_cache_.find(key);
    if (found != f32_cache_.end()) return found->second->get();
    std::vector<float> host = host_f32(name);
    auto buffer = std::make_unique<DeviceBuffer<float>>(host.size());
    check(cudaMemcpy(buffer->get(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice),
          "upload cached FP32 tensor");
    const float *pointer = buffer->get();
    f32_cache_.emplace(key, std::move(buffer));
    return pointer;
}

void Runner::linear(std::string_view name, const float *input, float *output, int rows, int cols) {
    const TensorLocation &weight = model_.tensor(name);
    require(weight.shape.size() == 2 && weight.shape[0] == uint32_t(rows) &&
                weight.shape[1] == uint32_t(cols),
            "wrong linear geometry for " + std::string(name));
    if (q8_index_) {
        // The 8-bit cache holds fabricated layer-45 entries (it lists
        // shared-expert tensors that do not exist in the MTP layer), so its
        // whole layer-45 region is untrustworthy: stream BF16 instead.
        const bool layer45 = name.find("layers.45.") != std::string_view::npos;
        if (mtp_bf16_ && layer45) {
            const uint32_t *device = reinterpret_cast<const uint32_t *>(stager_.load(weight));
            insignia::bf16_gemv_v2(device, input, output, rows, cols);
            debug_linear_output(name, output, rows);
            return;
        }
        if (const Q8TensorLocation *q8 = q8_index_->find(name)) {
            require(q8->rows == uint32_t(rows) && q8->cols == uint32_t(cols),
                    "wrong Q8 linear geometry for " + std::string(name));
            if (std::getenv("INSIGNIA_GLM53_CHECK_LINEAR"))
                debug_linear_output(std::string(name) + " input", input, cols);
            q8_stager_->load(name, *q8, 0, uint32_t(rows));
            if (q8_index_->format() == Cache8Format::fp8_e4m3)
                check(insignia::glm53::fp8_tc_gemv(q8_stager_->weight_bytes(), q8_stager_->scales(),
                      input, output, rows, cols, q8_workspace_), "FP8 linear");
            else
                check(insignia::glm53::q8_gemv(q8_stager_->weights(), q8_stager_->scales(),
                      input, output, rows, cols, q8_workspace_), "Q8 linear");
            debug_linear_output(name, output, rows);
            return;
        }
    }
    require(weight.type != TensorType::external_fp8,
            "external FP8 tensor is missing from cache: " + std::string(name));
    require(weight.type == TensorType::bf16,
            "wrong BF16 fallback type for " + std::string(name));
    const uint32_t *device = reinterpret_cast<const uint32_t *>(stager_.load(weight));
    insignia::bf16_gemv_v2(device, input, output, rows, cols);
}

// Two equal-row matrices over one activation: one quantize pass and one
// paired launch. Falls back to plain linear() when either matrix is not
// FP8-resident (streaming slots alias, so the pair would read clobbered
// weights).
void Runner::linear_pair(std::string_view name_a, std::string_view name_b,
                         const float *input, float *out_a, float *out_b, int rows, int cols) {
    if (q8_index_ && q8_index_->format() == Cache8Format::fp8_e4m3) {
        const Q8TensorLocation *q8_a = q8_index_->find(name_a);
        const Q8TensorLocation *q8_b = q8_index_->find(name_b);
        if (q8_a && q8_b && q8_a->rows == uint32_t(rows) && q8_a->cols == uint32_t(cols) &&
            q8_b->rows == q8_a->rows && q8_b->cols == q8_a->cols) {
            q8_stager_->load(name_a, *q8_a, 0, uint32_t(rows));
            const uint8_t *weights_a = q8_stager_->weight_bytes();
            const uint16_t *scales_a = q8_stager_->scales();
            q8_stager_->load(name_b, *q8_b, 0, uint32_t(rows));
            const uint8_t *weights_b = q8_stager_->weight_bytes();
            if (weights_b != weights_a) {
                if (std::getenv("INSIGNIA_GLM53_CHECK_LINEAR"))
                    debug_linear_output(std::string(name_a) + " input", input, cols);
                check(insignia::glm53::fp8_tc_gemv2(weights_a, scales_a, weights_b,
                      q8_stager_->scales(), input, out_a, out_b, rows, cols,
                      q8_workspace_), "FP8 pair");
                debug_linear_output(name_a, out_a, rows);
                debug_linear_output(name_b, out_b, rows);
                return;
            }
        }
    }
    linear(name_a, input, out_a, rows, cols);
    linear(name_b, input, out_b, rows, cols);
}

// The prefill workhorse: inputs is [tokens,cols] and outputs [tokens,rows].
// Each weight chunk is staged once and multiplied against every token, so a
// prompt chunk costs one streaming pass per matrix instead of one per token.
void Runner::linear_multi(std::string_view name, const float *inputs, float *outputs,
                          int tokens, int rows, int cols) {
    require(tokens >= 1 && tokens <= kMaxChunk(), "prefill chunk out of range");
    const TensorLocation &weight = model_.tensor(name);
    require(weight.shape.size() == 2 && weight.shape[0] == uint32_t(rows) &&
                weight.shape[1] == uint32_t(cols),
            "wrong multi-token linear geometry for " + std::string(name));
    const auto run_chunk = [&](int row, int chunk_rows) {
        for (int token = 0; token < tokens; ++token)
            insignia::bf16_gemv_v2(chunk_bf16_weights_, inputs + size_t(token) * cols,
                                   outputs + size_t(token) * rows + row, chunk_rows, cols);
    };
    if (q8_index_) {
        if (const Q8TensorLocation *q8 = q8_index_->find(name)) {
            require(q8->rows == uint32_t(rows) && q8->cols == uint32_t(cols),
                    "wrong multi-token Q8 linear geometry");
            const uint64_t capacity = Q8Stager::kWeightCapacity / cols;
            for (uint64_t row = 0; row < uint32_t(rows); row += capacity) {
                const uint32_t chunk_rows = uint32_t(std::min<uint64_t>(capacity, rows - row));
                q8_stager_->load(name, *q8, uint32_t(row), chunk_rows);
                static const bool scalar_fp8 =
                    std::getenv("INSIGNIA_GLM53_SCALAR_FP8_PREFILL") != nullptr;
                if (q8_index_->format() == Cache8Format::fp8_e4m3 && !scalar_fp8) {
                    check(insignia::glm53::fp8_tc_gemv_batch(
                          q8_stager_->weight_bytes(), q8_stager_->scales(), inputs,
                          outputs + row, tokens, int(chunk_rows), cols, rows,
                          q8_workspace_), "multi FP8 linear");
                } else for (int token = 0; token < tokens; ++token) {
                    if (q8_index_->format() == Cache8Format::fp8_e4m3)
                        check(insignia::glm53::fp8_tc_gemv(q8_stager_->weight_bytes(), q8_stager_->scales(),
                              inputs + size_t(token) * cols,
                              outputs + size_t(token) * rows + row, int(chunk_rows), cols,
                              q8_workspace_), "multi FP8 linear");
                    else
                        check(insignia::glm53::q8_gemv(q8_stager_->weights(), q8_stager_->scales(),
                              inputs + size_t(token) * cols,
                              outputs + size_t(token) * rows + row, int(chunk_rows), cols,
                              q8_workspace_), "multi Q8 linear");
                }
            }
            return;
        }
    }
    require(weight.type != TensorType::external_fp8,
            "external FP8 tensor is missing from multi-token cache: " + std::string(name));
    require(weight.type == TensorType::bf16,
            "wrong multi-token BF16 fallback type for " + std::string(name));
    const uint64_t row_bytes = uint64_t(cols) * 2;
    const uint64_t capacity = TensorStager::kCapacity / row_bytes;
    if (stager_.is_resident(weight)) {
        chunk_bf16_weights_ = reinterpret_cast<const uint32_t *>(stager_.load(weight));
        run_chunk(0, rows);
        return;
    }
    for (uint64_t row = 0; row < uint32_t(rows); row += capacity) {
        const uint64_t chunk_rows = std::min<uint64_t>(capacity, rows - row);
        chunk_bf16_weights_ = reinterpret_cast<const uint32_t *>(
            stager_.load(weight, row * row_bytes, chunk_rows * row_bytes));
        run_chunk(int(row), int(chunk_rows));
    }
}

void Runner::linear_rows(std::string_view name, const TensorLocation &weight, uint64_t row, int rows,
                         const float *input, float *output, int cols) {
    require(weight.shape.size() == 2 && weight.shape[1] == uint32_t(cols) &&
                row + rows <= weight.shape[0],
            "wrong chunked linear geometry");
    if (q8_index_) {
        if (const Q8TensorLocation *q8 = q8_index_->find(name)) {
            require(q8->rows == weight.shape[0] && q8->cols == uint32_t(cols),
                    "wrong chunked Q8 linear geometry");
            q8_stager_->load(name, *q8, uint32_t(row), uint32_t(rows));
            if (q8_index_->format() == Cache8Format::fp8_e4m3)
                check(insignia::glm53::fp8_tc_gemv(q8_stager_->weight_bytes(), q8_stager_->scales(),
                      input, output, rows, cols, q8_workspace_), "chunked FP8 linear");
            else
                check(insignia::glm53::q8_gemv(q8_stager_->weights(), q8_stager_->scales(),
                      input, output, rows, cols, q8_workspace_), "chunked Q8 linear");
            debug_linear_output(name, output, rows);
            return;
        }
    }
    require(weight.type != TensorType::external_fp8,
            "external FP8 tensor is missing from chunked cache: " + std::string(name));
    require(weight.type == TensorType::bf16, "wrong chunked BF16 fallback type");
    const uint64_t row_bytes = uint64_t(cols) * 2;
    const uint32_t *device = reinterpret_cast<const uint32_t *>(
        stager_.load(weight, row * row_bytes, uint64_t(rows) * row_bytes));
    insignia::bf16_gemv_v2(device, input, output, rows, cols);
}

void Runner::rms(std::string_view name, const float *input, float *output, int rows, int cols) {
    const TensorLocation &weight = model_.tensor(name);
    require(weight.type == TensorType::bf16 && weight.bytes == uint64_t(cols) * 2,
            "wrong RMSNorm geometry for " + std::string(name));
    const uint16_t *device = reinterpret_cast<const uint16_t *>(stager_.load(weight));
    launch_rms(input, device, output, rows, cols);
}

void Runner::mhc(std::string_view stem, std::string_view norm,
                 const float *streams, float *normalized) {
    const float *base = device_f32(std::string(stem) + "_base");
    const float *scale = device_f32(std::string(stem) + "_scale");
    const TensorLocation &fn = model_.tensor(std::string(stem) + "_fn");
    require(fn.type == TensorType::bf16 &&
            fn.shape == std::vector<uint32_t>({24, uint32_t(kStreams) * hidden_}),
            "wrong mHC projection geometry");
    const uint16_t *device_fn = reinterpret_cast<const uint16_t *>(stager_.load(fn));
    const TensorLocation &weight = model_.tensor(norm);
    require(weight.type == TensorType::bf16 && weight.bytes == uint64_t(hidden_) * 2,
            "wrong RMSNorm geometry for " + std::string(norm));
    const uint16_t *device_weight = reinterpret_cast<const uint16_t *>(stager_.load(weight));
    check(insignia::glm53::mhc_analyze(device_fn, base, scale, streams, device_weight,
        post_, comb_, normalized, mhc_workspace_.get(), kStreams * hidden_), "mHC analyze");
}

void Runner::kda(int layer, const float *input, float *output, int position) {
    const std::string stem = layer_stem(layer) + "self_attn.";
    const int row = kda_row_[size_t(layer)];
    require(row >= 0, "KDA state requested for a non-KDA layer");
    float *history = conv_history_.get() + size_t(row) * 9 * kda_width_;
    const bool conv_fp32 = model_.tensor(stem + "q_conv1d.weight").type == TensorType::f32;
                    linear(stem + "q_proj.weight", input, q_, kda_width_, hidden_);
                    const void *conv_q = stager_.load(stem + "q_conv1d.weight");
                    linear(stem + "k_proj.weight", input, k_, kda_width_, hidden_);
                    const void *conv_k = stager_.load(stem + "k_conv1d.weight");
                    linear(stem + "v_proj.weight", input, v_, kda_width_, hidden_);
                    const void *conv_v = stager_.load(stem + "v_conv1d.weight");
                    check(insignia::glm53::kda_conv_silu3(q_, k_, v_, conv_q, conv_k, conv_v,
                          history, position, kda_width_, conv_fp32), "KDA qkv convolution");

    linear(stem + "f_a_proj.weight", input, small_a_, f_a_rows_, hidden_);
    linear(stem + "f_b_proj.weight", small_a_, gate_8192_, kda_width_, f_a_rows_);
    const float *dt_bias = device_f32(stem + "dt_bias");
    const float *a_log = device_f32(stem + "A_log");
    linear(stem + "b_proj.weight", input, beta_, kda_heads_, hidden_);
    kda_gate_kernel<<<32, 256>>>(gate_8192_, dt_bias, a_log, beta_, kda_heads_, kda_width_);
    check(cudaGetLastError(), "KDA gate launch");

    float *state = kda_states_.get() + size_t(row) * kda_width_ * kda_head_dim_;
    check(insignia::glm53::kda_decode(state, q_, k_, v_, gate_8192_, beta_, core_,
          kda_heads_, kda_head_dim_), "KDA recurrence");
    linear(stem + "g_a_proj.weight", input, small_a_, f_a_rows_, hidden_);
    linear(stem + "g_b_proj.weight", small_a_, gate_8192_, kda_width_, f_a_rows_);
    const uint16_t *norm = reinterpret_cast<const uint16_t *>(stager_.load(stem + "o_norm.weight"));
    kda_output_kernel<<<kda_heads_, kda_head_dim_>>>(core_, gate_8192_, norm, projected_8192_);
    check(cudaGetLastError(), "KDA output norm launch");
    linear(stem + "o_proj.weight", projected_8192_, output, hidden_, kda_width_);
}

bool Runner::bind_mla_absorb_fp8() {
    // The diagnostic exports materialized W_uk/W_uv arrays; retain its old
    // contract instead of silently changing the dump surface.
    if (!q8_index_ || !q8_stager_ ||
        q8_index_->format() != Cache8Format::fp8_e4m3 ||
        std::getenv("INSIGNIA_GLM53_MLA_BF16_ABSORB") ||
        std::getenv("INSIGNIA_GLM53_MLA_DUMP"))
        return false;
    mla_absorb_fp8_views_.assign(size_t(mla_layers_), {});
    for (int slot = 0; slot < mla_layers_; ++slot) {
        const std::string name =
            layer_stem(mla_slot_[size_t(slot)]) + "self_attn.kv_b_proj.weight";
        const Q8TensorLocation *quantized = q8_index_->find(name);
        if (!quantized || quantized->rows != uint32_t(kv_b_rows_) ||
            quantized->cols != uint32_t(kv_a_rows_))
            return false;
        if (!q8_stager_->is_resident(name) && !q8_stager_->try_pin(name, *quantized))
            return false;
        if (!q8_stager_->resident_view(name, &mla_absorb_fp8_views_[size_t(slot)]))
            return false;
    }
    return true;
}

// Extracts per-head absorb weights from kv_b_proj. With the normal dense FP8
// cache, widen the exact e4m3*FP16 values used by fp8_tc_gemv so absorbed MLA
// remains the same quantized model. Raw BF16 is only the cache-less fallback.
// The store's kv_b output rows are h*(2*head_dim) + j for K and
// h*(2*head_dim) + head_dim + j for V, each [latent] wide.  W_uk[h] is the
// head's K block and W_uv[h] its V block, both copied straight into
// [head_dim, latent] rows (row j-major matches the kernel index math).
void Runner::extract_mla_absorb() {
    const size_t heads = size_t(mla_heads_), head_dim = size_t(mla_head_dim_);
    const size_t latent = size_t(kv_a_rows_);
    std::vector<float> host_uk(absorb_per_layer_), host_uv(absorb_per_layer_);
    std::array<float, 256> fp8_table{};
    for (int value = 0; value < 256; ++value)
        fp8_table[size_t(value)] = from_fp8_e4m3_host(uint8_t(value));
    const bool use_fp8 = q8_index_ && q8_index_->format() == Cache8Format::fp8_e4m3 &&
                         !std::getenv("INSIGNIA_GLM53_MLA_BF16_ABSORB");
    for (int slot = 0; slot < mla_layers_; ++slot) {
        const std::string name =
            layer_stem(mla_slot_[size_t(slot)]) + "self_attn.kv_b_proj.weight";
        const TensorLocation &location = model_.tensor(name);
        if (use_fp8) {
            const Q8TensorLocation *quantized = q8_index_->find(name);
            require(quantized && quantized->rows == uint32_t(kv_b_rows_) &&
                    quantized->cols == uint32_t(latent),
                    "FP8 cache misses MLA absorb tensor " + name);
            std::vector<uint8_t> weights(size_t(quantized->weight_bytes));
            std::vector<uint16_t> scales(size_t(quantized->scale_bytes) / sizeof(uint16_t));
            q8_index_->read_rows(*quantized, 0, quantized->rows,
                                 weights.data(), scales.data());
            constexpr size_t group = insignia::glm53::kQ8GroupSize;
            const size_t groups = latent / group;
            const auto dequant_row = [&](size_t row, float *destination) {
                for (size_t g = 0; g < groups; ++g) {
                    const float scale = from_f16_host(scales[row * groups + g]);
                    for (size_t within = 0; within < group; ++within) {
                        const size_t column = g * group + within;
                        destination[column] = fp8_table[weights[row * latent + column]] * scale;
                    }
                }
            };
            for (size_t h = 0; h < heads; ++h)
                for (size_t j = 0; j < head_dim; ++j) {
                    dequant_row(h * 2 * head_dim + j,
                                host_uk.data() + (h * head_dim + j) * latent);
                    dequant_row(h * 2 * head_dim + head_dim + j,
                                host_uv.data() + (h * head_dim + j) * latent);
                }
        } else {
            require(location.type == TensorType::bf16 &&
                        location.bytes == size_t(kv_b_rows_) * latent * sizeof(uint16_t),
                    "unexpected BF16 kv_b_proj tensor size");
            std::vector<uint16_t> source(size_t(kv_b_rows_) * latent);
            model_.read(location, source.data());
            for (size_t h = 0; h < heads; ++h)
                for (size_t j = 0; j < head_dim; ++j)
                    for (size_t column = 0; column < latent; ++column) {
                        host_uk[(h * head_dim + j) * latent + column] = from_bf16_host(
                            source[(h * 2 * head_dim + j) * latent + column]);
                        host_uv[(h * head_dim + j) * latent + column] = from_bf16_host(
                            source[(h * 2 * head_dim + head_dim + j) * latent + column]);
                    }
        }
        check(cudaMemcpy(w_uk_.get() + size_t(slot) * absorb_per_layer_, host_uk.data(),
                         host_uk.size() * sizeof(float), cudaMemcpyHostToDevice),
              "upload W_uk");
        check(cudaMemcpy(w_uv_.get() + size_t(slot) * absorb_per_layer_, host_uv.data(),
                         host_uv.size() * sizeof(float), cudaMemcpyHostToDevice),
              "upload W_uv");
    }
}

void Runner::reconstruct_mla_prefix(int slot, int positions, int dirty_base) {
    require(mla_prefix_reconstruct_ && slot >= 0 && slot < mla_layers_ &&
            positions >= 1 && positions <= kLegacyMlaContext &&
            dirty_base >= 0 && dirty_base < positions,
            "invalid MLA exact-prefix reconstruction request");
    const Q8Stager::ResidentView &view = mla_absorb_fp8_views_[size_t(slot)];
    const size_t latent_stride = size_t(kLegacyMlaContext) * kv_a_rows_;
    const float *latents = mla_prefix_latent_.get() + size_t(slot) * latent_stride;
    // Full-prompt layer-major prefill visits consecutive chunks of one MLA
    // layer. Preserve the already reconstructed rows in the shared scratch;
    // decode/verify naturally changes slot and rebuilds from zero.
    int first = 0;
    if (mla_prefix_kv_slot_ == slot)
        first = std::min(dirty_base, mla_prefix_kv_positions_);
    for (int base = first; base < positions; base += kMaxChunk()) {
        const int count = std::min(kMaxChunk(), positions - base);
        check(insignia::glm53::fp8_tc_gemv_batch(
              view.weights, view.scales, latents + size_t(base) * kv_a_rows_,
              mla_prefix_kv_.get() + size_t(base) * kv_b_rows_, count,
              kv_b_rows_, kv_a_rows_, kv_b_rows_, q8_workspace_),
              "reconstruct exact MLA prefix K/V");
    }
    mla_prefix_kv_slot_ = slot;
    mla_prefix_kv_positions_ = positions;
}

void Runner::mla(int layer, const float *input, float *output, int position) {
    const std::string stem = layer_stem(layer) + "self_attn.";
    linear(stem + "q_a_proj.weight", input, small_a_, q_a_rows_, hidden_);
    rms(stem + "q_a_layernorm.weight", small_a_, small_b_, 1, q_a_rows_);
    linear(stem + "q_b_proj.weight", small_b_, mla_query_, q_b_rows_, q_a_rows_);
    linear(stem + "kv_a_proj_with_mqa.weight", input, small_a_, kv_a_rows_, hidden_);
    rms(stem + "kv_a_layernorm.weight", small_a_, small_b_, 1, kv_a_rows_);
    const int slot = int(std::find(mla_slot_.begin(), mla_slot_.end(), layer) - mla_slot_.begin());
    if (mla_legacy_) {
        require(position < kLegacyMlaContext, "legacy MLA context exceeds 256 tokens");
        linear(stem + "kv_b_proj.weight", small_b_, kv_, kv_b_rows_, kv_a_rows_);
        const size_t expanded_stride = size_t(kLegacyMlaContext) * q_b_rows_;
        check(insignia::glm53::mla_decode(mla_query_, kv_,
              mla_keys_.get() + size_t(slot) * expanded_stride,
              mla_values_.get() + size_t(slot) * expanded_stride,
              mla_output_, position, mla_heads_, mla_head_dim_), "legacy MLA attention");
        if (std::getenv("INSIGNIA_GLM53_MLA_DUMP") && layer == mla_slot_.front() &&
            position >= 4 && position <= 6) {
            const std::filesystem::path dir = std::getenv("INSIGNIA_GLM53_MLA_DUMP");
            std::filesystem::create_directories(dir);
            const auto save_dev = [&](const std::string &name, const void *device, size_t bytes) {
                std::vector<uint8_t> host(bytes);
                check(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost), name.c_str());
                std::FILE *file = std::fopen((dir / name).string().c_str(), "wb");
                std::fwrite(host.data(), 1, bytes, file);
                std::fclose(file);
            };
            const std::string tag = "legacy.dec" + std::to_string(position) + ".";
            save_dev(tag + "q.bin", mla_query_.get(), size_t(2) * mla_head_dim_ * sizeof(float));
            save_dev(tag + "out.bin", mla_output_.get(), size_t(2) * mla_head_dim_ * sizeof(float));
            save_dev(tag + "keys.bin", mla_keys_.get() + size_t(slot) * expanded_stride,
                     size_t(position + 1) * q_b_rows_ * sizeof(float));
            save_dev(tag + "values.bin", mla_values_.get() + size_t(slot) * expanded_stride,
                     size_t(position + 1) * q_b_rows_ * sizeof(float));
        }
        linear(stem + "o_proj.weight", mla_output_, output, hidden_, q_b_rows_);
        return;
    }
    const size_t layer_stride = size_t(kMaxContext()) * kv_a_rows_;
    uint8_t *cache_u8 = kv_fp8_
        ? mla_latent_u8_.get() + size_t(slot) * layer_stride
        : nullptr;
    float *cache_scale = mla_latent_scale_.get() + size_t(slot) * kMaxContext() *
                         insignia::glm53::kMlaLatentGroups;
    float *cache_f32 = kv_fp8_
        ? nullptr
        : mla_latent_f32_.get() + size_t(slot) * layer_stride;
    if (position < kLegacyMlaContext) {
        check(insignia::glm53::mla_store_latent(
              small_b_.get(), cache_u8, cache_scale, cache_f32, 1, position,
              kv_a_rows_), "MLA latent shadow store");
        const size_t prefix_stride = size_t(kLegacyMlaContext) * kv_a_rows_;
        if (mla_prefix_reconstruct_ || mla_cross_head_fp8_) {
            check(cudaMemcpyAsync(
                  mla_prefix_latent_.get() + size_t(slot) * prefix_stride +
                      size_t(position) * kv_a_rows_,
                  small_b_.get(), size_t(kv_a_rows_) * sizeof(float),
                  cudaMemcpyDeviceToDevice), "save exact MLA prefix latent");
        }
        if (mla_prefix_reconstruct_) {
            reconstruct_mla_prefix(slot, position + 1, position);
            check(insignia::glm53::mla_decode_reconstructed(
                  mla_query_.get(), mla_prefix_kv_.get(), mla_output_, position,
                  mla_heads_, mla_head_dim_), "reconstructed exact MLA prefix attention");
        } else {
            linear(stem + "kv_b_proj.weight", small_b_, kv_, kv_b_rows_, kv_a_rows_);
            const size_t expanded_stride = size_t(kLegacyMlaContext) * q_b_rows_;
            check(insignia::glm53::mla_decode(
                  mla_query_.get(), kv_.get(),
                  mla_keys_.get() + size_t(slot) * expanded_stride,
                  mla_values_.get() + size_t(slot) * expanded_stride,
                  mla_output_, position, mla_heads_, mla_head_dim_),
                  "exact MLA prefix attention");
        }
    } else {
        if (mla_fp8_absorb_) {
            const Q8Stager::ResidentView &view = mla_absorb_fp8_views_[size_t(slot)];
            if (mla_cross_head_fp8_)
                check(insignia::glm53::mla_decode_latent_cross_head_fp8_absorb(
                      mla_query_.get(), small_b_.get(), cache_u8, cache_scale,
                      view.weights, view.scales, mla_qeff_u8_.get(),
                      mla_qeff_scale_.get(), mla_partial_.get(), mla_output_, position,
                      mla_heads_, mla_head_dim_, kv_a_rows_,
                      mla_prefix_latent_.get() + size_t(slot) *
                          kLegacyMlaContext * kv_a_rows_,
                      mla_qeff_f32_.get(),
                      mla_prefix_parallel_ ? mla_prefix_partial_.get() : nullptr,
                      mla_prefix_parallel_),
                      "MLA cross-head FP8 attention");
            else
                check(insignia::glm53::mla_decode_latent_fp8_absorb(
                      mla_query_.get(), small_b_.get(), cache_u8, cache_scale, cache_f32,
                      view.weights, view.scales, mla_partial_.get(), mla_output_, position,
                      mla_heads_, mla_head_dim_, kv_a_rows_), "MLA compact absorb attention");
        } else {
            check(insignia::glm53::mla_decode_latent(
                  mla_query_.get(), small_b_.get(), nullptr, cache_u8, cache_scale,
                  cache_f32, nullptr,
                  w_uk_.get() + size_t(slot) * absorb_per_layer_,
                  w_uv_.get() + size_t(slot) * absorb_per_layer_,
                  mla_partial_.get(), mla_output_, position,
                  mla_heads_, mla_head_dim_, kv_a_rows_), "MLA attention");
        }
    }
    if (std::getenv("INSIGNIA_GLM53_MLA_DUMP") && layer == mla_slot_.front() &&
        position >= 4 && position <= 6) {
        const std::filesystem::path dir = std::getenv("INSIGNIA_GLM53_MLA_DUMP");
        auto save_dev = [&](const std::string &name, const void *dev, size_t bytes) {
            std::vector<uint8_t> host(bytes);
            check(cudaMemcpy(host.data(), dev, bytes, cudaMemcpyDeviceToHost), name.c_str());
            std::FILE *file = std::fopen((dir / name).string().c_str(), "wb");
            std::fwrite(host.data(), 1, bytes, file);
            std::fclose(file);
        };
        const std::string tag = "dec" + std::to_string(position) + ".";
        save_dev(tag + "q.bin", mla_query_.get(), size_t(2) * mla_head_dim_ * sizeof(float));
        save_dev(tag + "latent.bin", small_b_.get(), size_t(kv_a_rows_) * sizeof(float));
        save_dev(tag + "out.bin", mla_output_.get(), size_t(2) * mla_head_dim_ * sizeof(float));
        save_dev(tag + "cache.bin",
                 kv_fp8_ ? (const void *)(mla_latent_u8_.get() + size_t(slot) * size_t(kMaxContext()) * kv_a_rows_)
                         : (const void *)(mla_latent_f32_.get() + size_t(slot) * size_t(kMaxContext()) * kv_a_rows_),
                 size_t(position + 1) * kv_a_rows_ * (kv_fp8_ ? 1 : 4));
        save_dev(tag + "scales.bin", mla_latent_scale_.get() + size_t(slot) * kMaxContext() *
                     insignia::glm53::kMlaLatentGroups,
                 size_t(position + 1) * insignia::glm53::kMlaLatentGroups * sizeof(float));
    }
    linear(stem + "o_proj.weight", mla_output_, output, hidden_, q_b_rows_);
}

void Runner::compute_mlp(std::string_view stem, const float *input, float *output, int intermediate) {
    linear_pair(std::string(stem) + "gate_proj.weight", std::string(stem) + "up_proj.weight",
                input, gate_, up_, intermediate, hidden_);
    launch_clamped_swiglu(gate_, up_, activation_, intermediate);
    linear(std::string(stem) + "down_proj.weight", activation_, output, hidden_, intermediate);
}

void Runner::dense_mlp(std::string_view stem, const float *input, float *output, int intermediate) {
    compute_mlp(stem, input, output, intermediate);
}

// Pre-attention route hint. The normal post-attention router remains the
// authority; this duplicate 288x4096 projection only buys the reader pool a
// head start while attention/KDA runs.
void Runner::early_route(int layer, const float *input) {
    const std::string stem = layer_stem(layer) + "mlp.";
    linear(stem + "gate.weight", input, router_, moe_experts_, hidden_);
    std::vector<float> logits(static_cast<size_t>(moe_experts_));
    check(cudaMemcpy(logits.data(), router_.get(), logits.size() * sizeof(float),
                     cudaMemcpyDeviceToHost), "download early router logits");
    const std::vector<float> &bias = host_f32(stem + "gate.e_score_correction_bias");
    std::vector<float> choice(static_cast<size_t>(moe_experts_));
    for (int expert = 0; expert < moe_experts_; ++expert)
        choice[size_t(expert)] =
            1.0f / (1.0f + std::exp(-logits[size_t(expert)])) + bias[size_t(expert)];
    std::vector<int> order(static_cast<size_t>(moe_experts_));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + moe_topk_, order.end(),
        [&](int left, int right) { return choice[size_t(left)] > choice[size_t(right)]; });
    for (int slot = 0; slot < moe_topk_; ++slot)
        early_routing_[size_t(layer)][size_t(slot)] = order[size_t(slot)];
    if (early_route_prefetch_ && prefetch_on_)
        expert_stager_->prefetch(layer, early_routing_[size_t(layer)].data(),
                                 early_route_prefetch_n_);
}

void Runner::early_route_multi(int layer, const float *input, int tokens) {
    const std::string stem = layer_stem(layer) + "mlp.";
    linear_multi(stem + "gate.weight", input, c_router_, tokens,
                 moe_experts_, hidden_);
    std::vector<float> logits(size_t(tokens) * moe_experts_);
    check(cudaMemcpy(logits.data(), c_router_.get(), logits.size() * sizeof(float),
                     cudaMemcpyDeviceToHost), "download early router logits (multi)");
    const std::vector<float> &bias = host_f32(stem + "gate.e_score_correction_bias");
    auto &rows = early_multi_rows_[size_t(layer)];
    rows.resize(size_t(tokens));
    std::vector<float> choice(static_cast<size_t>(moe_experts_));
    std::vector<int> order(static_cast<size_t>(moe_experts_));
    for (int token = 0; token < tokens; ++token) {
        const float *row = logits.data() + size_t(token) * moe_experts_;
        for (int expert = 0; expert < moe_experts_; ++expert)
            choice[size_t(expert)] =
                1.0f / (1.0f + std::exp(-row[expert])) + bias[size_t(expert)];
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + moe_topk_, order.end(),
            [&](int left, int right) { return choice[size_t(left)] > choice[size_t(right)]; });
        for (int slot = 0; slot < moe_topk_; ++slot)
            rows[size_t(token)][size_t(slot)] = order[size_t(slot)];
    }
    if (!early_multi_prefetch_ || !prefetch_on_) return;
    std::vector<int> picks;
    picks.reserve(size_t(std::min(early_multi_max_, tokens * early_multi_n_)));
    for (int rank = 0; rank < early_multi_n_ && int(picks.size()) < early_multi_max_; ++rank)
        for (int token = 0; token < tokens && int(picks.size()) < early_multi_max_; ++token) {
            const int expert = rows[size_t(token)][size_t(rank)];
            if (std::find(picks.begin(), picks.end(), expert) == picks.end())
                picks.push_back(expert);
        }
    early_multi_hints_ += picks.size();
    if (!picks.empty())
        early_multi_started_ +=
            expert_stager_->prefetch(layer, picks.data(), int(picks.size()));
}

// Routing-locality trace: one line per (token, sparse layer) after CPU routing,
// "token_index layer e0..e7 s0..s7\n" (s = sigmoid score of e). When the env
// var is unset the cost is one getenv probe plus a null check per layer.
void Runner::route_trace(int layer, const std::vector<int> &selected,
                         const std::vector<float> &scores) {
    if (!route_trace_probed_) {
        route_trace_probed_ = true;
        if (const char *path = std::getenv("INSIGNIA_GLM53_ROUTE_TRACE"))
            route_trace_ = std::fopen(path, "w");
    }
    if (!route_trace_)
        return;
    std::fprintf(route_trace_, "%ld %d", token_index_, layer);
    for (int slot = 0; slot < moe_topk_; ++slot)
        std::fprintf(route_trace_, " %d", selected[slot]);
    for (int slot = 0; slot < moe_topk_; ++slot)
        std::fprintf(route_trace_, " %.6e", scores[size_t(selected[slot])]);
    std::fputc('\n', route_trace_);
    std::fflush(route_trace_);
}

// CCT table: ST-MoE-style static cross-layer correlation. File format:
// magic "CCT01", u32 layers, u32 experts, u32 topk, then one
// experts*topk uint16 successor-id table per adjacent sparse layer pair,
// in layer order (layer i's table predicts layer i+1). INSIGNIA_GLM53_CCT
// names the file; no file means no cross-layer prefetch.
void Runner::load_cct() {
    cct_offset_.assign(model_.layers(), -1);
    cct_.clear();
    cct_experts_ = cct_topk_ = 0;
    const char *path = std::getenv("INSIGNIA_GLM53_CCT");
    if (!path)
        return;
    std::FILE *file = std::fopen(path, "rb");
    if (!file) {
        std::printf("cct: %s missing, cross-layer prefetch disabled\n", path);
        return;
    }
    char magic[5] = {};
    uint32_t header[3] = {};
    const bool ok = std::fread(magic, 4, 1, file) == 1 && std::strcmp(magic, "CCT0") == 0 &&
                    std::fread(header, sizeof(header), 1, file) == 1 &&
                    header[0] == model_.layers() && header[1] == uint32_t(moe_experts_) &&
                    header[2] >= 1 && header[2] <= 16;
    if (!ok) {
        std::printf("cct: bad table header, disabled\n");
        std::fclose(file);
        return;
    }
    cct_experts_ = int(header[1]);
    cct_topk_ = int(header[2]);
    for (int layer = 0; layer + 1 < int(model_.layers()); ++layer) {
        if (!is_sparse_[size_t(layer)] || !is_sparse_[size_t(layer) + 1])
            continue;
        cct_offset_[size_t(layer)] = std::ptrdiff_t(cct_.size());
        cct_.resize(cct_.size() + size_t(cct_experts_) * cct_topk_);
        if (std::fread(cct_.data() + cct_offset_[size_t(layer)],
                       sizeof(uint16_t) * size_t(cct_experts_) * cct_topk_, 1, file) != 1) {
            std::printf("cct: truncated table, disabled at layer %d\n", layer);
            cct_offset_.assign(model_.layers(), -1);
            cct_.clear();
            std::fclose(file);
            return;
        }
    }
    std::fclose(file);
    std::printf("cct: loaded %zu pair tables (experts %d, topk %d)\n",
                std::count_if(cct_offset_.begin(), cct_offset_.end(),
                              [](int64_t v) { return v >= 0; }),
                cct_experts_, cct_topk_);
}

void Runner::cct_prefetch(int layer) {
    if (size_t(layer) >= cct_offset_.size() || cct_offset_[size_t(layer)] < 0)
        return;
    // Union of the routed experts' successor lists. The cap defaults to 8:
    // at the measured ~2.4x table overfetch a 16-record union costs more disk
    // than it saves (see audits/mla-latent-session.md section 6).
    static const int cct_max = [] {
        const char *value = std::getenv("INSIGNIA_GLM53_CCT_MAX");
        return std::clamp(value ? std::atoi(value) : 8, 1, 16);
    }();
    int picks[16];
    int count = 0;
    const uint16_t *table = cct_.data() + cct_offset_[size_t(layer)];
    for (int slot = 0; slot < moe_topk_ && count < cct_max; ++slot) {
        const int routed = prev_routing_[size_t(layer)][size_t(slot)];
        if (routed < 0 || routed >= cct_experts_)
            continue;  // routing not yet computed for this token
        const uint16_t *row = table + size_t(routed) * cct_topk_;
        for (int k = 0; k < cct_topk_ && count < cct_max; ++k) {
            bool duplicate = false;
            for (int j = 0; j < count; ++j)
                duplicate = duplicate || picks[j] == int(row[k]);
            if (!duplicate)
                picks[count++] = int(row[k]);
        }
    }
    if (count)
        expert_stager_->prefetch(layer + 1, picks, count);
}

void Runner::sparse_moe(int layer, const float *input, float *output) {
    const std::string stem = layer_stem(layer) + "mlp.";
    linear(stem + "gate.weight", input, router_, moe_experts_, hidden_);
    std::vector<float> logits(moe_experts_);
    check(cudaMemcpy(logits.data(), router_.get(), logits.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "download router logits");
    const std::vector<float> &bias = host_f32(stem + "gate.e_score_correction_bias");
    require(bias.size() == size_t(moe_experts_), "wrong router bias geometry");
    std::vector<float> scores(moe_experts_), choice(moe_experts_);
    for (int expert = 0; expert < moe_experts_; ++expert) {
        scores[expert] = 1.0f / (1.0f + std::exp(-logits[expert]));
        choice[expert] = scores[expert] + bias[expert];
    }
    std::vector<int> order(moe_experts_);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + moe_topk_, order.end(),
        [&](int left, int right) { return choice[left] > choice[right]; });
    float denominator = 0.0f;
    for (int slot = 0; slot < moe_topk_; ++slot) denominator += scores[order[slot]];
    std::vector<int> selected(order.begin(), order.begin() + moe_topk_);
    if (early_route_on_) {
        int overlap = 0;
        for (int predicted : early_routing_[size_t(layer)])
            overlap += std::find(selected.begin(), selected.end(), predicted) != selected.end();
        early_route_hits_ += uint64_t(overlap);
        early_route_total_ += uint64_t(moe_topk_);
        if (early_route_trace_) {
            std::fprintf(early_route_trace_, "%ld %d %d", token_index_, layer, overlap);
            for (int predicted : early_routing_[size_t(layer)])
                std::fprintf(early_route_trace_, " %d", predicted);
            for (int actual : selected)
                std::fprintf(early_route_trace_, " %d", actual);
            std::fputc('\n', early_route_trace_);
            std::fflush(early_route_trace_);
        }
    }
    route_trace(layer, selected, scores);

    check(cudaMemset(routed_, 0, hidden_ * sizeof(float)), "clear routed output");
    if (!nvfp4_experts_ && !q3_experts_) {
        // BF16 checkpoints (the tiny oracle among them) take the plain linear
        // path; there are no block scales to stage.
        for (int slot = 0; slot < moe_topk_; ++slot) {
            const int expert = selected[slot];
            compute_mlp(stem + "experts." + std::to_string(expert) + ".", input, ffn_, moe_intermediate_);
            const float weight = 2.5f * scores[expert] / denominator;
            scale_add_kernel<<<16, 256>>>(routed_, ffn_, weight, hidden_);
        }
    } else {
        // Record this token's routing, then speculatively start the next
        // layer's records (the previous token's choices) while this
        // layer's demand reads still own the disk.
        for (int slot = 0; slot < moe_topk_; ++slot)
            prev_routing_[size_t(layer)][size_t(slot)] = selected[slot];
        if (prefetch_on_ && size_t(layer) + 1 < prev_routing_.size() &&
            is_sparse_[size_t(layer) + 1])
            expert_stager_->prefetch(int(layer) + 1, prev_routing_[size_t(layer) + 1].data(),
                                     moe_topk_);
        expert_stager_->load_batch(layer, {selected[0], selected[1], selected[2], selected[3],
                                           selected[4], selected[5], selected[6], selected[7]});
        // Cross-layer prefetch queues behind this layer's own demand records
        // so speculative reads can never delay the current GEMVs.
        if (prefetch_on_ && !cct_.empty())
            cct_prefetch(layer);
        constexpr int direct_id = 0;
        if (q3_experts_)
            check(insignia::glm53::iq_quantize_activation_rows(
                      input, hidden_, &direct_id, 1, iq_workspace_4096_.get()),
                  "quantize Q3 expert input");
        else
            check(insignia::glm53::nvfp4_quantize_activation(
                      input, hidden_, nv_workspace_4096_),
                  "quantize expert input");
        // The shared expert is dense-resident FP8; running it first lets its
        // GEMVs hide under the routed records' disk reads. Routed downs land
        // in a scratch buffer so `output` keeps the shared result.
        compute_mlp(stem + "shared_experts.", input, output, shared_intermediate_);
        static const bool q3_topk = [] {
            const char *value = std::getenv("INSIGNIA_GLM53_Q3_TOPK");
            return !value || std::atoi(value) != 0;
        }();
        if (q3_experts_ && q3_topk) expert_stager_->prime_device_arena();
        if (q3_experts_ && q3_topk && expert_stager_->device_topk_capable()) {
            std::array<const uint8_t *, 8> gate_weights{}, up_weights{}, down_weights{};
            std::array<float, 8> combine{};
            std::array<int, 8> device_slots{};
            TensorType gate_type = TensorType::iq3_xxs;
            TensorType down_type = TensorType::iq4_xs;
            for (int slot = 0; slot < moe_topk_; ++slot) {
                const int expert = selected[size_t(slot)];
                expert_stager_->upload(slot);
                gate_weights[size_t(slot)] = expert_stager_->gate_weight();
                up_weights[size_t(slot)] = expert_stager_->up_weight();
                down_weights[size_t(slot)] = expert_stager_->down_weight();
                combine[size_t(slot)] = 2.5f * scores[size_t(expert)] / denominator;
                device_slots[size_t(slot)] = expert_stager_->active_device_slot();
                require(device_slots[size_t(slot)] >= 0,
                        "Q3 top-k execution requires a persistent device slot");
                for (int earlier = 0; earlier < slot; ++earlier)
                    require(device_slots[size_t(slot)] != device_slots[size_t(earlier)],
                            "Q3 top-k collection recycled a live device slot");
                const TensorType this_gate = expert_stager_->projection_type(1);
                require(this_gate == expert_stager_->projection_type(2),
                        "Q3 expert gate/up formats must match");
                const TensorType this_down = expert_stager_->projection_type(0);
                if (slot == 0) {
                    gate_type = this_gate;
                    down_type = this_down;
                } else {
                    require(gate_type == this_gate && down_type == this_down,
                            "Q3 expert formats must be uniform within a layer");
                }
                // Keep the original copy/compute pipeline alive: as soon as
                // one normal IQ3 record reaches VRAM, launch its gate/up pair
                // while the copy engine fetches the next record.  Waiting for
                // all eight records before any work made cold and host-tier
                // hits serialize behind eight H2D copies.
                if (gate_type == TensorType::iq3_xxs &&
                    down_type == TensorType::iq4_xs) {
                    const int output_id = slot;
                    check(insignia::glm53::iq3_xxs_gemv2_rows(
                              gate_weights[size_t(slot)], up_weights[size_t(slot)],
                              iq_workspace_4096_.get(), 1, c_gateu_.get(), c_up_.get(),
                              &output_id, moe_intermediate_, hidden_),
                          "Q3 pipelined IQ3 gate/up");
                }
            }
            if (gate_type == TensorType::iq3_xxs && down_type == TensorType::iq4_xs) {
                check(insignia::glm53::iq4_xs_swiglu_gemv_acc_topk_x1(
                          down_weights.data(), c_gateu_.get(), c_up_.get(), combine.data(),
                          moe_topk_, routed_.get(), hidden_, moe_intermediate_),
                      "Q3 raw top-k IQ4 down");
            } else {
                // Three model layers carry higher-precision exception tensors.
                // Keep their canonical expert order while still amortizing all
                // eight uploads ahead of the compute launches.
                for (int slot = 0; slot < moe_topk_; ++slot) {
                    if (gate_type == TensorType::iq3_xxs) {
                        check(insignia::glm53::iq3_xxs_gemv2_rows(
                                  gate_weights[size_t(slot)], up_weights[size_t(slot)],
                                  iq_workspace_4096_.get(), 1, gate_.get(), up_.get(),
                                  &direct_id, moe_intermediate_, hidden_),
                              "Q3 exception IQ3 gate/up");
                    } else if (gate_type == TensorType::iq4_xs) {
                        check(insignia::glm53::iq4_xs_gemv_rows(
                                  gate_weights[size_t(slot)], iq_workspace_4096_.get(), 1,
                                  gate_.get(), &direct_id, moe_intermediate_, hidden_),
                              "Q3 exception IQ4 gate");
                        check(insignia::glm53::iq4_xs_gemv_rows(
                                  up_weights[size_t(slot)], iq_workspace_4096_.get(), 1,
                                  up_.get(), &direct_id, moe_intermediate_, hidden_),
                              "Q3 exception IQ4 up");
                    } else {
                        require(false, "unsupported Q3 gate/up format");
                    }
                    if (down_type == TensorType::iq4_xs) {
                        check(insignia::glm53::iq4_xs_swiglu_gemv_acc_fused_x1(
                                  down_weights[size_t(slot)], gate_.get(), up_.get(), 0,
                                  routed_.get(), 0, combine[size_t(slot)], hidden_,
                                  moe_intermediate_),
                              "Q3 exception IQ4 fused routed down");
                    } else if (down_type == TensorType::q6_k) {
                        check(insignia::glm53::q6_k_swiglu_gemv_acc_fused_x1(
                                  down_weights[size_t(slot)], gate_.get(), up_.get(), 0,
                                  routed_.get(), 0, combine[size_t(slot)], hidden_,
                                  moe_intermediate_),
                              "Q3 exception Q6 fused routed down");
                    } else {
                        require(false, "unsupported Q3 down format");
                    }
                }
            }
            expert_stager_->fence_device_slots(device_slots.data(), moe_topk_);
        } else for (int slot = 0; slot < moe_topk_; ++slot) {
            const int expert = selected[slot];
            expert_stager_->upload(slot);
            if (q3_experts_) {
                const TensorType gate_type = expert_stager_->projection_type(1);
                const TensorType up_type = expert_stager_->projection_type(2);
                require(gate_type == up_type,
                        "Q3 expert gate/up formats must match");
                if (gate_type == TensorType::iq3_xxs) {
                    check(insignia::glm53::iq3_xxs_gemv2_rows(
                              expert_stager_->gate_weight(),
                              expert_stager_->up_weight(), iq_workspace_4096_.get(),
                              1, gate_.get(), up_.get(), &direct_id,
                              moe_intermediate_, hidden_),
                          "Q3 IQ3 gate/up");
                } else if (gate_type == TensorType::iq4_xs) {
                    check(insignia::glm53::iq4_xs_gemv_rows(
                              expert_stager_->gate_weight(), iq_workspace_4096_.get(),
                              1, gate_.get(), &direct_id,
                              moe_intermediate_, hidden_),
                          "Q3 IQ4 gate");
                    check(insignia::glm53::iq4_xs_gemv_rows(
                              expert_stager_->up_weight(), iq_workspace_4096_.get(),
                              1, up_.get(), &direct_id,
                              moe_intermediate_, hidden_),
                          "Q3 IQ4 up");
                } else {
                    require(false, "unsupported Q3 gate/up format");
                }
                const float weight = 2.5f * scores[expert] / denominator;
                const TensorType down_type = expert_stager_->projection_type(0);
                if (down_type == TensorType::iq4_xs) {
                    check(insignia::glm53::iq4_xs_swiglu_gemv_acc_fused_x1(
                              expert_stager_->down_weight(), gate_.get(), up_.get(),
                              0, routed_.get(), 0, weight, hidden_, moe_intermediate_),
                          "Q3 IQ4 fused routed down");
                } else if (down_type == TensorType::q6_k) {
                    check(insignia::glm53::q6_k_swiglu_gemv_acc_fused_x1(
                              expert_stager_->down_weight(), gate_.get(), up_.get(),
                              0, routed_.get(), 0, weight, hidden_, moe_intermediate_),
                          "Q3 Q6 fused routed down");
                } else {
                    require(false, "unsupported Q3 down format");
                }
            } else {
                check(expert_stager_->packed_direct_active()
                      ? insignia::glm53::nvfp4_gemv2_dp4a_quantized_rows_packed(
                            expert_stager_->gate_weight(),
                            expert_stager_->gate_packed_scale(),
                            expert_stager_->gate_global(slot),
                            expert_stager_->up_weight(),
                            expert_stager_->up_packed_scale(),
                            expert_stager_->up_global(slot),
                            nv_workspace_4096_, 1, gate_, up_, &direct_id,
                            moe_intermediate_, hidden_, kNvfp4PackedPairCtaWarps[1],
                            expert_stager_->packed_tablefree())
                      : insignia::glm53::nvfp4_gemv2_dp4a_quantized(
                            expert_stager_->gate_weight(), expert_stager_->gate_scale(),
                            expert_stager_->gate_global(slot), expert_stager_->up_weight(),
                            expert_stager_->up_scale(), expert_stager_->up_global(slot),
                            nv_workspace_4096_, gate_, up_, moe_intermediate_, hidden_),
                      "routed expert gate/up");
                check(insignia::glm53::quantize_swiglu_activation(gate_, up_, moe_intermediate_, nv_workspace_2048_),
                      "quantize routed SwiGLU");
                const float weight = 2.5f * scores[expert] / denominator;
                check(expert_stager_->packed_direct_active()
                          ? insignia::glm53::nvfp4_gemv_dp4a_acc_quantized_rows_packed(
                                expert_stager_->down_weight(),
                                expert_stager_->down_packed_scale(),
                                expert_stager_->down_global(slot), nv_workspace_2048_,
                                1, routed_, &direct_id, &weight, hidden_, moe_intermediate_,
                                kNvfp4PackedDownAccCtaWarps[1],
                                expert_stager_->packed_tablefree())
                          : insignia::glm53::nvfp4_gemv_dp4a_acc_quantized(
                                expert_stager_->down_weight(), expert_stager_->down_scale(),
                                expert_stager_->down_global(slot), nv_workspace_2048_,
                                routed_, weight, hidden_, moe_intermediate_),
                      "routed expert down");
            }
            }
        add_kernel<<<16, 256>>>(output, routed_, hidden_);
    }
    if (!nvfp4_experts_ && !q3_experts_) {
        compute_mlp(stem + "shared_experts.", input, output, shared_intermediate_);
        add_kernel<<<16, 256>>>(output, routed_, hidden_);
    } else {
        check(cudaGetLastError(), "MoE combine launch");
    }
}

// Copies one projected KDA tensor (pre-conv q/k/v, raw gate, raw beta) into
// the replay archive during a verify pass. slot: 0=q 1=k 2=v 3=gate 4=beta.
void Runner::archive_kda_rows(int layer, const float *src, int rows, int slot) {
    if (!kda_archive_) return;
    const int width = kda_width_;
    const int row = kda_row_[size_t(layer)];
    require(row >= 0, "archiving a non-KDA layer");
    const size_t stride = 4 * size_t(width) + kda_heads_;
    const size_t span = slot == 4 ? size_t(kda_heads_) : size_t(width);
    const size_t layer_base = size_t(row) * kMaxVerify * stride;
    // `src` is projection-major ([token][span]), while the replay archive is
    // token-major ([token][q,k,v,gate,beta]). A contiguous rows*span copy
    // overlaps the following fields and corrupts every row after token 0.
    for (int token = 0; token < rows; ++token) {
        const size_t offset = layer_base + size_t(token) * stride + size_t(slot) * width;
        check(cudaMemcpyAsync(kda_arch_.get() + offset, src + size_t(token) * span,
                              span * sizeof(float), cudaMemcpyDeviceToDevice),
              "archive KDA row");
    }
}

// Layer 45 (MTP) forward: one draft token from (embed(token) | hidden).
// Mirrors the GLM-4.5/GLM-5 nextn block: enorm(embed) ++ hnorm(hidden) ->
// eh_proj -> one pre-norm MLA/DSA block (no hyper-connections) -> noaux_tc
// MoE without a shared expert -> shared_head.norm -> tied lm_head.
// `hidden_in` is the pre-final-norm mean of the 4 hyper-connection streams
// (target step) or the recycled draft hidden from the previous draft step.
int Runner::mtp_forward(int token, const float *hidden_in, int position) {
    const std::string stem = layer_stem(45);
    const TensorLocation &embedding = model_.tensor("model.language_model.embed_tokens.weight");
    const uint16_t *row = reinterpret_cast<const uint16_t *>(
        stager_.load(embedding, uint64_t(token) * hidden_ * 2, hidden_ * 2));
    float *embed_half = mtp_variant_ == 2 ? mtp_eh_in_ + hidden_ : mtp_eh_in_;
    float *hidden_half = mtp_variant_ == 2 ? mtp_eh_in_ : mtp_eh_in_ + hidden_;
    if (position == 0 && mtp_variant_ != 4) {
        // The GLM-4.5 reference zeroes the embedding at position 0 before
        // enorm; variant 4 disables it in case GLM-5.3 does not.
        check(cudaMemsetAsync(embed_half, 0, size_t(hidden_) * sizeof(float)),
              "zero position-0 MTP embed");
    } else if (mtp_variant_ == 3) {
        bf16_rms_kernel<<<16, 256>>>(row,
            reinterpret_cast<const uint16_t *>(stager_.load(stem + "hnorm.weight")),
            embed_half, hidden_);
        check(cudaGetLastError(), "MTP crossed enorm launch");
    } else {
        bf16_rms_kernel<<<16, 256>>>(row,
            reinterpret_cast<const uint16_t *>(stager_.load(stem + "enorm.weight")),
            embed_half, hidden_);
        check(cudaGetLastError(), "MTP enorm launch");
    }
    if (mtp_variant_ == 3)
        rms(stem + "enorm.weight", hidden_in, hidden_half, 1, hidden_);
    else
        rms(stem + "hnorm.weight", hidden_in, hidden_half, 1, hidden_);
    if (std::getenv("INSIGNIA_GLM53_MTP_BF16")) {
        // Bypass the FP8 cache for eh_proj (rules out cache corruption on the
        // 8192-wide matrix): stage the BF16 weights and GEMV directly.
        const TensorLocation &eh = model_.tensor(stem + "eh_proj.weight");
        const uint32_t *device = reinterpret_cast<const uint32_t *>(stager_.load(eh));
        insignia::bf16_gemv_v2(device, mtp_eh_in_, mtp_hidden_, hidden_, 2 * hidden_);
    } else {
        linear(stem + "eh_proj.weight", mtp_eh_in_, mtp_hidden_, hidden_, 2 * hidden_);
    }
    launch_rms(mtp_hidden_,
        reinterpret_cast<const uint16_t *>(stager_.load(stem + "input_layernorm.weight")),
        normalized_, 1, hidden_);
    mla(45, normalized_, mtp_attn_, position);
    add_kernel<<<16, 256>>>(mtp_hidden_, mtp_attn_, hidden_);
    check(cudaGetLastError(), "MTP attention residual");
    launch_rms(mtp_hidden_,
        reinterpret_cast<const uint16_t *>(stager_.load(stem + "post_attention_layernorm.weight")),
        normalized_, 1, hidden_);
    mtp_moe(normalized_, mtp_moe_);
    add_kernel<<<16, 256>>>(mtp_hidden_, mtp_moe_, hidden_);
    check(cudaGetLastError(), "MTP MoE residual");
    launch_rms(mtp_hidden_,
        reinterpret_cast<const uint16_t *>(stager_.load(stem + "shared_head.norm.weight")),
        normalized_, 1, hidden_);
    linear("lm_head.weight", normalized_, mtp_logits_.get(), int(model_.vocab_size()), hidden_);
    check(cudaMemcpyAsync(mtp_recycle_.get(), mtp_hidden_, size_t(hidden_) * sizeof(float),
                          cudaMemcpyDeviceToDevice), "recycle MTP hidden");
    rows_argmax_kernel<<<dim3(1, 1), 1024>>>(mtp_logits_.get(), verify_arg_.get(),
                                             1, int(model_.vocab_size()));
    check(cudaGetLastError(), "MTP argmax launch");
    int token_out = 0;
    check(cudaMemcpy(&token_out, verify_arg_.get(), sizeof(token_out), cudaMemcpyDeviceToHost),
          "download MTP argmax");
    if (std::getenv("INSIGNIA_GLM53_MTP_DEBUG")) {
        std::vector<float> probe(model_.vocab_size());
        check(cudaMemcpy(probe.data(), mtp_logits_.get(), probe.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), "download MTP probe logits");
        std::vector<int> order(probe.size());
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + 5, order.end(),
            [&](int left, int right) { return probe[left] > probe[right]; });
        std::fprintf(stderr, "mtp draft in=%d pos=%d top5", token, position);
        for (int index = 0; index < 5; ++index)
            std::fprintf(stderr, " %d:%.3f", order[index], probe[order[index]]);
        std::fprintf(stderr, "\n");
        if (const char *path = std::getenv("INSIGNIA_GLM53_MTP_DUMP")) {
            // Binary oracle record: token, position, hidden_in[4096], argmax.
            static FILE *dump = std::fopen(path, "wb");
            if (dump) {
                std::vector<float> hidden = std::vector<float>(size_t(hidden_));
                check(cudaMemcpy(hidden.data(), hidden_in, size_t(hidden_) * sizeof(float),
                                 cudaMemcpyDeviceToHost), "download MTP hidden");
                std::fwrite(&token, sizeof(token), 1, dump);
                std::fwrite(&position, sizeof(position), 1, dump);
                std::fwrite(hidden.data(), sizeof(float), size_t(hidden_), dump);
                const int best = token_out;
                std::fwrite(&best, sizeof(best), 1, dump);
                std::fflush(dump);
            }
        }
    }
    return token_out;
}

void Runner::retain_df_prior_logits(const float *device_logits, const float *host_logits) {
    const size_t vocab = model_.vocab_size();
    require(device_logits, "DFlash prior logits are missing");
    if (df_calibration_guard_js_ > 0.0f) {
        require(df_prior_logits_device_.size() == vocab,
                "DFlash calibration prior buffer has wrong geometry");
        check(cudaMemcpyAsync(df_prior_logits_device_.get(), device_logits,
                              vocab * sizeof(float), cudaMemcpyDeviceToDevice),
              "retain accepted DFlash target logits on device");
        df_prior_logits_ready_ = true;
    }
    if (df_retry_top1_drop_ > 0.0f) {
        df_prior_logits_host_.resize(vocab);
        if (host_logits) {
            std::copy_n(host_logits, vocab, df_prior_logits_host_.data());
        } else {
            check(cudaMemcpy(df_prior_logits_host_.data(), device_logits,
                             vocab * sizeof(float), cudaMemcpyDeviceToHost),
                  "download accepted DFlash retry prior logits");
        }
    }
}

// The first-pass verifier has already paid for target logits but has not yet
// committed recurrent state.  A sharp confidence collapse inside the block is
// an unusually strong on-policy failure signal.  Spend CPU softmax work and,
// when it fires, restore the round-start KDA/conv snapshot and execute the
// identical candidate block once more with all eight experts.
bool Runner::dflash_retry_needed(int rows) {
    if (df_retry_top1_drop_ <= 0.0f) return false;
    require(rows >= 1 && rows <= kMaxVerify && df_retry_logits_host_,
            "DFlash retry row geometry is invalid");
    const size_t vocab = model_.vocab_size();
    require(df_prior_logits_host_.size() == vocab,
            "DFlash retry has no previous target logits");
    check(cudaMemcpy(df_retry_logits_host_, verify_logits_.get(),
                     size_t(rows) * vocab * sizeof(float), cudaMemcpyDeviceToHost),
          "download DFlash retry target logits");
    const auto probability = [&](const float *logits) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (size_t token = 0; token < vocab; ++token)
            maximum = std::max(maximum, double(logits[token]));
        double normalizer = 0.0;
        for (size_t token = 0; token < vocab; ++token)
            normalizer += std::exp(double(logits[token]) - maximum);
        return 1.0 / normalizer;
    };
    double previous = probability(df_prior_logits_host_.data());
    double largest_drop = -std::numeric_limits<double>::infinity();
    std::array<double, kMaxVerify> row_probability{};
    for (int row = 0; row < rows; ++row) {
        const double current = probability(
            df_retry_logits_host_ + size_t(row) * vocab);
        row_probability[size_t(row)] = current;
        largest_drop = std::max(largest_drop, previous - current);
        previous = current;
    }
    ++df_retry_rounds_;
    df_retry_drop_sum_ += largest_drop;
    df_retry_drop_max_ = std::max(df_retry_drop_max_, largest_drop);
    const bool retry = largest_drop >= df_retry_top1_drop_;
    df_retry_triggered_rounds_ += retry;
    if (std::getenv("INSIGNIA_GLM53_DF_DEBUG")) {
        std::fprintf(stderr, "df post-verify target p");
        for (int row = 0; row < rows; ++row)
            std::fprintf(stderr, " %.6f", row_probability[size_t(row)]);
        std::fprintf(stderr, " max_drop %.6f retry %d\n", largest_drop, retry);
    }
    return retry;
}

// DFlash2 draft round: stage [anchor, mask x7] embeds, one block forward,
// lm_head the 7 draft hiddens through the target's FP8 head, then the host
// selector walks the top-16 candidate lattice.
std::vector<int> Runner::df_draft(int anchor, int position) {
    // Same reader-pool warm-up the scalar step() path performs: the drafter
    // phase is otherwise dead time for the disks, and the first sparse
    // layers' records (predicted from the previous round's routing) are what
    // the verify pass demands first.
    if (prefetch_on_ && expert_stager_) {
        for (size_t layer = 3; layer < prev_routing_.size() && layer < 6; ++layer)
            if (is_sparse_[layer])
                expert_stager_->prefetch(int(layer), prev_routing_[layer].data(), moe_topk_);
        if (!cct_.empty())
            for (size_t layer = 3; layer + 1 < prev_routing_.size() && layer < 6; ++layer)
                cct_prefetch(int(layer));
    }
    const TensorLocation &embedding = model_.tensor("model.language_model.embed_tokens.weight");
    const auto stage_row = [&](int token, int t) {
        const uint16_t *row = reinterpret_cast<const uint16_t *>(
            stager_.load(embedding, uint64_t(token) * hidden_ * 2, hidden_ * 2));
        df_->set_block_row(t, row);
    };
    stage_row(anchor, 0);
    for (int t = 1; t < insignia::glm53::DFlash2Drafter::kBlock; ++t)
        stage_row(insignia::glm53::DFlash2Drafter::kMaskToken, t);
    df_->forward(anchor, position);
    // 7 separate GEMVs: each re-reads lm_head, but the BF16 stager keeps it
    // VRAM-resident so reads cost microseconds. (linear_multi measured slower —
    // lm_head FP8 exceeds the Q8 slot, forcing per-round NVMe re-streams.)
    for (int t = 0; t < insignia::glm53::DFlash2Drafter::kDrafts; ++t)
        linear("lm_head.weight", df_->draft_hidden() + size_t(t) * hidden_,
               verify_logits_.get() + size_t(t) * model_.vocab_size(),
               int(model_.vocab_size()), hidden_);
    constexpr int draft_rows = insignia::glm53::DFlash2Drafter::kDrafts;
    const int vocab = int(model_.vocab_size());
    // The engine already enforces finite residuals before lm_head. Keep that
    // invariant here rather than paying another full-vocabulary validation pass.
    if (df_calibration_guard_js_ > 0.0f) {
        require(df_prior_logits_ready_, "DFlash calibration has no target-logit prior");
        check(insignia::glm53::logit_metrics_async(
                  df_prior_logits_device_.get(), verify_logits_.get(), vocab,
                  df_logit_metrics_workspace_.get(), df_logit_metrics_device_.get()),
              "launch DFlash2 calibration metrics");
    }
    if (df_uncertainty_top1_p_ > 0.0f || df_uncertainty_top1_drop_ > 0.0f)
        check(insignia::glm53::logit_row_stats_async(
                  verify_logits_.get(), draft_rows, vocab,
                  df_logit_row_stats_workspace_.get(), df_logit_row_stats_device_.get()),
              "launch DFlash2 row statistics");
    if (df_calibration_guard_js_ > 0.0f)
        check(cudaMemcpyAsync(df_logit_metrics_host_, df_logit_metrics_device_.get(),
                              sizeof(insignia::glm53::LogitMetrics),
                              cudaMemcpyDeviceToHost),
              "download DFlash2 calibration metrics");
    if (df_uncertainty_top1_p_ > 0.0f || df_uncertainty_top1_drop_ > 0.0f)
        check(cudaMemcpyAsync(df_logit_row_stats_host_, df_logit_row_stats_device_.get(),
                              size_t(draft_rows) * sizeof(insignia::glm53::LogitRowStats),
                              cudaMemcpyDeviceToHost),
              "download DFlash2 row statistics");
    check(cudaMemcpyAsync(df_logits_host_, verify_logits_.get(),
                          df_->logits_span() * sizeof(float), cudaMemcpyDeviceToHost),
          "download DFlash2 logits");
    check(cudaMemcpyAsync(df_hp_host_, df_->hidden_projection(),
                          size_t(draft_rows) * insignia::glm53::DFlash2Drafter::kRank *
                              sizeof(float),
                          cudaMemcpyDeviceToHost),
          "download DFlash2 hp");
    check(cudaStreamSynchronize(nullptr), "complete DFlash2 logit decisions");
    df_logit_guard_exact_.fill(0);
    df_logit_guard_k_.fill(0);
    const int draft_round = int(df_draft_round_++);
    if (std::binary_search(df_diagnostic_exact_rounds_.begin(),
                           df_diagnostic_exact_rounds_.end(), draft_round))
        for (int row = 0; row < kMaxVerify; ++row)
            guard_dflash_row(row, 8);
    if (df_uncertainty_hold_left_ > 0) {
        for (int row = 0; row < kMaxVerify; ++row)
            guard_dflash_row(row, df_uncertainty_guard_k_);
        --df_uncertainty_hold_left_;
    }
    if (df_calibration_guard_js_ > 0.0f) {
        const double js = df_logit_metrics_host_->js;
        ++df_calibration_guard_rounds_;
        df_calibration_js_sum_ += js;
        df_calibration_js_max_ = std::max(df_calibration_js_max_, js);
        if (js > df_calibration_guard_js_) {
            for (int row = 0; row < kMaxVerify; ++row)
                guard_dflash_row(row, 8);
            ++df_calibration_guarded_rounds_;
        }
        if (std::getenv("INSIGNIA_GLM53_DF_DEBUG"))
            std::fprintf(stderr, "df calibration JS %.9f guard %d\n", js,
                         js > df_calibration_guard_js_);
    }
    if (df_uncertainty_top1_p_ > 0.0f || df_uncertainty_top1_drop_ > 0.0f) {
        constexpr int rows = insignia::glm53::DFlash2Drafter::kDrafts;
        std::array<double, rows> top1_probability{};
        for (int draft_row = 0; draft_row < rows; ++draft_row)
            top1_probability[size_t(draft_row)] =
                df_logit_row_stats_host_[draft_row].top1_probability;
        int highest_guard = -1;
        // Target output after verify row r predicts candidate r+1, whose
        // causal uncertainty is DFlash row r+1.  The probability drop from
        // row r to r+1 measures within-block confidence collapse and was the
        // strongest held-out hard-prompt failure signal (12/12 capture when
        // composed with an absolute p threshold).
        for (int verify_row = 0; verify_row + 1 < rows; ++verify_row) {
            const int current = verify_row + 1;
            const double probability = top1_probability[size_t(current)];
            const double drop = top1_probability[size_t(current - 1)] - probability;
            const bool low = df_uncertainty_top1_p_ > 0.0f &&
                probability <= df_uncertainty_top1_p_;
            const bool collapsed = df_uncertainty_top1_drop_ > 0.0f &&
                drop >= df_uncertainty_top1_drop_;
            if (low || collapsed) {
                guard_dflash_row(verify_row, df_uncertainty_guard_k_);
                highest_guard = verify_row;
            }
        }
        if (df_logit_guard_prefix_)
            for (int verify_row = 0; verify_row <= highest_guard; ++verify_row)
                guard_dflash_row(verify_row, df_uncertainty_guard_k_);
        if (highest_guard >= 0)
            df_uncertainty_hold_left_ = std::max(
                df_uncertainty_hold_left_, df_uncertainty_hold_rounds_);
        if (std::getenv("INSIGNIA_GLM53_DF_DEBUG")) {
            std::fprintf(stderr, "df distribution p");
            for (double probability : top1_probability)
                std::fprintf(stderr, " %.6f", probability);
            std::fprintf(stderr, " highest_guard %d k%d hold_left %d\n", highest_guard,
                         df_uncertainty_guard_k_, df_uncertainty_hold_left_);
        }
    }
    if (df_logit_guard_margin_ > 0.0f) {
        int highest_margin_guard = -1;
        for (int verify_row = 0; verify_row < kMaxVerify; ++verify_row) {
            // Target output after candidate r predicts candidate r+1. DFlash
            // row r+1 is the same-position risk signal. The final draft row
            // has no look-ahead row, so keep it exact.
            if (verify_row + 1 >= insignia::glm53::DFlash2Drafter::kDrafts) {
                guard_dflash_row(verify_row, 8);
                continue;
            }
            const float *row = df_logits_host_ + size_t(verify_row + 1) * vocab;
            float first = -std::numeric_limits<float>::infinity();
            float second = first;
            for (int token = 0; token < vocab; ++token) {
                const float value = row[token];
                if (value > first) {
                    second = first;
                    first = value;
                } else if (value > second) {
                    second = value;
                }
            }
            if (first - second < df_logit_guard_margin_) {
                guard_dflash_row(verify_row, 8);
                highest_margin_guard = verify_row;
            }
        }
        // KDA recurrent state and causal attention carry every earlier row's
        // approximation error into later rows. Exactifying only the flagged
        // row cannot repair that state, so a risky row makes its whole causal
        // prefix exact. Do not propagate the final no-lookahead sentinel above.
        if (df_logit_guard_prefix_)
            for (int verify_row = 0; verify_row <= highest_margin_guard; ++verify_row)
                guard_dflash_row(verify_row, 8);
    }
    static const bool df_debug = std::getenv("INSIGNIA_GLM53_DF_DEBUG") != nullptr;
    if (df_debug) {
        for (int t = 0; t < 5; ++t) {
            const float *row = df_logits_host_ + size_t(t) * vocab;
            std::vector<int> order(vocab);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                              [&](int a, int b) { return row[a] > row[b]; });
            std::fprintf(stderr, "df logits t%d top5", t);
            for (int k = 0; k < 5; ++k) std::fprintf(stderr, " %d:%.3f", order[k], row[order[k]]);
            std::fprintf(stderr, "\n");
        }
    }
    return df_->select(df_logits_host_, df_hp_host_, anchor);
}

// Layer-45 routed MoE: identical noaux_tc routing to the main sparse layers
// but with no shared expert and no next-layer prefetch (there is no layer 46).
void Runner::mtp_moe(const float *input, float *output) {
    const std::string stem = layer_stem(45) + "mlp.";
    linear(stem + "gate.weight", input, router_, moe_experts_, hidden_);
    std::vector<float> logits(moe_experts_);
    check(cudaMemcpy(logits.data(), router_.get(), logits.size() * sizeof(float),
                     cudaMemcpyDeviceToHost), "download MTP router logits");
    const std::vector<float> &bias = host_f32(stem + "gate.e_score_correction_bias");
    require(bias.size() == size_t(moe_experts_), "wrong MTP router bias geometry");
    std::vector<float> scores(moe_experts_), choice(moe_experts_);
    for (int expert = 0; expert < moe_experts_; ++expert) {
        scores[expert] = 1.0f / (1.0f + std::exp(-logits[expert]));
        choice[expert] = scores[expert] + bias[expert];
    }
    std::vector<int> order(moe_experts_);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + moe_topk_, order.end(),
        [&](int left, int right) { return choice[left] > choice[right]; });
    float denominator = 0.0f;
    for (int slot = 0; slot < moe_topk_; ++slot) denominator += scores[order[slot]];
    std::vector<int> selected(order.begin(), order.begin() + moe_topk_);
    check(cudaMemset(routed_, 0, hidden_ * sizeof(float)), "clear MTP routed output");
    expert_stager_->load_batch(45, {selected[0], selected[1], selected[2], selected[3],
                                    selected[4], selected[5], selected[6], selected[7]});
    check(insignia::glm53::nvfp4_quantize_activation(input, hidden_, nv_workspace_4096_),
          "quantize MTP expert input");
    for (int slot = 0; slot < moe_topk_; ++slot) {
        const int expert = selected[slot];
        expert_stager_->upload(slot);
        constexpr int direct_id = 0;
        check(expert_stager_->packed_direct_active()
                  ? insignia::glm53::nvfp4_gemv2_dp4a_quantized_rows_packed(
                        expert_stager_->gate_weight(), expert_stager_->gate_packed_scale(),
                        expert_stager_->gate_global(slot), expert_stager_->up_weight(),
                        expert_stager_->up_packed_scale(), expert_stager_->up_global(slot),
                        nv_workspace_4096_, 1, gate_, up_, &direct_id,
                        moe_intermediate_, hidden_, kNvfp4PackedPairCtaWarps[1],
                        expert_stager_->packed_tablefree())
                  : insignia::glm53::nvfp4_gemv2_dp4a_quantized(
                        expert_stager_->gate_weight(), expert_stager_->gate_scale(),
                        expert_stager_->gate_global(slot), expert_stager_->up_weight(),
                        expert_stager_->up_scale(), expert_stager_->up_global(slot),
                        nv_workspace_4096_, gate_, up_, moe_intermediate_, hidden_),
              "MTP expert gate/up");
        check(insignia::glm53::quantize_swiglu_activation(gate_, up_, moe_intermediate_,
              nv_workspace_2048_), "quantize MTP SwiGLU");
        const float weight = 2.5f * scores[expert] / denominator;
        check(expert_stager_->packed_direct_active()
                  ? insignia::glm53::nvfp4_gemv_dp4a_acc_quantized_rows_packed(
                        expert_stager_->down_weight(), expert_stager_->down_packed_scale(),
                        expert_stager_->down_global(slot), nv_workspace_2048_, 1,
                        routed_, &direct_id, &weight, hidden_, moe_intermediate_,
                        kNvfp4PackedDownAccCtaWarps[1],
                        expert_stager_->packed_tablefree())
                  : insignia::glm53::nvfp4_gemv_dp4a_acc_quantized(
                        expert_stager_->down_weight(), expert_stager_->down_scale(),
                        expert_stager_->down_global(slot), nv_workspace_2048_, routed_,
                        weight, hidden_, moe_intermediate_),
              "MTP expert down");
    }
    add_kernel<<<16, 256>>>(output, routed_, hidden_);
    check(cudaGetLastError(), "MTP MoE combine launch");
}

// Restores the pre-verify recurrent state and replays the accepted prefix
// through the KDA recurrence from the archived pre-conv projections (the
// layer weights are already resident, so the replay is compute-only).
void Runner::rollback_kda(int accepted, int position_base) {
    require(!forced_sequential_verify_,
            "rollback is unreachable when DFlash sequential verify is forced");
    check(cudaMemcpyAsync(kda_states_.get(), kda_snap_.get(),
                          kda_states_.size() * sizeof(float), cudaMemcpyDeviceToDevice),
          "restore KDA states");
    check(cudaMemcpyAsync(conv_history_.get(), conv_snap_.get(),
                          conv_history_.size() * sizeof(float), cudaMemcpyDeviceToDevice),
          "restore conv history");
    const int width = kda_width_;
    const size_t stride = 4 * size_t(width) + kda_heads_;
    for (int layer = 0; layer < int(model_.layers()); ++layer) {
        if (is_mla_[layer]) continue;
        const int row = kda_row_[size_t(layer)];
        const std::string stem = layer_stem(layer) + "self_attn.";
        const bool conv_fp32 = model_.tensor(stem + "q_conv1d.weight").type == TensorType::f32;
        const uint16_t *conv_q = reinterpret_cast<const uint16_t *>(stager_.load(stem + "q_conv1d.weight"));
        const uint16_t *conv_k = reinterpret_cast<const uint16_t *>(stager_.load(stem + "k_conv1d.weight"));
        const uint16_t *conv_v = reinterpret_cast<const uint16_t *>(stager_.load(stem + "v_conv1d.weight"));
        const float *dt_bias = device_f32(stem + "dt_bias");
        const float *a_log = device_f32(stem + "A_log");
        float *history = conv_history_.get() + size_t(row) * 9 * width;
        float *state = kda_states_.get() + size_t(row) * width * kda_head_dim_;
        for (int token = 0; token < accepted; ++token) {
            const float *arch = kda_arch_.get() + (size_t(row) * kMaxVerify + token) * stride;
            check(cudaMemcpy(q_, arch, size_t(width) * sizeof(float), cudaMemcpyDeviceToDevice),
                  "replay q");
            check(cudaMemcpy(k_, arch + width, size_t(width) * sizeof(float),
                             cudaMemcpyDeviceToDevice), "replay k");
            check(cudaMemcpy(v_, arch + 2 * width, size_t(width) * sizeof(float),
                             cudaMemcpyDeviceToDevice), "replay v");
            check(insignia::glm53::kda_conv_silu(q_, conv_q, history,
                  position_base + token, width, conv_fp32), "replay q convolution");
            check(insignia::glm53::kda_conv_silu(k_, conv_k, history + 3 * width,
                  position_base + token, width, conv_fp32), "replay k convolution");
            check(insignia::glm53::kda_conv_silu(v_, conv_v, history + 6 * width,
                  position_base + token, width, conv_fp32), "replay v convolution");
            check(cudaMemcpy(gate_8192_, arch + 3 * width, size_t(width) * sizeof(float),
                             cudaMemcpyDeviceToDevice), "replay gate");
            check(cudaMemcpy(beta_, arch + 4 * width, size_t(kda_heads_) * sizeof(float),
                             cudaMemcpyDeviceToDevice), "replay beta");
            kda_gate_kernel<<<32, 256>>>(gate_8192_, dt_bias, a_log, beta_, kda_heads_, width);
            check(cudaGetLastError(), "replay gate launch");
            check(insignia::glm53::kda_decode(state, q_, k_, v_, gate_8192_, beta_, core_,
                  kda_heads_, kda_head_dim_), "replay KDA recurrence");
        }
    }
}

// One verify pass: run the candidate tokens through the full target stack
// (prefill machinery) and return each row's argmax for acceptance checking.
std::pair<int, std::vector<int>> Runner::verify_round(const std::vector<int> &candidates,
                                                      int position_base) {
    prefill(candidates, position_base, true);
    const int count = int(candidates.size());
    // One block per row: the shared-memory reduction assumes a single block
    // owns the row, so gridDim.x must stay 1.
    rows_argmax_kernel<<<dim3(1, count), 1024>>>(verify_logits_.get(), verify_arg_.get(),
                                                 count, int(model_.vocab_size()));
    check(cudaGetLastError(), "verify argmax launch");
    std::vector<int> arg = std::vector<int>(size_t(count));
    check(cudaMemcpy(arg.data(), verify_arg_.get(), size_t(count) * sizeof(int),
                     cudaMemcpyDeviceToHost), "download verify argmax");
    return {count, arg};
}

// Row-sequential verify: forward ONE candidate at a time and return its
// target argmax. Unlike the batched verify_round, the caller stops the round
// as soon as acceptance fails, so the rejected tail's experts are never
// read and no state rollback is ever required (the recurrent state always
// stands at the accepted boundary). verify_may_rollback_ stays false for
// the whole sequential round so prefill skips the state snapshots; the
// drafter capture rows land in slot capture_offset_ (set by the caller).
int Runner::verify_token(int token, int position) {
    prefill({token}, position, true);
    rows_argmax_kernel<<<dim3(1, 1), 1024>>>(verify_logits_.get(), verify_arg_.get(),
                                             1, int(model_.vocab_size()));
    check(cudaGetLastError(), "verify argmax launch");
    int arg = 0;
    check(cudaMemcpy(&arg, verify_arg_.get(), sizeof(int), cudaMemcpyDeviceToHost),
          "download verify argmax");
    return arg;
}

// Teacher-force one fixed token stream through the exact or approximate
// multi-row verifier and dump full-vocabulary logits for apples-to-apples
// quality comparisons. Record 0 is the already-computed prompt-final logits;
// record i predicts forced token i under the same forced prefix in every arm.
void Runner::force_logits(const std::vector<int> &tokens, int position_base, int anchor,
                          const char *dump_path) {
    require(df_ && !tokens.empty(), "target-forced logits require DFlash2 and tokens");
    require(dump_path && *dump_path, "target-forced logits require an output path");
    require(position_base + int(tokens.size()) <= kMaxContext(),
            "target-forced stream exceeds context");
    for (int token : tokens)
        require(token >= 0 && token < int(model_.vocab_size()),
                "target-forced token is outside vocabulary");
    std::FILE *dump = std::fopen(dump_path, "wb");
    require(dump, "cannot open target-forced logits dump");
    const char *draft_path = std::getenv("INSIGNIA_GLM53_FORCE_DF_LOGITS_DUMP");
    std::FILE *draft_dump = draft_path ? std::fopen(draft_path, "wb") : nullptr;
    require(!draft_path || draft_dump, "cannot open target-forced DFlash logits dump");
    const int vocab = int(model_.vocab_size());
    std::vector<float> host(size_t(kMaxVerify) * vocab);
    check(cudaMemcpy(host.data(), logits_.get(), size_t(vocab) * sizeof(float),
                     cudaMemcpyDeviceToHost), "download prompt-final forced logits");
    require(std::fwrite(host.data(), sizeof(float), size_t(vocab), dump) == size_t(vocab),
            "write prompt-final forced logits");

    const int verify_k = [] {
        const char *value = std::getenv("INSIGNIA_GLM53_DF_VERIFY_K");
        return std::clamp(value ? std::atoi(value) : 4, 1, kMaxVerify);
    }();
    // Teacher forcing normally consumes every row and therefore elides the
    // rollback snapshot. Post-verify exact retry is the one exception: it
    // must restore the state that preceded this approximate chunk.
    verify_may_rollback_ = df_retry_top1_drop_ > 0.0f;
    for (size_t consumed = 0; consumed + 1 < tokens.size(); ) {
        begin_verify_epoch();
        const int count = int(std::min<size_t>(verify_k, tokens.size() - 1 - consumed));
        if (draft_dump || df_logit_guard_on()) {
            (void)df_draft(anchor, position_base + int(consumed) - 1);
            if (draft_dump) {
                const size_t draft_values =
                    size_t(insignia::glm53::DFlash2Drafter::kDrafts) * vocab;
                require(std::fwrite(df_logits_host_, sizeof(float), draft_values, draft_dump) ==
                            draft_values,
                        "write target-forced DFlash logits");
            }
        }
        std::vector<int> chunk(tokens.begin() + consumed, tokens.begin() + consumed + count);
        capture_offset_ = 0;
        prefill(chunk, position_base + int(consumed), true);
        if (dflash_retry_needed(count)) {
            rollback_kda(0, position_base + int(consumed));
            begin_dflash_exact_retry();
            prefill(chunk, position_base + int(consumed), true);
            end_dflash_exact_retry();
        }
        const size_t values = size_t(count) * vocab;
        check(cudaMemcpy(host.data(), verify_logits_.get(), values * sizeof(float),
                         cudaMemcpyDeviceToHost), "download target-forced logits");
        require(std::fwrite(host.data(), sizeof(float), values, dump) == values,
                "write target-forced logits");
        if (df_calibration_guard_js_ > 0.0f || df_retry_top1_drop_ > 0.0f)
            retain_df_prior_logits(
                verify_logits_.get() + size_t(count - 1) * vocab,
                host.data() + size_t(count - 1) * vocab);
        df_commit(count, position_base + int(consumed));
        anchor = tokens[consumed + size_t(count) - 1];
        consumed += size_t(count);
    }
    verify_may_rollback_ = true;
    std::fclose(dump);
    if (draft_dump) std::fclose(draft_dump);
    std::printf("target-forced logits: %zu records -> %s\n", tokens.size(), dump_path);
    if (draft_path) std::printf("target-forced DFlash logits -> %s\n", draft_path);
}

void Runner::mhc_multi(std::string_view stem, const float *streams, float *collapsed, int tokens) {
    const float *base = device_f32(std::string(stem) + "_base");
    const float *scale = device_f32(std::string(stem) + "_scale");
    const TensorLocation &fn = model_.tensor(std::string(stem) + "_fn");
    require(fn.type == TensorType::bf16 &&
            fn.shape == std::vector<uint32_t>({24, uint32_t(kStreams) * hidden_}),
            "wrong mHC projection geometry");
    const uint16_t *device_fn = reinterpret_cast<const uint16_t *>(stager_.load(fn));
    const int width = kStreams * hidden_;
    for (int token = 0; token < tokens; ++token)
        check(insignia::glm53::mhc_analyze(device_fn, base, scale, streams + size_t(token) * width,
            nullptr,
            c_post_.get() + size_t(token) * 4, c_comb_.get() + size_t(token) * 16,
            collapsed + size_t(token) * hidden_, mhc_workspace_.get(), width), "mHC analyze (prefill)");
}

void Runner::kda_multi(int layer, const float *input, float *output, int tokens, int position_base) {
    const std::string stem = layer_stem(layer) + "self_attn.";
    const int width = kda_width_;
    const int row = kda_row_[size_t(layer)];
    require(row >= 0, "KDA state requested for a non-KDA layer");
    float *history = conv_history_.get() + size_t(row) * 9 * width;
    const bool conv_fp32 = model_.tensor(stem + "q_conv1d.weight").type == TensorType::f32;
    linear_multi(stem + "q_proj.weight", input, c_q_, tokens, width, hidden_);
    if (kda_archive_) archive_kda_rows(layer, c_q_, tokens, 0);
    const uint16_t *conv = reinterpret_cast<const uint16_t *>(stager_.load(stem + "q_conv1d.weight"));
    for (int token = 0; token < tokens; ++token)
        check(insignia::glm53::kda_conv_silu(c_q_.get() + size_t(token) * width, conv, history,
              position_base + token, width, conv_fp32), "KDA q convolution (prefill)");
    linear_multi(stem + "k_proj.weight", input, c_k_, tokens, width, hidden_);
    if (kda_archive_) archive_kda_rows(layer, c_k_, tokens, 1);
    conv = reinterpret_cast<const uint16_t *>(stager_.load(stem + "k_conv1d.weight"));
    for (int token = 0; token < tokens; ++token)
        check(insignia::glm53::kda_conv_silu(c_k_.get() + size_t(token) * width, conv,
              history + 3 * width, position_base + token, width, conv_fp32), "KDA k convolution (prefill)");
    linear_multi(stem + "v_proj.weight", input, c_v_, tokens, width, hidden_);
    if (kda_archive_) archive_kda_rows(layer, c_v_, tokens, 2);
    conv = reinterpret_cast<const uint16_t *>(stager_.load(stem + "v_conv1d.weight"));
    for (int token = 0; token < tokens; ++token)
        check(insignia::glm53::kda_conv_silu(c_v_.get() + size_t(token) * width, conv,
              history + 6 * width, position_base + token, width, conv_fp32), "KDA v convolution (prefill)");

    linear_multi(stem + "f_a_proj.weight", input, c_small_, tokens, f_a_rows_, hidden_);
    linear_multi(stem + "f_b_proj.weight", c_small_, c_gate_, tokens, width, f_a_rows_);
    if (kda_archive_) archive_kda_rows(layer, c_gate_, tokens, 3);
    const float *dt_bias = device_f32(stem + "dt_bias");
    const float *a_log = device_f32(stem + "A_log");
    linear_multi(stem + "b_proj.weight", input, c_beta_, tokens, kda_heads_, hidden_);
    if (kda_archive_) archive_kda_rows(layer, c_beta_, tokens, 4);
    float *state = kda_states_.get() + size_t(row) * width * kda_head_dim_;
    for (int token = 0; token < tokens; ++token) {
        float *gate = c_gate_.get() + size_t(token) * width;
        float *beta = c_beta_.get() + size_t(token) * kda_heads_;
        kda_gate_kernel<<<32, 256>>>(gate, dt_bias, a_log, beta, kda_heads_, width);
        check(cudaGetLastError(), "KDA gate launch (prefill)");
        check(insignia::glm53::kda_decode(state, c_q_.get() + size_t(token) * width,
              c_k_.get() + size_t(token) * width, c_v_.get() + size_t(token) * width,
              gate, beta, c_core_.get() + size_t(token) * width, kda_heads_, kda_head_dim_),
              "KDA recurrence (prefill)");
    }
    linear_multi(stem + "g_a_proj.weight", input, c_small_, tokens, f_a_rows_, hidden_);
    linear_multi(stem + "g_b_proj.weight", c_small_, c_gate_, tokens, width, f_a_rows_);
    const uint16_t *norm = reinterpret_cast<const uint16_t *>(stager_.load(stem + "o_norm.weight"));
    for (int token = 0; token < tokens; ++token)
        kda_output_kernel<<<kda_heads_, kda_head_dim_>>>(c_core_.get() + size_t(token) * width,
            c_gate_.get() + size_t(token) * width, norm, c_proj_.get() + size_t(token) * width);
    check(cudaGetLastError(), "KDA output norm launch (prefill)");
    linear_multi(stem + "o_proj.weight", c_proj_, output, tokens, hidden_, width);
}

void Runner::mla_multi(int layer, const float *input, float *output, int tokens, int position_base) {
    const std::string stem = layer_stem(layer) + "self_attn.";
    linear_multi(stem + "q_a_proj.weight", input, c_small_, tokens, q_a_rows_, hidden_);
    rms(stem + "q_a_layernorm.weight", c_small_, c_small_, tokens, q_a_rows_);
    linear_multi(stem + "q_b_proj.weight", c_small_, c_mlaq_, tokens, q_b_rows_, q_a_rows_);
    linear_multi(stem + "kv_a_proj_with_mqa.weight", input, c_small_, tokens, kv_a_rows_, hidden_);
    rms(stem + "kv_a_layernorm.weight", c_small_, c_small_, tokens, kv_a_rows_);
    const int slot = int(std::find(mla_slot_.begin(), mla_slot_.end(), layer) - mla_slot_.begin());
    if (mla_legacy_) {
        require(position_base + tokens <= kLegacyMlaContext,
                "legacy MLA prefill exceeds 256 tokens");
        linear_multi(stem + "kv_b_proj.weight", c_small_, c_kv_, tokens,
                     kv_b_rows_, kv_a_rows_);
        const size_t expanded_stride = size_t(kLegacyMlaContext) * q_b_rows_;
        check(insignia::glm53::mla_flash2_prefill(
              c_mlaq_, c_kv_,
              mla_keys_.get() + size_t(slot) * expanded_stride,
              mla_values_.get() + size_t(slot) * expanded_stride,
              c_mlao_, tokens, position_base, mla_heads_, mla_head_dim_),
              "legacy FlashAttention-2 MLA prefill");
        linear_multi(stem + "o_proj.weight", c_mlao_, output, tokens, hidden_, q_b_rows_);
        return;
    }
    const size_t layer_stride = size_t(kMaxContext()) * kv_a_rows_;
    uint8_t *cache_u8 = kv_fp8_ ? mla_latent_u8_.get() + size_t(slot) * layer_stride : nullptr;
    float *cache_f32 = kv_fp8_ ? nullptr : mla_latent_f32_.get() + size_t(slot) * layer_stride;
    const float *w_uk = mla_fp8_absorb_ ? nullptr :
        w_uk_.get() + size_t(slot) * absorb_per_layer_;
    const float *w_uv = mla_fp8_absorb_ ? nullptr :
        w_uv_.get() + size_t(slot) * absorb_per_layer_;
    const Q8Stager::ResidentView *absorb_view = mla_fp8_absorb_ ?
        &mla_absorb_fp8_views_[size_t(slot)] : nullptr;
    float *cache_scale = mla_latent_scale_.get() + size_t(slot) * kMaxContext() *
                         insignia::glm53::kMlaLatentGroups;
    // A legal prefill chunk may straddle 256 (for example 192..287 with
    // PREFILL_CHUNK=96). Execute and save the overlapping exact rows first,
    // then dispatch only the suffix from position 256.
    const int prefix_tokens =
        insignia::glm53::mla_exact_prefix_overlap_tokens(
            tokens, position_base);
    const int long_tokens = tokens - prefix_tokens;
    const int long_position_base = position_base + prefix_tokens;
    if (prefix_tokens) {
        check(insignia::glm53::mla_store_latent(
              c_small_.get(), cache_u8, cache_scale, cache_f32,
              prefix_tokens, position_base, kv_a_rows_),
              "MLA exact-prefix latent shadow prefill store");
        const size_t prefix_stride = size_t(kLegacyMlaContext) * kv_a_rows_;
        if (mla_prefix_reconstruct_ || mla_cross_head_fp8_) {
            check(cudaMemcpyAsync(
                  mla_prefix_latent_.get() + size_t(slot) * prefix_stride +
                      size_t(position_base) * kv_a_rows_,
                  c_small_.get(),
                  size_t(prefix_tokens) * kv_a_rows_ * sizeof(float),
                  cudaMemcpyDeviceToDevice), "save exact MLA prefix latents (prefill)");
        }
        if (mla_prefix_reconstruct_) {
            reconstruct_mla_prefix(
                slot, position_base + prefix_tokens, position_base);
            check(insignia::glm53::mla_flash2_prefill_reconstructed(
                  c_mlaq_.get(), mla_prefix_kv_.get(), c_mlao_.get(),
                  prefix_tokens,
                  position_base, mla_heads_, mla_head_dim_),
                  "reconstructed exact FlashAttention-2 MLA prefix");
        } else {
            linear_multi(stem + "kv_b_proj.weight", c_small_, c_kv_,
                         prefix_tokens, kv_b_rows_, kv_a_rows_);
            const size_t expanded_stride = size_t(kLegacyMlaContext) * q_b_rows_;
            check(insignia::glm53::mla_flash2_prefill(
                  c_mlaq_.get(), c_kv_.get(),
                  mla_keys_.get() + size_t(slot) * expanded_stride,
                  mla_values_.get() + size_t(slot) * expanded_stride,
                  c_mlao_.get(), prefix_tokens, position_base,
                  mla_heads_, mla_head_dim_),
                  "exact FlashAttention-2 MLA prefix");
        }
    }
    static const bool scalar_attention =
        std::getenv("INSIGNIA_GLM53_SCALAR_MLA_PREFILL") != nullptr;
    const float *long_query =
        c_mlaq_.get() + size_t(prefix_tokens) * q_b_rows_;
    const float *long_latent =
        c_small_.get() + size_t(prefix_tokens) * kv_a_rows_;
    float *long_output = c_mlao_.get() + size_t(prefix_tokens) * q_b_rows_;
    auto decode_long_rows = [&] {
        for (int token = 0; token < long_tokens; ++token) {
            if (mla_fp8_absorb_) {
                if (mla_cross_head_fp8_)
                    check(insignia::glm53::mla_decode_latent_cross_head_fp8_absorb(
                          long_query + size_t(token) * q_b_rows_,
                          long_latent + size_t(token) * kv_a_rows_,
                          cache_u8, cache_scale,
                          absorb_view->weights, absorb_view->scales,
                          mla_qeff_u8_.get(), mla_qeff_scale_.get(),
                          mla_partial_.get(),
                          long_output + size_t(token) * q_b_rows_,
                          long_position_base + token,
                          mla_heads_, mla_head_dim_, kv_a_rows_,
                          mla_prefix_latent_.get() + size_t(slot) *
                              kLegacyMlaContext * kv_a_rows_,
                          mla_qeff_f32_.get(),
                          mla_prefix_parallel_ ? mla_prefix_partial_.get() : nullptr,
                          mla_prefix_parallel_),
                          "cross-head exact-prefix MLA decode rows (prefill)");
                else
                    check(insignia::glm53::mla_decode_latent_fp8_absorb(
                          long_query + size_t(token) * q_b_rows_,
                          long_latent + size_t(token) * kv_a_rows_,
                          cache_u8, cache_scale, cache_f32,
                          absorb_view->weights, absorb_view->scales,
                          mla_partial_.get(),
                          long_output + size_t(token) * q_b_rows_,
                          long_position_base + token,
                          mla_heads_, mla_head_dim_, kv_a_rows_),
                          "scalar compact-absorb MLA attention (prefill)");
            } else {
                check(insignia::glm53::mla_decode_latent(
                      long_query + size_t(token) * q_b_rows_,
                      long_latent + size_t(token) * kv_a_rows_,
                      nullptr, cache_u8, cache_scale, cache_f32, nullptr,
                      w_uk, w_uv, mla_partial_.get(),
                      long_output + size_t(token) * q_b_rows_,
                      long_position_base + token,
                      mla_heads_, mla_head_dim_, kv_a_rows_),
                      "scalar MLA attention (prefill)");
            }
        }
    };
    if (long_tokens && scalar_attention) {
        decode_long_rows();
    } else if (long_tokens) {
        const bool cross_prefill = mla_cross_head_fp8_ &&
            insignia::glm53::mla_cross_head_use_fused_prefill(
                long_tokens, long_position_base);
        if (mla_fp8_absorb_ && cross_prefill)
            check(insignia::glm53::mla_prefill_latent_cross_head_fp8_absorb(
                  long_query, long_latent, cache_u8, cache_scale,
                  absorb_view->weights, absorb_view->scales,
                  mla_qeff_u8_.get(), mla_qeff_scale_.get(),
                  long_output, long_tokens, long_position_base,
                  mla_heads_, mla_head_dim_, kv_a_rows_,
                  mla_prefix_latent_.get() + size_t(slot) *
                      kLegacyMlaContext * kv_a_rows_,
                  mla_qeff_f32_.get()),
                  "cross-head FP8 MLA latent prefill");
        else if (mla_fp8_absorb_ && mla_cross_head_fp8_)
            decode_long_rows();
        else if (mla_fp8_absorb_)
            check(insignia::glm53::mla_prefill_latent_fp8_absorb(
                  long_query, long_latent, cache_u8, cache_scale, cache_f32,
                  absorb_view->weights, absorb_view->scales, long_output,
                  long_tokens, long_position_base,
                  mla_heads_, mla_head_dim_, kv_a_rows_),
                  "compact-absorb MLA latent prefill");
        else
            check(insignia::glm53::mla_prefill_latent(long_query, long_latent,
                  nullptr, cache_u8, cache_scale, cache_f32, nullptr,
                  w_uk, w_uv, long_output, long_tokens, long_position_base,
                  mla_heads_, mla_head_dim_, kv_a_rows_),
                  "MLA latent prefill");
    }
    if (std::getenv("INSIGNIA_GLM53_MLA_DUMP") && layer == mla_slot_.front() &&
        position_base == 0 && tokens > 1) {
        static bool dumped = false;
        if (!dumped) {
            dumped = true;
            const std::filesystem::path dir = std::getenv("INSIGNIA_GLM53_MLA_DUMP");
            std::filesystem::create_directories(dir);
            auto save_dev = [&](const char *name, const void *dev, size_t bytes) {
                std::vector<uint8_t> host(bytes);
                check(cudaMemcpy(host.data(), dev, bytes, cudaMemcpyDeviceToHost), name);
                std::FILE *file = std::fopen((dir / name).string().c_str(), "wb");
                std::fwrite(host.data(), 1, bytes, file);
                std::fclose(file);
            };
            save_dev("q.bin", c_mlaq_.get(), size_t(2) * mla_head_dim_ * sizeof(float));
            save_dev("latent.bin", c_small_.get(), size_t(kv_a_rows_) * sizeof(float));
            save_dev("out.bin", c_mlao_.get(), size_t(2) * mla_head_dim_ * sizeof(float));
            save_dev("cache.bin", kv_fp8_ ? (const void *)cache_u8 : (const void *)cache_f32,
                     size_t(tokens) * kv_a_rows_ * (kv_fp8_ ? 1 : 4));
            save_dev("scales.bin", mla_latent_scale_.get() + size_t(slot) * kMaxContext() *
                         insignia::glm53::kMlaLatentGroups,
                     size_t(tokens) * insignia::glm53::kMlaLatentGroups * sizeof(float));
            save_dev("wuk.bin", w_uk, size_t(mla_head_dim_) * kv_a_rows_ * sizeof(float));
            save_dev("wuv.bin", w_uv, size_t(mla_head_dim_) * kv_a_rows_ * sizeof(float));
            std::fprintf(stderr, "mla dump written to %s\n", dir.string().c_str());
        }
    }
    linear_multi(stem + "o_proj.weight", c_mlao_, output, tokens, hidden_, q_b_rows_);
}

void Runner::mlp_multi(std::string_view stem, const float *input, float *output,
                       int tokens, int intermediate) {
    linear_multi(std::string(stem) + "gate_proj.weight", input, c_gateu_, tokens, intermediate, hidden_);
    linear_multi(std::string(stem) + "up_proj.weight", input, c_up_, tokens, intermediate, hidden_);
    launch_clamped_swiglu(c_gateu_, c_up_, c_act_, tokens * intermediate);
    linear_multi(std::string(stem) + "down_proj.weight", c_act_, output, tokens, hidden_, intermediate);
}

void Runner::report_moe_metrics(
    int layer, const std::vector<std::vector<std::pair<int, float>>> &selection,
    const std::vector<std::array<uint16_t, 32>> &candidate_experts,
    const std::vector<std::array<float, 32>> &candidate_logits,
    const std::vector<std::array<float, 32>> &candidate_choice,
    const std::vector<std::array<float, 8>> &router_summary,
    const std::vector<std::array<uint32_t, 4>> &candidate_residency,
    const float *input, int tokens) {
    const bool write_exact_trace = falsifier_trace_ && !falsifier_feature_only_;
    require((moe_metrics_ || write_exact_trace) && moe_topk_ == 8,
            "invalid MoE diagnostics state");
    require(selection.size() == size_t(tokens) &&
                (!write_exact_trace ||
                 (candidate_experts.size() == size_t(tokens) &&
                  candidate_logits.size() == size_t(tokens) &&
                  candidate_choice.size() == size_t(tokens) &&
                  router_summary.size() == size_t(tokens) &&
                  candidate_residency.size() == size_t(tokens))),
            "incomplete MoE diagnostics context");
    const size_t expert_values = size_t(tokens) * moe_topk_ * hidden_;
    const size_t exact_values = size_t(tokens) * hidden_;
    moe_metrics_expert_.resize(expert_values);
    moe_metrics_exact_.resize(exact_values);
    check(cudaMemcpy(moe_metrics_expert_.data(), c_expert_out_.get(),
                     expert_values * sizeof(float), cudaMemcpyDeviceToHost),
          "download MoE expert contributions");
    check(cudaMemcpy(moe_metrics_exact_.data(), c_routed_.get(),
                     exact_values * sizeof(float), cudaMemcpyDeviceToHost),
          "download exact routed output");
    if (write_exact_trace) {
        moe_metrics_input_.resize(exact_values);
        check(cudaMemcpy(moe_metrics_input_.data(), input,
                         exact_values * sizeof(float), cudaMemcpyDeviceToHost),
              "download falsifier hidden input");
    }

    for (int row = 0; row < tokens; ++row) {
        const float *expert = moe_metrics_expert_.data() +
            size_t(row) * moe_topk_ * hidden_;
        const float *exact = moe_metrics_exact_.data() + size_t(row) * hidden_;
        std::array<double, 8> term_norm{};
        float original_sum = 0.0f;
        for (int slot = 0; slot < moe_topk_; ++slot) {
            const float weight = selection[size_t(row)][size_t(slot)].second;
            original_sum += weight;
            double norm2 = 0.0;
            for (int column = 0; column < hidden_; ++column) {
                const double value = double(weight) *
                    expert[size_t(slot) * hidden_ + size_t(column)];
                norm2 += value * value;
            }
            term_norm[size_t(slot)] = std::sqrt(norm2);
        }
        double exact_norm2 = 0.0, exact_term_sum = 0.0, replay_max = 0.0;
        for (double norm : term_norm) exact_term_sum += norm;
        for (int column = 0; column < hidden_; ++column) {
            const double value = exact[column];
            exact_norm2 += value * value;
            float replay = 0.0f;
            for (int slot = 0; slot < moe_topk_; ++slot)
                replay = std::fmaf(selection[size_t(row)][size_t(slot)].second,
                                   expert[size_t(slot) * hidden_ + size_t(column)], replay);
            replay_max = std::max(replay_max, std::fabs(double(replay) - value));
        }
        const double exact_norm = std::sqrt(exact_norm2);
        const double exact_cancel = exact_norm > 0.0 ? exact_term_sum / exact_norm : 0.0;

        if (write_exact_trace) {
            DfFalsifierEventV2 record{};
            record.epoch = verify_epoch_;
            record.layer = uint16_t(layer);
            record.row = uint8_t(row);
            record.tokens = uint8_t(tokens);
            const int verify_row = capture_offset_ + row;
            require(verify_row >= 0 && verify_row <= 255,
                    "falsifier verify row is out of range");
            record.verify_row = uint8_t(verify_row);
            record.exec_k = 8;
            record.flags = uint16_t((is_mla_[size_t(layer)] ? 1u : 0u) |
                                    (kda_archive_ ? 2u : 0u));
            record.candidate_residency = candidate_residency[size_t(row)];
            for (int slot = 0; slot < moe_topk_; ++slot) {
                record.expert[size_t(slot)] =
                    uint16_t(selection[size_t(row)][size_t(slot)].first);
                record.weight[size_t(slot)] =
                    selection[size_t(row)][size_t(slot)].second;
            }
            record.candidate_expert = candidate_experts[size_t(row)];
            record.candidate_logit = candidate_logits[size_t(row)];
            record.candidate_choice = candidate_choice[size_t(row)];
            record.router_summary = router_summary[size_t(row)];

            const float *hidden = moe_metrics_input_.data() + size_t(row) * hidden_;
            double input_norm2 = 0.0;
            for (int column = 0; column < hidden_; ++column) {
                uint32_t hash = uint32_t(column + 1) * 0x9e3779b1u;
                hash ^= hash >> 16;
                hash *= 0x85ebca6bu;
                hash ^= hash >> 13;
                const int bucket = int(hash & 63u);
                record.hidden_countsketch[size_t(bucket)] +=
                    (hash & 64u) ? hidden[column] : -hidden[column];
                input_norm2 += double(hidden[column]) * hidden[column];
            }
            const float sketch_scale = std::sqrt(64.0f / float(hidden_));
            for (float &value : record.hidden_countsketch) value *= sketch_scale;

            std::array<double, 36> gram{};
            for (int column = 0; column < hidden_; ++column) {
                std::array<double, 8> contribution{};
                for (int slot = 0; slot < moe_topk_; ++slot)
                    contribution[size_t(slot)] =
                        double(selection[size_t(row)][size_t(slot)].second) *
                        expert[size_t(slot) * hidden_ + size_t(column)];
                int index = 0;
                for (int left = 0; left < moe_topk_; ++left)
                    for (int right = left; right < moe_topk_; ++right)
                        gram[size_t(index++)] += contribution[size_t(left)] *
                                                 contribution[size_t(right)];
            }
            for (size_t index = 0; index < gram.size(); ++index)
                record.contribution_gram[index] = float(gram[index] / hidden_);
            record.tail = {float(exact_norm2 / hidden_), float(exact_cancel),
                           float(replay_max), float(input_norm2 / hidden_)};
            require(std::fwrite(&record, sizeof(record), 1, falsifier_trace_) == 1,
                    "write DFlash falsifier event");
        }

        if (!moe_metrics_) continue;

        float retained_sum = 0.0f;
        double retained_term_sum = 0.0;
        for (int topm = 1; topm < moe_topk_; ++topm) {
            retained_sum += selection[size_t(row)][size_t(topm - 1)].second;
            retained_term_sum += term_norm[size_t(topm - 1)];
            for (int renorm = 0; renorm <= 1; ++renorm) {
                const float weight_scale = renorm ? original_sum / retained_sum : 1.0f;
                double error2 = 0.0, dot = 0.0, approx_norm2 = 0.0, max_abs = 0.0;
                for (int column = 0; column < hidden_; ++column) {
                    float approximation = 0.0f;
                    for (int slot = 0; slot < topm; ++slot) {
                        float weight = selection[size_t(row)][size_t(slot)].second;
                        weight *= weight_scale;
                        approximation = std::fmaf(
                            weight, expert[size_t(slot) * hidden_ + size_t(column)], approximation);
                    }
                    const double approximate = approximation;
                    const double reference = exact[column];
                    const double difference = approximate - reference;
                    error2 += difference * difference;
                    dot += approximate * reference;
                    approx_norm2 += approximate * approximate;
                    max_abs = std::max(max_abs, std::fabs(difference));
                }
                const double approx_norm = std::sqrt(approx_norm2);
                const double cosine = exact_norm > 0.0 && approx_norm > 0.0
                    ? std::clamp(dot / (exact_norm * approx_norm), -1.0, 1.0) : 0.0;
                const double approx_cancel = approx_norm > 0.0
                    ? retained_term_sum * std::fabs(double(weight_scale)) / approx_norm : 0.0;
                std::fprintf(moe_metrics_,
                    "%u,%d,%d,%d,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                    verify_epoch_, layer, row, topm, renorm ? "renorm" : "zero",
                    error2 / hidden_, exact_norm2 > 0.0 ? std::sqrt(error2 / exact_norm2) : 0.0,
                    cosine, max_abs, exact_norm > 0.0 ? approx_norm / exact_norm : 0.0,
                    retained_sum / original_sum, exact_cancel, approx_cancel, replay_max);
            }
        }
    }
    if (moe_metrics_) std::fflush(moe_metrics_);
    if (write_exact_trace) std::fflush(falsifier_trace_);
}

void Runner::report_falsifier_features(
    int layer, const std::vector<std::vector<std::pair<int, float>>> &selection,
    const std::vector<int> &exec_count,
    const std::vector<std::array<uint16_t, 32>> &candidate_experts,
    const std::vector<std::array<float, 32>> &candidate_logits,
    const std::vector<std::array<float, 32>> &candidate_choice,
    const std::vector<std::array<float, 8>> &router_summary,
    const std::vector<std::array<uint32_t, 4>> &candidate_residency,
    const float *input, int tokens) {
    require(falsifier_trace_ && falsifier_feature_only_ && kda_archive_ && moe_topk_ == 8,
            "invalid feature-only falsifier trace state");
    require(selection.size() == size_t(tokens) && exec_count.size() == size_t(tokens) &&
                candidate_experts.size() == size_t(tokens) &&
                candidate_logits.size() == size_t(tokens) &&
                candidate_choice.size() == size_t(tokens) &&
                router_summary.size() == size_t(tokens) &&
                candidate_residency.size() == size_t(tokens),
            "incomplete feature-only falsifier context");

    const size_t input_values = size_t(tokens) * hidden_;
    moe_metrics_input_.resize(input_values);
    check(cudaMemcpy(moe_metrics_input_.data(), input, input_values * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "download feature-only falsifier hidden input");
    for (int row = 0; row < tokens; ++row) {
        require(selection[size_t(row)].size() == size_t(moe_topk_) &&
                    exec_count[size_t(row)] >= 1 && exec_count[size_t(row)] <= moe_topk_,
                "invalid feature-only falsifier action");
        DfFalsifierEventV2 record{};
        record.epoch = verify_epoch_;
        record.layer = uint16_t(layer);
        record.row = uint8_t(row);
        record.tokens = uint8_t(tokens);
        const int verify_row = capture_offset_ + row;
        require(verify_row >= 0 && verify_row <= 255,
                "feature-only falsifier verify row is out of range");
        record.verify_row = uint8_t(verify_row);
        record.exec_k = uint8_t(exec_count[size_t(row)]);
        record.flags = uint16_t((is_mla_[size_t(layer)] ? 1u : 0u) |
                                (kda_archive_ ? 2u : 0u) | 4u);
        record.candidate_residency = candidate_residency[size_t(row)];
        for (int slot = 0; slot < moe_topk_; ++slot) {
            record.expert[size_t(slot)] =
                uint16_t(selection[size_t(row)][size_t(slot)].first);
            record.weight[size_t(slot)] =
                selection[size_t(row)][size_t(slot)].second;
        }
        record.candidate_expert = candidate_experts[size_t(row)];
        record.candidate_logit = candidate_logits[size_t(row)];
        record.candidate_choice = candidate_choice[size_t(row)];
        record.router_summary = router_summary[size_t(row)];

        const float *hidden = moe_metrics_input_.data() + size_t(row) * hidden_;
        double input_norm2 = 0.0;
        for (int column = 0; column < hidden_; ++column) {
            uint32_t hash = uint32_t(column + 1) * 0x9e3779b1u;
            hash ^= hash >> 16;
            hash *= 0x85ebca6bu;
            hash ^= hash >> 13;
            const int bucket = int(hash & 63u);
            record.hidden_countsketch[size_t(bucket)] +=
                (hash & 64u) ? hidden[column] : -hidden[column];
            input_norm2 += double(hidden[column]) * hidden[column];
        }
        const float sketch_scale = std::sqrt(64.0f / float(hidden_));
        for (float &value : record.hidden_countsketch) value *= sketch_scale;
        // Gram and the first three tail values are labels and deliberately
        // remain zero in an on-policy feature trace. The input norm is causal.
        record.tail[3] = float(input_norm2 / hidden_);
        require(std::fwrite(&record, sizeof(record), 1, falsifier_trace_) == 1,
                "write feature-only DFlash falsifier event");
    }
    std::fflush(falsifier_trace_);
}

// Router scores for the whole chunk download once; each distinct expert's
// three matrices stage once and multiply against only the tokens that
// selected it, so expert I/O scales with distinct selections, not tokens.
void Runner::moe_multi(int layer, const float *input, float *output, int tokens) {
    const std::string stem = layer_stem(layer) + "mlp.";
    const int experts = moe_experts_;
    const int topk = moe_topk_;
    // Approximation normally remains restricted to provisional DFlash verify
    // rows. An explicit full-layer-major prefill experiment may reuse the same
    // policy; scalar decode and the unset prefill path retain all eight routed
    // experts and keep the existing arithmetic and traffic exactly intact.
    const bool approximate_prefill =
        full_layer_major_active_ && prefill_approx_moe_ &&
        layer >= prefill_approx_first_layer_;
    const bool approximate_pass = kda_archive_ || approximate_prefill;
    const bool approximate_moe = approximate_pass &&
        (df_approx_topm_ || df_approx_mass_ > 0.0f || df_cache_route_k_);
    std::vector<int> exec_count(size_t(tokens), topk);
    linear_multi(stem + "gate.weight", input, c_router_, tokens, experts, hidden_);
    std::vector<float> logits(size_t(tokens) * experts);
    check(cudaMemcpy(logits.data(), c_router_.get(), logits.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "download router logits (prefill)");
    const std::vector<float> &bias = host_f32(stem + "gate.e_score_correction_bias");
    require(bias.size() == size_t(experts), "wrong router bias geometry");
    std::vector<std::vector<std::pair<int, float>>> selection(tokens);
    // Prompt prefill is not part of the speculative verify trajectory; only
    // widen and emit the fixed records once the KDA replay archive is active.
    const bool trace_falsifier = falsifier_trace_ != nullptr && kda_archive_ &&
        !df_retry_replay_;
    const bool widen_router = trace_falsifier ||
        (approximate_pass && df_cache_route_k_);
    const size_t widened_rows = widen_router ? size_t(tokens) : 0;
    std::vector<std::array<uint16_t, 32>> candidate_experts(widened_rows);
    std::vector<std::array<float, 32>> candidate_logits(widened_rows);
    std::vector<std::array<float, 32>> candidate_choice(widened_rows);
    std::vector<std::array<float, 8>> router_summary(widened_rows);
    std::vector<std::array<uint32_t, 4>> candidate_residency(widened_rows);
    std::vector<float> baseline_denominator(widened_rows);
    std::vector<std::array<int, 8>> baseline_experts(
        approximate_pass && df_cache_route_k_ ? size_t(tokens) : 0);
    for (int token = 0; token < tokens; ++token) {
        const float *row = logits.data() + size_t(token) * experts;
        std::vector<float> scores(experts), choice(experts);
        for (int expert = 0; expert < experts; ++expert) {
            scores[expert] = 1.0f / (1.0f + std::exp(-row[expert]));
            choice[expert] = scores[expert] + bias[expert];
        }
        std::vector<int> order(experts);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + topk, order.end(),
            [&](int left, int right) { return choice[left] > choice[right]; });
        float denominator = 0.0f;
        for (int slot = 0; slot < topk; ++slot) denominator += scores[order[slot]];
        for (int slot = 0; slot < topk; ++slot) {
            selection[token].emplace_back(order[slot], 2.5f * scores[order[slot]] / denominator);
            if (!baseline_experts.empty())
                baseline_experts[size_t(token)][size_t(slot)] = order[slot];
        }
        if (widen_router) {
            baseline_denominator[size_t(token)] = denominator;
            double raw_sum = 0.0, raw_square_sum = 0.0, score_sum = 0.0;
            float raw_max = -std::numeric_limits<float>::infinity();
            float raw_second = raw_max;
            for (int expert = 0; expert < experts; ++expert) {
                raw_sum += row[expert];
                raw_square_sum += double(row[expert]) * row[expert];
                score_sum += scores[expert];
                if (row[expert] > raw_max) {
                    raw_second = raw_max;
                    raw_max = row[expert];
                } else if (row[expert] > raw_second) {
                    raw_second = row[expert];
                }
            }
            float selected_entropy = 0.0f;
            for (int slot = 0; slot < topk; ++slot) {
                const float probability = scores[order[slot]] / denominator;
                selected_entropy -= probability * std::log(std::max(probability, 1.0e-30f));
            }
            std::vector<int> candidates(experts);
            std::iota(candidates.begin(), candidates.end(), 0);
            std::partial_sort(candidates.begin(), candidates.begin() + 32, candidates.end(),
                [&](int left, int right) { return choice[left] > choice[right]; });
            for (int rank = 0; rank < 32; ++rank) {
                const int expert = candidates[size_t(rank)];
                candidate_experts[size_t(token)][size_t(rank)] = uint16_t(expert);
                candidate_logits[size_t(token)][size_t(rank)] = row[expert];
                candidate_choice[size_t(token)][size_t(rank)] = choice[expert];
            }
            const double raw_mean = raw_sum / experts;
            const double raw_variance =
                std::max(0.0, raw_square_sum / experts - raw_mean * raw_mean);
            router_summary[size_t(token)] = {
                float(raw_mean), float(std::sqrt(raw_variance)), raw_max, raw_second,
                float(score_sum), float(denominator / score_sum),
                choice[order[0]] - choice[order[1]], selected_entropy};
            candidate_residency[size_t(token)] = expert_stager_->residency_masks(
                layer, candidate_experts[size_t(token)].data(), 32);
        }
        const int verify_row = capture_offset_ + token;
        const bool logit_guarded = kda_archive_ && df_logit_guard_on() &&
            verify_row >= 0 && verify_row < kMaxVerify &&
            df_logit_guard_exact_[size_t(verify_row)];
        const int cache_route_retain = logit_guarded
            ? df_cache_guard_retain_ : df_cache_route_retain_;
        if (approximate_pass && df_cache_route_k_ && !df_approx_topm_ &&
            df_cache_route_regret_ > 0.0f &&
            cache_route_retain < 8 && !df_cache_joint_options_) {
            const auto &candidates = candidate_experts[size_t(token)];
            const auto &residency = candidate_residency[size_t(token)];
            std::array<int, 288> candidate_rank{};
            candidate_rank.fill(-1);
            for (int rank = 0; rank < df_cache_route_k_; ++rank)
                candidate_rank[size_t(candidates[size_t(rank)])] = rank;
            const auto &baseline = baseline_experts[size_t(token)];
            double baseline_score = 0.0, score_scale = 0.0;
            for (int slot = 0; slot < topk; ++slot) {
                baseline_score += choice[size_t(baseline[size_t(slot)])];
                score_scale += std::fabs(choice[size_t(baseline[size_t(slot)])]);
            }
            auto transfer = [&](int expert) {
                const int rank = candidate_rank[size_t(expert)];
                require(rank >= 0, "cache-aware baseline expert missing from candidate pool");
                const uint32_t bit = uint32_t(1u) << rank;
                const bool device = (residency[2] & bit) != 0;
                const bool host = (residency[0] & bit) != 0;
                const bool inflight = (residency[1] & bit) != 0;
                return std::pair<int, int>{device ? 0 : !(host || inflight), device ? 0 : 1};
            };
            std::array<int, 8> best = baseline;
            int best_disk = 0, best_h2d = 0;
            for (int expert : baseline) {
                const auto [disk, h2d] = transfer(expert);
                best_disk += disk;
                best_h2d += h2d;
            }
            const int baseline_disk = best_disk, baseline_h2d = best_h2d;
            double best_regret = 0.0;
            int best_substitutions = 0;
            auto consider = [&](int first_tail, int second_tail) {
                std::array<int, 8> trial{};
                for (int slot = 0; slot < cache_route_retain; ++slot)
                    trial[size_t(slot)] = baseline[size_t(slot)];
                trial[size_t(cache_route_retain)] = first_tail;
                if (cache_route_retain == 6) trial[7] = second_tail;
                std::sort(trial.begin(), trial.end(), [&](int left, int right) {
                    return candidate_rank[size_t(left)] < candidate_rank[size_t(right)];
                });
                double selected_score = 0.0;
                int disk = 0, h2d = 0, substitutions = 0;
                for (int expert : trial) {
                    selected_score += choice[size_t(expert)];
                    const auto [d, h] = transfer(expert);
                    disk += d;
                    h2d += h;
                    substitutions += std::find(baseline.begin(), baseline.end(), expert) ==
                                     baseline.end();
                }
                const double regret = std::max(0.0, baseline_score - selected_score);
                const double ratio = score_scale > 0.0 ? regret / score_scale : 0.0;
                if (ratio > double(df_cache_route_regret_) + 2.0e-7) return;
                const bool better = disk < best_disk ||
                    (disk == best_disk && (h2d < best_h2d ||
                     (h2d == best_h2d && (regret < best_regret ||
                      (regret == best_regret && substitutions < best_substitutions)))));
                if (better) {
                    best = trial;
                    best_disk = disk;
                    best_h2d = h2d;
                    best_regret = regret;
                    best_substitutions = substitutions;
                }
            };
            std::vector<int> pool;
            for (int rank = 0; rank < df_cache_route_k_; ++rank) {
                const int expert = candidates[size_t(rank)];
                if (std::find(baseline.begin(),
                              baseline.begin() + cache_route_retain, expert) ==
                    baseline.begin() + cache_route_retain)
                    pool.push_back(expert);
            }
            if (cache_route_retain == 7) {
                for (int expert : pool) consider(expert, -1);
            } else {
                for (size_t first = 0; first < pool.size(); ++first)
                    for (size_t second = first + 1; second < pool.size(); ++second)
                        consider(pool[first], pool[second]);
            }
            if (best_substitutions) {
                float selected_denominator = 0.0f;
                for (int expert : best) selected_denominator += scores[size_t(expert)];
                selection[size_t(token)].clear();
                for (int expert : best)
                    selection[size_t(token)].emplace_back(
                        expert, 2.5f * scores[size_t(expert)] / selected_denominator);
            }
            ++df_cache_route_rows_;
            df_cache_route_changed_ += best_substitutions != 0;
            df_cache_route_substitutions_ += uint64_t(best_substitutions);
            const double regret_ratio = score_scale > 0.0 ? best_regret / score_scale : 0.0;
            df_cache_route_regret_sum_ += regret_ratio;
            df_cache_route_regret_max_ = std::max(df_cache_route_regret_max_, regret_ratio);
            df_cache_route_disk_saved_ += baseline_disk - best_disk;
            df_cache_route_h2d_saved_ += baseline_h2d - best_h2d;
        }
        if (logit_guarded) {
            const int guarded_k = df_logit_guard_k_[size_t(verify_row)]
                ? int(df_logit_guard_k_[size_t(verify_row)]) : topk;
            exec_count[size_t(token)] = guarded_k;
        }
        else if (approximate_pass && df_approx_topm_)
            exec_count[size_t(token)] = df_approx_topm_;
        else if (approximate_pass && df_approx_mass_ > 0.0f) {
            float original_sum = 0.0f;
            for (int slot = 0; slot < topk; ++slot)
                original_sum += selection[token][size_t(slot)].second;
            float retained_sum = 0.0f;
            int chosen = df_approx_max_k_;
            for (int slot = 0; slot < df_approx_max_k_; ++slot) {
                retained_sum += selection[token][size_t(slot)].second;
                if (slot + 1 >= df_approx_min_k_ &&
                    retained_sum >= df_approx_mass_ * original_sum) {
                    chosen = slot + 1;
                    break;
                }
            }
            exec_count[size_t(token)] = chosen;
        }
        if (exec_count[size_t(token)] != topk && df_approx_renorm_) {
            float original_sum = 0.0f, retained_sum = 0.0f;
            for (int slot = 0; slot < topk; ++slot)
                original_sum += selection[token][size_t(slot)].second;
            for (int slot = 0; slot < exec_count[size_t(token)]; ++slot)
                retained_sum += selection[token][size_t(slot)].second;
            const float scale = original_sum / retained_sum;
            for (int slot = 0; slot < exec_count[size_t(token)]; ++slot)
                selection[token][size_t(slot)].second *= scale;
        }
    }
    // Compute-heavy cache-aware pruning: preserve the strongest M-1 experts
    // and choose only the Mth executed expert from a widened router frontier.
    // The CPU evaluates the candidate residency and, for k<=4 verify rows, an
    // exact Cartesian layer-union objective.  This deliberately spends cheap
    // Raptor Lake scalar work to avoid NVMe reads and PCIe uploads.  We retain
    // the original Top-8 denominator, so this is fixed-Top-M zero-fill with a
    // bounded tail substitution rather than a hidden renormalization.
    if (approximate_pass && df_approx_topm_ && df_cache_route_k_ &&
        df_cache_route_regret_ > 0.0f) {
        const auto selector_started = std::chrono::steady_clock::now();
        const bool mask_search = df_cache_mask_search_ && df_cache_joint_options_ &&
            tokens >= 2 && tokens <= 4;
        struct PrunedCacheChoice {
            int expert = -1;
            double regret = 0.0, ratio = 0.0;
            int disk = 0, h2d = 0;
        };
        const int retained = df_cache_route_retain_;
        const int topm = df_approx_topm_;
        std::vector<std::vector<PrunedCacheChoice>> options(
            static_cast<size_t>(tokens));
        std::vector<std::array<int, 288>> candidate_rank(
            static_cast<size_t>(tokens));
        std::vector<uint8_t> guarded(size_t(tokens), 0);
        std::vector<uint8_t> guarded_exec_k(
            static_cast<size_t>(tokens), static_cast<uint8_t>(topm));
        std::array<int, 288> union_disk{}, union_h2d{};
        union_disk.fill(-1);
        union_h2d.fill(-1);
        for (int token = 0; token < tokens; ++token) {
            auto &rank = candidate_rank[size_t(token)];
            rank.fill(-1);
            const auto &candidates = candidate_experts[size_t(token)];
            const auto &residency = candidate_residency[size_t(token)];
            for (int candidate = 0; candidate < df_cache_route_k_; ++candidate) {
                const int expert = candidates[size_t(candidate)];
                rank[size_t(expert)] = candidate;
                const uint32_t bit = uint32_t(1u) << candidate;
                const bool device = (residency[2] & bit) != 0;
                const bool host = (residency[0] & bit) != 0;
                const bool inflight = (residency[1] & bit) != 0;
                if (union_disk[size_t(expert)] < 0) {
                    union_disk[size_t(expert)] = device ? 0 : !(host || inflight);
                    union_h2d[size_t(expert)] = device ? 0 : 1;
                }
            }
            const int verify_row = capture_offset_ + token;
            guarded[size_t(token)] = kda_archive_ && df_logit_guard_on() &&
                verify_row >= 0 &&
                verify_row < kMaxVerify &&
                df_logit_guard_exact_[size_t(verify_row)];
            if (guarded[size_t(token)])
                guarded_exec_k[size_t(token)] = df_logit_guard_k_[size_t(verify_row)]
                    ? df_logit_guard_k_[size_t(verify_row)] : uint8_t(topk);
            const auto &baseline = baseline_experts[size_t(token)];
            double baseline_score = 0.0, score_scale = 0.0;
            for (int slot = 0; slot < topm; ++slot) {
                const int candidate = rank[size_t(baseline[size_t(slot)])];
                require(candidate >= 0,
                        "pruned cache baseline missing from candidate pool");
                const double value = candidate_choice[size_t(token)][size_t(candidate)];
                baseline_score += value;
                score_scale += std::fabs(value);
            }
            double prefix_score = 0.0;
            for (int slot = 0; slot < retained; ++slot) {
                const int candidate = rank[size_t(baseline[size_t(slot)])];
                prefix_score += candidate_choice[size_t(token)][size_t(candidate)];
            }
            auto append = [&](int expert) {
                if (std::find(baseline.begin(), baseline.begin() + retained, expert) !=
                    baseline.begin() + retained)
                    return;
                const int candidate = rank[size_t(expert)];
                if (candidate < 0) return;
                PrunedCacheChoice action;
                action.expert = expert;
                const double selected_score = prefix_score +
                    candidate_choice[size_t(token)][size_t(candidate)];
                action.regret = std::max(0.0, baseline_score - selected_score);
                action.ratio = score_scale > 0.0 ? action.regret / score_scale : 0.0;
                if (action.ratio > double(df_cache_route_regret_) + 2.0e-7) return;
                for (int slot = 0; slot < retained; ++slot) {
                    const int prefix_expert = baseline[size_t(slot)];
                    require(union_disk[size_t(prefix_expert)] >= 0,
                            "pruned cache prefix has no residency");
                    action.disk += union_disk[size_t(prefix_expert)];
                    action.h2d += union_h2d[size_t(prefix_expert)];
                }
                action.disk += union_disk[size_t(expert)];
                action.h2d += union_h2d[size_t(expert)];
                options[size_t(token)].push_back(action);
            };
            // The baseline is an invariant action even when tied partial sorts
            // place it outside the expected candidate rank.
            append(baseline[size_t(retained)]);
            if (!guarded[size_t(token)])
                for (int candidate = 0; candidate < df_cache_route_k_; ++candidate)
                    append(candidates[size_t(candidate)]);
            require(!options[size_t(token)].empty(),
                    "pruned cache route has no feasible action");
            auto local_less = [&](const PrunedCacheChoice &left,
                                  const PrunedCacheChoice &right) {
                if (left.disk != right.disk) return left.disk < right.disk;
                if (left.h2d != right.h2d) return left.h2d < right.h2d;
                if (left.regret != right.regret) return left.regret < right.regret;
                return rank[size_t(left.expert)] < rank[size_t(right.expert)];
            };
            std::sort(options[size_t(token)].begin(), options[size_t(token)].end(),
                      local_less);
            options[size_t(token)].erase(
                std::unique(options[size_t(token)].begin(), options[size_t(token)].end(),
                            [](const PrunedCacheChoice &left,
                               const PrunedCacheChoice &right) {
                                return left.expert == right.expert;
                            }),
                options[size_t(token)].end());
            if (df_cache_joint_options_ &&
                int(options[size_t(token)].size()) > df_cache_joint_options_) {
                const int baseline_tail = baseline[size_t(retained)];
                const auto baseline_action = std::find_if(
                    options[size_t(token)].begin(), options[size_t(token)].end(),
                    [&](const PrunedCacheChoice &action) {
                        return action.expert == baseline_tail;
                    });
                require(baseline_action != options[size_t(token)].end(),
                        "pruned cache route lost its baseline action");
                const PrunedCacheChoice baseline_copy = *baseline_action;
                options[size_t(token)].resize(size_t(df_cache_joint_options_));
                if (std::find_if(options[size_t(token)].begin(),
                                 options[size_t(token)].end(),
                                 [&](const PrunedCacheChoice &action) {
                                     return action.expert == baseline_tail;
                                 }) == options[size_t(token)].end())
                    options[size_t(token)].back() = baseline_copy;
            }
        }

        // DFlash verify has <=8 rows, but the same policy can serve a 128-row
        // prefill chunk. Keep one action per actual row; a fixed-eight array
        // here corrupted the stack as soon as prefill widened the caller.
        std::vector<int> chosen(size_t(tokens), 0);
        std::array<std::array<ExpertMask288, 8>, 4> option_masks{};
        ExpertMask288 disk_mask, h2d_mask;
        if (mask_search) {
            for (int expert = 0; expert < 288; ++expert) {
                if (union_disk[size_t(expert)] > 0) disk_mask.add(expert);
                if (union_h2d[size_t(expert)] > 0) h2d_mask.add(expert);
            }
            for (int token = 0; token < tokens; ++token) {
                ExpertMask288 prefix;
                const auto &baseline = baseline_experts[size_t(token)];
                const int prefix_count = guarded[size_t(token)]
                    ? guarded_exec_k[size_t(token)] : retained;
                for (int slot = 0; slot < prefix_count; ++slot)
                    prefix.add(baseline[size_t(slot)]);
                for (int option = 0; option < int(options[size_t(token)].size()); ++option) {
                    option_masks[size_t(token)][size_t(option)] = prefix;
                    if (!guarded[size_t(token)])
                        option_masks[size_t(token)][size_t(option)].add(
                            options[size_t(token)][size_t(option)].expert);
                }
            }
        }
        if (df_cache_joint_options_ && tokens <= 4) {
            std::array<int, 8> trial{};
            int best_disk = std::numeric_limits<int>::max();
            int best_h2d = best_disk, best_union = best_disk;
            double best_regret = std::numeric_limits<double>::infinity();
            auto search = [&](auto &&self, int row) -> void {
                if (row != tokens) {
                    for (int option = 0;
                         option < int(options[size_t(row)].size()); ++option) {
                        trial[size_t(row)] = option;
                        self(self, row + 1);
                    }
                    return;
                }
                int disk = 0, h2d = 0, union_count = 0;
                double regret = 0.0;
                if (mask_search) {
                    ExpertMask288 union_mask;
                    for (int token = 0; token < tokens; ++token) {
                        union_mask.merge(option_masks[size_t(token)]
                                                     [size_t(trial[size_t(token)])]);
                        if (!guarded[size_t(token)])
                            regret += options[size_t(token)]
                                             [size_t(trial[size_t(token)])].regret;
                    }
                    union_count = expert_mask_count(union_mask);
                    disk = expert_mask_intersection_count(union_mask, disk_mask);
                    h2d = expert_mask_intersection_count(union_mask, h2d_mask);
                    if (df_cache_mask_verify_) {
                        std::array<uint8_t, 288> seen{};
                        int ref_disk = 0, ref_h2d = 0, ref_union = 0;
                        double ref_regret = 0.0;
                        auto add = [&](int expert) {
                            if (seen[size_t(expert)]) return;
                            seen[size_t(expert)] = 1;
                            ++ref_union;
                            require(union_disk[size_t(expert)] >= 0,
                                    "pruned shadow union contains an unranked expert");
                            ref_disk += union_disk[size_t(expert)];
                            ref_h2d += union_h2d[size_t(expert)];
                        };
                        for (int token = 0; token < tokens; ++token) {
                            const auto &baseline = baseline_experts[size_t(token)];
                            if (guarded[size_t(token)]) {
                                for (int slot = 0;
                                     slot < guarded_exec_k[size_t(token)]; ++slot)
                                    add(baseline[size_t(slot)]);
                            } else {
                                for (int slot = 0; slot < retained; ++slot)
                                    add(baseline[size_t(slot)]);
                                const auto &action = options[size_t(token)]
                                    [size_t(trial[size_t(token)])];
                                add(action.expert);
                                ref_regret += action.regret;
                            }
                        }
                        require(ref_disk == disk && ref_h2d == h2d &&
                                    ref_union == union_count && ref_regret == regret,
                                "POPCNT pruned cache objective disagrees with byte shadow");
                    }
                } else {
                    std::array<uint8_t, 288> seen{};
                    auto add = [&](int expert) {
                        if (seen[size_t(expert)]) return;
                        seen[size_t(expert)] = 1;
                        ++union_count;
                        require(union_disk[size_t(expert)] >= 0,
                                "pruned joint union contains an unranked expert");
                        disk += union_disk[size_t(expert)];
                        h2d += union_h2d[size_t(expert)];
                    };
                    for (int token = 0; token < tokens; ++token) {
                        const auto &baseline = baseline_experts[size_t(token)];
                        if (guarded[size_t(token)]) {
                            for (int slot = 0; slot < guarded_exec_k[size_t(token)]; ++slot)
                                add(baseline[size_t(slot)]);
                        } else {
                            for (int slot = 0; slot < retained; ++slot)
                                add(baseline[size_t(slot)]);
                            const PrunedCacheChoice &action =
                                options[size_t(token)][size_t(trial[size_t(token)])];
                            add(action.expert);
                            regret += action.regret;
                        }
                    }
                }
                const bool better = disk < best_disk ||
                    (disk == best_disk && (h2d < best_h2d ||
                     (h2d == best_h2d && (union_count < best_union ||
                      (union_count == best_union && regret < best_regret)))));
                if (better) {
                    best_disk = disk;
                    best_h2d = h2d;
                    best_union = union_count;
                    best_regret = regret;
                    std::copy_n(trial.begin(), tokens, chosen.begin());
                }
            };
            search(search, 0);
        }

        auto group_cost = [&](bool baseline_cost) {
            std::array<uint8_t, 288> seen{};
            std::array<int, 3> cost{};  // union, disk, H2D
            auto add = [&](int expert) {
                if (seen[size_t(expert)]) return;
                seen[size_t(expert)] = 1;
                ++cost[0];
                cost[1] += union_disk[size_t(expert)];
                cost[2] += union_h2d[size_t(expert)];
            };
            for (int token = 0; token < tokens; ++token) {
                const auto &baseline = baseline_experts[size_t(token)];
                if (guarded[size_t(token)]) {
                    for (int slot = 0; slot < guarded_exec_k[size_t(token)]; ++slot)
                        add(baseline[size_t(slot)]);
                } else {
                    for (int slot = 0; slot < retained; ++slot)
                        add(baseline[size_t(slot)]);
                    add(baseline_cost ? baseline[size_t(retained)] :
                        options[size_t(token)][size_t(chosen[size_t(token)])].expert);
                }
            }
            return cost;
        };
        if (df_cache_joint_options_ && tokens <= 4) {
            const auto baseline_cost = group_cost(true);
            const auto selected_cost = group_cost(false);
            ++df_cache_joint_groups_;
            df_cache_joint_baseline_union_ += uint64_t(baseline_cost[0]);
            df_cache_joint_selected_union_ += uint64_t(selected_cost[0]);
            df_cache_joint_disk_saved_ += baseline_cost[1] - selected_cost[1];
            df_cache_joint_h2d_saved_ += baseline_cost[2] - selected_cost[2];
        }
        for (int token = 0; token < tokens; ++token) {
            if (guarded[size_t(token)]) continue;
            const auto &baseline = baseline_experts[size_t(token)];
            const int baseline_tail = baseline[size_t(retained)];
            const PrunedCacheChoice &action =
                options[size_t(token)][size_t(chosen[size_t(token)])];
            const auto baseline_action = std::find_if(
                options[size_t(token)].begin(), options[size_t(token)].end(),
                [&](const PrunedCacheChoice &candidate) {
                    return candidate.expert == baseline_tail;
                });
            require(baseline_action != options[size_t(token)].end(),
                    "pruned cache stats lost baseline action");
            if (action.expert != baseline_tail) {
                const int candidate =
                    candidate_rank[size_t(token)][size_t(action.expert)];
                const float score = 1.0f /
                    (1.0f + std::exp(-candidate_logits[size_t(token)][size_t(candidate)]));
                selection[size_t(token)][size_t(retained)] = {
                    action.expert,
                    2.5f * score / baseline_denominator[size_t(token)]};
            }
            ++df_cache_route_rows_;
            df_cache_route_changed_ += action.expert != baseline_tail;
            df_cache_route_substitutions_ += action.expert != baseline_tail;
            df_cache_route_regret_sum_ += action.ratio;
            df_cache_route_regret_max_ =
                std::max(df_cache_route_regret_max_, action.ratio);
            df_cache_route_disk_saved_ += baseline_action->disk - action.disk;
            df_cache_route_h2d_saved_ += baseline_action->h2d - action.h2d;
        }
        df_cache_selector_seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - selector_started).count();
        ++df_cache_selector_calls_;
        df_cache_selector_mask_calls_ += mask_search;
    }
    if (approximate_pass && !df_approx_topm_ && df_cache_joint_options_ &&
        df_cache_route_regret_ > 0.0f) {
        const auto selector_started = std::chrono::steady_clock::now();
        const bool mask_search = df_cache_mask_search_ && tokens >= 2 && tokens <= 4;
        struct CacheChoice {
            std::array<int, 8> experts{};
            double regret = 0.0, ratio = 0.0;
            int substitutions = 0, disk = 0, h2d = 0;
        };
        std::vector<std::vector<CacheChoice>> options(static_cast<size_t>(tokens));
        std::vector<std::array<int, 288>> candidate_rank(static_cast<size_t>(tokens));
        std::array<int, 288> union_disk{}, union_h2d{};
        union_disk.fill(-1);
        union_h2d.fill(-1);
        for (int token = 0; token < tokens; ++token) {
            auto &rank = candidate_rank[size_t(token)];
            rank.fill(-1);
            const auto &candidates = candidate_experts[size_t(token)];
            const auto &residency = candidate_residency[size_t(token)];
            for (int candidate = 0; candidate < df_cache_route_k_; ++candidate) {
                const int expert = candidates[size_t(candidate)];
                rank[size_t(expert)] = candidate;
                const uint32_t bit = uint32_t(1u) << candidate;
                const bool device = (residency[2] & bit) != 0;
                const bool host = (residency[0] & bit) != 0;
                const bool inflight = (residency[1] & bit) != 0;
                const int disk = device ? 0 : !(host || inflight);
                const int h2d = device ? 0 : 1;
                if (union_disk[size_t(expert)] < 0) {
                    union_disk[size_t(expert)] = disk;
                    union_h2d[size_t(expert)] = h2d;
                }
            }
            const auto &baseline = baseline_experts[size_t(token)];
            double baseline_score = 0.0, score_scale = 0.0;
            for (int expert : baseline) {
                const int candidate = rank[size_t(expert)];
                require(candidate >= 0, "joint cache baseline missing from candidate pool");
                const double value = candidate_choice[size_t(token)][size_t(candidate)];
                baseline_score += value;
                score_scale += std::fabs(value);
            }
            const int verify_row = capture_offset_ + token;
            const bool guarded = kda_archive_ && df_logit_guard_on() &&
                verify_row >= 0 &&
                verify_row < kMaxVerify && df_logit_guard_exact_[size_t(verify_row)];
            const int cache_route_retain = guarded
                ? df_cache_guard_retain_ : df_cache_route_retain_;
            auto append_action = [&](int first_tail, int second_tail) {
                CacheChoice action;
                if (first_tail < 0) {
                    action.experts = baseline;
                } else {
                    for (int slot = 0; slot < cache_route_retain; ++slot)
                        action.experts[size_t(slot)] = baseline[size_t(slot)];
                    action.experts[size_t(cache_route_retain)] =
                        candidates[size_t(first_tail)];
                    if (cache_route_retain == 6)
                        action.experts[7] = candidates[size_t(second_tail)];
                }
                double selected_score = 0.0;
                for (int expert : action.experts) {
                    const int candidate = rank[size_t(expert)];
                    selected_score += candidate_choice[size_t(token)][size_t(candidate)];
                    action.disk += union_disk[size_t(expert)];
                    action.h2d += union_h2d[size_t(expert)];
                }
                action.regret = std::max(0.0, baseline_score - selected_score);
                action.ratio = score_scale > 0.0 ? action.regret / score_scale : 0.0;
                if (action.ratio > double(df_cache_route_regret_) + 2.0e-7) return;
                for (int slot = cache_route_retain; slot < 8; ++slot)
                    action.substitutions +=
                        std::find(baseline.begin(), baseline.end(),
                                  action.experts[size_t(slot)]) == baseline.end();
                options[size_t(token)].push_back(action);
            };
            // Tied router scores can make the independent top-8 and top-32
            // partial sorts disagree in order. Insert the true baseline
            // explicitly instead of assuming candidate ranks 6/7 recreate it.
            append_action(-1, -1);
            if (cache_route_retain == 7) {
                for (int tail = 7; tail < df_cache_route_k_; ++tail)
                    append_action(tail, -1);
            } else if (cache_route_retain == 6) {
                for (int first = 6; first < df_cache_route_k_; ++first)
                    for (int second = first + 1; second < df_cache_route_k_; ++second)
                        append_action(first, second);
            }
            require(!options[size_t(token)].empty(), "joint cache route has no feasible action");
            auto local_less = [](const CacheChoice &left, const CacheChoice &right) {
                if (left.disk != right.disk) return left.disk < right.disk;
                if (left.h2d != right.h2d) return left.h2d < right.h2d;
                if (left.regret != right.regret) return left.regret < right.regret;
                if (left.substitutions != right.substitutions)
                    return left.substitutions < right.substitutions;
                return left.experts < right.experts;
            };
            std::sort(options[size_t(token)].begin(), options[size_t(token)].end(), local_less);
            options[size_t(token)].erase(
                std::unique(options[size_t(token)].begin(), options[size_t(token)].end(),
                            [](const CacheChoice &left, const CacheChoice &right) {
                                return left.experts == right.experts;
                            }),
                options[size_t(token)].end());
            if (int(options[size_t(token)].size()) > df_cache_joint_options_) {
                const auto baseline_action = std::find_if(
                    options[size_t(token)].begin(), options[size_t(token)].end(),
                    [&](const CacheChoice &action) { return action.experts == baseline; });
                require(baseline_action != options[size_t(token)].end(),
                        "joint cache route lost its baseline action");
                const CacheChoice baseline_copy = *baseline_action;
                options[size_t(token)].resize(size_t(df_cache_joint_options_));
                if (std::find_if(options[size_t(token)].begin(), options[size_t(token)].end(),
                                 [&](const CacheChoice &action) {
                                     return action.experts == baseline;
                                 }) == options[size_t(token)].end())
                    options[size_t(token)].back() = baseline_copy;
            }
        }

        std::vector<int> chosen(size_t(tokens), 0);
        std::array<std::array<ExpertMask288, 8>, 4> option_masks{};
        ExpertMask288 disk_mask, h2d_mask;
        if (mask_search) {
            for (int expert = 0; expert < 288; ++expert) {
                if (union_disk[size_t(expert)] > 0) disk_mask.add(expert);
                if (union_h2d[size_t(expert)] > 0) h2d_mask.add(expert);
            }
            for (int token = 0; token < tokens; ++token)
                for (int option = 0; option < int(options[size_t(token)].size()); ++option)
                    for (int expert : options[size_t(token)][size_t(option)].experts)
                        option_masks[size_t(token)][size_t(option)].add(expert);
        }
        if (tokens <= 4) {
            std::array<int, 8> trial{};
            int best_disk = std::numeric_limits<int>::max();
            int best_h2d = best_disk, best_union = best_disk, best_substitutions = best_disk;
            double best_regret = std::numeric_limits<double>::infinity();
            auto search = [&](auto &&self, int row) -> void {
                if (row != tokens) {
                    for (int option = 0; option < int(options[size_t(row)].size()); ++option) {
                        trial[size_t(row)] = option;
                        self(self, row + 1);
                    }
                    return;
                }
                int disk = 0, h2d = 0, union_count = 0, substitutions = 0;
                double regret = 0.0;
                if (mask_search) {
                    ExpertMask288 union_mask;
                    for (int token = 0; token < tokens; ++token) {
                        const int option = trial[size_t(token)];
                        const CacheChoice &action =
                            options[size_t(token)][size_t(option)];
                        union_mask.merge(option_masks[size_t(token)][size_t(option)]);
                        regret += action.regret;
                        substitutions += action.substitutions;
                    }
                    union_count = expert_mask_count(union_mask);
                    disk = expert_mask_intersection_count(union_mask, disk_mask);
                    h2d = expert_mask_intersection_count(union_mask, h2d_mask);
                    if (df_cache_mask_verify_) {
                        std::array<uint8_t, 288> seen{};
                        int ref_disk = 0, ref_h2d = 0, ref_union = 0;
                        int ref_substitutions = 0;
                        double ref_regret = 0.0;
                        for (int token = 0; token < tokens; ++token) {
                            const CacheChoice &action = options[size_t(token)]
                                [size_t(trial[size_t(token)])];
                            ref_regret += action.regret;
                            ref_substitutions += action.substitutions;
                            for (int expert : action.experts)
                                if (!seen[size_t(expert)]) {
                                    seen[size_t(expert)] = 1;
                                    ++ref_union;
                                    require(union_disk[size_t(expert)] >= 0,
                                            "joint shadow union contains an unranked expert");
                                    ref_disk += union_disk[size_t(expert)];
                                    ref_h2d += union_h2d[size_t(expert)];
                                }
                        }
                        require(ref_disk == disk && ref_h2d == h2d &&
                                    ref_union == union_count && ref_regret == regret &&
                                    ref_substitutions == substitutions,
                                "POPCNT cache objective disagrees with byte shadow");
                    }
                } else {
                    std::array<uint8_t, 288> seen{};
                    for (int token = 0; token < tokens; ++token) {
                        const CacheChoice &action =
                            options[size_t(token)][size_t(trial[size_t(token)])];
                        regret += action.regret;
                        substitutions += action.substitutions;
                        for (int expert : action.experts)
                            if (!seen[size_t(expert)]) {
                                seen[size_t(expert)] = 1;
                                ++union_count;
                                require(union_disk[size_t(expert)] >= 0,
                                        "joint cache union contains an unranked expert");
                                disk += union_disk[size_t(expert)];
                                h2d += union_h2d[size_t(expert)];
                            }
                        }
                }
                const bool better = disk < best_disk ||
                    (disk == best_disk && (h2d < best_h2d ||
                     (h2d == best_h2d && (union_count < best_union ||
                      (union_count == best_union && (regret < best_regret ||
                       (regret == best_regret && substitutions < best_substitutions)))))));
                if (better) {
                    best_disk = disk;
                    best_h2d = h2d;
                    best_union = union_count;
                    best_regret = regret;
                    best_substitutions = substitutions;
                    std::copy_n(trial.begin(), tokens, chosen.begin());
                }
            };
            search(search, 0);
        }

        auto group_cost = [&](bool baseline) {
            std::array<uint8_t, 288> seen{};
            std::array<int, 3> cost{};  // union, disk, H2D
            for (int token = 0; token < tokens; ++token) {
                const auto &experts = baseline
                    ? baseline_experts[size_t(token)]
                    : options[size_t(token)][size_t(chosen[size_t(token)])].experts;
                for (int expert : experts)
                    if (!seen[size_t(expert)]) {
                        seen[size_t(expert)] = 1;
                        ++cost[0];
                        cost[1] += union_disk[size_t(expert)];
                        cost[2] += union_h2d[size_t(expert)];
                    }
            }
            return cost;
        };
        if (tokens <= 4) {
            const auto baseline_cost = group_cost(true);
            const auto selected_cost = group_cost(false);
            ++df_cache_joint_groups_;
            df_cache_joint_baseline_union_ += uint64_t(baseline_cost[0]);
            df_cache_joint_selected_union_ += uint64_t(selected_cost[0]);
            df_cache_joint_disk_saved_ += baseline_cost[1] - selected_cost[1];
            df_cache_joint_h2d_saved_ += baseline_cost[2] - selected_cost[2];
        }
        for (int token = 0; token < tokens; ++token) {
            const int verify_row = capture_offset_ + token;
            const bool guarded = kda_archive_ && df_logit_guard_on() &&
                verify_row >= 0 &&
                verify_row < kMaxVerify && df_logit_guard_exact_[size_t(verify_row)];
            const CacheChoice &action =
                options[size_t(token)][size_t(chosen[size_t(token)])];
            if (action.substitutions) {
                float denominator = 0.0f;
                for (int expert : action.experts) {
                    const int rank = candidate_rank[size_t(token)][size_t(expert)];
                    denominator += 1.0f /
                        (1.0f + std::exp(-candidate_logits[size_t(token)][size_t(rank)]));
                }
                selection[size_t(token)].clear();
                for (int expert : action.experts) {
                    const int rank = candidate_rank[size_t(token)][size_t(expert)];
                    const float score = 1.0f /
                        (1.0f + std::exp(-candidate_logits[size_t(token)][size_t(rank)]));
                    selection[size_t(token)].emplace_back(expert, 2.5f * score / denominator);
                }
            }
            if (guarded) continue;
            const auto baseline_action = std::find_if(
                options[size_t(token)].begin(), options[size_t(token)].end(),
                [&](const CacheChoice &candidate) {
                    return candidate.experts == baseline_experts[size_t(token)];
                });
            require(baseline_action != options[size_t(token)].end(),
                    "joint cache stats lost baseline action");
            ++df_cache_route_rows_;
            df_cache_route_changed_ += action.substitutions != 0;
            df_cache_route_substitutions_ += uint64_t(action.substitutions);
            df_cache_route_regret_sum_ += action.ratio;
            df_cache_route_regret_max_ = std::max(df_cache_route_regret_max_, action.ratio);
            df_cache_route_disk_saved_ += baseline_action->disk - action.disk;
            df_cache_route_h2d_saved_ += baseline_action->h2d - action.h2d;
        }
        df_cache_selector_seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - selector_started).count();
        ++df_cache_selector_calls_;
        df_cache_selector_mask_calls_ += mask_search;
    }
    if (trace_falsifier && falsifier_feature_only_)
        report_falsifier_features(layer, selection, exec_count, candidate_experts,
                                  candidate_logits, candidate_choice, router_summary,
                                  candidate_residency, input, tokens);
    std::vector<int> distinct;
    for (int token = 0; token < tokens; ++token) {
        const auto &picks = selection[size_t(token)];
        for (int slot = 0; slot < exec_count[size_t(token)]; ++slot) {
            const int expert = picks[size_t(slot)].first;
            if (std::find(distinct.begin(), distinct.end(), expert) == distinct.end())
                distinct.push_back(expert);
        }
    }
    if (approximate_moe) {
        std::vector<uint8_t> exact_seen(size_t(experts), 0);
        uint64_t exact_union = 0;
        for (int token = 0; token < tokens; ++token)
            for (int slot = 0; slot < topk; ++slot) {
                const int expert = df_cache_route_k_
                    ? baseline_experts[size_t(token)][size_t(slot)]
                    : selection[size_t(token)][size_t(slot)].first;
                if (!exact_seen[size_t(expert)]) {
                    exact_seen[size_t(expert)] = 1;
                    ++exact_union;
                }
            }
        df_approx_rows_ += uint64_t(tokens);
        df_approx_union_ += distinct.size();
        df_approx_exact_union_ += exact_union;
        for (int count : exec_count) {
            df_approx_slots_ += uint64_t(count);
            ++df_approx_k_hist_[size_t(count)];
        }
        if (layer == 3 && df_logit_guard_on())
            for (int token = 0; token < tokens; ++token) {
                ++df_logit_guard_rows_;
                const int verify_row = capture_offset_ + token;
                if (verify_row >= 0 && verify_row < kMaxVerify &&
                    df_logit_guard_exact_[size_t(verify_row)])
                    ++df_logit_guarded_rows_;
            }
    }
    if (early_multi_route_on_) {
        const auto &predicted_rows = early_multi_rows_[size_t(layer)];
        require(predicted_rows.size() == size_t(tokens), "missing early multi-route rows");
        std::vector<int> predicted;
        for (int rank = 0; rank < topk; ++rank)
            for (int token = 0; token < tokens; ++token) {
                const int expert = predicted_rows[size_t(token)][size_t(rank)];
                if (std::find(predicted.begin(), predicted.end(), expert) == predicted.end())
                    predicted.push_back(expert);
            }
        int overlap = 0;
        for (int expert : predicted)
            overlap += std::find(distinct.begin(), distinct.end(), expert) != distinct.end();
        early_multi_hits_ += uint64_t(overlap);
        early_multi_predicted_ += predicted.size();
        early_multi_actual_ += distinct.size();
        if (early_multi_trace_) {
            std::fprintf(early_multi_trace_, "%llu %d %d %d %zu %zu",
                         (unsigned long long)early_multi_batch_, layer, tokens, overlap,
                         predicted.size(), distinct.size());
            for (int token = 0; token < tokens; ++token) {
                for (int expert : predicted_rows[size_t(token)])
                    std::fprintf(early_multi_trace_, " %d", expert);
                for (const auto &[expert, weight] : selection[size_t(token)])
                    std::fprintf(early_multi_trace_, " %d", expert);
            }
            std::fputc('\n', early_multi_trace_);
            std::fflush(early_multi_trace_);
        }
    }
    if (whole_moe_route_sink_) {
        require(nvfp4_experts_ && expert_stager_ && topk == 8 && !approximate_moe &&
                    !kda_archive_,
                "whole-layer route sink requires exact Top-8 NVFP4 prompt routing");
        for (int token = 0; token < tokens; ++token) {
            require(exec_count[size_t(token)] == topk,
                    "whole-layer route sink cannot retain a pruned row");
            const int row = whole_moe_route_sink_->row_base + token;
            for (int slot = 0; slot < topk; ++slot) {
                whole_moe_route_sink_->experts[size_t(row)][size_t(slot)] =
                    selection[size_t(token)][size_t(slot)].first;
                whole_moe_route_sink_->weights[size_t(row)][size_t(slot)] =
                    selection[size_t(token)][size_t(slot)].second;
            }
        }
        // Retain the ordinary prompt bookkeeping even though expert compute
        // is deferred.  Successive original chunks therefore leave exactly
        // the same final prev_routing_/row_routing_ state as moe_multi.
        for (int slot = 0; slot < topk; ++slot)
            prev_routing_[size_t(layer)][size_t(slot)] =
                selection[size_t(tokens - 1)][size_t(slot)].first;
        for (int row = 0; row < tokens && row < 8; ++row) {
            row_vein_[size_t(layer)][size_t(row)] = verify_epoch_;
            for (int slot = 0; slot < topk; ++slot)
                row_routing_[size_t(layer)][size_t(row * topk + slot)] =
                    selection[size_t(row)][size_t(slot)].first;
        }
        return;
    }
    check(cudaMemset(c_routed_, 0, size_t(tokens) * hidden_ * sizeof(float)), "clear routed (prefill)");
    if ((nvfp4_experts_ || q3_experts_) && expert_stager_) {
        // Routing bookkeeping + speculative next-layer reads, mirroring the
        // decode path. Demand staging for the whole deduplicated union comes
        // first so all four readers stream the layer while its first GEMV
        // batches run; speculative queues can never jump a demand record.
        for (int slot = 0; slot < topk; ++slot)
            prev_routing_[size_t(layer)][size_t(slot)] = selection[size_t(tokens - 1)][size_t(slot)].first;
        for (int row = 0; row < tokens && row < 8; ++row) {
            // Freshness vein for acceptance-prefix demotion: this round's
            // verify chunks stamp the rows they actually routed.
            row_vein_[size_t(layer)][size_t(row)] = verify_epoch_;
            for (int slot = 0; slot < topk; ++slot)
                row_routing_[size_t(layer)][size_t(row * topk + slot)] =
                    selection[size_t(row)][size_t(slot)].first;
        }
        expert_stager_->stage_layer(layer, distinct.data(), int(distinct.size()));
        if (!full_layer_major_active_ && prefetch_on_ &&
            size_t(layer) + 1 < prev_routing_.size() &&
            is_sparse_[size_t(layer) + 1])
            expert_stager_->prefetch(int(layer) + 1, prev_routing_[size_t(layer) + 1].data(),
                                     moe_topk_);
        if (!full_layer_major_active_ && prefetch_on_ && !cct_.empty())
            cct_prefetch(layer);
        // The shared expert is dense-resident FP8; running it first lets its
        // GEMVs hide under the routed records' disk reads (same ordering as
        // the decode path). It writes `output`; routed experts accumulate
        // into c_routed_ and the combine order below is unchanged.
        mlp_multi(stem + "shared_experts.", input, output, tokens, shared_intermediate_);
    }

    if (!nvfp4_experts_ && !q3_experts_) {
        for (int expert : distinct) {
            std::vector<int> users;
            for (int token = 0; token < tokens; ++token)
                for (int pick_slot = 0; pick_slot < exec_count[size_t(token)]; ++pick_slot)
                    if (selection[size_t(token)][size_t(pick_slot)].first == expert)
                        users.push_back(token);
            const std::string estem = stem + "experts." + std::to_string(expert) + ".";
            const TensorLocation &gate_w = model_.tensor(estem + "gate_proj.weight");
            const uint32_t *device = reinterpret_cast<const uint32_t *>(stager_.load(gate_w));
            for (int token : users)
                insignia::bf16_gemv_v2(device, input + size_t(token) * hidden_,
                                       c_gateu_.get() + size_t(token) * moe_intermediate_,
                                       moe_intermediate_, hidden_);
            const TensorLocation &up_w = model_.tensor(estem + "up_proj.weight");
            device = reinterpret_cast<const uint32_t *>(stager_.load(up_w));
            for (int token : users)
                insignia::bf16_gemv_v2(device, input + size_t(token) * hidden_,
                                       c_up_.get() + size_t(token) * moe_intermediate_,
                                       moe_intermediate_, hidden_);
            for (int token : users)
                launch_clamped_swiglu(c_gateu_.get() + size_t(token) * moe_intermediate_,
                                      c_up_.get() + size_t(token) * moe_intermediate_,
                                      c_act_.get() + size_t(token) * moe_intermediate_, moe_intermediate_);
            const TensorLocation &down_w = model_.tensor(estem + "down_proj.weight");
            device = reinterpret_cast<const uint32_t *>(stager_.load(down_w));
            for (int token : users) {
                insignia::bf16_gemv_v2(device, c_act_.get() + size_t(token) * moe_intermediate_,
                                       c_proj_.get() + size_t(token) * hidden_, hidden_, moe_intermediate_);
                float weight = 0.0f;
                for (int pick_slot = 0; pick_slot < exec_count[size_t(token)]; ++pick_slot)
                    if (selection[size_t(token)][size_t(pick_slot)].first == expert)
                        weight = selection[size_t(token)][size_t(pick_slot)].second;
                scale_add_kernel<<<16, 256>>>(c_routed_.get() + size_t(token) * hidden_,
                                              c_proj_.get() + size_t(token) * hidden_, weight, hidden_);
            }
        }
    } else {
        // A union batches experts in first-seen order, which differs from a
        // later token's router top-k order. Verification must retain each
        // down result and reproduce the scalar fmaf order exactly; otherwise
        // close target-logit ties can flip and break greedy equivalence.
        // Likewise, a >64-row prompt union changes the effective accumulation
        // order of its second 64 rows. Retain those results and replay the
        // legacy per-64-row union order while keeping expert compute and I/O
        // deduplicated across the full chunk.
        const bool ordered_accumulation = kda_archive_;
        const bool legacy_chunk_accumulation = !kda_archive_ && tokens > 64;
        const bool retain_down_results = ordered_accumulation || legacy_chunk_accumulation;
        for (size_t base_slot = 0; base_slot < distinct.size(); base_slot += 8) {
            std::array<int, 8> batch{};
            for (size_t slot = 0; slot < 8; ++slot)
                batch[slot] = distinct[std::min(base_slot + slot, distinct.size() - 1)];
            const int batch_count = int(std::min<size_t>(8, distinct.size() - base_slot));
            bool populate = false;
            uint8_t populate_mask = 0;
            if (full_layer_major_active_) {
                // All chunks of this layer execute consecutively and the
                // complete layer has at most 288 records, far below the
                // 2,425-slot production host tier. Admit the whole union so
                // later chunks become RAM hits; arithmetic order is unchanged.
                populate = true;
                populate_mask = uint8_t((1u << batch_count) - 1u);
            } else if (verify_populate_) {
                // Explicit legacy A/B: admit the entire verify union.
                populate = true;
                populate_mask = uint8_t((1u << batch_count) - 1u);
            } else if (kda_archive_) {
                // Partition the tier evenly by sparse layer while reserving
                // transient windows for the read/copy pipeline. The default
                // 379 slots retain 8 experts/layer; a lockable 6.6-GiB tier
                // retains 11. Fill from consecutive verify positions so the
                // extra RAM captures temporal alternatives without admitting
                // the whole ~2,000-record scan. Admission is per slot, leaving
                // processing order (and routed-sum rounding) untouched.
                static const int cache_token = [] {
                    const char *value = std::getenv("INSIGNIA_GLM53_VERIFY_CACHE_TOKEN");
                    return value ? std::max(0, std::atoi(value)) : 0;
                }();
                const int chosen_token = std::min(cache_token, tokens - 1);
                const int sparse_layers = int(std::count(is_sparse_.begin(), is_sparse_.end(), true));
                const int quota = std::max(1, (expert_stager_->cache_slots() - 16) /
                                                   std::max(1, sparse_layers));
                std::vector<int> retained;
                retained.reserve(size_t(quota));
                for (int offset = 0; offset < tokens && int(retained.size()) < quota; ++offset) {
                    const int token = (chosen_token + offset) % tokens;
                    for (int pick_slot = 0; pick_slot < exec_count[size_t(token)]; ++pick_slot) {
                        const int expert =
                            selection[size_t(token)][size_t(pick_slot)].first;
                        if (std::find(retained.begin(), retained.end(), expert) == retained.end())
                            retained.push_back(expert);
                        if (int(retained.size()) == quota) break;
                    }
                }
                for (int slot = 0; slot < batch_count; ++slot)
                    for (int expert : retained)
                        if (batch[size_t(slot)] == expert)
                            populate_mask |= uint8_t(1u << slot);
                populate = populate_mask != 0;
            } else if (!kda_archive_) {
                // Prompt prefill: retain a bounded per-layer slice of the
                // chunk union so the next chunk's overlapping picks hit the
                // pinned tier instead of re-reading disk. Zero disables.
                static const int retain_quota = [] {
                    const char *value = std::getenv("INSIGNIA_GLM53_PREFILL_RETAIN");
                    return value ? std::max(0, std::atoi(value)) : 24;
                }();
                for (int slot = 0; slot < batch_count; ++slot)
                    if (int(base_slot) + slot < retain_quota)
                        populate_mask |= uint8_t(1u << slot);
                populate = populate_mask != 0;
            }
            expert_stager_->load_batch(layer, batch, batch_count, populate, populate_mask);
            for (int slot = 0; slot < batch_count; ++slot) {
                const int expert = distinct[base_slot + slot];
                expert_stager_->upload(slot);
                // Collect this expert's users in ascending token order — the
                // same visit order the per-token loop had — then serve each
                // group of up to 8 rows in one weight pass. The batched
                // kernels preserve every per-row accumulation chain, so the
                // routed-sum rounding is untouched.
                std::array<int, kMaxChunkCap> users{}, out_ids{};
                std::array<float, kMaxChunkCap> combine{};
                int total = 0;
                for (int token = 0; token < tokens; ++token)
                    for (int pick_slot = 0; pick_slot < exec_count[size_t(token)]; ++pick_slot)
                        if (selection[size_t(token)][size_t(pick_slot)].first == expert &&
                            selection[size_t(token)][size_t(pick_slot)].second != 0.0f) {
                            users[size_t(total)] = token;
                            out_ids[size_t(total)] = token * topk + pick_slot;
                            combine[size_t(total)] = selection[size_t(token)][size_t(pick_slot)].second;
                            ++total;
                }
                for (int base = 0; base < total; base += kMaxVerify) {
                    const int count = std::min(kMaxVerify, total - base);
                    if (q3_experts_) {
                        check(insignia::glm53::iq_quantize_activation_rows(
                                  input, hidden_, &users[size_t(base)], count,
                                  iq_workspace_4096_.get()),
                              "quantize Q3 expert input (batched prefill)");
                        const TensorType gate_type = expert_stager_->projection_type(1);
                        const TensorType up_type = expert_stager_->projection_type(2);
                        require(gate_type == up_type,
                                "Q3 expert gate/up formats must match");
                        if (gate_type == TensorType::iq3_xxs) {
                            check(insignia::glm53::iq3_xxs_gemv2_rows(
                                      expert_stager_->gate_weight(),
                                      expert_stager_->up_weight(),
                                      iq_workspace_4096_.get(), count,
                                      c_gateu_.get(), c_up_.get(),
                                      &users[size_t(base)], moe_intermediate_, hidden_),
                                  "Q3 IQ3 gate/up (batched prefill)");
                        } else if (gate_type == TensorType::iq4_xs) {
                            check(insignia::glm53::iq4_xs_gemv_rows(
                                      expert_stager_->gate_weight(),
                                      iq_workspace_4096_.get(), count,
                                      c_gateu_.get(), &users[size_t(base)],
                                      moe_intermediate_, hidden_),
                                  "Q3 IQ4 gate (batched prefill)");
                            check(insignia::glm53::iq4_xs_gemv_rows(
                                      expert_stager_->up_weight(),
                                      iq_workspace_4096_.get(), count,
                                      c_up_.get(), &users[size_t(base)],
                                      moe_intermediate_, hidden_),
                                  "Q3 IQ4 up (batched prefill)");
                        } else {
                            require(false, "unsupported Q3 gate/up format");
                        }
                        check(insignia::glm53::iq_quantize_swiglu_rows(
                                  c_gateu_.get(), c_up_.get(), moe_intermediate_,
                                  &users[size_t(base)], count,
                                  iq_workspace_2048_.get()),
                              "quantize Q3 routed SwiGLU (batched prefill)");
                        const TensorType down_type = expert_stager_->projection_type(0);
                        if (retain_down_results) {
                            if (down_type == TensorType::iq4_xs)
                                check(insignia::glm53::iq4_xs_gemv_rows(
                                          expert_stager_->down_weight(),
                                          iq_workspace_2048_.get(), count,
                                          c_expert_out_.get(), &out_ids[size_t(base)],
                                          hidden_, moe_intermediate_),
                                      "Q3 IQ4 down (ordered batched)");
                            else if (down_type == TensorType::q6_k)
                                check(insignia::glm53::q6_k_gemv_rows(
                                          expert_stager_->down_weight(),
                                          iq_workspace_2048_.get(), count,
                                          c_expert_out_.get(), &out_ids[size_t(base)],
                                          hidden_, moe_intermediate_),
                                      "Q3 Q6 down (ordered batched)");
                            else
                                require(false, "unsupported Q3 down format");
                        } else {
                            if (down_type == TensorType::iq4_xs)
                                check(insignia::glm53::iq4_xs_gemv_acc_rows(
                                          expert_stager_->down_weight(),
                                          iq_workspace_2048_.get(), count,
                                          c_routed_.get(), &users[size_t(base)],
                                          &combine[size_t(base)], hidden_, moe_intermediate_),
                                      "Q3 IQ4 down (batched prefill)");
                            else if (down_type == TensorType::q6_k)
                                check(insignia::glm53::q6_k_gemv_acc_rows(
                                          expert_stager_->down_weight(),
                                          iq_workspace_2048_.get(), count,
                                          c_routed_.get(), &users[size_t(base)],
                                          &combine[size_t(base)], hidden_, moe_intermediate_),
                                      "Q3 Q6 down (batched prefill)");
                            else
                                require(false, "unsupported Q3 down format");
                        }
                    } else {
                        check(insignia::glm53::nvfp4_quantize_activation_rows(
                                  input, hidden_, &users[size_t(base)], count,
                                  nv_workspace_4096_),
                              "quantize expert input (batched prefill)");
                        check(expert_stager_->packed_direct_active()
                              ? insignia::glm53::nvfp4_gemv2_dp4a_quantized_rows_packed(
                                    expert_stager_->gate_weight(),
                                    expert_stager_->gate_packed_scale(),
                                    expert_stager_->gate_global(int(slot)),
                                    expert_stager_->up_weight(),
                                    expert_stager_->up_packed_scale(),
                                    expert_stager_->up_global(int(slot)),
                                    nv_workspace_4096_, count, c_gateu_.get(), c_up_.get(),
                                    &users[size_t(base)], moe_intermediate_, hidden_,
                                    kNvfp4PackedPairCtaWarps[size_t(count)],
                                    expert_stager_->packed_tablefree())
                              : insignia::glm53::nvfp4_gemv2_dp4a_quantized_rows(
                                    expert_stager_->gate_weight(), expert_stager_->gate_scale(),
                                    expert_stager_->gate_global(int(slot)),
                                    expert_stager_->up_weight(), expert_stager_->up_scale(),
                                    expert_stager_->up_global(int(slot)),
                                    nv_workspace_4096_, count, c_gateu_.get(), c_up_.get(),
                                    &users[size_t(base)], moe_intermediate_, hidden_),
                              "routed expert gate/up (batched prefill)");
                        check(insignia::glm53::quantize_swiglu_activation_rows(
                                  c_gateu_.get(), c_up_.get(), moe_intermediate_,
                                  &users[size_t(base)], count, nv_workspace_2048_),
                              "quantize routed SwiGLU (batched prefill)");
                        if (retain_down_results) {
                            check(expert_stager_->packed_direct_active()
                                  ? insignia::glm53::nvfp4_gemv_dp4a_quantized_rows_packed(
                                        expert_stager_->down_weight(),
                                        expert_stager_->down_packed_scale(),
                                        expert_stager_->down_global(int(slot)),
                                        nv_workspace_2048_, count, c_expert_out_.get(),
                                        &out_ids[size_t(base)], hidden_, moe_intermediate_,
                                        kNvfp4PackedDownStoreCtaWarps[size_t(count)],
                                        expert_stager_->packed_tablefree())
                                  : nvfp4_fixed_rows_
                                  ? insignia::glm53::nvfp4_gemv_dp4a_quantized_rows_fixed(
                                        expert_stager_->down_weight(),
                                        expert_stager_->down_scale(),
                                        expert_stager_->down_global(int(slot)),
                                        nv_workspace_2048_, count, c_expert_out_.get(),
                                        &out_ids[size_t(base)], hidden_, moe_intermediate_,
                                        kNvfp4DownStoreCtaWarps[size_t(count)])
                                  : insignia::glm53::nvfp4_gemv_dp4a_quantized_rows(
                                        expert_stager_->down_weight(),
                                        expert_stager_->down_scale(),
                                        expert_stager_->down_global(int(slot)),
                                        nv_workspace_2048_, count, c_expert_out_.get(),
                                        &out_ids[size_t(base)], hidden_, moe_intermediate_),
                                  "routed expert down (ordered batched)");
                        } else {
                            check(expert_stager_->packed_direct_active()
                                  ? insignia::glm53::nvfp4_gemv_dp4a_acc_quantized_rows_packed(
                                        expert_stager_->down_weight(),
                                        expert_stager_->down_packed_scale(),
                                        expert_stager_->down_global(int(slot)), nv_workspace_2048_,
                                        count, c_routed_.get(), &users[size_t(base)],
                                        &combine[size_t(base)], hidden_, moe_intermediate_,
                                        kNvfp4PackedDownAccCtaWarps[size_t(count)],
                                        expert_stager_->packed_tablefree())
                                  : insignia::glm53::nvfp4_gemv_dp4a_acc_quantized_rows(
                                        expert_stager_->down_weight(), expert_stager_->down_scale(),
                                        expert_stager_->down_global(int(slot)), nv_workspace_2048_,
                                        count, c_routed_.get(), &users[size_t(base)],
                                        &combine[size_t(base)], hidden_, moe_intermediate_),
                                  "routed expert down (batched prefill)");
                        }
                    }
                }
            }
        }
        if (ordered_accumulation)
            for (int token = 0; token < tokens; ++token)
                for (int pick_slot = 0; pick_slot < exec_count[size_t(token)]; ++pick_slot)
                    scale_add_kernel<<<16, 256>>>(
                        c_routed_.get() + size_t(token) * hidden_,
                        c_expert_out_.get() +
                            (size_t(token) * topk + size_t(pick_slot)) * hidden_,
                        selection[token][size_t(pick_slot)].second, hidden_);
        else if (legacy_chunk_accumulation)
            for (int chunk_base = 0; chunk_base < tokens; chunk_base += 64) {
                const int chunk_end = std::min(chunk_base + 64, tokens);
                std::vector<int> chunk_distinct;
                chunk_distinct.reserve(distinct.size());
                for (int token = chunk_base; token < chunk_end; ++token)
                    for (int pick_slot = 0; pick_slot < exec_count[size_t(token)]; ++pick_slot) {
                        const int expert =
                            selection[size_t(token)][size_t(pick_slot)].first;
                        if (std::find(chunk_distinct.begin(), chunk_distinct.end(), expert) ==
                            chunk_distinct.end())
                            chunk_distinct.push_back(expert);
                    }
                for (int expert : chunk_distinct)
                    for (int token = chunk_base; token < chunk_end; ++token)
                        for (int pick_slot = 0; pick_slot < exec_count[size_t(token)]; ++pick_slot)
                            if (selection[size_t(token)][size_t(pick_slot)].first == expert)
                                scale_add_kernel<<<16, 256>>>(
                                    c_routed_.get() + size_t(token) * hidden_,
                                    c_expert_out_.get() +
                                        (size_t(token) * topk + size_t(pick_slot)) * hidden_,
                                    selection[size_t(token)][size_t(pick_slot)].second, hidden_);
            }
        if (ordered_accumulation &&
            (moe_metrics_ || (falsifier_trace_ && !falsifier_feature_only_)))
            report_moe_metrics(layer, selection, candidate_experts, candidate_logits,
                               candidate_choice, router_summary, candidate_residency,
                               input, tokens);
    }
    if (!nvfp4_experts_ && !q3_experts_)
        mlp_multi(stem + "shared_experts.", input, output, tokens, shared_intermediate_);
    for (int token = 0; token < tokens; ++token)
        add_kernel<<<16, 256>>>(output + size_t(token) * hidden_,
                                c_routed_.get() + size_t(token) * hidden_, hidden_);
    check(cudaGetLastError(), "MoE combine launch (prefill)");
}

void Runner::prefill_layer_chunk_exact(int layer, float *in_place, float *scratch,
                                       int count, int position_base,
                                       float *dflash_capture) {
    require(count >= 1 && count <= kMaxChunk(), "prefill layer chunk out of range");
    const auto begin = std::chrono::steady_clock::now();
    const std::string base = layer_stem(layer);
    float *streams = in_place;
    float *next_streams = scratch;

    mhc_multi(base + "hc_attn", streams, c_collapsed_, count);
    rms(base + "input_layernorm.weight", c_collapsed_, c_normalized_, count, hidden_);
    const int retry_seam_bias = df_retry_replay_ ? 10 : 0;
    if (kda_archive_)
        seam(layer, 11 + retry_seam_bias, c_normalized_, count * hidden_);
    if (early_multi_route_on_ && is_sparse_[size_t(layer)])
        early_route_multi(layer, c_normalized_, count);
    if (is_mla_[layer])
        mla_multi(layer, c_normalized_, c_attn_, count, position_base);
    else
        kda_multi(layer, c_normalized_, c_attn_, count, position_base);
    if (kda_archive_)
        seam(layer, 12 + retry_seam_bias, c_attn_, count * hidden_);
    for (int token = 0; token < count; ++token)
        check(insignia::glm53::mhc_mix(streams + size_t(token) * kStreams * hidden_,
            c_attn_.get() + size_t(token) * hidden_, c_post_.get() + size_t(token) * 4,
            c_comb_.get() + size_t(token) * 16,
            next_streams + size_t(token) * kStreams * hidden_, hidden_),
            "attention mHC mix (prefill)");
    std::swap(streams, next_streams);
    if (kda_archive_)
        seam(layer, 13 + retry_seam_bias, streams,
             count * kStreams * hidden_);

    mhc_multi(base + "hc_ffn", streams, c_collapsed_, count);
    rms(base + "post_attention_layernorm.weight", c_collapsed_, c_normalized_, count, hidden_);
    if (kda_archive_)
        seam(layer, 14 + retry_seam_bias, c_normalized_, count * hidden_);
    if (is_sparse_[layer])
        moe_multi(layer, c_normalized_, c_ffn_, count);
    else
        mlp_multi(base + "mlp.", c_normalized_, c_ffn_, count, dense_intermediate_);
    if (kda_archive_)
        seam(layer, 15 + retry_seam_bias, c_ffn_, count * hidden_);
    for (int token = 0; token < count; ++token)
        check(insignia::glm53::mhc_mix(streams + size_t(token) * kStreams * hidden_,
            c_ffn_.get() + size_t(token) * hidden_, c_post_.get() + size_t(token) * 4,
            c_comb_.get() + size_t(token) * 16,
            next_streams + size_t(token) * kStreams * hidden_, hidden_),
            "FFN mHC mix (prefill)");
    std::swap(streams, next_streams);
    // Two mHC swaps deliberately return the result to the caller's in-place
    // buffer. The full-prompt scheduler depends on this exact ownership rule.
    require(streams == in_place, "prefill layer changed residual-buffer parity");
    if (kda_archive_)
        seam(layer, 16 + retry_seam_bias, streams,
             count * kStreams * hidden_);

    if (dflash_capture)
        for (int token = 0; token < count; ++token)
            average_streams_kernel<<<16, 256>>>(
                streams + size_t(token) * kStreams * hidden_,
                dflash_capture + size_t(token) * hidden_, hidden_);
    if (deep_checks_) {
        const int one = 1;
        int valid = 0;
        check(cudaMemcpy(finite_, &one, sizeof(one), cudaMemcpyHostToDevice),
              "initialize finite flag");
        finite_kernel<<<16, 256>>>(streams, count * kStreams * hidden_, finite_);
        check(cudaMemcpy(&valid, finite_.get(), sizeof(valid), cudaMemcpyDeviceToHost),
              "read finite flag");
        require(valid, "non-finite residual stream after prefill layer " +
                       std::to_string(layer));
    }
    if (trace_layers_) {
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        std::printf("prefill layer %02d %-3s %d tokens %.3f s\n", layer,
                    is_mla_[layer] ? "MLA" : "KDA", count, seconds);
        std::fflush(stdout);
    }
}

void Runner::prefill_prompt_full_layer_major(const std::vector<int> &tokens) {
    const int prompt_tokens = int(tokens.size());
    require(prompt_tokens >= 1 && prompt_tokens < kMaxContext(),
            "full-layer-major prompt out of range");
    require(!full_layer_major_active_, "nested full-layer-major prefill");

    const char *store_arg = std::getenv("INSIGNIA_GLM53_PREFILL_STORE");
    const std::string store = store_arg ? store_arg : "host";
    require(store == "host" || store == "vram",
            "INSIGNIA_GLM53_PREFILL_STORE must be host or vram");
    const bool vram_store = store == "vram";
    const size_t stream_stride = size_t(kStreams) * hidden_;
    const size_t prompt_floats = size_t(prompt_tokens) * stream_stride;
    const int df_tokens = df_ ? std::min(
        prompt_tokens, insignia::glm53::DFlash2Drafter::kMaxCtx) : 0;
    const size_t capture_floats = size_t(5) * df_tokens * hidden_;

    constexpr int kWholeMoeMaxPrompt = 8192;
    bool whole_moe = prefill_whole_layer_moe_;
    const char *whole_fallback = nullptr;
    if (whole_moe && prompt_tokens > kWholeMoeMaxPrompt)
        whole_fallback = "prompt exceeds the 8192-row sidecar cap";
    else if (whole_moe && (!nvfp4_experts_ || !expert_stager_ ||
                           moe_experts_ != 288 || moe_topk_ != 8))
        whole_fallback = "engine is not exact Top-8 NVFP4";
    else if (whole_moe && prefill_approx_moe_)
        whole_fallback = "approximate/cache-aware prompt routing is enabled";
    else if (whole_moe && (kda_archive_ || df_retry_replay_))
        whole_fallback = "verify/retry state is active";
    else if (whole_moe && (moe_metrics_ || falsifier_trace_))
        whole_fallback = "MoE metrics/falsifier instrumentation is active";
    if (whole_fallback) whole_moe = false;

    // The arena's first allocation permanently fixes the decode slot count
    // from then-free VRAM.  Prime before prompt/capture/sidecar allocations.
    if (whole_moe) expert_stager_->prime_device_arena();

    // Every persistent allocation happens before embedding or recurrent-state
    // mutation. A failed explicit mode therefore leaves the Runner untouched.
    DeviceBuffer<float> prompt_device(vram_store ? prompt_floats : 0);
    DeviceBuffer<float> capture_device(vram_store ? capture_floats : 0);
    std::vector<float> prompt_host(vram_store ? 0 : prompt_floats);
    std::vector<float> capture_host(vram_store ? 0 : capture_floats);
    DeviceBuffer<float> whole_normalized;
    DeviceBuffer<float> whole_expert_out;
    DeviceBuffer<float> whole_post;
    DeviceBuffer<float> whole_comb;
    if (whole_moe) {
        const size_t whole_floats = size_t(prompt_tokens) *
            (size_t(hidden_) + size_t(moe_topk_) * hidden_ + 4 + 16);
        size_t free_bytes = 0, total_bytes = 0;
        const cudaError_t info = cudaMemGetInfo(&free_bytes, &total_bytes);
        constexpr size_t kAllocationReserve = 64ull << 20;
        if (info != cudaSuccess || whole_floats >
                (free_bytes > kAllocationReserve
                    ? (free_bytes - kAllocationReserve) / sizeof(float) : 0)) {
            cudaGetLastError();
            whole_moe = false;
            whole_fallback = "insufficient post-arena VRAM for exact sidecars";
        } else {
            try {
                whole_normalized.reset(size_t(prompt_tokens) * hidden_);
                whole_expert_out.reset(
                    size_t(prompt_tokens) * moe_topk_ * hidden_);
                whole_post.reset(size_t(prompt_tokens) * 4);
                whole_comb.reset(size_t(prompt_tokens) * 16);
            } catch (const std::runtime_error &) {
                cudaGetLastError();
                whole_normalized.reset(0);
                whole_expert_out.reset(0);
                whole_post.reset(0);
                whole_comb.reset(0);
                whole_moe = false;
                whole_fallback = "exact sidecar allocation failed";
            }
        }
    }
    if (prefill_whole_layer_moe_ && !whole_moe)
        std::printf("whole-layer MoE fallback: %s\n",
                    whole_fallback ? whole_fallback : "unsupported mode");

    std::vector<std::array<int, 8>> whole_routes(
        whole_moe ? size_t(prompt_tokens) : 0);
    std::vector<std::array<float, 8>> whole_weights(
        whole_moe ? size_t(prompt_tokens) : 0);
    std::vector<int> whole_users(
        whole_moe ? size_t(prompt_tokens) * moe_topk_ : 0);
    std::vector<int> whole_out_ids(
        whole_moe ? size_t(prompt_tokens) * moe_topk_ : 0);

    const TensorLocation &embedding = model_.tensor("model.language_model.embed_tokens.weight");
    require(embedding.type == TensorType::bf16 && embedding.shape.size() == 2 &&
            embedding.shape[1] == uint32_t(hidden_), "wrong embedding geometry");
    for (int token : tokens)
        require(token >= 0 && token < int(model_.vocab_size()),
                "prefill token is outside the vocabulary");

    struct ActiveScope {
        bool &active;
        explicit ActiveScope(bool &flag) : active(flag) { active = true; }
        ~ActiveScope() { active = false; }
    } active_scope(full_layer_major_active_);

    const uint64_t records_before = expert_stager_ ? expert_stager_->records_read() : 0;
    const uint64_t io_before = expert_stager_ ? expert_stager_->io_bytes() : 0;
    const uint64_t h2d_records_before = expert_stager_ ?
        expert_stager_->packed_h2d_records() : 0;
    uint64_t prompt_h2d = 0, prompt_d2h = 0;
    uint64_t whole_layers = 0, whole_union_experts = 0;
    uint64_t whole_upload_calls = 0, whole_legacy_upload_calls = 0;
    uint64_t whole_user_rows = 0;
    double whole_phase_a = 0.0, whole_phase_b = 0.0, whole_phase_c = 0.0;
    const auto all_begin = std::chrono::steady_clock::now();
    const int layers = int(model_.layers());

    for (int layer = 0; layer < layers; ++layer) {
        const auto layer_begin = std::chrono::steady_clock::now();
        if (whole_moe && is_sparse_[size_t(layer)]) {
            ++whole_layers;
            const std::string base = layer_stem(layer);
            const std::string moe_stem = base + "mlp.";

            // Phase A: run attention and the exact CPU router at the original
            // <=128-row call boundaries.  FFN-normalized rows and its mHC
            // coefficients survive until the globally staged expert pass.
            auto phase_begin = std::chrono::steady_clock::now();
            for (int pos0 = 0; pos0 < prompt_tokens; pos0 += kMaxChunk()) {
                const int count = std::min(kMaxChunk(), prompt_tokens - pos0);
                if (early_multi_route_on_) ++early_multi_batch_;
                float *incoming = vram_store ?
                    prompt_device.get() + size_t(pos0) * stream_stride : c_stream_a_.get();
                if (layer == 0) {
                    for (int row = 0; row < count; ++row) {
                        uint16_t *device_row = reinterpret_cast<uint16_t *>(stager_.load(
                            embedding, uint64_t(tokens[size_t(pos0 + row)]) * hidden_ * 2,
                            hidden_ * 2));
                        embed_repeat_kernel<<<16, 256>>>(
                            device_row, incoming + size_t(row) * stream_stride, hidden_);
                    }
                    check(cudaGetLastError(),
                          "embedding repeat launch (whole-layer MoE)");
                } else if (!vram_store) {
                    const size_t bytes = size_t(count) * stream_stride * sizeof(float);
                    check(cudaMemcpy(incoming,
                                     prompt_host.data() + size_t(pos0) * stream_stride,
                                     bytes, cudaMemcpyHostToDevice),
                          "restore prompt streams for whole-layer attention");
                    prompt_h2d += bytes;
                }

                mhc_multi(base + "hc_attn", incoming, c_collapsed_, count);
                rms(base + "input_layernorm.weight", c_collapsed_, c_normalized_,
                    count, hidden_);
                if (early_multi_route_on_)
                    early_route_multi(layer, c_normalized_, count);
                if (is_mla_[size_t(layer)])
                    mla_multi(layer, c_normalized_, c_attn_, count, pos0);
                else
                    kda_multi(layer, c_normalized_, c_attn_, count, pos0);
                for (int row = 0; row < count; ++row)
                    check(insignia::glm53::mhc_mix(
                              incoming + size_t(row) * stream_stride,
                              c_attn_.get() + size_t(row) * hidden_,
                              c_post_.get() + size_t(row) * 4,
                              c_comb_.get() + size_t(row) * 16,
                              c_stream_b_.get() + size_t(row) * stream_stride,
                              hidden_),
                          "attention mHC mix (whole-layer MoE)");

                mhc_multi(base + "hc_ffn", c_stream_b_, c_collapsed_, count);
                rms(base + "post_attention_layernorm.weight", c_collapsed_,
                    c_normalized_, count, hidden_);
                check(cudaMemcpy(whole_normalized.get() + size_t(pos0) * hidden_,
                                 c_normalized_.get(),
                                 size_t(count) * hidden_ * sizeof(float),
                                 cudaMemcpyDeviceToDevice),
                      "retain whole-layer FFN normalized rows");
                check(cudaMemcpy(whole_post.get() + size_t(pos0) * 4,
                                 c_post_.get(), size_t(count) * 4 * sizeof(float),
                                 cudaMemcpyDeviceToDevice),
                      "retain whole-layer FFN post coefficients");
                check(cudaMemcpy(whole_comb.get() + size_t(pos0) * 16,
                                 c_comb_.get(), size_t(count) * 16 * sizeof(float),
                                 cudaMemcpyDeviceToDevice),
                      "retain whole-layer FFN combine coefficients");

                WholeMoeRouteSink route_sink{
                    whole_routes.data(), whole_weights.data(), pos0};
                require(!whole_moe_route_sink_, "nested whole-layer route sink");
                whole_moe_route_sink_ = &route_sink;
                try {
                    moe_multi(layer, c_normalized_, c_ffn_, count);
                } catch (...) {
                    whole_moe_route_sink_ = nullptr;
                    throw;
                }
                whole_moe_route_sink_ = nullptr;

                const size_t stream_bytes =
                    size_t(count) * stream_stride * sizeof(float);
                if (vram_store) {
                    check(cudaMemcpy(prompt_device.get() + size_t(pos0) * stream_stride,
                                     c_stream_b_.get(), stream_bytes,
                                     cudaMemcpyDeviceToDevice),
                          "retain whole-layer post-attention streams");
                } else {
                    check(cudaMemcpy(prompt_host.data() + size_t(pos0) * stream_stride,
                                     c_stream_b_.get(), stream_bytes,
                                     cudaMemcpyDeviceToHost),
                          "spill whole-layer post-attention streams");
                    prompt_d2h += stream_bytes;
                }
            }
            whole_phase_a += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - phase_begin).count();

            // Phase B: deduplicate the complete prompt's expert union.  Each
            // expert is loaded/uploaded once, and every down row lands at the
            // stable token*8+pick sidecar ID used by Phase C.
            phase_begin = std::chrono::steady_clock::now();
            std::array<int, 288> union_experts{};
            int union_count = 0;
            for (int row = 0; row < prompt_tokens; ++row)
                for (int pick = 0; pick < moe_topk_; ++pick) {
                    const int expert = whole_routes[size_t(row)][size_t(pick)];
                    if (std::find(union_experts.begin(),
                                  union_experts.begin() + union_count, expert) ==
                        union_experts.begin() + union_count)
                        union_experts[size_t(union_count++)] = expert;
                }
            for (int pos0 = 0; pos0 < prompt_tokens; pos0 += kMaxChunk()) {
                const int end = std::min(pos0 + kMaxChunk(), prompt_tokens);
                std::array<int, 288> chunk_experts{};
                int chunk_count = 0;
                for (int row = pos0; row < end; ++row)
                    for (int pick = 0; pick < moe_topk_; ++pick) {
                        const int expert = whole_routes[size_t(row)][size_t(pick)];
                        if (std::find(chunk_experts.begin(),
                                      chunk_experts.begin() + chunk_count, expert) ==
                            chunk_experts.begin() + chunk_count)
                            chunk_experts[size_t(chunk_count++)] = expert;
                    }
                whole_legacy_upload_calls += uint64_t(chunk_count);
            }
            require(union_count > 0 && union_count <= moe_experts_,
                    "invalid whole-layer expert union");
            whole_union_experts += uint64_t(union_count);
            // stage_layer() claims every union window until its batch is
            // consumed.  A small host tier cannot legally claim more windows
            // than it owns: take_window() would otherwise find no victim and
            // abort before Phase B starts.  Per-batch load_batch() already
            // provides the exact fallback and still uploads every union expert
            // once, so only issue whole-union read-ahead when it fits.
            if (expert_stager_->cache_slots() >= union_count)
                expert_stager_->stage_layer(layer, union_experts.data(), union_count);
            constexpr std::array<int, 8> local_ids{0, 1, 2, 3, 4, 5, 6, 7};
            for (int base_expert = 0; base_expert < union_count; base_expert += 8) {
                const int batch_count = std::min(8, union_count - base_expert);
                std::array<int, 8> batch{};
                for (int slot = 0; slot < 8; ++slot)
                    batch[size_t(slot)] = union_experts[size_t(
                        std::min(base_expert + slot, union_count - 1))];
                const uint8_t populate_mask =
                    uint8_t((1u << unsigned(batch_count)) - 1u);
                expert_stager_->load_batch(
                    layer, batch, batch_count, true, populate_mask);
                for (int slot = 0; slot < batch_count; ++slot) {
                    const int expert = union_experts[size_t(base_expert + slot)];
                    expert_stager_->upload(slot);
                    ++whole_upload_calls;
                    int user_count = 0;
                    for (int row = 0; row < prompt_tokens; ++row)
                        for (int pick = 0; pick < moe_topk_; ++pick)
                            if (whole_routes[size_t(row)][size_t(pick)] == expert) {
                                whole_users[size_t(user_count)] = row;
                                whole_out_ids[size_t(user_count)] = row * moe_topk_ + pick;
                                ++user_count;
                            }
                    whole_user_rows += uint64_t(user_count);
                    for (int base_user = 0; base_user < user_count;
                         base_user += kMaxVerify) {
                        const int count = std::min(kMaxVerify, user_count - base_user);
                        check(insignia::glm53::nvfp4_quantize_activation_rows(
                                  whole_normalized.get(), hidden_,
                                  whole_users.data() + base_user, count,
                                  nv_workspace_4096_),
                              "quantize whole-layer expert input");
                        check(expert_stager_->packed_direct_active()
                                  ? insignia::glm53::nvfp4_gemv2_dp4a_quantized_rows_packed(
                                        expert_stager_->gate_weight(),
                                        expert_stager_->gate_packed_scale(),
                                        expert_stager_->gate_global(slot),
                                        expert_stager_->up_weight(),
                                        expert_stager_->up_packed_scale(),
                                        expert_stager_->up_global(slot),
                                        nv_workspace_4096_, count, c_gateu_.get(),
                                        c_up_.get(), local_ids.data(),
                                        moe_intermediate_, hidden_,
                                        kNvfp4PackedPairCtaWarps[size_t(count)],
                                        expert_stager_->packed_tablefree())
                                  : insignia::glm53::nvfp4_gemv2_dp4a_quantized_rows(
                                        expert_stager_->gate_weight(),
                                        expert_stager_->gate_scale(),
                                        expert_stager_->gate_global(slot),
                                        expert_stager_->up_weight(),
                                        expert_stager_->up_scale(),
                                        expert_stager_->up_global(slot),
                                        nv_workspace_4096_, count, c_gateu_.get(),
                                        c_up_.get(), local_ids.data(),
                                        moe_intermediate_, hidden_),
                              "whole-layer routed expert gate/up");
                        check(insignia::glm53::quantize_swiglu_activation_rows(
                                  c_gateu_.get(), c_up_.get(), moe_intermediate_,
                                  local_ids.data(), count, nv_workspace_2048_),
                              "quantize whole-layer routed SwiGLU");
                        check(expert_stager_->packed_direct_active()
                                  ? insignia::glm53::nvfp4_gemv_dp4a_quantized_rows_packed(
                                        expert_stager_->down_weight(),
                                        expert_stager_->down_packed_scale(),
                                        expert_stager_->down_global(slot),
                                        nv_workspace_2048_, count,
                                        whole_expert_out.get(),
                                        whole_out_ids.data() + base_user,
                                        hidden_, moe_intermediate_,
                                        kNvfp4PackedDownStoreCtaWarps[size_t(count)],
                                        expert_stager_->packed_tablefree())
                                  : nvfp4_fixed_rows_
                                  ? insignia::glm53::nvfp4_gemv_dp4a_quantized_rows_fixed(
                                        expert_stager_->down_weight(),
                                        expert_stager_->down_scale(),
                                        expert_stager_->down_global(slot),
                                        nv_workspace_2048_, count,
                                        whole_expert_out.get(),
                                        whole_out_ids.data() + base_user,
                                        hidden_, moe_intermediate_,
                                        kNvfp4DownStoreCtaWarps[size_t(count)])
                                  : insignia::glm53::nvfp4_gemv_dp4a_quantized_rows(
                                        expert_stager_->down_weight(),
                                        expert_stager_->down_scale(),
                                        expert_stager_->down_global(slot),
                                        nv_workspace_2048_, count,
                                        whole_expert_out.get(),
                                        whole_out_ids.data() + base_user,
                                        hidden_, moe_intermediate_),
                              "whole-layer routed expert down");
                    }
                }
            }
            whole_phase_b += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - phase_begin).count();

            // Phase C: restore each post-attention chunk, replay the original
            // independent 64-row first-seen expert orders, then perform the
            // one deferred FFN mHC mix.  Its result is scratch, so persist it
            // explicitly as the next layer's prompt state.
            phase_begin = std::chrono::steady_clock::now();
            for (int pos0 = 0; pos0 < prompt_tokens; pos0 += kMaxChunk()) {
                const int count = std::min(kMaxChunk(), prompt_tokens - pos0);
                const size_t stream_bytes =
                    size_t(count) * stream_stride * sizeof(float);
                if (vram_store) {
                    check(cudaMemcpy(c_stream_b_.get(),
                                     prompt_device.get() + size_t(pos0) * stream_stride,
                                     stream_bytes, cudaMemcpyDeviceToDevice),
                          "restore whole-layer post-attention streams");
                } else {
                    check(cudaMemcpy(c_stream_b_.get(),
                                     prompt_host.data() + size_t(pos0) * stream_stride,
                                     stream_bytes, cudaMemcpyHostToDevice),
                          "restore spilled whole-layer post-attention streams");
                    prompt_h2d += stream_bytes;
                }

                const float *normalized =
                    whole_normalized.get() + size_t(pos0) * hidden_;
                mlp_multi(moe_stem + "shared_experts.", normalized,
                          c_ffn_, count, shared_intermediate_);
                check(cudaMemset(c_routed_, 0,
                                 size_t(count) * hidden_ * sizeof(float)),
                      "clear whole-layer routed output");
                for (int block0 = 0; block0 < count; block0 += 64) {
                    const int block_end = std::min(block0 + 64, count);
                    std::array<int, 288> block_experts{};
                    int block_count = 0;
                    for (int local_row = block0; local_row < block_end; ++local_row) {
                        const int row = pos0 + local_row;
                        for (int pick = 0; pick < moe_topk_; ++pick) {
                            const int expert = whole_routes[size_t(row)][size_t(pick)];
                            if (std::find(block_experts.begin(),
                                          block_experts.begin() + block_count, expert) ==
                                block_experts.begin() + block_count)
                                block_experts[size_t(block_count++)] = expert;
                        }
                    }
                    for (int expert_slot = 0; expert_slot < block_count; ++expert_slot) {
                        const int expert = block_experts[size_t(expert_slot)];
                        for (int local_row = block0; local_row < block_end; ++local_row) {
                            const int row = pos0 + local_row;
                            for (int pick = 0; pick < moe_topk_; ++pick)
                                if (whole_routes[size_t(row)][size_t(pick)] == expert)
                                    scale_add_kernel<<<16, 256>>>(
                                        c_routed_.get() + size_t(local_row) * hidden_,
                                        whole_expert_out.get() +
                                            (size_t(row) * moe_topk_ + size_t(pick)) * hidden_,
                                        whole_weights[size_t(row)][size_t(pick)], hidden_);
                        }
                    }
                }
                check(cudaGetLastError(), "whole-layer routed replay launch");
                for (int row = 0; row < count; ++row)
                    add_kernel<<<16, 256>>>(
                        c_ffn_.get() + size_t(row) * hidden_,
                        c_routed_.get() + size_t(row) * hidden_, hidden_);
                check(cudaGetLastError(), "whole-layer MoE combine launch");

                float *final_streams = vram_store ?
                    prompt_device.get() + size_t(pos0) * stream_stride : c_stream_a_.get();
                for (int row = 0; row < count; ++row)
                    check(insignia::glm53::mhc_mix(
                              c_stream_b_.get() + size_t(row) * stream_stride,
                              c_ffn_.get() + size_t(row) * hidden_,
                              whole_post.get() + size_t(pos0 + row) * 4,
                              whole_comb.get() + size_t(pos0 + row) * 16,
                              final_streams + size_t(row) * stream_stride,
                              hidden_),
                          "FFN mHC mix (whole-layer MoE)");

                int capture_idx = -1;
                if (df_ && pos0 < df_tokens)
                    for (int ci = 0; ci < 5; ++ci)
                        if (kDfCaptureLayers[ci] == layer) capture_idx = ci;
                if (capture_idx >= 0) {
                    for (int row = 0; row < count; ++row)
                        average_streams_kernel<<<16, 256>>>(
                            final_streams + size_t(row) * stream_stride,
                            c_collapsed_.get() + size_t(row) * hidden_, hidden_);
                    const int valid = std::min(count, df_tokens - pos0);
                    const size_t bytes = size_t(valid) * hidden_ * sizeof(float);
                    const size_t offset =
                        (size_t(capture_idx) * df_tokens + size_t(pos0)) * hidden_;
                    if (vram_store) {
                        check(cudaMemcpy(capture_device.get() + offset,
                                         c_collapsed_.get(), bytes,
                                         cudaMemcpyDeviceToDevice),
                              "retain whole-layer DFlash capture");
                    } else {
                        check(cudaMemcpy(capture_host.data() + offset,
                                         c_collapsed_.get(), bytes,
                                         cudaMemcpyDeviceToHost),
                              "spill whole-layer DFlash capture");
                        prompt_d2h += bytes;
                    }
                }

                if (deep_checks_) {
                    const int one = 1;
                    int valid = 0;
                    check(cudaMemcpy(finite_, &one, sizeof(one),
                                     cudaMemcpyHostToDevice),
                          "initialize whole-layer finite flag");
                    finite_kernel<<<16, 256>>>(
                        final_streams, count * kStreams * hidden_, finite_);
                    check(cudaMemcpy(&valid, finite_.get(), sizeof(valid),
                                     cudaMemcpyDeviceToHost),
                          "read whole-layer finite flag");
                    require(valid, "non-finite residual stream after whole-layer prefill layer " +
                                   std::to_string(layer));
                }
                if (!vram_store && layer + 1 < layers) {
                    check(cudaMemcpy(
                              prompt_host.data() + size_t(pos0) * stream_stride,
                              final_streams, stream_bytes, cudaMemcpyDeviceToHost),
                          "persist whole-layer FFN result");
                    prompt_d2h += stream_bytes;
                }
                if (!deep_checks_ && layer + 1 == layers) {
                    const int one = 1;
                    int valid = 0;
                    check(cudaMemcpy(finite_, &one, sizeof(one),
                                     cudaMemcpyHostToDevice),
                          "initialize finite flag");
                    finite_kernel<<<16, 256>>>(
                        final_streams, count * kStreams * hidden_, finite_);
                    check(cudaMemcpy(&valid, finite_.get(), sizeof(valid),
                                     cudaMemcpyDeviceToHost),
                          "read finite flag");
                    require(valid,
                            "non-finite residual stream after full-layer-major prefill");
                }
            }
            whole_phase_c += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - phase_begin).count();
            if (trace_layers_) {
                std::printf("prefill full-lm layer %02d/%02d whole-moe %.3f s\n",
                            layer + 1, layers,
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - layer_begin).count());
                std::fflush(stdout);
            }
            continue;
        }
        for (int pos0 = 0; pos0 < prompt_tokens; pos0 += kMaxChunk()) {
            const int count = std::min(kMaxChunk(), prompt_tokens - pos0);
            if (early_multi_route_on_) ++early_multi_batch_;
            float *in_place = vram_store ?
                prompt_device.get() + size_t(pos0) * stream_stride : c_stream_a_.get();
            if (layer == 0) {
                for (int row = 0; row < count; ++row) {
                    uint16_t *device_row = reinterpret_cast<uint16_t *>(stager_.load(
                        embedding, uint64_t(tokens[size_t(pos0 + row)]) * hidden_ * 2,
                        hidden_ * 2));
                    embed_repeat_kernel<<<16, 256>>>(
                        device_row, in_place + size_t(row) * stream_stride, hidden_);
                }
                check(cudaGetLastError(), "embedding repeat launch (full layer-major)");
            } else if (!vram_store) {
                const size_t bytes = size_t(count) * stream_stride * sizeof(float);
                check(cudaMemcpy(in_place,
                                 prompt_host.data() + size_t(pos0) * stream_stride,
                                 bytes, cudaMemcpyHostToDevice),
                      "restore prompt streams");
                prompt_h2d += bytes;
            }

            int capture_idx = -1;
            if (df_ && pos0 < df_tokens)
                for (int ci = 0; ci < 5; ++ci)
                    if (kDfCaptureLayers[ci] == layer) capture_idx = ci;
            prefill_layer_chunk_exact(layer, in_place, c_stream_b_.get(), count, pos0,
                                      capture_idx >= 0 ? c_collapsed_.get() : nullptr);

            if (capture_idx >= 0) {
                const int valid = std::min(count, df_tokens - pos0);
                const size_t bytes = size_t(valid) * hidden_ * sizeof(float);
                const size_t offset =
                    (size_t(capture_idx) * df_tokens + size_t(pos0)) * hidden_;
                if (vram_store) {
                    check(cudaMemcpy(capture_device.get() + offset, c_collapsed_.get(), bytes,
                                     cudaMemcpyDeviceToDevice),
                          "retain full-layer-major DFlash capture");
                } else {
                    check(cudaMemcpy(capture_host.data() + offset, c_collapsed_.get(), bytes,
                                     cudaMemcpyDeviceToHost),
                          "spill full-layer-major DFlash capture");
                    prompt_d2h += bytes;
                }
            }

            if (!vram_store && layer + 1 < layers) {
                const size_t bytes = size_t(count) * stream_stride * sizeof(float);
                check(cudaMemcpy(prompt_host.data() + size_t(pos0) * stream_stride,
                                 in_place, bytes, cudaMemcpyDeviceToHost),
                      "spill prompt streams");
                prompt_d2h += bytes;
            }
            if (!deep_checks_ && layer + 1 == layers) {
                const int one = 1;
                int valid = 0;
                check(cudaMemcpy(finite_, &one, sizeof(one), cudaMemcpyHostToDevice),
                      "initialize finite flag");
                finite_kernel<<<16, 256>>>(in_place, count * kStreams * hidden_, finite_);
                check(cudaMemcpy(&valid, finite_.get(), sizeof(valid), cudaMemcpyDeviceToHost),
                      "read finite flag");
                require(valid, "non-finite residual stream after full-layer-major prefill");
            }
        }
        if (trace_layers_) {
            std::printf("prefill full-lm layer %02d/%02d %.3f s\n", layer + 1, layers,
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - layer_begin).count());
            std::fflush(stdout);
        }
    }

    // DFlash target captures are independent of target prefill state. Replay
    // the exact five planes through the original <=128-row commit surface in
    // the original ascending chunk order after the target stack is complete.
    if (df_)
        for (int pos0 = 0; pos0 < df_tokens; pos0 += kMaxChunk()) {
            const int count = std::min(kMaxChunk(), df_tokens - pos0);
            const size_t bytes = size_t(count) * hidden_ * sizeof(float);
            for (int ci = 0; ci < 5; ++ci) {
                const size_t offset = (size_t(ci) * df_tokens + size_t(pos0)) * hidden_;
                if (vram_store) {
                    check(cudaMemcpy(df_->capture_row(ci, 0), capture_device.get() + offset,
                                     bytes, cudaMemcpyDeviceToDevice),
                          "replay full-layer-major DFlash capture");
                } else {
                    check(cudaMemcpy(df_->capture_row(ci, 0), capture_host.data() + offset,
                                     bytes, cudaMemcpyHostToDevice),
                          "restore full-layer-major DFlash capture");
                    prompt_h2d += bytes;
                }
            }
            df_->commit(count, pos0);
        }

    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - all_begin).count();
    const uint64_t records_after = expert_stager_ ? expert_stager_->records_read() : 0;
    const uint64_t io_after = expert_stager_ ? expert_stager_->io_bytes() : 0;
    const uint64_t h2d_records_after = expert_stager_ ?
        expert_stager_->packed_h2d_records() : 0;
    const uint64_t whole_sidecar_bytes = uint64_t(
        whole_normalized.size() + whole_expert_out.size() +
        whole_post.size() + whole_comb.size()) * sizeof(float);
    std::printf("prefill_full_lm store=%s tokens=%d prompt_h2d=%llu prompt_d2h=%llu "
                "records_read=%llu io_bytes=%llu expert_h2d_records=%llu "
                "whole_moe=%d whole_layers=%llu whole_union=%llu whole_uploads=%llu "
                "whole_legacy_uploads=%llu whole_uploads_saved=%llu "
                "whole_rows=%llu whole_sidecar_bytes=%llu phase_a=%.3f phase_b=%.3f "
                "phase_c=%.3f wall=%.3f s\n",
                store.c_str(), prompt_tokens,
                (unsigned long long)prompt_h2d, (unsigned long long)prompt_d2h,
                (unsigned long long)(records_after - records_before),
                (unsigned long long)(io_after - io_before),
                (unsigned long long)(h2d_records_after - h2d_records_before),
                whole_moe ? 1 : 0,
                (unsigned long long)whole_layers,
                (unsigned long long)whole_union_experts,
                (unsigned long long)whole_upload_calls,
                (unsigned long long)whole_legacy_upload_calls,
                (unsigned long long)(whole_legacy_upload_calls - whole_upload_calls),
                (unsigned long long)whole_user_rows,
                (unsigned long long)whole_sidecar_bytes,
                whole_phase_a, whole_phase_b, whole_phase_c, seconds);
    std::fflush(stdout);
}

void Runner::prefill(const std::vector<int> &tokens, int position_base, bool capture) {
    const int count = int(tokens.size());
    require(count >= 1 && count <= kMaxChunk(), "prefill chunk out of range");
    if (early_multi_route_on_) ++early_multi_batch_;
    if (capture) {
        require(mtp_draft_total_, "verify capture requested with MTP disabled");
        // Snapshot the recurrent state so a rejected draft prefix can be
        // restored and replayed to the accepted boundary. Row-sequential
        // verify rounds never roll back (their state always stands at the
        // accepted boundary), so they skip these two whole-state copies.
        if (verify_may_rollback_) {
            require(!forced_sequential_verify_,
                    "batch verify is forbidden after sequential snapshot elision");
            check(cudaMemcpyAsync(kda_snap_.get(), kda_states_.get(),
                                  kda_states_.size() * sizeof(float), cudaMemcpyDeviceToDevice),
                  "snapshot KDA states");
            check(cudaMemcpyAsync(conv_snap_.get(), conv_history_.get(),
                                  conv_history_.size() * sizeof(float), cudaMemcpyDeviceToDevice),
                  "snapshot conv history");
        }
        kda_archive_ = true;
        // A verify chunk touches roughly 2,000 distinct records, far beyond
        // the 379-record host tier.  Admitting that scan evicts every useful
        // per-layer/token-0 record and measured just 0.3% hits.  Keep only the
        // first eight distinct experts per layer (the normal moe_multi policy),
        // a 42*8=336-record working set that actually fits.  Retain the old
        // behavior as an explicit A/B switch.
        verify_populate_ = std::getenv("INSIGNIA_GLM53_VERIFY_CACHE_ALL") != nullptr;
    }
    const TensorLocation &embedding = model_.tensor("model.language_model.embed_tokens.weight");
    require(embedding.type == TensorType::bf16 && embedding.shape.size() == 2 &&
            embedding.shape[1] == uint32_t(hidden_), "wrong embedding geometry");
    for (int token = 0; token < count; ++token) {
        require(tokens[token] >= 0 && tokens[token] < int(model_.vocab_size()),
                "prefill token is outside the vocabulary");
        uint16_t *device_row = reinterpret_cast<uint16_t *>(
            stager_.load(embedding, uint64_t(tokens[token]) * hidden_ * 2, hidden_ * 2));
        embed_repeat_kernel<<<16, 256>>>(device_row, c_stream_a_.get() + size_t(token) * kStreams * hidden_,
                                         hidden_);
    }
    check(cudaGetLastError(), "embedding repeat launch (prefill)");
    float *streams = c_stream_a_;
    float *next_streams = c_stream_b_;
    const int layers = int(model_.layers());
    for (int layer = 0; layer < layers; ++layer) {
        float *dflash_capture = nullptr;
        if (df_)
            for (int ci = 0; ci < 5; ++ci)
                if (kDfCaptureLayers[ci] == layer)
                    dflash_capture = df_->capture_row(ci, capture_offset_);
        prefill_layer_chunk_exact(layer, streams, next_streams, count, position_base,
                                  dflash_capture);
    }
    if (!deep_checks_) {
        const int one = 1;
        int valid = 0;
        check(cudaMemcpy(finite_, &one, sizeof(one), cudaMemcpyHostToDevice), "initialize finite flag");
        finite_kernel<<<16, 256>>>(streams, count * kStreams * hidden_, finite_);
        check(cudaMemcpy(&valid, finite_.get(), sizeof(valid), cudaMemcpyDeviceToHost), "read finite flag");
        require(valid, "non-finite residual stream after prefill stack");
    }
    // Prompt chunks (capture=false) commit immediately; verify rounds
    // (capture=true) commit only the accepted prefix from main().
    if (df_ && !capture)
        df_->commit(count, position_base);
    if (capture) {
        kda_archive_ = false;
        verify_populate_ = false;
        // Per-position final hidden (mean of the 4 mHC streams) plus one
        // greedy logits row per candidate: the acceptance-check targets.
        for (int token = 0; token < count; ++token)
            average_streams_kernel<<<16, 256>>>(
                streams + size_t(token) * kStreams * hidden_,
                verify_means_.get() + size_t(token) * hidden_, hidden_);
        check(cudaGetLastError(), "verify mean launch");
        for (int token = 0; token < count; ++token) {
            rms("model.language_model.norm.weight",
                verify_means_.get() + size_t(token) * hidden_,
                c_normalized_.get() + size_t(token) * hidden_, 1, hidden_);
            if (mtp_variant_ == 1)
                check(cudaMemcpyAsync(verify_normed_.get() + size_t(token) * hidden_,
                                      c_normalized_.get() + size_t(token) * hidden_,
                                      size_t(hidden_) * sizeof(float), cudaMemcpyDeviceToDevice),
                      "capture verify normed hidden");
            linear("lm_head.weight", c_normalized_.get() + size_t(token) * hidden_,
                   verify_logits_.get() + size_t(token) * model_.vocab_size(),
                   int(model_.vocab_size()), hidden_);
        }
    }
}

std::vector<std::pair<int, float>> Runner::step(
    int token, int position, int layer_limit, bool produce_logits) {
    const int layers = int(model_.layers());
    ++token_index_;
    require(token >= 0 && token < int(model_.vocab_size()), "input token is outside vocabulary");
    require(layer_limit >= 1 && layer_limit <= layers,
            "layer limit must be between 1 and the indexed layer count");
    require(position >= 0 && position < kMaxContext(), "position exceeds the exact-attention cache");
    const TensorLocation &embedding = model_.tensor("model.language_model.embed_tokens.weight");
    require(embedding.type == TensorType::bf16 && embedding.shape.size() == 2 &&
            embedding.shape[0] == model_.vocab_size() && embedding.shape[1] == uint32_t(hidden_),
            "wrong embedding geometry");
    uint16_t *device_row = reinterpret_cast<uint16_t *>(
        stager_.load(embedding, uint64_t(token) * hidden_ * 2, hidden_ * 2));
    embed_repeat_kernel<<<16, 256>>>(device_row, streams_a_, hidden_);
    check(cudaGetLastError(), "embedding repeat launch");

    float *streams = streams_a_;
    float *next_streams = streams_b_;
    // Layers 0-2 are dense; kick the first sparse layers' records before the
    // stack starts so their reads overlap attention compute.
        if (prefetch_on_ && expert_stager_)
            for (size_t layer = 3; layer < prev_routing_.size() && layer < 6; ++layer)
                if (is_sparse_[layer])
                    expert_stager_->prefetch(int(layer), prev_routing_[layer].data(), moe_topk_);
        if (prefetch_on_ && !cct_.empty()) {
            // Warm the reader pool with the prompt-side layers before the
            // stack starts (same window the prev-token prefetch uses).
            for (size_t layer = 3; layer + 1 < prev_routing_.size() && layer < 6; ++layer)
                cct_prefetch(int(layer));
        }
    const auto layers_begin = std::chrono::steady_clock::now();
    for (int layer = 0; layer < layer_limit; ++layer) {
        const auto begin = std::chrono::steady_clock::now();
        const std::string base = layer_stem(layer);
        mhc(base + "hc_attn", base + "input_layernorm.weight", streams, normalized_);
        seam(layer, 1, normalized_, hidden_);
        if (early_route_on_ && is_sparse_[size_t(layer)])
            early_route(layer, normalized_);
        if (is_mla_[layer])
            mla(layer, normalized_, attention_, position);
        else
            kda(layer, normalized_, attention_, position);
        seam(layer, 2, attention_, hidden_);
        check(insignia::glm53::mhc_mix(streams, attention_, post_, comb_, next_streams, hidden_), "attention mHC mix");
        seam(layer, 3, next_streams, kStreams * hidden_);
        std::swap(streams, next_streams);

        mhc(base + "hc_ffn", base + "post_attention_layernorm.weight", streams, normalized_);
        seam(layer, 4, normalized_, hidden_);
        if (is_sparse_[layer])
            sparse_moe(layer, normalized_, ffn_);
        else
            dense_mlp(base + "mlp.", normalized_, ffn_, dense_intermediate_);
        seam(layer, 5, ffn_, hidden_);
        check(insignia::glm53::mhc_mix(streams, ffn_, post_, comb_, next_streams, hidden_), "FFN mHC mix");
        seam(layer, 6, next_streams, kStreams * hidden_);
        std::swap(streams, next_streams);
        if (df_) {
            for (int ci = 0; ci < 5; ++ci)
                if (kDfCaptureLayers[ci] == layer)
                    average_streams_kernel<<<16, 256>>>(streams, df_->capture_row(ci, 0), hidden_);
        }
        if (!layer_dump_probed_) {
            layer_dump_probed_ = true;
            if (const char *path = std::getenv("INSIGNIA_GLM53_LAYER_DUMP"))
                layer_dump_ = std::fopen(path, "wb");
        }
        if (layer_dump_) {
            average_streams_kernel<<<16, 256>>>(streams, collapsed_, hidden_);
            layer_dump_host_.assign(size_t(hidden_), 0.0f);
            check(cudaMemcpy(layer_dump_host_.data(), collapsed_,
                             size_t(hidden_) * sizeof(float), cudaMemcpyDeviceToHost),
                  "download layer dump");
            const int32_t header[3] = {int32_t(token_index_), layer, int32_t(hidden_)};
            std::fwrite(header, sizeof(header), 1, layer_dump_);
            std::fwrite(layer_dump_host_.data(), sizeof(float), size_t(hidden_), layer_dump_);
            std::fflush(layer_dump_);
        }
        if (deep_checks_) {
            const int one = 1;
            int valid = 0;
            check(cudaMemcpy(finite_, &one, sizeof(one), cudaMemcpyHostToDevice), "initialize finite flag");
            finite_kernel<<<16, 256>>>(streams, kStreams * hidden_, finite_);
            check(cudaMemcpy(&valid, finite_.get(), sizeof(valid), cudaMemcpyDeviceToHost), "read finite flag");
            require(valid, "non-finite residual stream after layer " + std::to_string(layer));
        }
        if (trace_layers_) {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            std::printf("layer %02d %-3s %.3f s\n", layer, is_mla_[layer] ? "MLA" : "KDA", seconds);
            std::fflush(stdout);
        }
    }
    if (!deep_checks_) {
        // Two device syncs per layer cost ~100 ms across the 45-layer stack;
        // one drain at the end catches the same non-finite corruption.
        const int one = 1;
        int valid = 0;
        check(cudaMemcpy(finite_, &one, sizeof(one), cudaMemcpyHostToDevice), "initialize finite flag");
        finite_kernel<<<16, 256>>>(streams, kStreams * hidden_, finite_);
        check(cudaMemcpy(&valid, finite_.get(), sizeof(valid), cudaMemcpyDeviceToHost), "read finite flag");
        require(valid, "non-finite residual stream after the layer stack");
    }
    if (trace_layers_ && layer_limit == layers)
        std::fprintf(stderr, "profile: %d layers %.3f ms\n", layer_limit,
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - layers_begin).count());
    // The token this step() processed is committed by construction (seed or
    // empty-round truth0); its captures feed the drafter K/V cache.
    if (df_ && layer_limit == layers)
        df_->commit(1, position);

    if (layer_limit != layers || !produce_logits) {
        const double q8_seconds = q8_stager_ ? q8_stager_->io_seconds() : 0.0;
        const uint64_t q8_bytes = q8_stager_ ? q8_stager_->io_bytes() : 0;
        const double io_seconds = stager_.io_seconds() + expert_io_seconds() + q8_seconds;
        const uint64_t io_bytes = stager_.io_bytes() + expert_io_bytes() + q8_bytes;
        if (layer_limit != layers)
            std::printf("partial %d-layer smoke pass streamed %.3f GiB in %.3f s (%.2f GB/s)\n",
                        layer_limit, io_bytes / double(1ull << 30), io_seconds,
                        io_bytes / io_seconds / 1.0e9);
        return {};
    }

    average_streams_kernel<<<16, 256>>>(streams, collapsed_, hidden_);
    if (mtp_draft_total_)
        check(cudaMemcpyAsync(last_avg_.get(), collapsed_, size_t(hidden_) * sizeof(float),
                              cudaMemcpyDeviceToDevice), "capture decode hidden");
    rms("model.language_model.norm.weight", collapsed_, normalized_, 1, hidden_);
    if (mtp_draft_total_ && mtp_variant_ == 1)
        check(cudaMemcpyAsync(last_normed_.get(), normalized_, size_t(hidden_) * sizeof(float),
                              cudaMemcpyDeviceToDevice), "capture decode normed hidden");
    const auto head_begin = std::chrono::steady_clock::now();
    const TensorLocation &head = model_.tensor("lm_head.weight");
    const Q8TensorLocation *q8_head =
        q8_index_ ? q8_index_->find("lm_head.weight") : nullptr;
    if (stager_.is_resident(head) ||
        (q8_head && q8_stager_->try_pin("lm_head.weight", *q8_head))) {
        // A resident head skips the 8192-row streaming chunks: one full-width
        // GEMV launch covers the vocabulary.
        linear("lm_head.weight", normalized_, logits_.get(), int(head.shape[0]), hidden_);
    } else {
        constexpr int rows_per_chunk = 8192;
        for (uint64_t row = 0; row < head.shape[0]; row += rows_per_chunk) {
            const int rows = int(std::min<uint64_t>(rows_per_chunk, head.shape[0] - row));
            linear_rows("lm_head.weight", head, row, rows, normalized_, logits_.get() + row, hidden_);
        }
    }
    check(cudaDeviceSynchronize(), "LM head completion");
    const auto head_end = std::chrono::steady_clock::now();
    std::vector<float> host_logits(model_.vocab_size());
    check(cudaMemcpy(host_logits.data(), logits_.get(), host_logits.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "download logits");
    if (df_calibration_guard_js_ > 0.0f || df_retry_top1_drop_ > 0.0f)
        retain_df_prior_logits(logits_.get(), host_logits.data());
    if (const char *dump_path = std::getenv("INSIGNIA_GLM53_LOGITS_DUMP")) {
        static std::FILE *dump = nullptr;
        if (!dump) dump = std::fopen(dump_path, "wb");
        if (dump) std::fwrite(host_logits.data(), sizeof(float), host_logits.size(), dump);
    }
    std::vector<int> order(host_logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + 10, order.end(),
        [&](int left, int right) { return host_logits[left] > host_logits[right]; });
    std::vector<std::pair<int, float>> top;
    for (int index = 0; index < 10; ++index) top.emplace_back(order[index], host_logits[order[index]]);
    if (std::getenv("INSIGNIA_GLM53_PROFILE"))
        std::fprintf(stderr, "profile: lm_head+topk %.3f ms\n",
                     std::chrono::duration<double, std::milli>(head_end - head_begin).count() +
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - head_end).count());
    const double q8_seconds = q8_stager_ ? q8_stager_->io_seconds() : 0.0;
    const uint64_t q8_bytes = q8_stager_ ? q8_stager_->io_bytes() : 0;
    const double io_seconds = stager_.io_seconds() + expert_io_seconds() + q8_seconds;
    const uint64_t io_bytes = stager_.io_bytes() + expert_io_bytes() + q8_bytes;
    std::printf("streamed %.3f GiB in %.3f s (%.2f GB/s hierarchy aggregate)\n",
                io_bytes / double(1ull << 30), io_seconds, io_bytes / io_seconds / 1.0e9);
    std::printf("  source BF16 %.3f GiB / %.3f s (%.2f GB/s)\n",
                stager_.io_bytes() / double(1ull << 30), stager_.io_seconds(),
                stager_.io_bytes() / stager_.io_seconds() / 1.0e9);
    if (expert_stager_)
        std::printf("  QD8 expert O_DIRECT %.3f GiB / %.3f s (%.2f GB/s)\n",
                    expert_io_bytes() / double(1ull << 30), expert_io_seconds(),
                    expert_io_bytes() / expert_io_seconds() / 1.0e9);
    if (expert_stager_ &&
        (stripe_model_ || model_.alt_shard_count() || expert_stager_->stripe_fallbacks()))
        std::printf("  expert drives C %llu records/%.3f GiB, E %llu records/%.3f GiB, "
                    "%llu exact-C fallbacks\n",
                    (unsigned long long)expert_stager_->drive_records(0),
                    expert_stager_->drive_bytes(0) / double(1ull << 30),
                    (unsigned long long)expert_stager_->drive_records(1),
                    expert_stager_->drive_bytes(1) / double(1ull << 30),
                    (unsigned long long)expert_stager_->stripe_fallbacks());
    if (expert_stager_ && expert_stager_->cache_lookups())
        std::printf("  NVFP4 cache %llu/%llu hits (%.1f%%, %.3f GiB NVMe+H2D avoided; %d slots)\n",
                    (unsigned long long)expert_stager_->cache_hits(),
                    (unsigned long long)expert_stager_->cache_lookups(),
                    100.0 * expert_stager_->cache_hits() / expert_stager_->cache_lookups(),
                    expert_stager_->cache_hits() *
                        (ExpertStager::kBodyBytes + ExpertStager::kScaleBytes + 3 * sizeof(float)) /
                        double(1ull << 30),
                    expert_stager_->cache_slots());
    if (expert_stager_ && expert_stager_->device_lookups())
        std::printf("  VRAM expert tier %llu/%llu hits (%.1f%%, %.3f GiB PCIe avoided; %d slots)\n",
                    (unsigned long long)expert_stager_->device_hits(),
                    (unsigned long long)expert_stager_->device_lookups(),
                    100.0 * expert_stager_->device_hits() / expert_stager_->device_lookups(),
                    expert_stager_->device_hits() * ExpertStager::kPayloadCapacity / double(1ull << 30),
                    expert_stager_->device_slots());
    if (expert_stager_ && expert_stager_->f3_rescued())
        std::printf("  F3 device-consult rescued %llu reads (%.3f GiB NVMe avoided)\n",
                    (unsigned long long)expert_stager_->f3_rescued(),
                    expert_stager_->f3_rescued() *
                        (ExpertStager::kBodyBytes + ExpertStager::kScaleBytes) /
                        double(1ull << 30));
    if (expert_stager_ && expert_stager_->demoted_cold())
        std::printf("  SLRU demoted %llu rejected-row records to the probationary tail\n",
                    (unsigned long long)expert_stager_->demoted_cold());
    if (expert_stager_ && expert_stager_->cache_lookups())
        std::printf("  expert read-wait %.3f s of %.3f s expert wall (demand blocks on NVMe/pool)\n",
                    expert_stager_->read_wait_seconds(), expert_io_seconds());
    if (expert_stager_ && expert_stager_->prefetch_started())
        std::printf("  expert prefetch %llu started, %llu adopted, %llu wasted (%.3f GiB speculative)\n",
                    (unsigned long long)expert_stager_->prefetch_started(),
                    (unsigned long long)expert_stager_->prefetch_useful(),
                    (unsigned long long)expert_stager_->prefetch_wasted_observable(),
                    expert_stager_->prefetch_bytes() / double(1ull << 30));
    if (expert_stager_ && expert_stager_->packed_gpu_scales() &&
        expert_stager_->packed_h2d_records()) {
        const uint64_t expanded = expert_stager_->packed_h2d_records() *
            (ExpertStager::kBodyBytes + ExpertStager::kScaleBytes);
        std::printf("  packed GPU H2D %.3f GiB (%.3f GiB / %.2f%% PCIe bytes avoided)\n",
                    expert_stager_->packed_h2d_bytes() / double(1ull << 30),
                    (expanded - expert_stager_->packed_h2d_bytes()) / double(1ull << 30),
                    100.0 * (expanded - expert_stager_->packed_h2d_bytes()) / expanded);
    }
    if (expert_stager_ && expert_stager_->packed_experts() &&
        expert_stager_->packed_expanded_bytes())
        std::printf("  packed expand %.3f GiB in %.3f s (%.2f GiB/s; %.3f ms/record)\n",
                    expert_stager_->packed_expanded_bytes() / double(1ull << 30),
                    expert_stager_->packed_expand_seconds(),
                    expert_stager_->packed_expanded_bytes() /
                        (expert_stager_->packed_expand_seconds() * double(1ull << 30)),
                    expert_stager_->packed_expand_seconds() * 1.0e3 /
                        double(expert_stager_->packed_expanded_bytes() /
                               (ExpertStager::kBodyBytes + ExpertStager::kScaleBytes)));
    if (early_route_total_)
        std::printf("  pre-attention route recall %llu/%llu (%.1f%%)\n",
                    (unsigned long long)early_route_hits_,
                    (unsigned long long)early_route_total_,
                    100.0 * early_route_hits_ / early_route_total_);
    if (early_multi_actual_)
        std::printf("  batched pre-attention union recall %.1f%%, precision %.1f%% "
                    "(%llu useful / %llu predicted / %llu actual; %llu/%llu hints started)\n",
                    100.0 * early_multi_hits_ / early_multi_actual_,
                    100.0 * early_multi_hits_ / early_multi_predicted_,
                    (unsigned long long)early_multi_hits_,
                    (unsigned long long)early_multi_predicted_,
                    (unsigned long long)early_multi_actual_,
                    (unsigned long long)early_multi_started_,
                    (unsigned long long)early_multi_hints_);
    if (df_approx_rows_) {
        const double exact_slots = double(df_approx_rows_) * moe_topk_;
        std::printf("  DFlash approximate k %.3f mean (%llu/%llu slots, %.1f%% removed)\n",
                    double(df_approx_slots_) / df_approx_rows_,
                    (unsigned long long)df_approx_slots_,
                    (unsigned long long)(df_approx_rows_ * uint64_t(moe_topk_)),
                    100.0 * (1.0 - df_approx_slots_ / exact_slots));
        std::printf("  DFlash expert union %llu/%llu records (%.1f%% removed); k histogram",
                    (unsigned long long)df_approx_union_,
                    (unsigned long long)df_approx_exact_union_,
                    100.0 * (1.0 - double(df_approx_union_) / df_approx_exact_union_));
        for (int k = 1; k <= moe_topk_; ++k)
            if (df_approx_k_hist_[size_t(k)])
                std::printf(" k%d=%llu", k, (unsigned long long)df_approx_k_hist_[size_t(k)]);
        std::fputc('\n', stdout);
        if (df_logit_guard_rows_)
            std::printf("  DFlash logit guard %s %llu/%llu verify rows (%.1f%%)\n",
                        (df_uncertainty_top1_p_ > 0.0f ||
                         df_uncertainty_top1_drop_ > 0.0f) &&
                                df_uncertainty_guard_k_ < 8
                            ? "raised-k" :
                        (df_cache_route_k_ && df_cache_guard_retain_ < 8
                            ? "tightened" : "exactified"),
                        (unsigned long long)df_logit_guarded_rows_,
                        (unsigned long long)df_logit_guard_rows_,
                        100.0 * df_logit_guarded_rows_ / df_logit_guard_rows_);
    }
    if (df_cache_route_rows_)
        std::printf("  DFlash cache route changed %llu/%llu rows with %llu substitutions "
                    "(regret mean %.6f max %.6f; immediate disk/H2D records saved %lld/%lld)\n",
                    (unsigned long long)df_cache_route_changed_,
                    (unsigned long long)df_cache_route_rows_,
                    (unsigned long long)df_cache_route_substitutions_,
                    df_cache_route_regret_sum_ / df_cache_route_rows_,
                    df_cache_route_regret_max_,
                    (long long)df_cache_route_disk_saved_,
                    (long long)df_cache_route_h2d_saved_);
    if (df_cache_joint_groups_)
        std::printf("  DFlash joint cache union %llu/%llu records (%.1f%% removed; "
                    "union disk/H2D saved %lld/%lld across %llu layer groups)\n",
                    (unsigned long long)df_cache_joint_selected_union_,
                    (unsigned long long)df_cache_joint_baseline_union_,
                    100.0 * (df_cache_joint_baseline_union_ - df_cache_joint_selected_union_) /
                        df_cache_joint_baseline_union_,
                    (long long)df_cache_joint_disk_saved_,
                    (long long)df_cache_joint_h2d_saved_,
                    (unsigned long long)df_cache_joint_groups_);
    report_cache_selector();
    if (df_calibration_guard_rounds_)
        std::printf("  DFlash calibration guard exactified %llu/%llu rounds "
                    "(JS mean %.6f max %.6f)\n",
                    (unsigned long long)df_calibration_guarded_rounds_,
                    (unsigned long long)df_calibration_guard_rounds_,
                    df_calibration_js_sum_ / df_calibration_guard_rounds_,
                    df_calibration_js_max_);
    if (df_retry_rounds_)
        std::printf("  DFlash post-verify retried %llu/%llu rounds "
                    "(target-p drop mean %.6f max %.6f)\n",
                    (unsigned long long)df_retry_triggered_rounds_,
                    (unsigned long long)df_retry_rounds_,
                    df_retry_drop_sum_ / df_retry_rounds_, df_retry_drop_max_);
    if (q8_stager_)
        std::printf("  %s matrix cache %.3f GiB / %.3f s (%.2f GB/s)\n",
                    q8_index_->format() == Cache8Format::q8 ? "Q8" : "FP8",
                    q8_bytes / double(1ull << 30), q8_seconds,
                    q8_bytes / q8_seconds / 1.0e9);
    return top;
}

std::vector<int> parse_token_list(std::string encoded) {
    if (!encoded.empty() && encoded[0] == '@') {
        std::FILE *file = std::fopen(encoded.c_str() + 1, "r");
        require(file, "cannot open token file");
        std::vector<char> payload;
        char chunk[65536];
        size_t read = 0;
        while ((read = std::fread(chunk, 1, sizeof(chunk), file)) > 0)
            payload.insert(payload.end(), chunk, chunk + read);
        std::fclose(file);
        encoded.assign(payload.begin(), payload.end());
        while (!encoded.empty() && std::isspace(static_cast<unsigned char>(encoded.back())))
            encoded.pop_back();
    }
    std::vector<int> tokens;
    size_t begin = 0;
    while (begin < encoded.size()) {
        const size_t comma = encoded.find(',', begin);
        tokens.push_back(std::stoi(encoded.substr(begin, comma - begin)));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return tokens;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 7) {
        std::fprintf(stderr,
            "usage: %s MODEL_ROOT MODEL.index [TOKENS=154820] [LAYERS=0(all)] [GENERATE=1] [8BIT_PREFIX]\n"
            "  INSIGNIA_GLM53_MTP=K enables K-token MTP speculative decode (greedy-exact)\n"
            "  INSIGNIA_GLM53_PREFILL_FULL_LAYER_MAJOR=0|1 overrides automatic multi-chunk prefill\n"
            "  INSIGNIA_GLM53_PREFILL_WHOLE_LAYER_MOE=1 enables exact three-phase sparse prefill\n",
            argv[0]);
        return 64;
    }
    try {
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        require(properties.major == 8 && properties.minor == 9,
                "GLM-5.3 runner is deliberately compiled only for sm_89");
        // Prompt tokens: a literal CSV, or "@file" to read the CSV from a
        // file (long contexts do not fit any command line).
        std::vector<int> tokens = parse_token_list(argc >= 4 ? argv[3] : "154820");
        require(!tokens.empty() && tokens.size() <= kMaxContext(), "token list exceeds the context limit");
        const int layers_argc = argc >= 5 ? std::atoi(argv[4]) : 0;
        const int generate = argc >= 6 ? std::atoi(argv[5]) : 1;
        require(generate >= 1 && tokens.size() + size_t(generate) - 1 <= kMaxContext(),
                "generation must fit the exact-attention cache");
        Runner runner(argv[1], argv[2], argc >= 7 ? argv[6] : "");
        const int layers = layers_argc > 0 ? layers_argc : runner.layer_count();

        const auto begin = std::chrono::steady_clock::now();
        std::vector<std::pair<int, float>> top;
        // A single <=128-row chunk already stays layer-major on device. Longer
        // prompts default to the full-prompt scheduler: spilling residuals
        // between layers is vastly cheaper than rereading the sparse experts
        // for every chunk. The final token keeps the per-token logits path.
        if (tokens.size() > 1) {
            const size_t prefill_count = tokens.size() - 1;
            const char *full_lm = std::getenv("INSIGNIA_GLM53_PREFILL_FULL_LAYER_MAJOR");
            const bool full_lm_on = full_lm ? std::atoi(full_lm) != 0 :
                prefill_count > size_t(Runner::kMaxChunk());
            if (full_lm_on) {
                std::printf("prefill scheduler: full-prompt layer-major (%s)\n",
                            full_lm ? "forced" : "auto multi-chunk");
                std::fflush(stdout);
                std::vector<int> prompt(tokens.begin(), tokens.end() - 1);
                runner.prefill_prompt_full_layer_major(prompt);
            } else {
                for (size_t consumed = 0; consumed < prefill_count; ) {
                    const size_t take = std::min<size_t>(Runner::kMaxChunk(), prefill_count - consumed);
                    std::vector<int> chunk(tokens.begin() + int(consumed),
                                           tokens.begin() + int(consumed + take));
                    runner.prefill(chunk, int(consumed));
                    consumed += take;
                    std::printf("prompt %zu/%zu tokens prefilled\n", consumed, tokens.size());
                    std::fflush(stdout);
                }
            }
        }
        top = runner.step(tokens.back(), int(tokens.size()) - 1, layers, true);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        if (const char *forced = std::getenv("INSIGNIA_GLM53_FORCE_TOKENS")) {
            std::vector<int> forced_tokens = parse_token_list(forced);
            require(!forced_tokens.empty(), "target-forced token list is empty");
            runner.force_logits(forced_tokens, int(tokens.size()), tokens.back(),
                                std::getenv("INSIGNIA_GLM53_FORCE_LOGITS_DUMP"));
            return 0;
        }
        if (!top.empty()) {
            std::vector<int> generated;
            generated.reserve(generate);
            const int mtp_k = runner.mtp_k();
            if (runner.dflash2_on() && generate > 1) {
                // DFlash2 block-diffusion rounds: one drafter forward
                // proposes 7 candidates for the anchor (whose pending target
                // argmax is truth0), one target verify forward checks them,
                // exactly like the MTP flow below. Committed output is
                // identical to plain greedy decode by construction.
                int position = int(tokens.size()) - 1;
                int root = tokens.back();
                int truth0 = top.front().first;
                // Renewal reward is accepted draft work, not merely output
                // committed by the round.  Empty rounds and the k=1 scalar
                // bypass advance the output stream but accept no draft.
                double committed_total = 0.0, accepted_draft_total = 0.0;
                double draft_total = 0.0, verify_total = 0.0;
                double fallback_total = 0.0;
                int rounds = 0, verified_rounds = 0, empty_rounds = 0;
                // kDrafts is eight; index eight must be representable.
                std::array<int, 9> accept_hist{};
                const int verify_k = [] {
                    const char *value = std::getenv("INSIGNIA_GLM53_DF_VERIFY_K");
                    return std::clamp(value ? std::atoi(value) : 4, 1,
                                      insignia::glm53::DFlash2Drafter::kDrafts);
                }();
                // Verify mode: batch (one k-row pass) wins when acceptance is
                // high because the expert-union batching amortizes; row-
                // sequential wins when rounds reject early because the
                // rejected tail's experts are never read. Track a rolling
                // acceptance EMA and pick per round; env overrides force.
                const int df_verify_mode_env = [] {
                    if (std::getenv("INSIGNIA_GLM53_DF_SEQ_VERIFY")) return 1;
                    if (std::getenv("INSIGNIA_GLM53_DF_BATCH_VERIFY")) return 2;
                    return 0;
                }();
                // Adaptive draft length: on real text the acceptance EMA
                // sits well below the block size, and every drafted position
                // beyond it widens the verify expert union for tokens that
                // get rejected. Cap the draft at EMA-driven headroom; the
                // verify pass itself stays greedy-exact either way.
                const int adaptive_k_mode = [] {
                    const char *value = std::getenv("INSIGNIA_GLM53_DF_ADAPTIVE_K");
                    return value ? std::atoi(value) : 1;
                }();
                const bool adaptive_k_on = adaptive_k_mode != 0;
                double accept_ema = 0.0;
                bool accept_ema_init = false;
                // Adaptive draft length v2 (DF_ADAPTIVE_K=2): the corrected
                // speculative economics from audits/s8 §2 --
                //   T(k) = [D + (1-p1)F + p1 * b * d(k)] / [(1-p1) + sum_j S(j)]
                // with draft cost D and empty-round fallback F from running
                // EMAs, per-record cost b from a decayed least-squares
                // regression of verify wall vs demand record reads (U3),
                // survival S(j) from censoring-correct per-position hazards,
                // and d(k) = 42*8*k*ratio[k] from the measured sticky union
                // curve. 8-point argmax with 1% hysteresis plus a mandatory
                // +/-1 probe every 16 rounds (probe-less argmax deadlocks at
                // k=1 with 11-46% regret). Per-round CSV via DF_COSTTRACE.
                double v2_dhat_ms = 17.0, v2_fhat_ms = 450.0;
                double v2_qhat[9] = {0.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7};
                double v2_sxx = 0, v2_sxy = 0, v2_sx = 0, v2_sy = 0, v2_sn = 0;
                double v2_bhat = 1.4;
                int v2_kstar = 0;
                int v2_probe_dir = 1;
                const bool v2_costtrace = std::getenv("INSIGNIA_GLM53_DF_COSTTRACE") != nullptr;
                const bool v2_k1_scalar =
                    std::getenv("INSIGNIA_GLM53_DF_K1_SCALAR") != nullptr;
                int v2_scalar_rounds = 0;
                // Sticky union curve d(k)/336k: s6 measured K=2..5 (0.903,
                // 0.859, 0.825, 0.785), s7 extrapolated K=6..8 (0.754, 0.727,
                // 0.700); verifies at 1067/1109 records at k=4.
                const double v2_union_ratio[9] = {0.0, 1.0, 0.903, 0.859, 0.825,
                                                  0.785, 0.754, 0.727, 0.700};
                auto v2_pick_k = [&](int fallback_k) {
                    if (!adaptive_k_on) return fallback_k;
                    if (adaptive_k_mode != 2)
                        return accept_ema_init
                                   ? std::clamp(int(accept_ema * 1.3) + 1, 2, fallback_k)
                                   : fallback_k;
                    const double p1 = v2_qhat[1];
                    double denom = 1.0 - p1;
                    double surv = p1;
                    double best_t = 1.0e30, t_at_star = 1.0e30;
                    int best_k = 1;
                    for (int k = 1; k <= verify_k; ++k) {
                        denom += surv;  // S(k) enters the accepted-token sum
                        const double cost = v2_dhat_ms + (1.0 - p1) * v2_fhat_ms +
                                            p1 * v2_bhat * (42.0 * 8.0 * k * v2_union_ratio[k]);
                        const double t = cost / denom;
                        if (k == v2_kstar) t_at_star = t;
                        if (t < best_t) {
                            best_t = t;
                            best_k = k;
                        }
                        surv *= v2_qhat[k + 1 > 8 ? 8 : k + 1];
                    }
                    if (v2_kstar < 1)
                        v2_kstar = best_k;
                    else if (best_k != v2_kstar && best_t < t_at_star * 0.99)
                        v2_kstar = best_k;  // 1% hysteresis against oscillation
                    int chosen = v2_kstar;
                    // Mandatory exploration: a wrong-surface argmax otherwise
                    // deadlocks (s8: k=1 deadlock at 11-46% regret).
                    if (rounds && rounds % 16 == 0) {
                        chosen = v2_kstar + v2_probe_dir;
                        if (chosen > verify_k || chosen < 1) {
                            v2_probe_dir = -v2_probe_dir;
                            chosen = v2_kstar + v2_probe_dir;
                        }
                        v2_probe_dir = -v2_probe_dir;
                    }
                    return std::clamp(chosen, 1, fallback_k);
                };
                const auto decode_begin = std::chrono::steady_clock::now();
                std::printf("position %zu top10", tokens.size());
                for (const auto &[id, logit] : top) std::printf(" %d:%.6f", id, logit);
                std::printf("\n");
                std::fflush(stdout);
                while (int(generated.size()) < generate) {
                    // The drafter attends over its checkpoint's 2048-position window
                    // and the block adds kBlock keys on top of the anchor, so
                    // the last safe anchor leaves room for the whole block;
                    // past it, fall back to plain greedy steps.
                    if (position + 1 + insignia::glm53::DFlash2Drafter::kBlock >
                        insignia::glm53::DFlash2Drafter::kMaxCtx) break;
                    int round_verify_k = v2_pick_k(verify_k);
                    const int draft_k = std::min(round_verify_k, generate - int(generated.size()));
                    // k=1 cannot expose an extra accepted token: truth0 is
                    // already known from the previous exact target step.
                    // Skip the drafter entirely and execute that exact scalar
                    // transition. Periodic k=2 probes still pass through the
                    // speculative path and can move the controller off k=1.
                    if (v2_k1_scalar && adaptive_k_mode == 2 && draft_k == 1) {
                        const uint64_t records_before = runner.expert_records_read();
                        const auto scalar_begin = std::chrono::steady_clock::now();
                        generated.push_back(truth0);
                        double scalar_ms = 0.0;
                        if (int(generated.size()) < generate) {
                            top = runner.step(truth0, position + 1, layers, true);
                            scalar_ms = 1000.0 * std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - scalar_begin).count();
                            v2_fhat_ms = 0.9 * v2_fhat_ms + 0.1 * scalar_ms;
                            root = truth0;
                            truth0 = top.front().first;
                            ++position;
                        }
                        ++rounds;
                        ++v2_scalar_rounds;
                        ++accept_hist[0];
                        committed_total += 1;
                        if (v2_costtrace)
                            std::printf("costtrace,%d,0,1,0.000,%.3f,0,%.0f,%.4f,%.3f,%d\n",
                                        rounds, scalar_ms,
                                        double(runner.expert_records_read() - records_before),
                                        v2_qhat[1], v2_bhat, v2_kstar);
                        std::fflush(stdout);
                        continue;
                    }
                    if (std::getenv("INSIGNIA_GLM53_DF_DEBUG"))
                        std::fprintf(stderr, "df round %d: anchor %d pos %d truth0 %d\n",
                                     rounds, root, position, truth0);
                    const auto draft_begin = std::chrono::steady_clock::now();
                    const std::vector<int> candidates = runner.df_draft(root, position);
                    const double draft_ms = 1000.0 * std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - draft_begin).count();
                    draft_total += draft_ms / 1000.0;
                    if (adaptive_k_mode == 2)
                        v2_dhat_ms = 0.9 * v2_dhat_ms + 0.1 * draft_ms;

                    // d1 is checked against the exact target argmax carried
                    // from the preceding committed token. A mismatch proves
                    // this round accepts zero drafts; running the 7-token
                    // target verifier cannot change that verdict.
                    if (candidates[0] != truth0) {
                        ++empty_rounds;
                        ++accept_hist[0];
                        generated.push_back(truth0);
                        ++rounds;
                        committed_total += 1;
                        // Empty rounds are matched=0 samples; skipping the
                        // EMA update leaves k inflated through empty streaks,
                        // widening verify unions exactly when acceptance is
                        // worst.
                        accept_ema = accept_ema_init ? 0.75 * accept_ema : 0.0;
                        accept_ema_init = true;
                        double fallback_ms = 0.0;
                        if (int(generated.size()) < generate) {
                            const auto fallback_begin = std::chrono::steady_clock::now();
                            top = runner.step(truth0, position + 1, layers, true);
                            fallback_ms = 1000.0 * std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - fallback_begin).count();
                            fallback_total += fallback_ms / 1000.0;
                            if (adaptive_k_mode == 2) {
                                v2_fhat_ms = 0.9 * v2_fhat_ms + 0.1 * fallback_ms;
                                v2_qhat[1] = 0.9 * v2_qhat[1];  // reject at position 1
                            }
                            root = truth0;
                            truth0 = top.front().first;
                            ++position;
                        }
                        if (v2_costtrace && adaptive_k_mode == 2)
                            std::printf("costtrace,%d,%d,0,%.3f,0,%.3f,0,%.4f,%.3f,%d\n",
                                        rounds, draft_k, draft_ms, fallback_ms,
                                        v2_qhat[1], v2_bhat, v2_kstar);
                        std::fflush(stdout);
                        continue;
                    }

                    const auto verify_begin = std::chrono::steady_clock::now();
                    const uint64_t verify_records_before = runner.expert_records_read();
                    runner.begin_verify_epoch();
                    std::vector<int> arg(static_cast<size_t>(draft_k), 0);
                    int matched = 1;
                    bool df_seq_verify =
                        df_verify_mode_env == 1 ||
                        (df_verify_mode_env == 0 && accept_ema_init &&
                         accept_ema < 0.70 * draft_k);
                    if (df_seq_verify) {
                        // Row-sequential verify: stop forwarding as soon as
                        // acceptance fails; rejected-tail experts are never
                        // read, and the recurrent state always stands at the
                        // accepted boundary so no rollback is ever needed.
                        runner.verify_may_rollback_ = false;
                        for (int r = 0; r < draft_k; ++r) {
                            runner.capture_offset_ = r;
                            arg[size_t(r)] = runner.verify_token(candidates[size_t(r)],
                                                                 position + 1 + r);
                            if (r + 1 >= draft_k) { matched = draft_k; break; }
                            if (arg[size_t(r)] != candidates[size_t(r + 1)]) {
                                matched = r + 1;
                                break;
                            }
                        }
                        runner.capture_offset_ = 0;
                        runner.verify_may_rollback_ = true;
                    } else {
                        const std::vector<int> verify_candidates(
                            candidates.begin(), candidates.begin() + draft_k);
                        std::pair<int, std::vector<int>> verdict =
                            runner.verify_round(verify_candidates, position + 1);
                        if (runner.dflash_retry_needed(draft_k)) {
                            runner.rollback_kda(0, position + 1);
                            runner.begin_dflash_exact_retry();
                            verdict = runner.verify_round(verify_candidates, position + 1);
                            runner.end_dflash_exact_retry();
                        }
                        arg = verdict.second;
                        while (matched < draft_k && arg[size_t(matched - 1)] == candidates[size_t(matched)])
                            ++matched;
                    }
                    const double verify_ms = 1000.0 * std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - verify_begin).count();
                    verify_total += verify_ms / 1000.0;
                    ++verified_rounds;
                    if (adaptive_k_mode == 2) {
                        // Censoring-correct hazard updates: positions 1..matched
                        // passed; position matched+1 rejected (observable only
                        // when the round drafted that wide).
                        for (int j = 1; j <= matched && j <= 8; ++j)
                            v2_qhat[j] = 0.9 * v2_qhat[j] + 0.1;
                        if (matched < draft_k && matched + 1 <= 8)
                            v2_qhat[matched + 1] = 0.9 * v2_qhat[matched + 1];
                        // U3: online per-record cost. Decayed least squares of
                        // verify wall on demand record reads; clamped to the
                        // measured regime band (0.4-2.5 ms/record).
                        const double records =
                            double(runner.expert_records_read() - verify_records_before);
                        constexpr double lambda = 0.95;
                        v2_sn = lambda * v2_sn + 1.0;
                        v2_sx = lambda * v2_sx + records;
                        v2_sy = lambda * v2_sy + verify_ms;
                        v2_sxx = lambda * v2_sxx + records * records;
                        v2_sxy = lambda * v2_sxy + records * verify_ms;
                        if (v2_sn > 20.0 && v2_sxx > v2_sx * v2_sx / v2_sn + 1.0) {
                            const double slope =
                                (v2_sxy - v2_sx * v2_sy / v2_sn) /
                                (v2_sxx - v2_sx * v2_sx / v2_sn);
                            if (slope > 0.0)
                                v2_bhat = std::clamp(slope, 0.4, 2.5);
                        }
                        if (v2_costtrace)
                            std::printf("costtrace,%d,%d,%d,%.3f,%.3f,0,%.0f,%.4f,%.3f,%d\n",
                                        rounds, draft_k, matched, draft_ms, verify_ms,
                                        records, v2_qhat[1], v2_bhat, v2_kstar);
                    }
                    accept_ema = accept_ema_init ? 0.75 * accept_ema + 0.25 * matched
                                                 : double(matched);
                    accept_ema_init = true;
                    ++accept_hist[size_t(matched)];
                    for (int i = 0; i < matched && int(generated.size()) < generate; ++i)
                        generated.push_back(candidates[size_t(i)]);
                    runner.adopt_df_prior_logits(df_seq_verify ? 0 : matched - 1);
                    if (matched < draft_k && !df_seq_verify)
                        runner.rollback_kda(matched, position + 1);
                    // The verify captures cover the candidates; append the
                    // accepted prefix (step() commits its own token).
                    runner.adopt_anchor_routing(df_seq_verify ? 0 : matched - 1);
                    runner.df_commit(matched, position + 1);
                    // SLRU insert-then-demote: batch verify read the whole
                    // union; experts that served only the rejected tail get
                    // pushed to the cold end of the host tier.
                    if (!df_seq_verify)
                        runner.demote_rejected_routing(matched, draft_k);
                    root = candidates[size_t(matched - 1)];
                    truth0 = arg[size_t(matched - 1)];
                    position += matched;
                    committed_total += matched;
                    accepted_draft_total += matched;
                    ++rounds;
                    std::fflush(stdout);
                }
                // Drafter window exhausted (position >= 2040): plain greedy
                // steps for the remainder, same shape as the empty-round
                // fallback.
                while (int(generated.size()) < generate) {
                    generated.push_back(truth0);
                    committed_total += 1;
                    if (int(generated.size()) >= generate) break;
                    top = runner.step(truth0, position + 1, layers, true);
                    truth0 = top.front().first;
                    ++position;
                }
                const double decode_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - decode_begin).count();
                std::printf("greedy IDs");
                for (int id : generated) std::printf(" %d", id);
                std::printf("\n%zu-token prompt %.3f s; %d greedy tokens in %d DFLASH2-k%d rounds "
                            "(%.2f accepted/round, %d empty; %.1f ms/token; "
                            "draft %.1f ms/spec round + verify %.1f ms/verified round (%d verified), "
                            "fallback %.1f ms)\n",
                            tokens.size(), elapsed, generate, rounds, verify_k,
                            accepted_draft_total / std::max(1, rounds), empty_rounds,
                            1000.0 * decode_seconds / generate,
                            1000.0 * draft_total /
                                std::max(1, rounds - v2_scalar_rounds),
                            1000.0 * verify_total / std::max(1, verified_rounds),
                            verified_rounds,
                            1000.0 * fallback_total);
                std::printf("  accepted histogram");
                for (int count = 0; count <= verify_k; ++count)
                    if (accept_hist[size_t(count)])
                        std::printf(" %d:%d", count, accept_hist[size_t(count)]);
                std::printf("\n");
                std::printf("  renewal rewards %.2f accepted-draft/round, "
                            "%.2f committed/round; %.3f accepted-draft/s, "
                            "%.3f committed/s\n",
                            accepted_draft_total / std::max(1, rounds),
                            committed_total / std::max(1, rounds),
                            accepted_draft_total / std::max(1.0e-9, decode_seconds),
                            committed_total / std::max(1.0e-9, decode_seconds));
                runner.report_cache_selector();
                if (v2_scalar_rounds)
                    std::printf("  adaptive controller bypassed speculation for %d k=1 rounds\n",
                                v2_scalar_rounds);
            } else if (mtp_k >= 2 && generate > 1) {
                // Faithful MTP flow (vLLM deepseek_mtp semantics): each round
                // drafts K tokens from (embed(root), h(root)) chained through
                // the draft's own recycled hidden states, then ONE target
                // verify forward checks them. d1 is checked host-side against
                // the target argmax carried from the previous round (truth0);
                // rows 0..K-2 check d2..dK. The committed sequence is
                // therefore identical to plain greedy decode.
                int position = int(tokens.size()) - 1;
                int root = tokens.back();
                const float *root_hidden = runner.chain_root_hidden();
                int truth0 = top.front().first;
                double accept_total = 0.0;
                int rounds = 0, empty_rounds = 0;
                const auto decode_begin = std::chrono::steady_clock::now();
                // Same anchor line the plain greedy loop prints for token 0.
                std::printf("position %zu top10", tokens.size());
                for (const auto &[id, logit] : top) std::printf(" %d:%.6f", id, logit);
                std::printf("\n");
                std::fflush(stdout);
                while (int(generated.size()) < generate) {
                    std::vector<int> candidates;
                    candidates.reserve(size_t(mtp_k));
                    int chain_token = root;
                    const float *chain_hidden = root_hidden;
                    int chain_position = position;
                    for (int i = 0; i < mtp_k; ++i) {
                        const int draft = runner.mtp_forward(chain_token, chain_hidden,
                                                             chain_position);
                        candidates.push_back(draft);
                        chain_token = draft;
                        chain_hidden = runner.mtp_recycle();
                        ++chain_position;
                    }
                    const std::pair<int, std::vector<int>> verdict =
                        runner.verify_round(candidates, position + 1);
                    const std::vector<int> &arg = verdict.second;
                    int matched = 0;
                    if (candidates[0] == truth0) {
                        matched = 1;
                        while (matched < mtp_k && arg[size_t(matched - 1)] == candidates[size_t(matched)])
                            ++matched;
                    }
                    for (int i = 0; i < matched && int(generated.size()) < generate; ++i)
                        generated.push_back(candidates[size_t(i)]);
                    if (matched < mtp_k)
                        runner.rollback_kda(matched, position + 1);
                    if (matched > 0) {
                        // Row matched-1's input is c_matched (the last accepted
                        // token, and the next chain root); its mean hidden and
                        // its argmax seed the next round.
                        root = candidates[size_t(matched - 1)];
                        root_hidden = runner.verify_row_hidden(matched - 1);
                        truth0 = arg[size_t(matched - 1)];
                        position += matched;
                        accept_total += matched;
                    } else {
                        // Empty round: commit the known-correct token with one
                        // exact target step; its hidden state seeds the chain.
                        ++empty_rounds;
                        if (int(generated.size()) < generate)
                            generated.push_back(truth0);
                        top = runner.step(truth0, position + 1, layers, true);
                        root = truth0;
                        root_hidden = runner.chain_root_hidden();
                        truth0 = top.front().first;
                        ++position;
                        accept_total += 1;
                    }
                    ++rounds;
                    std::fflush(stdout);
                }
                const double decode_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - decode_begin).count();
                std::printf("greedy IDs");
                for (int id : generated) std::printf(" %d", id);
                std::printf("\n%zu-token prompt %.3f s; %d greedy tokens in %d MTP rounds "
                            "(%.2f accepted/round, %d empty; %.1f ms/token)\n",
                            tokens.size(), elapsed, generate, rounds,
                            accept_total / rounds, empty_rounds,
                            1000.0 * decode_seconds / generate);
            } else {
                for (int index = 0; index < generate; ++index) {
                    std::printf("position %zu top10", tokens.size() + size_t(index));
                    for (const auto &[id, logit] : top) std::printf(" %d:%.6f", id, logit);
                    const int next = top.front().first;
                    generated.push_back(next);
                    std::printf("\n");
                    if (index + 1 < generate)
                        top = runner.step(next, int(tokens.size()) + index, layers, true);
                }
                const double total = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - begin).count();
                std::printf("greedy IDs");
                for (int id : generated) std::printf(" %d", id);
                std::printf("\n%zu-token prompt %.3f s; %d greedy token%s total %.3f s\n",
                            tokens.size(), elapsed, generate, generate == 1 ? "" : "s", total);
            }
        } else {
            std::printf("%d-layer, %zu-token smoke pass %.3f s\n", layers, tokens.size(), elapsed);
        }
    } catch (const std::exception &error) {
        std::fprintf(stderr, "glm53-generate: %s\n", error.what());
        return 1;
    }
    return 0;
}
