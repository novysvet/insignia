#include "insignia_bf16.cuh"
#include "insignia_glm53.cuh"
#include "insignia_glm53_dflash2.cuh"
#include "insignia_glm53_fp8.cuh"
#include "insignia_glm53_index.hpp"
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

using insignia::glm53::ShardedIndex;
using insignia::glm53::TensorLocation;
using insignia::glm53::TensorType;
using insignia::glm53::Q8Index;
using insignia::glm53::Q8TensorLocation;
using insignia::glm53::Cache8Format;
constexpr int kStreams = insignia::glm53::kHyperStreams;
static int kMaxContext() { static const int limit = [] { const char *v = std::getenv("INSIGNIA_GLM53_CONTEXT"); return std::clamp(v ? std::atoi(v) : 8192, 512, 262144); }(); return limit; }
constexpr int kLegacyMlaContext = 256;

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
    static constexpr size_t kWindowBytes =
        (kPayloadCapacity + 2 * kAlignment - 2) & ~(kAlignment - 1);
    static constexpr uint32_t kNoKey = 0xffffffffu;

    explicit ExpertStager(ShardedIndex &model, uint64_t host_cache_bytes) : model_(model) {
        // A full decode token needs 336 records; default the tier just above
        // that and let the environment shrink it on smaller hosts.
        window_count_ = int(std::clamp<uint64_t>(host_cache_bytes / kWindowBytes, 64, 4096));
        size_t attempt = size_t(window_count_);
        while (attempt >= 64) {
            void *block = nullptr;
            const cudaError_t status = cudaHostAlloc(&block, attempt * kWindowBytes + kAlignment - 1,
                                                     cudaHostAllocDefault);
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
        check(cudaMalloc(&device_, kPayloadCapacity), "cudaMalloc expert record");
        overlap_reads_ = std::getenv("INSIGNIA_GLM53_EAGER_EXPERT_JOIN") == nullptr;
        l2_mode_ = std::getenv("INSIGNIA_GLM53_PAGECACHE_L2") != nullptr;
        if (const char *filter = std::getenv("INSIGNIA_GLM53_ADMIT_N"))
            admit_threshold_ = std::max(1, std::atoi(filter));
        else if (const char *filter = std::getenv("INSIGNIA_GLM53_ADMIT"))
            admit_threshold_ = std::atoi(filter) != 0 ? 2 : 1;
        if (const char *budget = std::getenv("INSIGNIA_GLM53_EXPERT_VRAM_MB"))
            vram_budget_mb_ = std::max(0, std::atoi(budget));
        // F3 residency ordering: consult the VRAM expert tier before starting
        // an NVMe read. The legacy order only noticed device residency at
        // upload time, after the bytes had already been re-read from disk.
        if (const char *f3 = std::getenv("INSIGNIA_GLM53_F3"))
            f3_device_consult_ = std::atoi(f3) != 0;
        if (const char *path = std::getenv("INSIGNIA_GLM53_PACKED_EXPERTS")) {
            const char *gpu = std::getenv("INSIGNIA_GLM53_PACKED_GPU");
            packed_gpu_scales_ = !gpu || std::atoi(gpu) != 0;
            if (const char *merge = std::getenv("INSIGNIA_GLM53_PACKED_V2"))
                packed_merge_h2d_ = std::atoi(merge) != 0;
            if (const char *kernel = std::getenv("INSIGNIA_GLM53_PACKED_KERNEL"))
                packed_kernel_v2_ = std::atoi(kernel) == 2;
            open_packed_experts(path);
            if (packed_gpu_scales_)
                check(cudaMalloc(&packed_scale_device_, kPackedDeviceCapacity),
                      "cudaMalloc packed scale transport");
        }
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
            staged.push_back(window);
            if (taken < per_layer_device) pinned_device_keys_.insert(key);
            ++taken;
            ++taken_device;
            ++pinned_count;
        }
        std::fclose(file);
        for (int window : staged) wait_and_consume_error(window);
        for (int window : staged)
            if (windows_[size_t(window)].error)
                windows_[size_t(window)].pinned = false;
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
                batch_window_[slot] = resident->second;
                if (window_done(resident->second)) {
                    // Host-tier hit: the record is already pinned in RAM and
                    // only owes the 13.5 MiB H2D copy.
                    batch_cached_[slot] = true;
                    ++state.hits;
                    ++hits;
                    if (!state.demand) ++prefetch_useful_;
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
        batch_demand_count_ = count - hits - f3_rescued;
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
                device_slot = take_device_slot(state.layer);
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
        active_ = state.layout;
        active_globals_ = state.globals;
        if (state.pinned || (batch_populate_[size_t(slot)] && batch_admit_[size_t(slot)])) {            // Admitted: the window stays resident in the host LRU; eviction
            // re-checks copy_done before the slot can be refilled. The pinned
            // window now owns the bytes, so drop the page-cache shadow.
            if (l2_mode_ && state.l2_shard >= 0)
                model_.evict_span_cache(uint16_t(state.l2_shard), state.l2_offset, state.l2_bytes);
            state.claimed = false;
            state.stamp = ++stamp_;
        } else {
            // Passed through: drop it from the resident index immediately so
            // later lookups re-read, and free the window once the async copy
            // has drained (reap_released polls the event).
            if (state.key != kNoKey) flight_index_.erase(state.key);
            state.key = kNoKey;
            state.releasing = true;
            releasing_.push_back(window);
            if (!state.demand) ++prefetch_wasted_;
        }
    }
    const uint8_t *down_weight() const { return active_device_ + active_.body[0]; }
    const uint8_t *gate_weight() const { return active_device_ + active_.body[1]; }
    const uint8_t *up_weight() const { return active_device_ + active_.body[2]; }
    const uint8_t *down_scale() const { return active_device_ + active_.scales[0]; }
    const uint8_t *gate_scale() const { return active_device_ + active_.scales[1]; }
    const uint8_t *up_scale() const { return active_device_ + active_.scales[2]; }
    float down_global(int) const { return active_globals_[0]; }
    float gate_global(int) const { return active_globals_[1]; }
    float up_global(int) const { return active_globals_[2]; }
    double io_seconds() const { return io_seconds_; }
    uint64_t io_bytes() const { return io_bytes_; }
    uint64_t prefetch_bytes() const { return prefetch_bytes_; }
    uint64_t cache_hits() const { return cache_hits_; }
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
    bool packed_experts() const { return packed_fd_ >= 0; }
    bool packed_gpu_scales() const { return packed_gpu_scales_; }
    uint64_t packed_h2d_bytes() const { return packed_h2d_bytes_.load(); }
    uint64_t packed_h2d_records() const { return packed_h2d_records_.load(); }
    uint64_t packed_expanded_bytes() const { return packed_expanded_bytes_.load(); }
    double packed_expand_seconds() const {
        return packed_expand_nanoseconds_.load() * 1.0e-9;
    }
private:
    struct Layout {
        std::array<size_t, 3> body{};
        std::array<size_t, 3> scales{};
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
        bool demand = false, done = true, claimed = false, copy_issued = false;
        bool releasing = false, pinned = false;
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
        require(std::memcmp(header.magic, "IG53XPK1", 8) == 0 && header.version == 1,
                "bad packed expert sidecar header");
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
            packed_scratch_bytes_ = std::max(packed_scratch_bytes_, size_t(entry.padded_bytes));
            ++populated;
        }
        require(populated == header.records && packed_scratch_bytes_,
                "packed expert index record count mismatch");
#ifdef O_DIRECT
        packed_direct_fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
#endif
        const double fraction = header.source_bytes
            ? double(header.stored_bytes) / double(header.source_bytes) : 1.0;
        std::printf("packed experts: %zu records, %.3f GiB logical, %.2f%% smaller, %s + %s expand\n",
                    populated, header.stored_bytes / double(1ull << 30),
                    100.0 * (1.0 - fraction),
                    packed_direct_fd_ >= 0 ? "O_DIRECT" : "buffered I/O",
                    packed_gpu_scales_ ? "GPU/packed-H2D" : "AVX2/expanded-H2D");
    }
    const PackedExpertIndexEntry &packed_entry(int layer, int expert) const {
        const size_t index = size_t(layer) * model_.experts() + size_t(expert);
        require(index < packed_entries_.size() && packed_entries_[index].offset,
                "expert is missing from packed sidecar");
        return packed_entries_[index];
    }
    uint64_t source_bytes(int layer, int expert) const {
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
    void stage_packed(Layout &layout, std::array<float, 3> &globals, uint8_t *&payload,
                      uint8_t *window, int layer, int expert, void *scratch) {
        if (packed_gpu_scales_)
            stage_packed_gpu(layout, globals, payload, window, layer, expert, scratch);
        else
            stage_packed_cpu(layout, globals, payload, window, layer, expert, scratch);
    }
    void enqueue_record_copy(WindowState &state, uint8_t *destination) {
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
        const size_t stride = (kPayloadCapacity + kAlignment - 1) & ~(kAlignment - 1);
        size_t attempt = budget / stride;
        while (attempt > 1) {
            if (cudaMalloc(&device_arena_, attempt * stride) == cudaSuccess) break;
            cudaGetLastError();  // clear the sticky error and retry smaller
            attempt /= 2;
        }
        if (attempt < 2) {
            // A one-slot arena cannot recycle (the active slot is never its
            // own victim); stay on the legacy single-scratch path instead.
            if (device_arena_) {
                cudaFree(device_arena_);
                device_arena_ = nullptr;
            }
            return;
        }
        device_stride_ = stride;
        device_slot_count_ = int(attempt);
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
    int take_device_slot(int layer) {
        const int segments = std::max(1, std::min(46, device_slot_count_));
        const int begin = size_t(layer) % size_t(segments) * device_slot_count_ / segments;
        const int end = size_t(layer) % size_t(segments) + 1 == segments
                            ? device_slot_count_
                            : (size_t(layer) % size_t(segments) + 1) * device_slot_count_ / segments;
        int victim = -1;
        for (int index = begin; index < end; ++index) {
            if (index == active_device_slot_) continue;  // fence not recorded yet
            if (device_slot_pinned_[size_t(index)]) continue;
            if (device_slot_keys_[size_t(index)] == kNoKey) return index;
            if (victim < 0 ||
                device_slot_stamps_[size_t(index)] < device_slot_stamps_[size_t(victim)])
                victim = index;
        }
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
        flight_index_.emplace(key, window);
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
        free_windows_.push_back(window);
    }
    // Drive of a routing target: 0 = main store, 1 = ALT_SHARD_DIR (second
    // physical drive). Cached per (layer,expert); the lookup walks 9 index
    // entries once per distinct key.
    int drive_of(uint32_t key, int layer, int expert) {
        if (packed_fd_ >= 0) return 0;
        std::lock_guard<std::mutex> lock(pool_mutex_);
        const auto found = drive_cache_.find(key);
        if (found != drive_cache_.end()) return found->second;
        const ExpertLocations tensors = locate_expert(model_, layer, expert);
        const int drive = model_.shard_is_alt(tensors.body[0]->shard) ? 1 : 0;
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
        for (int drive = 0; drive < 2; ++drive) {
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
        const WindowState &state = windows_[size_t(window)];
        const int drive = drive_of(state.key, state.layer, state.expert);
        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            (demand ? demand_queue_[drive] : prefetch_queue_[drive]).push_back(window);
        }
        // notify_one can land on a wrong-drive worker whose predicate is
        // false; with partitioned queues it must be notify_all or the last
        // record in a drive's queue waits forever.
        pool_cv_.notify_all();
    }
    void read_window(int window, void *packed_scratch) {
        WindowState &state = windows_[size_t(window)];
        try {
            if (packed_fd_ >= 0) {
                stage_packed(state.layout, state.globals, state.payload,
                             host_ + size_t(window) * kWindowBytes,
                             state.layer, state.expert, packed_scratch);
            } else {
                const ExpertLocations tensors = locate_expert(model_, state.layer, state.expert);
                stage(state.layout, state.globals, state.payload, tensors,
                      host_ + size_t(window) * kWindowBytes, state);
            }
            if (!state.demand) prefetch_bytes_ += state.source_bytes;
        } catch (...) {
            state.error = std::current_exception();
        }
    }
    void stage(Layout &layout, std::array<float, 3> &globals, uint8_t *&payload,
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
                delta = model_.read_span_cached_window(
                    first.shard, first.offset, layout.bytes, window, kWindowBytes);
                wstate.l2_shard = first.shard;
                wstate.l2_offset = first.offset;
                wstate.l2_bytes = layout.bytes;
            } else {
                delta = model_.read_span_direct_window(
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
            model_.read_span_direct(body.shard, body.offset, body.bytes + scales.bytes,
                                    window + cursor);
            cursor += size_t(body.bytes + scales.bytes);
            model_.read(*tensors.globals[projection], globals.data() + projection);
        }
        layout.bytes = cursor;
    }

    ShardedIndex &model_;
    int packed_fd_ = -1, packed_direct_fd_ = -1;
    std::vector<PackedExpertIndexEntry> packed_entries_;
    size_t packed_scratch_bytes_ = 0;
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
    cudaStream_t copy_stream_ = nullptr;
    bool stop_ = false, packed_gpu_scales_ = false;
    bool packed_merge_h2d_ = false, packed_kernel_v2_ = false;
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
    double read_wait_seconds_ = 0.0;
    // VRAM tier: an arena of per-record device slots acts as an LRU above the
    // pinned host windows. A device hit skips both the NVMe read and the PCIe
    // H2D entirely; a miss streams into its own slot, so copies for different
    // records pipeline instead of serializing through one shared scratch.
    uint8_t *device_arena_ = nullptr;
    size_t device_stride_ = 0;
    int device_slot_count_ = 0;
    int active_device_slot_ = -1;
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
};

class Q8Stager {
public:
    static constexpr size_t kWeightCapacity = 128ull << 20;
    static constexpr size_t kScaleCapacity = 4ull << 20;

    explicit Q8Stager(Q8Index &index) : index_(index) {
        check(cudaHostAlloc(&host_weights_, kWeightCapacity, cudaHostAllocDefault),
              "cudaHostAlloc Q8 weights");
        check(cudaHostAlloc(&host_scales_, kScaleCapacity, cudaHostAllocDefault),
              "cudaHostAlloc Q8 scales");
        check(cudaMalloc(&stream_weights_, kWeightCapacity), "cudaMalloc Q8 weights");
        check(cudaMalloc(&stream_scales_, kScaleCapacity), "cudaMalloc Q8 scales");
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
        : model_(index, root), stager_(model_),
          q8_index_(open_q8(q8_prefix)),
          q8_stager_(q8_index_ ? std::make_unique<Q8Stager>(*q8_index_) : nullptr),
          logits_(0), finite_(1) {
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
            if (nvfp4_experts_) {
                // Host-RAM LRU over whole expert records: one decode token
                // touches 42 x 8 = 336 records, so the tier must exceed that
                // to hit at all. Default 5 GiB pinned (~370 records) inside
                // the 14 GiB WSL VM; VRAM keeps only the 13.5 MiB H2D target.
                uint64_t host_cache = 32768ull << 20;
                if (const char *budget = std::getenv("INSIGNIA_GLM53_EXPERT_CACHE_MB"))
                    host_cache = uint64_t(std::max(0, std::atoi(budget))) << 20;
                expert_stager_ = std::make_unique<ExpertStager>(model_, host_cache);
            }
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
        kda_states_.reset(size_t(layer_count) * kda_width_ * kda_head_dim_);
        conv_history_.reset(size_t(layer_count) * 9 * kda_width_);
        mla_legacy_ = std::getenv("INSIGNIA_GLM53_MLA_LEGACY") != nullptr;
        kv_fp8_ = !std::getenv("INSIGNIA_GLM53_KV_FP8") ||
                  std::atoi(std::getenv("INSIGNIA_GLM53_KV_FP8")) != 0;
        if (mla_legacy_) {
            const size_t expanded_stride = size_t(kLegacyMlaContext) * q_b_rows_;
            mla_keys_.reset(size_t(mla_layers_) * expanded_stride);
            mla_values_.reset(size_t(mla_layers_) * expanded_stride);
        } else {
            const size_t latent_stride = size_t(kMaxContext()) * kv_a_rows_;
            const size_t expanded_stride = size_t(kLegacyMlaContext) * q_b_rows_;
            mla_latent_u8_.reset(kv_fp8_ ? size_t(mla_layers_) * latent_stride : 0);
            mla_latent_f32_.reset(kv_fp8_ ? 0 : size_t(mla_layers_) * latent_stride);
            mla_latent_scale_.reset(size_t(mla_layers_) * kMaxContext() *
                                    insignia::glm53::kMlaLatentGroups);
            mla_keys_.reset(size_t(mla_layers_) * expanded_stride);
            mla_values_.reset(size_t(mla_layers_) * expanded_stride);
            mla_partial_.reset(size_t(mla_heads_) * ((kMaxContext() + 511) / 512) *
                               (size_t(kv_a_rows_) + 2));
            absorb_per_layer_ = size_t(mla_heads_) * mla_head_dim_ * kv_a_rows_;
            w_uk_.reset(absorb_per_layer_ * mla_layers_);
            w_uv_.reset(absorb_per_layer_ * mla_layers_);
            extract_mla_absorb();
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
            kda_snap_.reset(kda_states_.size());
            conv_snap_.reset(conv_history_.size());
            // Per (kda layer, token): pre-conv q,k,v + raw gate + raw beta, the
            // exact inputs the recurrence replay needs after a rejected draft.
            kda_arch_.reset(size_t(kda_layers_) * kMaxVerify * (4 * size_t(kda_width_) + kda_heads_));
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
                    nvfp4_experts_ ? ", NVFP4 experts" : ", BF16 experts");
        if (expert_stager_ && expert_stager_->cache_slots())
            std::printf("expert cache: %d pinned host-RAM NVFP4 records (%.1f MiB)\n",
                        expert_stager_->cache_slots(),
                        expert_stager_->cache_slots() * ExpertStager::kWindowBytes / double(1 << 20));
        else if (expert_stager_)
            std::printf("expert cache: disabled (no pinned host records)\n");
        load_cct();
        if (stager_.resident_bytes() || (q8_stager_ && q8_stager_->resident_bytes()))
            std::printf("resident: %llu MiB BF16 + %llu MiB %s pinned in VRAM\n",
                        (unsigned long long)(stager_.resident_bytes() >> 20),
                        (unsigned long long)(q8_stager_ ? q8_stager_->resident_bytes() : 0) >> 20,
                        q8_index_->format() == Cache8Format::q8 ? "Q8" : "FP8");
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
    void prefill(const std::vector<int> &tokens, int position_base, bool capture = false);
    // MTP speculative decoding surface.
    int mtp_k() const { return mtp_draft_total_; }
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
    // Verifies candidate tokens at position_base+1.. against the target and
    // returns (accepted_count, argmax rows) — see the caller in main().
    int verify_token(int token, int position);
    std::pair<int, std::vector<int>> verify_round(const std::vector<int> &candidates,
                                                  int position_base);
    int mtp_forward(int token, const float *hidden_in, int position);
    void mtp_moe(const float *input, float *output);
    void rollback_kda(int accepted, int position_base);
    // Row-sequential verify controls: snapshot suppression + drafter capture slot.
    bool verify_may_rollback_ = true;
    int capture_offset_ = 0;
    void set_last_avg(const float *device_row) {
        check(cudaMemcpyAsync(last_avg_.get(), device_row, size_t(hidden_) * sizeof(float),
                              cudaMemcpyDeviceToDevice), "copy pending hidden");
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
    void extract_mla_absorb();
    void compute_mlp(std::string_view stem, const float *input, float *output, int intermediate);
    void dense_mlp(std::string_view stem, const float *input, float *output, int intermediate);
    void sparse_moe(int layer, const float *input, float *output);
    void mhc_multi(std::string_view stem, const float *streams, float *collapsed, int tokens);
    void kda_multi(int layer, const float *input, float *output, int tokens, int position_base);
    void mla_multi(int layer, const float *input, float *output, int tokens, int position_base);
    void mlp_multi(std::string_view stem, const float *input, float *output, int tokens, int intermediate);
    void moe_multi(int layer, const float *input, float *output, int tokens);
    const float *device_f32(std::string_view name);
    const std::vector<float> &host_f32(std::string_view name);
    void archive_kda_rows(int layer, const float *src, int rows, int slot);
    double expert_io_seconds() const { return expert_stager_ ? expert_stager_->io_seconds() : 0.0; }
    uint64_t expert_io_bytes() const { return expert_stager_ ? expert_stager_->io_bytes() : 0; }

    ShardedIndex model_;
    TensorStager stager_;
    std::unique_ptr<ExpertStager> expert_stager_;
    std::vector<std::array<int, 8>> prev_routing_;
    std::vector<std::array<int, 64>> row_routing_;  // [layer][row*8+slot], last multi-row chunk
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
    // names the file, INSIGNIA_GLM53_SEAM_LAYER the layer to trace (default 0).
    // Record: i32[4] {token_index, layer, tag, count} + count f32 values.
    // 1 attn-norm  2 attn-out  3 streams after attn mix  4 ffn-norm
    // 5 ffn-out    6 streams after ffn mix.
    FILE *seam_dump_ = nullptr;
    bool seam_dump_probed_ = false;
    int seam_layer_ = 0;
    std::vector<float> seam_host_;
    bool prefetch_on_ = true, deep_checks_ = false, trace_layers_ = false;
    bool early_route_on_ = false, early_route_prefetch_ = false;
    int early_route_prefetch_n_ = 8;
    uint64_t early_route_hits_ = 0, early_route_total_ = 0;
    bool early_multi_route_on_ = false, early_multi_prefetch_ = false;
    int early_multi_n_ = 4, early_multi_max_ = 64;
    uint64_t early_multi_batch_ = 0, early_multi_hits_ = 0;
    uint64_t early_multi_predicted_ = 0, early_multi_actual_ = 0;
    uint64_t early_multi_hints_ = 0, early_multi_started_ = 0;
    std::vector<std::vector<std::array<int, 8>>> early_multi_rows_;

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
        if (!seam_dump_ || layer != seam_layer_) return;
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
    float *df_logits_host_ = nullptr, *df_hp_host_ = nullptr;
    std::unique_ptr<Q8Index> q8_index_;
    std::unique_ptr<Q8Stager> q8_stager_;
    int hidden_ = 0, kda_width_ = 0, kda_heads_ = 0, kda_head_dim_ = 0, f_a_rows_ = 0;
    int q_a_rows_ = 0, q_b_rows_ = 0, kv_a_rows_ = 0, kv_b_rows_ = 0;
    int mla_heads_ = 0, mla_head_dim_ = 0, mla_layers_ = 0, kda_layers_ = 0;
    int moe_experts_ = 0, moe_topk_ = 0, moe_intermediate_ = 0;
    int dense_intermediate_ = 0, shared_intermediate_ = 0;
    bool nvfp4_experts_ = false;
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
    DeviceBuffer<uint8_t> mla_latent_u8_;
    DeviceBuffer<float> mla_latent_f32_, mla_latent_scale_, mla_partial_, w_uk_, w_uv_;
    DeviceBuffer<float> mla_keys_, mla_values_;
    bool kv_fp8_ = true, mla_legacy_ = false;
    size_t absorb_per_layer_ = 0;
    const uint32_t *chunk_bf16_weights_ = nullptr;
    DeviceBuffer<float> c_stream_a_, c_stream_b_, c_collapsed_, c_normalized_, c_attn_, c_ffn_;
    DeviceBuffer<float> c_routed_, c_expert_out_, c_q_, c_k_, c_v_, c_gate_, c_core_, c_proj_, c_small_;
    DeviceBuffer<float> c_kv_, c_mlaq_, c_mlao_, c_gateu_, c_up_, c_act_, c_router_;
    DeviceBuffer<float> c_post_, c_comb_, c_beta_;
    DeviceBuffer<uint8_t> nv_workspace_4096_, nv_workspace_2048_, q8_workspace_;
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
    require(weight.type == TensorType::bf16 && weight.shape.size() == 2 &&
            weight.shape[0] == uint32_t(rows) && weight.shape[1] == uint32_t(cols),
            "wrong BF16 linear geometry for " + std::string(name));
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
    require(weight.type == TensorType::bf16 && weight.shape.size() == 2 &&
            weight.shape[0] == uint32_t(rows) && weight.shape[1] == uint32_t(cols),
            "wrong multi-token BF16 linear geometry for " + std::string(name));
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
    require(weight.type == TensorType::bf16 && weight.shape.size() == 2 &&
            weight.shape[1] == uint32_t(cols) && row + rows <= weight.shape[0],
            "wrong chunked BF16 linear geometry");
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
    float *history = conv_history_.get() + size_t(layer) * 9 * kda_width_;
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

    float *state = kda_states_.get() + size_t(layer) * kda_width_ * kda_head_dim_;
    check(insignia::glm53::kda_decode(state, q_, k_, v_, gate_8192_, beta_, core_,
          kda_heads_, kda_head_dim_), "KDA recurrence");
    linear(stem + "g_a_proj.weight", input, small_a_, f_a_rows_, hidden_);
    linear(stem + "g_b_proj.weight", small_a_, gate_8192_, kda_width_, f_a_rows_);
    const uint16_t *norm = reinterpret_cast<const uint16_t *>(stager_.load(stem + "o_norm.weight"));
    kda_output_kernel<<<kda_heads_, kda_head_dim_>>>(core_, gate_8192_, norm, projected_8192_);
    check(cudaGetLastError(), "KDA output norm launch");
    linear(stem + "o_proj.weight", projected_8192_, output, hidden_, kda_width_);
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
        require(location.bytes == size_t(kv_b_rows_) * latent * sizeof(uint16_t),
                "unexpected kv_b_proj tensor size");
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
        linear(stem + "kv_b_proj.weight", small_b_, kv_, kv_b_rows_, kv_a_rows_);
        check(insignia::glm53::mla_store_latent(
              small_b_.get(), cache_u8, cache_scale, cache_f32, 1, position,
              kv_a_rows_), "MLA latent shadow store");
        const size_t expanded_stride = size_t(kLegacyMlaContext) * q_b_rows_;
        check(insignia::glm53::mla_decode(
              mla_query_.get(), kv_.get(),
              mla_keys_.get() + size_t(slot) * expanded_stride,
              mla_values_.get() + size_t(slot) * expanded_stride,
              mla_output_, position, mla_heads_, mla_head_dim_),
              "exact MLA prefix attention");
    } else {
        check(insignia::glm53::mla_decode_latent(
              mla_query_.get(), small_b_.get(), nullptr, cache_u8, cache_scale,
              cache_f32, nullptr,
              w_uk_.get() + size_t(slot) * absorb_per_layer_,
              w_uv_.get() + size_t(slot) * absorb_per_layer_,
              mla_partial_.get(), mla_output_, position,
              mla_heads_, mla_head_dim_, kv_a_rows_), "MLA attention");
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
    if (!nvfp4_experts_) {
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
        check(insignia::glm53::nvfp4_quantize_activation(input, hidden_, nv_workspace_4096_),
              "quantize expert input");
        // The shared expert is dense-resident FP8; running it first lets its
        // GEMVs hide under the routed records' disk reads. Routed downs land
        // in a scratch buffer so `output` keeps the shared result.
        compute_mlp(stem + "shared_experts.", input, output, shared_intermediate_);
        for (int slot = 0; slot < moe_topk_; ++slot) {
            const int expert = selected[slot];
            expert_stager_->upload(slot);
            check(insignia::glm53::nvfp4_gemv2_dp4a_quantized(
                expert_stager_->gate_weight(), expert_stager_->gate_scale(), expert_stager_->gate_global(slot),
                expert_stager_->up_weight(), expert_stager_->up_scale(), expert_stager_->up_global(slot),
                nv_workspace_4096_, gate_, up_, moe_intermediate_, hidden_), "routed expert gate/up");
                check(insignia::glm53::quantize_swiglu_activation(gate_, up_, moe_intermediate_, nv_workspace_2048_),
                      "quantize routed SwiGLU");
                const float weight = 2.5f * scores[expert] / denominator;
                check(insignia::glm53::nvfp4_gemv_dp4a_acc_quantized(
                    expert_stager_->down_weight(), expert_stager_->down_scale(), expert_stager_->down_global(slot),
                    nv_workspace_2048_, routed_, weight, hidden_, moe_intermediate_), "routed expert down");
            }
        add_kernel<<<16, 256>>>(output, routed_, hidden_);
    }
    if (!nvfp4_experts_) {
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
    check(cudaMemcpy(df_logits_host_, verify_logits_.get(),
                     df_->logits_span() * sizeof(float), cudaMemcpyDeviceToHost),
          "download DFlash2 logits");
    check(cudaMemcpy(df_hp_host_, df_->hidden_projection(),
                     size_t(insignia::glm53::DFlash2Drafter::kDrafts) * insignia::glm53::DFlash2Drafter::kRank *
                         sizeof(float),
                     cudaMemcpyDeviceToHost),
          "download DFlash2 hp");
    static const bool df_debug = std::getenv("INSIGNIA_GLM53_DF_DEBUG") != nullptr;
    if (df_debug) {
        const int vocab = int(model_.vocab_size());
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
        check(insignia::glm53::nvfp4_gemv2_dp4a_quantized(
            expert_stager_->gate_weight(), expert_stager_->gate_scale(),
            expert_stager_->gate_global(slot),
            expert_stager_->up_weight(), expert_stager_->up_scale(),
            expert_stager_->up_global(slot),
            nv_workspace_4096_, gate_, up_, moe_intermediate_, hidden_), "MTP expert gate/up");
        check(insignia::glm53::quantize_swiglu_activation(gate_, up_, moe_intermediate_,
              nv_workspace_2048_), "quantize MTP SwiGLU");
        const float weight = 2.5f * scores[expert] / denominator;
        check(insignia::glm53::nvfp4_gemv_dp4a_acc_quantized(
            expert_stager_->down_weight(), expert_stager_->down_scale(),
            expert_stager_->down_global(slot), nv_workspace_2048_, routed_, weight,
            hidden_, moe_intermediate_), "MTP expert down");
    }
    add_kernel<<<16, 256>>>(output, routed_, hidden_);
    check(cudaGetLastError(), "MTP MoE combine launch");
}

// Restores the pre-verify recurrent state and replays the accepted prefix
// through the KDA recurrence from the archived pre-conv projections (the
// layer weights are already resident, so the replay is compute-only).
void Runner::rollback_kda(int accepted, int position_base) {
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
        float *history = conv_history_.get() + size_t(layer) * 9 * width;
        float *state = kda_states_.get() + size_t(layer) * width * kda_head_dim_;
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
    float *history = conv_history_.get() + size_t(layer) * 9 * width;
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
    float *state = kda_states_.get() + size_t(layer) * width * kda_head_dim_;
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
    const float *w_uk = w_uk_.get() + size_t(slot) * absorb_per_layer_;
    const float *w_uv = w_uv_.get() + size_t(slot) * absorb_per_layer_;
    float *cache_scale = mla_latent_scale_.get() + size_t(slot) * kMaxContext() *
                         insignia::glm53::kMlaLatentGroups;
    const bool exact_prefix = position_base + tokens <= kLegacyMlaContext;
    if (exact_prefix) {
        linear_multi(stem + "kv_b_proj.weight", c_small_, c_kv_, tokens,
                     kv_b_rows_, kv_a_rows_);
        check(insignia::glm53::mla_store_latent(
              c_small_.get(), cache_u8, cache_scale, cache_f32,
              tokens, position_base, kv_a_rows_), "MLA latent shadow prefill store");
        const size_t expanded_stride = size_t(kLegacyMlaContext) * q_b_rows_;
        check(insignia::glm53::mla_flash2_prefill(
              c_mlaq_.get(), c_kv_.get(),
              mla_keys_.get() + size_t(slot) * expanded_stride,
              mla_values_.get() + size_t(slot) * expanded_stride,
              c_mlao_.get(), tokens, position_base, mla_heads_, mla_head_dim_),
              "exact FlashAttention-2 MLA prefix");
    }
    static const bool scalar_attention =
        std::getenv("INSIGNIA_GLM53_SCALAR_MLA_PREFILL") != nullptr;
    if (!exact_prefix && scalar_attention) {
        for (int token = 0; token < tokens; ++token)
            check(insignia::glm53::mla_decode_latent(
                  c_mlaq_.get() + size_t(token) * q_b_rows_,
                  c_small_.get() + size_t(token) * kv_a_rows_,
                  nullptr, cache_u8, cache_scale, cache_f32, nullptr,
                  w_uk, w_uv, mla_partial_.get(),
                  c_mlao_.get() + size_t(token) * q_b_rows_, position_base + token,
                  mla_heads_, mla_head_dim_, kv_a_rows_), "scalar MLA attention (prefill)");
    } else if (!exact_prefix) {
        check(insignia::glm53::mla_prefill_latent(c_mlaq_.get(), c_small_.get(),
              nullptr, cache_u8, cache_scale, cache_f32, nullptr,
              w_uk, w_uv, c_mlao_, tokens,
              position_base, mla_heads_, mla_head_dim_, kv_a_rows_), "MLA latent prefill");
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

// Router scores for the whole chunk download once; each distinct expert's
// three matrices stage once and multiply against only the tokens that
// selected it, so expert I/O scales with distinct selections, not tokens.
void Runner::moe_multi(int layer, const float *input, float *output, int tokens) {
    const std::string stem = layer_stem(layer) + "mlp.";
    const int experts = moe_experts_;
    const int topk = moe_topk_;
    linear_multi(stem + "gate.weight", input, c_router_, tokens, experts, hidden_);
    std::vector<float> logits(size_t(tokens) * experts);
    check(cudaMemcpy(logits.data(), c_router_.get(), logits.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "download router logits (prefill)");
    const std::vector<float> &bias = host_f32(stem + "gate.e_score_correction_bias");
    require(bias.size() == size_t(experts), "wrong router bias geometry");
    std::vector<std::vector<std::pair<int, float>>> selection(tokens);
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
        for (int slot = 0; slot < topk; ++slot)
            selection[token].emplace_back(order[slot], 2.5f * scores[order[slot]] / denominator);
    }
    std::vector<int> distinct;
    for (const auto &picks : selection)
        for (const auto &[expert, weight] : picks)
            if (std::find(distinct.begin(), distinct.end(), expert) == distinct.end())
                distinct.push_back(expert);
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
    check(cudaMemset(c_routed_, 0, size_t(tokens) * hidden_ * sizeof(float)), "clear routed (prefill)");
    if (nvfp4_experts_ && expert_stager_) {
        // Routing bookkeeping + speculative next-layer reads, mirroring the
        // decode path. Demand staging for the whole deduplicated union comes
        // first so all four readers stream the layer while its first GEMV
        // batches run; speculative queues can never jump a demand record.
        for (int slot = 0; slot < topk; ++slot)
            prev_routing_[size_t(layer)][size_t(slot)] = selection[size_t(tokens - 1)][size_t(slot)].first;
        for (int row = 0; row < tokens && row < 8; ++row)
            for (int slot = 0; slot < topk; ++slot)
                row_routing_[size_t(layer)][size_t(row * topk + slot)] =
                    selection[size_t(row)][size_t(slot)].first;
        expert_stager_->stage_layer(layer, distinct.data(), int(distinct.size()));
        if (prefetch_on_ && size_t(layer) + 1 < prev_routing_.size() &&
            is_sparse_[size_t(layer) + 1])
            expert_stager_->prefetch(int(layer) + 1, prev_routing_[size_t(layer) + 1].data(),
                                     moe_topk_);
        if (prefetch_on_ && !cct_.empty())
            cct_prefetch(layer);
        // The shared expert is dense-resident FP8; running it first lets its
        // GEMVs hide under the routed records' disk reads (same ordering as
        // the decode path). It writes `output`; routed experts accumulate
        // into c_routed_ and the combine order below is unchanged.
        mlp_multi(stem + "shared_experts.", input, output, tokens, shared_intermediate_);
    }

    if (!nvfp4_experts_) {
        for (int expert : distinct) {
            std::vector<int> users;
            for (int token = 0; token < tokens; ++token)
                for (const auto &[picked, weight] : selection[token])
                    if (picked == expert) users.push_back(token);
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
                for (const auto &[picked, pick_weight] : selection[token])
                    if (picked == expert) weight = pick_weight;
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
            if (verify_populate_) {
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
                    for (const auto &[expert, weight] : selection[size_t(token)]) {
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
                    for (int pick_slot = 0; pick_slot < topk; ++pick_slot)
                        if (selection[size_t(token)][size_t(pick_slot)].first == expert &&
                            selection[size_t(token)][size_t(pick_slot)].second != 0.0f) {
                            users[size_t(total)] = token;
                            out_ids[size_t(total)] = token * topk + pick_slot;
                            combine[size_t(total)] = selection[size_t(token)][size_t(pick_slot)].second;
                            ++total;
                        }
                for (int base = 0; base < total; base += kMaxVerify) {
                    const int count = std::min(kMaxVerify, total - base);
                    check(insignia::glm53::nvfp4_quantize_activation_rows(
                              input, hidden_, &users[size_t(base)], count, nv_workspace_4096_),
                          "quantize expert input (batched prefill)");
                    check(insignia::glm53::nvfp4_gemv2_dp4a_quantized_rows(
                              expert_stager_->gate_weight(), expert_stager_->gate_scale(),
                              expert_stager_->gate_global(int(slot)),
                              expert_stager_->up_weight(), expert_stager_->up_scale(),
                              expert_stager_->up_global(int(slot)),
                              nv_workspace_4096_, count,
                              c_gateu_.get(), c_up_.get(), &users[size_t(base)],
                              moe_intermediate_, hidden_), "routed expert gate/up (batched prefill)");
                    check(insignia::glm53::quantize_swiglu_activation_rows(
                              c_gateu_.get(), c_up_.get(), moe_intermediate_, &users[size_t(base)],
                              count, nv_workspace_2048_), "quantize routed SwiGLU (batched prefill)");
                    if (retain_down_results) {
                        check(insignia::glm53::nvfp4_gemv_dp4a_quantized_rows(
                                  expert_stager_->down_weight(), expert_stager_->down_scale(),
                                  expert_stager_->down_global(int(slot)), nv_workspace_2048_,
                                  count, c_expert_out_.get(), &out_ids[size_t(base)],
                                  hidden_, moe_intermediate_), "routed expert down (ordered batched)");
                    } else {
                        check(insignia::glm53::nvfp4_gemv_dp4a_acc_quantized_rows(
                                  expert_stager_->down_weight(), expert_stager_->down_scale(),
                                  expert_stager_->down_global(int(slot)), nv_workspace_2048_,
                                  count, c_routed_.get(), &users[size_t(base)], &combine[size_t(base)],
                                  hidden_, moe_intermediate_), "routed expert down (batched prefill)");
                    }
                }
            }
        }
        if (ordered_accumulation)
            for (int token = 0; token < tokens; ++token)
                for (int pick_slot = 0; pick_slot < topk; ++pick_slot)
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
                    for (const auto &[expert, weight] : selection[size_t(token)])
                        if (std::find(chunk_distinct.begin(), chunk_distinct.end(), expert) ==
                            chunk_distinct.end())
                            chunk_distinct.push_back(expert);
                for (int expert : chunk_distinct)
                    for (int token = chunk_base; token < chunk_end; ++token)
                        for (int pick_slot = 0; pick_slot < topk; ++pick_slot)
                            if (selection[size_t(token)][size_t(pick_slot)].first == expert)
                                scale_add_kernel<<<16, 256>>>(
                                    c_routed_.get() + size_t(token) * hidden_,
                                    c_expert_out_.get() +
                                        (size_t(token) * topk + size_t(pick_slot)) * hidden_,
                                    selection[size_t(token)][size_t(pick_slot)].second, hidden_);
            }
    }
    if (!nvfp4_experts_)
        mlp_multi(stem + "shared_experts.", input, output, tokens, shared_intermediate_);
    for (int token = 0; token < tokens; ++token)
        add_kernel<<<16, 256>>>(output + size_t(token) * hidden_,
                                c_routed_.get() + size_t(token) * hidden_, hidden_);
    check(cudaGetLastError(), "MoE combine launch (prefill)");
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
        const auto begin = std::chrono::steady_clock::now();
        const std::string base = layer_stem(layer);
        mhc_multi(base + "hc_attn", streams, c_collapsed_, count);
        rms(base + "input_layernorm.weight", c_collapsed_, c_normalized_, count, hidden_);
        if (early_multi_route_on_ && is_sparse_[size_t(layer)])
            early_route_multi(layer, c_normalized_, count);
        if (is_mla_[layer])
            mla_multi(layer, c_normalized_, c_attn_, count, position_base);
        else
            kda_multi(layer, c_normalized_, c_attn_, count, position_base);
        for (int token = 0; token < count; ++token)
            check(insignia::glm53::mhc_mix(streams + size_t(token) * kStreams * hidden_,
                c_attn_.get() + size_t(token) * hidden_, c_post_.get() + size_t(token) * 4,
                c_comb_.get() + size_t(token) * 16,
                next_streams + size_t(token) * kStreams * hidden_, hidden_), "attention mHC mix (prefill)");
        std::swap(streams, next_streams);

        mhc_multi(base + "hc_ffn", streams, c_collapsed_, count);
        rms(base + "post_attention_layernorm.weight", c_collapsed_, c_normalized_, count, hidden_);
        if (is_sparse_[layer])
            moe_multi(layer, c_normalized_, c_ffn_, count);
        else
            mlp_multi(base + "mlp.", c_normalized_, c_ffn_, count, dense_intermediate_);
        for (int token = 0; token < count; ++token)
            check(insignia::glm53::mhc_mix(streams + size_t(token) * kStreams * hidden_,
                c_ffn_.get() + size_t(token) * hidden_, c_post_.get() + size_t(token) * 4,
                c_comb_.get() + size_t(token) * 16,
                next_streams + size_t(token) * kStreams * hidden_, hidden_), "FFN mHC mix (prefill)");
        std::swap(streams, next_streams);
        if (df_) {
            for (int ci = 0; ci < 5; ++ci)
                if (kDfCaptureLayers[ci] == layer)
                    for (int token = 0; token < count; ++token)
                        average_streams_kernel<<<16, 256>>>(
                            streams + size_t(token) * kStreams * hidden_,
                            df_->capture_row(ci, capture_offset_ + token), hidden_);
        }
        if (deep_checks_) {
            const int one = 1;
            int valid = 0;
            check(cudaMemcpy(finite_, &one, sizeof(one), cudaMemcpyHostToDevice), "initialize finite flag");
            finite_kernel<<<16, 256>>>(streams, count * kStreams * hidden_, finite_);
            check(cudaMemcpy(&valid, finite_.get(), sizeof(valid), cudaMemcpyDeviceToHost), "read finite flag");
            require(valid, "non-finite residual stream after prefill layer " + std::to_string(layer));
        }
        if (trace_layers_) {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            std::printf("prefill layer %02d %-3s %d tokens %.3f s\n", layer,
                        is_mla_[layer] ? "MLA" : "KDA", count, seconds);
            std::fflush(stdout);
        }
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
    if (q8_stager_)
        std::printf("  %s matrix cache %.3f GiB / %.3f s (%.2f GB/s)\n",
                    q8_index_->format() == Cache8Format::q8 ? "Q8" : "FP8",
                    q8_bytes / double(1ull << 30), q8_seconds,
                    q8_bytes / q8_seconds / 1.0e9);
    return top;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 7) {
        std::fprintf(stderr,
            "usage: %s MODEL_ROOT MODEL.index [TOKENS=154820] [LAYERS=0(all)] [GENERATE=1] [8BIT_PREFIX]\n"
            "  INSIGNIA_GLM53_MTP=K enables K-token MTP speculative decode (greedy-exact)\n",
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
        std::string encoded = argc >= 4 ? argv[3] : "154820";
        if (!encoded.empty() && encoded[0] == '@') {
            std::FILE *prompt_file = std::fopen(encoded.c_str() + 1, "r");
            require(prompt_file, "cannot open prompt file");
            std::vector<char> payload;
            char chunk[65536];
            size_t read = 0;
            while ((read = std::fread(chunk, 1, sizeof(chunk), prompt_file)) > 0)
                payload.insert(payload.end(), chunk, chunk + read);
            std::fclose(prompt_file);
            encoded.assign(payload.begin(), payload.end());
            while (!encoded.empty() && std::isspace(static_cast<unsigned char>(encoded.back())))
                encoded.pop_back();
        }
        std::vector<int> tokens;
        size_t begin_token = 0;
        while (begin_token < encoded.size()) {
            const size_t comma = encoded.find(',', begin_token);
            tokens.push_back(std::stoi(encoded.substr(begin_token, comma - begin_token)));
            if (comma == std::string::npos) break;
            begin_token = comma + 1;
        }
        require(!tokens.empty() && tokens.size() <= kMaxContext(), "token list exceeds the context limit");
        const int layers_argc = argc >= 5 ? std::atoi(argv[4]) : 0;
        const int generate = argc >= 6 ? std::atoi(argv[5]) : 1;
        require(generate >= 1 && tokens.size() + size_t(generate) - 1 <= kMaxContext(),
                "generation must fit the exact-attention cache");
        Runner runner(argv[1], argv[2], argc >= 7 ? argv[6] : "");
        const int layers = layers_argc > 0 ? layers_argc : runner.layer_count();

        const auto begin = std::chrono::steady_clock::now();
        std::vector<std::pair<int, float>> top;
        // Prompt tokens run layer-major in chunks so each weight streams once
        // per chunk; the final token keeps the per-token path that emits
        // logits.
        if (tokens.size() > 1) {
            const size_t prefill_count = tokens.size() - 1;
            for (size_t consumed = 0; consumed < prefill_count; ) {
                const size_t take = std::min<size_t>(Runner::kMaxChunk(), prefill_count - consumed);
                std::vector<int> chunk(tokens.begin() + int(consumed), tokens.begin() + int(consumed + take));
                runner.prefill(chunk, int(consumed));
                consumed += take;
                std::printf("prompt %zu/%zu tokens prefilled\n", consumed, tokens.size());
                std::fflush(stdout);
            }
        }
        top = runner.step(tokens.back(), int(tokens.size()) - 1, layers, true);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
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
                double accept_total = 0.0, draft_total = 0.0, verify_total = 0.0;
                double fallback_total = 0.0;
                int rounds = 0, verified_rounds = 0, empty_rounds = 0;
                std::array<int, 8> accept_hist{};
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
                const bool adaptive_k_on = [] {
                    const char *value = std::getenv("INSIGNIA_GLM53_DF_ADAPTIVE_K");
                    return !value || std::atoi(value) != 0;
                }();
                double accept_ema = 0.0;
                bool accept_ema_init = false;
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
                    int round_verify_k = verify_k;
                    if (adaptive_k_on && accept_ema_init)
                        round_verify_k = std::clamp(int(accept_ema * 1.3) + 1, 2, verify_k);
                    const int draft_k = std::min(round_verify_k, generate - int(generated.size()));
                    if (std::getenv("INSIGNIA_GLM53_DF_DEBUG"))
                        std::fprintf(stderr, "df round %d: anchor %d pos %d truth0 %d\n",
                                     rounds, root, position, truth0);
                    const auto draft_begin = std::chrono::steady_clock::now();
                    const std::vector<int> candidates = runner.df_draft(root, position);
                    draft_total += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - draft_begin).count();

                    // d1 is checked against the exact target argmax carried
                    // from the preceding committed token. A mismatch proves
                    // this round accepts zero drafts; running the 7-token
                    // target verifier cannot change that verdict.
                    if (candidates[0] != truth0) {
                        ++empty_rounds;
                        ++accept_hist[0];
                        generated.push_back(truth0);
                        ++rounds;
                        accept_total += 1;
                        // Empty rounds are matched=0 samples; skipping the
                        // EMA update leaves k inflated through empty streaks,
                        // widening verify unions exactly when acceptance is
                        // worst.
                        accept_ema = accept_ema_init ? 0.75 * accept_ema : 0.0;
                        accept_ema_init = true;
                        if (int(generated.size()) < generate) {
                            const auto fallback_begin = std::chrono::steady_clock::now();
                            top = runner.step(truth0, position + 1, layers, true);
                            fallback_total += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - fallback_begin).count();
                            root = truth0;
                            truth0 = top.front().first;
                            ++position;
                        }
                        std::fflush(stdout);
                        continue;
                    }

                    const auto verify_begin = std::chrono::steady_clock::now();
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
                        const std::pair<int, std::vector<int>> verdict =
                            runner.verify_round(verify_candidates, position + 1);
                        arg = verdict.second;
                        while (matched < draft_k && arg[size_t(matched - 1)] == candidates[size_t(matched)])
                            ++matched;
                    }
                    verify_total += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - verify_begin).count();
                    ++verified_rounds;
                    accept_ema = accept_ema_init ? 0.75 * accept_ema + 0.25 * matched
                                                 : double(matched);
                    accept_ema_init = true;
                    ++accept_hist[size_t(matched)];
                    for (int i = 0; i < matched && int(generated.size()) < generate; ++i)
                        generated.push_back(candidates[size_t(i)]);
                    if (matched < draft_k && !df_seq_verify)
                        runner.rollback_kda(matched, position + 1);
                    // The verify captures cover the candidates; append the
                    // accepted prefix (step() commits its own token).
                    runner.adopt_anchor_routing(df_seq_verify ? 0 : matched - 1);
                    runner.df_commit(matched, position + 1);
                    root = candidates[size_t(matched - 1)];
                    truth0 = arg[size_t(matched - 1)];
                    position += matched;
                    accept_total += matched;
                    ++rounds;
                    std::fflush(stdout);
                }
                // Drafter window exhausted (position >= 2040): plain greedy
                // steps for the remainder, same shape as the empty-round
                // fallback.
                while (int(generated.size()) < generate) {
                    generated.push_back(truth0);
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
                            "draft %.1f ms/round + verify %.1f ms/verified round (%d verified), "
                            "fallback %.1f ms)\n",
                            tokens.size(), elapsed, generate, rounds, verify_k,
                            accept_total / rounds, empty_rounds,
                            1000.0 * decode_seconds / generate,
                            1000.0 * draft_total / rounds,
                            1000.0 * verify_total / std::max(1, verified_rounds),
                            verified_rounds,
                            1000.0 * fallback_total);
                std::printf("  accepted histogram");
                for (int count = 0; count <= verify_k; ++count)
                    if (accept_hist[size_t(count)])
                        std::printf(" %d:%d", count, accept_hist[size_t(count)]);
                std::printf("\n");
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
