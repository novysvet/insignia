#include <cuda_runtime.h>

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr size_t kBytes = 2660ull * 4096;
constexpr int kWarmup = 4;
constexpr int kRounds = 128;

void check(cudaError_t status, const char *what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

__attribute__((noinline)) void copy_memcpy(void *destination, const void *source,
                                           size_t bytes) {
    std::memcpy(destination, source, bytes);
}

__attribute__((target("avx2"), noinline))
void copy_avx2_temporal(void *destination, const void *source, size_t bytes) {
    auto *out = static_cast<uint8_t *>(destination);
    const auto *in = static_cast<const uint8_t *>(source);
    size_t offset = 0;
    for (; offset + 256 <= bytes; offset += 256) {
        const __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(in + offset));
        const __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(in + offset + 32));
        const __m256i v2 = _mm256_load_si256(reinterpret_cast<const __m256i *>(in + offset + 64));
        const __m256i v3 = _mm256_load_si256(reinterpret_cast<const __m256i *>(in + offset + 96));
        const __m256i v4 = _mm256_load_si256(reinterpret_cast<const __m256i *>(in + offset + 128));
        const __m256i v5 = _mm256_load_si256(reinterpret_cast<const __m256i *>(in + offset + 160));
        const __m256i v6 = _mm256_load_si256(reinterpret_cast<const __m256i *>(in + offset + 192));
        const __m256i v7 = _mm256_load_si256(reinterpret_cast<const __m256i *>(in + offset + 224));
        _mm256_store_si256(reinterpret_cast<__m256i *>(out + offset), v0);
        _mm256_store_si256(reinterpret_cast<__m256i *>(out + offset + 32), v1);
        _mm256_store_si256(reinterpret_cast<__m256i *>(out + offset + 64), v2);
        _mm256_store_si256(reinterpret_cast<__m256i *>(out + offset + 96), v3);
        _mm256_store_si256(reinterpret_cast<__m256i *>(out + offset + 128), v4);
        _mm256_store_si256(reinterpret_cast<__m256i *>(out + offset + 160), v5);
        _mm256_store_si256(reinterpret_cast<__m256i *>(out + offset + 192), v6);
        _mm256_store_si256(reinterpret_cast<__m256i *>(out + offset + 224), v7);
    }
    if (offset != bytes) std::memcpy(out + offset, in + offset, bytes - offset);
}

__attribute__((target("avx2"), noinline))
void copy_avx2_stream(void *destination, const void *source, size_t bytes) {
    auto *out = static_cast<uint8_t *>(destination);
    const auto *in = static_cast<const uint8_t *>(source);
    size_t offset = 0;
    for (; offset + 256 <= bytes; offset += 256) {
        _mm_prefetch(reinterpret_cast<const char *>(in + offset + 1024), _MM_HINT_NTA);
        const __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(in + offset));
        const __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(in + offset + 32));
        const __m256i v2 = _mm256_load_si256(reinterpret_cast<const __m256i *>(in + offset + 64));
        const __m256i v3 = _mm256_load_si256(reinterpret_cast<const __m256i *>(in + offset + 96));
        const __m256i v4 = _mm256_load_si256(reinterpret_cast<const __m256i *>(in + offset + 128));
        const __m256i v5 = _mm256_load_si256(reinterpret_cast<const __m256i *>(in + offset + 160));
        const __m256i v6 = _mm256_load_si256(reinterpret_cast<const __m256i *>(in + offset + 192));
        const __m256i v7 = _mm256_load_si256(reinterpret_cast<const __m256i *>(in + offset + 224));
        _mm256_stream_si256(reinterpret_cast<__m256i *>(out + offset), v0);
        _mm256_stream_si256(reinterpret_cast<__m256i *>(out + offset + 32), v1);
        _mm256_stream_si256(reinterpret_cast<__m256i *>(out + offset + 64), v2);
        _mm256_stream_si256(reinterpret_cast<__m256i *>(out + offset + 96), v3);
        _mm256_stream_si256(reinterpret_cast<__m256i *>(out + offset + 128), v4);
        _mm256_stream_si256(reinterpret_cast<__m256i *>(out + offset + 160), v5);
        _mm256_stream_si256(reinterpret_cast<__m256i *>(out + offset + 192), v6);
        _mm256_stream_si256(reinterpret_cast<__m256i *>(out + offset + 224), v7);
    }
    if (offset != bytes) std::memcpy(out + offset, in + offset, bytes - offset);
    _mm_sfence();
}

using Copy = void (*)(void *, const void *, size_t);

void copy_stream_parallel(void *destination, const void *source, size_t bytes, int threads) {
    const size_t slice = (bytes / size_t(threads)) & ~size_t(4095);
    std::vector<std::thread> workers;
    workers.reserve(size_t(threads - 1));
    for (int thread = 0; thread + 1 < threads; ++thread) {
        const size_t begin = size_t(thread) * slice;
        workers.emplace_back(copy_avx2_stream,
                             static_cast<uint8_t *>(destination) + begin,
                             static_cast<const uint8_t *>(source) + begin, slice);
    }
    const size_t begin = size_t(threads - 1) * slice;
    copy_avx2_stream(static_cast<uint8_t *>(destination) + begin,
                     static_cast<const uint8_t *>(source) + begin, bytes - begin);
    for (std::thread &worker : workers) worker.join();
}

template <typename Function>
void run(const char *name, Function copy, void *destination, const void *source) {
    for (int round = 0; round < kWarmup; ++round) copy(destination, source, kBytes);
    const auto begin = std::chrono::steady_clock::now();
    for (int round = 0; round < kRounds; ++round) copy(destination, source, kBytes);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    const double gib = double(kBytes) * kRounds / double(1ull << 30);
    const volatile uint8_t witness = static_cast<const uint8_t *>(destination)[kBytes / 2];
    std::printf("%-22s %8.3f ms/record  %8.2f GiB/s  witness=%u\n",
                name, seconds * 1000.0 / kRounds, gib / seconds, unsigned(witness));
}

}  // namespace

int main() {
    void *pinned = nullptr, *pageable = nullptr;
    check(cudaHostAlloc(&pinned, kBytes, cudaHostAllocDefault), "cudaHostAlloc source");
    if (::posix_memalign(&pageable, 4096, kBytes)) throw std::bad_alloc();
    auto *source = static_cast<uint8_t *>(pinned);
    auto *destination = static_cast<uint8_t *>(pageable);
    for (size_t byte = 0; byte < kBytes; ++byte)
        source[byte] = uint8_t((byte * 1315423911u + (byte >> 12)) >> 17);
    std::memset(destination, 0, kBytes);

    std::printf("pinned -> pageable, %.3f MiB record, %d timed copies\n",
                kBytes / double(1ull << 20), kRounds);
    run("glibc memcpy", copy_memcpy, destination, source);
    run("AVX2 temporal", copy_avx2_temporal, destination, source);
    run("AVX2 stream", copy_avx2_stream, destination, source);
    run("AVX2 stream x2", [](void *out, const void *in, size_t bytes) {
        copy_stream_parallel(out, in, bytes, 2);
    }, destination, source);
    run("AVX2 stream x4", [](void *out, const void *in, size_t bytes) {
        copy_stream_parallel(out, in, bytes, 4);
    }, destination, source);

    cudaFreeHost(pinned);
    std::free(pageable);
    return 0;
}
