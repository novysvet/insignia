#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "insignia_glm53.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif
#if !defined(INSIGNIA_GLM53_NO_MAIN)
#include <array>
#include <chrono>
#endif

namespace {

struct FixtureHeader {
    char magic[8];
    uint32_t version;
    uint32_t rows;
    uint32_t cols;
    uint32_t group;
    float global_scale;
    uint32_t reserved;
    uint64_t nv_weight_bytes;
    uint64_t nv_scale_bytes;
    uint64_t i4_weight_bytes;
    uint64_t i4_scale_bytes;
    uint64_t e2_weight_bytes;
    uint64_t e2_scale_bytes;
};
static_assert(sizeof(FixtureHeader) == 80);

__constant__ float c_e4m3[256];
__constant__ int8_t c_e2i[16];

[[noreturn]] void die(const char *message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

void cuda_check(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(2);
    }
}

std::vector<uint8_t> read_file(const char *path) {
    std::FILE *file = std::fopen(path, "rb");
    if (!file) die("cannot open fixture");
    if (std::fseek(file, 0, SEEK_END) || std::ftell(file) < 0) die("cannot size fixture");
    const long size = std::ftell(file);
    std::rewind(file);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (std::fread(bytes.data(), 1, bytes.size(), file) != bytes.size()) die("short fixture read");
    std::fclose(file);
    return bytes;
}

template <typename T>
T *device_copy(const void *source, size_t count) {
    T *device = nullptr;
    cuda_check(cudaMalloc(&device, count * sizeof(T)), "cudaMalloc");
    cuda_check(cudaMemcpy(device, source, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    return device;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, offset);
    return value;
}

__device__ __forceinline__ void finish_row(float value, float *output, int row) {
    __shared__ float partial[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (!lane) partial[warp] = value;
    __syncthreads();
    if (!warp) {
        value = lane < 8 ? partial[lane] : 0.0f;
        value = warp_sum(value);
        if (!lane) output[row] = value;
    }
}

__global__ __launch_bounds__(256) void nvfp4_f32_kernel(
    const uint8_t *__restrict__ weights,
    const uint8_t *__restrict__ scales,
    const float *__restrict__ x,
    float *__restrict__ y,
    int rows,
    int cols,
    float global_scale) {
    __shared__ float e2[16];
    if (threadIdx.x < 16) e2[threadIdx.x] = 0.5f * float(c_e2i[threadIdx.x]);
    __syncthreads();
    const int row = blockIdx.x;
    const int group = threadIdx.x;
    float sum = 0.0f;
    if (row < rows && group < (cols >> 4)) {
        const uint8_t *row_weights = weights + static_cast<size_t>(row) * (cols >> 1);
        const uint64_t packed = *reinterpret_cast<const uint64_t *>(row_weights + group * 8);
        const float *xg = x + group * 16;
#pragma unroll
        for (int i = 0; i < 16; ++i)
            sum = fmaf(e2[(packed >> (4 * i)) & 15u], __ldg(xg + i), sum);
        sum *= c_e4m3[scales[static_cast<size_t>(row) * (cols >> 4) + group]] * global_scale;
    }
    if (row < rows) finish_row(sum, y, row);
}

__global__ __launch_bounds__(256, 1) void quantize_x16_kernel(
    const float *__restrict__ x,
    uint32_t *__restrict__ xq,
    float *__restrict__ xscale,
    int groups) {
    const int group = threadIdx.x;
    if (group >= groups) return;
    const float4 *source = reinterpret_cast<const float4 *>(x + group * 16);
    float values[16];
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float4 value = __ldg(source + i);
        values[4 * i + 0] = value.x;
        values[4 * i + 1] = value.y;
        values[4 * i + 2] = value.z;
        values[4 * i + 3] = value.w;
        maximum = fmaxf(maximum, fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                                       fmaxf(fabsf(value.z), fabsf(value.w))));
    }
    const float inverse = maximum > 0.0f ? 127.0f / maximum : 0.0f;
    xscale[group] = maximum * (1.0f / 127.0f);
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        uint32_t packed = 0;
#pragma unroll
        for (int byte = 0; byte < 4; ++byte)
            packed |= uint32_t(uint8_t(__float2int_rn(values[word * 4 + byte] * inverse))) << (byte * 8);
        xq[group * 4 + word] = packed;
    }
}

__device__ __forceinline__ void unpack_e2(
    uint32_t packed,
    const unsigned long long *__restrict__ table,
    uint32_t &first,
    uint32_t &second) {
    const unsigned long long b0 = table[packed & 0xff];
    const unsigned long long b1 = table[(packed >> 8) & 0xff];
    const unsigned long long b2 = table[(packed >> 16) & 0xff];
    const unsigned long long b3 = table[packed >> 24];
    const uint32_t a0 = __byte_perm(uint32_t(b0), uint32_t(b1), 0xc480);
    const uint32_t a1 = __byte_perm(uint32_t(b0 >> 32), uint32_t(b1 >> 32), 0x4c80);
    first = __byte_perm(a0, a1, 0x6240);
    const uint32_t a2 = __byte_perm(uint32_t(b2), uint32_t(b3), 0xc480);
    const uint32_t a3 = __byte_perm(uint32_t(b2 >> 32), uint32_t(b3 >> 32), 0x4c80);
    second = __byte_perm(a2, a3, 0x6240);
}

__global__ __launch_bounds__(256) void nvfp4_dp4a_kernel(
    const uint32_t *__restrict__ weights,
    const uint8_t *__restrict__ scales,
    const uint32_t *__restrict__ xq,
    const float *__restrict__ xscale,
    float *__restrict__ y,
    int rows,
    int groups,
    float global_scale) {
    __shared__ unsigned long long table[256];
    const int code = threadIdx.x;
    const uint32_t lo = uint32_t(uint8_t(c_e2i[code & 15])) * 0x01010101u;
    const uint32_t hi = uint32_t(uint8_t(c_e2i[code >> 4])) * 0x01010101u;
    table[code] = static_cast<unsigned long long>(lo) | (static_cast<unsigned long long>(hi) << 32);
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_weights = weights + static_cast<size_t>(row) * groups * 2;
    const uint8_t *row_scales = scales + static_cast<size_t>(row) * groups;
    float sum = 0.0f;
#pragma unroll 8
    for (int group = lane; group < groups; group += 32) {
        const uint2 packed = __ldcs(reinterpret_cast<const uint2 *>(row_weights + group * 2));
        uint32_t w0, w1, w2, w3;
        unpack_e2(packed.x, table, w0, w1);
        unpack_e2(packed.y, table, w2, w3);
        const uint32_t *xg = xq + group * 4;
        int dot = __dp4a(int(w0), int(xg[0]), 0);
        dot = __dp4a(int(w1), int(xg[1]), dot);
        dot = __dp4a(int(w2), int(xg[2]), dot);
        dot = __dp4a(int(w3), int(xg[3]), dot);
        const float scale = 0.5f * c_e4m3[row_scales[group]] * global_scale * xscale[group];
        sum = fmaf(float(dot), scale, sum);
    }
    sum = warp_sum(sum);
    if (!lane) y[row] = sum;
}

__global__ __launch_bounds__(256) void nvfp4_dp4a_acc_kernel(
    const uint32_t *__restrict__ weights,
    const uint8_t *__restrict__ scales,
    const uint32_t *__restrict__ xq,
    const float *__restrict__ xscale,
    float *__restrict__ y,
    float combine_weight,
    int rows,
    int groups,
    float global_scale) {
    __shared__ unsigned long long table[256];
    const int code = threadIdx.x;
    const uint32_t lo = uint32_t(uint8_t(c_e2i[code & 15])) * 0x01010101u;
    const uint32_t hi = uint32_t(uint8_t(c_e2i[code >> 4])) * 0x01010101u;
    table[code] = static_cast<unsigned long long>(lo) | (static_cast<unsigned long long>(hi) << 32);
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_weights = weights + static_cast<size_t>(row) * groups * 2;
    const uint8_t *row_scales = scales + static_cast<size_t>(row) * groups;
    float sum = 0.0f;
#pragma unroll 8
    for (int group = lane; group < groups; group += 32) {
        const uint2 packed = __ldcs(reinterpret_cast<const uint2 *>(row_weights + group * 2));
        uint32_t w0, w1, w2, w3;
        unpack_e2(packed.x, table, w0, w1);
        unpack_e2(packed.y, table, w2, w3);
        const uint32_t *xg = xq + group * 4;
        int dot = __dp4a(int(w0), int(xg[0]), 0);
        dot = __dp4a(int(w1), int(xg[1]), dot);
        dot = __dp4a(int(w2), int(xg[2]), dot);
        dot = __dp4a(int(w3), int(xg[3]), dot);
        const float scale = 0.5f * c_e4m3[row_scales[group]] * global_scale * xscale[group];
        sum = fmaf(float(dot), scale, sum);
    }
    sum = warp_sum(sum);
    if (!lane) y[row] = fmaf(sum, combine_weight, y[row]);
}

__global__ __launch_bounds__(256) void nvfp4_dp4a_pair_kernel(
    const uint32_t *__restrict__ weights_a,
    const uint8_t *__restrict__ scales_a,
    const uint32_t *__restrict__ weights_b,
    const uint8_t *__restrict__ scales_b,
    const uint32_t *__restrict__ xq,
    const float *__restrict__ xscale,
    float *__restrict__ y_a,
    float *__restrict__ y_b,
    int rows,
    int groups,
    float global_a,
    float global_b) {
    __shared__ unsigned long long table[256];
    const int code = threadIdx.x;
    const uint32_t lo = uint32_t(uint8_t(c_e2i[code & 15])) * 0x01010101u;
    const uint32_t hi = uint32_t(uint8_t(c_e2i[code >> 4])) * 0x01010101u;
    table[code] = static_cast<unsigned long long>(lo) | (static_cast<unsigned long long>(hi) << 32);
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_a = weights_a + static_cast<size_t>(row) * groups * 2;
    const uint32_t *row_b = weights_b + static_cast<size_t>(row) * groups * 2;
    const uint8_t *scale_a = scales_a + static_cast<size_t>(row) * groups;
    const uint8_t *scale_b = scales_b + static_cast<size_t>(row) * groups;
    float sum_a = 0.0f, sum_b = 0.0f;
#pragma unroll 8
    for (int group = lane; group < groups; group += 32) {
        const uint2 packed_a = __ldcs(reinterpret_cast<const uint2 *>(row_a + group * 2));
        const uint2 packed_b = __ldcs(reinterpret_cast<const uint2 *>(row_b + group * 2));
        uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
        unpack_e2(packed_a.x, table, a0, a1);
        unpack_e2(packed_a.y, table, a2, a3);
        unpack_e2(packed_b.x, table, b0, b1);
        unpack_e2(packed_b.y, table, b2, b3);
        const uint32_t *xg = xq + group * 4;
        int dot_a = __dp4a(int(a0), int(xg[0]), 0);
        dot_a = __dp4a(int(a1), int(xg[1]), dot_a);
        dot_a = __dp4a(int(a2), int(xg[2]), dot_a);
        dot_a = __dp4a(int(a3), int(xg[3]), dot_a);
        int dot_b = __dp4a(int(b0), int(xg[0]), 0);
        dot_b = __dp4a(int(b1), int(xg[1]), dot_b);
        dot_b = __dp4a(int(b2), int(xg[2]), dot_b);
        dot_b = __dp4a(int(b3), int(xg[3]), dot_b);
        const float activation_scale = 0.5f * xscale[group];
        sum_a = fmaf(float(dot_a), activation_scale * c_e4m3[scale_a[group]] * global_a, sum_a);
        sum_b = fmaf(float(dot_b), activation_scale * c_e4m3[scale_b[group]] * global_b, sum_b);
    }
    sum_a = warp_sum(sum_a);
    sum_b = warp_sum(sum_b);
    if (!lane) {
        y_a[row] = sum_a;
        y_b[row] = sum_b;
    }
}

__global__ __launch_bounds__(256, 1) void quantize_swiglu_x16_kernel(
    const float *__restrict__ gate,
    const float *__restrict__ up,
    uint32_t *__restrict__ xq,
    float *__restrict__ xscale,
    int groups) {
    const int group = threadIdx.x;
    if (group >= groups) return;
    const float4 *gate4 = reinterpret_cast<const float4 *>(gate + group * 16);
    const float4 *up4 = reinterpret_cast<const float4 *>(up + group * 16);
    float values[16];
    float maximum = 0.0f;
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        const float4 g = __ldg(gate4 + word);
        const float4 u = __ldg(up4 + word);
        const float gv[4] = {g.x, g.y, g.z, g.w};
        const float uv[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float gc = fminf(gv[i], 10.0f);
            const float uc = fminf(fmaxf(uv[i], -10.0f), 10.0f);
            const float value = (gc / (1.0f + __expf(-gc))) * uc;
            values[word * 4 + i] = value;
            maximum = fmaxf(maximum, fabsf(value));
        }
    }
    const float inverse = maximum > 0.0f ? 127.0f / maximum : 0.0f;
    xscale[group] = maximum * (1.0f / 127.0f);
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        uint32_t packed = 0;
#pragma unroll
        for (int byte = 0; byte < 4; ++byte)
            packed |= uint32_t(uint8_t(__float2int_rn(values[word * 4 + byte] * inverse))) << (byte * 8);
        xq[group * 4 + word] = packed;
    }
}

// ---------------------------------------------------------------------------
// Multi-row (R <= 8) variants for the verify path: one weight pass serves all
// token rows that share the expert. Every per-row arithmetic chain (group
// order, scale association, warp_sum tree) is a verbatim transplant of the
// 1-row kernels above, so each output element stays bit-identical to running
// the single-row kernel row by row.
// ---------------------------------------------------------------------------

struct Nvfp4RowIds {
    int count;
    int ids[8];
};

struct Nvfp4RowOut {
    int count;
    int ids[8];
    float weights[8];
};

__global__ __launch_bounds__(256, 1) void quantize_x16_rows_kernel(
    const float *__restrict__ x,
    uint32_t *__restrict__ xq,        // [rows][words_per_row]
    float *__restrict__ xscale,       // [rows][groups]
    int groups,
    int words_per_row,
    Nvfp4RowIds rows) {
    const int group = threadIdx.x;
    if (group >= groups) return;
    const float4 *source = reinterpret_cast<const float4 *>(
        x + static_cast<size_t>(rows.ids[blockIdx.x]) * (groups * 16) + group * 16);
    uint32_t *row_q = xq + static_cast<size_t>(blockIdx.x) * words_per_row;
    float *row_scale = xscale + static_cast<size_t>(blockIdx.x) * groups;
    float values[16];
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float4 value = __ldg(source + i);
        values[4 * i + 0] = value.x;
        values[4 * i + 1] = value.y;
        values[4 * i + 2] = value.z;
        values[4 * i + 3] = value.w;
        maximum = fmaxf(maximum, fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                                       fmaxf(fabsf(value.z), fabsf(value.w))));
    }
    const float inverse = maximum > 0.0f ? 127.0f / maximum : 0.0f;
    row_scale[group] = maximum * (1.0f / 127.0f);
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        uint32_t packed = 0;
#pragma unroll
        for (int byte = 0; byte < 4; ++byte)
            packed |= uint32_t(uint8_t(__float2int_rn(values[word * 4 + byte] * inverse))) << (byte * 8);
        row_q[group * 4 + word] = packed;
    }
}

__global__ __launch_bounds__(256, 1) void quantize_swiglu_x16_rows_kernel(
    const float *__restrict__ gate,
    const float *__restrict__ up,
    uint32_t *__restrict__ xq,
    float *__restrict__ xscale,
    int groups,
    int words_per_row,
    Nvfp4RowIds rows) {
    const int group = threadIdx.x;
    if (group >= groups) return;
    const size_t base = static_cast<size_t>(rows.ids[blockIdx.x]) * (groups * 16);
    const float4 *gate4 = reinterpret_cast<const float4 *>(gate + base + group * 16);
    const float4 *up4 = reinterpret_cast<const float4 *>(up + base + group * 16);
    uint32_t *row_q = xq + static_cast<size_t>(blockIdx.x) * words_per_row;
    float *row_scale = xscale + static_cast<size_t>(blockIdx.x) * groups;
    float values[16];
    float maximum = 0.0f;
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        const float4 g = __ldg(gate4 + word);
        const float4 u = __ldg(up4 + word);
        const float gv[4] = {g.x, g.y, g.z, g.w};
        const float uv[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float gc = fminf(gv[i], 10.0f);
            const float uc = fminf(fmaxf(uv[i], -10.0f), 10.0f);
            const float value = (gc / (1.0f + __expf(-gc))) * uc;
            values[word * 4 + i] = value;
            maximum = fmaxf(maximum, fabsf(value));
        }
    }
    const float inverse = maximum > 0.0f ? 127.0f / maximum : 0.0f;
    row_scale[group] = maximum * (1.0f / 127.0f);
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        uint32_t packed = 0;
#pragma unroll
        for (int byte = 0; byte < 4; ++byte)
            packed |= uint32_t(uint8_t(__float2int_rn(values[word * 4 + byte] * inverse))) << (byte * 8);
        row_q[group * 4 + word] = packed;
    }
}

__global__ __launch_bounds__(256) void nvfp4_dp4a_rows_kernel(
    const uint32_t *__restrict__ weights,
    const uint8_t *__restrict__ scales,
    const uint32_t *__restrict__ xq,        // [rows][words_per_row]
    const float *__restrict__ xscale,       // [rows][groups]
    float *__restrict__ y,
    int rows,
    int groups,
    int words_per_row,
    float global_scale,
    Nvfp4RowOut out) {
    __shared__ unsigned long long table[256];
    const int code = threadIdx.x;
    const uint32_t lo = uint32_t(uint8_t(c_e2i[code & 15])) * 0x01010101u;
    const uint32_t hi = uint32_t(uint8_t(c_e2i[code >> 4])) * 0x01010101u;
    table[code] = static_cast<unsigned long long>(lo) | (static_cast<unsigned long long>(hi) << 32);
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_weights = weights + static_cast<size_t>(row) * groups * 2;
    const uint8_t *row_scales = scales + static_cast<size_t>(row) * groups;
    float sums[8] = {};
#pragma unroll 8
    for (int group = lane; group < groups; group += 32) {
        const uint2 packed = __ldcs(reinterpret_cast<const uint2 *>(row_weights + group * 2));
        uint32_t w0, w1, w2, w3;
        unpack_e2(packed.x, table, w0, w1);
        unpack_e2(packed.y, table, w2, w3);
        const float base_scale = 0.5f * c_e4m3[row_scales[group]] * global_scale;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= out.count) break;
            const uint32_t *xg = xq + static_cast<size_t>(r) * words_per_row + group * 4;
            int dot = __dp4a(int(w0), int(xg[0]), 0);
            dot = __dp4a(int(w1), int(xg[1]), dot);
            dot = __dp4a(int(w2), int(xg[2]), dot);
            dot = __dp4a(int(w3), int(xg[3]), dot);
            sums[r] = fmaf(float(dot), base_scale * xscale[static_cast<size_t>(r) * groups + group],
                           sums[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        if (r >= out.count) break;
        const float sum = warp_sum(sums[r]);
        if (!lane) y[static_cast<size_t>(out.ids[r]) * rows + row] = sum;
    }
}

__global__ __launch_bounds__(256) void nvfp4_dp4a_acc_rows_kernel(
    const uint32_t *__restrict__ weights,
    const uint8_t *__restrict__ scales,
    const uint32_t *__restrict__ xq,
    const float *__restrict__ xscale,
    float *__restrict__ y,
    int rows,
    int groups,
    int words_per_row,
    float global_scale,
    Nvfp4RowOut out) {
    __shared__ unsigned long long table[256];
    const int code = threadIdx.x;
    const uint32_t lo = uint32_t(uint8_t(c_e2i[code & 15])) * 0x01010101u;
    const uint32_t hi = uint32_t(uint8_t(c_e2i[code >> 4])) * 0x01010101u;
    table[code] = static_cast<unsigned long long>(lo) | (static_cast<unsigned long long>(hi) << 32);
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_weights = weights + static_cast<size_t>(row) * groups * 2;
    const uint8_t *row_scales = scales + static_cast<size_t>(row) * groups;
    float sums[8] = {};
#pragma unroll 8
    for (int group = lane; group < groups; group += 32) {
        const uint2 packed = __ldcs(reinterpret_cast<const uint2 *>(row_weights + group * 2));
        uint32_t w0, w1, w2, w3;
        unpack_e2(packed.x, table, w0, w1);
        unpack_e2(packed.y, table, w2, w3);
        const float base_scale = 0.5f * c_e4m3[row_scales[group]] * global_scale;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= out.count) break;
            const uint32_t *xg = xq + static_cast<size_t>(r) * words_per_row + group * 4;
            int dot = __dp4a(int(w0), int(xg[0]), 0);
            dot = __dp4a(int(w1), int(xg[1]), dot);
            dot = __dp4a(int(w2), int(xg[2]), dot);
            dot = __dp4a(int(w3), int(xg[3]), dot);
            sums[r] = fmaf(float(dot), base_scale * xscale[static_cast<size_t>(r) * groups + group],
                           sums[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        if (r >= out.count) break;
        const float sum = warp_sum(sums[r]);
        if (!lane) {
            float *slot = y + static_cast<size_t>(out.ids[r]) * rows;
            slot[row] = fmaf(sum, out.weights[r], slot[row]);
        }
    }
}

__global__ __launch_bounds__(256) void nvfp4_dp4a_pair_rows_kernel(
    const uint32_t *__restrict__ weights_a,
    const uint8_t *__restrict__ scales_a,
    const uint32_t *__restrict__ weights_b,
    const uint8_t *__restrict__ scales_b,
    const uint32_t *__restrict__ xq,
    const float *__restrict__ xscale,
    float *__restrict__ y_a,
    float *__restrict__ y_b,
    int rows,
    int groups,
    int words_per_row,
    float global_a,
    float global_b,
    Nvfp4RowIds out) {
    __shared__ unsigned long long table[256];
    const int code = threadIdx.x;
    const uint32_t lo = uint32_t(uint8_t(c_e2i[code & 15])) * 0x01010101u;
    const uint32_t hi = uint32_t(uint8_t(c_e2i[code >> 4])) * 0x01010101u;
    table[code] = static_cast<unsigned long long>(lo) | (static_cast<unsigned long long>(hi) << 32);
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_a = weights_a + static_cast<size_t>(row) * groups * 2;
    const uint32_t *row_b = weights_b + static_cast<size_t>(row) * groups * 2;
    const uint8_t *scale_a = scales_a + static_cast<size_t>(row) * groups;
    const uint8_t *scale_b = scales_b + static_cast<size_t>(row) * groups;
    float sums_a[8] = {}, sums_b[8] = {};
#pragma unroll 8
    for (int group = lane; group < groups; group += 32) {
        const uint2 packed_a = __ldcs(reinterpret_cast<const uint2 *>(row_a + group * 2));
        const uint2 packed_b = __ldcs(reinterpret_cast<const uint2 *>(row_b + group * 2));
        uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
        unpack_e2(packed_a.x, table, a0, a1);
        unpack_e2(packed_a.y, table, a2, a3);
        unpack_e2(packed_b.x, table, b0, b1);
        unpack_e2(packed_b.y, table, b2, b3);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= out.count) break;
            const uint32_t *xg = xq + static_cast<size_t>(r) * words_per_row + group * 4;
            const float activation_scale = 0.5f * xscale[static_cast<size_t>(r) * groups + group];
            int dot_a = __dp4a(int(a0), int(xg[0]), 0);
            dot_a = __dp4a(int(a1), int(xg[1]), dot_a);
            dot_a = __dp4a(int(a2), int(xg[2]), dot_a);
            dot_a = __dp4a(int(a3), int(xg[3]), dot_a);
            int dot_b = __dp4a(int(b0), int(xg[0]), 0);
            dot_b = __dp4a(int(b1), int(xg[1]), dot_b);
            dot_b = __dp4a(int(b2), int(xg[2]), dot_b);
            dot_b = __dp4a(int(b3), int(xg[3]), dot_b);
            sums_a[r] = fmaf(float(dot_a),
                             activation_scale * c_e4m3[scale_a[group]] * global_a, sums_a[r]);
            sums_b[r] = fmaf(float(dot_b),
                             activation_scale * c_e4m3[scale_b[group]] * global_b, sums_b[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        if (r >= out.count) break;
        const float sum_a = warp_sum(sums_a[r]);
        const float sum_b = warp_sum(sums_b[r]);
        if (!lane) {
            y_a[static_cast<size_t>(out.ids[r]) * rows + row] = sum_a;
            y_b[static_cast<size_t>(out.ids[r]) * rows + row] = sum_b;
        }
    }
}

__global__ __launch_bounds__(256, 1) void quantize_x64_kernel(
    const float *__restrict__ x,
    uint32_t *__restrict__ xq,
    float *__restrict__ xscale) {
    const int group = threadIdx.x >> 2;
    const int quarter = threadIdx.x & 3;
    const float4 *source = reinterpret_cast<const float4 *>(x + group * 64 + quarter * 16);
    float values[16];
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float4 value = __ldg(source + i);
        values[4 * i + 0] = value.x;
        values[4 * i + 1] = value.y;
        values[4 * i + 2] = value.z;
        values[4 * i + 3] = value.w;
        maximum = fmaxf(maximum, fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                                       fmaxf(fabsf(value.z), fabsf(value.w))));
    }
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 2));
    const float inverse = maximum > 0.0f ? 127.0f / maximum : 0.0f;
    if (!quarter) xscale[group] = maximum * (1.0f / 127.0f);
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        uint32_t packed = 0;
#pragma unroll
        for (int byte = 0; byte < 4; ++byte)
            packed |= uint32_t(uint8_t(__float2int_rn(values[word * 4 + byte] * inverse))) << (byte * 8);
        xq[group * 16 + quarter * 4 + word] = packed;
    }
}

__device__ __forceinline__ uint32_t sign_extend_i4(uint32_t values) {
    const uint32_t sign = values & 0x08080808u;
    return values | (sign << 1) | (sign << 2) | (sign << 3) | (sign << 4);
}

__device__ __forceinline__ void unpack_i4(uint32_t packed, uint32_t &first, uint32_t &second) {
    const uint32_t even = packed & 0x0f0f0f0fu;
    const uint32_t odd = (packed >> 4) & 0x0f0f0f0fu;
    first = sign_extend_i4(__byte_perm(even, odd, 0x5140));
    second = sign_extend_i4(__byte_perm(even, odd, 0x7362));
}

template <bool E2M1_CODES>
__global__ __launch_bounds__(256) void grouped_i4_dp4a_kernel(
    const uint32_t *__restrict__ weights,
    const __half *__restrict__ scales,
    const uint32_t *__restrict__ xq,
    const float *__restrict__ xscale,
    float *__restrict__ y,
    int rows,
    int groups) {
    __shared__ unsigned long long table[256];
    if constexpr (E2M1_CODES) {
        const int code = threadIdx.x;
        const uint32_t lo = uint32_t(uint8_t(c_e2i[code & 15])) * 0x01010101u;
        const uint32_t hi = uint32_t(uint8_t(c_e2i[code >> 4])) * 0x01010101u;
        table[code] = static_cast<unsigned long long>(lo) | (static_cast<unsigned long long>(hi) << 32);
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_weights = weights + static_cast<size_t>(row) * groups * 8;
    const __half *row_scales = scales + static_cast<size_t>(row) * groups;
    float sum = 0.0f;
#pragma unroll 2
    for (int group = lane; group < groups; group += 32) {
        const uint4 first_pack = __ldcs(reinterpret_cast<const uint4 *>(row_weights + group * 8));
        const uint4 second_pack = __ldcs(reinterpret_cast<const uint4 *>(row_weights + group * 8 + 4));
        const uint32_t packed[8] = {
            first_pack.x, first_pack.y, first_pack.z, first_pack.w,
            second_pack.x, second_pack.y, second_pack.z, second_pack.w,
        };
        const uint32_t *xg = xq + group * 16;
        int dot = 0;
#pragma unroll
        for (int word = 0; word < 8; ++word) {
            uint32_t w0, w1;
            if constexpr (E2M1_CODES)
                unpack_e2(packed[word], table, w0, w1);
            else
                unpack_i4(packed[word], w0, w1);
            dot = __dp4a(int(w0), int(xg[word * 2]), dot);
            dot = __dp4a(int(w1), int(xg[word * 2 + 1]), dot);
        }
        float scale = __half2float(row_scales[group]) * xscale[group];
        if constexpr (E2M1_CODES) scale *= 0.5f;
        sum = fmaf(float(dot), scale, sum);
    }
    sum = warp_sum(sum);
    if (!lane) y[row] = sum;
}

struct Metrics {
    double relative_l2;
    double cosine;
    double max_abs;
};

Metrics compare(const std::vector<float> &actual, const float *reference) {
    double dot = 0.0, aa = 0.0, rr = 0.0, error = 0.0, maximum = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], r = reference[i], d = a - r;
        dot += a * r;
        aa += a * a;
        rr += r * r;
        error += d * d;
        maximum = fmax(maximum, fabs(d));
    }
    return {sqrt(error / rr), dot / sqrt(aa * rr), maximum};
}

template <typename Launch>
float benchmark(Launch launch, int iterations = 2000) {
    for (int i = 0; i < 50; ++i) launch();
    cuda_check(cudaDeviceSynchronize(), "warmup");
    cudaEvent_t begin, end;
    cuda_check(cudaEventCreate(&begin), "cudaEventCreate");
    cuda_check(cudaEventCreate(&end), "cudaEventCreate");
    cuda_check(cudaEventRecord(begin), "cudaEventRecord");
    for (int i = 0; i < iterations; ++i) launch();
    cuda_check(cudaEventRecord(end), "cudaEventRecord");
    cuda_check(cudaEventSynchronize(end), "cudaEventSynchronize");
    float elapsed = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed, begin, end), "cudaEventElapsedTime");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return elapsed / iterations;
}

void print_result(const char *name, float milliseconds, uint64_t bytes, const Metrics &metrics) {
    const double bandwidth = (double(bytes) / 1.0e9) / (double(milliseconds) / 1.0e3);
    std::printf("%-23s %8.3f us  %7.1f GB/s  rel=%9.6f cos=%.9f max=%g\n",
                name, milliseconds * 1000.0f, bandwidth,
                metrics.relative_l2, metrics.cosine, metrics.max_abs);
}

// One thread expands one packed byte (two output scale codes). The host
// supplies an exclusive escape count at every 256-byte block boundary; an
// in-block scan recovers the exact global escape cursor without serializing
// the 512 KiB output. All operations are integer/byte moves. Shared by the
// single-projection and fused three-projection launches so both execute an
// instruction-identical sequence per block.
__device__ __forceinline__ void expand_scale_block_bytes(
    const uint8_t *__restrict__ packed,
    const uint8_t *__restrict__ escapes,
    const uint8_t *__restrict__ codebook,
    const uint32_t *__restrict__ block_prefix,
    uint8_t *__restrict__ output, unsigned block) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const size_t packed_index = size_t(block) * blockDim.x + threadIdx.x;
    const uint8_t byte = packed[packed_index];
    const uint8_t low = byte & 15u;
    const uint8_t high = byte >> 4;
    const uint32_t mine = uint32_t(low == 15u) + uint32_t(high == 15u);

    uint32_t inclusive = mine;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const uint32_t prior = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive += prior;
    }
    __shared__ uint32_t warp_totals[8], warp_offsets[8];
    __shared__ uint8_t codes[16];
    if (threadIdx.x < 16) codes[threadIdx.x] = codebook[threadIdx.x];
    if (lane == 31) warp_totals[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
        uint32_t value = lane < 8 ? warp_totals[lane] : 0u;
        uint32_t warp_inclusive = value;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const uint32_t prior = __shfl_up_sync(0xffffffffu, warp_inclusive, offset);
            if (lane >= offset) warp_inclusive += prior;
        }
        if (lane < 8) warp_offsets[lane] = warp_inclusive - value;
    }
    __syncthreads();

    uint32_t escape_at = block_prefix[block] + warp_offsets[warp] + inclusive - mine;
    const uint8_t first = low == 15u ? escapes[escape_at++] : codes[low];
    const uint8_t second = high == 15u ? escapes[escape_at] : codes[high];
    output[2 * packed_index] = first;
    output[2 * packed_index + 1] = second;
}

__global__ __launch_bounds__(256) void expand_scale_nibbles_kernel(
    const uint8_t *__restrict__ packed,
    const uint8_t *__restrict__ escapes,
    const uint8_t *__restrict__ codebook,
    const uint32_t *__restrict__ block_prefix,
    uint8_t *__restrict__ output) {
    expand_scale_block_bytes(packed, escapes, codebook, block_prefix, output,
                             unsigned(blockIdx.x));
}

// v2: warp uint32 expansion batches. One thread owns one packed uint32 word
// (= 8 nibbles = 8 output scale codes = one 64-bit store). 512 KiB output =
// 65536 words = 128 blocks x 512 threads exactly: one 4-byte load and one
// 8-byte store per thread, no tail block, no bounds checks (the wrapper
// enforces the shape and the natural alignments). A 256-entry shared pair
// table answers a whole packed byte per lookup and folds the escape flags
// into bits 16/17, so the per-word escape count is free popc. A block spans
// 512 words = 2048 packed bytes = eight 256-byte host prefix entries, so the
// block base is block_prefix[8*blockIdx.x]. Output is byte-identical to the
// byte-per-thread kernel and the AVX2 sidecar decoder: little-endian nibble
// order puts output byte 8q+n at nibble n of word q, and escape tail bytes
// are consumed strictly in ascending output position.
__global__ __launch_bounds__(512, 3) void expand_scale_nibbles_v2_kernel(
    const uint32_t *__restrict__ packed_words,
    const uint8_t *__restrict__ escapes,
    const uint8_t *__restrict__ codebook,
    const uint32_t *__restrict__ block_prefix,
    uint2 *__restrict__ output_words) {
    __shared__ uint32_t pair_table[256];
    __shared__ uint32_t warp_totals[16], warp_offsets[16];

    const uint32_t t = threadIdx.x;
    if (t < 256u) {
        const uint32_t lo = t & 15u, hi = t >> 4;
        pair_table[t] = uint32_t(codebook[lo]) | (uint32_t(codebook[hi]) << 8) |
                        ((lo == 15u ? 1u : 0u) << 16) | ((hi == 15u ? 1u : 0u) << 17);
    }
    const uint32_t word = blockIdx.x * 512u + t;  // < 65536 by construction
    const uint32_t w = packed_words[word];        // overlaps the table build
    __syncthreads();

    const uint32_t e0 = pair_table[w & 0xffu];
    const uint32_t e1 = pair_table[(w >> 8) & 0xffu];
    const uint32_t e2 = pair_table[(w >> 16) & 0xffu];
    const uint32_t e3 = pair_table[w >> 24];
    const uint32_t mine = __popc(e0 >> 16) + __popc(e1 >> 16) +
                          __popc(e2 >> 16) + __popc(e3 >> 16);

    const uint32_t lane = t & 31u, warp = t >> 5;
    uint32_t inclusive = mine;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const uint32_t prior = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive += prior;
    }
    if (lane == 31u) warp_totals[warp] = inclusive;
    __syncthreads();

    if (!warp) {
        const uint32_t value = lane < 16u ? warp_totals[lane] : 0u;
        uint32_t warp_inclusive = value;
#pragma unroll
        for (int offset = 1; offset < 16; offset <<= 1) {
            const uint32_t prior = __shfl_up_sync(0xffffffffu, warp_inclusive, offset);
            if (lane >= offset) warp_inclusive += prior;
        }
        if (lane < 16u) warp_offsets[lane] = warp_inclusive - value;
    }
    __syncthreads();

    const uint32_t cursor = block_prefix[blockIdx.x * 8u] +
                            warp_offsets[warp] + inclusive - mine;

    uint32_t lo4 = __byte_perm(e0, e1, 0x5410);
    uint32_t hi4 = __byte_perm(e2, e3, 0x5410);
    if (mine) {  // ~6% of threads at the 0.782% production escape rate
        uint32_t mask = ((e0 >> 16) & 3u) | (((e1 >> 16) & 3u) << 2) |
                        (((e2 >> 16) & 3u) << 4) | (((e3 >> 16) & 3u) << 6);
        uint32_t next = 0u;
        do {  // escape tail consumed in ascending nibble order
            const uint32_t n = uint32_t(__ffs(mask)) - 1u;
            const uint32_t byte = uint32_t(escapes[cursor + next++]);
            // selector 0x3210 with nibble (n&3) replaced by pool byte 4
            // (byte 0 of `byte`): +4->0x3214, +0x30->0x3240,
            // +0x200->0x3410, +0x1000->0x4210.
            const uint32_t replace = 0x3210u + ((4u - (n & 3u)) << ((n & 3u) << 2));
            if (n < 4u) lo4 = __byte_perm(lo4, byte, replace);
            else        hi4 = __byte_perm(hi4, byte, replace);
            mask &= mask - 1u;
        } while (mask);
    }
    output_words[word] = make_uint2(lo4, hi4);  // one STG.64, fully coalesced
}

// Fused three-projection launch, byte worker: grid = 3 * blocks_per_proj,
// projection = blockIdx.x / blocks_per_proj. The per-projection blob offsets
// inside the shared device scratch are runtime values (variable escape
// tails), passed as scalar kernel parameters; the branches are uniform per
// block. Per-block arithmetic is identical to the single-projection kernel,
// so the expanded bytes are bit-identical.
__global__ __launch_bounds__(256) void expand_scale_nibbles3_kernel(
    const uint8_t *__restrict__ scratch,
    unsigned p0, unsigned e0, unsigned c0, unsigned x0,
    unsigned p1, unsigned e1, unsigned c1, unsigned x1,
    unsigned p2, unsigned e2, unsigned c2, unsigned x2,
    uint8_t *__restrict__ output_base, size_t output_pitch, unsigned scale_off,
    unsigned blocks_per_proj) {
    const unsigned projection = blockIdx.x / blocks_per_proj;
    const unsigned block = blockIdx.x - projection * blocks_per_proj;
    uint8_t *output = output_base + size_t(projection) * output_pitch + scale_off;
    if (projection == 0)
        expand_scale_block_bytes(scratch + p0, scratch + e0, scratch + c0,
                                 reinterpret_cast<const uint32_t *>(scratch + x0),
                                 output, block);
    else if (projection == 1)
        expand_scale_block_bytes(scratch + p1, scratch + e1, scratch + c1,
                                 reinterpret_cast<const uint32_t *>(scratch + x1),
                                 output, block);
    else
        expand_scale_block_bytes(scratch + p2, scratch + e2, scratch + c2,
                                 reinterpret_cast<const uint32_t *>(scratch + x2),
                                 output, block);
}

// Fused three-projection launch, uint32 worker: same shape at 128 blocks per
// projection (512 threads, 512 words each).
__global__ __launch_bounds__(512, 3) void expand_scale_nibbles3_v2_kernel(
    const uint8_t *__restrict__ scratch,
    unsigned p0, unsigned e0, unsigned c0, unsigned x0,
    unsigned p1, unsigned e1, unsigned c1, unsigned x1,
    unsigned p2, unsigned e2, unsigned c2, unsigned x2,
    uint8_t *__restrict__ output_base, size_t output_pitch, unsigned scale_off,
    unsigned blocks_per_proj) {
    const unsigned projection = blockIdx.x / blocks_per_proj;
    const unsigned block = blockIdx.x - projection * blocks_per_proj;
    const uint8_t *packed = nullptr, *escapes = nullptr, *codebook = nullptr;
    const uint32_t *prefix = nullptr;
    if (projection == 0) {
        packed = scratch + p0; escapes = scratch + e0;
        codebook = scratch + c0; prefix = reinterpret_cast<const uint32_t *>(scratch + x0);
    } else if (projection == 1) {
        packed = scratch + p1; escapes = scratch + e1;
        codebook = scratch + c1; prefix = reinterpret_cast<const uint32_t *>(scratch + x1);
    } else {
        packed = scratch + p2; escapes = scratch + e2;
        codebook = scratch + c2; prefix = reinterpret_cast<const uint32_t *>(scratch + x2);
    }
    __shared__ uint32_t pair_table[256];
    __shared__ uint32_t warp_totals[16], warp_offsets[16];

    const uint32_t t = threadIdx.x;
    if (t < 256u) {
        const uint32_t lo = t & 15u, hi = t >> 4;
        pair_table[t] = uint32_t(codebook[lo]) | (uint32_t(codebook[hi]) << 8) |
                        ((lo == 15u ? 1u : 0u) << 16) | ((hi == 15u ? 1u : 0u) << 17);
    }
    const uint32_t word = block * 512u + t;
    const uint32_t w = reinterpret_cast<const uint32_t *>(packed)[word];
    __syncthreads();

    const uint32_t q0 = pair_table[w & 0xffu];
    const uint32_t q1 = pair_table[(w >> 8) & 0xffu];
    const uint32_t q2 = pair_table[(w >> 16) & 0xffu];
    const uint32_t q3 = pair_table[w >> 24];
    const uint32_t mine = __popc(q0 >> 16) + __popc(q1 >> 16) +
                          __popc(q2 >> 16) + __popc(q3 >> 16);

    const uint32_t lane = t & 31u, warp = t >> 5;
    uint32_t inclusive = mine;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const uint32_t prior = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive += prior;
    }
    if (lane == 31u) warp_totals[warp] = inclusive;
    __syncthreads();
    if (!warp) {
        const uint32_t value = lane < 16u ? warp_totals[lane] : 0u;
        uint32_t warp_inclusive = value;
#pragma unroll
        for (int offset = 1; offset < 16; offset <<= 1) {
            const uint32_t prior = __shfl_up_sync(0xffffffffu, warp_inclusive, offset);
            if (lane >= offset) warp_inclusive += prior;
        }
        if (lane < 16u) warp_offsets[lane] = warp_inclusive - value;
    }
    __syncthreads();

    const uint32_t cursor = prefix[block * 8u] +
                            warp_offsets[warp] + inclusive - mine;

    uint32_t lo4 = __byte_perm(q0, q1, 0x5410);
    uint32_t hi4 = __byte_perm(q2, q3, 0x5410);
    if (mine) {
        uint32_t mask = ((q0 >> 16) & 3u) | (((q1 >> 16) & 3u) << 2) |
                        (((q2 >> 16) & 3u) << 4) | (((q3 >> 16) & 3u) << 6);
        uint32_t next = 0u;
        do {
            const uint32_t n = uint32_t(__ffs(mask)) - 1u;
            const uint32_t byte = uint32_t(escapes[cursor + next++]);
            const uint32_t replace = 0x3210u + ((4u - (n & 3u)) << ((n & 3u) << 2));
            if (n < 4u) lo4 = __byte_perm(lo4, byte, replace);
            else        hi4 = __byte_perm(hi4, byte, replace);
            mask &= mask - 1u;
        } while (mask);
    }
    reinterpret_cast<uint2 *>(output_base + size_t(projection) * output_pitch +
                              scale_off)[word] = make_uint2(lo4, hi4);
}

}  // namespace

namespace insignia::glm53 {

size_t nvfp4_workspace_bytes(int cols) {
    if (cols <= 0 || (cols & 15) || cols > 4096) return 0;
    const size_t quantized = static_cast<size_t>(cols);
    const size_t aligned = (quantized + 255) & ~size_t(255);
    return aligned + static_cast<size_t>(cols / 16) * sizeof(float);
}

cudaError_t initialize_nvfp4() {
    float e4m3[256];
    for (int code = 0; code < 256; ++code) {
        const int exponent = (code >> 3) & 15;
        const int mantissa = code & 7;
        float value = exponent ? std::ldexp(float(8 + mantissa), exponent - 10)
                               : float(mantissa) * std::ldexp(1.0f, -9);
        e4m3[code] = code & 128 ? -value : value;
    }
    const int8_t e2i[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    cudaError_t status = cudaMemcpyToSymbol(c_e4m3, e4m3, sizeof(e4m3));
    if (status != cudaSuccess) return status;
    return cudaMemcpyToSymbol(c_e2i, e2i, sizeof(e2i));
}

cudaError_t expand_nvfp4_scale_nibbles(
    const uint8_t *packed, const uint8_t *escapes, const uint8_t *codebook,
    const uint32_t *block_prefix, uint8_t *output, size_t output_bytes,
    cudaStream_t stream) {
    constexpr size_t packed_per_block = 256;
    if (!packed || !escapes || !codebook || !block_prefix || !output ||
        !output_bytes || (output_bytes & 1) ||
        (output_bytes / 2) % packed_per_block)
        return cudaErrorInvalidValue;
    const size_t blocks = output_bytes / 2 / packed_per_block;
    expand_scale_nibbles_kernel<<<unsigned(blocks), 256, 0, stream>>>(
        packed, escapes, codebook, block_prefix, output);
    return cudaPeekAtLastError();
}

cudaError_t expand_nvfp4_scale_nibbles_v2(
    const uint8_t *packed, const uint8_t *escapes, const uint8_t *codebook,
    const uint32_t *block_prefix, uint8_t *output, size_t output_bytes,
    cudaStream_t stream) {
    constexpr size_t kOutputPerBlock = 512u * 8u;  // 512 words x 8 codes
    if (!packed || !escapes || !codebook || !block_prefix || !output ||
        output_bytes < kOutputPerBlock || output_bytes % kOutputPerBlock ||
        reinterpret_cast<uintptr_t>(packed) & 3 ||
        reinterpret_cast<uintptr_t>(block_prefix) & 3 ||
        reinterpret_cast<uintptr_t>(output) & 7)
        return cudaErrorInvalidValue;
    const size_t blocks = output_bytes / kOutputPerBlock;  // 128 for 512 KiB
    expand_scale_nibbles_v2_kernel<<<unsigned(blocks), 512, 0, stream>>>(
        reinterpret_cast<const uint32_t *>(packed), escapes, codebook,
        block_prefix, reinterpret_cast<uint2 *>(output));
    return cudaPeekAtLastError();
}

namespace {

// Packs the nine per-projection scratch offsets into the scalar kernel
// parameter lists of the fused launches (arrays in kernel parameter space
// would need dynamic indexing; scalars stay in the parameter registers).
struct FusedOffsets {
    unsigned packed[3], escapes[3], codebooks[3], prefixes[3];
};

}  // namespace

cudaError_t expand_nvfp4_scale_nibbles3(
    const uint8_t *scratch, const size_t *packed_offsets,
    const size_t *escape_offsets, const size_t *codebook_offsets,
    const size_t *prefix_offsets, uint8_t *output_base, size_t output_pitch,
    size_t scale_offset, size_t projection_bytes, cudaStream_t stream) {
    constexpr size_t packed_per_block = 256;
    if (!scratch || !packed_offsets || !escape_offsets || !codebook_offsets ||
        !prefix_offsets || !output_base || !projection_bytes ||
        (projection_bytes & 1) || (projection_bytes / 2) % packed_per_block)
        return cudaErrorInvalidValue;
    const unsigned blocks_per_proj = unsigned(projection_bytes / 2 / packed_per_block);
    const FusedOffsets off = {{unsigned(packed_offsets[0]), unsigned(packed_offsets[1]), unsigned(packed_offsets[2])},
                              {unsigned(escape_offsets[0]), unsigned(escape_offsets[1]), unsigned(escape_offsets[2])},
                              {unsigned(codebook_offsets[0]), unsigned(codebook_offsets[1]), unsigned(codebook_offsets[2])},
                              {unsigned(prefix_offsets[0]), unsigned(prefix_offsets[1]), unsigned(prefix_offsets[2])}};
    expand_scale_nibbles3_kernel<<<3u * blocks_per_proj, 256, 0, stream>>>(
        scratch,
        off.packed[0], off.escapes[0], off.codebooks[0], off.prefixes[0],
        off.packed[1], off.escapes[1], off.codebooks[1], off.prefixes[1],
        off.packed[2], off.escapes[2], off.codebooks[2], off.prefixes[2],
        output_base, output_pitch, unsigned(scale_offset), blocks_per_proj);
    return cudaPeekAtLastError();
}

cudaError_t expand_nvfp4_scale_nibbles3_v2(
    const uint8_t *scratch, const size_t *packed_offsets,
    const size_t *escape_offsets, const size_t *codebook_offsets,
    const size_t *prefix_offsets, uint8_t *output_base, size_t output_pitch,
    size_t scale_offset, size_t projection_bytes, cudaStream_t stream) {
    constexpr size_t kOutputPerBlock = 512u * 8u;
    if (!scratch || !packed_offsets || !escape_offsets || !codebook_offsets ||
        !prefix_offsets || !output_base ||
        projection_bytes < kOutputPerBlock || projection_bytes % kOutputPerBlock)
        return cudaErrorInvalidValue;
    for (int projection = 0; projection < 3; ++projection)
        if ((reinterpret_cast<uintptr_t>(scratch + packed_offsets[projection]) & 3) ||
            (reinterpret_cast<uintptr_t>(scratch + prefix_offsets[projection]) & 3) ||
            (reinterpret_cast<uintptr_t>(output_base + size_t(projection) * output_pitch +
                                         scale_offset) & 7))
            return cudaErrorInvalidValue;
    const FusedOffsets off = {{unsigned(packed_offsets[0]), unsigned(packed_offsets[1]), unsigned(packed_offsets[2])},
                              {unsigned(escape_offsets[0]), unsigned(escape_offsets[1]), unsigned(escape_offsets[2])},
                              {unsigned(codebook_offsets[0]), unsigned(codebook_offsets[1]), unsigned(codebook_offsets[2])},
                              {unsigned(prefix_offsets[0]), unsigned(prefix_offsets[1]), unsigned(prefix_offsets[2])}};
    expand_scale_nibbles3_v2_kernel<<<3u * unsigned(projection_bytes / kOutputPerBlock),
                                     512, 0, stream>>>(
        scratch,
        off.packed[0], off.escapes[0], off.codebooks[0], off.prefixes[0],
        off.packed[1], off.escapes[1], off.codebooks[1], off.prefixes[1],
        off.packed[2], off.escapes[2], off.codebooks[2], off.prefixes[2],
        output_base, output_pitch, unsigned(scale_offset),
        unsigned(projection_bytes / kOutputPerBlock));
    return cudaPeekAtLastError();
}

cudaError_t nvfp4_gemv_f32(
    const uint8_t *weights, const uint8_t *scales, float global_scale,
    const float *x, float *y, int rows, int cols, cudaStream_t stream) {
    if (!weights || !scales || !x || !y || rows <= 0 || cols <= 0 || (cols & 15) || cols > 4096)
        return cudaErrorInvalidValue;
    nvfp4_f32_kernel<<<rows, 256, 0, stream>>>(weights, scales, x, y, rows, cols, global_scale);
    return cudaPeekAtLastError();
}

cudaError_t nvfp4_gemv_dp4a(
    const uint8_t *weights, const uint8_t *scales, float global_scale,
    const float *x, float *y, int rows, int cols, void *workspace, cudaStream_t stream) {
    const size_t bytes = nvfp4_workspace_bytes(cols);
    if (!bytes || !weights || !scales || !x || !y || !workspace || rows <= 0)
        return cudaErrorInvalidValue;
    cudaError_t status = nvfp4_quantize_activation(x, cols, workspace, stream);
    if (status != cudaSuccess) return status;
    return nvfp4_gemv_dp4a_quantized(weights, scales, global_scale, workspace, y, rows, cols, stream);
}

cudaError_t nvfp4_quantize_activation(const float *x, int cols, void *workspace, cudaStream_t stream) {
    const size_t bytes = nvfp4_workspace_bytes(cols);
    if (!bytes || !x || !workspace) return cudaErrorInvalidValue;
    auto *xq = reinterpret_cast<uint32_t *>(workspace);
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    auto *xscale = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(workspace) + aligned);
    quantize_x16_kernel<<<1, 256, 0, stream>>>(x, xq, xscale, cols / 16);
    return cudaPeekAtLastError();
}

cudaError_t nvfp4_gemv_dp4a_quantized(
    const uint8_t *weights, const uint8_t *scales, float global_scale,
    const void *workspace, float *y, int rows, int cols, cudaStream_t stream) {
    const size_t bytes = nvfp4_workspace_bytes(cols);
    if (!bytes || !weights || !scales || !workspace || !y || rows <= 0) return cudaErrorInvalidValue;
    const auto *xq = reinterpret_cast<const uint32_t *>(workspace);
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    const auto *xscale = reinterpret_cast<const float *>(reinterpret_cast<const uint8_t *>(workspace) + aligned);
    const int groups = cols / 16;
    nvfp4_dp4a_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t *>(weights), scales, xq, xscale,
        y, rows, groups, global_scale);
    return cudaPeekAtLastError();
}

cudaError_t nvfp4_gemv_dp4a_acc_quantized(
    const uint8_t *weights, const uint8_t *scales, float global_scale,
    const void *workspace, float *y, float combine_weight,
    int rows, int cols, cudaStream_t stream) {
    const size_t bytes = nvfp4_workspace_bytes(cols);
    if (!bytes || !weights || !scales || !workspace || !y || rows <= 0) return cudaErrorInvalidValue;
    const auto *xq = reinterpret_cast<const uint32_t *>(workspace);
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    const auto *xscale = reinterpret_cast<const float *>(reinterpret_cast<const uint8_t *>(workspace) + aligned);
    const int groups = cols / 16;
    nvfp4_dp4a_acc_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t *>(weights), scales, xq, xscale,
        y, combine_weight, rows, groups, global_scale);
    return cudaPeekAtLastError();
}

cudaError_t nvfp4_gemv2_dp4a_quantized(
    const uint8_t *weights_a, const uint8_t *scales_a, float global_scale_a,
    const uint8_t *weights_b, const uint8_t *scales_b, float global_scale_b,
    const void *workspace, float *y_a, float *y_b, int rows, int cols, cudaStream_t stream) {
    const size_t bytes = nvfp4_workspace_bytes(cols);
    if (!bytes || !weights_a || !scales_a || !weights_b || !scales_b || !workspace ||
        !y_a || !y_b || rows <= 0) return cudaErrorInvalidValue;
    const auto *xq = reinterpret_cast<const uint32_t *>(workspace);
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    const auto *xscale = reinterpret_cast<const float *>(reinterpret_cast<const uint8_t *>(workspace) + aligned);
    nvfp4_dp4a_pair_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t *>(weights_a), scales_a,
        reinterpret_cast<const uint32_t *>(weights_b), scales_b,
        xq, xscale, y_a, y_b, rows, cols / 16, global_scale_a, global_scale_b);
    return cudaPeekAtLastError();
}

cudaError_t quantize_swiglu_activation(
    const float *gate, const float *up, int cols, void *workspace, cudaStream_t stream) {
    const size_t bytes = nvfp4_workspace_bytes(cols);
    if (!bytes || !gate || !up || !workspace) return cudaErrorInvalidValue;
    auto *xq = reinterpret_cast<uint32_t *>(workspace);
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    auto *xscale = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(workspace) + aligned);
    quantize_swiglu_x16_kernel<<<1, 256, 0, stream>>>(gate, up, xq, xscale, cols / 16);
    return cudaPeekAtLastError();
}

size_t nvfp4_workspace_rows_bytes(int cols, int rows) {
    return nvfp4_workspace_bytes(cols) * static_cast<size_t>(rows);
}

cudaError_t nvfp4_quantize_activation_rows(
    const float *x, int cols, const int *row_ids, int count, void *workspace,
    cudaStream_t stream) {
    const size_t bytes = nvfp4_workspace_bytes(cols);
    if (!bytes || !x || !row_ids || !workspace || count <= 0 || count > 8)
        return cudaErrorInvalidValue;
    Nvfp4RowIds batch{};
    batch.count = count;
    for (int r = 0; r < count; ++r) batch.ids[r] = row_ids[r];
    auto *xq = reinterpret_cast<uint32_t *>(workspace);
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    auto *xscale = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(workspace) +
                                             count * aligned);
    quantize_x16_rows_kernel<<<count, 256, 0, stream>>>(
        x, xq, xscale, cols / 16, int(aligned / 4), batch);
    return cudaPeekAtLastError();
}

cudaError_t quantize_swiglu_activation_rows(
    const float *gate, const float *up, int cols, const int *row_ids, int count,
    void *workspace, cudaStream_t stream) {
    const size_t bytes = nvfp4_workspace_bytes(cols);
    if (!bytes || !gate || !up || !row_ids || !workspace || count <= 0 || count > 8)
        return cudaErrorInvalidValue;
    Nvfp4RowIds batch{};
    batch.count = count;
    for (int r = 0; r < count; ++r) batch.ids[r] = row_ids[r];
    auto *xq = reinterpret_cast<uint32_t *>(workspace);
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    auto *xscale = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(workspace) +
                                             count * aligned);
    quantize_swiglu_x16_rows_kernel<<<count, 256, 0, stream>>>(
        gate, up, xq, xscale, cols / 16, int(aligned / 4), batch);
    return cudaPeekAtLastError();
}

cudaError_t nvfp4_gemv_dp4a_quantized_rows(
    const uint8_t *weights, const uint8_t *scales, float global_scale,
    const void *workspace, int count, float *y, const int *y_ids,
    int rows, int cols, cudaStream_t stream) {
    const size_t bytes = nvfp4_workspace_bytes(cols);
    if (!bytes || !weights || !scales || !workspace || !y || !y_ids ||
        rows <= 0 || count <= 0 || count > 8) return cudaErrorInvalidValue;
    Nvfp4RowOut out{};
    out.count = count;
    for (int r = 0; r < count; ++r) out.ids[r] = y_ids[r];
    const auto *xq = reinterpret_cast<const uint32_t *>(workspace);
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    const auto *xscale = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(workspace) + count * aligned);
    nvfp4_dp4a_rows_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t *>(weights), scales, xq, xscale,
        y, rows, cols / 16, int(aligned / 4), global_scale, out);
    return cudaPeekAtLastError();
}

cudaError_t nvfp4_gemv_dp4a_acc_quantized_rows(
    const uint8_t *weights, const uint8_t *scales, float global_scale,
    const void *workspace, int count, float *y, const int *y_ids, const float *combine,
    int rows, int cols, cudaStream_t stream) {
    const size_t bytes = nvfp4_workspace_bytes(cols);
    if (!bytes || !weights || !scales || !workspace || !y || !y_ids || !combine ||
        rows <= 0 || count <= 0 || count > 8) return cudaErrorInvalidValue;
    Nvfp4RowOut out{};
    out.count = count;
    for (int r = 0; r < count; ++r) {
        out.ids[r] = y_ids[r];
        out.weights[r] = combine[r];
    }
    const auto *xq = reinterpret_cast<const uint32_t *>(workspace);
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    const auto *xscale = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(workspace) + count * aligned);
    nvfp4_dp4a_acc_rows_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t *>(weights), scales, xq, xscale,
        y, rows, cols / 16, int(aligned / 4), global_scale, out);
    return cudaPeekAtLastError();
}

cudaError_t nvfp4_gemv2_dp4a_quantized_rows(
    const uint8_t *weights_a, const uint8_t *scales_a, float global_scale_a,
    const uint8_t *weights_b, const uint8_t *scales_b, float global_scale_b,
    const void *workspace, int count, float *y_a, float *y_b, const int *y_ids,
    int rows, int cols, cudaStream_t stream) {
    const size_t bytes = nvfp4_workspace_bytes(cols);
    if (!bytes || !weights_a || !scales_a || !weights_b || !scales_b || !workspace ||
        !y_a || !y_b || !y_ids || rows <= 0 || count <= 0 || count > 8)
        return cudaErrorInvalidValue;
    Nvfp4RowIds out{};
    out.count = count;
    for (int r = 0; r < count; ++r) out.ids[r] = y_ids[r];
    const auto *xq = reinterpret_cast<const uint32_t *>(workspace);
    const size_t aligned = (static_cast<size_t>(cols) + 255) & ~size_t(255);
    const auto *xscale = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(workspace) + count * aligned);
    nvfp4_dp4a_pair_rows_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t *>(weights_a), scales_a,
        reinterpret_cast<const uint32_t *>(weights_b), scales_b,
        xq, xscale, y_a, y_b, rows, cols / 16, int(aligned / 4),
        global_scale_a, global_scale_b, out);
    return cudaPeekAtLastError();
}

}  // namespace insignia::glm53

#ifndef INSIGNIA_GLM53_NO_MAIN

// ---------------------------------------------------------------------------
// --expand mode: synthesized packed-scale-plane expansion microbenchmark.
// For each escape rate, draws a 512 KiB code stream (nibble 15 = escape),
// packs it 2:1, builds the escape tail + per-256-byte block prefix exactly
// like ExpertStager (src/glm53_generate.cu), then measures the GPU expand
// kernels (byte worker, warp uint32 worker, fused 3-projection variants),
// the full host staging decode, the pinned H2D transports of both payload
// shapes, the record-level v1 vs v2 enqueue mixes, and the empty-launch
// floor. All synthesis is splitmix64 seeded and byte-for-byte reproducible.
// ---------------------------------------------------------------------------
namespace {

constexpr size_t kPackedScaleBytes = 256ull << 10;                  // per projection
constexpr size_t kExpandedScaleBytes = 512ull << 10;                // per projection
constexpr size_t kScalePrefixEntries = kPackedScaleBytes / 256 + 1; // 1025
constexpr int kLaunchCostLaunches = 400;
constexpr int kDecodeReps = 200;
constexpr int kMixReps = 200;
constexpr size_t kRecordSlotPitch = (4ull << 20) + (512ull << 10);  // body + scale

struct SplitMix64 {
    uint64_t state;
    uint64_t next() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
};

struct SynthScales {
    std::vector<uint8_t> packed;    // 256 KiB, nibble 15 = escape
    std::vector<uint8_t> escapes;   // escape tail, consumed low nibble first
    std::vector<uint8_t> codebook;  // 16 distinct bytes, index 15 unused
    std::vector<uint32_t> prefix;   // 1025 exclusive per-block escape counts
    std::vector<uint8_t> truth;     // 512 KiB ground-truth expansion
    std::vector<uint8_t> blob;      // packed|escapes|codebook|pad4|prefix
    size_t escape_count = 0;
};

double bandwidth_gbps(double bytes, double microseconds) {
    return (bytes / 1.0e9) / (microseconds / 1.0e6);
}

#if defined(__AVX2__)
uint32_t count_escape_nibbles_256(const uint8_t *packed) {
    const __m256i nibble_mask = _mm256_set1_epi8(15);
    const __m256i escape_code = _mm256_set1_epi8(15);
    uint32_t count = 0;
    for (size_t offset = 0; offset < 256; offset += 32) {
        const __m256i chunk =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(packed + offset));
        const __m256i low = _mm256_and_si256(chunk, nibble_mask);
        const __m256i high = _mm256_and_si256(_mm256_srli_epi16(chunk, 4), nibble_mask);
        count += uint32_t(__builtin_popcount(uint32_t(_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(low, escape_code)))));
        count += uint32_t(__builtin_popcount(uint32_t(_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(high, escape_code)))));
    }
    return count;
}

void expand_scale_nibbles_host(const uint8_t *packed, const uint8_t *escapes,
                               uint32_t escape_count, const uint8_t *codebook,
                               uint8_t *output, size_t bytes) {
    const __m128i lookup = _mm_loadu_si128(reinterpret_cast<const __m128i *>(codebook));
    const __m128i nibble_mask = _mm_set1_epi8(15);
    const __m128i escape_code = _mm_set1_epi8(15);
    uint32_t escape_at = 0;
    for (size_t index = 0; index < bytes; index += 32) {
        const __m128i chunk =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(packed + index / 2));
        const __m128i low = _mm_and_si128(chunk, nibble_mask);
        const __m128i high = _mm_and_si128(_mm_srli_epi16(chunk, 4), nibble_mask);
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
                if (escape_at >= escape_count) die("escape underflow");
                const unsigned bit = unsigned(__builtin_ctz(mask));
                output[index + size_t(half) * 16 + bit] = escapes[escape_at++];
                mask &= mask - 1;
            }
        }
    }
    if (escape_at != escape_count) die("escape overflow");
}
#else
uint32_t count_escape_nibbles_256(const uint8_t *packed) {
    uint32_t count = 0;
    for (size_t i = 0; i < 256; ++i) {
        const uint8_t byte = packed[i];
        count += uint32_t((byte & 15u) == 15u) + uint32_t((byte >> 4) == 15u);
    }
    return count;
}

void expand_scale_nibbles_host(const uint8_t *packed, const uint8_t *escapes,
                               uint32_t escape_count, const uint8_t *codebook,
                               uint8_t *output, size_t bytes) {
    uint32_t escape_at = 0;
    for (size_t i = 0; i < bytes; ++i) {
        const uint8_t code = (i & 1) ? uint8_t(packed[i >> 1] >> 4)
                                     : uint8_t(packed[i >> 1] & 15u);
        if (code == 15u) {
            if (escape_at >= escape_count) die("escape underflow");
            output[i] = escapes[escape_at++];
        } else {
            output[i] = codebook[code];
        }
    }
    if (escape_at != escape_count) die("escape overflow");
}
#endif

void build_escape_prefix_host(const uint8_t *packed, uint32_t escape_count,
                              uint32_t *prefix) {
    uint32_t count = 0;
    for (size_t block = 0; block + 1 < kScalePrefixEntries; ++block) {
        prefix[block] = count;
        count += count_escape_nibbles_256(packed + block * 256);
    }
    prefix[kScalePrefixEntries - 1] = count;
    if (count != escape_count) die("escape-count mismatch");
}

// Draws the expanded code stream, packs it 2:1 (even index = low nibble,
// matching tools/pack_glm53_experts.py), and derives the escape tail, the
// 1025-entry block prefix, and the transport blob exactly as
// ExpertStager::stage_packed_gpu lays it out.
SynthScales synthesize_scales(double rate, uint64_t seed) {
    SplitMix64 rng{seed};
    uint8_t codebook[16];
    bool seen[256] = {};
    for (int filled = 0; filled < 16; ) {
        const uint8_t value = uint8_t(rng.next());
        if (seen[value]) continue;
        seen[value] = true;
        codebook[filled++] = value;
    }
    const uint32_t threshold = uint32_t(rate * 4294967296.0);
    SynthScales plane;
    plane.packed.assign(kPackedScaleBytes, 0);
    plane.truth.resize(kExpandedScaleBytes);
    for (size_t i = 0; i < kExpandedScaleBytes; ++i) {
        const uint32_t coin = uint32_t(rng.next() >> 32);
        uint8_t code;
        if (coin < threshold) {
            code = 15;
            plane.truth[i] = uint8_t(rng.next());
            plane.escapes.push_back(plane.truth[i]);
        } else {
            code = uint8_t(uint32_t(rng.next() >> 32) % 15u);
            plane.truth[i] = codebook[code];
        }
        if (i & 1) plane.packed[i >> 1] |= uint8_t(code << 4);
        else       plane.packed[i >> 1]  = code;
    }
    plane.escape_count = plane.escapes.size();
    plane.codebook.assign(codebook, codebook + 16);
    plane.prefix.resize(kScalePrefixEntries);
    build_escape_prefix_host(plane.packed.data(), uint32_t(plane.escape_count),
                             plane.prefix.data());
    plane.blob.assign(plane.packed.begin(), plane.packed.end());
    plane.blob.insert(plane.blob.end(), plane.escapes.begin(), plane.escapes.end());
    plane.blob.insert(plane.blob.end(), codebook, codebook + 16);
    plane.blob.resize((plane.blob.size() + 3) & ~size_t(3));
    const uint8_t *prefix_bytes = reinterpret_cast<const uint8_t *>(plane.prefix.data());
    plane.blob.insert(plane.blob.end(), prefix_bytes,
                      prefix_bytes + kScalePrefixEntries * sizeof(uint32_t));
    return plane;
}

__global__ void noop_kernel() {}

int expand_bench_main(int argc, char **argv) {
    if (argc > 4) {
        std::fprintf(stderr, "usage: %s --expand [seed] [iters]\n", argv[0]);
        return 64;
    }
    uint64_t seed = 0x5eed5301ull;
    int iterations = 2000;
    if (argc >= 3) seed = std::strtoull(argv[2], nullptr, 0);
    if (argc >= 4) {
        iterations = std::atoi(argv[3]);
        if (iterations <= 0) die("iteration count must be positive");
    }
    cudaDeviceProp device{};
    cuda_check(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties");
    if (device.major != 8 || device.minor != 9) die("GLM-5.3 kernels require sm_89");
    std::printf("GLM-5.3 scale expansion bench on %s (sm_%d%d), seed %#llx, %d iters\n",
                device.name, device.major, device.minor,
                static_cast<unsigned long long>(seed), iterations);

    uint8_t *d_packed = nullptr, *d_escapes = nullptr, *d_codebook = nullptr;
    uint8_t *d_output = nullptr, *d_sink = nullptr;
    uint32_t *d_prefix = nullptr;
    cuda_check(cudaMalloc(&d_packed, kPackedScaleBytes), "cudaMalloc packed");
    cuda_check(cudaMalloc(&d_escapes, kExpandedScaleBytes), "cudaMalloc escapes");
    cuda_check(cudaMalloc(&d_codebook, 16), "cudaMalloc codebook");
    cuda_check(cudaMalloc(&d_prefix, kScalePrefixEntries * sizeof(uint32_t)),
               "cudaMalloc prefix");
    cuda_check(cudaMalloc(&d_output, kExpandedScaleBytes), "cudaMalloc output");
    cuda_check(cudaMalloc(&d_sink, kPackedScaleBytes + kExpandedScaleBytes),
               "cudaMalloc H2D sink");
    uint8_t *p_expanded = nullptr, *p_blob = nullptr;
    cuda_check(cudaHostAlloc(&p_expanded, kExpandedScaleBytes, cudaHostAllocDefault),
               "cudaHostAlloc expanded");
    cuda_check(cudaHostAlloc(&p_blob,
                             kPackedScaleBytes + kExpandedScaleBytes + 16 +
                                 kScalePrefixEntries * sizeof(uint32_t),
                             cudaHostAllocDefault),
               "cudaHostAlloc blob");

    const double rates[] = {0.0, 0.00782, 0.02, 0.05, 0.125};
    const int rate_count = int(sizeof(rates) / sizeof(rates[0]));
    bool all_exact = true;
    for (int rate_index = 0; rate_index < rate_count; ++rate_index) {
        const double rate = rates[rate_index];
        const SynthScales scales = synthesize_scales(
            rate, seed + 0x9e3779b97f4a7c15ull * uint64_t(rate_index + 1));
        cuda_check(cudaMemcpy(d_packed, scales.packed.data(), kPackedScaleBytes,
                              cudaMemcpyHostToDevice), "packed H2D");
        if (scales.escape_count)
            cuda_check(cudaMemcpy(d_escapes, scales.escapes.data(), scales.escape_count,
                                  cudaMemcpyHostToDevice), "escapes H2D");
        cuda_check(cudaMemcpy(d_codebook, scales.codebook.data(), 16,
                              cudaMemcpyHostToDevice), "codebook H2D");
        cuda_check(cudaMemcpy(d_prefix, scales.prefix.data(),
                              kScalePrefixEntries * sizeof(uint32_t),
                              cudaMemcpyHostToDevice), "prefix H2D");

        const float v1_ms = benchmark([&] {
            cuda_check(insignia::glm53::expand_nvfp4_scale_nibbles(
                           d_packed, d_escapes, d_codebook, d_prefix, d_output,
                           kExpandedScaleBytes),
                       "expand kernel launch");
        }, iterations);
        const float v2_ms = benchmark([&] {
            cuda_check(insignia::glm53::expand_nvfp4_scale_nibbles_v2(
                           d_packed, d_escapes, d_codebook, d_prefix, d_output,
                           kExpandedScaleBytes),
                       "expand v2 kernel launch");
        }, iterations);

        // Byte exactness of both workers against the ground truth.
        std::vector<uint8_t> gpu_out(kExpandedScaleBytes);
        cuda_check(insignia::glm53::expand_nvfp4_scale_nibbles(
                       d_packed, d_escapes, d_codebook, d_prefix, d_output,
                       kExpandedScaleBytes),
                   "expand kernel launch");
        cuda_check(cudaDeviceSynchronize(), "expand kernel completion");
        cuda_check(cudaMemcpy(gpu_out.data(), d_output, kExpandedScaleBytes,
                              cudaMemcpyDeviceToHost), "expanded D2H");
        const bool v1_exact =
            std::memcmp(gpu_out.data(), scales.truth.data(), kExpandedScaleBytes) == 0;
        cuda_check(insignia::glm53::expand_nvfp4_scale_nibbles_v2(
                       d_packed, d_escapes, d_codebook, d_prefix, d_output,
                       kExpandedScaleBytes),
                   "expand v2 kernel launch");
        cuda_check(cudaDeviceSynchronize(), "expand v2 kernel completion");
        cuda_check(cudaMemcpy(gpu_out.data(), d_output, kExpandedScaleBytes,
                              cudaMemcpyDeviceToHost), "expanded D2H v2");
        const bool v2_exact =
            std::memcmp(gpu_out.data(), scales.truth.data(), kExpandedScaleBytes) == 0;
        all_exact = all_exact && v1_exact && v2_exact;

        // Full host staging decode: prefix pass + AVX2 expand, chrono timed.
        std::vector<uint32_t> prefix_scratch(kScalePrefixEntries);
        std::vector<uint8_t> cpu_out(kExpandedScaleBytes);
        build_escape_prefix_host(scales.packed.data(), uint32_t(scales.escape_count),
                                 prefix_scratch.data());
        expand_scale_nibbles_host(scales.packed.data(), scales.escapes.data(),
                                  uint32_t(scales.escape_count), scales.codebook.data(),
                                  cpu_out.data(), kExpandedScaleBytes);
        const bool cpu_exact =
            std::memcmp(cpu_out.data(), scales.truth.data(), kExpandedScaleBytes) == 0;
        all_exact = all_exact && cpu_exact;
        const auto prefix_begin = std::chrono::steady_clock::now();
        for (int rep = 0; rep < kDecodeReps; ++rep)
            build_escape_prefix_host(scales.packed.data(), uint32_t(scales.escape_count),
                                     prefix_scratch.data());
        const auto prefix_end = std::chrono::steady_clock::now();
        for (int rep = 0; rep < kDecodeReps; ++rep)
            expand_scale_nibbles_host(scales.packed.data(), scales.escapes.data(),
                                      uint32_t(scales.escape_count), scales.codebook.data(),
                                      cpu_out.data(), kExpandedScaleBytes);
        const auto decode_end = std::chrono::steady_clock::now();
        const double prefix_us =
            std::chrono::duration<double, std::micro>(prefix_end - prefix_begin).count() /
            kDecodeReps;
        const double decode_us =
            std::chrono::duration<double, std::micro>(decode_end - prefix_end).count() /
            kDecodeReps;

        // Pinned H2D transport references for both payload shapes.
        std::memcpy(p_expanded, scales.truth.data(), kExpandedScaleBytes);
        std::memcpy(p_blob, scales.blob.data(), scales.blob.size());
        const float expanded_ms = benchmark([&] {
            cuda_check(cudaMemcpyAsync(d_sink, p_expanded, kExpandedScaleBytes,
                                       cudaMemcpyHostToDevice), "expanded H2D");
        }, iterations);
        const float blob_ms = benchmark([&] {
            cuda_check(cudaMemcpyAsync(d_sink, p_blob, scales.blob.size(),
                                       cudaMemcpyHostToDevice), "blob H2D");
        }, iterations);

        std::printf("rate %6.3f%%  escapes %6zu  blob %7.1f KiB\n",
                    100.0 * rate, scales.escape_count, scales.blob.size() / 1024.0);
        std::printf("  gpu v1      %8.2f us  %7.1f GB/s  exact=%s\n",
                    v1_ms * 1.0e3, bandwidth_gbps(double(kExpandedScaleBytes), v1_ms * 1.0e3),
                    v1_exact ? "PASS" : "FAIL");
        std::printf("  gpu v2      %8.2f us  %7.1f GB/s  exact=%s\n",
                    v2_ms * 1.0e3, bandwidth_gbps(double(kExpandedScaleBytes), v2_ms * 1.0e3),
                    v2_exact ? "PASS" : "FAIL");
        std::printf("  cpu stage   %8.2f us  %7.1f GB/s  (prefix %6.2f + decode %6.2f us)  exact=%s\n",
                    prefix_us + decode_us,
                    bandwidth_gbps(double(kExpandedScaleBytes), prefix_us + decode_us),
                    prefix_us, decode_us, cpu_exact ? "PASS" : "FAIL");
        std::printf("  h2d pinned  expanded %7.2f us (%5.1f GB/s)  blob %7.2f us (%5.1f GB/s)\n",
                    expanded_ms * 1.0e3,
                    bandwidth_gbps(double(kExpandedScaleBytes), expanded_ms * 1.0e3),
                    blob_ms * 1.0e3,
                    bandwidth_gbps(double(scales.blob.size()), blob_ms * 1.0e3));

        if (rate_index == 1) {  // production escape rate: record-level mixes
            // Build the full 3-projection record: pinned window layout
            // (12 MiB bodies + contiguous blobs) and a 4.5 MiB-stride device
            // slot, then time the v1 enqueue mix (6 memcpy + 3 launches)
            // against the v2 mix (2D bodies + merged blob + fused launch),
            // each synchronized per rep like the engine's copy stream.
            const size_t window_bytes = (12ull << 20) + 3 * scales.blob.size();
            uint8_t *window = nullptr;
            cuda_check(cudaHostAlloc(&window, window_bytes, cudaHostAllocDefault),
                       "cudaHostAlloc bench window");
            uint8_t *slot = nullptr;
            cuda_check(cudaMalloc(&slot, 3 * kRecordSlotPitch), "cudaMalloc bench slot");
            uint8_t *scratch = nullptr;
            cuda_check(cudaMalloc(&scratch, 3 * scales.blob.size() + 64),
                       "cudaMalloc bench scratch");
            for (int projection = 0; projection < 3; ++projection) {
                std::memcpy(window + size_t(projection) * (4ull << 20),
                            scales.packed.data(), 4ull << 20);  // filler bodies
                std::memcpy(window + (12ull << 20) + projection * scales.blob.size(),
                            scales.blob.data(), scales.blob.size());
            }
            size_t dev_cursor = 0;
            std::array<size_t, 3> dev{}, esc{}, cb{}, pre{};
            for (int projection = 0; projection < 3; ++projection) {
                dev[projection] = dev_cursor;
                dev_cursor += scales.blob.size();
            }
            // Inner offsets exactly as stage_packed_gpu records them:
            // packed | escapes | codebook(16) | align4 | prefix(4100).
            for (int projection = 0; projection < 3; ++projection) {
                esc[projection] = dev[projection] + kPackedScaleBytes;
                // Engine blob order: packed | escapes | codebook | align4 pad
                // | prefix - the codebook sits immediately after the escapes
                // (stage_packed_gpu packed_codebook), NOT +16 beyond. The +16
                // here made every record-mix launch read a shifted codebook
                // while truth stayed correct: the s8 fused spot-check FAIL.
                cb[projection] = dev[projection] + kPackedScaleBytes +
                                 scales.escape_count;
                size_t inner = kPackedScaleBytes + scales.escape_count + 16;
                inner = (inner + 3) & ~size_t(3);
                pre[projection] = dev[projection] + inner;
            }
            cudaStream_t stream;
            cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate bench");

            const auto mix_time = [&](auto &&enqueue) {
                for (int rep = 0; rep < 10; ++rep) enqueue();  // warmup
                cuda_check(cudaStreamSynchronize(stream), "warmup sync");
                const auto begin = std::chrono::steady_clock::now();
                for (int rep = 0; rep < kMixReps; ++rep) enqueue();
                cuda_check(cudaStreamSynchronize(stream), "mix sync");
                return std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - begin).count() / kMixReps;
            };

            const double v1_mix_us = mix_time([&] {
                for (int projection = 0; projection < 3; ++projection)
                    cuda_check(cudaMemcpyAsync(slot + size_t(projection) * kRecordSlotPitch,
                                               window + size_t(projection) * (4ull << 20),
                                               4ull << 20, cudaMemcpyHostToDevice, stream),
                               "body H2D");
                for (int projection = 0; projection < 3; ++projection)
                    cuda_check(cudaMemcpyAsync(scratch + dev[projection],
                                               window + (12ull << 20) +
                                                   projection * scales.blob.size(),
                                               scales.blob.size(),
                                               cudaMemcpyHostToDevice, stream),
                               "blob H2D");
                for (int projection = 0; projection < 3; ++projection)
                    cuda_check(insignia::glm53::expand_nvfp4_scale_nibbles(
                                   scratch + dev[projection], scratch + esc[projection],
                                   scratch + cb[projection],
                                   reinterpret_cast<const uint32_t *>(scratch + pre[projection]),
                                   slot + size_t(projection) * kRecordSlotPitch + (4ull << 20),
                                   kExpandedScaleBytes, stream),
                               "expand launch");
            });
            const double v2_mix_us = mix_time([&] {
                cuda_check(cudaMemcpy2DAsync(slot, kRecordSlotPitch,
                                             window, 4ull << 20, 4ull << 20, 3,
                                             cudaMemcpyHostToDevice, stream),
                           "bodies H2D (2D)");
                cuda_check(cudaMemcpyAsync(scratch, window + (12ull << 20),
                                           3 * scales.blob.size(),
                                           cudaMemcpyHostToDevice, stream),
                           "blobs H2D (merged)");
                cuda_check(insignia::glm53::expand_nvfp4_scale_nibbles3_v2(
                               scratch, dev.data(), esc.data(), cb.data(), pre.data(),
                               slot, kRecordSlotPitch, 4ull << 20, kExpandedScaleBytes,
                               stream),
                           "expand fused v2");
            });

            // Fused-path byte exactness across the whole slot.
            std::vector<uint8_t> slot_host(3 * kRecordSlotPitch, 0xA5);
            cuda_check(cudaMemcpy(slot_host.data(), slot, slot_host.size(),
                                  cudaMemcpyDeviceToHost), "slot D2H");
            bool fused_exact = true;
            for (int projection = 0; projection < 3; ++projection)
                fused_exact = fused_exact &&
                    std::memcmp(slot_host.data() + size_t(projection) * kRecordSlotPitch +
                                    (4ull << 20),
                                scales.truth.data(), kExpandedScaleBytes) == 0;
            all_exact = all_exact && fused_exact;
            std::printf("  record mix  v1 (6cpy+3k) %8.2f us  v2 (2D+1cpy+1k) %8.2f us  "
                        "delta %+.1f us/record  fused exact=%s\n",
                        v1_mix_us, v2_mix_us, v2_mix_us - v1_mix_us,
                        fused_exact ? "PASS" : "FAIL");
            for (const int records : {336, 1500})
                std::printf("    @ %d records: %+.2f ms side (v2 vs v1)\n", records,
                            (v2_mix_us - v1_mix_us) * records / 1.0e3);
            cudaStreamDestroy(stream);
            cudaFree(slot);
            cudaFree(scratch);
            cudaFreeHost(window);
        }
    }

    // Empty-launch floor (repo cross-check: ~8.1 us/launch on WSL/WDDM).
    const auto launch_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < kLaunchCostLaunches; ++i) noop_kernel<<<1, 32>>>();
    cuda_check(cudaDeviceSynchronize(), "noop completion");
    const double wall_us =
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - launch_begin).count() /
        kLaunchCostLaunches;
    const float event_ms = benchmark([] { noop_kernel<<<1, 32>>>(); }, kLaunchCostLaunches);
    std::printf("empty launch  %8.2f us/launch wall  %8.2f us/launch events  (N=%d)\n",
                wall_us, event_ms * 1.0e3f, kLaunchCostLaunches);
    std::printf("overall byte-exactness: %s\n", all_exact ? "PASS" : "FAIL");

    cudaFree(d_packed); cudaFree(d_escapes); cudaFree(d_codebook); cudaFree(d_prefix);
    cudaFree(d_output); cudaFree(d_sink);
    cudaFreeHost(p_expanded); cudaFreeHost(p_blob);
    return all_exact ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc >= 2 && !std::strcmp(argv[1], "--expand"))
        return expand_bench_main(argc, argv);
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s FIXTURE.ig53 | --expand [seed] [iters]\n", argv[0]);
        return 64;
    }
    cudaDeviceProp device{};
    cuda_check(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties");
    if (device.major != 8 || device.minor != 9) die("GLM-5.3 kernels require sm_89");

    std::vector<uint8_t> file = read_file(argv[1]);
    if (file.size() < sizeof(FixtureHeader)) die("truncated fixture header");
    FixtureHeader header{};
    std::memcpy(&header, file.data(), sizeof(header));
    if (std::memcmp(header.magic, "IG53X001", 8) || header.version != 1) die("bad fixture magic/version");
    if (header.cols != 4096 || header.group != 64 || (header.rows & 7)) die("fixture is not the sm_89 GLM expert shape");

    const uint8_t *cursor = file.data() + sizeof(header);
    const uint8_t *end = file.data() + file.size();
    auto take = [&](uint64_t bytes) {
        if (bytes > uint64_t(end - cursor)) die("truncated fixture payload");
        const uint8_t *result = cursor;
        cursor += bytes;
        return result;
    };
    const uint8_t *nv_weight = take(header.nv_weight_bytes);
    const uint8_t *nv_scale = take(header.nv_scale_bytes);
    const uint8_t *i4_weight = take(header.i4_weight_bytes);
    const uint8_t *i4_scale = take(header.i4_scale_bytes);
    const uint8_t *e2_weight = take(header.e2_weight_bytes);
    const uint8_t *e2_scale = take(header.e2_scale_bytes);
    const float *x = reinterpret_cast<const float *>(take(uint64_t(header.cols) * sizeof(float)));
    const float *nv_reference = reinterpret_cast<const float *>(take(uint64_t(header.rows) * sizeof(float)));
    const float *i4_reference = reinterpret_cast<const float *>(take(uint64_t(header.rows) * sizeof(float)));
    const float *e2_reference = reinterpret_cast<const float *>(take(uint64_t(header.rows) * sizeof(float)));
    if (cursor != end) die("fixture has trailing bytes");

    cuda_check(insignia::glm53::initialize_nvfp4(), "NVFP4 LUT upload");

    uint8_t *d_nvw = device_copy<uint8_t>(nv_weight, header.nv_weight_bytes);
    uint8_t *d_nvs = device_copy<uint8_t>(nv_scale, header.nv_scale_bytes);
    uint8_t *d_i4w = device_copy<uint8_t>(i4_weight, header.i4_weight_bytes);
    __half *d_i4s = device_copy<__half>(i4_scale, header.i4_scale_bytes / sizeof(__half));
    uint8_t *d_e2w = device_copy<uint8_t>(e2_weight, header.e2_weight_bytes);
    __half *d_e2s = device_copy<__half>(e2_scale, header.e2_scale_bytes / sizeof(__half));
    float *d_x = device_copy<float>(x, header.cols);
    float *d_y = nullptr;
    float *d_y2 = nullptr;
    uint32_t *d_xq = nullptr;
    float *d_xscale = nullptr;
    cuda_check(cudaMalloc(&d_y, uint64_t(header.rows) * sizeof(float)), "cudaMalloc y");
    cuda_check(cudaMalloc(&d_y2, uint64_t(header.rows) * sizeof(float)), "cudaMalloc y2");
    cuda_check(cudaMalloc(&d_xq, uint64_t(header.cols) * sizeof(int8_t)), "cudaMalloc xq");
    cuda_check(cudaMalloc(&d_xscale, uint64_t(header.cols / 16) * sizeof(float)), "cudaMalloc xscale");

    const int nv_groups = header.cols / 16;
    const int i4_groups = header.cols / header.group;
    const dim3 nv_grid(header.rows);
    const dim3 packed_grid(header.rows / 8);
    const dim3 block(256);
    std::vector<float> output(header.rows);

    auto fetch_metrics = [&](const float *reference) {
        cuda_check(cudaDeviceSynchronize(), "kernel completion");
        cuda_check(cudaMemcpy(output.data(), d_y, uint64_t(header.rows) * sizeof(float), cudaMemcpyDeviceToHost), "output D2H");
        return compare(output, reference);
    };

    nvfp4_f32_kernel<<<nv_grid, block>>>(d_nvw, d_nvs, d_x, d_y, header.rows, header.cols, header.global_scale);
    Metrics nv_f32_metrics = fetch_metrics(nv_reference);
    const float nv_f32_ms = benchmark([&] {
        nvfp4_f32_kernel<<<nv_grid, block>>>(d_nvw, d_nvs, d_x, d_y, header.rows, header.cols, header.global_scale);
    });

    quantize_x16_kernel<<<1, block>>>(d_x, d_xq, d_xscale, nv_groups);
    nvfp4_dp4a_kernel<<<packed_grid, block>>>(reinterpret_cast<const uint32_t *>(d_nvw), d_nvs,
        d_xq, d_xscale, d_y, header.rows, nv_groups, header.global_scale);
    Metrics nv_dp_metrics = fetch_metrics(nv_reference);
    const float nv_dp_ms = benchmark([&] {
        quantize_x16_kernel<<<1, block>>>(d_x, d_xq, d_xscale, nv_groups);
        nvfp4_dp4a_kernel<<<packed_grid, block>>>(reinterpret_cast<const uint32_t *>(d_nvw), d_nvs,
            d_xq, d_xscale, d_y, header.rows, nv_groups, header.global_scale);
    });
    const float nv_dp_gemv_ms = benchmark([&] {
        nvfp4_dp4a_kernel<<<packed_grid, block>>>(reinterpret_cast<const uint32_t *>(d_nvw), d_nvs,
            d_xq, d_xscale, d_y, header.rows, nv_groups, header.global_scale);
    });
    const float nv_two_gemv_ms = benchmark([&] {
        nvfp4_dp4a_kernel<<<packed_grid, block>>>(reinterpret_cast<const uint32_t *>(d_nvw), d_nvs,
            d_xq, d_xscale, d_y, header.rows, nv_groups, header.global_scale);
        nvfp4_dp4a_kernel<<<packed_grid, block>>>(reinterpret_cast<const uint32_t *>(d_nvw), d_nvs,
            d_xq, d_xscale, d_y2, header.rows, nv_groups, header.global_scale);
    });
    const float nv_pair_ms = benchmark([&] {
        nvfp4_dp4a_pair_kernel<<<packed_grid, block>>>(
            reinterpret_cast<const uint32_t *>(d_nvw), d_nvs,
            reinterpret_cast<const uint32_t *>(d_nvw), d_nvs,
            d_xq, d_xscale, d_y, d_y2, header.rows, nv_groups,
            header.global_scale, header.global_scale);
    });

    quantize_x64_kernel<<<1, block>>>(d_x, d_xq, d_xscale);
    grouped_i4_dp4a_kernel<false><<<packed_grid, block>>>(reinterpret_cast<const uint32_t *>(d_i4w), d_i4s,
        d_xq, d_xscale, d_y, header.rows, i4_groups);
    Metrics i4_metrics = fetch_metrics(i4_reference);
    const float i4_ms = benchmark([&] {
        quantize_x64_kernel<<<1, block>>>(d_x, d_xq, d_xscale);
        grouped_i4_dp4a_kernel<false><<<packed_grid, block>>>(reinterpret_cast<const uint32_t *>(d_i4w), d_i4s,
            d_xq, d_xscale, d_y, header.rows, i4_groups);
    });

    quantize_x64_kernel<<<1, block>>>(d_x, d_xq, d_xscale);
    grouped_i4_dp4a_kernel<true><<<packed_grid, block>>>(reinterpret_cast<const uint32_t *>(d_e2w), d_e2s,
        d_xq, d_xscale, d_y, header.rows, i4_groups);
    Metrics e2_metrics = fetch_metrics(e2_reference);
    const float e2_ms = benchmark([&] {
        quantize_x64_kernel<<<1, block>>>(d_x, d_xq, d_xscale);
        grouped_i4_dp4a_kernel<true><<<packed_grid, block>>>(reinterpret_cast<const uint32_t *>(d_e2w), d_e2s,
            d_xq, d_xscale, d_y, header.rows, i4_groups);
    });

    std::printf("GLM-5.3 expert %ux%u on %s (sm_%d%d)\n", header.rows, header.cols, device.name, device.major, device.minor);
    print_result("NVFP4 float decode", nv_f32_ms, header.nv_weight_bytes + header.nv_scale_bytes, nv_f32_metrics);
    print_result("NVFP4 DP4A total", nv_dp_ms, header.nv_weight_bytes + header.nv_scale_bytes, nv_dp_metrics);
    print_result("NVFP4 DP4A GEMV", nv_dp_gemv_ms, header.nv_weight_bytes + header.nv_scale_bytes, nv_dp_metrics);
    print_result("NVFP4 two GEMVs", nv_two_gemv_ms, 2 * (header.nv_weight_bytes + header.nv_scale_bytes), nv_dp_metrics);
    print_result("NVFP4 fused pair", nv_pair_ms, 2 * (header.nv_weight_bytes + header.nv_scale_bytes), nv_dp_metrics);
    print_result("uniform INT4-g64", i4_ms, header.i4_weight_bytes + header.i4_scale_bytes, i4_metrics);
    print_result("E2M1 INT4-g64", e2_ms, header.e2_weight_bytes + header.e2_scale_bytes, e2_metrics);

    cudaFree(d_nvw); cudaFree(d_nvs); cudaFree(d_i4w); cudaFree(d_i4s);
    cudaFree(d_e2w); cudaFree(d_e2s); cudaFree(d_x); cudaFree(d_y); cudaFree(d_y2); cudaFree(d_xq); cudaFree(d_xscale);
    return 0;
}
#endif
