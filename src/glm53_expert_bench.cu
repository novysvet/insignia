#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "insignia_glm53.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s FIXTURE.ig53\n", argv[0]);
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
