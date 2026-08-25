#include "insignia_layout.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#define CUDA_OK(call) do{cudaError_t e_=(call);if(e_!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e_));return 2;}}while(0)

// ---- experimental variants (bench-only) ----
namespace bench {
__device__ __forceinline__ float dec_arith(uint32_t w, int j) {
    const uint32_t n = (w >> (j * 4)) & 15u;
    const uint32_t mag = n & 7u;
    const uint32_t bits = ((126u + (mag >> 1)) << 23) | ((mag & 1u) << 22);
    const float v = mag >= 2u ? __uint_as_float(bits ^ ((n & 8u) << 28)) : mag * 0.5f * ((n & 8u) ? -1.f : 1.f);
    return v;
}
// A: v2 structure, arithmetic decode instead of shared LUT
__global__ __launch_bounds__(256) void gemv_a(const uint32_t *__restrict__ weights, const uint8_t *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y, int rows, int groups) {
    constexpr int LANES = 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ float sx[];
    for (int c0 = threadIdx.x * 32; c0 < groups * 32; c0 += blockDim.x * 32) {
        const float4 *xr = reinterpret_cast<const float4 *>(x + c0);
        float *sr = sx + (c0 >> 5);
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const float4 v = __ldg(xr + q);
            sr[(q * 4 + 0) * groups] = v.x; sr[(q * 4 + 1) * groups] = v.y;
            sr[(q * 4 + 2) * groups] = v.z; sr[(q * 4 + 3) * groups] = v.w;
        }
    }
    __syncthreads();
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_w = weights + static_cast<size_t>(row) * groups * 4;
    const uint8_t *row_s = scales + static_cast<size_t>(row) * groups;
    float acc = 0.f;
    #pragma unroll 4
    for (int g0 = lane; g0 < groups; g0 += LANES) {
        const uint4 p = __ldcs(reinterpret_cast<const uint4 *>(row_w + static_cast<size_t>(g0) * 4));
        const float scale = __int_as_float(static_cast<uint32_t>(row_s[g0]) << 23);
        const float *xg = sx + g0;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        #define BW(word, kb) { const uint32_t w_=(word); \
            s0=fmaf(dec_arith(w_,0),xg[(kb+0)*groups],s0); s1=fmaf(dec_arith(w_,1),xg[(kb+1)*groups],s1); \
            s2=fmaf(dec_arith(w_,2),xg[(kb+2)*groups],s2); s3=fmaf(dec_arith(w_,3),xg[(kb+3)*groups],s3); \
            s0=fmaf(dec_arith(w_,4),xg[(kb+4)*groups],s0); s1=fmaf(dec_arith(w_,5),xg[(kb+5)*groups],s1); \
            s2=fmaf(dec_arith(w_,6),xg[(kb+6)*groups],s2); s3=fmaf(dec_arith(w_,7),xg[(kb+7)*groups],s3); }
        BW(p.x,0) BW(p.y,8) BW(p.z,16) BW(p.w,24)
        #undef BW
        acc = fmaf((s0 + s1) + (s2 + s3), scale, acc);
    }
    float sum = acc;
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, mask);
    if (lane == 0) y[row] = sum;
}
void gemv_a_launch(const uint32_t *w, const uint8_t *s, const float *x, float *y, int rows, int cols) {
    static const bool ok = [] { return cudaFuncSetAttribute(gemv_a, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024) == cudaSuccess; }();
    (void)ok;
    gemv_a<<<(rows + 7) >> 3, 256, size_t(cols) * 4, nullptr>>>(w, s, x, y, rows, cols >> 5);
}
// C: two groups per lane per iteration (2x uint4), LUT decode
__global__ __launch_bounds__(256) void gemv_c(const uint32_t *__restrict__ weights, const uint8_t *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y, int rows, int groups) {
    constexpr int LANES = 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ float sx[];
    float *lut = sx + groups * 32;
    for (int c0 = threadIdx.x * 32; c0 < groups * 32; c0 += blockDim.x * 32) {
        const float4 *xr = reinterpret_cast<const float4 *>(x + c0);
        float *sr = sx + (c0 >> 5);
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const float4 v = __ldg(xr + q);
            sr[(q * 4 + 0) * groups] = v.x; sr[(q * 4 + 1) * groups] = v.y;
            sr[(q * 4 + 2) * groups] = v.z; sr[(q * 4 + 3) * groups] = v.w;
        }
    }
    if (threadIdx.x < 16) lut[threadIdx.x] = insignia::fp4_e2m1(uint8_t(threadIdx.x));
    __syncthreads();
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_w = weights + static_cast<size_t>(row) * groups * 4;
    const uint8_t *row_s = scales + static_cast<size_t>(row) * groups;
    float acc0 = 0.f, acc1 = 0.f;
    int g0 = lane;
    for (; g0 + LANES < groups; g0 += 2 * LANES) {
        const uint4 pa = __ldcs(reinterpret_cast<const uint4 *>(row_w + static_cast<size_t>(g0) * 4));
        const uint4 pb = __ldcs(reinterpret_cast<const uint4 *>(row_w + static_cast<size_t>(g0 + LANES) * 4));
        const float sa = __int_as_float(static_cast<uint32_t>(row_s[g0]) << 23);
        const float sb = __int_as_float(static_cast<uint32_t>(row_s[g0 + LANES]) << 23);
        const float *xa = sx + g0, *xb = sx + g0 + LANES;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        #define BW2(word, kb, xr_) { const uint32_t w_=(word); \
            s0=fmaf(lut[(w_>>0)&15u],xr_[(kb+0)*groups],s0); s1=fmaf(lut[(w_>>4)&15u],xr_[(kb+1)*groups],s1); \
            s2=fmaf(lut[(w_>>8)&15u],xr_[(kb+2)*groups],s2); s3=fmaf(lut[(w_>>12)&15u],xr_[(kb+3)*groups],s3); \
            s0=fmaf(lut[(w_>>16)&15u],xr_[(kb+4)*groups],s0); s1=fmaf(lut[(w_>>20)&15u],xr_[(kb+5)*groups],s1); \
            s2=fmaf(lut[(w_>>24)&15u],xr_[(kb+6)*groups],s2); s3=fmaf(lut[(w_>>28)&15u],xr_[(kb+7)*groups],s3); }
        BW2(pa.x,0,xa) BW2(pa.y,8,xa) BW2(pa.z,16,xa) BW2(pa.w,24,xa)
        acc0 = fmaf((s0 + s1) + (s2 + s3), sa, acc0);
        s0 = s1 = s2 = s3 = 0.f;
        BW2(pb.x,0,xb) BW2(pb.y,8,xb) BW2(pb.z,16,xb) BW2(pb.w,24,xb)
        acc1 = fmaf((s0 + s1) + (s2 + s3), sb, acc1);
        #undef BW2
    }
    for (; g0 < groups; g0 += LANES) {
        const uint4 p = __ldcs(reinterpret_cast<const uint4 *>(row_w + static_cast<size_t>(g0) * 4));
        const float scale = __int_as_float(static_cast<uint32_t>(row_s[g0]) << 23);
        const float *xg = sx + g0;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        #define BW1(word, kb) { const uint32_t w_=(word); \
            s0=fmaf(lut[(w_>>0)&15u],xg[(kb+0)*groups],s0); s1=fmaf(lut[(w_>>4)&15u],xg[(kb+1)*groups],s1); \
            s2=fmaf(lut[(w_>>8)&15u],xg[(kb+2)*groups],s2); s3=fmaf(lut[(w_>>12)&15u],xg[(kb+3)*groups],s3); \
            s0=fmaf(lut[(w_>>16)&15u],xg[(kb+4)*groups],s0); s1=fmaf(lut[(w_>>20)&15u],xg[(kb+5)*groups],s1); \
            s2=fmaf(lut[(w_>>24)&15u],xg[(kb+6)*groups],s2); s3=fmaf(lut[(w_>>28)&15u],xg[(kb+7)*groups],s3); }
        BW1(p.x,0) BW1(p.y,8) BW1(p.z,16) BW1(p.w,24)
        #undef BW1
        acc0 = fmaf((s0 + s1) + (s2 + s3), scale, acc0);
    }
    float sum = acc0 + acc1;
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, mask);
    if (lane == 0) y[row] = sum;
}
void gemv_c_launch(const uint32_t *w, const uint8_t *s, const float *x, float *y, int rows, int cols) {
    static const bool ok = [] { return cudaFuncSetAttribute(gemv_c, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024) == cudaSuccess; }();
    (void)ok;
    gemv_c<<<(rows + 7) >> 3, 256, size_t(cols) * 4 + 64, nullptr>>>(w, s, x, y, rows, cols >> 5);
}
// D: two rows per warp, shared x loads feed both rows' FMAs
__global__ __launch_bounds__(256) void gemv_d(const uint32_t *__restrict__ weights, const uint8_t *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y, int rows, int groups) {
    constexpr int LANES = 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ float sx[];
    float *lut = sx + groups * 32;
    for (int c0 = threadIdx.x * 32; c0 < groups * 32; c0 += blockDim.x * 32) {
        const float4 *xr = reinterpret_cast<const float4 *>(x + c0);
        float *sr = sx + (c0 >> 5);
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const float4 v = __ldg(xr + q);
            sr[(q * 4 + 0) * groups] = v.x; sr[(q * 4 + 1) * groups] = v.y;
            sr[(q * 4 + 2) * groups] = v.z; sr[(q * 4 + 3) * groups] = v.w;
        }
    }
    if (threadIdx.x < 16) lut[threadIdx.x] = insignia::fp4_e2m1(uint8_t(threadIdx.x));
    __syncthreads();
    const int row0 = (blockIdx.x * 8 + warp) * 2, row1 = row0 + 1;
    if (row0 >= rows) return;
    const bool has1 = row1 < rows;
    const uint32_t *w0 = weights + static_cast<size_t>(row0) * groups * 4;
    const uint32_t *w1 = weights + static_cast<size_t>(has1 ? row1 : row0) * groups * 4;
    const uint8_t *s0 = scales + static_cast<size_t>(row0) * groups;
    const uint8_t *s1 = scales + static_cast<size_t>(has1 ? row1 : row0) * groups;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
    #pragma unroll 4
    for (int g0 = lane; g0 < groups; g0 += LANES) {
        const uint4 pa = __ldcs(reinterpret_cast<const uint4 *>(w0 + static_cast<size_t>(g0) * 4));
        const uint4 pb = __ldcs(reinterpret_cast<const uint4 *>(w1 + static_cast<size_t>(g0) * 4));
        const float sa = __int_as_float(static_cast<uint32_t>(s0[g0]) << 23);
        const float sb = __int_as_float(static_cast<uint32_t>(s1[g0]) << 23);
        const float *xg = sx + g0;
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f, q0 = 0.f, q1 = 0.f, q2 = 0.f, q3 = 0.f;
        #define BWD(wa_, wb_, kb) { const uint32_t wa=wa_, wb=wb_; const float xk0=xg[(kb+0)*groups], xk1=xg[(kb+1)*groups], xk2=xg[(kb+2)*groups], xk3=xg[(kb+3)*groups], xk4=xg[(kb+4)*groups], xk5=xg[(kb+5)*groups], xk6=xg[(kb+6)*groups], xk7=xg[(kb+7)*groups]; \
            p0=fmaf(lut[(wa>>0)&15u],xk0,p0); p1=fmaf(lut[(wa>>4)&15u],xk1,p1); p2=fmaf(lut[(wa>>8)&15u],xk2,p2); p3=fmaf(lut[(wa>>12)&15u],xk3,p3); \
            p0=fmaf(lut[(wa>>16)&15u],xk4,p0); p1=fmaf(lut[(wa>>20)&15u],xk5,p1); p2=fmaf(lut[(wa>>24)&15u],xk6,p2); p3=fmaf(lut[(wa>>28)&15u],xk7,p3); \
            q0=fmaf(lut[(wb>>0)&15u],xk0,q0); q1=fmaf(lut[(wb>>4)&15u],xk1,q1); q2=fmaf(lut[(wb>>8)&15u],xk2,q2); q3=fmaf(lut[(wb>>12)&15u],xk3,q3); \
            q0=fmaf(lut[(wb>>16)&15u],xk4,q0); q1=fmaf(lut[(wb>>20)&15u],xk5,q1); q2=fmaf(lut[(wb>>24)&15u],xk6,q2); q3=fmaf(lut[(wb>>28)&15u],xk7,q3); }
        BWD(pa.x, pb.x, 0) BWD(pa.y, pb.y, 8) BWD(pa.z, pb.z, 16) BWD(pa.w, pb.w, 24)
        #undef BWD
        a0 = fmaf((p0 + p1) + (p2 + p3), sa, a0);
        b0 = fmaf((q0 + q1) + (q2 + q3), sb, b0);
    }
    float sa = (a0 + a1) + (a2 + a3), sb = (b0 + b1) + (b2 + b3);
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) { sa += __shfl_xor_sync(0xffffffff, sa, mask); sb += __shfl_xor_sync(0xffffffff, sb, mask); }
    if (lane == 0) { y[row0] = sa; if (has1) y[row1] = sb; }
}
void gemv_d_launch(const uint32_t *w, const uint8_t *s, const float *x, float *y, int rows, int cols) {
    static const bool ok = [] { return cudaFuncSetAttribute(gemv_d, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024) == cudaSuccess; }();
    (void)ok;
    const int pairs = (rows + 1) / 2;
    gemv_d<<<(pairs + 7) >> 3, 256, size_t(cols) * 4 + 64, nullptr>>>(w, s, x, y, rows, cols >> 5);
}
// E: four rows per warp; one x load feeds four rows' FMAs
__global__ __launch_bounds__(256) void gemv_e(const uint32_t *__restrict__ weights, const uint8_t *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y, int rows, int groups) {
    constexpr int LANES = 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ float sx[];
    float *lut = sx + groups * 32;
    for (int c0 = threadIdx.x * 32; c0 < groups * 32; c0 += blockDim.x * 32) {
        const float4 *xr = reinterpret_cast<const float4 *>(x + c0);
        float *sr = sx + (c0 >> 5);
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const float4 v = __ldg(xr + q);
            sr[(q * 4 + 0) * groups] = v.x; sr[(q * 4 + 1) * groups] = v.y;
            sr[(q * 4 + 2) * groups] = v.z; sr[(q * 4 + 3) * groups] = v.w;
        }
    }
    if (threadIdx.x < 16) lut[threadIdx.x] = insignia::fp4_e2m1(uint8_t(threadIdx.x));
    __syncthreads();
    const int rowb = (blockIdx.x * 8 + warp) * 4;
    if (rowb >= rows) return;
    const int n = rowb + 4 <= rows ? 4 : rows - rowb;
    const uint32_t *rw[4]; const uint8_t *rs[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int r = i < n ? rowb + i : rowb;
        rw[i] = weights + static_cast<size_t>(r) * groups * 4;
        rs[i] = scales + static_cast<size_t>(r) * groups;
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    #define EWORD(w_, p_) { const uint32_t w__=(w_); (p_) += lut[(w__>>0)&15u]*xk0 + lut[(w__>>4)&15u]*xk1 + lut[(w__>>8)&15u]*xk2 + lut[(w__>>12)&15u]*xk3 + lut[(w__>>16)&15u]*xk4 + lut[(w__>>20)&15u]*xk5 + lut[(w__>>24)&15u]*xk6 + lut[(w__>>28)&15u]*xk7; }
    #pragma unroll 2
    for (int g0 = lane; g0 < groups; g0 += LANES) {
        uint4 pw0 = __ldcs(reinterpret_cast<const uint4 *>(rw[0] + static_cast<size_t>(g0) * 4));
        uint4 pw1 = __ldcs(reinterpret_cast<const uint4 *>(rw[1] + static_cast<size_t>(g0) * 4));
        uint4 pw2 = __ldcs(reinterpret_cast<const uint4 *>(rw[2] + static_cast<size_t>(g0) * 4));
        uint4 pw3 = __ldcs(reinterpret_cast<const uint4 *>(rw[3] + static_cast<size_t>(g0) * 4));
        const float sc0 = __int_as_float(static_cast<uint32_t>(rs[0][g0]) << 23);
        const float sc1 = __int_as_float(static_cast<uint32_t>(rs[1][g0]) << 23);
        const float sc2 = __int_as_float(static_cast<uint32_t>(rs[2][g0]) << 23);
        const float sc3 = __int_as_float(static_cast<uint32_t>(rs[3][g0]) << 23);
        const float *xg = sx + g0;
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
        {
            const float xk0 = xg[0 * groups], xk1 = xg[1 * groups], xk2 = xg[2 * groups], xk3 = xg[3 * groups];
            const float xk4 = xg[4 * groups], xk5 = xg[5 * groups], xk6 = xg[6 * groups], xk7 = xg[7 * groups];
            EWORD(pw0.x, p0) EWORD(pw1.x, p1) EWORD(pw2.x, p2) EWORD(pw3.x, p3)
        }
        {
            const float xk0 = xg[8 * groups], xk1 = xg[9 * groups], xk2 = xg[10 * groups], xk3 = xg[11 * groups];
            const float xk4 = xg[12 * groups], xk5 = xg[13 * groups], xk6 = xg[14 * groups], xk7 = xg[15 * groups];
            EWORD(pw0.y, p0) EWORD(pw1.y, p1) EWORD(pw2.y, p2) EWORD(pw3.y, p3)
        }
        {
            const float xk0 = xg[16 * groups], xk1 = xg[17 * groups], xk2 = xg[18 * groups], xk3 = xg[19 * groups];
            const float xk4 = xg[20 * groups], xk5 = xg[21 * groups], xk6 = xg[22 * groups], xk7 = xg[23 * groups];
            EWORD(pw0.z, p0) EWORD(pw1.z, p1) EWORD(pw2.z, p2) EWORD(pw3.z, p3)
        }
        {
            const float xk0 = xg[24 * groups], xk1 = xg[25 * groups], xk2 = xg[26 * groups], xk3 = xg[27 * groups];
            const float xk4 = xg[28 * groups], xk5 = xg[29 * groups], xk6 = xg[30 * groups], xk7 = xg[31 * groups];
            EWORD(pw0.w, p0) EWORD(pw1.w, p1) EWORD(pw2.w, p2) EWORD(pw3.w, p3)
        }
        acc[0] = fmaf(p0, sc0, acc[0]); acc[1] = fmaf(p1, sc1, acc[1]);
        acc[2] = fmaf(p2, sc2, acc[2]); acc[3] = fmaf(p3, sc3, acc[3]);
    }
    #undef EWORD
    float s0 = acc[0], s1 = acc[1], s2 = acc[2], s3 = acc[3];
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) { s0 += __shfl_xor_sync(0xffffffff, s0, mask); s1 += __shfl_xor_sync(0xffffffff, s1, mask); s2 += __shfl_xor_sync(0xffffffff, s2, mask); s3 += __shfl_xor_sync(0xffffffff, s3, mask); }
    if (lane == 0) { y[rowb] = s0; if (n > 1) y[rowb + 1] = s1; if (n > 2) y[rowb + 2] = s2; if (n > 3) y[rowb + 3] = s3; }
}
void gemv_e_launch(const uint32_t *w, const uint8_t *s, const float *x, float *y, int rows, int cols) {
    static const bool ok = [] { return cudaFuncSetAttribute(gemv_e, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024) == cudaSuccess; }();
    (void)ok;
    const int quads = (rows + 3) >> 2;
    gemv_e<<<(quads + 7) >> 3, 256, size_t(cols) * 4 + 64, nullptr>>>(w, s, x, y, rows, cols >> 5);
}


// F: pair dp4a - both activation rows quantized to per-group int8 once; weights decoded to
// signed int8 (x2 E2M1 table, 0.5 folded into the group scale) through one 256-entry u64
// broadcast table (LO nibble in low word, HI in high) and __byte_perm interleave. x lives
// group-major so each row costs 2x LDS.128 per group. Assumes cols==4096 (groups==128).
__global__ __launch_bounds__(256) void gemv_f(const uint32_t *__restrict__ weights, const uint8_t *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y, int rows, int groups) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ char smem[];
    uint32_t *xq = reinterpret_cast<uint32_t *>(smem);                        // [2][groups][8]
    float *xs = reinterpret_cast<float *>(smem + size_t(16) * groups * 4);    // [2][groups]
    unsigned long long *btab = reinterpret_cast<unsigned long long *>(smem + size_t(16) * groups * 4 + 2 * groups * 4);  // [256]
    const int r = threadIdx.x >> 7, g = threadIdx.x & 127;
    {   // one thread quantizes one 32-wide group of one row to int8 + fp32 group scale
        const float *xg = x + r * (groups * 32) + g * 32;
        float v[32], m = 0.f;
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const float4 f4 = __ldg(reinterpret_cast<const float4 *>(xg) + q);
            v[q * 4 + 0] = f4.x; v[q * 4 + 1] = f4.y; v[q * 4 + 2] = f4.z; v[q * 4 + 3] = f4.w;
            m = fmaxf(m, fmaxf(fmaxf(fabsf(f4.x), fabsf(f4.y)), fmaxf(fabsf(f4.z), fabsf(f4.w))));
        }
        const float scale = m * (1.f / 127.f);
        xs[r * groups + g] = scale;
        const float inv = m > 0.f ? 127.f / m : 0.f;
        uint32_t *dst = xq + (r * groups + g) * 8;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            uint32_t packed = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) packed |= uint32_t(uint8_t(__float2int_rn(v[w * 4 + j] * inv))) << (8 * j);
            dst[w] = packed;
        }
    }
    {   // btab[b] = { LO-nibble code broadcast x4 | HI-nibble code broadcast x4 }
        static const signed char tbl[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        const int b = threadIdx.x;
        const int lo = b & 15, hi = b >> 4;
        const uint32_t clo = uint32_t(uint8_t((lo & 8) ? -tbl[lo & 7] : tbl[lo & 7])) * 0x01010101u;
        const uint32_t chi = uint32_t(uint8_t((hi & 8) ? -tbl[hi & 7] : tbl[hi & 7])) * 0x01010101u;
        btab[threadIdx.x] = unsigned long long(clo) | (unsigned long long(chi) << 32);
    }
    __syncthreads();
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_w = weights + static_cast<size_t>(row) * groups * 4;
    const uint8_t *row_s = scales + static_cast<size_t>(row) * groups;
    float acc0 = 0.f, acc1 = 0.f;
    #pragma unroll 4
    for (int g0 = lane; g0 < groups; g0 += 32) {
        const uint4 P = __ldcs(reinterpret_cast<const uint4 *>(row_w + static_cast<size_t>(g0) * 4));
        const uint4 *x0 = reinterpret_cast<const uint4 *>(xq + g0 * 8);
        const uint4 *x1 = reinterpret_cast<const uint4 *>(xq + (groups + g0) * 8);
        int d0a = 0, d0b = 0, d1a = 0, d1b = 0;
        const uint32_t pw[4] = {P.x, P.y, P.z, P.w};
        #pragma unroll
        for (int wi = 0; wi < 4; wi++) {
            const uint32_t w_ = pw[wi];
            const unsigned long long b0 = btab[w_ & 0xff], b1 = btab[(w_ >> 8) & 0xff];
            const unsigned long long b2 = btab[(w_ >> 16) & 0xff], b3 = btab[w_ >> 24];
            const uint32_t L0 = uint32_t(b0), H0 = uint32_t(b0 >> 32);
            const uint32_t L1 = uint32_t(b1), H1 = uint32_t(b1 >> 32);
            const uint32_t L2 = uint32_t(b2), H2 = uint32_t(b2 >> 32);
            const uint32_t L3 = uint32_t(b3), H3 = uint32_t(b3 >> 32);
            const uint32_t A = __byte_perm(L0, L1, 0xc480);       // [c0,0,c2,0]
            const uint32_t B = __byte_perm(H0, H1, 0x4c80);       // [0,c1,0,c3]
            const uint32_t w0 = __byte_perm(A, B, 0x6240);        // [c0,c1,c2,c3]
            const uint32_t A2 = __byte_perm(L2, L3, 0xc480);
            const uint32_t B2 = __byte_perm(H2, H3, 0x4c80);
            const uint32_t w1 = __byte_perm(A2, B2, 0x6240);      // [c4,c5,c6,c7]
            const uint32_t xx0 = reinterpret_cast<const uint32_t *>(x0)[wi * 2];
            const uint32_t xx1 = reinterpret_cast<const uint32_t *>(x0)[wi * 2 + 1];
            const uint32_t yy0 = reinterpret_cast<const uint32_t *>(x1)[wi * 2];
            const uint32_t yy1 = reinterpret_cast<const uint32_t *>(x1)[wi * 2 + 1];
            d0a = __dp4a(int(w0), int(xx0), d0a);
            d1a = __dp4a(int(w0), int(yy0), d1a);
            d0b = __dp4a(int(w1), int(xx1), d0b);
            d1b = __dp4a(int(w1), int(yy1), d1b);
        }
        const float ws = __int_as_float(uint32_t(row_s[g0]) << 23) * 0.5f;
        acc0 = fmaf(float(d0a + d0b), ws * xs[g0], acc0);
        acc1 = fmaf(float(d1a + d1b), ws * xs[groups + g0], acc1);
    }
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) { acc0 += __shfl_xor_sync(0xffffffff, acc0, mask); acc1 += __shfl_xor_sync(0xffffffff, acc1, mask); }
    if (lane == 0) { y[row] = acc0; y[rows + row] = acc1; }
}
void gemv_f_launch(const uint32_t *w, const uint8_t *s, const float *x, float *y, int rows, int cols) {
    const int groups = cols >> 5;
    gemv_f<<<(rows + 7) >> 3, 256, size_t(16) * groups * 4 + 2 * groups * 4 + 2048, nullptr>>>(w, s, x, y, rows, cols >> 5);
}
}

int run_gemm_bench() {

    const int shapes[][2] = {{8192,4096},{12288,4096},{4096,12288},{248320,4096}};
    int rc = 0;
    for (auto &sh : shapes) {
        const int rows = sh[0], cols = sh[1], groups = cols / 32, T = 64;
        std::vector<uint32_t> w(size_t(rows) * groups * 4);
        std::vector<uint8_t> s(size_t(rows) * groups);
        std::vector<float> x(size_t(T) * cols), y(size_t(T) * rows);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> fx(-1.5f, 1.5f);
        for (auto &v : x) v = fx(rng);
        for (int r = 0; r < rows; r++)
            for (int g = 0; g < groups; g++) {
                s[size_t(r) * groups + g] = uint8_t(120 + (r + g) % 9);
                for (int lane = 0; lane < 32; lane++) {
                    uint8_t q = uint8_t((r * 5 + g * 3 + lane * 7) & 15);
                    w[(size_t(r) * groups + g) * 4 + (lane >> 3)] |= uint32_t(q) << ((lane & 7) * 4);
                }
            }
        uint32_t *dw; uint8_t *ds; float *dx, *dy;
        CUDA_OK(cudaMalloc(&dw, w.size() * 4)); CUDA_OK(cudaMalloc(&ds, s.size()));
        CUDA_OK(cudaMalloc(&dx, x.size() * 4)); CUDA_OK(cudaMalloc(&dy, y.size() * 4));
        CUDA_OK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(ds, s.data(), s.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        {   // v2.1 path: bf16 A scratch + cp.async pipeline
            void *x16;
            CUDA_OK(cudaMalloc(&x16, size_t(T) * cols * 2));
            insignia::f32_to_bf16(dx, x16, T * cols, nullptr);
            cudaEvent_t c, d; cudaEventCreate(&c); cudaEventCreate(&d);
            insignia::mxfp4_gemm_v21(dw, ds, x16, dy, rows, cols, T);
            CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
            float err2 = 0;
            for (int si = 0; si < 8; si++) {
                const int r = (si * 2654435761u) % rows;
                for (int t = 0; t < T; t += 7) {
                    double z = 0, mag = 0;
                    for (int g = 0; g < groups; g++) {
                        const float scale = insignia::e8m0(s[size_t(r) * groups + g]);
                        for (int lane = 0; lane < 32; lane++) {
                            const uint8_t q = uint8_t(w[(size_t(r) * groups + g) * 4 + (lane >> 3)] >> ((lane & 7) * 4));
                            const double term = double(insignia::fp4_e2m1(q) * scale) * x[size_t(t) * cols + g * 32 + lane];
                            z += term; mag += fabs(term);
                        }
                    }
                    err2 = fmaxf(err2, float(fabs(z - y[size_t(t) * rows + r]) / (mag + 1e-6)));
                }
            }
            cudaEventRecord(c);
            for (int i = 20; i; i--) insignia::mxfp4_gemm_v21(dw, ds, x16, dy, rows, cols, T);
            cudaEventRecord(d); cudaEventSynchronize(d);
            float ms2; cudaEventElapsedTime(&ms2, c, d); ms2 /= 20;
            std::printf("%dx%d T=%d v21 max_rel=%g %.3f ms | %.0f GiB/s | %.1f TFLOPS\n", rows, cols, T, err2, ms2, double(w.size() * 4 + s.size()) / 1073741824.0 / (ms2 / 1000.0), double(T) * rows * cols * 2 / 1e12 / (ms2 / 1000.0));
            cudaFree(x16); cudaEventDestroy(c); cudaEventDestroy(d);
        }
        insignia::mxfp4_gemm_v2(dw, ds, dx, dy, rows, cols, T);
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
        // Host check on 8 sample rows across several t, normalized by the
        // cancellation-free magnitude (near-zero dots make plain relative error meaningless).
        float err = 0;
        for (int si = 0; si < 8; si++) {
            const int r = (si * 2654435761u) % rows;
            for (int t = 0; t < T; t += 7) {
                double z = 0, mag = 0;
                for (int g = 0; g < groups; g++) {
                    const float scale = insignia::e8m0(s[size_t(r) * groups + g]);
                    for (int lane = 0; lane < 32; lane++) {
                        const uint8_t q = uint8_t(w[(size_t(r) * groups + g) * 4 + (lane >> 3)] >> ((lane & 7) * 4));
                        const double term = double(insignia::fp4_e2m1(q) * scale) * x[size_t(t) * cols + g * 32 + lane];
                        z += term;
                        mag += fabs(term);
                    }
                }
                err = fmaxf(err, float(fabs(z - y[size_t(t) * rows + r]) / (mag + 1e-6)));
            }
        }
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        const int iters = 30;
        cudaEventRecord(a);
        for (int i = iters; i; i--) insignia::mxfp4_gemm_v2(dw, ds, dx, dy, rows, cols, T);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b); ms /= iters;
        const double wbytes = double(w.size() * 4 + s.size());
        const double flops = double(T) * rows * cols * 2;
        (void)wbytes; (void)flops;
        std::printf("%dx%d T=%d max_rel=%g %.3f ms | %.0f GiB/s weights | %.1f TFLOPS\n",
                    rows, cols, T, err, ms, wbytes / 1073741824.0 / (ms / 1000.0), flops / 1e12 / (ms / 1000.0));
        if (err > 2e-2) { std::printf("FAIL\n"); rc = 1; }
        cudaFree(dw); cudaFree(ds); cudaFree(dx); cudaFree(dy);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    return rc;
}

struct Shape { int rows, cols; };
int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "gemm") == 0) return run_gemm_bench();
    const Shape shapes[] = {{8192,4096},{4096,4096},{12288,4096},{4096,12288},{1024,4096},{248320,4096}};
    int rc = 0;
    for (const Shape &sh : shapes) {
        const int rows = sh.rows, cols = sh.cols, groups = cols / 32;
        std::vector<uint32_t> w(size_t(rows) * groups * 4);
        std::vector<uint8_t> s(size_t(rows) * groups);
        std::vector<float> x(cols), ref(rows), y(rows);
        for (int i = 0; i < cols; i++) x[i] = float((i * 17) % 31 - 15) / 16.f;
        for (int r = 0; r < rows; r++) {
            double z = 0;
            for (int g = 0; g < groups; g++) {
                s[size_t(r) * groups + g] = uint8_t(124 + (r + g) % 7);
                float scale = insignia::e8m0(s[size_t(r) * groups + g]);
                for (int lane = 0; lane < 32; lane++) {
                    uint8_t q = uint8_t((r * 5 + g + lane * 7) & 15);
                    w[(size_t(r) * groups + g) * 4 + (lane >> 3)] |= uint32_t(q) << ((lane & 7) * 4);
                    z += double(insignia::fp4_e2m1(q) * scale) * x[g * 32 + lane];
                }
            }
            ref[r] = float(z);
        }
        uint32_t *dw; uint8_t *ds; float *dx, *dy;
        CUDA_OK(cudaMalloc(&dw, w.size() * 4)); CUDA_OK(cudaMalloc(&ds, s.size()));
        CUDA_OK(cudaMalloc(&dx, x.size() * 4)); CUDA_OK(cudaMalloc(&dy, y.size() * 4));
        CUDA_OK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(ds, s.data(), s.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        double bytes = double(w.size() * 4 + s.size() + x.size() * 4);
        auto bench_one = [&](const char *name, int warps, bool v2) {
            auto run = [&] { v2 ? insignia::mxfp4_gemv_v2(dw, ds, dx, dy, rows, cols)
                                 : insignia::mxfp4_gemv_mlx(dw, ds, dx, dy, rows, cols, warps); };
            run(); CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
            float err = 0;
            for (int r = 0; r < rows; r++) err = fmaxf(err, fabsf(y[r] - ref[r]) / (fabsf(ref[r]) + 1e-5f));
            constexpr int iters = 100;
            cudaEventRecord(a);
            for (int i = iters; i; i--) run();
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b); ms /= iters;
            std::printf("%dx%d %-10s w=%d max_rel=%g %.3f ms %.1f GiB/s\n", rows, cols, name, warps, err, ms, (bytes / 1073741824.) / (ms / 1000.));
            if (err > 3e-4) rc = 1;
        };
        auto bench_lambda = [&](const char *name, void (*launch)(const uint32_t *, const uint8_t *, const float *, float *, int, int)) {
            launch(dw, ds, dx, dy, rows, cols); CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
            float err = 0;
            for (int r = 0; r < rows; r++) err = fmaxf(err, fabsf(y[r] - ref[r]) / (fabsf(ref[r]) + 1e-5f));
            constexpr int iters = 100;
            cudaEventRecord(a);
            for (int i = iters; i; i--) launch(dw, ds, dx, dy, rows, cols);
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b); ms /= iters;
            std::printf("%dx%d %-10s max_rel=%g %.3f ms %.1f GiB/s\n", rows, cols, name, err, ms, (bytes / 1073741824.) / (ms / 1000.));
            if (err > 3e-4) rc = 1;
        };
        if (rows == 8192 || rows == 248320) {
            bench_one("mlx", 2, false);
            bench_one("v2", 0, true);
            bench_lambda("A-arith", bench::gemv_a_launch);
            bench_lambda("C-2grp", bench::gemv_c_launch);
            bench_lambda("D-2rows", bench::gemv_d_launch);
            bench_lambda("E-4rows", bench::gemv_e_launch);
        }
        {   // pair (2 activation rows) path used by speculative decode
            float *dx2, *dy2;
            CUDA_OK(cudaMalloc(&dx2, size_t(cols) * 2 * 4));
            CUDA_OK(cudaMalloc(&dy2, size_t(rows) * 2 * 4));
            CUDA_OK(cudaMemcpy(dx2, x.data(), cols * 4, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(dx2 + cols, x.data(), cols * 4, cudaMemcpyHostToDevice));
            auto timeit = [&](const char *name, auto &&run2) {
                run2(); CUDA_OK(cudaDeviceSynchronize());
                constexpr int iters = 100;
                cudaEventRecord(a);
                for (int i = iters; i; i--) run2();
                cudaEventRecord(b); cudaEventSynchronize(b);
                float ms; cudaEventElapsedTime(&ms, a, b); ms /= iters;
                std::printf("%dx%d %-10s %.3f ms %.1f GiB/s weights\n", rows, cols, name, ms, (bytes / 1073741824.) / (ms / 1000.));
            };
            timeit("gemv2(2x)", [&] { insignia::mxfp4_gemv2_v2(dw, ds, dx2, dy2, rows, cols); });
            {   // prequantized pair path with correctness check against ref
                uint32_t *xq; float *xs;
                CUDA_OK(cudaMalloc(&xq, size_t(2 * (cols >> 5)) * 8 * 4));
                CUDA_OK(cudaMalloc(&xs, 2 * (cols >> 5) * 4));
                auto runq = [&] { insignia::quantize_x8(dx2, xq, xs, 2, cols); insignia::mxfp4_gemv2_q8g(dw, ds, xq, xs, dy2, rows, cols); };
                runq(); CUDA_OK(cudaDeviceSynchronize());
                std::vector<float> y2(rows * 2);
                CUDA_OK(cudaMemcpy(y2.data(), dy2, rows * 2 * 4, cudaMemcpyDeviceToHost));
                float err = 0;
                for (int r = 0; r < rows; r++) err = fmaxf(err, fabsf(y2[r] - ref[r]));  // bench uses the same x for both rows
                std::printf("%dx%d q8g-pair   max_abs=%g ref0=%.1f\n", rows, cols, err, fabsf(ref[0]));
                timeit("q8g(2x)", runq);
            {   // fused a+b kernel check: A = rows 0..31, B = rows 32..63 of the same weights
                uint32_t *xq; float *xs; float *ya, *yb;
                CUDA_OK(cudaMalloc(&xq, size_t(2 * (cols >> 5)) * 8 * 4));
                CUDA_OK(cudaMalloc(&xs, 2 * (cols >> 5) * 4));
                CUDA_OK(cudaMalloc(&ya, 64 * 4)); CUDA_OK(cudaMalloc(&yb, 64 * 4));
                insignia::quantize_x8(dx2, xq, xs, 2, cols);
                insignia::mxfp4_gemv_ab2_q8g(dw, ds, dw + size_t(32) * groups * 4, ds + size_t(32) * groups, xq, xs, ya, yb, cols);
                CUDA_OK(cudaDeviceSynchronize());
                std::vector<float> ha(64), hb(64);
                CUDA_OK(cudaMemcpy(ha.data(), ya, 64 * 4, cudaMemcpyDeviceToHost));
                CUDA_OK(cudaMemcpy(hb.data(), yb, 64 * 4, cudaMemcpyDeviceToHost));
                float ea = 0, eb = 0;
                for (int r = 0; r < 32; r++) { ea = fmaxf(ea, fabsf(ha[r] - ref[r]) / (fabsf(ref[r]) + 1e-3)); eb = fmaxf(eb, fabsf(hb[r] - ref[32 + r]) / (fabsf(ref[32 + r]) + 1e-3)); }
                std::printf("%dx%d ab2q8g    a_max_rel=%g b_max_rel=%g\n", rows, cols, ea, eb);
                cudaFree(xq); cudaFree(xs); cudaFree(ya); cudaFree(yb);
            }
                cudaFree(xq); cudaFree(xs);
            }
            if (cols == 4096) {
                bench_lambda("F-dp4a2x", bench::gemv_f_launch);
            }
            timeit("v2 x2", [&] { insignia::mxfp4_gemv_v2(dw, ds, dx2, dy2, rows, cols); insignia::mxfp4_gemv_v2(dw, ds, dx2 + cols, dy2 + rows, rows, cols); });
            cudaFree(dx2); cudaFree(dy2);
        }
        cudaFree(dw); cudaFree(ds); cudaFree(dx); cudaFree(dy);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    return rc;
}
