#include "insignia_layout.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
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
}

struct Shape { int rows, cols; };
int main() {
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
        cudaFree(dw); cudaFree(ds); cudaFree(dx); cudaFree(dy);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    return rc;
}
