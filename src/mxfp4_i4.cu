#include "insignia_layout.cuh"
#include <cuda_bf16.h>
namespace insignia {

// ---- INSIG4: E2M1 codes + fp16 scale shared per 64-element super-group ----
// Scale bytes identical to MXFP4 (1B/group-equivalent); +5.9dB SQNR (tools/quant_study.py).
__device__ __forceinline__ float i4_scale(const uint16_t *s, int g) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(s + (g >> 1)));
}

// Single-row decode GEMV (INSIG4).
__global__ __launch_bounds__(256) void mxfp4_gemv_v2_i4_kernel(const uint32_t *__restrict__ weights, const uint16_t *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y, int rows, int groups) {
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
    if (threadIdx.x < 16) lut[threadIdx.x] = decode4(threadIdx.x, 0);
    __syncthreads();
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_w = weights + static_cast<size_t>(row) * groups * 4;
    const uint16_t *row_s = scales + static_cast<size_t>(row) * (groups >> 1);
    float acc = 0.f;
    #define V2I(word, kb) { \
        const uint32_t w_ = (word); \
        _Pragma("unroll") \
        for (int j = 0; j < 8; j++) { \
            const float v = lut[(w_ >> (j * 4)) & 15u]; \
            const float xv = xg[(kb + j) * groups]; \
            const int which = j & 3; \
            if (which == 0) p0 = fmaf(v, xv, p0); \
            else if (which == 1) p1 = fmaf(v, xv, p1); \
            else if (which == 2) p2 = fmaf(v, xv, p2); \
            else p3 = fmaf(v, xv, p3); \
        } \
    }
    #pragma unroll 4
    for (int g0 = lane; g0 < groups; g0 += LANES) {
        const uint4 packed = __ldcs(reinterpret_cast<const uint4 *>(row_w + static_cast<size_t>(g0) * 4));
        const float scale = i4_scale(row_s, g0);
        const float *xg = sx + g0;
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
        V2I(packed.x, 0)
        V2I(packed.y, 8)
        V2I(packed.z, 16)
        V2I(packed.w, 24)
        acc = fmaf((p0 + p1) + (p2 + p3), scale, acc);
    }
    #undef V2I
    float sum = acc;
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, mask);
    if (lane == 0) y[row] = sum;
}
void mxfp4_gemv_v2_i4(const uint32_t *weights, const uint16_t *scales, const float *x, float *y, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 1023)) return;
    static const bool configured = [] { return cudaFuncSetAttribute(mxfp4_gemv_v2_i4_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024) == cudaSuccess; }();
    (void)configured;
    const int groups = cols >> 5;
    mxfp4_gemv_v2_i4_kernel<<<(rows + 7) >> 3, 256, size_t(cols) * 4 + 64, stream>>>(weights, scales, x, y, rows, groups);
}

// Pair dp4a GEMV (INSIG4): both rows staged+quantized per block.
__global__ __launch_bounds__(256) void mxfp4_gemv2_q8_i4_kernel(const uint32_t *__restrict__ weights, const uint16_t *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y, int rows, int groups) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ char smem[];
    uint32_t *xq = reinterpret_cast<uint32_t *>(smem);
    float *xs = reinterpret_cast<float *>(smem + size_t(16) * groups * 4);
    unsigned long long *btab = reinterpret_cast<unsigned long long *>(smem + size_t(16) * groups * 4 + 2 * groups * 4);
    for (int rg = threadIdx.x; rg < 2 * groups; rg += 256) {
        const int r = rg / groups, g = rg % groups;
        const float *xg = x + r * (groups * 32) + g * 32;
        float v[32], m = 0.f;
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const float4 f4 = __ldg(reinterpret_cast<const float4 *>(xg) + q);
            v[q * 4 + 0] = f4.x; v[q * 4 + 1] = f4.y; v[q * 4 + 2] = f4.z; v[q * 4 + 3] = f4.w;
            m = fmaxf(m, fmaxf(fmaxf(fabsf(f4.x), fabsf(f4.y)), fmaxf(fabsf(f4.z), fabsf(f4.w))));
        }
        xs[r * groups + g] = m * (1.f / 127.f);
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
    if (threadIdx.x < 256) {
        const signed char tbl[8] = {0, 1, 2, 3, 4, 6, 8, 12};
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
    const uint16_t *row_s = scales + static_cast<size_t>(row) * (groups >> 1);
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
            const uint32_t A = __byte_perm(uint32_t(b0), uint32_t(b1), 0xc480);
            const uint32_t B = __byte_perm(uint32_t(b0 >> 32), uint32_t(b1 >> 32), 0x4c80);
            const uint32_t w0 = __byte_perm(A, B, 0x6240);
            const uint32_t A2 = __byte_perm(uint32_t(b2), uint32_t(b3), 0xc480);
            const uint32_t B2 = __byte_perm(uint32_t(b2 >> 32), uint32_t(b3 >> 32), 0x4c80);
            const uint32_t w1 = __byte_perm(A2, B2, 0x6240);
            d0a = __dp4a(int(w0), int(reinterpret_cast<const uint32_t *>(x0)[wi * 2]), d0a);
            d1a = __dp4a(int(w0), int(reinterpret_cast<const uint32_t *>(x1)[wi * 2]), d1a);
            d0b = __dp4a(int(w1), int(reinterpret_cast<const uint32_t *>(x0)[wi * 2 + 1]), d0b);
            d1b = __dp4a(int(w1), int(reinterpret_cast<const uint32_t *>(x1)[wi * 2 + 1]), d1b);
        }
        const float ws = i4_scale(row_s, g0) * 0.5f;
        acc0 = fmaf(float(d0a + d0b), ws * xs[g0], acc0);
        acc1 = fmaf(float(d1a + d1b), ws * xs[groups + g0], acc1);
    }
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) { acc0 += __shfl_xor_sync(0xffffffff, acc0, mask); acc1 += __shfl_xor_sync(0xffffffff, acc1, mask); }
    if (lane == 0) { y[row] = acc0; y[rows + row] = acc1; }
}
void mxfp4_gemv2_q8_i4(const uint32_t *weights, const uint16_t *scales, const float *x, float *y, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 31)) return;
    const int groups = cols >> 5;
    mxfp4_gemv2_q8_i4_kernel<<<(rows + 7) >> 3, 256, size_t(16) * groups * 4 + 2 * groups * 4 + 2048, stream>>>(weights, scales, x, y, rows, groups);
}

// Fused in_proj_a+b pair kernel (INSIG4).
__global__ __launch_bounds__(256) void mxfp4_gemv_ab2_q8_i4_kernel(const uint32_t *__restrict__ wa, const uint16_t *__restrict__ sa, const uint32_t *__restrict__ wb, const uint16_t *__restrict__ sb, const float *__restrict__ x, float *__restrict__ ya, float *__restrict__ yb, int groups) {
    extern __shared__ char smem[];
    uint32_t *xq = reinterpret_cast<uint32_t *>(smem);
    float *xs = reinterpret_cast<float *>(smem + size_t(16) * groups * 4);
    unsigned long long *btab = reinterpret_cast<unsigned long long *>(smem + size_t(16) * groups * 4 + 2 * groups * 4);
    {
        const int r = threadIdx.x >> 7, g = threadIdx.x & 127;
        const float *xg = x + r * (groups * 32) + g * 32;
        float v[32], m = 0.f;
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const float4 f4 = __ldg(reinterpret_cast<const float4 *>(xg) + q);
            v[q * 4 + 0] = f4.x; v[q * 4 + 1] = f4.y; v[q * 4 + 2] = f4.z; v[q * 4 + 3] = f4.w;
            m = fmaxf(m, fmaxf(fmaxf(fabsf(f4.x), fabsf(f4.y)), fmaxf(fabsf(f4.z), fabsf(f4.w))));
        }
        xs[r * groups + g] = m * (1.f / 127.f);
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
    {
        const signed char tbl[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        const int b = threadIdx.x;
        const int lo = b & 15, hi = b >> 4;
        const uint32_t clo = uint32_t(uint8_t((lo & 8) ? -tbl[lo & 7] : tbl[lo & 7])) * 0x01010101u;
        const uint32_t chi = uint32_t(uint8_t((hi & 8) ? -tbl[hi & 7] : tbl[hi & 7])) * 0x01010101u;
        btab[threadIdx.x] = unsigned long long(clo) | (unsigned long long(chi) << 32);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint32_t *xq0 = xq, *xq1 = xq + groups * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int rr = warp * 8 + i;
        const bool is_a = rr < 32;
        const int row = is_a ? rr : rr - 32;
        const uint32_t *row_w = (is_a ? wa : wb) + static_cast<size_t>(row) * groups * 4;
        const uint16_t *row_s = (is_a ? sa : sb) + static_cast<size_t>(row) * (groups >> 1);
        float acc0 = 0.f, acc1 = 0.f;
        #pragma unroll 4
        for (int g0 = lane; g0 < groups; g0 += 32) {
            const uint4 P = __ldcs(reinterpret_cast<const uint4 *>(row_w + static_cast<size_t>(g0) * 4));
            const uint4 *x0 = reinterpret_cast<const uint4 *>(xq0 + g0 * 8);
            const uint4 *x1 = reinterpret_cast<const uint4 *>(xq1 + g0 * 8);
            int d0a = 0, d0b = 0, d1a = 0, d1b = 0;
            const uint32_t pw[4] = {P.x, P.y, P.z, P.w};
            #pragma unroll
            for (int wi = 0; wi < 4; wi++) {
                const uint32_t w_ = pw[wi];
                const unsigned long long b0 = btab[w_ & 0xff], b1 = btab[(w_ >> 8) & 0xff];
                const unsigned long long b2 = btab[(w_ >> 16) & 0xff], b3 = btab[w_ >> 24];
                const uint32_t A = __byte_perm(uint32_t(b0), uint32_t(b1), 0xc480);
                const uint32_t B = __byte_perm(uint32_t(b0 >> 32), uint32_t(b1 >> 32), 0x4c80);
                const uint32_t w0 = __byte_perm(A, B, 0x6240);
                const uint32_t A2 = __byte_perm(uint32_t(b2), uint32_t(b3), 0xc480);
                const uint32_t B2 = __byte_perm(uint32_t(b2 >> 32), uint32_t(b3 >> 32), 0x4c80);
                const uint32_t w1 = __byte_perm(A2, B2, 0x6240);
                d0a = __dp4a(int(w0), int(reinterpret_cast<const uint32_t *>(x0)[wi * 2]), d0a);
                d1a = __dp4a(int(w0), int(reinterpret_cast<const uint32_t *>(x1)[wi * 2]), d1a);
                d0b = __dp4a(int(w1), int(reinterpret_cast<const uint32_t *>(x0)[wi * 2 + 1]), d0b);
                d1b = __dp4a(int(w1), int(reinterpret_cast<const uint32_t *>(x1)[wi * 2 + 1]), d1b);
            }
            const float ws = i4_scale(row_s, g0) * 0.5f;
            acc0 = fmaf(float(d0a + d0b), ws * xs[g0], acc0);
            acc1 = fmaf(float(d1a + d1b), ws * xs[groups + g0], acc1);
        }
        #pragma unroll
        for (int mask = 16; mask; mask >>= 1) { acc0 += __shfl_xor_sync(0xffffffff, acc0, mask); acc1 += __shfl_xor_sync(0xffffffff, acc1, mask); }
        if (lane == 0) {
            if (is_a) { ya[row] = acc0; ya[32 + row] = acc1; }
            else { yb[row] = acc0; yb[32 + row] = acc1; }
        }
    }
}
void mxfp4_gemv_ab2_q8_i4(const uint32_t *wa, const uint16_t *sa, const uint32_t *wb, const uint16_t *sb, const float *x, float *ya, float *yb, int cols, cudaStream_t stream) {
    const int groups = cols >> 5;
    mxfp4_gemv_ab2_q8_i4_kernel<<<1, 256, size_t(16) * groups * 4 + 2 * groups * 4 + 2048, stream>>>(wa, sa, wb, sb, x, ya, yb, groups);
}

// Embedding row gather (INSIG4).
__global__ void mxfp4_get_row_i4_kernel(const uint32_t *weights, const uint16_t *scales, float *out, const int *row_dev, int groups) {
    const int row = __ldg(row_dev);
    for (int group = blockIdx.x; group < groups; group += gridDim.x) {
        const int lane = threadIdx.x;
        const uint32_t packed = weights[(static_cast<size_t>(row) * groups + group) * 4 + (lane >> 3)];
        const uint8_t q = uint8_t(packed >> ((lane & 7) * 4));
        out[group * 32 + lane] = fp4_e2m1(q) * i4_scale(scales + static_cast<size_t>(row) * (groups >> 1), group);
    }
}
void mxfp4_get_row_i4(const uint32_t *w, const uint16_t *s, float *out, const int *row_dev, int cols, cudaStream_t stream) {
    mxfp4_get_row_i4_kernel<<<((cols >> 5) < 128 ? (cols >> 5) : 128), 32, 0, stream>>>(w, s, out, row_dev, cols >> 5);
}

}
