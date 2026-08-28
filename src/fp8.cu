#include <stdexcept>
#include <string>
#include "insignia_fp8.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
namespace insignia {
using namespace nvcuda;

// Decode GEMV: Y[rows] = W[rows,cols]·x, W e4m3, scales bf16 per 128x128 tile.
// One warp per row, 8 warps per block; x staged once per block in shared memory.
// Per round a lane eats 16 consecutive weights (uint4) — always inside one 128-col
// scale block — applies that block's scale to its partial, and FMA-reduces at warp end.
// STAGE=false skips the smem copy for wide matrices (cols*4 > 48KB would cap occupancy
// at 1 block/SM via the opt-in): x is tiny against the 48MB L2, direct __ldg reads win.
template <bool STAGE>
__global__ __launch_bounds__(256) void fp8_gemv_kernel(const uint8_t *__restrict__ weights, const uint16_t *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y, int rows, int cols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ float sx[];
    if (STAGE) {
        for (int c0 = threadIdx.x * 16; c0 < cols; c0 += blockDim.x * 16) {
            const float4 *xr = reinterpret_cast<const float4 *>(x + c0);
            float4 *sr = reinterpret_cast<float4 *>(sx + c0);
            #pragma unroll
            for (int q = 0; q < 4; q++) sr[q] = __ldg(xr + q);
        }
        __syncthreads();
    }
    const float *const xs = STAGE ? sx : x;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const int kblocks = cols >> 7;
    const uint8_t *row_w = weights + static_cast<size_t>(row) * cols;
    const uint16_t *row_s = scales + static_cast<size_t>(row >> 7) * kblocks;
    float acc = 0.f;
    #pragma unroll 2
    for (int c0 = lane * 16; c0 < cols; c0 += 512) {
        const uint4 packed = __ldcs(reinterpret_cast<const uint4 *>(row_w + c0));
        const float sc = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(row_s + (c0 >> 7)));
        const uint32_t pw[4] = {packed.x, packed.y, packed.z, packed.w};
        const float4 *xv = reinterpret_cast<const float4 *>(xs + c0);
        float part = 0.f;
        #pragma unroll
        for (int wi = 0; wi < 4; wi++) {
            const float2 a = e4m3x2(pw[wi]), b2 = e4m3x2(pw[wi] >> 8);
            const float4 v = xv[wi];
            part = fmaf(a.x, v.x, part);
            part = fmaf(b2.x, v.y, part);
            part = fmaf(a.y, v.z, part);
            part = fmaf(b2.y, v.w, part);
        }
        acc = fmaf(part, sc, acc);
    }
    #pragma unroll
    for (int m = 16; m; m >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, m);
    if (!lane) y[row] = acc;
}
void fp8_gemv(const uint8_t *weights, const uint16_t *scales, const float *x, float *y, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 127)) throw std::runtime_error("insignia: bad GEMV/GEMM dims rows=" + std::to_string(rows) + " cols=" + std::to_string(cols));
    if (reinterpret_cast<uintptr_t>(weights) & 15) throw std::runtime_error("insignia: fp8 weights not 16B aligned (shard rebase missing)");
    const size_t smem = size_t(cols) * 4;
    if (smem <= 48 * 1024) {
        fp8_gemv_kernel<true><<<(rows + 7) >> 3, 256, smem, stream>>>(weights, scales, x, y, rows, cols);
    } else {
        fp8_gemv_kernel<false><<<(rows + 7) >> 3, 256, 0, stream>>>(weights, scales, x, y, rows, cols);
    }
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("insignia: fp8_gemv launch failed cols=" + std::to_string(cols));
}

// Pair decode GEMV: two activation rows against one weight pass (spec verify).
__global__ __launch_bounds__(256) void fp8_gemv2_kernel(const uint8_t *__restrict__ weights, const uint16_t *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y, int rows, int cols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ float sx[];
    for (int c0 = threadIdx.x * 16; c0 < 2 * cols; c0 += blockDim.x * 16) {
        const float4 *xr = reinterpret_cast<const float4 *>(x + c0);
        float4 *sr = reinterpret_cast<float4 *>(sx + c0);
        #pragma unroll
        for (int q = 0; q < 4; q++) sr[q] = __ldg(xr + q);
    }
    __syncthreads();
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const int kblocks = cols >> 7;
    const uint8_t *row_w = weights + static_cast<size_t>(row) * cols;
    const uint16_t *row_s = scales + static_cast<size_t>(row >> 7) * kblocks;
    float acc0 = 0.f, acc1 = 0.f;
    #pragma unroll 2
    for (int c0 = lane * 16; c0 < cols; c0 += 512) {
        const uint4 packed = __ldcs(reinterpret_cast<const uint4 *>(row_w + c0));
        const float sc = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(row_s + (c0 >> 7)));
        const uint32_t pw[4] = {packed.x, packed.y, packed.z, packed.w};
        const float4 *x0 = reinterpret_cast<const float4 *>(sx + c0);
        const float4 *x1 = reinterpret_cast<const float4 *>(sx + cols + c0);
        float p0 = 0.f, p1 = 0.f;
        #pragma unroll
        for (int wi = 0; wi < 4; wi++) {
            const float2 a = e4m3x2(pw[wi]), b2 = e4m3x2(pw[wi] >> 8);
            const float4 v0 = x0[wi], v1 = x1[wi];
            p0 = fmaf(a.x, v0.x, p0); p0 = fmaf(b2.x, v0.y, p0); p0 = fmaf(a.y, v0.z, p0); p0 = fmaf(b2.y, v0.w, p0);
            p1 = fmaf(a.x, v1.x, p1); p1 = fmaf(b2.x, v1.y, p1); p1 = fmaf(a.y, v1.z, p1); p1 = fmaf(b2.y, v1.w, p1);
        }
        acc0 = fmaf(p0, sc, acc0);
        acc1 = fmaf(p1, sc, acc1);
    }
    #pragma unroll
    for (int m = 16; m; m >>= 1) { acc0 += __shfl_xor_sync(0xffffffff, acc0, m); acc1 += __shfl_xor_sync(0xffffffff, acc1, m); }
    if (!lane) { y[row] = acc0; y[rows + row] = acc1; }
}
void fp8_gemv2(const uint8_t *weights, const uint16_t *scales, const float *x, float *y, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 127)) throw std::runtime_error("insignia: bad GEMV/GEMM dims rows=" + std::to_string(rows) + " cols=" + std::to_string(cols));
    if (reinterpret_cast<uintptr_t>(weights) & 15) throw std::runtime_error("insignia: fp8 weights not 16B aligned (shard rebase missing)");
    const size_t smem = size_t(2) * cols * 4;
    if (smem > 99 * 1024) throw std::runtime_error("insignia: fp8_gemv2 staging exceeds 99KB smem at cols=" + std::to_string(cols) + " (use fp8_gemm for the pair path)");
    static const bool configured = [] { return cudaFuncSetAttribute(fp8_gemv2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024) == cudaSuccess; }();
    (void)configured;
    fp8_gemv2_kernel<<<(rows + 7) >> 3, 256, smem, stream>>>(weights, scales, x, y, rows, cols);
}

// Prefill GEMM: Y[T,rows] = X16[T,cols]·W^T with W e4m3 + bf16 128x128 block scales.
// v2.1-style cp.async pipeline; KT=64 (two steps share one 128-col scale block).
// B dequant e4m3->bf16 (exact) in registers. Requires rows%32==0, cols%128==0,
// 1<=T<=64, x16 zero-padded to 64 rows.
__global__ __launch_bounds__(256) void fp8_gemm_kernel(const uint8_t *__restrict__ weights, const uint16_t *__restrict__ scales, const __nv_bfloat16 *__restrict__ x16, float *__restrict__ y, int rows, int cols, int T) {
    constexpr int KT = 64, NT = 32;
    constexpr int APAD = 8, BPAD = 8;
    __shared__ __nv_bfloat16 As[2][64][KT + APAD];
    __shared__ __nv_bfloat16 Bs[2][NT][KT + BPAD];
    __shared__ uint8_t Braw[2][NT][KT];
    const int n0 = blockIdx.x * NT;
    const int tid = threadIdx.x;
    const int kblocks = cols >> 7;

    auto prefetch = [&](int kb, int buf) {
        const int k = kb * KT;
        {   // A: 64 rows x 128B (64 bf16) = 512 chunks of 16B
            const char *ag = reinterpret_cast<const char *>(x16);
            for (int i = tid; i < 64 * (KT / 8); i += 256) {
                const int m = i / (KT / 8), c8 = i % (KT / 8);
                const size_t srcoff = (size_t(m) * cols + k) * 2 + c8 * 16;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((unsigned)__cvta_generic_to_shared(&As[buf][m][c8 * 8])), "l"(ag + srcoff));
            }
        }
        {   // B: NT rows x 64B = NT*4 chunks of 16B
            for (int i = tid; i < NT * 4; i += 256) {
                const int n = i >> 2, chunk = i & 3;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((unsigned)__cvta_generic_to_shared(&Braw[buf][n][chunk * 16])), "l"(weights + (size_t(n0 + n) * cols + k) + chunk * 16));
            }
        }
    };
    auto dequant = [&](int kb, int buf) {
        // 256 threads = 32 rows x 8 threads; each converts 8 e4m3 -> 8 bf16 (16B store)
        const int n = tid >> 3, w = tid & 7;
        const uint32_t sb = uint32_t(__ldg(scales + size_t((n0 + n) >> 7) * kblocks + (kb >> 1))) << 16;
        const float sc = __uint_as_float(sb);
        const uint2 p = *reinterpret_cast<const uint2 *>(&Braw[buf][n][w * 8]);
        const uint32_t pw[2] = {p.x, p.y};
        __nv_bfloat16 out[8];
        #pragma unroll
        for (int wi = 0; wi < 2; wi++) {
            const float2 a = e4m3x2(pw[wi]), b2 = e4m3x2(pw[wi] >> 8);
            out[wi * 4 + 0] = __float2bfloat16(a.x * sc);
            out[wi * 4 + 1] = __float2bfloat16(b2.x * sc);
            out[wi * 4 + 2] = __float2bfloat16(a.y * sc);
            out[wi * 4 + 3] = __float2bfloat16(b2.y * sc);
        }
        *reinterpret_cast<uint4 *>(&Bs[buf][n][w * 8]) = *reinterpret_cast<const uint4 *>(out);
    };

    const int ksteps = cols / KT;
    prefetch(0, 0);
    asm volatile("cp.async.commit_group;\n");

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    const int warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;

    for (int kb = 0; kb < ksteps; kb++) {
        const int buf = kb & 1;
        if (kb + 1 < ksteps) { prefetch(kb + 1, buf ^ 1); asm volatile("cp.async.commit_group;\n"); }
        if (kb + 2 < ksteps) asm volatile("cp.async.wait_group 1;\n");
        else asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        dequant(kb, buf);
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[4];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf[4];
        #pragma unroll
        for (int kh = 0; kh < 4; kh++) {
            wmma::load_matrix_sync(af[kh], &As[buf][wm * 16][kh * 16], KT + APAD);
            wmma::load_matrix_sync(bf[kh], &Bs[buf][wn * 16][kh * 16], KT + BPAD);
            wmma::mma_sync(acc, af[kh], bf[kh], acc);
        }
        __syncthreads();
    }
    // Y is written as full 16-row tiles: the caller's buffer must be 64-row padded
    // (rows T..63 stay stale, never read). T>64 would leave rows unwritten silently.
    if (wm * 16 < T) wmma::store_matrix_sync(y + size_t(wm * 16) * rows + n0 + wn * 16, acc, rows, wmma::mem_row_major);
}
void fp8_gemm(const uint8_t *weights, const uint16_t *scales, const void *x16, float *y, int rows, int cols, int T, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 127) || T <= 0 || (rows & 31)) throw std::runtime_error("insignia: bad GEMM dims rows=" + std::to_string(rows) + " cols=" + std::to_string(cols));
    if (T > 64) throw std::runtime_error("insignia: fp8_gemm T=" + std::to_string(T) + " exceeds the 64-row A tile");
    fp8_gemm_kernel<<<rows >> 5, 256, 0, stream>>>(weights, scales, (const __nv_bfloat16 *)x16, y, rows, cols, T);
}

// Row gather from a bf16 matrix (embed table).
__global__ void bf16_get_row_kernel(const uint16_t *__restrict__ w, float *__restrict__ out, const int *__restrict__ row_dev, int cols) {
    const int row = __ldg(row_dev);
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += gridDim.x * blockDim.x)
        out[c] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(w + static_cast<size_t>(row) * cols + c));
}
void bf16_get_row(const uint16_t *w, float *out, const int *row_dev, int cols, cudaStream_t stream) {
    bf16_get_row_kernel<<<(cols + 255) / 256, 256, 0, stream>>>(w, out, row_dev, cols);
}

}  // namespace insignia
