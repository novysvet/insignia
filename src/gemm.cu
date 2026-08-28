#include <stdexcept>
#include <string>
#include "insignia_layout.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
namespace insignia {
using namespace nvcuda;

// cp.async 16B global->shared + group waits (sm_80+; latency-hiding staging)
__device__ __forceinline__ void cp_async16(void *smem, const char *global) {
    const unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(global));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_prev() { asm volatile("cp.async.wait_group 1;\n"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n"); }


// Batched prefill GEMM for sm_89: Y[T,rows] = X[T,cols] * W[rows,cols]^T with W in MLX MXFP4.
// One 256-thread block computes a 64x64 output tile; 8 warps as a 4x2 grid of 16x32 wmma
// tiles. K advances in 32-column steps (one E8M0 scale group per step). A comes from fp32
// activations converted to bf16 (rows beyond T stage as zero), B is fp4*scale dequant
// (exact in bf16), accumulate fp32. Requires rows%64==0, cols%32==0, and Y with 64 allocated
// rows regardless of T.
__global__ __launch_bounds__(256) void mxfp4_gemm_mlx_kernel(const uint32_t *__restrict__ weights, const uint8_t *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y, int rows, int cols, int T) {
    constexpr int KT = 32, NT = 64;
    const int n0 = blockIdx.x * NT;
    __shared__ __nv_bfloat16 At[64][KT + 16];  // 48-half rows: 16B-aligned for ldmatrix
    __shared__ __nv_bfloat16 Bt[NT][KT + 16];  // 48-half rows: 16B-aligned for ldmatrix
    const int groups = cols >> 5;
    const int tid = threadIdx.x;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.f);
    wmma::fill_fragment(acc1, 0.f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af0, af1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf00, bf01, bf10, bf11;
    for (int k0 = 0; k0 < cols; k0 += KT) {
        if (tid < 64) {  // Stage A: one activation row per thread, 32 cols as 8 float4 reads.
            for (int k = 0; k < KT; k += 4) {
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (tid < T) v = *reinterpret_cast<const float4 *>(x + static_cast<size_t>(tid) * cols + k0 + k);
                At[tid][k] = __float2bfloat16(v.x);
                At[tid][k + 1] = __float2bfloat16(v.y);
                At[tid][k + 2] = __float2bfloat16(v.z);
                At[tid][k + 3] = __float2bfloat16(v.w);
            }
        }
        {   // Stage B: dequant 64 rows x 32 cols (thread -> one row's one word).
            const int n = tid >> 2, w = tid & 3;
            const uint32_t word = weights[static_cast<size_t>(n0 + n) * groups * 4 + (k0 >> 3) + w];
            const float scale = __int_as_float(static_cast<uint32_t>(scales[static_cast<size_t>(n0 + n) * groups + (k0 >> 5)]) << 23);
            alignas(16) __nv_bfloat16 outv[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) outv[j] = __float2bfloat16(decode4(word, j) * scale);
            *reinterpret_cast<uint4 *>(&Bt[n][w << 3]) = *reinterpret_cast<const uint4 *>(outv);
        }
        __syncthreads();
        const int warp = tid >> 5, warp_m = warp >> 1, warp_n = warp & 1;
        wmma::load_matrix_sync(af0, &At[warp_m << 4][0], KT + 16);
        wmma::load_matrix_sync(af1, &At[warp_m << 4][16], KT + 16);
        wmma::load_matrix_sync(bf00, &Bt[warp_n << 5][0], KT + 16);
        wmma::load_matrix_sync(bf01, &Bt[(warp_n << 5) + 16][0], KT + 16);
        wmma::load_matrix_sync(bf10, &Bt[warp_n << 5][16], KT + 16);
        wmma::load_matrix_sync(bf11, &Bt[(warp_n << 5) + 16][16], KT + 16);
        wmma::mma_sync(acc0, af0, bf00, acc0);
        wmma::mma_sync(acc0, af1, bf10, acc0);
        wmma::mma_sync(acc1, af0, bf01, acc1);
        wmma::mma_sync(acc1, af1, bf11, acc1);
        __syncthreads();
    }
    const int warp = tid >> 5, warp_m = warp >> 1, warp_n = warp & 1;
    wmma::store_matrix_sync(y + static_cast<size_t>(warp_m << 4) * rows + n0 + (warp_n << 5), acc0, rows, wmma::mem_row_major);
    wmma::store_matrix_sync(y + static_cast<size_t>(warp_m << 4) * rows + n0 + (warp_n << 5) + 16, acc1, rows, wmma::mem_row_major);
}

void mxfp4_gemm_mlx(const uint32_t *weights, const uint8_t *scales, const float *x, float *y, int rows, int cols, int T, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 31) || T <= 0 || (rows & 63)) throw std::runtime_error("insignia: bad GEMV/GEMM dims rows=" + std::to_string(rows) + " cols=" + std::to_string(cols));
    const int blocks = rows >> 6;
    mxfp4_gemm_mlx_kernel<<<blocks, 256, 0, stream>>>(weights, scales, x, y, rows, cols, T);
}
}

namespace insignia {
using namespace nvcuda;

// Pipelined MXFP4 GEMM for prefill: Y[T,rows] = X[T,cols] * W[rows,cols]^T.
// Block computes a 64x64 tile; K advances in 64-column steps (two E8M0 scale groups).
// Staging is fully parallel: 256 threads = 64 B-rows x 4 u32-words, each thread
// dequants its 8 nibble-pairs through the 256-entry bf16 pair-LUT with the group
// scale folded in; A converts fp32->bf16 across all threads. Two-stage pipeline.
// 8 warps: warp w owns m-tile (w&3) and the two n-tiles at n0+32*(w>>2).
// Requires rows%64==0, cols%64==0, 1<=T<=64 (rows beyond T contribute zeros).
__global__ __launch_bounds__(256) void mxfp4_gemm_v2_kernel(const uint32_t *__restrict__ weights, const uint8_t *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y, int rows, int cols, int T) {
    constexpr int KT = 64;
    constexpr int APAD = 8, BPAD = 8;
    __shared__ __nv_bfloat16 As[2][64][KT + APAD];
    __shared__ __nv_bfloat16 Bs[2][64][KT + BPAD];
    __shared__ uint32_t lut[256];
    {
        const unsigned tbl[16] = {0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
                                  0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0};
        lut[threadIdx.x] = tbl[threadIdx.x & 15] | (tbl[(threadIdx.x >> 4) & 15] << 16);
    }
    const int groups = cols >> 5;
    const int n0 = blockIdx.x * 64;
    const int tid = threadIdx.x;

    auto stage = [&](int kb, int buf) {   // K step kb covers cols [kb*KT, kb*KT+KT)
        const int k = kb * KT;
        for (int i = tid; i < 64 * KT; i += 256) {
            const int m = i / KT, c = i & (KT - 1);
            As[buf][m][c] = __float2bfloat16(m < T ? x[size_t(m) * cols + k + c] : 0.f);
        }
        {   // B: thread = (row, word): 64 rows x 16 words of 4 nibble-pairs each
            const int n = tid >> 2, w = tid & 3;                  // word w in [0,4): 16B = 4 u32
            const uint32_t word = weights[size_t(n0 + n) * groups * 4 + k / 8 + w];
            const int g = kb * 2;                                  // words 0..3 all live in group kb*2
            const float sc = __int_as_float(uint32_t(scales[size_t(n0 + n) * groups + g]) << 23);
            __nv_bfloat16 out[8];
            #pragma unroll
            for (int byt = 0; byt < 4; byt++) {
                const uint32_t pair = lut[(word >> (byt * 8)) & 0xff];
                const float lo = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&pair));
                const float hi = __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(&pair)[1]);
                out[byt * 2 + 0] = __float2bfloat16(lo * sc);
                out[byt * 2 + 1] = __float2bfloat16(hi * sc);
            }
            *reinterpret_cast<uint4 *>(&Bs[buf][n][w * 8]) = *reinterpret_cast<const uint4 *>(out);
        }
        {   // second scale group of this K step: cols [k+32, k+64), words 4..7
            const int n = tid >> 2, w = (tid & 3) + 4;
            const uint32_t word = weights[size_t(n0 + n) * groups * 4 + k / 8 + w];
            const float sc = __int_as_float(uint32_t(scales[size_t(n0 + n) * groups + kb * 2 + 1]) << 23);
            __nv_bfloat16 out[8];
            #pragma unroll
            for (int byt = 0; byt < 4; byt++) {
                const uint32_t pair = lut[(word >> (byt * 8)) & 0xff];
                const float lo = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&pair));
                const float hi = __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(&pair)[1]);
                out[byt * 2 + 0] = __float2bfloat16(lo * sc);
                out[byt * 2 + 1] = __float2bfloat16(hi * sc);
            }
            *reinterpret_cast<uint4 *>(&Bs[buf][n][w * 8]) = *reinterpret_cast<const uint4 *>(out);
        }
    };

    stage(0, 0);
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    wmma::fill_fragment(acc[0], 0.f);
    wmma::fill_fragment(acc[1], 0.f);
    const int warp = tid >> 5;
    const int wm = warp & 3, wn = warp >> 2;

    const int ksteps = cols / KT;
    for (int kb = 0; kb < ksteps; kb++) {
        const int buf = kb & 1;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[4];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b0[4], b1[4];
        #pragma unroll
        for (int kh = 0; kh < 4; kh++) {
            wmma::load_matrix_sync(af[kh], &As[buf][wm * 16][kh * 16], KT + APAD);
            wmma::load_matrix_sync(b0[kh], &Bs[buf][wn * 32][kh * 16], KT + BPAD);
            wmma::load_matrix_sync(b1[kh], &Bs[buf][wn * 32 + 16][kh * 16], KT + BPAD);
        }
        #pragma unroll
        for (int kh = 0; kh < 4; kh++) {
            wmma::mma_sync(acc[0], af[kh], b0[kh], acc[0]);
            wmma::mma_sync(acc[1], af[kh], b1[kh], acc[1]);
        }
        __syncthreads();
        if (kb + 1 < ksteps) stage(kb + 1, buf ^ 1);
        __syncthreads();
    }
    wmma::store_matrix_sync(y + size_t(wm * 16) * rows + n0 + wn * 32, acc[0], rows, wmma::mem_row_major);
    wmma::store_matrix_sync(y + size_t(wm * 16) * rows + n0 + wn * 32 + 16, acc[1], rows, wmma::mem_row_major);
}

void mxfp4_gemm_v2(const uint32_t *weights, const uint8_t *scales, const float *x, float *y, int rows, int cols, int T, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 63) || T <= 0 || (rows & 63)) throw std::runtime_error("insignia: bad GEMV/GEMM dims rows=" + std::to_string(rows) + " cols=" + std::to_string(cols));
    mxfp4_gemm_v2_kernel<<<rows >> 6, 256, 0, stream>>>(weights, scales, x, y, rows, cols, T);
}
}

namespace insignia {

// fp32 -> bf16 activation copy (GEMM A-side scratch; T rows x cols).
__global__ void f32_to_bf16_kernel(const float *__restrict__ x, __nv_bfloat16 *__restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}
void f32_to_bf16(const float *x, void *y, int n, cudaStream_t stream) {
    f32_to_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, (__nv_bfloat16 *)y, n);
}

}

namespace insignia {
using namespace nvcuda;
using namespace ::insignia;

// Pipelined MXFP4 GEMM v2.1: Y[T,rows] = X16[T,cols] * W[rows,cols]^T with X16 bf16.
// 64x32 tile per 256-thread block (8 warps, one 16x16 output tile each); K advances
// 64 columns per step (two E8M0 groups). cp.async stages the next A tile and B
// nibbles while the warps dequant-and-mma the current step, so global latency hides.
// B dequant uses the 256-entry bf16 pair-LUT with group scales folded in.
// Requires rows%32==0, cols%64==0, 1<=T<=64 (A rows beyond T must be zero).
__global__ __launch_bounds__(256) void mxfp4_gemm_v21_kernel(const uint32_t *__restrict__ weights, const uint8_t *__restrict__ scales, const __nv_bfloat16 *__restrict__ x16, float *__restrict__ y, int rows, int cols, int T) {
    constexpr int KT = 64, NT = 32;
    constexpr int APAD = 8, BPAD = 8;
    __shared__ __nv_bfloat16 As[2][64][KT + APAD];
    __shared__ __nv_bfloat16 Bs[2][NT][KT + BPAD];
    __shared__ uint32_t Braw[2][NT][8];       // packed nibbles, 8 u32 per row per step
    __shared__ uint32_t lut[256];
    {
        const unsigned tbl[16] = {0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
                                  0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0};
        lut[threadIdx.x] = tbl[threadIdx.x & 15] | (tbl[(threadIdx.x >> 4) & 15] << 16);
    }
    const int groups = cols >> 5;
    const int n0 = blockIdx.x * NT;
    const int tid = threadIdx.x;
    const uint32_t *b16base = reinterpret_cast<const uint32_t *>(x16);  // A as raw 16B chunks

    auto prefetch = [&](int kb, int buf) {   // cp.async: A tile + B nibbles for K step kb
        const int k = kb * KT;
        {   // A: 64 rows x 128B (64 bf16) = 512 chunks of 16B; thread strides
            const char *ag = reinterpret_cast<const char *>(x16);
            for (int i = tid; i < 64 * (KT / 8); i += 256) {
                const int m = i / (KT / 8), c8 = i % (KT / 8);   // c8: which 8-bf16 chunk
                const size_t srcoff = (size_t(m) * cols + k) * 2 + c8 * 16;
                cp_async16(&As[buf][m][c8 * 8], ag + srcoff);
            }
        }
        {   // B: NT rows x 32B nibbles = NT*2 chunks of 16B
            for (int i = tid; i < NT * 2; i += 256) {
                const int n = i >> 1, half = i & 1;
                cp_async16(&Braw[buf][n][half * 4], reinterpret_cast<const char *>(weights + size_t(n0 + n) * groups * 4 + k / 8 + half * 4));
            }
        }
    };
    auto dequant = [&](int kb, int buf) {    // Braw -> Bs with group scales (regs -> shared)
        const int n = tid >> 3, w = tid & 7;  // 256 threads = 32 rows x 8 words
        if (n < NT) {
            const int half = w >> 2;
            const uint32_t word = Braw[buf][n][w];
            const float sc = __int_as_float(uint32_t(__ldg(scales + size_t(n0 + n) * groups + kb * 2 + half)) << 23);
            __nv_bfloat16 out[8];
            #pragma unroll
            for (int byt = 0; byt < 4; byt++) {
                const uint32_t pair = lut[(word >> (byt * 8)) & 0xff];
                const float lo = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&pair));
                const float hi = __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(&pair)[1]);
                out[byt * 2 + 0] = __float2bfloat16(lo * sc);
                out[byt * 2 + 1] = __float2bfloat16(hi * sc);
            }
            *reinterpret_cast<uint4 *>(&Bs[buf][n][w * 8]) = *reinterpret_cast<const uint4 *>(out);
        }
    };

    const int ksteps = cols / KT;
    prefetch(0, 0);
    cp_async_commit();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    const int warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;   // warp tile: m16 x n16

    for (int kb = 0; kb < ksteps; kb++) {
        const int buf = kb & 1;
        if (kb + 1 < ksteps) { prefetch(kb + 1, buf ^ 1); cp_async_commit(); }
        if (kb + 2 < ksteps) cp_async_wait_prev();   // current buf's loads complete (one group back)
        else cp_async_wait_all();                    // tail: no group was committed this step
        __syncthreads();
        dequant(kb, buf);       // Bs[buf] ready after this sync
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[4];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf[4];
        #pragma unroll
        for (int kh = 0; kh < 4; kh++) {
            wmma::load_matrix_sync(af[kh], &As[buf][wm * 16][kh * 16], KT + APAD);
            wmma::load_matrix_sync(bf[kh], &Bs[buf][wn * 16][kh * 16], KT + BPAD);
            wmma::mma_sync(acc, af[kh], bf[kh], acc);
        }
        __syncthreads();       // everyone done with buf before it is reused two steps later
    }
    wmma::store_matrix_sync(y + size_t(wm * 16) * rows + n0 + wn * 16, acc, rows, wmma::mem_row_major);
}

void mxfp4_gemm_v21(const uint32_t *weights, const uint8_t *scales, const void *x16, float *y, int rows, int cols, int T, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 63) || T <= 0 || (rows & 31)) throw std::runtime_error("insignia: bad GEMM dims rows=" + std::to_string(rows) + " cols=" + std::to_string(cols));
    mxfp4_gemm_v21_kernel<<<rows >> 5, 256, 0, stream>>>(weights, scales, (const __nv_bfloat16 *)x16, y, rows, cols, T);
}
}

namespace insignia {
using namespace nvcuda;

// INSIG4 prefill GEMM: same tiling as mxfp4_gemm_mlx; scales are fp16 per 64-group.
__global__ __launch_bounds__(256) void mxfp4_gemm_mlx_i4_kernel(const uint32_t *__restrict__ weights, const uint16_t *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y, int rows, int cols, int T) {
    constexpr int KT = 32, NT = 64;
    const int n0 = blockIdx.x * NT;
    __shared__ __nv_bfloat16 At[64][KT + 16];
    __shared__ __nv_bfloat16 Bt[NT][KT + 16];
    const int groups = cols >> 5;
    const int tid = threadIdx.x;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.f);
    wmma::fill_fragment(acc1, 0.f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af0, af1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf00, bf01, bf10, bf11;
    const uint16_t *sg_scales = scales;  // [rows][groups/2]
    for (int k0 = 0; k0 < cols; k0 += KT) {
        if (tid < 64) {
            for (int k = 0; k < KT; k += 4) {
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (tid < T) v = *reinterpret_cast<const float4 *>(x + static_cast<size_t>(tid) * cols + k0 + k);
                At[tid][k] = __float2bfloat16(v.x);
                At[tid][k + 1] = __float2bfloat16(v.y);
                At[tid][k + 2] = __float2bfloat16(v.z);
                At[tid][k + 3] = __float2bfloat16(v.w);
            }
        }
        {
            const int n = tid >> 2, w = tid & 3;
            const uint32_t word = weights[static_cast<size_t>(n0 + n) * groups * 4 + (k0 >> 3) + w];
            const float scale = __half2float(*reinterpret_cast<const __half *>(sg_scales + static_cast<size_t>(n0 + n) * (groups >> 1) + ((k0 >> 5) >> 1)));
            alignas(16) __nv_bfloat16 outv[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) outv[j] = __float2bfloat16(decode4(word, j) * scale);
            *reinterpret_cast<uint4 *>(&Bt[n][w << 3]) = *reinterpret_cast<const uint4 *>(outv);
        }
        __syncthreads();
        const int warp = tid >> 5, warp_m = warp >> 1, warp_n = warp & 1;
        wmma::load_matrix_sync(af0, &At[warp_m << 4][0], KT + 16);
        wmma::load_matrix_sync(af1, &At[warp_m << 4][16], KT + 16);
        wmma::load_matrix_sync(bf00, &Bt[warp_n << 5][0], KT + 16);
        wmma::load_matrix_sync(bf01, &Bt[(warp_n << 5) + 16][0], KT + 16);
        wmma::load_matrix_sync(bf10, &Bt[warp_n << 5][16], KT + 16);
        wmma::load_matrix_sync(bf11, &Bt[(warp_n << 5) + 16][16], KT + 16);
        wmma::mma_sync(acc0, af0, bf00, acc0);
        wmma::mma_sync(acc0, af1, bf10, acc0);
        wmma::mma_sync(acc1, af0, bf01, acc1);
        wmma::mma_sync(acc1, af1, bf11, acc1);
        __syncthreads();
    }
    const int warp = tid >> 5, warp_m = warp >> 1, warp_n = warp & 1;
    wmma::store_matrix_sync(y + static_cast<size_t>(warp_m << 4) * rows + n0 + (warp_n << 5), acc0, rows, wmma::mem_row_major);
    wmma::store_matrix_sync(y + static_cast<size_t>(warp_m << 4) * rows + n0 + (warp_n << 5) + 16, acc1, rows, wmma::mem_row_major);
}
void mxfp4_gemm_mlx_i4(const uint32_t *weights, const uint16_t *scales, const float *x, float *y, int rows, int cols, int T, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 31) || T <= 0 || (rows & 63)) throw std::runtime_error("insignia: bad GEMV/GEMM dims rows=" + std::to_string(rows) + " cols=" + std::to_string(cols));
    mxfp4_gemm_mlx_i4_kernel<<<rows >> 6, 256, 0, stream>>>(weights, scales, x, y, rows, cols, T);
}
}
namespace insignia {
using namespace nvcuda;

// Pipelined INSIG4 prefill GEMM v2.1: Y[T,rows] = X16[T,cols] * W[rows,cols]^T, X16 bf16.
// Identical pipeline to mxfp4_gemm_v21 (64x32 tile, 256 threads, KT=64, cp.async double
// buffer, wmma 16x16x16 bf16 / f32 acc, 256-entry pair-LUT dequant). Only the scale path
// differs: INSIG4 has ONE fp16 scale per 64-element super-group per row, and a KT=64 step
// covers exactly one super-group -> scale index = kb (e8m0 needed kb*2 + half).
// B nibble packing is identical to MLX MXFP4 (4 u32 per 32-elt group), so the prefetch
// geometry is v21 verbatim. Requires rows%32==0, cols%1024==0, 1<=T<=64, X16 zero-padded
// to 64 rows. Store is guarded: only A-tiles with wm*16 < T are written (callers' pf_*
// buffers are 64-row, so a tile straddling T still writes harmless scratch rows).
__global__ __launch_bounds__(256) void mxfp4_gemm_v21_i4_kernel(const uint32_t *__restrict__ weights, const uint16_t *__restrict__ scales, const __nv_bfloat16 *__restrict__ x16, float *__restrict__ y, int rows, int cols, int T) {
    constexpr int KT = 64, NT = 32;
    constexpr int APAD = 8, BPAD = 8;
    __shared__ __nv_bfloat16 As[2][64][KT + APAD];
    __shared__ __nv_bfloat16 Bs[2][NT][KT + BPAD];
    __shared__ uint32_t Braw[2][NT][8];
    __shared__ uint32_t lut[256];             // INSIG4 E2M1 == MXFP4 code space: v21 table verbatim
    {
        const unsigned tbl[16] = {0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
                                  0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0};
        lut[threadIdx.x] = tbl[threadIdx.x & 15] | (tbl[(threadIdx.x >> 4) & 15] << 16);
    }
    const int groups = cols >> 5;             // 32-elt nibble groups per row; scales stride groups>>1
    const int n0 = blockIdx.x * NT;
    const int tid = threadIdx.x;

    auto prefetch = [&](int kb, int buf) {
        const int k = kb * KT;
        {
            const char *ag = reinterpret_cast<const char *>(x16);
            for (int i = tid; i < 64 * (KT / 8); i += 256) {
                const int m = i / (KT / 8), c8 = i % (KT / 8);
                const size_t srcoff = (size_t(m) * cols + k) * 2 + c8 * 16;
                cp_async16(&As[buf][m][c8 * 8], ag + srcoff);
            }
        }
        {
            for (int i = tid; i < NT * 2; i += 256) {
                const int n = i >> 1, half = i & 1;
                cp_async16(&Braw[buf][n][half * 4], reinterpret_cast<const char *>(weights + size_t(n0 + n) * groups * 4 + k / 8 + half * 4));
            }
        }
    };
    auto dequant = [&](int kb, int buf) {    // ONE fp16 super-group scale per row per step
        const int n = tid >> 3, w = tid & 7;
        const float sc = __half2float(__ldg(reinterpret_cast<const __half *>(scales) + size_t(n0 + n) * (groups >> 1) + kb));
        const uint32_t word = Braw[buf][n][w];
        __nv_bfloat16 out[8];
        #pragma unroll
        for (int byt = 0; byt < 4; byt++) {
            const uint32_t pair = lut[(word >> (byt * 8)) & 0xff];
            const float lo = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&pair));
            const float hi = __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(&pair)[1]);
            out[byt * 2 + 0] = __float2bfloat16(lo * sc);
            out[byt * 2 + 1] = __float2bfloat16(hi * sc);
        }
        *reinterpret_cast<uint4 *>(&Bs[buf][n][w * 8]) = *reinterpret_cast<const uint4 *>(out);
    };

    const int ksteps = cols / KT;
    prefetch(0, 0);
    cp_async_commit();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    const int warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;

    for (int kb = 0; kb < ksteps; kb++) {
        const int buf = kb & 1;
        if (kb + 1 < ksteps) { prefetch(kb + 1, buf ^ 1); cp_async_commit(); }
        if (kb + 2 < ksteps) cp_async_wait_prev();
        else cp_async_wait_all();
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
    if (wm * 16 < T) wmma::store_matrix_sync(y + size_t(wm * 16) * rows + n0 + wn * 16, acc, rows, wmma::mem_row_major);
}

void mxfp4_gemm_v21_i4(const uint32_t *weights, const uint16_t *scales, const void *x16, float *y, int rows, int cols, int T, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 1023) || T <= 0 || (rows & 31)) throw std::runtime_error("insignia: bad GEMM dims rows=" + std::to_string(rows) + " cols=" + std::to_string(cols));
    if (T > 64) throw std::runtime_error("insignia: mxfp4_gemm_v21_i4 T=" + std::to_string(T) + " exceeds the 64-row A tile");
    mxfp4_gemm_v21_i4_kernel<<<rows >> 5, 256, 0, stream>>>(weights, scales, (const __nv_bfloat16 *)x16, y, rows, cols, T);
}

// a/b monster-killer: BOTH in_proj_a and in_proj_b for all T tokens in ONE launch
// (replaces the T x 2 per-token GEMV loop: at T=64 that was 3072 launches ~ 9-12 ms
// per 64-token chunk). Two blocks, v21_i4 tile pipeline verbatim; block 0 walks a's
// 32 rows -> ya[T,32], block 1 walks b's 32 rows -> yb[T,32]. A (x16) shared, block 1
// hits L2. Both tensors are 9B in_proj_a/b with EXACTLY 32 rows (baked in).
__global__ __launch_bounds__(256) void mxfp4_gemm_ab_i4_kernel(const uint32_t *__restrict__ wa, const uint16_t *__restrict__ sa, const uint32_t *__restrict__ wb, const uint16_t *__restrict__ sb, const __nv_bfloat16 *__restrict__ x16, float *__restrict__ ya, float *__restrict__ yb, int cols, int T) {
    constexpr int KT = 64, NT = 32;
    constexpr int APAD = 8, BPAD = 8;
    __shared__ __nv_bfloat16 As[2][64][KT + APAD];
    __shared__ __nv_bfloat16 Bs[2][NT][KT + BPAD];
    __shared__ uint32_t Braw[2][NT][8];
    __shared__ uint32_t lut[256];
    {
        const unsigned tbl[16] = {0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
                                  0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0};
        lut[threadIdx.x] = tbl[threadIdx.x & 15] | (tbl[(threadIdx.x >> 4) & 15] << 16);
    }
    const int groups = cols >> 5;
    const uint32_t *weights = blockIdx.x ? wb : wa;   // uniform per block: no divergence
    const uint16_t *scales = blockIdx.x ? sb : sa;
    float *y = blockIdx.x ? yb : ya;                  // [T,32] outputs, row stride 32
    const int tid = threadIdx.x;

    auto prefetch = [&](int kb, int buf) {
        const int k = kb * KT;
        {
            const char *ag = reinterpret_cast<const char *>(x16);
            for (int i = tid; i < 64 * (KT / 8); i += 256) {
                const int m = i / (KT / 8), c8 = i % (KT / 8);
                const size_t srcoff = (size_t(m) * cols + k) * 2 + c8 * 16;
                cp_async16(&As[buf][m][c8 * 8], ag + srcoff);
            }
        }
        {
            for (int i = tid; i < NT * 2; i += 256) {
                const int n = i >> 1, half = i & 1;
                cp_async16(&Braw[buf][n][half * 4], reinterpret_cast<const char *>(weights + size_t(n) * groups * 4 + k / 8 + half * 4));
            }
        }
    };
    auto dequant = [&](int kb, int buf) {
        const int n = tid >> 3, w = tid & 7;
        const float sc = __half2float(__ldg(reinterpret_cast<const __half *>(scales) + size_t(n) * (groups >> 1) + kb));
        const uint32_t word = Braw[buf][n][w];
        __nv_bfloat16 out[8];
        #pragma unroll
        for (int byt = 0; byt < 4; byt++) {
            const uint32_t pair = lut[(word >> (byt * 8)) & 0xff];
            const float lo = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&pair));
            const float hi = __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(&pair)[1]);
            out[byt * 2 + 0] = __float2bfloat16(lo * sc);
            out[byt * 2 + 1] = __float2bfloat16(hi * sc);
        }
        *reinterpret_cast<uint4 *>(&Bs[buf][n][w * 8]) = *reinterpret_cast<const uint4 *>(out);
    };

    const int ksteps = cols / KT;
    prefetch(0, 0);
    cp_async_commit();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    const int warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;

    for (int kb = 0; kb < ksteps; kb++) {
        const int buf = kb & 1;
        if (kb + 1 < ksteps) { prefetch(kb + 1, buf ^ 1); cp_async_commit(); }
        if (kb + 2 < ksteps) cp_async_wait_prev();
        else cp_async_wait_all();
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
    if (wm * 16 < T) wmma::store_matrix_sync(y + size_t(wm * 16) * 32 + wn * 16, acc, 32, wmma::mem_row_major);
}

void mxfp4_gemm_ab_i4(const uint32_t *wa, const uint16_t *sa, const uint32_t *wb, const uint16_t *sb, const void *x16, float *ya, float *yb, int T, int cols, cudaStream_t stream) {
    if (cols <= 0 || (cols & 1023)) throw std::runtime_error("insignia: bad ab GEMM cols=" + std::to_string(cols));
    if (T <= 0 || T > 64) throw std::runtime_error("insignia: ab GEMM T=" + std::to_string(T) + " outside 1..64");
    mxfp4_gemm_ab_i4_kernel<<<2, 256, 0, stream>>>(wa, sa, wb, sb, (const __nv_bfloat16 *)x16, ya, yb, cols, T);
}
}
