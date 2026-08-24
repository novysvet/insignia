#include "insignia_layout.cuh"
#include <cuda_bf16.h>
#include <mma.h>
namespace insignia {
using namespace nvcuda;

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
    if (rows <= 0 || cols <= 0 || (cols & 31) || T <= 0 || (rows & 63)) return;
    const int blocks = rows >> 6;
    mxfp4_gemm_mlx_kernel<<<blocks, 256, 0, stream>>>(weights, scales, x, y, rows, cols, T);
}
}
