#include "insignia_bf16.cuh"
#include <cuda_bf16.h>
#include <mma.h>
#include <stdexcept>
#include <string>
namespace insignia {
using namespace nvcuda;

// bf16 weights viewed as u32 pairs (lo = element 2u, hi = element 2u+1). bf16->f32 is
// EXACT bit surgery: lo<<16 / hi&0xffff0000 — no cvt instructions (AGENTS: bit-manipulate).
__device__ __forceinline__ void cp_async16(void *smem, const char *global) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((unsigned)__cvta_generic_to_shared(smem)), "l"(global));
}

// ---- decode GEMV: one warp per row, x staged in smem, 16B weight loads -----------------
template <bool PAIR>
__global__ __launch_bounds__(256) void bf16_gemv_v2_kernel(const uint32_t *__restrict__ w, const float *__restrict__ x, float *__restrict__ y, int rows, int cols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ float sx[];
    const int xfloats = PAIR ? 2 * cols : cols;
    for (int c0 = threadIdx.x * 16; c0 < xfloats; c0 += blockDim.x * 16) {
        const float4 *xr = reinterpret_cast<const float4 *>(x + c0);
        float4 *sr = reinterpret_cast<float4 *>(sx + c0);
        #pragma unroll
        for (int q = 0; q < 4; q++) sr[q] = __ldg(xr + q);
    }
    __syncthreads();
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_w = w + static_cast<size_t>(row) * (cols >> 1);
    const int nu = cols >> 1;
    float acc0 = 0.f, acc1 = 0.f;
    #pragma unroll 2
    for (int u0 = lane * 4; u0 < nu; u0 += 128) {
        const uint4 p = __ldcs(reinterpret_cast<const uint4 *>(row_w + u0));
        const float4 *x0 = reinterpret_cast<const float4 *>(sx + u0 * 2);
        float part0 = 0.f;
        part0 = fmaf(__uint_as_float(p.x << 16), x0[0].x, part0);
        part0 = fmaf(__uint_as_float(p.x & 0xffff0000u), x0[0].y, part0);
        part0 = fmaf(__uint_as_float(p.y << 16), x0[0].z, part0);
        part0 = fmaf(__uint_as_float(p.y & 0xffff0000u), x0[0].w, part0);
        part0 = fmaf(__uint_as_float(p.z << 16), x0[1].x, part0);
        part0 = fmaf(__uint_as_float(p.z & 0xffff0000u), x0[1].y, part0);
        part0 = fmaf(__uint_as_float(p.w << 16), x0[1].z, part0);
        part0 = fmaf(__uint_as_float(p.w & 0xffff0000u), x0[1].w, part0);
        acc0 += part0;
        if (PAIR) {
            const float4 *x1 = reinterpret_cast<const float4 *>(sx + cols + u0 * 2);
            float part1 = 0.f;
            part1 = fmaf(__uint_as_float(p.x << 16), x1[0].x, part1);
            part1 = fmaf(__uint_as_float(p.x & 0xffff0000u), x1[0].y, part1);
            part1 = fmaf(__uint_as_float(p.y << 16), x1[0].z, part1);
            part1 = fmaf(__uint_as_float(p.y & 0xffff0000u), x1[0].w, part1);
            part1 = fmaf(__uint_as_float(p.z << 16), x1[1].x, part1);
            part1 = fmaf(__uint_as_float(p.z & 0xffff0000u), x1[1].y, part1);
            part1 = fmaf(__uint_as_float(p.w << 16), x1[1].z, part1);
            part1 = fmaf(__uint_as_float(p.w & 0xffff0000u), x1[1].w, part1);
            acc1 += part1;
        }
    }
    #pragma unroll
    for (int m = 16; m; m >>= 1) { acc0 += __shfl_xor_sync(0xffffffff, acc0, m); if (PAIR) acc1 += __shfl_xor_sync(0xffffffff, acc1, m); }
    if (!lane) { y[row] = acc0; if (PAIR) y[rows + row] = acc1; }
}
void bf16_gemv_v2(const uint32_t *w, const float *x, float *y, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 15)) throw std::runtime_error("insignia: bad bf16 GEMV dims rows=" + std::to_string(rows) + " cols=" + std::to_string(cols));
    if (reinterpret_cast<uintptr_t>(w) & 15) throw std::runtime_error("insignia: bf16 weights not 16B aligned (rebase missing)");
    const size_t shared = size_t(cols) * 4;
    if (shared > 48 * 1024) {
        const cudaError_t opt_in = cudaFuncSetAttribute(
            bf16_gemv_v2_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared));
        if (opt_in != cudaSuccess) throw std::runtime_error(std::string("insignia: bf16 GEMV shared-memory opt-in: ") + cudaGetErrorString(opt_in));
    }
    bf16_gemv_v2_kernel<false><<<(rows + 7) >> 3, 256, shared, stream>>>(w, x, y, rows, cols);
    const cudaError_t e27 = cudaGetLastError(); if (e27 != cudaSuccess) throw std::runtime_error(std::string("insignia: bf16_gemv_v2 launch: ") + cudaGetErrorString(e27));
}
void bf16_gemv2_v2(const uint32_t *w, const float *x, float *y, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 15)) throw std::runtime_error("insignia: bad bf16 GEMV dims rows=" + std::to_string(rows) + " cols=" + std::to_string(cols));
    if (reinterpret_cast<uintptr_t>(w) & 15) throw std::runtime_error("insignia: bf16 weights not 16B aligned (rebase missing)");
    const size_t smem = size_t(2) * cols * 4;
    if (smem > 48 * 1024) throw std::runtime_error("insignia: bf16_gemv2_v2 smem exceeds 48KB at cols=" + std::to_string(cols));
    bf16_gemv_v2_kernel<true><<<(rows + 7) >> 3, 256, smem, stream>>>(w, x, y, rows, cols);
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("insignia: bf16_gemv2_v2 launch failed");
}

// ---- pipelined GEMM: Y[T,rows] = X16[T,cols]·W[rows,cols]^T, all bf16 ------------------
// mxfp4_gemm_v21 geometry verbatim (64x32 tile, KT=64, cp.async double buffer, wmma f32
// acc) minus the nibble dequant: B stages straight into wmma fragments. cols%64==0.
__global__ __launch_bounds__(256) void bf16_gemm_kernel(const __nv_bfloat16 *__restrict__ w, const __nv_bfloat16 *__restrict__ x16, float *__restrict__ y, int rows, int cols, int T) {
    constexpr int KT = 64, NT = 32;
    constexpr int APAD = 8, BPAD = 8;
    __shared__ __nv_bfloat16 As[2][64][KT + APAD];
    __shared__ __nv_bfloat16 Bs[2][NT][KT + BPAD];
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
            const char *bg = reinterpret_cast<const char *>(w);
            for (int i = tid; i < NT * (KT / 8); i += 256) {
                const int n = i / (KT / 8), c8 = i % (KT / 8);
                cp_async16(&Bs[buf][n][c8 * 8], bg + (size_t(n0 + n) * cols + k) * 2 + c8 * 16);
            }
        }
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
void bf16_gemm(const uint32_t *w, const void *x16, float *y, int rows, int cols, int T, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 63) || T <= 0 || (rows & 31)) throw std::runtime_error("insignia: bad bf16 GEMM dims rows=" + std::to_string(rows) + " cols=" + std::to_string(cols));
    if (T > 64) throw std::runtime_error("insignia: bf16_gemm T=" + std::to_string(T) + " exceeds the 64-row A tile");
    if (reinterpret_cast<uintptr_t>(w) & 15) throw std::runtime_error("insignia: bf16 weights not 16B aligned (rebase missing)");
    bf16_gemm_kernel<<<rows >> 5, 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16 *>(w), static_cast<const __nv_bfloat16 *>(x16), y, rows, cols, T);
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("insignia: bf16_gemm launch failed");
}

// ---- embed row gather: T rows from a bf16 table (device or UVA host-pinned) ------------
__global__ void embed_gather_bf16_kernel(const uint16_t *__restrict__ w, const int *__restrict__ tokens, float *__restrict__ out, int cols) {
    const int t = blockIdx.x;
    const uint32_t *row = reinterpret_cast<const uint32_t *>(w) + size_t(__ldg(tokens + t)) * (cols >> 1);
    float *o = out + size_t(t) * cols;
    for (int u0 = threadIdx.x * 4; u0 < (cols >> 1); u0 += blockDim.x * 4) {
        const uint4 p = *reinterpret_cast<const uint4 *>(row + u0);
        float4 *of = reinterpret_cast<float4 *>(o + u0 * 2);
        of[0].x = __uint_as_float(p.x << 16); of[0].y = __uint_as_float(p.x & 0xffff0000u);
        of[0].z = __uint_as_float(p.y << 16); of[0].w = __uint_as_float(p.y & 0xffff0000u);
        of[1].x = __uint_as_float(p.z << 16); of[1].y = __uint_as_float(p.z & 0xffff0000u);
        of[1].z = __uint_as_float(p.w << 16); of[1].w = __uint_as_float(p.w & 0xffff0000u);
    }
}
void embed_gather_bf16(const uint16_t *w, const int *tokens_dev, float *out, int cols, int T, cudaStream_t stream) {
    embed_gather_bf16_kernel<<<T, 256, 0, stream>>>(w, tokens_dev, out, cols);
}

// ---- a/b pair: two [heads,cols] bf16 matrices, both decode tokens, ONE launch ----------
__global__ __launch_bounds__(256) void bf16_gemv_ab2_kernel(const uint32_t *__restrict__ wa, const uint32_t *__restrict__ wb, const float *__restrict__ x0, const float *__restrict__ x1, float *__restrict__ a0, float *__restrict__ a1, float *__restrict__ b0, float *__restrict__ b1, int cols, int heads) {
    const int is_b = blockIdx.x >= heads ? 1 : 0, h = blockIdx.x - is_b * heads;   // grid = 2*heads
    const uint32_t *row = (is_b ? wb : wa) + static_cast<size_t>(h) * (cols >> 1);
    const int nu = cols >> 1;
    float s0 = 0.f, s1 = 0.f;
    for (int u = threadIdx.x; u < nu; u += blockDim.x) {
        const uint32_t p = __ldg(row + u);
        const float w_lo = __uint_as_float(p << 16), w_hi = __uint_as_float(p & 0xffff0000u);
        const float v0lo = __ldg(x0 + u * 2), v0hi = __ldg(x0 + u * 2 + 1);
        const float v1lo = __ldg(x1 + u * 2), v1hi = __ldg(x1 + u * 2 + 1);
        s0 = fmaf(w_lo, v0lo, s0); s0 = fmaf(w_hi, v0hi, s0);
        s1 = fmaf(w_lo, v1lo, s1); s1 = fmaf(w_hi, v1hi, s1);
    }
    #pragma unroll
    for (int m = 16; m; m >>= 1) { s0 += __shfl_xor_sync(0xffffffff, s0, m); s1 += __shfl_xor_sync(0xffffffff, s1, m); }
    __shared__ float r[8], r2[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (!lane) { r[warp] = s0; r2[warp] = s1; }
    __syncthreads();
    if (!warp) {
        s0 = lane < 8 ? r[lane] : 0.f; s1 = lane < 8 ? r2[lane] : 0.f;
        #pragma unroll
        for (int m = 16; m; m >>= 1) { s0 += __shfl_xor_sync(0xffffffff, s0, m); s1 += __shfl_xor_sync(0xffffffff, s1, m); }
        if (!lane) { (is_b ? b0 : a0)[h] = s0; (is_b ? b1 : a1)[h] = s1; }
    }
}
void bf16_gemv_ab2_pair(const uint32_t *wa, const uint32_t *wb, const float *x0, const float *x1, float *a0, float *a1, float *b0, float *b1, int cols, int heads, cudaStream_t stream) {
    bf16_gemv_ab2_kernel<<<2 * heads, 256, 0, stream>>>(wa, wb, x0, x1, a0, a1, b0, b1, cols, heads);
}

// ---- a/b rows for prefill: both [heads,cols] matrices against T bf16 token rows ----
// grid = 2*heads blocks; each block owns one weight row and dots it against every x16 row.
__global__ __launch_bounds__(256) void bf16_gemv_ab_rows_kernel(const uint32_t *__restrict__ wa, const uint32_t *__restrict__ wb, const __nv_bfloat16 *__restrict__ x16, float *__restrict__ a, float *__restrict__ b, int cols, int heads, int T) {
    const int is_b = blockIdx.x >= heads ? 1 : 0, h = blockIdx.x - is_b * heads;
    const uint32_t *row = (is_b ? wb : wa) + static_cast<size_t>(h) * (cols >> 1);
    const int nu = cols >> 1;
    float *out = (is_b ? b : a);
    for (int t = 0; t < T; t++) {
        const uint32_t *xr = reinterpret_cast<const uint32_t *>(x16 + static_cast<size_t>(t) * cols);
        float s = 0;
        for (int u = threadIdx.x; u < nu; u += blockDim.x) {
            const uint32_t p = __ldg(row + u);
            const uint32_t q = __ldg(xr + u);
            s = fmaf(__uint_as_float(p << 16), __uint_as_float(q << 16), s);          // lo * lo
            s = fmaf(__uint_as_float(p & 0xffff0000u), __uint_as_float(q & 0xffff0000u), s);  // hi * hi
        }
        #pragma unroll
        for (int m = 16; m; m >>= 1) s += __shfl_xor_sync(0xffffffff, s, m);
        __shared__ float r[8];
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        if (!lane) r[warp] = s;
        __syncthreads();
        if (!warp) {
            s = lane < 8 ? r[lane] : 0.f;
            #pragma unroll
            for (int m = 16; m; m >>= 1) s += __shfl_xor_sync(0xffffffff, s, m);
            if (!lane) out[t * heads + h] = s;
        }
        __syncthreads();
    }
}
void bf16_gemv_ab_rows(const uint32_t *wa, const uint32_t *wb, const void *x16, float *a, float *b, int cols, int heads, int T, cudaStream_t stream) {
    bf16_gemv_ab_rows_kernel<<<2 * heads, 256, 0, stream>>>(wa, wb, static_cast<const __nv_bfloat16 *>(x16), a, b, cols, heads, T);
}
}
