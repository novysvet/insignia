// Qwen3.8-27B-FP8 tiered engine: 27B-specialized kernels + TieredStorage27 + Qwen38Decode.
// The 9B files (decode.cu/attention.cu/deltanet.cu/prefill.cu) stay frozen regression
// assets; everything here is 27B-hardcoded per AGENTS.md (bake every assumption).
#include "insignia_27b.hpp"
#include "insignia_fp8.cuh"
#include "insignia_bf16.cuh"
#include "insignia_ops.cuh"
#include "insignia_qwen_kernels.cuh"
#include "insignia_prefill.cuh"
#include "insignia_layout.cuh"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace insignia {
bool g_dump_stage27 = false;   // test hook: dump layer-0 prefill intermediates
static void dump_stage(const float *dev, int n, const char *name) {
    if (!g_dump_stage27) return;
    std::vector<float> h(n);
    cudaMemcpy(h.data(), dev, size_t(n) * 4, cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(nullptr);
    char path[128];
    snprintf(path, sizeof path, "build/stage-%s.f32", name);
    FILE *f = fopen(path, "wb");
    fwrite(h.data(), 4, h.size(), f);
    fclose(f);
    printf("stage %s (%d) [newline]", name, n);
}
static void launch_check(const char *where) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("27B launch ") + where + ": " + cudaGetErrorString(e));
}

// ===========================================================================
// 27B kernels. Deltas vs the 9B twins: grids 24/28/48, kvh=head/6, kh=head/3,
// widths 5120/6144/10240/17408, norm weights f32 with +1 BAKED AT LOAD
// (input/post/q/k/final — HF zero-centered); linear_attn.norm stays raw bf16.
// ===========================================================================

__global__ void rmsnorm_f32w_kernel(const float *__restrict__ x, const float *__restrict__ w, float *__restrict__ y, int cols) {
    const int row = blockIdx.x;
    float ss = 0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) { const float v = x[row * cols + i]; ss = fmaf(v, v, ss); }
    for (int m = 16; m; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
    __shared__ float p[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (!lane) p[warp] = ss;
    __syncthreads();
    if (!warp) { ss = lane < 8 ? p[lane] : 0; for (int m = 16; m; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m); if (!lane) p[0] = rsqrtf(ss / cols + 1e-6f); }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) y[row * cols + i] = x[row * cols + i] * p[0] * w[i];
}
static void rmsnorm_f32w(const float *x, const float *w, float *y, int rows, int cols, cudaStream_t s) {
    rmsnorm_f32w_kernel<<<rows, 256, 0, s>>>(x, w, y, cols); launch_check("rmsnorm_f32w");
}

// pinned host bf16 row -> device f32 (UVA read; embed stays on NVMe, row pread earlier)
__global__ void bf16_row_load_kernel(const uint32_t *__restrict__ src, float *__restrict__ dst, int cols) {
    const int nu = cols >> 1;
    for (int u = threadIdx.x * 4; u < nu; u += blockDim.x * 4) {
        const uint4 p = *reinterpret_cast<const uint4 *>(src + u);
        float4 *o = reinterpret_cast<float4 *>(dst + u * 2);
        o[0].x = __uint_as_float(p.x << 16); o[0].y = __uint_as_float(p.x & 0xffff0000u);
        o[0].z = __uint_as_float(p.y << 16); o[0].w = __uint_as_float(p.y & 0xffff0000u);
        o[1].x = __uint_as_float(p.z << 16); o[1].y = __uint_as_float(p.z & 0xffff0000u);
        o[1].z = __uint_as_float(p.w << 16); o[1].w = __uint_as_float(p.w & 0xffff0000u);
    }
}
__global__ void bf16_rows_load_kernel(const uint32_t *__restrict__ src, float *__restrict__ dst, int cols) {
    const int t = blockIdx.x, nu = cols >> 1;
    const uint32_t *row = src + size_t(t) * nu;
    for (int u = threadIdx.x * 4; u < nu; u += blockDim.x * 4) {
        const uint4 p = *reinterpret_cast<const uint4 *>(row + u);
        float4 *o = reinterpret_cast<float4 *>(dst + size_t(t) * cols + u * 2);
        o[0].x = __uint_as_float(p.x << 16); o[0].y = __uint_as_float(p.x & 0xffff0000u);
        o[0].z = __uint_as_float(p.y << 16); o[0].w = __uint_as_float(p.y & 0xffff0000u);
        o[1].x = __uint_as_float(p.z << 16); o[1].y = __uint_as_float(p.z & 0xffff0000u);
        o[1].z = __uint_as_float(p.w << 16); o[1].w = __uint_as_float(p.w & 0xffff0000u);
    }
}

// QK norm (f32 +1-baked) + partial RoPE: pairs (i, i+32) over dims 0..63, theta 1e7.
__global__ void qk_norm_rope27_kernel(float *__restrict__ q, float *__restrict__ k, const float *__restrict__ qw, const float *__restrict__ kw, const int *__restrict__ pos_dev) {
    const int head = blockIdx.x, tid = threadIdx.x;
    const bool isq = head < 24;
    float *p = isq ? q + head * 256 : k + (head - 24) * 256;
    const float *w = isq ? qw : kw;
    float v = p[tid], ss = v * v;
    for (int m = 16; m; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
    __shared__ float mem[64];
    __shared__ float nsc;
    const int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) mem[warp] = ss;
    __syncthreads();
    if (warp == 0) { ss = lane < 8 ? mem[lane] : 0; for (int m = 16; m; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m); if (lane == 0) nsc = rsqrtf(ss / 256 + 1e-6f); }
    __syncthreads();
    v *= nsc * w[tid];
    const int pos = __ldg(pos_dev);
    if (pos != 0 && tid < 64) mem[tid] = v;
    __syncthreads();
    if (pos != 0 && tid < 64) {
        const int half = tid & 31;
        const float inv = __powf(10000000.0f, -float(2 * half) / 64.0f), a = float(pos) * inv, c = __cosf(a), s = __sinf(a);
        const float other = mem[tid < 32 ? tid + 32 : tid - 32];
        v = v * c + (tid < 32 ? -other : other) * s;
    }
    p[tid] = v;
}
static void qk_norm_rope27(float *q, float *k, const float *qw, const float *kw, const int *pos_dev, cudaStream_t s) {
    qk_norm_rope27_kernel<<<28, 256, 0, s>>>(q, k, qw, kw, pos_dev);
}

__global__ __launch_bounds__(256, 2) void gqa_decode27_kernel(const float *__restrict__ q, const float *__restrict__ kc, const float *__restrict__ vc, float *__restrict__ out, const int *__restrict__ pos_dev, int base) {
    const int head = blockIdx.x, tid = threadIdx.x, kvh = head / 6;
    const int tokens = __ldg(pos_dev) + base + 1;
    __shared__ float score[4096];
    const float scale = .0625f;
    float mx = -3.402823466e+38F;
    for (int t = tid; t < tokens; t += 256) { float z = 0; for (int d = 0; d < 256; d++) z = fmaf(q[head * 256 + d], kc[(size_t(t) * 4 + kvh) * 256 + d], z); z *= scale; score[t] = z; mx = fmaxf(mx, z); }
    for (int m = 16; m; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, m));
    __shared__ float red[8], smx, sden;
    const int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    if (warp == 0) { mx = lane < 8 ? red[lane] : -3.402823466e+38F; for (int m = 16; m; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, m)); if (lane == 0) smx = mx; }
    __syncthreads();
    mx = smx;
    float den = 0;
    for (int t = tid; t < tokens; t += 256) { const float e = __expf(score[t] - mx); score[t] = e; den += e; }
    for (int m = 16; m; m >>= 1) den += __shfl_xor_sync(0xffffffff, den, m);
    if (lane == 0) red[warp] = den;
    __syncthreads();
    if (warp == 0) { den = lane < 8 ? red[lane] : 0; for (int m = 16; m; m >>= 1) den += __shfl_xor_sync(0xffffffff, den, m); if (lane == 0) sden = 1.f / den; }
    __syncthreads();
    float z = 0;
    for (int t = 0; t < tokens; t++) z = fmaf(score[t] * sden, vc[(size_t(t) * 4 + kvh) * 256 + tid], z);
    out[head * 256 + tid] = z;
}
static void gqa_decode27(const float *q, const float *k, const float *v, float *out, const int *pos_dev, int base, cudaStream_t s) {
    gqa_decode27_kernel<<<24, 256, 0, s>>>(q, k, v, out, pos_dev, base); launch_check("gqa_decode27");
}

__global__ void split_q_gate27_kernel(const float *__restrict__ src, float *__restrict__ q, float *__restrict__ g) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    const int h = i >> 8, d = i & 255;
    q[i] = src[h * 512 + d];
    g[i] = src[h * 512 + 256 + d];
}
static void split_q_gate27(const float *s, float *q, float *g, cudaStream_t stream) {
    split_q_gate27_kernel<<<24, 256, 0, stream>>>(s, q, g);
}
__global__ void expand_gate27_kernel(const float *__restrict__ g, float *__restrict__ out) { const int i = blockIdx.x * 256 + threadIdx.x; if (i < 6144) out[i] = g[i]; }
static void expand_gate_heads27(const float *g, float *out, cudaStream_t s) { expand_gate27_kernel<<<24, 256, 0, s>>>(g, out); }

__global__ __launch_bounds__(128, 4) void deltanet_decode27_kernel(float *__restrict__ state, const float *__restrict__ q16, const float *__restrict__ k16, const float *__restrict__ v, const float *__restrict__ g, const float *__restrict__ beta, float *__restrict__ out) {
    const int head = blockIdx.x, tid = threadIdx.x, kh = head / 3;
    float q = q16[kh * 128 + tid], k = k16[kh * 128 + tid];
    float qn = q * q, kn = k * k;
    for (int m = 16; m; m >>= 1) { qn += __shfl_xor_sync(0xffffffff, qn, m); kn += __shfl_xor_sync(0xffffffff, kn, m); }
    __shared__ float sq[4], sk[4], delta[128];
    const int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) { sq[warp] = qn; sk[warp] = kn; }
    __syncthreads();
    if (tid == 0) { const float a = sq[0] + sq[1] + sq[2] + sq[3], b = sk[0] + sk[1] + sk[2] + sk[3]; sq[0] = rsqrtf(a + 1e-6f) * 0.08838834764831845f; sk[0] = rsqrtf(b + 1e-6f); }
    __syncthreads();
    q *= sq[0]; k *= sk[0];
    float *S = state + head * 128 * 128;
    const float decay = expf(g[head]);
    float dot = 0;
    for (int i = 0; i < 128; i++) dot = fmaf(S[i * 128 + tid] * decay, k16[kh * 128 + i] * sk[0], dot);
    delta[tid] = (v[head * 128 + tid] - dot) * beta[head];
    __syncthreads();
    float acc = 0;
    for (int i = 0; i < 128; i++) { float &cell = S[i * 128 + tid]; cell = fmaf(cell, decay, k16[kh * 128 + i] * sk[0] * delta[tid]); acc = fmaf(cell, q16[kh * 128 + i] * sq[0], acc); }
    out[head * 128 + tid] = acc;
}
static void deltanet_decode27(float *s, const float *q, const float *k, const float *v, const float *g, const float *b, float *o, cudaStream_t stream) {
    deltanet_decode27_kernel<<<48, 128, 0, stream>>>(s, q, k, v, g, b, o); launch_check("deltanet_decode27");
}

// ---- prefill kernels (T<=64) ------------------------------------------------
__global__ void split_q_gate_batch27_kernel(const float *__restrict__ src, float *__restrict__ q, float *__restrict__ gate) {
    const int t = blockIdx.x / 24, h = blockIdx.x % 24, d = threadIdx.x;
    const size_t base = static_cast<size_t>(t) * 12288 + h * 512;
    q[static_cast<size_t>(t) * 6144 + h * 256 + d] = src[base + d];
    gate[static_cast<size_t>(t) * 6144 + h * 256 + d] = src[base + 256 + d];
}
static void split_q_gate_batch27(const float *src, float *q, float *gate, int T, cudaStream_t stream) {
    split_q_gate_batch27_kernel<<<T * 24, 256, 0, stream>>>(src, q, gate);
}

__global__ void qk_norm_rope_batch27_kernel(float *__restrict__ q, float *__restrict__ k, const float *__restrict__ qw, const float *__restrict__ kw, const int *__restrict__ pos_dev) {
    const int t = blockIdx.y, head = blockIdx.x, tid = threadIdx.x;
    const bool isq = head < 24;
    float *p = isq ? q + (static_cast<size_t>(t) * 24 + head) * 256 : k + (static_cast<size_t>(t) * 4 + head - 24) * 256;
    const float *w = isq ? qw : kw;
    float v = p[tid], ss = v * v;
    for (int m = 16; m; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
    __shared__ float mem[64];
    __shared__ float nsc;
    const int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) mem[warp] = ss;
    __syncthreads();
    if (warp == 0) { ss = lane < 8 ? mem[lane] : 0; for (int m = 16; m; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m); if (lane == 0) nsc = rsqrtf(ss / 256 + 1e-6f); }
    __syncthreads();
    v *= nsc * w[tid];
    const int pos = __ldg(pos_dev) + t;
    if (pos != 0 && tid < 64) mem[tid] = v;
    __syncthreads();
    if (pos != 0 && tid < 64) {
        const int half = tid & 31;
        const float inv = __powf(10000000.0f, -float(2 * half) / 64.0f), a = float(pos) * inv, c = __cosf(a), s = __sinf(a);
        const float other = mem[tid < 32 ? tid + 32 : tid - 32];
        v = v * c + (tid < 32 ? -other : other) * s;
    }
    p[tid] = v;
}
static void qk_norm_rope_batch27(float *q, float *k, const float *qw, const float *kw, const int *pos_dev, int T, cudaStream_t stream) {
    qk_norm_rope_batch27_kernel<<<dim3(28, T), 256, 0, stream>>>(q, k, qw, kw, pos_dev);
}

__global__ __launch_bounds__(256, 2) void gqa_prefill27_kernel(const float *__restrict__ q, const float *__restrict__ kc, const float *__restrict__ vc, float *__restrict__ out, const int *__restrict__ pos_dev) {
    const int head = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, kvh = head / 6;
    const int tokens = __ldg(pos_dev) + t + 1;
    __shared__ float qs[256];
    __shared__ float score[4096];
    __shared__ float red[8], smx, sden;
    __shared__ float part[8][256];
    const float *qrow = q + (static_cast<size_t>(t) * 24 + head) * 256;
    qs[tid] = qrow[tid];
    __syncthreads();
    const int lane = tid & 31, warp = tid >> 5;
    float mx = -3.402823466e+38F;
    for (int j = warp; j < tokens; j += 8) {
        const float *krow = kc + (size_t(j) * 4 + kvh) * 256 + lane * 8;
        float z = 0;
        #pragma unroll
        for (int d = 0; d < 8; d++) z = fmaf(qs[lane * 8 + d], __ldg(krow + d), z);
        for (int m = 16; m; m >>= 1) z += __shfl_xor_sync(0xffffffff, z, m);
        if (lane == 0) { z *= .0625f; score[j] = z; mx = fmaxf(mx, z); }
    }
    for (int m = 16; m; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, m));
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    if (warp == 0) { mx = lane < 8 ? red[lane] : -3.402823466e+38F; for (int m = 16; m; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, m)); if (lane == 0) smx = mx; }
    __syncthreads();
    mx = smx;
    float den = 0;
    for (int j = warp; j < tokens; j += 8) { const float e = __expf(score[j] - mx); if (lane == 0) { score[j] = e; den += e; } }
    for (int m = 16; m; m >>= 1) den += __shfl_xor_sync(0xffffffff, den, m);
    if (lane == 0) red[warp] = den;
    __syncthreads();
    if (warp == 0) { den = lane < 8 ? red[lane] : 0; for (int m = 16; m; m >>= 1) den += __shfl_xor_sync(0xffffffff, den, m); if (lane == 0) sden = 1.f / den; }
    __syncthreads();
    const float inv_den = sden;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int j = warp; j < tokens; j += 8) {
        const float p = score[j] * inv_den;
        const float *vrow = vc + (size_t(j) * 4 + kvh) * 256 + lane * 8;
        #pragma unroll
        for (int d = 0; d < 8; d++) acc[d] = fmaf(p, __ldg(vrow + d), acc[d]);
    }
    #pragma unroll
    for (int d = 0; d < 8; d++) part[warp][lane * 8 + d] = acc[d];
    __syncthreads();
    float o = 0;
    #pragma unroll
    for (int w2 = 0; w2 < 8; w2++) o += part[w2][tid];
    out[(static_cast<size_t>(t) * 24 + head) * 256 + tid] = o;
}
static void gqa_prefill27(const float *q, const float *kc, const float *vc, float *out, const int *pos_dev, int T, int max_context, cudaStream_t stream) {
    (void)max_context;
    gqa_prefill27_kernel<<<dim3(24, T), 256, 0, stream>>>(q, kc, vc, out, pos_dev);
}

__global__ void conv_prefill27_kernel(const float *__restrict__ x, float *__restrict__ out, const float *__restrict__ state, const uint16_t *__restrict__ w) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = idx / 10240, c = idx % 10240;
    float z = 0;
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        const float hist = t >= 3 - i ? x[static_cast<size_t>(t - (3 - i)) * 10240 + c] : state[c * 3 + i];
        z = fmaf(hist, __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(w + c * 4 + i)), z);
    }
    z = fmaf(x[static_cast<size_t>(t) * 10240 + c], __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(w + c * 4 + 3)), z);
    out[idx] = z / (1.f + __expf(-z));
}
__global__ void conv_roll_state27_kernel(const float *__restrict__ x, float *__restrict__ state, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 10240) return;
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        const int j = T - 3 + i;
        state[c * 3 + i] = j >= 0 ? x[static_cast<size_t>(j) * 10240 + c] : state[c * 3 + 3 + j];
    }
}
static void conv_prefill_silu27(float *x, float *scratch, float *state, const uint16_t *w, int T, cudaStream_t stream) {
    const int n = T * 10240;
    conv_prefill27_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, scratch, state, w);
    conv_roll_state27_kernel<<<(10240 + 255) / 256, 256, 0, stream>>>(x, state, T);
}

__global__ __launch_bounds__(128) void deltanet_prefill27_kernel(float *__restrict__ state, const float *__restrict__ qkv, const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ out, int T) {
    extern __shared__ float sh[];
    const int head = blockIdx.x, tid = threadIdx.x, kh = head / 3;
    float *S = sh;
    float *gstate = state + static_cast<size_t>(head) * 128 * 128;
    for (int i = tid; i < 128 * 128; i += 128) S[i] = gstate[i];
    __syncthreads();
    for (int t = 0; t < T; t++) {
        const float *qt = qkv + static_cast<size_t>(t) * 10240 + kh * 128;
        const float *kt = qkv + static_cast<size_t>(t) * 10240 + 2048 + kh * 128;
        const float *vt = qkv + static_cast<size_t>(t) * 10240 + 4096 + head * 128;
        float q = qt[tid], k = kt[tid];
        float qn = q * q, kn = k * k;
        for (int m = 16; m; m >>= 1) { qn += __shfl_xor_sync(0xffffffff, qn, m); kn += __shfl_xor_sync(0xffffffff, kn, m); }
        __shared__ float sq[4], sk[4], delta[128];
        const int lane = tid & 31, warp = tid >> 5;
        if (lane == 0) { sq[warp] = qn; sk[warp] = kn; }
        __syncthreads();
        if (tid == 0) {
            const float qa = sq[0] + sq[1] + sq[2] + sq[3], ka = sk[0] + sk[1] + sk[2] + sk[3];
            sq[0] = rsqrtf(qa + 1e-6f) * 0.08838834764831845f;
            sk[0] = rsqrtf(ka + 1e-6f);
        }
        __syncthreads();
        q *= sq[0]; k *= sk[0];
        const float decay = expf(a[static_cast<size_t>(t) * 48 + head]);
        const float beta = b[static_cast<size_t>(t) * 48 + head];
        float dot = 0;
        for (int i = 0; i < 128; i++) dot = fmaf(S[i * 128 + tid] * decay, kt[i] * sk[0], dot);
        delta[tid] = (vt[tid] - dot) * beta;
        __syncthreads();
        float acc = 0;
        for (int i = 0; i < 128; i++) {
            float &cell = S[i * 128 + tid];
            cell = fmaf(cell, decay, kt[i] * sk[0] * delta[tid]);
            acc = fmaf(cell, qt[i] * sq[0], acc);
        }
        out[(static_cast<size_t>(t) * 48 + head) * 128 + tid] = acc;
        __syncthreads();
    }
    for (int i = tid; i < 128 * 128; i += 128) gstate[i] = S[i];
}
static void deltanet_prefill27(float *state, const float *qkv, const float *a, const float *b, float *out, int T, cudaStream_t stream) {
    // (launch_check below after the launch)
    static const bool ok = [] { return cudaFuncSetAttribute(deltanet_prefill27_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024 + 512) == cudaSuccess; }();
    if (!ok) throw std::runtime_error("insignia: deltanet_prefill27 smem opt-in failed (66048 B)");
    deltanet_prefill27_kernel<<<48, 128, 64 * 1024 + 512, stream>>>(state, qkv, a, b, out, T); launch_check("deltanet_prefill27");
}

__global__ void bumpi27_kernel(int *p) { (*p)++; }

// ===========================================================================
// TieredStorage27
// ===========================================================================
namespace {
void cuda_check(cudaError_t e, const char *what) { if (e != cudaSuccess) throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(e)); }
constexpr int kNF8 = 10;   // view slot order: 0 qkv,1 z,2 out,3 q,4 k,5 v,6 o,7 gate,8 up,9 down
}

TieredStorage27::TieredStorage27(const wchar_t *index_path, const wchar_t *manifest_path, cudaStream_t stream) : model_(index_path), stream_(stream) {
    if (!model_.v2()) throw std::runtime_error("27B storage needs an INSIDX02 index");
    if (model_.shape_hdr()[0] != Q38Shape::hidden || model_.shape_hdr()[1] != Q38Shape::layers)
        throw std::runtime_error("index shape header does not match the 27B engine");
    for (int i = 0; i < Q38Shape::layers; i++) tier_[i] = Tier::N;
    {
        FILE *mf = _wfopen(manifest_path, L"rb");
        if (!mf) throw std::runtime_error("cannot open placement manifest");
        char line[128];
        while (fgets(line, sizeof line, mf)) {
            int lo, hi; char t = 0;
            if (sscanf(line, " %c %d %d", &t, &lo, &hi) != 3) continue;
            const Tier tier = t == 'V' ? Tier::V : t == 'Z' ? Tier::Z : t == 'C' ? Tier::C : Tier::N;
            for (int l = lo; l <= hi && l < Q38Shape::layers; l++) tier_[l] = tier;
        }
        fclose(mf);
    }
    // The 8 shards with F8 bases ==8 mod 16 are layers 0..9 — they MUST be V (ring/UVA
    // uint4 loads would be misaligned). Enforce loudly rather than corrupt.
    for (int l = 0; l < Q38Shape::layers; l++) {
        if (tier_[l] == Tier::Z || tier_[l] == Tier::C) throw std::runtime_error("Z/C tiers arrive with the v1.5/v2 manifests; v1 is V+N only");
        if (tier_[l] != Tier::V && misaligned_shard(l)) throw std::runtime_error("manifest puts a 16B-misaligned shard outside V (layers 0..9 must be V)");
    }
    bounce_bytes_ = 64ull << 20;
    cuda_check(cudaMallocHost(&bounce_, bounce_bytes_), "bounce");
    cuda_check(cudaHostAlloc(&embed_, 64 * Q38Shape::hidden * 2, cudaHostAllocDefault), "embed rows");
    load_smalls_and_scales();
    load_v_layers();
    build_n_plans();
    printf("storage27: V=%d Z=%d N=%d | VRAM=%.0f MB free after load\n",
           count_tier(Tier::V), count_tier(Tier::Z), count_tier(Tier::N), free_vram_mb());
}

TieredStorage27::~TieredStorage27() {
    if (bounce_) cudaFreeHost(bounce_);
    if (embed_) cudaFreeHost(embed_);
    for (void *p : slabs_) cudaFree(p);
    if (n_stage_) cudaFree(n_stage_);
    if (lm_head_) cudaFree(lm_head_);
    if (norm_arena_) cudaFree(norm_arena_);
    if (bf16_arena_) cudaFree(bf16_arena_);
    if (alog_arena_) cudaFree(alog_arena_);
    if (scales_arena_) cudaFree(scales_arena_);
    if (embed_file_) CloseHandle(static_cast<HANDLE>(embed_file_));
}

bool TieredStorage27::misaligned_shard(int l) const {
    const std::string p = "language_model.model.layers." + std::to_string(l);
    for (const char *suffix : {".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z", ".linear_attn.out_proj", ".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj", ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj"}) {
        const TensorView *t = model_.find((p + suffix + ".weight").c_str());
        if (t && (t->off & 15)) return true;
    }
    return false;
}
int TieredStorage27::count_tier(Tier t) const { int n = 0; for (int i = 0; i < Q38Shape::layers; i++) n += tier_[i] == t; return n; }
double TieredStorage27::free_vram_mb() const { size_t f = 0, tot = 0; cudaMemGetInfo(&f, &tot); return double(f) / (1024.0 * 1024.0); }

// device alloc + chunked staged copy (reads through the pinned bounce; alignment by construction)
void *TieredStorage27::dev_tensor_chunked(const TensorView &t) {
    void *d;
    cuda_check(cudaMalloc(&d, size_t(t.bytes)), ("alloc " + t.name).c_str());
    for (uint64_t off = 0; off < t.bytes; ) {
        const size_t chunk = size_t(std::min<uint64_t>(t.bytes - off, bounce_bytes_));
        model_.read_into_of(t, off, chunk, bounce_);
        cuda_check(cudaMemcpyAsync(d, bounce_, chunk, cudaMemcpyHostToDevice, stream_), "H2D tensor chunk");
        off += chunk;
    }
    cudaStreamSynchronize(stream_);
    return d;
}

void TieredStorage27::load_smalls_and_scales() {
    // ---- exact arena sizing from the index ----
    size_t norm_f = size_t(Q38Shape::layers) * 2 * Q38Shape::hidden + 2 * Q38Shape::head_dim * Q38Shape::full_attn_layers + Q38Shape::hidden;  // in/post per layer + q/k norms + final
    size_t ab_b = 0, conv_b = 0, dt_b = 0, lanorm_b = 0, alog_b = 0;
    for (int l = 0; l < Q38Shape::layers; l++) if (!Q38Shape::full_attention(l)) {
        ab_b += 2 * size_t(48) * Q38Shape::hidden * 2;
        conv_b += size_t(10240) * 4 * 2;
        dt_b += 48 * 2;
        lanorm_b += 128 * 2;
        alog_b += 48 * 4;
    }
    size_t scale_b = 0;
    for (const auto &t : model_.tensors()) if (t.name.find(".scales") != std::string::npos) scale_b += size_t(t.bytes);
    cuda_check(cudaMalloc(&norm_arena_, norm_f * 4), "norm arena");
    cuda_check(cudaMalloc(&bf16_arena_, ab_b + conv_b + dt_b + lanorm_b), "bf16 arena");
    cuda_check(cudaMalloc(&alog_arena_, alog_b), "alog arena");
    cuda_check(cudaMalloc(&scales_arena_, scale_b), "scales arena");
    cuda_check(cudaMemset(norm_arena_, 0, norm_f * 4), "memset norms");  // q/k norms absent on linear layers stay 0 (never read)

    // ---- fill: host staging in bounce, transform, H2D ----
    std::vector<float> hnorm(norm_f);
    std::vector<uint16_t> hbf16((ab_b + conv_b + dt_b + lanorm_b) / 2);
    std::vector<float> halog(alog_b / 4);
    std::vector<uint16_t> hscales(scale_b / 2);
    size_t no = 0, bo = 0, ao = 0, so = 0;
    auto norm_load = [&](const char *name, bool bake_plus1) -> float * {
        const TensorView *t = model_.find(name);
        if (!t) throw std::runtime_error(std::string("missing tensor ") + name);
        if (t->dtype != DType::bf16) throw std::runtime_error(std::string("expected bf16: ") + name);
        std::vector<uint16_t> raw(size_t(t->bytes) / 2);
        model_.read_into(*t, raw.data());
        const size_t n = raw.size();
        if (bake_plus1) for (size_t i = 0; i < n; i++) hnorm[no + i] = 1.f + __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&raw[i]));
        else            for (size_t i = 0; i < n; i++) hnorm[no + i] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&raw[i]));
        float *ret = norm_arena_ + no;
        no += n;
        return ret;
    };
    auto bf16_load = [&](const char *name) -> uint16_t * {
        const TensorView *t = model_.find(name);
        if (!t) throw std::runtime_error(std::string("missing tensor ") + name);
        uint16_t *ret = hbf16.data() + bo;
        model_.read_into(*t, ret);
        bo += size_t(t->bytes) / 2;
        return nullptr;   // device pointer resolved after the single bulk upload below
    };
    // per-layer offsets recorded, one bulk H2D per arena at the end
    struct BOff { size_t bf16_off; };
    std::vector<size_t> la_norm_off(Q38Shape::layers), conv_off(Q38Shape::layers), dt_off(Q38Shape::layers), ab_off[2]{std::vector<size_t>(Q38Shape::layers), std::vector<size_t>(Q38Shape::layers)};
    std::vector<size_t> qn_off(Q38Shape::layers), kn_off(Q38Shape::layers);
    for (int l = 0; l < Q38Shape::layers; l++) {
        const std::string p = "language_model.model.layers." + std::to_string(l);
        views_[l].in_norm = norm_load((p + ".input_layernorm.weight").c_str(), true);
        views_[l].post_norm = norm_load((p + ".post_attention_layernorm.weight").c_str(), true);
        if (Q38Shape::full_attention(l)) {
            views_[l].q_norm = norm_load((p + ".self_attn.q_norm.weight").c_str(), true);
            views_[l].k_norm = norm_load((p + ".self_attn.k_norm.weight").c_str(), true);
        } else {
            const std::string a = p + ".linear_attn";
            la_norm_off[l] = bo; bf16_load((a + ".norm.weight").c_str());
            conv_off[l] = bo;    bf16_load((a + ".conv1d.weight").c_str());
            dt_off[l] = bo;      bf16_load((a + ".dt_bias").c_str());
            ab_off[0][l] = bo;   bf16_load((a + ".in_proj_a.weight").c_str());
            ab_off[1][l] = bo;   bf16_load((a + ".in_proj_b.weight").c_str());
            // A_log: bf16 -> f32 widen
            const TensorView *t = model_.find(a + ".A_log");
            if (!t) throw std::runtime_error("missing A_log " + a);
            std::vector<uint16_t> raw(size_t(t->bytes) / 2);
            model_.read_into(*t, raw.data());
            for (size_t i = 0; i < raw.size(); i++) halog[ao + i] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&raw[i]));
            views_[l].a_log = alog_arena_ + ao;
            ao += raw.size();
        }
        views_[l].tier = tier_[l];
    }
    final_norm_ = norm_load("language_model.model.norm.weight", true);
    // scales: raw copy
    for (const auto &t : model_.tensors()) if (t.name.find(".scales") != std::string::npos) {
        model_.read_into(t, hscales.data() + so);
        so += size_t(t.bytes) / 2;
    }
    cuda_check(cudaMemcpyAsync(norm_arena_, hnorm.data(), norm_f * 4, cudaMemcpyHostToDevice, stream_), "norm H2D");
    cuda_check(cudaMemcpyAsync(bf16_arena_, hbf16.data(), hbf16.size() * 2, cudaMemcpyHostToDevice, stream_), "bf16 H2D");
    cuda_check(cudaMemcpyAsync(alog_arena_, halog.data(), halog.size() * 4, cudaMemcpyHostToDevice, stream_), "alog H2D");
    cuda_check(cudaMemcpyAsync(scales_arena_, hscales.data(), hscales.size() * 2, cudaMemcpyHostToDevice, stream_), "scales H2D");
    cudaStreamSynchronize(stream_);
    // resolve bf16 arena pointers now that offsets are known
    for (int l = 0; l < Q38Shape::layers; l++) if (!Q38Shape::full_attention(l)) {
        views_[l].la_norm_bf16 = bf16_arena_ + la_norm_off[l];
        views_[l].conv = bf16_arena_ + conv_off[l];
        views_[l].dt = bf16_arena_ + dt_off[l];
        views_[l].ab[0] = bf16_arena_ + ab_off[0][l];
        views_[l].ab[1] = bf16_arena_ + ab_off[1][l];
    }
    // scale pointers per matrix
    auto scale_ptr = [&](const std::string &base) -> const uint16_t * {
        const TensorView *t = model_.find(base + ".scales");
        if (!t) throw std::runtime_error("missing scales for " + base);
        // offset within the scales arena = sum of bytes of all scale tensors sorted before it
        static std::vector<const TensorView *> order;
        if (order.empty()) for (const auto &x : model_.tensors()) if (x.name.find(".scales") != std::string::npos) order.push_back(&x);
        size_t off = 0;
        for (const TensorView *x : order) { if (x == t) break; off += size_t(x->bytes); }
        return scales_arena_ + off / 2;
    };
    for (int l = 0; l < Q38Shape::layers; l++) {
        const std::string p = "language_model.model.layers." + std::to_string(l);
        LayerView27 &v = views_[l];
        auto setv = [&](Fp8View &f, const std::string &base) {
            const TensorView *t = model_.find(base + ".weight");
            if (!t || t->dtype != DType::f8_e4m3) throw std::runtime_error("missing F8 " + base + ".weight");
            f.rows = int(t->shape[0]); f.cols = int(t->shape[1]); f.s = scale_ptr(base);
        };
        if (Q38Shape::full_attention(l)) {
            const std::string a = p + ".self_attn";
            setv(v.q, a + ".q_proj"); setv(v.k, a + ".k_proj"); setv(v.v, a + ".v_proj"); setv(v.o, a + ".o_proj");
        } else {
            const std::string a = p + ".linear_attn";
            setv(v.qkv, a + ".in_proj_qkv"); setv(v.z, a + ".in_proj_z"); setv(v.out, a + ".out_proj");
        }
        setv(v.gate, p + ".mlp.gate_proj"); setv(v.up, p + ".mlp.up_proj"); setv(v.down, p + ".mlp.down_proj");
    }
    // lm_head: device copy (2.54 GB) — refuse to start if VRAM short
    {
        const TensorView *t = model_.find("language_model.lm_head.weight");
        if (!t || t->dtype != DType::bf16) throw std::runtime_error("missing bf16 lm_head");
        size_t free_b = 0, tot = 0; cudaMemGetInfo(&free_b, &tot);
        if (free_b < t->bytes + (256ull << 20)) throw std::runtime_error("not enough free VRAM for lm_head (need 2.54 GB)");
        lm_head_ = static_cast<uint32_t *>(dev_tensor_chunked(*t));
    }
    // embed file handle + row offset (rows stay on NVMe)
    {
        const TensorView *t = model_.find("language_model.model.embed_tokens.weight");
        if (!t) throw std::runtime_error("missing embed table");
        const ShardInfo &sh = model_.shards()[t->shard];
        embed_file_ = CreateFileW(sh.path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (embed_file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open embed shard");
        embed_off_ = t->off;
    }
}

void TieredStorage27::load_v_layers() {
    for (int l = 0; l < Q38Shape::layers; l++) {
        if (tier_[l] != Tier::V) continue;
        LayerView27 &v = views_[l];
        const std::string p = "language_model.model.layers." + std::to_string(l);
        auto load1 = [&](Fp8View &f, const std::string &base) {
            const TensorView *t = model_.find(base + ".weight");
            f.w = static_cast<const uint8_t *>(dev_tensor_chunked(*t));
            slabs_.push_back(const_cast<uint8_t *>(f.w));
        };
        if (Q38Shape::full_attention(l)) {
            const std::string a = p + ".self_attn";
            load1(v.q, a + ".q_proj"); load1(v.k, a + ".k_proj"); load1(v.v, a + ".v_proj"); load1(v.o, a + ".o_proj");
        } else {
            const std::string a = p + ".linear_attn";
            load1(v.qkv, a + ".in_proj_qkv"); load1(v.z, a + ".in_proj_z"); load1(v.out, a + ".out_proj");
        }
        load1(v.gate, p + ".mlp.gate_proj"); load1(v.up, p + ".mlp.up_proj"); load1(v.down, p + ".mlp.down_proj");
    }
}

void TieredStorage27::build_n_plans() {
    n_layer_ids_.clear();
    n_plans_.clear();
    n_tensors_.clear();
    n_paths_.clear();
    n_paths_.reserve(Q38Shape::layers * 7);   // c_str() stability: no reallocation after reserve
    for (int l = 0; l < Q38Shape::layers; l++) {
        if (tier_[l] != Tier::N) continue;
        const std::string p = "language_model.model.layers." + std::to_string(l);
        LayerView27 &v = views_[l];
        ReadPlan plan;
        std::vector<NTensor> its;
        auto add = [&](Fp8View &f, int slot, const std::string &base) {
            const TensorView *t = model_.find(base + ".weight");
            if (!t) throw std::runtime_error("missing F8 " + base + ".weight");
            if (t->off & 15) throw std::runtime_error("N-tier tensor " + base + " not 16B aligned in shard — put layer " + std::to_string(l) + " in V");
            n_paths_.push_back(model_.shards()[t->shard].path);
            plan.push_back(ReadRequest{n_paths_.back().c_str(), t->off, t->bytes});
            its.push_back(NTensor{t->shard, t->off, t->bytes, slot});
        };
        if (Q38Shape::full_attention(l)) {
            const std::string a = p + ".self_attn";
            add(v.q, 3, a + ".q_proj"); add(v.k, 4, a + ".k_proj"); add(v.v, 5, a + ".v_proj"); add(v.o, 6, a + ".o_proj");
        } else {
            const std::string a = p + ".linear_attn";
            add(v.qkv, 0, a + ".in_proj_qkv"); add(v.z, 1, a + ".in_proj_z"); add(v.out, 2, a + ".out_proj");
        }
        add(v.gate, 7, p + ".mlp.gate_proj"); add(v.up, 8, p + ".mlp.up_proj"); add(v.down, 9, p + ".mlp.down_proj");
        n_plans_.push_back(std::move(plan));
        n_tensors_.push_back(std::move(its));
        n_layer_ids_.push_back(l);
    }
}

void TieredStorage27::begin_epoch() { feeder_.begin_epoch(n_plans_); }

const void *TieredStorage27::acquire(int l) {
    int i = 0;
    while (i < int(n_layer_ids_.size()) && n_layer_ids_[i] != l) i++;
    if (i >= int(n_layer_ids_.size())) throw std::runtime_error("acquire on non-N layer");
    const void *base = feeder_.acquire_layer(i);
    if (!base) throw std::runtime_error("stream fatal error (reader unhealthy)");
    LayerView27 &v = views_[l];
    for (int r = 0; r < int(n_tensors_[i].size()); r++) {
        const uint8_t *ptr = static_cast<const uint8_t *>(feeder_.map(i, r));
        switch (n_tensors_[i][r].view_slot) {
            case 0: v.qkv.w = ptr; break; case 1: v.z.w = ptr; break; case 2: v.out.w = ptr; break;
            case 3: v.q.w = ptr; break;  case 4: v.k.w = ptr; break;  case 5: v.v.w = ptr; break; case 6: v.o.w = ptr; break;
            case 7: v.gate.w = ptr; break; case 8: v.up.w = ptr; break; case 9: v.down.w = ptr; break;
        }
        if (reinterpret_cast<uintptr_t>(ptr) & 15) throw std::runtime_error("slot tensor misaligned");
    }
    return base;
}
const void *TieredStorage27::acquire_staged(int l) {
    const void *base = acquire(l);
    // VRAM staging slab sized to the largest N payload (once)
    if (!n_stage_) {
        size_t need = 0;
        for (size_t i = 0; i < n_layer_ids_.size(); i++) {
            size_t sum = 0;
            for (const auto &t : n_tensors_[i]) sum += size_t(t.bytes);
            need = std::max(need, sum);
        }
        size_t free_b = 0, tot = 0; cudaMemGetInfo(&free_b, &tot);
        if (free_b < need + (64ull << 20)) throw std::runtime_error("not enough VRAM for the N staging slab");
        cuda_check(cudaMalloc(&n_stage_, need), "n staging");
        n_stage_bytes_ = need;
    }
    int i = 0;
    while (i < int(n_layer_ids_.size()) && n_layer_ids_[i] != l) i++;
    uint8_t *dst = n_stage_;
    LayerView27 &v = views_[l];
    struct Copy { Fp8View *f; const void *src; size_t bytes; };
    Copy copies[kNF8];
    int nc = 0;
    auto take = [&](Fp8View &f) { if (f.w && f.rows) { copies[nc++] = {&f, f.w, size_t(f.rows) * size_t(f.cols)}; } };
    take(v.qkv); take(v.z); take(v.out); take(v.q); take(v.k); take(v.v); take(v.o); take(v.gate); take(v.up); take(v.down);
    size_t off = 0;
    for (int c = 0; c < nc; c++) {
        if ((off & 15)) off += 16 - (off & 15);   // per-tensor 16B alignment inside the slab
        cudaMemcpyAsync(dst + off, copies[c].src, copies[c].bytes, cudaMemcpyHostToDevice, stream_);
        copies[c].f->w = dst + off;
        off += copies[c].bytes;
    }
    if (off > n_stage_bytes_) throw std::runtime_error("staging slab overflow");
    return base;
}
void TieredStorage27::release(int l) {
    int i = 0;
    while (i < int(n_layer_ids_.size()) && n_layer_ids_[i] != l) i++;
    if (i < int(n_layer_ids_.size())) feeder_.release_layer(i);
}

const uint16_t *TieredStorage27::embed_row(int token) {
    LARGE_INTEGER li; li.QuadPart = embed_off_ + uint64_t(token) * Q38Shape::hidden * 2;
    OVERLAPPED ov{}; ov.Offset = li.LowPart; ov.OffsetHigh = li.HighPart;
    DWORD got = 0;
    if (!ReadFile(static_cast<HANDLE>(embed_file_), embed_, Q38Shape::hidden * 2, &got, &ov) || got != Q38Shape::hidden * 2)
        throw std::runtime_error("embed row pread failed");
    return embed_;
}
const uint16_t *TieredStorage27::embed_rows(const int *tokens, int T) {
    for (int t = 0; t < T; t++) {
        LARGE_INTEGER li; li.QuadPart = embed_off_ + uint64_t(tokens[t]) * Q38Shape::hidden * 2;
        OVERLAPPED ov{}; ov.Offset = li.LowPart; ov.OffsetHigh = li.HighPart;
        uint16_t *dst = embed_ + size_t(t) * Q38Shape::hidden;
        DWORD got = 0;
        if (!ReadFile(static_cast<HANDLE>(embed_file_), dst, Q38Shape::hidden * 2, &got, &ov) || got != Q38Shape::hidden * 2)
            throw std::runtime_error("embed row pread failed");
    }
    return embed_;
}

// ===========================================================================
// Workspace27
// ===========================================================================
Workspace27::Workspace27(int max_context) : max_context(max_context) {
    const int H = Q38Shape::hidden, I = Q38Shape::inter;
    auto M = [&](void **p, size_t b) { cuda_check(cudaMalloc(p, b), "workspace"); };
    M((void **)&hidden, H * 4); M((void **)&norm, H * 4); M((void **)&qkv, 10240 * 4); M((void **)&z, 6144 * 4);
    M((void **)&a, 48 * 4); M((void **)&b, 48 * 4); M((void **)&core, 6144 * 4);
    M((void **)&gate, I * 4); M((void **)&up, I * 4); M((void **)&down, H * 4); M((void **)&key, 1024 * 4); M((void **)&value, 1024 * 4);
    M((void **)&logits, size_t(Q38Shape::vocab) * 4);
    M((void **)&delta_state, size_t(48) * 48 * 128 * 128 * 4);
    M((void **)&conv_state, size_t(48) * 10240 * 3 * 4);
    M((void **)&kv_keys, size_t(16) * max_context * 1024 * 4);
    M((void **)&kv_values, size_t(16) * max_context * 1024 * 4);
    M((void **)&pf_x, size_t(64) * H * 4); M((void **)&pf_n, size_t(64) * H * 4);
    M((void **)&pf_qkv, size_t(64) * 10240 * 4); M((void **)&pf_z, size_t(64) * 6144 * 4);
    M((void **)&pf_a, size_t(64) * 48 * 4); M((void **)&pf_b, size_t(64) * 48 * 4);
    M((void **)&pf_scratch, size_t(64) * 12288 * 4);
    M((void **)&pf_gate, size_t(64) * I * 4); M((void **)&pf_up, size_t(64) * I * 4); M((void **)&pf_down, size_t(64) * H * 4);
    M((void **)&pf_core, size_t(64) * 6144 * 4); M((void **)&pf_q, size_t(64) * 6144 * 4); M((void **)&pf_g, size_t(64) * 6144 * 4);
    M((void **)&pf_k, size_t(64) * 1024 * 4); M((void **)&pf_v, size_t(64) * 1024 * 4);
    M((void **)&pf_bf16, size_t(64) * I * 2);
    M((void **)&pf_tokens, 64 * 4); M((void **)&pos_dev, 8 * 4); M((void **)&token_dev, 4); M((void **)&next_dev, 4);
    cuda_check(cudaMallocHost(&next_host, 4), "next_host"); cuda_check(cudaMallocHost(&pos_host, 4), "pos_host");
    M((void **)&am_scratch, 8);
    cuda_check(cudaMemset(delta_state, 0, size_t(48) * 48 * 128 * 128 * 4), "delta memset");
    cuda_check(cudaMemset(conv_state, 0, size_t(48) * 10240 * 3 * 4), "conv memset");
    cuda_check(cudaMemset(kv_keys, 0, size_t(16) * max_context * 1024 * 4), "kv memset");
    cuda_check(cudaMemset(kv_values, 0, size_t(16) * max_context * 1024 * 4), "kv memset");
    cuda_check(cudaStreamCreate(&stream), "stream");
}
Workspace27::~Workspace27() {
    #define F(p) if (p) cudaFree(p)
    F(hidden); F(norm); F(qkv); F(z); F(a); F(b); F(core); F(gate); F(up); F(down); F(key); F(value); F(logits);
    F(delta_state); F(conv_state); F(kv_keys); F(kv_values);
    F(pf_x); F(pf_n); F(pf_qkv); F(pf_z); F(pf_a); F(pf_b); F(pf_scratch); F(pf_gate); F(pf_up); F(pf_down); F(pf_core); F(pf_q); F(pf_g); F(pf_k); F(pf_v); F(pf_bf16);
    F(pf_tokens); F(pos_dev); F(token_dev); F(next_dev); F(am_scratch);
    #undef F
    if (next_host) cudaFreeHost(next_host);
    if (pos_host) cudaFreeHost(pos_host);
    if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
}

// ===========================================================================
// Qwen38Decode
// ===========================================================================
void Qwen38Decode::linear(const Fp8View &m, const float *in, float *out) {
    fp8_gemv(m.w, m.s, in, out, m.rows, m.cols, x_.stream);
}
void Qwen38Decode::linear_batch(const Fp8View &m, const float *in, float *out, int T) {
    if (T < 64) cudaMemsetAsync((char *)x_.pf_bf16 + size_t(T) * m.cols * 2, 0, size_t(64 - T) * m.cols * 2, x_.stream);
    f32_to_bf16(in, x_.pf_bf16, size_t(T) * m.cols, x_.stream);
    fp8_gemm(m.w, m.s, x_.pf_bf16, out, m.rows, m.cols, T, x_.stream);
}

void Qwen38Decode::delta_layer(int l) {
    const LayerView27 &v = st_.layer(l);
    Workspace27 &x = x_;
    rmsnorm_f32w(x.hidden, v.in_norm, x.norm, 1, Q38Shape::hidden, x.stream);
    linear(v.qkv, x.norm, x.qkv);
    linear(v.z, x.norm, x.z);
    bf16_gemv_ab2_pair((const uint32_t *)v.ab[0], (const uint32_t *)v.ab[1], x.norm, x.norm, x.a, x.pf_a + 48, x.b, x.pf_b + 48, Q38Shape::hidden, 48, x.stream);
    const int di = l - l / 4;
    causal_conv4_silu(x.qkv, x.conv_state + size_t(di) * 10240 * 3, v.conv, 10240, x.stream);
    deltanet_parameters(x.a, x.b, v.a_log, v.dt, 48, x.stream);
    deltanet_decode27(x.delta_state + size_t(di) * 48 * 128 * 128, x.qkv, x.qkv + 2048, x.qkv + 4096, x.a, x.b, x.core, x.stream);
    gated_rmsnorm_bf16(x.core, v.la_norm_bf16, x.z, x.core, 48, 128, x.stream);
    linear(v.out, x.core, x.down);
    residual_add(x.hidden, x.down, Q38Shape::hidden, x.stream);
    rmsnorm_f32w(x.hidden, v.post_norm, x.norm, 1, Q38Shape::hidden, x.stream);
    linear(v.gate, x.norm, x.gate);
    linear(v.up, x.norm, x.up);
    silu_mul(x.gate, x.up, x.gate, Q38Shape::inter, x.stream);
    linear(v.down, x.gate, x.down);
    residual_add(x.hidden, x.down, Q38Shape::hidden, x.stream);
}

void Qwen38Decode::attention_layer(int l) {
    const LayerView27 &v = st_.layer(l);
    Workspace27 &x = x_;
    if (x.position >= x.max_context) throw std::runtime_error("KV cache full");
    rmsnorm_f32w(x.hidden, v.in_norm, x.norm, 1, Q38Shape::hidden, x.stream);
    linear(v.q, x.norm, x.gate);                       // raw q+gate [12288]
    split_q_gate27(x.gate, x.qkv, x.z, x.stream);      // q -> qkv[6144], gate -> z[6144]
    linear(v.k, x.norm, x.key);
    linear(v.v, x.norm, x.value);
    qk_norm_rope27(x.qkv, x.key, v.q_norm, v.k_norm, x.pos_dev, x.stream);
    const int ai = l / 4;
    float *kc = x.kv_keys + size_t(ai) * x.max_context * 1024, *vc = x.kv_values + size_t(ai) * x.max_context * 1024;
    store_kv(x.key, x.value, kc, vc, x.pos_dev, 0, x.stream);
    gqa_decode27(x.qkv, kc, vc, x.core, x.pos_dev, 0, x.stream);
    expand_gate_heads27(x.z, x.qkv, x.stream);         // qkv <- gate broadcast
    sigmoid_mul(x.core, x.qkv, 6144, x.stream);
    linear(v.o, x.core, x.down);
    residual_add(x.hidden, x.down, Q38Shape::hidden, x.stream);
    rmsnorm_f32w(x.hidden, v.post_norm, x.norm, 1, Q38Shape::hidden, x.stream);
    linear(v.gate, x.norm, x.gate);
    linear(v.up, x.norm, x.up);
    silu_mul(x.gate, x.up, x.gate, Q38Shape::inter, x.stream);
    linear(v.down, x.gate, x.down);
    residual_add(x.hidden, x.down, Q38Shape::hidden, x.stream);
}

void Qwen38Decode::forward_body() {
    cudaEvent_t ev[2] = {};
    for (int i = 0; i < 2; i++) cudaEventCreate(&ev[i]);
    int held[2] = {-1, -1}, nheld = 0;   // N layers with work possibly in flight (ascending)
    auto retire_one = [&]() {           // release the OLDEST held slot (strict feeder order); the
        cudaEventSynchronize(ev[nheld - 1]);   // NEWEST event implies all earlier ones (same stream)
        st_.release(held[0]);
        held[0] = held[1]; held[1] = -1; nheld--;
    };
    st_.begin_epoch();
    for (int l = 0; l < Q38Shape::layers; l++) {
        if (st_.tier_of(l) == Tier::N) {
            if (nheld == 2) retire_one();
            st_.acquire_staged(l);   // v1 bring-up: stage to VRAM (UVA zero-copy decode GEMV lands in the perf phase)
            if (Q38Shape::full_attention(l)) attention_layer(l); else delta_layer(l);
            cudaEventRecord(ev[nheld], x_.stream);
            held[nheld++] = l;
        } else {
            if (Q38Shape::full_attention(l)) attention_layer(l); else delta_layer(l);
        }
    }
    while (nheld) retire_one();
    for (int i = 0; i < 2; i++) cudaEventDestroy(ev[i]);
    rmsnorm_f32w(x_.hidden, st_.final_norm(), x_.norm, 1, Q38Shape::hidden, x_.stream);
    bf16_gemv_v2(st_.lm_head(), x_.norm, x_.logits, Q38Shape::lm_rows, Q38Shape::hidden, x_.stream);
}

void Qwen38Decode::forward_token(int token) {
    if (token < 0 || token >= Q38Shape::vocab) throw std::runtime_error("token out of range");
    if (x_.position >= x_.max_context) throw std::runtime_error("KV cache full");
    const uint16_t *row = st_.embed_row(token);            // pinned host row (NVMe pread)
    bf16_row_load_kernel<<<4, 256, 0, x_.stream>>>(reinterpret_cast<const uint32_t *>(row), x_.hidden, Q38Shape::hidden); launch_check("bf16_row_load");
    forward_body();
    bumpi27_kernel<<<1, 1, 0, x_.stream>>>(x_.pos_dev);
    x_.position++;
}
int Qwen38Decode::logits_argmax() {
    argmax_fast(x_.logits, Q38Shape::vocab, x_.next_dev, x_.am_scratch, x_.stream);
    cudaMemcpyAsync(x_.next_host, x_.next_dev, sizeof(int), cudaMemcpyDeviceToHost, x_.stream);
    cudaStreamSynchronize(x_.stream);
    return *x_.next_host;
}
int Qwen38Decode::decode_token(int token) { forward_token(token); return logits_argmax(); }
void Qwen38Decode::set_position(int pos) { *x_.pos_host = pos; cudaMemcpyAsync(x_.pos_dev, x_.pos_host, sizeof(int), cudaMemcpyHostToDevice, x_.stream); }

void Qwen38Decode::prefill_chunk_seam(const int *tokens, int T, void (*seam)(int, const float *, int, void *), void *user) {
    if (T <= 0 || T > 64) throw std::runtime_error("prefill chunk must be 1..64 tokens");
    if (x_.position + T > x_.max_context) throw std::runtime_error("KV cache full");
    Workspace27 &x = x_;
    const uint16_t *rows = st_.embed_rows(tokens, T);      // pinned [T][hidden] bf16
    bf16_rows_load_kernel<<<T, 256, 0, x.stream>>>(reinterpret_cast<const uint32_t *>(rows), x.pf_x, Q38Shape::hidden); launch_check("bf16_rows_load");
    cudaMemcpyAsync(x.pf_tokens, tokens, sizeof(int) * T, cudaMemcpyHostToDevice, x.stream);
    cudaEvent_t ev[2] = {};
    for (int i = 0; i < 2; i++) cudaEventCreate(&ev[i]);
    int held[2] = {-1, -1}, nheld = 0;
    auto retire_one = [&]() {
        cudaEventSynchronize(ev[held[0] & 1]);
        st_.release(held[0]);
        held[0] = held[1]; held[1] = -1; nheld--;
    };
    st_.begin_epoch();
    for (int l = 0; l < Q38Shape::layers; l++) {
        const LayerView27 &v = st_.layer(l);
        if (st_.tier_of(l) == Tier::N) {
            if (nheld == 2) retire_one();
            st_.acquire_staged(l);   // fp8_gemm cp.async cannot read pinned: stage to VRAM first
        }
        rmsnorm_f32w(x.pf_x, v.in_norm, x.pf_n, T, Q38Shape::hidden, x.stream);
        if (l == 0) dump_stage(x.pf_x, T * 5120, "pf_x0");
        if (l == 0) dump_stage(x.pf_n, T * 5120, "norm0");
        if (Q38Shape::full_attention(l)) {
            linear_batch(v.q, x.pf_n, x.pf_scratch, T);           // raw q+gate [T,12288]
            split_q_gate_batch27(x.pf_scratch, x.pf_q, x.pf_g, T, x.stream);
            linear_batch(v.k, x.pf_n, x.pf_k, T);                  // [T,1024] (64-row padded buffer)
            linear_batch(v.v, x.pf_n, x.pf_v, T);
            qk_norm_rope_batch27(x.pf_q, x.pf_k, v.q_norm, v.k_norm, x.pos_dev, T, x.stream);
            const int ai = l / 4;
            float *kc = x.kv_keys + size_t(ai) * x.max_context * 1024, *vc = x.kv_values + size_t(ai) * x.max_context * 1024;
            store_kv_batch(x.pf_k, x.pf_v, kc, vc, x.pos_dev, T, x.max_context, x.stream);
            gqa_prefill27(x.pf_q, kc, vc, x.pf_core, x.pos_dev, T, x.max_context, x.stream);
            sigmoid_mul(x.pf_core, x.pf_g, size_t(T) * 6144, x.stream);
            linear_batch(v.o, x.pf_core, x.pf_down, T);
        } else {
            linear_batch(v.qkv, x.pf_n, x.pf_qkv, T);
            linear_batch(v.z, x.pf_n, x.pf_z, T);
            if (l == 0) { dump_stage(x.pf_qkv, T * 10240, "qkv0"); dump_stage(x.pf_z, T * 6144, "z0"); }
            {
                // a/b: one bf16 pass for all T tokens (rows=48 not %32 -> dedicated kernel)
                if (T < 64) cudaMemsetAsync((char *)x.pf_bf16 + size_t(T) * Q38Shape::hidden * 2, 0, size_t(64 - T) * Q38Shape::hidden * 2, x.stream);
                f32_to_bf16(x.pf_n, x.pf_bf16, size_t(T) * Q38Shape::hidden, x.stream);
                bf16_gemv_ab_rows((const uint32_t *)v.ab[0], (const uint32_t *)v.ab[1], x.pf_bf16, x.pf_a, x.pf_b, Q38Shape::hidden, 48, T, x.stream);
                if (l == 0) { dump_stage(x.pf_a, T * 48, "ab0"); dump_stage(x.pf_b, T * 48, "ab1"); }
            }
            conv_prefill_silu27(x.pf_qkv, x.pf_scratch, x.conv_state + size_t(l - l / 4) * 10240 * 3, v.conv, T, x.stream);
            if (l == 0) dump_stage(x.pf_scratch, T * 10240, "conv0");
            deltanet_params_batch_h(x.pf_a, x.pf_b, v.a_log, v.dt, T, 48, x.stream);
            if (l == 0) { dump_stage(x.pf_a, T * 48, "pa0"); dump_stage(x.pf_b, T * 48, "pb0"); }
            deltanet_prefill27(x.delta_state + size_t(l - l / 4) * 48 * 128 * 128, x.pf_scratch, x.pf_a, x.pf_b, x.pf_core, T, x.stream);
            if (l == 0) dump_stage(x.pf_core, T * 6144, "core0");
            gated_rmsnorm_bf16(x.pf_core, v.la_norm_bf16, x.pf_z, x.pf_core, size_t(T) * 48, 128, x.stream);
            if (l == 0) dump_stage(x.pf_core, T * 6144, "gated0");
            linear_batch(v.out, x.pf_core, x.pf_down, T);
            if (l == 0) dump_stage(x.pf_down, T * 5120, "block0");
        }
        residual_add(x.pf_x, x.pf_down, size_t(T) * Q38Shape::hidden, x.stream);
        if (l == 0) dump_stage(x.pf_x, T * 5120, "res0");
        rmsnorm_f32w(x.pf_x, v.post_norm, x.pf_n, T, Q38Shape::hidden, x.stream);
        if (l == 0) dump_stage(x.pf_n, T * 5120, "norm2");
        linear_batch(v.gate, x.pf_n, x.pf_gate, T);
        linear_batch(v.up, x.pf_n, x.pf_up, T);
        if (l == 0) { dump_stage(x.pf_gate, T * 17408, "gate"); dump_stage(x.pf_up, T * 17408, "up"); }
        silu_mul(x.pf_gate, x.pf_up, x.pf_gate, size_t(T) * Q38Shape::inter, x.stream);
        if (l == 0) dump_stage(x.pf_gate, T * 17408, "act");
        linear_batch(v.down, x.pf_gate, x.pf_down, T);
        if (l == 0) dump_stage(x.pf_down, T * 5120, "mlpblock");
        residual_add(x.pf_x, x.pf_down, size_t(T) * Q38Shape::hidden, x.stream);
        if (st_.tier_of(l) == Tier::N) { cudaEventRecord(ev[nheld], x.stream); held[nheld++] = l; }
        if (seam) {
            while (nheld) retire_one();
            cudaStreamSynchronize(x.stream);
            seam(l, x.pf_x, T, user);
        }
    }
    while (nheld) retire_one();
    for (int i = 0; i < 2; i++) cudaEventDestroy(ev[i]);
    rmsnorm_f32w(x.pf_x, st_.final_norm(), x.pf_n, T, Q38Shape::hidden, x.stream);
    bf16_gemv_v2(st_.lm_head(), x.pf_n + size_t(T - 1) * Q38Shape::hidden, x.logits, Q38Shape::lm_rows, Q38Shape::hidden, x.stream);
    argmax_fast(x.logits, Q38Shape::vocab, x.next_dev, x.am_scratch, x.stream);
    cudaMemcpyAsync(x.hidden, x.pf_x + size_t(T - 1) * Q38Shape::hidden, Q38Shape::hidden * 4, cudaMemcpyDeviceToDevice, x.stream);
    addi_kernel_launch(x.pos_dev, T, x.stream);
    x.position += T;
}
int Qwen38Decode::prefill_chunk(const int *tokens, int T) {
    prefill_chunk_seam(tokens, T, nullptr, nullptr);
    cudaMemcpyAsync(x_.next_host, x_.next_dev, sizeof(int), cudaMemcpyDeviceToHost, x_.stream);
    cudaStreamSynchronize(x_.stream);
    return *x_.next_host;
}

}  // namespace insignia
