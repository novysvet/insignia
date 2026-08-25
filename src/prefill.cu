#include "insignia_prefill.cuh"
#include "insignia_layout.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
namespace insignia {

// Gather T embedding rows (MXFP4 dequant) selected by device token ids.
__global__ void embed_gather_kernel(const uint32_t *__restrict__ w, const uint8_t *__restrict__ s, const int *__restrict__ tokens, float *__restrict__ out) {
    const int t = blockIdx.x, g = threadIdx.x;  // 128 groups per row, one thread per group
    const size_t row = __ldg(tokens + t);
    const uint4 packed = reinterpret_cast<const uint4 *>(w + row * 512)[g];  // 128 groups x 4 words
    const float scale = __int_as_float(static_cast<uint32_t>(s[row * 128 + g]) << 23);
    float *o = out + static_cast<size_t>(t) * 4096 + g * 32;
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    #pragma unroll
    for (int wi = 0; wi < 4; wi++)
        #pragma unroll
        for (int j = 0; j < 8; j++) o[wi * 8 + j] = decode4(words[wi], j) * scale;
}
void embed_gather(const uint32_t *w, const uint8_t *s, const int *tokens_dev, float *out, int T, cudaStream_t stream) {
    embed_gather_kernel<<<T, 128, 0, stream>>>(w, s, tokens_dev, out);
}

// INSIG4 embedding gather: fp16 super-group scales.
__global__ void embed_gather_i4_kernel(const uint32_t *__restrict__ w, const uint16_t *__restrict__ s, const int *__restrict__ tokens, float *__restrict__ out) {
    const int t = blockIdx.x, g = threadIdx.x;
    const size_t row = __ldg(tokens + t);
    const uint4 packed = reinterpret_cast<const uint4 *>(w + row * 512)[g];
    const float scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(s + row * 64 + (g >> 1)));
    float *o = out + static_cast<size_t>(t) * 4096 + g * 32;
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    #pragma unroll
    for (int wi = 0; wi < 4; wi++)
        #pragma unroll
        for (int j = 0; j < 8; j++) o[wi * 8 + j] = decode4(words[wi], j) * scale;
}
void embed_gather_i4(const uint32_t *w, const uint16_t *s, const int *tokens_dev, float *out, int T, cudaStream_t stream) {
    embed_gather_i4_kernel<<<T, 128, 0, stream>>>(w, s, tokens_dev, out);
}


__global__ void split_q_gate_batch_kernel(const float *__restrict__ src, float *__restrict__ q, float *__restrict__ gate) {
    const int t = blockIdx.x >> 4, h = blockIdx.x & 15, d = threadIdx.x;
    const size_t base = static_cast<size_t>(t) * 8192 + h * 512;
    q[static_cast<size_t>(t) * 4096 + h * 256 + d] = src[base + d];
    gate[static_cast<size_t>(t) * 4096 + h * 256 + d] = src[base + 256 + d];
}
void split_q_gate_batch(const float *src, float *q, float *gate, int T, cudaStream_t stream) {
    split_q_gate_batch_kernel<<<T * 16, 256, 0, stream>>>(src, q, gate);
}

// QK norm + partial RoPE for T tokens; token t uses position pos_dev[0] + t.
__global__ void qk_norm_rope_batch_kernel(float *__restrict__ q, float *__restrict__ k, const uint16_t *__restrict__ qw, const uint16_t *__restrict__ kw, const int *__restrict__ pos_dev) {
    const int t = blockIdx.y, head = blockIdx.x, tid = threadIdx.x;
    const bool isq = head < 16;
    float *p = isq ? q + (static_cast<size_t>(t) * 16 + head) * 256 : k + (static_cast<size_t>(t) * 4 + head - 16) * 256;
    const uint16_t *w = isq ? qw : kw;
    float v = p[tid], ss = v * v;
    for (int m = 16; m; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
    __shared__ float mem[64];
    int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) mem[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        ss = lane < 8 ? mem[lane] : 0;
        for (int m = 16; m; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
        if (lane == 0) mem[0] = rsqrtf(ss / 256 + 1e-6f);
    }
    __syncthreads();
    v *= mem[0] * __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(w + tid));
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
void qk_norm_rope_batch(float *q, float *k, const uint16_t *qw, const uint16_t *kw, const int *pos_dev, int T, cudaStream_t stream) {
    qk_norm_rope_batch_kernel<<<dim3(20, T), 256, 0, stream>>>(q, k, qw, kw, pos_dev);
}

__global__ void store_kv_batch_kernel(const float *__restrict__ k, const float *__restrict__ v, float *__restrict__ kc, float *__restrict__ vc, const int *__restrict__ pos_dev) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1024) return;
    const size_t pos = static_cast<size_t>(__ldg(pos_dev) + t);
    kc[pos * 1024 + i] = k[static_cast<size_t>(t) * 1024 + i];
    vc[pos * 1024 + i] = v[static_cast<size_t>(t) * 1024 + i];
}
void store_kv_batch(const float *k, const float *v, float *kc, float *vc, const int *pos_dev, int T, int max_context, cudaStream_t stream) {
    (void)max_context;
    store_kv_batch_kernel<<<dim3(4, T), 256, 0, stream>>>(k, v, kc, vc, pos_dev);
}

// One (query head, token) per block; causal over pos_dev[0]+t+1 cache keys.
// Warps own whole key rows (8 dims per lane) so K/V reads stay coalesced.
__global__ __launch_bounds__(256, 2) void gqa_prefill_kernel(const float *__restrict__ q, const float *__restrict__ kc, const float *__restrict__ vc, float *__restrict__ out, const int *__restrict__ pos_dev) {
    const int head = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, kvh = head >> 2;
    const int tokens = __ldg(pos_dev) + t + 1;
    __shared__ float qs[256];
    __shared__ float score[4096];
    __shared__ float red[8];
    __shared__ float part[8][256];
    const float *qrow = q + (static_cast<size_t>(t) * 16 + head) * 256;
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
    if (warp == 0) {
        mx = lane < 8 ? red[lane] : -3.402823466e+38F;
        for (int m = 16; m; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, m));
        if (lane == 0) red[0] = mx;
    }
    __syncthreads();
    mx = red[0];
    float den = 0;
    for (int j = warp; j < tokens; j += 8) {
        float e = __expf(score[j] - mx);
        if (lane == 0) { score[j] = e; den += e; }
    }
    for (int m = 16; m; m >>= 1) den += __shfl_xor_sync(0xffffffff, den, m);
    if (lane == 0) red[warp] = den;
    __syncthreads();
    if (warp == 0) {
        den = lane < 8 ? red[lane] : 0;
        for (int m = 16; m; m >>= 1) den += __shfl_xor_sync(0xffffffff, den, m);
        if (lane == 0) red[0] = 1.f / den;
    }
    __syncthreads();
    const float inv_den = red[0];
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
    out[(static_cast<size_t>(t) * 16 + head) * 256 + tid] = o;
}
void gqa_prefill(const float *q, const float *kc, const float *vc, float *out, const int *pos_dev, int T, int max_context, cudaStream_t stream) {
    (void)max_context;
    gqa_prefill_kernel<<<dim3(16, T), 256, 0, stream>>>(q, kc, vc, out, pos_dev);
}

// Depthwise causal conv4 + SiLU over T tokens; state carries the 3 pre-chunk inputs.
// Output goes to a separate buffer: x must stay raw until every thread has read it.
// Also snapshots the state as of after the FIRST row (spec-decode rollback point).
__global__ void conv_prefill_kernel(const float *__restrict__ x, float *__restrict__ out, const float *__restrict__ state, const uint16_t *__restrict__ w) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // t*8192 + c
    const int t = idx / 8192, c = idx % 8192;
    float z = 0;
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        const float hist = t >= 3 - i ? x[static_cast<size_t>(t - (3 - i)) * 8192 + c] : state[c * 3 + i];
        z = fmaf(hist, __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(w + c * 4 + i)), z);
    }
    z = fmaf(x[static_cast<size_t>(t) * 8192 + c], __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(w + c * 4 + 3)), z);
    out[idx] = z / (1.f + __expf(-z));
}
// Roll the last 3 raw inputs of the combined [state, x] window into the state, and
// write the row-0 checkpoint [s1, s2, x0] for speculative rollback.
__global__ void conv_roll_state_kernel(const float *__restrict__ x, float *__restrict__ state, float *__restrict__ snap, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 8192) return;
    snap[c * 3 + 0] = state[c * 3 + 1];
    snap[c * 3 + 1] = state[c * 3 + 2];
    snap[c * 3 + 2] = x[c];  // row 0's raw input
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        const int j = T - 3 + i;
        state[c * 3 + i] = j >= 0 ? x[static_cast<size_t>(j) * 8192 + c] : state[c * 3 + 3 + j];
    }
}
void conv_prefill_silu(float *x, float *scratch, float *state, const uint16_t *w, int T, cudaStream_t stream, float *row0_snap) {
    const int n = T * 8192;
    conv_prefill_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, scratch, state, w);
    conv_roll_state_kernel<<<(8192 + 255) / 256, 256, 0, stream>>>(x, state, row0_snap ? row0_snap : state, T);
}

// Per-token decay/beta: a[t][h] = -exp(A_log)*softplus(a+dt), b[t][h] = sigmoid(b).
__global__ void params_batch_kernel(float *__restrict__ a, float *__restrict__ b, const float *__restrict__ A_log, const uint16_t *__restrict__ dt, int T) {
    const int t = blockIdx.x, h = threadIdx.x;
    if (h >= 32) return;
    float *ar = a + static_cast<size_t>(t) * 32, *br = b + static_cast<size_t>(t) * 32;
    br[h] = 1.f / (1.f + __expf(-br[h]));
    const float z = ar[h] + __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(dt + h));
    const float soft = z > 20 ? z : log1pf(__expf(z));
    ar[h] = -__expf(A_log[h]) * soft;
}
void deltanet_params_batch(float *a, float *b, const float *A_log, const uint16_t *dt_bias, int T, cudaStream_t stream) {
    params_batch_kernel<<<T, 32, 0, stream>>>(a, b, A_log, dt_bias, T);
}

// Sequential gated DeltaNet over T tokens; one block per value head, state in shared.
// qkv layout per token: q[16][128] | k[16][128] | v[32][128]; a/b are per-token [32].
// After the first token the state is also written to `snap` (speculative rollback point).
__global__ __launch_bounds__(128) void deltanet_prefill_kernel(float *__restrict__ state, const float *__restrict__ qkv, const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ out, int T, float *__restrict__ snap) {
    extern __shared__ float sh[];  // [128*128] state + [4] scratch
    const int head = blockIdx.x, tid = threadIdx.x, kh = head >> 1;
    float *S = sh;
    float *gstate = state + static_cast<size_t>(head) * 128 * 128;
    for (int i = tid; i < 128 * 128; i += 128) S[i] = gstate[i];
    __syncthreads();
    for (int t = 0; t < T; t++) {
        const float *qt = qkv + static_cast<size_t>(t) * 8192 + kh * 128;
        const float *kt = qkv + static_cast<size_t>(t) * 8192 + 2048 + kh * 128;
        const float *vt = qkv + static_cast<size_t>(t) * 8192 + 4096 + head * 128;
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
        const float decay = expf(a[static_cast<size_t>(t) * 32 + head]);
        const float beta = b[static_cast<size_t>(t) * 32 + head];
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
        out[(static_cast<size_t>(t) * 32 + head) * 128 + tid] = acc;
        __syncthreads();
        if (t == 0 && snap) {  // committed row-0 checkpoint for speculative rollback
            float *gsnap = snap + static_cast<size_t>(head) * 128 * 128;
            for (int i = tid; i < 128 * 128; i += 128) gsnap[i] = S[i];
        }
    }
    for (int i = tid; i < 128 * 128; i += 128) gstate[i] = S[i];
}
void deltanet_prefill(float *state, const float *qkv, const float *a, const float *b, float *out, int T, cudaStream_t stream, float *row0_snap) {
    static const bool ok = [] { return cudaFuncSetAttribute(deltanet_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024 + 512) == cudaSuccess; }();
    (void)ok;
    deltanet_prefill_kernel<<<32, 128, 64 * 1024 + 512, stream>>>(state, qkv, a, b, out, T, row0_snap);
}

__global__ void addi_kernel(int *p, int v) { *p += v; }
void addi_kernel_launch(int *p, int v, cudaStream_t stream) { addi_kernel<<<1, 1, 0, stream>>>(p, v); }

// ---- device-state speculative step plumbing (CUDA-graph replayable) ----
// pos_dev layout: 0 position, 1 pending token, 2 next/after, 3 t2(row0 argmax), 4 draft,
// 5 committed count, 6 accept flag, 7 mtp position.
__global__ void spec_prologue_kernel(int *__restrict__ pos) { pos[7] = pos[0] - 1; }
void spec_prologue(int *pos, cudaStream_t stream) { spec_prologue_kernel<<<1, 1, 0, stream>>>(pos); }

__global__ void spec_setup_kernel(int *__restrict__ pos, int *__restrict__ pf_tokens) {
    pos[4] = pos[2];  // draft
    pf_tokens[0] = pos[1];
    pf_tokens[1] = pos[4];
}
void spec_setup(int *pos, int *pf_tokens, cudaStream_t stream) { spec_setup_kernel<<<1, 1, 0, stream>>>(pos, pf_tokens); }

__global__ void spec_commit_kernel(int *__restrict__ pos, int *__restrict__ committed) {
    const bool acc = pos[4] == pos[3];  // draft == t2
    const int c = pos[5];
    committed[c] = pos[1];
    if (acc) {  // both rows valid: commit [t0, draft], pending = after
        committed[c + 1] = pos[4];
        pos[1] = pos[2];
        pos[5] = c + 2;
    } else {    // only t0 was truly processed: commit [t0], pending = t2 (forwarded next step)
        pos[1] = pos[3];
        pos[5] = c + 1;
    }
    pos[6] = acc ? 1 : 0;
    if (!acc) pos[0] -= 1;  // only the verified token advanced the model state
}
void spec_commit(int *pos, int *committed, cudaStream_t stream) { spec_commit_kernel<<<1, 1, 0, stream>>>(pos, committed); }

// Undo the speculative row's recurrent state (accepted steps early-exit everywhere).
__global__ void spec_rollback_kernel(const float *__restrict__ snap_delta, const float *__restrict__ snap_conv, float *__restrict__ delta_state, float *__restrict__ conv_state, const float *__restrict__ pf_x, float *__restrict__ hidden, const int *__restrict__ pos) {
    if (pos[6]) return;
    const int n = 24 * 32 * 128 * 128;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) delta_state[i] = snap_delta[i];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 24 * 8192 * 3; i += gridDim.x * blockDim.x) conv_state[i] = snap_conv[i];
    if (blockIdx.x == 0 && threadIdx.x < 4096) hidden[threadIdx.x] = pf_x[threadIdx.x];  // row0's hidden
}
void spec_rollback(const float *snap_delta, const float *snap_conv, float *delta_state, float *conv_state, const float *pf_x, float *hidden, const int *pos, cudaStream_t stream) {
    spec_rollback_kernel<<<512, 256, 0, stream>>>(snap_delta, snap_conv, delta_state, conv_state, pf_x, hidden, pos);
}
}
