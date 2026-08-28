// DFlash2 block-diffusion drafter implementation. See the header for the
// algorithm contract; the reference is z-lab/dflash model.py (one-pass block
// forward, KV-injected target features, two-tap grouped dynamic convs, greedy
// top-16 selector walk).

#include "insignia_glm53_dflash2.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

#include <cuda_runtime.h>

#include "insignia_glm53_fp8.cuh"
#include "insignia_glm53_index.hpp"

namespace insignia::glm53 {

namespace {

void check(cudaError_t error, const char *what) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
}

__device__ __forceinline__ float bf16_f(uint16_t bits) {
    return __uint_as_float(uint32_t(bits) << 16);
}

inline float bf16_h(uint16_t bits) {
    uint32_t wide = uint32_t(bits) << 16;
    float out;
    std::memcpy(&out, &wide, sizeof(out));
    return out;
}

// RoPE frequency table for head_dim 128, theta 1e4: freq[i] = theta^(-2i/128),
// i = 0..63, computed on host in double precision.
__constant__ float c_rope_freq[64];

// Single shared oracle-dump handle: two independent "wb" fopen calls on the
// same path would truncate each other's output.
FILE *df_dump_file() {
    static FILE *file = [] {
        const char *path = std::getenv("INSIGNIA_GLM53_DF_DUMP");
        return path ? std::fopen(path, "wb") : nullptr;
    }();
    return file;
}

// Convert one staged BF16 embedding row into the block buffer.
__global__ void df_embed_row_kernel(const uint16_t *__restrict__ row, float *__restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < DFlash2Drafter::kHidden) out[i] = bf16_f(row[i]);
}

// RMSNorm over each row of [rows, cols] with an fp32 weight.
__global__ void df_rms_rows_kernel(const float *__restrict__ x, const float *__restrict__ w,
                                    float *__restrict__ y, int rows, int cols) {
    const int row = blockIdx.y;
    if (row >= rows) return;
    extern __shared__ float warp_sums[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float *xr = x + size_t(row) * cols;
    float partial = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = xr[i];
        partial = fmaf(v, v, partial);
    }
    // warp reduce
    for (int off = 16; off; off >>= 1) partial += __shfl_down_sync(~0u, partial, off);
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    if (warp == 0) {
        float total = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.f;
        for (int off = 16; off; off >>= 1) total += __shfl_down_sync(~0u, total, off);
        if (lane == 0) warp_sums[0] = rsqrtf(total / float(cols) + 1.0e-5f);
    }
    __syncthreads();
    const float scale = warp_sums[0];
    float *yr = y + size_t(row) * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) yr[i] = xr[i] * scale * w[i];
}

// Two-tap grouped dynamic causal conv. dyn[t, (side*2+tap)*256 + group] in
// the full [T, 1024] projection output; the side-1 finish pass instead reads
// the stashed [T, 512] buffer (tap-major). base[side*2*4096 + tap*4096 + c].
template <int SIDE>
__global__ void df_conv_kernel(const float *__restrict__ x, const float *__restrict__ dyn,
                               const float *__restrict__ base, float *__restrict__ y,
                               float *__restrict__ dyn1, int rows) {
    const int global = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = global >> 12, c = global & 4095;
    if (t >= rows) return;
    const int g = c >> 4;
    const int row_stride = SIDE == 0 ? 1024 : 512;
    const float k0 = base[SIDE * 2 * 4096 + 0 * 4096 + c] + dyn[t * row_stride + 0 * 256 + g];
    const float k1 = base[SIDE * 2 * 4096 + 1 * 4096 + c] + dyn[t * row_stride + 1 * 256 + g];
    const float x0 = x[t * 4096 + c];
    const float xp = t > 0 ? x[(t - 1) * 4096 + c] : 0.f;
    y[t * 4096 + c] = fmaf(k0, x0, k1 * xp);
    if (SIDE == 0 && (c & 15) == 0) {
        dyn1[t * 512 + 0 * 256 + g] = dyn[t * 1024 + (1 * 2 + 0) * 256 + g];
        dyn1[t * 512 + 1 * 256 + g] = dyn[t * 1024 + (1 * 2 + 1) * 256 + g];
    }
}

// Per-head RMSNorm (head_dim 128) + neox RoPE (pairs (i, i+64), freq index =
// i) for q rows [rows, heads*128] or k rows [rows, kvh*128]. One warp per
// (row, head); lane l owns dims l, l+32, l+64, l+96 so both rope pairs are
// lane-local.
template <int HEADS>
__global__ void df_norm_rope_kernel(float *__restrict__ x, const float *__restrict__ w,
                                    int pos0, int rows) {
    const int warp_id = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int row = warp_id / HEADS, head = warp_id % HEADS;
    if (row >= rows) return;
    float *xr = x + size_t(row) * HEADS * 128 + size_t(head) * 128;
    float v[4] = {xr[lane], xr[lane + 32], xr[lane + 64], xr[lane + 96]};
    float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    for (int off = 16; off; off >>= 1) ss += __shfl_down_sync(~0u, ss, off);
    const float inv = rsqrtf(__shfl_sync(~0u, ss, 0) * (1.0f / 128.f) + 1.0e-5f);
    v[0] = v[0] * inv * w[lane];
    v[1] = v[1] * inv * w[lane + 32];
    v[2] = v[2] * inv * w[lane + 64];
    v[3] = v[3] * inv * w[lane + 96];
    const float pos = float(pos0 + row);
    const float a0 = pos * c_rope_freq[lane], a1 = pos * c_rope_freq[lane + 32];
    const float c0 = __cosf(a0), s0 = __sinf(a0), c1 = __cosf(a1), s1 = __sinf(a1);
    // pair (d, d+64): rot(d) = d*cos - (d+64)*sin; rot(d+64) = (d+64)*cos + d*sin
    xr[lane] = v[0] * c0 - v[2] * s0;
    xr[lane + 64] = v[2] * c0 + v[0] * s0;
    xr[lane + 32] = v[1] * c1 - v[3] * s1;
    xr[lane + 96] = v[3] * c1 + v[1] * s1;
}

// Block attention: 8 queries against ctx_len cached context keys plus the 8
// block keys, all visible (the training mask is a 2048 two-sided window that
// never bites below 2048 context). One warp per (position, q head); each warp
// scores ctx_len+8 keys. GQA: q head h reads kv head h>>2.
__global__ void df_attn_kernel(const float *__restrict__ q,      // [8, 32*128]
                               const float *__restrict__ kcache, // [5][264][8*128]
                               const float *__restrict__ vcache,
                               const float *__restrict__ kblk,    // [8, 8*128]
                               const float *__restrict__ vblk,
                               float *__restrict__ out,           // [8, 32*128]
                               int layer, int ctx_len) {
    const int warp_id = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int t = warp_id >> 5, h = warp_id & 31;
    const int kvh = h >> 2;
    const int keys = ctx_len + DFlash2Drafter::kBlock;
    __shared__ float tile[8][DFlash2Drafter::kMaxCtx];
    __shared__ float scale_inv;
    if (threadIdx.x == 0) scale_inv = rsqrtf(128.f);
    __syncthreads();

    const float *qh = q + size_t(t) * 4096 + size_t(h) * 128;
    const float qv[4] = {qh[lane], qh[lane + 32], qh[lane + 64], qh[lane + 96]};
    // Every warp scores ALL keys itself (its shared tile row is private);
    // 264 dots per warp, no cross-warp reduction needed until softmax.
    const int warp = threadIdx.x >> 5;
    for (int j = 0; j < keys; ++j) {
        const float *kh = j < ctx_len
            ? kcache + (size_t(layer) * 264 + j) * 1024 + size_t(kvh) * 128
            : kblk + size_t(j - ctx_len) * 1024 + size_t(kvh) * 128;
        const float k0 = kh[lane], k1 = kh[lane + 32], k2 = kh[lane + 64], k3 = kh[lane + 96];
        float dot = qv[0] * k0 + qv[1] * k1 + qv[2] * k2 + qv[3] * k3;
        for (int off = 16; off; off >>= 1) dot += __shfl_down_sync(~0u, dot, off);
        tile[warp][j] = __shfl_sync(~0u, dot, 0) * scale_inv;
    }
    __syncthreads();
    float m = -1e30f;
    for (int j = lane; j < keys; j += 32) m = fmaxf(m, tile[warp][j]);
    for (int off = 16; off; off >>= 1) m = fmaxf(m, __shfl_down_sync(~0u, m, off));
    m = __shfl_sync(~0u, m, 0);
    float denom = 0.f;
    for (int j = lane; j < keys; j += 32) {
        const float e = __expf(tile[warp][j] - m);
        tile[warp][j] = e;
        denom += e;
    }
    for (int off = 16; off; off >>= 1) denom += __shfl_down_sync(~0u, denom, off);
    denom = __shfl_sync(~0u, denom, 0);
    const float inv_denom = 1.f / denom;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < keys; ++j) {
        const float p = tile[warp][j] * inv_denom;
        const float *vh = j < ctx_len
            ? vcache + (size_t(layer) * 264 + j) * 1024 + size_t(kvh) * 128
            : vblk + size_t(j - ctx_len) * 1024 + size_t(kvh) * 128;
        acc[0] = fmaf(p, vh[lane], acc[0]);
        acc[1] = fmaf(p, vh[lane + 32], acc[1]);
        acc[2] = fmaf(p, vh[lane + 64], acc[2]);
        acc[3] = fmaf(p, vh[lane + 96], acc[3]);
    }
    float *oh = out + size_t(t) * 4096 + size_t(h) * 128;
    oh[lane] = acc[0];
    oh[lane + 32] = acc[1];
    oh[lane + 64] = acc[2];
    oh[lane + 96] = acc[3];
}

// Gather the 5 per-layer captures into the fc input halves: out_a[t, c] =
// cap[l][t][c] for l<... — each half carries specific capture layers
// (a: layers 0-1 plus 2048 of layer 2, b: the rest) is overkill; instead the
// two halves are pure column splits of the concatenated 20480 vector:
// a = captures 0..1 + 2048 cols of capture 2 handled by a strided gather.
__global__ void df_gather_kernel(const float *__restrict__ cap,  // [5][tokens][4096]
                                 float *__restrict__ out_a,      // [tokens, 10240]
                                 float *__restrict__ out_b,      // [tokens, 10240]
                                 int tokens) {
    const int global = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = global >> 12, c = global & 4095;
    if (t >= tokens) return;
    // concatenated column = l*4096 + c for capture l; halves split at 10240.
    const float v0 = cap[(size_t(0) * 32 + t) * 4096 + c];
    const float v1 = cap[(size_t(1) * 32 + t) * 4096 + c];
    const float v2 = cap[(size_t(2) * 32 + t) * 4096 + c];
    const float v3 = cap[(size_t(3) * 32 + t) * 4096 + c];
    const float v4 = cap[(size_t(4) * 32 + t) * 4096 + c];
    out_a[t * 10240 + c] = v0;
    out_a[t * 10240 + 4096 + c] = v1;
    if (c < 2048) {
        out_a[t * 10240 + 8192 + c] = v2;
    } else {
        const int cc = c - 2048;
        out_b[t * 10240 + cc] = v2;
    }
    out_b[t * 10240 + 2048 + c] = v3;
    out_b[t * 10240 + 6144 + c] = v4;
}

// K/V cache append: per-head k_norm + RoPE on k rows, then both k and v land
// in the layer's cache rows pos0..pos0+rows-1.
__global__ void df_kv_append_kernel(const float *__restrict__ k, const float *__restrict__ v,
                                    const float *__restrict__ kw, float *__restrict__ kcache,
                                    float *__restrict__ vcache, int rows, int pos0,
                                    int layer) {
    const int warp_id = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int t = warp_id >> 3, kvh = warp_id & 7;
    if (t >= rows) return;
    const float *kr = k + size_t(t) * 1024 + size_t(kvh) * 128;
    float v0 = kr[lane], v1 = kr[lane + 32], v2 = kr[lane + 64], v3 = kr[lane + 96];
    float ss = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    for (int off = 16; off; off >>= 1) ss += __shfl_down_sync(~0u, ss, off);
    const float inv = rsqrtf(__shfl_sync(~0u, ss, 0) * (1.0f / 128.f) + 1.0e-5f);
    v0 = v0 * inv * kw[lane];
    v1 = v1 * inv * kw[lane + 32];
    v2 = v2 * inv * kw[lane + 64];
    v3 = v3 * inv * kw[lane + 96];
    const float pos = float(pos0 + t);
    const float a0 = pos * c_rope_freq[lane], a1 = pos * c_rope_freq[lane + 32];
    const float c0 = __cosf(a0), s0 = __sinf(a0), c1 = __cosf(a1), s1 = __sinf(a1);
    float *krow = kcache + (size_t(layer) * 264 + pos0 + t) * 1024 + size_t(kvh) * 128;
    krow[lane] = v0 * c0 - v2 * s0;
    krow[lane + 64] = v2 * c0 + v0 * s0;
    krow[lane + 32] = v1 * c1 - v3 * s1;
    krow[lane + 96] = v3 * c1 + v1 * s1;
    const float *vr = v + size_t(t) * 1024 + size_t(kvh) * 128;
    float *vrow = vcache + (size_t(layer) * 264 + pos0 + t) * 1024 + size_t(kvh) * 128;
    vrow[lane] = vr[lane];
    vrow[lane + 32] = vr[lane + 32];
    vrow[lane + 64] = vr[lane + 64];
    vrow[lane + 96] = vr[lane + 96];
}

__global__ void df_add_kernel(float *__restrict__ dst, const float *__restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

__global__ void df_silu_mul_kernel(const float *__restrict__ g, const float *__restrict__ u,
                                   float *__restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = g[i];
        y[i] = x / (1.f + __expf(-x)) * u[i];
    }
}

}  // namespace

DFlash2Drafter::DFlash2Drafter(const std::string &index_path, const std::string &model_root,
                               const std::string &fp8_prefix, int vocab)
    : vocab_(vocab) {
    float freqs[64];
    for (int i = 0; i < 64; ++i)
        freqs[i] = float(std::pow(10000.0, -2.0 * i / 128.0));
    check(cudaMemcpyToSymbol(c_rope_freq, freqs, sizeof(freqs)), "df rope freq upload");

    Q8Index fp8(fp8_prefix);
    if (fp8.format() != Cache8Format::fp8_e4m3)
        throw std::runtime_error("DFlash2 FP8 cache has the wrong format");
    auto load = [&](Fp8Mat &into, const char *name) {
        const Q8TensorLocation *loc = fp8.find(name);
        if (!loc) throw std::runtime_error(std::string("DFlash2 FP8 cache misses ") + name);
        std::vector<uint8_t> host(loc->weight_bytes + loc->scale_bytes);
        fp8.read_rows(*loc, 0, loc->rows, host.data(), host.data() + loc->weight_bytes);
        into.rows = int(loc->rows);
        into.cols = int(loc->cols);
        check(cudaMalloc(&into.w, loc->weight_bytes), "df fp8 weights alloc");
        check(cudaMalloc(&into.s, loc->scale_bytes), "df fp8 scales alloc");
        check(cudaMemcpy(into.w, host.data(), loc->weight_bytes, cudaMemcpyHostToDevice),
              "df fp8 weights upload");
        check(cudaMemcpy(into.s, host.data() + loc->weight_bytes, loc->scale_bytes,
                         cudaMemcpyHostToDevice), "df fp8 scales upload");
    };
    for (int l = 0; l < kLayers; ++l) {
        const std::string p = "L" + std::to_string(l) + ".";
        load(q_[l], (p + "q").c_str());
        load(k_[l], (p + "k").c_str());
        load(v_[l], (p + "v").c_str());
        load(o_[l], (p + "o").c_str());
        load(gate_[l], (p + "gate").c_str());
        load(up_[l], (p + "up").c_str());
        load(down_[l], (p + "down").c_str());
        load(akp_[l], (p + "akp").c_str());
        load(mkp_[l], (p + "mkp").c_str());
    }
    load(fc_a_, "fc.a");
    load(fc_b_, "fc.b");
    load(hp_, "hp");

    ShardedIndex bf16(index_path, model_root);
    auto small_f32 = [&](const char *name, float **dst, size_t count) {
        const TensorLocation &loc = bf16.tensor(name);
        if (loc.type != TensorType::bf16 || loc.bytes != count * 2)
            throw std::runtime_error(std::string("DFlash2 small tensor geometry: ") + name);
        std::vector<uint16_t> bits(count);
        bf16.read(loc, bits.data());
        std::vector<float> wide(count);
        for (size_t i = 0; i < count; ++i) wide[i] = bf16_h(bits[i]);
        check(cudaMalloc(dst, count * sizeof(float)), "df small alloc");
        check(cudaMemcpy(*dst, wide.data(), count * sizeof(float), cudaMemcpyHostToDevice),
              "df small upload");
    };
    for (int l = 0; l < kLayers; ++l) {
        const std::string s = "layers." + std::to_string(l) + ".";
        small_f32((s + "input_layernorm.weight").c_str(), &input_ln_[l], kHidden);
        small_f32((s + "post_attention_layernorm.weight").c_str(), &post_ln_[l], kHidden);
        small_f32((s + "self_attn.q_norm.weight").c_str(), &q_norm_[l], kHeadDim);
        small_f32((s + "self_attn.k_norm.weight").c_str(), &k_norm_[l], kHeadDim);
        small_f32((s + "attention_conv.base_kernel").c_str(), &conv_base_[2 * l], 2 * 2 * kHidden);
        small_f32((s + "mlp_conv.base_kernel").c_str(), &conv_base_[2 * l + 1], 2 * 2 * kHidden);
    }
    small_f32("hidden_norm.weight", &hidden_norm_, kHidden);
    small_f32("norm.weight", &final_norm_, kHidden);

    auto codebook = [&](const char *name, std::vector<uint16_t> &bits) {
        const TensorLocation &loc = bf16.tensor(name);
        if (loc.type != TensorType::bf16)
            throw std::runtime_error("DFlash2 codebook dtype");
        bits.resize(loc.bytes / 2);
        bf16.read(loc, bits.data());
    };
    codebook("candidate_selector.predecessor_codebook", pred_bits_);
    codebook("candidate_selector.successor_codebook", succ_bits_);

    auto alloc = [](float **p, size_t floats) {
        check(cudaMalloc(p, floats * sizeof(float)), "df workspace alloc");
    };
    check(cudaMalloc(&kcache_, size_t(kLayers) * 264 * 1024 * sizeof(float)), "df kcache");
    check(cudaMalloc(&vcache_, size_t(kLayers) * 264 * 1024 * sizeof(float)), "df vcache");
    check(cudaMemset(kcache_, 0, size_t(kLayers) * 264 * 1024 * sizeof(float)), "df kcache zero");
    alloc(&x_block_, kBlock * kHidden);
    alloc(&xn_, kBlock * kHidden);
    alloc(&branch_, kBlock * kHidden);
    alloc(&sub_, kBlock * kHidden);
    alloc(&qbuf_, kBlock * kQHeads * kHeadDim);
    alloc(&kblk_, kBlock * kKVHeads * kHeadDim);
    alloc(&vblk_, kBlock * kKVHeads * kHeadDim);
    alloc(&dyn_, kBlock * 1024);
    alloc(&dyn1_, kBlock * 512);
    alloc(&gbuf_, kBlock * kIntermediate);
    alloc(&ubuf_, kBlock * kIntermediate);
    alloc(&hidden_, kDrafts * kHidden);
    alloc(&fc_in_a_, kMaxTokens * 10240);
    alloc(&fc_in_b_, kMaxTokens * 10240);
    alloc(&fc_out_, kMaxTokens * kHidden);
    alloc(&fc_tmp_, kMaxTokens * kHidden);
    alloc(&ctx_x_, kMaxTokens * kHidden);
    alloc(&ck_, kMaxTokens * kKVHeads * kHeadDim);
    alloc(&cv_, kMaxTokens * kKVHeads * kHeadDim);
    alloc(&capture_, size_t(kLayers) * kMaxTokens * kHidden);
    alloc(&hp_dev_, kDrafts * kRank);
    // fp8 activation workspace: largest cols among down (12288) and the fc
    // halves (10240), batch 32.
    const size_t ws_bytes = fp8_batch_workspace_bytes(12288, kMaxTokens);
    check(cudaMalloc(&workspace_, ws_bytes), "df fp8 workspace");
    std::printf("dflash2: drafter resident (%d FP8 matrices, codebooks %zu+%zu rows host)\n",
                9 * kLayers + 3, pred_bits_.size() / kRank, succ_bits_.size() / kRank);
}

DFlash2Drafter::~DFlash2Drafter() {
    auto free_mat = [](Fp8Mat &m) { cudaFree(m.w); cudaFree(m.s); };
    for (int l = 0; l < kLayers; ++l) {
        free_mat(q_[l]); free_mat(k_[l]); free_mat(v_[l]); free_mat(o_[l]);
        free_mat(gate_[l]); free_mat(up_[l]); free_mat(down_[l]);
        free_mat(akp_[l]); free_mat(mkp_[l]);
        cudaFree(input_ln_[l]); cudaFree(post_ln_[l]);
        cudaFree(q_norm_[l]); cudaFree(k_norm_[l]);
        cudaFree(conv_base_[2 * l]); cudaFree(conv_base_[2 * l + 1]);
    }
    free_mat(fc_a_); free_mat(fc_b_); free_mat(hp_);
    cudaFree(hidden_norm_); cudaFree(final_norm_);
    cudaFree(kcache_); cudaFree(vcache_);
    cudaFree(x_block_); cudaFree(xn_); cudaFree(branch_); cudaFree(sub_);
    cudaFree(qbuf_); cudaFree(kblk_); cudaFree(vblk_);
    cudaFree(dyn_); cudaFree(dyn1_);
    cudaFree(gbuf_); cudaFree(ubuf_); cudaFree(hidden_);
    cudaFree(fc_in_a_); cudaFree(fc_in_b_); cudaFree(fc_out_); cudaFree(fc_tmp_);
    cudaFree(ctx_x_); cudaFree(ck_); cudaFree(cv_); cudaFree(capture_);
    cudaFree(hp_dev_); cudaFree(workspace_);
}

float *DFlash2Drafter::capture_row(int capture_idx, int token) {
    return capture_ + (size_t(capture_idx) * kMaxTokens + token) * kHidden;
}

void DFlash2Drafter::set_block_row(int t, const uint16_t *device_row) {
    df_embed_row_kernel<<<8, 512>>>(device_row, x_block_ + size_t(t) * kHidden);
    check(cudaGetLastError(), "df embed row launch");
}

void DFlash2Drafter::forward(int anchor, int anchor_position) {
    static const bool ltrace = std::getenv("INSIGNIA_GLM53_DF_LTRACE") != nullptr;
    static const bool itrace = std::getenv("INSIGNIA_GLM53_DF_ITRACE") != nullptr;
    const auto trace_layer = [&](int l) {
        if (!ltrace || !df_dump_file())
            return;
        std::vector<float> host(8 * kHidden);
        check(cudaMemcpy(host.data(), x_block_, host.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), "df layer trace");
        const uint8_t tag = 4;
        std::fwrite(&tag, 1, 1, df_dump_file());
        const int8_t layer_i8 = int8_t(l);
        std::fwrite(&layer_i8, 1, 1, df_dump_file());
        int n = int(host.size());
        std::fwrite(&n, sizeof(n), 1, df_dump_file());
        std::fwrite(host.data(), sizeof(float), host.size(), df_dump_file());
        std::fflush(df_dump_file());
    };
    // [DEBUG-DF-ITRACE] Binary layer-0 boundary trace for the NumPy oracle.
    // Tag 5: i8 layer, i8 stage, i32 rows, i32 cols, followed by fp32 data.
    const auto trace_stage = [&](int l, int stage, const float *device, int rows, int cols) {
        if (!itrace || l != 0 || !df_dump_file())
            return;
        std::vector<float> host(size_t(rows) * cols);
        check(cudaMemcpy(host.data(), device, host.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), "df intermediate trace");
        const uint8_t tag = 5;
        const int8_t layer_i8 = int8_t(l), stage_i8 = int8_t(stage);
        std::fwrite(&tag, 1, 1, df_dump_file());
        std::fwrite(&layer_i8, 1, 1, df_dump_file());
        std::fwrite(&stage_i8, 1, 1, df_dump_file());
        std::fwrite(&rows, sizeof(rows), 1, df_dump_file());
        std::fwrite(&cols, sizeof(cols), 1, df_dump_file());
        std::fwrite(host.data(), sizeof(float), host.size(), df_dump_file());
        std::fflush(df_dump_file());
    };
    if (FILE *dump = df_dump_file()) {
        const uint8_t tag = 2;
        std::fwrite(&tag, 1, 1, dump);
        std::fwrite(&anchor, sizeof(anchor), 1, dump);
        std::fwrite(&anchor_position, sizeof(anchor_position), 1, dump);
        std::fflush(dump);
    }
    const int ctx_len = anchor_position + 1;
    trace_layer(-1);
    for (int l = 0; l < kLayers; ++l) {
        // attention sublayer
        df_rms_rows_kernel<<<dim3(2, kBlock), 256, 256 * sizeof(float)>>>(
            x_block_, input_ln_[l], xn_, kBlock, kHidden);
        check(cudaGetLastError(), "df rms launch");
        trace_stage(l, 0, xn_, kBlock, kHidden);
        check(fp8_tc_gemv_batch(akp_[l].w, akp_[l].s, xn_, dyn_, kBlock, akp_[l].rows,
                                kHidden, 1024, workspace_), "df akp gemv");
        trace_stage(l, 1, dyn_, kBlock, 1024);
        df_conv_kernel<0><<<128, 256>>>(xn_, dyn_, conv_base_[2 * l], branch_, dyn1_, kBlock);
        check(cudaGetLastError(), "df conv prepare");
        trace_stage(l, 2, branch_, kBlock, kHidden);
        check(fp8_tc_gemv_batch(q_[l].w, q_[l].s, branch_, qbuf_, kBlock, q_[l].rows,
                                kHidden, kQHeads * kHeadDim, workspace_), "df q gemv");
        check(fp8_tc_gemv2_batch(k_[l].w, k_[l].s, v_[l].w, v_[l].s,
                                 branch_, kblk_, vblk_, kBlock, k_[l].rows,
                                 kHidden, kKVHeads * kHeadDim, workspace_),
              "df kv gemv batch pair");
        trace_stage(l, 3, qbuf_, kBlock, kQHeads * kHeadDim);
        trace_stage(l, 4, kblk_, kBlock, kKVHeads * kHeadDim);
        trace_stage(l, 5, vblk_, kBlock, kKVHeads * kHeadDim);
        df_norm_rope_kernel<32><<<32, 256>>>(qbuf_, q_norm_[l], anchor_position, kBlock);
        df_norm_rope_kernel<8><<<8, 256>>>(kblk_, k_norm_[l], anchor_position, kBlock);
        check(cudaGetLastError(), "df norm rope");
        trace_stage(l, 6, qbuf_, kBlock, kQHeads * kHeadDim);
        trace_stage(l, 7, kblk_, kBlock, kKVHeads * kHeadDim);
        trace_stage(l, 21, kcache_ + size_t(l) * 264 * 1024, ctx_len,
                    kKVHeads * kHeadDim);
        trace_stage(l, 22, vcache_ + size_t(l) * 264 * 1024, ctx_len,
                    kKVHeads * kHeadDim);
        df_attn_kernel<<<32, 256>>>(qbuf_, kcache_, vcache_, kblk_, vblk_, branch_, l, ctx_len);
        check(cudaGetLastError(), "df attn");
        trace_stage(l, 8, branch_, kBlock, kHidden);
        check(fp8_tc_gemv_batch(o_[l].w, o_[l].s, branch_, sub_, kBlock, o_[l].rows,
                                kHidden, kHidden, workspace_), "df o gemv");
        trace_stage(l, 9, sub_, kBlock, kHidden);
        df_conv_kernel<1><<<128, 256>>>(sub_, dyn1_, conv_base_[2 * l], branch_, nullptr, kBlock);
        check(cudaGetLastError(), "df conv finish");
        trace_stage(l, 10, branch_, kBlock, kHidden);
        df_add_kernel<<<128, 256>>>(x_block_, branch_, kBlock * kHidden);
        trace_stage(l, 11, x_block_, kBlock, kHidden);

        // MLP sublayer
        df_rms_rows_kernel<<<dim3(2, kBlock), 256, 256 * sizeof(float)>>>(
            x_block_, post_ln_[l], xn_, kBlock, kHidden);
        check(cudaGetLastError(), "df rms 2 launch");
        trace_stage(l, 12, xn_, kBlock, kHidden);
        check(fp8_tc_gemv_batch(mkp_[l].w, mkp_[l].s, xn_, dyn_, kBlock, mkp_[l].rows,
                                kHidden, 1024, workspace_), "df mkp gemv");
        trace_stage(l, 13, dyn_, kBlock, 1024);
        df_conv_kernel<0><<<128, 256>>>(xn_, dyn_, conv_base_[2 * l + 1], branch_, dyn1_, kBlock);
        check(cudaGetLastError(), "df conv prepare 2");
        trace_stage(l, 14, branch_, kBlock, kHidden);
        check(fp8_tc_gemv2_batch(gate_[l].w, gate_[l].s, up_[l].w, up_[l].s,
                                 branch_, gbuf_, ubuf_, kBlock, gate_[l].rows,
                                 kHidden, kIntermediate, workspace_),
              "df gate/up batch pair");
        trace_stage(l, 15, gbuf_, kBlock, kIntermediate);
        trace_stage(l, 16, ubuf_, kBlock, kIntermediate);
        df_silu_mul_kernel<<<(kBlock * kIntermediate + 255) / 256, 256>>>(
            gbuf_, ubuf_, gbuf_, kBlock * kIntermediate);
        trace_stage(l, 17, gbuf_, kBlock, kIntermediate);
        check(fp8_tc_gemv_batch(down_[l].w, down_[l].s, gbuf_, sub_, kBlock, down_[l].rows,
                                kIntermediate, kHidden, workspace_), "df down gemv");
        trace_stage(l, 18, sub_, kBlock, kHidden);
        df_conv_kernel<1><<<128, 256>>>(sub_, dyn1_, conv_base_[2 * l + 1], branch_, nullptr, kBlock);
        check(cudaGetLastError(), "df conv finish 2");
        trace_stage(l, 19, branch_, kBlock, kHidden);
        df_add_kernel<<<128, 256>>>(x_block_, branch_, kBlock * kHidden);
        trace_stage(l, 20, x_block_, kBlock, kHidden);
        trace_layer(l);
    }
    // Final norm over rows 1..7 into the draft-hidden buffer (row stride
    // differs from the block buffer, so run the norm on the whole block and
    // copy rows 1..7 by addressing xn_ rows directly through a strided rms).
    df_rms_rows_kernel<<<dim3(2, kBlock), 256, 256 * sizeof(float)>>>(
        x_block_, final_norm_, xn_, kBlock, kHidden);
    check(cudaGetLastError(), "df final norm");
    check(cudaMemcpyAsync(hidden_, xn_ + kHidden, size_t(kDrafts) * kHidden * sizeof(float),
                          cudaMemcpyDeviceToDevice), "df hidden copy");
    check(fp8_tc_gemv_batch(hp_.w, hp_.s, hidden_, hp_dev_, kDrafts, hp_.rows, kHidden,
                            kRank, workspace_), "df hp gemv");
    if (FILE *dump = df_dump_file()) {
        // tag 3: forward intermediates for oracle comparison.
        std::vector<float> block(8 * kHidden), hid(kDrafts * kHidden);
        check(cudaMemcpy(block.data(), x_block_, block.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), "df xblock dump");
        check(cudaMemcpy(hid.data(), hidden_, hid.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), "df hidden dump");
        const uint8_t tag = 3;
        std::fwrite(&tag, 1, 1, dump);
        std::fwrite(&anchor_position, sizeof(anchor_position), 1, dump);
        int n = int(block.size());
        std::fwrite(&n, sizeof(n), 1, dump);
        std::fwrite(block.data(), sizeof(float), block.size(), dump);
        n = int(hid.size());
        std::fwrite(&n, sizeof(n), 1, dump);
        std::fwrite(hid.data(), sizeof(float), hid.size(), dump);
        std::fflush(dump);
    }
}

void DFlash2Drafter::commit(int count, int pos0) {
    if (FILE *dump = df_dump_file()) {
        // Oracle record: every committed token's five captured target
        // features, packed token-major. capture_ itself is layer-major with a
        // fixed kMaxTokens stride, so a single contiguous copy is incorrect
        // whenever count < kMaxTokens.
        std::vector<float> host(size_t(kLayers) * count * kHidden);
        for (int t = 0; t < count; ++t)
            for (int l = 0; l < kLayers; ++l)
                check(cudaMemcpy(host.data() + (size_t(t) * kLayers + l) * kHidden,
                                 capture_row(l, t), size_t(kHidden) * sizeof(float),
                                 cudaMemcpyDeviceToHost), "df capture dump");
        const uint8_t tag = 1;
        std::fwrite(&tag, 1, 1, dump);
        std::fwrite(&count, sizeof(count), 1, dump);
        std::fwrite(&pos0, sizeof(pos0), 1, dump);
        std::fwrite(host.data(), sizeof(float), host.size(), dump);
        std::fflush(dump);
    }
    df_gather_kernel<<<(kMaxTokens * 4096 + 255) / 256, 256>>>(capture_, fc_in_a_, fc_in_b_,
                                                               count);
    check(cudaGetLastError(), "df gather");
    check(fp8_tc_gemv_batch(fc_a_.w, fc_a_.s, fc_in_a_, fc_out_, count, fc_a_.rows,
                            10240, kHidden, workspace_), "df fc.a gemv");
    check(fp8_tc_gemv_batch(fc_b_.w, fc_b_.s, fc_in_b_, fc_tmp_, count, fc_b_.rows,
                            10240, kHidden, workspace_), "df fc.b gemv");
    df_add_kernel<<<(kMaxTokens * kHidden + 255) / 256, 256>>>(fc_out_, fc_tmp_,
                                                               count * kHidden);
    df_rms_rows_kernel<<<dim3(2, kMaxTokens), 256, 256 * sizeof(float)>>>(
        fc_out_, hidden_norm_, ctx_x_, count, kHidden);
    check(cudaGetLastError(), "df hidden norm");
    for (int l = 0; l < kLayers; ++l) {
        check(fp8_tc_gemv2_batch(k_[l].w, k_[l].s, v_[l].w, v_[l].s,
                                 ctx_x_, ck_, cv_, count, k_[l].rows,
                                 kHidden, kKVHeads * kHeadDim, workspace_),
              "df commit kv batch pair");
        df_kv_append_kernel<<<count, 256>>>(ck_, cv_, k_norm_[l],
                                            kcache_, vcache_, count, pos0, l);
        check(cudaGetLastError(), "df kv append");
    }
}

std::vector<int> DFlash2Drafter::select(const float *logits_host, const float *hp_host,
                                        int anchor) const {
    std::vector<int> path;
    path.reserve(kDrafts);
    int pred = anchor;
    float pred_row[kRank];
    for (int t = 0; t < kDrafts; ++t) {
        // top-16 linear scan of the logits row
        float topv[kTopK];
        int topi[kTopK];
        for (int k = 0; k < kTopK; ++k) { topv[k] = -1e30f; topi[k] = -1; }
        const float *row = logits_host + size_t(t) * vocab_;
        for (int v = 0; v < vocab_; ++v) {
            const float x = row[v];
            if (x > topv[kTopK - 1]) {
                int k = kTopK - 1;
                while (k > 0 && topv[k - 1] < x) { topv[k] = topv[k - 1]; topi[k] = topi[k - 1]; --k; }
                topv[k] = x; topi[k] = v;
            }
        }
        // predecessor gating vector: A[pred] (elementwise) times hp[t]
        const uint16_t *pbits = pred_bits_.data() + size_t(pred) * kRank;
        const float *hp = hp_host + size_t(t) * kRank;
        for (int i = 0; i < kRank; ++i)
            pred_row[i] = bf16_h(pbits[i]) * hp[i];
        int best = -1;
        float best_score = -1e30f;
        for (int k = 0; k < kTopK; ++k) {
            const uint16_t *sbits = succ_bits_.data() + size_t(topi[k]) * kRank;
            float dot = 0.f;
            for (int i = 0; i < kRank; ++i) dot = fmaf(pred_row[i], bf16_h(sbits[i]), dot);
            const float score = topv[k] + dot;
            if (score > best_score) { best_score = score; best = topi[k]; }
        }
        path.push_back(best);
        pred = best;
    }
    return path;
}

}  // namespace insignia::glm53
