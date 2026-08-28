# INSIG4 v21 GEMM port + a/b monster-killer — full paste-ready spec (w3, 2026-08-25)

Deliverable of audits/w3/insig4-perf.md §3.1/§3.2. Read-only session: no builds run, no git
changes, no source modified. Everything below is verified against the tree as of today.

- `mxfp4_gemm_v21_i4` — cp.async-pipelined INSIG4 prefill GEMM (twin of `mxfp4_gemm_v21`,
  src/gemm.cu:210-296), replacing the unpipelined v1 clone `mxfp4_gemm_mlx_i4`
  (src/gemm.cu:302-357) that decode.cu:35 currently routes ALL INSIG4 batch GEMMs through.
- `mxfp4_gemm_ab_i4` — one-launch a+b GEMM killing the T×2 per-token GEMV loop at
  decode.cu:72-73 (3072 launches per 64-token chunk).

---

## 1. Verified facts the port rests on

### 1.1 Scale indexing: one fp16 scale per KT=64 step, index = kb — CONFIRMED

INSIG4 (src/mxfp4_i4.cu:8-13): scales are `__half`, one per **64-element super-group**,
layout `[rows][groups>>1]` where `groups = cols>>5`:

- `i4_scale(s, g)` reads `s[g >> 1]` (mxfp4_i4.cu:11-12) — 32-elt group index `g` ⇒
  super-group index `g>>1` = `col>>6`.
- GEMV twin: `row_s = scales + row*(groups>>1)` (mxfp4_i4.cu:36), scale per 32-elt group `g0`
  is `row_s[g0>>1]` — i.e. two consecutive nibble groups share ONE u16.
- Writer side confirms layout: `s16[r*(cols>>6) + sg]` (src/test_i4.cu:36) and
  `mxfp4_gemm_mlx_i4_kernel` reads `sg_scales[(n0+n)*(groups>>1) + ((k0>>5)>>1)]`
  (gemm.cu:315,330).

KT=64 step `kb` covers cols `[64*kb, 64*kb+64)` = exactly super-group `64*kb/64 = kb`.
Therefore the per-row scale for the whole step is:

```
scales[(n0 + n) * (groups>>1) + kb]        // groups = cols>>5;  groups>>1 = cols>>6
```

Cleaner than e8m0, which needed `kb*2 + half` because its u8 scale covers only 32 cols
(gemm.cu:249). Concrete check, cols=4096: groups=128, stride 64 super-groups/row; step kb=5
covers cols [320,384) = super-group 5 ⇒ index `(n0+n)*64 + 5`. ✓

### 1.2 The 256-entry bf16 pair-LUT — reuse verbatim — CONFIRMED

The LUT (gemm.cu:216-221 / 99-104) maps one byte to two packed bf16 via
`tbl = {0x0000,0x3F00,0x3F80,0x3FC0,0x4000,0x4040,0x4080,0x40C0}` + sign copies
(`0x8000...0xC0C0`). INSIG4's code space is defined by `fp4_e2m1` / `decode4`
(include/insignia_layout.cuh:20-26,38-45) and `e2m1_host` (src/test_i4.cu:14-17):
`{0,.5,1,1.5,2,3,4,6}` + sign bit 8 — **identical to MXFP4's E2M1**. INSIG4 differs only in
the scale encoding (fp16/64-elt vs e8m0/32-elt), never in the nibble semantics. The bf16
patterns are exact for all 16 codes (audit §1.2). ⇒ LUT block copied verbatim.

### 1.3 B nibble staging geometry — identical to v21 — CONFIRMED

INSIG4 weight rows are `groups*4` u32: 4 nibble-packed u32 per 32-elt group
(mxfp4_i4.cu:35 `row_w = weights + row*groups*4`, uint4 read at `g0*4`; mlx_i4 gemm.cu:329
`weights[row*groups*4 + (k0>>3) + w]`). MXFP4 MLX layout is the same expression
(gemm.cu:117). Per KT=64 step a 32-row tile needs 32 nibble-bytes/row = 2×16 B `cp.async.cg`
chunks into `Braw[2][32][8]` — v21 lines 237-242 verbatim. ✓

### 1.4 Pipeline, tiling, smem — carried over unchanged

64×32 tile, 256 threads (8 warps, warp = m16×n16), KT=64, `As[2][64][72]` +
`Bs[2][32][72]` + `Braw[2][32][8]` + `lut[256]` = **30,720 B static smem** (< 48 KB, no
opt-in needed) ⇒ 3 blocks/SM at 30 KB (audit §3.1.6). The double-buffer cp.async schedule
(commit / wait_group 1 / tail wait_group 0) is v21's, reviewed correct: the buffer being
dequanted at step kb was prefetched at kb and completed by `wait_prev`; it is not reused
until kb+2, after the loop-end `__syncthreads()`.

Numerics: A side rounds fp32→bf16 once (same `__float2bfloat16` as the v1 clone's staging),
B side = lut-bf16 × fp32(fp16 scale) rounded once to bf16 — **the same one-rounding-per-
element path as `mxfp4_gemm_mlx_i4`**, only fp32 accumulation order inside wmma differs
(cos ≈ 1−1e-7 vs the v1 clone). No new parity risk. (The `__hmul2` Design-A dequant from
audit §1.2 remains a separate measured follow-up; not bundled here.)

---

## 2. Paste-ready code — append at end of `src/gemm.cu`

```cuda
namespace insignia {
using namespace nvcuda;

// Pipelined INSIG4 prefill GEMM v2.1: Y[T,rows] = X16[T,cols] * W[rows,cols]^T, X16 bf16.
// Identical pipeline to mxfp4_gemm_v21 (64x32 tile, 256 threads, KT=64, cp.async double
// buffer, wmma 16x16x16 bf16 / f32 acc, 256-entry pair-LUT dequant). Only the scale path
// differs: INSIG4 has ONE fp16 scale per 64-element super-group per row, and a KT=64 step
// covers exactly one super-group -> scale index = kb (e8m0 needed kb*2 + half).
// B nibble packing is identical to MLX MXFP4 (4 u32 per 32-elt group), so the prefetch
// geometry is v21 verbatim. Requires rows%32==0, cols%1024==0 (kernel itself needs %64;
// %1024 keeps parity with mxfp4_gemv_v2_i4's fast-path gate), 1<=T<=64, and X16 zero-padded
// to 64 rows. Store is fp8_gemm's guarded pattern: only A-tiles with wm*16 < T are written
// (a tile straddling T still writes 16 rows - callers' pf_* buffers are 64-row).
__global__ __launch_bounds__(256) void mxfp4_gemm_v21_i4_kernel(const uint32_t *__restrict__ weights, const uint16_t *__restrict__ scales, const __nv_bfloat16 *__restrict__ x16, float *__restrict__ y, int rows, int cols, int T) {
    constexpr int KT = 64, NT = 32;
    constexpr int APAD = 8, BPAD = 8;
    __shared__ __nv_bfloat16 As[2][64][KT + APAD];
    __shared__ __nv_bfloat16 Bs[2][NT][KT + BPAD];
    __shared__ uint32_t Braw[2][NT][8];       // packed nibbles, 8 u32 per row per step
    __shared__ uint32_t lut[256];             // INSIG4 E2M1 == MXFP4 code space: v21 table verbatim
    {
        const unsigned tbl[16] = {0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
                                  0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0};
        lut[threadIdx.x] = tbl[threadIdx.x & 15] | (tbl[(threadIdx.x >> 4) & 15] << 16);
    }
    const int groups = cols >> 5;             // 32-elt nibble groups per row; scales stride groups>>1
    const int n0 = blockIdx.x * NT;
    const int tid = threadIdx.x;

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
        {   // B: NT rows x 32B nibbles (2x 16B) - same packing as MLX MXFP4
            for (int i = tid; i < NT * 2; i += 256) {
                const int n = i >> 1, half = i & 1;
                cp_async16(&Braw[buf][n][half * 4], reinterpret_cast<const char *>(weights + size_t(n0 + n) * groups * 4 + k / 8 + half * 4));
            }
        }
    };
    auto dequant = [&](int kb, int buf) {    // Braw -> Bs; ONE fp16 super-group scale per row per step
        const int n = tid >> 3, w = tid & 7;  // 256 threads = 32 rows x 8 words (n < NT always)
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
    // Guarded store (fp8_gemm pattern): only tiles holding at least one valid row;
    // the tile straddling T still writes 16 rows (64-row pf_* buffers make that scratch).
    if (wm * 16 < T) wmma::store_matrix_sync(y + size_t(wm * 16) * rows + n0 + wn * 16, acc, rows, wmma::mem_row_major);
}

void mxfp4_gemm_v21_i4(const uint32_t *weights, const uint16_t *scales, const void *x16, float *y, int rows, int cols, int T, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 1023) || T <= 0 || (rows & 31)) throw std::runtime_error("insignia: bad GEMM dims rows=" + std::to_string(rows) + " cols=" + std::to_string(cols));
    if (T > 64) throw std::runtime_error("insignia: mxfp4_gemm_v21_i4 T=" + std::to_string(T) + " exceeds the 64-row A tile");
    mxfp4_gemm_v21_i4_kernel<<<rows >> 5, 256, 0, stream>>>(weights, scales, (const __nv_bfloat16 *)x16, y, rows, cols, T);
}

// a/b monster-killer (audit w3 perf §3.2): computes BOTH in_proj_a (32 rows) and in_proj_b
// (32 rows) for all T tokens in ONE launch, replacing the T x 2 per-token GEMV loop in
// decode.cu (at T=64: 128 launches x 24 delta layers = 3072 launches ~ 9-12 ms per chunk).
// Two blocks, v21_i4 tile pipeline verbatim: block 0 walks a's 32 rows -> ya[T,32]; block 1
// walks b's 32 rows -> yb[T,32]. A (x16) is shared by both blocks (block 1 hits L2).
// Both tensors are assumed to have EXACTLY 32 rows (in_proj_a/b of the 9B - baked in,
// no rows parameter). Requires cols%1024==0, 1<=T<=64, x16 zero-padded to 64 rows;
// ya/yb need ceil(T/16)*16 rows (pf_a/pf_b are [64,32]).
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

    auto prefetch = [&](int kb, int buf) {   // cp.async: A tile + B nibbles for K step kb
        const int k = kb * KT;
        {   // A: 64 rows x 128B (64 bf16) = 512 chunks of 16B; thread strides
            const char *ag = reinterpret_cast<const char *>(x16);
            for (int i = tid; i < 64 * (KT / 8); i += 256) {
                const int m = i / (KT / 8), c8 = i % (KT / 8);
                const size_t srcoff = (size_t(m) * cols + k) * 2 + c8 * 16;
                cp_async16(&As[buf][m][c8 * 8], ag + srcoff);
            }
        }
        {   // B: NT rows x 32B nibbles (2x 16B); n0 = 0 within each tensor
            for (int i = tid; i < NT * 2; i += 256) {
                const int n = i >> 1, half = i & 1;
                cp_async16(&Braw[buf][n][half * 4], reinterpret_cast<const char *>(weights + size_t(n) * groups * 4 + k / 8 + half * 4));
            }
        }
    };
    auto dequant = [&](int kb, int buf) {    // Braw -> Bs; ONE fp16 super-group scale per row per step
        const int n = tid >> 3, w = tid & 7;  // 256 threads = 32 rows x 8 words
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
```

Notes on deliberate details:

- `__ldg(reinterpret_cast<const __half*>(...) + i)` — the `__half __ldg(const __half*)`
  overload exists in the installed CUDA 13.3 (cuda_fp16.hpp:1881), read-only-cache path,
  8 threads of a B-row hitting the same u16 coalesce to one transaction.
- Dequant drops v21's dead `if (n < NT)` guard (256 = 32×8 exactly; assumption baked in).
- Shared-declaration order is byte-identical to v21 so the (working, known-good-build)
  static-smem layout — including Braw's 16 B cp.async alignment — carries over unchanged.
- cols%1024 ⇒ ksteps = cols/64 is a multiple of 16 ⇒ pipeline warm-up/tail always well-formed,
  and `groups` even ⇒ `groups>>1` scale stride exact.
- Store guard matches fp8.cu:182-184 exactly; callers keep 64-row buffers, so tiles
  straddling T write harmless scratch rows (never read: pf_* consumers index t < T).

## 3. Header declarations — `include/insignia_layout.cuh`

Add after the `mxfp4_gemm_mlx_i4` declaration (line 67):

```cpp
void mxfp4_gemm_v21_i4(const uint32_t *weights, const uint16_t *scales, const void *x16, float *y, int rows, int cols, int T, cudaStream_t stream = nullptr);
void mxfp4_gemm_ab_i4(const uint32_t *wa, const uint16_t *sa, const uint32_t *wb, const uint16_t *sb, const void *x16, float *ya, float *yb, int T, int cols, cudaStream_t stream = nullptr);
```

## 4. Call-site diffs — `src/decode.cu`

### 4.1 `linear_batch` (line 33): route INSIG4 through the pipelined GEMM

Replace the `if(m.insig4){...}` arm (line 35) with the same zero-padded bf16 staging the
e8m0 arm already does (`pf_bf16` is 64×12288×2 B — large enough for every call site):

```cpp
void Qwen35Decode::linear_batch(const std::string&base,const float*in,float*out,int T){
 auto m=w_.matrix(base);
 if(m.insig4){  // pipelined GEMM: A staged as zero-padded bf16 (same contract as the e8m0 path)
  if(T<64)cudaMemsetAsync((char*)x_.pf_bf16+size_t(T)*m.cols*2,0,size_t(64-T)*m.cols*2,x_.stream);  // tail rows must read as zero
  f32_to_bf16(in,x_.pf_bf16,size_t(T)*m.cols,x_.stream);
  mxfp4_gemm_v21_i4((const uint32_t*)m.weight.data,(const uint16_t*)m.scales.data,x_.pf_bf16,out,m.rows,m.cols,T,x_.stream);
 }
 else{  // pipelined GEMM: A staged as zero-padded bf16
  if(T<64)cudaMemsetAsync((char*)x_.pf_bf16+size_t(T)*m.cols*2,0,size_t(64-T)*m.cols*2,x_.stream);  // tail rows must read as zero
  f32_to_bf16(in,x_.pf_bf16,size_t(T)*m.cols,x_.stream);
  mxfp4_gemm_v21((const uint32_t*)m.weight.data,(const uint8_t*)m.scales.data,x_.pf_bf16,out,m.rows,m.cols,T,x_.stream);
 }
 w_.release(base);}
```

### 4.2 The a/b per-token loop (lines 71-73): ONE launch for both tensors

Replace the whole `else{...}` arm that follows `if(pair){...}` in the delta-layer branch:

```cpp
   else{linear_batch(a+".in_proj_qkv",x_.pf_n,x_.pf_qkv,T);linear_batch(a+".in_proj_z",x_.pf_n,x_.pf_z,T);
    {auto ma=w_.matrix(a+".in_proj_a"),mb=w_.matrix(a+".in_proj_b");
     if(ma.insig4){  // BOTH 32-row projections in ONE launch (block0=a, block1=b): kills 2*T-2 launches
      if(T<64)cudaMemsetAsync((char*)x_.pf_bf16+size_t(T)*ma.cols*2,0,size_t(64-T)*ma.cols*2,x_.stream);  // tail rows must read as zero
      f32_to_bf16(x_.pf_n,x_.pf_bf16,size_t(T)*ma.cols,x_.stream);
      mxfp4_gemm_ab_i4((const uint32_t*)ma.weight.data,(const uint16_t*)ma.scales.data,(const uint32_t*)mb.weight.data,(const uint16_t*)mb.scales.data,x_.pf_bf16,x_.pf_a,x_.pf_b,T,ma.cols,x_.stream);
     }else{  // e8m0 twin: two 1-block v21 GEMMs (was T x 2 GEMV launches)
      if(T<64)cudaMemsetAsync((char*)x_.pf_bf16+size_t(T)*ma.cols*2,0,size_t(64-T)*ma.cols*2,x_.stream);
      f32_to_bf16(x_.pf_n,x_.pf_bf16,size_t(T)*ma.cols,x_.stream);
      mxfp4_gemm_v21((const uint32_t*)ma.weight.data,(const uint8_t*)ma.scales.data,x_.pf_bf16,x_.pf_a,ma.rows,ma.cols,T,x_.stream);
      mxfp4_gemm_v21((const uint32_t*)mb.weight.data,(const uint8_t*)mb.scales.data,x_.pf_bf16,x_.pf_b,mb.rows,mb.cols,T,x_.stream);
     }
     w_.release(a+".in_proj_a");w_.release(a+".in_proj_b");}
```

Wiring notes:

- The T==2 pair path (decode.cu:66-70, `mxfp4_gemv_ab2_q8_i4`) is untouched; so is the
  graph-captured spec step. The new path only fires for T != 2 prefill chunks.
- `ma.insig4 ⇒ mb.insig4` follows the existing convention (decode.cu:68 checks only ma).
- The ab branch re-stages `pf_n` even though `linear_batch(in_proj_qkv)` staged the identical
  [64,4096] bf16 two calls earlier (in_proj_qkv/z/a/b all have cols=4096). Dropping the
  memset+convert there would save ~1.5 MB of traffic (~3 µs/layer) but couples the ab call
  to a staging side-effect two calls away — kept explicit; dedup only after a measurement
  says those 3 µs matter.
- All staging ops (`cudaMemsetAsync`, `f32_to_bf16`, GEMM) are graph-capturable; no host syncs
  introduced.
- The spec pair lm_head/GEMV family is unchanged — decode-path routing (decode.cu:31-32)
  never reaches `linear_batch`.

---

## 5. Expected speedup — the math

### 5.1 The a/b monster (the point of §4.2)

Current path, T=64 chunk (all numbers per 64-token prefill chunk):

- Launches: 24 delta layers × 64 tokens × 2 tensors = **3072 × `mxfp4_gemv_v2_i4`**,
  rows=32 ⇒ grid=(32+7)>>3 = **4 blocks on 56 SMs** (7% of the machine), 69.6 KB weights
  and 16 KB x-smem staging per launch.
- Cost/launch: ~2-3 µs kernel (latency-bound, 4 blocks) + ~1.5-2 µs in-stream submit gap
  (WDDM, no graph on the prefill path) ⇒ ~3.5-5 µs ⇒ **3072 × 3.5-5 µs ≈ 10.8-15.4 ms**
  (matches the audit's on-record 9-12 ms estimate; ±WDDM weather).

New path, same chunk:

- 24 × [1 × `f32_to_bf16` (64×4096: 1.0 MB R + 0.5 MB W ⇒ ~3-4 µs) + 1 × `mxfp4_gemm_ab_i4`]
  + 24 × memset (zero bytes at T=64 — skipped).
- `mxfp4_gemm_ab_i4` internals: B = 64 rows × 4096 × 0.53125 B/elt = **139.3 KB**
  (131.1 KB nibbles + 8.2 KB fp16 scales — the "131 KB L2-resident" figure); A = 512 KB
  bf16 read by both blocks (block 1 hits L2) ⇒ ~1.16 MB moved; bandwidth floor at the full
  504 GB/s = 2.3 µs — but grid=2 ⇒ 2 SMs, so the honest bound is per-SM issue/compute:
  33.6 MFLOP at ~1.3 TF/SM × 2 ≈ **~13-30 µs incl. launch** per layer.
- Chunk total: 24 × (4 + 20) µs ≈ **0.5-0.8 ms**. The mission's "~0.1 ms compute" is the
  24-layer aggregate bandwidth floor (24 × 1.16 MB = 27.8 MB / 504 GB/s ≈ 55 µs) — reachable
  only if the grid is widened (2-tiles-per-block or fusing a+b rows into one 64-row tile);
  at 24 × ~25 µs the path is already ~2% of the chunk, so stop here.

**Net: 9-12 ms → ~0.5-0.8 ms ≈ 12-20× on the a/b path; every 64-token chunk saves ~8.5-11 ms.
This routing change alone is worth more than the GEMM pipeline itself** (audit §3.2 verdict
confirmed).

### 5.2 The general prefill GEMMs (§4.1)

200 GEMM calls/chunk move ≈ 2.0 GB of INSIG4 weights (24×58.9 MB delta + 8×55.8 MB attn +
lm_head is GEMV) ⇒ DRAM floor ~4 ms. The v1 clone spends it poorly: syncs between stages,
fp32 A staged by only 64/256 threads, dequant on the critical path, unhidden global latency
— plausibly 2-3× over floor (~8-12 ms of GEMM time). v21_i4 hides global latency behind
wmma and stages with all 256 threads: audit §3.3's per-GEMM estimate ~2× (250-350 µs →
150-180 µs on a 4096×12288 down_proj at T=64, compute-bound at the wmma f32-acc rate).
Chunk-level: expect **−2 to −4 ms** on top of §5.1, i.e. the audit's "prefill chunk −30-50%"
overall.

### 5.3 What does NOT change

Decode/spec single- and pair-token paths (GEMV family, 554/610 launches, 4.99 GB/step at
~378 GB/s) are untouched by this port; their levers are audit items #3-#5.

---

## 6. Test plan (wiring is yours; both targets already link `src\gemm.cu`)

`build/test-i4.bat` compiles `src\test_i4.cu + src\mxfp4_i4.cu + src\gemm.cu` and
`build/nll.bat` compiles the full model including both — **no .bat changes needed**, only
code additions.

1. **Unit parity in `src/test_i4.cu`** (new section 5, reuse the existing wref/w/s16
   quantization at lines 26-51 — rows=8192, cols=4096):
   - For T ∈ {1, 3, 17, 64}: build xT [T,cols]; pack bf16 `x16[64][cols]` on host
     (`__float2bfloat16`, rows ≥ T zeroed — this also tests the zero-pad contract);
     `mxfp4_gemm_v21_i4(dw, ds, dx16, dyT, rows, cols, T)`; compare y rows [0,T) vs the
     f64 ref: gates `max_rel ≤ 2e-2` (bf16 A adds ≤2^-8 rel per element; the existing
     gemm/gemv gates in this file use 2e-2) and **cos ≥ 0.99999**.
   - Cross-check vs `mxfp4_gemm_mlx_i4` on identical inputs: cos ≥ 0.999999 (only fp32
     accumulation order differs).
   - ab kernel: wa/wb = rows 0..31 / 32..63 of the same matrix, same x16; check
     `ya[t*32+r] ≈ xT[t]·wref[r]` and `yb[t*32+r] ≈ xT[t]·wref[32+r]` for t < T, same gates.
   - Dim gates throw: rows=33, cols=4032, T=65.
2. **End-to-end**: `build/nll.bat` NLL vs the mlx_i4 baseline — **delta < 0.01** (mission
   gate); plus `tools/reference_multistep_i4.py` worst_layer_cos ≥ 0.999 (audit gate; today
   0.9998+).
3. **Bench (audit §2.4 protocol — cold-L2 only)**: extend `src/bench_gemm.cu` with the
   `l2_sweep` flush between reps; measure (a) mlx_i4 vs v21_i4 on [8192,4096] and
   [12288,4096] at T=64, (b) the ab kernel vs the 128-launch GEMV loop at T=64, cols=4096.
   Record in build/*.log so the next audit has on-disk GEMM numbers (today there are none).

Risk register (small): Braw cp.async 16 B alignment inherited verbatim from the proven v21
build (same declaration order); guarded stores write 16-row tiles straddling T (all pf_*
buffers are 64-row; test buffers must be too — the existing yT already is); if parity ever
regresses, the only numeric difference vs today is bf16 A — same as the e8m0 path, so
suspect staging first (memset tail, stale pf_bf16 rows ≥ T for cols-mismatched reuse).
