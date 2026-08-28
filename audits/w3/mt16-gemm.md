# mt16: 16-row-A-tile GEMM for small-T spec verify (w3)

Design report for `mxfp4_gemm_v21_i4_mt16` (INSIG4), `mxfp4_gemm_v21_mt16` (MLX E8M0 twin,
needed because the production 9B checkpoint is MLX), and `fp8_gemm_mt16` (27B FP8).
Read of AGENTS.md, spec-deepen.md §3/F4, trtllm-fp8-deep.md §1-3/§7, src/gemm.cu and
src/fp8.cu in full. Read-only audit; nothing here is built or committed.

## 0. TL;DR

1. v21/fp8_gemm hardcode a 64-row A tile (`As[2][64][…]`, gemm.cu:213 / fp8.cu:112): every
   T∈[2,64] pays 64 rows of wmma. fp8_gemm's `if (wm*16 < T)` store guard (fp8.cu:184)
   skips dead *stores*, not the dead *MMA* — the waste is compute-issue, only a 16-row A
   tile removes it.
2. mt16 restructure: 256 threads = 8 warps, A tile = 16 rows (the single wmma m16), warps
   split N: warp w owns columns [16w,16w+16); block tile 16×128 (NT=128), KT=64 per step,
   cp.async double-buffer on A + raw nibbles, Bs single-buffered (v21's second Bs buffer
   was dead weight — dequant always sits between two `__syncthreads`).
3. NT=128 == the FP8 128×128 N-scale period → exactly ONE bf16 scale scalar per (block,
   k-step-pair), TRT-style broadcast (trtllm-fp8-deep §1.3). i4 scale = one f16 per KT=64
   step (super-group); MLX = two E8M0 bytes per step. Same shell, three scale lines.
4. Smem: mxfp4 31.5 KB (3 CTAs/SM), fp8 38.5 KB (2 CTAs/SM) — both under the 48 KB static
   limit, no opt-in needed. Masked epilogue: accumulator stages through reused Bs as
   16×128 f32, guarded copy writes only rows t0+m < T, so y may be sized exactly [T,rows]
   (lm_head logits — this is required, not cosmetic: a full-tile store would OOB).
5. Math answer: yes, m16 alone hits the BW floor at T=2. Compute at T_pad=16 is 16/64 of
   today: 34.8 µs vs 176.9 µs floor on 17408×5120; 19.6 µs vs 53.1 µs on 9B gate —
   2.7-5.1× margin at every useful shape. The 8× pad waste vs T=2 rides tensor pipes that
   are >2.7× idle anyway. T=2/4/8/16 are literally the same kernel time (BW-bound).
6. Today's 64-row kernel is compute-bound on 9B MXFP4 (78.6 µs vs 53.1-60.8 µs floor —
   the 78-vs-59 number) and only 1.27× from compute-bound on 27B FP8; mt16 deletes the
   compute term everywhere.
7. CTA count = rows/128: lm_head 1940 and gate 136 saturate; rows=4096 → 32 CTAs is the
   bench gate (13.75 GB/s per CTA needed — marginal, ≈ wash vs v21); rows≤2048 starved →
   keep pair at T=2 and v21 at T≥3 there (their absolute padding waste is tiny).
8. Dispatch rule: mt16 iff T∈[2,16] ∧ rows≥4096. T∈[17,64] stays v21 — grid.y=ceil(T/16)
   re-reads weights per m-tile, so 2 passes at T=17..32 always lose to v21's one pass.
9. `argmax_rows` (new): batched T-row argmax, atomic-free two-stage, 2 launches, no
   memset (replaces 3T launches of argmax_fast; T=16: 50→2).
10. Migration: swap decode.cu:94-101 lm_head (pair + 2×argmax_fast, or last-row-only GEMV
    at T≠2) → f32_to_bf16_pad + mt16 + argmax_rows; widen logits to 248320×16 f32 (+14 MB)
    and am_scratch to 8 KB; gates = host-double rel-err ≤2e-2, T=16 cross-check vs v21,
    pair-parity cos ≥0.9999, bench ≥420 GB/s on lm_head and spec step ≤13.1 ms.

---

## 1. The problem, precisely

`mxfp4_gemm_v21_kernel` (gemm.cu:210-291) and `fp8_gemm_kernel` (fp8.cu:109-185) share the
v2.1 shell: 256 threads, 64×32 block tile, 8 warps as 4×2 (wm=warp>>1 picks the m16 row
tile, wn=warp&1 the n16), KT=64, cp.async double buffer, B dequant through a register→smem
stage. The A tile is `As[2][64][72]` regardless of T: the caller (decode.cu:36-39
`linear_batch`) zero-pads `pf_bf16` to 64 rows, and all 4 m16 row tiles go through
`mma_sync` even at T=2. Weight traffic is already one pass per output tile (T-independent)
— so at small T the kernel is pure-compute-wasteful:

- 9B gate_proj 12288×4096 (MXFP4, 0.53125 B/elem): weights+scales 26.74 MB → 53.1 µs at
  504 GB/s, 60.8 µs at the 430-450 GB/s real-streaming band. 64-row compute =
  2·64·12288·4096 = 6.44 GFLOP → 78.6 µs at 82 TF (the sustained bf16-fp32acc band;
  71 TF dense-peak conservative → 90.7 µs). **Compute-bound by 1.48×** — spec-deepen's
  78 µs vs 59 µs, confirmed.
- 27B gate 17408×5120 (FP8, 1 B/elem): 89.14 MB → 176.9 µs floor. 64-row compute 11.41
  GFLOP → 139.2 µs @82 / 160.7 @71 — nominally BW-bound but with only a 1.27× margin; any
  tensor-rate dip below ~64.5 TF sustained makes it compute-bound.
- 27B lm_head 248320×5120: 1271.5 MB → 2522.8 µs floor; 64-row compute 162.8 GFLOP →
  1984.9 µs @82. Same 1.27× knife edge.

A T=2-specific fix already exists (the dp4a pair kernels, one weight pass, hard
memory-bound at ~430 GB/s), but it caps at T=2; spec-deepen §3 wants one kernel for
T=2..16 that ties the pair at T=2 and is free beyond. The wmma m16 granularity is the
floor: bf16 wmma has no m8 shape, so 16 rows is the smallest honest MMA tile.

## 2. Tiling design

Per block (256 threads, 8 warps):

- **A tile: 16 rows × 64 cols** (bf16, from x16 with the SAME zero-pad contract as today —
  rows [T, 16·ceil(T/16)) zero — pad 16, not 64).
- **B tile: 128 rows (NT=128) × 64 cols** per K step; warps split N, one n16 each:
  `warp w computes Y[t0..t0+16, n0+16w .. n0+16w+16]` — exactly one 16×16 f32 accumulator
  per warp, 4 `mma_sync(16,16,16)` per warp per K step (KT=64 = 4×k16).
- **Why warps on N, not M**: with A=16 rows there is exactly one m16 tile; any warp mapped
  to another m-tile would be dead (that is the v21 disease in miniature). 8×n16 forces
  NT=128, which is also the minimum N-footprint keeping 8 warps busy — and it aligns the
  block with the FP8 128×128 N-scale period, so each fp8 dequant pass reads ONE scale
  scalar (uniform `__ldg` broadcast; TRT's `tXrSFB(0)` trick, trtllm-fp8-deep §1.3).
- **Pipeline** (v2.1 shape, verbatim discipline): 2-stage cp.async on `As[2][16][72]` +
  `Braw[2]` raw nibbles/bytes; `Bs[1][128][72]` dequant target is SINGLE-buffered — in v21
  the second Bs buffer is never overlapped (dequant(kb) runs between the wait-sync and the
  mma-sync; Bs[buf^1] is written only next iteration after another sync), so dropping it
  saves 18.4 KB and buys the third CTA per SM. Loop body per kb: prefetch(kb+1)+commit →
  wait_group 1 (0 at tail) → sync → dequant(kb) → sync → 4×(load A/B frag, mma) → sync.
- **K step**: KT=64 = two MLX E8M0 groups | one INSIG4 f16 super-group | half an FP8
  128-col scale block. Identical to v21/fp8_gemm — all scale indexing proven there.
- **Masked epilogue** (the point of "t < T rows only"): after the K loop each warp
  `store_matrix_sync` its m16n16 tile into a 16×128 f32 staging area aliased over Bs
  (8 KB ≤ 18.4 KB; disjoint per warp, one sync after), then all 256 threads copy with a
  row guard `if (t0 + m < T) y[(t0+m)*rows + n0 + n] = stage[i]` (8 elements per thread).
  Full 16-row tiles (t0+16 ≤ T) take the same path — uniform, and the staging cost is
  noise next to a ≥53 µs memory floor. A plain wmma store cannot be predicated per row;
  without staging, a T%16≠0 last tile would write up to 15 rows past the end of a
  [T,rows]-sized y (lm_head logits) — staging is a correctness requirement, not a style
  choice.
- **Grid**: `dim3(rows/128, ceil(T/16))`. T=2..16 → grid.y=1, one A tile, the only waste
  is the ≤14 zero rows of MMA inside it (bounded, and free per §3). T up to 64 supported
  for API completeness/parity tests; the dispatch rule keeps it at T≤16 (§5).

Shared memory (static, no opt-in; sm_89 allows 48 KB static / 100 KB per SM):

| kernel | As | Braw | Bs | lut | total | CTAs/SM |
|---|---|---|---|---|---|---|
| mxfp4 mt16 | 2·16·72·2 = 4.6 KB | 2·128·8·4 = 8.2 KB | 128·72·2 = 18.4 KB | 1.0 KB | **31.5 KB** | 3 |
| fp8 mt16 | 4.6 KB | 2·128·64 = 16.4 KB | 18.4 KB | — | **38.5 KB** | 2 |

Register budget per thread: 8 f32 accumulator + 4 A-frags + 4 B-frags (16+16 regs of
bf16 pairs) + addressing ≈ 80 regs — same class as v21, fine under `__launch_bounds__(256)`.

## 3. Kernel 1 — `mxfp4_gemm_v21_mt16` (MLX) / `mxfp4_gemm_v21_i4_mt16` (INSIG4)

One template, two instantiations; the ONLY format difference is the scale fetch (E8M0 per
32-group with per-word half select vs f16 per 64 super-group, one per K step). Paste into
`src/gemm.cu` (reuses `cp_async16`/`cp_async_commit`/`cp_async_wait_*` from gemm.cu:11-17);
add `#include <type_traits>` at the top and the two declarations to
`include/insignia_layout.cuh`.

```cpp
namespace insignia {
using namespace nvcuda;

// Small-T MXFP4 GEMM (mt16): Y[T,rows] = X16[T,cols] * W[rows,cols]^T, A tile = 16 rows.
// v21's 64-row A tile makes every T<=16 pay 64 rows of wmma (compute-bound on 9B:
// 78us vs 53-61us floor); mt16 pays exactly one m16 per 16 tokens. 256 threads = 8 warps,
// all on the single m16 A tile, split along N: warp w owns cols [16w,16w+16); block tile
// 16x128, K steps of 64 cols. cp.async double-buffers A + raw B nibbles; Bs is
// single-buffered (v21's second Bs buffer was never overlapped). Masked epilogue stages
// the tile through smem so y may be sized exactly [T,rows].
// Contract: x16 (bf16, row stride cols) zero-padded to 16*ceil(T/16) rows — SAME rule as
// v21, 16 not 64. Requires rows%128==0, cols%64==0, 2<=T<=64 (grid.y = ceil(T/16); the
// useful range is T<=16 — above that v21's single 64-row weight pass wins).
template <typename ScaleT>
__global__ __launch_bounds__(256) void mxfp4_gemm_mt16_kernel(
        const uint32_t *__restrict__ weights, const ScaleT *__restrict__ scales,
        const __nv_bfloat16 *__restrict__ x16, float *__restrict__ y,
        int rows, int cols, int T) {
    constexpr int KT = 64, NT = 128, MPT = 16;
    constexpr int APAD = 8, BPAD = 8;
    __shared__ __nv_bfloat16 As[2][MPT][KT + APAD];              // 4.6 KB
    __shared__ uint32_t Braw[2][NT][8];                          // 8.2 KB (packed nibbles)
    __shared__ __align__(16) __nv_bfloat16 Bs[NT][KT + BPAD];    // 18.4 KB
    __shared__ uint32_t lut[256];                                // bf16 E2M1 pair LUT
    {
        const unsigned tbl[16] = {0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
                                  0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0};
        lut[threadIdx.x] = tbl[threadIdx.x & 15] | (tbl[(threadIdx.x >> 4) & 15] << 16);
    }
    const int groups = cols >> 5;
    const int n0 = blockIdx.x * NT;
    const int t0 = blockIdx.y * MPT;
    const int tid = threadIdx.x;

    auto prefetch = [&](int kb, int buf) {       // cp.async: A tile + B nibbles for step kb
        const int k = kb * KT;
        for (int i = tid; i < MPT * (KT >> 3); i += 256) {       // 16 rows x 8 chunks of 16B
            const int m = i / (KT >> 3), c8 = i % (KT >> 3);
            cp_async16(&As[buf][m][c8 * 8],
                       reinterpret_cast<const char *>(x16) + (size_t(t0 + m) * cols + k) * 2 + c8 * 16);
        }
        {   // B: NT rows x 32B of nibbles = 256 chunks, one per thread
            const int n = tid >> 1, half = tid & 1;
            cp_async16(&Braw[buf][n][half * 4],
                       reinterpret_cast<const char *>(weights + size_t(n0 + n) * groups * 4 + (k >> 3) + half * 4));
        }
    };
    auto dequant = [&](int kb, int buf) {        // Braw -> Bs, group scale folded in regs
        for (int i = tid; i < NT * 8; i += 256) {                // 1024 words, 4 per thread
            const int n = i >> 3, w = i & 7;
            const uint32_t word = Braw[buf][n][w];
            float sc;
            if constexpr (std::is_same_v<ScaleT, uint8_t>)       // MLX E8M0: byte per 32-group
                sc = __int_as_float(uint32_t(__ldg(scales + size_t(n0 + n) * groups + kb * 2 + (w >> 2))) << 23);
            else                                                // INSIG4 f16: half per 64 super-group
                sc = __half2float(*reinterpret_cast<const __half *>(scales + size_t(n0 + n) * (groups >> 1) + kb));
            __nv_bfloat16 out[8];
            #pragma unroll
            for (int byt = 0; byt < 4; byt++) {
                const uint32_t pair = lut[(word >> (byt * 8)) & 0xff];
                const float lo = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&pair));
                const float hi = __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(&pair)[1]);
                out[byt * 2 + 0] = __float2bfloat16(lo * sc);
                out[byt * 2 + 1] = __float2bfloat16(hi * sc);
            }
            *reinterpret_cast<uint4 *>(&Bs[n][w << 3]) = *reinterpret_cast<const uint4 *>(out);
        }
    };

    const int ksteps = cols / KT;
    prefetch(0, 0);
    cp_async_commit();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    const int warp = tid >> 5;                   // warp w == n-tile [16w, 16w+16)

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
            wmma::load_matrix_sync(af[kh], &As[buf][0][kh * 16], KT + APAD);   // single m16 tile: row 0
            wmma::load_matrix_sync(bf[kh], &Bs[warp * 16][kh * 16], KT + BPAD);
            wmma::mma_sync(acc, af[kh], bf[kh], acc);
        }
        __syncthreads();
    }
    // Masked epilogue: stage the 16x128 tile through Bs (reused as f32; warps write
    // disjoint n16 columns, ldm = NT) then copy only rows t0+m < T.
    float *stage = reinterpret_cast<float *>(&Bs[0][0]);        // 16*128 f32 = 8 KB <= Bs
    wmma::store_matrix_sync(stage + warp * 16, acc, NT, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < MPT * NT; i += 256) {
        const int m = i >> 7, n = i & (NT - 1);
        if (t0 + m < T) y[size_t(t0 + m) * rows + n0 + n] = stage[i];
    }
}

void mxfp4_gemm_v21_mt16(const uint32_t *weights, const uint8_t *scales, const void *x16,
                         float *y, int rows, int cols, int T, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 63) || T < 2 || T > 64 || (rows & 127))
        throw std::runtime_error("insignia: bad mt16 GEMM dims rows=" + std::to_string(rows) +
                                 " cols=" + std::to_string(cols) + " T=" + std::to_string(T));
    mxfp4_gemm_mt16_kernel<uint8_t><<<dim3(rows >> 7, (T + 15) >> 4), 256, 0, stream>>>(
        weights, scales, (const __nv_bfloat16 *)x16, y, rows, cols, T);
}

void mxfp4_gemm_v21_i4_mt16(const uint32_t *weights, const uint16_t *scales, const void *x16,
                            float *y, int rows, int cols, int T, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 63) || T < 2 || T > 64 || (rows & 127))
        throw std::runtime_error("insignia: bad mt16 GEMM dims rows=" + std::to_string(rows) +
                                 " cols=" + std::to_string(cols) + " T=" + std::to_string(T));
    mxfp4_gemm_mt16_kernel<uint16_t><<<dim3(rows >> 7, (T + 15) >> 4), 256, 0, stream>>>(
        weights, scales, (const __nv_bfloat16 *)x16, y, rows, cols, T);
}

// f32 -> bf16 with zero tail: converts n real elements and zero-fills [n, pad) in ONE
// launch — replaces linear_batch's separate memset + convert (2 launches, 4x the bytes at
// T=2) and guarantees the mt16/v21 zero-pad contract per call. Note the pad must be
// rewritten per GEMM: pf_bf16 is a flat buffer and the cols=4096 pad region overlaps the
// cols=12288 live region of another view.
__global__ void f32_to_bf16_pad_kernel(const float *__restrict__ x, __nv_bfloat16 *__restrict__ y, int n, int pad) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pad) y[i] = i < n ? __float2bfloat16(x[i]) : __float2bfloat16(0.f);
}
void f32_to_bf16_pad(const float *x, void *y, int n, int pad, cudaStream_t stream) {
    f32_to_bf16_pad_kernel<<<(pad + 255) / 256, 256, 0, stream>>>(x, (__nv_bfloat16 *)y, n, pad);
}
}
```

Scale-index derivation (checked against existing kernels): MLX — v21's own line is
`scales[(n0+n)*groups + kb*2 + (w>>2)]` (gemm.cu:249), words 0-3 = first 32-group, 4-7 =
second. INSIG4 — a KT=64 step covers exactly one 64-element super-group, so the index is
`(n0+n)*(groups>>1) + kb` for every word (consistent with `i4_scale(s, g)=s[g>>1]`,
mxfp4_i4.cu:11-13: g=2kb+h → s[kb] for both halves h).

## 4. Kernel 2 — `fp8_gemm_mt16` (27B, e4m3 + bf16 128×128 block scales)

Paste into `src/fp8.cu` (inline cp.async asm matches that file's local style; `e4m3x2`
comes from insignia_fp8.cuh). W8A16 discipline per trtllm-fp8-deep §4.3: activations stay
bf16, weights dequant exact to bf16 in registers — no activation quant at T≤16.

```cpp
// Small-T FP8 blockwise GEMM (mt16): Y[T,rows] = X16[T,cols] * W[rows,cols]^T, W e4m3,
// scales bf16 [rows/128][cols/128]. Same 16x128 mt16 tiling as the MXFP4 twin. NT=128 ==
// the N scale period, so every dequant pass reads ONE scalar: scales[blockIdx.x*kblocks
// + (kb>>1)] (n0 is a multiple of 128; n>>7 == 0 for the whole tile). Requires
// rows%128==0, cols%128==0, 2<=T<=64. x16 zero-padded to 16*ceil(T/16) rows.
__global__ __launch_bounds__(256) void fp8_gemm_mt16_kernel(
        const uint8_t *__restrict__ weights, const uint16_t *__restrict__ scales,
        const __nv_bfloat16 *__restrict__ x16, float *__restrict__ y,
        int rows, int cols, int T) {
    constexpr int KT = 64, NT = 128, MPT = 16;
    constexpr int APAD = 8, BPAD = 8;
    __shared__ __nv_bfloat16 As[2][MPT][KT + APAD];              // 4.6 KB
    __shared__ uint8_t Braw[2][NT][KT];                          // 16.4 KB raw e4m3
    __shared__ __align__(16) __nv_bfloat16 Bs[NT][KT + BPAD];    // 18.4 KB
    const int kblocks = cols >> 7;
    const int n0 = blockIdx.x * NT;
    const int t0 = blockIdx.y * MPT;
    const int tid = threadIdx.x;

    auto cp16 = [](void *smem, const char *global) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"((unsigned)__cvta_generic_to_shared(smem)), "l"(global));
    };
    auto prefetch = [&](int kb, int buf) {
        const int k = kb * KT;
        for (int i = tid; i < MPT * (KT >> 3); i += 256) {       // A: 16 rows x 8 chunks
            const int m = i / (KT >> 3), c8 = i % (KT >> 3);
            cp16(&As[buf][m][c8 * 8],
                 reinterpret_cast<const char *>(x16) + (size_t(t0 + m) * cols + k) * 2 + c8 * 16);
        }
        for (int i = tid; i < NT * 4; i += 256) {                // B: 128 rows x 64B = 4 chunks
            const int n = i >> 2, c = i & 3;
            cp16(&Braw[buf][n][c << 4], weights + size_t(n0 + n) * cols + k + (c << 4));
        }
    };
    auto dequant = [&](int kb, int buf) {        // 128 rows x 2 threads, 32 e4m3 each
        const int n = tid >> 1, w = tid & 1;
        const float sc = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(
                scales + size_t(blockIdx.x) * kblocks + (kb >> 1)));   // ONE scalar per pass
        __nv_bfloat16 out[32];
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const uint4 p = *reinterpret_cast<const uint4 *>(&Braw[buf][n][(w << 5) + (half << 4)]);
            const uint32_t pw[4] = {p.x, p.y, p.z, p.w};
            #pragma unroll
            for (int wi = 0; wi < 4; wi++) {
                const float2 a = e4m3x2(pw[wi]), b2 = e4m3x2(pw[wi] >> 8);
                out[half * 16 + wi * 4 + 0] = __float2bfloat16(a.x * sc);
                out[half * 16 + wi * 4 + 1] = __float2bfloat16(b2.x * sc);
                out[half * 16 + wi * 4 + 2] = __float2bfloat16(a.y * sc);
                out[half * 16 + wi * 4 + 3] = __float2bfloat16(b2.y * sc);
            }
        }
        *reinterpret_cast<uint4 *>(&Bs[n][w << 5])        = *reinterpret_cast<const uint4 *>(&out[0]);
        *reinterpret_cast<uint4 *>(&Bs[n][(w << 5) + 16]) = *reinterpret_cast<const uint4 *>(&out[16]);
    };

    const int ksteps = cols / KT;
    prefetch(0, 0);
    asm volatile("cp.async.commit_group;\n");

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    const int warp = tid >> 5;

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
            wmma::load_matrix_sync(af[kh], &As[buf][0][kh * 16], KT + APAD);
            wmma::load_matrix_sync(bf[kh], &Bs[warp * 16][kh * 16], KT + BPAD);
            wmma::mma_sync(acc, af[kh], bf[kh], acc);
        }
        __syncthreads();
    }
    float *stage = reinterpret_cast<float *>(&Bs[0][0]);
    wmma::store_matrix_sync(stage + warp * 16, acc, NT, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < MPT * NT; i += 256) {
        const int m = i >> 7, n = i & (NT - 1);
        if (t0 + m < T) y[size_t(t0 + m) * rows + n0 + n] = stage[i];
    }
}

void fp8_gemm_mt16(const uint8_t *weights, const uint16_t *scales, const void *x16,
                   float *y, int rows, int cols, int T, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 127) || T < 2 || T > 64 || (rows & 127))
        throw std::runtime_error("insignia: bad mt16 GEMM dims rows=" + std::to_string(rows) +
                                 " cols=" + std::to_string(cols) + " T=" + std::to_string(T));
    fp8_gemm_mt16_kernel<<<dim3(rows >> 7, (T + 15) >> 4), 256, 0, stream>>>(
        weights, scales, (const __nv_bfloat16 *)x16, y, rows, cols, T);
}
```

Declaration for `include/insignia_fp8.cuh`:
`void fp8_gemm_mt16(const uint8_t*, const uint16_t*, const void*, float*, int, int, int, cudaStream_t = nullptr);`

Dim checks (`rows%128`, `cols%64` mxfp4 / `cols%128` fp8, `2<=T<=64`) are enforced at the
launcher — no T%anything requirement, grid.y absorbs any tail. Every current call site is
clean: 9B rows ∈ {1024, 4096, 8192, 12288, 248320}, 27B rows per trtllm-fp8-deep §1.1 all
multiples of 128; `in_proj_a/b` (32 rows) never route through the batch GEMM (they stay on
the per-row GEMV loop, decode.cu:72-73) — if that ever changes the throw fires loudly.

## 5. Performance model

Constants: 504 GB/s peak; 430-450 GB/s real streaming (pair measured 430 on lm_head shapes;
Marlin-class lands 80-90% of peak, trtllm-fp8-deep §7.2). bf16-wmma-fp32acc: 82 TF
sustained band / 71 TF dense-peak conservative. Compute = 2·T_pad·R·C with
T_pad = 16·ceil(T/16); memory = weight bytes once (T-independent — the whole point).
Activation/logits traffic is a ≤3-5% footnote (worst case: 9B lm_head T=16 writes
16·248320·4 = 15.9 MB vs 540.6 MB weights, +2.9%; A re-reads per n-block hit L2 — the
16·cols·2 ≤ 640 KB tile is resident).

**27B gate_proj 17408×5120 (FP8): weights+scales 89.14 MB → floor 176.9 µs @504 / 202.6 @440.**

| T | T_pad | compute @82/@71 µs | binding | predicted | today (64-pad) compute |
|---|---|---|---|---|---|
| 2 | 16 | 34.8 / 40.2 | BW (5.1× margin) | **177-203 µs** | 139.2 / 160.7 |
| 4 | 16 | 34.8 / 40.2 | BW | 177-203 µs | 139.2 / 160.7 |
| 8 | 16 | 34.8 / 40.2 | BW | 177-203 µs | 139.2 / 160.7 |
| 16 | 16 | 34.8 / 40.2 | BW | 177-203 µs | 139.2 / 160.7 |

Today's kernel is only 1.27× under the floor at 82 TF and flips compute-bound below ~64.5
TF sustained; mt16 removes the compute term (margin 5.1×) — identical predicted time for
T=2 through 16 (the T=2 "8× pad waste" is free because those pipes idle either way).

**27B lm_head 248320×5120 (FP8): 1271.5 MB → floor 2522.8 / 2890.6 µs.**

| T | T_pad | compute @82/@71 µs | binding | predicted | today compute |
|---|---|---|---|---|---|
| 2-16 | 16 | 496.2 / 573.1 | BW (5.1×) | **2523-2891 µs** | 1984.9 / 2292.4 |

**9B references (MXFP4, the live spec-verify shapes):**

| shape | bytes | floor @504/@440 | mt16 compute @82/@71 | today (64-pad) @82/@71 | verdict |
|---|---|---|---|---|---|
| gate 12288×4096 | 26.74 MB | 53.1 / 60.8 µs | 19.6 / 22.7 µs | **78.6 / 90.7 (compute-bound)** | mt16 = floored; the 78-vs-59 fix |
| lm_head 248320×4096 | 540.6 MB | 1072.7 / 1228.7 µs | 397.0 / 458.4 µs | **1587.8 / 1833.6 (compute-bound)** | mt16 = floored (spec-deepen's 0.4 ms « 1.3 ms) |

Direct answer to the worked math: "compute 16/64 × 140 µs" is the 27B gate at 82 TF —
16/64·139.2 = 34.8 µs, far under the 176.9 µs floor of that same matrix; on the 9B gate it
is 16/64·78.6 = 19.6 µs vs a 53-61 µs floor. **m16 is enough at T=2 for every shape with
rows ≥ 4096.** No MT=8 (no bf16 wmma m8) and no T=2-special kernel is needed beyond the
existing pair, which stays optimal for the narrow matrices below.

**Occupancy / grid fill** (CTAs = rows/128; per-CTA streaming need at 440 GB/s total =
440·128/rows):

| rows | CTAs | waves @56 SMs | GB/s per CTA | call |
|---|---|---|---|---|
| 248320 | 1940 | 34.6 | 0.23 | lm_head — trivially saturated |
| 17408 | 136 | 2.43 | 3.2 | 27B gate/up — fine |
| 12288 | 96 | 1.71 | 4.6 | 9B gate/up — fine |
| 8192 | 64 | 1.14 | 6.9 | qkv — fine |
| 5120 | 40 | 0.71 | 11.0 | 27B down — ok |
| 4096 | 32 | 0.57 | 13.8 | 9B down/o/z — **bench gate**: ≈ wash vs v21 (v21 4096-row compute 26.2 µs ≈ mt16 at ~320 GB/s effective); keep whichever measures faster |
| 1024 | 8 | 0.14 | 55 | k/v — starved: keep pair at T=2 / v21 at T≥3 (their absolute padding waste is 6.5 µs on a 5 µs floor) |

**Dispatch rule (wire into `linear_batch`): mt16 iff `T∈[2,16] ∧ rows≥4096 ∧ rows%128==0`;
T∈[17,64] → v21** (grid.y re-reads the weight matrix ceil(T/16)× — at T=17..32 that is
2× traffic vs v21's one pass with 2× compute; v21 always wins above 16). T=2 per-layer
linears stay on the pair (`linear2`) until the bench says mt16 ≥ pair per shape; k/v
(1024) stay pair/v21 permanently per the table.

Mission point 3 confirmed: at T≤8 (indeed T≤16) grid.y=1 — there is exactly one m16 tile,
all 8 warps compute useful N-columns, and the only waste is the ≤14 zero rows inside the
single m16 MMA, which §5 shows is free. There are no dead A-tiles to skip.

## 6. `argmax_rows` — batched T-row argmax (lm_head epilogue fusion)

spec-deepen §3: the [T,248320] logits tile wants one winner per row in ONE launch pair;
today's per-row `argmax_fast` = 3T launches (2T kernels + T memsets) — 0.12 ms and 48+
graph nodes at T=16. Paste into `src/qwen_kernels.cu` (reuses the monotone u64 key trick
of argmax_stage1, qwen_kernels.cu:25-63; ties resolve to the smallest index exactly like
argmax_fast — deterministic):

```cpp
// Batched row argmax over x[T, n] (row stride `stride`), atomic-free two-stage.
// Stage 1: grid (BS=64, T) x 512 — each block reduces its chunk of its row to one u64
// key (order bits << 32 | index) with a plain store to scratch[row*64 + b].
// Stage 2: grid (T) x 64 — max over the row's 64 keys, unpack index. No memset needed:
// stage 1 overwrites every slot stage 2 reads.
__global__ void argmax_rows_s1_kernel(const float *__restrict__ x, int n, int stride, unsigned long long *__restrict__ scr) {
    const float *xr = x + size_t(blockIdx.y) * stride;
    float bv = -3.402823466e38F; int bi = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = __ldg(xr + i);
        if (v > bv) { bv = v; bi = i; }
    }
    for (int m = 16; m; m >>= 1) {
        const float v = __shfl_xor_sync(0xffffffff, bv, m);
        const int j = __shfl_xor_sync(0xffffffff, bi, m);
        if (v > bv) { bv = v; bi = j; }
    }
    __shared__ float bv16[16]; __shared__ int bi16[16];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (!lane) { bv16[warp] = bv; bi16[warp] = bi; }
    __syncthreads();
    if (!warp) {
        bv = lane < 16 ? bv16[lane] : -3.402823466e38F;
        bi = lane < 16 ? bi16[lane] : 0;
        for (int m = 16; m; m >>= 1) {
            const float v = __shfl_xor_sync(0xffffffff, bv, m);
            const int j = __shfl_xor_sync(0xffffffff, bi, m);
            if (v > bv) { bv = v; bi = j; }
        }
        if (!lane) {
            uint32_t bits = __float_as_uint(bv);
            bits ^= (uint32_t(int32_t(bits) >> 31)) | 0x80000000u;
            scr[size_t(blockIdx.y) * gridDim.x + blockIdx.x] = (unsigned long long(bits) << 32) | unsigned(bi);
        }
    }
}
__global__ void argmax_rows_s2_kernel(const unsigned long long *__restrict__ scr, int bs, int *__restrict__ out) {
    unsigned long long best = 0;
    for (int b = threadIdx.x; b < bs; b += blockDim.x) best = max(best, scr[size_t(blockIdx.x) * bs + b]);
    __shared__ unsigned long long red[16];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (!lane) red[warp] = best;
    __syncthreads();
    if (!warp) {
        best = lane < 16 ? red[lane] : 0;
        for (int m = 16; m; m >>= 1) best = max(best, __shfl_xor_sync(0xffffffff, best, m));
        if (!lane) out[blockIdx.x] = int(best & 0xffffffffu);
    }
}
void argmax_rows(const float *x, int n, int stride, int T, int *out, unsigned long long *scratch, cudaStream_t s) {
    argmax_rows_s1_kernel<<<dim3(64, T), 512, 0, s>>>(x, n, stride, scratch);
    argmax_rows_s2_kernel<<<T, 64, 0, s>>>(scratch, 64, out);
}
```

Scratch: 64·T u64 (T=16 → 8 KB — widen `am_scratch`, see §7). Winner per row lands in
`out[t]` = the t*_i slots of spec-deepen §2/§4. Launch count 3T+T memsets → 2, zero
memsets; in the captured graph that is ~50 nodes → 2 at T=16.

## 7. Migration — decode.cu swap plan + parity gates

Target: `prefill_chunk_device` lm_head block (decode.cu:93-101). Today: T==2 →
`mxfp4_gemv2_q8_i4`/`mxfp4_gemv2_q8` + 2×`argmax_fast`; T≠2 → LAST-ROW-ONLY GEMV
(`x_.pf_n + (T-1)*4096`) + 1 argmax — T>2 verify currently gets logits for one row only,
which blocks spec-deepen's D≥2 chain. Replace both with the T-general path:

```cpp
{auto m=w_.matrix("language_model.lm_head");
 const int pad=(T+15)&~15;                                       // 16-row A tile pad
 f32_to_bf16_pad(x_.pf_n,x_.pf_bf16,size_t(T)*m.cols,size_t(pad)*m.cols,x_.stream);
 if(m.insig4) mxfp4_gemm_v21_i4_mt16((const uint32_t*)m.weight.data,(const uint16_t*)m.scales.data,x_.pf_bf16,x_.logits,m.rows,m.cols,T,x_.stream);
 else         mxfp4_gemm_v21_mt16  ((const uint32_t*)m.weight.data,(const uint8_t*)m.scales.data,x_.pf_bf16,x_.logits,m.rows,m.cols,T,x_.stream);
 argmax_rows(x_.logits,Qwen35Shape::vocab,Qwen35Shape::vocab,T,x_.tstar_dev,x_.am_scratch,x_.stream);
 w_.release("language_model.lm_head");}
```

Workspace deltas (DecodeWorkspace, decode.cu:14-26): `logits` 248320·2 → 248320·16 f32
(+14 MB; masked store means T rows used, 16-row headroom per spec-deepen §7);
`am_scratch` 8 B → 8 KB (64·16 u64); `tstar_dev` = pos_dev slots [2..2+T-1] per
spec-deepen §5. `capture_spec` must be re-captured once after the alloc change (pointers
move); buffers are stable thereafter, pin-set logic (graph-hazards) untouched. For 27B the
same block uses `fp8_gemm_mt16` once the fp8 lm_head path exists (27B lm_head is bf16-dense
per spec-deepen §6 — that one needs the streaming bf16 GEMM, not this kernel).

`linear_batch` (decode.cu:33-41) second step: route per §5's rule —
`if (T<=16 && rows>=4096 && !(rows&127)) mt16 else v21` (MLX: swap the memset+f32_to_bf16
tail to `f32_to_bf16_pad` with pad 16·ceil(T/16) when going to mt16, keep 64 for v21);
INSIG4 branch moves from the v1-clone `mxfp4_gemm_mlx_i4` (single-buffered, no cp.async)
to `mxfp4_gemm_v21_i4_mt16` for T≤16 — that also retires the "no test touches i4 kernels"
gap (internals.md) since mt16_i4 enters the unit/bench rotation. k/v (1024 rows) and the
a/b 32-row per-row loops stay as-is. T=2 per-layer linears stay on `linear2`/pair until
per-shape benches say otherwise (§5 table).

**Parity gates, in order (all must pass before the pair leaves the spec path):**

1. Unit: host-double reference, bench_gemm.cu pattern — shapes {12288×4096, 248320×4096}
   × T ∈ {2,3,15,16} (plus one T∈[17,32] run to exercise grid.y=2 + masked last tile),
   rel-err ≤ 2e-2, for MLX, INSIG4, and FP8 variants.
2. Cross-tiling: mt16 at T=16 vs `mxfp4_gemm_v21` on identical x16 — must agree to
   bf16-accumulation reorder tolerance (cos ≥ 0.9999); same for fp8 pair.
3. Pair parity at T=2: logits rows vs `mxfp4_gemv2_q8_i4`/`mxfp4_gemv2_q8`, cos ≥ 0.9999
   (dump_multistep harness). NOTE: the pair is dp4a (int8 activation quant); mt16 is
   bf16-exact dequant — closer to the NumPy reference (dp4a drift is a listed parity
   suspect, internals.md). Near-tie tokens may legitimately flip, so the greedy-chain
   regression must compare against the reference stream, not the old pair output.
4. Bench gates: lm_head mt16 ≥ 420 GB/s effective at T=2 (pair = 430); 9B gate ≤ 65 µs
   (floor 60.8); eager spec step ≤ 13.1 ms (not regressed); rows=4096 matrices measured
   both ways before the dispatch keeps/changes them.
5. Only then delete pair from the spec path (keep it in eager diff-testing, per
   spec-deepen §3), and delete the T≠2 last-row-only lm_head branch with it.

## 8. Risks / open items

- **Bs single-buffering** is a deliberate delta from v21 — argued safe in §2 (dequant
  always runs between two syncs; v21's second Bs buffer was never overlapped). If SASS
  ever reorders the epilogue reuse, the staging sync covers it; still, gate 2 (T=16 vs
  v21) is the tripwire.
- **rows=4096 (32 CTAs)** is the one genuinely marginal occupancy case; predicted wash vs
  v21. If the bench says short, options in order: 3-stage Braw pipeline (+4-8 KB smem,
  still 2-3 CTAs/SM), or leave 4096-row matrices on v21 (their 64-row compute 26.2 µs is
  only 1.29× their 20.3 µs floor — cheap waste).
- **Dequant on the critical path** between two syncs — inherited from v21 (internals.md
  ranked opt), unchanged here; at T≤16 the kernel is ≥2.7× memory-bound so dequant ALU
  has slack.
- **Static smem 31.5/38.5 KB** compiles without opt-in; if a future variant adds stages
  past 48 KB it must switch to dynamic + `cudaFuncSetAttribute` (fp8_gemv2 pattern,
  fp8.cu:100).
- **L2 note**: A tile re-read by every n-block stays L2-resident (≤640 KB); first-touch
  DRAM once — same behavior as v21, no new traffic class.
- **grid.y>1 never ships in the hot path** (dispatch caps at T≤16); it exists for
  correctness testing of the masked epilogue at T%16≠0 with multiple tiles.
- The `tstar_dev` slot layout and per-row snapshots that consume argmax_rows output are
  spec-deepen §4/§5 machinery — this report only provides the GEMM + argmax primitives.
