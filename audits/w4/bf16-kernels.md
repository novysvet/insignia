# w4: bf16 kernel family for Qwen3.8-27B-FP8 — inventory + concrete sm_89 designs

Date 2026-08-25. Audit agent: w4/bf16-kernels. Read-only except this file.
Mission: MASTER-PLAN Phase C.4 (bf16 family + lm_head) with `audits/w3/embed-lmhead.md`
and `audits/w3/ab2-redesign.md` as the w3 baselines. Everything below was re-verified
against the live tree and the checkpoint headers this session (shapes read directly out
of `Qwen3.8-27B-FP8/*.safetensors`, not copied from an audit).

Hardware: 4070 SUPER, sm_89, 56 SMs, 504.2 GB/s GDDR6X, 48 MB L2, 100 KB smem/SM
(99 KB max dynamic/block), max 1536 threads/SM, bf16 wmma tensor pipe ~71-83 TFLOPS.

---

## 0. Tensor facts (verified from checkpoint headers this session)

| tensor | dtype | shape | bytes | shard / header offset |
|---|---|---|---|---|
| `lm_head.weight` | BF16 | **[248320, 5120]** | 2,542,796,800 (2.368 GiB) | `outside.safetensors` off 0 |
| `model.language_model.embed_tokens.weight` | BF16 | **[248320, 5120]** | 2,542,796,800 | `outside.safetensors` off 2,542,796,800 |
| `mtp.fc.weight` | BF16 | **[5120, 10240]** | 104,857,600 (100 MiB) | `mtp.safetensors` off 0 |
| `layers.N.linear_attn.in_proj_a.weight` | BF16 | **[48, 5120]** | 491,520 | every linear shard, e.g. layers-0 off 92,352 |
| `layers.N.linear_attn.in_proj_b.weight` | BF16 | **[48, 5120]** | 491,520 | layers-0 off 583,872 |
| `layers.N.linear_attn.conv1d.weight` | BF16 | **[10240, 1, 4]** | 81,920 | layers-0 off 10,336 |

Derived traffic: lm_head sweep = 2.543 GB → **5.05 ms @504 / 5.30 @480 / 5.41 @470 /
5.91 @430 GB/s** (the mission's 5.4 ms budget = 470.9 GB/s, the optimistic edge of the
honest 430-480 band, w3/embed-lmhead §4.2). embed row = 10,240 B. a+b = 983,040 B.
mtp.fc = 104.86 MB → 0.70 ms @150 GB/s (today's kernel) vs 0.24 ms @440. NLL logits
64×248320 f32 = 63.57 MB.

---

## 1. Inventory — what exists today (file:line)

### 1.1 bf16 GEMV — `bf16_gemv` (src/qwen_kernels.cu:67-68)

- Kernel `bf16_gemv_kernel` (:67): **one 256-thread block per row**, block-strided
  scalar loop `for(i=threadIdx.x; i<cols; i+=blockDim.x)` with scalar `bf()` cvt
  (:4) and **scalar u16 loads** (2 B/lane → 64 B/warp transaction), warp+block tree
  reduce. Launcher (:68) `<<<rows,256>>>` — **rows/cols-generic, no 4096/8192
  hardcode in the kernel**. The hardcode is at the only caller: `decode.cu:155`
  passes literal `4096, 8192` (9B mtp.fc `[out,in]`); at 27B the same site must pass
  `5120, 10240`.
- Speed class: ~120-180 GB/s est. (w3/embed-lmhead §4.1: scalar cvt + 2 B loads,
  x re-read from L2 by every block). Fine for nothing; 2.543 GB lm_head would take
  14-21 ms. **Must be replaced / demoted to a fallback.**
- fp8.cu also has `bf16_get_row` (src/fp8.cu:193-200, decl include/insignia_fp8.cuh:26):
  scalar one-row gather from a bf16 table by device id, `(cols+255)/256` blocks.
  **Zero callers today.** (Correction to w3/ab2-redesign §3: it claims
  "`bf16_gemv_rows` is declared in insignia_fp8.cuh:27 and unimplemented" — no such
  declaration exists; the line is `bf16_get_row`, which IS implemented. Neither name
  has a GEMV.)

### 1.2 embed gather — MXFP4-only

- `embed_gather` (src/prefill.cu:9-23) and `embed_gather_i4` (:26-40): `<<<T,128>>>`,
  one thread per 32-elt group, **hardcoded 9B geometry** (`w + row*512` u32, scales
  `row*128+g` / `row*64+(g>>1)`, out stride 4096). No bf16 variant.
- Single-row decode twins `mxfp4_get_row_mlx` / `_i4` (src/mxfp4.cu:277-278,
  src/mxfp4_i4.cu:245-255), called from `Qwen35Weights::embed_dev`
  (src/qwen35.cu:33) — **MXFP4-only too**: a bf16 embed tensor would be reinterpreted
  as nibbles. `embed_dev` needs the bf16 branch.
- Loader side is ready: `Qwen35Weights::matrix` (src/qwen35.cu:25-28) already returns
  `WKind::bf16` (`{w, DeviceView{}, rows, cols, false, WKind::bf16, false}`) and
  `WKind`/`QuantMatrix` exist (include/insignia_qwen35.hpp:8-9). **Nothing dispatches
  on it**: `linear/linear2/linear_batch` (decode.cu:31-41) test only `m.insig4`, so a
  bf16 matrix today falls into the mxfp4 arm = garbage.

### 1.3 bf16 GEMM — none

GEMM family is mxfp4/mlx/i4/fp8 only: `mxfp4_gemm_mlx` (gemm.cu:26, launcher :77-81),
`mxfp4_gemm_v2` (:94, :181-184), **`mxfp4_gemm_v21` (:210-291, launcher :293-296)** —
the pipelined cp.async+wmma skeleton to clone — `mxfp4_gemm_mlx_i4` (:303),
`mxfp4_gemm_v21_i4` (:371), `mxfp4_gemm_ab_i4` (:461), `fp8_gemm` (fp8.cu:109-190).
Shared helper `f32_to_bf16` (gemm.cu:190-196) and cp.async intrinsics (gemm.cu:11-17).
**A raw-bf16 GEMM does not exist anywhere in src/.**

### 1.4 argmax

- `argmax_fast` (qwen_kernels.cu:59-63): memset 8 B + `argmax_stage1_kernel<<<64,512>>>`
  (:25-57, grid-stride, warp+block reduce, monotonic-u64 `(orderbits<<32)|idx`
  atomicMax — 64 atomics) + `argmax_stage2_kernel<<<1,1>>>` (:58). Handles any n;
  at n=248320 → 32768 threads ≈ 7.6 elems each, ~4-6 µs/call, logits L2-resident.
- `argmax_logits` (single-block reference, :20-21). `src/test_argmax.cu` covers both,
  including exact ties (nit: `argmax_fast` breaks ties toward the higher index,
  `argmax_logits` toward the lower — valid both, noted asymmetry).
- **No T-row batched variant** — the T=2 verify path calls it twice back-to-back
  (decode.cu:101-102).

### 1.5 conv1d bf16 — already exists

`conv4`/`causal_conv4_silu` (qwen_kernels.cu:7-8) takes `const uint16_t* w` (bf16
weights, `w[c*4+i]` layout = checkpoint `[10240,1,4]` linearization) and is
**n-generic**; `conv_prefill_silu` (prefill.cu:170-199) likewise but with 8192
literals. 27B only changes call-site literals (decode.cu:128 `8192` → 10240; the
prefill.cu conv kernels' 8192s). **No new kernel.**

### 1.6 ab2 pair

9B fused pair kernels exist and are now guarded: `mxfp4_gemv_ab2_q8` (mxfp4.cu:589,
launcher :669-673 **throws cols!=4096**), `mxfp4_gemv_ab2_q8g` (:521, :577-580
throws), `mxfp4_gemv_ab2_q8_i4` (mxfp4_i4.cu:157, launcher :237-241 **throws**) —
the w3/ab2-redesign §7 guard recommendation has already landed at all three.
**The bf16 [48,5120] pair kernel does not exist** (design in §5).

### 1.7 lm_head call sites (all currently MXFP4-only)

| site | what happens today | 27B need |
|---|---|---|
| decode.cu:133 `forward_body` | `linear(lm_head, norm, logits)` (mxfp4 GEMV) | bf16 GEMV (or GEMM T=1) + argmax |
| decode.cu:97-105 `prefill_chunk_device` | T==2 → `mxfp4_gemv2_q8[_i4]` pair sweep + 2× `argmax_fast` (:101-102); T>2 → last-row GEMV (:103-104) | bf16 GEMM T=2 (+merged argmax) |
| decode.cu:190-191 `mtp_layer` | `linear(lm_head)` + `argmax_fast` | bf16 GEMV/GEMM T=1 + argmax |
| nll.cu:78-82 | `mxfp4_gemm_mlx[_i4]` T=chunk (the OLD non-pipelined KT=32 GEMM, not v21!) | bf16 GEMM T≤64 |
| generate.cu:76-81 `run_nll` | same `mxfp4_gemm_mlx[_i4]` | bf16 GEMM T≤64 |
| decode.cu:152-157 `mtp.fc` | `bf16_gemv(...,4096,8192)` literal dims | `bf16_gemv` v2 rows=5120 cols=10240 |

### 1.8 VRAM workspace status

- `DecodeWorkspace::logits` = `248320*2` f32 = **1.99 MB** (decode.cu:14) — enough for
  spec verify T=2 only. `pf_bf16` = 64×12288×2 B = 1.57 MB (decode.cu:26; 27B needs
  64×17408×2 = 2.23 MB per MASTER-PLAN Phase B).
- The **63.57 MB NLL logits buffer already exists**, allocated locally in both NLL
  drivers: nll.cu:62 `cudaMalloc(&logitsT, chunk*vocab*4)` (chunk default 64) and
  generate.cu:62 `cudaMalloc(&logitsT, 64*vocab*4)`. It is NOT in DecodeWorkspace and
  not in the MASTER-PLAN §2.2 fixed block explicitly ("workspace 250 MB" absorbs it).
  No new allocation is needed for NLL — only the bf16 branch at :78-82/:76-81.
- lm_head itself must be VRAM-resident 2.543 GB (MASTER-PLAN §2.2, embed-lmhead §2 —
  20-22× faster than PCIe per sweep; pinned-RAM alternatives don't fit 15.9 GiB).

---

## 2. Design A — `bf16_gemv` v2: persistent warp-per-row GEMV (lm_head decode T=1)

Build order note (MASTER-PLAN C.4): build `bf16_gemm` (§3) FIRST and measure T=1
through it; add this GEMV only if GEMM-T1 loses. The GEMV is also the right kernel for
mtp.fc (5120 rows — a 64-row-tile GEMM wastes grid) and any small-row bf16 matrix,
so it gets built either way; the open question is only which one serves lm_head T=1.

Structure = `mxfp4_gemv_v2` (mxfp4.cu:90-144) with the decode path swapped and the
w3/insig4-perf §2.2 persistent wrapper adopted from day one:

- **Persistent grid**: `blocks = min((rows+7)/8, 56 * occ)` with occ from
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (smem 20 KB + 256 thr → 5 blocks/SM
  at cols=5120 (100/20; thread cap 1536/256=6), grid = **280 blocks**). Each block
  stages x ONCE, then grid-strides rows: 280×8 = 2240 rows in flight, 248320/2240 =
  110.9 passes — no 31040-block launch tail, no 31040× x re-staging (the v2 kernel
  re-stages x per block = 509 MB of L2 traffic + a sync per 8 rows, insig4-perf §2.1).
- **Warp-per-row, lane owns 32-weight groups**: cols=5120 → 160 groups; 5 iterations
  of `g0 = lane; g0 += 32`. Per iteration a lane eats 4× `uint4` (64 B = 32 bf16);
  the warp's 32 lanes cover groups 0..159 consecutively → **2 KB contiguous per
  warp-iteration, 10 KB per row** (10240 B row = 640 uint4 = 20/lane).
- **Weights `__ldcs`** (evict-first: the 2.543 GB stream flushes 48 MB L2
  continuously; correct per insig4-perf §2.4).
- **x staged transposed** into dynamic smem (`sx[k*160+g] == x[g*32+k]`, 20 KB) so
  the per-lane x reads `sx[k*groups+g0]` are bank-conflict-free (same staging loop as
  mxfp4.cu:95-106 — for fixed k the lanes touch consecutive g0). One `__syncthreads()`
  after staging; the row loop needs no further syncs (warps independent).
- **bf16→f32 by bit surgery** (project rule "bit manipulate floats"): a bf16 IS the
  high half of an f32 — per u32: low element `__uint_as_float(u<<16)`, high element
  `__uint_as_float(u & 0xffff0000u)`. Exact, zero cvt instructions; both cvt-pipe and
  ALU-pipe are ≤3% of capacity so this is a style choice, verified once in SASS
  (embed-lmhead §3.1).
- **fp32 accumulation, 4 partial sums** for ILP (`p0..p3` like mxfp4.cu:121-124).
  bf16×bf16 products are exact in fp32; only the 5120-term accumulation rounds
  (rel err ~3.6e-6 = logit noise floor). fp16/bf16 accumulation rejected — 1-5% /
  worse error flips near-tie argmaxes (embed-lmhead §3.1).
- **Compute check** (embed-lmhead §3, corrected units): 504 GB/s ÷ 2 B = 252 G
  weights/s = 1.4% of the 17.74 T fp32-FMA/s pipe; whole-kernel warp-instruction issue
  ≈ 2.5-5% — pure DRAM-bound, no tensor cores needed at T=1.

```cuda
// Persistent bf16 GEMV, sm_89: y[row] = w[row,:]·x. Warp-per-row, lane owns
// 32-bf16 groups (4x uint4 __ldcs), x staged transposed once per block, fp32 acc,
// bf16->f32 by u<<16 / u&0xffff0000 bit surgery. cols%32==0; rows arbitrary.
__global__ __launch_bounds__(256) void bf16_gemv_v2_kernel(
        const uint16_t *__restrict__ w, const float *__restrict__ x,
        float *__restrict__ y, int rows, int groups /* = cols>>5 */) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    extern __shared__ float sx[];                       // [groups*32] transposed x
    for (int c0 = threadIdx.x * 32; c0 < groups * 32; c0 += blockDim.x * 32) {
        const float4 *xr = reinterpret_cast<const float4 *>(x + c0);
        float *sr = sx + (c0 >> 5);
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const float4 v = __ldg(xr + q);
            sr[(q*4+0)*groups] = v.x; sr[(q*4+1)*groups] = v.y;
            sr[(q*4+2)*groups] = v.z; sr[(q*4+3)*groups] = v.w;
        }
    }
    __syncthreads();                                    // the ONLY sync
    for (int row = blockIdx.x * 8 + warp; row < rows; row += gridDim.x * 8) {
        const uint4 *wr = reinterpret_cast<const uint4 *>(w + size_t(row) * groups * 32);
        const float *xg = sx;                           // xg[k*groups+g] == x[g*32+k]
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
        #pragma unroll 4
        for (int g0 = lane; g0 < groups; g0 += 32) {
            #define BF16X2(u, pa) { \
                const uint32_t u_ = (u); \
                pa = fmaf(__uint_as_float(u_ << 16),       xg[(kb+0)*groups+g0], pa); \
                pa = fmaf(__uint_as_float(u_ & 0xffff0000u), xg[(kb+1)*groups+g0], pa); }
            const uint4 P0 = __ldcs(wr + g0 * 4 + 0);  // cols [g0*32, g0*32+8)
            float s0 = 0.f, s1 = 0.f; { const int kb = 0; BF16X2(P0.x, s0) BF16X2(P0.y, s0) BF16X2(P0.z, s0) BF16X2(P0.w, s0) }
            { const int kb = 8;  BF16X2(P0.x, s1) BF16X2(P0.y, s1) BF16X2(P0.z, s1) BF16X2(P0.w, s1) }
            // ... same for P1 = __ldcs(wr + g0*4 + 1) (kb 16/24), P2, P3 omitted for brevity:
            // 4 uint4 = 32 weights per lane per group; 8 FFMA per uint4, 2 regs per u32
            p0 += s0; p1 += s1; /* p2 += s2; p3 += s3; */
        }
        #undef BF16X2
        float sum = (p0 + p1) + (p2 + p3);
        #pragma unroll
        for (int m = 16; m; m >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, m);
        if (!lane) y[row] = sum;
    }
}

void bf16_gemv_v2(const uint16_t *w, const float *x, float *y, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 31)) throw std::runtime_error("insignia: bad bf16 GEMV dims rows=" + std::to_string(rows) + " cols=" + std::to_string(cols));
    static int blocks = 0, sm = 0;
    if (!sm) { cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0); }
    const int groups = cols >> 5;
    const size_t smem = size_t(cols) * 4;
    if (smem > 99 * 1024) throw std::runtime_error("insignia: bf16 GEMV staging exceeds 99KB at cols=" + std::to_string(cols));
    static const bool cfg = [] { return cudaFuncSetAttribute(bf16_gemv_v2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024) == cudaSuccess; }();
    (void)cfg;
    int occ = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, bf16_gemv_v2_kernel, 256, smem);
    if (!blocks) blocks = std::min((rows + 7) >> 3, sm * std::max(occ, 1));
    bf16_gemv_v2_kernel<<<blocks, 256, smem, stream>>>(w, x, y, rows, groups);
}
```

(The body shown is the skeleton; the 4-uint4 unroll is mechanical — each `uint4`
contributes 8 weights, 2 f32 per u32 via the two masks. smem 20 KB @5120 → 5 blk/SM;
@10240 (mtp.fc x) 40 KB → 2 blk/SM, still fine.)

**Expected: 430-480 GB/s cold-L2 (85-95% of 504.2) → 5.3-5.9 ms/sweep**, anchored by
(a) this repo's own 379-435 GiB/s on the same [248320,·] shape with a strictly heavier
MXFP4 kernel (audits/internals.md:18,28), (b) Ada F16 GEMV production kernels at
85-93%. The mission's ≥470 GB/s target is the top edge — bench, don't promise.
mtp.fc: 104.86 MB → ~0.22-0.24 ms (vs 0.70 ms through the old scalar kernel).

Bench protocol (mandatory, insig4-perf §2.4): `l2_sweep` 256 MB evict-first sweep +
`cudaCtxResetPersistingL2Cache` between reps; cold numbers only are citable
(lm_head 2.543 GB ≫ 48 MB L2 is honest by construction; a/b at 0.94 MB is L2-resident
— report latency + launch count instead).

---

## 3. Design B — `bf16_gemm`: exact diff vs `mxfp4_gemm_v21` (gemm.cu:210-296)

`Y[T,rows] = X16[T,cols] · W[rows,cols]^T`, X16 bf16 zero-padded to 64 rows,
Y f32 row-major ldm=rows (what `row_logp_kernel` and `argmax` consume). Requirements
for lm_head: rows=248320 (%32=0 → grid 7760 at NT=32), cols=5120 (%64=0 → 80 K-steps),
T≤64. **The diff is almost entirely deletion**: the B tile is already wanted as bf16
in smem — v21 just gets there through nibbles.

Line-numbered diff against `mxfp4_gemm_v21_kernel` (gemm.cu:210) /
`mxfp4_gemm_v21` (gemm.cu:293):

1. **Signature** (v21 :210): drop scales, weights become bf16:
   `__global__ __launch_bounds__(256) void bf16_gemm_kernel(const __nv_bfloat16 *__restrict__ w, const __nv_bfloat16 *__restrict__ x16, float *__restrict__ y, int rows, int cols, int T)`.
2. **DELETE** `__shared__ uint32_t Braw[2][NT][8];` (v21 :215).
3. **DELETE** `__shared__ uint32_t lut[256];` and its 16-entry `tbl` init block
   (v21 :216-221).
4. **DELETE** `const int groups = cols >> 5;` (v21 :222) — no scale groups anywhere.
5. **A prefetch** (v21 :229-235): **unchanged verbatim** (cp_async16 of 64 rows ×
   128 B from `x16`).
6. **B prefetch** (v21 :237-242): replace the 2×16 B nibble chunks into `Braw` with
   **direct cp.async of raw bf16 tiles into `Bs`** — geometry becomes the A-path with
   `m`→`n0+n`, destination `As`→`Bs`:
   ```cuda
   {   // B: NT rows x 128B (64 bf16) = NT*8 chunks of 16B — raw bf16, no dequant
       const char *bg = reinterpret_cast<const char *>(w);
       for (int i = tid; i < NT * (KT / 8); i += 256) {
           const int n = i / (KT / 8), c8 = i % (KT / 8);
           const size_t srcoff = (size_t(n0 + n) * cols + k) * 2 + c8 * 16;
           cp_async16(&Bs[buf][n][c8 * 8], bg + srcoff);
       }
   }
   ```
   (NT=32: 256 chunks vs A's 512 → 1 iteration of the strided loop; smem alignment:
   `Bs[buf][n]` rows are (64+8)×2=144 B apart — `cp_async16` dst is 16 B aligned
   since `c8*8` elements = 16 B steps from a 144 B-aligned row base... note 144%16=0
   ✓ and `&Bs[buf][n][c8*8]` is 16 B aligned for every c8 ✓.)
7. **DELETE the whole `dequant` lambda** (v21 :244-261) and its call + the extra
   `__syncthreads()` (v21 :278-279). After `cp_async_wait_prev/_all` + the single
   `__syncthreads()` (v21 :277), `Bs[buf]` is already final bf16.
8. **Main loop** (v21 :272-289): keep the FIXED tail-wait semantics now in the tree
   (gemm.cu:275-276 `if (kb+2<ksteps) cp_async_wait_prev(); else cp_async_wait_all();`
   — the Phase-0 race fix; the clone must inherit it), then the wmma block
   (v21 :280-287) verbatim, then the trailing `__syncthreads()` (v21 :288).
9. **Epilogue** (v21 :290): adopt the guarded store from `mxfp4_gemm_v21_i4`/
   `fp8_gemm` (gemm.cu:447, fp8.cu:184):
   `if (wm * 16 < T) wmma::store_matrix_sync(y + size_t(wm*16) * rows + n0 + wn * 16, acc, rows, wmma::mem_row_major);`
   (16-row tile granularity: rows T..ceil(T,16) write zeros given zero-padded X16;
   caller's y must be 64-row padded — nll/generate logitsT already is, and the spec
   verify y = `x_.logits` [2,vocab] needs either a 64-row scratch or the T=2 call
   targets a [64,vocab] scratch. Simplest: verify path reuses a 64×vocab scratch
   (63.57 MB, see §7) or the guard is tightened to `wm*16 < T` with per-row clamped
   stores — recommend the padded-scratch contract, matching fp8_gemm's.)
10. **Launcher** (v21 :293-296):
    ```cuda
    void bf16_gemm(const void *w, const void *x16, float *y, int rows, int cols, int T, cudaStream_t stream) {
        if (rows <= 0 || cols <= 0 || (cols & 63) || T <= 0 || (rows & 31))
            throw std::runtime_error("insignia: bad bf16 GEMM dims rows=" + std::to_string(rows) + " cols=" + std::to_string(cols));
        if (T > 64) throw std::runtime_error("insignia: bf16_gemm T=" + std::to_string(T) + " exceeds the 64-row A tile");
        bf16_gemm_kernel<<<rows >> 5, 256, 0, stream>>>((const __nv_bfloat16 *)w, (const __nv_bfloat16 *)x16, y, rows, cols, T);
    }
    ```
11. A-side staging at the call site reuses `stage_a` (decode.cu:35-38: tail-zero
    memset + `f32_to_bf16`) verbatim.

Numbers: smem 2×64×72×2 + 2×32×72×2 = 27 KB → 3 blocks/SM (168 resident), grid 7760
blocks = 46 waves. IO-bound: 2.543 GB @460-480 GB/s ≈ **5.3-5.5 ms regardless of T**
(T=2 verify: one launch, one weight pass — compute 5.09 GFLOP ≈ 0.07 ms; T=64 NLL:
162.7 GFLOP ≈ 2.0-2.3 ms at 71-83 TF, fully hidden under the stream, 39-43% tensor
utilization). NT=64 variant (Bs 2×64×72×2, total 36 KB, 2 blocks/SM, grid 3880) is a
one-line bench knob if 46-wave NT=32 underperforms; ship NT=32 first (minimal diff).
T=1 through the GEMM ≈ 460-490 GB/s (embed-lmhead §4.3) — hence "measure GEMM-T1
before building the GEMV for lm_head"; the GEMV still lands for mtp.fc.

---

## 4. Design C — `embed_gather_bf16`: T rows from staged-pinned (UVA) or pinned table

Placement ground truth (MASTER-PLAN §0.5/§2.4): **embed stays on NVMe**, consumed as
a 10 KB row-pread issued a step ahead (row for [pending] known at commit time; row for
[draft] staged post-argmax in the same window; 27B has no CUDA graphs so the per-step
host window exists). Whole-embed-pinned (2.543 GB, w3/embed-lmhead §7's primary) is
**v1-only** (v1 RAM slack is huge: 1.76 GB used of 13.5 GB); v2's 12.8/13.5 GB ledger
has no room, so the staged-row path is the production design. One kernel serves both:
the only difference is whether `w` points at the whole table or at a T-row staging
buffer, and whether `tokens[t]` is a real id or a staging slot (0..T-1).

```cuda
// Gather T bf16 embed rows (device ids) into f32 activations. `w` may be a pinned
// host pointer (UVA: whole table in v1, or a host-staged row set on NVMe placements).
// 10240 B rows are 16 B aligned (10240%16==0), so uint4 loads are legal over PCIe.
__global__ void embed_gather_bf16_kernel(const uint16_t *__restrict__ w,
                                         const int *__restrict__ tokens,
                                         float *__restrict__ out, int hidden /*=5120*/) {
    const int t = blockIdx.x;                        // <<<T, 256>>>, T <= 64
    const size_t row = size_t(__ldg(tokens + t));
    const uint4 *src = reinterpret_cast<const uint4 *>(w + row * hidden);
    float4 *dst = reinterpret_cast<float4 *>(out + size_t(t) * hidden);
    for (int i = threadIdx.x; i < hidden / 8; i += 256) {   // 640 uint4 = 10 KiB
        const uint4 p = src[i];                      // warp = 512 B contiguous over PCIe
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&p.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&p.y));
        const float2 c = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&p.z));
        const float2 d = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&p.w));
        dst[2 * i]     = make_float4(a.x, a.y, b.x, b.y);
        dst[2 * i + 1] = make_float4(c.x, c.y, d.x, d.y);
    }
}
void embed_gather_bf16(const uint16_t *w, const int *tokens_dev, float *out, int T, int hidden, cudaStream_t stream) {
    if (hidden <= 0 || (hidden & 7)) throw std::runtime_error("insignia: embed_gather_bf16 hidden must be %8");
    embed_gather_bf16_kernel<<<T, 256, 0, stream>>>(w, tokens_dev, out, hidden);
}
```

(Conversion here can equally use the §2 bit-surgery masks; `__bfloat1622float2` on a
pure-copy kernel is immaterial — 640 cvt-pairs per 10 KiB.)

- **T=1 decode** (`forward_token`/`mtp_layer`): 1 block; cost = PCIe UVA latency
  ~2-4 µs + 10.24 KiB at link rate ≈ **3-5 µs/step** (0.0002% of the 1.63-5.21 s
  step). Host side: one 10 KB buffered `read_once` pread into the pinned row
  (NvReader's buffered twin exists for exactly this, nvme-reader §3), ~1 ms at
  3.3 GB/s device-level QD but amortized/hidden by issue-a-step-ahead prefetch.
- **T=2 spec pair** (decode.cu:46): 2 blocks; host stages [pending, draft] rows.
- **T≤64 prefill**: 64 blocks; 640 KiB over UVA @18 GB/s ≈ 36 µs, fully hidden.
- Alignment trap (MASTER-PLAN Phase D "CRITICAL"): `data_start ≡ 8 mod 16` in shards —
  applies to mmap/NVMe-mapped bases. A pinned **copy** (cudaHostAlloc +
  cudaMemcpy / pread into pinned) is host-page aligned by construction → all uint4
  loads legal. If the whole-table variant ever points at a file-mapped view: don't —
  Windows file maps aren't UVA-visible to the GPU anyway; always copy into pinned.
- Graph-safety: fixed pinned address ⇒ capture-safe (moot at 27B — no graphs).
- The old MXFP4 `embed_gather*` stay for the 9B; dispatch on `m.kind == WKind::bf16`.

---

## 5. Design D — `bf16_gemv_ab2_pair` (in_proj_a+b, [48,5120]×2, one launch)

w3/ab2-redesign §3 "design B" is confirmed correct against the current tree; full
source lives there (kernel + launcher + header line + call site). Summary + deltas:

- **Why**: 4 GEMVs (a@x0, a@x1, b@x0, b@x1) per DeltaNet layer per pair step → one
  launch. Weights 0.94 MB ≈ 2% of L2 → **latency/launch-bound**; the only levers are
  launch count (1) and SM spread (96 blocks = 48 A-rows + 48 B-rows, 256 threads,
  one block reduces one row over 5120 cols = 20 cols/thread = 5 rounds of
  `uint2`(w) + `float4`×2(x)). Expected **~1.5-2.5 µs warm** (DRAM floor 1.9 µs cold).
- **Exact `u<<16` bit surgery** per u32: `w01.x = __uint_as_float(p.x << 16)` (element
  2k), `w01.y = __uint_as_float(p.x & 0xffff0000u)` (element 2k+1) — two of four
  weights per u32 are pure AND-masks; exact, no cvt.
- **fp32 accumulation, deterministic tree reduce** (no atomics — parity reruns must be
  bit-stable), one `__syncthreads()` at the cross-warp handoff (`red[2][8]` written
  strictly before, read strictly after — immune to the ops.cu read-overwrite race
  class), no dynamic smem. `__ldcs` on weights, `__ldg` on x (`pf_n` just written by
  rmsnorm, L1-hot).
- Output layout `ya[row]`, `ya[heads+row]` = [T=2][heads] — exactly what
  `deltanet_params_batch` consumes (decode.cu:82).
- Launcher **throws** on `cols&3` (8 B alignment) — shape-generic (48/5120 call site;
  also serves 32/4096). Guards on the three 9B ab2 launchers already landed
  (mxfp4.cu:578, mxfp4.cu:670, mxfp4_i4.cu:239) — nothing more to do there.
- Call-site delta (decode.cu:66-70): add the bf16 arm
  `if (ma.kind == WKind::bf16) bf16_gemv_ab2_pair(...)` selecting on the kind the
  loader already returns (qwen35.cu:25-28).
- **Dependency**: `pf_a`/`pf_b` workspace is 64×32 (decode.cu:24) → must be 64×48 for
  27B (Phase B cluster-3 list).
- CPU twin (ab2-redesign §5, AVX2 `vpmovzxwd+vpslld`+FMA, ~20 µs/layer) exists as a
  spec for the CPU tier; not needed until v2.
- qkv-fusion quantified and rejected (ab2-redesign §4: ≤0.2% even all-VRAM, for a
  second dtype inside the hottest kernel). Do not revisit.

Test: `src/test_bf16_ab2.cu` per ab2-redesign §6 (cos > 1-1e-6 vs exact-double dots at
48/5120 AND 32/4096, negative throw tests, cross-check vs old `bf16_gemv` per row).

---

## 6. lm_head argmax at 27B

- **Keep `argmax_fast`** (qwen_kernels.cu:59-63): correct, ~4-6 µs, 0.001% of the 27B
  step. Vocab is identical (248320) so nothing changes shape-wise.
- **Merge the two T=2 calls** (decode.cu:101-102) into one launch — the one cheap
  upgrade worth doing while touching the file: `argmax_fast_rows(x, vocab, T, out)`
  with grid `(64*T, 512)`, row = `blockIdx.x >> 6`, T u64 scratch slots (memset
  T×8 B), stage 2 `<<<T,1>>>` unpacks. Saves a memset + 2 launches (~2-3 µs/step);
  also serves D=4 verify T=5 unchanged.
- **Fused GEMV/GEMM+argmax**: feasible — the persistent GEMV can carry a block-local
  best (value,idx) across its ~110×8 rows and issue ONE atomicMax per block (280
  atomics; the GEMM can do one atomic per (block,row-tile)). Saves the 993 KB logits
  re-read (~2 µs) + 3 launches. Verdict (embed-lmhead §8.3, agreed): **not worth the
  coupling now** (~0.04% of step); fold in only if the launch-count crusade reaches
  it. Logits are still written in all fused variants — NLL needs them.
- NLL T=64 uses `row_logp_kernel` (nll.cu:12-43, generate.cu:15-46), not argmax; the
  `red[8]/red[9]` dedicated-slot race fix has landed there already (verified this
  session — max lands in `red[8]`, sum in `red[9]`, no overwrite window).

---

## 7. VRAM / workspace ledger for the bf16 family

| item | size | status |
|---|---|---|
| lm_head weights (VRAM, mandatory) | 2,542.8 MB | MASTER-PLAN §2.2 fixed block; pin at load, refuse to start if short |
| NLL logitsT 64×248320 f32 | 63.57 MB | already allocated at nll.cu:62 + generate.cu:62 — no change |
| decode `logits` 2×vocab f32 | 1.99 MB | decode.cu:14 — sufficient for T=2 verify; T>2 verify/NLL uses logitsT |
| spec-verify GEMM y scratch (64-row padded contract) | 63.57 MB | NEW if verify routes through bf16_gemm with the padded-y contract (option: reuse logitsT-shaped scratch allocated once; or clamp-store variant at T=2 to skip it — pick padded-scratch for uniformity with fp8_gemm) |
| `pf_bf16` A-staging 64×cols×2 | 2.23 MB @17408 (1.57 MB today @12288) | decode.cu:26 — Phase B resize |
| embed staging (pinned host, not VRAM) | 64 rows × 10,240 B = 640 KiB ring + one 10 KB decode row | NEW (host pinned; counts against the 8,531 MB WDDM pinned cap — 0.66 MB, noise) |
| x smem (GEMV) / tiles (GEMM) | 20 KB / 27 KB per block | on-chip |

---

## 8. Plug-in map (every call site, with the branch condition)

Dispatch key everywhere: `QuantMatrix::kind == WKind::bf16` (already produced by
`Qwen35Weights::matrix`, qwen35.cu:25-28). The `linear*` trio (decode.cu:31-41) gains
the bf16 arm (and the fp8 arm per Phase C.3 — same edit):

1. **decode.cu:31 `linear`** → `bf16_gemv_v2(m.weight.data, in, out, m.rows, m.cols)`
   (serves lm_head T=1 via :133/:190 and any bf16 per-token GEMV).
2. **decode.cu:32 `linear2`** (pair) → for bf16: `bf16_gemm` T=2 (or a bf16 pair-GEMV
   twin of `mxfp4_gemv2_v2` if benches favor it — start with GEMM).
3. **decode.cu:33-41 `linear_batch`** → `stage_a(m.cols); bf16_gemm(w, pf_bf16, out,
   rows, cols, T)` alongside the v21/v21_i4 arms.
4. **decode.cu:46** (pair embed): bf16 → `embed_gather_bf16(staged, pf_tokens, pf_x,
   2, 5120)`. **decode.cu:71-77** non-pair path same with T.
5. **decode.cu:66-70** (pair a/b): bf16 → `bf16_gemv_ab2_pair` (§5).
6. **decode.cu:97-105** (verify lm_head): T==2 → `stage_a(5120)` on `pf_n` +
   `bf16_gemm(w, pf_bf16, y64, 248320, 5120, 2)` + `argmax_fast_rows` on rows 0/1;
   T>2 → GEMM T (verify D=4) or GEMV last row (current :103-104 semantics).
7. **decode.cu:133 `forward_body`**: `linear` dispatch covers it; then `argmax_fast`
   at :135 unchanged.
8. **decode.cu:134 `forward_token` → qwen35.cu:33 `embed_dev`**: add bf16 branch →
   `bf16_get_row` (fp8.cu:198, exists, scalar — fine for 10 KB) or the staged-row
   variant of §4 (`embed_gather_bf16` with T=1).
9. **decode.cu:137-144 `mtp_layer` embed**: bf16 → `embed_gather_bf16(..., 1, 5120)`.
10. **decode.cu:152-157 mtp.fc**: replace literal `4096, 8192` with shape constants →
    `bf16_gemv_v2(w, x_.qkv, x_.hidden, 5120, 10240)` (~0.23 ms vs 0.70 today).
11. **decode.cu:190-191** (draft lm_head): `linear` dispatch + `argmax_fast`.
12. **nll.cu:78-82 / generate.cu:76-81**: bf16 → `stage_a` + `bf16_gemm(w, pf_bf16,
    logitsT, 248320, 5120, T)` (also fixes that both still call the OLD
    non-pipelined `mxfp4_gemm_mlx`, not v21 — same edit should move mxfp4 to v21).
13. **conv1d**: no new kernel — `causal_conv4_silu` is bf16 and n-generic; flip
    decode.cu:128 literal 8192→10240 and prefill.cu conv kernels' 8192s (Phase B).
14. `capture_spec`/`capture_step` (decode.cu:238/255): re-capture after any launch
    change (9B standing rule); 27B driver doesn't capture at all.

---

## 9. Correctness + acceptance gates (per AGENTS.md: measure + parity)

1. Unit tests, house style of `test_fp8.cu` / `test_argmax.cu`:
   - `bf16_gemv_v2` / `bf16_gemm`: cosine vs f64 reference > 0.999999 AND max-rel
     < 1e-4 at [248320,5120] slices + T∈{1,2,3,33,64} (GEMM; include the tile-
     boundary Ts), throw-tests T=65, cols=5122.
   - `embed_gather_bf16`: **exact** row equality vs host bf16 decode (pure copy+cvt).
   - `bf16_gemv_ab2_pair`: per ab2-redesign §6 (cos > 1-1e-6, throws, CPU twin
     determinism, per-row cross-check vs old `bf16_gemv`).
   - `argmax_fast_rows` agrees per-row with `argmax_fast` (T=2..5).
2. Benchmarks (cold-L2 protocol, insig4-perf §2.4; only cold citable):
   - bf16 GEMV [248320,5120] ≥ 470 GB/s target / 430 GB/s floor (5.3-5.9 ms).
   - bf16 GEMM T=1/2/64 same band (one weight pass for all T).
   - mtp.fc GEMV ≥ 400 GB/s; ab2 pair ≤ 3 µs; embed T=1 ≤ 5 µs end-to-end.
3. End-to-end: R7 (4-step multistep with bf16 embed slice + chunked lm_head argmax,
   argmax 4/4), R8 (NLL |ΔNLL| < 0.02 first run), R9 greedy vs `tools/reference27.py`.

## 10. Build-order recommendation (smallest risk path)

1. `bf16_gemm` (§3, pure deletion off v21 — half a day incl. test) → bench T=1/2/64.
2. `embed_gather_bf16` + `embed_dev`/decode.cu branches (§4, trivial).
3. `bf16_gemv_ab2_pair` (§5, full source already written in w3) + `pf_a/pf_b` 64×48.
4. `bf16_gemv_v2` (§2) — build for mtp.fc immediately; adopt for lm_head T=1 only if
   GEMM-T1 loses the cold-L2 bench.
5. `argmax_fast_rows` merged launch + the nll/generate v21-vs-mlx modernization.
6. All behind `WKind::bf16` dispatch at decode.cu:31-41 + the §8 call sites; 9B
   regression suite must stay green (kernels are shape-generic; no 9B path changes
   except the mtp.fc dims literal → constants).

## 11. Corrections to the w3 baselines found during this audit

- ab2-redesign §3: "`bf16_gemv_rows` is already declared in insignia_fp8.cuh:27 and
  unimplemented" — **wrong**: line 26 is `bf16_get_row` (implemented, fp8.cu:193-200,
  zero callers). No `bf16_gemv_rows` exists anywhere.
- ab2-redesign §7 (guard the 9B ab2 launchers) — **already landed** (mxfp4.cu:578,
  mxfp4.cu:670, mxfp4_i4.cu:239).
- embed-lmhead §7 recommends whole-embed pinned+UVA as primary — superseded by
  MASTER-PLAN §2.4 (embed on NVMe, staged row-pread; whole-table pinned only fits v1).
- embed-lmhead §9/MASTER-PLAN Phase 0 "row_logp red[0] race" — **already fixed** in
  the tree (nll.cu:24/27/36, generate.cu:27/30/39 use red[8]/red[9]).
- nll.cu:78-82 and generate.cu:76-79 call `mxfp4_gemm_mlx[_i4]` (the original
  non-pipelined KT=32 GEMM) for the NLL lm_head — not v21 — an easy modernization
  that rides the same bf16-branch edit.
- `matrix()` bf16 kind exists (qwen35.cu:25-28) but ZERO dispatch sites use it; a
  27B bf16 tensor routed through today's `linear*` silently takes the mxfp4 path
  (same class as synthesis bug #5 — loud-throw or dispatch is mandatory Phase C work).
