# ggml (standalone v0.21.0-2-gd99724f2) audit — CUDA MMVQ/MMQ + CPU AVX2 GEMV

Scope: `E:\coding\Insignia\ggml\` (read-only clone, synced with llama.cpp @ 8599e0ea).
Relevance filter: RTX 4070 SUPER = **sm_89 / Ada / cc 8.9**, Ryzen 5 5600X = **Zen 3, AVX2+F16C, no AVX-512, no AVX-VNNI, no AVX_VNNI_INT8, no AVX512-BF16**.
Everything below states what **Ada/Zen3 actually gets**, not what other arches get.

---

## 0. High-level dispatch (what runs when)

`ggml-cuda.cu:1815-1869` `ggml_cuda_mul_mat()` order (src1 must be F32, else cuBLAS):

1. **MMVF** — f32/f16/bf16 vector kernel, batch <= 8 (`mmvf.cu`)
2. (transposed-vector special case, `ggml-cuda.cu:1843-1855`)
3. **MMF** — f32/f16/bf16 small-batch **tensor-core** kernel, ncols <= 16 (`mmf.cu`)
4. **MMVQ** — quantized vector kernel, batch <= 8 (`mmvq.cu`)
5. **MMQ** — quantized GEMM, everything else (int8 tensor cores on Ada)
6. cuBLAS fallback

### On Ada specifically (cc == GGML_CUDA_CC_ADA_LOVELACE)

- `ggml_cuda_should_use_mmvf` (`mmvf.cu:786-869`):
  - F32: `ne11 <= 3` (ampere_mma_available → but Ada falls to `cc >= GGML_CUDA_CC_ADA_LOVELACE` → `ne11 <= 4`? No: `ampere_mma_available(8.9)` is true (>= AMPERE), so **F32: ne11 <= 3**).
  - F16/BF16: `ampere_mma_available` true → `src0_small && ne11 == 1` where `src0_small = (src0_ne[1] <= 512 || ne[2]*ne[3] == 1)`. So plain GEMV with <= 512 rows... effectively **ne11 == 1 only**; multi-token goes to MMF.
- `ggml_cuda_should_use_mmf` (`mmf.cu:133-191`): requires `src0_ne[0] % (32*(4/ts)) == 0` (K multiple of 128 for f16/bf16, 256 for f32), `ne01 % 32 == 0`, `src1_ncols <= 16`. BF16 needs `ampere_mma_available` → true on Ada.
- `ggml_cuda_should_use_mmvq` (`mmvq.cu:289-373`): Ada branch exists but only restricts K-quants (Q2_K<=4, Q3_K<=6, Q4_K/Q5_K<=7); **Q4_0/Q8_0/MXFP4/NVFP4: ne11 <= MMVQ_MAX_BATCH_SIZE (=8)**.
- `ggml_cuda_should_use_mmq` (`mmq.cu:259-330`): `turing_mma_available(8.9)` → **always true** (int8 tensor cores used for all batch sizes >= threshold; the `ne11 < MMQ_DP4A_MAX_BATCH_SIZE (64)` cuBLAS comparison only matters pre-Turing).

Also note `mmvq.cu:1316-1324`: if src0 lives in a compute buffer with padding, MMVQ/MMQ zero the padding via `cudaMemsetAsync` before running (correctness crutch for the padded last row).

---

## 1. CUDA — MMVQ (quantized GEMV, batch <= 8)

### 1.1 Kernel structure (`mmvq.cu:544-765` `mul_mat_vec_q<type, ncols_dst, has_fusion, ...>`)

- **Grid**: `(ceil(nrows_x / rows_per_block), nchannels_dst, nsamples)`; **Block**: `(32, nwarps)`.
- Launch params from per-arch tables (`mmvq.cu:397-542`). Ada hits `MMVQ_PARAMETERS_GENERIC` (line 90-92): `ncols_dst 1-4 → nwarps=4`; `5-8 → nwarps=2`; `rows_per_block`: 1 for ncols=1 (or nwarps when small_k), else 2.
- The whole vec_dot type + VDR is `constexpr` — zero indirect calls at runtime (`mmvq.cu:11-38, 558-566`).
- **K-loop** (`mmvq.cu:658-678`):

```cuda
for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kby = kbx * (qk/QK8_1);          // y block index that aligns with kbx
    const int kqs = vdr * (tid % (qi/vdr));    // x block quant index
    #pragma unroll
    for (int j = 0; j < ncols_dst; ++j)
        #pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i)
            tmp[j][i] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
}
```
`blocks_per_iter = vdr * nwarps*32 / qi` (`mmvq.cu:571`). For Q4_0 (qi=8, vdr=2) with 4 warps: 32 blocks/iter = 1024 K-elements per pass; a 4096-wide row is fully consumed in one loop trip by one block; each thread's `vec_dot` handles `vdr*(32/qi)` int32 chunks.
- Reduction: per-warp partials to smem `tmp_shared[nwarps-1][ncols][rows][32]` + `warp_reduce_sum` (`mmvq.cu:680-757`). Thread i==row writes dst — no atomics.
- **L2 structure**: consecutive blocks x consume x blocks in the same order across all rows-blocks; the activation (`block_q8_1 * y`) is read by every row block — it stays hot in L2 because it is tiny (<= 8 rows × K).
- **Fusion epilogue** (`mmvq.cu:599-649, 725-755`): optional gate matmul + bias + SwiGLU/GEGLU/SwiGLU-OAI fused into the same kernel (only compiled when `has_fusion` and ncols=1). Biases prefetched early "to hide latency" (comment at 621).
- **MoE multi-token variant** `mul_mat_vec_q_moe` (`mmvq.cu:771-835`): block=(32, ncols_dst), one warp per token, warp-local reduce only, `rows_per_block=2` ("2 gives best perf based on tuning").
- **PDL** (programmatic dependent launch): `ggml_cuda_pdl_sync()` at 579 / `pdl_lc()` at 823 — but these only fire `__CUDA_ARCH__ >= HOPPER` (`common.cuh:114-133`); on Ada they compile to nothing.

### 1.2 Activation pre-quantization

`ggml_cuda_mul_mat_vec_q` (`mmvq.cu:1249-1362`): src1 F32 → `block_q8_1` via `quantize_row_q8_1_cuda` into pool memory, row padded to `MATRIX_ROW_PADDING = 512` (`common.cuh:176`) — the padded tail keeps `vec_dot` reads in bounds when K is not a multiple of the block.

### 1.3 The vec_dot microkernels (`vecdotq.cuh`) — dp4a core

Q4_0 (`vecdotq.cuh:115-137`):

```cuda
template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);   // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }
    const float2 ds8f = __half22float2(ds8);
    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
}
```
The `-8` bias is cancelled **once per block** using the q8_1 partial sum `s` — no per-element subtraction. Scale math kept in half2 until the very end.

Q8_0 (`vecdotq.cuh:243-258`) is the minimal ideal: `sumi = dp4a(v[i], u[i], sumi); return d8_0*d8_1 * sumi;` (fp16 scales preloaded as scalars).

MXFP4 (`vecdotq.cuh:307-329`): FP4 values are **not** affine in the nibble, so it decodes through a 16-entry LUT with the `__byte_perm` trick (`vecdotq.cuh:34-95`):

```cuda
static __device__ __forceinline__ float vec_dot_mxfp4_q8_1(...) {
    const block_mxfp4 * bq4 = (const block_mxfp4 *) vbq + kbx;
    const int * q8 = (const int *) bq8_1->qs + iqs;
    int sumi = 0;
    #pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b1(bq4->qs, iqs + l);        // byte-gather to int (uncoalesced qs!)
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = ggml_cuda_e8m0_to_fp32(bq4->e) * 0.5f * __low2float(bq8_1->ds);
    return d * sumi;
}
```
`get_int_from_table_16` (`vecdotq.cuh:57-80`) does the 4-bit-index LUT with 6 `__byte_perm` per int32 (2 iterations × low/high half + 2 reorder perms). This is the pattern to steal for Insignia's MLX-MXFP4: LUT-in-register + dp4a, scale `e8m0 * 0.5` (0.5 = mean of LUT magnitudes folded out).
`ggml_cuda_dp4a` (`common.cuh:704-736`) maps straight to `__dp4a` on NVIDIA >= 6.1.

---

## 2. CUDA — MMQ (quantized GEMM)

### 2.1 Config for sm_89 (`mmq-config-ampere.cuh`)

Used when `highest_compiled_arch >= VOLTA` (`mmq.cuh:247-250`) — Ada included. Every quant type gets the **same shape table**:
```
CASE(<type>, 256 threads, occupancy 1, I=128, J ∈ {8,16,24,...,128},
     <sram layout per type>, K_vram=256 (MMQ_ITER_K), stream_k=true, fallback as needed)
```
i.e. 128×J output tile per block, 8 warps × 32 lanes, K sliced in 256-element VRAM chunks (QK_FP4_MMQ=512 for fp4 on Blackwell only). J is picked at launch (`mmq.cuh:1469-1550`): iterate J from 8 up while it reduces the number of column tiles, respecting smem per block (`smpbo`).

### 2.2 Shared-memory staging (no cp.async!)

`block_q8_1_mmq` (`mmq.cuh:27-46`): the activation is pre-quantized into a **transposed, 16B-padded** layout where each 128-value group becomes one contiguous smem blob; padding lanes carry the scales/partial sums (union d4/ds4/d2s6, `mmq.cuh:18-22`). x tiles are staged to smem with padded strides to dodge bank conflicts (`MMQ_TILE_NE_K=32`, stride `% 8 == 4`, `mmq.cuh:103-161`).

Tile load (`mmq-load-tiles.cuh`) is **plain global→register→shared** — e.g. Q4_0 at `mmq-load-tiles.cuh:179-241`: `get_int_b2` gathers, `__vsubss4` dequantizes nibbles to signed bytes in the **mma layout** branch; dp4a layout keeps raw nibble ints with stride `MMQ_TILE_NE_K + 1` padding. **`cp-async.cuh` (cp.async.cg 16B with L2::64/128/256B prefetch hints) exists but is only used by `fattn-mma-f16.cuh:400,475`** — MMQ dropped cp.async; it relies on `__syncthreads()` double-buffer-free staging (two half-tile phases inside `mul_mat_q_process_tile`, `mmq.cuh:901-934`) and stream-K + occupancy for latency hiding.

### 2.3 The kernel (`mmq.cuh:867-1231`)

`mul_mat_q_process_tile`: per K-chunk: `load_tiles(x → smem)`, cooperative copy of `J` y-blocks, `__syncthreads()`, `vec_dot(tile_x, tile_y, sum, 0)`, again for the second 32-K half, `__syncthreads()`.

- **Stream-K decomposition** (`mmq.cuh:944-1231`, ref arXiv 2301.03598): blocks linearize (i,j,k-chunk) space and each takes `kbc..kbc_stop` contiguous slice; partial tiles go to a `tmp_fixup` buffer; `mul_mat_q_stream_k_fixup` (`mmq.cuh:1233-1369`) folds fixups back. On NVIDIA, grid = number of tiles (not SMs) when tile efficiency >= 90% (`mmq.cuh:1433-1436`) which skips the fixup kernel entirely.
- Launch heuristics: `mmq.cuh:1387-1467` (shared mem limit via `CUDA_SET_SHARED_MEMORY_LIMIT`, fastdiv `uint3` magic constants — `common.cuh:903-960` — instead of division in hot loops).

### 2.4 vec_dot paths (`mmq-vec-dot.cuh`)

Two families selected at compile time by `use_mma_data_layout()` (`mmq.cuh:189-204`) — **Ada has `turing_mma_available` → MMA layout is always used for MMQ on sm_89**:

- **dp4a family** (kept for pre-Turing): `ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a` (`mmq-vec-dot.cuh:110-140`) — each warp lane owns one output row i, iterates k01 over the 32-wide tile, pulls y from smem via `ggml_cuda_memcpy_1<16>` (16B copies, `common.cuh:791-820`), calls the same `vec_dot_*_impl` as MMVQ with VDR_Q8_0_Q8_1_MMQ=8.
- **MMA family** (what Ada runs): `ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma` (`mmq-vec-dot.cuh:142-280`), NVIDIA branch:

```cuda
typedef tile<16, 8, int> tile_A;   // s8 tensor core m16n8k32
typedef tile< 8, 8, int> tile_B;
typedef tile<16, 8, int> tile_C;
...
tile_A A[ntx][MMQ_TILE_NE_K/QI8_0];
float  dA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_0];
// A tiles preloaded once per k-chunk via load_ldmatrix (ldmatrix.sync.aligned.m8n8.x4.b16)
for (j0 ...) for (k01 ...) {
    load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // comment: "faster than load_ldmatrix"
    mma(C, A[n][k01/QI8_0], B);                                   // mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
    sum[...] += C.x[l] * dA[...] * dB[...];                        // per-32 scale folded in fp32
}
```
The underlying PTX: `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (`mma.cuh:946`; Turing fallback at 950-960 uses 4× m8n8k16). Scale application is deferred to the epilogue per 32-K group (per-block scales live in fp32 smem `x_df`).

**There is no warp specialization** in MMQ — all 8 warps do load+math with `__syncthreads()` barriers; the split is temporal (half-tile ping-pong), not role-based.

### 2.5 MXFP4/NVFP4 in MMQ

On Ada: `GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1` (mxfp4) / `_NVFP4`, dp4a-style `ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma` with `MMQ_Q8_1_DS_LAYOUT_D4`, y converted so each int32 `d4` holds 2×e8m0 (mxfp4) or 4×ue4m3 (nvfp4) scales (`mmq.cuh:48-54`). The fp4×fp4 block-scaled `mma.sync.aligned.kind::mxf4nvf4...ue4m3` path (`mma.cuh:1145`, `mmq-vec-dot.cuh:1184+`) is **Blackwell-only** (`#ifdef BLACKWELL_MMA_AVAILABLE`) — matches Insignia's "Ada has no block-scaled FP4 MMA" note.

---

## 3. CUDA — MMVF/MMF (f32/f16/bf16) and the FP8 story

### 3.1 MMVF (`mmvf.cu:7-379`)

One block per row, threads stride K by block_size (`mmvf.cu:18-21, 136-158`): f32 path loads `float2`, accumulates with `ggml_cuda_mad` (plain fp32 FMA, `common.cuh:745`). f16 path either converts each `half2` to fp32 (`type_acc=float`) or accumulates directly in half2 when `FP16_AVAILABLE` (`mmvf.cu:191-229`). bf16 (non-HIP) accumulates in fp32 after element widening (`mmvf.cu:271-299`). Multi-warp variants reduce via smem. Includes the same GLU/bias fusion hooks as MMVQ (cols=1 only).

### 3.2 MMF (`mmf.cuh:48-294`)

Small-batch (<=16 cols) **tensor-core** GEMM over f32/f16/bf16: 32 rows/block, each warp stages a 16-row × (warp_size+4) smem tile, `ldmatrix` + f16/bf16 `mma.m16n8k16` (`mma.cuh:977-1005`), fp32 accumulate, cross-warp combine buffer with padding. `nwarps` auto-tuned 1..8 by minimizing K-loop trips (`mmf.cuh:646-655`). This replaced cuBLAS for decode-batch f16/bf16 on tensor-core GPUs.

### 3.3 FP8 e4m3 handling — **there is none for storage/GEMM**

Full inventory of fp8 in ggml-cuda:
- `FP8_AVAILABLE` = CUDART >= 11.8 → `<cuda_fp8.h>` (`vendors/cuda.h:13-16`). Used **only** by `ggml_cuda_ue4m3_to_fp32` / `ggml_cuda_fp32_to_ue4m3` (`common.cuh:840-879`) which convert **NVFP4 sub-block scales** (unsigned e4m3), including a scalar bit-twiddle fallback.
- NVFP4 quantization of activations (`quantize.cu:234-324`).
- Blackwell `kind::mxf4nvf4` scale_vec::4X ue4m3 MMA (`mma.cuh:1145`).

**No GGML_TYPE for e4m3 weights, no e4m3 GEMV/GEMM kernel, no cuBLAS fp8 path (no cublasGemmEx with CUDA_R_8F_E4M3 anywhere).** ggml's answer to "FP8 on Ada" is: quantize to q8_0/K-quants, or MXFP4/NVFP4 with dp4a/s8-MMA decode. Insignia's 128×128-bf16-block-scaled e4m3 plan has **no template to copy from ggml**; the closest analogues are (a) block_q8_1_mmq's "scales in smem padding" trick and (b) the deferred per-block fp32 scale epilogue of the s8 MMA path.

### 3.4 CUDA graphs (`ggml-cuda.cu:2543-2650, 4189-4300`)

- Single-graph-per-key (`nodes[0]` as key), stored per backend context.
- **Warmup**: at least 2 evaluations with unchanged node properties before first capture (4243-4282); property fingerprint = memcmp of each tensor struct + src data pointers/ne/nb (`ggml_cuda_graph_update_required`, 2581-2621).
- Capture on the backend stream `cudaStreamCaptureModeRelaxed` (4294); exec update via `cudaGraphExecUpdate` with re-instantiate fallback (2623-2649); launch `cudaGraphLaunch` (4218).
- Graphs are **disabled** when any `MUL_MAT_ID` needs the sync fallback path (2556-2567, ref PR #18958) or pre-Volta.
- Optional QKV branch concurrency via multi-stream + capture, opt-in env `GGML_CUDA_GRAPH_OPT=1` (4327+).
- Kernel launches go through `ggml_cuda_kernel_launch` with `cudaLaunchKernelEx` + PDL attributes when `GGML_CUDA_USE_PDL` (Hopper+ only, `common.cuh:114-133, 1567-1586`) — on Ada kernels launch normally.

---

## 4. CPU — AVX2 GEMV kernels (Zen 3 / 5600X)

### 4.1 The x86 SIMD layer (`simd-mappings.h`)

AVX2-only CPUs (Zen 3) match the `__AVX__` section (581-683):
- F32: `GGML_F32_STEP 32`, `GGML_F32_EPR 8` (4× `__m256` per step), `_mm256_fmadd_ps` under `__FMA__` (595-599), reduce via lane-fold + `hadd` (602-620).
- F16: `GGML_F16_STEP 32`, `GGML_F16_EPR 8`; **F16C** cvt on load: `_mm256_cvtph_ps(_mm_loadu_si128(...))` (644-647); arithmetic in fp32, conversion back only on store.
- GGML_BF16_* macros only exist for POWER/s390; **x86 bf16 is handled directly in the kernels** (below).

### 4.2 `ggml_vec_dot_bf16` (`vec.cpp:139-262`) — the bf16→f32 trick

```cpp
#elif defined(__AVX2__) || defined(__AVX__)
#define LOAD(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16))
    __m256 c1..c4 = zero;
    for (; i + 32 <= n; i += 32) {
        c1 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i),      LOAD(y + i)),      c1);
        c2 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i +  8), LOAD(y + i +  8)), c2);
        c3 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c3);
        c4 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 24), LOAD(y + i + 24)), c4);
    }
    // lane-fold reduction (188-195)
```
- **bf16→f32 = widen u16→u32 + shift-left-16** (`vec.cpp:174`): 2 instructions per 8 values, bit-exact, no LUT, no rounding. This is *the* pattern to copy for AVX2 bf16 decode.
- 4 independent accumulators (32 elems/iter) to hide FMA/mul latency; note the code writes `mul+add` explicitly — with `-ffp-contract=fast` compilers fuse it to FMA anyway.
- The AVX512 variants: `AVX512BF16` uses `_mm512_dpbf16_ps` (2×32 per iter, `vec.cpp:148-158`); plain AVX512F uses the same shift trick 512-bit wide (160-171). Zen 3 gets the 256-bit path.
- Leftover tail uses scalar `GGML_BF16_TO_FP32` (257-260).

### 4.3 `ggml_vec_dot_f16` (`vec.cpp:264-378`)

Generic SIMD path (344-370): 4× `__m256` accumulators, per-step `cvtph_ps` both operands, `_mm256_fmadd_ps`, tree-reduce. F16C makes this ~memory bound already; no unroll-by-2 of blocks, no prefetch (contrasts with the SSSE3 q4_0 path which double-blocks + `_mm_prefetch T0`, see 4.5).

### 4.4 Integer primitives (`arch/x86/quants.c:28-275`) — VNNI ladder

```c
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
    const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);        // true s8×s8 dot
#elif __AVX512VNNI__+__AVX512VL__ || __AVXVNNI__
    ... _mm256_dpbusd_epi32(zero, ax, sy)                                // u8×s8
#else
    const __m256i ax = _mm256_sign_epi8(x, x);      // abs
    const __m256i sy = _mm256_sign_epi8(y, x);      // sign transfer
    return mul_sum_us8_pairs_float(ax, sy);         // maddubs + madd(1)
#endif
}
```
- **Zen 3 (no VNNI of any kind) gets**: `sign/sign/maddubs/madd(ones)` = 4 ops per 32 int8 MACs + `cvtepi32_ps`. `maddubs` saturates at ±127×127×16? — avoided by the abs/sign split keeping the i16 partials within range for typical quant magnitudes (classic llama.cpp trick, `quants.c:105-134`).
- `bytes_from_nibbles_32` (90-96): one 16B load + interleave-shift-mask = 32 nibbles → bytes [0..15].
- `hsum_float_8` (43-49): extract128+add, movehl, movehdup — 5 ops.

### 4.5 `ggml_vec_dot_q4_0_q8_0` (`arch/x86/quants.c:701-857`) — AVX2 body

```c
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps( GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d) );
        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        qx = _mm256_sub_epi8(qx, _mm256_set1_epi8(8));           // [0..15] -> [-8..7]
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        const __m256 q = mul_sum_i8_pairs_float(qx, qy);
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    sumf = hsum_float_8(acc);
```
One block (32 elems) per iteration; 18B weight block + 34B activation block; scale via scalar fp16→f32 LUT (`GGML_CPU_FP16_TO_FP32` — a 128-entry table, faster than F16C scalar cvt per the comment at 265-269). The SSE path (770-837) does **2 blocks/iter + `_mm_prefetch(T0)`** — the AVX2 path relies on hardware prefetch across sequential blocks instead.

### 4.6 `ggml_vec_dot_q8_0_q8_0` (`arch/x86/quants.c:1308-1374`)

Identical shape, minus nibble decode: `loadu/loadu/mul_sum_i8_pairs_float/fmadd(d,q,acc)` per 34B block. This is the cleanest template for a CPU int8 GEMV microkernel.

### 4.7 `ggml_vec_dot_mxfp4_q8_0` (`arch/x86/quants.c:918-1002`) — directly relevant to Insignia

```c
#if defined __AVX2__
    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_fp4);  // {0,.5,1,1.5,2,3,4,6} x2
    const __m256i mone = _mm256_set1_epi16(1);
    __m256 accum1, accum2 = zero;
    for (; ib + 1 < nb; ib += 2) {                       // 2 blocks/iter
        ... q4bits_1/2 load (16B each)
        const __m256i q4b_1 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
        const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1);      // maddubs: LUT values are non-negative!
        const __m256i p_1   = _mm256_madd_epi16(p16_1, mone);
        const __m256 scale0 = _mm256_set1_ps(FP16(y.d) * GGML_CPU_E8M0_TO_FP32_HALF(x.e));
        accum1 = _mm256_fmadd_ps(scale0, _mm256_cvtepi32_ps(p_1), accum1);
    }
```
Because the MXFP4 LUT `kvalues_fp4` ∈ {0,0.5,...,6.0} is **unsigned**, the weight bytes feed `maddubs` (u8×s8) directly with **no sign-fix** — cheaper than q4_0's abs/sign dance. Scale = `e8m0 * 0.5` folded with the q8_0 scale; `_E8M0_TO_FP32_HALF` is a scalar helper. Two independent accumulators to cover latency. The scalar tail (991-1000) is the parity reference.

### 4.8 Orchestration (`ggml-cpu.c:1162-1452`)

- src1 (F32) is first converted to `vec_dot_type` (q8_0/q8_1/etc.) into `wdata`, **parallelized across threads by K-slice** (1344-1355), then `ggml_barrier`.
- `ggml_compute_forward_mul_mat_one_chunk` (1164-1252): 16×16 block tiling of the output (`blck_0/blck_1 = 16`) purely for L2/src1 reuse — `vec_dot` writes to a `float tmp[32]` stack buffer then `memcpy` to dst (false-sharing mitigation, comment at 1207).
- Work distribution (1396-1451): chunk size 16 (64 if either dim is 1); if `nchunk0*nchunk1 < nth*4` or NUMA → split by thread over the larger dimension; otherwise **atomic chunk stealing** via `current_chunk` (1426-1451).
- `nrc=2` vec_dot variants exist for NEON mmla; x86 `nrows=1`.
- **llamafile tinyBLAS** (`llamafile/sgemm.cpp`): tried first for contiguous src1 (1295-1320, and post-quantization at 1366-1388). `tinyBLAS_Q0_AVX` (1353+) does register-blocked GEMM (4×N tiles, `__m256 Cv[RN][4]`, scale broadcast trick packing 4 fp16 d's into one `_mm_set_epi64x` + `_mm_cvtph_ps`, lines 1538-1549) covering q4_0/q5_0/q8_0/iq4_nl and f32/f16/bf16 — it generally wins over the vec_dot path for batch >= 2 on AVX2.

### 4.9 Arithmetic intensity for hidden 4096-5120 GEMV (weights, per element)

| type | bytes/elem (weights) | MACs/byte | AVX2 decode cost per 32 elems (Zen3) |
|---|---|---|---|
| f32 | 4.0 | 0.25 | 4× (fmadd) |
| bf16 | 2.0 | 0.5 | widen+shift ×2 + 4 fmadd — ~1.5× f32 cost/2 data |
| f16 | 2.0 | 0.5 | 2 cvtph + 4 fmadd |
| **fp8 e4m3 (planned)** | ~1.0 (+2B/16K block scale ≈ 0) | ~1.0 | no native cvt; LUT (pshufb×2) or shift-mul path ≈ q8_0 cost, **but dp4a unusable** (nonuniform exponents) → fp32 FMA after decode |
| q8_0 | 1.0625 | 0.94 | 1 load + sign/sign/maddubs/madd |
| q4_0 | 0.5625 | 1.78 | nibble unpack + sub + i8 dot |
| mxfp4 | 0.53125 | 1.88 | 2 pshufb + maddubs + madd (unsigned LUT!) |

On a 5600X (~445 GMAC/s vector fp32 FMA ceiling, ~35-45 GB/s DDR4 effective): every row of a 4096-5120 GEMV is **bandwidth-bound by 5-10×** for all table rows; the compute headroom is why ggml tolerates the fp32-accumulate q4_0/mxfp4 decode. For Insignia's heterogeneous 27B decode: CPU share should be sized by `bytes(weights-on-CPU)/DDR-bandwidth`; bf16/fp8 decode cost is irrelevant next to memory time — but the **activation** (one K-vector per layer) must be converted once per layer and kept in L2 (ggml does exactly this via wdata + 16-row chunk tiling).

FP8 e4m3 CPU note: with 128×128 bf16 block scales the per-element cost is 1 decode + 1 FMA; a 256-entry (or 16-entry×2) `pshufb` LUT on the 8-bit code → fp32-mantissa trick: e4m3 = `((code&0x7f) << 20)` magic for normals + subnormal LUT; or shift to f16 and use F16C `cvtph` on a masked subset (e4m3 ⊂ f16 normals for exp>=1, only 8 subnormal codes need LUT) — the mmvf/AVX512-BF16 `_mm512_dpbf16_ps` shows what a native path would look like but Zen 3 has neither.

---

## 5. What Insignia should steal (ranked)

1. **MMVQ kernel skeleton** (`mmvq.cu:544-765`): block=(32,nwarps), rows_per_block 1-2, constexpr-typed vec_dot, warp-stride K loop with `blocks_per_iter = vdr*nwarps*32/qi`, smem partial reduce. Maps 1:1 to an FP8-e4m3 GEMV with a per-thread decode (LUT) + fp32 FMA microkernel replacing dp4a.
2. **MXFP4 tricks**: `__byte_perm` 4-bit LUT (`vecdotq.cuh:34-95`) on CUDA; unsigned-LUT + `maddubs` without sign-fix on CPU (`arch/x86/quants.c:944-960`); `0.5` folded into the scale (`vecdotq.cu:327`).
3. **bf16→f32 by widen+shift16** (`vec.cpp:174`) for both CPU decode of 128×128 block scales and bf16 tensors.
4. **Block-scale epilogue deferral**: int-MMA accumulate raw dots, multiply per-block fp32 scales at writeback (`mmq-vec-dot.cuh:196-199, 274`) — the exact shape Insignia needs for 128×128-scaled e4m3 on the s8/fp32 CUDA path and the CPU fp32 path.
5. **`block_q8_1_mmq` smem layout** (`mmq.cuh:27-58`): scales/partial sums hidden in transposed-block padding.
6. **CUDA graph lifecycle** (`ggml-cuda.cu:2581-2650`): node-property fingerprint + warmup-2 + `cudaGraphExecUpdate` — cheap to replicate and removes launch overhead for the 27B decode loop.
7. **Stream-K MMQ partitioning** (`mmq.cuh:944-1231`) if Insignia ever does batch/GEMM prefill in fp8: fixes wave quantization; grid=tiles when efficiency ≥ 90% to skip the fixup pass.
8. **CPU threading** (`ggml-cpu.c:1396-1451`): 16-row chunks (L2-friendly for the shared activation) + atomic chunk stealing; activation converted once per layer, split by K across threads.
9. **Do not bother**: cp.async pipelines for MMQ-style kernels (ggml dropped them there), warp specialization (absent), PDL (Hopper-only), VNNI paths (Zen 3 lacks all), `GGML_CUDA_GRAPH_OPT` multi-stream (opt-in, niche).

## 6. Gotchas / negative findings

- **No e4m3 GEMM anywhere in ggml** (storage, CUDA, CPU, cuBLAS) — Insignia's FP8 path is greenfield; expect no parity reference beyond the scale handling analogues above.
- MMVQ quantizes activations to q8_1 with **512-element row padding** (`mmvq.cu:1326-1333`, `common.cuh:176`) and memset-clears padded weight rows (`mmvq.cu:1316-1324`) — padding bookkeeping is a real correctness surface.
- CUDA graphs get disabled by the mul_mat_id fallback sync (`ggml-cuda.cu:2556-2567`) — heterogeneous MoE routing must avoid stream syncs inside captured regions.
- CPU `vec_dot_bf16` on AVX2 does **no FMA in source** (mul+add); correctness-parity builds should pin `-ffp-contract`; MSVC never fuses by default.
- `get_int_b1` in the CUDA MXFP4 vec_dot does byte-granular gathers (`vecdotq.cuh:7-16`) — fine on GPU (L1), terrible if copied to CPU.
