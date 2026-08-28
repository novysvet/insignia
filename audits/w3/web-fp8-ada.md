# Web audit: FP8 on Ada (sm_89 / RTX 4070 SUPER) — e4m3 MMA, cuBLASLt, Marlin W8A16, roofline

2026-08-25. Web research for Insignia (Qwen3.8-27B-FP8 on 4070 SUPER: sm_89, 504 GB/s DRAM,
48 MB L2, 56 SMs @ 2.475 GHz). Companion to `audits/w2/trtllm-sm89-fp8.md` and
`audits/w2/vllm-marlin-fp8.md` (local-clone audits, referenced as [TRT] and [VM]).

## TL;DR (10 lines)

1. `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` is **legal sm_89+**, introduced
   PTX ISA 7.8 / CUDA 11.8; nvcc 13.3 accepts it (CUDA 13.3's own `cuda_fp8.hpp` emits FP8
   PTX behind `__CUDA_ARCH__ >= 890`; TRT-LLM + vLLM ship it for sm_89). No errata found.
2. Ada FP8 tensor rate (FP32 accum) = **2× the BF16-FP32acc rate**. Whitepaper (4090):
   FP8/FP32acc = 330.3 dense (660.6 sparse), BF16 = 165.2 dense — the prompt's
   "330 sparse → 165 dense" reading is 2× too low.
3. Corrected 4070 SUPER rates: **71 TF bf16-f32acc / 142 TF e4m3-f32acc dense**
   (35.5 TF FP32 shader × 2 / × 4) — not 82/165 (those are 4090-class halves).
4. cuBLASLt on Ada: **tensorwide (per-tensor) FP8 scaling only**; VEC128/BLK128x128 are
   Hopper(sm_90)+; Ada FP8 also forces TN layout. **Verdict: unusable for our 128×128-block
   scales without requantizing weights — write our own kernel** ([TRT]/Marlin pattern).
5. Best published T=1 blockwise-FP8 path = **vLLM Marlin W8A16**: "up to 2× in memory-bound
   scenarios" vs bf16 (PR #5975), i.e. ~full-bandwidth weight streaming; dequant = 3 logic
   ops per bf16x2 pair with the 2^120 exponent fold (steal verbatim, [VM] §3.3).
6. `cvt.rn.f16x2.e4m3x2`: 2 packed e4m3 → 2 f16 in **1 instruction, EXACT** (e4m3 ⊂ f16,
   subnormals 2^-9 and NaN included). But it only produces **f16**, never bf16 → for
   bf16-MMA the bit trick wins (18 ops/8 weights, no cvt); for f32-FMA GEMV the cvt wins
   (12 vs 24-40 ops per 8 weights).
7. Roofline (504 GB/s): weights-bound until **T* ≈ 70** (bf16, realistic 71 TF) /
   **T* ≈ 141** (e4m3-MMA). At T=64 bf16 needs 91% of tensor peak (marginal); e4m3 needs
   45%. **Dequant-bf16 is sufficient for all our T ≤ 32; T=64 prefill is the first
   e4m3-MMA customer.**
8. Activation dynamic quant is NOT needed for W8A16 — skipping it is strictly *higher*
   precision (bf16 acts ⊃ e4m3); vLLM itself serves this checkpoint class on Ada with
   unquantized activations. Only a W8A8 e4m3-MMA path would need
   `per_token_group_quant_fp8` (fusable into RMSNorm; FlashInfer/SGLang ship fused variants).
9. Ada L2 = 48 MB; scales (≤ 305 KiB for lm_head) + x (10-640 KiB) = ≤ 2% of L2 and are
   re-referenced every CTA → natural ~100% hit rate. **accessPolicyWindow: skip it**;
   optionally mark the *weight stream* `cudaAccessPropertyStreaming` instead.
10. Bottom line: implement Marlin-style **W8A16 dequant-bf16 GEMV/GEMM first** (T=1..32),
    defer e4m3-MMA (W8A8 + act-quant + TRT-LLM promote pattern) until T≥64 prefill actually
    shows up compute-bound in nsys.

---

## 1. sm_89 e4m3 MMA: legality, PTX ISA, throughput

### 1.1 Legality + citation

- Instruction: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (also `.e5m2`).
  A = 4×u32 registers (16 fp8), B = 2×u32 (8 fp8), C/D = 4×f32. Only FP32 accumulation is
  exposed for the k32 FP8 shape on sm_89 (no f16-acc variant).
- PTX ISA 9.3 doc (ships with CUDA 13.3; current page
  https://docs.nvidia.com/cuda/parallel-thread-execution/):
  - §9.7.15.5.10 "Matrix Fragments for mma.m16n8k32" (anchor
    `#warp-level-matrix-fragment-mma-16832`) — fragment layouts for `.u8/.s8/.e4m3/.e5m2`.
  - §9.7.15.5.14 "Multiply-and-Accumulate Instruction: mma" — instruction + target gating.
  - FP8 mma + FP8 `cvt` forms were introduced in **PTX ISA 7.8 = CUDA 11.8**, the release
    that added `sm_89`/`sm_90` targets (Ada/Hopper launch vehicle). Cross-checked via the
    CUDA↔PTX version mapping (https://milthorpe.org/2022/05/09/matching-cuda-and-nvptx-isa-versions/,
    https://www.hpcwire.com/off-the-wire/nvidia-announces-cuda-toolkit-11-8-new-features/)
    and the ptxas validator reference ("mma with FP8 floating point type" is gated sm_89+,
    https://gh.evko.io/crucible-notes/ptxas/intrinsics/tensor.html). Note: the packed
    `.e4m3x2/.e4m3x4` **mma operands** (mx block-scale, `.kind::mxf8f6f4`) are a different,
    PTX ISA 8.6+ thing — irrelevant on Ada.
- nvcc 13.3 (V13.3.73, ptxas built 2026-06-09, verified locally) accepts it: CUDA 13.3's
  own `cuda_fp8.hpp` compiles FP8 cvt PTX under `__CUDA_ARCH__ >= 890` (read locally,
  lines 567-573); TRT-LLM builds `fp8_blockscale_gemm` for real sm_89
  (`set_cuda_architectures(fp8_blockscale_gemm_src 89 90 100f 120f)`, [TRT] §1); vLLM
  carries the identical asm (`marlin_mma.h:94-101`, [VM] §4.4). **No known ptxas errata
  for e4m3 mma on sm_89** (nothing in NVIDIA forums / GitHub issues found by search).
  Only footgun: it must be guarded to sm_89 (sm_80/86 targets reject it).

### 1.2 Ada FP8 tensor rate vs FP16 — whitepaper numbers (4090)

NVIDIA Ada whitepaper, Appendix A peak-performance table
(https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf),
corroborated by TechPowerUp (https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889):

| RTX 4090 (dense / with sparsity) | TFLOPS-TOPS |
|---|---|
| FP16 tensor, FP16 acc | 330.3 / 660.6 |
| FP16/BF16 tensor, FP32 acc | **165.2 / 330.3** |
| **FP8 tensor, FP32 acc** | **330.3 / 660.6** |
| FP8 tensor, FP16 acc | 660.6 / 1321.2 |
| INT8 tensor | 660.6 / 1321.2 |
| INT4 tensor | 1321.2 / 2642.4 |

- So: **e4m3 MMA with FP32 accumulation = exactly 2× the BF16/FP16-FP32acc rate.**
  The prompt's reading ("4090 = 330 sparse INT8/FP8 → 165 dense; FP16 165 sparse → 82.5
  dense") is wrong by 2× — 330.3 is the *dense* FP8-FP32acc / FP16-FP16acc number.
- FP8 with FP16 accumulate (4×) is a whitepaper rate with no exposed mma.sync shape on
  sm_89 (k32 FP8 is f32-acc only) — ignore it for kernels; cuBLASLt hits it internally
  where legal. The 660-vs-330 forum confusion
  (https://forums.developer.nvidia.com/t/ada-geforce-rtx-4090-fp8-cublaslt-performance/250737)
  is exactly this accumulate-mode distinction, not an erratum.

### 1.3 Scaled to RTX 4070 SUPER (our part)

7168 CUDA cores, boost 2475 MHz → FP32 shader = 35.5 TFLOPS
(TechPowerUp https://www.techpowerup.com/gpu-specs/geforce-rtx-4070-super.c4186). Applying
the whitepaper multipliers (tensor = 2/4/8× shader for BF16-f32acc / FP8-f32acc / FP8-f16acc):

| | dense, FP32 acc |
|---|---|
| BF16 mma (`m16n8k16...bf16.bf16.f32`) | **71.0 TF** |
| e4m3 mma (`m16n8k32...e4m3.e4m3.f32`) | **141.9 TF (≈142)** |
| INT8 / FP8-f16acc | 283.9 |

The task brief's "82 TF bf16 / 165 TF FP8" corresponds to a 4090 halved and is ~15%
optimistic for the 4070 SUPER. Matches `audits/synthesis.md` ("4070S ≈ 142 dense TFLOPS").
Roofline below uses both; conclusions are insensitive to the 71-vs-82 spread.

## 2. cuBLASLt FP8 on sm_89 — verdict: not usable for blockwise

Scaling-mode enum (local CUDA 13.3 `cublasLt.h:932-956` — verbatim definitions):

| enum | meaning |
|---|---|
| `CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F` | one fp32 scalar per tensor (per-tensor) |
| `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` | MX: 1 UE4M3 scale per 16 elems (FP4) |
| `CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` | MX: 1 UE8M0 scale per 32 elems (FP8) |
| `CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F` | per-row-A / per-col-B fp32 |
| `CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F` | 1 fp32 per 128-elem block along K |
| `CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F` | 1 fp32 per 128×128 block (DeepSeek style) |

Architecture support (cuBLAS 13.3 docs §3.1.4 "Scaling factor layouts" /
"Scaling Mode Support Overview" table, https://docs.nvidia.com/cuda/cublas/#scaling-factor-layouts;
NVIDIA cuBLAS 12.9 blog, https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/):

- **Ada (sm_89): tensorwide (per-tensor scalar) FP8 scaling only.** Blog: "Previous cuBLAS
  versions enabled FP8 tensor-wide scaling (single scaling factor) on NVIDIA Hopper and
  **NVIDIA Ada** GPUs"; the new schemes (outer-vector, 1D-128, 2D-128×128) were added in
  12.9 "**on Hopper GPUs**"; MX 16/32-elem modes are Blackwell tensor-core native.
- cuBLAS 13.3 docs additionally note Ada FP8 restrictions: scaling must satisfy the
  support-overview table, and "**A must be transposed and B non-transposed (the 'TN'
  format) on Ada**".
- The open, unanswered NVIDIA question "minimum CUDA version / GPU requirements for
  VEC128_32F + BLK128x128_32F" (https://github.com/nvidia/cudalibrarysamples/issues/310)
  shows even NVIDIA users can't get block modes below Hopper.
- `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER`-style per-tensor FP8 works on Ada since CUDA 11.8.

**Verdict for Qwen3.8-27B-FP8 (128×128 blocks) on 4070S: cuBLASLt cannot express our
scales.** Folding 128×128 blocks into one per-tensor scalar would require re-scaling every
weight block to a common exponent — i.e. requantization (precision loss, exactly what we
refuse). Per-tensor mode is also where cuBLASLt's FP8 GEMM perf on Ada is underwhelming
(forum thread above). Hand-rolled kernels win: TRT-LLM `ada_blockwise_gemm` ([TRT], the
m16n8k32 + promote reference) or Marlin W8A16 ([VM]). cuBLASLt remains useful only as a
bf16 baseline for benchmarking.

## 3. Best published FP8-blockwise GEMV/GEMM at small T

### 3.1 vLLM Marlin FP8 (W8A16) — the reference decode path on sm_89

- Origin: PR #5975 "[Kernel] Expand FP8 support to Ampere GPUs using FP8 Marlin"
  (mgoin/Neural Magic, merged 2024-07-03,
  https://github.com/vllm-project/vllm/pull/5975): original kernel
  `csrc/quantization/fp8/fp8_marlin.cu` ("efficient 4xFP8 → 4xFP16/BF16 dequant using bit
  arithmetic and SIMT ops", lineage IST-DASLab Marlin + FasterTransformer
  `interleaved_numeric_conversion.h`). Reported: **"improves performance up to 2× in
  memory-bound scenarios"** (T small), 2× weight-memory reduction, "accuracy comparable to
  FP16" because **activations are never quantized**; slight *slowdown* at M > 256 (prefill)
  accepted for decode wins; gains largest on bandwidth-starved GPUs (A10, RTX 3090).
- Blockwise (128×128) support + merge into `gptq_marlin`: PR #16850
  (https://github.com/vllm-project/vllm/pull/16850, benchmarks on A800, images only in PR).
  This is what serves DeepSeek/Qwen-FP8 checkpoints on sm_89 in vLLM today
  (`MarlinFP8ScaledMMLinearKernel` is the last reachable kernel in the priority list, [VM] §2).
- Kernel files (current main / our local clone): 
  - `csrc/libtorch_stable/quantization/marlin/marlin_template.h` (kernel; `matmul`
    inner loop :1177-1293), `dequant.h` (bit-trick dequant :321-395), `marlin_mma.h`
    (mma wrappers incl. the sm89 e4m3 one :94-101), `gptq_marlin_repack.cu` (weight repack),
  - `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py` (repack + scale
    expansion + 2^120 fold). GitHub root:
    https://github.com/vllm-project/vllm/tree/main/csrc/libtorch_stable/quantization/marlin
- **Inner loop** (from [VM] §4.3-4.4, verified in clone): 4-stage cp.async gmem→smem
  pipeline carries *raw packed fp8 bytes* (never unpacked in smem) + bf16 group scales;
  A fragments via `ldmatrix.x4`; per k16 slice and per 4 n-sub-tiles: read 2×u32 of packed
  fp8 (`frag_b_quant`), dequant each u32 → 2×bf16x2 with
  `(q&0x80008000) | ((q&0x7F007F00)>>4)` (+ second word from `q<<8`) — values come out as
  `fp8 × 2^-120`, cancelled by the 2^120 folded into the bf16 scales at load; apply scale
  with one `__hmul2` per bf16x2; feed `mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32`
  (FP32 accum). The m-loop is deliberately innermost so dequant ALU overlaps tensor-core
  math. Epilogue: lock-based FP32 global reduce through L2 (bf16 atomicAdd unavailable on
  sm8x). Decode specialization `m_block_size_8` swaps operands (`mma_trans`) for M ≤ 8.
- Effective bandwidth: "up to 2×" vs bf16 GEMM in the memory-bound regime ≈ the full
  theoretical ceiling (fp8 = half of bf16's bytes ⇒ 2× at equal achieved BW ⇒ ~504 GB/s
  weight streaming on our card). No public 4090 GB/s plot for FP8-W8A16 GEMV exists; the
  MARLIN paper (arXiv 2408.11743, https://arxiv.org/html/2408.11743v1) shows the design
  point: INT4 Marlin holds the *ideal 3.87×* speedup vs FP16 GEMM up to batch 16-32 on A10,
  roofline crossover ~batch 50-64, and dequant fused as single `lop3`-class ops per value
  pair — same architecture as the FP8 variant at a 2× ideal ratio.

### 3.2 TRT-LLM / llm.c / other

- TRT-LLM: **no dedicated FP8 GEMV** — the sm89 blockwise kernel ([TR]) tolerates M=1..8
  through the M=32 tile with cp.async predication; separate sm89 FP8 *rowwise* (per-tensor/
  per-channel) CUTLASS kernel with StreamK exists for non-block FP8. Nothing T=1-specific
  to steal beyond the promote pattern.
- llm.c (Karpathy): pretraining-only (GPT-2, bf16/ABF16 matmuls), no FP8 decode GEMV
  (https://github.com/karpathy/llm.c). Not a source.
- No blog with measured GB/s of fp8-blockwise GEMV on 4090 surfaced (searched "fp8
  blockwise gemv bandwidth", "w8a16 marlin bandwidth 4090"); closest quantitative anchors
  are PR #5975 ("up to 2×"), the MARLIN-paper roofline (ideal to batch 16-32), and
  vLLM docs' "~1.6× throughput, 2× memory" for FP8 generally
  (https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/fp8/).
  For our shapes the yardstick is our own: T=1 fp8 GEMV at 504 GB/s ⇒ qkv [10240,5120]
  52.4 MB = 104 µs, MLP up [17408,5120] 89.1 MB = 177 µs, lm_head [248320,5120]
  1.271 GB = 2.52 ms (100%-BW numbers; ~85-90% realistic).

## 4. `cvt.rn.f16x2.e4m3x2` — semantics, throughput, vs our bit trick

### 4.1 Semantics (PTX ISA 9.3 §9.7.9.22 "cvt")

- Form: `cvt.rn.f16x2.e4m3x2 d, a;` — `a` is a 16-bit register holding **two packed e4m3
  values**; `d` is a 32-bit register holding two f16 (f16x2). (Reverse direction is
  `cvt.rn.satfinite.e4m3x2.f16x2`, which does round + saturate-to-±448.) Introduced with
  the sm_89 FP8 cvt family in PTX ISA 7.8.
- `.rn` = round-to-nearest-even, but the conversion is **EXACT for every e4m3 input**:
  e4m3 (4-bit exp, bias 7, 3-bit mantissa) has range ±448 with min subnormal 2^-9; f16
  (5-bit exp, bias 15, 10-bit mantissa) covers ±65504 with min subnormal 2^-24 — every
  e4m3 magnitude (normals *and* subnormals, 3-bit mantissa ⊂ 10-bit) is exactly
  representable; e4m3's single NaN encoding (S.1111.111) maps to an f16 NaN. So `.rn`
  never actually rounds. (e5m2 → f16 is likewise exact: min subnormal 2^-16, inf → inf.)
  This is *unlike* the manual f32 trick in synthesis.md, which is subnormal-wrong.
- CUDA C++ intrinsic: `__half2_raw __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t,
  __NV_E4M3)` (declared `cuda_fp8.h:377-379`). Device implementation (CUDA 13.3
  `cuda_fp8.hpp:563-583`, read locally): for `__CUDA_ARCH__ >= 890` it is literally
  `asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(x));` — 1 PTX instruction per 2
  weights; pre-Ada/host falls back to bit-emulation (with full subnormal normalization).
  **There is no `cvt` to bf16x2 from e4m3x2** (neither in PTX nor in `cuda_fp8.hpp` —
  bf16x2 destinations exist only from ue8m0/f32/bf16). That single fact decides the
  bf16-MMA path must stay on the bit trick.
- Throughput: PTX is an ISA spec (no cycle numbers); the CUDA "Throughput of Native
  Arithmetic Instructions" table
  (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#arithmetic-instructions)
  has **no separate FP8-conversion row** — it is an ordinary ALU conversion (16-bit-class
  conversions run at 64 results/SM/clk on cc 8.x/9.x; int8/16→32 conversions likewise 64),
  **not an SFU op** (SFU = rcp/sqrt/etc. at 16/SM/clk). Practically: treat it as a
  1-instruction, 2-result ALU op that issues like any other cvt; the instruction *count*
  is what matters.

### 4.2 Instruction count for 8 fp8 weights (all paths)

| path | instructions for 8 weights | output | notes |
|---|---|---|---|
| **Marlin bit trick → bf16x2** (2×u32 in) | **18** (9 per u32: and/and/shr/or ×2 + shl) + 4 `__hmul2` if scales not pre-folded | 8×bf16 | feeds `mma.m16n8k16.bf16` directly; exact incl. subnormals; scale = 2^120-folded bf16 |
| `cvt.rn.f16x2.e4m3x2` ×4 | **4** (+4 `hmul2` if f16 compute with 2^8 fold) | 8×f16 | f16 compute only; f16 accumulation over K=5120 is a precision risk |
| cvt ×4 + 8× `cvt.f32.f16` | **12** | 8×f32 | the FMA/GEMV route; exact incl. subnormals |
| manual scalar trick (`(b&0x7f)<<20 + 0x3C000000`, +sign) | **24-40** | 8×f32 | 3-5 ops/value, one value per op-slot; subnormal-wrong |

**Which wins:** for the **bf16 tensor-core path** the bit trick (18 ops) is the only
option (no e4m3→bf16 cvt exists) and is cheaper than cvt+f16→bf16 anyway. For a **CUDA-core
f32-FMA GEMV** (our MXFP4-style decode kernels), the cvt route (12 ops) halves to thirds
the manual trick's count and fixes subnormals — adopt `__nv_cvt_fp8x2_to_halfraw2` there.
Caveat: 12 vs 18 only matters if ALU-issue bound; at T=1 both are far below the memory
floor (8 weights = 8 B streamed; at 504 GB/s an SM has ~100 B/cycle of budget — issue is
never the limit at T≤8 for a well-pipelined kernel).

## 5. Roofline: dequant-bf16 vs e4m3-MMA for our shapes (K=5120, N=5120..248320)

Arithmetic intensity of a weight-streaming GEMM = 2T FLOP per weight byte (activations +
scales negligible). Compute keeps up with 504 GB/s iff `2T × 0.504 TF ≤ TF_peak`, i.e.
**T* = TF_peak / (2·BW)**:

| rates used | bf16-f32acc T* | e4m3-f32acc T* |
|---|---|---|
| 4070S realistic (**71 / 142 TF**) | **70.5** | **141.1** |
| task brief (82 / 165 TF) | 81.4 | 163.7 |

Utilization table (TF demanded = 1.008·T to stay memory-bound at 504 GB/s):

| T | AI (FLOP/B) | TF needed | % of 71 (bf16) | % of 142 (e4m3) |
|---|---|---|---|---|
| 1 (pair decode) | 2 | 1.0 | 1.4% | 0.7% |
| 2 | 4 | 2.0 | 2.8% | 1.4% |
| 4 | 8 | 4.0 | 5.7% | 2.8% |
| 8 (spec verify) | 16 | 8.1 | 11.4% | 5.7% |
| 16 | 32 | 16.1 | 22.7% | 11.4% |
| 32 | 64 | 32.3 | 45.5% | 22.7% |
| **64 (prefill chunk)** | **128** | **64.5** | **90.9% ← marginal** | **45.5%** |
| 128 | 256 | 129.0 | 182% (compute-bound) | 91.0% |
| 141 | 282 | 142.2 | 200% | 100% ← crossover |

Two corrections to the pure roofline:

1. **Efficiency**: real GEMMs reach 60-80% of tensor peak at M=64 (thin-M wastes tiles;
   the TRT-LLM sm89 kernel uses an M=32 tile — half empty at M=16, full at M=32+). At
   T=64, demanding 91% of bf16 peak means **in practice T=64 prefill is compute-limited
   on the dequant-bf16 path** (realistic effective T* ≈ 45-55), while e4m3-MMA (45%
   demand) keeps it comfortably bandwidth-bound.
2. **Dequant ALU overhead** (bf16 path only): ~9 LOP/shift ops + 1-2 `__hmul2` per
   4 B-values ⇒ ~11 warp-ALU instructions per `m16n8k16` warp-mma. At 71 TF the SM must
   retire one such mma per 8 SM-cycles → 11/32 issue slots ≈ 34% of the SM's total issue
   bandwidth on dequant alone — co-issuable (Marlin proves it, holding ideal speedups to
   batch 16-32 on A10), but it eats into the 60-80% efficiency above. The e4m3-MMA path
   consumes B as **raw fp8 with zero dequant** (scales via the TRT promote: 1 FFMA per
   accumulator per 128-K block, [TRT] §2.4), eliminating that entire tax.

**Answers:** weights-bound until T* ≈ 70 (realistic bf16) — so dequant-bf16 is already
sufficient for *all* T ≤ 32 regimes including 8-wide spec verify, marginal-but-usable at
T=64, insufficient from T ≈ 128. e4m3-MMA doubles the ceiling (T* ≈ 141) and deletes the
dequant ALU tax; it is worth implementing **only for the T=64 prefill GEMMs (and future
T=128 chunks)** — for decode (T ≤ 8) it buys nothing (both paths sit at <12% of compute).

## 6. Activation dynamic quant (per-token-group 1×128): needed? faster variants?

- Reference kernel: vLLM `per_token_group_quant_8bit` —
  `csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu`
  (https://github.com/vllm-project/vllm/tree/main/csrc/libtorch_stable/quantization/w8a8/fp8);
  audited in [VM] §5: 16 threads per (row, 128-elem group), smem two-pass, xor-butterfly
  absmax, scale = absmax/448 (e4m3 max), eps clamp 1e-10, clamp-to-±448 on requantize,
  PDL launch hooks; register-resident fast path (8 thr/group, no smem) for packed UE8M0.
  TRT-LLM's equivalent is `scale_1x128_kernel` / `trtllm::fp8_quantize_1x128` ([TRT] §5).
- Faster/fused variants (2025-2026): FlashInfer
  [`fused_add_rmsnorm_quant`](https://docs.flashinfer.ai/generated/flashinfer.norm.fused_add_rmsnorm_quant.html)
  (add+RMSNorm+quant in one kernel, https://github.com/flashinfer-ai/flashinfer); SGLang
  JIT "[Fused Add+RMSNorm+per-token FP8 quant]" (PR #28101, reviving #21403) incl. direct
  UE8M0 pow2-scale emission; vLLM torch.compile fusion passes
  (`FusedAddRMSNormDynamicQuantPattern`, https://docs.vllm.ai/en/latest/design/fusions/),
  with Inductor sometimes beating the hand CUDA kernel; TRT-LLM's SM100 packed variant
  (lane-subgroup amax, PDL). For us the right move (if ever needed) is the one already in
  the backlog: **fuse group-quant into the preceding RMSNorm** (as done for MXFP4), zero
  extra launches.
- **Do we even need it? No — for W8A16.** Reasoning, confirmed on three legs:
  1. *Quantization math*: DeepSeek-style blockwise FP8 is **weight-only** quantization —
     `weight_scale_inv` is derived from weight magnitudes alone; `activation_scheme:
     dynamic` in the config is a **serving-time instruction** to the runtime, not a
     property of the weights. Nothing in the checkpoint assumes quantized activations.
  2. *Reference behavior*: on sm_89 vLLM itself serves this checkpoint class as Marlin
     **W8A16 with unquantized bf16 activations** ([VM] §1: "the Marlin W8A16 path does
     not quantize activations at all"; PR #5975: "activations have no need to be
     quantized"). The reference deployment *on our own hardware* skips it too.
  3. *Numerics*: e4m3 activations ⊂ bf16 activations in precision; skipping the quant
     strictly **reduces** error vs the unquantized model. Our parity target is the
     fp32/bf16 reference model (per AGENTS), not the FP8-served logits — so skipping is
     not a parity risk at all; it is strictly closer to ground truth.
  Only the W8A8 e4m3-MMA path (§5, T=64 prefill) would reintroduce it — and then it is a
  *new* error source (per-128-group activation rounding + absmax scaling), exactly the
  trade TRT-LLM makes for speed ([TRT] §4).

## 7. Ada L2 (48 MB on 4070 SUPER) and accessPolicyWindow

Sizes (TechPowerUp: 4070 SUPER L2 = 48 MB, https://www.techpowerup.com/gpu-specs/geforce-rtx-4070-super.c4186):

- lm_head (worst case, N=248320, K=5120): scales [1940][40] = 77,600 values = **302 KiB
  (fp32)** / 152 KiB (bf16) — 0.65% of L2. Typical layer (e.g. [10240,5120]): 40×40
  scales = 6.2 KiB.
- Activation x: 5120×2 B = **10 KiB** (T=1, bf16) … 640 KiB (T=64). fp32 20 KiB.
- Everything "small" totals ≤ 2% of L2 even at T=64.

Hit-rate reasoning (no policy): within one GEMM launch every CTA reads x and the scale
rows for its N-tile. Inter-reference interval for x = the weight bytes streamed between
two CTAs touching it ≈ (weights/grid) ≈ 1.3-2.5 MB for lm_head with a ~500-1000-CTA grid
— versus 48 MB of L2 between evictions under perfect LRU: **x is re-referenced ~20-40×
more often than L2 turns over ⇒ natural hit rate ≈ 100%**, scales likewise (they are
touched once per CTA-tile and are 6-302 KiB). The streaming weight bytes cannot plausibly
evict them. Cross-*launch* persistence (the actual purpose of `cudaAccessPropertyPersisting`)
is useless for us: x is rewritten every token/layer and scales change per layer.

**Verdict: the accessPolicyWindow carveout is not needed** — drop it from the backlog
item (synthesis #2); everything small already sits in L2 naturally. If profiling ever
shows scale/x misses during 1.27 GB lm_head streams, the correct (inverse) use of the API
is to mark the *weight stream* `cudaAccessPropertyStreaming` (evict-first) so the default
partition protects x/scales automatically — cheaper than managing a persisting window +
`cudaLimitPersistingL2CacheSize` per stream, and it also keeps `__ldcs` semantics
honest for the big shapes. (Keep the other half of that backlog item: do drop `__ldcs`
on genuinely L2-resident shapes — at ≤12288 rows the weights themselves fit in 48 MB and
benchmarks lie.)

## 8. Recommended idioms per regime (4070 SUPER, Qwen3.8-27B-FP8)

| regime | idiom |
|---|---|
| **T=1-2 (pair decode)** | Marlin-style W8A16: persistent GEMV, raw fp8 bytes in smem via cp.async, dequant-to-bf16 in registers (`(q&0x80008000)\|((q&0x7F007F00)>>4)` + `q<<8`), bf16 scales with 2^120 folded **but kept fp32 in smem and applied via FP32 promote** (our edge over vLLM's bf16 scales, [VM] §6.3), `mma.m16n8k16.bf16` or FMA + `__nv_cvt_fp8x2_to_halfraw2` for CUDA-core path. Compute is ≤3% of peak — bandwidth is everything. |
| **T=3-8 (spec verify)** | same kernel, M-predicated tile (TRT-LLM tolerates M≤8 in an M=32 tile); compute ≤12% — still pure bandwidth. MTP verify reads weights once ⇒ free second token. |
| **T=64 (prefill chunk)** | bf16-dequant path is at 91% of tensor peak — measure; if nsys shows tensor-pipe saturation, port TRT-LLM's `ada_blockwise_gemm` (m16n8k32 e4m3, 32×128×128 tile, 3-stage cp.async, temp-accum + FP32-scale promote) which needs only 45% of peak and zero dequant ALU; costs: per-token-group-128 act quant (fuse into RMSNorm, §6) + FP32 scale upcast at load. |

## Sources

- PTX ISA 9.3 (current, CUDA 13.x): https://docs.nvidia.com/cuda/parallel-thread-execution/ (§9.7.15.5.10 m16n8k32 fragments, §9.7.15.5.14 mma, §9.7.9.22 cvt); archived 8.4 PDF: https://docs.nvidia.com/cuda/archive/12.4.0/pdf/ptx_isa_8.4.pdf
- CUDA↔PTX version mapping: https://milthorpe.org/2022/05/09/matching-cuda-and-nvptx-isa-versions/ ; CUDA 11.8 launch: https://www.hpcwire.com/off-the-wire/nvidia-announces-cuda-toolkit-11-8-new-features/
- ptxas intrinsics reference (FP8 mma sm_89 gate): https://gh.evko.io/crucible-notes/ptxas/intrinsics/tensor.html
- NVIDIA Ada whitepaper: https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf ; 4090/4070S specs: https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889 , https://www.techpowerup.com/gpu-specs/geforce-rtx-4070-super.c4186
- cuBLAS 13.3 docs (scaling modes): https://docs.nvidia.com/cuda/cublas/#scaling-factor-layouts ; cuBLAS 12.9 blog: https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/ ; per-tensor/per-block scaling: https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/ ; open HW-requirement question: https://github.com/nvidia/cudalibrarysamples/issues/310
- vLLM FP8 Marlin PR #5975: https://github.com/vllm-project/vllm/pull/5975 ; blockwise merge PR #16850: https://github.com/vllm-project/vllm/pull/16850 ; Marlin kernel tree: https://github.com/vllm-project/vllm/tree/main/csrc/libtorch_stable/quantization/marlin ; vLLM FP8 docs: https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/fp8/
- MARLIN paper: https://arxiv.org/html/2408.11743v1 ; Red Hat/Neural Magic Marlin article: https://developers.redhat.com/articles/2024/04/17/how-marlin-pushes-boundaries-mixed-precision-llm-inference
- Instruction-throughput table: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#arithmetic-instructions
- FlashInfer fused_add_rmsnorm_quant: https://docs.flashinfer.ai/generated/flashinfer.norm.fused_add_rmsnorm_quant.html ; SGLang PR #28101: https://github.com/sgl-project/sglang/pull/28101 ; vLLM fusion passes: https://docs.vllm.ai/en/latest/design/fusions/
- 4090 FP8 cuBLASLt 660-vs-330 thread: https://forums.developer.nvidia.com/t/ada-geforce-rtx-4090-fp8-cublaslt-performance/250737
- Local: CUDA 13.3 `cuda_fp8.h:377-379`, `cuda_fp8.hpp:563-583` (intrinsic → PTX), `cublasLt.h:932-956` (scale enums); `audits/w2/trtllm-sm89-fp8.md`, `audits/w2/vllm-marlin-fp8.md`, `audits/synthesis.md`.
