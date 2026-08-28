# Audit: exllamav3 GPU kernel architecture (for Insignia's 4-bit + FP8 engine)

Source: `E:\coding\Insignia\exllamav3\` — clone at v1.4.2 (master `5f3c537`, "Bump to v1.4.2").
All paths below are relative to `E:\coding\Insignia\exllamav3\`. Line numbers verified against this checkout.

---

## 1. What EXL3 is (the format we're reading)

EXL3 = QTIP-derived trellis quantization, 1–8 bpw integer bitrates ("K" = bits per weight), with
procedural codebooks and QuIP#-style randomized-Hadamard incoherence on **both** sides of W:

```
W = diag(suh) . H128 . W_hat . H128 . diag(svh)      (reconstruct.cu:147-149)
```

- `trellis` tensor: shape `(k/16, n/16, 16*K)` uint16 (`exl3_gemm.cu:27`). One 16x16 weight tile =
  256 weights x K bits = 16*K uint16 words, bit-packed as a tail-biting trellis stream.
- `suh`/`svh`: per-channel sign flips (+/-1) packed 16-per-int16 at rest (`su`/`sv`), unpacked to
  fp16 vectors for the kernels (`modules/quant/exl3.py:142-158`). No magnitude scales at runtime —
  the Hadamard whitening is what makes fixed codebooks work.
- The Hadamard + sign-flip transform is **fused into every GEMM/GEMV launch** (see §3/§4) — there is
  no separate transform kernel on the hot path.
- Codebooks: three procedural variants, decoded in ~3 integer instructions
  (`exllamav3_ext/quant/codebook.cuh`):
  - cb0 (default "3inst"): `x = x*89226354 + 64248484; lop3(x, 0x8fff8fff, 0x3b603b60, 0x6a); hadd2` (codebook.cuh:59-66)
  - cb1 ("mcg"): `x *= 0xCBAC1FED; lop3(...); hadd2` (codebook.cuh:67-75)
  - cb2 ("mul1"): `x *= 0x83DCD12D; sum = dp4a(x, 0x01010101, 0x6400); hfma(h(sum), 0x1eee /*1/147.7*/, 0xc931 /*-10.39*/)` (codebook.cuh:76-89).
    The codebook value is **affine in the byte-sum of `x*M`** — this one property powers the entire
    int8/dp4a GEMV path (§5).

Per-arch note in codebook.cuh:3-5: integer-MAD inline asm used to beat IMUL on sm_86; as of CUDA
13.2 the plain multiply wins — they keep the hook.

## 2. Kernel inventory / dispatch chain

Entry: `LinearEXL3.forward` (`modules/quant/exl3.py:114-139`) → C++ `exl3_gemm_gr`
(`exllamav3_ext/quant/exl3_gemm.cu:110`), which tries in order:

1. **int8-activation GEMV** (mul1 tensors only, env `EXL3_INT8_GEMV`, default on) — `exl3_gemm.cu:182-186`
2. **QTIP-style fp16 GEMV** for m<=8 (env `EXL3_GEMV`) — `exl3_gemm.cu:222-236`
3. **Autotuned cooperative block-pipelined GEMM** (m in 16-row strips) — `exl3_gemm.cu:238-279`
4. Python-side: **rows > 144** (`AUTO_RECONSTRUCT_THRESHOLD`, `modules/quant/exl3.py:10`) →
   full dequant to fp16 + cuBLAS `hgemm` (`modules/quant/exl3.py:161-218`).

So the small-batch GEMV vs batched GEMM switch points are:
- m == 1..8: GEMV kernels (int8 first, then fp16 QTIP-style, then the regular kernel as fallback)
- m == 9..144: cooperative GEMM with TILESIZE_M = 16 tiles (`exl3_gemm_kernel.cuh:37-50` loops m in strips of 16)
- m > 144: dequant + cuBLAS (prefill). Fused `reconstruct_had_slice` emits original-basis weights
  (both Hadamards folded into the dequant kernel) so prefill needs **zero** extra Hadamard launches
  — saves ~14% of long-chunk prefill GPU time; kernel costs ~4x plain reconstruct, breakeven at
  rows ~400-900, enabled at rows >= 1024 (`modules/quant/exl3.py:172-184`, kernel at
  `exllamav3_ext/quant/reconstruct.cu:147-308`). Column-sliced at 32768 (`MAX_RECONSTRUCT_SLICE_N`)
  to bound the fp16 workspace.

FP8 note: **there are no fp8 compute kernels**. FP8 e4m3 checkpoints (Mistral-Small-4 per-expert
scalar scales, DeepSeek-V4 [128,128] bf16 blocks / E8M0 grid) are dequantized to fp16 at load
(`modules/linear.py:133-217`, block shape derived from the scale grid) and run through the fp16
paths. cuBLAS is fp16-in/fp32-compute (`exllamav3_ext/hgemm.cu:65-75`). dspark has an e4m3
KV-cache fake-quant (groups of 64, ue8m0 scale) but that's attention-side
(`modules/arch_specific/dspark.py:31-38`). **Implication for Insignia: on Ada, treat FP8 as a
storage format; dequant-to-fp16 tiles + cuBLAS (or custom mma.m16n8k16.f16) is the proven recipe —
Ada has no fp8 tensor cores anyway, only fp16/bf16 at 2x fp32 rate.**

## 3. The fp16 GEMV decode kernel (QTIP-style) — `exllamav3_ext/quant/exl3_gemv_kernel.cuh`

Header comment (lines 1-22) is a complete design doc. Key structure:

- **Eligibility**: K in 2..4, m <= 8 (EXL3_GEMV_MAX_M), size_k/n % 128 == 0 (`exl3_gemv.cu:110-114`).
- **Two configs** (exl3_gemv_kernel.cuh:143-146):
  - CFG0 "narrow": 512 threads = 16 warps (WK=16 k-splits), 2 n-tiles/warp = 32 output cols/block,
    prefetch ring PF=4, fp32-fold cadence FOLD=4. Wins at attention-projection sizes (n <= 4096).
  - CFG1 "wide": 256 threads = 8 warps (WK=8), 4 n-tiles/warp = 64 cols/block, PF=2, FOLD=2.
    Wins at large-n/small-k FFN sizes.
- **MMODE 0** is the m==1 fast path (lane<4 guard); **MMODE 1** covers m 2..8 with row-guarded
  fragment loads (exl3_gemv_kernel.cuh:196, 341-342).
- **No block-wide sync in the main loop** — warps split k and never synchronize (line 7-9).
  B (trellis) streams straight to **registers** with `__ldcs` (evict-first; weights are single-use)
  behind a compile-time-sized register prefetch ring `uint32_t pf[PF][LOADS]`
  (exl3_gemv_kernel.cuh:226-263). The ring indices MUST be compile-time or the array lands in
  local memory (comment at line 225).
- **Trellis window extraction in-warp without smem**: each lane's two source words are at
  lane-computable indices, fetched via `__shfl_sync` from the words already in registers
  (exl3_gemv_kernel.cuh:297-316). Optional warp-private smem staging variant (SMEM_STAGE) for A/B.
- **MMA trick for GEMV**: `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` with the **A operand
  supplied as two FragB halves** (`mma_ab_h`, exl3_gemv_kernel.cuh:35-49) — the m<=8 activation
  rows live permanently in registers (2 half2 loads per lane per k16 slice, lines 275-281), no
  ldmatrix, no smem for A. B tile decodes directly into FragB register layout.
- **fp16 accumulation folded to fp32 every FOLD iterations** (lines 322-333): 2 fp16 accumulators
  per (tile, frag) folded into fp32 on a fixed cadence; final cross-warp reduce in fp32 smem
  `sh_red[WK][ROWS][COLS]` (lines 337-371) — atomics-free, one __syncthreads, grid-stride over
  32/64-col output groups.
- **Cooperative launch** with exactly two grid.sync: one after the fused input Hadamard stage, one
  before the output Hadamard stage (lines 156-175, 374-401). The Hadamard stage itself is
  warp-strided over the whole grid (one warp = 128 elements), fused with suh sign flips.
- **Heuristic** (`exl3_gemv_cfg`, exl3_gemv.cu:46-72): narrow config wins (up to ~30%) when its
  grid (size_n/32 blocks) fits a single co-resident wave (`size_n/32 <= occupancy*num_sms`,
  cached via cudaOccupancyMaxActiveBlocksPerMultiprocessor, exl3_gemv.cu:125-135); in the 1-2 wave
  zone it loses unless k is small. Wide covers n>=8192 && k<=4096. **Measured claims (3090, 4bpw):
  narrow +15-60% at n<=4096, wide +8% at large-n; "Ada/Blackwell are memory-bound here and keep
  the regular kernel"** (exl3_gemv.cu:22-26) — but note K==2 always routes here, and K==3 routes
  here on Ada (`K == 3 && cc == CC_ADA`, exl3_gemv.cu:65). **For 4bpw on Ada the QTIP GEMV is
  declined and the pipelined GEMM kernel serves decode.**
- Grid capped at max co-resident blocks; env `EXL3_GEMV_SMEM` toggles extraction style.

## 4. The main cooperative GEMM (prefill/batch) — `exllamav3_ext/quant/exl3_gemm_inner.cuh`

Classic Ampere-style block-pipelined tensor-core GEMM, specialized hard:

- **Four template shapes** (`exl3_kernel_map.cuh:53-60`), TILESIZE_M always 16:

  | shape | K-tile | N-tile | SH_STAGES | FRAG_STAGES | blockDim |
  |-------|--------|--------|-----------|-------------|----------|
  | 1 (unused) | 16 | 128 | 6 | 5 | 256 |
  | 2 | 32 | 128 | 4 | 3 | 512 |
  | 3 | 32 | 256 | 4 | 3 | 512 |
  | 4 | 16 | 512 | 4 | 3 | 256 |

  blockDim = 256 * TILESIZE_K/16; `sub_k = threadIdx.x / 256` splits the k-tile across
  256-thread groups (exl3_gemm_inner.cuh:74-77). SMEM_MAX = 90 KB (sm_86 cap; they never raised
  it for Ada's 100 KB).
- **cp.async pipeline**: 16-byte `cp.async.cg` (ptx.cuh:159-168), A and B tiles, SH_STAGES deep;
  wait allows SH_STAGES-2 groups pending (`cp_async_wait<SH_STAGES - 2>`, line 633). B loads can
  use `createpolicy.fractional.L2::evict_first` cache hints (`cp_async_stream`, ptx.cuh:172-184).
- **A staged through smem with XOR swizzle** (`col ^ ((row >> shift) & mask)`, lines 52-54,
  120-124) for bank-conflict-free `ldmatrix.x4` (`ldsm4`, ptx.cuh:203-212).
- **B staged raw (packed trellis words), dequantized on the way to registers** (`load_frags`,
  lines 287-299, calls `dq_dispatch` from exl3_dq.cuh), with FRAG_STAGES (3) deep register
  buffering. The main loop is hand-unrolled via FSTAGE macros with per-FRAG_STAGES orderings
  (lines 666-732) — the comment at 662-664 explains two different loop iterations are needed so
  the compiler doesn't spill the fragment arrays to local memory.
- **fp16 accumulation (sm_86 only)**: `EXL3_GEMM_H_ACC` (lines 11-20, 571-581) — GA10x runs
  fp32-accum HMMA at half rate; accumulating in fp16 and folding into fp32 once per k-slice is
  ~14% faster at bsz1 with ~1% output RMS error at k=4096. Gated to `__CUDA_ARCH__ == 860`;
  Ada runs full fp32 accumulate.
- **Split-k across blocks**: grid = num_sms slices over (tiles_k x tiles_n); per-column cross-block
  reduction via a **spin-lock global barrier**: thread 0 polls `ld.global.acquire.gpu.b32` until
  the lock equals its slice index, non-first blocks read back the fp16/fp32 partial sum from
  global C, non-last blocks write it back, last block writes row-major (lines 588-627);
  `barrier_release` uses `fence.acq_rel.gpu` + `red.relaxed.gpu.global.add`, last resets to 0
  (ptx.cuh:103-139). Slices within a column are processed in **reverse order** so the block doing
  the bottom slice proceeds immediately (comment 586-587).
- **Atomics-free threadblock reduction** across sub_k groups via sh_c smem with hand-unrolled
  store/add sequences per TILEBLOCKS_K in {2,3,4} (lines 315-423) — a tree, no atomics, and a
  "small" variant (m<=8) that only moves 2 of 4 accumulators.
- **Epilogue**: last block writes the 16xN tile to smem, then warp-per-128-col fused output
  Hadamard + svh + optional fp32->fp16 (lines 426-480).
- **Autotuner** (`coop_autotune.cu`): FNV-hashes (m bucketed to pow2<=16, k, n, K, c_fp32, device,
  cc, num_sms, cb) — `gemm_autotune_hash` (exl3_gemm.cu:53-82) — benchmarks all compatible shape
  candidates with CUDA events on first encounter, caches in-memory and to disk
  (`%LOCALAPPDATA%/exllamav3/autotune/coop_autotune_v1.bin`, coop_autotune.cu:75-118, versioned
  magic header). Also tunes num_sms.
- **Shape selection per arch** (`select_gemm_shape`, exl3_kernel_map.cu:23-75). Ada branch (8.9):
  mod-256 && K<=3: k<=2048 non-multi → shape2; n<4096 && k<=12288 → shape2 else shape3;
  n<=16384 → shape2; mod-512 && n>=32768 → shape4; mod-256 → shape3; else shape2.
- Each (K, cb) compiles as a separate translation unit (`comp_units/exl3_comp_unit_K_cbX.cu`) to
  keep register allocation per-file; kernel pointer tables `[K][cb][shape]` (exl3_kernel_map.cu:95-130).
- All launches are `cudaLaunchCooperativeKernel` with SMEM_MAX dynamic smem
  (exl3_gemm.cu:296-304); `cudaFuncAttributeMaxDynamicSharedMemorySize` set once per kernel
  (tracked in a set, lines 290-295).

## 5. The int8/dp4a GEMV — `exllamav3_ext/quant/exl3_gemv_int8_kernel.cuh`

The most Insignia-relevant kernel (it is their answer to our Q8/DP4A experiment). For mul1 (cb2)
tensors only. Design doc in the header, lines 20-53:

- **Core identity**: `dp4a(x * 0x83DCD12D, splat(int8 a), acc)` evaluates
  `codebook(x) * a` in one instruction with exact int32 accumulation, because the codebook value
  is affine in the byte sum. Epilogue recovers
  `y = k_inv * (q*acc1 + q2*acc2) + (1024*k_inv + k_bias) * (q*sum1 + q2*sum2)`
  (lines 24-31, 729-735).
- **Residual mode** (`EXL3_INT8_GEMV=1`): activation rounding error r = a - q*i (+/-q/2 by
  construction) is quantized at q2 = q/254 and accumulated by a **second dp4a chain sharing the
  decoded weight products** → ~15-16 bit effective activation precision, KL at parity or better
  vs the fp16 kernel (lines 28-31). Plain mode (=2, default) is ~0.9% output RMS deviation.
- **Two kernels**:
  - `exl3_gemv_int8_sq_kernel` (m==1, plus m==2 plain mode): **regular launch, NO cooperative
    machinery, NO atomics, NO grid sync**. Per-k-slice activation scales (q_s = slicemax/127,
    deterministic because fmax is order-independent — duplicating blocks agree bit-exactly);
    units write exclusive per-slice int32 partials with plain stores; a per-256-column
    completion counter (`atomicAdd(&counters[nb256],1) == ksplit-1`) gates an inline epilogue that
    combines `sum_s q_s*acc_s` in fixed slice order. Partial reads bypass L1 with `__ldcg`
    (lines 764-776, 890-947, 1027-1038). Workspace counters are zeroed once at allocation and
    self-reset (lines 970-776 comment).
  - `exl3_gemv_int8_coop_kernel` (m==1 fallback): cooperative, atomicAdd into 2*size_n int32
    accumulators that double as the `locks` arg; phase 1a fuses input Hadamard + accumulator
    zeroing + per-block partial max of transformed activations (lines 1069-1115); m==1 row scale
    re-derived per block from the partial maxes (fmax order-independence = determinism, no extra
    sync); block 0 computes the exact epilogue sums concurrently with other blocks' GEMV work
    (lines 1168-1192).
- **Wide unit (K=4)** `gemv_int8_unit_wide` (lines 320-460): a warp owns an **adjacent 256-column
  block pair**; lane l's uint2 = words {2m, 2m+1} of a contiguous 64-word region = exactly the
  primary words of runs t = 8*(2m), 8*(2m+1), which scatter to the SAME two n values — splats
  merge into two uint4 smem loads, single-stage butterfly (xor-1 shuffle), boundary word via one
  shuffle. **256 B contiguous per warp per block row, with 2-row register prefetch (r0/r1/r2
  rotation)** (lines 349-356, 414-415). This is the DRAM-saturation geometry.
- **Narrow unit** (generic K): pointer-based extraction straight from global (lines 611-644).
- **Smem-staged unit** (K = 3, 5, 7): warp-private cp.async ring, `GEMV_STAGE_D = 4` rows deep,
  one commit group per row, warp-private slice → **no block-level sync, only __syncwarp**
  (lines 650-701; routing predicate `gemv_int8_stage_smem` line 705-708, measured on 3090:
  K=3 -13%, narrow wins 2/6/8 which are ALU-floor/DRAM-bound).
- **Work decomposition**: units = (n/256) x ksplit, fine-grained ~4 per block "because with
  grid.sync at the end of the phase, tail imbalance stalls the whole grid" (lines 1145-1148);
  rows_per multiple of 8; single-wave rule with half-wave floor up to 32 rows (sq kernel,
  lines 976-989).
- **Occupancy**: "Natural register allocation (2-3 resident blocks/SM) measures faster than
  forcing higher occupancy: with wide loads the per-warp ILP is worth more than the extra warps"
  (line 51-53). Grid = MIN(occupancy*num_sms, 1024).
- **Workspace**: fixed 16 MB per device, allocated once, NEVER reallocated because the pointer is
  baked into captured CUDA graphs (exl3_gemv_int8.cu:97-115).
- **Shared-memory carveout pinning**: `cudaFuncAttributePreferredSharedMemoryCarveout =
  cudaSharedmemCarveoutMaxShared` — these kernels interleave with the tensor-core kernels hundreds
  of times per token, and a smaller carveout makes the GPU drain and reconfigure SMs on every
  transition, "measured at ~4 us per launch in graph replay" (exl3_gemv_int8.cu:153-163, 279-290).
- **Per-arch K gate** (`exl3_gemv_int8_max_k`, exl3_gemv_int8.cu:37-52): int8 path accepts K<=5 on
  Ampere/Ada, K<=6 on Hopper/Blackwell. Measured: 3090 -29/-22/-9/-6% at K=2/3/4/5, DRAM-bound
  from K=6; **"Ada is marginal at K=6 (4090: -0..+7%, residual mode loses) and keeps the
  conservative gate"**; H200 +16% e2e at K=6 because the fp16 kernel is INT-throughput-bound there.
  → On Ada at 4bpw the int8 trick is roughly a wash (DRAM-bound), which matches Insignia's own
  finding that direct FP32 accumulation from packed nibbles beat Q8/DP4A on decode GEMV.
- **Gate/up unfusing**: when int8 GEMV is on, same-input tensor pairs are split out of the batched
  MGEMM when each matrix is wide enough to fill the GPU alone (env `EXL3_MGEMM_N_THRESHOLD`
  default 8192) — batching small GEMVs into one launch hurts when each could saturate
  (doc/env_vars.md "EXL3 GEMM / GEMV" section).

## 6. Reduction / synchronization toolkit

- `reduction.cuh`: standard warp-xor + block-via-32-slot-smem reductions; noteworthy:
  `warp_reduce_sum_last_k/first_k` (zero lanes then reduce, no separate mask) and
  `shuffle_had_f4x32` (hadamard_inner.cuh:17-44) packs 4 floats into 2x 64-bit shuffles per
  butterfly stage — halves the shuffle count of a 32-point transform.
- **Spin-lock split-k barrier** (ptx.cuh:103-139): acquire = thread-0 polling
  `ld.global.acquire.gpu.b32` + __syncthreads; release = `fence.acq_rel.gpu` +
  `red.relaxed.gpu.global.add`, last block plain-stores 0. No atomics in the data path.
- **Sense-reversing group barrier** for >SM-count groups (Blackwell path, ptx.cuh:317-348):
  cuda::atomic_ref counter + sense flag + `__nanosleep(32)` backoff.
- **Completion-counter epilogue gate** (sq kernel, exl3_gemv_int8_kernel.cuh:1027-1038):
  `__threadfence(); __syncthreads(); if (t==0) sh_last = (atomicAdd(&counters[nb256],1)==ksplit-1)`
  — ksplit-th contributor runs the epilogue and resets the counter. Reads of other blocks'
  partials use `__ldcg` (L2, skip L1).
- MoE scheduler: self-resetting ticket system (atomicAdd next-ticket, acq_rel retirement,
  exl3_moe_kernel.cuh:261-282).

## 7. MoE fused expert MLP — `exllamav3_ext/quant/exl3_moe_kernel.cuh`

Single launch per layer for the whole expert FFN: gather+input-Hadamard → g/u GEMM (calls
`exl3_gemm_kernel_inner` directly, MOE tiles 16x32xN, 3/3 stages, exl3_moe_common.cuh:12-15) →
fused out-Hadamard(gate)*act*in-Hadamard(d) (`had_hf_r_128_guad_inner`, hadamard_inner.cuh:283-413)
→ d GEMM → out-Hadamard + weighted scatter-add with atomicAdd reshuffled through smem for
coalescing (`had_hf_r_128_d_inner`, hadamard_inner.cuh:417-473). Dynamic ticket scheduling for
load balance; CPU-expert offload path uses AVX512-VBMI byte-gather extraction + a band-swizzled
weight layout so "each GEMV band streams sequentially from DRAM instead of short strided runs
(+45-75% cold decode GEMV throughput on a 7960X, reaching the sequential-read roofline)"
(doc/env_vars.md EXL3_MOE_CPU_SWIZZLE) and a `cuStreamWaitValue32/WriteValue32` front-end
handshake instead of spin kernels.

## 8. CUDA graph parameter patching — `exllamav3_ext/graph.cu`

- Kernels record their mutable arg slots at capture time (`record_param(kernel, param_id,
  arg_index)`, e.g. exl3_gemm.cu:206-218); at capture_end the host walks graph nodes in order and
  binds sites to node+kernelParams slots (graph.cu:68-111). At replay only **changed** args are
  memcmp'd and patched via `cudaGraphExecKernelNodeSetParams` (graph.cu:142-183).
- GEMV/GEMM/int8 kernels share the **identical 10-arg signature** precisely so graph patch sites
  are interchangeable between kernel choices (exl3_gemv_kernel.cuh:16-17, exl3_gemv.cu:15-17).
- cuBLAS nodes handled by re-binding stream+workspace at replay (graph.cu:131-140).
- BC_* C++ paths capture the whole attention/MLP block as one graph per shape and patch only
  input/output/position/cache pointers (doc/env_vars.md EXL3_BC_ATTN).

## 9. lm_head / embedding gathers

Nothing exotic: lm_head is a regular (usually 6-bpw, `-hb` flag) EXL3 Linear — decode logits go
through the same GEMV/GEMM chain; tied embeddings alias the embed tensor
(architecture/*.py `tie_word_embeddings` blocks). Embedding forward is a torch index_select with
pinned-staging async upload when the table lives on CPU (`modules/embedding.py:146-168`).

## 10. Steal list for Insignia (INSIG4 + FP8, RTX 4070 SUPER / sm_89)

Ranked by expected value:

1. **L2 policy discipline for the weight stream**: `__ldcs` (evict-first) on GEMV weight loads
   (exl3_gemv_kernel.cuh:226-232) and `createpolicy.L2::evict_first` cp.async in GEMM
   (ptx.cuh:172-184). Single-use quantized weights should not pollute L2 that the KV cache /
   activations need. Trivial to add to Insignia's MXFP4/INSIG4 matvec.
2. **The dp4a-affine-codebook result** (their int8 GEMV) is directly comparable to our Q8/DP4A
   experiment — and their measurement agrees with ours: on Ada at 4bpw the decode GEMV is
   DRAM-bound, int8 activation tricks are a wash (exl3_gemv_int8.cu:38-45: "Ada is marginal",
   4090 -0..+7%, residual loses). Their escape hatch for precision loss (error-feedback residual
   second dp4a chain, exl3_gemv_int8_kernel.cuh:28-31) is the interesting bit if we ever revisit
   integer accumulation.
3. **FP8 strategy**: dequant-at-load (or dequant-in-kernel) to fp16 + tensor-core mma.m16n8k16 /
   cuBLAS. They fold the [128,128]-block bf16 scales into an fp16 weight copy at load
   (linear.py:133-146). For a VRAM-tight engine, Insignia can instead dequant fp8→fp16 tiles
   inside the GEMM's smem staging (same place EXL3 dequants its trellis: exl3_gemm_inner.cuh:287-299)
   and keep fp8 in VRAM.
4. **Fused Hadamard/sign/scale into GEMM+GEMV launch** (one cooperative kernel per linear, two
   grid.syncs total). Insignia's equivalent: fuse MXFP4/INSIG4 scale multiply into the matvec
   main loop (we already do this) and consider fusing activation quantization + partial-max
   tracking into the same launch like the int8 coop kernel phase 1a
   (exl3_gemv_int8_kernel.cuh:1069-1115).
5. **Prefill switch point + original-basis dequant**: m>threshold → dequant whole layer to fp16
   (threshold 144 rows, their number; ours will differ) and cuBLAS it, with the dequant kernel
   emitting already-transformed weights so zero extra elementwise kernels run
   (modules/quant/exl3.py:161-218). For INSIG4: a `reconstruct` that emits fp16 W (fold group
   scales) sliced at ~32K columns.
6. **Register prefetch ring for the B stream** in GEMV (pf[PF][LOADS] with compile-time indices,
   exl3_gemv_kernel.cuh:225-263) — replaces smem staging entirely for the weight stream in
   small-m kernels; the "indices must be compile-time or pf lands in local memory" warning is
   load-bearing.
7. **A-in-registers MMA for GEMV** (`mma_ab_h`, exl3_gemv_kernel.cuh:35-49): for m<=8, put the
   activation fragment in registers once per k-slice and issue mma against every n-tile without
   ldmatrix or smem. Pairs beautifully with nibble decode straight into FragB layout.
8. **Immediate-argument bitfield macros** FSHF_IMM / BFE16_IMM (ptx.cuh:314-315) — funnel-shift
   and bfe with literal shift baked into the instruction string; cheaper than the bfe-with-register
   form the compiler picks on its own. Directly applicable to E2M1 nibble unpacking (our
   nibble->LUT could use BFE16_IMM + a 16-entry half table via byte-perm / lop3 pairs).
9. **Shared-memory carveout pinning** (`cudaSharedmemCarveoutMaxShared` on every kernel that
   alternates with big-smem kernels) to avoid ~4 us/launch SM reconfigure in graph replay
   (exl3_gemv_int8.cu:153-163). Matters once Insignia graphs the decode loop.
10. **Fixed never-reallocated workspace + self-resetting counters** so graph-captured pointers
    never dangle (exl3_gemv_int8.cu:97-115), and **memcmp-gated graph arg patching**
    (graph.cu:142-183) instead of re-capture.
11. **Autotune cache disk persistence + per-(K,cb) translation units**: cheap idea for keeping
    nvcc from cross-contaminating register allocation between kernel variants
    (exl3_kernel_map.cu / comp_units layout), and the FNV shape hash → disk cache
    (coop_autotune.cu) is a nice pattern for Insignia's build bench harness.
12. **Completion-counter atomics-free m==1 GEMV** (sq kernel pattern, §5): per-slice exclusive
    partials + counter-gated deterministic epilogue — removes ALL global coordination (no
    cooperative launch, no locks); attractive for Insignia's GEMV if we ever split k across blocks.
13. **Band-swizzled weight layout for CPU GEMV** (+45-75% cold decode throughput on Zen 3 by
    making the weight stream sequential per band, doc/env_vars.md EXL3_MOE_CPU_SWIZZLE) — highly
    relevant to Insignia's CPU+GPU mixed compute / NVMe paging hierarchy.
14. **Things NOT to copy**: fp16 accumulation (Ada HMMA fp32-acc is full rate; it's an sm_86-only
    fix), the 90 KB smem cap (Ada has 100), cooperative launches everywhere (they need them for
    the fused had stages; a plain kernel + separate tiny had kernel can be faster to schedule at
    bsz1 — their own int8 sq kernel abandons coop for exactly this reason).

## Appendix: exact constants quick-reference

- GEMV narrow: blockDim 512, 16 warps, WK=16, WNT=2, PF=4, FOLD=4, 32 cols/block, launch_bounds 512
  (exl3_gemv_kernel.cuh:139-149).
- GEMV wide: blockDim 256, WK=8, WNT=4, PF=2, FOLD=2, 64 cols/block.
- GEMM shapes table: exl3_kernel_map.cuh:53-62 (see §4).
- cp.async: 16 B, `cp.async.cg`, wait_group SH_STAGES-2 (exl3_gemm_inner.cuh:630-635).
- int8 sq kernel: NUM_THREADS 256, GEMV_STAGE_D 4, SQ_MINROWS 16, SQ_ROWS_MAX 512,
  SQ_COUNTERS_CAP 4096, workspace 16 MB (exl3_gemv_int8_kernel.cuh:54, 310-318;
  exl3_devctx.cuh:18).
- Locks buffer: MAX_TILES_C 1M ints + 2*1024 barrier counters + MoE sched (exl3_devctx.cuh:7-15).
- Hadamard scale constant 0.088388347648f = 1/sqrt(128), hardcoded everywhere.
- int8 max K: 5 (Ampere/Ada), 6 (Hopper/Blackwell) (exl3_gemv_int8.cu:46-52).
- Autoreconstruct threshold 144 rows; reconstruct slice 32768 cols; fused-reconstruct breakeven
  rows>=1024 (modules/quant/exl3.py:10-12, 184).
