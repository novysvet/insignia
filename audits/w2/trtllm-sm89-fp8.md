# TRT-LLM sm89 (Ada) FP8 Blockwise GEMM — Audit

Clone audited: `E:\coding\Insignia\TensorRT-LLM` @ `v1.3.0rc24-454-gd9329fb8d3` (Aug 2026).
Scope: sm_89 FP8 blockwise (DeepSeek-style 1x128 activation x 128x128 weight scales) GEMM
kernels, scale promotion, scale dtype handling, W8A16 vs W8A8 on Ada, per-token-group
activation quant. All paths below are relative to `E:\coding\Insignia\TensorRT-LLM\`.

Headline: TRT-LLM ships a **hand-rolled CuTe sm89 kernel** (`ada_blockwise_gemm`) that runs
DeepSeek FP8 checkpoints W8A8 on Ada using `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
with FP32 scales and a per-128-K-block promote. It is compiled for real sm_89
(`cpp/tensorrt_llm/kernels/cutlass_kernels/CMakeLists.txt:213`:
`set_cuda_architectures(fp8_blockscale_gemm_src 89 90 100f 120f)`) and is a first-class
dispatch target (`case 89:` in the torch op). There is **no FP8 W8A16 path** anywhere.

---

## 1. Kernel inventory

| File | Role |
|---|---|
| `cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/ada_blockwise_gemm/sm89_fp8_gemm_1d1d.cuh` (454 L) | Hand-rolled CuTe sm89 blockwise GEMM kernel (dense + bmm) |
| `.../ada_blockwise_gemm/sm89_utils.cuh` (249 L) | MMA atom (inline PTX), traits: tiles, stages, cp.async atoms, smem layouts, scale copies |
| `.../fp8_blockscale_gemm_kernel.cuh` (1191 L) | Quant kernels (scale_1x128 / scale_128x128), arch dispatch (`gemm_dispatch_sm89` etc.), sm120/deep_gemm paths |
| `.../fp8_blockscale_gemm.cu` (335 L) | Runner: workspace math, internal-quantize entry points |
| `.../fp8_blockscale_mma_utils.cuh` (620 L) | Hopper wgmma e4m3 variants (m64n{16..192}k32) — NOT used on Ada |
| `.../fp8_blockscale_quant_packed.cu` (283 L) | SM100-only fused 1x128 quant + UE8M0 packing (contrast) |
| `.../sm120_blockwise_gemm/*` | SM120 path; scales as packed UE8M0 int32 |
| `.../fp8_rowwise_gemm/fp8_rowwise_gemm_kernel_template_sm89.h` | Cutlass 2.x sm89 FP8 **rowwise** (per-tensor/per-channel) GEMM — StreamK |
| `cpp/tensorrt_llm/thop/fp8BlockScalingGemm.cpp` (453 L) | Torch op `trtllm::fp8_block_scaling_gemm[_impl|_bmm|_moe_gemm]`, arch switch incl. `case 89` |
| `cpp/tensorrt_llm/thop/fp8Quantize.cpp` (263 L) | Torch op `trtllm::fp8_quantize_1x128` (the per-token-group quant entry) |
| `tensorrt_llm/_torch/modules/linear.py:1158-1240` | `FP8BlockScalesLinearMethod.apply` — model-level flow incl. sm89 |
| `tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/trtllm_quant.py:99-175` | `trtllm_finegrained_fp8_linear` — HF FineGrained FP8 → TRT-LLM op fusion |

Naming: "1d1d" = 1D activation scales (per row x per 128-K group) x 1D-per-tile weight scales
(one scalar per 128x128 block). This is exactly the DeepSeek/Qwen3.8-FP8 recipe.

## 2. The sm89 blockwise kernel — configuration

Dispatch (`fp8_blockscale_gemm_kernel.cuh:681-715`, dense; `:1082-1122`, strided-batch):

```cpp
static constexpr int Stages = 3;
using TileShape = cutlass::gemm::GemmShape<32, 128, 128>;   // M=32, N=128, K=128
using KT = ada_blockwise_gemm::AdaBlockwiseGemmTraits<float_e4m3_t, bfloat16_t, float,
            float /*ElementBlockScale*/, Stages, 32, 128, 128>;
// block = 128 threads (4 warps), grid = (ceil(M/32), ceil(N/128), [num_problems])
cudaFuncSetAttribute(sm89_fp8_gemm_1d1d_impl<GemmKernel>,
    cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);   // kSmemSize ≈ 61,952 B
```

Exactly **one tile config is instantiated**: 32x128x128, 3 stages. (sm120 gets 32/64x128x128x4.)

Traits (`sm89_utils.cuh`):
- `kWarpsCount = 4`, `kThreadCount = 128` (`:135-136`).
- Scale granularity (`:138-140`): `ScaleGranularityM = 1`, `ScaleGranularityN = 128`,
  `ScaleGranularityK = 128` → SFA shape [M, K/128] (per-token-group), SFB [N/128, K/128].
- Static asserts (`:126-128`): `TileM % 16 == 0`, `TileN % 32 == 0`, `TileK % 32 == 0`;
  `Stages >= 2` (`:224`).
- MMA permutation (`:150-159`): `kMmaPerm{M,N,K} = 32`, so `NUM_GROUP_N = TileN/32 = 4`,
  `NUM_GROUP_K = TileK/32 = 4` (4 mma k-steps per k-tile per n-group).
- Shared memory (`:226-245`): union of load (A + B + SFA + SFB, staged) and store (O) structs,
  `aligned_struct<128>`. For 32x128x128x3: A 3*32*128 = 12,288 B + B 3*128*128 = 49,152 B +
  SFA 3*32*4 = 384 B + SFB 3*4 = 12 B (padded to 128s) ≈ **61.5 KB > 48 KB** → opt-in smem
  required; only **1 CTA/SM** on Ada (100 KB smem/SM).

### 2.1 MMA instruction (sm89_utils.cuh:41-82)

Ada FP8 MMA is hand-written PTX wrapped in a cute atom:

```cpp
struct SM89_16x8x32_F32F8F8F32_TN   // cute namespace, sm89_utils.cuh:41
{
    using DRegisters = float[4]; using ARegisters = uint32_t[4]; using BRegisters = uint32_t[2];
    CUTE_HOST_DEVICE static void fma(...) {
#if defined(CUTE_ARCH_MMA_F32_SM89_ENABLED)   // __CUDA_ARCH__ >= 890  (line 33-35)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
              "f"(c0),"f"(c1),"f"(c2),"f"(c3));
#endif
    }
};
template <> struct MMA_Traits<SM89_16x8x32_F32F8F8F32_TN> {   // :69-82
    using ValTypeD = float; using ValTypeA = float_e4m3_t; using ValTypeB = float_e4m3_t;
    using Shape_MNK = Shape<_16,_8,_32>; using ThrID = Layout<_32>;
    using ALayout = Layout<Shape<Shape<_4,_8>,Shape<_4,_2,_2>>, Stride<Stride<_64,_1>,Stride<_16,_8,_256>>>;
    using BLayout = Layout<Shape<Shape<_4,_8>,Shape<_4,_2>>,  Stride<Stride<_32,_1>,Stride<_8,_128>>>;
    using CLayout = SM80_16x8_Row;
};
```

- Yes: `mma.sync m16n8k32` with **e4m3 x e4m3 inputs, f32 accumulate** (A regs 4xu32 = 16 fp8,
  B regs 2xu32 = 8 fp8). This is the Ada-only FP8 tensor-op shape (Hopper uses wgmma
  m64nXk32 instead — see `fp8_blockscale_mma_utils.cuh:29+`).
- TiledMMA (`sm89_utils.cuh:156-159`): `TiledMMA<MMA_Atom, Layout<2,2,1>, Tile<32,32,32>>`
  → CTA-wide mma tile 32x32 per k-step, 4 warps. A BF16 fallback atom
  `SM80_16x8x16_F32BF16BF16F32_TN` also exists in the traits (`:97-104`) — same kernel shell
  can run BF16 x BF16 (not used by the dispatch, but available).

### 2.2 cp.async pipeline (gmem → smem)

Classic sm80-style multi-stage cp.async, **no TMA, no mbarriers**:

- Copy atom (`sm89_utils.cuh:162-163`):
  `Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, ElementInput>` — 16 B `cp.async.cg`
  per thread for A and B. Thread/value layout `Layout<Shape<_16,_8>>` x `Shape<_1,_16>>`
  → 128 threads move a 16x128 fp8 slab (2 KB) per iteration.
- Scale copies (`:193-201`): `Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<float>, float>` —
  4 B cp.async; SFA tiled copy distributes TileM(=32) row-scales across threads;
  SFB copy uses a **stride-0 broadcast layout** (`GmemLayoutTVSFB ... Stride<_0,_0>`, `:199`)
  so all threads load the same one scalar for the N=128 tile.
- Pipeline (`sm89_fp8_gemm_1d1d.cuh:262-345`): prologue fills `Stages-1` buffers with
  `cute::copy_if` (predication on M/N residue) + `cute::cp_async_fence()`; steady state uses
  `cute::cp_async_wait<KT::Stages - 2>()` + `__syncthreads()` and rotates
  `smem_pipe_write = smem_pipe_read` ring indices (`:378-380`); the gmem→smem refill is
  issued **inside** the `n_block == 0` iteration (`:365-377`) to overlap with the 4 n-group
  mmas; explicit "load tail" loop with `cp_async_wait<Stages-3-WaitIndex>()` (`:393-430`)
  and a final "mma tail" (`:431-445`).
- smem→RF (`sm89_utils.cuh:173-175`): `Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>` —
  `ldmatrix.x4` per warp. smem A/B layout: `composition(Swizzle<3,4,3>{}, Layout<16x128>)`
  tiled to (TileM, TileK, Stages) (`:169-171`) — 128 B rows, K-major, 8-row XOR swizzle.
  B is pre-partitioned per n-group: `tXrB(_, _, n_block)` fragments for all 4 n-groups live
  in registers simultaneously (`sm89_fp8_gemm_1d1d.cuh:303-316, 362-364`).

### 2.3 Scale layouts in gmem

`sm89_fp8_gemm_1d1d.cuh:78-92`:

```cpp
uint32_t const ScaleM = (((M + 3) >> 2) << 2);          // A-scales padded to 4 rows
uint32_t const ScaleN = (N + 127) / 128, ScaleK = (K + 127) / 128;
mSFA_mk = make_tensor(..., shape(ScaleM, ScaleK), stride(_1, ScaleM)); // [K/128][M] col-major
mSFB_nk = make_tensor(..., shape(ScaleN, ScaleK), stride(ScaleK, _1)); // [N/128][K/128] row-major
```

So: **SFA = k-block-major, rows contiguous, M padded to multiple of 4; SFB = plain row-major
[N/128, K/128]** (matches DeepSeek HF `weight_scale_inv` layout directly). For bmm,
per-batch stride = `(N/128)*(K/128)` (`fp8_blockscale_gemm_kernel.cuh:1110`).

### 2.4 THE scale-promotion pattern (exact code)

`promote()` — `sm89_fp8_gemm_1d1d.cuh:167-191`:

```cpp
CUTE_DEVICE void promote(TensorD& accum, TensorC const& temp_accum, TensorScale const& scale, Index n_block)
{
    for (int mma_m = 0; mma_m < cute::get<1>(cute::shape<0>(accum)); ++mma_m)
    CUTE_UNROLL for (int mma_n = 0; mma_n < cute::get<0>(cute::shape<0>(accum)); ++mma_n)
    CUTE_UNROLL for (int mma_iter_m = 0; mma_iter_m < cute::size<1>(accum); ++mma_iter_m)
    CUTE_UNROLL for (int mma_iter_n = 0; mma_iter_n < cute::size<2>(accum); ++mma_iter_n)
    {
        auto coord_d = cute::make_coord(cute::make_coord(mma_n, mma_m), mma_iter_m, mma_iter_n, n_block);
        auto coord_c = cute::make_coord(cute::make_coord(mma_n, mma_m), mma_iter_m, mma_iter_n);
        accum(coord_d) += temp_accum(coord_c) * scale(mma_m, mma_iter_m, cute::_0{});
    }
}
```

Mainloop scale computation + use — `sm89_fp8_gemm_1d1d.cuh:381-390` (identical at `:419-428`
and `:440-444` for the tails):

```cpp
if constexpr (n_block == 0) {
    ... issue next gmem->smem cp.async stage ...
    cute::for_each(cute::make_int_sequence<cute::size(scale)>{},
        [&](auto i) { scale(i) = tXrSFA(i) * tXrSFB(0); });   // per-row a_scale * scalar b_scale
}
cute::clear(temp);                                            // fresh FP32 accum per 128-K block
cute::gemm(mma, tCrA, tCrB(cute::_, cute::_, cute::_, n_block), temp);
promote(accum, temp, scale, n_block);                         // accum += temp * (sa*sb)
```

Semantics: because `TileK == ScaleGranularityK == 128`, **one k-tile == one scale group**.
Per k-tile: clear a temp register accumulator, run 4 n-groups x (32x32x128 worth of)
`m16n8k32` mmas into it, then FMA it into the persistent accumulator scaled by
`SFA[row, k_tile] * SFB[0, k_tile]` (per-row x per-tile-scalar). N-direction: with TileN=128
there is exactly one SFB scalar per tile; the 4 n_groups (32 cols each) all share it, so the
promote only indexes `scale(mma_m, mma_iter_m, _0)` — the row (M) dimension. `tXrSFB` is a
broadcast fragment (SFB smem→RF copy TV layout is all stride-0, `sm89_utils.cuh:214-217`;
`SmemLayoutSFA` distributes the 32 row-scales so each thread holds the scales for its
accumulator rows, `:205-212`). Cost note: promote = 1 FFMA per accumulator element per
k-tile — i.e. per 128 mma-k, one extra FFMA per output reg, amortized 4 k-steps of mma per
FFMA (128/32). Register budget per thread: accum 32 f32 (32x128/128 thr) + temp 8 f32
(32x32/128) + scale fragment + A/B frags.

Caveat: the kernel reads only `tXrSFB(0)` — a **single B scale per (N-tile, k-tile)**. Only
valid because TileN = 128 == ScaleGranularityN. Reusing this shell for TileN > 128 requires
generalizing the SFB fragment (TRT-LLM never instantiates such a config).

### 2.5 Epilogue

`sm89_fp8_gemm_1d1d.cuh:114-165` (`epilogue_with_smem`): regs → smem via
`make_tiled_copy_C(AutoVectorizingCopy, mma)` into `Swizzle<3,3,3>` 8x64 bf16 atom
(`sm89_utils.cuh:178-190`), `__syncthreads()`, smem → regs, then 16 B `UniversalCopy<uint128_t>`
stores to gmem with per-element M/N residue bounds checks (`:151-164`). Output dtype BF16.
No beta/C, no bias fusion in-kernel (bias added in torch code, `linear.py:1231`).

## 3. Scale dtype handling

- **sm89 (and sm90 deep_gemm): FP32 scales, period.** Torch op asserts
  `CHECK_INPUT(matScale, FP8_BLOCK_SCALING_SF_DTYPE)` where
  `FP8_BLOCK_SCALING_SF_DTYPE = torch::ScalarType::Float`
  (`cpp/tensorrt_llm/thop/thUtils.h:67`, used at `fp8BlockScalingGemm.cpp:44`).
- **The kernel stores the DEQUANT scale** (amax/448), not the quant scale
  (`scale_1x128_kernel:247-252`: `dequant_scale = 1/quant_scale` stored; quant multiplies
  by `448/amax`). Same convention as HF `weight_scale_inv`.
- **BF16 checkpoint scales must be upcast**: `trtllm_quant.py:130-132` —
  `"TRT-LLM fp8_block_scaling_gemm requires float32 scales; HF checkpoints may store
  weight_scale_inv in bfloat16 ... so cast here"`, and `linear.py:1178-1180` stores
  `weight_scale` as an FP32 Parameter at load. => For Qwen3.8-27B-FP8 (BF16 128x128 scales)
  we must preconvert BF16 → FP32 scales (trivial one-shot at load, or fuse into the
  checkpoint indexer).
- Contrast — Blackwell: sm100 uses UE8M0 packed-int scales (`fp8_blockscale_quant_packed.cu`,
  `fp8_blockscale_gemm_kernel.cuh:730` `ElementBlockScale = int32_t` for sm120; torch op
  `fp8_block_scale_gemm_blackwell_geforce` checks `ScalarType::Int`). The
  `use_ue8m0` template param on `scale_1x128_kernel` (`:182-244`) exists to emit
  power-of-2-rounded scales for those paths; sm89 uses `use_ue8m0=false` → plain FP32.
- Ada entry constraints (`fp8BlockScalingGemm.cpp:99-100`): **K % 128 == 0** (Hopper only
  needs %16) and N % 16 == 0. M unconstrained (residue predication in kernel).

## 4. W8A16 vs W8A8 on Ada

- **W8A8 is the only FP8 execution mode on sm89.** The whole `fp8_block_scaling_gemm` family
  requires both operands as `Float8_e4m3fn` ("Matrix dtype must be FP8",
  `fp8BlockScalingGemm.cpp:41-42`); activations are dynamically quantized per-token-group
  upstream. Model flow (`linear.py:1219-1228`):
  ```python
  else:   # sm89 and sm90
      act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(input)
      output = torch.ops.trtllm.fp8_block_scaling_gemm(act_input_fp8, module.weight,
                                                       act_input_sf, module.weight_scale)
  ```
- **No FP8 W8A16 exists**: `QuantAlgo.W8A16` in `tensorrt_llm/quantization/mode.py:24,385`
  is INT8 weight-only (dequant-GEMM via `fpA_intB_gemm` cutlass kernels — that is where
  TRT-LLM's "dequant-in-mainloop" pattern lives, for INT4/INT8, not FP8).
- Runtime B requantization (BF16→FP8 128x128, `scale_128x128_kernel`) is guarded
  `__CUDA_ARCH__ >= 900` (`fp8_blockscale_gemm_kernel.cuh:492`) → compiles to a **no-op on
  sm89**; internal_quantize_b must be false on Ada (weights pre-quantized, which is our case).
  The A quant kernel `scale_1x128_kernel` is guarded `>= 890` (`:186`) → works on Ada.
- If block geometry is not exactly 128x128, `trtllm_finegrained_fp8_linear` **falls back to
  BF16 dequant + cuBLAS** rather than an FP8 kernel (`trtllm_quant.py:148-159`).
- MoE/grouped blockscale GEMM: **sm89 not supported** (`fp8_block_scaling_moe_gemm` switch
  only has `case 90/120`, `fp8BlockScalingGemm.cpp:343-349`). Dense + strided-batch only.
  (Dense BMM on sm89: `strided_batch_gemm_dispatch_sm89`, `fp8_blockscale_gemm_kernel.cuh:1082`.)

## 5. Per-token-group activation quant (the "per_token_group_quant" equivalent)

This clone has **no kernel named `per_token_group_quant`** (older TRT-LLM / vLLM name). The
equivalent is `scale_1x128_kernel` exposed as torch op `trtllm::fp8_quantize_1x128`.

`scale_1x128_kernel` (`fp8_blockscale_gemm_kernel.cuh:182-274`), guard `>= 890` (Ada OK):

- Grid `SMs*8` x 256 threads; **one warp per (row, 128-elem K-group)**
  (`:191-195`); grid-stride loop over groups.
- Each lane loads 2x `__nv_bfloat162` (4 elems, 64-elem stride) → warp covers 128 elems
  (`:202-215`); tail-clamped by bounds check against `dim_x`.
- Amax: `__hmax(__habs(...))` per lane + 5-step `__shfl_xor` warp reduce
  (`find_max_elem_in_warp`, `:172-180`); clamp `amax >= 1e-10` (`:230`).
- `quant_scale = 448.f / amax` (448 = e4m3 max); stores **dequant** scale `1/quant_scale`
  at `scales[k_block * M_padded + row]` (`:231-252`) — the col-major SFA layout the GEMM
  kernel expects (`M_padded = ceil(M/4)*4`).
- Requantized store: `output_line[lane] = OutputType(x * quant_scale)` as bf162 pairs
  (`:256-271`).
- `USE_UE8M0` variant (`:234-244`): rounds dequant scale to a power of 2 via
  `__nv_cvt_float_to_e8m0(..., cudaRoundPosInf)` and recomputes quant scale — for SM100
  deep_gemm compatibility. Ada ignores it.
- Launcher `fp8_1x128_cs` (`:599-616`): `<<<SMs*8, 256>>>`. MoE variant with fast-divmod
  (`kernel_utils::find_divmod` magic-number division, `:46-115`) and optional binary search
  over per-expert row offsets, guarded `>= 900` (`:276-401`). Reshape/permute102 variant for
  attention BMM (`:405-486`, `>= 890`).
- Newer fused alternative `fp8_quantize_1x128_packed_bf16_e4m3`
  (`fp8_blockscale_quant_packed.cu:240-260`) — one warp does row x 512 K-elems, 8-lane
  subgroups per 128-block, packs 4 UE8M0 bytes into uint32, PDL launch hooks
  (`cudaGridDependencySynchronize` / `cudaTriggerProgrammaticLaunchCompletion`) — **SM100
  only** (guarded in `fp8Quantize.cpp:160-161,220-221`). Interesting pattern to steal for a
  fused quant even with FP32 scales (the lane-subgroup amax shuffle reduction
  `__shfl_xor_sync(...,4,8)` at `:127-129` is cheaper than full-warp reduce).

Weight-side 128x128 quant (`scale_128x128_kernel`, `:489-569`, `>= 900`): one warp per
128x128 block, 128 rows x 4x32 lanes strided reads, warp amax, `scale = 448/amax`, stores
`scales[n_block * (K/128) + k_block]` (row-major SFB). Offline-only (weights); public API
`fp8CS128x128` (`:629-638`) actually just casts + fills scales with 1.0 (used when scales
are provided separately).

## 6. sm89 FP8 rowwise GEMM (context — per-tensor/per-channel scales)

`fp8_rowwise_gemm_kernel_template_sm89.h` + `fp8_rowwise_gemm_template.h:221-296`:
- Cutlass 2.x `DefaultGemmWithVisitor`, `InstructionShape = 16x8x32` (same Ada FP8 mma),
  `ArchTag = Sm89`, `ThreadblockSwizzleStreamK` (**StreamK** — noteworthy for Ada GEMMs
  with small M x large N), stages 2/3/4.
- Scales applied **in the EVT epilogue**, not the mainloop: per-col A scale
  (`VisitorColBroadcast`) x per-row B scale (`VisitorRowBroadcast`) — fine for rowwise
  quant, NOT usable for 128x128 blocks (needs K-direction partitioning; that's why the
  hand-rolled kernel exists).
- Instantiated CTA shapes (`fp8_rowwise_gemm_template.h:230-296`): 32x128x64/w32x32,
  64x128x64/w32x64, 64x64x128/w32x64, 64x128x64/w64x32, 128x64x64, 128x128x64 (three warp
  splits), 128x256x64/w64x64. K-tiles of 64 or 128, FP8 alignment 16 elems (128b).
- Runs on sm89 or sm12x only (runtime trap otherwise, `:107-122`).

## 7. Actionable takeaways for Insignia (Qwen3.8-27B-FP8 on 4070 SUPER)

1. **The mma exists and is enough**: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
   (copy the exact PTX + cute ALayout/BLayout from `sm89_utils.cuh:41-82` — the register
   fragment layouts are fiddly; reusing them verbatim avoids ldmatrix layout bugs).
2. **Reference design**: 32x128x128 CTA tile, 3-stage cp.async (16 B cg for data, 4 B for
   scales), Swizzle<3,4,3> smem, ldmatrix.x4, temp-accumulator-per-128-K-block +
   `accum += temp * (sa[row,kb] * sb[nb,kb])` promote, ~61.5 KB smem, 1 CTA/SM, 128 threads.
   For decode GEMV (M=1..8) the M=32 tile with cp.async predication is exactly TRT-LLM's
   answer — no separate GEMV kernel; waste is tolerated.
3. **Scale layout contract**: SFA `[K/128][M4]` col-major (M padded to 4), dequant-scale
   convention, FP32 dtype; SFB `[N/128][K/128]` row-major = HF layout. BF16→FP32 scale
   upcast must happen once at load.
4. **Activation quant kernel shape**: warp-per-(row,128-group), 4 bf16/lane, hmax-abs warp
   reduce, 1e-10 clamp, 448/amax — trivially fusable into preceding RMSNorm like we do for
   MXFP4, removing a separate kernel launch + gmem roundtrip.
5. **Things TRT-LLM does NOT give us on sm89**: FP8 W8A16/dequant path (none), grouped/MoE
   blockscale (Hopper/Blackwell only), TMA (Ada has none for this), and their promote cost
   (1 FFMA/output/k-tile) — a 64-wide K micro-tile with 2 promotes per 128-K block or an
   fp16x2 packed promote could shave ALU if it ever shows up in nsys.
6. StreamK scheduling from the rowwise sm89 template is worth stealing for N-heavy decode
   GEMMs on the 48-SM 4070 SUPER (grid = N/128 CTAs only ~32 CTAs for N=4096 → half idle).
