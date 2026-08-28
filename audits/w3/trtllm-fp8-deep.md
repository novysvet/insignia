# w3 deep dive: TRT-LLM sm89 FP8-blockscale GEMM/GEMV for Insignia

Clone: `E:\coding\Insignia\TensorRT-LLM` @ `v1.3.0rc24-454-gd9329fb8d3` (read-only, all paths
below relative to it unless absolute). Cross-referenced against `E:\coding\Insignia\vllm`
(read-only), `audits/w2/trtllm-sm89-fp8.md`, `audits/w2/vllm-marlin-fp8.md`,
`audits/w2/loader-27b-spec.md`. Mission: everything needed for the best sm_89 FP8-blockscale
(e4m3 weights, 128x128 bf16 scales) GEMM/GEMV for T<=64 prefill and T=1-5 decode of
Qwen3.8-27B-FP8 on the 4070 SUPER.

## 0. TL;DR

1. The only sm89 blockwise-FP8 GEMM in existence is TRT-LLM's hand-rolled CuTe kernel
   (`ada_blockwise_gemm/`): CTA tile 32x128x128, 3 cp.async stages, 61,952 B smem, 128
   threads, 1 CTA/SM. Core trick: raw e4m3 MMA per 128-K slab into a *temp* fp32
   accumulator, then `accum += temp * (SFA[row] * SFB_tile_scalar)` in registers.
2. The MMA is `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` — legal on sm_89+ only,
   2x the bf16-fp32acc rate. Copy the PTX + fragment layouts verbatim from
   `sm89_utils.cuh:41-82`; they are the fiddliest part.
3. 4070S roofline (intensity 2T FLOP/byte for every weight matrix): balance points
   T ~= 70 (bf16-wmma) and T ~= 141 (e4m3-mma). T<=64 is bandwidth-bound (177 us for the
   89 MB gate/up) — bf16-wmma W8A16 *suffices* at T=64 but with only 9-21% compute headroom.
4. T=128: bf16 is compute-bound (278-321 us) while fp8 is still BW-bound (138-161 us) ->
   fp8-mma wins 1.8-2.0x. T=256: exactly 2x. (Mission's "T=128 BW = 354 us" double-counts —
   weights stream once per batch; BW time is constant in T.)
5. For any e4m3-mma path, A must be e4m3 too: per-token 1x128 absmax/448 quant, floor 1e-10.
   TRT kernel: warp-per-group, grid `SMs*8` x 256 thr; fuse it into RMSNorm epilogue.
6. Numerics verdict: W8A16-dequant is strictly better quality (no activation rounding) and
   matches Insignia's NumPy reference; W8A8 only matches *TRT-LLM serving* numerics (vLLM on
   Ada is W8A16 Marlin anyway). Use W8A16 everywhere except T>=128 prefill chunks.
7. cuBLASLt is a dead end on sm89: VEC128_32F / BLK128x128_32F / OUTER_VEC_32F block-scale
   modes are Hopper(+Blackwell)-only (CUDA >= 12.9); Ada gets per-tensor scalar FP8 only.
8. CUTLASS has NO sm89 blockwise (3.x collectives sm90+; 2.x sm89 fp8 is per-tensor scale
   only). CUTLASS is not vendored in the TRT tree (fetched, pinned v4.4.2). Nothing to lift
   headers-only; reimplement the TRT pattern in Insignia's existing cp.async GEMM shell.
9. No blockwise-FP8 GEMV exists anywhere (TRT pads M into a 32-row tile; Marlin uses 16-row
   thread tiles). Insignia's f32-accum GEMV is already SOTA-shaped; steal Marlin's 3-logic-op
   e4m3->bf16 shift dequant (2^120 folded into scales) for any bf16 path.
10. Regime map: T=1-5 -> f32-acc GEMV + shift-dequant (W8A16); T<=64 -> W8A16 bf16-wmma GEMM
    at roofline (e4m3-mma as compute-risk insurance); T>=128 -> TRT-style e4m3-mma + promote
    + fused per-group act quant.

## 1. The sm89 blockwise kernel, verbatim

Files: `cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/ada_blockwise_gemm/
sm89_fp8_gemm_1d1d.cuh` (454 L) and `.../sm89_utils.cuh` (249 L). Dispatch:
`.../fp8_blockscale_gemm_kernel.cuh:681-715`; torch op `cpp/tensorrt_llm/thop/
fp8BlockScalingGemm.cpp:85-116` (arch switch `case 89:` at :246-253).

### 1.1 Instantiated config (exactly one; `fp8_blockscale_gemm_kernel.cuh:688-711`)

```cpp
using ElementInput = cute::float_e4m3_t;      // A and B both e4m3
using ElementOutput = cute::bfloat16_t;
using ElementAccum = float;
using ElementBlockScale = float;              // FP32 scales, period (sm89)
static constexpr int Stages = 3;
using TileShape = cutlass::gemm::GemmShape<32, 128, 128>;   // M=32, N=128, K=128
using KT = ada_blockwise_gemm::AdaBlockwiseGemmTraits<ElementInput, ElementOutput,
            ElementAccum, ElementBlockScale, Stages, 32, 128, 128>;
using GemmKernel = ada_blockwise_gemm::AdaBlockwiseGemmKernel<KT>;
static constexpr int kSmemSize = KT::kSmemSize;             // = 61,952 B
int grid_m = (shape_m + 32 - 1) / 32, grid_n = (shape_n + 128 - 1) / 128;
dim3 grid = dim3(grid_m, grid_n, 1); dim3 block = dim3(128, 1, 1);
cudaFuncSetAttribute(...sm89_fp8_gemm_1d1d_impl<GemmKernel>,
    cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
sm89_fp8_gemm_1d1d_impl<GemmKernel><<<grid, block, kSmemSize, stream>>>(...);
```

Smem budget (`sm89_utils.cuh:226-245`): A 3x32x128 = 12,288 B + B 3x128x128 = 49,152 B +
SFA 3x32 f32 = 384 B + SFB 3x1 f32 = 12 B, struct aligned to 128 -> **61,952 B**, opt-in
dynamic smem, **1 CTA/SM** on Ada (100 KB/SM). Static asserts (`sm89_utils.cuh:126-128,
224`): TileM%16==0, TileN%32==0, TileK%32==0, Stages>=2. 4 warps / 128 threads
(`:135-136`). MMA permutation 32x32x32 -> `NUM_GROUP_N = 4` n-groups of 32 cols
(`:150-159`). Register cost per thread: 32 f32 accum + 8 f32 temp + 32-reg B fragments
(all 4 n-groups resident) + 8-reg A + scale frags -> ~120 regs, fits.

Entry constraints (`fp8BlockScalingGemm.cpp:99-100`): **K % 128 == 0, N % 16 == 0, M
unconstrained** (per-row cp.async predication). Qwen3.8-27B is fully 128-clean: K in
{5120, 6144, 17408} = 40/48/136 x128; every N (17408, 5120, 10240, 6144, 12288, 2048,
1024) is a multiple of 128, so every N-tile maps to exactly one SFB scalar — the kernel's
`tXrSFB(0)` single-B-scale assumption (below) holds everywhere in this model.

### 1.2 Scale tensors in gmem (`sm89_fp8_gemm_1d1d.cuh:78-92`)

```cpp
uint32_t const ScaleM = (((M + 3) >> 2) << 2);              // A-scales: M padded to 4
uint32_t const ScaleN = (N + 127) / 128, ScaleK = (K + 127) / 128;
mSFA_mk = make_tensor(..., shape(ScaleM, ScaleK), stride(_1, ScaleM)); // [K/128][M4] col-major
mSFB_nk = make_tensor(..., shape(ScaleN, ScaleK), stride(ScaleK, _1)); // [N/128][K/128] row-major
```

SFB layout == HF `weight_scale_inv` exactly (only needs the one-time bf16->f32 upcast, see
§4.4). Scales are DEQUANT scales (amax/448), fp32 dtype enforced by the op
(`thop/thUtils.h:67` via `fp8BlockScalingGemm.cpp:44`).

### 1.3 How the scales load

- SFA (per-row, 32 per tile): 4 B `cp.async` per thread through
  `GmemTiledCopySFA` (`sm89_utils.cuh:193-197`); smem->RF copy distributes the 32 row
  scales so each thread holds the scales for its accumulator rows (`:204-212`).
- SFB (per 128x128 tile, ONE scalar per k-tile): gmem copy uses an **all-stride-0 broadcast
  TV layout** (`GmemLayoutTVSFB ... Stride<_0,_0>`, `sm89_utils.cuh:199-201`) so every
  thread issues the same 4 B `cp.async`; smem->RF fragment is also stride-0 broadcast
  (`SmemLayoutTVSFB`, `:214-217`) -> every thread has the same SFB scalar in a register.
- Caveat (unchanged from w2): the promote reads only `tXrSFB(0)` — valid only because
  TileN = 128 == ScaleGranularityN (`sm89_utils.cuh:138-140`). Never widen TileN without
  generalizing the SFB fragment.

### 1.4 The promote-accumulate core (verbatim)

`promote()` — `sm89_fp8_gemm_1d1d.cuh:167-191`:

```cpp
template <class TensorD, class TensorC, class TensorScale, class Index>
CUTE_DEVICE void promote(TensorD& accum, TensorC const& temp_accum, TensorScale const& scale, Index n_block)
{
    using AccumType = typename TensorD::value_type;
    for (int mma_m = 0; mma_m < cute::get<1>(cute::shape<0>(accum)); ++mma_m)
    {
        CUTE_UNROLL for (int mma_n = 0; mma_n < cute::get<0>(cute::shape<0>(accum)); ++mma_n)
        CUTE_UNROLL for (int mma_iter_m = 0; mma_iter_m < cute::size<1>(accum); ++mma_iter_m)
        CUTE_UNROLL for (int mma_iter_n = 0; mma_iter_n < cute::size<2>(accum); ++mma_iter_n)
        {
            auto coord_d = cute::make_coord(cute::make_coord(mma_n, mma_m), mma_iter_m, mma_iter_n, n_block);
            auto coord_c = cute::make_coord(cute::make_coord(mma_n, mma_m), mma_iter_m, mma_iter_n);
            accum(coord_d) += temp_accum(coord_c) * scale(mma_m, mma_iter_m, cute::_0{});
        }
    }
}
```

Mainloop (steady state) — `sm89_fp8_gemm_1d1d.cuh:346-392`, the part that matters:

```cpp
int k_tile_iter = KT::Stages - 1;                    // prologue filled Stages-1 stages
while (k_tile_iter < k_tile_count)
{
    cute::for_each(cute::make_int_sequence<KT::NUM_GROUP_N>{}, [&](auto n_block)
    {
        if constexpr (n_block == KT::NUM_GROUP_N - 1) {          // last n-group of the tile:
            ...rotate ring: tXsX_read = tXsX(_,_,_,smem_pipe_read);//
            cute::cp_async_wait<KT::Stages - 2>(); __syncthreads();
            cute::copy(s2r_copy_SFA, tXsSFA_read, tXrSFA);        // new row scales -> RF
            cute::copy(s2r_copy_SFB, tXsSFB_read, tXrSFB);        // new tile scalar -> RF
        }
        auto n_block_next = (n_block + 1) % KT::NUM_GROUP_N;
        cute::copy(s2r_copy_B, tXsB_read(_, n_block_next, _), tXrB(_, _, n_block_next)); // prefetch next B frag
        if constexpr (n_block == 0) {
            // gmem -> smem refill for k_tile_iter (A 4KB + B 16KB + scales), fence, ring++
            cute::copy_if(g2s_copy_A, tApA, tAgA(_,_,_,k_tile_iter), tAsA(_,_,_,smem_pipe_write));
            cute::copy_if(g2s_copy_B, tBpB, tBgB(_,_,_,k_tile_iter), tBsB(_,_,_,smem_pipe_write));
            cute::copy_if(g2s_copy_SFA, tApSFA, tAgSFA(_,_,_,k_tile_iter), tAsSFA(_,_,_,smem_pipe_write));
            cute::copy(g2s_copy_SFB, tBgSFB(_,_,_,k_tile_iter), tBsSFB(_,_,_,smem_pipe_write));
            cute::cp_async_fence();  k_tile_iter++;
            smem_pipe_write = smem_pipe_read;
            ++smem_pipe_read;  smem_pipe_read = smem_pipe_read == KT::Stages ? 0 : smem_pipe_read;
            // ==== SCALE PRODUCT: SFA[row] * SFB, once per k-tile, fp32 registers ====
            cute::for_each(cute::make_int_sequence<cute::size(scale)>{},
                [&](auto i) { scale(i) = tXrSFA(i) * tXrSFB(0); });
        }
        cute::clear(temp);                                       // fresh fp32 accum per 128-K block
        cute::gemm(mma, tCrA, tCrB(_, _, _, n_block), temp);      // 4 k-steps of m16n8k32 into temp
        if constexpr (n_block == KT::NUM_GROUP_N - 1) { cute::copy(s2r_copy_A, tXsA_read, tCrA); } // A refetch
        promote(accum, temp, scale, n_block);                    // accum += temp * (sa*sb)
    });
}
// load tail (:393-430): same body under for_each<Stages-2> with cp_async_wait<Stages-3-WaitIndex>
// mma tail (:431-445): last k-tile, same body without gmem refill
// epilogue (:446-451 -> :114-165): f32->bf16, regs->smem (Swizzle<3,3,3>), 16B stores w/ M/N residue checks
```

Semantics: `TileK == ScaleGranularityK == 128` -> **one k-tile == one scale group**. Per
k-tile: `scale[row] = SFA[row,ktile] * SFB[ntile,ktile]` computed once (fp32, registers),
raw e4m3 MMAs accumulate into `temp` (cleared each k-tile), then one FFMA per accumulator
element folds `temp` into the persistent accumulator. A is reloaded once per 4 n-groups
(reused across the tile); B fragments for all 4 n-groups live in registers simultaneously.

**Promote cost** (per thread per k-tile): 32 FFMA (32 accum values) vs 4096 tensor-core MAC
(32x128x128 / 128 thr). FFMA pipe = 128 FMA/SM/clk vs fp8 tensor ~512 MAC/SM/clk -> 32 clk
of FFMA per 1024 clk of MMA = **3.1% tensor-time overhead**. Cheap; keep it.

Pipeline: classic sm80-style cp.async, no TMA, no mbarriers (`sm89_utils.cuh:162-163`:
16 B `SM80_CP_ASYNC_CACHEALWAYS<uint128_t>` for A/B; `:193` 4 B for scales; `:173-175`
`SM75_U32x4_LDSM_N` = `ldmatrix.x4`; smem A/B = `Swizzle<3,4,3>` over 16x128, `:169-171`).
Prologue fills Stages-1 stages with `copy_if` residue predication (`:262-289`); steady
state waits `cp_async_wait<Stages-2>` + `__syncthreads()` (`:357-358`).

### 1.5 Grid utilization on the 4070 SUPER (48 SMs)

- gate/up (N=17408): grid_n = 136 CTAs -> 2.83 waves, fine.
- **down_proj / out_proj / o_proj (N=5120): grid_n = 40 CTAs < 48 SMs -> 8 SMs idle.**
  TRT never fixed this (no streamK on the sm89 path; only the sm89 *rowwise* template has
  `ThreadblockSwizzleStreamK`, `fp8_rowwise_gemm_kernel_template_sm89.h`). If we lift the
  design, an N-split (2 CTAs per N-tile halving K, reduce in smem/L2) or a smaller-TileN
  variant (64) is a free win on the 5120-wide matrices.

### 1.6 Model-level flow (where the A-quant sits)

`tensorrt_llm/_torch/modules/linear.py:1193-1229` (`FP8BlockScalesLinearMethod.apply`) —
sm89 branch (the `else:` at :1225):

```python
else:   # sm89 and sm90 (non-100f)
    act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(input)   # per-token 1x128 quant
    output = torch.ops.trtllm.fp8_block_scaling_gemm(
        act_input_fp8, module.weight, act_input_sf, module.weight_scale)       # e4m3-mma blockwise
```

`module.weight_scale` is created as an **FP32 Parameter** (`linear.py:1177-1181`); HF bf16
scales are upcast at load — `trtllm_quant.py:130-132`: "TRT-LLM fp8_block_scaling_gemm
requires float32 scales; HF checkpoints may store weight_scale_inv in bfloat16 ... cast
here". Dense + strided-batch only on sm89; grouped/MoE is `case 90/120` only
(`fp8BlockScalingGemm.cpp:343-349`).

## 2. The e4m3 MMA on sm_89

### 2.1 Exact instruction and wrapper (verbatim, `sm89_utils.cuh:33-82`)

```cpp
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
#define CUTE_ARCH_MMA_F32_SM89_ENABLED
#endif
...
struct SM89_16x8x32_F32F8F8F32_TN
{
    using DRegisters = float[4];
    using ARegisters = uint32_t[4];    // 16 e4m3
    using BRegisters = uint32_t[2];    // 8 e4m3
    using CRegisters = float[4];

    CUTE_HOST_DEVICE static void fma(float& d0, float& d1, float& d2, float& d3,
        uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
        uint32_t const& b0, uint32_t const& b1,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
#if defined(CUTE_ARCH_MMA_F32_SM89_ENABLED)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#endif
    }
};

template <>
struct MMA_Traits<SM89_16x8x32_F32F8F8F32_TN>
{
    using ValTypeD = float; using ValTypeA = float_e4m3_t; using ValTypeB = float_e4m3_t;
    using ValTypeC = float;
    using Shape_MNK = Shape<_16, _8, _32>;
    using ThrID = Layout<_32>;
    using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _2, _2>>,
                           Stride<Stride<_64, _1>, Stride<_16, _8, _256>>>;
    using BLayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _2>>,
                           Stride<Stride<_32, _1>, Stride<_8, _128>>>;
    using CLayout = SM80_16x8_Row;   // each thread: 4 f32, rows (groupID + {0,8}), cols 2j+{0,1}
};
```

CTA tiling (`sm89_utils.cuh:156-159`): `TiledMMA<MMA_Atom, Layout<2,2,1>, Tile<32,32,32>>`
-> 4 warps each own an m16n8k32 atom quadruple per 32x32x32 k-step; A operand feeds
`ldmatrix.x4`, B is 16 B smem reads. A **bf16 fallback atom**
(`SM80_16x8x16_F32BF16BF16F32_TN`, `sm89_utils.cuh:97-104`) exists in the same traits —
the identical kernel shell can run W8A16-bf16mma by swapping the atom and smem swizzle.
That is precisely the "one shell, two backends" Insignia wants.

### 2.2 Legality + throughput vs bf16

- Legality: `m16n8k32.f32.e4m3.e4m3.f32` is sm_89+ (PTX ISA 7.8+; TRT gates on
  `__CUDA_ARCH__ >= 890`, `sm89_utils.cuh:33-35`). vLLM's Marlin confirms from the other
  side: its fp8-as-A mma is hard-restricted to `sm89 || sm12x` with the comment "It is
  slower than Marlin W4A16 on other devices" (`vllm/csrc/libtorch_stable/quantization/
  marlin/marlin.cu:413-421`); fp8 weights require `>= 89` (`marlin.cu:413-416`).
- On sm_90 this mma.sync shape is legal but strictly worse than wgmma (Hopper path uses
  `fp8_blockscale_mma_utils.cuh` m64nXk32 wgmma instead). It is an Ada-native instruction.
- Throughput: **2x bf16** per the Ada tensor-core design (FP8 = 2x FP16 rate; m16n8k32
  carries 2x the K of m16n8k16 per instruction at the same issue rate). fp32 accumulate is
  the only accumulator for e4m3 (no fp16-acc variant), so the comparison that matters:
  e4m3-mma vs `m16n8k16.f32.bf16.bf16.f32`.
- Absolute rates for the 4070 SUPER (56 SM, 7168 lanes, 2.475 GHz boost -> 35.5 TF FP32):
  bf16-wmma-fp32acc ~= 2x FP32 = **~71 TF dense**; e4m3-mma-fp32acc ~= 4x FP32 =
  **~142 TF dense** (= synthesis.md's 142; TechPowerUp lists 141.9 "FP16" / 283.9 "FP8"
  dense, which are the FP16-accumulate-class/marketing rates — consumer GeForce halves
  FP32-accumulate throughput on 16-bit MMA, while e4m3 is f32-acc-only at the 4x rate).
  Real kernels reach 70-85% of these. The mission's 82/165 TF band sits between the two
  conventions; both give identical conclusions (§3).
- Also on sm_89 (PTX): `cvt.rn.f16x2.e4m3x2` — 2-way fp8->f16 conversion instruction,
  useful for any dequant-to-bf16/f16 path (already noted in synthesis.md).

## 3. Crossover math for our shapes (4070 SUPER)

Shape family [T,K]x[N,K]^T, e.g. gate/up [T,5120]x[17408,5120]^T. Weight bytes = N*K (1
B/elem) = 89,128,960 B for 17408x5120. FLOPs = 2*T*N*K = T * 178.26 MFLOP.
**Arithmetic intensity is 2T FLOP per weight byte for every matrix in the model** — so the
crossovers below are universal (down_proj, out_proj, qkv, ... all identical):

- BW time = 89.129 MB / 504 GB/s = **176.8 us, constant in T** (weights streamed once).
- Machine balance: bf16 71 TF / 504 GB/s = 141 FLOP/B -> **T = 70.4**; at 82 TF -> T = 81.3.
  fp8 142 TF / 504 GB/s = 282 FLOP/B -> **T = 140.9**; at 165 TF -> T = 163.7.

| T | GFLOP | BW (us) | bf16@71TF | bf16@82TF | fp8@142TF | fp8@165TF | binding limit | verdict |
|---|-------|---------|-----------|-----------|-----------|-----------|---------------|---------|
| 16 | 2.85 | 176.8 | 40.2 | 34.8 | 20.1 | 17.3 | BW | dtype irrelevant |
| 32 | 5.70 | 176.8 | 80.3 | 69.6 | 40.2 | 34.6 | BW | dtype irrelevant |
| 64 | 11.41 | 176.8 | 160.7 | 139.1 | 80.3 | 69.1 | **BW (bf16 margin 9-21%)** | bf16-wmma suffices, tightly; fp8-mma = 2.2x compute headroom |
| 128 | 22.82 | 176.8 | 321.4 | 278.3 | 160.7 | 138.3 | bf16: **compute**; fp8: BW (margin 9-22%) | **fp8-mma wins 1.8-2.0x** |
| 256 | 45.63 | 176.8 | 642.7 | 556.5 | 321.4 | 276.6 | compute (both) | fp8-mma exactly 2x (minus 3.1% promote) |

Checks against the mission's numbers: 89 MB -> 177 us at 504 GB/s **confirmed**;
2*64*17408*5120 = 11.41 GFLOP **confirmed**; 140 us @ 82 TF / 70 us @ 165 TF arithmetic
**confirmed**; "bf16-wmma suffices at T=64" **confirmed with caveat** — at the more
conservative 71 TF the margin is 10% (161 vs 177 us), i.e. the bf16 kernel must sustain
>=89% of peak tensor throughput *while* saturating HBM; with realistic 80-90% tensor
efficiency bf16 becomes the binding constraint at T=64. The mission's "T=128: 354 us BW"
is a **double-count** (that is two T=64 passes; one T=128 pass still reads the weights
once -> 177 us). Corrected conclusion is stronger than the mission's: at T=128 bf16 is
already compute-bound (278-321 us > 177 us) while fp8 is still BW-bound -> fp8 wins
1.8-2x; at T=256 fp8 wins 2.0x.

Two 4070S-specific footnotes:
- 48 MB L2 (AD103, full cache enabled — Tom's Hardware correction; TechPowerUp). The 89 MB
  gate/up and 52 MB qkv matrices exceed it (true DRAM streaming), but 31 MB matrices
  (in_proj_z, out_proj, o_proj) fit — repeated reads across MTP draft steps or chunked
  prefill can hit L2 at multi-TB/s, moving their crossover up.
- At T=64 chunked prefill of the 27B: per linear-attention layer, weight streaming =
  382.73 MB = 759 us/layer floor, x64 layers ~= 48.6 ms/chunk minimum — prefill is a
  weight-streaming problem here too (matches synthesis.md), so GEMM dtype only matters
  once T>=128 chunks or the bf16 margin above bites.

## 4. Activation quant (the A side of W8A8)

### 4.1 TRT-LLM `scale_1x128_kernel` (verbatim core, `fp8_blockscale_gemm_kernel.cuh:182-274`)

Launch (`fp8_1x128_cs`, `:599-616`): `<<<SMs * 8, 256>>>` (grid-stride), guard
`__CUDA_ARCH__ >= 890`. One warp per (row, 128-elem K-group); each lane reads 2x
`__nv_bfloat162` at 64-elem stride (warp covers 128), tail-clamped by bounds check:

```cpp
Input2Type input_frag2[2] = {Input2Type(0,0), Input2Type(0,0)};
for (int i = 0; i < 2; i++) {                       // 2 iterations x 64 elems
    if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x) break;
    input_frag2[i] = *((Input2Type*) (input_line) + lane_id / 2);
    input_line += 64;
}
for (int i = 0; i < 2; i++) {                       // absmax in bf16 ops
    if (...) break;
    input_amax = InputType(__hmax(input_amax, __hmax(__habs(input_frag2[i].x), __habs(input_frag2[i].y))));
}
InputType amax = find_max_elem_in_warp(input_amax);      // 5x __shfl_down + broadcast (:171-180)
amax = tensorrt_llm::common::cuda_max(amax, InputType(1e-10f));   // floor
ScaleType quant_scale = 448.f / ScaleType(amax);
ScaleType dequant_scale;
if constexpr (USE_UE8M0) { /* SM100-only pow2-rounded scale, :234-244 */ }
else { dequant_scale = 1.f / quant_scale; }
if (lane_id == 0)
    scales[(size_t) scales_idx_x * stride_scale_dim_y + scales_idx_y] = dequant_scale;  // [K/128][M4] col-major
...
    ScaleType value_1 = ScaleType(input_frag2[i].x) * quant_scale;   // multiply by 448/amax
    output_line[lane_id]     = OutputType(value_1);                  // __nv_fp8_e4m3 ctor (satfinite)
    output_line[lane_id + 1] = OutputType(value_2);
```

absmax/448, clamp via amax-floor 1e-10, dequant-scale stored (SFA layout of §1.2), fp8
store through the saturating `__nv_fp8_e4m3` constructor. Where it sits in their graph:
immediately before the GEMM, once per linear (linear.py:1225-1229); the reshape/permute102
variant (`:405-486`) serves attention BMMs; the SM100 fused variant
(`fp8_blockscale_quant_packed.cu:240-260`) packs UE8M0 for Blackwell — not for us.

### 4.2 vLLM `per_token_group_quant_8bit` (local tree, for contrast)

`vllm/csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu`:
16 threads/group, group=128; two-pass through **smem** (single gmem read): pass 1 copies
gmem->smem while accumulating `local_absmax` (init eps=1e-10), 4-step xor-butterfly
`GroupReduceMax` over 16 lanes (`:21-40`); scale `y_s = absmax / 448` (`:43-74`); lane 0
stores the fp32 scale; pass 2 re-reads smem and stores `clamp(src/y_s, -448, 448)` -> fp8
(`:76-96`). Launch (`:197-242`): grid = num_groups/groups_per_block, block =
groups_per_block x 16 with `GetGroupsPerBlock` picking the largest divisor <=16
(`:166-180`), dynamic smem = groups_per_block x 128 x sizeof(T), `cudaLaunchKernelEx` with
PDL (programmatic stream serialization). vLLM divides by y_s; TRT multiplies by 448/amax —
same math, one reciprocal apart.

For Insignia: fuse either into the preceding RMSNorm epilogue (Insignia already fuses
MXFP4 activation packing there) — one warp per (row,128) with the 2-loads-per-lane TRT
shape is the cheaper template; no separate launch, no gmem roundtrip for the bf16
intermediate.

### 4.3 Do we need W8A8? (numerics decision)

Facts from the loader spec (`audits/w2/loader-27b-spec.md:76-83`): quant_method fp8 e4m3,
`activation_scheme: dynamic`, `weight_block_size [128,128]`, all 407 F8 tensors have linked
bf16 scales; the checkpoint contains NO activation scale tensors. "dynamic" is a *serving*
directive: it tells the runtime to quantize activations online (per-token 1x128). It is not
a property of the weights, and the weights were converted offline from a bf16 model
(modules_to_not_convert list, PTQ conversion). Evidence of what "reference serving" does:
TRT-LLM sm89 = W8A8 (§1.6); vLLM sm89 = **W8A16** (Marlin; "Marlin W8A8 is not supported",
`marlin_utils_fp8.py:79-81`); HF transformers fp8 with dynamic scheme = dequant-to-bf16 +
bf16 matmul (W8A16) when no fp8 kernel. So there is no single "reference numeric" — the
stacks disagree, and vLLM-on-Ada (the closest public analog to our rig) is W8A16.

Decision:
- **W8A16 (dequant weights, bf16/f32 activations) is strictly better quality**: e4m3
  activation quant adds ~2^-3.5 relative rounding (~4% worst-case, ~0.5-1% RMS) on every
  activation element; W8A16 eliminates it entirely. It also keeps bit-relevant parity with
  Insignia's NumPy reference (which dequantizes weights only) — the parity harness stays
  meaningful.
- **W8A8 is only ever a performance tool** (needed to feed e4m3-mma), and only pays at
  T >= ~70-141 (§3). Cost side is negligible: T x K elements quantized per matrix
  (T=64: 327K elems, ~µs) and fusable into RMSNorm.
- Recommendation: **W8A16 for decode (T=1-5) and T<=64 prefill; W8A8-with-dynamic-1x128
  only if/when T>=128 chunked prefill matters.** If ever comparing token streams against
  vLLM on this same GPU, both engines run W8A16 at decode anyway; TRT-LLM comparisons
  would need W8A8 — document the switch.

### 4.4 Scale dtype chores at load (one-time)

bf16 `weight_scale_inv` -> fp32 upcast (TRT enforces f32: `linear.py:1177`,
`trtllm_quant.py:130-132`); expand nothing — SFB stays [N/128, K/128] row-major. If we
instead run the Marlin-style bf16 path (§7), fold `2^120` into the bf16 scales at load and
assert scales in (2^-120, 2^7) (real block scales are ~1e-3 — fine).

## 5. cuBLASLt on sm_89: verdict — NOT usable for blockwise

- The only block-scale matmul descriptors in cuBLASLt (CUDA >= 12.9):
  `CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F` (A, 1x128) +
  `CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F` (B, 128x128), plus
  `OUTER_VEC_32F` (channelwise) and `VEC16/VEC32_UE4M3/UE2M3` (FP4/MX). These exactly
  match our formats — **on Hopper**: NVIDIA's cuBLAS 12.9 announcement scopes channel- and
  block-scaled FP8 matmuls to "NVIDIA Hopper GPUs"; 16/32-element 1D block scaling to
  Blackwell; Ada appears only as "previous cuBLAS versions enabled FP8 tensor-wide scaling
  ... on Hopper and Ada". cuBLAS 13.3 docs list the modes under "On GPUs with compute
  capability 9.0" for outer-vector/128-block. Confirmed no sm89 support anywhere.
- The TRT tree itself uses none of this on the FP8-blockwise path — its only
  `CUBLASLT_MATMUL_DESC_*_SCALE_MODE` usage is `VEC16_UE4M3` for **FP4** GEMM
  (`cpp/tensorrt_llm/common/cublasMMWrapper.cpp:88-118`, gated on `CUDA_R_4F_E2M1`).
- Verdict: on the 4070 SUPER cuBLASLt offers per-tensor-scaled FP8 e4m3 GEMM only
  (sm89-legal since CUDA 12.4 via the plain tensor-wide path). Wrong scaling granularity
  (would require requantizing weights to a global scale — exactly what the mission
  forbids). **Dead end; hand kernel it is.**

## 6. CUTLASS on sm_89: verdict — no blockwise exists

- CUTLASS is **not vendored** in the TRT tree: it is a CMake fetch pinned to **v4.4.2**
  (`3rdparty/fetch_content.json:12-15`); only `cpp/tensorrt_llm/cutlass_extensions`
  (TRT's own headers) is in-tree. So "lift headers-only" would first require fetching
  cutlass v4.4.2 anyway.
- CUTLASS support matrix (changelog + example 67): FP8 blockwise collective mainloops
  exist for **sm90 (3.7.0+, ex. 67), sm100/101/103, sm120** only. sm89 FP8 support is
  "Ada (SM89) FP8 tensor cores via the 2.x API" (3.5.0) = per-tensor scale (rowwise via
  epilogue visitors), **no K-direction block scaling** — a 128x128 block scale cannot be
  applied in a 2.x epilogue because the K loop is inside the mainloop. There is no
  `CollectiveBuilder` sm89 blockwise and no sm89 2.x blockwise. (vLLM's gate agrees:
  `cutlass_scaled_mm_supports_block_fp8` returns false below cc 90,
  `scaled_mm_entry.cu:161-174`.)
- Answer to "can any be lifted headers-only": **no** — the only sm89 blockwise
  implementation in the wild is TRT's `ada_blockwise_gemm` itself. It is 2 files (~700
  LOC) but cute-dependent (TiledMMA/copy atoms/swiddles). Two options for Insignia:
  (a) vendor cutlass v4.4.2 headers + lift `sm89_utils.cuh`/`sm89_fp8_gemm_1d1d.cuh`
  verbatim; (b) re-express the pattern in Insignia's existing cp.async GEMM shell (it
  already has stages/predication) using only the verbatim PTX + ALayout/BLayout/CLayout
  of §2.1 and the promote structure of §1.4. Given AGENTS.md's hyper-specialization
  rules, (b) is the better fit; the fragment layouts are the only correctness trap.

## 7. GEMV (T=1): no one has a blockwise-FP8 GEMV; Marlin deep dive

### 7.1 Confirmation of absence

- TRT-LLM sm89: `case 89:` dispatches every M (including M=1) to the 32x128x128 tile with
  per-row cp.async predication (`gemm_dispatch_sm89` unconditional,
  `fp8BlockScalingGemm.cpp:252` -> `fp8_blockscale_gemm_kernel.cuh:681-715`; residue logic
  `sm89_fp8_gemm_1d1d.cuh:233-260`). A is still quantized 1x128 first (linear.py:1225) —
  even M=1 pays a W8A8 round trip. No GEMV.
- vLLM sm89: Marlin W8A16 is the only blockwise path; minimum problem granularity is a
  16-row thread tile (`thread_m_blocks` counted in 16s, `marlin.cu:423-438`), with an
  operand-swapped `mma_trans` 8-row specialization for `prob_m_split <= 8 && a_type
  16-bit` (`marlin.cu:438`; `marlin_mma.h:137+`). No GEMV.
- CUTLASS: 2.x sm89 fp8 tiles bottom out at 32x128/64x64. No GEMV.
- (For contrast, Hopper deep_gemm in TRT even does swap-AB for M < 32 —
  `runGemmSwapAB`, `fp8_blockscale_gemm_kernel.cuh:666-678` — Ada has no analog.)

At M=1 the tensor paths waste 16-32x compute, but intensity 2T = 2 FLOP/B << 141 so they
stay BW-bound; the real inefficiencies are activation-quant overhead (TRT), scale-precision
loss (Marlin, below), and launch tail (136-CTA grids). Insignia's dedicated f32-accum FP8
GEMV with fp32 scales is therefore *beyond* the published state of the art for this format
— nothing to copy, only things to beat.

### 7.2 Marlin FP8 mechanics (all local-tree citations, `E:\coding\Insignia\vllm\`)

- **Repack format**: weights bitwise-reinterpreted as int32 (4 e4m3 bytes) via
  `pack_fp8_to_int32` (`vllm/model_executor/layers/quantization/utils/
  marlin_utils_fp8.py:377-391`), padded to `(n%64&&k%128)` or `(n%128&&k%64)` tiles, then
  `ops.gptq_marlin_repack(num_bits=8)` (`csrc/libtorch_stable/quantization/marlin/
  gptq_marlin_repack.cu:117-236`): 16x64 tiles, 4 warps, 8-stage cp.async; for 8-bit
  weights the byte shuffle `pack_idx[4] = {0,2,1,3}` with `tc_offsets {0,1,8,9}` puts each
  output int's bytes at k = [r, r+8, r+1, r+9] (`:222-236`) — pre-arranged for the
  dequant's lane permutation and the `ldmatrix`-free 16 B B-fragment reads.
- **Scales**: 128x128 block scales -> group scales by `repeat_interleave(128, dim=N)`
  (exact, weights untouched) + `marlin_permute_scales` 64-wide `scale_perm` interleave;
  then the **2^120 fold** `fp8_fused_exponent_bias_into_scales`
  (`marlin_utils_fp8.py:35-46, 156-214`) — bf16 scale store = 8-bit mantissa = ~0.4%
  per-block scale rounding (the known quality tax; Insignia's fp32-scale design beats it).
- **Dequant idiom** (3 logic ops per 4 values, `csrc/.../marlin/dequant.h:357-373`):
  ```cpp
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;   // 4
  constexpr int MASK = 0x7F007F00;
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);   // e4m3 -> bf16 * 2^-120, exact incl. subnormals
  ```
  The 2^-120 bias cancels the folded scale; fp16 variant uses shift 1 / fold 2^8
  (`:321-336`). Scale applied by one `__hmul2` per bf16x2 against `frag_s`
  (`marlin_template.h:106-115, 1275-1277`); fp32 accumulation via
  `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (`marlin_mma.h:74-81`).
- **FP8 MMA in Marlin** (W4A8 only): `m16n8k16.f32.e4m3.e4m3.f32` and
  `m16n8k32.f32.e4m3.e4m3.f32` wrappers exist (`marlin_mma.h:93-116`) — the k16 shape is
  the half-throughput fp8 instruction; both are gated to sm89/sm12x for A-fp8
  (`marlin.cu:413-421`), "slower than Marlin W4A16 on other devices".
- **Pipeline/scheduling** (worth mirroring for T<=8): 4-stage cp.async (`marlin.cu:407`),
  persistent striped (k_tiles x n_tiles) partitioning with lock-based FP32 global reduce in
  L2 (bf16 atomicAdd unavailable on sm8x), scales refetched only when a stage starts a new
  group (`marlin_template.h:881-891`), `max_par` up to 128 stripes for n <= 4096 to keep
  SMs busy at small M (`marlin.cu:427-438`).
- **Reported performance**: no official GB/s for the fp8 variant; the Marlin paper
  (arXiv:2408.11743) claims near-optimal batch-1-16 decode ("within 5-10% of ideal
  speedup"); the repo notes real kernels land 10-15% below spec-sheet bandwidth; community
  benchmarks put Marlin-class W8A16 decode GEMV at ~80-90% of peak effective bandwidth
  (e.g. ~84% peak GEMV utilization on DGX Spark Qwen3.5 runs; ZipServ/ASPLOS'26 benchmarks
  the Marlin W8A16 FP8 kernel on an RTX 4090 as its baseline). For a 4070 SUPER that
  means ~400-450 GB/s of the 504 GB/s peak — Insignia's 504 GB/s-class GEMV target is the
  right bar.

## 8. Recommended idiom per regime (port checklist)

- **T=1-5 (decode, incl. MTP pair)**: keep the dedicated GEMV — f32 (or bf16x2) A, e4m3
  weights decoded in registers. Steal Marlin's shift dequant (§7.2) for a bf16-accumulating
  dot, or the `cvt.rn.f16x2.e4m3x2`/bit-trick decode for f32; apply 128x128 scale once per
  k-block in fp32 (SFB[kb] read as fp32 from the upcast buffer; fold per-row SFA is absent
  — activations unquantized). Bandwidth-bound; nothing else matters.
- **T<=8**: same GEMV widened to 2-8 rows (A in registers, Marlin `mma_trans`-style if
  bf16-mma is used), still BW-bound (intensity <= 16 FLOP/B).
- **T<=64 prefill**: W8A16 bf16-wmma GEMM: e4m3 B in smem, Marlin shift-dequant ->
  `m16n8k16.f32.bf16.bf16.f32` (or TRT's bf16 fallback atom in the same shell), block
  scales as fp32 in smem applied per 128-K slab with the promote trick but bf16-mma's temp
  accumulator. Roofline-sufficient (§3) with 9-21% headroom at T=64; NO activation quant —
  better numerics, matches the NumPy reference.
- **T>=128 chunks**: the TRT design verbatim-in-spirit: e4m3 A (per-token 1x128 dynamic
  quant fused into RMSNorm) + e4m3 B, `m16n8k32` MMA, temp-accum + `accum += temp *
  (SFA[row]*SFB)` promote per 128-K slab, 3+ stages, Swizzle<3,4,3>, ldmatrix.x4, fp32
  scales. Expected ~2x over bf16 at T>=141 (machine balance), 3.1% promote tax.
- Fixes to TRT's design when porting: N-split/streamK for N=5120 grids (40 CTAs < 48 SMs);
  optional TileN=64 variant for small-N matrices; keep SFB fp32 (never fold to bf16);
  fuse bias/gating epilogues per Insignia backlog.

## 9. File index

| What | Where (base `E:\coding\Insignia\TensorRT-LLM\`) |
|---|---|
| sm89 kernel | `cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/ada_blockwise_gemm/sm89_fp8_gemm_1d1d.cuh` (promote :167-191, mainloop :346-392, tails :393-445, epilogue :114-165, scale layouts :78-92) |
| traits + MMA PTX | `.../ada_blockwise_gemm/sm89_utils.cuh` (mma :41-82, bf16 fallback :97-104, granularity :138-140, copies :162-221, smem :226-245) |
| dispatch sm89 | `.../fp8_blockscale_gemm_kernel.cuh:681-715` (dense), `:1082-1122` (bmm); A-quant kernel `:182-274`, launcher `:599-616`; deep_gemm sm90 swap-AB `:640-679` |
| torch op + constraints | `cpp/tensorrt_llm/thop/fp8BlockScalingGemm.cpp:85-116` (K%128, N%16), `:246-253` (case 89), `:343-349` (no sm89 MoE); `thop/fp8Quantize.cpp:32` |
| model flow / f32 scales | `tensorrt_llm/_torch/modules/linear.py:1177-1181, 1193-1229`; `tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/trtllm_quant.py:130-132` |
| cutlass pin | `3rdparty/fetch_content.json:12-15` (v4.4.2, fetched not vendored) |
| cuBLASLt FP4-only modes | `cpp/tensorrt_llm/common/cublasMMWrapper.cpp:80-118` |
| vLLM Marlin FP8 | `E:\coding\Insignia\vllm\` — `csrc/libtorch_stable/quantization/marlin/{dequant.h:321-395, marlin_mma.h:74-116, marlin.cu:400-441, marlin_template.h, gptq_marlin_repack.cu:117-236}`; `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py:35-46,107-219,377-391` |
| vLLM act quant | `E:\coding\Insignia\vllm\csrc\libtorch_stable\quantization\w8a8\fp8\per_token_group_quant.cu:21-96,166-242` |
| config facts | `audits/w2/loader-27b-spec.md:76-83` (activation_scheme dynamic, weight_block_size [128,128]) |

Web sources: [TechPowerUp RTX 4070 SUPER specs](https://www.techpowerup.com/gpu-specs/geforce-rtx-4070-super.c4186),
[Tom's Hardware 48 MB L2 correction](https://www.tomshardware.com/pc-components/gpus/nvidia-corrects-mistake-with-one-of-its-new-rtx-40-super-gpus),
[NVIDIA cuBLAS 12.9 blog (block scaling = Hopper; Ada = tensor-wide only)](https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/),
[cuBLAS docs (outer vector / 1D-2D block scaling, cc 9.0+)](https://docs.nvidia.com/cuda/cublas/index.html),
[cuBLAS 13.0.1 PDF](https://docs.nvidia.com/cuda/archive/13.0.1/pdf/CUBLAS_Library.pdf),
[cudalibrarysamples #310 (VEC128_32F / BLK128x128_32F min CUDA 12.9)](https://github.com/nvidia/cudalibrarysamples/issues/310),
[CUTLASS changelog (sm89 fp8 2.x-only; blockwise sm90+)](https://docs.nvidia.com/cutlass/4.2.1/CHANGELOG.html),
[CUTLASS example 67 Hopper blockwise](https://github.com/NVIDIA/cutlass/blob/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu),
[MARLIN paper](https://arxiv.org/abs/2408.11743), [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin),
[vLLM FP8 docs (Marlin W8A16 on pre-Hopper)](https://docs.vllm.ai/en/stable/features/quantization/llm-compressor/fp8/).
