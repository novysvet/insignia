# TRT-LLM sm_89 blockwise FP8 GEMM — extraction report (agent w4)

Repo: `E:\coding\Insignia\TensorRT-LLM` (read-only clone). All paths below relative to that root
unless prefixed `Insignia\`. Target HW: RTX 4070 SUPER, sm_89, 56 SMs, ~504 GB/s GDDR6X, 100 KiB
max smem/CTA, 36 MB L2.

---

## 1. Exact file list + line-precise walkthrough

The sm_89 hand-rolled blockwise FP8 GEMM lives in
`cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/`:

| File | Role |
|---|---|
| `ada_blockwise_gemm/sm89_fp8_gemm_1d1d.cuh` (454 L) | **the kernel** — cute-atom-level (no CUTLASS collective), cp.async pipeline + raw fp8 mma + register promote |
| `ada_blockwise_gemm/sm89_utils.cuh` (248 L) | PTX for `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, TiledMMA traits, smem/swizzle/layout traits, SharedStorage |
| `fp8_blockscale_gemm_kernel.cuh` (1191 L) | dispatch: `gemm_dispatch_sm89` L681-715, `strided_batch_gemm_dispatch_sm89` L1082-1122, arch routing L814-826; also the quant kernels `scale_1x128_kernel` L183-274 (+MoE variant L277-401) and `scale_128x128_kernel` L489-569 |
| `fp8_blockscale_gemm.cu` (335 L) | runner wrapper, workspace sizing (irrelevant: it throws "only support Hopper" unless the quantize path is bypassed — the raw-fp8 `gemm()` overload at L83-90 reaches sm89 dispatch fine) |
| `fp8_blockscale_gemm.h` (171 L) | runner interface |
| `sm120_blockwise_gemm/*`, `fp8_blockscale_tma_utils.cuh`, `fp8_blockscale_quant_packed.cu` | Blackwell/Hopper-only (tcgen05 block-scaled MMA, TMA tensormaps, UE8M0 packing) — **irrelevant to sm_89** |

### 1.1 Launch config (`fp8_blockscale_gemm_kernel.cuh:681-715`)

```cpp
static constexpr int Stages = 3;
using TileShape = cutlass::gemm::GemmShape<32, 128, 128>;   // L692-693
// grid = (ceil(M/32), ceil(N/128), 1); block = 128 threads (4 warps)   L700-704
cudaFuncSetAttribute(sm89_fp8_gemm_1d1d_impl<GemmKernel>,
    cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);           // L706-707
sm89_fp8_gemm_1d1d_impl<GemmKernel><<<grid, block, kSmemSize, stream>>>(...)
```
Arch routing at L814-818: `arch == 89 -> gemm_dispatch_sm89` (120 -> sm120 hand-rolled, else
deep_gemm JIT/cuBLAS Hopper paths). BMM variant identical tile, `blockIdx.z` = problem
(`sm89_fp8_gemm_1d1d.cuh:33-48`, dispatch L1082-1122).

### 1.2 Data layouts & scale granularity (`sm89_utils.cuh:116-159`, `sm89_fp8_gemm_1d1d.cuh:72-112`)

- A = activations `[M,K]` row-major e4m3, ld = K. B = weights `[N,K]` row-major e4m3 ("TN" GEMM).
- Scale granularity (L138-140): **SFA = 1x128** (per row per 128-K), **SFB = 128x128**, both fp32.
- SFA gmem layout is **transposed, M-contiguous, padded to 4 rows**
  (`sm89_fp8_gemm_1d1d.cuh:78-89`): `ScaleM=((M+3)>>2)<<2`, tensor `(ScaleM, ScaleK)` stride
  `(1, ScaleM)` → `sfa[k_block][m]`, m fastest. The quant kernel must write exactly this.
- SFB layout `(ScaleN, ScaleK)` stride `(K/128, 1)` → row-major `sfb[n/128][k/128]` — **identical
  to Insignia's current `scales[row>>7][kblock]` bf16 layout** (modulo f32 vs bf16).
- Constraint notes: `TileM%16==0, TileN%32==0, TileK%32==0` (L126-128); since `ScaleGranularityK
  == TileK == 128`, **one K-tile = exactly one scale slab** — the structural invariant that makes
  the register promote correct.

### 1.3 cp.async staging (gmem→smem), 3 stages

Copy atom (`sm89_utils.cuh:162-163`): `SM80_CP_ASYNC_CACHEALWAYS<uint128_t>` = 16 B
`cp.async.cg`, thread-value layout `Layout<Shape<16,8>> × <1,16>` → each of 128 threads moves
16 B; a 32×128 A tile = 4 KiB/stage, B 128×128 = 16 KiB/stage.

Smem swizzle (L169-171): `composition(Swizzle<3,4,3>{}, Layout<Shape<16,128>, Stride<128,1>>)`
— 128-B swizzle over a 16×128-B atom, `tile_to_shape` to `(TileM, TileK, Stages)`. Bank-conflict
free for the `ldmatrix` reads below; also makes each 16-B cp.async land in a distinct bank pair.

Pipeline (`sm89_fp8_gemm_1d1d.cuh`):
- L262-289 prologue: `cute::clear` smem (zero-fill = OOB predication), then for `k_pipe in
  [0, Stages-1)`: **predicated** `copy_if` for A/B/SFA (`tApA` = row-coord < residue_m, L238-260),
  unconditional for SFB, then `cp_async_fence()`. If `k_pipe >= k_tile_count` predicates are
  cleared and the *last valid* tile is re-fetched (no OOB when K-tiles < stages).
- L336-337 main-loop head: `cp_async_wait<Stages-2>()` (= `cp.async.wait_group 1`, one group in
  flight) + `__syncthreads()`, then prefetch smem→rf for SFA/SFB/A/B of stage 0.
- L346-392 main loop, per K-tile: a `for_each<NUM_GROUP_N>` (4 iterations of 32-N) where at
  `n_block==0` the **next K-tile's gmem→smem cp.async is issued** + `cp_async_fence()` +
  pipe-index rotation (`smem_pipe_write = smem_pipe_read; ++smem_pipe_read mod Stages`, L377-380)
  and `scale(i) = SFA(i)*SFB(0)` computed (L381-382); at `n_block==NUM_GROUP_N-1` the wait for
  the freshly issued tile happens (`cp_async_wait<Stages-2>` + `__syncthreads()`, L357-358) and
  new SFA/SFB + A fragments are reloaded (L359-360, 388). B fragments for the *next* n_block are
  prefetched to registers each iteration (L362-364) — 2-level (smem-stage × reg) double buffer.
- L394-430 "load tail": consumes the last `Stages-2` prefetched tiles with shrinking
  `cp_async_wait<Stages-3-WaitIndex>`; L432-445 "mma tail": final pipe's 4 n_blocks, no waits.
- Only **one `__syncthreads()` per K-tile** (inside the n_block==3 branch).

### 1.4 mma.sync m16n8k32 e4m3 — the exact PTX (`sm89_utils.cuh:41-82`)

Raw inline PTX, no wmma, no CUTLASS hardware abstraction:

```cpp
// sm89_utils.cuh:41-67 — the entire hardware interface
struct SM89_16x8x32_F32F8F8F32_TN
{
    using DRegisters = float[4];
    using ARegisters = uint32_t[4];   // 16 x e4m3
    using BRegisters = uint32_t[2];   // 8  x e4m3
    using CRegisters = float[4];

    CUTE_HOST_DEVICE static void fma(float& d0, float& d1, float& d2, float& d3,
        uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
        uint32_t const& b0, uint32_t const& b1,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
#if defined(CUTE_ARCH_MMA_F32_SM89_ENABLED)   // __CUDA_ARCH__ >= 890  (L33-35)
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

// sm89_utils.cuh:69-82 — fragment layouts (paste-ready)
template <>
struct MMA_Traits<SM89_16x8x32_F32F8F8F32_TN>
{
    using ValTypeD = float; using ValTypeA = float_e4m3_t;
    using ValTypeB = float_e4m3_t; using ValTypeC = float;
    using Shape_MNK = Shape<_16, _8, _32>;
    using ThrID = Layout<_32>;
    using ALayout = Layout<Shape<Shape<_4,_8>, Shape<_4,_2,_2>>,
                           Stride<Stride<_64,_1>, Stride<_16,_8,_256>>>;
    using BLayout = Layout<Shape<Shape<_4,_8>, Shape<_4,_2>>,
                           Stride<Stride<_32,_1>, Stride<_8,_128>>>;
    using CLayout = SM80_16x8_Row;   // c0,c1 = row r, cols n,n+1; c2,c3 = row r+8
};
```

Notes for lifting into Insignia (no cute needed):
- A per thread = 4×b32 = 16 e4m3; per-lane row mapping for a 16×32 A tile: lane quad q∈{0..3}
  holds row `q + 4*(lane/4)`? — decode from ALayout: thread t: `(t%8)*... ` simplest is to keep
  cute's `make_tiled_copy_A` with the ldmatrix atom below, or hand-code ldmatrix as in §1.5.
- TiledMMA (L107-114, 156-159): atom × `Layout<2,2,1>` (4 warps: 2×2 warp grid) ×
  `Tile<32,32,32>` permutation → per-warp 16×16 output tile, CTA mma tile 32×32 per call; whole
  CTA tile 32×128 (N split into `NUM_GROUP_N=4` 32-wide n_blocks, K=128 = 4 k-iters inside
  `cute::gemm` on register fragments).

### 1.5 ldmatrix usage (`sm89_utils.cuh:173-175`)

```cpp
using SmemCopyAtomLoad = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;  // A and B
```
`SM75_U32x4_LDSM_N` = `ldmatrix.sync.aligned.m8n8.x4.shared.b16 {4 regs}, [addr]`
(non-transposed). e4m3 pairs are treated as one b16 element, so one `x4` ldmatrix per thread
loads 16 B = 16 e4m3 = exactly one A-fragment (4×u32); two per B... per k-slice B-fragment is
2×u32 (8 e4m3). The `Swizzle<3,4,3>` smem layout + LDSM_N combine so that the ldmatrix addresses
are swizzle-XOR'd by the iterator (`tXsA` cute tensor does the XOR arithmetic in address
computation). Accumulator A-frag regs are loaded **once per K-tile and reused across all 4
n_blocks** (L385 gemm per n_block with same `tCrA`); B-frag regs are loaded per n_block.

Scale smem→rf: plain `UniversalCopy<float>` (`SmemCopyAtomScale`, L204) with TV layouts
L194-221: SFA read so each thread gets the scale of its accumulator rows (`Shape<(4,8,2,2),2>`);
SFB TV layout has all-zero strides = **one broadcast scalar per CTA** (TileN 128 == granularity
128 → SFB is a single float per K-tile), `tXrSFB(0)`.

### 1.6 Where/how block scales are applied — register promote per 128-K slab

The heart (`sm89_fp8_gemm_1d1d.cuh:167-191` + call sites L381-390):

```cpp
// per K-tile, at n_block==0 (L381-382):
cute::for_each(..., [&](auto i) { scale(i) = tXrSFA(i) * tXrSFB(0); });  // f32 SFA[row]*SFB

// per n_block (L384-390):
cute::clear(temp);                                  // temp = fresh f32 accumulator
cute::gemm(mma, tCrA, tCrB(_,_,n_block), temp);     // RAW e4m3 mma, f32 accum, NO scales
promote(accum, temp, scale, n_block);               // accum += temp * scale  (FFMA)

// promote body (L167-191): per accumulator element, in registers:
accum(coord_d) += temp_accum(coord_c) * scale(mma_m, mma_iter_m, _0{});
```

So: raw fp8 MMA runs 128-K deep into a *temporary* f32 accumulator; the running result is
promoted with one FFMA per output element per 128-K slab. **No dequant of the fp8 data ever
happens**; scales stay f32 end-to-end. Per-thread cost accounting for tile 32×128×128:
4096 MACs/thread per slab (each thread owns 32 accum elements) vs 32 FFMA + (32/4 SFA muls once
per tile) → promote overhead ≈ 1.6% of mma issue slots.

### 1.7 Epilogue (`sm89_fp8_gemm_1d1d.cuh:114-165`)

Register f32 → bf16 convert → swizzled smem staging (`Swizzle<3,3,3>` 8×64 atom,
`SmemLayoutO` L178-181) via `tiled_copy_C` of the mma (`AutoVectorizingCopy`) + `__syncthreads()`
→ re-read smem with a coalescing tiled copy `TiledCopyS2G` (16×64, `UniversalCopy<uint128_t>`
16-B gmem stores, L189-190) → predicated gmem store with `residue_m/residue_n` bounds check
(L151-164). The load and store shared storages are a **union** (`sm89_utils.cuh:239-243`), so the
epilogue adds 0 smem. Output bf16 row-major `[M,N]`, ld=N.

---

## 2. Quantitative design deltas vs Insignia `src/fp8.cu` (fp8_gemm_kernel, L109-190)

| Axis | Insignia `src/fp8.cu` | TRT-LLM ada kernel |
|---|---|---|
| MMA | `wmma::mma_sync` bf16 16x16x16 (= 2× slower HMMA path) | raw `mma.sync m16n8k32 e4m3` at 2× bf16 rate |
| B (weights) path | cp.async raw e4m3 smem → **dequant to bf16 into a second smem buffer** (L136-153: e4m3→f32→×bf16-scale→bf16 round trip, 2 extra `__syncthreads()` per 64-K step L169-171) | cp.async e4m3 smem → `ldmatrix` → mma directly. **Zero conversion instructions**, one sync per 128-K tile |
| Scales | baked into B values *before* bf16 rounding → each weight re-rounded to bf16 after scale (quantization-error double-dip) | f32 in registers; `accum += temp*(SFA*SFB)` — data stays e4m3-exact |
| Activations | bf16 `x16` consumed as-is (bf16 mma) | quantized e4m3 1×128 + f32 scales (same promote) |
| Scale granularity supported | weight-only 128×128 (bf16) | A: 1×128 (per token per 128-K), B: 128×128, both f32 |
| Pipeline | 2 stages, KT=64, manual double buffer | 3 stages × K=128, cp.async wait_group 1, reg-level B double-buffer across 4 n_blocks |
| smem/CTA | As 2×64×72×2B=18 KiB + Bs 2×32×72×2B=9 KiB + Braw 2×32×64=4 KiB ≈ 31.5 KiB | 61 952 B (see §3) |
| Tile | 64(M)×32(N)×64(K), 256 thr, grid=rows/32 | 32×128×128, 128 thr, grid=(M/32, N/128) |
| Epilogue | `wmma::store_matrix_sync` f32 straight to gmem (strided 32-B stores at our Y layout y[m*rows+n]) | rf→swizzled smem→128-B coalesced bf16 stores |

The delta that matters at our shapes is **not** mma rate (see §4: we are DRAM-bound at T≤64) but
issue-slot and smem-bandwidth relief: our dequant spends ~6 instructions per weight plus a full
extra smem round trip and two barriers per 64-K; theirs spends 0 per weight and one barrier per
128-K. Plus precision: no bf16 re-rounding of scaled weights.

### Paste-ready idiom (the promote + fp8 mma, PTX level)

```cpp
// per 128-K slab, per output element (thread-local f32 regs):
//   acc[i] = fma(tmp[i], sfa_row * sfb_block, acc[i]);
// with tmp[] produced by:
asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3));
// a0..a3: 16 x e4m3 packed 4/regs (from ldmatrix.x4 on the Swizzle<3,4,3> smem);
// b0..b1: 8 x e4m3; c/d: f32. Repeat K/32 times into a *temp* accumulator,
// then one FFMA per element with sfa[row]*sfb[n>>7] (f32), accumulate across slabs.
```

---

## 3. Their handling of the ugly edges

- **K not multiple of tile**: there is *no K predication*. Because `TileK == ScaleGranularityK ==
  128`, K must effectively be a multiple of 128 (16-B cp.async would read past row end otherwise;
  SFB count `ceil(K/128)` also assumes full slabs). Prologue guards only the *tile-count <
  stages* case by re-fetching the last valid tile (L272-278). **Our K=5120 (40 slabs) is safe.**
- **Scale boundary alignment**: K-tile == scale block, so the promote at tile boundaries is
  automatically scale-aligned — no intra-tile scale switch exists by construction. SFA rows are
  padded to a multiple of 4 (`ScaleM=((M+3)>>2)<<2`, L78; quant kernel pads identically,
  `stride_scale_dim_y = div_up(dim_y,4)*4`, L189) because the SFA gmem→smem copy does 4-B
  contiguous reads down M. SFB needs `N%128` granularity = 1 block per CTA-tile, read
  unconditionally (safe only if N%128==0 or scales over-allocated; our 10240/17408 are exact).
- **T (M) tail**: `residue_m = M - kTileM*blockIdx.x` clamped to kTileM (L233-236). Two layers:
  (a) gmem loads predicated per 16-B row chunk (`tApA(m,0) = coord < residue_m`, L238-244) into
  smem that was pre-zeroed with `cute::clear` (L263-266) → OOB rows contribute exact zeros to
  the mma; (b) epilogue store re-checks `coord < residue_m/residue_n` (L159). No wasted-store
  masking needed since output is guarded.
- **f32 accumulation promote pattern**: dual accumulator sets in registers — persistent `accum`
  (f32, 32 regs/thread at 32×128 tile) + per-slab `temp` (8 regs/thread per 32×32 n_block);
  `clear(temp)` → raw fp8 gemm → `promote()` FFMA into `accum`. Scale product `SFA*SFB` hoisted
  to once per K-tile per thread (32 muls) at n_block==0 (L381-382).
- **Occupancy**: 128 threads, smem = A 3×4096 + B 3×16384 + SFA 3×32×4 + SFB 3×4 = 61 836 B →
  `aligned_struct<128>` rounds `kSmemSize` to **61 952 B (~60.5 KiB)** (`sm89_utils.cuh:226-245`)
  ⇒ exactly **1 CTA/SM** (2 CTAs would need 121 KiB > 100 KiB sm_89 cap), 4 warps/SM = 6.25%
  warp occupancy — latency hidden by cp.async stages + reg-level B buffering, not threads.
  Register inventory floor (derived): 32 accum + 8 temp + 8 A-frag + 32 B-frag (4 n_blocks) +
  ~2 scale + ~1 SFB ≈ 83 data regs + addressing → comfortably < 255 even with TileM=64 variant
  (would be 64+16+16+32 ≈ 128 data regs).

---

## 4. What we can lift for T≤64 prefill (rows N ∈ {10240, 17408}, cols K=5120) on 56 SMs

**The math says DRAM-bound — TFLOPS is the wrong metric here.** 4070 SUPER dense FP8 tensor peak
≈ 285 TOPS (8× the 35.6 shader TF; per SM/clk ≈ 2048 MAC), DRAM ≈ 504 GB/s:

| shape | 2·T·N·K | mma floor | W bytes (N·K) | DRAM floor | bound |
|---|---|---|---|---|---|
| T=64, N=10240, K=5120 | 6.7 GF | 23.6 µs | 52.4 MB | 104 µs | DRAM 4.4× |
| T=64, N=17408, K=5120 | 11.4 GF | 40.1 µs | 89.1 MB | 177 µs | DRAM 4.4× |
| T=8, N=10240 | 0.84 GF | 2.9 µs | 52.4 MB | 104 µs | DRAM 36× |
| T=1 (GEMV) | 0.10 GF | 0.4 µs | 52.4 MB | 104 µs | DRAM 260× |

Crossover to mma-bound is T ≈ 282 (2·T·504/285e3). So for T≤64:

- **Recommended tile: their exact 32×128×128/3-stage config works as-is.** Grid =
  (ceil(T/32), N/128) = (2, 80) = 160 CTAs or (2,136) = 272 CTAs on 56 SMs → 3-5 waves, 1 CTA/SM.
  Weight-slice per CTA (128×5120 = 640 KiB) streams L2-friendly: the two m-tiles of the same
  n-tile read the same W slice, second read hits L2 in-wave → W crosses DRAM ~once. Expected:
  ≥ 85% of 504 GB/s ⇒ **~110-125 µs for N=10240 (≈ 60 TFLOPS effective at T=64), ~190-210 µs for
  N=17408** — a TFLOPS number nobody should brag about; the right target is "W-read time".
- A TileM=64 variant (traits allow it; sm120 uses 64×128 for M>64) halves grid_m to 1 and
  eliminates the double W read entirely, smem ≈ 74.6 KiB still 1 CTA/SM — worth benchmarking
  both; TileM=64×TileN=128 = 8192 outputs/128 threads = 64 accum regs.
- **Activation quant becomes mandatory**: their kernel eats e4m3 A + f32 SFA (1×128). We must add
  `scale_1x128_kernel` (below) for the T×5120 activation per layer (trivially cheap: 64×40
  warps total). Alternative if we refuse activation quant: keep bf16 A and hand-write an mma
  bf16-A × e4m3-B? — *no such mixed mma exists*; per-tensor-A e4m3 (single f32 scale per GEMM)
  would also work with this kernel by making SFA constant, but per-128-K dynamic quant is both
  more accurate and what the kernel was built for.
- **Our current wmma kernel is also DRAM-bound** at these shapes (same W bytes), so raw speedup
  from the mma swap alone will be modest (~1.1-1.3×, from removed dequant issue pressure and
  better cp.async overlap); the real wins are (a) precision (no bf16 re-round of scaled weights),
  (b) unifying decode GEMV (T=1) and prefill into one scale system, (c) headroom to grow T.
- **GEMV (T=1)**: our `fp8_gemv_kernel` (warp-per-row, f32 promote per 128-block — same idiom!)
  is already DRAM-bound at ~104 µs floor; keep it. The TRT kernel at M=1 wastes 31/32 of the mma
  but is equally DRAM-bound; nothing to lift there.

**Irrelevant / do-not-lift**: `fp8_blockscale_tma_utils.cuh` (CUtensorMap, Hopper+ only),
deep_gemm JIT dispatch (`fp8_blockscale_gemm_kernel.cuh:640-679` — requires sm90 wgmma +
runtime nvrtc/cubin compile), sm120 blockwise (`tcgen05.mma` block-scaled, Blackwell-only),
`fp8_blockscale_quant_packed.cu` (UE8M0 packing for sm100 deep_gemm format),
grouped/MoE variants, cuBLAS wrappers.

**CUTLASS sm89 fp8 blockwise**: does **not** exist, in the clone or upstream — TRT-LLM fetches
CUTLASS `v4.4.2` at build time (`3rdparty/fetch_content.json:12-17`; not vendored, so nothing to
link standalone anyway), and CUTLASS blockwise-scaled FP8 collectives are sm90/sm100/sm120 only
(TMA/UMMA-dependent). That absence is precisely *why* TRT-LLM hand-rolled the ada kernel — it is
the only sm_89 128-block-scaled fp8 GEMM in the tree. Lifting `sm89_utils.cuh` +
`sm89_fp8_gemm_1d1d.cuh` requires cute headers (CUTLASS include path) — or port the ~450 lines to
raw PTX (the kernel only uses: cp.async 16B, ldmatrix.x4, the one mma, FFMA, swizzle XOR math).

---

## 5. Their fp8 activation quant (`per_token_group_quant` equivalent)

No `per_token_group_quant` exists in this TRT-LLM version (checked `kernels/quantization.cu`:
only per-tensor/per-token `invokePerTokenQuantization` L76, FP4/MXFP8 quant). The 1×128 dynamic
activation quant we need is `scale_1x128_kernel` — `fp8_blockscale_gemm_kernel.cuh:182-274`,
launched at L843-846 / L608-615 as `<<<SMs*8, 256>>>`:

- One warp per (row, 128-K block), grid-stride (L191-192). Each lane loads 2×bf16 via bf162
  loads at `lane*2`, two 64-elem chunks (L200-215) → warp covers exactly 128 elems.
- Amax: `__hmax/__habs` tree + warp shuffle reduce (L216-229, `find_max_elem_in_warp` L171-180),
  clamp `amax = max(amax, 1e-10)` (L230).
- Scale: `quant_scale = 448/amax`; stores **dequant** scale `1/quant_scale` f32 at
  `scales[kblock * ceil(M/4)*4 + row]` (L250-252) — the transposed, 4-padded SFA layout the GEMM
  reads. Optional UE8M0 power-of-2 rounding branch L234-244 (sm100-only need, skip).
- Quantize: `output[lane] = e4m3(x * quant_scale)` in-place re-read of the register frags (L256-271).
  K-tail predicated by `kblock*128 + i*64 + lane*2 < dim_x` (L259).
- B-side static weight quant `scale_128x128_kernel` L488-569 (warp per 128×128 block, scalar
  loops, `448/amax`, row-major scale out) — we already have equivalent offline quant for our
  checkpoint, but note theirs emits **f32** scales, ours are packed bf16 (`uint16`); the ada
  kernel requires f32 (cp.async 4-B scale loads + f32 promote). Our quantizer would need a f32
  scale sidecar (40 KiB per N=10240 layer — noise).

---

## 6. Bottom line

Everything needed for a paste-ready Insignia port is in two files:
`ada_blockwise_gemm/sm89_utils.cuh` (PTX mma + traits + swizzles) and
`ada_blockwise_gemm/sm89_fp8_gemm_1d1d.cuh` (pipeline + promote + epilogue), plus
`scale_1x128_kernel` for activations. Tile 32×128×128, 3 stages, 128 threads, 61 952 B smem,
1 CTA/SM, raw `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` into a per-slab temp
accumulator, `accum += temp*(SFA[row]*SFB_tile)` in f32 registers once per 128-K, K%128==0 and
N%128==0 assumed (our shapes satisfy both), M tail handled by predicated zero-fill loads +
guarded epilogue. At T≤64 the kernel is DRAM-bound (104 µs / 177 µs floors per GEMM at
N=10240/17408); the port's value is precision + issue efficiency + a unified scale system, not
TFLOPS.
