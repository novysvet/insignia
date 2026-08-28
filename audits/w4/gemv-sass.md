# w4 audit: decode GEMV family — measurement + SASS

Scope: `src/mxfp4.cu`, `src/mxfp4_i4.cu`, `src/fp8.cu` (GEMV parts), `src/bench_mxfp4_mlx.cu`,
`src/bench_gemm.cu`. All benches run read-only via `python tools/rundll.py` on existing DLLs;
all SASS from `cuobjdump -sass / -res-usage` (CUDA 13.3) on `build/bench-mxfp4.dll`,
`build/test-i4.dll`, `build/test-fp8.dll`, `build/generate.dll`. Nothing compiled, nothing modified.

Hardware frame: RTX 4070 SUPER, sm_89, 56 SMs, 504 GB/s DRAM (= 469.5 GiB/s), 48 MB L2,
64K regs/SM, 1536 threads/SM, ~2.48 GHz. GiB/s below are as printed by the benches.

Production decode path (checked in `src/decode.cu`): every linear goes through
`mxfp4_gemv_v2` / `mxfp4_gemv_v2_i4` (single token) and the q8 pair kernels for the
speculative/MTP pair path — so those two families are *the* hot kernels.

---

## 1. Measured bandwidth (bench-mxfp4.dll, this run)

Weights per pass = rows·cols·0.53125 B (16B nibbles + scale bytes). 100 iters, cuda events.

### Single-token GEMV (rows == 8192 / 248320 only, per bench harness)

| shape | kernel | ms | GiB/s | note |
|---|---|---|---|---|
| 8192x4096  (17.8 MB w) | mlx w=2 | 0.105 | 157.6 | old layout |
| 8192x4096 | **v2** (prod) | 0.050 | 333.9 | L2-resident (17.8 MB < 48 MB L2) |
| 8192x4096 | A-arith | 0.054 | 308.2 | arith decode, no LUT |
| 8192x4096 | C-2grp | 0.044 | 373.7 | |
| 8192x4096 | D-2rows | 0.043 | 390.6 | |
| 8192x4096 | **E-4rows** | 0.035 | **475.5** | >DRAM peak → L2 is serving this |
| 248320x4096 (540 MB w) | mlx w=2 | 2.854 | 176.3 | DRAM truth |
| 248320x4096 | **v2** (prod) | 1.154 | **436.1** | 468 GB/s = 93% of 504 |
| 248320x4096 | A-arith | 1.411 | 356.7 | |
| 248320x4096 | C-2grp | 1.189 | 423.2 | |
| 248320x4096 | **D-2rows** | 1.119 | **449.8** | 483 GB/s = 95.8% of peak |
| 248320x4096 | E-4rows | 1.123 | 448.1 | |

**L2 warning (confirms the w3 caveat):** every shape with rows ≤ ~150K at cols 4096 keeps
the whole weight matrix inside 48 MB L2 after the first (untimed) warmup call, so the 8192-row
numbers measure L2/issue behavior, not DRAM. Only 248320x4096 (540 MB) is honest DRAM
streaming. Real decode (32 layers sweeping 4.8 GB/token) is DRAM-bound like the 248320 row.

### Pair path (2 activation rows, one weight pass — speculative decode)

| shape | gemv2(2x) fp32 | q8g(2x) dp4a | F-dp4a2x | v2 run twice |
|---|---|---|---|---|
| 8192x4096 | 233.4 | 302.9 | 408.7 | 189.0 |
| 4096x4096 | 215.5 | 291.6 | 381.4 | 169.7 |
| 12288x4096 | 228.7 | 356.8 | 371.1 | 193.3 |
| 4096x12288 | 176.8 | 342.6 | — | 164.0 |
| 1024x4096 | 139.2 | 126.7 | 230.3 | 102.3 |
| 248320x4096 | 280.6 | 431.8 | **446.8** | 215.9 |

Per token delivered: pair q8g at 248320 gives 431.8 GiB/s of weights for **two** tokens =
~2x the per-token efficiency of v2 (436.1 for one token). The dp4a pair kernels are not a
loss — they are the best per-token path measured.

### GEMM (bench-gemm.dll, T=64 prefill — for context only)

8192x4096 76 GiB/s / 19.7 TF · 12288x4096 96 / 24.9 · 4096x12288 42 / 10.7 · 248320x4096
101 GiB/s / 26.1 TF. (Bench prints max_rel up to 0.092 → "FAIL" lines are the harness
threshold vs random-float cancellation, not kernel crashes; timing valid. Separate w4 topic.)

`AGENTS.md`'s "~150 GiB/s on 4096x4096" for the MLX matvec is stale — that is the old
`mxfp4_gemv_mlx` number (157.6 today); production `v2` is at 334–436.

---

## 2. SASS: main-loop anatomy (cuobjdump -sass, sm_89)

Per **32-element group** (one group = 16 B packed nibbles + scale byte; "warp-instrs" are
SIMT warp instructions; a group is owned by one lane, so /32 for warp-level counts).
`unroll` = groups per unrolled body. Main loop = largest backward-BRA span.

| kernel | DLL | regs | occ. blocks/SM (thr/SM) | unroll | instr/group (warp) | LSU per group (warp) | spills |
|---|---|---|---|---|---|---|---|
| mxfp4_gemv_v2 | bench-mxfp4 | 64 | 4 (1024, 67%) | 4 | 6.2 | 2.06 (64×LDS.32 + 2 LDG /32 lanes) | 0 |
| mxfp4_gemv_v2_i4 | test-i4 | 80 | 3 (768, 50%) | 4 | 7.1 | 2.06 | 0 (STACK:64 unused) |
| bench::gemv_d (2 rows) | bench-mxfp4 | 80 | 3 (768, 50%) | 4 | 6.2/row | ~1.3/row | 0 |
| bench::gemv_e (4 rows) | bench-mxfp4 | **128** | **2 (512, 33%)** | 2 | 5.4/row | ~1.0/row | 0 |
| mxfp4_gemv2_q8g (pair) | bench-mxfp4 | 44 | 5 (1280, 83%) | 4 | 3.8/row | LDS.64×2 + LDG.32×4.5 + LDG.128 | 3 (prologue) |
| mxfp4_gemv_q8g (1 row, generate.dll) | generate | 40 | 5+ (83%+) | 4 | ~3.2/row | same pattern | 8 (prologue) |
| bench::gemv_f (pair, smem x) | bench-mxfp4 | 46 | 5 (1280, 83%) | 4 | 3.4/row | LDS.64×2 + LDS.128×0.5 + LDG | 0 |
| mxfp4_gemv_ab2_q8_i4 (fused qkv) | generate | 40 | 5 (83%) | 4 | ~4/row | LDS.64×2 + 2.25 LDS.32 | 0 |
| fp8_gemv | test-fp8 | ~46 | 5-6 (83%+) | 2 | 2.8 | LDS.128×0.25 | 0 |
| fp8_gemv2 | test-fp8 | 46 | 5 (83%) | 2 | 3.45 | LDS.128×0.5 | 0 |
| mxfp4_gemv_dp4a<2> (old) | bench-mxfp4 | 28 | high | ~1 | ~12+ | 4-lane LDG.32 only | 0 |

Instruction mix per 32-elt group (per owning lane; from loop bodies):

* **v2 (e8m0)**: 32×(SHF + LOP3&0xf + IMAD.IADD + LDS[LUT]) + 32×(IMAD + LDS[x]) +
  32–33 FFMA + 4 FADD + 1 LDG.E.EF.128 + 1 LDG.E.U8. `__ldcs` correctly emitted
  (`LDG.E.EF.128`). The `if (which==…)` p0..p3 routing compiled to *static* FFMA chains —
  zero branches, 1 BRA (back-edge), 16 predicated instrs in the whole kernel.
* **v2_i4**: identical + ~0.9 extra instr/element for `__half2float(scale)` (HADD2.F32 path);
  scale is shared per 64-elt super-group but re-decoded **every group**; 80 regs → 50% occ.
* **gemv_e**: per iteration 4 rows share the 32 x-LDS and staging; LUT LDS still 32/row.
  128 regs is the price of 4×uint4 + 4×scale + 4 accum chains in flight.
* **q8g pair**: per group 16 IDP.4A.S8.S8 + 24 PRMT + 16 LDS.64 (btab) + ~9/group scalar
  `LDG.E.CONSTANT` 4 B loads reading xq/xs **from global** (xq layout is lane-contiguous
  32 B — could be 2×LDG.128) + I2FP + 3 FMUL + FFMA epilogue.
* **fp8_gemv**: decode = `((u&0x007f007f)<<7)|((u&0x8080)<<8)` (2 LOP3) → `HADD2.F32`×2
  (half22float2) → `FMUL.FTZ ×256`×2, i.e. ~5–6 instrs per 2 weights; x read as 8×LDS.128
  broadcast-style (consecutive lanes, conflict-free), zero LUT, zero IMAD in the loop
  (immediate offsets `+0x10/+0x20` — groups is baked out).
* **old dp4a<2>**: the `int vals[8]` table with dynamic index `q&7` compiles to an
  **ISETP.EQ-chain + 8 predicated IMAD.MOV per nibble** (~16 instrs/nibble), plus BSSY/BSYNC
  stacks, 132 ISETP + 18 BRA in the loop, and only lanes 0–3 load weights (16 B/warp/group
  instead of 512 B). 28/32 lanes idle during compute.

Spills: **no LDL/STL in any hot loop** (q8g/q8g-single have 3–8 prologue-only LDLs; v2_i4
carries an unused 64 B stack frame). Unrolls are clean: fully unrolled bodies, 1 back-edge
branch, tails separated. The codegen quality is high; the inefficiencies are algorithmic.

### Top-3 SASS-level inefficiencies per production kernel

**mxfp4_gemv_v2 (+i4 twin)**
1. **2 LSU ops per element**: one LDS.32 for the e2m1 LUT + one LDS.32 for x, per element.
   The x LDS each need their own IMAD (address = k·groups·4, `groups` runtime) — 32 IMADs +
   32 LDS.32 per group just to feed FFMA.
2. **LUT index plumbing**: SHF+LOP3+IMAD.IADD per element (~3 instrs) before every LUT LDS.
   A-arith proved removing the LDS but keeping a select-chain decode is a net loss (308 vs
   334) — the fix must remove instructions, not move them.
3. v2_i4 only: redundant per-group `__half2float` of a scale shared by 2 groups (+29
   instrs/group, +16 regs, occupancy 67%→50%).

**mxfp4_gemv2_q8g / gemv_q8g / ab2 (pair-LUT dp4a family)**
1. **xq/xs read from global as scalar LDG.E 4 B** (9/group; 72 per unrolled body) — lane's
   32 B is contiguous; should be 2×LDG.E.128 (or LDS after staging). ~4x LSU waste.
2. btab fetch chain per weight byte: SHF + IMAD.SHL×8 + LOP3 0x7f8 + LDS.64 = 5 instrs to
   get 2 nibbles' broadcasts; PRMT×24/group to re-interleave. vs direct nibble arithmetic
   (fp8-style bit-build) this is LSU traffic, but it replaces 8 FFMA with 4 IDP.4A — it wins
   on instr count and is within noise of v2 on DRAM-bound shapes.
3. Epilogue I2FP + 3×FMUL + 2×FFMA per group — could fold `*0.5f` into a pre-scaled btab
   (store 2×vals in the table) to drop one FMUL; minor.

**fp8_gemv / fp8_gemv2**
1. `FMUL.FTZ ×256` per value (16/group) is pure overhead — fold 2^8 into the per-128-tile
   bf16 scale (one FMUL per 128 elements instead of 16 per 32).
2. The LOP3×2 + HADD2×2 + FMUL×2 software decode per u32 ignores that **sm_89 has hardware
   `cvt.rn.f16x2.e4m3x2`** (PTX 7.8+, one F2F.F16X2.E4M3X2 per pair). ~5 instrs/pair → ~1.
3. Nothing else: no spills, immediate-offset LDS.128, 83% occupancy. Best codegen of the
   family; missing only a bench (test-fp8.dll prints cosines, no timing).

---

## 3. Occupancy / latency analysis (56 SMs)

* **DRAM-bound regime (248320x4096, the one that matters for decode):** v2 = 436.1 GiB/s
  (93% of 504 GB/s), D/E = ~450 (96%). Issue-rate math: v2 needs 0.37 warp-instr/B
  (6.2/17B) → at 468 GB/s that is ~172 G warp-instr/s vs ~554 G/s machine issue capacity,
  and ~57 G LSU-instr/s vs ~554 G/s LSU capacity. **Neither issue- nor LSU-bound; the
  remaining 4–7% is the per-group LDG.E.U8 scale byte (uncoalesced 1 B traffic interleaved
  with 16 B streams) + tail waves.** Latency hiding is adequate: unroll-4 keeps 4
  LDG.128/warp in flight; E keeps 16 (why E/D edge out v2 despite lower occupancy).
* **Small-row regime is where bandwidth dies and it is a grid/occupancy artifact:**
  grid = rows/8 blocks. rows=1024 → 128 blocks of 256 thr = 0.57 waves at 4 blocks/SM —
  half the SMs never get a block; measured 1024–139 GiB/s. rows=4096 → 512 blocks = 2.3
  waves, last wave 30% idle + 33 KB smem pair staging caps 3 blocks/SM → 164–216 GiB/s.
  rows=8192 → 4.6 waves, fine. This is why "below 400 GB/s": **wave quantization + block
  granularity, not memory latency, not L2 thrash.**
* gemv_e at 128 regs (2 blocks/SM, 33% occupancy) still wins L2-resident shapes because
  4-row MLP per warp (16 outstanding LDG.128) beats occupancy there; at DRAM it ties D.
* x (16 KB at cols 4096) is staged per block into smem — it is L2-hot by construction;
  `cudaAccessPolicyWindow` pinning of x would buy nothing measurable. Weights already use
  `__ldcs` (LDG.E.EF.128 = evict-first) — correct for one-touch streams.

---

## 4. Proposals, ranked by expected GB/s at the shapes that matter

(P1) **Split-K GEMV for rows ≤ 8192** (o_proj 4096, small heads, any skinny matrix).
Fixes the 0.57–2.3-wave regime: 1024-row 139 → ~300+ GiB/s, 4096-row ~216 → ~380+.
Mechanical, no numerics change (fp32 partials, one atomicAdd or 2-pass reduce per row).

(P2) **v3 staging: natural-order x + LDS.128 + LUT-free nibble decode** (kill 2-LSU/element).
Removes 32 LDS.32 + 32 IMAD + ~32 LUT-index instrs per group; expected L2-resident v2
334 → ~420–470 (E territory without E's register cost), DRAM-bound +0–4% but frees issue
slots to push unroll to 8. Sketch below.

(P3) **Promote the q8 pair path to the whole spec-decode forward pass** (exists for
in_proj a+b only): MLP gate+up fused single pass (24576 rows), o_proj pair. At 248320 the
pair path already delivers 432–447 GiB/s of weights for 2 tokens; production currently
uses v2 fp32 single (436 for 1 token). Per-token cost of spec-verify ≈ halves.

(P4) **q8g xq loads → 2×LDG.E.128 per lane** (and xs via one LDS/64B): −7 LSU/group,
expected pair 431.8 → ~450 at 248320, more at L2-resident sizes. One-line-ish change.

(P5) **fp8_gemv: `cvt.rn.f16x2.e4m3x2` + fold ×256 into tile scale**: 90 → ~50 instrs/group.
fp8 has no bench number today; expect ≥ v2-class per-byte efficiency at 2x bytes/weight.
Required for the 27B fp8 decode to not be decode-instruction-bound.

(P6) **E-style 4-row register trim (≤96 regs)** so the best measured kernel also runs at
4 blocks/SM; then E likely becomes the universal kernel (475 L2 / ~450 DRAM).

(P7) **Persistent whole-model GEMV chaining**: saves ~224 launches/token (~2–3 μs each ≈
0.5–0.7 ms of the ~10.7 ms/token streaming budget, ~5%). Only worth it after P1–P4.

Rejected on evidence: cp.async weight double-buffer (weights one-touch; measured 93–96% of
DRAM peak says prefetch is not the gap), warp specialization (same), accessPolicyWindow on
x (x already L2/smem hot), wider-than-uint4 weight loads (uint4 already emitted;
LDG.256 does not exist on Ada).

### Sketch 1 — v3 single-token GEMV (P2)

```cuda
// natural-order x in smem (no transpose!), lane owns one 32-elt group per uint4,
// x fetched 4 floats at a time (LDS.128, immediate offsets), weights decoded without LUT:
// E2M1 val*2 in {0,1,2,3,4,6,8,12} == low 3 bits (b) and sign (s):
//   mag = (n&7); norm = mag>=2; f32bits = norm ? ((117+(mag>>1))<<23 | (mag&1)<<22) : mag<<22;
//   (== decode4 of insignia_layout.cuh with the 0.5 folded into the group scale as *0.5f)
__global__ __launch_bounds__(256) void mxfp4_gemv_v3_kernel(
        const uint32_t* __restrict__ w, const uint8_t* __restrict__ s,
        const float* __restrict__ x, float* __restrict__ y, int rows, int groups) {
    extern __shared__ float sx[];                       // sx[c] == x[c]
    for (int c0 = threadIdx.x*16; c0 < groups*32; c0 += blockDim.x*16)
        reinterpret_cast<float4*>(sx+c0)[0] = __ldg(reinterpret_cast<const float4*>(x+c0));  // +1,2,3 unrolled
    __syncthreads();
    const int warp = threadIdx.x>>5, lane = threadIdx.x&31, row = blockIdx.x*8+warp;
    if (row >= rows) return;
    const uint32_t* rw = w + size_t(row)*groups*4;
    float acc = 0.f;
    #pragma unroll 4
    for (int g0 = lane; g0 < groups; g0 += 32) {
        const uint4 P = __ldcs(reinterpret_cast<const uint4*>(rw + size_t(g0)*4));   // 32 nibbles
        const float4 x0 = reinterpret_cast<const float4*>(sx + g0*32)[0];            // k=0..3
        const float4 x1 = reinterpret_cast<const float4*>(sx + g0*32)[1];            // k=4..7
        // ... x2..x7; then per word j: 8x FMA chain into p0..p3 with decode4(w,k)*0.5f
        // (identical V2_WORD macro body, but lut[] replaced by decode4() and xg by float4 regs)
        float p0=0,p1=0,p2=0,p3=0;
        #define W3(word,kb) { const uint32_t w_=(word); _Pragma("unroll") \
            for(int j=0;j<8;++j){ const float v=decode4(w_,j)*0.5f; \
                const float xv = (kb+j)<8 ? (&x0.x)[(kb+j)&7] : (&x1.x)[(kb+j)&7]; /* or keep 8 float4 */ \
                ((j&3)==0?p0:(j&3)==1?p1:(j&3)==2?p2:p3) = fmaf(v,xv,(j&3)==0?p0:(j&3)==1?p1:(j&3)==2?p2:p3);} }
        // 8 LDS.128 total for x, 32 FFMA, 0 LDS for LUT, 0 IMAD (immediates)
        W3(P.x,0) W3(P.y,8) W3(P.z,16) W3(P.w,24)
        acc = fmaf((p0+p1)+(p2+p3), __int_as_float(uint32_t(s[g0])<<23), acc);
    }
    #pragma unroll
    for (int m=16;m;m>>=1) acc += __shfl_xor_sync(~0u,acc,m);
    if (!lane) y[row]=acc;
}
```
Careful: float4-per-k-octet indexing above wants 8 named float4 (x0..x7) so every xv is a
plain register read — write it as `const float4 xk[8]` loaded with 8 LDS.128. Numerics:
`decode4*0.5f` is exact (powers of two); parity-check against `tools/reference_*.py` as usual.

### Sketch 2 — split-K for skinny rows (P1)

```cuda
// 2 blocks per row-block: block.z halves the group range; lane stride over half-groups.
// Partial sums to y2[2][rows], then a 1-warp reduce kernel (or atomicAdd, fp32, 2 adds/row:
// nondeterministic order — the NumPy reference check tolerates 1e-6 rel; use the 2-pass
// variant if bitwise-stable parity dumps are needed).
__global__ __launch_bounds__(256) void mxfp4_gemv_v2k_kernel(
        const uint32_t* __restrict__ w, const uint8_t* __restrict__ s,
        const float* __restrict__ x, float* __restrict__ y, int rows, int groups) {
    // ... identical staging (sx natural order reuses P2's) ...
    const int row = blockIdx.x*8 + warp;
    const int half = groups>>1;                          // groups is a multiple of 32 (cols%1024==0 enforced)
    const int base = blockIdx.y*half;                    // gridDim.y == 2
    float acc = 0.f;
    for (int g0 = base+lane; g0 < base+half; g0 += 32) { /* same body as v3 */ }
    // warp-reduce, then:
    if (!lane) atomicAdd(y+row, acc);                    // y pre-zeroed by 128B memset (async on stream)
}
// launch: dim3 grid((rows+7)/8, 2);  rows=1024 -> 256 blocks (fills 224 slots: 1.14 waves)
```
For cols 4096 (groups 128) split 2 is the sweet spot (64 groups/half ≥ 2 unroll-4 bodies +
tail). Combine with P2 body. Expected 1024x4096: 139 → ≥300 GiB/s; 4096-row matrices:
216 → ~380.

---

## 5. Sanity check of the AGENTS.md DP4A claim

Claim: "direct FP32 accumulation from packed nibbles currently beats the Q8/DP4A
experiment by a wide margin on decode GEMV." **True for the experiment it names, false as
a statement about DP4A on sm_89.** SASS of `mxfp4_gemv_dp4a_kernel<2>` shows why it lost —
none of the reasons is the DP4A instruction itself:

1. `const int vals[8]` indexed by `q&7` compiles to an ISETP.EQ ladder + 8 predicated
   IMAD.MOVs **per nibble** (~16 instrs to materialize one weight byte);
2. the `lane<4` guard leaves 28/32 lanes idle and makes weight loads 4×LDG.32 (16 B/warp)
   instead of LDG.128 (512 B/warp) — ~4% of achievable load efficiency;
3. 132 ISETP + 18 BRA + BSSY/BSYNC in the loop vs 1 back-edge branch everywhere else.

The newer pair-LUT family (`btab` u64 broadcast table + `__byte_perm` + IDP.4A.S8.S8)
fixes all three and *matches or beats* the fp32 v2 family per token: at 248320x4096,
F-dp4a2x 446.8 vs v2 436.1 GiB/s of weights, and it does so for two activation rows —
~2x the per-token arithmetic efficiency (3.4–3.8 instrs/row-group vs 6.2). IDP.4A.S8.S8
issues on the integer pipe and does 4 MACs/instr; the fp32 path costs 1 FFMA + 2 LDS +
~4 integer ops per MAC. The fp32 path's only structural advantages are exact numerics
(no activation quantization: q8g max_abs ≈ 0.5 on the bench vector) and no per-block
quantization stage. Recommendation: reword AGENTS.md — "FP32-LUT beats the *scalar-lane*
DP4A experiment; the pair-LUT DP4A path is the per-token winner for the speculative path."

---

## 6. Fast wins list (in merge order)

1. P4 (xq vector loads in q8g family) — trivial, ~+4% pair path.
2. P1 split-K — small, unlocks small-row shapes.
3. P2 v3 staging/decode — the main single-token kernel rewrite; bench + NumPy parity gate.
4. P5 fp8 cvt.rn.f16x2.e4m3x2 + scale-fold — before any 27B decode tuning; add an fp8 GEMV
   bench (test-fp8.dll has none).
5. P3 pair-path promotion in qwen35 spec decode — engine routing, kernels exist.
