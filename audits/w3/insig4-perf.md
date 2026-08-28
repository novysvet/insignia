# INSIG4 kernel performance audit — w3 (2026-08-25)

Scope: `src/mxfp4_i4.cu` (mxfp4_gemv_v2_i4, mxfp4_gemv2_q8_i4, ab2, get_row), `src/gemm.cu`
(mxfp4_gemm_mlx_i4 + v21 pattern), vs MXFP4 siblings in `src/mxfp4.cu`. Call-flow evidence from
`src/decode.cu`, `src/prefill.cu`, `src/ops.cu`, `src/qwen_kernels.cu`. Timing evidence on disk:
`build/last-test.log` (GEMV family only: 150-156 GiB/s L2-resident 4096x4096, 54 GiB/s legacy),
`build/i4-ref.log` + `build/multistep-parity.log` (parity only). **No GEMM timing exists on disk
for v21, v2, or any i4 kernel** — every GEMM number below is analytic; bench protocol in §2.4.

Hardware facts used (corrected): **4070 SUPER = 56 SMs** (AD104, 7168 cores, verified in
audits/research.md:7 and internals.md:42 — the task brief's "128 SMs" is wrong; 128 is the CUDA-core
count per SM). L2 = 48 MB, DRAM 504 GB/s, smem 100 KB/SM opt-in (99 KB/block max dynamic),
bf16 dense tensor ≈ 71-83 TFLOPS (fp8 = 2x bf16 ≈ 142 TF, per research.md:7).

---

## 0. Baseline recomputation (from synthesis/internals numbers)

Per-token weight traffic, 9B INSIG4 (0.5 B/elt nibbles + 2 B / 64 elts scales = **0.53125 B/elt**):

- Delta layer (24x): qkv 8192x4096 + z 4096x4096 + a/b 2x32x4096 + out 4096x6144 + gate/up 2x
  12288x4096 + down 4096x12288 = 226.75M elts = **120.5 MB**
- Attn layer (8x): q 8192 + k 1024 + v 1024 + o 4096 (all x4096) + mlp 3x(12288x4096-equiv)
  = 209.7M elts = **111.4 MB**
- lm_head 248320x4096 = **541 MB**; mtp layer + fc ≈ **130 MB**

| path | traffic | time | effective BW |
|---|---|---|---|
| single token (11.8 ms, pre-graph) | 4.33 GB | 11.8 ms | 367 GB/s = 73% peak |
| spec pair step (121 tok/s, p≈0.6 ⇒ 1.6 tok/step ⇒ 13.2 ms/step, 8.26 ms/token) | 3.78 + 0.13 + 2x0.541 ≈ **4.99 GB** | 13.2 ms | 378 GB/s = 75% peak |

The task brief's "8.3 ms/pair" is 1000/121 — per *emitted token*, not per step; per pair-step is
13.2 ms. Implication: the spec decode path is weight-traffic bound at 75% of DRAM peak. If GEMV
work reaches ~90% (454 GB/s), step → ~11 ms ⇒ ~145 tok/s (+20%). That is the **ceiling for all
decode-side kernel work in this audit**; anything beyond needs fewer weight bytes (lm_head dedup,
deeper verify — synthesis backlog items, out of scope but referenced).

Launch census (counted in decode.cu): single token `forward_body` = 554 launches
(24x17 delta + 8x18 attn + 2), of which **313 elementwise**. Spec pair step
(mtp_layer 26 + 32x18 + lm_head/argmax 4 + plumbing 4) ≈ **610 launches**, of which fusable
elementwise ≈ 110-180 (silu 33, sigmoid 9, residual 66, split/concat 10+).

---

## 1. Kill the fp32 round-trip in dequant (packed E2M1→bf16)

### 1.1 Where it happens

`mxfp4_gemm_mlx_i4_kernel` (gemm.cu:330-333) and both v2/v21 e8m0 GEMMs
(gemm.cu:55, :122-144, :250-258) dequant every weight as
`decode4(word,j)` (int ops → float) `* scale` (FMUL) `__float2bfloat16` (cvt) — i.e. per element:
~6-7 int SASS for decode4 (SHF/AND/ISETP/SEL/SHF/IADD/LOP3) + 1 FMUL + 1 F2FP-class cvt ≈ **8-9
SASS per weight**, ~64-70 SASS per u32 word (8 weights). INSIG4 makes it worse per group: fp16
scale → `__half2float` adds another cvt per word-thread.

Answer to the register-LUT question first: a 16-entry bf16 LUT in "two u32 registers" is **not
dynamically indexable** — GPUs have no register-indirect addressing; the compiler would spill it
to local memory. The two real mechanisms are (a) shared-memory LUT (v21 already stages a 256-entry
*pair* LUT, gemm.cu:216-221) or (b) PRMT byte-select arithmetic. Both worked out below.

### 1.2 Design A (recommended): smem 256-entry pair-LUT + one `__hmul2` per byte

The v21 LUT `lut[b] = tbl[b&15] | (tbl[b>>4]<<16)` already produces **two packed bf16 from one
byte** in a single LDS. What must change is only what follows: replace
bf16→f32→FMUL→f32→bf16 with a packed bf16x2 multiply. Exact bf16 patterns (verified):
`{0,0.5,1,1.5,2,3,4,6} = {0x0000,0x3F00,0x3F80,0x3FC0,0x4000,0x4040,0x4080,0x40C0}` — hi byte
∈ {0x00,0x3F,0x3F,0x3F,0x40,0x40,0x40,0x40}, lo byte ∈ {0x00,0x00,0x80,0xC0,0x00,0x40,0x80,0xC0}
(negatives = MSB of the hi byte). All values exact in bf16.

```cuda
// once per block (exists in v21; reuse verbatim):
__shared__ uint32_t lut[256];  // lut[b] = bf16pair(b&15, b>>4)

// per word (8 weights), INSIG4: ONE scale per 64-elt super-group = per KT=64 step = per word x8:
const __nv_bfloat162 sc2 = __float2bfloat162_rn(__half2float(
    *reinterpret_cast<const __half*>(scales + size_t(n0+n)*(groups>>1) + kb)));
const uint32_t word = Braw[buf][n][w];
__nv_bfloat16 out[8];
#pragma unroll
for (int byt = 0; byt < 4; byt++) {
    const uint32_t pair = lut[(word >> (8*byt)) & 0xff];           // 1 LDS per 2 weights
    reinterpret_cast<__nv_bfloat162&>(out[byt*2]) =
        __hmul2(reinterpret_cast<const __nv_bfloat162&>(pair), sc2); // 1 HMUL2.BF16 per 2 weights
}
*reinterpret_cast<uint4*>(&Bs[buf][n][w*8]) = *reinterpret_cast<const uint4*>(out);
```

Per word: 4x(SHF+LOP3 extract/index + LDS + HMUL2) ≈ **16 SASS for 8 weights** vs ~64-70 today —
a ~4x cut of dequant instructions, all register+LDS, no FP32 pipe, no cvt. Scale cost (2 cvt) is
amortized over the whole 64-elt super-group. Optional LDS index trick: `idx = ((b<<4)|b) & 0x0F0F`
style selector pre-building is unnecessary here — direct byte indexing is 1 SHF+1 AND, and random
LDS.32 over a 1 KB table costs ~3-4 way bank conflicts on average (acceptable; measured LDS budget
in §1.5 says we're not near the limit).

**Correctness analysis (must-record):** current path computes `e2m1 * fp16_scale` exactly in fp32
then rounds once to bf16 (RN). Proposed path rounds the *scale* fp16→bf16 (≤2^-9 rel) then rounds
the product (≤2^-9 rel) — two roundings, worst ~0.4% per weight vs ~0.2%. The E2M1 quantization
noise itself is ~3% (+5.9 dB SQNR vs MXFP4 ⇒ ~30 dB), so the added deviation is ~15x below
quantization noise and averages out over 4096-elt dots (~0.006% on the dot). No subnormal/overflow
hazards (bf16 exponent range = fp32; products of fp16 scales with {0..6} cannot leave range).
Acceptance gate: `tools/reference_multistep_i4.py` worst_layer_cos ≥ 0.999 (current 0.9998+) and
`nll_compare.py` A/B unchanged argmax. If parity regresses, fall back to
`__floats2bfloat162_rn(lo*scf, hi*scf)` with the fp32 scale (still kills decode4 + one cvt per
element, ~2x instead of 4x).

### 1.3 Design B (LDS-free): pure PRMT decode — exact mechanism

For completeness, the no-smem path. From one byte `b` (nibbles lo=b&15, hi=b>>4) build the packed
bf16 pair using the two byte-pools LO={00,00,80,C0,00,40,80,C0}, HI={00,3F,3F,3F,40,40,40,40}:

```cuda
const uint32_t sel = ((b << 4) | b) & 0x0F0F;          // SHF + LOP3: lo in sel-nibble0, hi in nibble1
uint32_t P1 = __byte_perm(LO_A, LO_B, sel);            // bytes [LO[lo], x, LO[hi], x]
uint32_t P2 = __byte_perm(HI_A, HI_B, sel);            // bytes [HI[lo], x, HI[hi], x]
uint32_t w2 = __byte_perm(P1, P2, 0x5410);             // [LO[lo],HI[lo],LO[hi],HI[hi]] = bf16pair
w2 |= ((b & 0x08) << 12) | ((b & 0x80) << 24);         // sign_lo->bit15, sign_hi->bit31 (LOP3+SHF)
val2 = __hmul2(w2, sc2);                                // HMUL2.BF16
```

This is 3 PRMT + ~4 aux + 1 HMUL2 ≈ 7-8 SASS per **2** weights (~3.5-4/element): ~2x better than
today but ~2x worse than Design A. Sign cannot fold into the HI pool (16 signed entries needed,
PRMT pool = 8 bytes; one PRMT output = 4 bytes = max 2 bf16, so "2 PRMT per 4 nibbles" is
structurally impossible — minimum is 3 PRMT per byte incl. merge). Arithmetic decode (hi = 0x3F +
(mag>=4), lo = ((m&1)<<7)|((m&5==5)<<6)) costs about the same as decode4 already does — not worth
it. **Use Design A everywhere; keep B in the back pocket for a register-pressure-starved variant.**

### 1.4 Where it plugs in + expected gain

- `mxfp4_gemm_mlx_i4` / the v21-i4 port (§3): B-side dequant, weights-dominated inner loop. At
  T=64 the GEMM is compute-bound (§3.3), and dequant+scale+cvt is roughly 30-40% of issued
  instructions — 4x cut ⇒ est **15-30% kernel time**. This is the single cheapest large GEMM win.
- GEMV (`mxfp4_gemv_v2_i4`): **do NOT port this** — the GEMV accumulates in fp32 and applies the
  scale once per group; there is no round-trip to kill. Its per-element cost is
  SHF+AND+LDS+FMA which is already minimal. (Optional: swap the 16-float LUT for the 256-entry
  pair LUT to halve LDS count — 8→4 LDS/word — but §1.5 shows LDS is not the limiter.)
- `mxfp4_gemv2_q8_i4` / ab2: integer path, no round-trip. Untouched.

### 1.5 LDS budget sanity (why A is safe)

To saturate 504 GB/s each SM must stream ~5.1 B/cycle of weights = ~1.3 u32 words/cycle/SM.
Design A costs 0.5 LDS/word ⇒ ~0.65 LDS/cycle/SM vs ~32 LDS.32/cycle/SM shared throughput: 2%
of the LSU budget. Not a constraint.

---

## 2. Persistent grid-stride GEMV + L2 policy + honest benching

### 2.1 Current cost structure (`mxfp4_gemv_v2_i4`, mxfp4_i4.cu:16-75)

- Grid = `(rows+7)/8` blocks x 256 thr; lm_head pair (248320 rows) = 31040 blocks; up/gate
  (12288 rows) = 1536 blocks. Each block: (a) re-stages **x into smem transposed** (cols*4 B —
  16 KB for cols=4096, 48 KB for down_proj's cols=12288) and (b) `__syncthreads()`, before the
  first weight load. x re-staging for lm_head = 31040 x 16 KB = 509 MB of L2 reads + smem writes
  (x is L2-resident so it's not DRAM, but it is per-block latency + LSU traffic), and the
  transposed store pattern `sr[(q*4+i)*groups]` is a stride-`groups`-float scatter = 4-way bank
  conflict per lane group (groups=128 ⇒ 128%32==0 ⇒ same bank).
- The pair kernel `mxfp4_gemv2_q8_i4` (used 5-6x per layer in the spec path, decode.cu:32) is
  worse: **every one of the 31040 blocks re-quantizes both activation rows** (256 thr x ~100 flops
  over 32 KB) before touching weights — the same 8192 elements quantized 31040 times, and it
  delays each block's weight stream by ~1-2 µs. The fix pattern already exists in mxfp4.cu
  (`quantize_x8` :378-404 + `mxfp4_gemv2_q8g` :408-464, e8m0 twins, currently used only in
  bench_mxfp4_mlx.cu:542) — x quantization is format-independent, so it is reused verbatim.

### 2.2 Design

```cuda
// persistent, grid-stride, stage-once (single-row variant shown; pair = same + xq from quantize_x8)
__global__ __launch_bounds__(256) void mxfp4_gemv_v2i4_pers_kernel(
        const uint32_t* __restrict__ weights, const uint16_t* __restrict__ scales,
        const float* __restrict__ x, float* __restrict__ y, int rows, int groups) {
    extern __shared__ float sx[];  float *lut = sx + groups*32;
    stage_x_transposed(sx, x, groups);          // once per block lifetime
    if (threadIdx.x < 16) lut[threadIdx.x] = decode4(threadIdx.x, 0);
    __syncthreads();                            // the ONLY sync
    for (int row = blockIdx.x*8 + (threadIdx.x>>5); row < rows; row += gridDim.x*8) {
        // inner loop identical to today (uint4 __ldcs, i4_scale, 4x V2I, shfl reduce)
        if ((threadIdx.x&31)==0) y[row] = sum;  // warps independent: no per-row syncs
    }
}
```

Host: `blocks = min((rows+7)/8, sm_count * occ)` with `sm_count` from
`cudaDevAttrMultiProcessorCount` (56) and `occ` from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` (smem 16.4 KB for cols=4096 ⇒ ~6/SM; 49 KB for
cols=12288 ⇒ 2/SM; pick 2 grids or fix at min). Effects:

- x staged ≤ 336 times instead of 31040 (lm_head): −509 MB of L2/smem staging traffic and one
  sync per ~92 blocks.
- Quantization staged exactly once by switching the spec pair path to
  `quantize_x8` + an i4 twin of `mxfp4_gemv2_q8g` (drop-in: change scale fetch
  `__int_as_float(row_s[g]<<23)` → `i4_scale(row_s, g0)`, mxfp4.cu:453 pattern). This also frees
  10.2 KB of the pair kernel's smem (xq/xs live in global), raising blocks/SM.
- Tail: 248320 rows / (336 blocks x 8 warps) = 92 clean waves, no launch quantization; small
  matrices (a/b: rows=32 ⇒ today grid=4 blocks = 4/56 SMs) become one launch covering all rows.

Expected: today's GEMV family runs ~367-378 GB/s effective (§0); the structural overheads above
are what separate it from the 379-435 GiB/s already demonstrated streaming on lm_head shapes
(internals.md:18). Estimate **5-15% on the spec step** (bigger on the small-matrix calls,
smaller on lm_head which is already stream-shaped), ceiling +20% at 454 GB/s.

### 2.3 L2 accessPolicyWindow — set it, but expect little

```cpp
cudaStreamAttrValue a{}; a.accessPolicyWindow = { const_cast<float*>(x), (size_t)cols*4,
    1.0f, cudaAccessPropertyPersisting, cudaAccessPropertyStreaming };
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &a);
```

x is only 16-48 KB and is touched by the first wave anyway, so the window mostly formalizes what
L2 already does. Two constraints: (1) with the persistent kernel it is nearly moot (x staged once,
then never re-read); (2) stream attributes are **not captured** by CUDA graphs — set once before
`capture_spec()`; the x buffers (pf_x/pf_n/pf_gate...) are stable workspace pointers so one window
over the workspace region is replay-safe. Verdict: implement with the persistent kernel (2 lines),
measure, keep only if it shows ≥1%.

### 2.4 `__ldcs` and the L2-resident bench lie — protocol

In production every matrix streams from DRAM once per step; 4.99 GB of traffic between reuses
flushes 48 MB L2 many times over, so `__ldcs` (evict-first) is *correct* for all weight reads and
swapping it for `__ldca` changes nothing end-to-end. The ≤12288-row bench shapes (≤25 MB weights)
sit entirely in 48 MB L2 across reps — those ~150 GiB/s numbers are fiction. Required protocol
for every future GEMV/GEMM bench (add to bench_mxfp4_mlx.cu / bench_gemm.cu):

```cuda
__global__ void l2_sweep(const float4* p, size_t n, float* sink) {   // 256 MB buffer
    float4 a{0,0,0,0};
    for (size_t i = threadIdx.x + blockIdx.x*blockDim.x; i < n; i += gridDim.x*blockDim.x)
        a = __ldcs(p+i), a.x += a.w;            // pure evict-first read sweep
    if (a.x == 12345.f) *sink = a.x;            // defeat DCE
}
// between timed reps: cudaCtxResetPersistingL2Cache(stream); l2_sweep<<<448,256>>>(...); sync;
```

Report cold (swept) and hot numbers; only cold is citable for shapes < 2x L2. One nuance worth
benching: in the spec step lm_head is read twice (draft sweep + verify sweep, 541 MB apart) — L2
cannot help; the fix is traffic elimination (synthesis backlog: fuse MTP lm_head into the verify
lm_head = −541 MB/step ≈ −1.2 ms ≈ +10% tok/s — still the single biggest lever anywhere, though
out of this audit's 5 items).

---

## 3. Prefill GEMM v2.1 port check — NOT done; full spec

**Verified: `mxfp4_gemm_mlx_i4` (gemm.cu:302-356) is still the unpipelined v1 clone** — KT=32,
single-buffered At/Bt, plain global loads, `__syncthreads()`-separated stages, dequant on the
critical path, fp32 A staged by only 64 of 256 threads. The synthesis line "port v2.1 (done this
session)" refers to the e8m0 `mxfp4_gemm_v21` (gemm.cu:210-295) existing; the i4 twin was never
written. decode.cu:35 routes ALL INSIG4 batch GEMMs (T>2 prefill, and the 24-layer a/b loops'
alternative) through this v1 clone.

### 3.1 Port spec: `mxfp4_gemm_v21_i4` (diff from v21, ~1 day)

KT=64 aligns **exactly** with the 64-elt super-group — cleaner than e8m0 (one scale per step, not
two):

1. Signature: `scales` becomes `const uint16_t*`; index
   `scales[(n0+n)*(groups>>1) + kb]` (KT=64 ⇒ scale idx = kb exactly). Rows per step: each of the
   32 row-threads loads one u16 (`__ldg`), converts once to `__nv_bfloat162` (§1.2) — fold into
   dequant, not a separate pass.
2. **B path unchanged**: 32 nibble-bytes/row/step = 2x 16 B `cp.async.cg` per row into
   `Braw[2][32][8]` (v21 lines 237-242 verbatim).
3. **A path unchanged**: bf16 tiles via cp.async (v21:229-236) — requires A in bf16, so keep the
   `f32_to_bf16` staging pass initially, then adopt §4.3 (rmsnorm emits bf16) to delete it.
4. **Dequant**: replace v21:250-258 with §1.2 (pair-LUT + `__hmul2`). Also relax `rows%64==0`
   (gemm.cu:354) to `rows%32==0` like v21:293 — this unlocks the a/b fix below.
5. **3-stage option**: v21's dequant sits between two `__syncthreads()` (gemm.cu:274-278) on the
   critical path. Go to `Bs[3]/Braw[3]` (keep `As[2]`): prefetch two groups ahead, dequant(kb+1)
   overlaps mma(kb) ⇒ dequant leaves the critical path entirely once §1.2 shrinks it. smem:
   v21 today 30.7 KB (audit); +1x Bs (9.2 KB) + Braw (1 KB) ≈ 41 KB ⇒ 2 blocks/SM at 99 KB
   (was 3). Measure 2-stage vs 3-stage; keep whichever wins.
6. **Occupancy correction to internals.md:42**: at 30.7 KB smem the v21/v2 GEMMs fit **3
   blocks/SM** (⇒ 168 concurrent), so the 4096-row down_proj grid of 128 blocks is a *single
   wave* — the "128/56 = 2.29 waves, 30% tail waste" claim wrongly assumed 1 block/SM. Tail waste
   only matters if register pressure limits occupancy to 1 (check with `-Xptxas -v` when builds
   are allowed). If it does, prefer NT=64 or 2-tiles-per-block over persistence here.
7. Grid/persistent: for prefill's long streams, `rows>>5` blocks is fine (17408 rows = 544
   blocks = 3.2 waves of 168 — tail ~5%); persistent variant optional.

### 3.2 The a/b per-token GEMV loop — bigger than the GEMM itself

decode.cu:72-73: for T>2 INSIG4 prefill, in_proj_a and in_proj_b are computed by **T separate
`mxfp4_gemv_v2_i4` launches each** (rows=32 ⇒ grid = 4 blocks = 4/56 SMs, ~3-5 µs latency-bound
each). At T=64: 128 launches/layer x 24 layers = **3072 tiny launches ≈ 9-12 ms per 64-token
chunk** — comparable to the entire 4.4 GB weight stream (~11 ms). Fix with the port (needs
`rows%32`): two `mxfp4_gemm_v21_i4` calls (a, b) with rows=32 ⇒ 1 block each, one launch each,
~70 µs total. **This one routing change is worth more than the GEMM pipeline itself.**

### 3.3 Ceiling and current-state honesty

T=64, 17408x5120 (27B down_proj; 9B's is 4096x12288, same math): weights 45.95 MB (44.56 nibbles
+ 1.39 scales) ⇒ **91 µs at 504 GB/s**; FLOPs 11.44 GF ⇒ **139-161 µs at 71-83 TF bf16** ⇒ the
T=64 GEMM is compute-bound by ~1.6x. Today's v1-style kernel with syncs + fp32 round-trip +
unhidden global latency plausibly runs 250-350 µs (no on-disk measurement exists — i4-ref.log is
parity-only). Port + §1.2 targets ~150-180 µs (~2x). Note the compute floor is the *wmma fp32-acc*
rate; going beyond needs larger T (T=128 doubles intensity to ~498 FLOP/B but also doubles the
compute — still compute-bound) or mma with fp16-acc fragments (2x rate, parity risk) — future work.

---

## 4. Micro-opts: fused epilogues (plug points + counts)

All fusions below keep bitwise-compatible outputs at fp32-tolerance (same argument as §1.2);
gate each on `reference_multistep_i4.py` worst_layer_cos ≥ 0.999 and an A/B nll_compare.

### 4.1 `silu_mul` → down_proj A-stage (kills 33 launches/step)

- Pair path (`mxfp4_gemv2_q8_i4`/its q8g twin, decode.cu:88): the staging loop (mxfp4_i4.cu:84-104)
  reads `x` rows and quantizes. Add a second pointer `u` (up), compute `v = silu(g)*u` before the
  absmax/quant — silu cost is 1 ex2+rcp per element inside a bandwidth-trivial loop. Signature:
  `..._silu(w, s, g, u, y, ...)`. Also delete the separate silu_mul launch at decode.cu:87.
- T>2 GEMM path: cp.async cannot transform, so instead make silu_mul write **bf16** directly into
  the A buffer (`silu_mul_bf16`): one kernel replaces silu_mul + f32_to_bf16 AND halves A-read
  traffic (12288x64: 3 MB fp32 → 1.5 MB bf16 read by the GEMM).
- Expected: −33 launches ≈ −45-70 µs/step (~+0.5%), prefill −32x(7 µs + pass) ≈ −0.3 ms/chunk.

### 4.2 `sigmoid_mul` → o_proj A-stage (kills 9 launches/step)

Identical shape: pair-kernel staging reads `core` + `gate` and quantizes `core*sigmoid(gate)`
(decode.cu:62/63 pair; attention.cu/o_proj). Decode path: variant of `mxfp4_gemv_v2_i4` staging
that reads two arrays. −9 launches ≈ −15-25 µs/step.

### 4.3 residual_add → following rmsnorm (`fused_add_rmsnorm_bf16`) — the best of the four

`residual_add(x,down)` (ops.cu:8) then `rmsnorm_bf16(x,...)` (qwen_kernels.cu:5) runs 2x per
layer + 2x mtp = **66 launches/step**. Fuse: rms reads `a=x, b=down`, computes `t=a[i]+b[i]`,
writes `x[i]=t` (residual stream must persist — pf_x is read at chunk end, decode.cu:102) and
`y[i]=t*inv*w` in one pass. Traffic per instance 96 KB → 64 KB (−33%) and −1 launch. Optionally
emit `y` as bf16 (§3.1 item 3) in the same kernel — this then deletes the `f32_to_bf16` pass
(gemm.cu:190-196) for all v21-family GEMMs: −66 residual launches, −32 conversion launches,
−1 MB/chunk traffic. ≈ −120-200 µs/step (~+1.5%), prefill −0.5-1 ms/chunk.

### 4.4 Summary of launch arithmetic per spec step

| fusion | launches removed / step | est time |
|---|---|---|
| silu→down A-stage | 33 | 45-70 µs |
| sigmoid→o A-stage | 9 | 15-25 µs |
| residual→rmsnorm (+bf16 out) | ~98 (66 res + 32 cvt) | 120-200 µs |
| total | ~140 of 610 | ~0.2-0.3 ms ≈ +1.5-2.5% tok/s |

Small but nearly free; do §4.3 first since the v21-i4 port wants bf16 A anyway.

---

## 5. Ranking by (expected gain)/(effort) against the §0 baseline

Baseline: 11.8 ms/token single pre-graph; spec 121 tok/s = 8.26 ms/token, 13.2 ms/step, 4.99
GB/step at 378 GB/s (75% of peak; kernel-work ceiling ≈ 145 tok/s at 90% DRAM).

| # | item | effort | expected gain | gain/effort |
|---|---|---|---|---|
| 1 | **v21→i4 cp.async port + rows%32 + route a/b loop to it (§3)** | ~1-1.5 d | prefill chunk −30-50% (of which the a/b routing is most); unblocks §4.3 | **highest** |
| 2 | **packed `__hmul2` dequant, Design A (§1)** | ~0.5 d | GEMM −15-30% (compute-bound side); zero decode effect | very high |
| 3 | **Persistent GEMV + quantize_x8/q8g-i4 pair path (§2.2)** | ~1 d | spec step −5-15% (75%→80-86% DRAM eff.) | high |
| 4 | **fused_add_rmsnorm_bf16 (§4.3)** | ~0.25 d | +1.5-2.5% decode, −0.5-1 ms/chunk, enables #1's A-side | high (do with #1) |
| 5 | silu→down, sigmoid→o A-stage (§4.1-4.2) | ~0.5 d | +0.5-1% decode | medium |
| 6 | L2-flush bench protocol + `__ldcs` audit freeze (§2.4) | ~2 h | 0 runtime; prevents false optimizations (the 150 GiB/s fiction) | medium (do FIRST) |
| 7 | accessPolicyWindow on x (§2.3) | ~1 h | ~0-1% | low |
| 8 | 3-stage Bs pipeline (§3.1.5) | ~0.5 d on top of #1 | +5-10% GEMM | low-medium |

Execution order: **6 → 1(+4) → 2 → 3 → 5 → 8**. Cross-reference: none of this beats the
out-of-scope lm_head dedup (−541 MB/step ≈ +10% tok/s, synthesis backlog #3) — schedule it
alongside.

Prereqs/gates for everything: the F16-scales-read-as-bf16 fix (already applied this session —
confirm before benching), `nll.bat` linking `src\mxfp4_i4.cu` (currently omitted,
synthesis bug #1), and each change lands with cold-L2 bench numbers + multistep parity
worst_layer_cos ≥ 0.999.
