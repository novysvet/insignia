# FP8 kernel audit (w4) — post-rewrite correctness + performance review

Date 2026-08-25. Scope: `src/fp8.cu`, `include/insignia_fp8.cuh`, `src/test_fp8.cu`
(Phase 0/A uncommitted state, git-untracked). Cross-checked against
`audits/w3/fp8-kernels.md` (F1–F8), `audits/w3/trtllm-fp8-deep.md`,
`audits/w3/MASTER-PLAN.md`, `include/insignia_streaming.hpp`, `src/mxfp4.cu`,
`src/gemm.cu`, the TensorRT-LLM clone, and the live checkpoint headers at
`E:\coding\Insignia\Qwen3.8-27B-FP8`. Method: line-by-line read, checkpoint header
enumeration (Python), occupancy/smem arithmetic, and one run of the already-built
test DLL (`python tools/rundll.py build/test-fp8.dll`). No builds, no file changes
except this report.

Test run (VERIFIED, this session):

```
fp8_gemv cos=1.00000000
fp8_gemv2 cos=1.00000000
gemm diag: nan=0 nonzero=30720 yT[0]=0.128349 yT[1]=0.294637 yT[rows]=0.139764
fp8_gemm cos=0.99999871
fp8_gemm T=33 (tile-boundary rows) cos=0.99999873
fp8_gemm T=65 throws: yes
```

---

## 1. Status of the w3 findings F1–F8 in current code

| # | w3 finding | status now | evidence |
|---|------------|-----------|----------|
| F1 | `fp8_gemm` ignored T, stored all 64 rows (OOB) | **FIXED (tile-granular)** | fp8.cu:184 `if (wm * 16 < T) wmma::store_matrix_sync(...)`; fp8.cu:188 throws on T>64 (runtime-verified "T=65 throws: yes"); header insignia_fp8.cuh:25 now documents y as `[64,rows] padded`; comment fp8.cu:182-183 states the contract. Residual: guard is 16-row tile granularity — rows T..ceil(T,16) are written (as exact zeros given zero-padded x16), rows ceil(T,16)..63 stay stale. Safe under the documented padded contract, **OOB if a caller passes y=[T,rows]** — the Phase C dispatch must allocate 64-row y (see N6). |
| F2 | `e4m3_host` reference decoder 64x wrong on normals | **FIXED** | test_fp8.cu:15 `ldexpf(1.f + m / 8.f, int(e) - 7)` — correct OCP bias-7. |
| F3 | test read x[2*cols..3*cols) OOB | **FIXED** | test_fp8.cu:76 `std::vector<float> x(64 * cols)` — all 64 rows allocated and filled. |
| F4 | `bf16_gemv_rows` declared, never defined | **FIXED (removed)** | insignia_fp8.cuh:23-26 declares only `fp8_gemv/fp8_gemv2/fp8_gemm/bf16_get_row`. |
| F5 | gemv2 smem ceiling, no opt-in, no launch checks | **PARTLY FIXED** | fp8.cu:98-101: explicit `> 99 KB` throw + lazy `cudaFuncSetAttribute(..., 99*1024)` static — done for gemv2 ONLY. `fp8_gemv` (fp8.cu:52-55) got **neither** — see N1 (new, blocking). No `cudaGetLastError` after any launch in the file (all three launchers) — leftover F5 half. |
| F6 | quantizer not RNE; benign (wref built from emitted code) | unchanged, still benign | test_fp8.cu:18-32; no "not for production" comment added (cosmetic). |
| F7 | `e4m1()` misnamed (decodes e4m3), dead | **NOT FIXED** | insignia_fp8.cuh:17-20, still present, still unused, still misnamed. |
| F8 | comment overstates "dequant exact" (bf16 product rounds ≤2^-8) | **NOT FIXED** | fp8.cu:107 comment + fp8.cu:147-150 `__float2bfloat16(a.x * sc)`. Numeric effect visible in the measured gemm cos 0.9999987 vs gemv 1.0. Fix = promote-style scale application (R4 below), which also speeds the GEMM. |

---

## 2. New findings (this audit)

### N1 (CRITICAL, blocking for 27B): `fp8_gemv` cannot launch on down_proj — 68 KB dynamic smem, no 99 KB opt-in, no size check — VERIFIED by static analysis + CUDA limit

`fp8_gemv` launches with `size_t(cols) * 4` dynamic smem (fp8.cu:54) and never calls
`cudaFuncSetAttribute(fp8_gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
(compare fp8.cu:100, which does exactly this for gemv2; also mxfp4.cu:145-147 does it
for `mxfp4_gemv_v2`). The default per-block dynamic-smem limit without opt-in is
48 KB. The four 27B GEMV geometries:

| matrix | rows×cols | gemv smem | launchable? | blocks/SM (smem) | weights | BW floor @504 |
|---|---|---|---|---|---|---|
| in_proj_qkvz | 16384×5120 | 20 KB | yes | 5 | 80.0 MB | 159 µs |
| out_proj | 5120×6144 | 24 KB | yes | 4 | 30.0 MB | 60 µs |
| gate / up | 17408×5120 | 20 KB | yes | 5 | 85.0 MB | 169 µs |
| **down_proj** | **5120×17408** | **68 KB** | **NO — cudaErrorInvalidValue** | (1 if opted in) | 85.0 MB | 169 µs |

Every MLP layer of every tier calls down_proj through `fp8_gemv` once the Phase C
dispatch ("fp8 → fp8_gemv/gemv2/gemm", MASTER-PLAN Phase C.3) lands. The launch
fails; no `cudaGetLastError` exists in the launcher, so the failure surfaces only at
the next sync as a confusing error with y left stale. Runtime reproduction was not
possible under this audit's no-build rule (test uses cols=5120 only); the 48 KB
default dynamic-smem limit is a documented CUDA constant, and the code path is
direct: **verified by static analysis, runtime confirmation blocked by rules.**

Even with a bare opt-in (68 KB ≤ 99 KB), occupancy collapses to 1 block/SM
(8 warps resident) — marginal for hiding DRAM latency; expect ~350-420 GB/s, i.e.
down_proj ~200-240 µs vs the 169 µs floor (+3-8% per layer end-to-end). Two better
fixes, ranked:
1. **Non-staged variant for cols>12288** (~30 LOC): drop the smem x mirror, read x
   via `__ldg` float4 straight from L2. x is 68 KB — permanently L2-resident on a
   48 MB L2; 640 blocks × 68 KB = 43 MB of L2 reads ≈ 11-14 µs, hidden under the
   169 µs weight stream. Occupancy stays 5 blocks/SM (no dynamic smem at all).
2. Split-K over cols halves (34 KB staging, ≤48 KB, 2 blocks/SM) + tiny add pass —
   more code, no advantage over (1) here.

Also add the missing symmetric guard the gemv2 launcher already has
(`if (smem > 99*1024) throw`) so future geometries fail loudly, and the same lazy
`cudaFuncSetAttribute` for the sub-99 KB opt-in range.

### N2 (CRITICAL for the streaming tier, refines MASTER-PLAN §Phase D.2 / risk 7): ring-slot 16B misalignment affects exactly 8 of 66 shards — VERIFIED against the live checkpoint

MASTER-PLAN says "`data_start ≡ 8 (mod 16)` **in every shard**". Actual enumeration
of all 66 safetensors headers (`data_start = 8 + hlen`, tools/index27.py:155):

- `data_start % 16 == 8`: **layers-0,1,2,4,5,6,8,9** (8 shards — all DeltaNet layers;
  the full-attn shards 3 and 7, mtp, outside, and the other 56 layer shards are ≡0).
- Consequence: in those 8 shards, **all F8 weight and scale tensors have file bases
  ≡ 8 mod 16** (96 of 814 F8+scale tensors; first-F8 `begin ≡ 0 mod 16` in every
  shard, so the whole shard inherits the header's phase).
- `PinnedRing` slots are 4096-aligned and `LayerFeeder::map` returns
  `slot_base + (off − align_down_4096(off))` (include/insignia_streaming.hpp:63-74,
  220) — mod-16 phase is preserved exactly, so streamed F8 bases for those 8 layers
  land **≡8 mod 16**.
- `fp8_gemv/gemv2` load weights as `uint4` (fp8.cu:32, 76) and `fp8_gemm` stages B
  with 16 B `cp.async` (fp8.cu:126, 132) — all require the weights base ≡0 mod 16.
  A ≡8 base is a guaranteed misaligned-address fault (loud) — on layers **0, 1, 2**,
  i.e. the R4 parity layer itself, plus 4,5,6,8,9.

Required (already planned, now with exact numbers): the Phase D 8-byte pad at the
BF16→F8 boundary in the plan builder, **plus** cheap acquire-time asserts in the
fp8 launchers — `if ((uintptr_t)weights & 15) throw` (weights, and x16 for the
GEMM's cp.async path) — so a plan-builder mistake dies at the call site instead of
as a device-side misaligned load. The engine's device-resident VRAM copies and
`cudaMalloc` buffers are always 256B-aligned; only the streamed/UVA path can
violate this.

### N3 (HIGH, test validity): still cosine-only scoring and one geometry — the R1 gate cannot be met as-is — VERIFIED

The Phase 0/R1 requirement (MASTER-PLAN Phase C gates: "cos > 0.999999 **AND
max-rel < 1e-4**", over "all 7 matrix geometries × T∈{3,33,64}") is unimplemented:
- No max-rel-err metric anywhere in test_fp8.cu (prints at :91, :99, :129, :143 are
  cosine only). Cosine remains invariant to a *uniform* scale bug in the kernel
  (proved in w3 F2) — e.g. folding 2^8 wrongly into `e4m3x2` or a
  `weight_scale_inv` ÷/× confusion would still print cos=1.0.
- Only rows=10240, cols=5120 is exercised. cols=17408 would have caught N1
  immediately; cols=6144 (out_proj, gemv2 exactly at the 48 KB boundary: 49152 B)
  is untested; rows%32 residue, T=64 (the wm=3 store tile active), and gemv2's
  `>99 KB` throw path are untested.
- The T=33 check covers one row (t=32) × 256 columns only — adequate for the
  tile-boundary question, not a parity statement.

Cheap completion: loop the existing generator over the geometry table, print
max-rel next to every cos (reference is already absolute-correct post-F2), add
expect-throw tests for gemv cols=17408-after-fix contract, gemv2 cols=17408 (throws
today by design, fp8.cu:99), gemm T=65 (already present).

### N4 (MEDIUM, perf): GEMM burns full T=64 MMA work at small T — VERIFIED

The w3 "fix item 3" (skip MMA for warp tiles with `wm*16 >= ceil16(T)`) was not
implemented: fp8.cu:172-179 runs `load_matrix_sync`+`mma_sync` for all 8 warps
regardless of T. Because gemv2 deliberately refuses cols=17408 (fp8.cu:99 message:
"use fp8_gemm for the pair path"), the down_proj pair-verify **must** route through
`fp8_gemm` at T=2 — where the kernel does 32x the useful MMA work. Numbers: gate/up
T=64-equivalent MMA ≈ 11.4 GFLOP; at a realistic 35-70 TF that is 160-320 µs vs a
177 µs weight-stream floor — i.e. compute binds at T=2 purely from wasted tiles.
The fix (guard only the load/mma, keep both `__syncthreads()` and cp.async uniform,
as in w3's design) cuts MMA 4x and returns the call to its BW bound: ~1.5-2x on
every small-T GEMM call, ~0 risk.

### N5 (LOW): dead/misnamed `e4m1()` and stale comment — carried from F7/F8

insignia_fp8.cuh:17-20 (decodes e4m3, named e4m1, unused) — delete or rename
`e4m3_lane`. fp8.cu:107 "dequant e4m3->bf16 (exact)" — the conversion is exact, the
`*sc` product is not (≤2^-8 rel; the measured gemm cos 0.9999987 is this rounding).

### N6 (LOW, forward-looking contract risks at dispatch time) — VERIFIED as unwired

`grep` finds zero references to `fp8_gemv/fp8_gemm/fp8_gemv2` outside
fp8.cu/test_fp8.cu — the dispatch (Phase C.3) does not exist yet, so N1 is a
landmine, not a live bug. When wiring: (a) y for fp8_gemm must be a full 64-row
buffer (header says so; the guard only trims at 16-row granularity); (b) x16 must
be zero-padded rows T..63 — the cp.async A-stage (fp8.cu:126) reads all 64 rows
unconditionally, so an unpadded [T,cols] x16 is an OOB read of up to 61×cols×2 B;
(c) suballocated workspace slices handed as x/x16 must keep 16B alignment
(cudaMalloc'd bases are fine; `pf_bf16`-style arena offsets are the thing to check).

---

## 3. Verified correct (proof or runtime evidence)

1. **Block-scale semantics (mission item 1) — CORRECT.** GEMV/GEMV2:
   `row_s = scales + (row>>7)*kblocks` with `kblocks = cols>>7` (fp8.cu:28, 72) and
   per-round `sc = scales[(row>>7)*kblocks + (c0>>7)]` applied as
   `acc = fmaf(part, sc, acc)` (fp8.cu:33+46, 77+89-90). A lane's 16-weight chunk
   starts at a multiple of 16 and never straddles a 128-col block boundary
   (exhaustively checked in w3; unchanged), so `Σ_b sc_b · dot_b ≡ Σ_c (fp8·scale)·x`
   exactly — **W_dequant = fp8 × scale, i.e. `weight_scale_inv` is multiplied**,
   matching the checkpoint layout `[ceil(r/128), ceil(c/128)]` row-major that
   tools/index27.py:195-200 enforces at index build (shape + BF16 dtype verified
   against all 407 links). GEMM: `((n0+n)>>7)*kblocks + (kb>>1)` (fp8.cu:139) —
   n0 is 32-aligned so a tile's 32 rows sit in one 128-row band, and kb>>1 shares
   one scale across the two KT=64 steps of a 128-col block. Correct.
2. **e4m3 decode exactness (item 2) — CORRECT, runtime-confirmed.** The bit trick
   (insignia_fp8.cuh:12-16) was exhaustively proven exact for all 254 finite codes
   incl. subnormals and both signs in w3; the code is unchanged; with the now-fixed
   host decoder the test prints gemv cos=1.00000000 / gemv2 cos=1.00000000, which
   is now *meaningful* (the reference is absolute-correct). Only 0x7F/0xFF (the
   e4m3 NaN encodings) decode to ±480·scale instead of NaN — documented
   ("every finite code"), acceptable for weight-only use. Subnormals: the fp16
   subnormal path (m·2^-17) ×256 = m·2^-9 is exact — fp16 subnormal steps divide
   e4m3 steps. No intermediate overflows (max 480).
3. **T-guard (item 3) — SAFE under the documented contract.** fp8.cu:184 stores a
   warp's 16×16 tile iff `wm*16 < T`; with y `[64,rows]` padded and rows%32==0
   (fp8.cu:187) there is no OOB write at any T∈[1,64]; T>64 throws (fp8.cu:188,
   runtime-verified). Rows in (T, ceil(T,16)) are written as exact zeros (x16
   zero-padded); rows ≥ ceil(T,16) are stale — documented "never read"
   (fp8.cu:182-183). The store `y + wm*16*rows + n0 + wn*16` with ldm=rows never
   crosses the column extent (rows%32==0 ⇒ n0+wn*16+15 < rows).
4. **GEMM pipeline — no race.** cp.async commit/wait trace (fp8.cu:156-181): at
   iteration kb, group G_kb is retired by `wait_group 1` (or `wait_group 0` for the
   last two iterations), results published by the `__syncthreads()` at :169 before
   `dequant` reads `Braw[buf]`, and the end-of-loop sync at :180 retires buf before
   the next-but-one prefetch refills it. This is the same corrected pattern as the
   Phase-0-fixed src/gemm.cu:275 (`if (kb+2<ksteps) wait_group 1 else wait_group 0`)
   — the fix landed in gemm.cu and was carried into fp8.cu. Double buffering is
   conflict-free; the early `wait_group 0` at kb=ksteps−2 is conservative (a few
   hundred ns of stall, not a bug).
5. **GEMV staging loop** covers [0,cols) exactly for any cols%16==0 (re-derived:
   grid-stride over 16-float chunks; at cols=5120/256 threads the second iteration
   engages tid<64 — complete, disjoint). gemv2 staging spans 2·cols and `sx+cols+c0`
   (fp8.cu:80) is in-bounds. Warp reductions (fp8.cu:49, 93) correct for both
   accumulators; `if (row >= rows) return` sits after the last `__syncthreads()`
   each thread participates in — no divergent-sync hazard.
6. **gemv2 48 KB boundary**: cols=6144 → 49152 B dynamic = exactly the no-opt-in
   ceiling — launches fine, and the lazy 99 KB attribute (fp8.cu:100) is set on
   first call regardless (function-local static, thread-safe, precedes the first
   launch). cols∈(6144,12672] covered by the opt-in; cols>12672 throws with a
   designed fallback message. The gemv2 launcher is the model the gemv launcher
   should copy (N1).
7. **bf16_get_row** (fp8.cu:193-200): grid-stride bf16→f32 row gather, device-side
   row id; trivially correct; 20-68 blocks at decode widths is fine for a 10-70 KB
   copy.
8. **Occupancy (item 5), computed**: gemv cols=5120 → 20 KB dyn smem, 5 blocks/SM
   (1280 thr, 83% of 1536); cols=6144 → 4; gemm 31,744 B static smem → 3 blocks/SM
   (24 warps) — all healthy; `__launch_bounds__(256)` on all three hot kernels;
   register pressure has no red flags (≤64 regs implied by successful 5-block
   residency in the mxfp4 analog). down_proj is the one pathological case (N1).

---

## 4. Performance headroom (item 6)

**Is GEMV at 504 GB/s-class? Nobody knows — no measurement exists.** VERIFIED: the
test has no timing, no bench-fp8 target exists, and no other harness touches the
fp8 kernels. The w3 analytic estimate stands as a hypothesis: ~15% issue utilization
(56 SM × 4 schedulers × ~2.48 GHz vs ~81 G warp-instr/s at 460 GB/s) → expected
430-470 GB/s (85-93% of peak). MASTER-PLAN Phase C gates kernels on a measured
cold-L2 bench (insig4-perf §2.4 protocol) — fp8 must get the same gate. Expected
per-geometry floors at 460 GB/s: qkvz 174 µs, out_proj 65 µs, gate/up 185 µs each,
down_proj 185 µs → ~830 µs F8 per linear layer (383 MB), consistent with the
placement tables' 0.78 ms/layer @504.

**vs the MXFP4 GEMV (src/mxfp4.cu):** the fp8 GEMV is structurally the v2 MXFP4
kernel minus the LUT (same warp-per-row, uint4 per lane, `__ldcs` streams, smem x
staging; compare mxfp4.cu:90-141 with fp8.cu:14-51). Differences that matter: fp8
needs no e2m1 LUT lookup per nibble and no transposed x layout (decode is 3 logic
ops + one half22float2 + FMULs), so instruction mix is *better* than the 150 GB/s
MXFP4 number (which is small-matrix latency-bound at 4096 cols, not decode-shape
comparable). The MXFP4 launcher's pattern of *fallback + attribute opt-in*
(mxfp4.cu:144-150: `cols&1023` → mlx variant, else 99 KB opt-in) is exactly what
`fp8_gemv` is missing.

**dp4a / marlin-style w8a16 vs the fp32-accum GEMV:** at T=1-5 every path is
bandwidth-bound (intensity 2T ≤ 10 FLOP/B vs a machine balance of ~141) — math dtype
is irrelevant; the f32-accum GEMV with in-register decode is already the
SOTA-shaped design (trtllm-fp8-deep §7.1: *no* blockwise-FP8 GEMV exists in TRT,
vLLM, or CUTLASS). dp4a additionally requires an int8 view of e4m3 that does not
exist without a repack; AGENTS.md already records the DP4A experiment losing to
direct FP32 accumulation. **Do not build a dp4a/marlin GEMV.** The one Marlin idea
worth stealing for any future bf16-accumulating variant is the 3-logic-op
shift-dequant (`(q&0x80008000)|((q&0x7F007F00)>>4)`, vLLM marlin dequant.h:357-373,
2^120 folded into scales) — strictly fewer ops than today's
shift-to-fp16+convert+multiply, but at 15% issue it buys ~0-5%.

**GEMM (T=64) is compute-bound; the TRT-LLM divergence is the scale placement:**
- Insignia (fp8.cu:139-152): multiplies the bf16 scale into every dequantized
  weight (4 extra FMULs + a bf16 rounding per 8 weights; F8).
- TRT-LLM sm89 (VERIFIED in clone: `sm89_utils.cuh:54`
  `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`; promote at
  `sm89_fp8_gemm_1d1d.cuh:168,390`): runs raw e4m3×e4m3 MMA per 128-K slab into a
  temp fp32 accumulator, then `accum += temp * (SFA[row]·SFB)` — 3.1% FFMA tax, no
  rounding, 2x the bf16 MMA rate, 3-stage cp.async, 61,952 B smem, 1 CTA/SM.
- The halfway step that fits the current shell without new MMA plumbing: keep bf16
  wmma, dequant **without** `*sc` (exact e4m3→bf16), accumulate per 64-K step into a
  temp fragment, then fold `acc.x[i] += temp.x[i]·sc` once per step (uniform scale —
  fragment layout is irrelevant to a uniform affine op on `.x[]`). Removes F8's
  rounding (gemm cos → ~1e-7) and ~10-15% of dequant cost, and is precisely the
  shape of the future e4m3-MMA promote. Worth ~5-10% on T=64 GEMMs today.
- Native e4m3 MMA (W8A8 with per-token 1x128 activation quant, TRT §1.6/§4.1) only
  pays at T≥~141 (machine balance) — i.e. never in the current plan's T≤64 prefill;
  correctly deferred to Phase G insurance per trtllm-fp8-deep §3/§8.

**Small open items from w3 perf list (still open, low priority):** `__ldcs`
everywhere on weights (fp8.cu:32, 76) is right for ≥48 MB mats but evicts the ≤32 MB
mats (out_proj 30 MB) that would otherwise stay L2-resident across MTP draft
re-reads — worth ~1-2% via a `rows*cols > 24MB ? __ldcs : __ldg` dispatch; folding
2^8 into the scale bits (`sb += 0x400000`) saves 2 FMULs per e4m3x2 (~0-3%).

---

## 5. Ranked recommendations

1. **Fix `fp8_gemv` for cols=17408 (N1)** — blocking; every layer's down_proj.
   Minimal: lazy `cudaFuncSetAttribute` + `cols*4 > 99KB` throw (copy fp8.cu:98-101).
   Right: `__ldg`-direct (non-staged) variant for cols>12288, 5 blocks/SM, ~460 GB/s
   vs ~350-420 for bare opt-in. Impact: 27B decode literally cannot run without it;
   with the good fix, ~30-70 µs/layer (~4-8% of step) vs the crippled version.
2. **Land the Phase D alignment pad + launcher asserts (N2)** — blocking for the
   streaming tier on layers 0,1,2,4,5,6,8,9 (96/814 F8+scale tensors at ≡8 mod 16;
   layer 0 is the R4 parity layer). Asserts are 3 lines in fp8.cu and convert a
   device-side misaligned-address crash into a named throw.
3. **Complete the test to the R1 gate (N3)** — max-rel-err next to every cosine,
   all geometries (incl. 17408×5120 both orientations), T∈{3,33,64}, expect-throw
   cases. Without it, uniform-scale kernel bugs remain invisible and the master
   plan's R1 blocker is unenforceable.
4. **T-aware MMA skip in fp8_gemm (N4)** — guard the load/mma pair with
   `wm*16 < ceil16(T)`; ~1.5-2x on T=2-5 GEMM calls (the forced down_proj pair path).
5. **Promote-style scale application in fp8_gemm (R4/F8)** — drop `*sc` from
   dequant, fold per-64-K-step into the fp32 accumulator; exactness + ~5-10% GEMM
   time; prepares the e4m3-MMA upgrade.
6. **Bench the GEMV/GEMM (cold-L2, insig4-perf protocol)** — the 430-470 GB/s
   figure is analytic; the Phase C gate requires a measurement. Add a timing loop
   to test-fp8 or a bench-fp8 target when builds are next allowed.
7. **Delete/rename `e4m1`, fix the "exact" comment (N5)**; optionally
   size-dependent `__ldcs/__ldg` and the 2^8 scale fold (µ-opts).

## 6. What was run / not run

- Ran: `python tools/rundll.py build/test-fp8.dll` (output above); Python header
  enumeration over all 66 shards of `Qwen3.8-27B-FP8` (alignment histograms); grep
  audits over src/include/tools/TensorRT-LLM; smem/occupancy arithmetic (table in N1).
- Not run (rules): any nvcc/mk.py build, so N1 is static-analysis-verified only and
  no bandwidth measurement exists; no files modified except this report.
