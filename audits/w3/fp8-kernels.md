# FP8 kernel audit (w3) — e4m3 + bf16 128x128 block scales

Date: 2026-08-25. Scope: `src/fp8.cu`, `include/insignia_fp8.cuh`, `src/test_fp8.cu`,
`build/test-fp8.bat`. Method: line-by-line reading plus pure-computation Python proofs
(exhaustive 256-code enumeration of the e4m3 decode, simulated staging/coverage loops,
simulated store addressing, roofline arithmetic). No files modified, no builds, no git.

## Verdict summary

| # | Finding | Severity | Location |
|---|---------|----------|----------|
| F1 | `fp8_gemm` ignores `T`, always computes and **stores all 64 output rows** -> OOB write into any honest `[T,rows]` buffer (21.3x past the end at T=3); `T>64` silently leaves rows 64..T-1 unwritten | **CRITICAL** | fp8.cu:114, 179, 181-183 |
| F2 | `e4m3_host` test reference is exactly **64x too large for every normal code** (bias bug: `2^(e-1)` instead of `2^(e-7)`); invisible because the test checks cosine only, and cosine is invariant to a uniform scale factor (proved: cos = 1.000000000 on 500k weights) | **CRITICAL (test validity)** | test_fp8.cu:15 |
| F3 | test reads `x[i]` for `i < 3*cols` but `x` has `2*cols` elements -> 20 KB out-of-bounds heap read (UB) building the GEMM's t=2 row | HIGH | test_fp8.cu:76, 102 |
| F4 | `bf16_gemv_rows` declared but never defined -> link error if ever called; zero call sites today | MEDIUM | insignia_fp8.cuh:27 |
| F5 | `fp8_gemv2` dynamic smem = `2*cols*4`: at `cols=6144` (out_proj, the largest pair-GEMV target) it is exactly 49152 B — the 48 KB no-opt-in ceiling; `cols>6144` fails to launch, and no launcher checks `cudaGetLastError` or calls `cudaFuncSetAttribute` | MEDIUM (latent) | fp8.cu:98 |
| F6 | Test blind spots: only T=3 exercised for GEMM; cosine-only scoring (no max-rel-err); `f32_to_e4m3` is not RNE (ties round half-up; `[7.5*2^-9, 2^-6)` clamps to 0x07 instead of promoting to 0x08) — benign *only* because `wref` is derived from the emitted code | MEDIUM | test_fp8.cu:18-32 |
| F7 | `e4m1()` helper is misnamed (it decodes e4m3, single lane) and unused — dead code trap | LOW | insignia_fp8.cuh:17-20 |
| F8 | fp8.cu:102 comment "dequant e4m3->bf16 (exact)": the e4m3->bf16 conversion is exact, but the `*sc` product rounds to bf16 (<=2^-8 rel) | LOW (doc) | fp8.cu:144-147 |

Two hypotheses from the audit brief were **refuted** by arithmetic (details in "Verified
correct"): the gemv shared-staging loop covers `[0,cols)` completely and without overlap
(second iteration engages tid<64, not tid<16), and the GEMM warp decomposition covers all
output rows t=0..63 (wm = warp>>1 with warp in 0..7 gives wm in 0..3, i.e. 4x2 16x16 tiles
= 64x32 per block). The GEMM's real defect is F1, not missing coverage.

---

## F1 (CRITICAL): `fp8_gemm` ignores `T` and writes all 64 output rows

Evidence:
- fp8.cu:114 — `(void)T;` — `T` is explicitly discarded.
- fp8.cu:179 — `wmma::store_matrix_sync(y + size_t(wm * 16) * rows + n0 + wn * 16, acc, rows, wmma::mem_row_major);` — unguarded, `wm*16` reaches 48, so the store touches `y[t*rows + n]` for every `t` in 0..63.
- fp8.cu:182 — launcher validates `rows%32==0`, `cols%128==0`, `T>0` but **not** `T<=64`.
- test_fp8.cu:107 — `cudaMalloc(&dyT, size_t(64) * rows * 4);` — the test quietly allocates
  64 rows, which is why this never crashed.

Address arithmetic (T=3, rows=10240): buffer is 30720 floats; kernel writes up to
`y[63*10240 + 10239] = y[655359]` — 21.3x past the end. Any caller following the header
contract (y shaped `[T,rows]`, insignia_fp8.cuh:25 documents x16's `[64,cols]` padding but
says nothing about y) gets silent heap corruption. Conversely if someone passes `T=100`,
rows 64..99 of y are never written (silent garbage), because the A tile is hardwired to 64
rows.

Fix design (minimal, keeps the 64x32x64 tile):
1. Launcher: `if (T > 64) throw ...` (fp8.cu:182), and document in insignia_fp8.cuh:25 that
   y must be `[64,rows]` **or** implement the guarded epilogue:
2. Guarded epilogue: `store_matrix_sync` cannot store a partial tile portably (accumulator
   fragment element mapping is implementation-defined). Stage the tile to shared memory
   (`__shared__ float ep[16][17]`), `store_matrix_sync` into it, then copy out with
   `if (wm*16 + i < T) y[...] = ep[i][j]`. Cost: one 1 KB tile + a 256-thread copy per
   warp tile — negligible vs the k-loop.
3. Cheap compute skip for small T: wrap the 4 `mma_sync` (fp8.cu:172-176) in
   `if (wm * 16 < ((T + 15) & ~15))` — warps still must hit both `__syncthreads()` in the
   loop, so no divergence hazard. At T=3 this idles 6 of 8 warps' MMAs and moves the kernel
   toward its B-stream bound (~116 us for qkv) instead of burning full T=64 compute.
4. If T>64 support is ever wanted: `grid.y = ceil(T/64)` over 64-row A chunks (B tiles get
   L2-hit across grid.y if blockIdx.y is scheduled innermost). Do not grow the A tile to 128
   rows — As alone would be 36.9 KB x2 buffers and occupancy collapses to 1 block/SM.

## F2 (CRITICAL, test validity): `e4m3_host` is 64x wrong on all normals — and the test cannot see it

test_fp8.cu:15:
```c
else v = float(1 << (e - 1)) * (1.f + m / 8.f);      // (1+m/8)*2^(e-1)
```
OCP E4M3 (bias 7) normal value is `(1+m/8) * 2^(E-7)`. The code computes `2^(E-1)`, i.e.
exactly **2^6 = 64x** the true value for every normal code. Exhaustive enumeration over
all 256 codes (Python):

- 0x00 -> 0 (correct), 0x01 -> 2^-9 (correct), 0x07 -> 7*2^-9 (correct; subnormals and zero
  are right, which makes the bug maximally deceptive)
- 0x08 -> host 1.0 vs true 2^-6 = 0.015625
- 0x38 -> host 64.0 vs true 1.0
- 0x7E -> host 28672.0 vs true 448.0
- 238/238 normal codes wrong, each by exactly 64.0x; 0 subnormal/zero errors.

Why the test still prints a beautiful cosine: `wref[i] = e4m3_host(code) * scf`
(test_fp8.cu:68) = `64 * true` for normals and `true` for the ~1e-4 subnormal mass.
Cosine is invariant under uniform positive scaling of one vector; simulation with the
test's exact distribution (500k weights, N(0,0.05)/sc): **cos(buggy ref, truth) =
1.000000000**. Consequences:

- The test cannot catch any *uniform* scale-factor error in the kernel decode (e.g. a
  2^7-vs-2^8 mistake in `e4m3x2`, or the x256 folded wrongly into the scale).
- It also cannot catch *subnormal-path* errors: subnormal codes carry ~1e-4 of the weight
  mass, so a 2x error there moves cos by ~1e-8.

Fix: test_fp8.cu:15 -> `else v = ldexpf(1.f + m / 8.f, int(e) - 7);` and add an absolute
check next to every cosine: `max_rel_err = max(|y-ref|/|ref|)` over |ref| above a floor
(expect ~1e-6 for gemv, ~4e-3 for the bf16-dequant gemm). The encoder `f32_to_e4m3`
(test_fp8.cu:18-32) is OCP-correct (verified: 0/200000 RNE mismatches on random values;
see F6 for the tie/boundary deviations), so the bug is purely in the decoder.

## F3 (HIGH): test builds the GEMM x16 tile with an out-of-bounds read

test_fp8.cu:76 declares `std::vector<float> x(2 * cols)` (the two gemv2 rows), but
test_fp8.cu:102:
```c
for (int i = 0; i < 64 * cols; i++) x16[i] = __float2bfloat16(i < 3 * cols ? x[i] : 0.f);
```
For `i` in `[2*cols, 3*cols)` (t=2 row, 10240 floats = 40 KB region, 20 KB past the end)
this reads past the vector — UB. It happens to be "self-consistent" because the t=2
reference (line 126) re-reads the already-converted `x16` values, so cos still passes, but
under ASAN/Debug or an unlucky heap layout this is a crash, and the t=2 slice of the check
is numerically meaningless (garbage in, garbage agreed upon). Fix: size x as `3*cols`
(line 76) and fill the third row, or make t=2 explicit: `x16[t*cols+c] = bf16(t<2 ? x[t*cols+c] : 0.f)`.

## F4 (MEDIUM): `bf16_gemv_rows` declared, never defined, never called

insignia_fp8.cuh:27 declares it; `src/fp8.cu` has no definition; repo-wide grep finds zero
call sites and zero other mentions. Any future caller gets an unresolved external at link
time. Remove the declaration or implement it (a bf16 row-gather GEMV is needed for
`mtp.fc` [5120,10240] bf16 and embed/lm_head bf16 paths — if that is the plan, implement;
otherwise delete).

## F5 (MEDIUM, latent): gemv2 shared-memory ceiling and unchecked launches

- fp8.cu:98 launches with `2*cols*4` bytes dynamic smem. Model mats: cols=5120 -> 40 KB
  (fine), cols=6144 (out_proj [5120,6144], the widest pair-GEMV target) -> **49152 B =
  exactly the 48 KB default per-block limit** — passes, with zero headroom. Any future
  cols=10240 (e.g. someone pointing gemv2 at a transposed mtp.fc layout) -> 80 KB ->
  launch failure. Fix: `cudaFuncSetAttribute(fp8_gemv2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024)`
  once at init (sm_89 allows 99 KB), and/or throw on `cols > 12672`.
- None of the three launchers checks `cudaGetLastError()` after `<<<>>>`; the kernel-side
  early throws (fp8.cu:53, 97, 182) cover dims but not launch-config failures (smem size,
  invalid stream). The test happens to check `cudaDeviceSynchronize` (test_fp8.cu:88 etc.);
  engine callers may not.

## F6 (MEDIUM): quantizer is not RNE — benign here, but document it

`f32_to_e4m3` (test_fp8.cu:18-32) deviates from round-to-nearest-even in exactly two ways,
both verified:
1. Ties round half-up (`+0.5f` then truncation), e.g. v = 2^-10 (exact midpoint of 0 and
   2^-9) -> 0x01, RNE -> 0x00.
2. The subnormal/normal boundary: for v in [7.5*2^-9, 2^-6) the `m = int(v*512+0.5)` path
   yields m=8 and the `m < 8 ? m : 7` clamp (line 26) returns 0x07 (7*2^-9) instead of
   promoting to 0x08 (2^-6) — up to 12.5% error on that 0.1%-wide band.

This does not invalidate the test because `wref` is computed **from the emitted code**
(test_fp8.cu:67-68), so quantizer infidelity is invisible by construction — the test
validates decode + scale indexing + accumulation, not the quantizer. Worth a comment in
the file so nobody reuses `f32_to_e4m3` as a production quantizer. (0/200000 mismatches
vs RNE on random non-tie inputs; max relative error 0.15 is inherent e4m3 subnormal
granularity, present in RNE too.)

## F7/F8 (LOW)

- F7: `e4m1()` (insignia_fp8.cuh:17-20) decodes a single **e4m3** byte (same bit trick as
  `e4m3x2`); the name refers to a different format. Unused everywhere. Delete or rename
  `e4m3_lane` before someone calls it expecting E4M1 semantics.
- F8: fp8.cu:102 comment says dequant is exact; the e4m3->bf16 conversion is exact, but
  `__float2bfloat16(a.x * sc)` rounds the *product* to bf16 (<= 2^-8 relative). Numerically
  harmless (e4m3 quantization noise is ~2^-4.3 rms), but the comment overstates. The
  promote-accumulator restructure in the perf section removes this rounding entirely.

---

## Verified correct (with proof)

### 1. `e4m3x2` bit trick — EXACT for all finite codes, both lanes, both signs (task 1)

insignia_fp8.cuh:12-16. Exhaustive enumeration of all 256 codes against exact OCP E4M3:
**zero mismatches over the 254 finite codes** (signs and +-0 included).

Why it is exact (the proof, not just the simulation):
- Masks: `0x007f007f` selects bytes 0 and 2 (the low byte of each 16-bit half);
  `0x00800080` selects bits 7 and 23 — the two sign bytes. `<<7` puts each magnitude byte's
  7 bits at half-lane bits 7..13; `<<8` puts each sign at half-lane bit 15. (Note the sign
  mask is {7,23}, not {7,15} — a easy-to-misread constant that is nevertheless right.)
- Normal e4m3, exponent field E in 1..15: constructed fp16 has exponent field E (bits
  10..13; bit 14 stays 0), value `(1 + m*8/1024) * 2^(E-15) = (1+m/8)*2^(E-15)`;
  `*256` gives `(1+m/8)*2^(E-7)` — the exact e4m3 normal value, since 3 mantissa bits
  drop into the fp16 mantissa without rounding.
- Subnormal e4m3 (E=0): constructed fp16 is the fp16 subnormal with mantissa `m<<7`,
  value `m*128 * 2^-24 = m*2^-17`; `*256` gives `m*2^-9` — exact, because fp16 subnormal
  steps (2^-24) divide the e4m3 subnormal steps (2^-17).
- The `*256` runs in fp32 (after `__half22float2`), and even in fp16 it would be exact:
  max pre-multiply magnitude is 1.875 (code 0x7F), post-multiply 480 — nowhere near
  fp16 max 65504. **No overflow anywhere.**

Spot values (code -> bit-trick result): 0x00 -> 0; 0x01 -> 2^-9; 0x07 -> 7*2^-9;
0x08 -> 2^-6; 0x38 -> 1.0; 0x7E -> 448.0; 0x80 -> -0.0; 0xFE -> -448.0. All exact.
Two-lane form verified on 20,000 random u32 words (bytes 0,2).

**One documented caveat**: 0x7F/0xFF — the only e4m3 NaN encodings (OCP E4M3 has no
Infinity) — decode to +-480, not NaN. The header comment scopes the claim to "every
finite code", so this is correct-by-documentation; garbage or NaN checkpoint bytes would
decode as +-480*scale instead of poisoning the output with NaN. For weight-only use this
is acceptable; know it exists.

### 2. `fp8_gemv` / `fp8_gemv2` indexing, alignment, staging (task 2) — all correct

- Scale indexing `scales[(row>>7)*kblocks + (c0>>7)]` (fp8.cu:28+33, 72+77) against the
  `[ceil(rows/128)][ceil(cols/128)]` layout: `kblocks = cols>>7` equals the true column
  count because `cols%128==0` is enforced (fp8.cu:53). `(row>>7) <= (rows-1)>>7` always
  lands inside `ceil(rows/128)` rows. Per-lane 16-weight chunks starting at `c0` (a
  multiple of 16) never straddle a 128-column boundary (checked exhaustively for all c0
  in 0..5104 step 16), so one scale per 512-column round is legitimate.
- 16B alignment: `row_w + c0` — `row*cols` is a multiple of 128 and `c0` of 16; base from
  cudaMalloc. `x + c0` as float4 — same argument. Both require the *base pointers* to be
  16B-aligned, which cudaMalloc guarantees and the engine's device-resident buffers must
  preserve (worth one assert at wire-up time if a mmap'd/partial-residency path ever
  hands over sub-row-aligned slices).
- Shared staging loop (fp8.cu:17-22, 61-66) — the brief suspected a coverage hole at
  c0=4096. Simulated for cols=5120, blockDim=256: iteration 1 covers floats 0..4095
  (tid 0..255), iteration 2 engages **tid 0..63** (c0 = tid*16+4096 < 5120 iff tid < 64;
  64 threads x 16 floats = the remaining 1024), tid 64..255 skip. Coverage complete
  (5120/5120), zero duplication. The brief's "tid<16" was an arithmetic slip; the loop is
  a standard grid-stride over 16-float chunks and handles any `cols%16==0` correctly.
- gemv2 specifics: staging spans `2*cols` (fp8.cu:61), `x1 = sx + cols + c0` (fp8.cu:80)
  is inside it; y layout `y[row], y[rows+row]` (fp8.cu:94) matches the test's `[2,rows]`
  view (test_fp8.cu:98). Warp reduction over both accumulators correct.
- The `pw[wi] >> 8` second decode (fp8.cu:39, 84, 143) recovers bytes 1 and 3 (bytes 0/2
  come from the unshifted call); all four products consume both lanes of both calls —
  nothing dead, nothing double-counted.

### 3. `fp8_gemm` tile coverage, staging, pipeline (task 3) — coverage complete; the defect is F1

- Warp decomposition: 256 threads = 8 warps; `wm = warp>>1` in {0,1,2,3}, `wn = warp&1`.
  The 8 warps own the 8 16x16 tiles of a 64x32 output block — simulated: 2048/2048
  (t,n) elements covered, t spans 0..63. **The brief's claim "rows 32..63 never computed"
  is refuted** (it assumed warp in 0..63). The real bug is the unconditional store (F1).
- dequant coverage (fp8.cu:135-149): 256 threads = 32 rows x 8 threads, each converts 8
  e4m3 (uint2 = 8 bytes) -> 8 bf16 stored as one uint4 — 64 bytes/row exactly matches
  KT=64 raw bytes. Scale index `((n0+n)>>7)*kblocks + (kb>>1)` (fp8.cu:136): `n0` is a
  multiple of 32 so `n0+n` spans 32 consecutive rows, always inside one 128-row scale
  band; `kb>>1` correctly shares one scale across the two KT=64 steps of a 128-col block
  (ksteps = cols/64 is even since 128|cols).
- cp.async staging: A = 64x8 chunks of 16 B (row stride cols, k 64-aligned -> srcoff
  16B-aligned given 16B-aligned x16); B = 32x4 chunks of 16 B — byte counts match
  KT=64 e4m3 bytes per row. Alignment verified for cols%128==0.
- Pipeline sync: trace of commit/wait_group pairs shows `wait_group 1` at iteration kb
  always retires group G_kb (the newest is G_{kb+1}), and the per-thread cp.async results
  are published block-wide by the `__syncthreads()` at fp8.cu:166 before `dequant` reads
  `Braw[buf]`. Double-buffering is conflict-free: iteration kb+1 prefetches buf^1 while
  mma reads buf; the fp8.cu:177 sync retires buf before it is refilled two iterations
  later. The final `wait_group 0` is conservatively correct.
- bf16 col_major B fragment vs row-major As with ldm 72 (APAD/BPAD=8): consistent with
  `Y[t,n] = sum_k X[t,k]*W[n,k]` and the store `y[t*rows+n]`, matching the test's
  `yT[t*rows+r]` view (for whatever rows it actually stores — F1).
- Static smem 31,744 B/block -> 3 blocks/SM; no overflow.

### 4. `bf16_get_row` (fp8.cu:187-194)

Grid-stride copy, bf16->f32, row read from device memory (no sync). Trivially correct;
grid (cols+255)/256 = 20 blocks at cols=5120 is fine for a 10 KB copy.

---

## Performance review (task 6)

### fp8_gemv on 10240x5120 (in_proj_qkv / in_proj_z shape) — expectation ~430-470 GB/s

- Traffic per call: 52.4 MB weights (+20 KB x, re-read from L2 by all 1280 blocks; +6.4 KB
  scales). Grid 1280 blocks = 4.6 waves at 5 resident blocks/SM (20 KB smem, 256 threads)
  — good tail behavior, as the brief noted.
- Instruction budget per 512 B warp-round (32 lanes x 16 weights): ~1 LDG.128 + 8x e4m3x2
  (~4 bit-ops each with LOP3 fusion) + 16 cvt/mul (half22float2 + x256) + 16 fmaf + LDS +
  scale -> ~90 warp-instructions. At 460 GB/s that is ~81 G warp-instr/s vs ~554 G/s issue
  capacity (56 SM x 4 schedulers x ~2.48 GHz) = **15% issue utilization**; FP32-family
  pipes ~8-10%. The FP32-decode path is *not* the bottleneck — the kernel should stream
  weights at 85-92% of the 504 GB/s peak, i.e. **~115-125 us** for this mat. This
  corroborates AGENTS.md's finding that direct FP32 accumulation beats DP4A variants.
- Per-token decode cost across one linear-attn layer (qkv 52.4 + z 31.5 + out 31.5 +
  a/b 0.5 MB ~ 116 MB) ~ 260 us + state I/O; the engine-level numbers in
  audits/synthesis.md stand.

Ranked improvements (expected end-to-end gain on this kernel family):

1. **L2-residency-aware loads for small mats** — `__ldcs` (evict-first) is right for
   52 MB streams but wrong for k/v_proj (5.2 MB) and a/b (0.25 MB), which fit the 48 MB
   L2 and are re-read every token. Switch those call sites to plain `__ldg` (or keep
   __ldcs only when `rows*cols > 24 MB`). Gain: those mats go from ~450 GB/s (DRAM) to
   ~1.5-2.5 TB/s (L2) — 5-15 us/token saved across 16 full-attn layers (~1-2% of decode).
   Free, zero risk.
2. **Split-K (or persistent grid-stride rows) for rows <= ~3584** — k_proj/v_proj at
   rows=1024 launch only 128 blocks on 56 SMs (46% of the 280 block-slot capacity at
   5 blocks/SM): latency-exposed, realistically 2-4x off peak. Split-K x4 over cols with
   a tiny second-pass reduce brings them near the L2/DRAM bound. Gain: ~10-20 us/token
   aggregate; also unblocks the L2 win in (1) from being latency-swamped.
3. **half2 arithmetic path (optional)** — keep `e4m3x2`'s half2 output, stage x as half2,
   `__hfma2` accumulate 8 pairs, promote to fp32 once per 512-col round. Cuts math
   instructions ~60% and halves sx (gemv2: 40 KB -> 10-20 KB => occupancy 2 -> 5
   blocks/SM). Since gemv is memory-bound at 15% issue, expect **~0-5% for fp8_gemv**;
   the real beneficiary is **fp8_gemv2 (+10-20%)** if it is latency-bound at 33%
   occupancy. Precision: 8-term fp16 partial sums with sc folded pre-hfma2 give ~0.1-0.3%
   error — well under e4m3's ~2% rms quantization noise; parity-check against the NumPy
   reference before adopting (per AGENTS.md conventions).
4. **Fold the x256 into the scale** (project-spirit micro-opt, provably exact): in
   fp8_gemv/gemv2, replace the two FMULs per e4m3x2 with one add per round: multiplying
   the bf16 scale by 2^8 is exactly `sb += 0x400000` on the fp32 bit pattern (fp32
   exponent += 8; bf16->fp32 has a zero low mantissa so no carry). Then
   `acc = fmaf(part, sc256, acc)` with `part` accumulated from un-scaled half values.
   Saves 16 of ~80 instructions/round (~20% issue headroom); end-to-end ~0-3% now, more
   valuable if a later variant becomes issue-bound.
5. Not worth it here: cp.async/cuda::pipeline for x staging (x staging is ~1% of the
   kernel; 26 MB of L2 x re-reads across 1280 blocks overlap fine), L2 accessPolicyWindow
   on x (already L2-resident via __ldg), uint4 x loads are already float4 and fine.

### fp8_gemm — compute-bound at T=64; wasteful at small T

- T=64, qkv: FLOPs = 2*64*10240*5120 = 6.87e10. WMMA bf16 peak on sm_89 ~82.6 TF ->
  832 us ideal; a 2-stage cp.async pipeline with 8 warps typically sustains 50-70% ->
  **~1.2-1.6 ms**. Weights (52.4 MB) and the 205 MB of A-tile L2 re-reads (A is 640 KB,
  L2-resident, ~50-100 us at 2-4 TB/s) hide completely under the MMA. So at T=64 this
  kernel is *compute*-bound, and the ranking of improvements flips:
  1. **Native e4m3 MMA** (`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, Ada-only,
     2x bf16 rate, per audits/synthesis.md) with TRT-LLM's blockwise promote: run raw fp8
     MMA per 128-K slab into a temp fp32 accumulator, then `acc += temp * (SFA[row]*SFB)`.
     This also deletes the dequant stage (Bs smem + cvt + the F8 bf16 rounding) — Braw
     feeds the MMA directly and the bf16 x16 path becomes a bf16-A/bf16... note A must be
     e4m3-quantized for the fp8xfp8 MMA (per-token-group 1x128 dynamic quant per
     synthesis.md), which is a bigger change; A-bf16 x B-fp8 mixed MMA does not exist, so
     the intermediate stepping stone is: keep bf16 MMA, drop `*sc` from dequant (pure
     exact e4m3->bf16), promote-scale every 2 kb in fp32 — removes F8's rounding and the
     4 FMULs, ~10-15% of dequant cost, and is the exact code shape the fp8 MMA upgrade
     needs later. Expected end state ~2x on T=64 GEMMs (600-750 us).
  2. **T-aware compute skip** (F1 fix item 3): at T=3 the kernel currently burns full
     T=64 MMA work (~1.2 ms) for 3 usable rows; skipping wm tiles >= T approaches the
     ~116 us B-stream bound — a ~10x for short-T calls.
  3. 3-stage pipeline / NT=64 tile: +10-20% MMA efficiency at T=64; do after the fp8 MMA
     rewrite, which changes the smem budget anyway.

### Cross-checks

- gemv2 at 40 KB smem (cols=5120) = 2 blocks/SM = 33% occupancy — acceptable given two
  independent accumulator chains per lane, but see improvement (3).
- All three kernels: no register-pressure red flags; `__launch_bounds__(256)` present on
  the two hot kernels; `__ldcs` only on the streamed weights (correct for the big mats).

## TL;DR

1. `fp8_gemm` ignores `T` (`(void)T`, fp8.cu:114) and stores all 64 output rows unguarded
   (fp8.cu:179) — 21x OOB write on a `[T,rows]` buffer at T=3; the test hides it by
   allocating 64 rows. Fix: guarded epilogue + throw on T>64; optionally skip mma for
   wm*16 >= T (10x at T=3).
2. The test's own reference decoder `e4m3_host` (test_fp8.cu:15) is exactly 64x too large
   for all 238 normal codes (`2^(e-1)` vs `2^(E-7)`); subnormals are right.
3. Cosine-only scoring cannot see that (proved: cos = 1.000000000) — add max-rel-err; fix
   with `ldexpf(1+m/8, e-7)`.
4. Test also reads 20 KB past `x` building the t=2 GEMM row (test_fp8.cu:102, vector is
   2*cols but loop goes to 3*cols) — UB, self-consistently masked.
5. The e4m3x2 bit trick is EXACT for all 254 finite codes incl. subnormals and both signs
   (exhaustively enumerated + proof); only 0x7F/0xFF (e4m3 NaN) decode to +-480 instead
   of NaN; no fp16 overflow anywhere (max intermediate 1.875 pre-scale, 480 after).
6. gemv/gemv2 scale indexing, 16B alignment, and the shared staging loop are all correct
   — the brief's "second iteration only tid<16" worry is refuted (it is tid<64, coverage
   complete and disjoint for cols=5120).
7. The GEMM warp tiling covers t=0..63 completely (wm in 0..3, not 0..31 as feared) —
   coverage was never the bug; the ungarded store is.
8. `bf16_gemv_rows` is declared (insignia_fp8.cuh:27), never defined, never called —
   link-error landmine; `e4m1()` is misnamed dead code.
9. gemv2 smem = 2*cols*4 sits exactly at the 48 KB limit for out_proj (cols=6144) and
   would fail >6144; no launcher checks launch errors or opts into 99 KB.
10. Perf: gemv should stream at ~430-470 GB/s (15% issue utilization — decode path is not
    the bottleneck); best wins are L2-resident loads for small mats, split-K for
    rows<=3584, and half2 staging for gemv2 occupancy; GEMM at T=64 is compute-bound —
    the native e4m3 MMA + fp32 block-scale promote (TRT-LLM pattern) is worth ~2x.
