# INSIG4 kernel audit — w4: correctness, q8 quality, perf, 27B viability (2026-08-25)

Scope: `src/mxfp4_i4.cu` (all), i4 kernels in `src/gemm.cu` (`mxfp4_gemm_mlx_i4_kernel` :303,
`mxfp4_gemm_v21_i4_kernel` :371, `mxfp4_gemm_ab_i4_kernel` :461), callers in `src/decode.cu` /
`src/prefill.cu`, format in `tools/quantize_insig4.py`. Evidence: re-ran `python
tools/rundll.py build/test-i4.dll` (green, numbers below), disassembled the **already-built**
`build/test-i4.dll` with cuobjdump (read-only), numpy emulation of the q8 path, SASS/PTX
cross-checks. No files modified. Cross-referenced `audits/internals.md`, `audits/w3/insig4-perf.md`.

Fresh test run (matches the brief):
```
gemv_v2_i4  max_rel=1.193e-05 cos=1.00000000   gemm_mlx_i4  max_rel=2.97e-01 cos=0.99999730
get_row_i4  max_abs=0.0                          gemv2_q8_i4  max_rel=7.555e-01 cos=0.99998530
gemm_v21_i4 T=33 max_rel=5.689e-02 cos=0.99999890   (vs mlx_i4 agreement 2.7e-04)
gemm_ab_i4  vs v21_i4(32 rows): a=0.000e+00 b=0.000e+00
```

## 1. Format and scale indexing — all verified correct

Format (quantize_insig4.py:59-64): E2M1 nibbles, 8/u32, MLX packing, 4 u32 per 32-elt group;
ONE fp16 scale per 64-elt super-group, stored `[rows][cols/64]` (py:64). Engine invariant:
`scales` stride per row = `groups>>1` where `groups = cols>>5`. `i4_scale(s,g) =
half2float(s[g>>1])` (mxfp4_i4.cu:11-13). Scales are F16 and every reader uses `__half2float`
(the old read-as-bf16 bug is gone everywhere).

| kernel | scale index | verdict |
|---|---|---|
| `mxfp4_gemv_v2_i4` mxfp4_i4.cu:54 | `i4_scale(row_s, g0)` = s[g0>>1], g0 = 32-group | correct |
| `mxfp4_gemv2_q8_i4` :142 | `i4_scale(row_s,g0)*0.5f` (0.5 folds out the 2x LUT magnitudes tbl={0,1,2,3,4,6,8,12} :106) | correct |
| `mxfp4_gemv_ab2_q8_i4` :225 | same as above | correct |
| `mxfp4_gemm_mlx_i4` gemm.cu:330 | `(k0>>5)>>1` = k0>>6 (KT=32 steps, 2 steps share one super-group scale) | correct |
| `mxfp4_gemm_v21_i4` gemm.cu:406 | `(groups>>1) row stride + kb` (KT=64 step == one super-group, scale idx = kb exactly) | correct |
| `mxfp4_gemm_ab_i4` gemm.cu:498 | `n*(groups>>1) + kb` (n0=0: each block owns a whole 32-row matrix) | correct |
| `mxfp4_get_row_i4` :251 | `s[row*(groups>>1) + group>>1]` | correct (max_abs=0.0) |
| `embed_gather_i4` prefill.cu:30 | `s[row*64 + (g>>1)]` (row*64 hardcodes cols=4096) | correct at 4096; see §6 |

No off-by-ones anywhere in the scale paths. Nibble->element mapping in every kernel: word w of
group g covers elements `g*32 + w*8 + j`, nibble j (little-endian nibble order matches the
quantizer's `packed |= nib << (4*j)`, py:62). `get_row_i4` thread map `word=lane>>3, nibble=(lane&7)*4`
(:249-250) matches; verified bit-exact by test.

## 2. Pipeline race (the `kb+2<ksteps` fix) — present in ALL THREE kernels

The hazard: at the tail iteration `kb = ksteps-1` no new group is committed, so exactly ONE
cp.async group is outstanding and `wait_group 1` is a **no-op** — dequant would read unloaded
`Braw`. The tail must `wait_group 0`. Verified:

- `mxfp4_gemm_v21` (e8m0) gemm.cu:275-276 — `if (kb+2<ksteps) cp_async_wait_prev(); else cp_async_wait_all();`
- `mxfp4_gemm_v21_i4` gemm.cu:432-433 — identical fix. **BOTH v21 and v21_i4 have it.**
- `mxfp4_gemm_ab_i4` gemm.cu:524-525 — identical fix.

Double-buffer reuse safety (all three): `prefetch(kb+1, buf^1)` is issued only after the prior
iteration's trailing `__syncthreads()` (:445/:537), so all reads of `buf^1` (wmma loads + dequant
reads of `Braw[buf^1]` from iteration kb-1) complete before cp.async overwrites it; dequant(kb,buf)
writes `Bs[buf]` (never cp.async'd) between its own two syncs. No race found.

Minor (perf, not correctness): the condition `kb+2<ksteps` fires `wait_all` one iteration early
(at `kb == ksteps-2` group kb+1 is committed and `wait_group 1` already suffices), serializing the
last prefetch — costs ~1 kstep of latency hiding (1/272 of the K loop at cols=17408). Could be
`kb+1<ksteps` with the `else wait_all`.

## 3. ab_i4 pair kernel — no smem aliasing, stores guarded

- Two blocks, `weights = blockIdx.x ? wb : wa` uniform per block (gemm.cu:474-476): **shared
  memory is per-block instantiation**, so blocks never alias each other's `As/Bs/Braw/lut`. `ya`
  and `yb` are distinct buffers; `y = blockIdx.x ? yb : ya` (:476) with stride-32 store :539.
  Test (`a=b=same weights`) exact-pair 0.0 + both-blocks-independent confirms.
- Store guard `if (wm*16 < T)` (:539, also v21_i4 :447): a tile straddling T writes up to 16
  scratch rows beyond T — safe because every caller output is a 64-row pf buffer (decode.cu:22-24).
  A-side is zero-padded to 64 rows by the callers (`cudaMemsetAsync` tail + `f32_to_bf16`,
  decode.cu:36-38, :73-74) before `prefetch` reads all 64 rows. Correct.
- `mxfp4_gemv_ab2_q8_i4` (dp4a T=2 twin, mxfp4_i4.cu:157-242): staging hardcodes r=tid>>7,
  g=tid&127 (2x128 groups) but the launcher **throws unless cols==4096** (:239), so the hardcode
  is guarded. One `__syncthreads()` total (:191), no early-return before it. ya/yb writes
  `[2,32]` into 64-row buffers (:232-233) — matches caller layout.

## 4. `__byte_perm` selectors 0xc480/0x4c80/0x6240 — safe on this toolchain (SASS-proven)

The q8 kernels (mxfp4_i4.cu:131-136, :214-219; mxfp4.cu:344-349 etc.) build `[v0,v1,v2,v3]`
int8 weight words via `btab` u64 LUT + 3 `__byte_perm` per half-word. Selector nibbles 8 and 0xC
appear (0xc480, 0x4c80): PTX `prmt` documents msb-of-nibble as sign-replication, under which
`w0.byte3` (element 3 of each quad) would become 0x00 and cos would collapse to ~0.87. Test says
cos=0.999985, and the SASS of build/test-i4.dll settles it: **ptxas normalized the 3-perm network
into 2 PRMTs with selectors 0x40 and 0x410 — all nibbles <= 7** (e.g. `PRMT R27, R4, 0x40, R5`;
`PRMT R6, R27, 0x410, R6`), i.e. ptxas treats nibble>=8 as index&7 and optimizes accordingly.
144 PRMT + 80 `IDP.4A.S8.S8` in the kernel. Recommendation (zero cost, portability insurance):
write the source selectors as 0x4440/0x4440/0x6240 — provably identical under ptxas, and inside
the documented 0-7 selector range so no other compiler/arch can reinterpret them.

## 5. QUALITY: why `gemv2_q8_i4 max_rel=0.755` — activation int8 quantization, measured

The kernel quantizes BOTH activation rows per 32-element group to int8 absmax
(`m = max|x_g|`, `xs = m/127`, codes `rn(x*127/m)`, mxfp4_i4.cu:84-104) and does the dot in
dp4a with weight scale folded (`ws = i4_scale*0.5`, :142-144). Weight side is exact (integer
tbl = 2xE2M1, compensated by 0.5). So the ONLY error vs the dequantized-weights reference is
activation quantization. numpy emulation of exactly this scheme (test_i4's N(0,0.03) data,
amax/6 INSIG4 weights, per-32-group absmax int8):

```
emulated: max_rel=0.527 cos=0.99998   sigma_e/sigma_ref = 0.66%  -> per-output SNR ~152:1
bf16 activations (mlx_i4/v21_i4 path): max_rel=0.118 cos=0.9999989  -> SNR ~1100:1
worst 200 rel errors sit at |ref| median 6.4e-4 vs overall 2.8e-2  (denominator floor 1e-3)
q8 with per-1024 groups: cos=0.9999795 (WORSE — wider groups are not a fix)
```

Verdict: **max_rel=0.755 is not a bug** — it is the 0.66% Gaussian noise of int8 activation
quantization landing on near-zero dot products where the rel denominator (|ref|+1e-3) floors
out. cos=0.999985 is the honest quality number. End-to-end: `linear2` routes the ENTIRE T=2
spec-verify path through this kernel (decode.cu:32,:53-66,:90-92,:99), so every projection
output in speculative verification carries 0.66% relative noise, compounding through 32 layers
to roughly 1.5-2.5% hidden-state noise — enough to flip greedy argmax on close tokens (spec
accept-rate) and worth ~0.005-0.02 nats; the current nll_compare.py cannot see it because the
mxfp4 A/B baseline (`mxfp4_gemv2_q8`) quantizes activations identically. Fixes:

1. **Recommended: retire activation int8 on this path.** Build `mxfp4_gemv2_v2_i4` — the
   e8m0 `mxfp4_gemv2_v2` pattern (insignia_layout.cuh:62 declares the twin; no i4 version
   exists) with both x rows staged fp32 and TWO fp32 accumulators per weight decode (weights
   still read ONCE — the point of the pair kernel; extra cost is 1 FMA per element per extra
   row, reusing the same LUT lookup). Quality becomes gemv_v2_i4-class (max_rel ~1e-5). Per
   AGENTS.md the fp32-nibble path already beats Q8/DP4A "by a wide margin" on decode GEMV, so
   this is a quality win AND a perf win. smem cost: 2*cols*4+64 = 32.8KB @4096 (3 blocks/SM),
   98.4KB @12288 (1 block/SM; if down_proj pair occupancy hurts, stage activations bf16 at
   49.2KB -> 2 blocks/SM, still SNR ~1100).
2. If dp4a must stay: hoist quantization out of the GEMV (the `quantize_x8` + `q8g` pattern,
   w3 §2.2) — fixes perf, does NOT fix the 0.66% noise.
3. "Per-token fp16 activation scale" / wider groups: refuted above (per-1024 cos drops to
   0.99998; per-token is worse still). Narrower than 32 is impossible at dp4a granularity.

## 6. PERF analysis at 9B (4096/12288) and 27B-linear (5120/10240/17408) shapes

Weight bytes = elts * 0.53125 (0.5 nibble + 2B/64 scales). Streaming BW: 379-435 GiB/s
demonstrated on real shapes (internals.md:24-25); the 150-156 GiB/s in build/last-test.log is
L2-resident fiction (w3 §2.4). 4070S: 56 SMs, 504 GB/s, issue ~554 G warp-instr/s.

- **gemv_v2_i4** (decode GEMV): hot loop 0x18f0-0x51f0 in SASS = 912 SASS per 4 groups per lane
  (64B weights) -> ~2.24 weight-bytes per issued warp instruction. Mix per 32 elts: 32 LDS
  (16-entry f32 LUT + transposed x) + 32 FFMA + ~96 LOP3/SHF/IMAD. Issue ceiling ~1.2 TB/s >>
  DRAM => memory-bound; `__ldcs` emitted as LDG.E.EF.128; scales compile to LDG.E.U16.CONSTANT
  (const __restrict__ gives the __ldg path even without the intrinsic). Times @407-450 GB/s:
  lm_head 248320x4096 (541MB) 1.20-1.33 ms; qkv 8192x4096 40-44us; gate/up/down (26.7MB)
  59-66us; o_proj 6144x4096 30-33us; 27B: out_proj 5120x6144 37-41us, gate/up 17408x5120 and
  down 5120x17408 (47.3MB) 105-116us, lm_head@5120 (676MB) 1.50-1.66ms. a/b (32 rows): grid=4
  blocks = 32 warps on 56 SMs, latency-bound 3-6us — the ab pair kernels exist to kill exactly
  this (and ab_i4 replaced a 3072-launch loop, gemm.cu:456-460).
- **gemv2_q8_i4**: inner loop ~42 instr per 32 elts (8 LDS.64 btab + 16 PRMT + 8 IDP.4A +
  2 FMA) — 6x leaner than v2_i4 per byte, but EVERY block re-quantizes both activation rows
  (mxfp4_i4.cu:84-104): for lm_head pair that is 31040 blocks x 32KB L2 re-reads ~ 1.0 GB of
  L2 traffic plus ~1us staging latency ahead of each 16KB weight stream. This is why fp32
  wins end-to-end (AGENTS.md). The q8 path's only defensible niche would be after fix #2
  (hoisted quantization), and even then fp32-pair (#1) dominates on both axes.
- **gemm_mlx_i4** (gemm.cu:303-357): single-buffered, no cp.async, dequant on critical path,
  fp32 A staged by only 64/256 threads. Now **test-only** (decode.cu:39-40 routes ALL INSIG4
  batch GEMM through v21_i4). No action.
- **v21_i4 / ab_i4**: 64x32 tile, KT=64. cp.async 24x LDGSTS 16B per step (A: 2x16B/thread;
  B: 2x16B per row per step, rows 32B apart — max cp.async width is 16B, nothing to widen on
  sm_89; TMA is Hopper+). Scales (2B/row/step) are NOT cp.async'd — read via __ldg in dequant
  (gemm.cu:406,:498), fine through L1. wmma load_matrix_sync from smem (LDSM) OK. Two findings:
  (a) smem reads inside the dequant/prefetch lambdas compile to **generic LD.E** (32 per
  kernel, 64-bit generic addressing) instead of LDS — identical in the e8m0 v21, so not an i4
  regression, but restructuring the lambdas to keep smem-typed pointers would shave address
  math; (b) dequant sits between two `__syncthreads()` on the critical path (:434-436) — the
  w3 §1.2/§3.1.5 plan (fold the fp16 scale into the 256-entry bf16 pair-LUT via `__hmul2`,
  4x fewer dequant SASS; 3-stage Bs so dequant overlaps mma) still applies verbatim to v21_i4
  and ab_i4; INSIG4 needs exactly ONE scale per KT=64 step (idx = kb) which is the cleanest
  possible case. Compute balance at T=64: 241 FLOP/weight-byte => mma-bound above ~300 GB/s of
  weight stream; 27B down 5120x17408 T=64: mma floor 138-161us (71-83 TF) vs 105-116us memory —
  compute-bound ~1.3x, so dequant instruction reduction directly buys time (est 10-20%).
- **Dequant LUT approaches** (task question): decode GEMV uses the 16-entry f32 LUT (1 LDS per
  nibble + 1 LDS for x): LDS pipe utilization ~4.4 TB/s-equivalent ceiling, NOT a limiter;
  switching it to the 256-entry pair-LUT would halve LDS but the kernel is DRAM-bound — low
  priority (w3 §1.4 concurs). GEMMs already use the 256-entry bf16 pair-LUT (1 LDS per 2
  weights) — correct choice. q8 kernels use the 256x u64 byte-broadcast table — optimal for
  dp4a (1 LDS.64 feeds 4 IDP.4A).
- **__ldg on scales**: effective everywhere (const __restrict__ -> LDG.*.CONSTANT in SASS,
  verified in v2_i4 and mlx_i4); explicit only in v21_i4 :406 and ab_i4 :498. Cosmetic.

## 7. 27B-shape viability — constraint matrix

27B linear shapes: hidden 5120 (x5*1024), out_proj cols 6144 (x6*1024), gate/up rows/cols
10240-family and 17408 (x10/x17*1024) — **every 27B cols value is a multiple of 1024**, and
5120/6144/17408/10240 are all %32/%64 clean, so:

| kernel | rows | cols | smem @cols=17408 | 27B verdict |
|---|---|---|---|---|
| gemv_v2_i4 (:70) | any (row<rows guard) | %1024 (throw) | 69.8KB <= 99KB (attr set :71) | OK as-is |
| gemv2_q8_i4 (:151) | any | %32 (throw) | 45.6KB | OK (but see §5: retire) |
| gemv_ab2_q8_i4 (:239) | 32 baked | ==4096 (throw) | — | **BLOCKED** (9B-specialized by design) |
| gemm_mlx_i4 (:355) | %64 (throw) | %32 (throw) | 12KB static | OK (test-only anyway); vocab 248320%64==0 |
| gemm_v21_i4 (:451) | %32 (throw) | %1024 (throw; kernel itself only needs %64) | 30.7KB static, 3 blocks/SM | OK; ksteps=80..272, scale idx=kb exact |
| gemm_ab_i4 (:543) | 32 baked (no param!) | %1024 (throw), T 1..64 (throw) | 30.7KB | **rows=32 hardcoded** — 27B in_proj_a/b row count must be checked before reuse |
| get_row_i4 (:254) | any | **NONE (silent)** | 0 | works @5120 (grid-stride) but add a throw (cols%32, cols>0) |
| embed_gather_i4 (prefill.cu:38) | — | 4096 hardcoded (:29-31), **silent** | 0 | **BLOCKED at hidden 5120** — reads wrong strides; generalize (160 groups) or throw |

So: hidden 5120 OK, out_proj 5120x6144 OK, down 5120x17408 OK on gemv_v2_i4 / gemv2_q8_i4 /
v21_i4. The three real 27B blockers are `embed_gather_i4` (silent!), `ab2_q8_i4` (at least it
throws), and `ab_i4`'s baked rows=32. DecodeWorkspace buffers (decode.cu:14-26) are all
4096/12288-hardcoded too but are out of this audit's file scope.

## 8. Dim-guard inventory (throw vs silent)

Throwing (good): gemv_v2_i4 :70; gemv2_q8_i4 :151; ab2_q8_i4 :239; gemm_mlx_i4 :355;
gemm_v21_i4 :451-452 (also T>64); gemm_ab_i4 :543-544. Missing: `mxfp4_get_row_i4` :254-256 and
`embed_gather_i4` :38-40 launch with zero validation (silent truncation/garbage on bad cols).
In-kernel early `return` after the single `__syncthreads()` in gemv_v2_i4 :34 / gemv2_q8_i4 :115
is safe (no syncs follow). No launcher checks cudaGetLastError; gemv_v2_i4 with cols>253,952
would exceed the 99KB opt-in and fail semi-silently — irrelevant for real shapes, worth one
line of defense. gemv_v2_i4's `%1024` could be relaxed to `%32` (kernel is fully generic over
groups; smem formula adapts) if a future shape demands it.

## 9. Ranked fixes

1. **Replace `mxfp4_gemv2_q8_i4`/`ab2_q8_i4` activation-quant path with fp32-accum dual-row
   GEMV** (`gemv2_v2_i4`, new): removes the 0.66% noise class (SNR 152 -> ~1e5 rel; max_rel
   0.755 -> ~1e-5), removes 31040x per-block requant prologue + ~1.0GB L2 re-reads on lm_head
   pair, and is the faster design per AGENTS.md. Est: lm_head pair 1.33ms -> ~1.2ms, spec step
   −5-10%, plus greedy-parity with the prefill path. smem 32.8KB@4096 / 98.4KB@12288 (bf16
   staging fallback 49.2KB if down_proj pair occupancy hurts).
2. **27B enablement**: generalize `embed_gather_i4` to cols (or throw), parameterize `ab_i4`
   rows, verify 27B in_proj_a/b row count vs the baked 32; add the missing throws in
   `get_row_i4`/`embed_gather_i4`. Small, purely enabling.
3. **v21_i4/ab_i4 dequant**: `__hmul2` pair-LUT scale folding + 3-stage Bs (w3 §1.2/§3.1.5,
   still unimplemented): est −10-20% on T=64 GEMMs (27B down: ~200-280us est today -> ~150-180us);
   INSIG4's scale-idx=kb makes it the easiest variant to do first.
4. **byte_perm selectors 0xc480/0x4c80 -> 0x4440/0x4440** (§4): identical semantics, documented
   range, zero cost, immune to future ptxas/arch reinterpretation.
5. **Tail wait conservatism**: `kb+2<ksteps` -> `kb+1<ksteps` for `wait_prev` in all three
   pipelined GEMMs (§2) — regains one K-step of prefetch overlap at the tail. Trivial.
6. Cosmetic: explicit `__ldg` on mlx_i4 scales :330; smem-typed pointers in the lambdas to get
   LDS instead of generic LD.E in the GEMMs.

Bottom line: no correctness bugs found in the INSIG4 kernels — scale indexing, pipeline
sync, pair-kernel isolation, and embedding gathers are all right (and ab_i4 is bit-exact vs
v21_i4). The one real quality liability is the int8 activation quantization in the q8 pair path
(§5), which is also a perf liability; the 27B gaps are the three hardcoded kernels (§7).
