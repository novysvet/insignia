# Session 14: Q3 compute-for-bandwidth wave

Date: 2026-09-01  
Branch: `glm53-dflash2-4070ti-super`  
Measurement host: `glm-box-wsl`, RTX 4070 Ti SUPER, sm_89, CUDA 13.3  
Fixture: real block-3 IQ3_XXS gate/up and IQ4_XS down tensors from shard 1

## Outcome

This wave optimized the formats that actually dominate `UD-Q3_K_XL`, not the
artifact name. The central result is that redundant compute is profitable when
it removes an intermediate that would otherwise cross global memory, but only
until reuse across routed experts amortizes the original conversion.

The strongest exact decode changes are:

- fused IQ3 gate/up: one CTA serves both matrices and improves x1 by about
  7--13% depending on layout/timing regime; the dedicated seven-run WIM32 x1
  comparison was 14.473 -> 13.493 us (1.073x);
- fused SwiGLU quantization + IQ4 down: 16.033 -> 7.850 us (2.042x) at the
  r32 tiling;
- the equivalent Q6_K exception path: 13.098 -> 10.035 us (1.305x);
- the complete resident single-expert path with WIM32 pair + fused down:
  36.185 -> 28.985 us by seven-run median time (1.248x; conservative paired
  ratio median 1.199x);
- double fusion, where hidden Q8 quantization is recomputed inside each IQ3
  gate/up CTA and SwiGLU Q8 is recomputed inside each down CTA: the complete
  single-expert median reached 17.759 us. Its paired speedup median is 1.644x
  over the old path and 1.487x over the prior optimized path.
- exact pointer-table batching collapses the ordinary top-8 path from sixteen
  expert launches to one gate/up launch plus one ordered down launch. The
  first isolated seven-run median improved 229.069 -> 129.467 us (1.769x by
  median times, 1.773x paired-ratio median). A follow-up complete k sweep
  measured 235.509 -> 129.370 us at k=8 (1.820x by median times).

All fused-path comparisons are bit-exact against their unfused GPU controls:
MSE 0, relative L2 0, cosine 1.0000000000, and max absolute error 0.

## Hidden-Q8 recomputation crossover

The fused hidden quantizer copies the exact operation order of the standalone
Q8-per-32 kernel. Each CTA reads the 4096-wide FP32 hidden row, constructs 128
Q8 groups in 4.5 KiB shared memory, then reuses those values across a gate/up
row tile. It deliberately repeats that work across CTAs to remove the global
Q8 workspace write and rereads.

The row-tile sweep, seven-run medians, was:

| IQ3 gate/up x1 path | Time (us) | Speedup vs separate |
|---|---:|---:|
| separate hidden-Q8 + WIM32 | 20.555 | 1.000x |
| fused r2 | 14.928 | 1.377x |
| fused r4 | 13.558 | 1.516x |
| fused r8 | 12.556 | 1.637x |
| fused r16 | 14.148 | 1.453x |

The paired-ratio median independently selects r8 at 1.564x. The r8 kernel uses
60 registers, one barrier, 4,608 bytes shared memory, and no spills.

The actual MoE reuses one hidden quantization across multiple distinct routed
experts. Eight consecutive real expert slices were therefore made resident
and measured as one serialized routed group. The exact crossover is:

| Active experts | Shared-Q8 median (us) | Double-fused median (us) | Paired ratio median |
|---:|---:|---:|---:|
| 1 | 32.555 | 23.503 | 1.405x |
| 2 | 58.832 | 49.306 | 1.161x |
| 4 | 101.588 | 97.615 | 1.045x, noisy |
| 8 | 181.071 | 195.607 | 0.924x |

That compute-only experiment did not include the real weighted ordered down
accumulation. The subsequent exact end-to-end expert-group sweep supersedes
its dispatch conclusion:

| Executed experts | Serial shared-Q8 (us) | Batched ordered (us) | Double fused (us) | Winner |
|---:|---:|---:|---:|---|
| 1 | 41.198 | 41.036 | 30.986 | double fused |
| 2 | 66.562 | 42.335 | 58.787 | batched |
| 4 | 123.249 | 71.043 | 116.160 | batched |
| 8 | 235.509 | 129.370 | 229.629 | batched |

Production rule: use double fusion only for k=1. For k>=2, quantize the hidden
row once and use the pointer-table batched path. All gate, up, and canonical
ordered down controls are bit-exact: MSE 0, relative L2 0, cosine 1.0, max
absolute error 0.

## Exact ordered top-k batching

The decode router already names at most eight experts. Their resident device
pointers are copied into persistent pointer tables. The IQ3 launch uses
`grid.y=expert` and writes compact `[expert][2048]` gate/up rows. The IQ4 down
launch keeps one 4096-row output CTA resident while it walks experts in router
order, regenerates the exact 2048-wide SwiGLU Q8 row in shared memory, and
performs the same sequence of FP32 `fmaf` operations as the serialized
reference. This removes fourteen launches without reassociating the MoE sum.

On sm_89, batched gate/up uses 91 registers and no spills. Batched down uses
113 registers, one barrier, 2,304 bytes of shared memory, and no spills. The
batched k=8 timing range was only 129.05--129.85 us in the first campaign and
128.50--130.14 us in the crossover campaign, while the serialized path varied
by tens of microseconds as CPU submission gaps repeatedly drained the GPU.

## Prefill and exception formats

The 32-token IQ3 gate/up pair now shares one FP32-to-FP16 activation conversion
and one launch while preserving the two independent HMMA accumulator chains.
Seven-run medians improve 182.215 -> 101.346 us (1.798x by median times;
1.795x paired-ratio median). Both gate and up are bit-exact against separate
tensor-core launches: MSE 0, relative L2 0, cosine 1.0, max error 0. The pair
kernel uses 53 registers, one barrier, 4 KiB shared memory, and no spills.

The faster quality-gated arm quantizes each 32-value activation group once and
feeds the decoded IQ3 bytes directly to Ada's signed
`mma.sync.aligned.m16n8k32`. Integer dot products stay in registers; the kernel
then applies the independent row and token scales in FP32. Seven-run medians
improve the FP16 pair from 101.419 us to 60.012 us including activation
quantization (1.690x by median times; 1.689x paired-ratio median). IMMA compute
alone is 57.487 us. The kernel uses 47 registers, one barrier, 2,176 bytes of
shared memory, and no spills.

Against the independent FP64-accumulating CPU oracle, the IMMA gate result has
MSE 3.417566e-6, relative L2 0.006917408, cosine 0.9999760766, and max error
0.01302025. Up has MSE 3.551246e-6, relative L2 0.006927118, cosine
0.9999760152, and max error 0.009515196. This passes the existing per-matrix
2%/0.9998 gate, but full-model PPL/KL/JS and hard-prompt checks remain required
before making it the unconditional production default.

The routed IQ4_XS down projection now uses the same signed-IMMA strategy while
retaining its nonlinear codebook exactly. Seven-run medians improve FP16 decode
66.896 -> 40.981 us for the complete Q8-activation plus IMMA pipeline (1.632x
by median times; 1.634x paired-ratio median). IMMA compute alone is about
38.5 us. The kernel uses 54 registers, one barrier, 1,600 bytes of shared
memory, and no spills. Against the independent FP64 CPU oracle it measures MSE
1.581668e-6, relative L2 0.006984302, cosine 0.9999756095, and max error
0.008243836. This gives native signed-IMMA prefill coverage to IQ3 gate/up,
IQ4 down, and the existing Q6 exception rows; all live sparse-expert formats in
Q3_K_XL now have compute-for-bandwidth tensor-core paths.

A decoder-granularity sweep made each thread expand one, two, or four IQ3 code
pairs to reduce redundant metadata reads. All variants were bit-identical, but
seven-run pipeline medians were 64.189, 65.370, and 66.085 us respectively.
The generalized p1 code also lost against the original 60.012 us specialization.
The sweep was removed and the original 128-way decoder restored.

A more aggressive prefill arm regenerated clamped SwiGLU inside every IQ4
down-row CTA to remove the 256 KiB intermediate. It was also bit-exact, but
duplicating the sigmoid work across CTAs regressed the seven-run median
69.152 -> 83.369 us (20.6% slower). That kernel and API were removed. The
profitable boundary is activation conversion shared across two weight matrices,
not nonlinear activation recomputation across output-row tiles.

Q6_K support is complete for the routed down projections in blocks 11, 12,
and 44. The direct decode kernel sustains about 10.07 us / 683 GB/s on the real
block-11 tensor. The 32-token tensor-core path improves 130.337 -> 48.728 us
(2.675x) versus the Q8 pipeline. Its independent CPU-oracle metrics are MSE
2.273117e-9, relative L2 0.0002819293, cosine 0.9999999603, and max absolute
error 0.0002374053.

Replacing four scalar FP32-to-FP16 conversion stores with two packed half2
stores retained bit-identical output and improved HMMA medians by 1.19% for
IQ3, 1.47% for IQ4, and 0.26% for Q6.

## WIM32 result

The byte-neutral WIM32 layout stores the FP16 scale plane first, then three
coalesced 128-byte warp fields per four-block wave. Its proof predicted fewer
sector/address requests than the earlier scale/index/aux SoA layout. Hardware
timing is workload-specific:

- x1 fused gate/up: 14.473 -> 13.493 us (1.073x), promoted;
- x8 fused gate/up: approximately 0.4% slower, not promoted;
- single-matrix x1/x8: effectively flat.

The store/dispatch must therefore select WIM32 for the x1 fused pair and retain
the earlier byte-neutral layout for wider verification batches unless a later
whole-store design proves a better compromise.

## Rejected arms

- Two-warp IQ3 HMMA remained exact but regressed 90.65 -> 134.01 us (47.8%);
  it was reverted.
- WIM64 emitted the intended aligned `LDG.E.EF.64` loads, used 80 registers for
  the 4096-column kernel, and had no spills. It nevertheless regressed the
  fused-pair median 14.031 -> 14.307 us (about 2.0%) and was reverted.
- Three exact sign circuits were tested: PRMT carrier masks plus packed
  subtraction, carrier masks plus VNEG/select, and a negative 1 KiB codebook
  plus two LOP3s. All produced MSE 0/cosine 1 against the current decoder. The
  best isolated paired median was only 1.009x and the complete expert path
  regressed about 1.9%; all were reverted.
- The concrete restartable rANS/bitpack expert codec reduces bytes by only
  1.694% (`io_ratio=0.983057`) while scoring 137,033,404 bounded scalar-equivalent
  decode operations per 10.88 MiB record. It can also expand raw-fallback I/O
  by 0.79%. This is not a viable SSD-bandwidth trade on Ada.

## Production integration and effective-bandwidth results

The earlier integration boundary is closed.  Native IQ3/IQ4/Q6 expert records
now run end-to-end through the typed Q3 stager and production generator.  The
largest full-model gains came from moving fewer bytes, not from the isolated
kernel winner alone.

### Variable-size pinned tier

The old fixed record pitch wasted the large-layer capacity on every ordinary
expert.  The compact tier reserves complete 288-record working sets for the
three medium/large exception geometries and fills the rest with common
10.39-MiB records.  At the former 32-GiB default it holds 2,958 records
(2,382/288/288 small/medium/large) and reclaims 14,588 MiB of padding.  On the
272-token ArXivLean hard prompt, a 20-token decode pair improved 559.9 ->
439.2 ms/token and reduced decode O_DIRECT traffic 32.430 -> 27.783 GiB
(14.3%).  IDs and printed top-10 logits were identical.

The WSL/DXG ceiling was then probed directly.  Touched single allocations up
through 34,816 MiB succeeded.  The first 35,072-MiB attempt blocked in
`dxgvmb_send_sync_msg` with about 40 GiB resident instead of returning a clean
CUDA error; it was terminated and all memory was recovered.  Therefore 62 GiB
of guest RAM is not a 62-GiB CUDA-pinned budget.  The production default is now
33.5 GiB (34,304 MiB), 768 MiB below the observed stall point.  It holds 3,106
records.  The adjacent 34-GiB arm beat 32 GiB in both 40-token pairs
(369.4 vs 387.7 and 358.5 vs 406.5 ms/token), and the headroomed 33.5-GiB
validation was 359.8 ms/token with exact logits.  Roll back with
`INSIGNIA_GLM53_EXPERT_CACHE_MB=32768`.

### Pageable victim tier

An exact mmap-backed Q3 victim cache can use ordinary RAM behind the pinned
tier.  A correctness bug was found during the experiment: synchronous restores
set `done` but did not claim the new batch window, allowing a following miss to
evict/refill it before `upload()`.  Claiming restored windows immediately made
all differential top-10 checks exact.

The policy remains opt-in with `INSIGNIA_GLM53_Q3_PAGEABLE_CACHE_MB`.  A 16-GiB
tier with minimum two L1 hits saved 9.027 GiB of 100-token NVMe reads but copied
22.569 GiB through DRAM and measured 281.5 vs 280.2 ms/token.  Minimum one hit
saved 18.703 GiB but copied 50.230 GiB and measured 282.3 ms/token.  Neither raw
wall result justifies a default.  A real i7-14700KF copy microbenchmark found
31.76 GiB/s for glibc `memcpy` and 39.20 GiB/s for two-way AVX2 non-temporal
copying, but the maximum copy saving is too small to overturn the policy result.

### Decode top-k overlap

The isolated exact top-8 kernels do not automatically improve streamed decode.
All-at-once batching removes launches but also removes down-projection compute
that hides the next H2D.  Exact grouped down fusion (1/2/4/8) was integrated
behind `INSIGNIA_GLM53_Q3_TOPK_GROUP`.  Group two retained the complete
position/log hash
`7e592f130c01c841555d4909ea76d3f6bb4efb380b446d801d10dd4b79c95e22`,
but its three-run raw median was 395.2 ms/token versus 347.5 ms/token for the
scalar control under large SSD swings.  Group four measured 371.5 ms/token in
its single screen.  The production default therefore remains the pipelined
scalar expert path; grouped top-k stays opt-in for trace replay.

## Exact whole-layer Q3 prefill

`INSIGNIA_GLM53_PREFILL_FULL_LAYER_MAJOR=1` alone retained layer-major prompt
state but did not activate the separate whole-layer MoE executor.  Enabling the
executor deduplicates the complete layer's expert union, uploads each expert
once, and scatters results back in exact token/router order.  On the 938-token
ArXivLean prompt it reduced uploads 52,954 -> 10,902 and prompt time
53.481 -> 41.590 s (17.54 -> 22.55 prompt tok/s, 22.2% lower latency).  The
top-10 logits were digit-identical.

The measured length crossover is:

| Reported prompt | Conventional (s) | Whole-layer (s) | Latency change | Exact |
|---:|---:|---:|---:|:---:|
| 272 | 29.681 | 30.276 | +2.0% | yes |
| 320 | 32.330 | 30.960 | -4.2% | yes |
| 384 | 34.267 | 32.119 | -6.3% | yes |
| 512 | 38.323 | 35.509 | -7.3% | yes |
| 938 | 53.481 | 41.590 | -22.2% | yes |

The runner receives one fewer retained row than the CLI reports, so automatic
Q3 dispatch starts at 319 internal rows (a reported 320-token prompt).  Explicit
`INSIGNIA_GLM53_PREFILL_WHOLE_LAYER_MOE=0|1` overrides the decision.  Remote
no-override smokes confirmed `whole_moe=1` at 319 rows and `whole_moe=0` at
271 rows.  A forced 320-token full-vocabulary A/B was byte-identical: both
154,880-float dumps have SHA-256
`55eadc82e0cb5c7c83f7e9cf90ab312ae1a8e35b9997a0bd6fc56c6b2d693213`.
MSE, relative L2, centered MSE/L2, KL, reverse KL, JS, maximum error, and mean
error are all zero; raw and centered cosine are 1.0; top-1 is 1/1 and top-10
overlap is 10/10.

The approximate signed-IMMA prefill arm remains opt-in.  Once tested on the
actual whole-layer path, minimum-16 IMMA measured 42.105 s and minimum-32
measured 42.360 s versus 41.590 s exact, while both changed the final top-10.
The compute-only win is real, but I/O plus partial-tile overhead prevents an
end-to-end win on this workload.
