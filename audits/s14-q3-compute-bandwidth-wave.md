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

Production rule: use double fusion only when the active expert count is at
most two. Keep one shared hidden-Q8 conversion for exact top-8. k=4 remains an
experimental arm because three of seven paired trials lost despite the median
gain. This makes the fused path useful to adaptive-k/mass-pruned execution
without regressing exact routing.

## Prefill and exception formats

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

## Integration boundary

These kernels and the real-tensor harness are tracked and pushed. The current
production generator still consumes the old NVFP4 compact-record schema, so
Q3 full-model throughput is not yet claimable. The next integration unit is a
typed GGUF expert index/stager with separate IQ3/IQ4/Q6 record geometry, WIM32
sidecars for the winning dispatch, and an active-expert-count dispatch between
shared-Q8 and double fusion. Dense/shared Q8_0 handling must follow before a
full MathArena/GSM teacher-forced PPL, cosine, KL/JS, and throughput campaign.

