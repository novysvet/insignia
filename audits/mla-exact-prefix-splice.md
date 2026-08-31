# MLA cross-head FP8 exact-prefix splice

Date: 2026-08-30  
Branch: `codex/glm53-dflash2-4070-super`  
Device: local RTX 4070 SUPER, sm_89, CUDA 13.3

## Decision

The long-context cross-head path previously read the group-64 FP8 shadow from
key zero, even though the runner retains compressed latents 0..255 in FP32.
Both decode and fused prefill now replay that FP32 prefix before the FP8
suffix.

The accepted decode path is an **ordered-partial FP32 splice**. It stages each
16-key FP32 prefix chunk once for eight heads, computes FP32 scores and values,
and writes 16 partials per head. A second kernel merges those chunks in key
order with the existing H8 tile-0 suffix partial. The existing outer 512-key
partial layout and outer merge are unchanged.

This arm is not called bit-exact: splitting online softmax into 16-key
summaries and changing the FP32 dot reduction reassociates arithmetic. It is
the default subpath when `INSIGNIA_GLM53_MLA_CROSS_HEAD_FP8=1` because it passed
all focused quality/hard-output gates and reduces the 8192 decode penalty from
about +29% to +4.10%. While that cross-head arm is enabled, set
`INSIGNIA_GLM53_MLA_PREFIX_PARALLEL=0` to select the scalar FP32 diagnostic.
Engine-wide MLA does not enable cross-head FP8.

The scalar path is deliberately named a **diagnostic**, not a reference. It
evaluates prefix rows 0..255 in FP32, but it also evaluates suffix rows
256..511 with scalar dequant/FMA instead of the shipping H8 MMA tile-0 path.
The fast arm preserves H8 arithmetic for that suffix half-tile.

No BF16 or TF32 score path is present in the accepted code.

## Decode design

For the fast subpath within the opt-in cross-head arm:

1. The existing absorb launch emits FP8 `q_eff`, group scales, and an FP32
   `q_eff` side output.
2. `mla_decode_exact_prefix_partial_kernel_v2` launches a 16 x 8 CTA grid.
   Each CTA stages 16 x 512 FP32 latent values once; eight warps compute one
   head each. It retains key order inside each 16-key online-softmax partial.
3. The existing cross-head H8 kernel evaluates tile 0 starting at key 256 and
   every subsequent 512-key tile without changing the partial ABI.
4. `mla_merge_exact_prefix_tile0_kernel` merges the 16 prefix summaries in
   ascending key order, then the H8 rows-256..511 summary, into the normal
   tile-0 slot.
5. The shipping outer-tile merge runs unchanged.

An early implementation used one CTA for eight heads over all 256 prefix keys.
It staged the prefix only once, but launched just eight CTAs on a 48-SM GPU and
barely improved 8192 decode (0.6738 vs 0.6805 ms scalar diagnostic). It was
removed. A BF16-MMA score prototype was also removed: on the earlier sentinel
fixture it reached 0.6415 ms at 8192, but was still +24.24% over no-splice and
was only approximate (seam rel-L2 3.16e-5; 8192 rel-L2 8.28e-4, KL 1.94e-7,
synthetic PPL +0.04228%). The accepted ordered-partial arm is both faster and
FP32-only, so a TF32 arm was not pursued.

The first ordered-partial prototype exposed an intermittent race: thread zero
overwrote the tile-0 max/denominator before other threads had loaded the H8
suffix header. The merge now snapshots both fields in shared memory and
synchronizes before any tile-0 write. Three final independent runs plus the
post-policy run produced identical quality metrics and passed every context.

## Prefill and production dispatch fixes

Fused prefill replays the 256 FP32 keys inside its persistent H4 x Q8 CTA, then
starts the FP8 suffix at key 256. It reuses dead suffix staging storage for
FP32 `q_eff`; dynamic shared memory remains 60,416 bytes.

Two production-dispatch holes found during review are fixed:

- DFlash verify chunks of at most eight rows previously bypassed the splice at
  contexts 256..4095 because only `tokens >= 16 || position >= 4096` selected
  fused cross-head prefill. Production now uses repeated ordered-partial decode
  for every chunk below 16 rows, at every context. Direct warmed A/B timings
  show it beats fused exact-prefix prefill in every measured 1/8-row case
  through 8192. Chunks of 16 or more rows use fused prefill.
- A legal non-divisor chunk such as 192..287 previously skipped the entire
  FP32 sidecar copy because its end exceeded 256, leaving rows 192..255
  uninitialized. `mla_multi` now computes the clipped overlap
  `min(tokens, max(0, 256 - position_base))`, executes/saves that exact prefix
  subrange, offsets all pointers, and dispatches only the post-256 suffix. This
  also avoids replaying future keys for the queries below 256.

The 16-row fused-prefill splice itself was not made faster. Its final repeated
median is 2.5104 ms versus 2.2014 ms no-splice (about +14.0%). The quality gate
passes, but reducing that overhead remains future work. DFlash's 1/8-row hot
path no longer pays it.

## Code and memory surface

- `include/insignia_glm53.cuh`: optional exact-prefix decode/prefill arguments,
  the 256-row constant, and shared overlap/dispatch policy helpers.
- `src/glm53_ops.cu`: FP32 `q_eff`, scalar diagnostic, ordered 16-key prefix
  partials, ordered tile-0 merge, H8 tile-0 skip, and fused-prefill replay.
- `src/glm53_generate.cu`: MLA-only runner hunks retain the exact sidecar,
  allocate scratch and default to the fast subpath only after cross-head FP8 is
  enabled, split crossing chunks, and route sub-16-row verifies through
  repeated spliced decode. This file also contains unrelated concurrent
  worktree changes; only the MLA hunks are part of this result.
- `tests/test_mla_exact_prefix_splice.cu`: deterministic GPU oracle, all metric
  gates, pointer validation, repeated timings, DFlash dispatch A/B, and the
  192+96 crossing regression.
- `audits/mla-exact-prefix-splice.md`: this record.

Production memory added by the complete splice is:

- 11 x 256 x 512 FP32 prefix sidecars: 5.50 MiB when reconstruction did not
  already own them;
- reusable 128 x 64 x 512 FP32 `q_eff`: 16.00 MiB;
- fast-subpath scratch `[64,16,514]` FP32: 2.008 MiB.

## API and ABI behavior

Decode appends these arguments before `cudaStream_t`:

```cpp
const float *exact_prefix = nullptr,
float *qeff_f32 = nullptr,
float *exact_prefix_partial = nullptr,
bool parallel_exact_prefix = false,
cudaStream_t stream = nullptr
```

Prefill appends paired `exact_prefix`/`qeff_f32` before its stream. The C++
mangled symbols therefore changed and binaries must be rebuilt. Source callers
that use defaults still compile. Callers that passed an explicit stream
positionally must insert the new optional arguments first.

The low-level null/no-splice call remains numerically and structurally
unchanged. Validation fails closed:

- `exact_prefix` and `qeff_f32` must be paired;
- `parallel_exact_prefix` must equal the presence of
  `exact_prefix_partial`;
- parallel mode requires the exact-prefix pair;
- scalar diagnostic mode requires null fast scratch.

The low-level boolean default remains false so a legacy null/no-splice call is
valid. When `MLA_CROSS_HEAD_FP8=1`, the production runner explicitly passes
true and allocates the required scratch; the environment value `0` is the
diagnostic opt-out. Otherwise this splice is not allocated or called.

## Focused test and timing method

Final build command:

```text
rtk proxy wsl -d Arch -- bash -lc '/opt/cuda/bin/nvcc -ccbin /usr/bin/g++-15 -arch=sm_89 -O3 --use_fast_math -lineinfo -Xptxas=-v -std=c++20 -I/mnt/e/coding/Insignia/include /mnt/e/coding/Insignia/src/glm53_ops.cu /mnt/e/coding/Insignia/tests/test_mla_exact_prefix_splice.cu -o /var/tmp/test-mla-exact-prefix-splice-final2'
```

The binary was run three independent times:

```text
rtk proxy wsl -d Arch -- bash -lc '/var/tmp/test-mla-exact-prefix-splice-final2'
```

All three ended with:

```text
pointer-pairing fail-closed=PASS
PASS exact-prefix decode and prefill splice
```

Each reported process timing is the median of seven CUDA-event samples. Every
sample has three warmups and averages 30 decode calls, 20 prefill calls, or 10
dispatch calls. Tables below take the median of the three process medians.

The fixture has nonzero rows throughout 0..8191, distributed query-aligned
keys on both sides of 256, and a fixed output-class margin so the hard argmax
gate is not a near-tie lottery. It does not use the old key-zero-dominant
sentinel.

## Decode medians

| Context | No splice ms | Scalar diagnostic ms | Ordered FP32 ms | Fast vs diagnostic | Fast vs no splice |
|---:|---:|---:|---:|---:|---:|
| 257 | 0.2675 | 0.2660 | 0.2207 | -17.03% | -17.50% |
| 512 | 0.3327 | 0.3551 | 0.2898 | -18.39% | -12.89% |
| 1024 | 0.3327 | 0.4850 | 0.3526 | -27.30% | +5.98% |
| 2048 | 0.3334 | 0.4851 | 0.3531 | -27.21% | +5.91% |
| 4096 | 0.4030 | 0.4863 | 0.4249 | -12.63% | +5.43% |
| 8192 | 0.5124 | 0.6703 | 0.5334 | -20.42% | +4.10% |

## Decode quality versus the full-FP32 latent oracle

The full-FP32 oracle is stricter than the effective hybrid model because the
accepted arm intentionally keeps the FP8 suffix.

| Context | Arm | MSE | rel-L2 | cosine | KL | JS | synthetic PPL delta | top-1 mismatches |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 257 | no splice | 6.355e-8 | 7.736e-3 | 0.999970134 | 3.137e-8 | 7.888e-9 | -0.03234% | 0 |
| 257 | scalar diagnostic | 4.087e-10 | 6.204e-4 | 0.999999808 | 3.951e-10 | 5.089e-11 | -0.00004% | 0 |
| 257 | ordered FP32 | 4.086e-10 | 6.204e-4 | 0.999999808 | 1.835e-10 | 5.089e-11 | -0.00004% | 0 |
| 512 | no splice | 3.721e-8 | 5.917e-3 | 0.999982570 | 1.874e-8 | 4.671e-9 | -0.02830% | 0 |
| 512 | scalar diagnostic | 2.218e-8 | 4.569e-3 | 0.999989590 | 1.124e-8 | 2.760e-9 | -0.01209% | 0 |
| 512 | ordered FP32 | 2.218e-8 | 4.569e-3 | 0.999989591 | 1.082e-8 | 2.760e-9 | -0.01209% | 0 |
| 1024 | no splice | 1.981e-8 | 4.316e-3 | 0.999990696 | 1.008e-8 | 2.471e-9 | -0.00965% | 0 |
| 1024 | scalar diagnostic | 1.534e-8 | 3.798e-3 | 0.999992790 | 7.983e-9 | 1.911e-9 | -0.00154% | 0 |
| 1024 | ordered FP32 | 1.534e-8 | 3.798e-3 | 0.999992790 | 7.841e-9 | 1.911e-9 | -0.00154% | 0 |
| 2048 | no splice | 1.064e-8 | 3.162e-3 | 0.999995026 | 5.507e-9 | 1.332e-9 | -0.01249% | 0 |
| 2048 | scalar diagnostic | 8.967e-9 | 2.903e-3 | 0.999995806 | 4.517e-9 | 1.119e-9 | -0.00843% | 0 |
| 2048 | ordered FP32 | 8.966e-9 | 2.902e-3 | 0.999995806 | 4.591e-9 | 1.119e-9 | -0.00843% | 0 |
| 4096 | no splice | 7.299e-9 | 2.619e-3 | 0.999996595 | 3.632e-9 | 9.130e-10 | -0.00911% | 0 |
| 4096 | scalar diagnostic | 6.722e-9 | 2.513e-3 | 0.999996862 | 3.524e-9 | 8.395e-10 | -0.00709% | 0 |
| 4096 | ordered FP32 | 6.722e-9 | 2.513e-3 | 0.999996862 | 3.446e-9 | 8.395e-10 | -0.00709% | 0 |
| 8192 | no splice | 5.313e-9 | 2.234e-3 | 0.999997507 | 2.783e-9 | 6.631e-10 | -0.00218% | 0 |
| 8192 | scalar diagnostic | 5.097e-9 | 2.188e-3 | 0.999997608 | 2.737e-9 | 6.358e-10 | -0.00117% | 0 |
| 8192 | ordered FP32 | 5.097e-9 | 2.188e-3 | 0.999997608 | 2.538e-9 | 6.358e-10 | -0.00117% | 0 |

Every accepted/diagnostic row passed explicit MSE, rel-L2, cosine, KL, JS,
PPL, and top-1 assertions. The broad full-FP32-vs-FP8-suffix limits are MSE
<1e-7, rel-L2 <2.5e-2, cosine >0.9997, KL <1e-7, JS <3e-8, synthetic PPL
<+3.5%, and zero top-1 mismatches. Seam gates are tighter.

## Ordered arm versus scalar diagnostic

This second gate isolates arm-to-arm drift so the broader FP8-suffix limits
cannot hide a reassociation failure.

| Context | MSE | rel-L2 | cosine | KL | JS | synthetic PPL delta | top-1 mismatches |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 257 | 1.814e-16 | 4.134e-7 | 1.000000000000 | 0 | 3.143e-17 | -0.00000% | 0 |
| 512 | 5.343e-14 | 7.089e-6 | 0.999999999976 | 0 | 6.677e-15 | +0.00000% | 0 |
| 1024 | 1.335e-14 | 3.542e-6 | 0.999999999994 | 0 | 1.656e-15 | -0.00000% | 0 |
| 2048 | 3.339e-15 | 1.771e-6 | 0.999999999998 | 7.499e-11 | 3.898e-16 | -0.00000% | 0 |
| 4096 | 8.405e-16 | 8.886e-7 | 1.000000000000 | 0 | 7.154e-17 | -0.00000% | 0 |
| 8192 | 2.122e-16 | 4.465e-7 | 1.000000000000 | 0 | 0 | -0.00000% | 0 |

The tight limits are MSE <1e-10, rel-L2 <1e-4, cosine >0.99999999, KL
<1e-9, JS <3e-10, synthetic PPL <+3.5%, and zero top-1 mismatches.

## Prefill, crossing, and DFlash dispatch

Fused 16-row prefill versus full FP32:

| Case | MSE | rel-L2 | cosine | KL | JS | synthetic PPL delta | top-1 mismatches |
|:---|---:|---:|---:|---:|---:|---:|---:|
| prefill16@256 | 3.526e-9 | 1.826e-3 | 0.999998334 | 1.817e-9 | 4.410e-10 | -0.00502% | 0 |
| crossing 192+96, suffix 256..287 | 5.317e-9 | 2.244e-3 | 0.999997493 | 2.659e-9 | 6.662e-10 | -0.00867% | 0 |

DFlash-size warmed medians:

| Context | Rows | Repeated ordered decode ms | Fused prefill ms | Repeated delta |
|---:|---:|---:|---:|---:|
| 257 | 1 | 0.2191 | 0.8601 | -74.53% |
| 257 | 8 | 1.7956 | 2.3301 | -22.94% |
| 512 | 1 | 0.2789 | 1.0772 | -74.11% |
| 512 | 8 | 2.2685 | 2.5715 | -11.78% |
| 1024 | 1 | 0.3416 | 1.5251 | -77.60% |
| 1024 | 8 | 2.8083 | 3.0402 | -7.63% |
| 2048 | 1 | 0.3414 | 2.4181 | -85.88% |
| 2048 | 8 | 2.8160 | 4.0078 | -29.74% |
| 4096 | 1 | 0.4137 | 4.2605 | -90.29% |
| 4096 | 8 | 3.3950 | 5.9370 | -42.82% |
| 8185 | 1 | 0.5191 | 7.8996 | -93.43% |
| 8185 | 8 | 4.2856 | 9.7727 | -56.15% |
| 8192 | 1 | 0.5211 | 7.9497 | -93.45% |

Both dispatch candidates passed the full-FP32 quality gates. Their direct
repeated-vs-fused tight comparison had worst MSE 3.465e-15, rel-L2 1.807e-6,
cosine 0.999999999998, KL 7.581e-11, JS 6.513e-16, synthetic PPL delta
+0.00001%, and zero top-1 mismatches.

## ptxas and compile checks

Final `ptxas -v` resources, all with zero spill loads/stores:

- ordered 16-key FP32 partial: 128 registers, 512 bytes static plus 32,768
  bytes dynamic shared memory;
- ordered tile-0 merge: 40 registers, 16 bytes shared memory;
- scalar diagnostic: 32 registers, 112 bytes shared memory;
- H8 suffix partial: 105 registers, 41,216 bytes shared memory;
- FP32/FP8 `q_eff`: 29 registers, 3,104 bytes shared memory;
- fused prefill: 118 registers, 128-byte stack, 60,416 bytes dynamic shared
  memory, zero spills.

The final production translation unit compiled after the seam/dispatch changes:

```text
rtk proxy wsl -d Arch -- bash -lc '/opt/cuda/bin/nvcc -ccbin /usr/bin/g++-15 -arch=sm_89 -O3 --use_fast_math -lineinfo -Xcompiler=-pthread -Xcompiler=-march=znver3 -Xcompiler=-mtune=znver3 -std=c++20 -DINSIGNIA_GLM53_NO_MAIN -I/mnt/e/coding/Insignia/include -c /mnt/e/coding/Insignia/src/glm53_generate.cu -o /var/tmp/glm53-generate-mla-prefix-final2.o'
```

Result: exit 0; only three pre-existing unused-method warnings. A full
`glm53-generate` link with all production translation units also succeeded
before the final dispatcher-only edit; the final generation TU and final ops
plus harness were rebuilt afterward.

The existing null-path cross-head decode and prefill harnesses were rebuilt
earlier in this focused session and passed all boundary cases. The final
splice harness additionally executes the null/no-splice path at every reported
decode context, so the final source is covered directly.

## Unrelated legacy-benchmark OOM caveat

An earlier broad `build/glm53.sh` invocation successfully compiled its targets
and smoke test, then the legacy monolithic `glm53-ops-bench` allocated expanded
MLA caches at the production maximum even for context-8/16 tests. That exceeded
the 14-GiB WSL guest cap and OOM-killed the distro. This happened outside the
optional cross-head splice kernels.

Root subsequently fixed `src/glm53_ops_bench.cu` to allocate only the exercised
rows. That file is not part of this MLA change, and no broad campaign was rerun.
All final commands above are focused harness/translation-unit builds and passed.

## Limitations

- “Exact prefix” means FP32 compressed-latent history and FP32 `q_eff` are
  replayed. It does not promise the historical expanded-K/V reduction order.
- The ordered arm is FP32 but reassociated; the tight direct gate quantifies
  that difference. It is not digit-parity.
- Rows 256+ intentionally remain group-64 FP8 with the existing H8 MMA path.
- Reported PPL is a deterministic synthetic 256-class, per-head proxy whose
  target is the full-FP32 fixture argmax. It is not full-vocabulary GLM PPL,
  MathArena teacher forcing, greedy sequence parity, or DFlash acceptance.
- No whole-checkpoint or full ABCD campaign was run. Engine-wide default-on
  promotion still needs the repository's normal checkpoint-level token/logit
  parity and full-vocabulary quality gates before a release claim.
