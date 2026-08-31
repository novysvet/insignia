# Session 12: table-free packed E2M1 on the 4070 Ti SUPER

Date: 2026-08-31

Branch: `glm53-dflash2-4070ti-super`

Commits under test: `c23081c`, `a005f34`, and `201a14d`

## Decision

Keep table-free E2M1 arithmetic enabled by default for direct XPR1-v2 expert
execution on glm-box.  Set `INSIGNIA_GLM53_NVFP4_TABLEFREE=0` for the exact
shared-LUT rollback.

This supersedes the local 4070 SUPER rejection in
`audits/s11-nvfp4-direct-execution.md`.  The production specialization and
dispatch were retested on the actual RTX 4070 Ti SUPER target.  No timing from
the local 4070 SUPER is part of this decision.

## What changed

The packed expert store, gate/up pair, and weighted-down kernels can now decode
each E2M1 nibble using packed integer arithmetic instead of loading a 2 KiB
shared-memory lookup table.  The arithmetic produces exactly the same signed
coefficient bytes consumed by DP4A.  Compile-time specialization removes the
unused table and its initialization/barrier work from the table-free arm.

The runtime switch reaches every packed direct execution site: scalar sparse
MoE, MTP MoE, multi-row verify/prefill, and full-prompt layer-major prefill.
Both 2048x4096 gate/up geometry and 4096x2048 down geometry are covered for all
active multiplicities 1 through 8 and both four- and eight-warp CTAs.

The Ti-specific gate/up pair dispatch is:

| multiplicity | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTA warps | 8 | 8 | 8 | 8 | 4 | 8 | 8 | 4 |

Serialized shape sweeps consistently selected four warps at multiplicity 5
(about 18% over eight warps) and multiplicity 8 (about 6--8%).

## Exactness gates

`/var/tmp/insignia-build/glm53-expert-bench
/var/lib/insignia/l3e0-gate.ig53` passed on glm-box:

- exhaustive E2M1 code gate: exact;
- packed store, pair, and accumulate, B=1..8, CTA4 and CTA8: byte-exact;
- packed down store and accumulate, B=1..8, CTA4 and CTA8: byte-exact;
- the ordinary numerical benchmark is unchanged: MSE `3.231e-5`, relative L2
  `0.004427`, cosine `0.999990218` against the float-decode reference.  These
  are the pre-existing activation-Q8 errors; LUT versus table-free is exact.

The uncombined fixture kernel measured 12.802 us for the LUT DP4A GEMV and
7.506 us for table-free arithmetic (368.6 versus 628.7 GB/s).  The production
packed path has shape-dependent gains because packed-scale decode and the
matrix reductions remain unchanged.  Resource inspection of the representative
packed store specialization showed the E2M1 shared plane falling from 2 KiB to
zero (3,072 to 1,024 bytes total shared memory), with the same register count
and no spills.

The whole-model 30-token full-vocabulary dumps were byte-identical:

```
SHA256  d0d1f16eef14f925efde75e4c28935e527355b16a792981fd8fd106a530d1a6a
bytes   18585600 per arm (30 x 154880 x FP32)
IDs     220 98546 24 13 171 105 224 198, then 200 through token 30
```

Thus the optimization adds zero MSE, zero KL/JS, zero PPL change, and zero
greedy/top-logit change relative to the LUT arm.

## Box-only end-to-end A/B

The timing harness deliberately used fixed allocations so every accepted run
reported exactly 1,212 pinned host records and 158 packed VRAM slots:

```
INSIGNIA_GLM53_EXPERT_CACHE_MB=16384
INSIGNIA_GLM53_EXPERT_VRAM_MB=2048
INSIGNIA_GLM53_DFLASH2=0
```

Each process started only after `nvidia-smi` returned to 287 MiB used / 15,775
MiB free.  This avoids WSL's delayed CUDA cleanup, which otherwise caused
`cudaHostAlloc` failure or allocator halving.  Five valid runs per arm were
collected; all ten had the same ledger and token sequence.

| arm | decode seconds for 30 tokens | median | median tok/s |
|---|---|---:|---:|
| shared LUT | 13.539, 13.104, 13.721, 14.315, 12.971 | 13.539 | 2.216 |
| table-free | 13.190, 13.381, 13.203, 13.556, 13.531 | 13.381 | 2.242 |

Table-free reduces median decode time by 1.17% and raises median throughput by
1.18%.  The three-point trimmed-mean time gain is 0.62%.  This modest
whole-engine result is expected: the harness still spends most decode time
waiting for O_DIRECT expert reads, and observed NVMe rate varied from about
3.6 to 4.4 GB/s.  The isolated-kernel result establishes the direction; the
end-to-end result establishes that the saved compute survives the hierarchy.

Median five-token prompt time was 4.589 s for LUT and 4.432 s for table-free.
That 3.42% difference is recorded but not promoted as a general prefill claim:
startup cache loading and NVMe variance are a large part of such a short
prompt.

## Invalid runs and benchmark hygiene

Runs with different allocation ledgers were excluded before examining their
timings.  In particular, a 32 GiB production pair produced 401 versus 200 VRAM
slots, and an immediate 3 GiB pair produced 238 versus 119 slots, because the
previous Windows/CUDA context had not released its resources.  Startup failures
with no model output were also excluded.  These are infrastructure artifacts,
not performance samples.

The Q3_K_XL download was not active during this campaign.  It was restarted
only after the last accepted timing as the persistent WSL unit
`insignia-q3-k-xl-download.service`.  Future performance measurements must stop
that unit or wait for completion.

