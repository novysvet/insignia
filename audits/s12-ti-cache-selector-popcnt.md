# Session 12: Raptor Lake POPCNT cache-union selector

Date: 2026-08-31

Branch: `glm53-dflash2-4070ti-super`

Implementation commit: `04bdab3`

## Decision

Keep the adaptive 288-bit POPCNT objective enabled for two- through four-row
joint cache-aware DFlash selection.  Keep the original byte-array objective for
one-row selection, where mask construction costs more than it saves.

`INSIGNIA_GLM53_DF_CACHE_MASK_SEARCH=0` is the byte-array rollback.
`INSIGNIA_GLM53_DF_CACHE_MASK_VERIFY=1` runs the optimized objective and the
byte reference over the same live residency snapshot and aborts on any cost
disagreement.

All performance measurements in this audit were made on glm-box's i7-14700KF.
No local-box timing is included.

## Mechanism

The old exact Cartesian selector evaluated up to `8^4 = 4096` assignments per
sparse layer.  At every leaf it cleared 288 bytes, walked every selected expert,
deduplicated with byte tests, and accumulated union, NVMe, and H2D costs.

The new objective represents the 288 experts in five 64-bit words.  Each row
action has a prebuilt mask.  A leaf ORs the row masks and computes:

```
union = popcount(mask)
disk  = popcount(mask & disk_miss_mask)
h2d   = popcount(mask & h2d_miss_mask)
```

The generator is compiled with `-march=raptorlake`; `_mm_popcnt_u64` therefore
maps directly to the verified POPCNT ISA rather than a runtime-dispatched
fallback.  Router regret and all tie-break fields retain their original
summation and comparison order.

## Exactness

The in-process shadow verifier was run on ArXivLean problem 16:

- 272-token hard prompt;
- 16 generated tokens;
- four DFlash k4 rounds, 4.00 accepted per round;
- every live multi-row Cartesian objective recomputed by both implementations
  from the identical expert residency state;
- no disk, H2D, union, regret, substitution, or selected-action disagreement;
- coherent output IDs:
  `1654 1184 311 12112 25 369 1449 7546 400 77 1124 709 80 220 16 54509`.

Separate cache-aware processes are not a valid full-logit equivalence test:
minor asynchronous differences in which experts reach the VRAM tier can change
the cache-aware action itself.  The shadow mode exists specifically to compare
the two objectives on one immutable snapshot.

## Timing

The default adaptive sequential/batch verifier used POPCNT for 42 of 588 layer
groups and correctly retained the byte path for the other 546 one-row groups:

| selector | total | mean per layer group |
|---|---:|---:|
| byte-only | 2.062 ms | 3.506 us |
| adaptive POPCNT | 1.886 ms | 3.208 us |

This is an 8.5% selector-time reduction in the mixed workload.

A forced-batch A/B exercised the Cartesian mask path for all 294 verified
layer groups:

| selector | total | mean per layer group |
|---|---:|---:|
| byte-array | 4.369 ms | 14.862 us |
| POPCNT | 1.875 ms | 6.377 us |

POPCNT reduces the intended multi-row selector cost by 57.1%, saving 2.494 ms
over the 294 calls.  Both arms generated the same 16 IDs and the same eight
round acceptance histogram (`0:1 2:6 3:1`, 1.88 accepted/round).

Whole decode measured 548.3 versus 543.6 ms/token in that one pair.  This audit
does not claim the noisy 0.86% wall difference: the measured selector saving is
only about 0.16 ms/token and the run streams hundreds of GiB through variable
WSL/NVMe I/O.  The accepted claim is the direct selector timer, not the wall.

## Deployment profile

The OpenAI-compatible web API now defaults to the already quality-green
`top6-cache` profile:

- fixed Top-6 target verification;
- K=32 cache-aware router frontier;
- normalized router regret 0.001;
- eight joint actions per row;
- no hardcoded 5 GiB expert-cache override, so glm-box uses its measured 32 GiB
  engine default.

`INSIGNIA_WEB_SPEED_PROFILE=exact` restores exact Top-8 verification.
`INSIGNIA_WEB_EXPERT_CACHE_MB` explicitly overrides the host tier.  The remote
self-test passed.  The performance evidence for this deployment remains the
existing ArXivLean-40 result: 1.678 to 2.491 tok/s (+48.4%) with PPL +0.86% and
coherent hard output; see `audits/s10-matharena-arxivlean.md`.

