# Session 9 — exact compute-for-bandwidth reclaims and focused validation

Date: 2026-08-29  
Target: glm-box, RTX 4070 Ti SUPER, sm_89, i7-14700KF, single NVMe  
Branch: `glm53-dflash2-4070ti-super`

## Outcome

This session converted two large duplicate/idle allocations into expert-tier
capacity without changing the model arithmetic:

1. MLA absorbed weights are reconstructed on consumption from the resident
   E4M3 + FP16-scale `kv_b_proj` cache. This removes the 704 MiB FP32
   `W_uk`/`W_uv` duplicate and deliberately spends otherwise-idle Ada ALU to
   increase effective model bandwidth.
2. A process forced to DFlash row-sequential verification no longer allocates
   145.6 MiB of KDA/conv rollback snapshots that it can never read.

Together with the low-risk Task-4 ledger changes (lazy Q8 stream slots, compact
34-row KDA state/history, and lazy single-expert fallback scratch), the live
VRAM expert tier moved as follows:

| Configuration | Expert slots | Change from old baseline |
|---|---:|---:|
| old scalar | 316 | — |
| old DFlash | 211 | — |
| compact MLA + low-risk reclaims, scalar | 383 | +67 (+21.2%) |
| compact MLA + low-risk reclaims, DFlash | 281 | +70 (+33.2%) |
| above + forced sequential snapshot elision | 292 | +81 (+38.4%) |

The final 292-slot glm-box run used the packed-v2 store, F3, O(1) host LRU,
SLRU, fixed k7, full dense FP8 residency, and the 32 GiB pinned tier.

## Exact compact MLA absorb

`INSIGNIA_GLM53_MLA_FP8_ABSORB` is now auto-on. Explicit `=0` retains the
materialized FP32 A/B path. It falls back automatically unless all required
`kv_b_proj` tensors have stable resident E4M3 weights and FP16 scales. BF16
absorb and MLA dump diagnostics retain their prior materialized contract.

The sm_89 kernels reconstruct each coefficient with the exact E4M3 bit decode,
FP16 widening, and `mul.rn.f32`, then preserve the old `fmaf` order. Validation:

- prefix 8: 0/16,384 decode mismatches and 0/131,072 prefill mismatches;
- prefix 520: zero mismatches;
- prefix 4096: zero mismatches;
- local real-model default smoke reproduced top-10 logits exactly and emitted
  IDs `220 98546`;
- four paired GSM8K/MATH-500 cases reproduced scalar/DFlash IDs exactly;
- final glm-box forced-sequential run reproduced IDs
  `1986 374 264 4285` and the known first-step top-10 digits.

Kernel timing on the local 4070 SUPER:

| Prefix | Path | Materialized | On-consumption | Ratio |
|---:|---|---:|---:|---:|
| 8 | decode, one MLA layer | 0.2849 ms | 0.2526 ms | 1.128x |
| 8 | prefill | 2.0288 ms | 1.9196 ms | 1.057x |
| 520 | decode | 0.5300 ms | 0.4470 ms | 1.186x |
| 4096 | decode | 0.9860 ms | 0.9858 ms | neutral |
| 4096 | prefill-8 | 7.3378 ms | 6.6176 ms | 1.109x |

`ptxas` reports 34 registers and no spills for compact partial decode versus
44/no-spill materialized. Compact prefill uses 48 registers and a tiny stack;
the old prefill kernel uses 80 registers and spills heavily. Reconstruction
therefore did not create the expected register-pressure penalty.

### Focused production A/B

The long factorial was stopped after the only necessary comparison. Each arm
used the packed-v2 + F3 + O(1) LRU + SLRU stack, four real math prompts, 32
generated tokens, and fixed k7.

| Arm | Scalar median | DFlash median | Parity |
|---|---:|---:|---|
| materialized absorb | 515.0 ms/token | 544.0 ms/token | 4/4 |
| compact absorb | 500.4 ms/token | 539.6 ms/token | 4/4 |

Run-order variance was large on the two MATH scalar cases. The more robust
median of paired per-prompt ratios is a 0.7% scalar improvement and a 0.8%
DFlash improvement. Claim only a small positive wall gain. The decisive result
is the exact 704 MiB reclaim and 67–70 added slots, whose value grows with a
longer cache-warm decode horizon.

## Forced-sequential snapshot elision

When `INSIGNIA_GLM53_DF_SEQ_VERIFY` is present and DFlash is active, the process
cannot enter batch verification. The constructor omits the compact 136 MiB KDA
state snapshot and 9.5625 MiB conv snapshot. Runtime guards hard-fail if a
future control-flow edit attempts a snapshot or rollback under this contract.
Adaptive, batch, and MTP modes are unchanged.

Local GSM8K row 895 exercised a verified block after the allocation removal and
reproduced exact IDs `1986 374 264 4285`. The final glm-box run printed the
145.6 MiB reclaim banner, exposed 292 expert slots, accepted histogram
`0:1 3:1`, and reproduced the same exact IDs.

## Full-prompt layer-major prefill

Task 1's exact full-prompt layer-major proposal was implemented behind
`INSIGNIA_GLM53_PREFILL_FULL_LAYER_MAJOR=1`, with host and VRAM state stores.
It preserves the existing <=128-row kernels and operation order while visiting
all chunks of a layer consecutively. Two-chunk toy host/VRAM tests reproduced
top-10 digits and IDs exactly. A real two-chunk host sample reduced prompt wall
from 8.570 to 7.188 seconds (16.1%) and increased host expert hits from 7.6% to
25.5%. Treat this as a promising single-sample result; no broad campaign was
run.

## Deliverable decisions

- Task 3 MLA exact absorption: implemented, GPU-exact, promoted auto-on.
- Task 4 VRAM ledger: its three low-risk changes were implemented. Its 352 MiB
  MLA phase-coloring proposal is superseded by the safer 704 MiB on-consumption
  reconstruction.
- Task 5 persistent KDA prefill: killed by its own modeled <3% ceiling.
- Task 7 wide-row exact INT8: killed because expert GPU work remains <10% of
  chunk wall even under favorable assumptions.
- Task 8 causal predictor: killed because held-out cross-family recall missed
  its gate.
- Task 9 staged DFlash verification: do not implement. Its exact shared-cost
  cold/hot maximin ceiling is 2.4369%, below the 3% gate; it also introduces
  new numerical-exactness boundaries. Pure sequential verification remains the
  useful endpoint.

## Operational result

No benchmark process remains. The 10-arm campaign was terminated after the
focused base/absorb pair, and its checked-in harness now defaults to only those
two arms. No current box contention was observed or assumed.

The GPU overclock showed no instability: no CUDA failure, Xid, PCIe fault,
clock collapse, or token divergence. After the final load it reported 48 C,
12,501 MHz memory, and normal P0 state.

## Commits

- `cac47b3` exact full-prompt layer-major prefill
- `c827b1e` low-risk idle-VRAM reclaims
- `05799cc` exact on-consumption MLA absorb kernels
- `1dafa4d` forced-sequential snapshot elision
- `c8f2e60` compact MLA auto-on and focused A/B harness
