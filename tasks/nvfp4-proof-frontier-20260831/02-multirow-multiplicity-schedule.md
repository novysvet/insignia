# Problem 2: multiplicity-aware multi-row NVFP4 scheduling

Repository: https://github.com/novysvet/insignia  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware required: none

## Engine facts

DFlash2 verifies up to eight candidate rows. In one sparse layer, each row
routes to eight of 288 experts. The union is staged once. For an expert `e`,
let `R_e` be the subset of verification rows that request it and let
`m_e=|R_e|`. The current CUDA multi-row kernel loads each NVFP4 weight group
once and evaluates every row in `R_e`, with eight statically allocated FP32
accumulators per output projection. Its gate/up pair variant can need sixteen
accumulators. This maximizes reuse at `m_e=8`, but register pressure and low
occupancy can make it wasteful for the far more common `m_e=1..3` cases.

One expanded expert is 13.5 MiB. Gate/up are `2048 x 4096`; down is
`4096 x 2048`. Each group of 16 weights consumes eight packed bytes and one
E4M3 scale byte. Arithmetic and expert accumulation order are model-visible.

## Mathematical problem

Design an optimal multiplicity-aware family of kernels and a dispatcher. A
kernel may process `b` rows at once, split `R_e` into several batches, pair or
unpair gate/up, and choose a different output-row/CTA shape. It may spend more
integer compute to reduce weight loads, but cannot assume free registers or
free occupancy.

For a kernel family `j`, model

```text
L_j(b)  = weight/scale transactions,
D_j(b)  = decode operations,
A_j(b)  = dot/FMA operations,
Q_j(b)  = registers per thread,
S_j(b)  = shared bytes per CTA,
O_j(b)  = occupancy/latency-hiding function,
H_j     = launch and dispatch cost.
```

The unknown hardware coefficients stay symbolic. The workload input is the
exact multiplicity histogram `h_m`, `m=1..8`, plus matrix dimensions.

## Proof obligations

1. Under a clearly stated roofline/occupancy model, prove the structure of an
   optimal partition of each `m` into kernel widths. Is an optimal policy a
   threshold rule, a shortest path over `m`, or neither?
2. Include discrete occupancy cliffs. Give a counterexample where the widest
   possible kernel minimizes bytes yet loses in time, and one where launching
   only scalar kernels loses despite higher occupancy.
3. Derive conditions under which gate/up should remain paired. Pairing shares
   activation quantization but doubles weight streams and accumulator state.
4. Account for output placement: rows in `R_e` need not be contiguous, and the
   down projection performs ordered weighted accumulation into each token's
   residual. Prove that the schedule neither drops nor duplicates a row and
   preserves per-row expert order.
5. Extend the policy from one expert to the layer union. Determine whether
   sorting experts by multiplicity is optimal when H2D copy and compute streams
   overlap, or construct a counterexample.
6. Quantify robustness: bound regret if the measured `Q_j/O_j` cost table is
   noisy or changes between the 56-SM local GPU and 66-SM larger GPU.

## CPU deliverables

- A dynamic program or exact integer solver for `m<=8`, with exhaustive tests
  against all partitions and synthetic kernel-cost tables.
- A generated dispatch table indexed by multiplicity and matrix role
  (gate/up pair or down).
- Sensitivity plots over register cost, launch cost, bandwidth, and `h_m`.
- A deterministic simulator for copy/compute overlap and adversarial examples
  that invalidate a simple widest-first rule.
- Exact proposed edits around `moe_multi` in `src/glm53_generate.cu` and the
  `nvfp4_*_rows_kernel` family in `src/glm53_expert_bench.cu`.

## Engine gate

The eventual CUDA A/B must report per-multiplicity kernel medians, SASS
registers/spills, achieved bandwidth, DFlash union/multiplicity histograms,
round wall time, committed ms/token, acceptance, and all quality metrics.
Exact IDs alone are not enough if accumulation order changed. Kill the family
if dispatch complexity cannot beat the current width-eight kernel by 5% on a
trace-weighted focused microbenchmark.
