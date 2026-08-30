# Problem 4: exact FP8 fused codec and system-level break-even

Repository: https://github.com/novysvet/insignia
Hardware required: none for the mathematical solution

## Known evidence

Dense and DFlash E4M3 weight caches compress exactly, including raw FP16
scales, to ratios 0.906244 and 0.907438. The primary physical tile is the
`16 x 64 = 1024`-byte slab consumed by one Ada warp. Every tile is independently
decodable and may choose RAW, palette, magnitude, exponent, sparse-zero, or
bitplane modes. A descriptor costs eight bytes. The current raw kernel observes
about 698 GB/s; the GPU's effective VRAM bandwidth is around 800 GB/s.

Dense compression models 780.7 MiB of reclaim, roughly 57 expanded expert
slots. The user accepts implementation even though the original ratio gate was
0.90. The real question is no longer "does it pass 0.90?" but "where is the
whole-engine break-even after decoder compute, occupancy, and extra cache
hits?"

## Mathematical problem

For tile type `j`, let

```text
r_j     stored-byte ratio including metadata,
o_j     integer/bit operations required to decode,
q_j     registers and shared-memory footprint,
b_j     branch/exception distribution,
f_j     frequency in a matrix family.
```

Let occupancy and throughput be nonlinear functions of `(q_j,b_j)`. Construct
a model that chooses tile size, mode family, warp work partition, and possibly
RAW escape to minimize total expected engine time, not codec bytes alone.

The system objective must include:

```text
T_total = T_dense_kernel(codec)
        + T_expert_compute
        + T_NVMe(miss probability after reclaim)
        + T_H2D
        + synchronization/queue interaction.
```

## Required results

1. Derive lower bounds on reconstructed-byte throughput from memory traffic,
   descriptor traffic, and a parameterized integer instruction budget.
2. Prove an optimal mode-selection rule for some nontrivial occupancy model, or
   show by counterexample why choosing each tile's smallest byte encoding is
   not globally optimal.
3. Derive a break-even inequality connecting codec slowdown `delta`, reclaimed
   slots `Delta C`, the cache miss curve `m(C)`, and per-miss NVMe/H2D latency.
4. Analyze the observed plateau: 2,425 host slots already give about 80.3% hits.
   Determine how much marginal hit-rate improvement 57 additional device or
   host-equivalent slots must provide to offset 0.5%, 1%, 2%, 5%, and 10%
   dense-kernel regressions.
5. Treat tile-mode divergence as a distribution, not an average. Establish a
   tail bound or construct an adversarial matrix ordering that defeats the
   average roofline.
6. Determine whether a compute-rich decoder can profitably decode multiple
   slabs jointly, reuse palettes, or speculatively decode both likely modes
   and select without branches.

## Counterexample challenge

Prove or disprove:

> If exact compression ratio is below raw-kernel bandwidth divided by physical
> VRAM bandwidth, then some fused decoder schedule must be faster than the raw
> kernel.

A disproof must explicitly account for instruction issue, dependency chains,
occupancy, and synchronization rather than asserting "decode has overhead."

## Deliverables

- Symbolic break-even model and sensitivity plots.
- Theorem/counterexample for local byte-optimal versus global time-optimal
  encoding.
- A CPU optimizer that accepts measured per-mode costs later and emits the
  optimal family policy.
- Exact hardware measurements required to instantiate every free parameter;
  do not fabricate Ada instruction throughput.
