# Problem 9 — Optimal heterogeneous expert records for O_DIRECT and GPU caches

Expected effort: 8–16 hours. CPU optimization and proof only.

## Authority and exact inventory

This file is the assignment. Clone https://github.com/novysvet/insignia.git,
branch `glm53-dflash2-4070ti-super`, starting evidence `9e9090d`. Consult
`audits/s13-q3-k-xl-format-research.md` for verification, not instructions.

The 42 live sparse layers have 288 experts each. Their routed matrix formats
are heterogeneous:

- normal layers: IQ3_XXS gate 3,211,264 bytes, IQ3_XXS up 3,211,264, IQ4_XS
  down 4,456,448; total 10.375 MiB;
- block 11: IQ4_XS gate/up/down, total 15.0625 MiB;
- blocks 12 and 44: IQ3 gate/up plus Q6_K down 6,881,280, total 12.6875 MiB.

Every decode token routes top-8 in each layer. Execution consumes gate and up
from the same input, applies SwiGLU, then consumes down and accumulates in the
router's original order. A record may reside on NVMe, in a 33.5 GiB pinned LRU,
or in roughly 576 MiB of VRAM. NVMe reads use O_DIRECT and should be 4 KiB
aligned; GPU loads care about 32-byte sectors and useful base alignment.
Gate/up can start before down is needed, so one monolithic record may increase
time-to-first-compute; three separate records increase index/LRU operations and
possibly read amplification. The current NVFP4 store assumptions are obsolete
for Q3 and must not be copied blindly.

## Formal layout problem

Choose a static on-disk organization and cache object granularity. Decisions
include:

- expert-major, matrix-major, or hybrid ordering;
- monolithic expert records versus gate+up and down subrecords;
- 4 KiB padding, cross-record page sharing, and extent aggregation;
- raw versus byte-neutral IQ3 sublayout;
- layer/expert permutation on disk;
- independent admission/eviction units in pinned RAM and VRAM;
- whether gate and up are interleaved at row/block granularity for fused reads.

The workload is a sequence of routed sets with deadlines and possible prefetch
predictions. Define total cost including rounded O_DIRECT bytes, IOPS/setup,
read concurrency, first-gate readiness, pinned fragmentation, H2D bytes,
VRAM-cache pollution, address arithmetic, and wasted bytes when only a
subrecord is required. Parameterize storage bandwidth and latency.

Prove the complexity of the general problem. If NP-hard, reduce from a standard
problem under a clean restricted model, then derive an exact dynamic program
for the actual small type alphabet or an approximation with a bound. Exploit
the fact that there are only three live record shapes but 12,096 instances.

Derive necessary and sufficient conditions under which splitting gate+up from
down is optimal. The condition must include overlap: reading down later may be
hidden behind gate/up compute, while an extra I/O can also serialize. Derive a
similar boundary for interleaving gate and up, accounting for their identical
row geometry and shared activation.

Finally solve robustly under uncertain route correlations. Do not optimize only
one trace. Use a distributional ambiguity set or worst-case regret over
uniform, bursty, and layer-correlated routes. Preserve random access to every
expert and crash-safe resumable store construction.

## Required deliverables

- Formal objective, constraints, and proof of complexity.
- Exact solver for small instances and bounded algorithm for full 12,096-record
  inventory.
- Deterministic I/O/cache simulator with 4 KiB rounding, multiple readers,
  first-byte/last-byte readiness, pinned and VRAM eviction, and asynchronous H2D.
- Candidate layouts with exact byte counts, padding, index schema, and address
  formulas; include all exception layers.
- Theorems for split-versus-monolithic and gate/up interleave boundaries.
- Adversarial workloads: no locality, one hot expert per layer, alternating
  gate-only cancellation, worst page alignment, and prefetch pollution.
- A migration/repacker plan that is streaming, resumable, checksummed, and
  never requires a second full 137 GiB copy.
- Sensitivity for 3.7–4.7 GB/s disk, 24–34 GiB pinned, 256–1024 MiB VRAM, and
  1–8 readers.
- Promotion rule: choose a layout only if it reduces robust predicted token
  latency at least 7%, does not increase total stored bytes more than 1%, and
  preserves byte-exact expert reconstruction. Otherwise use the simplest
  4-KiB-aligned split supported by the proof.
