# Problem 7 — Prove a barrier-minimal asynchronous Ada prefill schedule

Expected effort: 8–16 hours. CPU schedule search; GPU optional, not required.

## Authority and current kernel

This file is the assignment. Clone https://github.com/novysvet/insignia.git,
branch `glm53-dflash2-4070ti-super`, starting evidence `9e9090d`. Inspect
`iq3_xxs_wmma32_kernel` and `iq4_xs_wmma32_kernel` in `src/glm53_iq.cu` only as
the baseline to improve.

The promoted prefill CTA computes 16 output rows × 32 routed tokens. For every
K=32 group it:

1. decodes packed weights into a 16×32 FP16 shared tile;
2. converts a 32×32 FP32 activation tile into two col-major FP16 tiles;
3. executes a CTA barrier;
4. performs two 16×16×16 HMMA operations per consumer warp;
5. executes another CTA barrier before overwriting shared memory.

IQ3 uses 128 threads: every thread expands one four-weight codeword while two
warps also consume HMMA; two warps are producers only. IQ4 uses 64 threads and
both warps produce and consume. IQ3 compiles with 48 registers, one barrier
resource, 3,072 bytes shared memory, and no spills. Seven-run medians for a
32-row gate/down are 91.668 us IQ3 and 67.711 us IQ4. Local cosine is above
0.99999995.

Ada sm_89 has warp-level HMMA and Ampere-style asynchronous global-to-shared
copy operations, but no Hopper TMA, warp specialization hardware, or native
block-scaled FP4 MMA. Packed IQ weights still require integer decode before
HMMA. Treat exact instruction latencies and issue widths as parameters unless
verified from primary documentation; the schedule proof should not depend on
one folklore number.

## The scheduling problem

Model each K=32 iteration as a dependence DAG with global loads, codebook/sign
decode, FP16 conversion, shared stores, barriers, fragment loads, and HMMA.
Shared buffers have 32 banks of 4 bytes. A legal schedule must prevent a
producer from overwriting a tile before every consumer has loaded it and must
preserve the current FP32 accumulation order across K groups.

Find the minimum steady-state synchronization cost for single, double, and
triple buffering under these constraints. Consider CTA barriers, named
barriers/mbarrier if actually available on sm_89, warp barriers, producer
epochs, and `cp.async` for activation loads. Packed weight decode cannot be
performed by `cp.async`, but its raw bytes may be staged first.

Prove either:

- a two-buffer schedule with fewer than two full-CTA barriers per K group and
  no race, including prologue/epilogue; or
- a lower bound showing why the current barrier count is unavoidable under a
  stated producer/consumer ownership, followed by the minimal ownership change
  that breaks the bound.

Jointly solve shared-memory layout. Minimize bank conflicts for producer stores
and `ldmatrix`/WMMA fragment loads while keeping activation loads coalesced.
Give a formal bank mapping for every lane, not a diagram with missing cases.
Account for IQ3's 128 producer tasks, IQ4's paired-nibble decode, and 16-byte
alignment requirements.

Finally build a parametric throughput/occupancy bound. Include register growth,
shared-memory buffers, maximum active CTAs, memory sectors, decode instructions,
barrier throughput, and HMMA issue. Determine when 32, 64, or 128 routed rows
per expert should share a weight tile.

## Required deliverables

- Dependence DAG and formal race-free execution model.
- A theorem proving synchronization minimality or a strictly better schedule.
- A deterministic discrete-event simulator with configurable latency/issue
  parameters and exhaustive small-schedule search.
- A bank-conflict checker enumerating every shared access for all lanes.
- Pseudocode or compilable CUDA for prologue, steady state, and epilogue;
  include buffer indices and barrier epochs explicitly.
- Occupancy/resource certificate for sm_89 and sensitivity plots over uncertain
  latency parameters.
- Adversarial schedules that expose early overwrite, missing arrival, divergent
  warp, and tail-token bugs.
- Predicted performance against 91.668/67.711 us, clearly labeled unmeasured.
  Promotion requires a race proof, identical arithmetic order, no spill, and a
  robust predicted gain above 7%; otherwise preserve the simpler kernel.
