# Problem 8: a rigorous 12-token/s feasibility certificate

Repository: https://github.com/novysvet/insignia  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware required: none

## Fixed facts and notation

One scalar token requests 336 expert records before cache reuse. One expanded
record is

```text
B = 13.5 MiB = 14,155,776 bytes.
```

DFlash verification reduces distinct record reads per committed token through
multi-row unions. Existing analyses place this count around `N=237..264` in
useful regimes; treat that interval as supplied evidence, not a universal law.
Let:

```text
v = fraction of distinct records served from VRAM,
h = host-hit fraction conditional on an off-VRAM request,
r = transported record-size ratio (1 expanded, <1 exact packed),
A = committed tokens per verify round,
C = non-expert dense/drafter/attention/launch time per round.
```

Measured service ceilings to use as uncertain intervals are pinned H2D about
23.2 GB/s, one NVMe about 3.7--4.7 GB/s, and VRAM about 800 GB/s. The 32 GiB
host tier holds 2,425 expanded records and observed `h` plateaus around 0.803.
The current exact DFlash result is about 5.1--5.3 token/s. All decimal/binary
bandwidth units must be stated consistently.

At minimum, any steady state must satisfy byte cuts such as

```text
H2D bytes/token  >= N * (1-v) * r * B,
SSD bytes/token  >= N * (1-v) * (1-h) * r * B.
```

These cuts are necessary but not sufficient because 42 layers serialize and
copy/compute overlap has finite lookahead and buffers.

## Mathematical problem

Produce either:

1. a constructive, precedence-valid schedule and parameter certificate that
   reaches at least 12 committed token/s; or
2. an impossibility theorem under clearly stated capacities, bandwidth/service
   curves, route-union law, and quality mode.

Give separate certificates for exact Top-8, exact DFlash, and an approximate
expert-count arm. Do not mix their request counts or quality assumptions.

## Required analysis

1. Derive simultaneous SSD, H2D, VRAM, compute, launch, and dependency lower
   bounds. Combine them with max/cut arguments rather than naively summing all
   resource times.
2. From the H2D cut, calculate the minimum `v` at 12 token/s for every
   `N in [237,264]`; propagate bandwidth uncertainty. Then calculate the
   conditional `h` required by the SSD cut. These are only necessary targets.
3. Add finite VRAM accounting. Dense FP8 weights, drafter, KDA/MLA state,
   scratch, and expert slots compete for capacity. Prove the maximum attainable
   `v` from a supplied popularity trace or give a distribution-free bound.
4. Model DFlash explicitly: union size, accepted-prefix distribution, rejected
   tail pollution, draft cost, verify cost, and `C/A`. Prove why accepted
   tokens per round alone is not a throughput certificate.
5. Include 42-layer serialization and bounded prefetch. Derive a flow-shop or
   queueing stability condition that is stronger than aggregate bandwidth.
6. Determine the exact improvement required from each candidate lever:

   ```text
   direct packed-scale execution,
   more VRAM slots,
   multiplicity-aware multi-row kernels,
   cache-aware approximate Top-k,
   a second independent SSD,
   higher DFlash acceptance,
   reduced dense/guard overhead.
   ```

7. Solve the minimal-change problem: find the smallest weighted set of lever
   improvements that enters the feasible 12-token/s region. Prove optimality
   for the chosen cost model or bound its approximation ratio.
8. Give an adversarial workload showing why a certificate based on mean host
   hit rate or mean acceptance can fail over long bursts.
9. For approximate modes, attach the numerical/free-output quality gate as a
   hard constraint. A fast mode exceeding +3.5% PPL or collapsing on hard
   outputs is outside the feasible set, not a successful schedule.

## CPU deliverables

- A unit-consistent symbolic calculator with interval arithmetic.
- A discrete-event 42-layer simulator with SSD/H2D/GPU queues and finite
  buffers, plus an independent verifier for every precedence/capacity rule.
- Sensitivity surfaces over `(N,v,h,r,A,C)` and service intervals.
- Exact or mixed-integer optimization for the minimal lever set.
- Three verdict tables: exact current hardware, approximate current hardware,
  and hypothetical hardware changes. Every cell must say proved impossible,
  constructively feasible, or unknown - never merely “likely.”

## Falsification and engine mapping

A certificate is invalid if it counts host hits as avoiding H2D, adds the two
SSDs on a one-drive machine, ignores rejected-tail traffic, assumes perfect
overlap across sequential layers, omits fixed GPU allocations, or reports a
symbolic roofline as a measured speedup.

If feasible, translate the witness directly into cache capacities, packed-mode
choice, DFlash `k`, row-kernel widths, prefetch depth, and required per-stage
budgets. If impossible, identify the binding cut and the smallest quantitative
hardware or algorithm change that removes it.
