# Problem 10: Certified partial expert execution and early token decisions

Repository: https://github.com/novysvet/insignia  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware required: none; use a CPU reference implementation.

## Mission

Determine whether Insignia can stop computing an MoE layer, a down projection,
or the final vocabulary head once the unfinished work is provably unable to
change the relevant Top-8 route or greedy token. The target is exact early
termination, not a heuristic confidence threshold, with an optional separately
quality-gated approximate extension.

## Fixed computation

There are 42 sparse layers, 288 experts, and eight selected experts per token.
Each routed output is an ordered FP32 accumulation

`r = fmaf(E_0(x), a_0, fmaf-order continuation ...)`,

with gate/up matrices `2048 x 4096`, down matrices `4096 x 2048`, and a dense
shared expert. Four mHC residual streams and RMS normalization follow. The
vocabulary has 154,880 rows. Current exactness treats the FP32 FMA/reduction
order as part of the model. NVFP4 bodies use E2M1 plus one E4M3 scale per 16
weights and a global FP32 scale.

## Formal questions

1. Given computed partial sum `s_j` and certified sets for every uncomputed
   expert contribution, derive the tightest tractable set containing the final
   hidden vector. Compare norm balls, coordinate intervals, zonotopes,
   ellipsoids, and low-rank-plus-residual sets.
2. Propagate that set through mHC, RMSNorm, the next router, or selected rows of
   the LM head. Give a sound stopping test for unchanged Top-8 or Top-1.
3. Find the optimal order in which to evaluate eight experts or vocabulary
   blocks to maximize expected early certificates under known costs. Prove a
   structural rule, approximation ratio, or hardness result.
4. Establish lower bounds: construct instances where every expert and every
   relevant vocabulary row must be evaluated even with arbitrarily large
   current margins.
5. Design how bounds are obtained cheaply. Offline per-expert operator norms,
   activation-dependent NVFP4 group bounds, random sketches, and dual bounds
   are permitted; a certificate whose checking cost exceeds saved work fails.

## Supplied performance/quality contract

A decode token requests 336 expert records before cache hits. Avoiding even one
cold 13.5 MiB record can matter more than thousands of scalar operations.
Approximate variants must report MSE, relative-L2, cosine, both KL directions,
JS, same-prefix PPL, routes, Top-1, DFlash acceptance, and hard outputs; the PPL
ceiling is +3.5% only when hard outputs remain useful.

## Deliverable

Submit a theorem/counterexample report and `partial_certificate.py`. The program
must generate structured and adversarial matrices, execute exact ordered sums,
run each bounding family, optimize evaluation order, and report false-safe
count, fraction of work certified away, bound/check cost, and worst-case gap to
an oracle. Exhaustively enumerate a small integer model to validate soundness.

Completion requires zero false-safe decisions for the exact arm, a proved cost
model including certificate overhead, and an engine decision rule specifying
which tensors/metadata are stored and the minimum records or LM-head rows that
must be skipped to win. A rigorous impossibility bound is acceptable.
