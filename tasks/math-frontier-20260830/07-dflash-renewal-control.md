# Problem 7: DFlash drafting, verification, and retry as renewal control

Repository: https://github.com/novysvet/insignia
Hardware required: none

## Process

One DFlash round proposes a block. The target verifies up to `R <= 8` rows;
the current useful operating point is often four. Let `A` be the accepted
prefix length. The next round begins from the accepted anchor. Verification may
execute fewer than eight experts per row, choose cache-aware substitutes, or
retry the entire block exactly after observing target-logit collapse.

Costs include drafter time, target prefill/verify time, expert union I/O,
rollback snapshot/restore, and exact retry. Approximation changes the
distribution of `A` as well as quality.

## Main problem

Treat accepted tokens as rewards in a semi-Markov or renewal process. At each
round choose:

```text
block/verify width R,
per-row expert counts and cache-aware actions,
sequential versus batch verification,
whether/when to stop verifying,
whether to exact-retry after target observations.
```

Maximize long-run accepted tokens per second subject to a trajectory-level
quality constraint.

## Required results

1. Derive the renewal-reward objective correctly when zero-token rounds and
   variable accepted prefixes occur.
2. Under a defensible restricted observation model, characterize an optimal
   policy and prove whether retry/stop decisions are thresholds.
3. Produce a nonthreshold counterexample for the unrestricted problem.
4. Account for sunk costs: after verifying row `r`, some layer records are
   already cached and change the marginal cost of row `r+1` or a retry.
5. Model approximation as affecting both immediate latency and future state.
6. Give a safe fallback rule whose worst-case cost relative to always-exact is
   bounded.

## Numerical calibration

Use abstract cost parameters first, then sanity-check regimes around:

- exact DFlash approximately 187--194 ms per output token at its best;
- controller cost approximately 3.185 ms per four-row round;
- 13.5 MiB per missing expert record;
- acceptance distributions over `{0,...,R}`;
- exact retry restoring the pre-round recurrent snapshot.

Do not treat historical point estimates as stationary constants.

## Deliverables

- Semi-Markov/POMDP formulation and proofs.
- Exact small-state solver plus a scalable approximate policy.
- Synthetic evaluation across calibrated, miscalibrated, and adversarial
  acceptance models.
- A decision table describing which online probabilities/costs the engine must
  estimate before selecting a policy.
