# Problem 13: the MoE hierarchy separability conjecture

Repository: https://github.com/novysvet/insignia
Hardware required: none

This is an explicit prove-or-disprove challenge. A small counterexample is a
fully successful result.

## Conjecture

Consider a finite-horizon MoE inference system with:

1. additive transfer costs per distinct `(layer,expert)` record;
2. independent per-layer route processes conditioned on a shared causal state;
3. fixed host/device cache capacities;
4. no quality approximation: every demanded expert must eventually execute;
5. prefetches allowed, but all transfers and evictions have nonnegative cost.

**Hierarchy Separability Conjecture.** There exists an optimal policy whose
cache eviction and prefetch decisions decompose into independent per-layer
policies coupled only by a single scalar shadow price for cache capacity.

In other words, after choosing one global Lagrange multiplier, each layer may
optimize its own records without needing the identities or future request
structure of any other layer.

## Task

Prove the conjecture under its stated assumptions, find a finite explicit
counterexample, or identify a minimal additional condition that makes it true.

A counterexample must specify:

- number of layers, experts, cache slots, and horizon;
- complete conditional request distribution or deterministic scenarios;
- all transfer/prefetch/eviction costs;
- the best separable policy and its expected cost;
- a nonseparable policy with strictly smaller expected cost;
- why no alternate shadow price repairs separability.

Seek the lexicographically smallest witness in `(layers, experts, cache slots,
horizon, scenarios)`. Computer-assisted exhaustive search is allowed, but the
final witness and optimality proof must be human-checkable.

## Extensions

1. If the conjecture is false, characterize the obstruction: shared capacity,
   correlated deadlines, set requests, prefetch queueing, or value of future
   information.
2. Determine whether indexability in the sense of restless bandits holds for a
   restricted model.
3. Find an approximation ratio for the best scalar-price separable policy.
4. Repeat the analysis when records have unequal sizes and when a request is a
   Top-8 set.

## Why this matters

A true theorem would justify radically simpler per-layer cache controllers. A
small counterexample would prevent months of optimizing a decomposition that
can never be globally correct and would reveal exactly what context the learned
falsifier/cache policy must retain.

## Deliverables

- Proof or minimal counterexample.
- Exhaustive-search code with independently checkable output.
- A precise corrected theorem or approximation statement.
