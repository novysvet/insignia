# Problem 2: cache-aware routing as constrained hypergraph optimization

Repository: https://github.com/novysvet/insignia
Hardware required: none

## Finite optimization problem

A layer group has `R <= 8` rows. Row `r` exposes 32 candidate experts
`E_r`. Its exact Top-8 set is `B_r`. An admissible action `A_r subset E_r`
must have cardinality `k_r`, retain a required prefix of `B_r`, and satisfy a
router-regret budget

```text
g_r(A_r) <= epsilon_r.
```

Each expert `e` has state-dependent costs:

```text
d_e in {0,1}          missing host record / NVMe read,
h_e in {0,1}          missing device record / H2D upload,
c_e >= 0              compute cost,
p_e >= 0              eviction opportunity cost.
```

The layer-group cost is a union cost, not a sum over rows:

```text
F(A_1,...,A_R) =
  alpha sum_{e in union A_r} d_e
  + beta sum_{e in union A_r} h_e
  + sum_r sum_{e in A_r} c_e
  + Phi(union A_r; current cache).
```

`Phi` may account for evicting records useful at later layers. Current engine
code exhaustively searches at most `8^4` tail assignments; this does not solve
the general problem or explain when the objective has exploitable structure.

## Main questions

1. Classify the computational complexity under progressively stronger
   restrictions: fixed `R`, one replaceable tail, arbitrary `k_r`, laminar
   candidate sets, and additive versus cache-coupled `Phi`.
2. Determine whether an equivalent formulation is submodular minimization,
   submodular maximization under matroid constraints, facility location,
   weighted set cover, or none of these. Give explicit reductions.
3. Find the smallest counterexample to each plausible greedy rule:
   cheapest resident tail, smallest individual regret, largest pairwise
   overlap, and row-by-row optimization.
4. Design an exact fixed-parameter algorithm parameterized by `R`, number of
   replaceable slots, or union width. Prove its complexity and give pruning
   bounds suitable for `R <= 8`, 32 candidates.
5. If the general problem is hard, derive an approximation guarantee that
   respects every row's regret constraint rather than relaxing quality after
   optimization.

## Strong extension

Make `d_e` stochastic because an asynchronous read may complete before demand.
Each candidate has completion distribution `T_e`; selecting it incurs a stall
`max(0,T_e-deadline)`. Determine whether adaptive querying or a two-stage
stochastic program has a useful decomposition.

## Required experiments

Generate adversarial and structured instances with 288 experts and 1--8 rows.
Compare exhaustive search, integer programming, branch-and-bound, greedy,
local search, and the proposed algorithm. Report objective gaps and node counts,
not merely runtime.

## Deliverables

- Complexity theorem(s) with complete reductions.
- Minimal counterexamples in a human-checkable table.
- Exact/FPT or approximate solver with deterministic tests.
- A derived rule for when spending additional 14700KF compute can save at
  least one 13.5 MiB record without exceeding router regret.
