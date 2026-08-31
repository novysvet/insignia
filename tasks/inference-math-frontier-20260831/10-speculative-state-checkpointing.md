# Problem 10: Optimal checkpointing for speculative recurrent verification

Repository: https://github.com/novysvet/insignia  
Reference branch/commit: `codex/glm53-dflash2-4070-super` at `0740c63`

## Engine context

DFlash2 proposes up to eight rows and the 45-layer target verifies them. Eleven
layers use MLA caches; 34 use recurrent KDA state plus depth-4 convolution;
four mHC residual streams also evolve. If a draft fails after a prefix, the
engine commits accepted rows and must restore or reconstruct all target state
for the rejected suffix. Saving every intermediate is bandwidth- and VRAM-
heavy; saving too little forces replay. Acceptance length is random and depends
on the policy and prompt.

This is a weighted checkpointing/pebble-game problem. It can be solved using a
synthetic DAG and CPU algorithms without the model.

## Formal model

A verification round has rows `1..k` and layers `1..L`. Node `(l,r)` represents
the state after processing row `r` through layer `l`. Edges include ordinary
layer dependencies, per-row hidden flow, KDA recurrence across rows, convolution
history, MLA append semantics, and capture outputs consumed by the drafter.
Each node or state component has compute cost, save size, restore cost, and a
legal storage tier. After computation, random accepted length `J in {0,...,k}`
is revealed. The engine must materialize exactly the state corresponding to
the committed prefix and discard suffix effects.

Floating-point order is part of correctness: recomputation is legal only when
it replays the same operations and inputs.

## Main problem

1. Define the minimal state cut sufficient to restore any accepted prefix.
   Prove sufficiency and necessity for an abstract mixture of recurrent,
   append-only, and feed-forward nodes. Explain which tensors can be derived
   from others and which cannot.
2. Given a memory budget and distribution of `J`, find the checkpoint placement
   minimizing expected save/restore/recompute time. Give an exact dynamic
   program for a useful restricted graph and establish complexity or hardness
   for the general case.
3. Derive an approximation algorithm or competitive online policy when the
   acceptance distribution is unknown and changes causally. Include the cost of
   learning it and guard against a sudden collapse to `J=0`.
4. Jointly choose draft block size `k` and checkpoint plan. Larger `k` improves
   amortization but expands archive state and rejection work. Analyze the
   renewal reward in committed tokens per second, not accepted drafts alone.
5. Add storage tiers: registers/shared memory during a kernel, VRAM, pinned RAM,
   and recomputation. Determine when asynchronous spill can be hidden and when
   it lies on the critical path.

## Hard extensions

- Checkpoints may be compressed with a certified error or may store an exact
  seed/input sufficient for replay. Compare numerical and bandwidth costs.
- Multiple candidate verification policies share a common prefix. Model a
  verification lattice rather than one chain and find reusable checkpoints.
- The drafter consumes target captures; restoring target state must also keep
  drafter cache alignment exact. Add those cross-model edges to the graph.

## Required CPU artifact

Implement a typed weighted-DAG model and exact solver for small `L,k`, plus a
scalable heuristic. Test geometric, bimodal, and adversarial acceptance laws;
VRAM-rich and memory-starved regimes; and graphs containing KDA-like recurrence
and MLA append-only state. Validate every returned plan by replaying random
accept/reject outcomes and comparing final abstract state byte-for-byte.

## Engine decision

Return a concrete archive layout indexed by layer and row, the action taken for
each accepted length, its peak bytes, and expected/maximum latency. Identify
which existing snapshots are redundant. Kill any plan that assumes suffix
writes can simply be ignored despite recurrent state, changes operation order,
or optimizes accepted length without counting committed-tail behavior.
