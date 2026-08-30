# Problem 4: the bit-exact split-K frontier for NVFP4 expert GEMV

Repository: https://github.com/novysvet/insignia  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware required: none

## Engine facts

The current NVFP4 DP4A kernel assigns one warp to one output row. A lane visits
16-weight groups `lane, lane+32, lane+64, ...`, performs FP32 `fmaf` updates in
that order, and the warp finishes with a fixed shuffle reduction. Gate/up have
2,048 output rows and 4,096 columns; down has 4,096 rows and 2,048 columns. A
CTA contains eight warps. Multi-row verification adds as many as eight FP32
accumulators per projection.

The model obeys a strict determinism law: changing an FP32 association can
perturb a router logit, change a Top-8 expert set, and cascade into different
tokens. An exact optimization must reproduce the existing per-row result
bit-for-bit. A reassociated implementation is allowed only as an explicit
quality-gated arm with MSE/cos/KL/JS/PPL and free-output evaluation.

## Mathematical problem

Determine whether useful extra K-parallelism exists under the exact operation
tree, and characterize the fastest possible approximate split-K tree when it
does not. A split may use multiple warps/CTAs per output row, cooperative
groups, a two-stage reduction, atomics, or redundant recomputation.

## Proof obligations

1. Formalize the current IEEE-754 operation DAG, including E4M3 decode,
   integer dot products, scale multiplication, lane-local FMA chains, and warp
   shuffle additions.
2. Give a necessary and sufficient condition for another schedule to be
   bit-identical for all finite inputs. It is not enough to say that addition
   is commutative.
3. Prove one of the following:

   - a constructive split-K schedule with more simultaneously active warps
     that realizes the identical DAG; or
   - an impossibility/lower-bound result showing that cross-CTA parallelism
     necessarily changes a dependency or requires serialization that removes
     the benefit.

4. Treat signed zero, subnormals, overflow, NaN exclusion, FMA contraction,
   and compiler reassociation explicitly. State the engine's finite-input
   assumptions.
5. If exact split-K is impossible, derive a forward-error bound for a balanced
   or compensated tree in terms of group contributions and compare ordinary,
   pairwise, Kahan/Neumaier, and exact-superaccumulator merges. Determine which
   spends the least extra compute for a target error certificate.
6. Combine the reduction model with occupancy. Prove when split-K can improve
   latency for the actual row counts and when the existing 256/512 CTAs already
   saturate the machine, making split-K pure overhead.

## CPU deliverables

- A bit-level FP32 emulator or carefully controlled C/C++ harness that compares
  operation DAGs on random and adversarial group contributions.
- Automatically generated counterexamples for every non-equivalent merge tree.
- A symbolic occupancy/latency model parameterized by SM count, active CTAs,
  registers, memory latency, and merge cost.
- A recommendation for batch one, multiplicity `2..8`, and prefill separately.
- Exact CUDA/SASS checks needed to ensure the compiler emitted the intended DAG.

## Decision rule

A negative proof is a complete success: it permanently rules out split-K on
the exact arm. Implement an approximate arm only if the model predicts at least
10% focused-kernel latency reduction and the proposed quality campaign can
falsify route cascades. Never use a synthetic cosine alone as the gate.
