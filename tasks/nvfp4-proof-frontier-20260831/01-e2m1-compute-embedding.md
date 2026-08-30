# Problem 1: minimal exact E2M1 compute embedding on Ada

Repository: https://github.com/novysvet/insignia  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware required: none

## Engine facts

An NVFP4 group has 16 signed E2M1 weights, stored as eight nibbled bytes, and
one unsigned E4M3 scale. The magnitude alphabet is

```text
M = {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
```

Let `c_i` be a four-bit code and define the exact signed integer

```text
z(c_i) = 2 * E2M1(c_i) in {0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-12}.
```

For an E4M3 byte `u`, let `sigma=-1` when bit 7 is set and `+1`
otherwise, `e=(u>>3)&15`, and `m=u&7`. The engine's exact finite decoder is

```text
phi(u) = sigma * m * 2^-9              when e=0,
phi(u) = sigma * (8+m) * 2^(e-10)      when e>0.
```

Both E2M1 codes for positive and negative zero, and all E4M3 byte patterns,
must be covered explicitly. Do not silently canonicalize their bytes merely
because their real values compare equal.

After groupwise INT8 activation quantization `x_i ~= a*q_i`, the current
kernel evaluates

```text
dot ~= 0.5 * a * s * g * sum_i z(c_i) q_i,
```

where `s` is the E4M3 block scale and `g` the matrix FP32 global scale. It uses
a 256-entry shared table plus byte permutations to expand eight packed weight
bytes into sixteen signed INT8 lanes, then four DP4A instructions. Gate and up
are evaluated as a pair. Ada has DP4A and INT8 tensor cores but no native FP4
MMA.

## Mathematical problem

Find the minimum-cost exact embedding of the signed E2M1 alphabet into the
integer operations available to a compute-rich, bandwidth-limited machine.
Use a parameterized machine model rather than fabricating Ada latencies. The
model must charge separately for:

```text
packed bytes read, scale bytes read, table/shared-memory reads,
integer decode/permutation operations, dot-product operations,
registers, shared memory, dependency depth, and synchronization.
```

Consider at least these families:

1. direct LUT expansion followed by four-lane integer dot products;
2. an affine code representation `z(c)=alpha(code bits)+beta` with correction
   terms depending only on activation sums;
3. signed bitplane or small-basis decompositions of `z(c)`;
4. expansion of a tile in registers/shared memory followed by INT8 MMA;
5. redundant computation that removes tables, branches, or scale-plane reads.

The solution must distinguish batch one, DFlash multiplicity `r=2..8`, and
prefill tiles `r>=16` because an expansion can be amortized over rows.

## Proof obligations

1. Prove a lower bound on the number of independent four-lane dot products or
   basis contractions required to represent all 16 E2M1 products exactly.
   State the permitted instruction algebra precisely.
2. Find a minimal exact affine or bitplane representation of `z(c)`, or prove
   that no representation in the chosen class beats the LUT normal form after
   all correction terms are counted.
3. Give a sharp signed-accumulator overflow theorem for the actual column
   counts 2,048 and 4,096 and for arbitrary `K`. Distinguish the current
   per-group integer dot followed by FP32 scaling from a tensor-core design
   that might accidentally accumulate across independently scaled groups.
4. Solve the tensor-core scale-separation problem. A `K=32` MMA naturally
   mixes two independently E4M3-scaled groups of 16. Prove the minimum number
   of independently recoverable partials needed for arbitrary scale pairs, or
   construct an embedding that attains the bound through extra rows/columns or
   structural zero padding.
5. Characterize exactly when `phi(u) * E2M1(c)` is itself representable in
   E4M3. For the remaining pairs, find a minimum signed power-of-two
   decomposition or prove the optimum in a stated circuit model.
6. For a tile with `K` weights and `r` activation rows, derive the symbolic
   break-even between DP4A and expand-then-MMA, including expansion traffic,
   occupancy loss, and the fact that packed weights are read once for all `r`.
7. Prove or disprove:

   > For sufficiently large `r`, every exact four-bit alphabet admits an
   > expand-once INT8-MMA implementation that is faster than a DP4A GEMV on a
   > machine with unused tensor throughput.

   A valid disproof should construct a machine-parameter region where memory,
   issue, occupancy, or synchronization keeps DP4A optimal for all feasible
   `r`.
8. Show how E4M3 and global scales can be placed without changing the current
   FP32 group accumulation order. If the proposed algorithm cannot preserve
   order, identify the exact reassociation and classify it as a quality-gated
   approximate arm.

## CPU deliverables

- An exhaustive verifier over all 16 codes and adversarial INT8 activations
  proving the proposed algebra has no signedness, overflow, or scale-factor
  error.
- A symbolic cost enumerator that returns the optimal family as machine
  parameters vary and draws the `r` break-even regions.
- Counterexamples for at least one superficially attractive representation.
- A concrete CUDA mapping: tile dimensions, lane ownership, register/shared
  footprint, and the source functions to replace
  (`src/glm53_expert_bench.cu`, `nvfp4_dp4a*_kernel`).

## Acceptance and kill rules

Proceed to CUDA only if the proof leaves a nonempty plausible Ada parameter
region with at least 10% predicted kernel gain or enables direct packed-scale
execution without expanding record bytes. Kill any scheme whose advantage
requires nonexistent Ada FP4 MMA, omits scale work, or silently changes the
dot-product algebra.
