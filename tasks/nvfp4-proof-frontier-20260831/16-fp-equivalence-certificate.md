# Problem 16: Mechanical equivalence certificates for CUDA reduction rewrites

Repository: https://github.com/novysvet/insignia  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware required: none for the core proof/checker; supplied PTX/SASS may be
analyzed as text.

## Mission

Build a formal and executable method for deciding whether a proposed CUDA
NVFP4 kernel rewrite preserves the engine's effective floating-point model
bit-for-bit. Algebraic equivalence is insufficient: accumulation order,
rounding, FMA contraction, warp-shuffle topology, signed zero, NaN behavior,
and fast-math intrinsics can change later Top-8 routes.

## Fixed arithmetic path

Ada sm_89 has exact INT8 `DP4A` products/INT32 accumulation but no native FP4
MMA. The current NVFP4 path expands E2M1 codes to signed integers, evaluates
four DP4A operations per K16 group, converts the dot to FP32, applies E4M3,
activation, and global scales with a specific association, accumulates by
`fmaf`, and reduces lanes with a fixed XOR tree. Multi-row kernels must retain
that per-row chain. The build uses nvcc `-O3 --use_fast_math -arch=sm_89`.

## Required result

1. Specify an operational semantics for the relevant CUDA subset: INT32
   wrap/overflow assumptions, FP32 round-to-nearest behavior, FMA, conversion,
   `__expf` if included, shuffles, memory round-trips, signed zero, infinities,
   and NaNs.
2. Convert a reference and candidate kernel's arithmetic/reduction slice into
   canonical dependency DAGs annotated with rounding points and lane maps.
3. Give sound rewrite rules and a certificate checker. Prove soundness for the
   supported subset; when equivalence is undecidable or unsupported, return
   “unknown,” never “equal.”
4. Handle split-K and tensor-core candidates. State exactly when integer MMA
   can be decomposed into the legacy K16 scale boundaries and XOR order.
5. Produce counterexamples for reassociation, changed scale parentheses,
   online versus two-pass softmax, different shuffle trees, neutral partials,
   and signed-zero canonicalization.

## CPU artifact and completion gate

Submit `fp_equiv.py`, a compact input format for lane/reduction DAGs, a
bit-vector/SMT or exhaustive-small-domain backend, proof certificates, and a
test corpus. It must certify known identical transplants and reject known
non-identical rewrites. Include random differential testing over raw FP32 bit
patterns and exhaustive reduced formats, while clearly separating testing
from proof.

Completion requires a soundness argument, zero false-equal results in the
adversarial corpus, explicit compiler/SASS assumptions, and a recipe for CI:
extract operations, check the certificate, then run fixture byte gates. A
formal proof that a desired split-K transformation cannot preserve order is a
successful result.
