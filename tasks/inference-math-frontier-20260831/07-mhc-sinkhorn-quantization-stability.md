# Problem 7: Stability of quantized mHC/Sinkhorn residual dynamics

Repository: https://github.com/novysvet/insignia  
Reference branch/commit: `codex/glm53-dflash2-4070-super` at `0740c63`

## Engine context

GLM-5.3-Flash carries four residual streams through mHC hyper-connections. A
learned 4x4 mixing matrix is normalized by Sinkhorn iterations and used to mix
streams before or after nonlinear layer transformations. Quantizing mixer
inputs, logits, normalization factors, or residual streams can save bandwidth,
but a tiny per-layer perturbation is multiplied through 45 layers and can alter
later MoE routing. Treating each layer independently misses the dynamics of the
product of near-doubly-stochastic matrices.

No private weights are needed. The problem is finite-dimensional numerical
analysis and matrix dynamics runnable on any CPU.

## Formal model

Let `X_l in R^{4 x d}` contain four residual streams. An ideal layer has

`X_{l+1} = P_l X_l + R_l(X_l)`,

where `P_l = Sinkhorn(exp(M_l))` is a positive 4x4 doubly-stochastic matrix and
`R_l` is the layer residual contribution. The implemented layer uses finite
Sinkhorn iterations, quantized inputs/outputs, and perturbations
`P_hat_l = P_l + E_l`, `R_hat_l = R_l + r_l`. A final collapse combines the
four streams before RMS normalization.

State precisely whether norms act across stream space, feature space, or both.

## Main problem

1. Derive sharp perturbation bounds for finite Sinkhorn normalization of a 4x4
   positive matrix. Include sensitivity to dynamic range and to zeros created
   by FP8 underflow. Establish when a fixed iteration count is enough and when
   no uniform bound exists.
2. Analyze products `P_{L-1}...P_0`. Use Birkhoff contraction, Dobrushin
   coefficients, joint spectral radius, or another tool to separate the common
   residual mode from disagreement modes. Prove conditions under which mixer
   perturbations do not accumulate linearly with depth.
3. Couple the mixer to nonlinear residual maps `R_l`. Give a small-gain or
   shadowing theorem with measurable layer constants, and construct a
   nonnormal/adversarial sequence where every `P_l` is well-conditioned and
   doubly stochastic yet transient disagreement becomes large.
4. Analyze the final collapse and RMS normalization. Determine which stream
   errors are annihilated, which are amplified near small norms, and whether a
   cheap invariant (sum, energy, or disagreement norm) can certify safety.
5. Design an optimal precision schedule across layers and components under a
   byte/compute budget. The schedule may keep Sinkhorn scalars in FP32 while
   storing streams or mixer logits in FP8. Provide a dynamic program or proved
   approximation, not an unsupported heuristic.

## Hard extension: routing discontinuity

Let selected router logits at layer `l` be `g_l(X_l)` with a Top-8 boundary
margin `m_l`. Combine the continuous mHC error bound with margins to certify a
prefix of unchanged route sets. Do not union-bound 42 layers blindly; seek a
conditional or pathwise certificate. Show a case where all mixer errors are
small in Frobenius norm but one route changes immediately.

## Required CPU artifact

Implement high-precision and simulated E4M3/BF16 4-stream systems. Enumerate or
optimize adversarial 4x4 sequences for small depth and compare observed error
with every bound. Include nearly permutation matrices, nearly rank-one mixing,
large-logit Sinkhorn inputs, underflowed entries, and alternating nonnormal
layers. Verify the small cases with arbitrary precision.

## Engine decision

Specify which arrays can safely move to FP8, required group scales, retained
FP32 invariants, and an online fallback signal. Kill the optimization if the
sound bound requires unavailable router margins, if finite Sinkhorn error is
unbounded at observed dynamic ranges, or if checking the certificate costs as
much bandwidth as the representation it protects.
