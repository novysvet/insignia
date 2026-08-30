# Problem 6: quantization error through discontinuous routing and recurrent state

Repository: https://github.com/novysvet/insignia
Hardware required: none

## Motivation

Ordinary layerwise MSE bounds are misleading for this engine. A tiny numerical
perturbation can cross a router margin, change an expert set, alter recurrent
KDA/MLA state, and cause a later discrete divergence. Mathematically equivalent
floating-point reassociation has already changed greedy tokens. We need a theory
that is smooth away from routing boundaries and explicitly handles boundary
crossings.

## Model

At layer `l`, hidden state `h_l` produces router logits `g_l(h_l)`. A biased
Top-8 operator selects `S_l`. The transition combines attention, mHC residual
mixing, a shared expert, and routed experts:

```text
h_(l+1) = F_l(h_l, S_l, W_l) + eta_l.
```

`eta_l` represents FP8/NVFP4/UD-IQ3 approximation and arithmetic-order error.
KDA state carries across output tokens; MLA attention carries a latent history.
Let `margin_l` be the biased eighth-minus-ninth router score gap.

## Main problem

Derive a useful bound or probabilistic certificate for output damage over many
layers/tokens which separates:

1. continuous amplification while every route is unchanged;
2. probability and consequence of route-boundary crossings;
3. recurrence across decode tokens;
4. softmax/attention sensitivity;
5. mHC mixing stability.

## Required results

- A deterministic same-route bound in terms of local Jacobian/operator norms.
- A margin theorem: sufficient conditions on hidden/logit perturbation for
  Top-8 invariance, including correction bias and near ties.
- A stochastic bound using empirical margin and quantization-error
  distributions, with dependencies across layers allowed.
- At least one counterexample where every layer has tiny cosine/MSE but final
  token probability or free trajectory changes catastrophically.
- A result explaining when expected error is finite/small but tail failure is
  dominated by rare route crossings.

## Strong version

Construct a hybrid certificate that evaluates exact local router margins and a
cheap upper bound online. It should decide whether a proposed lower-precision
or Top-k action is provably route-safe for the next layer or whether exact
execution is required. Analyze false-safe probability when operator norms are
estimated rather than known.

## Deliverables

- Theorems with clearly stated norm and independence assumptions.
- Minimal numerical counterexamples.
- CPU simulation over 42 layers and multiple recurrent tokens.
- A practical list of trace quantities needed to turn the bound into a
  falsifier feature or precision-allocation constraint.
