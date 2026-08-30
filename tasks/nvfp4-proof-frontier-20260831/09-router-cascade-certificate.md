# Problem 9: A non-vacuous certificate for a 42-router error cascade

Repository: https://github.com/novysvet/insignia  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware required: none; the deliverable must run on an ordinary CPU.

## Mission

Derive a certificate that maps a local approximation error—NVFP4/FP8
attention, an approximate expert, or a pruned expert sum—to the probability or
certainty of changing any later Top-8 route or final greedy token. A global
Lipschitz product that merely explodes to infinity is a failed answer. The
certificate must exploit observed margins, sparsity, normalization, and the
fact that only discontinuity surfaces matter.

## Fixed engine facts

GLM-5.3-Flash has 45 layers and 42 sparse layers. Each sparse router produces
288 logits and selects Top-8 after adding its correction bias; selected sigmoid
scores are normalized and multiplied by 2.5. Hidden width is 4096 and mHC keeps
four residual streams. Tiny floating-point changes can flip a route and then
cascade. Approximation is acceptable only with full MSE, relative-L2, cosine,
forward/reverse KL, JS, same-prefix PPL, Top-1, route-flip, DFlash-acceptance,
and hard-output reporting. Same-prefix PPL may rise at most 3.5%, conditional on
useful hard outputs.

For layer `l`, model the exact state as `h_l`, the perturbed state as
`h_l + e_l`, router logits as `z_l(h)`, and the ordered exact logits as
`z_(1) >= ... >= z_(288)`. The decisive local margin is
`m_l = z_(8) - z_(9)`; top-1 and final-vocabulary margins are separate objects.
Previous-token target logits, router margins, route overlap, cache state, and
DFlash acceptance history are available causal features.

## Required result

1. Give deterministic sufficient conditions for no Top-8 change using local
   Jacobian-vector or norm bounds. Tighten them with directional, block, or
   randomized bounds rather than one global spectral norm.
2. When the deterministic condition fails, derive a calibrated probabilistic
   bound for at least one flip across 42 routers. It must handle dependence;
   an independence union bound may appear only as a labeled baseline.
3. Model post-flip propagation as a branching/coupled process and prove either
   a subcritical regime, an impossibility theorem, or a computable upper/lower
   sandwich. State exactly which quantities must be traced from the engine.
4. Incorporate previous target logits or margins and prove whether they can
   tighten the certificate without leaking future target computation.
5. Exhibit adversarial counterexamples: arbitrarily high cosine with a route
   flip, zero local flip but a later flip, and a margin-only estimator that is
   confidently wrong.

## CPU artifact and completion gate

Submit `router_cascade.py` with deterministic synthetic residual networks,
exact and perturbed rollouts, interval/directional certificates, correlated
Monte Carlo, and coverage plots. Include easy, boundary, heavy-tailed, and
adversarial tests. Report certificate coverage, false-safe count, tightness,
runtime, and sensitivity to 42-layer dependence.

The work is complete only when every claimed safe case has zero false-safe
events in exhaustive small models, empirical coverage meets its stated
confidence on held-out synthetic families, and the report gives a concrete
engine trace schema plus a go/no-go threshold. A proof that no useful
certificate is possible from the proposed features is a valid high-value
answer if accompanied by indistinguishable counterexamples.
