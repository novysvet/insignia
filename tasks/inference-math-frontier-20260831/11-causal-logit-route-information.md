# Problem 11: How much can previous logits reveal about future routes?

Repository: https://github.com/novysvet/insignia  
Reference branch/commit: `codex/glm53-dflash2-4070-super` at `0740c63`

## Engine context

An exact target logit vector has 154,880 entries. Previous target and DFlash
logits, route sets, hidden summaries, and cache state may reveal whether the
next block is difficult, which experts will be needed, or whether approximate
verification will fail. Earlier fixed CountSketch summaries have an explicit
collision: two radically different logit worlds map to identical features.
The engine can afford additional GPU compute, but not repeated full-vector CPU
transfers. It needs to know what causal information is fundamentally useful and
what representation is sufficient under stated distributional assumptions.

This is an information-theory/statistics problem. It requires no access to the
real model; synthetic distributions and finite alphabets are acceptable.

## Formal setup

At time `t`, let `L_t` be target/draft logits, `R_t` the 42-layer Top-8 route
object, `C_t` cache state, and `Y_t` a future quantity such as next routes,
verification failure, accepted length, or quality loss. A causal encoder with
state `S_t` observes some history and emits at most `b` bits or `m` floats:

`S_t = Phi(S_{t-1}, L_t, R_t, C_t)`.

A decision rule uses `S_t` and cheap current features. Full logits may remain
on GPU for fused reductions, but persistent state and host transfers are
bounded.

## Main problem

1. Give worst-case lower bounds on `b` for preserving a target functional of
   logits (JS, Top-k overlap, margin-tail events, or a future label). Extend the
   fixed-sketch collision idea to any deterministic or randomized public
   encoder under an adversarial input model.
2. Under a distributional model, characterize the minimal sufficient causal
   state for predicting `Y_{t+1:t+h}`. Relate achievable risk to conditional
   mutual information or predictive rate-distortion, and distinguish
   information about routes from information about approximation failure.
3. Prove when previous logits add value beyond previous routes and hidden
   summaries. Construct one process where they are sufficient for near-perfect
   prediction despite high marginal route entropy, and another where they add
   exactly zero conditional information.
4. Design a causal encoder that spends GPU computation to reduce transfer:
   exact scalar divergences, learned state, random features, top-head residuals,
   or combinations. Give a finite-sample guarantee or a clearly labeled
   asymptotic theorem and account for key/schema stability across checkpoints.
5. Optimize representation size jointly with decision value. One extra byte of
   state is useful only if its expected I/O/verification saving exceeds its
   compute, residency, and transfer cost.

## Hard extension: active information acquisition

The controller may request one of several measurements—full exact JS, selected
logit blocks, an early layer capture, a route prediction, or nothing—before
choosing the fast path. Formulate this as sequential experimental design and
derive an optimal policy or approximation bound. Include cases where the most
informative measurement arrives too late to prefetch.

## Required CPU artifact

Implement finite-state processes with controllable conditional information and
exact Bayes-risk calculation. Compare routes-only, scalar metrics, CountSketch,
random Fourier features, recurrent learned summaries, and an oracle. Include
adversarial collisions, high-entropy-but-conditionally-predictable routes, and
distribution shift. Report bits/floats, prediction risk, calibration, and a
symbolic system-value curve rather than invented GPU speed.

## Engine decision and kill criterion

Return the exact state schema, update equations, device/host placement, training
distribution, and mismatch fingerprint. Kill a representation if it cannot
distinguish a supplied adversarial family, if its claimed advantage disappears
after conditioning on existing features, or if acquiring it finishes after the
read/verification decision it is supposed to improve.
