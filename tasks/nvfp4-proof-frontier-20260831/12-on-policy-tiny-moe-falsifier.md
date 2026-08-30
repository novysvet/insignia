# Problem 12: A selectively safe on-policy tiny-MoE falsifier

Repository: https://github.com/novysvet/insignia  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware required for this assignment: none; the reference must train on CPU.

## Mission

Design the mathematical training and deployment protocol for a tiny, roughly
99%-sparse MoE that decides whether an aggressive inference shortcut is safe
for the current row. It may consume previous target logits, router margins,
hidden sketches, cache/residency state, DFlash draft logits and acceptance
history, and prompt-position features. Its output is a selective action:
approximate, exact fallback, or request more evidence.

Candidate engineering ingredients include native FP8, MLA, mHC, attention
residuals, Stable LatentMoE, Muon-Clip-style optimization, and Dion3-style
updates. Treat these as hypotheses, not proof by acronym: establish which
properties matter and ablate the rest.

## Fixed safety/quality contract

The target has 42 discontinuous Top-8 routers, so rare false-safe decisions can
cascade. Every approximate engine arm reports MSE, relative-L2, cosine,
forward/reverse KL, JS, same-prefix PPL, Top-1/routes, DFlash acceptance, and
readable hard outputs. A +3.5% same-prefix PPL ceiling is allowed only while
hard outputs remain useful. Exact fallback is always available but costs time.
Training examples are on-policy teacher traces: changing the falsifier changes
which future examples are visited.

## Required mathematics

1. Define the loss as saved wall time subject to a selective-risk constraint,
   including asymmetric catastrophic false-safe cost and evidence-request cost.
2. Give a finite-sample upper confidence bound on deployment false-safe risk.
   Address prompt clustering, temporal dependence, multiple shortcuts, and
   policy-induced covariate shift. Plain IID conformal prediction is only a
   baseline.
3. Derive an on-policy data-collection/exploration scheme with bounded risk and
   adequate support. Analyze importance weighting, doubly robust estimation,
   or sequential e-values; state where each breaks.
4. Propose a sparse architecture whose compute and memory overhead are priced.
   Prove or test whether expert sparsity helps conditional coverage rather than
   merely fitting the training set.
5. Construct adversarial collisions: identical supplied features but opposite
   shortcut safety. Use them to prove the irreducible abstention rate for the
   chosen context.

## CPU artifact and completion

Submit `selective_falsifier.py` with a synthetic on-policy teacher, tiny sparse
model, optimizer variants, exact fallback policy, calibration, and sequential
deployment simulator. Include family-held-out and regime-shift splits, risk-
coverage curves, expected speed value after model overhead, effective sample
size, abstention, and worst-family false-safe bounds.

The task is complete only if the stated confidence bound attains nominal
coverage in repeated held-out simulations, every feature is causal at decision
time, overhead is in the value calculation, and the report gives a trace
schema plus promotion/rollback thresholds. An impossibility result proving
that the available context cannot support useful coverage is valid.
