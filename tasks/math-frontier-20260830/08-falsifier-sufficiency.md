# Problem 8: sufficient causal state for a learned performance falsifier

Repository: https://github.com/novysvet/insignia
Hardware required: none

## Existing observation

For each target layer/verify row, a tiny causal controller receives:

- 16 scalar logit statistics, including previous-round temporal change;
- three 64-dimensional CountSketches of target prior, current draft, and their
  difference;
- Top-32 router IDs/logits/choice scores and residency;
- current Top-8 weights, route summaries, multiplicities, and overlap history;
- a 64-wide hidden CountSketch;
- block/layer/row metadata and a 64-wide recurrent MLA latent.

It predicts immediate damage, 8/16/32-horizon forced/free failure, collapse,
an 8x8 contribution Gram, action risk/cost, and DFlash acceptance. The native
runtime budget is about 3.185 ms per four-row round. Training data are on-policy
and dependent; easy average losses are not the target.

## Main problem

Determine what information is sufficient for safe action selection and how to
calibrate the resulting multi-horizon risk under policy shift.

Formalize latent environment state `Z_t`, observation `X_t`, action `A_t`, and
failure process `Y_t`. A useful result may be:

- a predictive-state representation theorem;
- an information-theoretic lower bound showing the current observation cannot
  distinguish two states requiring different actions;
- a minimal augmentation that restores approximate sufficiency;
- a finite-sample risk controller using e-values, conformal methods, martingale
  bounds, or distributionally robust optimization.

## Required challenges

1. Same-prefix PPL labels do not identify free-running collapse. Exhibit two
   processes with identical forced observations/labels and different free-run
   risk.
2. Data are collected by prior policies, so counterfactual action damage is
   missing. State identifiability/positivity assumptions explicitly.
3. Failures are rare and clustered by prompt family; ordinary IID calibration
   is invalid.
4. The controller changes caching/routing and therefore its future input
   distribution.
5. False negatives are more costly than false positives, but exact fallback
   has measurable latency.

## Strong deliverable

Design a sequential decision rule that consumes calibrated lower/upper risk
bounds for actions `k=3,...,8`, spends a global failure budget over a generation,
and falls back to exact execution when the uncertainty set is too wide. Prove a
trajectory-level statement under clearly named assumptions.

## Deliverables

- Sufficiency/impossibility theorem and explicit indistinguishable-state
  constructions.
- Calibration method robust to temporal dependence and prompt-family holdout.
- CPU simulation showing coverage, false-safe rate, saved I/O, and sensitivity
  to policy shift.
- A ranked list of additional causal features, each justified by conditional
  information gain rather than architectural fashion.
