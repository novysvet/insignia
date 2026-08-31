# Problem 8: An anytime-valid falsifier under its own policy feedback

Repository: https://github.com/novysvet/insignia  
Reference branch/commit: `codex/glm53-dflash2-4070-super` at `0740c63`

## Engine context

Insignia is developing a tiny, highly sparse causal MoE controller that sees
previous target and draft logits, routing/cache state, hidden summaries, and
history. It may decide whether an approximate DFlash verification, reduced
expert count, or other fast path is safe. The controller can abstain and force
exact work. Once deployed, its actions change which histories are visited,
which labels are observed, cache state, and future acceptance. Ordinary IID
validation and teacher-forced calibration do not survive this feedback loop.

The user tolerates up to +3.5% same-prefix perplexity only when genuinely hard
free-running answers remain useful. A falsifier must therefore control rare,
high-cost failures rather than maximize average classification accuracy.

## Formal setup

At round `t`, causal context `X_t` is observed. A policy chooses `A_t` from
`{fast, exact}` or a larger action set. Potential outcome `Y_t(a)` measures a
failure severity or whether the fast action would have violated an exact
verification criterion. `Y_t(fast)` may be unobserved when the policy chooses
exact unless the engine performs counterfactual shadow computation. Actions
affect the law of `X_{t+1}`. The controller emits a score and may abstain.

Define a selective-risk target precisely, such as

`E[Y_t(fast) | fast selected] <= epsilon`

uniformly over time, or a bound on cumulative catastrophic loss with probability
`1-delta`. Distinguish marginal, conditional, and pathwise guarantees.

## Main problem

1. Prove an impossibility theorem without overlap or shadow labels: construct
   two environments observationally identical under the logging policy but with
   different deployed fast-path risk.
2. Under explicit overlap, bounded importance ratios, or randomized auditing,
   construct an anytime-valid estimator/confidence sequence for selective risk
   under adaptive data. Delayed labels and abstentions must be handled.
3. Optimize the audit probability and abstention threshold to maximize saved
   inference time subject to the risk guarantee. Exact audits have context-
   dependent cost and alter caches; include that cost rather than counting only
   label frequency.
4. Extend to nonstationarity. Give a guarantee under bounded drift, change
   points, or a safe reset mechanism. State what is lost relative to the
   stationary case.
5. The model itself is selected and tuned using logged data. Prevent reuse of
   calibration evidence from invalidating coverage. Analyze held-out epochs,
   online e-values, reusable holdouts, or another valid mechanism.

## Hard extensions

- Use severity-weighted risk where one hard-prompt collapse costs far more than
  a benign Top-1 mismatch.
- Couple decisions over a DFlash block: an early risky row changes the meaning
  of all later rows, so row labels are not independent.
- Allow a tiny MoE to request additional features or exact partial computation
  before deciding. Solve this as sequential feature acquisition with a risk
  certificate.

## Required CPU artifact

Build a synthetic controlled process with covariate drift, policy feedback,
selective labels, delayed outcomes, and rare catastrophic modes. Compare naive
calibration, split conformal, importance weighting, and your method. Report
coverage/risk violations, abstention, exact-audit cost, and throughput reward.
Include an indistinguishable-environment test that demonstrates the impossibility
result.

## Engine acceptance rule

Return the exact random-audit protocol, logged propensities, state fingerprint,
confidence update, threshold policy, and reset condition. The default action
must be exact whenever the certificate is absent or invalid. Kill any proposal
that assumes labels for actions never taken, treats on-policy rows as IID, or
uses a nominal validation percentile as an anytime guarantee.
