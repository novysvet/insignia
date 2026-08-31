# Problem 12: Joint adaptive compute from causal difficulty evidence

Repository: https://github.com/novysvet/insignia  
Reference branch/commit: `codex/glm53-dflash2-4070-super` at `0740c63`

## Engine context

"Harder tasks need more experts" motivates adaptive Top-k, but expert count is
only one control. Per block Insignia could also choose DFlash draft length,
verification depth, exact versus approximate checks, expert precision,
prefetch aggressiveness, capture features, checkpoint layout, and fallback.
Previous logits and causal state provide evidence about difficulty. These
actions interact: changing expert count changes logits and future routes;
prefetch changes cache state; draft length changes both amortization and state
archive cost. A collection of independent threshold heuristics is unlikely to
be globally optimal.

This problem seeks a mathematical controller, not a larger unprincipled neural
network. It can be solved on a generic CPU using a symbolic/synthetic model.

## Formal setup

At decision epoch `t`, hidden state `D_t` describes latent difficulty and
machine conditions. Observation `O_t` contains causal logits-derived features,
routes, cache occupancy, previous acceptance, and timing counters. Action
`A_t` selects a tuple

`(expert_k, draft_k, verify_policy, precision, prefetch_budget, measurement)`.

The transition depends on the action. Reward is committed tokens minus weighted
latency/bytes; constraint cost is quality loss or catastrophic-failure risk.
The process is semi-Markov because actions have unequal durations and commit a
random number of tokens.

## Main problem

1. Formulate a constrained partially observed semi-Markov decision process whose
   objective is long-run committed tokens per second and whose quality
   constraint corresponds to a pathwise or risk-sensitive requirement. State
   why an average per-row classification loss is insufficient.
2. Determine conditions under which the action tuple decomposes or admits an
   index/threshold policy. Give explicit counterexamples showing pairwise
   interactions that destroy separability—especially expert count versus
   future routes and draft length versus checkpoint cost.
3. Derive an exact solution for a finite discretization and a scalable
   approximation with a performance or constraint-violation bound. Candidate
   tools include Lagrangian occupation measures, belief-state compression,
   robust dynamic programming, Lyapunov drift, or primal-dual online learning.
4. Use previous logits as observations through a specified likelihood model.
   Prove how posterior difficulty changes the optimal action, and find the value
   of information of one additional exact metric before acting.
5. Handle model error and distribution shift. Construct a robust/safe policy
   that reverts to exact inference when its uncertainty set or risk certificate
   is violated. Quantify the throughput price of safety.

## Hard extensions

- Add a request-level hard/easy latent variable and token-level regime changes.
  Determine whether the optimal controller has hysteresis.
- The quality budget is global PPL plus a separate rare hard-answer constraint.
  Analyze whether a single Lagrange multiplier can enforce both.
- Permit online learning but charge exact shadow evaluations and exploration
  against throughput. Give an anytime-safe exploration rule.
- Couple two storage devices and one GPU through queues, making the environment
  state partly controlled by earlier prefetch actions.

## Required CPU artifact

Implement an exact finite-state solver and a simulation environment with
endogenous cache state, stochastic acceptance, hidden difficulty, and correlated
quality loss. Compare fixed settings, independent thresholds, myopic value of
information, and your joint controller. Include a constructed case where every
individual heuristic looks locally sensible but their combination is unstable
or violates the quality budget.

## Required engine handoff

Provide the smallest observation/action schema, offline data requirements,
online state update, safety envelope, and a table of which decisions must be
made before which GPU/I/O deadline. Unknown transition parameters must remain
symbolic or be estimated with uncertainty. Kill the controller if its inference
and measurement cost consumes the predicted saving, if safe exploration lacks
overlap, or if no robust policy materially beats the best fixed configuration.
