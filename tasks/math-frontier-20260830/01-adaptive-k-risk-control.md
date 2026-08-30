# Problem 1: adaptive expert count as risk-constrained stochastic control

Repository: https://github.com/novysvet/insignia
Hardware required: none

## System

At target sparse layer `l in {1,...,42}` and speculative row
`r in {1,...,R}`, `R <= 8`, the router supplies eight ordered expert terms

```text
v[l,r,i] = w[l,r,i] e[l,r,i],  i=1,...,8.
```

Executing only the first `k` terms, `k in {3,...,8}`, saves transfer and matrix
work but perturbs a recurrent model. The action changes later hidden states,
routes, DFlash acceptance, and therefore future observations. A controller sees
only causal state `X[l,r]`: route margins, cache residency, previous target and
draft-logit sketches, hidden sketches, prior actions, and compressed history.

Let `C_t(k)` be token-time cost, including union-dependent NVMe/H2D work, and
let `Y_t^h(k_1,...,k_t)` be a downstream failure measure at horizon
`h in {1,8,16,32}`. Failure may combine Top-1 flip, log-MSE, KL/JS, repetition,
or entropy collapse. The policy must respect the user's hard condition:

```text
P(max_h Y_t^h > tau_h for some t <= T) <= delta,
```

while minimizing expected wall time per accepted output token.

## Main problem

Construct a causal policy with a finite-sample safety statement under
on-policy, temporally dependent data. The policy should exploit extra CPU
compute and may maintain a belief state, uncertainty set, or risk budget. It
must not assume independent layer errors or train/test stationarity.

At least one of the following must be achieved:

1. derive an exact dynamic program for a nontrivial model class and prove when
   an optimal policy is a threshold in posterior failure probability;
2. give a distributionally robust or conformal policy whose trajectory-level
   violation probability is bounded with finite samples;
3. prove an impossibility theorem showing what additional assumptions are
   necessary, then solve the strongest defensible restricted problem;
4. construct a counterexample in which every myopic `quality loss / bytes
   saved` rule violates the global constraint by an arbitrarily large factor.

## Required complications

- Costs are set-valued: two rows choosing the same expert pay one layer-group
  transfer, not two.
- Errors alter later routing, so `Y_t` is policy-dependent.
- Rare hard-prompt failures matter more than mean loss.
- Exact `k=8` is always available as a safe fallback, but an unnecessary
  fallback has measurable cost.
- The learned risk model may be miscalibrated under a new prompt family.

## Concrete finite instance

Provide a solver for `42 x 4` decisions with action set `{3,...,8}`. A useful
synthetic generator should include:

```text
latent difficulty z_t = rho z_(t-1) + epsilon_t,
route-set overlap controlled by q,
cache state with capacity C,
damage increments with both smooth and route-boundary components,
acceptance length A_t in {0,...,4}.
```

Sweep `rho`, tail heaviness, calibration error, overlap `q`, and cache pressure.
Compare exact DP where feasible, a proposed scalable policy, fixed Top-4/6/8,
and a myopic controller.

## Deliverables

- Formal objective and filtration.
- Theorem/proof or impossibility/counterexample.
- CPU reference solver and reproducible synthetic experiments.
- A mapping from the mathematical risk budget to measured MSE/cosine/KL/JS,
  +3.5% PPL, and free-run collapse gates.
- A clear statement of which quantities must be learned from new on-policy
  traces before deployment.
