# Problem 3: Optimal target-layer observations for a block drafter

Repository: https://github.com/novysvet/insignia  
Reference branch/commit: `codex/glm53-dflash2-4070-super` at `0740c63`

## Engine context

Insignia's DFlash2 drafter has five layers and consumes target-model captures
currently taken after layers 5, 14, 24, 33, and 42 of a 45-layer model. Captures
cost memory traffic and synchronization but provide information about future
target tokens and verification acceptance. The current locations are a design
choice, not a theorem. A better subset, a compressed capture, or an adaptive
sensor schedule could improve acceptance without changing target correctness.

This assignment is mathematical and CPU-only. Treat layer activations as
random variables; do not assume access to model weights or private traces.

## Formal setup

Let `H_0,...,H_L` be the hidden-state process of a fixed target transformer,
with `L=45`. Let `Y` denote a future object the drafter must predict: one token,
a block of up to eight tokens, target logits, or the exact verification-prefix
length. Choosing capture set `S subset {1,...,L-1}` reveals encoded observations
`Z_l = phi_l(H_l)` for `l in S`. A capture has byte cost `b_l`, latency cost
`c_l`, and optionally an encoder distortion/rank parameter. The drafter always
sees the prompt and its own causal state `D`.

Choose a loss that actually reflects the engine, such as conditional log loss,
Bayes block-error probability, or expected committed tokens per unit time. Do
not substitute unconditional correlation without justification.

## Main problem

1. Characterize when the value
   `f(S)=I(Y; Z_S | D)` is monotone submodular. Prove the claim under explicit
   graphical-model assumptions and give a smallest counterexample when hidden
   layers have synergy or redundant deterministic transforms.
2. For the non-submodular case, derive an algorithm with a stated guarantee
   under a defensible weaker property (submodularity ratio, curvature, bounded
   interaction order, adaptive submodularity, or another condition). Include
   nonuniform byte and latency costs.
3. Replace mutual information with the end-to-end renewal reward
   `accepted_tokens / round_time`. Explain why maximizing information can lose
   throughput and derive either an exact dynamic program or a surrogate with a
   regret/approximation bound.
4. Allow the capture set to be selected causally per request from cheap early
   statistics. Formulate the adaptive sensing problem and prove when adaptivity
   has positive value. Construct an example where a fixed five-layer set is
   arbitrarily worse, and one where adaptive selection cannot help.
5. Establish a lower bound on the number of capture bits needed to attain a
   target block-error or acceptance rate, using Fano, rate-distortion, or a
   sharper task-specific argument.

## Required CPU artifact

Implement exact enumeration for small layered Bayesian networks and a scalable
solver for synthetic `L=45` instances. It must compare the current five
locations, greedy information gain, cost-aware greedy, your proposed method,
and the true optimum on small cases. Include XOR synergy, near-duplicate
captures, a change-point/depth regime, and a case whose most informative layer
is too expensive in latency.

## Deliverable standard

Return a theorem/counterexample set, algorithm, tests, and a schema for the
minimum trace fields needed to estimate the objective from real requests.
Specify a kill criterion: for example, if the maximum statistically defensible
acceptance gain cannot repay capture D2D/H2D traffic and drafter compute. Never
claim that layers 5/14/24/33/42 are suboptimal without data; the mathematical
result must say exactly what evidence would decide it.
