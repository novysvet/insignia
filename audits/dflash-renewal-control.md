# DFlash drafting, verification, and retry as renewal control

## Status

This note supplies the mathematical formulation, threshold results, unrestricted counterexample, exact finite-state solver, scalable controller, safe fallback rule, and hardware-free synthetic evaluation requested by `tasks/math-frontier-20260830/07-dflash-renewal-control.md`.

Code and reproducible outputs:

- `tools/dflash_renewal_control.py`: exact time-normalized occupation-measure solver and theorem helpers.
- `tools/test_dflash_renewal_control.py`: hardware-free tests, including zero-token renewals and trajectory chance constraints.
- `tools/evaluate_dflash_renewal_control.py`: adaptive exact and risk-gated approximate policies plus synthetic environments.
- `scratch/dflash-renewal-control/summary.csv`: evaluation summary.
- `scratch/dflash-renewal-control/exact-lp-sweep.csv`: small-state constrained LP sweep.
- `scratch/dflash-renewal-control/demo-model.json`: example finite SMDP schema.

The current engine already contains the relevant mechanisms:

- row-sequential verification with stop on first mismatch;
- batch verification with recurrent snapshot and rollback;
- exact retry of the identical candidate block after a target-logit drop;
- online survival-hazard and per-record cost estimates for adaptive width;
- cache demotion for rejected-tail records.

The added engine counter separates speculative accepted reward from committed output. An empty DFlash round commits the known exact fallback token but contributes zero to accepted-draft reward.

## 1. Decision epochs, rewards, and clocks

Let round `n` begin at an exact accepted anchor. A round-level action is a contingent option, not only a scalar width. It contains:

\[
U_n=(R_n, e_{n,1:R_n}, m_n, \sigma_n, \kappa_n),
\]

where:

- \(R_n\in\{1,\ldots,8\}\) is the proposed and maximum verified width;
- \(e_{n,r}\) selects the expert count, subset, or cache-aware substitute action for row \(r\);
- \(m_n\) selects batch or row-sequential verification;
- \(\sigma_n\) is an observation-dependent stopping rule;
- \(\kappa_n\) is an observation-dependent exact-retry rule.

The option can inspect observations generated inside the round, such as target logits after row \(r\), cache completion events, and the measured union-I/O latency.

Define the round outcome:

- \(A_n\in\{0,\ldots,R_n\}\): accepted draft-prefix length;
- \(Y_n\ge A_n\): committed output tokens in the round;
- \(T_n>0\): total wall time from round start through commit and recurrent-state restoration;
- \(D_n\ge0\): quality damage or divergence cost;
- \(E_n\in\{0,1\}\): a generation trajectory ended in this transition;
- \(V_n\in\{0,1\}\): that completed trajectory violated its quality contract.

For the current DFlash path, an empty speculative round has \(A_n=0\), then emits the known exact target token. Thus \(Y_n=1\) for that round. A normal accepted prefix has \(Y_n=A_n\). The two counters answer different questions:

\[
\text{speculative accepted throughput} = \frac{\text{accepted draft tokens}}{\text{wall time}},
\]

\[
\text{end-to-end output throughput} = \frac{\text{committed output tokens}}{\text{wall time}}.
\]

The main objective in the task is the first quantity. End-to-end latency remains an essential report metric.

## 2. Renewal-reward objective

### 2.1 IID renewal cycles

Suppose first that rounds are IID under a fixed policy, with \(0<E[T_1]<\infty\) and \(0\le A_1\le8\). Let

\[
S_k=\sum_{n=1}^k T_n,\qquad
N(t)=\max\{k:S_k\le t\}.
\]

The cumulative accepted reward by time \(t\) is

\[
R(t)=\sum_{n=1}^{N(t)} A_n+R_{\mathrm{res}}(t),
\]

where the residual reward from an unfinished round is bounded by 8. The renewal-reward theorem gives

\[
\boxed{
\lim_{t\to\infty}\frac{R(t)}{t}
=\frac{E[A]}{E[T]}
}
\quad\text{almost surely and in mean.}
\]

#### Proof

The strong law gives \(S_k/k\to E[T]\), hence \(N(t)/t\to1/E[T]\). It also gives

\[
\frac{1}{N(t)}\sum_{n=1}^{N(t)}A_n\to E[A].
\]

Multiplying the two limits yields \(E[A]/E[T]\). Since \(|R_{\mathrm{res}}(t)|\le8\), its contribution divided by \(t\) vanishes. Nothing in this proof requires \(P(A=0)=0\). QED.

The following alternatives are incorrect:

\[
E[A/T], \qquad \frac{1}{E[T/A]}.
\]

The second is undefined on empty rounds. The first weights short rounds differently from the wall-time process. For example, let \((A,T)=(0,1)\) or \((2,3)\), each with probability one half. The correct throughput is \(E[A]/E[T]=1/2\) token per time unit. The average of per-round ratios is \(E[A/T]=1/3\).

### 2.2 Markov renewal and semi-Markov control

DFlash rounds are not IID. Cache state, recurrent trajectory state, hardware queues, prompt difficulty, and quality debt persist between rounds. Let \(S_n\) be the state at decision epoch \(n\). Under a stationary policy \(\pi\), suppose the embedded decision chain is positive recurrent with stationary decision-epoch distribution \(\eta_\pi\). Define

\[
a(s,u)=E[A_n\mid S_n=s,U_n=u],
\]

\[
\tau(s,u)=E[T_n\mid S_n=s,U_n=u].
\]

The long-run accepted throughput is

\[
\boxed{
\rho(\pi)=
\frac{\sum_s\eta_\pi(s)\sum_u\pi(u\mid s)a(s,u)}
{\sum_s\eta_\pi(s)\sum_u\pi(u\mid s)\tau(s,u)}.
}
\]

This is the Markov-renewal version of the ratio of expectations. Zero-reward rounds remain ordinary transitions with positive sojourn time.

### 2.3 Time-normalized occupation measure

For a finite SMDP, let \(x(s,u)\) be action starts per millisecond. The exact constrained problem is the linear program

\[
\max_{x\ge0}\quad \sum_{s,u}x(s,u)a(s,u)
\]

subject to flow conservation

\[
\sum_u x(j,u)
=\sum_{s,u}x(s,u)P(j\mid s,u),\quad\forall j,
\]

time normalization

\[
\sum_{s,u}x(s,u)\tau(s,u)=1,
\]

and quality constraints described below. The objective has units accepted tokens per millisecond. Multiplication by 1000 gives accepted tokens per second.

The solver in `tools/dflash_renewal_control.py` implements this LP with SciPy HiGHS. It reports accepted and committed reward separately. It is exact for a finite communicating SMDP. A discretized belief-state POMDP becomes such an SMDP. In a multichain model, the LP can choose any closed recurrent class; a caller with a fixed initial state must restrict the model to its reachable communicating class or use an initial-state constrained formulation.

## 3. Trajectory-level quality

A per-round approximation penalty is not a trajectory-level contract. A locally small logit change can alter future recurrent state, future routers, and eventual text behavior. The state must retain the information needed to decide whether a completed sequence violates the contract.

### 3.1 Completed-trajectory chance constraint

Augment the state with:

- a finite quality monitor state \(z\), including an absorbing `violated` flag;
- remaining generation length or an EOS model;
- a reset transition at trajectory completion.

Let

\[
e(s,u)=E[E_n\mid s,u],\qquad
v(s,u)=E[V_n\mid s,u].
\]

The completed-trajectory quality constraint

\[
P(\text{trajectory violates})\le\alpha
\]

becomes

\[
\boxed{
\sum_{s,u}x(s,u)\bigl(v(s,u)-\alpha e(s,u)\bigr)\le0.
}
\]

Both numerator and denominator are rates per wall time, so their ratio is the violation probability among completed trajectories. This linear constraint is implemented by the exact solver through `trajectory_end`, `trajectory_violation`, and `max_trajectory_violation_probability`.

A finite quality automaton can represent a thresholded cumulative divergence, an external judge failure, a collapse event, or a conjunction of checks. A richer learned evaluator requires belief-state approximation.

### 3.2 Average damage constraint

A softer contract can bound expected damage per accepted token:

\[
\frac{\sum x(s,u)d(s,u)}{\sum x(s,u)a(s,u)}\le\delta.
\]

The linearized form is

\[
\sum_{s,u}x(s,u)\bigl(d(s,u)-\delta a(s,u)\bigr)\le0.
\]

The exact solver supports this form as well. It should not be described as a trajectory chance guarantee.

### 3.3 Infinite-stream caution

For an unbounded stream, a fixed positive harmful-commit probability per approximate round eventually violates an `ever bad` contract with probability one. An anytime chance guarantee needs summable risk allocations, a finite generation boundary, or eventual exact operation. Section 10 gives a summable rule.

## 4. POMDP state and approximation-dependent dynamics

The fully observed physical state can be written as

\[
X_n=(H_n,G_n,C_n,Q_n,Z_n,B_n,L_n).
\]

Here:

- \(H_n\): exact target and drafter recurrent state at the anchor;
- \(G_n\): latent prompt or acceptance regime;
- \(C_n\): expert residency and recency state across VRAM, host cache, and in-flight reads;
- \(Q_n\): NVMe, host-to-device, and CUDA queue state;
- \(Z_n\): quality-monitor state;
- \(B_n\): remaining trajectory risk budget;
- \(L_n\): recent timing and observation-calibration state.

The engine observes only a function of this state. Let \(b_n=P(X_n\mid\mathcal H_n)\) be the belief state. A round option induces a semi-Markov kernel

\[
P(A_n,Y_n,T_n,D_n,O_n,b_{n+1}\mid b_n,U_n).
\]

Approximation must enter this kernel in two places:

1. It changes the immediate distribution of \(A_n\), target logits, and latency.
2. It changes \(b_{n+1}\) through recurrent divergence, router changes, cache residency, and the inferred latent regime.

A model that subtracts a fixed quality penalty from the current round but reuses the exact next-state kernel is structurally wrong. The synthetic model intentionally makes an un-retried approximate round more likely to enter a hard regime and less likely to retain exact-cache residency.

## 5. Average-cost Bellman and Dinkelbach form

For a trial throughput \(\rho\), average damage multiplier \(\lambda\ge0\), and trajectory multiplier \(\nu\ge0\), define the transformed one-transition reward

\[
\begin{aligned}
r_{\rho,\lambda,\nu}
={}& A-\rho T-\lambda(D-\delta A)-\nu(V-\alpha E)\\
={}&(1+\lambda\delta)A-\rho T-\lambda D-\nu V+\nu\alpha E.
\end{aligned}
\]

The belief-SMDP relative-value equation is

\[
h(b)=\max_U E\left[r_{\rho,\lambda,\nu}+h(b')\mid b,U\right]-g.
\]

At the optimal throughput and active multipliers, the maximal long-run transformed average is zero. This gives a practical controller architecture:

- Dinkelbach or stochastic root updates for \(\rho\);
- dual updates for \(\lambda\) and \(\nu\);
- dynamic programming, rollout, or MPC for the transformed additive objective.

The threshold proofs below operate on the marginal transformed value after sunk costs have been paid.

## 6. Restricted model: retry is a state-conditioned threshold

Consider a batch first pass that has already produced target observations. Let \(c\) summarize the known cache and queue state after that pass. Let \(B\in\{0,1\}\) denote a latent `bad approximation` condition. A scalar target observation \(z\) produces

\[
q(z,c)=P(B=1\mid z,c).
\]

The controller now chooses `commit first pass` or `restore the pre-round snapshot and exact-retry once`.

Let \(k_R(c)\) be the marginal retry time from the current post-first-pass state. Let \(G_b(c)\) be the transformed benefit of retry over commit in latent condition \(b\), excluding retry time. It can include recovered accepted reward, quality-risk reduction, and future-state value. The retry advantage is

\[
\Delta_R(q,c)
=(1-q)G_0(c)+qG_1(c)-\rho k_R(c).
\]

### Theorem 1: posterior retry threshold

Assume:

1. the current cache state \(c\) is fixed at the decision;
2. one exact retry is allowed;
3. \(G_1(c)>G_0(c)\);
4. no other hidden variable changes the marginal comparison once \((q,c)\) is given.

Then exact retry is optimal exactly when

\[
\boxed{
q\ge \theta_R(c)
=\frac{\rho k_R(c)-G_0(c)}{G_1(c)-G_0(c)}.
}
\]

Values \(\theta_R(c)\le0\) mean always retry. Values \(\theta_R(c)\ge1\) mean never retry.

#### Proof

\(\Delta_R(q,c)\) is affine in \(q\), with positive slope \(G_1(c)-G_0(c)\). It crosses zero at the displayed value. QED.

### Corollary 1: observation threshold

If the observation family has a monotone likelihood ratio, then \(q(z,c)\) is monotone in \(z\). The retry rule is a threshold in \(z\), with a cache-conditioned threshold \(z_R^*(c)\).

The current largest adjacent top-1 probability drop can support this theorem only after empirical validation of monotone likelihood ratio or a monotone calibrated posterior. A raw drop threshold alone does not establish optimality.

## 7. Restricted model: stop or continue is a threshold

Suppose rows \(1,\ldots,r\) have survived and row \(r+1\) is available. Let

\[
p=P(A\ge r+1\mid A\ge r,\text{current observations}).
\]

All prior draft and verification time is sunk. Let \(m_{r+1}(c_r)\) be the marginal time to verify the next row from the current cache state. Let \(G_s(c_r)\) be the transformed continuation gain if the row succeeds, excluding marginal time. Let \(G_f(c_r)\) be the gain if it fails. The advantage of continuing over stopping is

\[
\Delta_C(p,c_r)
=pG_s(c_r)+(1-p)G_f(c_r)-\rho m_{r+1}(c_r).
\]

### Theorem 2: stop/continue threshold

If \(G_s(c_r)>G_f(c_r)\), continue exactly when

\[
\boxed{
p\ge\theta_C(c_r)
=\frac{\rho m_{r+1}(c_r)-G_f(c_r)}{G_s(c_r)-G_f(c_r)}.}
\]

#### Proof

The advantage is affine in \(p\), with positive slope. QED.

A common special case has one immediate accepted-token reward on success, no extra reward on failure, and fixed continuation values. Then the threshold compares the conditional survival probability with marginal verification time measured in units of the current throughput shadow price.

### Sequential versus batch

Let

\[
S_i=P(A\ge i),\qquad i=1,\ldots,R.
\]

With DFlash's carried exact `truth0`, an empty round is rejected before target verification. Row-sequential target work executes \(A\) rows when \(A\ge1\), and zero rows when \(A=0\). If \(m_i(C_{i-1})\) is the state-dependent marginal cost of row \(i\), then

\[
E[T_{\mathrm{seq}}]
=D+(1-S_1)F+\sum_{i=1}^R S_i E[m_i(C_{i-1})\mid A\ge i].
\]

A batch pass has

\[
E[T_{\mathrm{batch}}]
=D+(1-S_1)F+S_1 E[m_{\mathrm{batch}}(C_0,U_R)]
+P(A<R)E[c_{\mathrm{restore}}].
\]

The controller compares transformed values, not only expected rows. Sequential verification saves rejected-tail I/O. Batch verification amortizes target execution and union construction. The checked-in empirical result that batch wins at high acceptance and sequential can win at low acceptance follows directly from these formulas.

A scalar threshold in \(E[A]\) exists only under additional ordering assumptions on row costs and the acceptance family. Cache-coupled costs can destroy it.

### Per-row expert counts

For a finite expert-action set, the exact choice is the action with the largest transformed Bellman value. A threshold in router mass or posterior quality follows if ordered expert actions satisfy increasing differences: the marginal quality benefit of adding an expert must increase monotonically with the scalar uncertainty signal, and marginal latency must not reverse that order. Cache-aware substitutes generally violate these assumptions, so the unrestricted policy should retain the cache state and compare actions directly.

## 8. Unrestricted nonthreshold counterexample

A global threshold in posterior collapse risk does not exist when the same observation also predicts marginal retry cost through cache state.

Set the current throughput shadow price to \(\rho=1\) token per millisecond. Exact retry has zero benefit in a good state and one token of benefit in a bad state. Let the scalar signal order posterior bad probability upward:

| signal | posterior bad \(q\) | marginal retry cost \(k_R\) | advantage \(q-k_R\) | decision |
|---|---:|---:|---:|---|
| low | 0.20 | 0.10 | 0.10 | retry |
| middle | 0.50 | 0.80 | -0.30 | commit |
| high | 0.80 | 0.10 | 0.70 | retry |

The optimal action pattern is `retry, commit, retry`. No threshold in the ordered signal or in \(q\) can represent it.

The costs are defensible for DFlash. The middle observation can correspond to a candidate whose exact expert union is largely absent from cache, while low and high observations correspond to unions mostly warmed by the first pass. Once cache state is exposed, Theorem 1 supplies a threshold conditional on cache state. Hiding cache state creates the nonthreshold policy.

`nonthreshold_counterexample()` in the solver returns this table and the unit test checks its action pattern.

## 9. Sunk costs and marginal record sets

After row \(r\), define:

- \(V_r\): exact records resident in VRAM;
- \(H_r\): exact records present in the host cache;
- \(I_r\): records with in-flight reads that will complete before use;
- \(E_{r+1}\): exact record set needed by row \(r+1\);
- \(E^*\): exact record union for a full retry.

The row-\(r+1\) disk-miss set is

\[
M^{\mathrm{disk}}_{r+1}
=E_{r+1}\setminus(V_r\cup H_r\cup I_r).
\]

The host-to-device miss set is

\[
M^{\mathrm{h2d}}_{r+1}
=E_{r+1}\setminus V_r.
\]

A queue-aware marginal-cost model is

\[
\begin{aligned}
m_{r+1}(C_r)={}&c_{\mathrm{router}}+c_{\mathrm{kernel}}\\
&+\Phi_{\mathrm{disk}}\left(B\lvert M^{\mathrm{disk}}_{r+1}\rvert,Q_r\right)\\
&+\Phi_{\mathrm{h2d}}\left(B\lvert M^{\mathrm{h2d}}_{r+1}\rvert,Q_r\right)\\
&-\Phi_{\mathrm{overlap}}(Q_r),
\end{aligned}
\]

where \(B\) is bytes per record. The functions must reflect queue depth, coalescing, and overlap. Multiplying every missing record by one stationary scalar is a first approximation, not a permanent model.

After an approximate first pass, exact retry has marginal sets

\[
M^{\mathrm{disk}}_R
=E^*\setminus(V_R\cup H_R\cup I_R),
\]

\[
M^{\mathrm{h2d}}_R
=E^*\setminus V_R.
\]

The retry comparison includes only:

- restore cost from the pre-round recurrent snapshot;
- marginal exact records still missing;
- exact target execution;
- any future-state difference.

First-pass drafting, controller execution, target work, and records already loaded are sunk. Charging them again in the retry decision biases against retry. Ignoring warmed exact records biases toward overestimating retry cost.

## 10. Safe fallback with a competitive cost bound

### 10.1 Strict subset rule

For a quality-uncertain approximate attempt:

1. Save the exact pre-round recurrent snapshot.
2. Restrict every approximate expert set to a subset of the exact expert set. Do not use substitutes outside the exact union in the strict mode.
3. Pin first-pass exact records until the retry decision, or account for any destructive eviction.
4. Run one approximate first pass.
5. Commit only if an anytime-valid quality-risk upper bound is inside the remaining trajectory budget and the observation is in-distribution.
6. Otherwise restore the snapshot and run the identical candidate block exactly once with all exact experts.
7. Never retry recursively. An exact rejection uses the ordinary exact fallback path.

Let \(T_E(C)\) be the always-exact round time from initial cache state \(C\). Let \(T_Q(C)\) be the subset first-pass time. Under monotone I/O and compute cost,

\[
T_Q(C)\le T_E(C).
\]

Let \(C'\) be the post-first-pass cache. With pinned subset records and no destructive eviction,

\[
T_E(C')\le T_E(C).
\]

If \(H\) is controller, snapshot, restore, and decision overhead not already included in those terms, then the worst retrying round satisfies

\[
T_{\mathrm{safe}}
\le T_Q(C)+T_E(C')+H
\le2T_E(C)+H.
\]

Therefore

\[
\boxed{
\frac{T_{\mathrm{safe}}}{T_E(C)}
\le2+\frac{H}{T_E(C)}.
}
\]

For an approximately 680 ms exact four-row round and \(H=3.185+5+5=13.185\) ms, the bound is about 2.020. The synthetic cost-bounded wrapper reached at most about 1.08 times its matched exact round because the first pass warmed most of the retry set.

### 10.2 Bounded substitutes

If cache-aware substitution can load at most \(M\) records outside the exact union and each such record costs at most \(c_{\max}\), then

\[
\boxed{
\frac{T_{\mathrm{safe}}}{T_E(C)}
\le2+\frac{H+Mc_{\max}}{T_E(C)}.
}
\]

This bound requires an enforced substitute cap and a valid worst-record cost. Without those conditions, cache-aware substitutes can evict exact records and remove the competitive guarantee.

### 10.3 Anytime trajectory risk

Let \(\bar q_n\) be an anytime-valid upper confidence bound on the conditional probability that committing approximate round \(n\) causes a trajectory violation not already present. Allocate

\[
\alpha_n=\frac{6\alpha}{\pi^2(n+1)^2},
\qquad \sum_{n=0}^{\infty}\alpha_n=\alpha.
\]

Commit the approximate result only when

\[
\bar q_n\le\alpha_n.
\]

Otherwise retry exactly. By the union bound,

\[
P(\text{any harmful approximate commit})
\le\sum_n\bar q_n
\le\sum_n\alpha_n
=\alpha.
\]

This guarantee is conditional on validity of the risk bounds under the online distribution. A calibration-validity test must fail closed. An adversary that produces in-distribution observations and hidden harm defeats any observational guard without additional trusted information. In that setting, only always-exact operation or an externally certified approximation can guarantee exact-equivalent quality.

## 11. Exact small-state solver

### 11.1 JSON schema

A model consists of named states, actions available in each state, and probabilistic outcomes. Each outcome includes:

```text
probability
next_state
accepted
committed
time_ms
damage
collapse
trajectory_end
trajectory_violation
```

Run:

```bash
python3 tools/dflash_renewal_control.py \
  --write-demo-model scratch/dflash-renewal-control/demo-model.json

python3 tools/dflash_renewal_control.py \
  --model scratch/dflash-renewal-control/demo-model.json \
  --max-damage-per-token 0.001
```

### 11.2 Demonstrated LP behavior

The four-state demonstration model uses `easy/hard` acceptance regimes and `cold/warm` cache states. Approximation changes immediate outcomes and next-state transitions. Selected sweep results are:

| record ms | damage budget/token | accepted tok/s | mean round ms | mean accepted/round | qualitative policy |
|---:|---:|---:|---:|---:|---|
| 0.613 | 0 | 6.592 | 512.4 | 3.378 | exact sequential |
| 0.613 | 0.001 | 7.236 | 448.3 | 3.244 | approximate only in easy states, one randomized boundary state |
| 0.613 | 0.01 | 8.147 | 372.9 | 3.038 | approximate in easy states, narrower exact action in hard/cold |
| 1.8 | 0 | 2.489 | 1278.9 | 3.184 | exact, cache-conditioned width |
| 1.8 | 0.001 | 2.813 | 1141.0 | 3.210 | constrained mixture |
| 1.8 | 0.01 | 3.382 | 814.4 | 2.754 | approximation admitted in easy states |

The LP can randomize at a quality boundary. `test_trajectory_constraint_randomizes` constructs a one-state example where a 5 percent trajectory budget forces an exact 50/50 mixture of a fast risky action and a safe action.

## 12. Scalable approximate policy

`RiskGatedMPC` in the synthetic evaluator is a compact implementation of the following design.

### 12.1 Online estimators

For each row position, maintain a beta posterior for the conditional hazard

\[
q_i=P(A\ge i\mid A\ge i-1).
\]

A prefix \(A=a\) supplies successes for positions \(1,\ldots,a\), then one observed failure at position \(a+1\) if \(a<R\). This is censoring-correct. It is the same statistical structure used by the current adaptive-width path.

Maintain separate estimators for:

- exact and approximate hazards;
- per-tier record cost and fixed target cost;
- snapshot and restore time;
- cache set differences for each candidate action;
- target-observation likelihood under benign and harmful states;
- action-dependent next-regime and cache transitions.

### 12.2 Candidate evaluation

For each candidate width, expert action, and verification mode:

1. Form a conservative lower confidence bound on accepted reward.
2. Form an upper confidence bound on marginal wall time from set differences and queue state.
3. Roll the belief state forward for a short horizon.
4. Score the candidate with the transformed Dinkelbach reward.
5. Add shadow costs for entering a difficult acceptance regime and evicting exact records.
6. Reject approximate candidates whose residual quality bound exceeds the round's risk allocation.

At a sequential row boundary, apply Theorem 2. After batch target observations, apply Theorem 1 with the post-first-pass cache state. Mandatory probes should use exact or certified-safe actions so exploration does not consume unbudgeted quality risk.

The supplied scalable policy uses beta hazards, an EWMA record-cost estimate, one-step rollout with future-state shadow prices, a trajectory risk budget, and an OOD timing envelope. It is intentionally conservative outside its calibrated fast-resident profile.

## 13. Numerical calibration

The formulas above use abstract costs. The checked-in data supplies the following sanity points:

| quantity | value used for check | role |
|---|---:|---|
| drafter time | 17.5 ms/round | fixed speculative cost |
| exact empty-round fallback | 643.2 ms | liveness cost when \(A=0\) |
| controller | 3.1849 ms per four-row round | approximate-action overhead |
| missing expert record | about 13.5 MiB | I/O unit |
| four-row exact union | 1067 records | measured/blended sticky union |
| seven-row union | 1506.2 records | measured point |
| PCIe-floor record cost | 0.613 ms | fast-resident floor |
| recommended historical effective record cost | about 1.8 ms | slower mixed regime |
| NVMe range | 2.93 to 3.88 ms/record | storage-bound regime |

For the campaign four-row distribution,

\[
P(A=0,1,2,3,4)=(1,1,1,0,24)/27,
\]

so

\[
E[A]=3.6667,\qquad P(A=0)=1/27.
\]

Using a 10 ms batch snapshot/fixed term and the 0.613 ms record floor gives

\[
\begin{aligned}
E[T]
={}&17.5+\frac1{27}(643.2)\\
&+\frac{26}{27}(10+1067\cdot0.613)\\
={}&680.8\ \mathrm{ms}.
\end{aligned}
\]

Thus

\[
E[T]/E[A]=185.7\ \mathrm{ms/accepted\ token}.
\]

Adding the 3.1849 ms controller produces 186.5 ms per accepted token. This is consistent with the cited 187 to 194 ms best regime after ordinary measurement noise and omitted fixed costs. The controller contributes about 0.87 ms per accepted token in this high-acceptance distribution.

At 1.8 ms per record, the same cold calculation is about 519 ms per accepted token. This large shift is why the policy must estimate current cost instead of freezing the best historical point.

The 1067-record four-row union represents roughly 14.1 GiB of record traffic before residency and reuse. A single missing record at 13.5 MiB costs about 0.613 ms only in the pinned-H2D floor regime. Storage-bound execution changes the marginal policy.

## 14. Synthetic evaluation

### 14.1 Protocol

The evaluation ran 400 independent trajectories per scenario, each ending after at least 128 committed output tokens. Seeds and scenario parameters are stored in `scratch/dflash-renewal-control/metadata.json`.

Scenarios:

- `calibrated`: 0.613 ms record floor, well-separated benign and harmful target signals, low approximation risk.
- `miscalibrated`: higher and drifting record cost, weaker signal separation, higher harmful probability, stronger future hard-regime transition.
- `adversarial`: approximate acceptance is immediate throughput bait, harmful observations mimic benign observations, hard bursts and record-cost bursts occur.

Policies:

- `always_exact_r4`: fixed exact four-row batch baseline.
- `adaptive_exact`: hazard- and cost-adaptive exact width with sequential/batch choice.
- `static_nominal_approx_r4`: fixed approximate action with one static retry threshold.
- `cost_bounded_subset_wrapper`: subset-only first pass with a conservative retry threshold; its name refers to latency competitiveness, not adversarial quality safety.
- `risk_gated_mpc`: adaptive exact controller plus approximation, risk allocation, OOD fallback, and future-state shadow costs.

### 14.2 Results

| scenario | policy | accepted tok/s | speedup vs exact R4 | committed ms/token | trajectory violation | approximate fraction | retry fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| calibrated | `always_exact_r4` | 5.300 | 1.00x | 185.5 | 0.0000 | 0.000 | 0.000 |
| calibrated | `static_nominal_approx_r4` | 8.341 | 1.57x | 116.1 | 0.0050 | 1.000 | 0.013 |
| calibrated | `cost_bounded_subset_wrapper` | 6.800 | 1.28x | 143.3 | 0.0000 | 1.000 | 0.295 |
| calibrated | `adaptive_exact` | 6.237 | 1.18x | 158.6 | 0.0000 | 0.000 | 0.000 |
| calibrated | `risk_gated_mpc` | 8.342 | 1.57x | 116.0 | 0.0175 | 1.000 | 0.002 |
| miscalibrated | `always_exact_r4` | 3.241 | 1.00x | 300.4 | 0.0000 | 0.000 | 0.000 |
| miscalibrated | `static_nominal_approx_r4` | 4.612 | 1.42x | 205.4 | 0.1025 | 1.000 | 0.047 |
| miscalibrated | `cost_bounded_subset_wrapper` | 3.609 | 1.11x | 265.5 | 0.0050 | 1.000 | 0.382 |
| miscalibrated | `adaptive_exact` | 4.141 | 1.28x | 237.2 | 0.0000 | 0.000 | 0.000 |
| miscalibrated | `risk_gated_mpc` | 4.130 | 1.27x | 237.8 | 0.0000 | 0.008 | 0.000 |
| adversarial | `always_exact_r4` | 3.249 | 1.00x | 288.0 | 0.0000 | 0.000 | 0.000 |
| adversarial | `static_nominal_approx_r4` | 6.459 | 1.99x | 149.8 | 0.9250 | 1.000 | 0.013 |
| adversarial | `cost_bounded_subset_wrapper` | 4.757 | 1.46x | 203.4 | 0.8175 | 1.000 | 0.297 |
| adversarial | `adaptive_exact` | 3.543 | 1.09x | 270.2 | 0.0000 | 0.000 | 0.000 |
| adversarial | `risk_gated_mpc` | 3.594 | 1.11x | 265.4 | 0.1625 | 0.114 | 0.000 |

### 14.3 Interpretation

The calibrated exact R4 baseline lands at 185.5 ms per committed token, close to the requested best-regime check. Adaptive exact control improves throughput without quality risk by choosing wider batches in easy states and narrower or sequential work in difficult states.

The static approximate rule is fast when calibration holds. Under calibration drift, its trajectory violation rate rises to 10.25 percent. The risk-gated controller detects that its fast-resident timing profile is invalid, fails closed to exact operation, and preserves zero observed violations in this run. Its approximate fraction falls below one percent.

The adversarial environment makes harmful observations look benign and raises immediate approximate acceptance. Static and cost-bounded observational guards fail badly. The risk-gated controller limits exposure through OOD and future-state logic, but its 16.25 percent trajectory violation rate still exceeds a 5 percent contract. This result is expected and important: no posterior threshold is safe when the posterior calibration itself is adversarially invalid. Always-exact or externally certified approximation is required.

The cost-bounded subset wrapper's maximum synthetic round ratio to its matched exact action was about 1.08 across these scenarios, below the theoretical 2 plus overhead bound. Its adversarial quality result confirms that latency competitiveness and quality validity are separate properties.

## 15. Engine decision table

The engine should estimate the following quantities before choosing a policy. A point estimate alone is insufficient for quality-sensitive decisions; use confidence bounds and a calibration-validity state.

| online quantity | conditioning state | estimator or measurement | decisions affected | fail-closed rule |
|---|---|---|---|---|
| \(P(A\ge i\mid A\ge i-1)\) for each row | prompt features, recent anchor state, exact/approx action | censoring-correct beta hazards with decay | width, stop, sequential vs batch | use conservative lower bound; narrow or exact when support is weak |
| full \(P(A=a)\) | width and action | derive from survival or Dirichlet counts | renewal reward and fallback probability | retain explicit mass at zero |
| drafter time \(D(R)\) | width, context length, queue state | EWMA and high quantile | all round choices | charge upper confidence cost |
| fallback time \(F\) | cache and context | measured exact steps | value of avoiding empty rounds | exact fallback remains mandatory |
| target fixed cost | batch/seq mode and row count | direct timing decomposition | sequential vs batch | prefer exact measured mode when model residual is high |
| exact expert set per row | target router and candidate | exact route extraction | marginal row and retry cost | never infer a zero miss from a stale cache tag |
| VRAM, host, and in-flight record sets | layer and expert record | cache metadata plus completion fences | stop, substitute, retry | synchronize uncertain in-flight state or price it as missing |
| disk and H2D cost functions | bytes, queue depth, tier, coalescing | per-tier regression and quantiles | expert count, width, retry | profile-specific envelope; invalidate on drift |
| snapshot and restore cost | context and recurrent layout | direct timing | batch and retry | cap one retry; account for restore tail |
| marginal exact-retry record set | post-first-pass cache | exact set difference \(E^*\setminus C_R\) | retry threshold | first-pass costs are sunk; do not recharge them |
| target-observation likelihoods | logit drop, entropy, margin, row index | calibrated conditional density or conformal score | retry posterior | exact on OOD or failed coverage test |
| harmful-commit upper probability | observation, action, trajectory state | anytime-valid upper confidence bound | quality gate | exact when no valid bound exists |
| approximation effect on next regime | current belief and action | transition counts or learned dynamics | MPC future value | pessimistic transition set under uncertainty |
| approximation effect on exact-cache state | selected records and eviction order | deterministic cache model plus measurements | MPC and retry | pin subset records in strict safe mode |
| remaining trajectory risk | sequence position and prior commitments | deterministic risk ledger | approximate admission | exact when remaining budget is exhausted |
| EOS or remaining token distribution | request limits and decoder state | known request cap or EOS model | per-round risk allocation | use summable anytime allocation for unknown length |
| substitute-record cap \(M\) | action | enforced action validator | competitive bound | set \(M=0\) when a bound is required |
| model-calibration validity | timing, feature density, delayed quality audit | OOD detector and confidence sequence | all approximate actions | disable approximation for the trajectory |

## 16. Integration notes

1. **Keep two reward counters.** The added `accepted_draft_total` counter increments only on a verified speculative prefix. Existing `accept_total` continues to count committed tokens in DFlash rounds, including empty-round fallback and scalar bypass.

2. **Expose post-first-pass cache state.** `dflash_retry_needed()` currently consumes target logits. The controller also needs exact marginal retry records, residency tiers, and measured restore cost.

3. **Do not use one global retry drop.** Store a calibrated posterior or monotone score conditioned on width, row, expert action, and cache-cost profile. The threshold is state-conditioned.

4. **Log trajectory joins.** Immediate forced-prefix parity is insufficient for free-trajectory quality. Log round IDs, request IDs, action, observation, retry decision, accepted prefix, delayed quality result, and next-state features.

5. **Separate profile calibration.** The 0.613 ms pinned-H2D floor, approximately 1.8 ms mixed regime, and storage-bound range need distinct timing profiles. Extrapolating a quality model across a timing-profile change should fail closed unless calibration evidence supports it.

6. **Preserve exact retry semantics.** Retry must restore the pre-round recurrent snapshot and execute the identical candidate block exactly once. Re-drafting creates a different control problem and loses the simple competitive bound.

## 17. Reproduction

Run all hardware-free checks:

```bash
PYTHONPATH=tools python3 -m unittest -v \
  tools/test_dflash_renewal_control.py

python3 tools/dflash_renewal_control.py --demo

python3 tools/evaluate_dflash_renewal_control.py \
  --episodes 400 \
  --episode-tokens 128 \
  --output-dir scratch/dflash-renewal-control
```

The checked-in evaluation summary was assembled from one scenario per invocation to keep individual jobs short. All scenarios use deterministic seeds recorded in metadata.

## 18. Main conclusions

- The correct accepted-token objective is a ratio of long-run reward and wall time. Empty rounds remain in the denominator with zero accepted reward.
- In a binary restricted observation model with monotone likelihood ratio and fixed cache state, exact retry and row continuation are posterior thresholds.
- The unrestricted cache-coupled problem need not have a threshold in collapse probability or logit drop.
- Sunk first-pass work cancels from retry comparisons. Marginal exact set differences determine retry cost.
- Approximation changes future recurrent and cache state, so a one-round latency penalty model is inadequate.
- A subset-only first pass followed by at most one exact retry has a worst-case cost no larger than \(2+H/T_E\) times always exact, under pinned-cache monotonicity.
- A trajectory quality guarantee additionally needs valid anytime risk bounds. Under hidden adversarial miscalibration, exact fallback is the only guarantee available from this observation set.

