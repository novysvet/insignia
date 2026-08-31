# Joint adaptive compute from causal difficulty evidence

## Status

This note answers `tasks/inference-math-frontier-20260831/12-joint-adaptive-compute-control.md` at reference commit `0740c63`. It supplies a constrained partially observed semi-Markov formulation, structural results, counterexamples, an exact finite solver, a scalable primal-dual approximation, a causal previous-logit likelihood, a value-of-information calculation, a robust exact-fallback rule, hard-extension analysis, pooled CPU simulation, and an engine handoff.

Code and reproducible outputs:

- `tools/joint_adaptive_compute.py`: finite belief-SMDP model, exact occupation-measure LP, likelihood and posterior helpers, value of information, robust certificate, safety-price calculation, and counterexamples.
- `tools/evaluate_joint_adaptive_compute.py`: hidden-difficulty simulation with stochastic DFlash acceptance, endogenous cache and I/O state, request-level hardness, correlated quality damage, fixed baselines, independent thresholds, myopic information acquisition, primal-dual control, exact joint control, and robust exact fallback.
- `tools/aggregate_joint_adaptive_compute.py`: pooling of independent replications, ratio-of-sums estimates, replicate dispersion, event-rate intervals, action frequencies, safety price, and controller kill checks.
- `tools/test_joint_adaptive_compute.py`: CPU unit tests for posterior motion, metric martingales, nonnegative gross information value, constrained occupation measures, nonseparability, robust economic gating, and safety price.
- `scratch/joint-adaptive-compute/`: final exact policy, model summary, fixed sweep, value-of-information table, counterexamples, pooled simulation, per-replication data, action frequencies, and validation output.
- `handoffs/joint-adaptive-compute-engine.md`: minimal engine schema, data contract, update rule, safety envelope, deadlines, and kill criteria.

Every numerical transition and likelihood parameter in this artifact is synthetic. The finite LP is exact for that specified finite model. It is not a measurement of GLM-5.3-Flash or either Insignia machine. Engine deployment requires logged estimates and uncertainty sets described below.

## 1. Constrained partially observed semi-Markov model

### 1.1 Physical state and causal observation

Let decision epoch `n` occur at a committed recurrent-state anchor, before the next block's irreversible GPU or I/O work. A useful physical state is

\[
X_n=(G_n,R_n,H_n,C_n,Q_n,Z_n,B_n,\Theta_n).
\]

The components are:

- \(G_n\): request-level hard/easy type, fixed until request end;
- \(R_n\): token- or block-level difficulty regime, allowed to change inside a request;
- \(H_n\): exact target state, draft state, route state, and any approximation divergence needed for a Markov description;
- \(C_n\): expert residency, cache recency, in-flight records, and checkpoint residency;
- \(Q_n\): storage, host-to-device, CUDA, and verification queues;
- \(Z_n\): finite quality monitor, including persistent quality debt and absorbing failure flags;
- \(B_n\): remaining request-level risk allocation;
- \(\Theta_n\): unknown likelihood, transition, latency, acceptance, and harm parameters.

The engine sees only causal information available before action launch:

\[
O_n=(\phi(L_{n-1}),\,R^{\rm route}_{n-1},\,C^{\rm obs}_n,\,Q^{\rm obs}_n,
A_{n-1}^{\rm acc},\,T_{n-1}^{\rm ctr},\,Z_n^{\rm obs}).
\]

Here \(\phi(L_{n-1})\) contains previous target-logit features such as the top-1/top-2 margin. Route overlap, cache occupancy, previous accepted prefix, timing counters, and monitor state are also pre-action observations. No current target result may leak into this vector.

Let \({\cal H}_n\) be the causal history. The sufficient information state is the joint belief

\[
b_n(x,\theta)=P(X_n=x,\Theta_n=\theta\mid {\cal H}_n).
\]

The implemented finite model compresses this belief to a hard-regime posterior, a three-level cache state, a three-level I/O queue, one quality-debt bit, and a measurement-phase bit.

### 1.2 Joint action and semi-Markov outcome

The block action is

\[
U_n=(k_n^{\rm exp},k_n^{\rm draft},v_n,p_n,f_n,m_n),
\]

where the coordinates select expert count, DFlash width, verification policy, expert precision, prefetch budget, and optional exact measurement. Capture features, checkpoint layout, verification depth, and retry can be added to the same option without changing the formulation.

An action produces random outcome

\[
(Y_n,\tau_n,W_n,L_n,E_n^H,V_n^H,K_n,O_{n+1},X_{n+1}).
\]

The counters are:

- \(Y_n\): committed output tokens, including the exact fallback token after an empty draft;
- \(\tau_n>0\): wall-clock sojourn time through commit and state restoration;
- \(W_n\): bytes charged to the action;
- \(L_n\): additive excess negative log-likelihood or another additive global quality damage;
- \(E_n^H\): indicator or expected count for a completed hard request;
- \(V_n^H\): hard-request violation at that completion;
- \(K_n\): catastrophic-failure indicator or severity.

The controlled semi-Markov kernel is

\[
{\cal K}(dx',do,d\tau,dy,dw,d\ell,de,dv,dk\mid x,u,\theta).
\]

Unequal block durations, positive-time zero-token measurements, stochastic accepted prefixes, retries, and checkpoint restoration are ordinary transitions in this kernel.

### 1.3 Throughput objective

For byte price \(\omega_B\ge0\), the long-run utility rate is

\[
\boxed{
\rho(\pi)=
\liminf_{N\to\infty}
\frac{E_\pi\!\left[\sum_{n<N}(Y_n-\omega_B W_n)\right]}
{E_\pi\!\left[\sum_{n<N}\tau_n\right]}.
}
\]

With \(\omega_B=0\), this is committed tokens per millisecond. Multiplication by 1000 gives committed tokens per second. Under positive recurrence and integrability, the same ratio is the almost-sure Markov-renewal reward rate.

Latency belongs in the denominator. A weighted additive latency form appears after introducing a candidate rate \(\rho\):

\[
r_\rho=Y-\omega_BW-\rho\tau.
\]

At the optimal rate, the maximal average transformed reward is zero. This Dinkelbach form is useful for dynamic programming and online control. The exact finite LP below avoids root finding by normalizing occupation time to one millisecond.

### 1.4 Pathwise and risk-sensitive quality contracts

A global excess-NLL budget is

\[
\boxed{
\limsup_{N\to\infty}
\frac{\sum_{n<N}L_n}{\sum_{n<N}Y_n}\le \beta
\quad\text{almost surely}.
}
\]

Bounding average excess NLL by \(\beta\) bounds the multiplicative PPL increase by \(e^\beta\). The synthetic artifact calls this counter `ppl_loss`, but its mathematical role is additive log-quality damage.

A separate rare hard-answer contract is

\[
\boxed{
\limsup_{N\to\infty}
\frac{\sum_{n<N}V_n^H}{\sum_{n<N}E_n^H}\le\alpha
\quad\text{almost surely},
}
\]

with the ratio evaluated after enough hard requests have completed. A request-level chance version is

\[
P_\pi(V_r^H=1\mid G_r=H)\le\alpha.
\]

A catastrophic tail can be constrained by an ordinary rate, a CVaR monitor, or the entropic risk rate

\[
\limsup_{N\to\infty}\frac1N
\log E_\pi\exp\!\left(\eta\sum_{n<N}K_n\right)\le \kappa.
\]

For a finite quality automaton, request completion and violation are additive transition counters. The occupation LP then enforces the completed-request probability ratio exactly in the finite model.

### 1.5 Why average per-row classification loss is insufficient

A per-row classifier can have a small average error and still violate the generation contract.

First, one early error changes the recurrent state, logits, routes, cache state, and future observations. The loss is not an exogenous label error with an unchanged next-state kernel.

Second, rare failures compound across a path. A row hazard of \(10^{-4}\) over 10,000 opportunities gives

\[
1-(1-10^{-4})^{10000}\approx0.632.
\]

The average row loss remains \(10^{-4}\), but the probability of at least one path failure is about 63.2%.

Third, quality damage is correlated. Approximate verification can create a debt state that raises later failure probability. Averaging rows erases this dependence.

Fourth, hard-answer failures have a different denominator. A policy can satisfy global average loss by spending almost all damage on rare hard requests. One scalar row average cannot detect that allocation.

Fifth, unequal action durations require time-normalized control. Averaging per-row losses and rewards gives the wrong weighting when fast and slow blocks occur at different rates.

## 2. Previous logits as a causal likelihood

### 2.1 Specified likelihood

The finite artifact uses two previous-block features:

- top-1/top-2 target-logit margin \(M_n\);
- route churn count \(J_n\), the number of changed routes among eight comparisons.

Conditional on latent token regime \(D_n\in\{0,1\}\), the model is

\[
M_n\mid D_n=d\sim {\cal N}(\mu_d,\sigma_d^2),
\qquad
J_n\mid D_n=d\sim {\rm Binomial}(8,\theta_d),
\]

with conditional independence. The margin is binned by cut points 0.90 and 1.75. Churn is represented by \(1\{J_n>2\}\). Synthetic parameters are

\[
(\mu_0,\sigma_0,\theta_0)=(2.25,0.70,0.13),
\qquad
(\mu_1,\sigma_1,\theta_1)=(0.65,0.85,0.52).
\]

For observation \(o=(i,r)\), let

\[
\ell_d(o)=P(M\in I_i\mid D=d)P(1\{J>2\}=r\mid D=d).
\]

All parameters remain replaceable. Production estimates must carry confidence intervals and calibration diagnostics.

### 2.2 Bayes update and likelihood-ratio ordering

Given prior \(q=P(D=1\mid{\cal H})\), the posterior is

\[
\boxed{
q^+(o)=\frac{q\ell_1(o)}{q\ell_1(o)+(1-q)\ell_0(o)}.
}
\]

Equivalently,

\[
\log\frac{q^+}{1-q^+}
=
\log\frac{q}{1-q}+
\log\frac{\ell_1(o)}{\ell_0(o)}.
\]

Therefore evidence with likelihood ratio above one increases posterior difficulty. Evidence with likelihood ratio below one decreases it. Low margins and high route churn have larger hard/easy likelihood ratios in the supplied table. `tools/test_joint_adaptive_compute.py` verifies both directions.

The optional exact metric \(Z\in\{0,1\}\) has synthetic sensitivity 0.91 and false-positive probability 0.09. Its branch posterior is

\[
q_z=\frac{qP(Z=z\mid D=1)}{qP(Z=z\mid D=1)+(1-q)P(Z=z\mid D=0)}.
\]

The posterior martingale identity

\[
\sum_zP(z\mid q)q_z=q
\]

is tested directly.

### 2.3 Posterior threshold theorem

Fix observed machine state \(x=(c,q^{\rm io},z)\), Lagrange multipliers, and candidate throughput \(\rho\). Let the transformed action value be

\[
Q(q,x,a)=E\left[Y-\rho\tau-\omega_BW
-\sum_i\lambda_i g_i+h(b',x')\mid q,x,a\right].
\]

Order two actions as low and high compute. Define

\[
\Delta(q,x)=Q(q,x,a_H)-Q(q,x,a_L).
\]

**Threshold proposition.** If \(\Delta(q,x)\) is nondecreasing in \(q\), then the set of posteriors for which high compute is optimal is an upper interval. If \(\Delta\) is continuous and changes sign once, the optimal action has threshold

\[
q\ge q^*(x),\qquad \Delta(q^*(x),x)=0.
\]

**Proof.** Suppose high compute is optimal at \(q_0\), so \(\Delta(q_0,x)\ge0\). For every \(q\ge q_0\), increasing differences give \(\Delta(q,x)\ge\Delta(q_0,x)\ge0\). Thus high compute remains optimal. Continuity and a single sign change give the displayed threshold. QED.

Sufficient primitives include an ordered latent difficulty state, monotone likelihood ratios, increasing differences in immediate transformed reward, stochastically monotone transitions, and a continuation value that preserves increasing differences. Cache or queue state changes the threshold. A single threshold independent of machine state is not implied.

The unrestricted joint controller can randomize at an active constraint boundary. It can also violate monotonicity when a larger action worsens future queues enough to dominate its quality benefit.

### 2.4 Value of one exact metric

Let \(Q_0(q,x,a)\) be the pre-measurement action value excluding metric cost. The gross value of information is

\[
\boxed{
G(q,x)=
\sum_zP(z\mid q)\max_a Q_0(q_z,x,a)
-
\max_a Q_0(q,x,a).
}
\]

It is nonnegative. Let \(a^*\) maximize the value without measuring. In every metric branch the controller can ignore the result and choose \(a^*\), so

\[
\sum_zP(z\mid q)\max_aQ_0(q_z,x,a)
\ge
\sum_zP(z\mid q)Q_0(q_z,x,a^*)
=Q_0(q,x,a^*).
\]

The last equality requires conditional expected metrics to be affine in the posterior. The implementation mixes latent easy and hard kernels before maximizing; it does not raise a posterior-averaged acceptance probability to a draft-length power.

Net value subtracts metric time, bytes, and downstream state cost:

\[
G_{\rm net}=G-\rho c_m-\omega_Bw_m.
\]

At cache level 1, empty I/O queue, no debt, and the best-fixed reference rate, the synthetic table gives:

| posterior hard | gross value | time value | net value | measure |
|---:|---:|---:|---:|:---:|
| 0.02 | 0.00000 | 0.03724 | -0.03724 | no |
| 0.10 | 0.00354 | 0.03724 | -0.03370 | no |
| 0.25 | 0.03389 | 0.03724 | -0.00335 | no |
| 0.50 | 0.07524 | 0.03724 | 0.03800 | yes |
| 0.75 | 0.02844 | 0.03724 | -0.00880 | no |
| 0.90 | 0.00036 | 0.03724 | -0.03688 | no |
| 0.98 | 0.00000 | 0.03724 | -0.03724 | no |

Information is valuable near an action boundary. It has little value when both metric outcomes choose the same tuple. The exact dynamic LP measures on 5.97% of decision starts because continuation value and machine state add boundaries absent from the one-step table.

## 3. When the action tuple decomposes

### 3.1 Exact decomposition after ratio and constraint relaxation

Introduce a candidate throughput \(\rho\) and quality multipliers \(\lambda\). Suppose state and action factor as

\[
x=(x_1,\ldots,x_m),\qquad a=(a_1,\ldots,a_m).
\]

The Bellman maximization decomposes into coordinate problems if all of the following hold:

1. The feasible action set is Cartesian, \({\cal A}(x)=\prod_i{\cal A}_i(x_i)\).
2. The transformed one-step value is additive:
   \[
   E[Y-\rho\tau-\omega_BW-\lambda^Tg\mid x,a]
   =\sum_i r_i(x_i,a_i).
   \]
3. The controlled transition factorizes:
   \[
   P(x'\mid x,a)=\prod_iP_i(x_i'\mid x_i,a_i).
   \]
4. The observation likelihood factorizes consistently with those state factors.
5. The continuation value is additive, \(h(x)=\sum_i h_i(x_i)\).
6. No coordinate changes a shared cache, queue, recurrent state, quality automaton, or duration term outside its own factor.

Then

\[
\arg\max_a\sum_i\left[r_i(x_i,a_i)+E h_i(x_i')\right]
=
\prod_i\arg\max_{a_i}\left[r_i(x_i,a_i)+E h_i(x_i')\right].
\]

These conditions are strong. DFlash width and checkpoint layout share duration and archive bytes. Expert count changes next logits and routes. Prefetch controls future cache and queues. Verification and precision share the same quality monitor. The Insignia action tuple therefore does not satisfy this factorization in general.

### 3.2 Index policy conditions

A shared linear resource can be relaxed with price \(\nu\). Conditional on \(\nu\), factors may decouple. An index exists only if each factor is indexable: as \(\nu\) increases, the set of states choosing the passive or lower-resource action must expand monotonically. The index is the break-even price.

This construction fails when action coordinates control the same future route or queue, when resource use is nonadditive, when one coordinate changes another coordinate's observation law, or when the passive set is nonmonotone. A learned collection of independent thresholds has no index guarantee without these properties.

### 3.3 Expert count versus future routes and prefetch

The explicit two-block counterexample is:

| expert k | prefetch | current ms | next route | next ms | myopic tok/s | two-block tok/s |
|---:|---:|---:|:---|---:|---:|---:|
| 4 | 0 | 1.0 | cold-B | 10.0 | 1000.0 | 181.8 |
| 4 | 1 | 1.4 | wrong-prefetch-B | 11.0 | 714.3 | 161.3 |
| 8 | 0 | 2.0 | warm-A | 1.0 | 500.0 | 666.7 |
| 8 | 1 | 2.2 | hot-A | 0.6 | 454.5 | 714.3 |

The myopic choice is \((k=4,\text{prefetch}=0)\). The dynamic choice is \((k=8,\text{prefetch}=1)\). The two-factor cross difference is 68.15 tok/s, not zero. Expert count changes the future route, which changes whether prefetch is useful.

### 3.4 Draft length versus checkpoint layout

| draft k | checkpoint | committed | ms | tok/s |
|---:|:---|---:|---:|---:|
| 2 | delta | 1.8 | 2.0 | 900.0 |
| 2 | full | 1.8 | 2.4 | 750.0 |
| 4 | delta | 3.2 | 11.0 | 290.9 |
| 4 | full | 3.2 | 3.6 | 888.9 |

Choosing draft length while holding full checkpoints selects \(k=4\). Choosing checkpoint layout while holding \(k=2\) selects delta. Combining those sensible coordinate choices gives 290.9 tok/s. Joint optimization selects \((2,\text{delta})\) at 900 tok/s. The cross difference is 747.98 tok/s.

### 3.5 Every local heuristic improves, but the combination fails

The constructed baseline score is 100. The one-coordinate scores are 104.0 for low expert count, 105.0 for draft four, 103.0 for approximate verification, 102.5 for FP8, and 104.5 for prefetch. An additive model predicts 119.0. Pairwise interaction penalties total 41.0, so the actual combination is 78.0. Every coordinate test passes; the combined policy is worse than baseline.

The `interaction_trap` simulator adds the same mechanisms dynamically. Pooled results show independent thresholds at 23.88 tok/s versus 28.29 tok/s for the best fixed tuple, with 7.91% hard-answer violation against a 3.5% budget.

## 4. Exact finite-state solution

### 4.1 Discretization

The implemented state space is

\[
S=\{1,\ldots,7\}_{q}\times
\{0,1,2\}_{C}\times
\{0,1,2\}_{Q}\times
\{0,1\}_{Z}\times
\{\text{ready},\text{measured}\}.
\]

It contains

\[
7\cdot3\cdot3\cdot2\cdot2=252
\]

states. The compute action space has

\[
2\cdot2\cdot2\cdot2\cdot2=32
\]

tuples. Every ready state also offers `measure_exact_metric`. Measured states cannot measure again. The model therefore has

\[
126\cdot33+126\cdot32=8190
\]

state-action starts.

The metric transition takes 1.25 synthetic milliseconds, reads 0.125 synthetic MB, commits zero tokens, and changes only the belief and phase. Compute transitions return to `ready` and jointly update posterior, cache, I/O queue, and quality debt.

### 4.2 Time-normalized occupation measure

Let \(x(s,a)\) be action starts per millisecond. For expected counters \(y,\tau,w,\ell,e,v,k\), solve

\[
\max_{x\ge0}\quad \sum_{s,a}x(s,a)(y(s,a)-\omega_Bw(s,a))
\]

subject to flow conservation

\[
\sum_a x(j,a)=\sum_{s,a}x(s,a)P(j\mid s,a),\qquad \forall j,
\]

time normalization

\[
\sum_{s,a}x(s,a)\tau(s,a)=1,
\]

and quality constraints

\[
\sum_{s,a}x(s,a)(\ell(s,a)-\beta y(s,a))\le0,
\]

\[
\sum_{s,a}x(s,a)(v(s,a)-\alpha e(s,a))\le0,
\]

\[
\sum_{s,a}x(s,a)(k(s,a)-\gamma y(s,a))\le0.
\]

The objective is utility per millisecond because total occupied wall time is one. The stationary randomized policy is

\[
\pi(a\mid s)=\frac{x(s,a)}{\sum_{a'}x(s,a')}.
\]

This LP is exact for the finite known kernel and stationary randomized policies in a communicating model. A fixed initial state in a multichain model requires restricting to its reachable recurrent class or using an initial-state formulation. The result is only as accurate as belief discretization and kernel estimation.

### 4.3 Exact synthetic result

At budgets \(\beta=0.0045\), \(\alpha=0.035\), and \(\gamma=0.00035\), the LP reports:

| quantity | exact finite value |
|:---|---:|
| committed throughput | 37.9613 tok/s |
| PPL-loss proxy per token | 0.001607 |
| hard-answer violation | 0.035000 |
| catastrophe per token | 0.0000425 |
| measurement fraction | 0.059739 |
| mean decision duration | 65.414 ms |

The hard-answer constraint is active. Its LP shadow price is 73.1318 transformed token units. The other two synthetic constraints are slack and have zero shadow price.

The best feasible fixed tuple in the same finite model is

`k4_d2_exact_bf16_pf1`

at 29.7881 tok/s. Exact joint control improves this finite-model rate by 27.44%.

## 5. Scalable primal-dual approximation and bound

### 5.1 Drift-plus-penalty controller

Define constraint increments

\[
g_{1,n}=L_n-\beta Y_n,
\quad
g_{2,n}=V_n^H-\alpha E_n^H,
\quad
g_{3,n}=K_n-\gamma Y_n.
\]

Maintain virtual queues

\[
Q_{i,n+1}=[Q_{i,n}+g_{i,n}]_+.
\]

For current rate estimate \(\hat\rho_n\), choose a joint tuple that approximately maximizes

\[
V E[Y_n-\hat\rho_n\tau_n-\omega_BW_n\mid b_n,a]
-
\sum_iQ_{i,n}E[g_{i,n}\mid b_n,a]
+
\widehat V_{\rm cache}(b_n,a)
-
\widehat V_{\rm queue}(b_n,a).
\]

The implementation evaluates all 32 tuples on a CPU, includes cache and future-route bonuses, penalizes prefetch into a loaded queue, updates three separate virtual queues, and updates the observed throughput ratio online. This is a small mathematical controller, not a generic neural policy.

### 5.2 Performance and violation bound

Assume:

- \(0<\tau_{\min}\le\tau_n\le\tau_{\max}\);
- \(|g_{i,n}|\le G_i\) and transformed rewards are bounded;
- a stationary Slater policy has \(E[g_i]\le-\epsilon\) for every constraint;
- the per-action conditional model error is at most \(\varepsilon_m\);
- belief compression has diameter \(\delta_b\), and exact action values are \(L_b\)-Lipschitz in belief;
- the online maximization has additive error \(\varepsilon_o\).

Let

\[
B=\frac12\sum_iG_i^2,
\qquad
\varepsilon_Q=\varepsilon_o+2\varepsilon_m+2L_b\delta_b.
\]

The standard quadratic Lyapunov drift argument gives transformed-objective gap

\[
\limsup_{T\to\infty}
\left(f^*-\frac1T\sum_{n<T}E[f_n]\right)
\le \frac{B}{V}+\varepsilon_Q.
\]

At the optimal Dinkelbach root \(\rho^*\), \(f^*=0\). Therefore the throughput gap obeys

\[
\boxed{
\rho^*-\rho_{\rm pd}
\le
\frac{B/V+\varepsilon_Q}{\tau_{\min}}.
}
\]

The virtual-queue recursion gives the pathwise finite-horizon certificate

\[
\boxed{
\left[\frac1T\sum_{n<T}g_{i,n}\right]_+
\le \frac{Q_{i,T}}{T}.
}
\]

Under the Slater margin, expected backlog is \(O(V)\). One explicit bound has the form

\[
\limsup_T\frac1T\sum_{n<T}\sum_iE[Q_{i,n}]
\le
\frac{B+V(F_{\max}-F_{\rm Slater})+V\varepsilon_Q}{\epsilon}.
\]

Thus utility loss is \(O(1/V)\), steady queue size is \(O(V)\), and finite-horizon violation is \(O(V/T)\), plus model and belief-compression error. These bounds require normalized updates and valid confidence bounds. The simulator's fixed gains are a concrete finite-run tuning, not a claim that its displayed constants are universal.

### 5.3 Two quality constraints need two prices

A single multiplier can enforce both PPL and hard-answer constraints only when one constraint is redundant, the two constraint vectors are collinear on every relevant policy, or a fixed scalarization is known to expose the desired feasible boundary point.

A simple counterexample has three policies:

\[
A:(r,g_1,g_2)=(10,-1,+1),
\]
\[
B:(r,g_1,g_2)=(10,+1,-1),
\]
\[
C:(r,g_1,g_2)=(9,0,0).
\]

With equal scalar weights, both \(A\) and \(B\) have zero combined violation and beat \(C\), yet each violates one separate contract. KKT conditions for two independent active constraints require a vector \((\lambda_{\rm PPL},\lambda_{\rm hard})\). The exact LP and online controller use separate multipliers or virtual queues.

## 6. Robust control under model error and shift

### 6.1 Uncertainty set and exact fallback

Let \({\cal U}_n\) contain plausible observation, transition, latency, acceptance, and quality parameters. The set may come from confidence sequences, calibrated bootstrap regions, or a deliberately inflated engineering envelope. A proposed approximate tuple \(a\) is admitted only if all of the following pass:

1. The uncertainty radius is below its deployment ceiling.
2. Logged-data overlap for \((b_n,a)\) exceeds \(p_{\min}\).
3. Worst-case PPL, hard-answer, and catastrophe ratios lie inside reserved envelopes.
4. A lower confidence bound on economic gain over exact fallback is positive:
   \[
   \inf_{\theta\in{\cal U}_n}
   \left[
   Y_a-Y_0-\hat\rho(\tau_a-\tau_0)-\omega_B(W_a-W_0)
   \right]
   -c_{\rm controller}>0.
   \]
5. The controller and any requested metric finish before their hardware deadlines.

The fallback is

`k8_d2_exact_bf16_pf0`.

It uses full verification and the conservative precision in the synthetic model. If the uncertainty set, risk certificate, overlap check, OOD score, or deadline check fails, the controller returns this exact action for the block.

The implementation maintains a previous-logit surprise score and timing-ratio score. Persistent surprise or low overlap latches exact mode. It does not silently extrapolate beyond support.

### 6.2 Safety statement

Suppose confidence sets satisfy

\[
P(\Theta^*\in{\cal U}_n\ \text{for all }n)\ge1-\delta.
\]

If every admitted action has worst-case conditional constraint drift at most zero, then the cumulative constraint process is a bounded supermartingale. Its long-run average is nonpositive almost surely under standard martingale-difference conditions. For request-level anytime safety, allocate \(\delta_r\) to request `r` with

\[
\sum_r\delta_r\le\delta.
\]

Admit exploration only when the robust certificate bounds that request's failure probability by \(\delta_r\). A union bound then gives probability at least \(1-\delta\) that no certified request exceeds its allocated event. If no nonexact action passes, exact inference is the only admissible action.

### 6.3 Throughput price of safety

For joint action averages \((\bar Y_J,\bar\tau_J)\), exact fallback averages \((\bar Y_0,\bar\tau_0)\), fallback fraction \(f\), and guard cost \(c_g\), a state-agnostic mixture has rate

\[
\boxed{
\rho_{\rm safe}=
1000\frac{(1-f)\bar Y_J+f\bar Y_0}
{(1-f)\bar\tau_J+f\bar\tau_0+c_g}.
}
\]

The price is \(\rho_J-\rho_{\rm safe}\). State-selective fallback can cost less because it falls back in states where the joint policy's advantage is already small.

In the pooled calibrated synthetic run:

- unguarded exact-joint simulation: 37.4895 tok/s;
- guarded robust simulation: 35.7783 tok/s;
- observed state-selective price: 1.7112 tok/s, or 4.56%;
- fallback fraction: 22.55%;
- state-agnostic mixture estimate: 34.2822 tok/s, an 8.56% price;
- robust gain over the best fixed simulation: 20.94%.

The global kill test remains false because the transformed gain is +0.4736 committed-token equivalents per round, overlap fallback is far below 95%, and the robust gain exceeds the 3% materiality threshold.

## 7. Hard extensions

### 7.1 Request type, token regimes, and hysteresis

Add request type \(G_r\in\{E,H\}\) and token regime \(R_n\in\{E,H\}\). The exact belief is the four-point posterior

\[
b_n(g,r)=P(G_r=g,R_n=r\mid{\cal H}_n).
\]

Request completion resets \(G\). Inside a request, \(R\) follows an action-dependent Markov transition. The hard-answer monitor depends on \(G\), while expert count and DFlash acceptance depend strongly on \(R\). Keeping both posteriors prevents a globally easy token from erasing evidence that the request is a rare hard-answer request.

Hysteresis requires persistent action state or switching cost. Let \(\Delta(q)=Q_H(q)-Q_L(q)\) be increasing. If the previous mode is low, switching high costs \(c_{LH}>0\), so the switch threshold is

\[
q_{\uparrow}=\Delta^{-1}(c_{LH}).
\]

If the previous mode is high, switching low costs \(c_{HL}>0\), so high mode remains optimal until

\[
q_{\downarrow}=\Delta^{-1}(-c_{HL}).
\]

Since \(-c_{HL}<c_{LH}\),

\[
q_{\downarrow}<q_{\uparrow}.
\]

This is a hysteresis band. Checkpoint reuse, CUDA graph choice, cache residency, and in-flight prefetch create exactly such persistence. Without switching cost or controlled persistent state, the thresholds coincide and there is no hysteresis. Nonmonotone queue effects can produce more complex regions instead of a single band.

### 7.2 Anytime-safe paid exploration

At epoch `n`, build confidence set \({\cal U}_n\) with failure allocation \(\delta_n\), where \(\sum_n\delta_n\le\delta\). Define a safe exploration set

\[
{\cal A}^{\rm safe}_n=
\left\{
 a:
 \sup_{\theta\in{\cal U}_n}g_i(b_n,a;\theta)\le-r_{i,n}
 \ \forall i,
 \quad p_{\rm log}(a\mid b_n)\ge p_{\min},
 \quad \mathrm{LCBgain}(a)>0
\right\}.
\]

Exact shadow evaluations are actions with their own duration and byte cost. Their cost enters \(Y-\rho\tau-\omega_BW\). Exploration samples only from \({\cal A}^{\rm safe}_n\) using logged propensities. If the set is empty, overlap is absent, or the exact-shadow cost exceeds its information value, choose exact fallback. Confidence-sequence validity plus the summable risk allocation gives an anytime guarantee with probability at least \(1-\delta\).

### 7.3 Two storage devices and one GPU

Let \(Q_n^1,Q_n^2,Q_n^H,Q_n^G\) be queues for two storage devices, host-to-device transfer, and GPU work. Prefetch action assigns records to devices and controls future arrivals:

\[
Q_{n+1}^d=[Q_n^d+A_n^d(a_n)-S_n^d]_+,
\]

\[
Q_{n+1}^H=[Q_n^H+A_n^H(a_n,C_n)-S_n^H]_+,
\]

\[
Q_{n+1}^G=[Q_n^G+A_n^G(a_n)-S_n^G]_+.
\]

Cache state updates from completed reads, not requested reads. This queue vector belongs in the controlled state because earlier prefetch actions change later feasibility and latency. A Lyapunov controller adds queue-weighted arrival penalties to the transformed utility. Independent per-device thresholds are valid only after a separable relaxation and indexability check.

## 8. CPU simulation

### 8.1 Environment

The simulator contains:

- hidden token difficulty and request-level hard/easy type;
- action-dependent regime transitions and future routes;
- stochastic accepted-prefix length with an exact fallback token on zero acceptance;
- a three-level expert-cache state;
- a three-level I/O queue controlled by earlier prefetch;
- a shared stress draw that couples low acceptance, quality harm, and future debt;
- persistent quality debt until request end;
- hard-request failure and catastrophic events correlated with debt and difficulty;
- observation and timing shift scenarios.

The compared policies are conservative exact fixed, best feasible fixed, independent coordinate thresholds, myopic value of information, primal-dual joint control, exact finite joint control, and robust safe joint control.

The final output pools 12 independent replications of 500 rounds for every policy and scenario. This is 6,000 rounds per row and 126,000 compute rounds overall, plus measurement starts. All values are synthetic.

### 8.2 Pooled results

| scenario | policy | tok/s | PPL proxy/token | hard fail | hard events | fallback |
|:---|:---|---:|---:|---:|---:|---:|
| calibrated | best fixed | 29.584 | 0.001470 | 4/122 = 3.28% | 122 | 0% |
| calibrated | independent thresholds | 27.688 | 0.000814 | 1/106 = 0.94% | 106 | 0% |
| calibrated | myopic VoI | 35.248 | 0.007572 | 19/103 = 18.45% | 103 | 0% |
| calibrated | primal-dual joint | 34.402 | 0.003286 | 11/123 = 8.94% | 123 | 0% |
| calibrated | exact joint | 37.489 | 0.001788 | 4/138 = 2.90% | 138 | 0% |
| calibrated | robust safe joint | 35.778 | 0.000342 | 0/104 = 0.00% | 104 | 22.55% |
| distribution shift | best fixed | 24.609 | 0.003558 | 10/136 = 7.35% | 136 | 0% |
| distribution shift | independent thresholds | 22.626 | 0.004216 | 11/126 = 8.73% | 126 | 0% |
| distribution shift | myopic VoI | 28.719 | 0.018775 | 28/104 = 26.92% | 104 | 0% |
| distribution shift | primal-dual joint | 26.191 | 0.003960 | 18/133 = 13.53% | 133 | 0% |
| distribution shift | exact joint | 30.590 | 0.004130 | 11/126 = 8.73% | 126 | 0% |
| distribution shift | robust safe joint | 27.471 | 0.001426 | 3/125 = 2.40% | 125 | 40.17% |
| interaction trap | best fixed | 28.294 | 0.002267 | 4/125 = 3.20% | 125 | 0% |
| interaction trap | independent thresholds | 23.878 | 0.002725 | 11/139 = 7.91% | 139 | 0% |
| interaction trap | myopic VoI | 30.561 | 0.043437 | 52/114 = 45.61% | 114 | 0% |
| interaction trap | primal-dual joint | 29.510 | 0.003619 | 17/132 = 12.88% | 132 | 0% |
| interaction trap | exact joint | 36.090 | 0.002053 | 3/109 = 2.75% | 109 | 0% |
| interaction trap | robust safe joint | 33.322 | 0.000427 | 2/102 = 1.96% | 102 | 28.72% |

The robust policy increases fallback under shift and keeps every point estimate inside the stated synthetic budgets. The exact policy is faster under the shifted simulator but violates the hard-answer point budget. Independent thresholds become slower than the best fixed tuple in the interaction trap and violate the hard-answer budget. Myopic information acquisition spends the metric only for current value and has no pathwise risk certificate; it violates both quality budgets badly.

Rare-event uncertainty remains material. The descriptive Wilson 95% upper bounds for robust hard failure are 3.56% in calibrated data, 6.82% under distribution shift, and 6.87% in the interaction trap. Serial correlation also weakens a simple Bernoulli interval. The simulation therefore supports the controller design but does not independently certify a 3.5% production hard-answer limit. Production safety must come from the robust confidence set, request-level risk allocation, exact shadows, and substantially larger held-out evaluation.

### 8.3 Reproduction

A minimal exact solve and test run is:

```bash
cd tools
python test_joint_adaptive_compute.py
python joint_adaptive_compute.py \
  --solve-demo \
  --json-out ../scratch/joint-adaptive-compute/exact-policy.json
python evaluate_joint_adaptive_compute.py \
  --rounds 500 \
  --seed 12012 \
  --out-dir ../scratch/joint-adaptive-compute-run
```

For cheap independent replications, solve once, then pass the first output as `--static-dir`. Pool completed `run-*` directories with:

```bash
python aggregate_joint_adaptive_compute.py \
  --root ../scratch/joint-adaptive-compute-replications \
  --out-dir ../scratch/joint-adaptive-compute
```

## 9. Deployment decision

The synthetic artifact passes its internal deployment gate:

- robust calibrated gain over best fixed: +20.94%;
- robust transformed gain per round: +0.4736;
- calibrated exact-fallback fraction: 22.55%;
- controller-cost kill: false;
- overlap/fallback kill: false;
- material-gain kill: false.

This is permission to collect engine data, not permission to enable approximation by default. The production controller remains disabled until the engine handoff's overlap, calibration, exact-shadow, timing, risk, and lower-confidence throughput gates all pass. If no uncertainty-certified policy materially beats the best fixed configuration, the correct result is to delete the hot-path controller and keep the fixed configuration.
