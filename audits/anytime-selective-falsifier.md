# Anytime-valid selective falsification under policy feedback

## Status

This note solves Problem 8 in `tasks/inference-math-frontier-20260831/08-selective-falsifier-under-policy-drift.md`.

Reproducible artifacts:

- `tools/anytime_selective_falsifier.py`: confidence ledger, audit protocol helpers, historical-policy optimizer, optional held-out helpers, block severity, impossibility construction, and sequential feature-acquisition solver.
- `tools/evaluate_anytime_selective_falsifier.py`: CPU-only controlled feedback process and four-policy comparison.
- `tools/test_anytime_selective_falsifier.py`: theorem and protocol tests.
- `scratch/anytime-selective-falsifier/runs.csv`: per-seed results.
- `scratch/anytime-selective-falsifier/summary.csv`: aggregate results.
- `scratch/anytime-selective-falsifier/report.md`: human-readable reproduction report.
- `scratch/anytime-selective-falsifier/protocol.json`: machine-readable engine protocol.
- `scratch/anytime-selective-falsifier/indistinguishable-environments.json`: exact no-overlap witness.

The engine safety target is the pathwise cumulative contract in Section 2.3. The selective-risk confidence sequence in Section 5 is also time-uniform, but a contextwise conditional guarantee is deliberately not claimed.

## 1. Causal protocol and filtration

At engine round \(t\), let \(\mathcal F_{t-1}\) contain all prior contexts, actions, audit coins, arrived labels, cache transitions, resets, and controller updates. The engine then observes causal context \(X_t\). Before the audit coin is revealed it freezes:

- controller score \(R_t\);
- terminal fast proposal \(S_t\in\{0,1\}\);
- block severity range \(0\le Y_t\le B_t\le B\);
- audit probability \(q_t\in[q_{\min},q_{\max}]\);
- certificate and runtime-state fingerprints;
- the support cell, threshold, and exact-relative cost estimates.

Every cost estimate used to choose \(q_t\) is a predictable function of the
frozen causal context and controller score. Realized latency, cache damage, or
failure severity is logged after the action for evaluation, never fed back into
the current audit propensity.

The potential severity \(Y_t=Y_t(\mathrm{fast})\) is the result that exact block verification would assign to the fast candidate. Enlarge the pre-coin sigma field to

\[
\mathcal G_t=\sigma(\mathcal F_{t-1},X_t,S_t,q_t,B_t,Y_t).
\]

The audit broker draws

\[
Z_t\mid\mathcal G_t\sim\operatorname{Bernoulli}(q_t)
\]

independently of \(Y_t\). The independence is design based: the environment, controller, and future contexts may otherwise be arbitrary. If \(Z_t=1\), the engine runs the fast block in a rollback shadow, runs exact verification, commits exact state, and eventually reveals \(Y_t\). If \(Z_t=0\), it commits fast and does not assume a counterfactual label. If \(S_t=0\), it abstains and performs exact work without entering the audit ledger.

Actions can change \(X_{t+1}\), routing state, and caches. The guarantee conditions on the adaptively generated potential outcomes and uses only the current audit coin as randomization. Permanently changing or removing the audit policy creates a different closed-loop policy and invalidates the fingerprint.

## 2. Three different risk statements

### 2.1 Marginal selective risk on the visited adaptive path

For proposal rounds through \(n\), define

\[
W_n=\sum_{t\le n}S_t(1-q_t),\qquad
I_n=\sum_{t\le n}S_t(1-q_t)Y_t.
\]

The intended-commit selective risk is

\[
\boxed{\mathcal R_n^{\mathrm{int}}=I_n/W_n}
\]

when \(W_n>0\). It is the audit-randomized policy's average fast severity after conditioning on the contexts and potential outcomes that its own feedback produced. The factor \(1-q_t\) is essential because an audit overrides the fast commit.

This is a marginal path-average target. It does not assert

\[
E[Y_t\mid X_t=x,S_t=1]\le\epsilon
\]

for every context \(x\), every support cell, or every hard-prompt class. Such a conditional assertion needs structural assumptions, enough data in each cell, or a separately certified model.

### 2.2 Proposal risk

For diagnostics, define

\[
L_n=\sum_{t\le n}S_tY_t,\qquad
N_n=\sum_{t\le n}S_t,
\]

and \(\mathcal R_n^{\mathrm{prop}}=L_n/N_n\). This asks how risky every terminal fast proposal was, including proposals intercepted by audits.

### 2.3 Pathwise committed-loss contract

Let

\[
C_n=\sum_{t\le n}S_t(1-Z_t)Y_t,
\qquad
K_n=\sum_{t\le n}S_t(1-Z_t).
\]

The primary safety contract is

\[
\boxed{
P\!\left(\forall n:\ C_n\le \beta_0+\epsilon K_n\right)\ge1-\delta.
}
\]

Here \(\beta_0\) is a finite startup reserve measured in severity units. Equivalently, whenever \(K_n>0\),

\[
C_n/K_n\le\epsilon+\beta_0/K_n.
\]

A nonzero reserve is not cosmetic. With no prior information, \(B>\epsilon\), and \(\beta_0=0\), the first unaudited fast commit can have severity \(B\). No procedure can permit that commit and guarantee \(C_1\le\epsilon\) for every bounded environment. The alternatives are an explicit reserve, an exact-shadow warmup, or exact operation until sufficient evidence exists.

## 3. Impossibility without overlap or shadow labels

### Theorem 1: observational equivalence

Let a logging policy choose exact with probability one at state `anchor`, and suppose exact work does not compute the fast counterfactual. There are two environments \(P_0,P_1\) with identical distributions for every logged variable at every sample size, but opposite deployed fast risk:

- Under \(P_0\), exact has observed loss zero and returns to `anchor`; fast has severity zero and returns to `anchor`.
- Under \(P_1\), exact has observed loss zero and returns to `anchor`; fast has severity one and enters absorbing state `collapsed`.

Consequently, any data-dependent deployment rule with positive probability of selecting fast has fast risk zero under \(P_0\) and one under \(P_1\), although its input log has exactly the same law.

#### Proof

Under the logger, only exact is executed. Both environments emit the same context `anchor`, action `exact`, observed exact loss zero, and next state `anchor` at every round. Induction gives equality of the entire observed-log distribution. Therefore every measurable statistic, confidence bound, trained model, and deployed policy derived from the log has the same distribution under \(P_0\) and \(P_1\).

If the derived policy selects fast with positive probability, its potential fast loss is zero in \(P_0\) and one in \(P_1\). A uniformly valid upper bound smaller than one, or a nontrivial rule certifying risk below any \(\epsilon<1\), must fail in one world. QED.

### Corollary 1: positivity is necessary on deployment support

More generally, if a target policy has \(\pi_t(a\mid x)>0\) on a set where the logger has \(h_t(a\mid x)=0\), and no shadow computation reveals \(Y_t(a)\), environments may be changed arbitrarily on that unobserved action without changing the log. Identification and nontrivial coverage are impossible there.

The executable witness hashes the two exact-policy logs and obtains identical SHA-256 digests, then deploys fast to obtain selective risks zero and one.

## 4. Exact random-audit protocol

The accepted production order is:

1. Observe \(X_t\), including the prefix/history digest, previous target and draft summaries, routing state, cache generations, hidden summaries, and prior certificate state.
2. Run any allowed feature-acquisition or partial-exact stages. The audit coin remains hidden.
3. Freeze the terminal decision tuple

   \[
   D_t=(S_t,R_t,\tau_e,q_t,B_t,k_t,F_e,H_t),
   \]

   where \(\tau_e\) is the epoch threshold, \(k_t\) is the support key, \(F_e\) is the certificate fingerprint, and \(H_t\) is the runtime-state fingerprint.
4. Append the tuple to an append-only decision log before revealing the coin.
5. Obtain \(U_t\in[0,1)\) from an audit broker and set \(Z_t=1\{U_t<q_t\}\).
6. If audited, snapshot recurrent and cache state, run fast in shadow, run exact verification, construct one block-level severity label, rollback the fast state, and commit exact. Log the real audit cost and audit-induced cache transition.
7. If unaudited, commit fast. Do not fabricate \(Y_t\).
8. Append the ledger record in decision order. A delayed label resolves only its own audited record.
9. Update the confidence sequences and prospective pathwise gate.
10. On any invalidity or reset trigger, commit exact.

### 4.1 Audit randomness

The mathematical protocol requires a fresh Bernoulli coin independent of the frozen potential outcome. The operational helper uses

\[
U_t=\operatorname{uint64}(\operatorname{HMAC\!\!-SHA256}(K_e,
\texttt{domain}\Vert\texttt{request}\Vert t\Vert H_t))/2^{64}.
\]

The epoch key \(K_e\) is held outside the controller. Its hash and key ID are committed before deployment; the key may be disclosed after epoch closure for replay. The controller cannot inspect \(U_t\) before freezing \(q_t\). Statistical validity assumes true randomness or standard PRF security against the controller and environment.

### 4.2 Certificate fingerprint

The epoch fingerprint \(F_e\) is a canonical SHA-256 digest of:

- controller weights and executable/config digest;
- feature schema and preprocessing digest;
- exact verifier and severity-schema digest;
- threshold, information-price, \(q_{\min}\), \(q_{\max}\), and maximum severity;
- the causal audit-cost model, including its allowed and forbidden inputs;
- block semantics, cache-transition semantics, and the fail-closed reset policy;
- calibration epoch ID;
- enabled support keys.

Changing any field invalidates old evidence.

### 4.3 Runtime-state fingerprint

The per-round fingerprint \(H_t\) binds the audit coin and eventual label to:

- \(F_e\), request ID, round ID, and prefix/history digest;
- target and draft logit-summary digests;
- routing, cache, recurrent-generation, and hidden-summary digests;
- score, threshold, support key, \(q_t\), and \(B_t\).

Large tensors need not be logged. Collision-resistant digests plus the exact schema/version are enough to reject mismatched labels and replay a coin.

### 4.4 Required log fields

Every terminal proposal records:

```text
request_id, round_id, wall_clock, epoch_id
certificate_fingerprint, runtime_state_fingerprint, support_key
score, threshold, proposal_indicator
severity_bound, q_t, q_min, q_max
fast_gain_ms, audit_gain_ms, cache_shadow_terms
random_key_id, random_key_commitment, random_domain, audit_uniform, audit_indicator
action_committed, block_id, block_width
label_due_round, label_arrival_round, block_severity
fast_shadow_digest, exact_digest, first_divergence_row
pre_state_digest, post_state_digest, cache_generation_before, cache_generation_after
proposal_loss_upper, intended_loss_upper, committed_loss_upper
fast_commit_count, pathwise_budget, reset_reason
```

A missing propensity, invalid coin relation, fingerprint mismatch, out-of-range label, duplicate label, or unknown round invalidates the ledger and forces exact work.

## 5. Audit-capture confidence sequence

The loss is nonnegative and rare. A specialized capture e-process is tighter than treating the Horvitz-Thompson pseudo-outcome as a generic signed variable.

### Lemma 1: one-step e-factor

Fix \(v\in[0,b]\), \(Z\sim\operatorname{Bernoulli}(q)\), and \(\rho>0\) satisfying \(\rho b<q\). Define

\[
\lambda(\rho,q,b)=-\frac{1}{b}\log\left(1-\frac{\rho b}{q}\right).
\]

Then

\[
E\left[\exp\{\rho v-\lambda(\rho,q,b)Zv\}\right]\le1.
\]

#### Proof

Let

\[
f(x)=\log(1-q+qe^{-\lambda x}).
\]

The function \(f\) is convex, \(f(0)=0\), and

\[
f(b)=\log(1-\rho b)\le-\rho b.
\]

A convex function lies below its secant chord on \([0,b]\), so

\[
f(v)\le(v/b)f(b)\le-\rho v.
\]

Exponentiating gives

\[
e^{\rho v}(1-q+qe^{-\lambda v})\le1.
\]

QED.

### Theorem 2: adaptive audit-capture CS

Let \(v_t\in[0,b_t]\) and \(q_t\) be frozen before the independent audit coin, with future \(v_s,b_s,q_s\) allowed to depend arbitrarily on prior actions and observations. For a fixed \(\rho\) satisfying \(\rho b_t<q_t\) for every \(t\),

\[
M_n(\rho)=\exp\left\{
\rho\sum_{t\le n}v_t-
\sum_{t\le n}\lambda(\rho,q_t,b_t)Z_tv_t
\right\}
\]

is a nonnegative supermartingale. For fixed weights \(w_j>0\), \(\sum_jw_j=1\), the mixture

\[
M_n=\sum_jw_jM_n(\rho_j)
\]

is also a supermartingale. Define \(U_n\) as the unique endpoint satisfying

\[
\sum_jw_j\exp\left\{
\rho_jU_n-
\sum_{t\le n}\lambda(\rho_j,q_t,b_t)Z_tv_t
\right\}=1/\delta,
\]

intersected with \(\sum_{t\le n}b_t\). Then

\[
\boxed{P(\forall n:\ \sum_{t\le n}v_t\le U_n)\ge1-\delta.}
\]

#### Proof

Lemma 1 applied conditionally on \(\mathcal G_t\) gives the supermartingale property for each \(M_n(\rho_j)\), hence for the fixed mixture. Ville's inequality gives \(P(\sup_nM_n\ge1/\delta)\le\delta\). The mixture is increasing in its candidate total, so on the complement event the true total never exceeds the inverted endpoint. QED.

### 5.1 Two simultaneous ledgers

The implementation splits \(\delta\) and applies Theorem 2 to:

1. proposal loss: \(v_t=S_tY_t\), \(b_t=S_tB_t\);
2. intended loss: \(v_t=S_t(1-q_t)Y_t\), \(b_t=S_t(1-q_t)B_t\).

This yields time-uniform upper bounds \(U^L_n\) and \(U^I_n\), and

\[
\mathcal R_n^{\mathrm{int}}\le U^I_n/W_n
\]

simultaneously over time when \(W_n>0\).

### 5.2 Delayed and outcome-dependent labels

Let \(m(n)\) be the longest decision-ordered proposal prefix whose audited labels have arrived. Labels after an unresolved audit gap are ignored temporarily, even if they arrived earlier. For either capture target,

\[
U_n=U_{m(n)}^{\mathrm{capture}}+
\sum_{t=m(n)+1}^{n}b_t.
\]

The suffix is charged at its full logged bound. This remains valid when severe outcomes take longer to label. The price is temporary width. If the oldest unresolved audit exceeds the declared delay contract, the certificate resets and the default is exact.

### 5.3 Abstentions

An abstention has \(S_t=0\). It contributes neither loss nor denominator and performs exact work. Treating exact rows as zero-severity fast rows would bias selective risk downward and is forbidden.

## 6. Committed-loss bound and prospective gate

For a resolved prefix \(m\), exact audit labels reveal

\[
A_m=\sum_{t\le m}S_tZ_tY_t.
\]

Because \(C_m=L_m-A_m\), a valid upper endpoint is

\[
U^C_n=
\max\{0,U^L_m-A_m\}
+
\sum_{t=m+1}^{n}S_t(1-Z_t)B_t.
\]

Only unaudited records in the unresolved suffix are charged; audited suffix records commit exact.

### Theorem 3: anytime pathwise budget

Assume the proposal CS covers at every time. Before permitting the unaudited branch of the next proposal, compute the ledger endpoint that would result from appending a nonaudit with bound \(B_t\). Permit that branch only if

\[
U^{C,+}_t\le \beta_0+\epsilon(K_{t-1}+1).
\]

Audits and abstentions commit exact. Then

\[
C_n\le\beta_0+\epsilon K_n
\]

for every \(n\).

#### Proof

Induct on decisions. An audit or abstention changes neither \(C\) nor \(K\). For an unaudited fast commit, the post-append confidence endpoint covers the actual new committed loss on the CS event, and the gate places that endpoint below the new budget. QED.

Combining Theorems 2 and 3 gives the probability statement in Section 2.3. A later audit label may widen the displayed endpoint; that disables future fast commits but cannot retroactively violate the actual budget already established by induction.

## 7. General overlap and importance ratios

Suppose a logger chooses action \(A_t\sim h_t(\cdot\mid X_t)\), a predictable target policy is \(\pi_t\), and

\[
w_t=\frac{\pi_t(A_t\mid X_t)}{h_t(A_t\mid X_t)}\le R.
\]

For bounded immediate loss \(Y_t(A_t)\in[0,B]\),

\[
H_t=w_tY_t(A_t)\in[0,RB]
\]

is an unbiased pseudo-outcome for the target action's immediate loss at the visited context. Any bounded-mean betting confidence sequence can be applied to \(H_t\) under adaptive predictable logging.

This does not automatically identify the trajectory law of a different policy when actions change future contexts. Full counterfactual trajectory evaluation requires products of sequential likelihood ratios and overlap at every decision, which can be unusably variable. Random auditing avoids that target mismatch by monitoring the actually deployed audited policy on the histories that it creates. Removing audits or changing cache semantics is a new policy, not a free extrapolation.

## 8. Audit probability and abstention threshold

### 8.1 Context-dependent audit cost

Use exact as the value-zero baseline. For context \(x\), define predictable
cost-model outputs

\[
G_F(x)=T_E(x)-T_F(x)+V(C_F')-V(C_E')
\]

and

\[
G_A(x)=T_E(x)-T_A(x)+V(C_A')-V(C_E'),
\]

where \(T_E,T_F,T_A\) are exact, fast, and audit wall times, and \(V(C')\) is the shadow value of the resulting cache/queue state. This charges rollback, duplicate compute, I/O, and future cache effects. Counting only the number of labels is wrong.

The quantities in the optimizer are forecasts frozen before the coin. Realized
times and cache transitions are used to score throughput and refit a later cost
model. Using the realized current transition would leak \(Y_t\) into \(q_t\)
and destroy the design-based proof.

For a terminal proposal with audit probability \(q\), expected exact-relative reward is

\[
(1-q)G_F+qG_A.
\]

A Horvitz-Thompson information penalty has leading form \(B^2(1-q)/q\). For multiplier \(\eta>0\), maximize

\[
(1-q)G_F+qG_A-\eta B^2\frac{1-q}{q}.
\]

Writing \(\kappa=G_F-G_A\), the interior solution is

\[
\boxed{q^*(x)=B(x)\sqrt{\eta/\kappa(x)}}
\]

when \(\kappa>0\), clipped to \([q_{\min},q_{\max}]\). If auditing is no more expensive than fast commit, use \(q_{\max}\). Large severity bounds and cheap audits receive more probability.

### 8.2 Joint threshold selection

For score threshold \(\tau\) and information price \(\eta\), let

\[
S_{\tau}(x)=1\{R(x)\le\tau\},
\qquad q_{\tau,\eta}(x)=q^*(x).
\]

The design objective is

\[
\max_{\tau,\eta}
E\left[S_{\tau}(X)\{(1-q)G_F+qG_A\}\right]
\]

subject to an engineering screen on

\[
\frac{E[S_{\tau}(X)(1-q)Y]}{E[S_{\tau}(X)(1-q)]}
\]

being below a design limit, plus minimum support and mass constraints. The
screen may use biased, adaptively collected, and repeatedly inspected history.
It chooses an economically plausible frozen policy; it contributes no coverage.

The implementation uses chronological train/design/later-screen thirds:

1. train the severity-weighted risk model on the first part;
2. search \((\tau,\eta)\) on design data using exact-relative reward, including the predicted cache shadow cost;
3. reject support cells or recent windows whose later empirical severity exceeds a conservative engineering margin;
4. freeze the model, pair, support set, cost model, and complete certificate fingerprint;
5. start the formal confidence ledger only on fresh future audit coins and labels.

This separates optimization evidence from coverage evidence. An optional helper
also supports a one-shot held-out betting screen under an explicitly justified
independent-epoch or exchangeability assumption. A predeclared finite grid can
instead use Bonferroni allocation. Neither option is needed for Theorem 4.

## 9. Nonstationarity

### 9.1 What remains valid under arbitrary drift

The capture CS and pathwise budget require no stationarity. They condition on the entire adaptive sequence \((Y_t,q_t,B_t)\) before each audit coin. Covariate drift, controller feedback, and action-dependent cache transitions therefore do not invalidate cumulative coverage.

What changes is interpretation. A cumulative path average may be low after a long safe history even when current risk has risen. Arbitrary-drift validity does not turn a historical marginal average into a current conditional guarantee.

### 9.2 Bounded-drift local guarantee

If a separate model assumes conditional selective means \(r_t\) satisfy

\[
|r_t-r_{t-1}|\le d,
\]

then a confidence upper bound \(U^{\mathrm{win}}_n\) for the weighted mean over a recent window implies

\[
r_n\le U^{\mathrm{win}}_n+d\,
\frac{\sum_{t\in\mathrm{win}}w_t(n-t)}{\sum_{t\in\mathrm{win}}w_t}.
\]

The additive drift term is the cost of localization. Short windows react faster and have wider statistical bounds. This guarantee is lost if the drift constant is wrong.

### 9.3 Declared change points and safe resets

The robust engine mechanism is a fail-closed epoch reset:

1. A declared traffic-version change, delay breach, fingerprint change, or anytime risk alarm deactivates the epoch. An unsupported state always takes exact; persistent unsupported traffic may optionally trigger retraining.
2. The engine commits exact and may run fast only in rollback shadow to collect fresh labels.
3. Fresh rows are split chronologically into train/design/later stress-screen pieces. These rows optimize the reset policy but make no coverage claim.
4. Let \(R_0\) and \(\tau_0\) be the immutable launch sentinel. The reset score is

   \[
   R_{e+1}(x)=\max\{R^{\mathrm{new}}_{e+1}(x),R_0(x)\},
   \]

   and its threshold satisfies \(\tau_{e+1}\le\tau_0\). A small reset sample may shrink the launch fast set but cannot relax it by rescaling the score.
5. Freeze the new model, threshold, support cells, audit-cost model, and fingerprint. Formal coverage starts with fresh post-freeze audit coins. Failure to find a screened plan means exact operation.
6. Freeze the closed epoch's committed-loss endpoint \(U_e^C\). The new ledger
   receives only the unspent global reserve

   \[
   \beta_{e+1}=\beta_0+\epsilon K_{\le e}-\sum_{j\le e}U_j^C.
   \]

   This reset is allowed only when the right side is nonnegative. A negative
   value means the closed evidence cannot certify remaining global reserve, so
   the engine stays exact. Clipping it to zero and restarting would be an
   unjustified new budget.

7. Old ledgers remain immutable; the new epoch receives fresh error budget \(\delta_e\), with \(\sum_e\delta_e\le\delta\).

Relative to a stationary all-history analysis, reset loses sample efficiency and throughput during recollection. It gives no guarantee for the delay between an unannounced change and its detection beyond the pathwise reserve. Pooling old and new labels requires an explicit drift model.

## 10. Model selection and evidence reuse

The controller, threshold, support cells, cost model, and audit rule jointly define the tested policy. Reusing the same labels to tune them and then reporting a nominal bound is selective inference and can under-cover.

Accepted mechanisms include:

- **Held-out epochs:** train, design, and one untouched certificate split, provided the certificate epoch has a defensible sampling assumption and is used once.
- **Predeclared online e-values:** allocate initial wealth/error to candidate models and combine only predictable e-processes. A model may switch when its own evidence crosses a boundary.
- **Reusable holdout machinery:** release only stability-preserving or differentially private answers and account for every query. This is more complex than the engine needs.

The checked-in benchmark takes the strongest reuse-safe route: every historical
row may be reused for training, threshold search, support selection, and a
later empirical stress screen. Those rows make no coverage statement. The
accepted anytime coverage relies only on fresh online audit coins generated
after the complete policy fingerprint is frozen.

### Theorem 4: arbitrary historical tuning, fresh-evidence validity

Let \(\mathcal H_0\) contain the complete historical log. The model, threshold,
support set, cost model, and audit rule may be any measurable function of
\(\mathcal H_0\), including the result of extensive tuning and reuse. Freeze
their fingerprint before deployment, and construct Theorem 2 only from future
audit coins and future labels. Conditional on \(\mathcal H_0\), every one-step
e-factor still has expectation at most one. Therefore the future e-process has
the stated coverage conditionally, and hence unconditionally.

The proof fails as soon as future audit labels are used to change the tested
policy without closing the epoch. A modified model receives a new fingerprint,
a fresh summable error allocation, and only later audit evidence. Historical
rows may train the new model; they are never inserted into its fresh online
confidence ledger.

## 11. Severity-weighted blocks

A benign Top-1 mismatch and a hard-prompt collapse should not have equal cost. Define one bounded block outcome

\[
Y_t=\min\left\{B,
 w_{\mathrm{collapse}}1\{\mathrm{collapse}\}
 +\sum_{r=1}^{R}w_r\ell_{t,r}
\right\}.
\]

Weights may encode perplexity damage, task failure, exact-logit discrepancy, or external usefulness loss. The bound \(B\) and schema are fingerprinted.

The label must match the user-facing claim. A same-prefix perplexity check can
certify only same-prefix perplexity. It cannot certify that a hard answer remains
useful after approximation changes its own continuation. For the stated 3.5%
perplexity tolerance, the production severity schema should combine the excess
same-prefix damage with a much larger bounded continuation-collapse penalty,
for example

\[
Y=\min\{B,
w_{\mathrm{ppl}}[\Delta\mathrm{PPL}-0.035]_+
w_{\mathrm{hard}}1\{\text{free-running hard-answer collapse}\}\}.
\]

Obtaining the second term requires a rollback fork that evaluates the complete
coupled continuation or a predeclared finite-horizon usefulness judge. Its cost
belongs in \(T_A\). Teacher-forced row labels must not be relabeled as
free-running usefulness evidence.

An early DFlash divergence changes the recurrent state and the meaning of every later row. Therefore:

- draw one audit coin for the block;
- construct one potential block outcome;
- stop or continue according to the exact block semantics;
- never count downstream rows as independent calibration examples.

The simulator generates a first bad row, causally dependent later losses, and a persistent catastrophic mode, then aggregates the block once.

## 12. Sequential feature acquisition

Let state \(z\) contain the features and partial exact results acquired so far. Relative to exact fallback, let terminal fast gain be \(g_F(z)\), and let \(U(z)\) be a valid leaf risk certificate. An acquisition \(j\) has real cost \(c_j(z)\), including cache effects, and outcomes \(o\) leading to \(z_{j,o}\).

For a finite acyclic acquisition graph,

\[
V(z)=\max\left\{
0,
1\{U(z)\le\epsilon\ \text{and valid}\}g_F(z),
\max_j\left[-c_j(z)+\sum_op_j(o\mid z)V(z_{j,o})\right]
\right\}.
\]

The zero action is exact. Unsafe or uncertified leaves cannot choose fast. Cycles are rejected to impose a hard work bound. The final audit probability and coin are frozen only after acquisition reaches a terminal fast proposal. Adaptive feature requests do not hurt the online audit theorem because they occur before the coin and are part of \(\mathcal G_t\).

## 13. CPU controlled process

The synthetic process includes:

- continuous covariate drift and an abrupt traffic phase change;
- policy-dependent divergence debt and cache residency;
- selective counterfactual labels from audits only;
- outcome-dependent delayed labels and out-of-order arrival;
- abstention and exact fallback;
- rare catastrophic blocks that create a persistent failure mode;
- four causally coupled DFlash rows per block;
- context-dependent exact, fast, rollback, and future-cache costs.

The fixed run uses 12,000 rounds, a change at round 4,200, 15,000 initial pilot rows, 2,400 exact-shadow reset rows, \(q\in[0.18,0.48]\), \(\epsilon=0.06\), \(\delta=0.05\), and startup reserve 60 severity units. These are synthetic severity units, not a claim about GLM-5.3-Flash perplexity or answer usefulness.

Compared policies:

- naive threshold tuning and risk measurement on reused calibration data;
- split conformal with a frozen nominal residual percentile;
- randomized importance weighting with a repeatedly peeked pointwise Gaussian plug-in bound;
- the historically tuned, fingerprinted random-audit e-process with fresh online evidence and a monotone-safe reset sentinel.

Sixteen deterministic seeds produced:

| policy | mean intended risk | mean max local risk | reported-coverage failures | local-risk violations | abstention | audit ms/round | reset ms/round | saved ms/round | committed catastrophes/run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive calibration | 0.0813 | 0.2194 | 16/16 | 16/16 | 30.4% | 0.0 | 0.0 | 325.8 | 536.06 |
| split conformal | 0.0756 | 0.1918 | 16/16 | 16/16 | 32.0% | 0.0 | 0.0 | 317.9 | 488.88 |
| nominal importance weighting | 0.0668 | 0.1944 | 16/16 | 16/16 | 42.7% | 113.4 | 0.0 | 185.9 | 285.38 |
| anytime random audit | 0.00845 | 0.01409 | 0/16 | 0/16 | 67.4% | 58.9 | 173.4 | 74.2 | 1.19 |

The anytime method also had zero committed-loss CS failures and zero pathwise-budget violations in all sixteen runs. Every run performed one fail-closed reset at the declared traffic change. Its reset controller was the maximum of the new score and the immutable launch sentinel, with a nonincreasing threshold. The three comparators violated the global and local severity targets in every run; the nominal importance method also violated its repeatedly peeked reported bound in every run.


The indistinguishable-environment artifact reports byte-identical exact logs and deployed fast risks zero versus one. The full per-seed rows include coverage failures, global and local risk violations, abstention, audit/reset cost, fast fraction, catastrophes, and throughput reward.

## 14. Engine acceptance and kill rules

A proposal is accepted only when all of the following hold:

- a current certificate fingerprint matches the executable policy and verifier;
- the runtime support key is certified;
- \(q_t\) is frozen, logged, and at least \(q_{\min}\);
- the audit coin is hidden until the decision tuple is immutable;
- the runtime-state fingerprint binds the coin and eventual label;
- the prospective committed-loss gate passes;
- label delay is within contract;
- no reset or drift alarm is active.

The default action is exact if any item is absent or invalid.

Kill the design if it:

- assumes fast labels on exact-only rows without shadow work;
- trains on on-policy rows as though they were IID;
- removes audits after certification without starting a new policy epoch;
- treats a validation percentile or fixed-time interval as anytime valid;
- ignores logged propensities or permits zero overlap;
- counts dependent DFlash rows as independent evidence;
- drops unresolved severe labels instead of charging their bounds;
- changes model, loss, block, or cache semantics without a fresh fingerprint and evidence allocation.

## 15. References

- N. Karampatziakis, P. Mineiro, and A. Ramdas, *Off-policy Confidence Sequences*, 2021.
- I. Waudby-Smith, L. Wu, A. Ramdas, N. Karampatziakis, and P. Mineiro, *Anytime-valid off-policy inference for contextual bandits*, 2022.
- I. Waudby-Smith and A. Ramdas, *Estimating means of bounded random variables by betting*, 2020.
- C. Dwork et al., *The reusable holdout: Preserving validity in adaptive data analysis*, 2015.
- *Anytime-Valid Inference Under Outcome Delay: A Design-Based Approach*, 2026.
