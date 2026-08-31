# Anytime-valid performance experiments under hostile WSL noise

## Status

This note supplies the causal design, impossibility result for untreated carryover, anytime-valid tests, multiplicity control, finite escalation dynamic program, nonstationarity policy, hard extensions, CPU simulator, and executable decision protocol requested by `tasks/inference-math-frontier-20260831/06-anytime-ab-under-wsl-noise.md`.

Code and replayable outputs:

- `tools/anytime_ab.py`: betting e-processes, bounded log-effect score, HMAC randomization, change alarm, long-format decision engine, alpha/beta spending, and escalation DP.
- `tools/simulate_anytime_ab.py`: deterministic CPU simulator and method comparison.
- `tools/test_anytime_ab.py`: exhaustive optional-stopping checks and protocol tests.
- `build/anytime-ab-cpu.sh`: one-command CPU reproduction.
- `scratch/anytime-ab/summary.csv`: Monte Carlo operating characteristics.
- `scratch/anytime-ab/replay-three-median-seed-0.csv`: slower contender promoted by a conventional three-run median.
- `scratch/anytime-ab/protocol-example.json`: editable operational protocol.
- `scratch/anytime-ab/pair-log-template.csv`: header-only required log schema.

The central decision rule is intentionally conservative. A candidate can be rejected by a cheap screen, but it cannot be promoted by a screen. Promotion requires the deterministic parity gate and every required full-campaign cell to cross its anytime evidence threshold in one change-point epoch. Weak evidence ends as `INCONCLUSIVE`.

## 1. What can be claimed

There are two useful estimands, and they should not be conflated.

### 1.1 Operational randomized-pair median

For a predeclared benchmark cell, let

\[
R_i = \frac{T_{B,i}}{T_{A,i}}
\]

be the oriented latency ratio produced by one guarded randomized pair. The default test targets the conditional median statement

\[
P(R_i<c\mid\mathcal F_{i-1})
\;\le\;
P(R_i>c\mid\mathcal F_{i-1})
\tag{1}
\]

under the null, where \(c=0.98\) for a required 2% speedup. Exact ties are neutral. Equation (1) is equivalent to saying that the conditional probability mass strictly below \(c\), with half the tie mass, is at most one half.

This is an operational claim about the result of rerunning the declared pair protocol. It needs no latency moments and remains meaningful when the raw latency distribution has a Pareto tail.

### 1.2 Average slot-specific log effect

A causal average multiplicative effect is available from randomization under a carryover-neutralized design. Let \(L_{ij}(a)\) be the potential log latency for arm \(a\in\{A,B\}\) in slot \(j\in\{1,2\}\), after the declared preparation policy. Define

\[
\tau_i
=\frac12\left[
L_{i1}(B)-L_{i1}(A)
+L_{i2}(B)-L_{i2}(A)
\right].
\tag{2}
\]

The fair AB/BA assignment identifies \(\tau_i\) without IID timing. With a predeclared timer floor and timeout cap, the bounded log score in `bounded_log_ratio_score` gives an anytime test of a margin such as \(\tau_i\ge\log(0.98)\).

The resulting estimand is a capped geometric-mean effect. It is not an uncapped arithmetic latency mean.

### 1.3 What is not identified for free

The median of the unobserved same-slot ratios

\[
\operatorname{median}\left\{
\exp(L_{ij}(B)-L_{ij}(A))
\right\}
\]

is not nonparametrically identified from one AB or BA sequence per block. The sign test identifies the randomized-pair median in Section 1.1. Interpreting it as a same-state causal median requires an additional structural assumption, for example a common multiplicative treatment effect or a condition making the cross-slot period component median-zero. The protocol does not silently make that assumption.

## 2. Filtration and weakest defensible assumptions

Let \(\mathcal F_{i-1}\) contain every completed pair, hardware counter, change alarm, previous stopping decision, and the experimenter's next-cell policy before pair \(i\). The experimenter may adapt the next stage, cell, and benchmark case using \(\mathcal F_{i-1}\).

The following assumptions are sufficient. No IID, stationarity, Gaussianity, finite latency variance, or fixed sample size is required.

1. **Case and preparation lock.** Before the order draw, the stage, metric/state cell, case, cache recipe, warmup count, timeout, and epoch are fixed and logged.
2. **Conditional randomization.** The AB/BA bit is conditionally fair and independent of the pair's potential outcomes given the locked information and \(\mathcal F_{i-1}\). `derive_randomization` implements a fair HMAC bit from a hidden seed committed before the campaign.
3. **Consistency.** The logged measurements equal the potential outcomes for the assigned sequence and declared preparation policy.
4. **No differential carryover for a direct-effect claim.** After preparation, the second measured arm's state and start time do not depend on which measured arm ran first, except through an arm-independent slot effect. This includes cache state, thermal state, queue state, and elapsed-time displacement.
5. **Atomic inclusion.** A pair is kept in full, recorded at a predeclared timeout, or omitted only for a pre-outcome external fault code. Inclusion cannot depend on either latency.
6. **Test null.** For the sign test, the bounded score has non-positive conditional mean under the null. Randomization proves this for the sharp null in Section 4.3; the operational median claim assumes Equation (1).
7. **Target sampling.** A benchmark-distribution claim requires cases drawn from a frozen distribution, a fixed stratified schedule, or known adaptive probabilities. Arbitrary adaptive case selection changes the estimand even though the e-process remains mathematically valid for the selected sequence.

The latent WSL state may be Markov, non-Markov, adversarially persistent, or affected by all prior completed pairs. It may also have heavy-tailed shocks. Those features are in \(\mathcal F_{i-1}\) or the pair's potential outcomes; they do not break the proof.

## 3. Carryover is an identification problem

### 3.1 Raw AB/BA does not cancel differential carryover

Consider an additive model for log latency:

\[
Y_{i1}(a)=\mu_i+p_1+\tau\,1\{a=B\},
\]

\[
Y_{i2}(a\mid h)=
\mu_i+p_2+\tau\,1\{a=B\}+\kappa_{h\to a},
\]

where \(h\) is the first arm. The signed B-minus-A contrast is

\[
D_i^{AB}
=\tau+(p_2-p_1)+\kappa_{A\to B},
\]

\[
D_i^{BA}
=\tau-(p_2-p_1)-\kappa_{B\to A}.
\]

Fair order randomization therefore identifies

\[
E[D_i\mid\mathcal F_{i-1}]
=\tau+\frac12
\left(\kappa_{A\to B}-\kappa_{B\to A}\right).
\tag{3}
\]

The period effect cancels. Differential carryover does not. For any observed value in (3), infinitely many combinations of \(\tau\) and carryover difference produce the same distribution. The direct effect is not identified.

This is why the unguarded simulator makes every method fail. The anytime test controls its error only when its conditional null is true; it cannot repair a confounded estimand.

### 3.2 Guarded pair construction

A direct-effect pair uses two independently prepared subtrials:

1. Lock the case, cell, cache target, warmup recipe, timeout, and epoch.
2. Draw AB or BA after that lock.
3. Before the first measured arm, execute the neutralizer.
4. For a warm cell, self-warm that same arm with the fixed untimed workload.
5. Measure once. Record timeout at the cap rather than rerunning.
6. Pad the subtrial to a fixed slot boundary or otherwise ensure that the first arm's duration cannot shift the second arm into a different host-time regime.
7. Apply the same neutralizer before the second measured arm, self-warm it if required, and measure once.

A neutralizer need not reset the universe. It must erase arm-specific influence on the next measured arm. Residual host drift that is independent of the arm label is an arbitrary slot effect and is handled by randomization.

For cold-cache cells, the recipe should restart the engine process, clear engine-owned caches, execute a fixed scrub, and verify the declared page-fault or I/O counters. WSL may not provide a literal machine-wide cold cache. The honest estimand is therefore “cold-prep policy X,” not an unobservable absolute cold state.

For warm-cache cells, each implementation warms itself from the same neutral start. The estimand includes the declared self-warm policy. Arm A must never be used to warm arm B.

### 3.3 Fallback when neutralization is not credible

Use one of these alternatives:

- Randomize only the first measured slot and use only that outcome for the direct-effect analysis. The second run may be logged as a carryover diagnostic.
- Estimate and report AB and BA sequence-policy effects separately.
- Build an explicit transition model and label the result a policy effect, not a direct implementation effect.

A carryover test followed by choosing an analysis is not a free fix. That second-stage choice conditions on observed data and needs its own valid procedure.

## 4. Identification by randomized order

Assume the guarded no-carryover potential outcomes \(L_{ij}(a)\) are fixed immediately before the fair order coin. Define the observed B-minus-A log contrast

\[
D_i=
\begin{cases}
L_{i2}(B)-L_{i1}(A), & AB,\\
L_{i1}(B)-L_{i2}(A), & BA.
\end{cases}
\tag{4}
\]

Then

\[
\begin{aligned}
E[D_i\mid\mathcal F_{i-1},\{L_{ij}(a)\}]
&=\frac12\left[
L_{i2}(B)-L_{i1}(A)
+L_{i1}(B)-L_{i2}(A)
\right]\\
&=\tau_i.
\end{aligned}
\tag{5}
\]

Equation (5) is a design identity. The two slots may have completely different levels. The machine state may depend on every prior pair. No timing distribution is invoked.

The same argument works with a conditionally biased assignment probability \(\pi_i\) if a Horvitz-Thompson score is used. The executable protocol uses \(\pi_i=1/2\) because it has lower variance and is easier to audit.

## 5. Anytime-valid heavy-tail test

### 5.1 Sign score

For a lower-is-better ratio and margin \(c\), define

\[
X_i(c)
=1\{R_i<c\}-1\{R_i>c\}
\in\{-1,0,1\}.
\tag{6}
\]

The null is

\[
H_0(c):
E[X_i(c)\mid\mathcal F_{i-1}]\le0
\quad\text{for every included pair }i.
\tag{7}
\]

Equation (7) is the conditional-median null in Equation (1). It allows arbitrary autocorrelation and conditional heteroskedasticity.

### 5.2 Betting theorem

For any fixed \(\lambda\in[0,1)\), define

\[
M_t(\lambda)
=\prod_{i=1}^t(1+\lambda X_i).
\tag{8}
\]

**Theorem 1.** Under (7), \(M_t(\lambda)\) is a nonnegative supermartingale with initial value one.

**Proof.** Because \(X_i\in[-1,1]\) and \(\lambda<1\), every factor is nonnegative. Also

\[
E[M_t(\lambda)\mid\mathcal F_{t-1}]
=M_{t-1}(\lambda)
\left(1+\lambda E[X_t\mid\mathcal F_{t-1}]\right)
\le M_{t-1}(\lambda).
\]

QED.

Let \(w_j\ge0\), \(\sum_jw_j=1\), and use a finite grid \(\lambda_j\). The mixture

\[
E_t=\sum_jw_jM_t(\lambda_j)
\tag{9}
\]

is also a nonnegative supermartingale. Ville's inequality gives

\[
P_{H_0}\left(\sup_{t\ge0}E_t\ge1/\alpha\right)\le\alpha.
\tag{10}
\]

This proves validity under arbitrary stopping. The default implementation mixes 49 fractions from 0.02 through 0.98.

Latency magnitude never enters (6). The latency distribution can have no finite variance. A single enormous run changes one score from \(+1\) to \(-1\); it cannot dominate an average.

### 5.3 Design-based sharp-null corollary

Suppose the slot-specific log effects obey the sharp null

\[
L_{ij}(B)-L_{ij}(A)\ge\delta=\log c,
\quad j=1,2.
\tag{11}
\]

Let \(D_i^{AB}\) and \(D_i^{BA}\) be the two values in (4). Their sum is

\[
D_i^{AB}+D_i^{BA}
=
[L_{i1}(B)-L_{i1}(A)]
+[L_{i2}(B)-L_{i2}(A)]
\ge2\delta.
\tag{12}
\]

Both possible contrasts cannot be below \(\delta\). Under the fair order coin, the average of the two sign scores is therefore at most zero. Thus (7) follows from randomization for the sharp null (11), with no stochastic timing assumption.

Rejecting (11) proves that the universal no-speedup claim is untenable. It does not by itself identify the median of the unobserved same-slot treatment effects; Section 1.3 still applies.

### 5.4 Bounded average log-effect score

Signs trade power for robustness. When a timer floor \(m>0\) and timeout cap \(M\) are predeclared, every arm value is deterministically mapped into \([m,M]\). Set

\[
K=\log(M/m),\qquad
D_i=\log R_i,\qquad
\delta=\log c,
\]

and

\[
X_i^{\mathrm{log}}
=\frac{\delta-D_i}{K+|\delta|}.
\tag{13}
\]

Because \(D_i\in[-K,K]\), (13) lies in \([-1,1]\). Under the direct-effect null

\[
E[D_i\mid\mathcal F_{i-1}]\ge\delta,
\tag{14}
\]

its conditional mean is non-positive. The same betting proof applies.

Timeouts must be kept at \(M\). Deleting them destroys the bound and changes the estimand. The claim is about capped log latency.

### 5.5 Optional skipping and adaptive measurement choice

Let \(I_i\in\{0,1\}\) indicate that a particular cell is selected for the next pair, with \(I_i\) measurable before the outcome. Replacing each factor by

\[
1+\lambda I_iX_i
\]

preserves the proof because

\[
E[I_iX_i\mid\mathcal F_{i-1}]
=I_iE[X_i\mid\mathcal F_{i-1}]\le0.
\]

The experimenter may therefore choose the next cell from all past data. Choosing whether to include the current pair after seeing its latency is not predictable and is invalid.

### 5.6 Confidence-sequence inversion

On a fixed threshold grid, compute \(E_t(c)\) for every \(c\). For the lower-is-better sign score, \(E_t(c)\) is nondecreasing in \(c\). The smallest threshold whose e-value crosses gives an anytime upper confidence bound for a common randomized-pair median. The engine uses explicit promotion margins instead because the required claims are known in advance.

## 6. Multiple metrics, states, candidates, and epochs

### 6.1 Intersection-union promotion

Let \(\mathcal C\) be the required cells, for example:

- short warm decode;
- long warm decode;
- short cold prefill guardrail;
- long cold prefill guardrail.

A candidate is promoted in epoch \(e\) only if every required cell crosses its promotion boundary and its minimum coverage requirement:

\[
\mathrm{PROMOTE}_e
=\bigcap_{c\in\mathcal C}
\left\{\sup_t E_{c,e,t}\ge1/\alpha_e\right\}.
\tag{15}
\]

The global null is that at least one required cell fails its claim. If cell \(c^*\) is a true null, then

\[
P(\mathrm{PROMOTE}_e)
\le
P\left(\sup_t E_{c^*,e,t}\ge1/\alpha_e\right)
\le\alpha_e.
\tag{16}
\]

No Bonferroni division across required cells is needed for this intersection-union test. Any alternative promotion route using “cell X or cell Y” creates a union and must receive a separate alpha allocation or a closed-testing construction.

### 6.2 Candidate and epoch spending

For candidate index \(k=0,1,\ldots\), allocate

\[
\alpha_k
=\alpha_{\mathrm{program}}
\frac{6}{\pi^2(k+1)^2}.
\tag{17}
\]

For change epoch \(e\), allocate

\[
\alpha_{k,e}
=\alpha_k\frac{6}{\pi^2(e+1)^2}.
\tag{18}
\]

Both series are summable. By a union bound, the probability of ever promoting any nonqualifying candidate in any epoch is at most \(\alpha_{\mathrm{program}}\). The epoch may begin at an adaptive stopping time.

The example protocol starts from a candidate allocation of 0.01. Its epoch-zero promotion allocation is approximately 0.006079, so the e-value threshold is approximately 164.49.

### 6.3 Rejection control

Rejection is an “any harmful cell” union. The rejection budget \(\beta\) is split across cells by predeclared weights and then across epochs by the same summable schedule. A harmful e-process uses

\[
X_i^{\mathrm{harm}}(r)
=1\{R_i>r\}-1\{R_i<r\}.
\]

Crossing any allocated rejection threshold rejects the candidate. The resulting probability of ever rejecting a candidate whose cells all satisfy their rejection guardrails is at most the total \(\beta\).

### 6.4 Adaptive cases

There are three defensible choices:

1. Make each important case or case family its own required cell.
2. Draw cases from a frozen distribution with a logged probability.
3. Use a predictable adaptive case policy and explicitly claim performance under that policy.

Selecting an easy case because the candidate looked weak changes a fixed-distribution claim. Logging the selection probability permits inverse-probability methods, but their bounded range worsens as the minimum probability approaches zero.

## 7. Adaptive escalation by value of information

The Bayesian escalation policy chooses where to measure. It never substitutes for the frequentist promotion gate.

### 7.1 Exact finite dynamic program

Let \(\theta\) be a finite latent candidate class, such as harmful, neutral, small win, or large win. Let \(b\) be the posterior over \(\theta\), \(m\) the mask of completed stages, and \(g\) the evidence/gate state. An available measurement \(a\) has cost \(c_a\), outcome \(y\), likelihood \(p(y\mid\theta,a)\), and successor state \(T(s,a,y)\).

With keeping A normalized to zero, the Bellman equation is

\[
V(s)=\max\left\{
0,
1\{g\text{ permits promotion}\}
E_b[u_{\mathrm{promote}}(\theta)],
\max_{a\in\mathcal A(s)}
\left[-c_a+\sum_y p(y\mid s,a)V(T(s,a,y))\right]
\right\}.
\tag{19}
\]

A stage can require predecessors. The full campaign is marked required for promotion. Because the horizon and state space are finite, backward induction is optimal. `solve_escalation_dp` implements (19).

The default demonstration uses fixture, short model, long prompt, and full campaign stages. Its exact first action is the fixture.

### 7.2 Practical rollout and certified bound

Use a feasible base policy that either keeps A or pays the complete remaining prerequisite closure, ignores those future observations, and promotes only when current posterior expected deployment utility covers that cost. One-step lookahead over this base policy is a feasible rollout, hence its value \(L(s)\) is a lower bound on \(V(s)\).

Perfect information with no measurement or gate cost gives

\[
U(s)=E_b\left[\max\{0,u_{\mathrm{promote}}(\theta)\}\right].
\tag{20}
\]

It is an upper bound on every admissible policy. Therefore

\[
L(s)\le V(s)\le U(s),
\qquad
V(s)-L(s)\le U(s)-L(s).
\tag{21}
\]

The executable solver reports the rollout action, lower bound, exact value, perfect-information upper bound, and the certified gap.

### 7.3 Economic stopping rule

The maximum useful value of another measurement is at most

\[
U(s)-V_{\mathrm{stop}}(s).
\]

If every available measurement costs at least this amount, stop. If promotion is not yet permitted, the result is `INCONCLUSIVE` rather than a favorable guess.

The protocol also uses a hard measurement cap

\[
C_{\max}
=\min\{C_{\mathrm{engineering}},
C_{\mathrm{plausible\ payback}}\}.
\tag{22}
\]

The plausible-payback term is the maximum credible deployment opportunities times baseline cost per opportunity times the largest defensible fractional saving, after discounting implementation and maintenance cost. Measuring longer than (22) cannot be rational under the declared model.

## 8. Nonstationarity and change points

### 8.1 What the e-process already tolerates

Autocorrelation and gradual drift do not invalidate the betting proof if the conditional null remains true. No stationary distribution is required.

A “current regime” claim is different. Old evidence cannot be silently pooled after a driver, clock, host-load, or cache-policy change.

### 8.2 Epoch reaction rule

When a change alarm fires after pair \(t\):

1. Pair \(t\) remains in the old epoch.
2. Close that epoch. Do not delete or relabel its observations.
3. Start epoch \(e+1\) at pair \(t+1\).
4. Reset performance evidence and use allocation \(\alpha_{k,e+1}\).
5. Require every promotion cell to cross within the same new epoch.

The union of promotion events across epochs is controlled by (18), even when alarm times are adaptive.

### 8.3 Anytime-valid bounded anchor alarm

Let an external anchor metric be mapped by fixed bounds into \(W_t\in[0,1]\). Under a stable epoch, declare

\[
\ell\le E[W_t\mid\mathcal F_{t-1}]\le u.
\tag{23}
\]

Define

\[
X_t^+=\frac{W_t-u}{\max(u,1-u)},
\qquad
X_t^-=\frac{\ell-W_t}{\max(\ell,1-\ell)}.
\tag{24}
\]

Both scores lie in \([-1,1]\). Under (23), both have non-positive conditional mean. Run two betting e-processes at \(\delta/2\). A union bound gives

\[
P(\text{any false anchor alarm})\le\delta.
\]

`TwoSidedAnchorAlarm` implements (24). Spend \(\delta\) across epochs if the detector is restarted.

The envelope must be frozen from external calibration or account for calibration uncertainty. Estimating it from the same campaign and then treating it as fixed is not valid.

No distribution-free method can promise both a small false-alarm probability and prompt detection of every possible change. The guarantee above is false-alarm control under (23), not universal detection.

### 8.4 Why outcome-based outlier deletion is invalid

Take a simple null with exchangeable, symmetric, nondegenerate paired differences \(D_1,\ldots,D_n\), each with mean zero. Suppose the experimenter deletes the largest difference because it is a “slow B outlier.” The retained mean is

\[
\bar D_{-\max}
=\frac{\sum_iD_i-\max_iD_i}{n-1}.
\]

Hence

\[
E[\bar D_{-\max}]
=-\frac{E[\max_iD_i]}{n-1}<0.
\tag{25}
\]

The deletion creates a systematic apparent B speedup under the null. A t test or ordinary bootstrap applied afterward ignores this selection and has the wrong null distribution.

Deleting the largest absolute residual symmetrically may avoid the sign bias in a special symmetric model, but a conventional test still fails to account for the data-dependent transformation. A predeclared robust statistic can be valid if its randomization distribution or e-process is derived for that statistic. Ad hoc deletion after inspection is not that procedure.

Infrastructure omissions must use external codes fixed without reference to latency. When in doubt, keep the pair and record a timeout/cap.

## 9. Variance and committed-token throughput

A candidate can improve the median and still be poor in deployment because its variance or retry tail increases.

### 9.1 Preferred fixed-workload estimand

For each arm, request the same fixed number \(Q\) of correct committed tokens and measure total wall time from start through retries, rollback, and commit. Compare

\[
R_i^{(Q)}=\frac{T_{B,i}(Q)}{T_{A,i}(Q)}.
\]

This directly optimizes committed tokens per wall-clock hour. A timeout is a capped latency or a correctness failure; it is not removed. Candidate-specific variance is part of the outcome.

A fixed wall-time design can instead compare correct committed-token counts, with a predeclared count cap. The bounded mean betting construction applies after orientation.

### 9.2 Renewal formulation

For variable cycles with committed tokens \(C\) and wall time \(T>0\), long-run throughput is

\[
\rho=\frac{E[C]}{E[T]},
\]

not \(E[C/T]\). For a trial value \(r\),

\[
E[C-rT]\ge0
\quad\Longleftrightarrow\quad
\rho\ge r.
\tag{26}
\]

With bounded token counts and timeout-capped time, bet on a normalized version of \(C-rT\) and invert over \(r\) to obtain an anytime throughput confidence sequence. Interpreting (26) as a long-run deployment rate requires a stable renewal or ergodic model; the finite-campaign conditional-mean test itself does not require IID cycles.

## 10. A/B/C/D factorial extension

Suppose \(k\) binary optimization factors produce \(2^k\) combinations. For A/B/C/D, \(k=2\). Let \(x\in\{-1,+1\}^k\), and let

\[
f_S(x)=\prod_{j\in S}x_j
\]

encode a main effect or interaction. At step \(i\), choose combination \(X_i=x\) with known predictable probability \(\pi_i(x)\ge\pi_{\min}>0\). Map the outcome into \(Y_i(x)\in[0,1]\).

The Horvitz-Thompson factorial score

\[
H_{i,S}
=\frac{f_S(X_i)Y_i(X_i)}{2^k\pi_i(X_i)}
\tag{27}
\]

satisfies

\[
E[H_{i,S}\mid\mathcal F_{i-1}]
=2^{-k}\sum_x f_S(x)
E[Y_i(x)\mid\mathcal F_{i-1}],
\tag{28}
\]

which is the uniform factorial contrast. Since

\[
|H_{i,S}|\le\frac{1}{2^k\pi_{\min}},
\]

it can be centered at a contrast margin, normalized to \([-1,1]\), and used in the same e-process. Adaptive assignment and sequential stopping are valid because \(\pi_i\) is chosen before the outcome.

If an expensive combination receives zero probability, the corresponding contrast is not identified. Very small probabilities preserve identification but inflate range and reduce power.

Strong heredity and sparsity are useful for screening, not a license for unadjusted post-selection claims. Use fresh confirmation data after selecting interactions, or allocate alpha over a predeclared hereditary family. Promotion can then require the selected main-effect and interaction cells through an intersection-union gate.

## 11. Correctness and quality

The deterministic parity gate is zero tolerance on a finite corpus:

- any mismatch, crash, invalid token, or required hash mismatch rejects the candidate;
- passing means exact parity only on the tested inputs;
- performance evidence can never compensate for a parity failure.

For a broader input distribution, add a separate statistical quality or failure-rate gate. Promotion is the intersection of deterministic corpus pass, statistical quality evidence, and all required performance evidence. This preserves performance false-promotion control. It does not turn a finite deterministic corpus into a universal correctness proof.

Rare catastrophic failures should be logged as failures, not omitted as timing outliers. Case selection for the correctness campaign must not be conditioned on favorable performance outcomes.

## 12. Executable engine protocol

### 12.1 Before the campaign

1. Freeze baseline and contender commits, binaries, model/checkpoint/tokenizer hashes, benchmark cases, metric definitions, cache recipes, warmup counts, timer floor, timeout cap, and economic horizon.
2. Compute the candidate alpha allocation using (17).
3. Generate `protocol.json` and record its SHA-256.
4. Create a private randomization seed. Record only its commitment before the first pair.
5. Freeze the case-selection policy and the external anchor envelope.

Commands:

```bash
python tools/anytime_ab.py protocol \
  --candidate-id <candidate> \
  --output scratch/anytime-ab/protocol.json

python tools/anytime_ab.py log-template \
  --protocol scratch/anytime-ab/protocol.json \
  --output scratch/anytime-ab/pairs.csv

python tools/anytime_ab.py make-seed \
  --output scratch/anytime-ab/randomization.seed
```

The seed file belongs to a trusted scheduler. Do not expose it to the person selecting cases until the campaign closes.
The scheduler must append every issued draw to an immutable log and reject duplicate or revised pair IDs. Querying several candidate case IDs and keeping a favorable order is outcome-independent assignment shopping and is not permitted.

### 12.2 One randomized pair

After locking the case and preparation plan:

```bash
python tools/anytime_ab.py randomize \
  --secret-file scratch/anytime-ab/randomization.seed \
  --protocol scratch/anytime-ab/protocol.json \
  --epoch 0 \
  --pair-id full-long-warm-0001 \
  --cell decode_long_warm \
  --case-id long-4096-001
```

Run the returned order. Apply the neutralizer independently before each measured arm. Use fixed slot pacing. Never rerun because the ratio looks implausible.

### 12.3 Required log content

The example JSON contains the complete field list. At minimum, log these groups:

| Group | Required content |
|---|---|
| Identity | protocol hash; candidate ID; baseline and contender commits; binary, model, checkpoint, and tokenizer hashes |
| Selection | stage; cell; case ID; case-selection probability; case-lock timestamp; epoch |
| Randomization | seed commitment; HMAC draw; AB/BA order |
| Preparation | cache recipe hash; process restart; scrub result; warmup count and wall time; preparation success |
| Outcomes | A and B total latency; prefill; decode; requested and committed tokens; expert bytes; acceptance; retries; timeout flags |
| Correctness | parity result; mismatch identifier; output or logit hash; quality-gate result |
| Hardware | host/WSL/kernel/driver/CUDA fingerprint; GPU clocks, power, and temperature; host load; page faults; disk bytes; cache counters |
| Validity | external fault code; inclusion decision time; total campaign cost; change-alarm state |

Extra hardware counters are diagnostics. They cannot be used to delete a run after seeing its latency unless that exclusion rule was predeclared and its validity was proved.

### 12.4 Online stopping

For candidate \(k\), epoch \(e\), and required cell \(c\):

- promote evidence threshold: \(E_{c,e}\ge1/\alpha_{k,e}\);
- reject evidence threshold: \(E^{\mathrm{harm}}_{c,e}\ge1/\beta_{c,e}\);
- default full-cell minimum: 12 guarded pairs;
- default full-cell maximum: 80 guarded pairs;
- deterministic parity failure: immediate `REJECT`;
- undeclared exclusion, failed neutralization, identity mismatch, or unverifiable assignment: `INVALID`;
- any allocated harm crossing: `REJECT`;
- every required cell crossing in the same epoch: `PROMOTE`;
- measurement cap, required-cell pair cap, or operator closure with weak evidence: `INCONCLUSIVE`;
- otherwise: `CONTINUE`.

The engine freezes the first terminal decision and ignores later evidence rows. This makes replay match the declared stopping rule.

Replay a campaign after revealing the seed:

```bash
python tools/anytime_ab.py decide \
  --protocol scratch/anytime-ab/protocol.json \
  --log scratch/anytime-ab/pairs.csv \
  --secret-file scratch/anytime-ab/randomization.seed \
  --closed
```

### 12.5 Conditions forcing the full campaign

A surviving candidate must enter the full campaign before promotion when any of the following holds:

- it changes floating-point order, routing, cache residency, asynchronous scheduling, or I/O;
- short and long prompts, or cold and warm states, disagree;
- multiple optimizations interact;
- evidence is near either practical margin;
- a change alarm creates a new epoch;
- the remaining full campaign still has positive value of information and can plausibly repay its cost.

The fixture and short-model stages may reject or stop for futility. They never promote.

### 12.6 Kill criteria

Do not use a procedure that:

- needs IID timing or a fixed sample size for its stated guarantee;
- peeks at a fixed-sample p-value and stops;
- conditions inclusion, reruns, case choice, or metric choice on the current outcome;
- claims a direct A/B effect from an unneutralized carryover sequence;
- pools evidence across a declared change point;
- treats passing a finite parity corpus as universal correctness;
- spends more measurement wall time than the largest credible deployment saving.

## 13. CPU simulation

### 13.1 Environment

`HostileWSLSimulator` includes:

- a three-state persistent Markov machine regime;
- arm- and sequence-dependent cache carryover;
- state-dependent log-normal noise;
- Pareto latency shocks with default shape 1.35, giving finite mean and infinite variance for the Pareto component;
- a deterministic clock multiplier change after run 58;
- optional contender bursts lasting two through eight runs;
- candidate-specific log-noise and Pareto-shock multipliers;
- guarded and deliberately confounded protocols.

The methods are a three-run ratio of medians, a naive ratio of sums, fixed paired t test on log ratios, repeatedly peeked paired t test, IID paired bootstrap median interval, and the anytime sign e-process.

### 13.2 Monte Carlo results

The checked-in run uses 1,000 campaigns per scenario, seed 20260831, 80 maximum pairs, and 399 bootstrap replicates. Equality at the 0.98 promotion margin is treated as the least-favorable nonqualifying boundary.
Regret is measurement wall time plus the deployment loss over 50,000 opportunities at a 500 ms baseline. `INCONCLUSIVE` is treated as keeping A; promoting a slower B incurs its excess deployment time.

#### Guarded boundary, structural ratio 0.98

| Method | Promotion / boundary error | Expected pairs | Expected campaign seconds | Expected regret seconds |
|---|---:|---:|---:|---:|
| Three-run median | 0.506 | 3.000 | 3.849 | 250.849 |
| Naive mean | 0.507 | 10.000 | 12.814 | 259.314 |
| Fixed paired t | 0.061 | 24.000 | 33.081 | 502.581 |
| Peeked paired t | 0.240 | 62.498 | 94.780 | 474.780 |
| IID bootstrap median | 0.053 | 24.000 | 33.081 | 506.581 |
| Anytime sign e-process | 0.031 | 77.912 | 119.074 | 603.574 |

The anytime method is conservative and frequently inconclusive at the exact margin. Repeatedly peeking at the fixed t test raises the boundary promotion rate to 0.240.

#### Guarded slower contender, structural ratio 1.04

| Method | False promotion | Rejection | Expected campaign seconds |
|---|---:|---:|---:|
| Three-run median | 0.312 | 0.570 | 4.044 |
| Naive mean | 0.300 | 0.562 | 13.346 |
| Fixed paired t | 0.002 | 0.092 | 32.073 |
| Peeked paired t | 0.027 | 0.395 | 86.885 |
| IID bootstrap median | 0.004 | 0.122 | 32.073 |
| Anytime sign e-process | 0.001 | 0.082 | 118.084 |

#### Guarded faster contender, structural ratio 0.92

| Method | Promotion | False rejection | Expected pairs | Expected campaign seconds |
|---|---:|---:|---:|---:|
| Three-run median | 0.725 | 0.209 | 3.000 | 3.874 |
| Naive mean | 0.693 | 0.206 | 10.000 | 13.595 |
| Fixed paired t | 0.283 | 0.000 | 24.000 | 31.241 |
| Peeked paired t | 0.786 | 0.005 | 34.926 | 49.762 |
| IID bootstrap median | 0.486 | 0.000 | 24.000 | 31.241 |
| Anytime sign e-process | 0.615 | 0.000 | 53.988 | 77.349 |

The sign method pays for robustness with more pairs. The bounded log-effect score is available when a capped average-log estimand is acceptable and more power is needed.

#### Slower contender under unguarded carryover

| Method | False promotion |
|---|---:|
| Three-run median | 0.801 |
| Naive mean | 0.890 |
| Fixed paired t | 0.733 |
| Peeked paired t | 0.985 |
| IID bootstrap median | 0.884 |
| Anytime sign e-process | 0.990 |

This is not a failure of Ville's inequality. The raw sequence makes the observed conditional median favor B, so the direct-effect null is false for the confounded measurement process even though B's structural multiplier is 1.04.

### 13.3 Replay seed

Seed 0 uses a guarded cold protocol and a structural B/A multiplier of 1.05. The first three pairs are:

| Pair | Order | A ms | B ms | B/A |
|---|---|---:|---:|---:|
| 1 | BA | 599.663 | 764.507 | 1.274894 |
| 2 | BA | 1007.990 | 552.822 | 0.548440 |
| 3 | AB | 726.202 | 585.079 | 0.805670 |

The conventional ratio of three-run medians is 0.805670 and promotes B, although B is structurally 5% slower. Every measured run starts with zero simulated cache hotness, so this replay is a hostile-noise failure rather than carryover confounding. The anytime procedure does not promote on the three-pair prefix.

### 13.4 Reproduction

```bash
build/anytime-ab-cpu.sh
```

Or run the components directly:

```bash
PYTHONPATH=tools python tools/test_anytime_ab.py -v

python tools/simulate_anytime_ab.py simulate \
  --trials 1000 \
  --seed 20260831 \
  --max-pairs 80 \
  --bootstrap-reps 399 \
  --output-dir scratch/anytime-ab

python tools/simulate_anytime_ab.py replay \
  --seed 0 \
  --true-ratio 1.05 \
  --output scratch/anytime-ab/replay-three-median-seed-0.csv

python tools/anytime_ab.py dp-demo
```

## 14. Guarantee ledger

| Claim | Guarantee | Required assumption |
|---|---|---|
| Sign promotion in one cell/epoch | Anytime false crossing at most allocated alpha | Conditional sign-null (7); predictable inclusion |
| Sharp direct-effect test | Design-based anytime validity | Fair order; fixed potential outcomes before draw; no differential carryover; sharp slot null (11) |
| Bounded average log effect | Anytime validity for capped conditional mean | Fair order; no differential carryover; predeclared bounds; conditional mean null (14) |
| Multi-cell promotion | False promotion at most epoch alpha | At least one required cell null is true; all required cells must cross |
| Program-wide candidates/epochs | Ever-false-promotion at most program alpha | Summable allocations; no reuse of spent alpha |
| Change alarm | Ever-false-alarm at most allocated delta | Bounded anchor and stable conditional mean envelope (23) |
| Exact escalation policy | Optimal for the finite supplied model | Correct finite prior, likelihoods, utilities, costs, and stage graph |
| Rollout policy | Value within reported upper-minus-lower gap | Same finite model; lower policy feasible; perfect-information bound valid |
| Deterministic parity | Exact on tested corpus | Deterministic harness and complete execution of listed cases |

No row claims universal correctness, universal change detection, or a direct effect under untreated carryover.

## References

- Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. “Time-uniform, nonparametric, nonasymptotic confidence sequences.” *Annals of Statistics* 49(2), 2021. DOI: 10.1214/20-AOS1991.
- Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. “Time-uniform Chernoff bounds via nonnegative supermartingales.” *Probability Surveys* 17, 2020. DOI: 10.1214/18-PS321.
- Waudby-Smith, I. and Ramdas, A. “Estimating means of bounded random variables by betting.” arXiv:2010.09686.
- Shi, D. and Ye, T. “Behavioral carry-over effect and power consideration in crossover trials.” *Biometrics* 80(2), 2024. DOI: 10.1093/biomtc/ujae023.
