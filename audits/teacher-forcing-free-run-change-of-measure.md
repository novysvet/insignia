# Teacher-forced metrics, free-running failure, and the minimum viable certificate

## Status

This note supplies the exact path-space and stopped-time identities, the sharp
minimax failure bound, finite-state impossibility examples, a separate greedy
analysis, an anytime data-collection rule, and the selected-controller
extension requested by
`tasks/inference-math-frontier-20260831/01-teacher-forcing-free-run-change-of-measure.md`.

Reproducible CPU artifacts:

- `tools/teacher_forcing_free_run.py`: exact finite-state oracle, adversary
  generator, sharp bounds, greedy checker, and sequential certifier;
- `tools/test_teacher_forcing_free_run.py`: exact path-law, stopped-law,
  cascade, greedy, search, and confidence-sequence tests;
- `tools/evaluate_teacher_forcing_free_run.py`: deterministic experiment driver;
- `scratch/teacher-forcing-free-run/`: exact rational examples, search output,
  confidence widths, coverage simulation, and the machine-readable summary.

The main decision is:

> For stochastic decoding, total next-token KL averaged under the actual exact
> sampling occupancy, together with exact-policy failure probability, gives a
> sharp and minimax-complete certificate through inverse binary KL. The current
> mean same-token PPL field is not that quantity. For greedy decoding, the exact
> certificate is pointwise top-1 preservation, equivalently a pairwise
> margin/error inequality, along the common causal prefix. A production failure
> gate still needs randomized candidate-policy trajectories and an anytime upper
> confidence sequence.

The rare-history example at the current 3.5% PPL threshold uses rational hazard
`1/30`. It has 100% local top-1 agreement and a per-token PPL increase of only
`30/29 - 1 = 3.4483%`, yet under stochastic decoding its failure probability is
`81.6415%` by 50 tokens and `99.9830%` by 256 tokens. This does not contradict
any KL theorem: the total trajectory KL is respectively `1.6951` and `8.6788`.
The example does not fail under greedy decoding because the safe token remains
the argmax.

## 1. Path-space setup

Let `X_t` be the token at step `t`, let `H_t=X_{1:t}`, and let
`F_t=sigma(H_t)`. For a finite vocabulary `V`, write

\[
p_t(x\mid h)=P(X_t=x\mid H_{t-1}=h),\qquad
q_t(x\mid h)=Q(X_t=x\mid H_{t-1}=h).
\]

The induced path laws are

\[
P^T(x_{1:T})=\prod_{t=1}^T p_t(x_t\mid x_{<t}),\qquad
Q^T(x_{1:T})=\prod_{t=1}^T q_t(x_t\mid x_{<t}).
\]

Let `nu_t^P` be the law of `H_{t-1}` under `P`, and define the local
teacher-forced divergence

\[
\ell_t(h)=D_{\rm KL}\!\left(p_t(\cdot\mid h)\middle\|q_t(\cdot\mid h)\right).
\]

All identities below hold in the extended nonnegative reals. If a token has
positive exact probability and zero candidate probability on a `P`-reachable
history, both the relevant local term and trajectory KL are infinite.

A crucial distinction is the decoding law. If deployment samples at a given
temperature or with top-p truncation, `p_t` and `q_t` must include that exact
sampler. If deployment is deterministic greedy decoding, the policy laws are
point masses and Section 7 applies instead. Softmax KL between two logit vectors
is not trajectory KL between their deterministic greedy policies.

## 2. Exact trajectory KL chain rule

### Theorem 1: path-space chain rule

For every finite horizon `T`,

\[
\boxed{
D_{\rm KL}(P^T\|Q^T)
=\sum_{t=1}^T
E_{H_{t-1}\sim\nu_t^P}\left[\ell_t(H_{t-1})\right].
}
\]

### Proof

On every `P`-positive path for which the likelihood ratio is finite,

\[
\log\frac{P^T(X_{1:T})}{Q^T(X_{1:T})}
=\sum_{t=1}^T
\log\frac{p_t(X_t\mid H_{t-1})}{q_t(X_t\mid H_{t-1})}.
\]

Taking expectation under `P` and conditioning each summand on `F_{t-1}` gives

\[
\begin{aligned}
E_P\left[\log\frac{p_t(X_t\mid H_{t-1})}
{q_t(X_t\mid H_{t-1})}\right]
&=E_P\left[
\sum_{x\in V}p_t(x\mid H_{t-1})
\log\frac{p_t(x\mid H_{t-1})}{q_t(x\mid H_{t-1})}
\right]\\
&=E_{\nu_t^P}[\ell_t(H_{t-1})].
\end{aligned}
\]

Summing proves the finite case. If a `P`-positive, `Q`-zero transition occurs,
the first such local term and the path-space divergence are both infinite. QED.

This identity already handles the difference between exact and candidate
free-running history laws. The orientation `P||Q` is important: the chain rule
uses exact-policy occupancy and still controls the entire candidate path law.
No `Q`-occupancy replacement is permitted. The practical problem is that a
finite corpus law `mu`, or a single greedy exact trajectory, need not equal
`nu_t^P` for the stochastic policy being deployed.

## 3. Adaptive stopping and stopped change of measure

Let `tau` be a stopping time for `(F_t)`, such as the first format violation or
entry into an absorbing failure state. Set `sigma=tau wedge T`. The stopped
transcript contains `(sigma,X_{1:sigma})`; when `sigma=T`, the transcript also
reveals whether failure occurred at token `T`.

### Theorem 2: stopped KL chain rule

\[
\boxed{
D_{\rm KL}(P^{\sigma}\|Q^{\sigma})
=\sum_{t=1}^T
E_P\left[\mathbf 1\{\sigma\ge t\}\ell_t(H_{t-1})\right].
}
\]

### Proof

The event `{sigma >= t}` is `F_{t-1}`-measurable. The Radon--Nikodym derivative
of the stopped law is the product of the transition ratios through the last
observed token:

\[
\frac{dP^{\sigma}}{dQ^{\sigma}}
=\prod_{t=1}^{\sigma}
\frac{p_t(X_t\mid H_{t-1})}{q_t(X_t\mid H_{t-1})}.
\]

Therefore

\[
\begin{aligned}
D_{\rm KL}(P^{\sigma}\|Q^{\sigma})
&=E_P\left[\sum_{t=1}^T
\mathbf 1\{\sigma\ge t\}
\log\frac{p_t(X_t\mid H_{t-1})}{q_t(X_t\mid H_{t-1})}\right]\\
&=\sum_{t=1}^T E_P\left[
\mathbf 1\{\sigma\ge t\}\ell_t(H_{t-1})\right].
\end{aligned}
\]

The same extended-real argument handles support failure. QED.

The stopped sum can be strictly smaller than full trajectory KL because local
mismatch after a failure has already been observed is irrelevant to the event
`{tau <= T}`.

### Likelihood-ratio martingales

Assume first `P^t << Q^t` and define

\[
Z_t=\frac{dP^t}{dQ^t}(H_t)
=\prod_{s=1}^t\frac{p_s(X_s\mid H_{s-1})}{q_s(X_s\mid H_{s-1})}.
\]

`(Z_t)` is a nonnegative `Q`-martingale. For bounded `sigma`, optional stopping
and the stopped Radon--Nikodym identity give, for every `A in F_sigma`,

\[
P(A)=E_Q[Z_{\sigma}\mathbf 1_A].
\]

In the reverse direction, if `Q^t << P^t`,

\[
L_t=\frac{dQ^t}{dP^t}(H_t)
\]

is a nonnegative `P`-martingale and

\[
\boxed{Q(A)=E_P[L_{\sigma}\mathbf 1_A].}
\]

Two useful stopped bounds follow immediately.

1. If the stopped density ratio satisfies `L_sigma <= C` almost surely under
   `P`, then

   \[
   Q(A)\le C P(A).
   \]

2. For `alpha>1`, Holder's inequality gives

   \[
   \begin{aligned}
   Q(A)
   &\le \left(E_P L_{\sigma}^{\alpha}\right)^{1/\alpha}
   P(A)^{(\alpha-1)/\alpha}\\
   &=\exp\left(\frac{\alpha-1}{\alpha}
   D_{\alpha}(Q^{\sigma}\|P^{\sigma})\right)
   P(A)^{(\alpha-1)/\alpha}.
   \end{aligned}
   \]

These statements remain valid for data-dependent stopping because they are
proved on the stopped sigma-field. They require reverse absolute continuity;
a candidate-only support branch makes a finite density cap impossible.

## 4. The sharp KL-only failure certificate

For `p,q in [0,1]`, let

\[
\operatorname{kl}(p\|q)
=p\log\frac pq+(1-p)\log\frac{1-p}{1-q}
\]

with the usual boundary conventions, and define

\[
U_{\rm KL}(p,d)
=\sup\{q\in[p,1]:\operatorname{kl}(p\|q)\le d\}.
\]

For `p<1` and finite `d`, the endpoint is the unique right-hand solution of
`kl(p||q)=d`. In particular,

\[
U_{\rm KL}(0,d)=1-e^{-d}.
\]

### Theorem 3: bounded-functional certificate

Let `F:V^T -> [0,1]`,

\[
p=E_{P^T}[F],\qquad q=E_{Q^T}[F],\qquad
D=D_{\rm KL}(P^T\|Q^T).
\]

Then

\[
\boxed{q\le U_{\rm KL}(p,D).}
\]

The result is minimax tight over all finite probability spaces, all bounded
`F`, and all autoregressive models with vocabulary size at least two.

### Proof

Apply the Markov kernel that, conditional on a path `x`, emits a Bernoulli
variable with success probability `F(x)`. Under `P` the output is
`Bernoulli(p)`; under `Q` it is `Bernoulli(q)`. KL data processing yields

\[
\operatorname{kl}(p\|q)\le D_{\rm KL}(P^T\|Q^T)=D.
\]

If `q<p`, the claimed upper bound is immediate. If `q>=p`, inversion on the
monotone right branch gives `q<=U_KL(p,D)`.

For tightness, take a two-point space, let `F` be the indicator of one point,
and assign that point probabilities `p` under `P` and
`q=U_KL(p,D)` under `Q`. Then the full KL is exactly `kl(p||q)=D`. Embed the
two points as the first token of a finite-vocabulary autoregressive model and
make all later tokens deterministic. QED.

### Corollary 3.1: first-failure certificate

For `A={tau<=T}`, let

\[
p_{\tau}=P(\tau\le T),\qquad
D_{\sigma}=\sum_{t=1}^T E_P[
\mathbf 1\{\sigma\ge t\}\ell_t(H_{t-1})].
\]

Then

\[
\boxed{Q(\tau\le T)\le U_{\rm KL}(p_{\tau},D_{\sigma}).}
\]

This is the strongest possible bound based only on the exact-policy failure
baseline and stopped `P||Q` KL.

### Exactly when teacher-forced KL is sufficient

The expected cumulative local KL

\[
E_P\left[\sum_{t=1}^T\ell_t(H_{t-1})\right]
\]

is sufficient and minimax complete when all of the following refer to the
actual deployed stochastic policies:

- histories are distributed according to the exact policy occupancy;
- the complete next-token distributions are used, not only the realized token;
- the terms are summed over the horizon rather than reported only as a
  per-token mean;
- the exact-policy baseline `E_P F` is supplied.

Without the baseline, the minimax upper bound is one even when trajectory KL is
zero: choose `P=Q` concentrated entirely on failure. This is a separate
impossibility from history-distribution shift.

There cannot be a fixed-horizon counterexample with `E_P F -> 0`, total
trajectory KL `D -> 0`, and `E_Q F` bounded away from zero. Theorem 3 rules it
out. Counterexamples with excellent reported teacher-forced metrics therefore
exploit normalization by horizon, incomplete token scoring, a corpus law not
covering exact occupancy, the absence of a baseline, or the use of stochastic
metrics for greedy decoding.

## 5. Finite-state counterexamples

### 5.1 Binary-KL-tight rare-history cascade

Use states `safe` and absorbing `failure`, with vocabulary `{0,1}`. Under `P`,
state `safe` emits token `0` with probability one. Under `Q`, it emits token `0`
with probability `1-r` and token `1` with probability `r`; token `1` enters
`failure`. The failure state emits token `0` under both models.

On every exact-policy history,

\[
\ell_t=-\log(1-r),
\]

and token `0` remains the candidate argmax whenever `r<1/2`. Thus

\[
D_{\rm KL}(P^T\|Q^T)=-T\log(1-r),
\]

\[
P(\tau\le T)=0,
\qquad
Q(\tau\le T)=1-(1-r)^T.
\]

Since

\[
1-e^{-D}=1-(1-r)^T,
\]

this finite-state model attains Theorem 3 exactly.

For `r=1/30`, the exact forced-token PPL ratio is `30/29=1.0344827586`, and
local top-1 agreement is 100%. Exact rational dynamic programming gives:

| horizon | total `P||Q` KL | `Q` failure | sharp bound | Pinsker upper |
|---:|---:|---:|---:|---:|
| 50 | 1.6950776 | 0.8164150 | 0.8164150 | 0.9206187 |
| 256 | 8.6787972 | 0.9998298 | 0.9998298 | 1.0000000 |

For any per-token target `delta>0`, set `r=1-e^{-delta}`. Taking
`T=ceil(c/delta)` keeps mean local KL at `delta` and makes failure approach
`1-e^{-c}` as `delta` tends to zero. Taking `T` faster than `1/delta` makes
failure tend to one. There is no horizon-free guarantee from arbitrarily small
mean local KL or PPL delta.

This construction concerns sampling. Under deterministic greedy decoding both
models emit token `0`, so their greedy trajectories coincide.

### 5.2 Zero PPL delta by unobserved-mass reallocation

A single forced token does not identify a distribution. Let the forced exact
token be token `0` and set

\[
P=(0.51,0.49,0),\qquad
Q=(0.51,\varepsilon,0.49-\varepsilon).
\]

The exact and candidate NLL of token `0` are identical, the PPL ratio is one,
and top-1 agrees. If token `2` is failure, the one-step failure probability
changes from zero to nearly `0.49`. Full-vocabulary `P||Q` KL detects the
change; forced-token PPL does not.

In `tools/compare_logits.py`, the PPL field is

\[
\exp\left(\frac1T\sum_t
[-\log q_t(y_t\mid h_t)+\log p_t(y_t\mid h_t)]\right)
\]

for the supplied tokens `y_t`. This equals an empirical estimate of local KL
only when each `y_t` is sampled from the matching exact conditional law and the
histories have the required exact-policy occupancy. The MathArena path instead
uses exact greedy output as the forced sequence, so that identification does
not hold for a stochastic deployment policy.

### 5.3 Corpus support hole

Let the corpus history law `mu` put all mass on an initial context `u`, and let
the deployment exact occupancy put all mass on context `v`. Define `P=Q` on
`u`. On `v`, let `P` emit a safe token and `Q` emit a failure token. Every
corpus local metric is exactly zero, yet candidate deployment failure is one.
The density ratio `d nu^P/d mu` is infinite.

Replacing the point masses by `mu(v)=epsilon` makes corpus average local
metrics arbitrarily small while preserving any fixed behavior on the dominant
deployment context. A qualitative claim that the corpus is "close" to exact
occupancy cannot support a certificate; a quantitative overlap or density
bound is required.

### 5.4 KL without exact-policy failure

Set `P=Q` and let both laws enter failure with probability one. Local KL, JS,
cosine error, and PPL delta are all zero, but free-run failure is one. A
trajectory comparison controls change from the exact model; it does not prove
that the exact model itself satisfies the failure contract.

### 5.5 Average top-1, cosine, and MSE

One greedy token mismatch can move the candidate to an entirely different
history and cause deterministic failure. Therefore 99.9% top-1 agreement is not
a path certificate; every tested step must agree for a deterministic prompt.

Average cosine and MSE are weaker. For `gamma>0` and large `C`, take

\[
z^P=(0,-\gamma,-C),\qquad
z^Q=(-\gamma,0,-C).
\]

The argmax flips. The MSE is `2 gamma^2/3`, which tends to zero with `gamma`.
Raw and mean-centered cosine tend to one as the common background magnitude
`C` grows. Repeating an exact pair on `T-1` steps and using this pair once makes
all trajectory averages arbitrarily favorable while preserving a greedy
mismatch.

## 6. What one additional coverage quantity buys

Let `mu_t` be the logged history law and `nu_t^P` the required exact-policy
history law.

### Theorem 4: corpus-to-policy transfer under a density cap

Suppose

\[
\nu_t^P\ll\mu_t,
\qquad
\frac{d\nu_t^P}{d\mu_t}(h)\le C_t
\quad\mu_t\text{-almost surely}.
\]

If `m_t=E_{mu_t}[ell_t]`, then

\[
\boxed{
D_{\rm KL}(P^T\|Q^T)
\le \sum_{t=1}^T C_t m_t.
}
\]

### Proof

Since `ell_t>=0`,

\[
E_{\nu_t^P}\ell_t
=E_{\mu_t}\left[\frac{d\nu_t^P}{d\mu_t}\ell_t\right]
\le C_t E_{\mu_t}\ell_t=C_t m_t.
\]

Sum and invoke Theorem 1. QED.

Combining with Theorem 3 gives the computable certificate

\[
E_QF\le U_{\rm KL}\left(E_PF,\sum_t C_t m_t\right).
\]

A designed mixture supplies a known cap. If

\[
\mu_t=\rho\,\nu_t^P+(1-\rho)R_t,
\]

then `d nu_t^P/d mu_t <= 1/rho`. A cap inferred from finite raw token histories
is much harder because most deployment histories are absent from a finite
corpus. The cap must apply to the representation on which the local loss is
conditioned, or be guaranteed by randomized data collection.

The bound can be paired with simultaneous upper confidence bounds on `m_t`, but
unbounded local KL requires either a logged per-history cap, clipping with an
explicit tail correction, or a robust mean method. Direct candidate-policy
sampling avoids this additional layer.

## 7. Greedy decoding: exact pointwise characterization

Fix deterministic tie-breaking. Let the exact greedy path be
`x_1^*,...,x_T^*`, with exact-prefix history `h_{t-1}^*`. Let `z_t^P(h)` and
`z_t^Q(h)` be exact and candidate logits. Assume the exact argmax at each tested
prefix is unique, and let

\[
i_t=\arg\max_j z_{t,j}^P(h_{t-1}^*).
\]

Define exact pairwise margins and logit errors

\[
m_{t,j}=z_{t,i_t}^P-z_{t,j}^P>0,
\qquad
e_{t,j}=z_{t,j}^Q-z_{t,j}^P.
\]

### Theorem 5: weakest strict margin/error condition

The candidate and exact greedy trajectories are identical for `T` steps if
and only if, for every `t<=T` and every `j != i_t`,

\[
\boxed{e_{t,j}-e_{t,i_t}<m_{t,j}.}
\]

Equivalently, the candidate top-1 token equals `i_t` at every exact causal
prefix.

### Proof

At an exact prefix,

\[
z_{t,i_t}^Q-z_{t,j}^Q
=m_{t,j}-(e_{t,j}-e_{t,i_t}).
\]

The candidate chooses `i_t` exactly when every displayed difference is
positive. If the condition holds at step one, both policies emit the same first
token. Inductively they then reach the same next prefix, so the condition at
step two applies, and so on. Necessity follows by reversing the same argument
on each common prefix. QED.

With ties, replace the strict inequalities by the corresponding weak or strict
conditions dictated by the common deterministic tie-break rule. Treating an
unresolved tie as a failure is the robust engineering choice.

Useful sufficient but not necessary summaries are

\[
2\|e_t\|_{\infty}<m_t,
\qquad
\sqrt 2\|e_t\|_2<m_t,
\]

where `m_t` is the exact top-1/top-2 margin. These must hold pointwise at every
step; averages do not suffice.

### Implication for the current benchmark

`tools/benchmark_matharena.py` uses the exact free greedy IDs as the forced
candidate sequence, and `tools/compare_logits.py` reports candidate top-1 at
each such prefix. Provided forced replay reproduces the same candidate causal
state that free greedy decoding would have after the same tokens, 100% top-1
agreement over a prompt and horizon already certifies identical greedy tokens
for that prompt and horizon by Theorem 5.

The campaign currently records top-1 but sets `ppl_gate_pass` solely from the
3.5% PPL threshold. For a greedy production mode, the quality gate should reject
any pointwise top-1 mismatch before considering PPL. It should also log the
minimum pairwise slack, because a zero-slack or tiny-margin pass is brittle to
unlogged numerical changes.

A forced replay is not a certificate if the approximation controller, cache,
random seed, or fallback state differs from free deployment despite identical
tokens. In that case the causal state `z_t` must be included in the replay
contract or checked through direct free runs.

## 8. Sequential exact/candidate/intervention collection

The clean production certificate is a randomized trajectory experiment.
Episodes, rather than tokens, are the statistical units because the target is a
trajectory failure functional.

### 8.1 Collection rule

At episode `i`:

1. Using only the previous-episode filtration, choose probabilities
   `(alpha_i,beta_i,gamma_i)` with `beta_i>=beta_min>0`.
2. Draw source `S_i` from `{P,Q,I}` before drawing the deployment prompt. This
   prevents prompt-dependent candidate selection from biasing the direct `Q`
   subsequence.
3. Draw a fresh prompt from the deployment prompt law.
4. Run the entire episode under the selected source. `I` is a predeclared
   causal intervention, such as exact prefix followed by candidate suffix, a
   randomized exact fallback time, or candidate prefix followed by exact
   suffix.
5. Log source, source propensity, prompt stratum, trajectory failure, first
   failure time, and decoding mode. If all-source importance weighting will be
   used, also log the complete path log-likelihood under `P`, `Q`, and `I`.

The probabilities may adapt after every completed episode. They may not depend
on the unseen outcome of the current episode. Prompt-dependent allocation is
possible with stratification or propensity weighting, but the simple direct
subsequence theorem below uses a source coin independent of the current prompt.

### 8.2 Direct candidate-policy confidence sequence

Let `Y_1,Y_2,...` be failure indicators from the candidate-source episodes in
the order they occur. Under stationary deployment they are IID
`Bernoulli(q)`, where `q=Q(F=1)`. Predictable thinning by the source coin does
not change this law.

For `S_n=sum_{i=1}^n Y_i`, choose `a,b>0` and define, for a candidate null
`q_0`,

\[
E_n(q_0)=
\frac{B(S_n+a,n-S_n+b)}
{B(a,b)q_0^{S_n}(1-q_0)^{n-S_n}}.
\]

This is the likelihood ratio between a `Beta(a,b)` mixture alternative and the
simple Bernoulli null. Under `q=q_0`, `(E_n(q_0))` is a nonnegative martingale
with initial mean one. Ville's inequality gives

\[
P_{q_0}\left(\sup_n E_n(q_0)\ge\frac1\delta\right)\le\delta.
\]

Hence

\[
\mathcal C_n=\{q_0:E_n(q_0)<1/\delta\},
\qquad
U_n=\sup\mathcal C_n
\]

satisfies

\[
\boxed{P_q(\forall n:\ q\le U_n)\ge1-\delta.}
\]

The endpoint may be inspected continuously, and collection may stop the first
time `U_n` crosses a release threshold. The artifact uses the Jeffreys mixture
`a=b=1/2` and numerical inversion.

With zero observed candidate failures, the 95% anytime upper endpoints are:

| candidate trajectories | upper failure probability |
|---:|---:|
| 100 | 0.05703 |
| 500 | 0.01326 |
| 1,000 | 0.00700 |
| 5,000 | 0.00156 |
| 10,000 | 0.000817 |

These widths do not grow with token horizon. Long trajectories may have a
larger true failure probability, but the certificate directly measures that
probability rather than paying a formal factor of `T`.

### 8.3 Using every mixture trajectory

Let `M_i` be the behavior trajectory law

\[
M_i=\alpha_iP+\beta_iQ+\gamma_iI_i.
\]

If the path likelihoods are available, define

\[
W_i(x)=\frac{Q(x)}{M_i(x)},\qquad
Z_i=W_i(X_i)F(X_i).
\]

Then

\[
0\le Z_i\le B_i:=1/\beta_i,
\qquad
E[Z_i\mid\mathcal G_{i-1}]=E_QF=q.
\]

For a fixed `n`, the martingale Hoeffding bound gives

\[
q\le \frac1n\sum_{i=1}^n Z_i
+\sqrt{\frac{\sum_{i=1}^nB_i^2\log(1/\delta_n)}{2n^2}}
\]

except with probability `delta_n`. Taking

\[
\delta_n=\frac{6\delta}{\pi^2n^2}
\]

and union bounding over all `n` makes the bound anytime valid. Split the error
budget between this bound and the direct candidate confidence sequence, then
return the smaller endpoint. If full path likelihoods are not trustworthy,
ignore this estimator; the direct randomized `Q` subsequence remains valid.

Intervention episodes are valuable for locating the causal step at which a
trajectory becomes unrecoverable. An unweighted intervention failure average
is not a certificate for `Q` because `I` has a different path law.

### 8.4 Coverage check

The evaluation driver ran 5,000 simulations for each of four Bernoulli risks,
checking every sample size through 300 and using early certification stops in
two scenarios. Observed simultaneous coverage was `1.000`, `1.000`, `1.000`,
and `0.9802` for true risks `0`, `0.02`, `0.10`, and `0.50`. This simulation is
a regression test, not the proof; the proof is the martingale argument above.

## 9. Controller actions and policy selection from the same log

Let `z_t` be the logged causal state and `a_t` the approximation action. A
policy `pi(a|z)` changes the next-token law and therefore future states. Let the
behavior policy that collected episode `i` be `b_i(a|z)`. Assume the state and
action process has a common causal transition kernel

\[
K_t(dz_{t+1},dx_t\mid z_t,h_{t-1},a_t)
\]

under behavior and target policies. The action may select expert count,
precision, draft width, or exact fallback; its effect is contained in `K_t`.

### Theorem 6: sequential policy likelihood ratio

If `b_i(a|z)>0` whenever `pi(a|z)>0`, then on a logged episode

\[
\boxed{
W_i^{\pi}
=\frac{dP_{\pi}}{dP_{b_i}}
=\prod_t\frac{\pi(a_t\mid z_t)}{b_i(a_t\mid z_t)}.
}
\]

### Proof

Factor both trajectory laws into initial-state law, action probabilities, and
conditional transition kernels. The initial law and every `K_t` factor cancel;
only action probabilities remain. The cancellation is valid even though the
actions alter later states, because it is evaluated on the same realized
state/action trajectory. QED.

Thus `W_i^pi F_i` is an unbiased off-policy observation of policy `pi`'s
failure probability. The logged state must include every variable used by the
behavior action rule. Hidden action confounding invalidates the ratio.

### Selection-safe certificate for a countable policy class

Let `Pi` be countable and choose prior masses `w_pi>0` with
`sum_pi w_pi<=1`. Construct an anytime UCB for each policy at error level
`delta w_pi`. A union bound gives

\[
P\left(\forall\pi\in\Pi,\forall n:\
q_{\pi}\le U_{\pi,n}\right)\ge1-\delta.
\]

Therefore any data-dependent selection `hat pi`, including the empirical best
controller on the same log, obeys

\[
\boxed{q_{\hat\pi}\le U_{\hat\pi,n}}
\]

on the same simultaneous event. For the alpha-spending Hoeffding construction,
the policy complexity appears as

\[
\log\frac{\pi^2n^2}{6\delta w_{\pi}}.
\]

A uniform prior over `N` candidates pays `log N`. An uncountable learned class
needs a PAC-Bayes/covering argument or a held-out direct-policy stream.

### Why the certificate can still be vacuous

Per-decision importance ratios multiply. If the behavior takes every target
action with probability only `epsilon`, a length-`T` deterministic target can
have weight cap `epsilon^{-T}`. No concentration refinement can rescue absent
or exponentially weak overlap.

A whole-trajectory experiment is preferable for finalists: choose policy `pi`
for the full episode with direct mixture probability `beta_pi`. Its trajectory
density ratio against the mixture is at most `1/beta_pi`, independent of token
horizon. A practical workflow may use the shared log for discovery, freeze a
small finalist set, and continue a randomized direct-policy confidence
sequence for certification.

### Impossibility 1: no action overlap

Suppose the behavior never takes action `a=1` at a reachable state, while the
selected policy always takes it. Two causal models can agree on every logged
transition and assign failure probability zero versus one after `a=1`. The log
has the same distribution in both worlds. Any universally valid upper bound
strictly below one would be wrong in one world.

### Impossibility 2: unrestricted post-hoc selection

Let contexts be uniform on `{1,...,M}`. Safe action has failure zero and unsafe
action has failure one. After observing `n` context values, select the policy
that takes the safe action on every observed context and the unsafe action on
all others. Its same-data empirical failure is zero, even with full-information
losses on the observed contexts, but its population failure is at least
`1-n/M`. As `M` grows, the true failure approaches one. A post-hoc empirical
average is not a certificate without class complexity control or independent
validation.

## 10. Oracle and adversary implementation

`FiniteStateAR` represents a time-inhomogeneous finite-state autoregressive pair
with deterministic state updates indexed by tokens. Stochastic state updates
can be represented by token splitting. Failure monitors are included in the
state, so format automata, absorbing nonsense states, and discretized cumulative
loss budgets use the same solver.

The forward dynamic program tracks `(state,failed)` occupancy under both laws.
Its time complexity is

\[
O(T|S||V|)
\]

and its memory complexity is `O(|S|)`. It supports vocabularies of size 2--8
and was exercised at horizon 256; horizon 50 with vocabulary 8 is part of the
unit suite.

For rational kernels, occupancies, terminal laws, first-failure laws, and
failure probabilities use `fractions.Fraction`. KL is generally
transcendental, so the exact representation is a symbolic sum

\[
\sum_k c_k\log r_k
\]

with rational coefficients `c_k` and rational arguments `r_k`. The reported
floating value is derived from that expression, not used as the exact oracle.
Small path spaces can be fully enumerated to materialize `P^T` and `Q^T`; tests
compare direct path KL with both the ordinary and stopped chain rules.

The random adversary generator creates rational finite-state `P,Q` pairs and
searches attack strength under a fixed exact-policy trajectory-KL budget. With
budget `0.5`, horizon 12, vocabulary 3, four states, and 250 generated base
models, the best retained pair had

\[
P(\text{failure})=0.04779,\quad
Q(\text{failure})=0.46478,\quad
D(P^T\|Q^T)=0.49719.
\]

The sharp inverse binary-KL upper was `0.49779`, and every searched pair obeyed
its bound.

## 11. Engine logging and gate changes

The following interpretation matches the existing fields in
`tools/benchmark_matharena.py` and `tools/compare_logits.py`.

| Existing field | What it proves | What it does not prove |
|---|---|---|
| forced-token PPL delta | relative likelihood of the supplied tokens | full local KL, stochastic path risk, or greedy identity |
| mean `KL(P||Q)` on exact greedy prefixes | local distribution change on those prefixes | trajectory KL for a sampling policy unless prefixes have its occupancy |
| top-1 agreement | greedy identity only when every step agrees on replay-equivalent causal prefixes | sampling safety or an average-case greedy guarantee |
| cosine/MSE | numerical similarity diagnostic | argmax preservation without a pointwise margin relation |
| free-run answer checks | observed candidate behavior on tested prompts | a confidence level without randomized sampling and a failure count |

Recommended stochastic-mode log fields:

- deployed sampler definition, including temperature and truncation;
- per-trajectory sum of full-vocabulary `P||Q` local KL on exact-policy sampled
  histories;
- exact-policy failure flag and stopped failure time;
- randomized source `P/Q/I` and its source propensity;
- candidate free-run failure flag and the current anytime UCB.

Recommended greedy-mode fields:

- pointwise top-1 agreement, with any mismatch treated as a gate failure;
- exact top-1/top-2 margin;
- maximum pairwise error `max_j(e_j-e_i)` and minimum slack;
- a replay-equivalence identifier for controller/cache/random state.

A defensible release rule is:

1. In greedy mode, require positive minimum slack on every certified step and a
   direct candidate free-run UCB below the trajectory failure budget.
2. In stochastic mode, compute the inverse binary-KL bound only from the actual
   exact sampling occupancy and total or stopped KL, then intersect it with the
   direct candidate free-run UCB.
3. Keep PPL, JS, cosine, and average MSE as diagnostics. Do not let them override
   either path certificate.

The direct candidate UCB is the minimum additional on-policy information needed
for an assumption-light production certificate: source assignment, source
propensity, and a bounded trajectory failure outcome. Exact and intervention
runs remain useful for localization and relative-quality analysis.

## 12. Reproduction

From the repository root:

```bash
python tools/test_teacher_forcing_free_run.py
python tools/evaluate_teacher_forcing_free_run.py

python tools/teacher_forcing_free_run.py cascade \
  --horizon 50 --hazard 1/30 --vocab 8

python tools/teacher_forcing_free_run.py search \
  --budget 0.5 --horizon 12 --vocab 3 --states 4 --trials 250 \
  --output scratch/teacher-forcing-free-run/adversary-search-replay.csv

python tools/teacher_forcing_free_run.py coverage \
  --failure 0.02 --alpha 0.05 --samples 300 --trials 5000 \
  --threshold 0.08 --seed 101
```

The unit suite has 15 tests and completes on ordinary CPU in under a second in
the reference environment. The full evaluation, including 250 adversary bases
and 20,000 coverage trials, completes in several seconds.

## 13. Conclusions

1. The exact cumulative `P`-occupancy local KL is trajectory KL. Combined with
   exact-model failure, inverse binary KL is the strongest possible universal
   teacher-forcing-to-free-run bound.
2. The 3.5% mean PPL gate has no horizon-free stochastic guarantee. A rational
   two-state model passes it and fails with probability above 0.81 by 50 tokens.
3. For deterministic greedy decoding, pointwise top-1 preservation along the
   common causal prefix is necessary and sufficient. Average cosine or MSE
   cannot replace it.
4. A density-ratio cap transfers a corpus mean to exact occupancy, but raw
   history support makes such caps difficult to justify post hoc.
5. Randomized direct candidate trajectories plus an anytime confidence
   sequence provide the practical, horizon-independent failure certificate.
6. A controller selected from the same log needs simultaneous per-policy
   bounds, a PAC-Bayes/covering correction, or a fresh certification stream.
   Without action overlap or selection control, no nontrivial certificate is
   possible.

## References

- S. R. Howard, A. Ramdas, J. McAuliffe, and J. Sekhon,
  "Time-uniform, nonparametric, nonasymptotic confidence sequences,"
  *Annals of Statistics* 49(2), 2021, arXiv:1810.08240.
- N. Karampatziakis, P. Mineiro, and A. Ramdas, "Off-policy confidence
  sequences," *ICML*, 2021, arXiv:2102.09540.
- I. Kuzborskij, C. Vernade, A. Gyorgy, and C. Szepesvari, "Confident
  off-policy evaluation and selection through self-normalized importance
  weighting," arXiv:2006.10460.
