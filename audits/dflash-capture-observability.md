# Optimal target-layer observations for the DFlash2 drafter

## Status

This note answers `tasks/inference-math-frontier-20260831/03-dflash-capture-observability.md` at reference commit `0740c63`.

The checked-in engine uses target captures after layers 5, 14, 24, 33, and 42. The implementation describes them as the mean-contracted completed target outputs consumed by the five-layer DFlash2 drafter. Exact target verification preserves the committed greedy sequence, so capture selection changes speed and drafter acceptance rather than target correctness.

This submission does **not** assert that the current five locations are suboptimal. It gives the conditions under which another set would be supported by evidence and supplies a CPU-only way to evaluate candidate policies.

Artifacts:

- `tools/dflash_capture_observability.py`: exact finite Bayesian-network information, analytic Gaussian information, exact subset enumeration, greedy baselines, pair-bundle selection, Dinkelbach throughput optimization, adaptivity examples, and bit lower bounds.
- `tools/evaluate_dflash_capture_observability.py`: deterministic synthetic evaluation for small networks and `L=45` instances.
- `tools/test_dflash_capture_observability.py`: unit tests for theorems, counterexamples, costs, throughput, adaptivity, Fano bounds, and the trace schema.
- `tools/schemas/dflash_capture_trace.schema.json`: minimum exploration trace.
- `scratch/dflash-capture-observability/summary.csv`: algorithm comparison.
- `scratch/dflash-capture-observability/adaptive.csv`: context-adaptive examples.
- `scratch/dflash-capture-observability/theorem-checks.json`: exact numerical witnesses.

All reported numerical values are synthetic. They are algorithm tests, not measurements of GLM-5.3-Flash.

## 1. Objectives that correspond to the engine

Let `D` contain the prompt-derived state and any drafter state available before capture selection. Let `Y` be a future token, a target block, target logits, or the exact verification-prefix length. A selected capture set `S` reveals `Z_S=(Z_l:l in S)`.

### 1.1 Conditional log loss

Suppose the prediction rule is Bayes optimal for logarithmic loss. Its minimum expected loss is

\[
L_{\log}(S)=H(Y\mid D,Z_S).
\]

The improvement relative to no target capture is

\[
L_{\log}(\varnothing)-L_{\log}(S)
=I(Y;Z_S\mid D)=f(S).
\]

This is the precise justification for mutual information. If the fixed drafter cannot realize the Bayes rule, `f(S)` is an information ceiling rather than the measured engine gain.

### 1.2 Exact block error

For finite block-valued `Y`, the Bayes block-success probability is

\[
p_{\mathrm{block}}(S)
=E\left[\max_y P(Y=y\mid D,Z_S)\right].
\]

The Bayes block-error probability is `1-p_block(S)`. The discrete oracle in the artifact computes it exactly.

### 1.3 Verification-prefix reward

Let `Y=(Y_1,...,Y_K)`, where `K<=8`. For a proposed block `yhat`, define

\[
A(Y,\hat y)=\max\{r:Y_{1:r}=\hat y_{1:r}\}.
\]

The Bayes-optimal expected accepted prefix is

\[
a(S)=E\left[
\max_{\hat y}
\sum_{r=1}^{K}P(Y_{1:r}=\hat y_{1:r}\mid D,Z_S)
\right].
\]

This loss is closer to DFlash2 than unconditional correlation. The exact finite oracle computes this quantity for sequence-valued `Y`.

### 1.4 Renewal throughput

Let `T(S)>0` be total round time, including capture traffic on the critical path, drafter work, verification, fallback, and state restoration. Under IID renewal rounds,

\[
\rho(S)=\frac{E[A(S)]}{E[T(S)]}.
\]

The same ratio holds under a stationary ergodic Markov-renewal policy after expectations are taken with respect to its stationary decision-epoch law. It is not `E[A/T]`.

The principal engine decision should use `rho`, with information retained as a diagnostic and model-selection quantity.

## 2. Exact characterization of conditional-information submodularity

Let `V={1,...,L-1}` and assume all conditional mutual informations below are finite. Define the marginal value

\[
\Delta_e(S)=f(S\cup\{e\})-f(S)
=I(Y;Z_e\mid D,Z_S).
\]

### Theorem 1: monotonicity needs no graphical assumption

For every joint law of `(D,Y,Z_V)`, `f` is normalized and monotone:

\[
f(\varnothing)=0,
\qquad
f(S\cup\{e\})-f(S)\ge 0.
\]

#### Proof

The marginal is conditional mutual information, which is nonnegative:

\[
f(S\cup\{e\})-f(S)
=I(Y;Z_e\mid D,Z_S)\ge0.
\]

QED.

Monotonicity is a statement about information before capture cost is charged. `f(S)-lambda c(S)` and throughput need not be monotone.

It also assumes that each random variable `Z_l=phi_l(H_l)` is fixed independently of the selected set. If a global byte budget causes the encoder rank of an already selected layer to be reduced when another layer is added, the observation channel itself has changed and monotonicity need not hold. A correct formulation makes each `(layer, encoder-mode)` pair an action, enforces at most one mode per physical layer, and treats a rank upgrade as revealing nested additional bits. The reference artifact evaluates one fixed encoder mode per layer; its byte field can be swept across separate runs.

### Theorem 2: exact submodularity criterion

For `A subseteq B` and `e notin B`, let `T=B\A`. Then

\[
\boxed{
\Delta_e(A)-\Delta_e(B)
=I(Z_e;Z_T\mid D,Z_A)
-I(Z_e;Z_T\mid Y,D,Z_A).
}
\]

Consequently, `f` is submodular if and only if

\[
I(Z_e;Z_T\mid D,Z_A)
\ge
I(Z_e;Z_T\mid Y,D,Z_A)
\]

for every such `(A,B,e)`.

#### Proof

Use the entropy form of a conditional mutual information:

\[
\Delta_e(A)
=H(Z_e\mid D,Z_A)-H(Z_e\mid Y,D,Z_A),
\]

\[
\Delta_e(B)
=H(Z_e\mid D,Z_A,Z_T)-H(Z_e\mid Y,D,Z_A,Z_T).
\]

Subtracting gives

\[
H(Z_e\mid D,Z_A)-H(Z_e\mid D,Z_A,Z_T)
-I(Z_e;Z_T\mid Y,D,Z_A),
\]

which is the displayed identity. Diminishing returns is exactly the nonnegativity of this difference. QED.

The criterion identifies the failure mode. Conditioning on the prediction target can create more dependence between captures than was present before conditioning. This is the information-theoretic form of synergy or explaining away.

### Corollary 2.1: a sufficient graphical model

Assume the captures are mutually conditionally independent given `(Y,D)`:

\[
p(z_V\mid y,d)=\prod_{l\in V}p_l(z_l\mid y,d).
\]

Equivalently, a valid directed graph has `(Y,D)` as parents of each `Z_l`, with independent encoder noise and no remaining path between capture nodes after conditioning on `(Y,D)`.

One explicit structural form is `Z_l=g_l(Y,D,N_l)` with mutually independent noises `(N_l)_l` conditional on `D`. This is a sufficient observation model; it is not the ordinary sequential transformer graph.

Then `f(S)=I(Y;Z_S\mid D)` is monotone submodular.

#### Proof

The factorization remains valid after conditioning on any subset `Z_A`, so

\[
I(Z_e;Z_T\mid Y,D,Z_A)=0.
\]

Theorem 2 leaves

\[
\Delta_e(A)-\Delta_e(B)
=I(Z_e;Z_T\mid D,Z_A)\ge0.
\]

QED.

This assumption is sufficient, not automatic for a transformer. A Markov chain

\[
H_0\to H_1\to\cdots\to H_L
\]

does not imply that encoded intermediate states are conditionally independent given a future target. Conditioning on a downstream object can couple earlier states. Deterministic layers can also preserve shared nuisance variables.

## 3. Smallest counterexamples and redundancy

### 3.1 Minimal non-submodular witness: XOR

Let `D` be constant. Let `X_1,X_2` be independent fair bits and define

\[
Y=X_1\oplus X_2,
\qquad Z_1=X_1,
\qquad Z_2=X_2.
\]

Then

\[
f(\varnothing)=0,
\quad f(\{1\})=0,
\quad f(\{2\})=0,
\quad f(\{1,2\})=1.
\]

Thus

\[
\Delta_2(\varnothing)=0
<1=\Delta_2(\{1\}).
\]

One capture cannot violate submodularity because the ground set has no nontrivial diminishing-returns comparison. Two binary captures and a binary target therefore form a smallest witness.

The witness can be embedded in a deterministic hidden-state process with redundant layers. Set

\[
H_0=(X_1,X_2),
\qquad H_1=H_2=H_0,
\qquad H_3=X_1\oplus X_2,
\qquad Y=H_3,
\]

and use

\[
\phi_1(H_1)=X_1,
\qquad
\phi_2(H_2)=X_2.
\]

`H_2` is an identity transform of `H_1`, yet the two encoders expose complementary coordinates and retain the XOR synergy. Deterministic layer evolution does not rescue submodularity.

### 3.2 Exact duplicate captures do not by themselves violate submodularity

Let `Y` be fair, `N` be Bernoulli with parameter `p`, and

\[
Z_1=Y\oplus N,
\qquad Z_2=Z_1.
\]

The captures share the same nuisance bit, so they are not conditionally independent given `Y` unless `p` is zero or one. Nevertheless,

\[
I(Y;Z_1)=I(Y;Z_2)=1-h_2(p),
\]

\[
I(Y;Z_1,Z_2)=1-h_2(p).
\]

The second copy has zero marginal value and the set function is submodular. This example establishes two points:

1. conditional independence in Corollary 2.1 is not necessary;
2. singleton scores can badly overstate the value of a near-duplicate collection if a method ranks layers once instead of recomputing conditional marginals.

The artifact includes this exact duplicate and a 45-layer near-duplicate Gaussian case.

## 4. Non-submodular selection with bytes and latency

### 4.1 Cost model

For a measured transfer bandwidth `beta`, define the time-equivalent additive capture cost

\[
w_l=c_l+\frac{b_l}{\beta}.
\]

This is a guarantee model, not a claim that GPU critical-path time is exactly additive. If transfers overlap or synchronize through a shared queue, use conservative additive upper bounds for the knapsack theorem, or put the measured nonadditive set time directly into the renewal objective. The latter preserves the Dinkelbach characterization but removes the density-greedy guarantee unless the set-cost structure is separately controlled.

A more general scalarization is

\[
w_l=\lambda_c c_l+\lambda_b b_l.
\]

The implementation also enforces separate hard limits on capture count, bytes, latency, and effective time. The theorem below applies when one scalar additive budget is the only binding feasibility constraint. Other limits must be inactive or encoded in that scalar cost. For count, byte, and latency limits, one conservative certified reduction is

\[
w_l=\max\left\{\frac{1}{K},\frac{b_l}{B},\frac{c_l}{C}\right\},
\qquad w(S)\le1.
\]

Every returned set then satisfies `|S|<=K`, `b(S)<=B`, and `c(S)<=C`. The approximation comparison is against the best set in this conservative region, which can be much smaller than the original three-constraint region. A grid of positive scalarizations exposes supported points on the byte-latency Pareto frontier. It does not certify unsupported nonconvex Pareto points.

### 4.2 Bundle submodularity ratio

For a bundle `B`, define

\[
\Delta(B\mid S)=f(S\cup B)-f(S).
\]

Fix a maximum interaction order `q`. For a comparator `O`, partition `O\S` into disjoint bundles with at most `q` layers. Assume each bundle costs at most `eta W`, where `W` is the scalar budget.

Define `gamma_q` as a lower bound such that, for every relevant `S` and comparator `O`, there is a permitted partition `P` satisfying

\[
\sum_{B\in P}\Delta(B\mid S)
\ge
\gamma_q\bigl(f(S\cup O)-f(S)\bigr).
\]

For `q=1`, this is the usual submodularity-ratio condition based on singleton marginals. For XOR, `gamma_1=0` and `gamma_2=1`.

The condition is weaker than submodularity and directly testable on small fitted distributions. A lower confidence bound can be estimated on held-out traces for larger instances.

### 4.3 Pair-bundle density greedy

Starting at `S=empty`, repeatedly enumerate all feasible bundles with at most `q` layers and choose

\[
B^*\in\arg\max_B
\frac{\Delta(B\mid S)}{w(B)}.
\]

Stop if no feasible bundle has positive marginal value. The implementation uses `q=2` by default and then performs improving replacements of at most two selected layers by at most two unselected layers. The polishing phase cannot reduce the guarantee.

For `L=45`, there are 44 candidate capture layers and only

\[
44+\binom{44}{2}=990
\]

singleton-or-pair candidates per iteration. With at most five captures, this is small on a CPU.

### Theorem 3: bundle-greedy guarantee

Let `f` be normalized and monotone. Suppose feasibility is exactly the scalar knapsack `w(S)<=W`, with any count, byte, or latency limits already encoded in `w`. Let `O` be any feasible set. Assume the bundle-ratio condition above holds with `gamma_q>0`, and every comparator bundle has cost at most `eta W`. Then bundle-density greedy returns `S_g` satisfying

\[
\boxed{
f(S_g)
\ge
\left(1-e^{-\gamma_q(1-\eta)}\right)f(O).
}
\]

If bundles are divisible, or the budget is filled fractionally, the factor improves to `1-exp(-gamma_q)` before rounding.

#### Proof

At a greedy state `S` with `w(S)<=(1-eta)W`, every bundle in the comparator partition fits the remaining scalar budget. The total partition cost is at most `w(O)<=W`. Therefore one comparator bundle has density at least

\[
\frac{\gamma_q(f(S\cup O)-f(S))}{W}.
\]

Greedy chooses a bundle with at least this density. If its cost is `d`,

\[
f(S\cup B^*)-f(S)
\ge
\frac{\gamma_q d}{W}
\bigl(f(O)-f(S)\bigr),
\]

where monotonicity gives `f(S union O)>=f(O)`. Hence the residual obeys

\[
f(O)-f(S\cup B^*)
\le
\left(1-\frac{\gamma_q d}{W}\right)
(f(O)-f(S)).
\]

Multiplying these inequalities and using `1-x<=exp(-x)` gives

\[
f(O)-f(S)
\le
f(O)\exp\left(-\gamma_q\frac{w(S)}{W}\right).
\]

Once `w(S)>=(1-eta)W`, the displayed guarantee follows. If the algorithm stops earlier, every comparator bundle is feasible and has nonpositive gain. The bundle-ratio condition then forces `f(S union O)-f(S)=0`, so the guarantee also holds. QED.

### 4.4 What the guarantee does not say

A transformer can have interaction order greater than two, and `gamma_2` can be zero. The pair solver remains a heuristic in that case. It is also a heuristic when separate hard constraints bind without being reduced to the certified scalar knapsack. Theorem 3 applies to a normalized monotone value such as information; it does not automatically apply to the Dinkelbach transform `a-r t`, which can be nonmonotone and negative. For throughput, the required certificate is the additive inner-oracle gap used in Theorem 5, estimated by exact small cases, a stronger optimizer, or held-out upper bounds.

The code retains exact enumeration. For five captures among 44 candidates, all subsets of size at most five total 1,235,994. This is practical for a cheap analytic oracle, but can be expensive when every objective evaluation runs a learned evaluator or a Monte Carlo posterior.

## 5. Renewal reward and why information can lose throughput

Information can choose the wrong capture set for two independent reasons.

First, log-loss reduction need not improve exact-prefix acceptance by the same amount. A capture can explain distinctions that do not change the drafter's proposed prefix.

Second, an informative capture can cost more critical-path time than its acceptance gain repays.

A minimal numeric example is:

- no capture: `E[A]=1`, `E[T]=1`, so throughput is 1;
- a perfect one-bit capture: `E[A]=8`, `E[T]=100`, so throughput is 0.08.

The perfect capture maximizes information and loses throughput.

The synthetic expensive-layer test is less extreme. Information greedy selects the strongest expensive sensor and obtains 0.935 bits, but only 97.16 synthetic tokens/s. The exact throughput optimum keeps a single cheaper sensor with 0.366 bits and obtains 475.76 synthetic tokens/s. These numbers validate the solver behavior only.

### Theorem 4: Dinkelbach characterization

Let `F` be a finite feasible family, `a(S)>=0`, and `t(S)>=t_min>0`. Define

\[
\Phi(r)=\max_{S\in F}\{a(S)-r t(S)\}.
\]

Then the optimal throughput

\[
r^*=\max_{S\in F}\frac{a(S)}{t(S)}
\]

is the unique root of `Phi(r)=0`. Exact Dinkelbach iterations

\[
S_k\in\arg\max_S\{a(S)-r_k t(S)\},
\qquad
r_{k+1}=\frac{a(S_k)}{t(S_k)}
\]

converge to `r*`. For a finite family they terminate when the maximizing ratio repeats and the transformed residual is zero.

#### Proof

For any `r`,

\[
\Phi(r)=\max_S t(S)\left(\frac{a(S)}{t(S)}-r\right).
\]

It is positive below `r*`, zero at `r*`, and negative above `r*`. The standard Dinkelbach update selects a ratio at least as large as the current one whenever the residual is positive. The finite set of attainable ratios prevents an infinite strictly increasing sequence without reaching the maximum. QED.

The artifact uses exact enumeration as the inner solver on small instances and pair-bundle local search on the 45-layer instances. The context-adaptive implementation uses one common Dinkelbach ratio across contexts; optimizing each context's ratio separately is generally incorrect for the aggregate renewal objective.

### Theorem 5: throughput regret from an approximate inner oracle

Suppose an inner solver at trial ratio `r` returns `S_hat` satisfying

\[
a(S_{\mathrm{hat}})-r t(S_{\mathrm{hat}})
\ge
\Phi(r)-\epsilon.
\]

If `r=a(S_hat)/t(S_hat)`, then

\[
\boxed{r^*-r\le\epsilon/t_{\min}.}
\]

#### Proof

At the returned ratio, the transformed value of `S_hat` is zero, so `Phi(r)<=epsilon`. If `r<r*`, an optimal set `S*` gives

\[
\Phi(r)
\ge t(S^*)(r^*-r)
\ge t_{\min}(r^*-r).
\]

Combine the two inequalities. If `r>=r*`, the claimed upper bound is immediate. QED.

### Corollary 5.1: validated surrogate bound

Assume uniform validation bounds

\[
|\hat a(S)-a(S)|\le\epsilon_a,
\qquad
|\hat t(S)-t(S)|\le\epsilon_t
\]

for every candidate set under consideration. Suppose the inner optimizer is within `epsilon_opt` of the surrogate transformed optimum. At ratio `r`, its true transformed additive error is at most

\[
\epsilon
\le
\epsilon_{\mathrm{opt}}
+2(\epsilon_a+r\epsilon_t).
\]

Therefore

\[
r^*-r
\le
\frac{
\epsilon_{\mathrm{opt}}+2(\epsilon_a+r\epsilon_t)
}{t_{\min}}.
\]

This gives a usable high-probability regret bound when the validation errors are simultaneous confidence bounds. A model of acceptance should be fit directly to accepted-prefix outcomes. Mutual information is not a substitute for `epsilon_a`.

## 6. Causal per-request capture selection

### 6.1 General adaptive sensing formulation

Let `X` be a cheap statistic available before the first capture decision. A partial realization is

\[
\psi=\{(l,z_l):l\text{ already captured}\}.
\]

At each step the controller can stop or select another feasible layer. For a Dinkelbach trial ratio `r`, an exact finite-state Bellman recursion is

\[
V_r(x,\psi,u)
=
\max\left\{
R_r(x,\psi),
\max_{l\in\mathcal A(x,\psi,u)}
\left[-r w_l+E\left[V_r(x,\psi\cup\{(l,Z_l)\},u-w_l)\mid x,\psi\right]\right]
\right\},
\]

where

\[
R_r(x,\psi)=a(x,\psi)-r t_{\mathrm{post}}(x,\psi)
\]

is the transformed value of stopping. With separate byte and latency budgets, `u` is a two-dimensional remaining-budget vector. A finite Bayesian network and integer budgets give an exact memoized dynamic program. Dinkelbach root finding around the Bellman solver yields the optimal adaptive throughput.

Adaptive greedy has a standard guarantee only after the realized utility is proved adaptive monotone and adaptive submodular. Conditional mutual information does not receive that guarantee automatically. Posterior observations can increase another layer's value, as XOR already demonstrates.

### 6.2 Value of a cheap observed regime

For a context `x`, define

\[
g_r(x,S)=a_x(S)-r t_x(S).
\]

A fixed set has transformed value

\[
\Phi_{\mathrm{fix}}(r)
=\max_S E[g_r(X,S)].
\]

A context-adaptive set has value

\[
\Phi_{\mathrm{ad}}(r)
=E\left[\max_S g_r(X,S)\right].
\]

### Theorem 6: adaptivity dominance and strict value

For every `r`,

\[
\Phi_{\mathrm{ad}}(r)\ge\Phi_{\mathrm{fix}}(r).
\]

Therefore the optimal adaptive renewal throughput is at least the optimal fixed-set throughput.

Let `r_fix` be the fixed optimum. Adaptivity has strictly positive throughput value if

\[
E\left[\max_S g_{r_{\mathrm{fix}}}(X,S)\right]>0.
\]

For a finite action family, equality holds when one set is a pointwise maximizer for almost every context at `r_fix`. A common pointwise maximizer is also a simple sufficient condition for no adaptive value.

#### Proof

For every fixed `S`, pointwise maximization gives

\[
\max_T g_r(X,T)\ge g_r(X,S).
\]

Taking expectations and then maximizing the right side proves the first inequality. At `r_fix`, `Phi_fix(r_fix)=0`. If `Phi_ad(r_fix)>0`, the adaptive root lies strictly above `r_fix` because `Phi_ad` is decreasing in `r`. The common-optimizer equality statement follows by equality in the pointwise maximum. QED.

### 6.3 A fixed five-layer set can be arbitrarily worse

Use two equally likely regimes. There are two disjoint five-layer actions:

\[
S_A=\{1,2,3,4,5\},
\qquad
S_B=\{6,7,8,9,10\}.
\]

A cheap statistic reveals the regime. The matched set yields one accepted token in one time unit. A mismatched set yields zero accepted tokens and triggers a verified fallback path taking `M` time units. Any other five-layer set behaves as a mismatch in both regimes.

The best fixed set chooses `S_A` or `S_B`:

\[
\rho_{\mathrm{fix}}
=\frac{1/2}{(1+M)/2}
=\frac{1}{M+1}.
\]

The adaptive policy chooses the matched set:

\[
\rho_{\mathrm{ad}}=1.
\]

The ratio is `M+1`, which is unbounded as `M` grows. Exact verification still protects target correctness. The gap comes from request-dependent acceptance and fallback time.

The checked-in synthetic rows use `M=10,100,1000` and obtain factors 11, 101, and 1001.

### 6.4 An example where adaptivity cannot help

Suppose the conditional law of every capture outcome and every timing component is the same for all values of `X`, or suppose one set `S*` maximizes `g_r(x,S)` for every `x` at the relevant root. Then the adaptive and fixed objectives coincide.

The artifact includes two contexts with a shared best action. Both policies obtain the same throughput and the adaptivity ratio is one.

## 7. Capture-bit lower bounds

Nominal bytes are not the right information count when captures are compressed or duplicated. Let `B` be the total conditional code length in bits. A fixed-width encoder gives

\[
H(Z_S\mid D)\le B.
\]

For a conditionally prefix-free variable-length code, the same inequality holds for expected length. By data processing,

\[
B\ge H(Z_S\mid D)\ge I(Y;Z_S\mid D).
\]

### Theorem 7: conditional Fano bound for block error

Assume `Y|D=d` has finite support of size at most `M` almost surely, and the overall Bayes block-error probability is at most `epsilon`. Put

\[
\bar\epsilon=\min\left\{\epsilon,1-\frac1M\right\}.
\]

Then any capture encoding must satisfy

\[
\boxed{
B
\ge
H(Y\mid D)
-h_2(\bar\epsilon)
-\bar\epsilon\log_2(M-1).
}
\]

For a conditionally uniform target over `M` blocks,

\[
B
\ge
\log_2 M-h_2(\bar\epsilon)-\bar\epsilon\log_2(M-1).
\]

#### Proof

Conditional Fano gives

\[
H(Y\mid Z_S,D)
\le
h_2(\bar\epsilon)+\bar\epsilon\log_2(M-1).
\]

Subtract this from `H(Y|D)` and use

\[
B\ge I(Y;Z_S\mid D).
\]

QED.

A sharper context-dependent version retains the conditional Bayes error `epsilon_D=P(hat Y != Y | D)` inside the expectation:

\[
B\ge H(Y\mid D)
-E\left[h_2(\epsilon_D)+\epsilon_D\log_2(M_D-1)\right].
\]

Replacing `epsilon_D` by one global average error inside the product with `log(M_D-1)` is not valid without an additional uniform conditional-error assumption.

For a uniform 256-way block and error at most 0.05, the artifact evaluates the bound as 7.3139 bits.

### Theorem 8: lower bound from expected exact-prefix acceptance

Let the block length be `K`, let `A` be the accepted exact-prefix length, and suppose

\[
E[A]\ge a.
\]

Define `s_r=P(A>=r)`. Since `s_r` is nonincreasing and

\[
E[A]=\sum_{j=1}^K s_j,
\]

for every `r` with `a>r-1`,

\[
s_r
\ge
\frac{a-r+1}{K-r+1}.
\]

Thus the error probability for predicting the first `r` tokens is at most

\[
\epsilon_r
=\frac{K-a}{K-r+1}.
\]

Let

\[
\bar\epsilon_r=\min\left\{\epsilon_r,1-\frac1{M_r}\right\}.
\]

Applying Fano to the `r`-token prefix gives

\[
\boxed{
B\ge
\max_{r:a>r-1}
\left[
H(Y_{1:r}\mid D)
-h_2(\bar\epsilon_r)
-\bar\epsilon_r\log_2(M_r-1)
\right]_+.
}
\]

Here `M_r` is an almost-sure upper bound on the conditional prefix support size. For conditionally uniform `q`-ary tokens, replace the prefix entropy by `r log_2 q` and `M_r` by `q^r`. A context-dependent refinement again retains the conditional error inside the expectation.

#### Proof of the survival inequality

Because `s_j<=1` before `r` and `s_j<=s_r` after `r`,

\[
a\le E[A]
\le(r-1)+(K-r+1)s_r.
\]

Rearrange and apply Theorem 7 to the prefix. QED.

A general and often sharper statement uses conditional rate-distortion. With distortion

\[
d(Y,\hat Y)=K-A(Y,\hat Y),
\]

any scheme with `E[A]>=a` must satisfy

\[
B\ge R_{Y\mid D}(K-a),
\]

where `R` is the conditional rate-distortion function. For target logits, choose an engine-relevant distortion such as KL, a calibrated top-token loss, or expected accepted-prefix loss. Mean-squared hidden-state distortion has no automatic acceptance interpretation.

## 8. CPU artifact results

Run:

```bash
PYTHONPATH=tools python tools/test_dflash_capture_observability.py -v
PYTHONPATH=tools python tools/evaluate_dflash_capture_observability.py
```

The reference code requires Python 3.10 or newer and NumPy; schema validation in the tests uses `jsonschema`. The test suite has 18 deterministic tests and runs without CUDA or model files.

Selected synthetic results follow.

| Scenario | Objective | Method | Selected layers | Value |
|---|---:|---|---|---:|
| XOR small | information | greedy information | 3, 4 | 0.3319 bits |
| XOR small | information | pair bundle | 1, 2 | 1.0000 bits |
| XOR small | information | true optimum | 1, 2 | 1.0000 bits |
| duplicate small | information | pair bundle | 1, 3 | 0.5161 bits |
| expensive small | throughput | greedy information | 1, 2 | 97.16 synthetic tok/s |
| expensive small | throughput | Dinkelbach pair | 2 | 475.76 synthetic tok/s |
| expensive small | throughput | true optimum | 2 | 475.76 synthetic tok/s |
| restricted 45-layer candidate set | throughput | current 5 | 5, 14, 24, 33, 42 | 62.23 synthetic tok/s |
| restricted 45-layer candidate set | throughput | true optimum | 8, 10, 14, 33, 38 | 88.85 synthetic tok/s |

The exact theorem checks report:

- the conditionally independent network has no negative submodularity gap within numerical tolerance;
- XOR has minimum diminishing-returns gap `-1` bit;
- XOR has `gamma_1=0` and `gamma_2=1`;
- the deterministic duplicate pair has the same information as either copy alone.
- in the correlated 45-layer near-duplicate construction, adding layer 24 after layer 23 contributes only about 0.0085 bit.

The restricted eight-candidate case contains all five current locations and admits exact enumeration of every subset up to size five. It exists only to verify that the current-set baseline and the true-optimum code path are compared on the same layer-indexed problem.

The full 45-layer tables compare the current set against the requested baselines and the proposed solver. A subset of the synthetic output is:

| Scenario | Method | Selected layers | Information | Throughput |
|---|---|---|---:|---:|
| near duplicate | current 5 | 5, 14, 24, 33, 42 | 1.8529 bits | 91.69 synthetic tok/s |
| near duplicate | greedy information | 19, 21, 23, 29, 30 | 2.9185 bits | 133.51 synthetic tok/s |
| near duplicate | pair bundle | 20, 22, 23, 30, 31 | 2.9606 bits | 134.94 synthetic tok/s |
| change point | current 5 | 5, 14, 24, 33, 42 | 1.2010 bits | 62.23 synthetic tok/s |
| change point | fixed Dinkelbach pair | 8, 10, 13, 34, 38 | 1.8874 bits | 93.10 synthetic tok/s |
| expensive late | greedy information | 37, 38, 39, 41, 42 | 3.0240 bits | 83.24 synthetic tok/s |
| expensive late | cost-aware greedy | 37, 38, 39, 40, 43 | 2.2545 bits | 100.13 synthetic tok/s |
| expensive late | Dinkelbach pair | 37, 38, 39, 40, 44 | 2.2589 bits | 100.28 synthetic tok/s |

In the expensive-late construction, layer 42 is the strongest singleton and has deliberately excessive latency. Information greedy selects it. Throughput optimization rejects it.

In the change-point construction, the cheap-context policy uses one early-depth set and one late-depth set. Its synthetic throughput is 119.31 tokens/s versus 93.10 for the best fixed pair-Dinkelbach result, a factor of 1.2815.

None of these rows estimates production performance.

## 9. Minimum trace needed from real requests

The JSON schema records one speculative round. The statistically essential fields are below.

### 9.1 Causal context

Record the exact cheap features available before selection, their schema version, and their availability timestamp. Features computed after a capture cannot be treated as pre-action context.

Separately record a versioned representation of `D`, the prompt-derived and drafter causal state used by the frozen evaluator. This may be a privacy-safe feature vector or a stable content-addressed reference. Cheap policy context `X` and evaluator conditioning state `D` need not be the same. Without `D`, conditional information and conditional Bayes risk are not identifiable as stated.

### 9.2 Exploration action and propensity

Record:

- the candidate layer list;
- the ordered selected captures;
- each conditional selection probability;
- the probability of the complete ordered action;
- policy version, exploration cohort, and random seed.

Without positive joint probability for a pair or set, its interaction value is not identifiable from inverse-propensity weighting. Independent marginal inclusion probabilities are insufficient when the policy selects without replacement or stops adaptively.

### 9.3 Encoded observations

For every selected layer, record the exact encoded `Z_l` consumed by the frozen evaluator or a stable content-addressed reference. Also record encoder identity, rank, distortion setting, encoded bits, logical bytes, and transferred bytes.

Raw target hidden states are not required. Omitting `Z_l` prevents direct estimation of conditional information and Bayes risk.

### 9.4 Engine labels

Record the draft block, exact target block, verification-prefix length, accepted draft tokens, committed tokens, and fallback use. A frozen target-logit summary may be added if logits are the chosen `Y`.

### 9.5 Critical-path timing

Record round boundaries, drafter boundaries, verification boundaries, fallback time, and capture critical-path time. Per-capture ready and transfer timestamps are needed because overlapping traffic must not be summed as if it were serial.

Hardware queue, bandwidth, and clock strata should be retained when they materially affect latency.

### 9.6 Estimation protocol

Use randomized on-policy exploration with support over the candidate sets being compared. Fit on one partition and evaluate accepted-prefix reward plus round time on held-out requests. Report paired or doubly robust estimates with simultaneous confidence bounds. Stratify by cheap context and prompt/workload class before pooling.

## 10. Kill criterion and engine decision rule

Let the current policy be `S_0` with baseline throughput

\[
r_0=\frac{E[A_0]}{E[T_0]}.
\]

For a candidate policy `pi`, define its incremental Dinkelbach net reward at the baseline rate:

\[
G(\pi)
=E[A_\pi-A_0]
-r_0 E[T_\pi-T_0].
\]

The candidate improves throughput if and only if `G(pi)>0`.

### Primary kill criterion

After multiplicity correction over the searched policy family, stop the capture-placement project if

\[
\boxed{
\max_{\pi\in\Pi}
\operatorname{UCB}_{1-\alpha}G(\pi)
\le0.
}
\]

This criterion charges all measured effects, including capture traffic, drafter compute, verification changes, and fallback changes.

### Hardware lower-bound kill criterion

If only an upper bound on acceptance gain and a lower bound on unavoidable added time are available, include D2D/H2D traffic, synchronization, encoder work, and any extra drafter compute in that lower bound, then kill when

\[
\max_{\pi\in\Pi}
\operatorname{UCB}\Delta A(\pi)
\le
\operatorname{LCB}(r_0)
\operatorname{LCB}\Delta T_{\mathrm{unavoidable}}(\pi).
\]

The helper `kill_criterion_net_upper_bound` computes the right net quantity. The checked-in negative example is only a unit test.

### Evidence that would decide the current five locations

A statement that another policy is better requires all of the following:

1. on-policy randomized support for the current set and candidate policies;
2. held-out estimates of accepted-prefix reward and full round time;
3. a positive simultaneous lower confidence bound for incremental net reward;
4. unchanged exact verification and committed-token parity;
5. stability across the relevant prompt and hardware strata.

If a candidate passes these gates, it is evidence against the current placement for the measured workload. If no candidate passes, the result supports retaining 5/14/24/33/42 within the resolution of the experiment. Neither outcome follows from the synthetic tables.

## 11. References

- Andreas Krause and Carlos Guestrin. *Near-Optimal Nonmyopic Value of Information in Graphical Models*. UAI, 2005.
- Abhimanyu Das and David Kempe. *Submodular Meets Spectral: Greedy Algorithms for Subset Selection, Sparse Approximation and Dictionary Selection*. ICML, 2011.
- Daniel Golovin and Andreas Krause. *Adaptive Submodularity: Theory and Applications in Active Learning and Stochastic Optimization*. Journal of Artificial Intelligence Research, 2011.
- Werner Dinkelbach. *On Nonlinear Fractional Programming*. Management Science, 1967.
- Robert M. Fano. *Transmission of Information: A Statistical Theory of Communications*. MIT Press, 1961.
- Claude E. Shannon. *Coding Theorems for a Discrete Source with a Fidelity Criterion*. IRE National Convention Record, 1959.
