# Finite-horizon shadowing and reset control for quantized KDA state

## Status

This note answers `tasks/inference-math-frontier-20260831/02-kda-recurrent-shadowing.md` at reference commit `0740c63`.

Reproducible CPU material:

- `tools/kda_shadowing.py`: recurrence algebra, deterministic simulator, all bounds, FP8 block quantization, symbolic checks, and reset solvers.
- `tools/test_kda_shadowing.py`: hardware-free unit tests.
- `scratch/kda-shadowing/trace.csv`: token-by-token observed error and every implemented bound.
- `scratch/kda-shadowing/summary.json`: scenario summaries, symbolic results, false-confidence examples, the FP8 scale discontinuity, and reset output.
- `scratch/kda-shadowing/reset-demo.json`: exact finite-horizon reset instance.
- `scratch/kda-shadowing/fp8-scale-demo.json`: discontinuous max-scale example.
- `scratch/kda-shadowing/fp8-rank-demo.json`: rank-one input whose E4M3 image has rank two.

Engine-specific derivations are grounded in `src/glm53_ops.cu::kda_decode_kernel`, `src/glm53_generate.cu::kda_gate_kernel`, the recurrent buffer allocation near `kda_states_`/`conv_history_`, and `Runner::rollback_kda`.

The central engine result is favorable for the current FP32 KDA recurrence and unfavorable for blind state compression:

1. Under arbitrary operator-norm gate bounds, the usual product bound is minimax sharp. There is no universal improvement without more structure.
2. The implemented GLM KDA map has more structure. Per head it is a diagonal decay followed by a normalized rank-one correction. Its homogeneous state-error map is nonexpansive in Frobenius norm and satisfies an exact dissipation identity. Switching cannot create the nonnormal transient amplification possible in the abstract model as long as key normalization, `beta in [0,1]`, and `decay <= 1` remain true.
3. A sound online certificate needs two radii per head, a bound on the exact state norm and a bound on state error. It can be updated from vector norms and quantizer error envelopes without comparing against an exact state every token.
4. State FP8 is acceptable only when the state quantizer accumulates its residual norm, or a comparably sharp bound, in the same pass that performs quantization. A separate FP32 state scan or a worst-cell E4M3 bound that consumes the loss budget in one step defeats the optimization.
5. With a hard numerical loss budget, fixed reset cost, and a genuine exact anchor available at each legal reset point, the causal policy that resets immediately before the first predicted violation is pathwise optimal among policies that keep using the approximate path. No future gate knowledge is needed.
6. The current one-block DFlash snapshot is an exact anchor only for the existing exact target pass. A block-local approximation may use it only if the accepted prefix is reconstructed from causally exact inputs; once approximation changes later hidden features, `rollback_kda` with the approximate pass's archived inputs is not an exact reconstruction. Persistent state compression is therefore a no-go in the current engine unless a separate exact-anchor and causal reconstruction mechanism is added.

No semantic-loss threshold can be chosen without either private weights or measured downstream sensitivity. The theory supplies a certified state/output error radius. The final conversion from that radius to a logit, route, or task-quality budget must be calibrated in-engine and backed by the existing exact retry guard.

## 1. Model and norms

Throughout, "exact" means the designated reference engine path used for verification and replay. It need not mean an infinite-precision realization of the entire neural network. Any baseline FP32 roundoff can be included in `S_t`; the analysis certifies the additional deviation of the approximate path from that reference.

For one head, write

\[
S_t=A_tS_{t-1}B_t+R_t,
\qquad
R_t=\beta_tu_tv_t^T.
\]

The approximate path computes

\[
\widehat X_t=\widehat A_t\widehat S_{t-1}\widehat B_t+
\widehat R_t,
\qquad
\widehat R_t=\widehat\beta_t\widehat u_t\widehat v_t^T,
\]

then stores

\[
\widehat S_t=Q_t(\widehat X_t).
\]

Define

\[
\Delta_t=\widehat S_t-S_t,
\qquad
q_t=Q_t(\widehat X_t)-\widehat X_t.
\]

State matrices use the Frobenius norm. Left and right gates use the spectral norm. This pairing is natural because

\[
\|AXB\|_F\le \|A\|_2\|B\|_2\|X\|_F.
\]

No regularity of `Q_t` is assumed unless stated. It may depend on a block maximum and may be discontinuous.

The general results allow arbitrary square gates. The KDA specialization later assumes:

- `beta` lies in `[0,1]` on both paths;
- normalized keys satisfy `||k||_2 <= 1`;
- diagonal decays lie in `[0,1]`;
- input and quantization perturbations have deterministic norm envelopes known before an approximate state is committed.

Those assumptions match the intended kernel contract. If a finite-precision implementation violates one of them, the structural KDA theorem must not be used for that step.

## 2. Exact error recurrences

Let

\[
E_{A,t}=\widehat A_t-A_t,
\qquad
E_{B,t}=\widehat B_t-B_t.
\]

There are two useful exact decompositions. They differ only in which propagator is assigned to the old state error.

### 2.1 Exact-gate propagator

A direct telescoping identity gives

\[
\boxed{
\begin{aligned}
\Delta_t={}&A_t\Delta_{t-1}B_t\\
&+E_{A,t}\widehat S_{t-1}\widehat B_t
  +A_t\widehat S_{t-1}E_{B,t}\\
&+(\widehat R_t-R_t)+q_t.
\end{aligned}}
\tag{2.1}
\]

The terms are, in order:

- carried state error;
- coefficient error;
- low-rank update error;
- finite-precision error.

An equivalent coefficient telescoping is

\[
E_{A,t}\widehat S_{t-1}B_t+
\widehat A_t\widehat S_{t-1}E_{B,t}.
\]

Therefore a sharp triangle bound available from a measured or fused `||hat S||_F` is

\[
\|C_t\|_F\le \|\widehat S_{t-1}\|_F
\min\left\{
\epsilon_{A,t}\widehat b_t+a_t\epsilon_{B,t},
\epsilon_{A,t}b_t+\widehat a_t\epsilon_{B,t}
\right\},
\tag{2.2}
\]

where

\[
a_t=\|A_t\|_2,
\quad b_t=\|B_t\|_2,
\quad \widehat a_t=\|\widehat A_t\|_2,
\quad \widehat b_t=\|\widehat B_t\|_2,
\]

and `epsilon_A`, `epsilon_B` bound the coefficient perturbations.

### 2.2 Approximate-gate propagator

A second exact identity is

\[
\boxed{
\begin{aligned}
\Delta_t={}&\widehat A_t\Delta_{t-1}\widehat B_t\\
&+E_{A,t}S_{t-1}B_t
  +A_tS_{t-1}E_{B,t}
  +E_{A,t}S_{t-1}E_{B,t}\\
&+(\widehat R_t-R_t)+q_t.
\end{aligned}}
\tag{2.3}
\]

This form avoids scanning `hat S` if a scalar majorant `M_{t-1} >= ||S_{t-1}||_F` is tracked. Its coefficient term obeys

\[
\|\widetilde C_t\|_F
\le
(\epsilon_{A,t}b_t+a_t\epsilon_{B,t}+
 \epsilon_{A,t}\epsilon_{B,t})M_{t-1}.
\tag{2.4}
\]

### 2.3 Exact low-rank update split

With

\[
\delta\beta_t=\widehat\beta_t-\beta_t,
\quad
\delta u_t=\widehat u_t-u_t,
\quad
\delta v_t=\widehat v_t-v_t,
\]

one exact telescoping is

\[
\boxed{
\widehat R_t-R_t=
\delta\beta_t u_tv_t^T+
\widehat\beta_t\delta u_t v_t^T+
\widehat\beta_t\widehat u_t\delta v_t^T.
}
\tag{2.5}
\]

Hence

\[
\begin{aligned}
\|\widehat R_t-R_t\|_F\le{}&
|\delta\beta_t|\|u_t\|_2\|v_t\|_2\\
&+|\widehat\beta_t|\|\delta u_t\|_2\|v_t\|_2\\
&+|\widehat\beta_t|\|\widehat u_t\|_2\|\delta v_t\|_2.
\end{aligned}
\tag{2.6}
\]

When `beta=0`, the factors `u,v` are not identifiable. A canonical zero factor or the directly computed matrix norm `||hat R-R||_F` is preferable to a factorwise bound that assigns a large norm to an inactive exact update.

### 2.4 Rounding and state quantization

The rounding term is defined after all perturbed coefficients have been applied:

\[
q_t=Q_t(\widehat X_t)-\widehat X_t.
\]

This definition is important. It remains exact even when the quantizer scale jumps as a function of the current block maximum. No derivative or Lipschitz constant of `Q_t` is needed for the forward recurrence.

## 3. Sharp finite-horizon theory under operator-norm bounds

Let the local defect in (2.1) be

\[
d_t=C_t+(\widehat R_t-R_t)+q_t.
\]

### Theorem 1: variation of constants

For `1 <= j <= t`, define

\[
L_{t:j}=A_tA_{t-1}\cdots A_j,
\qquad
R_{j:t}=B_jB_{j+1}\cdots B_t,
\]

with an empty product equal to the identity. Then

\[
\boxed{
\Delta_t=
L_{t:1}\Delta_0R_{1:t}
+
\sum_{j=1}^t L_{t:j+1}d_jR_{j+1:t}.
}
\tag{3.1}
\]

Consequently,

\[
\boxed{
\begin{aligned}
\|\Delta_t\|_F\le{}&
\|L_{t:1}\|_2\|R_{1:t}\|_2\|\Delta_0\|_F\\
&+\sum_{j=1}^t
\|L_{t:j+1}\|_2\|R_{j+1:t}\|_2\|d_j\|_F.
\end{aligned}}
\tag{3.2}
\]

#### Proof

Substitution proves (3.1) by induction. Applying `||AXB||_F <= ||A||_2 ||B||_2 ||X||_F` to every term proves (3.2). QED.

Equation (3.2) is the best inexpensive analytic bound when actual transition products are retained. It captures cancellation and changing singular directions inside each product. Computing every suffix norm is not a production recommendation for `d=128`; it is a useful oracle in the CPU artifact.

### Corollary 1: only per-step norm bounds

Suppose only

\[
\|A_t\|_2\le \alpha_t,
\qquad
\|B_t\|_2\le \gamma_t,
\qquad
\|d_t\|_F\le \eta_t
\]

are known. Define `g_t=alpha_t gamma_t`. Then

\[
\boxed{
\|\Delta_t\|_F\le
\left(\prod_{k=1}^tg_k\right)\|\Delta_0\|_F+
\sum_{j=1}^t
\left(\prod_{k=j+1}^tg_k\right)\eta_j.
}
\tag{3.3}
\]

Equivalently, the online scalar recurrence

\[
r_t=g_tr_{t-1}+\eta_t,
\qquad r_0\ge\|\Delta_0\|_F
\tag{3.4}
\]

is safe.

### Theorem 2: minimax sharpness of the product bound

No smaller universal bound can be proved from only the nonnegative scalar data `alpha_t`, `gamma_t`, and `eta_t`.

#### Construction

Take `d=1`, choose

\[
A_t=\alpha_t,
\qquad B_t=\gamma_t,
\qquad d_t=\eta_t,
\qquad \Delta_0=r_0,
\]

with every scalar nonnegative. The recurrence is exactly

\[
\Delta_t=g_t\Delta_{t-1}+\eta_t,
\]

so equality holds in (3.3) at every time.

The same construction works for any `d >= 2` with all states and defects proportional to `e_1e_1^T`, and with gates scalar multiples of the identity. Every defect can be rank one. Therefore rank-one update structure alone does not improve the norm-only minimax result. A useful improvement requires a restriction that prevents repeated alignment, such as diagonal shared directions or the actual KDA projection-decay form.

### 3.1 Spectral radius is not a substitute for an induced norm

Let

\[
A_0=\begin{pmatrix}r&K\\0&r\end{pmatrix},
\qquad
A_1=\begin{pmatrix}r&0\\K&r\end{pmatrix},
\qquad 0<r<1.
\]

Each matrix has spectral radius `r`. Their two-step product is

\[
A_1A_0=
\begin{pmatrix}
r^2&rK\\
rK&K^2+r^2
\end{pmatrix}.
\]

For sufficiently large `K`, its largest eigenvalue exceeds one. Repeating the pair causes exponential growth. The deterministic artifact uses `r=1/2`, `K=2`; every individual spectral radius is `0.5`, while a `1e-6` initial error is amplified by more than `1700x` in ten steps.

This example applies to the broad abstract recurrence and to implementation failures that break KDA's normalization or decay constraints. It does not contradict the KDA Lyapunov identity below.

## 4. Structural improvements

### 4.1 Diagonal gates and shared directions

Suppose

\[
A_t=\operatorname{diag}(a_{t,1},\ldots,a_{t,d}),
\qquad
B_t=\operatorname{diag}(b_{t,1},\ldots,b_{t,d}).
\]

Then every state cell evolves independently:

\[
\Delta_{t,ij}=a_{t,i}b_{t,j}\Delta_{t-1,ij}+d_{t,ij}.
\]

Define the entrywise majorant

\[
\boxed{
r_{t,ij}=|a_{t,i}b_{t,j}|r_{t-1,ij}+|d_{t,ij}|,
\qquad
r_{0,ij}\ge|\Delta_{0,ij}|.
}
\tag{4.1}
\]

Induction gives

\[
|\Delta_{t,ij}|\le r_{t,ij},
\qquad
\|\Delta_t\|_F\le\|r_t\|_F.
\tag{4.2}
\]

This bound is cellwise sharp. It also applies after fixed orthogonal changes of basis when the left gates are simultaneously orthogonally diagonalizable and the right gates are simultaneously orthogonally diagonalizable. Merely sharing singular values, or sharing unrelated left and right singular vectors, is insufficient because successive products can rotate between the two bases.

The distinction from (3.3) can be large. The artifact alternates diagonal factors `diag(1.2,0.5)` and `diag(0.5,1.2)`. The per-step norm bound grows as if `1.2` acted on the same coordinate forever. Each coordinate actually receives a two-step factor `0.6`. At step 24, the scalar bound is about `1.12e-1`, while the exact diagonal majorant and observed error are about `3.08e-6`.

If every coordinate factor is at most one, (4.1) becomes a weighted path-length bound rather than a worst-case exponential.

### 4.2 The implemented GLM KDA recurrence

The CUDA kernel normalizes `k`, computes a diagonal decay `D=diag(exp(g))`, forms

\[
m_t=k_t^TD_tS_{t-1},
\]

and updates

\[
S_t=D_tS_{t-1}+\beta_tk_t(v_t^T-m_t).
\]

Therefore

\[
\boxed{
S_t=(I-\beta_tk_tk_t^T)D_tS_{t-1}+
\beta_tk_tv_t^T.
}
\tag{4.3}
\]

For this specialization,

\[
A_t=P_tD_t,
\qquad
P_t=I-\beta_tk_tk_t^T,
\qquad
B_t=I.
\]

The gate kernel maps the learned log-decay to a negative value and then exponentiates it in the KDA kernel. Under the intended contract, every diagonal decay lies in `(0,1]`. The key norm is at most one because the implementation uses reciprocal-square-root normalization with a positive epsilon.

### Theorem 3: exact KDA Lyapunov identity

Assume `0 <= beta <= 1`, `||k||_2 <= 1`, and `0 <= D_ii <= 1`. For every matrix `X`, let

\[
c=2\beta-\beta^2\|k\|_2^2\ge0.
\]

Then

\[
\boxed{
\begin{aligned}
\|(I-\beta kk^T)DX\|_F^2
={}&\|X\|_F^2\\
&-\sum_i(1-D_{ii}^2)\|X_{i,:}\|_2^2\\
&-c\|k^TDX\|_2^2.
\end{aligned}}
\tag{4.4}
\]

In particular,

\[
\|(I-\beta kk^T)DX\|_F\le\|X\|_F,
\tag{4.5}
\]

and

\[
\|A_t\|_2\le\|D_t\|_2=\max_iD_{t,ii}\le1.
\tag{4.6}
\]

#### Proof

Let `P=I-beta kk^T`. Since

\[
P^2=I-(2\beta-\beta^2\|k\|_2^2)kk^T,
\]

we have

\[
\|PDX\|_F^2=
\operatorname{tr}(X^TD P^2DX)
=\|DX\|_F^2-c\|k^TDX\|_2^2.
\]

Expanding `||DX||_F^2` around `||X||_F^2` gives (4.4). QED.

This identity rules out switching-induced Frobenius amplification for homogeneous KDA state error. The projection term may leave a large subspace unchanged at one token, but it never increases energy. Key diversity and diagonal decay dissipate additional directions over time.

### Exact driven-state energy identity

The delta rule also controls the norm of the exact state itself more sharply than the generic triangle recurrence. Let

\[
Y=D S,
\qquad
m=k^T Y,
\qquad
z=v-m.
\]

The full update is `S^+=Y+beta k z^T`, and direct expansion gives

\[
\boxed{
\begin{aligned}
\|S^+\|_F^2={}&\|Y\|_F^2
-\beta\|m\|_2^2+\beta\|v\|_2^2\\
&-\beta(1-\beta\|k\|_2^2)\|z\|_2^2.
\end{aligned}}
\tag{4.7}
\]

Indeed,

\[
\|Y+\beta kz^T\|_F^2
=\|Y\|_F^2+2\beta\langle m,z\rangle
 +\beta^2\|k\|_2^2\|z\|_2^2,
\]

and substituting `z=v-m` yields (4.7). The last term in (4.7) is nonpositive under the KDA contract. Therefore a state-norm majorant can use

\[
\boxed{
M_t^2=\rho_t^2M_{t-1}^2+\overline\beta_tV_t^2,
\qquad
M_t\ge\|S_t\|_F,
}
\tag{4.8}
\]

where `rho_t` upper-bounds the exact maximum decay, `overline beta_t` upper-bounds `beta_t`, and `V_t` upper-bounds `||v_t||_2`. This energy update is often much smaller than `rho M + beta V`. It uses only scalar/vector reductions already adjacent to the KDA kernel.

### Corollary 2: KDA path-length and input-to-state bounds

Let all coefficient, update, and quantization errors be collected into `d_t`, and define

\[
\rho_t=\max_iD_{t,ii}\le1,
\qquad
\|d_t\|_F\le\eta_t.
\]

Then

\[
\boxed{
r_t=\rho_tr_{t-1}+\eta_t}
\tag{4.9}
\]

is safe. Define accumulated log contraction

\[
\Gamma_t=\sum_{k=1}^t-\log\rho_k.
\]

The explicit bound is

\[
\boxed{
r_t\le e^{-\Gamma_t}r_0+
\sum_{j=1}^t e^{-(\Gamma_t-\Gamma_j)}\eta_j.}
\tag{4.10}
\]

If `rho_t <= rho < 1` and `eta_t <= eta`, then

\[
r_t\le \rho^tr_0+\eta\frac{1-\rho^t}{1-\rho}.
\tag{4.11}
\]

If no strict decay can be certified, `rho_t=1` still yields the nonexponential path-length bound

\[
r_t\le r_0+\sum_{j=1}^t\eta_j.
\tag{4.12}
\]

For the homogeneous recurrence, summing (4.4) gives an energy budget:

\[
\sum_t
\left[
\sum_i(1-D_{t,ii}^2)\|\Delta_{t-1,i,:}\|_2^2+
 c_t\|k_t^TD_t\Delta_{t-1}\|_2^2
\right]
\le\|\Delta_0\|_F^2.
\tag{4.13}
\]

### 4.3 A tighter block oracle and why it is not the default engine certificate

Because `A_t` is diagonal plus rank one, a product matrix `P <- A_t P` can be updated in `O(d^2)` rather than a generic `O(d^3)` multiply. A safe product norm follows from

\[
\|P\|_2\le\sqrt{\|P\|_1\|P\|_\infty}.
\]

For a DFlash block of at most eight rows, this can produce a sharper suffix-product bound than the scalar recurrence (4.9), especially when keys cover different directions. It costs an additional `d x d` matrix per head and another pass worth of arithmetic over it. That is suitable for a CPU oracle or a sampled diagnostic kernel. It should not be the production requirement for state compression. If the scalar KDA certificate is vacuous and only this extra matrix propagation makes compression appear safe, the optimization has failed its cost test unless a benchmark shows the work is hidden inside an existing state pass.

## 5. Online certificate without exact-state comparison

### 5.1 Generic two-scalar recurrence

Track, for every head,

\[
M_t\ge\|S_t\|_F,
\qquad
R_t\ge\|\Delta_t\|_F.
\]

Suppose the engine has certified bounds

\[
\|A_t\|_2\le a_t,
\quad
\|B_t\|_2\le b_t,
\quad
\|\widehat A_t\|_2\le\widehat a_t,
\quad
\|\widehat B_t\|_2\le\widehat b_t,
\]

as well as coefficient perturbation bounds `epsilon_A`, `epsilon_B`. Let

\[
U_t\ge\|u_t\|_2,
\qquad
V_t\ge\|v_t\|_2,
\qquad
E_{R,t}\ge\|\widehat R_t-R_t\|_F,
\qquad
E_{Q,t}\ge\|q_t\|_F.
\]

Update

\[
\boxed{
M_t=a_tb_tM_{t-1}+|\beta_t|U_tV_t,
}
\tag{5.1}
\]

and

\[
\boxed{
\begin{aligned}
R_t={}&\widehat a_t\widehat b_tR_{t-1}\\
&+(\epsilon_{A,t}b_t+a_t\epsilon_{B,t}
 +\epsilon_{A,t}\epsilon_{B,t})M_{t-1}\\
&+E_{R,t}+E_{Q,t}.
\end{aligned}}
\tag{5.2}
\]

### Theorem 4: certificate safety

If the initial inequalities hold and every scalar envelope in (5.1)-(5.2) is valid, then they hold for all later tokens.

#### Proof

Equation (5.1) follows from the exact recurrence and submultiplicativity. Equation (5.2) follows from the exact approximate-propagator decomposition (2.3), bound (2.4), and the triangle inequality. Induction completes the proof. QED.

This is `O(1)` certificate state per head. The required input reductions are `O(d)` for KDA vectors. No exact recurrent state is needed after an exact reset.

A tighter variant uses (2.1) and a fused measurement of `||hat S||_F`. That measurement is `O(d^2)`, but the KDA kernel already touches all `d^2` cells. It is acceptable only when the squares and reduction are fused into that pass, or into the state quantizer pass.

### 5.2 KDA-specific coefficient envelope

Let

\[
A=PD,
\quad \widehat A=\widehat P\widehat D,
\quad P=I-\beta kk^T.
\]

Assuming both projection factors are nonexpansive,

\[
\widehat A-A=
\widehat P(\widehat D-D)+
(\widehat P-P)D.
\]

Set

\[
\epsilon_D=\max_i|\widehat D_{ii}-D_{ii}|,
\qquad
\epsilon_k=\|\widehat k-k\|_2.
\]

Then

\[
\boxed{
\begin{aligned}
\|\widehat A-A\|_2\le{}&\epsilon_D+
\|D\|_2\Bigl(
|\widehat\beta-\beta|\|k\|_2^2\\
&+|\widehat\beta|\epsilon_k\|k\|_2
+|\widehat\beta|\|\widehat k\|_2\epsilon_k
\Bigr).
\end{aligned}}
\tag{5.3}
\]

For log-decay perturbation `|hat g_i-g_i| <= epsilon_g`, a safe exact decay upper endpoint is

\[
D_{ii}\le
\exp(\min\{0,\widehat g_i+\epsilon_g\}).
\tag{5.4}
\]

The same interval endpoints give `epsilon_D` without linearizing `exp`.

The key perturbation should be bounded after normalization, not guessed from an FP8 code distance. For the kernel map

\[
n_\varepsilon(x)=\frac{x}{\sqrt{\|x\|_2^2+\varepsilon}},
\]

suppose the raw-vector error is at most `delta_x` and `||hat x||_2 > delta_x`. The Jacobian norm on the line segment is at most

\[
\left((\|\widehat x\|_2-\delta_x)^2+\varepsilon\right)^{-1/2},
\]

so a causal bound is

\[
\epsilon_k\le
\frac{\delta_x}{\sqrt{(\|\widehat x\|_2-\delta_x)^2+\varepsilon}}.
\tag{5.5a}
\]

If the lower norm margin is lost, the global fallback is `delta_x/sqrt(epsilon)`, which will usually be too loose and should trigger exact replay rather than silently certify the row.

For the KDA update `R=beta k v^T`, equation (2.6) supplies `E_R`. The preferred exact-state majorant is the driven energy update

\[
M_t\le
\sqrt{\rho_t^2M_{t-1}^2+\overline\beta_tV_t^2}.
\tag{5.5}
\]

The looser linear update `rho M + beta V` remains valid for implementations that do not preserve the coupled delta-rule form.

### 5.3 Future envelope

At time `t`, suppose safe future gain and local-error envelopes are available through the next allowed reset boundary:

\[
\overline g_{t+1:t+h},
\qquad
\overline\eta_{t+1:t+h}.
\]

Then the future radius is bounded by recursively applying

\[
R\leftarrow\overline g_sR+\overline\eta_s.
\tag{5.6}
\]

The bound is safe even when the horizon is revealed causally. Before committing token `s`, the engine inserts the newly revealed coefficient envelopes and checks the resulting radius. For a DFlash candidate block, the same recurrence can be evaluated row by row once the approximate coefficients for those rows exist.

Using only global caps gives a valid but possibly loose forecast. In KDA, the structural cap is never larger than one, so the fallback forecast grows at most by the accumulated local-error budget.

### 5.4 Connecting state error to an engine loss budget

To avoid confusing the readout query with the quantization residual `q_t`, denote the query vector by `chi_t`. The KDA readout has the form

\[
y_t=\chi_t^TS_t.
\]

Therefore

\[
\|\widehat y_t-y_t\|_2
\le\|\chi_t\|_2R_t
\tag{5.7}
\]

when the readout vector is shared. If the query is also perturbed, the exact decomposition

\[
\widehat\chi_t^T\widehat S_t-\chi_t^TS_t
=\widehat\chi_t^T\Delta_t+(\widehat\chi_t-\chi_t)^TS_t
\]

gives

\[
\|\widehat y_t-y_t\|_2
\le\|\widehat\chi_t\|_2R_t+\epsilon_{\chi,t}M_t.
\tag{5.7a}
\]

The current kernel normalizes the query to approximately `1/sqrt(d)`, which attenuates the direct state-error term.

This does not certify final logits by itself. Downstream layers can amplify the readout and discrete expert routing can change. A semantic or logit loss budget needs a monotone certified map such as

\[
\mathcal L_t\le
\sum_{\ell,h}w_{\ell,h,t}R_{\ell,h,t},
\tag{5.8}
\]

where the weights come from measured local sensitivity or a conservative downstream norm product. If no useful weights are available, use the numerical certificate only as a prefilter and retain exact target-logit or route guards as the final authority.

### 5.5 False-confidence examples

#### Individual spectral radius

The alternating shear example has spectral radius `0.5` at every step and still creates a large transient. A per-step eigenvalue check is unsound for general structured gates.

#### Current output or a small sketch

Let

\[
\Delta=e_2e_2^T.
\]

The current query `q=e_1` observes zero error, while the next query `q=e_2` observes norm one. A query projection, checksum, or fixed low-dimensional random sketch can miss an error direction that later becomes visible.

#### Mean decay

For

\[
D=\operatorname{diag}(1,0.01,0.01,0.01),
\]

mean decay is `0.2575`, but error in the first row is not contracted. The maximum certified decay or a direction-aware bound is required.

## 6. Optimal finite-horizon reset and replay

Consider the scalar certificate

\[
r_t=f_t(r_{t-1})=g_tr_{t-1}+\eta_t,
\qquad g_t,\eta_t\ge0.
\tag{6.1}
\]

A reset at a legal boundary restores an exact state and sets `r=0`. A token is safe when `r_t <= B_t`; a monotone multi-head loss function can replace this scalar condition.

### 6.1 Causal lazy reset

At each token:

1. compute `z=f_t(r)`;
2. if `z <= B_t`, commit the approximate step and set `r=z`;
3. otherwise reset immediately before the token, then test `f_t(0)`;
4. if `f_t(0)>B_t`, that token cannot be approximated under the certificate and must run exact.

### Theorem 5: pathwise optimality

Assume resets are legal before every token, reset cost is fixed, and every token is otherwise processed on the approximate path. The lazy policy minimizes the number of resets on every realized prefix. Therefore it maximizes

\[
N C_s-N_r C_r
\tag{6.2}
\]

for every causally revealed gate and error sequence.

#### Proof

Suppose the current exact anchor is at boundary `i`, and lazy first predicts a violation at token `j`. Any feasible schedule with no reset in `(i,j]` has the same radius recursion and also violates at `j`. It must reset no later than `j`. Resetting exactly at `j` leaves radius `f_j(0)`, no larger than the radius produced by a reset at an earlier boundary because every `f_t` is monotone. Thus moving the first competing reset to `j` cannot hurt later feasibility. Apply the argument inductively after `j`. QED.

The result is online and exact. It does not need a horizon forecast.

### 6.2 Exact finite-horizon dynamic program

For verification and for restricted reset boundaries, precompute whether a segment `[i,j)` is feasible when started from radius zero. Let `V(j)` be maximal net saving through token `j`, ending at an approximate segment boundary. Then

\[
\boxed{
V(j)=
\max_{0\le i<j:\ [i,j)\ \mathrm{feasible}}
\left[
V(i)+(j-i)C_s-C_r\mathbf1\{i>0\}
\right].
}
\tag{6.3}
\]

If the final state must be exact, subtract one final `C_r`. The complexity is `O(N^2)` after segment simulation. The artifact also enumerates every reset subset for `N <= 20`; dynamic programming, exhaustive search, and the causal lazy reset agree on the minimum reset count across deterministic randomized tests.

### 6.3 Legal reset boundaries with causal interval revelation

Let

\[
0=b_0<b_1<\cdots<b_m=N
\]

be the legal reset boundaries. Suppose that at boundary `b_n` the engine knows,
or has safe upper envelopes for, every map and budget through `b_{n+1}` before
committing that interval. Propagate the current radius across the interval. If
all rows are safe, keep the current anchor. If any row would violate, reset at
`b_n` and reevaluate from radius zero. If the zero-start interval still
violates, that interval must run exact.

### Theorem 5a: boundary-lazy optimality

Under the preceding information pattern, fixed reset cost, and monotone maps
`f_t(r)=g_t r+eta_t`, boundary-lazy minimizes the reset count on every realized
prefix among schedules restricted to the same legal boundaries.

#### Proof

Begin at a boundary where boundary-lazy and a competing schedule have both
just installed an exact state. Let `b_n` be the start of the first interval
that boundary-lazy cannot traverse safely without another reset. The
boundary-lazy trajectory is safe up to `b_n`. Any feasible competitor must
place its next reset at some legal boundary no later than `b_n`; otherwise it
has the same no-reset trajectory and violates inside that interval. Move the
competitor's next reset to `b_n`. The prefix remains safe because it can follow
the boundary-lazy trajectory, and the moved reset produces radius zero at
`b_n`, no larger than the radius produced by the earlier reset. Monotonicity
preserves all later feasibility. The schedules are now synchronized at
`b_n`; repeat the exchange. QED.

The CPU artifact implements this policy as
`lazy_boundary_reset_schedule` and checks it against the restricted-boundary
dynamic program and exhaustive enumeration. If the interval envelopes are not
known at `b_n`, a DFlash block can still be executed speculatively from a
snapshot and rejected before commit. That preserves the loss guarantee but may
waste the speculative work. With adversarially revealed maps and no stochastic
model, no deterministic cost-competitive guarantee against an offline policy
is possible.

### 6.4 Profitability and the unknown-horizon limitation

For a realized run,

\[
\boxed{
G_N=N C_s-N_r C_r-C_{\mathrm{cert}},
}
\tag{6.4}
\]

where `C_cert` is certificate overhead. The average exact interval must exceed approximately

\[
\frac{C_r}{C_s}
\tag{6.5}
\]

before certificate overhead and risk margin.

If the engine may choose to disable approximation entirely and the horizon is adversarially unknown, no deterministic multiplicative guarantee against an offline controller exists. An adversary can stop before the first reset break-even when approximation is enabled, or continue for a very long safe interval when it is disabled. The engine therefore needs one of these operational assumptions:

- approximation is already selected, in which case lazy reset is exactly optimal;
- accepted-length and gate statistics are modeled, permitting a stochastic or receding-horizon decision;
- the optimization is enabled only after measured traces show positive gain with margin.

The sample reset instance intentionally has negative net gain under its chosen `C_r/C_s`. The correct decision for that instance is to disable the approximate state path.

## 7. DFlash block boundaries, rollback, and random accepted lengths

The current batch verification path copies KDA state and convolution history at the block start, then restores those copies and replays the accepted prefix from archived pre-convolution `q,k,v,gate,beta` values after a rejected batch. Under the current exact FP32 target path, both the block-start state and the archived recurrence inputs are exact relative to the designated target path.

A persistent approximate-state path changes the meaning of the same copy. Copying an inexact state into an FP32 buffer preserves its error; it does not create an exact checkpoint.

### 7.1 Three different transitions

Let block `n` start with certified radius `r_n`, candidate width `K_n`, and accepted length `A_n=a`. Define

\[
F^A_{n,a}(r)
\]

as the row-by-row approximate certificate map and

\[
F^E_{n,a}(r)
\]

as exact KDA replay of those rows with no new local approximation defects. The possible transitions are:

- commit the approximate prefix: `r_{n+1}=F^A_{n,a}(r_n)`;
- restore the block-start copy and replay accepted rows exactly: `r_{n+1}=F^E_{n,a}(r_n)`;
- restore a genuine exact anchor and replay exactly: `r_{n+1}=0` at the new accepted boundary.

For the scalar KDA certificate,

\[
F^E_{n,a}(r)
\le
\left(\prod_{j=1}^a\rho_{n,j}\right)r.
\tag{7.1}
\]

Thus rollback removes errors created inside the rejected block, but inherited error remains. The second transition equals zero only when `r_n=0`.

The CPU artifact includes this check explicitly: exact replay from a snapshot with incoming radius `0.2` remains positive, while replay from a genuine exact anchor remains zero.

### 7.2 Two implementable modes

**Block-local approximation.** Keep the committed target state exact at every DFlash boundary. Approximation may be used inside a candidate block, but an accepted boundary must be produced by an exact target reconstruction. The existing rollback buffers provide the exact starting anchor. KDA-only replay through `rollback_kda` is sufficient only when every archived recurrence input is the one the exact target path would have produced. If the approximate KDA output changed a later hidden state, layer input, convolution input, gate, or value vector, the archive from that approximate pass is not causally exact; the accepted prefix must instead be rerun through the full exact target stack from the anchor. The shadowing horizon is at most the candidate width, currently at most eight rows, but the full-reconstruction cost must be charged.

**Persistent approximate or compressed state.** Commit an approximate accepted state across blocks. The next block-start copy has radius `r_n>0` and is only a rollback point. A reset to zero now requires one of the following:

- retain a much older exact state and all causally sufficient recurrence inputs since that anchor;
- rerun the exact model from that anchor;
- maintain an exact recurrent shadow continuously.

The current archive is one-block only. Its per-row size is

\[
34\,[4(8192)+64]\,4\ \text{bytes}
=4.2583\ \text{MiB}.
\tag{7.2}
\]

Eight rows consume `34.0664 MiB`; one thousand rows would consume about `4.1585 GiB` before metadata or alignment. A long archive also assumes the stored recurrence inputs are the exact causal inputs. If upstream recurrent error changes later hidden features, replaying only KDA from approximate archived inputs is not an exact reset of the full network.

This distinction is decisive. The current one-block rollback mechanism supports block-local shadowing. It does not by itself support periodic exact resets of a state compressed across thousands of tokens.

### 7.3 Pathwise-safe block rule

At a block boundary with a genuine exact anchor:

1. snapshot KDA state, convolution history, and the scalar state-norm majorants;
2. process candidate rows through the approximate recurrence and certificate;
3. before declaring the next boundary exact, reconstruct the accepted prefix from causally exact recurrence inputs;
4. when those inputs may have been changed by the approximation, restore the anchor and rerun the full exact target prefix rather than using KDA-only replay;
5. on mismatch, certificate violation, or structural fault, take the same exact reconstruction path;
6. set `R_h=0` only after the exact reconstruction completes and its state is installed.

If an approximate prefix is committed without exact reconstruction, carry its nonzero radius into the next block and mark the new snapshot as rollback-only. Never reset the certificate to zero because the storage format is FP32.

When exact resets are legal only at selected block boundaries, the finite-horizon dynamic program (6.3) is restricted to those legal boundaries. `tools/kda_shadowing.py` implements this restriction and checks it against exhaustive enumeration.

### 7.4 Random accepted length

For a finite block horizon with a causal distribution `p_n(a)`, a Bellman recursion can use

\[
\begin{aligned}
V_n(r)=\max\{&V_n^{\mathrm{exact}}(r),\\
&\sum_a p_n(a)[aC_s+V_{n+1}(F^A_{n,a}(r))],\\
&\sum_a p_n(a)[aC_s-C_{\mathrm{rb}}(a)
                 +V_{n+1}(F^E_{n,a}(r))],\\
&\sum_a p_n(a)[aC_s-C_{\mathrm{reset}}(a)
                 +V_{n+1}(0)]\}.
\end{aligned}
\tag{7.3}
\]

The last action is admissible only when a genuine exact anchor and causally sufficient replay data exist. `C_rb` is block rollback plus accepted-prefix replay. `C_reset` includes reconstruction from the older exact anchor and can be much larger.

A pathwise loss contract checks every accepted length that can be committed. A chance contract can allocate a summable per-block risk budget, but it is weaker than deterministic shadowing. Random accepted lengths make the process semi-Markov; long-run profitability uses expected reward divided by expected wall time, not an average of per-block ratios.

## 8. Discontinuous FP8 block scaling

Consider max-scaled E4M3FN quantization

\[
Q_s(x)=s\,q_{\mathrm{E4M3}}(x/s),
\]

with upward power-of-two scale

\[
s(x)=2^{\lceil\log_2(\|x\|_\infty/448)\rceil}.
\tag{8.1}
\]

The scale jumps whenever the block maximum crosses `448 * 2^k`. A tiny perturbation in one maximal entry can therefore change the quantization grid for every other state cell. There is no finite global Lipschitz constant for `x -> Q_{s(x)}(x)`.

The artifact uses two blocks whose maxima differ by about `2e-7` around `448`. The scale jumps from `1` to `2`; another small entry changes from `0.001953125` to zero. The observed output/input difference ratio is about `9.77e3`, and it diverges as the maximum perturbation shrinks.

### 8.1 A safe bound that survives the discontinuity

Let `h_C` be the largest half-gap in the finite normalized codebook. For E4M3FN, `h_C=16`. If upward scaling prevents overflow and a block has `n` cells, then

\[
\boxed{
\|Q_s(x)-x\|_F\le h_Cs\sqrt n.
}
\tag{8.2}
\]

This bound treats the selected scale as observed data. It does not assume scale stability. It remains valid on both sides of a scale threshold.

For a `128 x 128` state block, `sqrt(n)=128`, so the worst-cell bound is `2048s`. That may be far too loose for a useful state-loss budget. The practical alternative is to accumulate the realized residual norm

\[
\sum_{ij}(Q_s(x)_{ij}-x_{ij})^2
\]

inside the quantizer kernel while the FP32 prequantized values are still available.

### 8.2 Stability margin

Expose the distance from the current block maximum to the adjacent scale bucket threshold. If the upstream infinity-norm uncertainty is smaller than that margin, the scale is guaranteed not to change. A runner-up maximum gap can certify which entry remains maximal, though it is not needed for the additive residual bound.

Scale stability alone does not remove elementwise rounding discontinuities. It allows a sharper fixed-scale analysis; the additive cell-radius term is still required.

### 8.3 Required failure flags

The certificate is invalid if the quantizer reports a nonfinite input, overflow, unsupported saturation, or a scale format whose rounding error is not included. Any such event forces exact replay.

## 9. Backward error

A broad backward statement is immediate if the model is enlarged to allow arbitrary additive matrix inputs:

\[
\widehat S_t=
\widehat A_t\widehat S_{t-1}\widehat B_t+
\widehat R_t+q_t.
\]

This is an exact recurrence with additive perturbation `q_t`. It is causal because `q_t` is determined by the current prequantized state.

A backward statement inside the original rank-one family is false in general.

### Theorem 6: rank obstruction

Start from `S_{t-1}=0`. Any recurrence with one rank-one update produces a next state of rank at most one, regardless of the gates. If a quantizer residual makes the approximate next state rank two, no perturbation of `beta,u,v,A,B` within the same one-update recurrence can reproduce it.

The artifact makes the obstruction concrete with E4M3FN. It starts from the rank-one matrix

\[
\begin{pmatrix}1/10\\29/195\end{pmatrix}
\begin{pmatrix}1/10&16/65\end{pmatrix}
\]

and applies one upward power-of-two max-scaled E4M3FN block quantization. The stored matrix is exactly

\[
\begin{pmatrix}
5/512&13/512\\
15/1024&9/256
\end{pmatrix},
\]

whose determinant is `-15/524288`, so its rank is two. SymPy checks the ranks and determinant exactly.

More generally, preserving the rank-one update requires the matrix remaining after subtracting the propagated state to have rank at most one and to admit factors consistent with the `beta` and normalization constraints. State quantization generally produces a full-rank residual, so this condition usually fails.

One can try to absorb `q_t` into an unstructured gate perturbation. If `widehat S_{t-1}widehat B_t` is invertible, a formal choice is

\[
\delta A_t=q_t(\widehat S_{t-1}\widehat B_t)^{-1}.
\]

Its norm is controlled by the reciprocal smallest singular value and can be arbitrarily large. Diagonal or other structured gates cannot represent an arbitrary residual. If the state is singular, a component outside the reachable row/column spaces remains impossible.

Causal realizability also matters. The quantization residual is known only after the coefficients have produced the prequantized state. Treating it as a perturbation to upstream token features may require those features to depend on a later rounding outcome. The enlarged additive-noise recurrence is causally sound. A general nearby-input statement for the original network is not.

## 10. CPU artifact results

Run:

```bash
python tools/kda_shadowing.py --out scratch/kda-shadowing --d 8 --steps 24
python tools/test_kda_shadowing.py
```

The 14-test unit suite performs exact numeric decompositions, SymPy rational checks for every `d=1,2,3,4`, a `d=32` simulator smoke test, homogeneous and driven KDA energy checks, all bound comparisons, unrestricted and block-boundary reset control versus dynamic programming and exhaustive search, scale-discontinuity checks, an actual E4M3 rank obstruction, fused-residual versus worst-cell FP8 certification, and artifact generation.

Representative deterministic output:

| Scenario | Peak observed error | Final norm-only bound | Online worst-cell | Online fused residual | Final transition/structural bound |
|---|---:|---:|---:|---:|---:|
| contractive KDA | `5.76e-3` | `1.92e-2` | `3.80e-2` | `3.77e-2` | transition `1.00e-2`, KDA `1.99e-2` |
| contractive KDA + E4M3 state | `3.30e-2` | `6.18e-1` | `6.37e-1` | `1.41e-1` | transition `6.72e-2` |
| marginal | `6.00e-3` | `6.00e-3` | `6.00e-3` | `6.00e-3` | `6.00e-3` |
| switching nonnormal | `3.56e-2` | `3.66e-2` | `3.66e-2` | `3.66e-2` | `3.66e-2` |
| aligned adversarial | `3.72e-3` | `3.72e-3` | `3.72e-3` | `3.72e-3` | `3.72e-3` |
| diagonal shared directions | `1.30e-3` peak | `1.12e-1` final | `1.12e-1` final | `1.12e-1` final | diagonal `3.08e-6` final |

No proposed bound is violated. The aligned adversarial sequence attains the product bound at every step. The switching sequence has spectral radius below one for every individual step and still amplifies its initial error by more than three orders of magnitude. The exact KDA scenarios satisfy both energy identities to floating-point tolerance. In the E4M3 state run, replacing the worst-cell quantizer envelope with the realized residual norm accumulated in the quantizer pass reduces the final online radius from about `0.637` to `0.141`; the worst-cell certificate is already much less useful after 24 rows.

The exact reset solvers agree on reset count and net saving. Different boundary placements can tie, which is expected. The tokenwise causal lazy schedule chooses the latest safe boundaries and has the same optimum reset count. The boundary-lazy controller also matches the restricted-boundary dynamic program and exhaustive enumeration. The rollback demo returns radius `0.171` from an inexact `0.2` snapshot and radius zero only from an exact anchor.

## 11. Engine decision

### 11.1 Scalars the CUDA KDA path must provide

For each of the `34*64=2176` KDA heads, maintain two FP32 certificate scalars in device memory:

- `M_h`: exact-state Frobenius norm majorant;
- `R_h`: state-error Frobenius radius.

The pair costs `17,408 bytes`, about `17.0 KiB` for the whole model. Snapshot `M_h` and `R_h` together with recurrent state at every rollback boundary.

For each token or candidate row, the KDA path must provide these inputs to the certificate update:

- `decay_max_upper_h`, or `-log(decay_max_upper_h)`;
- `beta_upper_h` and a flag that exact and approximate beta remain in `[0,1]`;
- normalized `||k||_2`, raw-key norm margin, and the post-normalization perturbation bound `epsilon_k_2`;
- `||v||_2` and `epsilon_v_2`;
- `epsilon_g_inf` or the resulting `epsilon_D_inf`;
- `epsilon_beta`;
- for state quantization, realized `||q_state||_F` or a deterministic residual bound accumulated in the same quantizer pass;
- state-block `amax`, selected scale, distance to the adjacent scale bucket, saturation/overflow status, and a nonfinite flag.

Use the driven energy recurrence (4.8) for `M_h` and the approximate-propagator recurrence (5.2) for `R_h`. The vectors are already touched by normalization, gating, and the recurrent update. Fuse their reductions. Do not copy one scalar per head to the host every token. Keep the per-head radii on device, reduce a calibrated loss ratio and fault word per layer or DFlash block, and expose only the aggregate maximum and OR-ed flags to the controller.

For exact FP32 state with perturbed coefficients, no separate `d*d` state scan is required. For compressed state, residual accumulation must ride the quantizer write pass. A second state-memory pass fails the cost test.

### 11.2 Snapshot placement and exact-anchor semantics

The current batch DFlash path has device buffers for:

- all KDA recurrent states: `136.0 MiB`;
- all KDA convolution histories: `9.5625 MiB`;
- total recurrent rollback image: `145.5625 MiB`.

Keep those buffers in VRAM for batch verification. Also snapshot the roughly `17 KiB` certificate state. The existing per-row recurrence-input archive costs `4.2583 MiB`; its eight-row allocation costs `34.0664 MiB`.

Under the current exact FP32 target path, the block-start recurrent image is a genuine exact anchor. A rejected block can restore it, replay the accepted prefix from the exact pass's archived inputs, and set `R_h=0` at the accepted boundary. If a proposed approximation changes those archived inputs, KDA-only replay no longer justifies zeroing `R_h`; a full exact target rerun is required.

Once an approximate or compressed state is committed across a boundary, a new FP32 copy is only a rollback snapshot. Store its `R_h` and restore that radius with it. Do not label it exact and do not zero the certificate after replay.

A persistent compressed-state design therefore needs a separate last-exact anchor plus causally sufficient data for reconstruction. The current one-block archive is not enough. Extending it to one thousand rows would require about `4.1585 GiB`, and KDA-only replay is exact only if those archived inputs are themselves exact. Otherwise the engine must rerun the full exact network from the anchor.

Row-sequential verification currently removes the recurrent snapshots because exact state naturally stops at the accepted row. That remains valid for the current exact path. It is incompatible with committing persistent approximate state unless another genuine exact-anchor mechanism is added.

### 11.3 Reset work, saved work, and accounting

Distinguish two costs.

For a block that starts from an exact anchor,

\[
C_{\mathrm{rb}}(a)=C_{\mathrm{restore\ D2D}}
+a\,C_{34\ \mathrm{KDA\ exact\ rows}}
+C_{\mathrm{certificate\ restore}},
\tag{11.1}
\]

where `a` is the accepted prefix length. With the existing archive, each exact KDA row includes three depth-4 convolutions, gate formation, and every `128*128` head-state update. It avoids dense projections, MLA, MoE expert reads, and full target-layer execution because the recurrence inputs were archived.

For a persistent approximate state whose last genuine exact anchor is `L` tokens old,

\[
C_{\mathrm{reset}}(L)=C_{\mathrm{restore\ anchor}}
+C_{\mathrm{reconstruct\ causal\ inputs}}(L)
+C_{\mathrm{exact\ replay}}(L).
\tag{11.2}
\]

This may be a full-model replay, not a KDA-only replay. The controller must never substitute `C_rb` for `C_reset` in this mode.

Measure both with CUDA events. Measure per-token approximate saving `C_s` around the same region and include snapshot copies, certificate reductions, scale selection, and residual accumulation. For a realized trace, enable the path only when

\[
N C_s>
\sum C_{\mathrm{rb}}+
\sum C_{\mathrm{reset}}+
C_{\mathrm{cert}}
\tag{11.3}
\]

with a margin that survives WSL timing variance.

### 11.4 Event that forces exact replay

Before committing each row, update all `M_h,R_h`, reduce the calibrated loss proxy, and force the exact path when any condition holds:

- the next committed row would exceed the numerical or calibrated loss budget;
- beta, normalized key norm, or decay violates the structural KDA contract;
- the raw-key norm margin collapses and only the `1/sqrt(epsilon)` normalization bound remains;
- a nonfinite value, unsupported saturation, scale overflow, or quantizer fault is reported;
- state-quantization residual norm is unavailable;
- a fixed-scale argument is used and the scale-bucket margin is no larger than upstream uncertainty;
- the existing target-logit, route, acceptance, or retry guard rejects the approximate result;
- batch verification advanced beyond the accepted prefix.

From a genuine exact block anchor, restore state, convolution history, and certificate scalars, replay the accepted prefix exactly, then set `R_h=0`. From a rollback-only snapshot, restore its saved `R_h`; exact replay removes only the new block defects. If no genuine exact anchor can be reconstructed at acceptable cost, disable persistent approximation rather than claiming an exact reset.

### 11.5 Go/no-go rule

**Proceed only with an instrumented block-local GPU experiment** when approximation never survives an accepted DFlash boundary without exact reconstruction. The horizon is at most eight rows and the existing snapshot is an exact starting anchor. Before reporting a speedup, include the cost of a full exact target rerun whenever approximation could have changed archived recurrence inputs; `rollback_kda` alone is not then an exact reset.

**Do not enable persistent compressed KDA state in the current engine.** The mathematical propagation certificate is nonexpansive and inexpensive, but the present engine has no long-horizon exact-reset mechanism. A one-block rollback snapshot does not erase inherited error, and a thousand-token exact-input archive would exceed `4 GiB`.

Reconsider persistent compression only after all of these are demonstrated:

1. a genuine exact-anchor and causal reconstruction design whose measured cost is included in `C_reset`;
2. fused state-quantization residual accumulation with no second state scan;
3. certified radii that remain below budget for several thousand tokens under format-derived perturbation envelopes;
4. average profitable interval longer than the measured break-even ratio;
5. no unexplained route or logit divergence inside the certified region.

Kill the experiment immediately when the only sound residual bound is the E4M3 worst-cell bound and it consumes the budget in one or a few tokens, when a useful certificate needs a separate `d*d` scan or an extra transition matrix per head, or when exact reconstruction costs more than the approximation saves. Under those conditions, FP32 recurrent state is the correct design. Approximate upstream vectors remain a separate option because their envelopes are cheap and the exact KDA map does not amplify homogeneous state error.
