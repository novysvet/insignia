# Causal expert residual coding: feasibility frontier and no-kernel gate

Date: 2026-08-31
Reference: `codex/glm53-dflash2-4070-super` at
`0740c63d1b7c24ff603bf81462f4d8430ad239a1`

## Status and engine decision

Do not write a CUDA residual-correction kernel from the information currently
available. No real expert-output trace accompanies the assignment, so there is
no evidence that GLM-5.3-Flash lies in a compressible conditional family. The
CPU artifact proves that a positive scheme is possible under explicit shared
basis or sparse-innovation assumptions. It also contains independent and
route-only adversarial families where the same idea has no byte advantage.

The current positive candidate is therefore a trace experiment, not a kernel:

1. collect causally timestamped expert inputs, routes, prior-logit summaries,
   exact projection outputs, proposed chunk contributions, cache state, and
   timing;
2. measure held-out conditional code length and conditional risk;
3. solve the representation/cache problem with the measured byte extents;
4. attempt an exact arithmetic-order certificate;
5. write a kernel only if every kill test in the final section passes.

The strongest synthetic positive case, an exactly shared rank-8 basis, stores
17.1875% of the raw synthetic matrices and reads no residual payload. A
shared-basis plus sparse-residual case stores 64.0625% and reads 46.875% per
random expert request. Independent random matrices read 100%, and a
route-predictive adversary also reads 100%. These are controlled constructions,
not estimates of the model.

## Repository facts and evidence boundary

The current engine facts used here come from `src/glm53_generate.cu` and
`AGENTS.md`:

- 42 sparse layers route top-8 experts, giving 336 layer-expert requests before
  cache hits for one decode token;
- a current expert payload has three projections, each with a 4 MiB packed
  nibble body and a 512 KiB scale region, for 13.5 MiB before small headers and
  globals;
- expert windows and packed records are 4096-byte aligned;
- gate and up projections feed a clamped SwiGLU, then the down projection;
- the engine already retains and replays routed down results when needed to
  preserve scalar FP32 accumulation order.

No route persistence, hidden-state smoothness, residual entropy, conditional
mutual information, or deadline slack is assumed. Every such quantity in the
artifact is generated from a named parameter in
`SyntheticParameters` and is written to
`scratch/causal-expert-residual/configuration.json`.

## 1. Query and computation model

### 1.1 Arbitrary finite expert maps

Restrict the input domain to a finite set
\(\mathcal X=\{x_1,\ldots,x_Q\}\). This restriction is enough for a lower
bound on any larger or continuous domain. There are \(E\) experts, and each
answer is a \(q\)-bit string:

\[
F=(f_e(x_i))_{e\in[E],i\in[Q]}
   \in (\{0,1\}^q)^{N},\qquad N=EQ.
\]

An offline encoder sees all of \(F\) and emits:

- a resident message \(M\) of at most \(s\) bits;
- an external random-access representation \(S\).

At query time the decoder receives \((e_t,x_t)\), the resident state, and a
causal context \(C_t\). Context may contain all earlier queries and exact
answers, routes, logits, summaries, prior external reads, and persistent cache
contents. It may not contain information from a future answer except through
\((M,S)\). The decoder may use unlimited computation and adaptive probes into
\(S\). All returned payload bits count. If access lengths, timing, or addresses
are allowed to carry source-dependent information, the complete interaction
transcript counts instead. Fixed aligned-sector reads are a special case.

The exact decoder must return every answer with zero error. Randomized
zero-error decoders are covered by fixing their random tape. Approximate codes
are treated later with an explicit distortion or selective-risk contract.

### 1.2 No-free-lunch storage theorem

**Theorem 1.** For the class of all maps \(F\), every zero-error representation
satisfies

\[
|M|+|S|\ge Nq.
\]

**Proof.** There are \(2^{Nq}\) possible maps. Two maps that produce the same
pair \((M,S)\) cannot be distinguished by an exhaustive sequence of the \(N\)
queries, contradicting zero-error decoding. The encoder therefore needs at
least \(Nq\) distinct bits of representation. QED.

This theorem permits arbitrary predictor computation. A predictor cannot
create one of the missing \(Nq\) independent bits.

### 1.3 Causal exhaustive-read theorem

Let the table cells \(Y_1,\ldots,Y_N\) be independent uniform \(q\)-bit
strings. Query all cells in an independent uniformly random permutation
\(\Pi\). Let \(T\) be the complete external transcript over the exhaustive
sequence, with already read data allowed to remain cached forever.

**Theorem 2.** If \(H(M)\le s\), then

\[
\mathbb E[|T|]\ge Nq-s,
\qquad
\frac{\mathbb E[|T|]}{N}\ge q-\frac{s}{N}.
\]

**Proof.** Exact exhaustive decoding gives
\(H(Y_{1:N}\mid M,\Pi,T)=0\). Hence

\[
\begin{aligned}
H(T\mid M,\Pi)
&\ge I(Y_{1:N};T\mid M,\Pi)\\
&=H(Y_{1:N}\mid M,\Pi)\\
&\ge H(Y_{1:N})-H(M)\\
&\ge Nq-s.
\end{aligned}
\]

A fixed-length bit transcript has length at least its entropy; the same holds
for a self-delimiting variable transcript in expectation. Divide by \(N\).
All causal history is already present in the conditioning and cannot lower the
total entropy of the unseen independent cells. QED.

The theorem is an average over an exhaustive random order. A representation may
make some cells cheap and others expensive, but it cannot make all independent
cells cheap without placing their information in resident memory.

### 1.4 Matrix corollary

For arbitrary \(b\)-bit matrices \(W_e\in\mathbb A^{d_o\times d_i}\), query
the standard basis vectors \(x_i\). Each answer exposes one matrix column and
contains \(q=bd_o\) bits. Theorem 1 gives

\[
E d_i d_o b
\]

bits, exactly the uncompressed matrix information. Thus a universal exact
causal compressor for arbitrary expert matrices does not exist. Nonlinear MLPs
cannot evade this lower bound because the lower-bound family may be chosen to
be linear.

## 2. A measurable positive structure

### 2.1 Shared right basis plus block residual

Assume a layer has

\[
W_e=A_eB+S_e,
\]

where \(B\in\mathbb R^{r\times d_i}\) is shared,
\(A_e\in\mathbb R^{d_o\times r}\), and \(S_e\) is divided into independently
addressable row blocks \(S_{e,1:J}\). The resident predictor is

\[
p_e(x)=A_e(Bx).
\]

The exact mathematical output is

\[
W_ex=p_e(x)+S_ex.
\]

With \(b_B\) and \(b_A\) bytes per stored basis/coefficient value, predictor
residency is

\[
R_{\rm pred}=b_B r d_i+b_A E d_o r.
\]

For one query, direct multiply-add accounting is

\[
2rd_i+2d_or+2\operatorname{nnz}(S_e)
\]

operations. The first term can be shared when several experts use the same
input and the same basis. Without that reuse it is paid per expert.

If the residual blocks have stored lengths \(\ell_{e,j}\) and alignment
\(a\), exact random access reads

\[
B_e=\sum_{j=1}^{J}a\left\lceil\frac{\ell_{e,j}}{a}\right\rceil.
\]

No earlier expert must be decoded. A resident fixed-size directory maps
\((e,j)\) to one extent.

This is a useful model only after \(r\), residual sparsity, byte entropy, and
reuse frequency are measured. A low numerical norm alone does not imply a
short lossless byte code.

### 2.2 Exact byte code `CER1`

Floating-point factorization introduces a second issue: independently
recomputing \(A_eB\) need not reproduce the original stored weight bits. The
CPU artifact therefore separates a structural predictor from exactness. It
generates deterministic predictor bytes \(P_e\) and stores

\[
X_e=W_e\mathbin{\mathtt{XOR}}P_e.
\]

The decoder regenerates \(P_e\), applies the XOR chunk, and checks the CRC of
the reconstructed original bytes. This makes exactness independent of the
quality of the predictor. A poor predictor merely creates a large residual.

`CER1` has this fixed little-endian layout:

- 64-byte file header;
- one 32-byte descriptor for every `(expert, row_chunk)`;
- one initial pad to the configured power-of-two alignment;
- independently aligned payload extents.

A descriptor is:

```text
uint64 offset
uint32 stored_bytes
uint32 extent_bytes
uint32 raw_bytes
uint32 reconstructed_crc32
uint32 mode
uint32 changed_bytes
```

Modes are:

- `ZERO`: no payload;
- `RAW_XOR`: one XOR byte for every raw byte;
- `SPARSE_XOR`: sorted `uint32` byte positions followed by one XOR byte per
  changed position.

For an \(n\)-byte chunk with \(m\) changed bytes, the logical payload is

\[
\ell=\min(n,5m)
\]

unless \(m=0\). The physical extent is
\(a\lceil\ell/a\rceil\). The complete file cost is

\[
64+32EJ+\text{initial pad}+
\sum_{e,j:m_{e,j}>0}a\left\lceil
\frac{\min(n_{e,j},5m_{e,j})}{a}\right\rceil.
\]

The directory is resident. One chunk decode needs one descriptor, one aligned
extent, either \(n\) byte XORs or \(m\) indexed XORs, and one CRC pass. The
format has no global dictionary, serial prefix, or cross-expert dependency.
Corruption, overlapping extents, invalid sparse positions, and CRC mismatch are
rejected by the reference decoder.

This code proves exact *weight-byte* recovery. It does not by itself prove that
an output computed as “predictor result plus correction” matches the current
kernel's FP32 arithmetic. The safe generic exact dispatch is to reconstruct the
complete current record and invoke the existing kernel. Section 6 gives the
narrow condition under which work can be continued instead.

### 2.3 Other structures covered by the same model

A conditional codebook uses resident anchors \(Q_z\), selected by a causal
cluster label \(z(C_t)\). Because the expert record \(W_e\) is fixed before the
future context is known, exact storage must contain a residual for every
supported `(expert, z)` pair, such as `W_e XOR Q_z`, or a context-independent
representation from which that residual can be derived. Its directory therefore
maps `(e, z, chunk)` to an extent, and its storage cost includes all supported
labels even though one label is read per query. It is random access when the
selected anchor is resident and no predecessor chain is required.

Block sparsity gives zero or sparse XOR extents and can also reduce online
residual multiply work. Sparse values must still be encoded losslessly in exact
mode.

If activations move with

\[
\|x_t-x_{t-1}\|\le\delta_t
\]

and \(f_e\) is \(L_e\)-Lipschitz on the visited region, then

\[
\|f_e(x_t)-f_e(x_{t-1})\|\le L_e\delta_t.
\]

This is an approximate-error certificate for a previous-output predictor. It
is not a lossless storage theorem. Exact mode still needs an innovation code
that distinguishes all possible outputs.

Repeated route clusters can reduce the entropy of a predictor or codebook
index. Route predictability alone says nothing about residual entropy; the
artifact includes a counterexample.

### 2.4 Synthetic exact-code results

The default evaluator uses eight `64 x 64` float32 matrices, four row chunks,
4096-byte extents, rank 8, and 4096 causal tokens. The records are intentionally
small, so directory and sector effects remain visible.

| family | exact total / raw | mean exact read / expert | changed-byte fraction | zero chunks |
|---|---:|---:|---:|---:|
| independent random | 1.03125 | 1.00000 | 0.99631 | 0/32 |
| exact shared basis | 0.171875 | 0.00000 | 0.00000 | 32/32 |
| shared basis + sparse residual | 0.640625 | 0.46875 | 0.04733 | 17/32 |
| slowly drifting from one anchor | 1.03125 | 0.87500 | 0.64809 | 4/32 |
| route-only adversary | 1.15625 | 1.00000 | 0.84607 | 0/32 |

All five containers reconstruct every float32 weight bit exactly. The slowly
moving family has a small output-residual RMS, but its dense floating-point byte
innovations remain expensive losslessly. This is the intended distinction
between approximate predictability and exact compressibility.

## 3. Optimal causal residual stopping

Let chunk \(j\) contribute \(\Delta_j\), and let

\[
\widehat Y_k=P+\sum_{j=1}^{k}\Delta_j
\]

be the result after a priority prefix. The information available after prefix
\(k\) is the filtration

\[
\mathcal G_k=\sigma(C_t,\Delta_{1:k},\text{completed reads and timings}).
\]

### 3.1 General Bellman rule

For a nonnegative loss \(L(Y,\widehat Y_k)\), read cost \(c_{k+1}\), and
Lagrange multiplier \(\lambda\), define the optimal remaining cost

\[
V_k(g)=\min\left\{
\lambda\,\mathbb E[L(Y,\widehat Y_k)\mid\mathcal G_k=g],
\ c_{k+1}(g)+
\mathbb E[V_{k+1}(\mathcal G_{k+1})\mid\mathcal G_k=g]
\right\}.
\]

At the final exact prefix, loss is zero. The optimal causal stopping rule is:
stop exactly when the first term is no larger than the continuation term. This
is not a confidence-threshold heuristic; it is the Bellman comparison after
all sunk work is removed.

A hard expected-error constraint is obtained by choosing the dual multiplier
that meets

\[
\mathbb E[L(Y,\widehat Y_K)]\le D.
\]

For finite contexts and prefixes, the primal is a linear program and can be
solved exactly.

### 3.2 Unequal independent chunk value

Suppose squared-error chunk cross terms vanish conditionally:

\[
\mathbb E[\langle\Delta_i,\Delta_j\rangle\mid C]=0,
\qquad i\ne j.
\]

Let \(d_j(C)=\mathbb E[\|\Delta_j\|^2\mid C]\). For arbitrary subset access,

\[
\mathbb E\left[\left\|\sum_{j\notin S}\Delta_j\right\|^2\middle|C\right]
=\sum_{j\notin S}d_j(C).
\]

The Lagrangian separates, so chunk \(j\) is fetched if and only if

\[
\lambda d_j(C)>c_j(C).
\]

If chunks are ordered by nonincreasing \(d_j/c_j\) and only prefixes are
allowed, stop at the first chunk whose marginal value fails this inequality.
When cross terms are nonzero, the value of a chunk depends on the selected set,
and the general Bellman rule is required.

### 3.3 Selective-risk constraint

Let approximate service mean \(A=1\{K<J\}\), and define

\[
r_k(C)=P(L(Y,\widehat Y_k)>\epsilon\mid C).
\]

The selective contract

\[
P(L>\epsilon\mid A=1)\le\alpha
\]

is equivalent, when approximation is used, to the linear inequality

\[
\mathbb E[(r_K(C)-\alpha)A]\le0.
\]

For finite context probabilities \(p_c\) and prefix probabilities \(x_{c,k}\),
the exact program is

\[
\begin{aligned}
\min_x\quad &\sum_{c,k}p_c x_{c,k}c_{c,k}\\
\text{s.t.}\quad&\sum_kx_{c,k}=1,\quad x_{c,k}\ge0,\\
&\sum_{c,k}p_cx_{c,k}(r_{c,k}-\alpha)1\{k<J\}\le0.
\end{aligned}
\]

With one global scalar constraint, an extreme optimum needs at most one
fractional context/action decision. Equivalently, it can be implemented by
randomizing between at most two deterministic policies. The CPU solver
enumerates all deterministic policies for small cases and all pairs for this
exact randomization.

### 3.4 What previous logits must predict

Let \(Z\) be a residual symbol or chunk-need event, \(B\) the base causal
context, and \(L\) a previous-logit statistic.

**Proposition 3, log loss.** The reduction in optimal held-out expected log loss
from adding \(L\) is exactly

\[
H(Z\mid B)-H(Z\mid B,L)=I(Z;L\mid B).
\]

Thus prior logits have no coding value once the base context is known if the
conditional mutual information is zero.

**Proposition 4, squared error.** For vector residual \(Z\), the reduction in
Bayes mean-squared error is

\[
\begin{aligned}
&\mathbb E\|Z-\mathbb E[Z\mid B]\|^2
-\mathbb E\|Z-\mathbb E[Z\mid B,L]\|^2\\
&=\mathbb E\|\mathbb E[Z\mid B,L]-\mathbb E[Z\mid B]\|^2.
\end{aligned}
\]

Both identities prove what the feature predicts. Concatenating logits into a
network without a held-out improvement does not establish value.

The synthetic held-out scalar probe reports:

| family | route accuracy | `I(route; prior top)` bits | `I(residual; prior logit | coordinate, route)` bits |
|---|---:|---:|---:|
| independent random | 0.8936 | 2.2266 | 0.7821 |
| exact shared basis | 0.9023 | 2.2802 | 0.0000 |
| shared basis + sparse residual | 0.8970 | 2.2322 | 0.3118 |
| slowly drifting | 0.8926 | 2.2105 | 0.7641 |
| route-only adversary | 0.90625 | 2.5511 | 0.0000 |

The positive values are generated by the explicit
`logit_activation_accuracy` parameter. In the adversary, every block contains
the full route-by-activation cross product. The prior top route is highly
predictive of route, but the residual is generated from an activation class
independent of that route-side context. Its conditional residual entropy is
2.03064 bits both before and after adding the prior-logit field.

### 3.5 Exact synthetic stopping examples

For the sparse-basis family, predictor-only expected squared error is
0.0236775 in synthetic output units. Reading every residual chunk costs 7744
mean aligned bytes. The exact finite solver obtains:

| allowed fraction of predictor-only error | optimal expected read bytes |
|---:|---:|
| 0.50 | 2523.35 |
| 0.10 | 6274.23 |
| 0.01 | 7558.42 |
| 0.00 | 7744.00 |

The solution may randomize between two deterministic contextual prefix
policies. Full policies and probabilities are in `stopping-policies.json`.
The independent family needs 14,462 mean bytes out of 16,384 to reach 10% of
predictor-only error, illustrating that causal stopping cannot manufacture a
useful residual hierarchy when chunks have similar value.

## 4. Conditional rate-distortion and decoder side information

### 4.1 Conditional rate-distortion lower bound

For a quantized residual source \(Z\), decoder context \(C\), reconstruction
\(\widehat Z\), and distortion \(d\), side information known to both encoder
and decoder gives

\[
R_{Z\mid C}(D)=
\min_{P(\widehat z\mid z,c):\mathbb E d(Z,\widehat Z)\le D}
I(Z;\widehat Z\mid C).
\]

The artifact estimates this finite-alphabet curve with conditional
Blahut-Arimoto updates:

\[
P_\beta(\widehat z\mid z,c)
\propto q_\beta(\widehat z\mid c)e^{-\beta d(z,\widehat z)},
\]

followed by

\[
q_\beta(\widehat z\mid c)
=\sum_zP(z\mid c)P_\beta(\widehat z\mid z,c).
\]

One shared \(\beta\) performs the optimal distortion allocation across context
values. The zero-distortion endpoint is the empirical
\(H(Z\mid C)\).

At normalized distortion 0.10, piecewise interpolation of the generated curves
gives:

| family | coordinate only | + route | + prior logit |
|---|---:|---:|---:|
| independent random | 1.6204 | 1.2804 | 0.4753 |
| shared basis + sparse residual | 0.8102 | 0.4112 | 0.1235 |
| slowly drifting | 1.6034 | 1.1396 | 0.4069 |
| route-only adversary | 1.3127 | 1.3127 | 1.3127 |

Units are bits per quantized probe scalar. The equality across all three
adversarial columns is exact in the finite construction.

### 4.2 Wyner-Ziv direction

If side information is available only at the decoder, the relevant idealized
quantity is

\[
R_{WZ}(D)=
\min_{U-Z-C,\ \widehat z=g(U,C)} I(Z;U\mid C).
\]

This rate is no smaller than the both-sides conditional rate-distortion bound
and can be strictly larger. In this engine, model storage is encoded before
future token context exists, so the both-sides curve is an optimistic lower
bound unless the stored representation can be binned in a way that the decoder
resolves using its causal context.

### 4.3 Why textbook source coding is not enough

The stored source is an expert map or weight record, reused over many future
queries. The decoder does not necessarily need the weights; it asks for the
nonlinear function

\[
f_e(x)=D_e\left(\operatorname{silu}(G_ex)\odot U_ex\right).
\]

An output-residual rate-distortion curve sampled from one input distribution
can understate the information needed to answer another input. For zero-error
functional compression, confusable expert maps form a characteristic graph:
two maps are adjacent when some decoder side information and query require
different outputs. Conditional graph entropy, rather than ordinary source
entropy, governs the idealized functional coding problem. For approximate
queries, one needs a functional rate-distortion analogue.

Consequently, the trace-estimated conditional curve is a necessary byte-rate
lower bound for the sampled output problem, not an achievability certificate
for a reusable expert file. A practical random-access codec supplies the upper
bound. A kernel proposal must show a gap between the current full-record rate,
the practical codec, and the conditional lower bound.

### 4.4 Estimable trace procedure

For a real trace:

1. split complete generations by time into train, calibration, and held-out
   test sets;
2. allow the offline lossless encoder to inspect the complete fixed expert
   file, but predeclare or train-select the representation family, rank, and
   chunking without using test-trace outcomes;
3. fit every trace-conditioned entropy model, context bin, importance model,
   quantizer scale, and stopping policy on train, calibrate risk on calibration,
   then freeze them;
4. estimate conditional code length, rate-distortion, Bayes error, aligned
   bytes, and latency on test;
5. use a paired block bootstrap over complete generations for confidence
   intervals;
6. report a practical random-access file size and aligned bytes per query
   beside every information-theoretic lower bound.

The artifact uses the first half of each synthetic trajectory to set residual
scale and the second half for all information summaries.

References for this section are Blahut, *IEEE Transactions on Information
Theory* 18(4), 1972, DOI `10.1109/TIT.1972.1054855`; Wyner and Ziv,
*IEEE Transactions on Information Theory* 22(1), 1976, DOI
`10.1109/TIT.1976.1055508`; and Orlitsky and Roche, “Coding for computing,”
*IEEE Transactions on Information Theory* 47(3), 2001, DOI
`10.1109/18.915643`.

## 5. Joint representation and cache placement

Let \(a\in\mathcal A_e\) choose one representation for expert \(e\): disk full,
RAM full, VRAM full, predictor plus selected residual chunks, or a
lower-precision copy. Let \(z_{e,a}\in\{0,1\}\). Let
\(b_{k,m}\in\{0,1\}\) place shared predictor object \(k\) in memory tier
\(m\). Each option has RAM bytes, VRAM bytes, expected read/H2D/compute/sync
cost \(c_{e,a}\), distortion \(d_{e,a}\), and risk \(r_{e,a}\).

A finite joint problem is

\[
\begin{aligned}
\min_{z,b}\quad&
\sum_e\pi_e\sum_a c_{e,a}z_{e,a}+\sum_{k,m}c^B_{k,m}b_{k,m}\\
\text{s.t.}\quad&
\sum_a z_{e,a}=1,\\
&\sum_{e,a}R_{e,a}z_{e,a}+\sum_{k,m}R^B_{k,m}b_{k,m}\le R_{\max},\\
&\sum_{e,a}V_{e,a}z_{e,a}+\sum_{k,m}V^B_{k,m}b_{k,m}\le V_{\max},\\
&\sum_m b_{k,m}\le 1\quad\forall k,\\
&z_{e,a}\le\sum_m b_{k(a),m}\quad\text{for predictor-dependent options},\\
&\sum_e\pi_e\sum_a d_{e,a}z_{e,a}\le D,\\
&\sum_e\pi_e\sum_a r_{e,a}z_{e,a}\le\alpha.
\end{aligned}
\]

Chunk-level placement is obtained by giving each residual chunk its own binary
object and adding dependency constraints. Cache replacement can be added by a
finite-state occupation measure or trace replay; the static integer program is
the exact small-instance core.

The reference solver exhaustively checks 6,591 configurations for three
experts, five representation choices, three basis placements, and independent
disk/RAM/VRAM placement of the `head` and `tail` residual chunks whenever a
predictor option is selected. The tail is requested with declared conditional
probability 0.35. Examples are:

| scenario | basis | expert forms | cached residual chunks | expected cost | distortion |
|---|---|---|---|---:|---:|
| tight exact | RAM | residual, residual, residual | e0 head/tail VRAM; e1/e2 head VRAM, tail RAM | 0.6490 | 0 |
| tight approximate | RAM | low precision, residual, residual | e1/e2 head VRAM, tail RAM | 0.6200 | 0.025 |
| balanced exact | RAM | residual, residual, residual | e0/e1 head/tail VRAM; e2 head VRAM, tail RAM | 0.5944 | 0 |
| roomy exact | RAM | VRAM full, residual, residual | e1/e2 head/tail VRAM | 0.4290 | 0 |

The units in this toy problem are declared abstract cost and byte blocks. Each
full copy, basis, low-precision copy, and residual chunk is charged to the same
capacity constraints. The purpose is to verify exact optimization and
dependency handling, not to estimate production cache value.

Predictor residency can itself be large. If gate/up share one 4096-wide right
basis, down uses one 2048-wide basis, every expert has rank-\(r\) coefficients,
and all 42 sparse layers are resident, then

\[
B_{\rm pred}=42b\left[r(4096+2048)+288r(2048+2048+4096)\right].
\]

At rank 32 and one byte per predictor value this is 2.9608 GiB, equal to 224.6
current 13.5 MiB expert records. At two bytes it is 5.9216 GiB. A predictor that
saves residual bytes but evicts a more valuable set of full experts is a net
loss; the cache solver must price that displacement.

## 6. Correctness-preserving speculative residuals

### 6.1 Canonical-prefix continuation theorem

For one output row, let the model-visible canonical FP32 update be

\[
a_{k+1}=\operatorname{FMA}_{32}(w_k,h_k,a_k),
\qquad k=0,\ldots,m-1,
\]

with a fixed initial accumulator, term order, rounding mode, denormal mode, and
compiler contraction behavior.

**Theorem 5.** Suppose the resident predictor contains exactly the canonical
terms \(0,\ldots,s-1\), the residual extent contains exactly the suffix terms
\(s,\ldots,m-1\), and the activation bits \(h_k\) are already exact. Computing
the prefix, storing the FP32 accumulator, reloading it, and continuing the
canonical suffix produces exactly the same FP32 bits as uninterrupted
execution.

**Proof.** The prefix computes the same recurrence through state \(a_s\). A
store/reload of FP32 preserves that state's bits. Induction on the remaining
canonical updates gives the same \(a_{s+1},\ldots,a_m\). QED.

The same proof applies independently to every output row. It does not apply to
a separately reduced predictor dot product and residual dot product.

### 6.2 Required buffer schedule

A valid speculative down-projection schedule is:

1. validate a descriptor whose kernel-ABI hash, projection shape, split index,
   quantization rules, and predictor fingerprint match the running binary;
2. issue the aligned suffix read;
3. compute gate and up exactly, then run the existing clamped SwiGLU and retain
   the exact activation vector `h`;
4. initialize every down accumulator exactly as the current kernel does and
   process only the canonical prefix terms;
5. store FP32 prefix accumulators and `h` in a continuation buffer; do not apply
   router scaling or expose the expert result;
6. after DMA completion and CRC validation, resume at the exact suffix index
   with the same FP32 update instruction and order;
7. only then enter the current order-preserving routed accumulation/replay path.

A continuation buffer for one expert needs the exact SwiGLU activation plus one
FP32 accumulator per output row, along with the split metadata and completion
event. The prefix and suffix must not be independently reduced by different
warps and added later.

The sufficient algebraic cases are narrow:

- gate and up are exact and shared/resident, while down is a canonical prefix
  plus suffix;
- the gate/up residual provably leaves every post-nonlinearity activation bit
  unchanged;
- the model's normative arithmetic is explicitly redefined to be predictor
  reduction followed by residual reduction. This last case is not parity with
  the current model and is therefore approximate unless separately accepted.

### 6.3 Nonlinear gate/up duplicate-work lower bound

Write

\[
g=Gx,\quad u=Ux,\quad h=\phi(g,u),\quad y=Dh,
\]

where \(\phi\) is the clamped elementwise SwiGLU. A predictor computes
\(g_0,u_0,h_0,D_0h_0\). After residual gate/up weights arrive,

\[
h=\phi(g_0+\delta g,u_0+\delta u).
\]

The exact down result is

\[
y=D_0h+\Delta D h
  =D_0h_0+D_0(h-h_0)+\Delta D h.
\]

For a non-affine \(\phi\), generic gate/up perturbations make \(h-h_0\) dense.
In the direct sparse/dense FMA model used by GEMV kernels, evaluating
\(D_0(h-h_0)\) for generic dense \(D_0\) requires one coefficient-activation
contribution for every nonzero coefficient:

\[
\Omega(\operatorname{nnz}(D_0))
\]

new multiply-add contributions. This is the operation count of the base down
GEMV. Thus predicted down work cannot be made asymptotically free by calling it
“correction.”

The bitwise condition is stronger. In general,

\[
\operatorname{dot}_{32}(D_0+\Delta D,h)
\ne
\operatorname{dot}_{32}(D_0,h_0)
+
\operatorname{dot}_{32}(D_0,h-h_0)
+
\operatorname{dot}_{32}(\Delta D,h).
\]

Separate reductions place roundings at different locations. To preserve the
current canonical order after gate/up changes, the down projection must be
restarted from its initial accumulator. Every speculative `D_0 h_0` FMA is
then unavoidable duplicate work.

The CPU artifact finds concrete counterexamples for both an explicit
multiply-then-add FP32 recurrence and a fused-FMA recurrence. The fused example
produces different final bit patterns for canonical combined weights and a
separately reduced predictor plus correction. Prefix continuation passes every
split in both recurrence models.

The lower bound is stated for the current direct coefficient-times-activation
kernel model. It does not rule out special structured matrices with a faster
linear circuit. Such structure would be another measurable prior and would
need its own bitwise continuation certificate.

## 7. Production layout and dispatch semantics for a positive result

`CER1` is a CPU reference, not a production ABI. A production sidecar should
retain its independence and add engine certificates.

### 7.1 Resident metadata

Use these fixed little-endian records. The hot header and every expert entry are
cache-line aligned; chunk extents are 4096-byte aligned.

```c
struct CausalExpertFileHeader {           // 64 bytes
    char     magic[8];                    //  0: "ICER0001"
    uint16_t version;                     //  8: 1
    uint16_t header_bytes;                // 10: 64
    uint16_t expert_entry_bytes;          // 12: 128
    uint16_t chunk_desc_bytes;            // 14: 32
    uint32_t layers;                      // 16
    uint32_t experts_per_layer;           // 20
    uint32_t records;                     // 24
    uint32_t alignment;                   // 28: 4096
    uint64_t predictor_catalog_offset;    // 32
    uint64_t expert_directory_offset;     // 40
    uint64_t payload_offset;              // 48
    uint64_t model_tag64;                 // 56
};
static_assert(sizeof(CausalExpertFileHeader) == 64);

struct alignas(64) CausalExpertEntry {    // 128 bytes
    uint64_t predictor_offset;            //   0
    uint64_t chunk_directory_offset;      //   8
    uint64_t predictor_bytes;             //  16
    uint64_t raw_record_bytes;            //  24
    uint8_t  source_sha256[32];           //  32
    uint8_t  predictor_sha256[32];        //  64
    uint32_t packed_schema_hash;          //  96
    uint32_t kernel_arithmetic_abi_hash;  // 100
    uint16_t chunk_count;                 // 104
    uint16_t flags;                       // 106
    uint16_t continuation_projection;     // 108
    uint16_t continuation_split;          // 110
    uint8_t  reserved[16];                // 112
};
static_assert(sizeof(CausalExpertEntry) == 128);

struct CausalResidualChunk {              // 32 bytes
    uint64_t offset;                      //  0, absolute file offset
    uint32_t stored_bytes;                //  8, excludes alignment pad
    uint32_t extent_bytes;                // 12, multiple of 4096
    uint32_t raw_bytes;                   // 16
    uint32_t reconstructed_crc32;         // 20
    uint16_t mode;                        // 24: ZERO/RAW_XOR/SPARSE_XOR
    uint16_t semantic;                    // 26: PACKED_BYTES/DOWN_K_SUFFIX/...
    uint16_t term_begin;                  // 28, canonical K index
    uint16_t term_count;                  // 30
};
static_assert(sizeof(CausalResidualChunk) == 32);
```

A `SPARSE_XOR` payload begins with a little-endian `uint32 changed_bytes`, then
that many sorted `uint32` positions, then the XOR values. `stored_bytes` includes
this four-byte count. `term_begin/term_count` are zero for ordinary packed-byte
reconstruction. A continuation descriptor is legal only for semantic
`DOWN_K_SUFFIX`, with a matching nonzero arithmetic-ABI hash and the
`CONTINUATION_CERTIFIED` expert flag. Flags also distinguish exact-byte
reconstruction, approximate eligibility, and predictor value format.

The full source-record SHA-256 is checked per expert. `model_tag64` is only a hot
reject tag and is never accepted as the sole identity check. Directories are
resident. Every nonempty payload has `offset % 4096 == 0`,
`extent_bytes % 4096 == 0`, and `stored_bytes <= extent_bytes`. No payload may
overlap another or depend on a previous expert.


### 7.2 Exact dispatch

For `(layer, expert)`:

1. use a full VRAM/RAM expert hit through the current path when available;
2. reject the predictor path on any model, schema, basis, coefficient, or
   kernel-ABI fingerprint mismatch;
3. if no continuation certificate exists, read all exact residual extents,
   regenerate predictor bytes, XOR, validate CRC/hash, materialize the current
   13.5 MiB record in the existing staging layout, and invoke the current
   kernel;
4. if a continuation certificate exists, use only the schedule in Section
   6.2;
5. retain final per-expert down results and replay router-weighted accumulation
   in the current top-k order;
6. never expose logits, route state, recurrent state, or a committed token
   before exact correction completes.

The fallback in step 3 is slower in compute but is the exact oracle for every
new optimized path.

### 7.3 Approximate dispatch

A resident policy table receives only fields whose availability timestamp
precedes the decision. It chooses a calibrated chunk prefix and issues those
extents. Priority order may differ from physical/canonical order for value, but
selected arithmetic contributions execute in canonical order. The policy logs
its context bin, selected chunks, risk upper bound, bytes, timing, and whether
an exact rescue was required. A missing or uncalibrated bin selects exact mode.

Lower-precision copies are separate cache options. They must not be silently
mixed with an “exact residual” flag.

## 8. Trace required before a kernel

One record per requested `(generation, token, layer, expert, router_rank)` must
contain the following.

### Identity and causality

- run/config/model hashes, generation ID, token index, layer, expert, router
  rank, router weight;
- monotonic timestamps for token start, each context field becoming available,
  route selection, read issue, host completion, H2D completion, predictor
  launch/end, correction launch/end, layer completion, and commit;
- previous routes and router logits/statistics actually available at decision
  time;
- previous target/draft logit summaries or a losslessly joinable logit-dump key;
- hidden/input summaries with a fixed seed and schema, plus the exact expert
  input or a losslessly joinable dump for offline oracle calculations;
- cache tier, slot, recency, in-flight state, queue depth, and prefetch origin.

### Exact targets

- exact gate and up preactivations;
- exact post-clamp SwiGLU activation;
- exact down output before router scaling;
- exact weighted expert contribution and its position in routed accumulation;
- exact proposed predictor output;
- exact per-chunk contribution in canonical order;
- exact final layer output, target logits, selected token, and next recurrent
  state hash;
- source expert-record hash and proposed residual extent IDs.

### Cost

- resident predictor bytes by tier;
- requested, completed, padded, and useful SSD/H2D bytes;
- CPU decode operations/time;
- GPU predictor, residual, and duplicate operations/time;
- synchronization/event count;
- deadline and slack;
- prediction miss, rescue, CRC failure, and cache displacement counters.

Raw vectors can be stored in sampled oracle runs rather than every production
run, but hashes and deterministic join keys are mandatory. A summary-only trace
cannot validate exact correction.

## 9. Statistical gate

Pre-register predictor families, ranks, chunk layouts, context fields, and
quality thresholds before looking at test data.

Use generation-level, time-ordered train/calibration/test splits. The same
prompt continuation or adjacent token block must not appear on both sides of a
split when it would leak the target. A static lossless transform may inspect
all bytes of the fixed model because that is the encoder's input; its family,
rank, and chunking must still be preregistered or selected without test-trace
outcomes. Fit every trace-conditioned codebook selector, quantizer, entropy
model, stopping policy, and context bin on train, calibrate risk on calibration,
and freeze them before test. An experiment claiming transfer to unseen experts
must additionally split experts or layers at that unit.

The primary side-information statistic is paired held-out code-length gain:

\[
\widehat\Delta_L=
\frac1n\sum_i
[-\log_2\widehat p(Z_i\mid B_i)
 +\log_2\widehat p(Z_i\mid B_i,L_i)].
\]

Under proper log loss this estimates the usable conditional mutual
information. Report practical aligned bytes as well. Use a block bootstrap over
complete generations for a confidence interval.

The null test for prior logits is a stratified conditional permutation:
permute the prior-logit field within predeclared strata containing layer,
current expert/route, router rank, and base hidden/input bin. This preserves its
route-predictive marginal while destroying incremental association with the
residual. Refit only where the predeclared cross-fitting protocol permits it.
Compare the observed held-out gain to the permutation distribution and control
familywise or false-discovery error across chunks and predictor families.

For squared error, also test the held-out reduction in Bayes residual MSE. For
selective mode, calibrate one-sided upper confidence bounds for each context
bin's tail risk. A bin is eligible only when its upper bound is no larger than
\(\alpha\). Merge or reject sparse bins rather than extrapolating.

Replay the exact cache trace under every candidate placement. The objective is
end-to-end time after full-expert hit loss from displaced RAM/VRAM, not residual
ratio in isolation. Use paired block bootstrap intervals for time saved.

Finally, run a parity gate over exact oracle traces:

- reconstructed record bytes identical;
- every expert output FP32 bit identical;
- routed accumulation bit-identical;
- layer output, logits, token ID, and recurrent state identical.

One unexplained bit mismatch rejects an “exact” dispatch.

## 10. Parametric system-cost projection

The evaluator includes a transparent projection using these defaults:

- 13.5 MiB current expert payload;
- eight requests per sparse layer and 336 per token;
- 4 GiB/s SSD, 24 GiB/s H2D;
- 20 effective GPU TFLOP/s, 50 CPU GOP/s;
- 5 microseconds per synchronization;
- 5 ms synthetic layer deadline;
- zero cache hits;
- rank-32 predictor;
- 10% synthetic correction-miss probability in truncated mode.

These are input parameters, not measured outcomes. Applying each synthetic
read ratio to one cold layer gives:

| family | full-record time | exact predictor+residual time | exact residual read |
|---|---:|---:|---:|
| independent random | 30.80 ms | 30.92 ms | 108.0 MiB |
| exact shared basis | 30.80 ms | 0.160 ms | 0 MiB |
| shared basis + sparse residual | 30.80 ms | 14.58 ms | 50.625 MiB |
| slowly drifting | 30.80 ms | 27.08 ms | 94.5 MiB |
| route-only adversary | 30.80 ms | 30.92 ms | 108.0 MiB |

The sparse synthetic family still misses the declared 5 ms layer deadline.
This table demonstrates how to turn a measured ratio into a decision; it does
not estimate the model's ratio.

## 11. CPU artifact and reproducibility

Implemented files:

- `tools/causal_expert_residual.py`;
- `tools/test_causal_expert_residual.py`;
- `tools/evaluate_causal_expert_residual.py`;
- `scratch/causal-expert-residual/` generated results and sample containers.

Run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 tools/test_causal_expert_residual.py -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 tools/evaluate_causal_expert_residual.py
```

The test suite has 19 hardware-free tests covering:

- counting and exhaustive-query lower-bound helpers;
- exact shared basis and route-only adversarial context;
- deterministic `CER1` round trip and random chunk access;
- corruption detection;
- conditional rate-distortion endpoints;
- exact deterministic and randomized prefix policies;
- exact cache enumeration, basis dependencies, and first-class residual
  chunk placement;
- FP32 prefix continuation and additive-correction counterexamples for both
  multiply/add and fused-FMA recurrences.

Generated outputs include:

- `representation-summary.csv`;
- `information-summary.csv`;
- `conditional-rate-distortion.csv`;
- `rate-at-fixed-distortion.csv`;
- `stopping-policies.json`;
- `cache-optima.json`;
- `engine-predictor-residency.csv`;
- `engine-parametric-cost.csv`;
- five exact `.cer` sample containers;
- FP32 counterexample fixtures.

## 12. Kill criteria

Kill the causal residual kernel project before CUDA work if any of the
following holds on held-out real traces.

1. **Conditional information failure.** The lower confidence bound on practical
   aligned-byte savings from context is nonpositive, or prior-logit conditional
   code-length gain is below the predeclared material threshold after the
   route-preserving permutation test.
2. **Lossless residual failure.** Exact aligned residual bytes plus directory,
   predictor storage, and decode overhead are no smaller than the current
   record path. Small residual norm without byte savings does not pass.
3. **Cache displacement failure.** Replaying the joint cache shows that basis,
   coefficients, chunks, or lower-precision copies evict full experts whose
   lost-hit time is at least the residual time saved.
4. **Deadline failure.** The one-sided high-quantile correction completion time,
   including miss rescue and synchronization, exceeds layer slack.
5. **Arithmetic-order failure.** Any exact trace changes a reconstructed byte,
   FP32 expert output, routed accumulation, layer output, logit, token, or
   recurrent-state bit. A generic separately reduced correction is already
   disqualified by the counterexample.
6. **Nonlinear duplicate-work failure.** Gate/up innovations change the SwiGLU
   activation and the required canonical down recomputation erases the overlap
   benefit.
7. **Selective calibration failure.** A context bin's one-sided risk upper bound
   exceeds the contract, drifts out of calibration, or lacks enough support.
8. **Complexity failure.** Descriptor traffic, CPU decode, extra events, or
   branch divergence makes measured end-to-end time no better than full expert
   reads despite nominal byte reduction.

Proceed only if a frozen real-trace analysis shows a positive lower confidence
bound on end-to-end time saved, the joint cache optimum selects the predictor,
the correction finishes inside deadline, and the exact path has a complete
model-visible parity certificate.
