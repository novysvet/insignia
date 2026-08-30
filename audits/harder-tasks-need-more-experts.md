# “Harder Task Needs More Experts” and adaptive-k for Insignia

Date: 2026-08-30

## Executive verdict

The paper’s mechanism is simple and directly causal: compute the router
distribution, then activate the shortest descending-probability prefix whose
cumulative mass exceeds a threshold. A diffuse router distribution therefore
uses more experts, while a concentrated distribution uses fewer. No separate
task-difficulty classifier is involved.

That makes **per-row adaptive k worth testing for Insignia’s approximate
DFlash2 verifier**, because the target router is already evaluated before
`stage_layer()` and expert reads. It does **not** justify pruning the exact
scalar decode path. The ACL paper’s own post-hoc experiment on a checkpoint
trained for fixed Top-2 found that reducing the average below two experts also
reduced quality; matching or exceeding Top-2 quality required *more* than two
experts. GLM-5.3-Flash is likewise a fixed-Top-8 checkpoint, not a model trained
with the paper’s dynamic-routing loss.

The best first policy is therefore a paper-inspired, zero-fill prefix rule,
restricted to provisional DFlash2 verification rows:

\[
m_{\ell,r}=\min\!\left(8,\ \max\!\left(m_{\min},\
\min\left\{m:\sum_{j=1}^{m}q_{\ell,r,j}\ge p_\ell\right\}\right)\right),
\]

where `q` is the normalized mass of the eight experts already selected by
GLM’s router, in the existing selection/accumulation order. Keep the retained
weights unchanged; do not renormalize. Calibrate `p`, `m_min`, and only then
any layer dependence from the existing per-row MSE/cosine traces before
spending time on another long benchmark campaign.

## Primary source

The final publication is *Harder Task Needs More Experts: Dynamic Routing in
MoE Models*, Huang et al., ACL 2024, pages 12883–12895, DOI
10.18653/v1/2024.acl-long.696. The earlier arXiv title uses “Tasks” rather than
“Task.” The authoritative final paper and authors’ implementation are:

- [ACL Anthology record](https://aclanthology.org/2024.acl-long.696/)
- [ACL paper PDF](https://aclanthology.org/2024.acl-long.696.pdf)
- [Authors’ code at commit `f56bc20`](https://github.com/ZhenweiAn/Dynamic_MoE/tree/f56bc20db176f22ab51cb56b7dcb96403fb2cd8a)

## What the paper actually does

For a token representation \(x\), a learned router produces a probability
distribution over all \(N\) experts:

\[
P=\operatorname{softmax}(W_r x^T).
\]

Let \(I\) sort experts by descending \(P\). Dynamic routing selects experts in
that order until cumulative probability exceeds confidence threshold \(p\):

\[
t=\min\left\{k\in\{1,\ldots,N\}:\sum_{j=1}^{k}P_{I_j}>p\right\},
\qquad S=\{e_{I_1},\ldots,e_{I_t}\}.
\]

The output coefficients are \(g_i(x)=P_i\) for selected experts and zero for
the rest. In particular, the dynamic rule does **not** renormalize the retained
experts. Algorithm 1 and the equations are in [§2 of the ACL
paper](https://aclanthology.org/2024.acl-long.696.pdf#page=3). The released
inference code implements the same sequence—softmax, descending sort,
cumulative sum, zero the tail—and feeds the original retained probabilities to
the expert combine
([`modeling_moe.py` lines 142–211](https://github.com/ZhenweiAn/Dynamic_MoE/blob/f56bc20db176f22ab51cb56b7dcb96403fb2cd8a/modeling/modeling_moe.py#L142-L211)).

The “difficulty” signal is therefore **router uncertainty**, not an external
task label:

- one dominant probability crosses \(p\) quickly and is treated as easy;
- dispersed mass requires a longer prefix and is treated as hard.

The authors prevent the router from making every distribution diffuse merely
to buy more computation by adding entropy minimization

\[
L_d=-\sum_i P_i\log P_i,
\]

and train with \(L=L_{LM}+10^{-2}L_{balance}+10^{-4}L_d\). This training term
is not incidental: their ablation increases average active experts from 1.8 to
2.0 while average task score falls from 42.8 to 40.0
([Appendix B.3](https://aclanthology.org/2024.acl-long.696.pdf#page=13)).

## Reported results and scope

The experiment is a 24-layer, hidden-1024 LLaMA-style model with 16 experts per
MoE layer, 3.5B total parameters, and roughly 374M/581M active parameters for
Top-1/Top-2. Models were trained from scratch on 100B RedPajama tokens with
context 2048, using as many as 128 A800 GPUs
([model/training settings](https://aclanthology.org/2024.acl-long.696.pdf#page=12)).

The principal quality result is:

| Model | Average active experts | Average score |
|---|---:|---:|
| fixed Top-1 | 1.0 | 40.5 |
| fixed Top-2 | 2.0 | 41.6 |
| Dynamic, trained with `p=0.4` | 1.8 | 42.3 |
| Dynamic, trained with `p=0.5` | 2.3 | 42.8 |

Thus the headline comparison is +0.7 average points over fixed Top-2 while
using about 90% as many expert activations. BBH used 1.87 experts on average
versus 1.76 across the five reported downstream tasks, and BBH improved by
more than two points over Top-2. Within two BBH task families, increasing the
number of objects from 3 to 7 increased mean active experts only slightly:
1.959→1.970 for shuffled-object tracking and 1.943→1.953 for logical
deduction. These results support a correlation with difficulty, but the size
of the within-task signal is modest
([§4–5 and Tables 3/5](https://aclanthology.org/2024.acl-long.696.pdf#page=6)).

The paper also reports that lower layers use more experts—up to four at the
lowest layer—while the top layer approaches one. The authors explicitly say
they do not claim that bottom-heavy allocation is generally optimal
([§6](https://aclanthology.org/2024.acl-long.696.pdf#page=8)).

### The post-hoc result is the decisive caveat for GLM

The final ACL appendix applies dynamic routing at inference to a model trained
with fixed Top-2:

| Inference on fixed-Top-2 checkpoint | Average active experts | Average score |
|---|---:|---:|
| fixed Top-2 baseline | 2.0 | 41.6 |
| dynamic `p=0.1` | 1.1 | 40.0 |
| dynamic `p=0.2` | 1.9 | 40.3 |
| dynamic `p=0.3` | 2.8 | 42.1 |
| dynamic `p=0.4` | 3.9 | 42.8 |

At `p=0.2`, saving only 0.1 expert per token costs 1.3 average score points.
The thresholds that outperform Top-2 activate 2.8–3.9 experts, so they improve
quality rather than efficiency. The paper concludes that direct dynamic
inference on a fixed-Top-K checkpoint “may not work”
([Appendix B.2 and Table 7](https://aclanthology.org/2024.acl-long.696.pdf#page=12)).

The paper’s wall-throughput evidence is also internally inconsistent. Its text
says Dynamic is about 5% faster during both training and inference, while
Table 4 reports 98.5K versus 93.9K training tokens/s (Dynamic is faster) but
0.19 versus 0.20 inference samples/s (Dynamic is slower). Those tests use
8×A800 for training and one A800 on BBH for inference; no token latency or
offloaded-expert I/O is reported
([Table 4](https://aclanthology.org/2024.acl-long.696.pdf#page=7)).

Other limits relevant to Insignia:

- The paper evaluates a 3.5B model with about 600M active parameters, not a
  288-expert GLM checkpoint streamed from NVMe.
- It trains the router for dynamic selection; GLM was trained for fixed Top-8
  `noaux_tc` routing and has no equivalent entropy objective.
- Downstream expert-count statistics concatenate each question with its gold
  answer, so they are not a clean measurement of online autoregressive prompt
  difficulty.
- It reports benchmark scores and parameter activation, not MSE, cosine,
  logit drift, PPL delta, DFlash acceptance, PCIe traffic, or SSD traffic.
- Parameter-count savings do not imply equal wall-time savings when a batched
  verifier stages the union of expert IDs across several rows.

## Exact mapping to GLM-5.3-Flash

Insignia’s current router is materially different from the paper’s router. For
expert \(e\), it computes

\[
s_e=\sigma(z_e),\qquad c_e=s_e+b_e,
\]

selects the eight largest corrected choices \(c_e\), but combines their expert
outputs using

\[
w_j=2.5\frac{s_{I_j}}{\sum_{r=1}^{8}s_{I_r}}.
\]

Selection order is therefore determined by corrected score \(c\), while
mixture mass is determined by uncorrected sigmoid score \(s\). This behavior is
implemented in `Runner::sparse_moe()` and `Runner::moe_multi()` in
`src/glm53_generate.cu`.

For a parity-aware first implementation, define

\[
q_j=\frac{w_j}{\sum_{r=1}^{8}w_r}=\frac{w_j}{2.5}
\]

and threshold cumulative \(q\) in the **existing corrected-choice order**.
This is not literally the paper’s descending-\(P\) Top-P, because GLM’s
selection and weighting signals differ, but it has three important properties:

1. it respects which experts the checkpoint’s correction bias intended to
   prioritize;
2. it preserves the existing expert accumulation order;
3. it can be decided before any expert record is requested.

Do not sort the selected eight by raw weight in the first version. That would
change which prefix survives and risks another routing-order sensitivity
failure. Also keep zero-fill semantics. It matches the paper’s unnormalized
tail deletion and Insignia’s measured renormalization result is worse.

### Causal I/O sequence

The current `moe_multi()` sequence already provides the needed seam:

1. run the 288-way router GEMV;
2. download router logits;
3. construct all eight exact selections and weights per row;
4. compute `exec_count[row]` from cumulative retained mass;
5. build the distinct expert union using only each row’s prefix;
6. call `stage_layer()` and execute those records.

Thus adaptive k can reduce expert reads *before* they begin. It does not need
an expert output, future token, target logit, or acceptance result. The actual
saved records are the reduction in the **union** across verify rows, not
`8 - mean(k)` by itself; this must be measured.

The current approximation uses one chunk-wide `exec_topk`. A per-row version
needs an `exec_count[tokens]` array and every union/user/retention/ordered-
accumulation loop must stop at `exec_count[token]`. The exact eight-entry
selection should remain available for diagnostics and routing traces. Scope it
behind `kda_archive_`, as the current approximate mode is, so scalar decode and
prompt prefill remain exact.

## What Insignia’s evidence already says

The earlier full-decode mass-pruning analysis remains a valid rejection for
the exact path. `audits/s10-router-mass-pruning.md` found:

- fixed Top-7 removes 12.5% of records but omits 7.6087% mean normalized mass;
- fixed Top-6 removes 25% but omits 15.8561% mean mass;
- even Top-6 had only a 45.135 ms/token transfer-channel upper bound under the
  measured host-hit/storage assumptions.

That conclusion does not automatically cover provisional DFlash verification.
The new verifier measurements expose a different cost surface:

- MATH prompt p12: fixed Top-4 zero-fill was 303.0 ms/token versus the faster
  bracketed exact run at 478.2 ms/token, a 1.58× speedup.
- MATH prompt p10: Top-3 was 213.3 versus exact 426.2 ms/token, 2.00×; Top-2
  reached 198.4 ms/token, only 7.5% faster than Top-3 despite much larger drift.
- On target-forced GSM8K p02 logits, Top-6 had mean cosine 0.994656, mean MSE
  0.1108, 31/32 top-1 agreement, and +3.25% PPL; Top-4 had mean cosine
  0.986254, mean MSE 0.2727, 31/32 top-1 agreement, and +5.41% PPL.
- On target-forced MATH p12 logits, Top-6 had mean cosine 0.988685, mean MSE
  0.1840, 31/32 top-1 agreement, and +1.02% PPL; Top-4 had mean cosine
  0.974096, mean MSE 0.4329, 31/32 top-1 agreement, and +8.47% PPL.

These are project measurements, not paper results. They show why an adaptive
mixture of Top-4/5/6/8 may dominate a single fixed `m`, but they do not yet show
that router mass predicts which rows need promotion.

There is a specific statistical hazard: the GSM8K contribution trace has
median exact cancellation around 2.22×. A low-weight expert can still matter
when expert vectors cancel, so retained router mass alone cannot guarantee low
routed-output error. That is why MSE/cosine—not a text “looks fine” check—must
choose the threshold.

## Minimal experiment that answers the question

No full ABCD campaign is needed yet.

### 1. Offline policy replay

Use the existing `INSIGNIA_GLM53_DF_MOE_METRICS` CSVs. For each unique
`(epoch, layer, row)`, choose the smallest measured `topm` crossing each
candidate threshold; synthesize `m=8` as exact when Top-7 does not cross.
Sweep, for example:

- `p`: 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95;
- `m_min`: 2, 3, 4;
- zero-fill only.

Report, separately for GSM8K and MATH:

- mean/p50/p90 `m` and fractions at each `m`;
- nominal selected-record reduction;
- routed-output MSE mean/p99;
- relative-L2 median/p99;
- cosine mean/p01/min;
- max-absolute-error p99;
- retained mass and norm ratio;
- layer-wise `m` distribution.

Tune on one prompt family and rank policies on the other. If all conservative
thresholds collapse to `m>=6`, the checkpoint’s router is too flat for this
idea to buy enough verifier speed. If a policy keeps Top-4-like traffic while
promoting only difficult rows, proceed.

### 2. Two or three target-forced logit checks

For only the offline Pareto frontier, run the already implemented forced-token
comparison and report full-vocabulary:

- logit cosine and MSE (mean/median/max);
- max logit delta;
- top-1 agreement and first mismatch;
- top-10 overlap;
- reference/approx PPL and relative PPL delta.

Suggested **project** gates—not claims from the paper—are:

- balanced: cosine ≥0.985, top-1 ≥95%, PPL increase ≤5%;
- aggressive: cosine ≥0.970, top-1 ≥90%, PPL increase ≤10%, provided measured
  wall speed is at least 1.5×.

### 3. Short wall-time A/B

Only after the quality frontier exists, bracket each candidate with exact k4
controls on one GSM8K and one MATH prompt. Add verifier cost tracing and report:

- ms/token and tokens/s;
- acceptance histogram and tokens/round;
- mean `m` and per-layer `m`;
- distinct expert union, staged records, cache hits, and bytes read;
- any PCIe/CUDA/clock instability.

## Decision

**Proceed with an offline adaptive-k replay and, if it produces a Pareto
frontier, a default-off per-row DFlash verifier knob. Do not apply it to exact
decode or cite the paper as evidence that post-hoc fixed-Top-8 pruning is
quality preserving.**

The paper supplies the right causal form and supports zero-fill semantics. Its
own appendix simultaneously explains why GLM-specific MSE, cosine, PPL, and
wall measurements—not the headline 0.7-point result—must decide whether the
controller survives.
