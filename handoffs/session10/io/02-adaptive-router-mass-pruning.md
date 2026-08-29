# Handoff 02 — adaptive router-mass pruning

## Objective

Falsify or justify an approximate mode that executes fewer than all eight
routed experts when the omitted routing mass is very small. This is the
largest remaining direct attack on expert bytes: dropping one expert in every
sparse layer removes 42 packed records/token, approximately 0.52 GiB before
cache effects.

The generic-PC phase is analysis only. It may recommend a tightly gated
runtime experiment, but it must not patch production code or claim quality
from routing scores alone.

## Repository and fresh-agent setup

- Repository: <https://github.com/novysvet/insignia.git>
- Branch: `glm53-dflash2-4070ti-super`
- Dispatch HEAD: `e48f633430c679ac6a30aae248159c887ac41601`
- Read from git: `AGENTS.md`, `progress.md`,
  `audits/s7-optimization-wave.md`, `audits/s8-gpu-expand-session.md`,
  `audits/s9-reclaim-session.md`, `src/glm53_generate.cu`, and
  `tools/glm53_route_analysis.py`.
- Make no production edit. H8 cross-head FP8 decode (`78e1a1c`) and fused
  H4×Q8 cross-head FP8 prefill (`e48f633`) are completed exclusions and must
  not be modified or reanalyzed.

The model has 42 sparse layers, 288 experts/layer, top 8 routing,
`norm_topk_prob`, and routed scaling 2.5. The trace stores the eight selected
expert IDs and their corresponding positive scores, but their storage order
must not be assumed to be descending.

## Verified performance anchors

- Baseline demand is 336 expert records/token if no record is resident.
- A packed-v2 expert averages about 12.765 MiB.
- Pinned H2D is 23.2 GB/s; single-NVMe bandwidth is 3.7–4.7 GB/s typical and
  5.84 GB/s best.
- Removing one routed expert from every layer removes about 536 MiB/token:
  approximately 24 ms of raw H2D and 92–138 ms of raw NVMe work if every
  removed record would otherwise miss. Cache residency and overlap reduce the
  wall benefit; model them explicitly.
- Routing changes cascade. Mathematically equivalent FP32 rewrites have
  changed future expert sets in this engine. This proposed mode is therefore
  approximate by construction, even if the first changed layer omits little
  router mass.
- The accepted FP8-latent quality reference incurred about +3% PPL with logit
  cosine 0.9957 and no greedy flip in a small 500-token test. That is a quality
  reference, not proof that expert pruning is equally safe.

## Required non-git artifact

Use `s9-campaign-handoff-20260829.tar.zst`:

- SHA-256:
  `819dcdb9e611f73a34c535292fed2a34da4b7fca5924cd6ca7bc69d817c94e56`
- Relevant paths after extraction:
  `var/lib/insignia/tracecampaign/merged/route-merged.trace` and
  `route-merged.manifest.tsv`.
- Schema: `token layer e0 e1 ... e7 s0 s1 ... s7`.
- Complete tokens contain exactly one row for every layer 3–44. Reject partial
  tokens, duplicate expert IDs, out-of-range IDs, non-finite scores, and
  malformed manifests.

The merged data contains legacy traces plus campaign/GSM8K/MATH-500 prompts.
Use manifest prompt boundaries. Legacy traces are diagnostics only and must
not drive the decision.

## Work

### 1. Establish score semantics

Report raw row-sum, minimum, maximum, rank-order, and concentration
distributions by layer and prompt. Independently verify whether normalizing
each row to sum one is appropriate for mass analysis. Keep raw scores for
runtime-policy reconstruction.

Sort only to choose candidates. Any future execution must retain the original
expert accumulation order among surviving experts.

### 2. Build the byte/quality-proxy frontier

Evaluate:

- fixed top-r, `r=1..8`;
- per-row minimum score thresholds;
- smallest r satisfying omitted normalized mass budgets of 0.1%, 0.25%,
  0.5%, 1%, 2%, and 5%;
- layer-specific thresholds trained without the held-out prompt;
- a robust per-token allocation of a total omission budget across 42 layers,
  with a per-layer cap so one layer cannot absorb the entire budget.

For every policy report mean, p50, p95, p99, and maximum omitted mass; records
retained/saved; prompt and family variation; and prompt-bootstrap intervals.
Compare two later runtime semantics without endorsing either: unchanged
survivor coefficients versus survivor renormalization. The latter perturbs
every retained contribution and is expected to be riskier.

Translate records saved into conservative traffic/time ranges under host-hit
rates 0%, 25%, 50%, and 80%, NVMe rates 3.7/4.7/5.84 GB/s, and measured H2D.
Do not add NVMe and H2D times when the production pipeline overlaps them;
provide serialized upper bounds and pipeline-aware estimates separately.

### 3. Design the later quality experiment

If the offline gate passes, produce an env-gated experiment blueprint and a
complete glm-box A/B matrix. It must measure at least:

- greedy IDs and full-vocabulary logit dumps;
- logit cosine, top-k overlap, and PPL on held-out text;
- route-set divergence after the first omitted expert;
- output repetition, NaN/Inf, empty-answer, and early-EOS degeneration;
- scalar and DFlash wall time, expert records, host/VRAM hits, and acceptance.

Define a conservative candidate gate (PPL regression <=5%, logit cosine
>=0.995) and a separately labelled aggressive gate allowed only for a major
speedup (PPL regression <=10%, logit cosine >=0.99, no catastrophic output
checks). These thresholds are hypotheses for measurement, not pre-approved
quality policy.

## Deliverables

- Strict parser and input-verification report.
- Reproducible analysis script with tests.
- `mass_frontier.csv`, `per_layer_thresholds.csv`, and
  `traffic_projection.csv`.
- Prompt/family bootstrap results and plots generated from the CSVs.
- `GLM_BOX_GATE.md` containing the later A/B protocol if warranted.
- `REPORT.md` with an explicit reject, conservative candidate, or aggressive
  candidate verdict.
- SHA-256 manifest for new outputs; no copied git files.

## Gates

Reject offline unless a held-out policy satisfies both:

- at least 15% mean expert-record reduction;
- mean omitted normalized mass <=1% and p99 omitted mass <=5% on both GSM8K
  and MATH-500 holdouts.

Reject any policy whose gain is driven by a single prompt, the legacy parrot
trace, survivor reordering, or post-hoc thresholds fitted on its test prompt.
No policy may be called successful until the later on-box quality and speed
gates both pass. A small quality loss earns acceptance only when the measured
speed gain is material; target at least 10% wall improvement.

## Forbidden duplication

This is not causal expert prediction, cache replacement, scale/body
compression, staged DFlash verification, adaptive draft length, CPU offload,
or an MLA change. In particular, H8 cross-head FP8 decode (`78e1a1c`) and
fused H4×Q8 cross-head FP8 prefill (`e48f633`) are complete. Do not reuse or
relabel rejected Task 8 or Task 9. Do not quantize expert weights below their
existing NVFP4 representation.
