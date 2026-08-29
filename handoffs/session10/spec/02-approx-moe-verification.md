# Task 2: Quality-constrained approximate MoE verification

## Mission

Determine whether DFlash target verification can gain a large bandwidth win
by executing fewer than eight routed experts for provisional verify rows.
This task intentionally relaxes bit-exactness. It must produce an honest
speed/quality Pareto frontier and a hard falsifier before any production
integration.

The simplest candidates are:

1. fixed top-m, `m in {2,3,4,5,6,7}`;
2. smallest m whose cumulative original top-8 route weight exceeds a threshold;
3. a per-layer threshold selected from held-out contribution error.

Keep the shared expert exact. Compare two accumulation semantics: omitted
experts contribute zero while retained weights stay unchanged, and retained
weights are renormalized to the original routed sum. Preserve retained expert
slot order. Do not add a learned expert predictor or cross-expert dictionary.

## Checkout and authority

- Repository: <https://github.com/novysvet/insignia.git>
- Branch and required committed/pushed HEAD: `glm53-dflash2-4070ti-super` at
  `e48f633`.
- Completed exclusions at this base: `78e1a1c` H8 cross-head FP8 MLA decode
  and `e48f633` fused H4 x Q8 cross-head FP8 MLA prefill. Work in a fresh
  clone and do not duplicate either.
- Read `AGENTS.md`, `progress.md`, `audits/s7-optimization-wave.md`,
  `audits/s8-gpu-expand-session.md`, `audits/s9-reclaim-session.md`,
  `audits/seqverify-session.md`, and `scratch/accept/analysis_digest.txt`.
- Source and committed measurements are authoritative; attachments are not.

## Why this may matter

GLM-5.3-Flash has 42 sparse layers, 288 experts/layer, top-8 routing, routed
scale 2.5, and a dense shared expert. One token can demand 336 routed records.
Each packed record is about 13.56 MiB. The production box has one NVMe and a
32 GiB pinned tier; verification remains dominated by expert transport rather
than expert arithmetic.

Committed real-text calibration:

- k4 verification: about 1,067 distinct records and 1,896 ms/verified round;
- k7 verification: about 1,506 distinct records and 2,781 ms/verified round;
- effective record cost: about 1.8 ms, with 0.613 ms PCIe floor and roughly
  2.93-3.88 ms cold-NVMe range;
- expert GPU work was modeled below 10% of chunk wall, which killed the exact
  INT8 kernel task but makes byte elimination interesting;
- mathematically equivalent FP32 rewrites have flipped routers. Approximate
  expert omission may cascade much more strongly.

Source anchors:

- `src/glm53_generate.cu`, `Runner::moe_multi` at line 4397: top-8
  selection, normalized weights, distinct union, retained verify results, and
  ordered accumulation.
- `src/glm53_generate.cu`, DFlash main loop near lines 5340-5645: verify
  batch, rollback, accepted-prefix commit, and quality-preserving baseline.
- `src/glm53_expert_bench.cu`: packed expert body geometry and kernels.
- `tools/compare_logits.py`, `tools/ppl.py`, and
  `tools/benchmark_math.py`: existing quality/performance tooling.
- `scratch/tracecampaign/TRACE-FORMAT.md`: routed expert trace format.

## Required contribution export

Local contribution error can be studied on any CPU if an operator exports
the eight exact down-projection outputs from real verify/scalar rows. The
export hook is diagnostic-only and must not change arithmetic or launch order.

Use chunk directories named `moe-contrib-v1-0000`, etc. Each chunk contains:

| File | dtype/shape | Meaning |
|---|---|---|
| `meta.npy` | `int32 [N,5]` | prompt id, dataset id, token position, layer, row-in-verify-batch. |
| `expert_id.npy` | `int16 [N,8]` | router top-8 in original accumulation order. |
| `route_weight.npy` | `float32 [N,8]` | exact `2.5*sigmoid(score)/sum(top8)` weights. |
| `expert_out.npy` | `float32 [N,8,4096]` | unweighted down-projection output before `scale_add`, slot ordered. |
| `exact_routed.npy` | `float32 [N,4096]` | shipping ordered-FMA routed sum for round-trip validation. |
| `manifest.json` | UTF-8 JSON | git commit, command/env, prompts, shapes, byte counts, endianness, and whether rows came from scalar or verify. |
| `SHA256SUMS` | text | SHA-256 for every file in the chunk. |

Little-endian NumPy v1/v2 files are the format contract. Limit each archive
part to 512 MiB; with float32 expert outputs, about 80 complete tokens x 42
sparse layers per part is a safe ceiling. Sample at least 10 GSM8K and 10
MATH-500 prompts with
prompt-group ids, spanning short/long prompts and easy/hard acceptance. Never
select only high-acceptance rounds.

Before shipping the artifact, replay all eight slots on-device with the same
ascending-slot `fmaf` kernel and require zero mismatches with `exact_routed`.
The receiving CPU agent verifies every hash, repeats the reconstruction with
IEEE `fmaf`, and reports exact-match rate plus max absolute/ULP error; isolate
subnormal/FTZ differences instead of silently accepting them.

The route-only traffic study may additionally consume the previously exported
`route-merged.trace`; it is not in git. If supplied, include its original
`route-merged.manifest.tsv`, SHA-256, line count, and a 100-line sample. The
line contract is documented in committed
`scratch/tracecampaign/TRACE-FORMAT.md`. Do not trust an unlabeled trace.

## Staged experiments

1. **Exact replay:** reproduce `exact_routed` from all eight expert outputs.
   This is the red loop and must pass before approximation.
2. **Local Pareto:** for every m and cumulative-mass threshold, compute
   relative L2, cosine, max error, direction/magnitude change, cancellation
   ratio, and retained route mass. Report per layer and prompt family, not
   only pooled means.
3. **Traffic Pareto:** recompute each verify batch's distinct expert union
   after truncation. Translate avoided records to wall time across the full
   0.613-3.88 ms/record range. Include shared/dense fixed costs.
4. **Risk rule:** fit only a transparent threshold/table using training
   prompts; evaluate prompt-group and cross-family holdouts. Features may use
   route weights, layer id, m, and local contribution norms available before
   omission. No neural model.
5. **Cascade bound:** estimate perturbation amplification using committed MLA
   audit observations and adjacent-layer contribution errors. Clearly label
   this as a bound, not an end-to-end quality result.
6. If and only if a useful offline frontier survives, prepare a minimal
   default-off experimental patch (`INSIGNIA_GLM53_DF_APPROX_TOPM` and/or a
   mass threshold). Preserve the exact path byte-for-byte when unset.
7. Hand the patch to a box operator for paired target-forced logit/PPL runs
   and free-generation smoke tests at 30/40/100/240 tokens. Use at least five
   GSM8K and five MATH-500 prompts; counterbalance run order and report paired
   medians.

## Deliverables

- `scratch/session10-approx-verify/analyze_contrib.py`, NumPy only, with a
  synthetic self-test for accumulation and union counting.
- `scratch/session10-approx-verify/FRONTIER.csv`: one row per policy, dataset,
  and record-cost point.
- `scratch/session10-approx-verify/RESULTS.md`: local errors, held-out results,
  traffic savings, cascade caveats, and a ranked live-test matrix.
- Optional `scratch/session10-approx-verify/approx-verify.patch` only after
  the offline gate; never make it default-on.
- An operator runbook covering artifact creation, SHA verification, build,
  paired tests, logit/PPL comparison, and immediate rollback to exact mode.

## Quality and speed gates

Report two explicit envelopes:

**Conservative candidate**

- at least 20% fewer distinct routed records in held-out verify batches;
- offline routed-output cosine median >=0.999 and first percentile >=0.99;
- live full-vocab logit cosine >=0.995, top-1 agreement >=99%, top-10 overlap
  >=9/10, and relative PPL increase <=3%;
- paired median wall improvement >=15% over exact DFlash.

**Extreme candidate**

- at least 40% fewer distinct routed records and projected/live speedup
  >=1.5x over exact DFlash;
- live top-1 agreement >=95%, top-10 overlap >=8/10, relative PPL increase
  <=10%, no NaN/Inf, no immediate EOS collapse, and no severe repetition or
  incoherent-output collapse on 100/240-token samples.

These are characterization gates, not permission to replace the exact
default. Publish representative changed outputs for the extreme arm.

Kill the task before runtime integration if top-6 cannot remove 15% of the
held-out union, if top-4 routed-output cosine first percentile is below 0.95,
or if savings exist only on the repetition campaign prompt. Kill each live
arm on CUDA errors, non-finite tensors, collapse within 16 tokens, PPL beyond
its envelope, or less than 10% measured wall gain.

## Forbidden duplication

Do not implement the rejected exact INT8 wide-row kernel, causal expert
prediction, staged verification, cross-expert dictionaries, new weight
quantization, layer-major prefill, compact MLA absorb, `78e1a1c` H8
cross-head decode, `e48f633` fused H4 x Q8 cross-head prefill, or VRAM
reclaim. This task changes only the number of already-selected routed experts
executed during experimental verification.
