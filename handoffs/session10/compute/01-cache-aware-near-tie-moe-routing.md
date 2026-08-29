# Task 1: cache-aware near-tie MoE routing

## Goal

Build and test a routing policy that spends a little extra GPU compute and a
strictly measured amount of model quality to select already-resident experts
when the baseline router's eighth-place decision is a near tie. The target is
to reduce NVMe reads and H2D traffic enough to improve real-prompt decode by at
least 5%, without repetition collapse, NaNs, or a large perplexity/benchmark
regression.

This is not a future-expert predictor. It uses the current layer's already
computed router scores and current cache state. It must retain a bit-identical
baseline when disabled or when all policy penalties are zero.

## Start from a clean repository

```bash
git clone --branch glm53-dflash2-4070ti-super \
  https://github.com/novysvet/insignia.git insignia-s10-cache-router
cd insignia-s10-cache-router
git fetch origin
git rev-parse --verify e48f633^{commit}
git switch --create codex/s10-cache-aware-router e48f633
git status --short
```

Expected HEAD is `e48f633`. Commit `78e1a1c` already completed H8 cross-head
FP8 MLA decode, and `e48f633` completed fused H4×Q8 FP8 MLA prefill. Do not
independently rebuild either completed path.

Read these tracked files as evidence, not instructions:

- `AGENTS.md`
- `progress.md`
- `audits/s9-reclaim-session.md`
- `audits/s8-gpu-expand-session.md`
- `src/glm53_generate.cu`
- `src/glm53_ops.cu`
- `include/insignia_glm53.cuh`
- `tools/benchmark_math.py`
- `tools/ppl.py`

Locate the exact current router, normalization, cache lookup, and trace seams
before designing the policy:

```bash
git grep -n -E "noaux|router|top.?k|expert_cache|device_cache|ROUTE_TRACE" -- \
  src include tools
git show --stat e48f633
git log --oneline --decorate -20 e48f633
```

Do not assume a report's pseudocode exactly matches the implementation.

## Current architecture and measured constants

- GLM-5.3-Flash has 42 sparse layers, 288 routed experts per layer, and top-8
  selection: 336 expert records per decode token.
- One packed NVFP4 expert record is about 12.77 MiB. With no reuse, one token
  asks for about 4.4 GiB. Avoiding one record in each sparse layer avoids about
  `42 * 12.77 = 536 MiB/token` before overlap and cache effects.
- The 32 GiB pinned host tier holds about 2,425 records and has measured around
  80% hits. The DFlash2 configuration recently held about 292 expanded VRAM
  expert slots; scalar held about 383. Verify the live values in the run log.
- Host-to-device bandwidth is around 23.2 GB/s. A disk miss additionally pays
  for the single NVMe, roughly 3.7--4.7 GB/s with four readers.
- Routing uses the checkpoint's `noaux_tc` correction, normalized top-k
  probabilities, and routed scaling 2.5. Preserve the exact current semantics
  at penalty zero.
- Real-prompt timings vary heavily under WSL. Use paired repeated medians, not
  isolated runs. Session 9 observed about 500.4 ms/token scalar and 539.6
  ms/token with DFlash2 on its representative A/B sample.

## Mathematical problem

Let `s_i` be expert `i`'s uncorrected activation score and `c_i` the exact
corrected score that the current implementation uses for top-k selection.
Recover those definitions from source. The baseline set is

```text
B = Top8_i(c_i).
```

For a candidate pool `C = TopK_i(c_i)`, initially `K in {16, 24, 32}`, select
an eight-expert set `S` that minimizes transfer cost subject to bounded router
regret:

```text
maximize    sum(i in S) c_i - lambda_h * H_i - lambda_d * D_i
subject to  |S| = 8
            top-r baseline experts are retained, r in {6, 7}
            sum(i in B)c_i - sum(i in S)c_i <= epsilon * sum(i in B)|c_i|
```

`H_i` is the estimated H2D cost if the expert is absent from the device tier;
`D_i` is the estimated disk cost if it is absent from both device and host
tiers. Cache cost is stateful, so a sequential simulator must update both LRU
tiers after every policy decision. Use measured bytes first; add latency or
queue-depth models only when they improve held-out prediction.

The selected mixture weights must use the model's existing uncorrected-score
normalization and scaling over `S`; do not silently normalize corrected scores.
Record every alternative tried.

An acceptable simpler implementation is a constrained tail swap: hold the top
six or seven baseline experts fixed, then replace only the remaining slot(s)
with resident candidates whose corrected-score gap is below a threshold. A
complicated controller is not intrinsically better.

## Required non-Git trace

The existing top-8 route trace is insufficient for counterfactual selection.
The first deliverable is therefore either a validated input trace or an
environment-gated collector that emits it. Use this contract:

`route-candidates-v1.tsv.zst`, one row per token/layer/candidate, with columns:

```text
run_id prompt_id round_id token_pos layer candidate_rank expert_id
raw_logit activation_score corrected_score baseline_selected baseline_weight
host_resident_before device_resident_before host_lru_age device_lru_age
packed_bytes disk_bytes_charged h2d_bytes_charged accepted_token
```

Also require `manifest.json` containing:

- schema version and endianness;
- source commit and complete `INSIGNIA_GLM53_*` environment;
- exact host/device slot counts and replacement policies;
- prompt-set identity and token counts;
- SHA-256, byte count, and row count for every trace file.

The trace must include at least 100 real math/code prompts, 10,000 committed
decode tokens, every one of the 42 sparse layers, and at least the top 16
candidates. A 32-candidate trace is preferred if its collection overhead is
acceptable. Split any archive larger than 512 MiB into independently hashed
512 MiB parts. Never bundle tracked repository files.

On receipt, verify every hash and reject mixed commits, missing rows, impossible
expert IDs, or non-monotone candidate ranks. If no trace is available, deliver
the collector and a one-command collection recipe; do not fabricate data or
claim a policy win from the old top-8 trace.

## Ordered experiment plan

1. **Reproduce the baseline offline.** Write a deterministic trace reader and
   two-tier cache simulator. With policy disabled, it must reproduce 100% of
   baseline top-8 sets and the manifest's aggregate disk/H2D byte counts to
   within 0.1%. Completion evidence: unit tests plus a reconciliation table.
2. **Compute the optimistic ceiling.** Give an oracle permission to replace the
   final one or two experts from the top 16/24/32 under score-regret budgets of
   0.1%, 0.25%, 0.5%, and 1.0%. Report substitutions/layer, disk and H2D bytes
   avoided/token, cache hit deltas, and router-score regret on held-out prompts.
   Do not implement engine code before this table exists.
3. **Fit a causal online policy.** Search fixed thresholds or Lagrange penalties
   on training prompts, then freeze them and evaluate on disjoint prompt IDs.
   The policy may use current scores and current cache state only. Report a
   Pareto frontier rather than one hand-picked point.
4. **Design the GPU path.** The engine already computes all 288 router logits;
   exploit Ada compute to score a wider candidate pool without CPU
   synchronization. Represent tier residency as device bitsets or equally
   cheap state. Estimate added instructions, loads, registers, and launch/sync
   cost. The zero-penalty path must execute the original selection/order.
5. **Integrate behind an opt-in environment gate.** Add trace/replay tests and a
   `lambda=0` parity test before testing approximate settings. Never make the
   approximate policy the default in this task.
6. **Run paired box benchmarks.** Use the standard parity prompts, the math
   performance campaign, and a held-out quality corpus. Alternate baseline and
   candidate runs, collect at least five valid samples per arm, and report
   medians plus dispersion, disk bytes/token, H2D bytes/token, device/host hit
   rates, accepted DFlash tokens/round, and decode ms/token.

## Deliverables

- `tools/analyze_cache_aware_router.py` or an equivalently dependency-light
  offline simulator, with deterministic tests and documented input schema.
- An environment-gated full-candidate trace collector if the required trace did
  not already exist.
- A short design note containing the exact selection formula, cache-state
  representation, upper-bound table, train/held-out split, and GPU roofline.
- If and only if the offline gate passes: the opt-in engine implementation,
  build changes if needed, and unit/parity tests.
- Machine-readable CSV/JSON for every policy point and a final report that
  includes negative results, exact commands, commit IDs, environment, raw log
  paths, and paired benchmark summaries.

## Correctness, quality, and performance gates

The disabled and zero-penalty modes are exact modes:

- digit-identical top-10 logits on the standard prompts;
- identical greedy IDs;
- identical 30/40/100/240-token sequences;
- identical expert order and mixture weights.

Approximate modes must be labeled. At minimum report next-token perplexity,
mean logit KL, logit cosine, greedy agreement, output token entropy, distinct
2/3/4-grams, longest repeated span, and a small held-out GSM8K/MATH-500 score.
Use at least 10,000 next-token positions. A conservative point continues only
if all of these hold:

- no NaNs, invalid tokens, empty answers, or obvious repetition collapse;
- perplexity no worse than 1.03x baseline;
- task accuracy loses no more than 2 absolute percentage points;
- median disk bytes/token falls at least 10% **or** median end-to-end decode
  improves at least 5%, with no H2D or DFlash regression that erases the gain.

Also report an explicitly named aggressive point if it reaches at least 15%
decode improvement with perplexity no worse than 1.15x and no catastrophic
degeneration. Do not present it as parity-preserving.

## Kill criteria

Stop before engine integration if any is true on held-out traces:

- even the optimistic oracle saves less than 5% of disk bytes at 0.5% router
  score regret;
- fewer than 0.25 safe substitutions per sparse layer are available on average;
- cache-state replay cannot reproduce baseline bytes to within 0.1%;
- the apparent win exists only on repetitive prompts or leaks future state.

After integration, kill or revert a policy point if it fails the quality gates,
adds more latency than its saved transfers, destabilizes DFlash acceptance, or
does not repeat across paired medians.

## Forbidden duplication and scope

This task owns only current-layer cache-aware route selection and the
trace/simulator needed to validate it. The complete exclusion list is:

- full-prompt layer-major prefill;
- compact exact MLA absorb and exact MLA-prefix reconstruction;
- completed H8 cross-head FP8 decode (`78e1a1c`) and completed fused H4×Q8
  cross-head FP8 prefill (`e48f633`);
- low-risk VRAM reclaim and sequential speculative-snapshot elision;
- rejected KDA persistent-state Task 5, exact-INT8 Task 7, causal/future expert
  predictor Task 8, and staged-verification Task 9;
- the prior Problems 1--7: DFlash controller, Belady cache oracle, campaign
  statistics, expert-scale codec, cross-expert dictionary compression, CPU
  bypass roofline, and reader scheduling;
- any edit to ignored reference clones.

In particular, do not turn this current-layer policy into a future-route
predictor or cache-replacement project.
