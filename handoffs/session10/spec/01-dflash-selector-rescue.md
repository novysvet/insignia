# Task 1: DFlash truth-seeded rescue and global selector search

## Mission

Find a better **single-path DFlash2 selector** using only information already
available before target verification. The primary experiment is an
empty-round rescue: when the drafter's first token differs from the exact
`truth0` already carried from the preceding target step, force the first
candidate to `truth0`, continue the learned selector from that predecessor,
and decide whether the remaining suffix is worth verifying. No extra drafter
forward is required. Target verification remains authoritative, so committed
tokens can remain greedy-exact.

Also test global Viterbi/beam decoding over the existing seven-position,
top-16 selector lattice. The shipping selector greedily maximizes each local
unary-plus-bilinear score; it need not maximize total path score.

## Checkout and authority

- Clone <https://github.com/novysvet/insignia.git> and check out branch
  `glm53-dflash2-4070ti-super` at committed/pushed HEAD `e48f633` in a
  separate directory.
- Read `AGENTS.md`, `progress.md`, `audits/dflash2-session.md`,
  `audits/dflash2-fixes-session.md`, `audits/quality-cct-session.md`,
  `audits/seqverify-session.md`, and `audits/s9-reclaim-session.md`.
- Use source and committed data as truth. Attachment/report claims are leads
  only.
- Completed exclusions at this base: `78e1a1c` H8 cross-head FP8 MLA decode
  and `e48f633` fused H4 x Q8 cross-head FP8 MLA prefill. Work in
  `scratch/session10-selector/` or `tools/` and do not duplicate either.

## Architecture and current behavior

The drafter runs one block `[anchor, mask x 7]`. For each of the seven mask
positions it produces target-lm-head logits and a 256-wide hidden projection.
The host selector takes the unary top-16 and scores candidate `c` after
predecessor `p` as

```text
score_t(p,c) = unary_t(c)
             + sum_{i=0}^{255} (A[p,i] * hp[t,i]) * B[c,i]
```

using ascending `i` and FP32 `fmaf`. It greedily chooses one token, then uses
that token as the next predecessor. The target already knows `truth0`, its
exact next-token argmax, before calling the drafter.

Source anchors at pinned base `e48f633`:

- `include/insignia_glm53_dflash2.cuh`: constants and `select` contract.
- `src/glm53_dflash2.cu`, `DFlash2Drafter::select` (near line 633): top-16
  scan and greedy bilinear walk.
- `src/glm53_generate.cu`, `Runner::df_draft` (line 3986): logits/HP
  download and selector call.
- `src/glm53_generate.cu`, DFlash main loop (near lines 5340-5645): `truth0`,
  empty short-circuit, batch/sequential verification, rollback, and commit.
- `tools/dflash2_oracle.py`: independent drafter replay and dump parser.
- `scratch/accept/out/position_acceptance_profile.csv`,
  `scratch/accept/out/calibration.csv`, and
  `scratch/accept/out/per_prompt.csv`: committed acceptance/cost baselines.

Real-text baseline: first-position survival is about 0.758 for GSM8K k4 and
0.711 for pooled GSM8K/MATH-500 k7. An empty round wastes the roughly 17.5 ms
draft then pays a roughly 643 ms scalar transition. k4/k7 verification costs
about 1,896/2,781 ms, so blindly rescuing every empty round at full width can
be worse. The result must include a cost-aware suffix-width/gating policy.

## Required lattice export

The committed aggregate CSVs are enough for cost modeling but not selector
counterfactuals. Use or request one diagnostic-only export; do not bundle the
2.34 GiB drafter checkpoint. The exporter runs where the checkpoint exists
and writes `selector-v1.npz` plus `manifest.json` and `SHA256SUMS`.

Required arrays:

| Name | dtype/shape | Meaning |
|---|---|---|
| `round_id` | `int64 [R]` | Stable round id. |
| `prompt_id` | `int32 [R]` | Group split key; never split rounds from one prompt across train/test. |
| `dataset_id` | `uint8 [R]` | 0=GSM8K, 1=MATH-500, 2=other. |
| `position` | `int32 [R]` | Absolute anchor position. |
| `anchor` | `int32 [R]` | Current committed token. |
| `truth` | `int32 [R,7]` | Exact scalar-greedy continuation from this anchor; `-1` when unavailable. |
| `candidate` | `int32 [R,7,16]` | Unary top-16 token ids, descending unary order. |
| `unary` | `float32 [R,7,16]` | Corresponding logits. |
| `edge` | `float32 [R,7,16,16]` | For `t>0`, bilinear term from each position `t-1` candidate to each position `t` candidate; t=0 unused/NaN. |
| `anchor_edge` | `float32 [R,16]` | Bilinear term from `anchor` to the t=0 candidates. |
| `truth0_edge` | `float32 [R,16]` | Bilinear term from exact `truth0` to the t=1 candidates, even if truth0 was outside t=0 top-16. |
| `shipping_path` | `int32 [R,7]` | Current greedy selector output, used to validate the export. |

Compute edge terms in the existing ascending-256 `fmaf` order. Include the
exact git commit, engine command/env, prompt row ids, array shapes, and byte
sizes in `manifest.json`. Generate hashes with SHA-256 after closing the file.
Keep every archive part below 512 MiB if a larger campaign is exported.

The exporter must pass this sample check before handoff:

```python
import numpy as np
d = np.load("selector-v1.npz")
assert d["candidate"].shape[1:] == (7, 16)
assert d["edge"].shape[1:] == (7, 16, 16)
assert np.all(d["shipping_path"][:, 0] >= 0)
```

The downstream agent must verify `SHA256SUMS` and reproduce
`shipping_path` exactly from `unary + edge` before trusting the artifact.

## Experiments

1. **Red loop:** write a small stdlib/NumPy evaluator that exactly reproduces
   the shipping greedy path and its accepted-prefix length against `truth`.
   Completion: every exported round matches `shipping_path` and the recorded
   acceptance histogram.
2. **Truth-seeded rescue:** for rounds where shipping t0 != truth0, set path0
   to truth0 and select t1 using `truth0_edge`; continue greedily thereafter.
   Measure the conditional suffix survival for widths 2-7.
3. **Global search:** evaluate exact Viterbi, beam widths 2/4/8/16, and a
   minimal calibration grid for unary weight, edge weight, and length
   normalization. Do not invent a neural predictor.
4. **Honest splits:** group by prompt and hold out both prompt ids and at
   least one dataset family. Report bootstrap confidence intervals. A policy
   selected on the campaign/repetition prompt is invalid.
5. **Cost policy:** choose fallback versus rescued verification width using
   only pre-verification features: truth0 unary rank/gap, forced-path score
   margins, entropy, position, and the committed k4/k7 cost table. Include
   the opportunity cost of 1,067/1,506 distinct records.
6. **Sensitivity:** sweep effective record cost from the 0.613 ms PCIe floor
   through 1.8 ms production and 3.88 ms cold-NVMe bound. Report where the
   policy changes.
7. Only after offline gates pass, provide a small default-off patch plan and
   exact GPU A/B command matrix. Keep the selector ABI narrow; a handful of
   loops is sufficient.

## Deliverables

- `scratch/session10-selector/analyze_selector.py` with a self-check and no
  dependencies beyond NumPy.
- `scratch/session10-selector/RESULTS.md` containing per-dataset confusion,
  survival curves, chosen paths, bootstrap intervals, and speed projections.
- `scratch/session10-selector/policy.csv`: one row per policy/width/cost point.
- `scratch/session10-selector/selector.patch` only if the held-out gate passes;
  keep it default-off until live validation.
- A one-page operator runbook for producing/verifying `selector-v1.npz` and
  running paired scalar/shipping/rescue tests.

## Gates and kill criteria

The offline proposal passes only if all hold:

- Shipping-path replay is exact on 100% of exported rounds.
- On held-out real prompts, projected median ms/token improves at least 5%
  versus the shipping DFlash policy at the same effective record cost.
- No held-out dataset loses more than 0.10 accepted drafts/round.
- Estimated selector overhead is below 1 ms/round on the i7-14700KF.
- The final target verifier remains authoritative; live greedy ids and top-10
  logits must therefore remain digit-identical.

Kill truth-seeded rescue if its conditional suffix survival cannot repay the
extra verify records at 1.8 ms/record. Kill Viterbi/beam if it adds less than
0.15 accepted drafts/round or regresses either held-out family. Report a
negative result instead of weakening the split or tuning on the campaign
prompt.

## Forbidden duplication

Do not work on adaptive-k v2, the rejected causal expert predictor, staged
verification, sequential snapshot elision, DFlash FP8 quantization,
`78e1a1c` H8 cross-head decode, `e48f633` fused H4 x Q8 cross-head prefill,
layer-major prefill, or drafter `lm_head` batching (it was measured 3-4x
slower and reverted).
