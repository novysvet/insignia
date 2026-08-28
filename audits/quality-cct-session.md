# Quality tooling, latent A/B, DFlash2 closure evidence, CCT repair (2026-08-28, parallel session)

Scope: the acceptance-regression discrimination chain that byte-level-closes
the session-5 "prompt artifact" verdict, the first latent-MLA validation past
position 256, the GSM8K pilot medians, published acceptance calibration, and
the CCT prefetch dead-on-arrival find + repair. All engine runs on glm-box
(32 GiB tier, Q8 10 GiB pin, fp8-fixed drafter cache). NOTE: another engine
session benchmarked concurrently for part of this window; paired-case ratios
are trustworthy, absolute ms/token are inflated.

## 1. DFlash2 acceptance: independent closure evidence

Session-5 closed the "collapse" as a prompt artifact (pre-bridge binary
reproduces 1.43/round on the oracle prompt). This session adds three
independent exonerations on the realistic `prompt_math.txt` prompt
(53 tokens), where acceptance measured 2.31/round k4 (5/13 empty) and
2.73/round k7:

1. **MLA path exonerated**: `INSIGNIA_GLM53_MLA_LEGACY=1` produces
   bit-identical acceptance histograms (k4 `0:5 1:1 2:1 3:2 4:4`,
   k7 `0:5 1:1 2:1 3:1 5:1 7:2`) and identical greedy IDs. The shadow
   bridge cannot have changed drafter behavior.
2. **Seq/batch verify latch exonerated**: DF_DEBUG 12-token trace shows the
   adaptive EMA never initializes (empty rounds don't update it), so every
   round ran BATCH verify — the new row-sequential path (verify_token,
   capture_offset_) was never exercised, yet rounds 0-1 were still empty.
3. **Drafter cache + numerics exonerated at byte level** (agent replay):
   - The glm-box-regenerated `glm53-dflash2-fp8-fixed` cache re-quantizes
     byte-identically from the BF16 checkpoint with
     `tools/quantize_dflash2.py::quantize_matrix` for ALL 48 tensors; dequant
     cos vs BF16 0.999585–1.000487; zero NaN codes.
   - A pure-BF16 NumPy oracle (no FP8 anywhere) replaying the DF_DUMP of the
     12-token run fails the SAME rounds from the SAME captures: truth0 ranks
     2nd/3rd with logit gaps 0.79–2.03 ("Let" vs "Step"/"**", "'s" vs " me",
     "\\n\\n" vs "**" — semantically reasonable alternates), and reproduces
     every accepted chain 9/9 exactly (incl. the 4-token "Let's work through
     this"). Also corrects the DF_DEBUG round reading: the last 12-token
     round accepted 2, it was not empty.

Combined with the healthy repetition-prompt numbers matching pre-bridge
digit-for-digit (k4 `0:1 1:1 2:1 4:24`, k7 `0:1 1:1 2:1 5:1 7:13`), the
drafter pipeline behaves exactly as designed.

## 2. Published acceptance calibration (DFlash arXiv:2602.06036, EAGLE-3, SPEED-Bench, specdecode-bench)

Metric: accepted drafts per round (ADPR) = τ − 1. Our 2.0–2.7 ADPR at ~35%
empty rounds on math CoT is the **EAGLE-3-small-tree floor (tree-16 ≈ 2.2
ADPR on GSM8K), low-normal, not broken**. Healthy for a 5-layer
feature-conditioned drafter: 2.5–3.5 ADPR; empty rounds 15–25%. Thinking
mode costs ~27% τ for SOTA drafters too. Ranked levers (published): capture
staleness/mismatch > conditioning pathway (ours already uses KV injection —
the +14–20% variant) > training data (target-regenerated CoT; fixed
checkpoint) > FP8 drafter weights (single-digit %, consistent with §1) >
verify-batch numerics (unquantified in literature). Block-8-budget peers
reach τ 4.2–4.8. Practical: k7 is squarely published practice; k4 is
conservative; DFlash trains at block 16 and generalizes DOWN cleanly.

## 3. GSM8K pilot (10 samples, k4, 32 gen, deterministic quantile spread)

Per-case parity 10/10 yes. Scalar median 545.0 ms/tok, DFlash2 median
612.0 ms/tok, aggregate median speedup **0.89x**; acceptance 2.00–3.20
(mean 2.64) per round. k4 does not beat scalar on real GSM8K at this
acceptance — the k7 decision needs the math500 half (killed mid-run by a
VM recycle; rerun pending). Absolute ms/token are contention-inflated;
per-case ratios stand. Reference: row 151 (119-tok prompt) hit 1.32x.

## 4. Latent MLA validation beyond position 256 (first numbers)

500-token GSM8K-derived prompt, 16 greedy tokens (positions ~484–516, all
latent territory), FP8-latent (default) vs FP32-latent (`KV_FP8=0`), scalar
decode, `INSIGNIA_GLM53_LOGITS_DUMP` + `tools/compare_logits.py`/`ppl.py`:

- greedy IDs identical 16/16; top-1 agreement 100%;
- logit cosine mean 0.9957 (median 0.9963), MSE mean 7.35e-2 (max 1.57e-1),
  top-10 overlap 9.1/10;
- PPL 1.4556 (FP8) vs 1.4136 (FP32) → the FP8 latent cache costs ~3.0% PPL
  with zero routing flips on this text.

Tooling landed in `c295638`: dump format = bare little-endian f32 records,
vocab 154880, one record per produce-logits step, no header. Also measured:
500-token prefill ≈ 178 s cold (~356 ms/token) — prefill is expert-I/O-bound;
the 8192-context benchmark plan must budget prefill separately.

## 5. CCT prefetch was dead on arrival — root causes and repair

The loader (`Runner::load_cct`) and the table builder (`tools/dump_cct.py`)
disagreed, so `INSIGNIA_GLM53_CCT` always printed "bad table header,
disabled":

1. Builder wrote magic `IGCCT1\0` + u32 pair-count + per-pair (u16,u16) tags;
   loader wants `CCT0` + u32 {layers, experts, topk} + positional tables.
2. Latent OOB: `cct_prefetch` indexed the successor table with
   `prev_routing_[layer][slot] == -1` (first `step()` warm-up before any
   routing exists) → `size_t(-1) * topk` wild row read → garbage expert ids.
3. `cct_prefetch` ran BEFORE `load_batch` (prefetch reads could start ahead
   of the current layer's demand) and was not gated by `INSIGNIA_GLM53_
   PREFETCH`.
4. No speculative-byte budget: union cap 16 at ~2.4x table overfetch costs
   more disk than it hides (bandwidth math in the CCT plan: ~510 ms/token of
   extra reads at cap 16 vs ~0.9 GiB/token of genuine misses at 80% hits).

Repairs landed (this session): builder emits the loader format with
zero-filled unobserved pairs; loader also validates `experts == 288`;
`cct_prefetch` guards `routed < 0 || routed >= cct_experts_` and caps the
union via `INSIGNIA_GLM53_CCT_MAX` (default 8); CCT issues AFTER
`load_batch` and only under `prefetch_on_`. Real-trace re-measurement
(train/test split, ROUTE_TRACE on GSM8K) is a prerequisite for enabling CCT
by default; the 73.7%/2.36x figures came from a repetition-loop trace.
A/B parity + perf runs pending GPU availability.

## 6. Operational

- glm-box cannot push to GitHub (no creds — pushes hang); commits land
  locally / via the bundle relay other sessions used.
- WSL VM recycled mid-run twice this window (SIGTERM on engine processes,
  harness died between cases); /var/tmp persists, /tmp does not.
- Parallel engine sessions contend for GPU + 2×32 GiB host pins:
  `pgrep -af glm53-generate` before benchmarking.
- `s6-bench.sh` (from another session) pointed at the raptor binary that was
  still the pre-bridge build; `/var/tmp/insignia-build-raptor/glm53-generate`
  has since been rebuilt from current HEAD.
