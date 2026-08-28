# Row-sequential verify + lm_head batching audit (2026-08-28)

Scope: DFlash2 verify loop restructure in `src/glm53_generate.cu`, plus a
measured micro-optimization attempt on the draft's lm_head. All benches on the
16-token math prompt, k4, striped store
(`INSIGNIA_GLM53_ALT_SHARD_DIR=/stripe`, tier 5120, `glm53-flash-text-striped.index`).

## 1. Row-sequential verify

### Motivation

Batched `verify_round` (one k-row prefill pass over the whole candidate block)
reads the union of all k rows' routed experts before any acceptance decision
is known. When a round rejects early, the experts for the rejected tail were
read for nothing. Row-sequential verify forwards one candidate at a time and
stops at the first acceptance failure, so the rejected tail's experts are
never read.

### Implementation

- New `Runner::verify_token(token, position)`: single-row prefill + argmax.
- `prefill` now honors two Runner fields set by the caller:
  - `verify_may_rollback_` — when false, skips the two whole-state snapshot
    copies (KDA states + conv history `cudaMemcpyAsync`). Row-sequential
    verify never rolls back because the recurrent state always stands at the
    accepted boundary (rejected rows were never forwarded).
  - `capture_offset_` — drafter capture slot for `df_->capture_row`, so
    per-row captures land in consecutive slots and `df_commit(matched, pos)`
    still commits the accepted prefix contiguously.
- Main loop: sequential loop per round when selected; batch path preserved.
- **Mode selection is adaptive by default**: a rolling acceptance EMA
  (`accept_ema = 0.75*ema + 0.25*matched`, seeded with the first verified
  round) picks sequential when `accept_ema < 0.70 * verify_k`. Env overrides:
  `INSIGNIA_GLM53_DF_SEQ_VERIFY=1` forces sequential,
  `INSIGNIA_GLM53_DF_BATCH_VERIFY=1` forces batch.

### Correctness

- Sequential and batch verify produce **digit-identical greedy IDs** on the
  math prompt (40 tokens, k4): `220 16 13 16 13 ...` in both modes.
- Acceptance histograms identical: `0:1 1:1 2:1 4:9` both modes.
- No rollback needed in sequential mode (state is always at the accepted
  boundary); `rollback_kda` retained for the MTP path and batch-verify mode.

### Measured (math prompt, 100 tok, k4)

| mode | ms/token | verify ms/verified round | accepted/round | histogram |
|------|----------|--------------------------|----------------|-----------|
| sequential (forced) | 690.3 | 2602.6 | 3.70 | 0:1 1:1 2:1 4:24 |
| batch | 401.4 | 1485.1 | 3.70 | 0:1 1:1 2:1 4:24 |

**Sequential loses on high-acceptance text.** When nearly every round accepts
all k drafts (histogram `4:24` here), row-sequential verify pays k separate
full-stack passes, each forcing its own expert-union read with no cross-row
amortization — the exact opposite of the win it was designed for. Verify time
per verified round grew 1.75x.

**The win case is low acceptance**: with acceptance 2.75/7 (real text per
earlier sessions), a round that rejects at row 1 skips ~6 rows' worth of
expert I/O. The 0.70·k EMA threshold targets that regime; the adaptive default
chose batch on the math prompt (acceptance 0.925) and matched the batch
number within noise (501.5 vs 451.6 ms/tok at 40 tok — residual gap is first-
round EMA warm-up and run-to-run variance, not mode cost: by the time the EMA
is warm every round is batch).

### Traps encountered

- `std::vector<int> arg(size_t(verify_k));` inside the loop = **most vexing
  parse** (declared a function). nvcc reported "expression must be a pointer
  to a complete object type" at `arg[r]`. Fixed with
  `std::vector<int> arg(static_cast<size_t>(verify_k), 0);`.
- Member placement: `verify_may_rollback_`/`capture_offset_` must be public
  (set by the main decode loop); `verify_token` needs a class declaration.

### What would make sequential strictly better

Per-row routing overlap: forward row r+1's expert reads while row r's GEMV
runs, so the sequential path keeps its skip-the-tail win without paying
serial full-stack cost per row. Not implemented (requires speculative
prefetch of the next row's union before acceptance is known — a measured
step, not done here).

## 2. lm_head draft batching — attempted, measured slower, reverted

**Hypothesis** (from prior session notes): `df_draft` calls
`linear("lm_head.weight", ...)` 7 times, each re-reading ~634 MB → batch via
`linear_multi` (which stages each FP8 weight chunk once and multiplies all
token rows).

**Measured**: draft time went from ~30.3 ms/round to 97–119 ms/round (40 tok,
k4). **3–4x slower. Reverted.**

**Root cause**: lm_head FP8 (~154k × 2048 weights + scales ≈ 330 MB) exceeds
`Q8Stager::kWeightCapacity` (128 MiB), so it cannot be VRAM-resident through
the Q8 path — `linear_multi` chunks it row-wise and re-streams **every chunk
from the FP8 host file over H2D every round** (~330 MB/round ≈ +60–90 ms at
the observed draft delta). The plain `linear()` path instead finds the tensor
in the **BF16 stager residency** (`stager_.load`), which keeps the whole
tensor in VRAM after first touch, making the 7 "re-reads" VRAM-local
(microseconds). The 7-call loop is already optimal under the current stager
split; the prior session's +8.5 ms/round estimate confused the BF16 re-read
(VRAM-cheap) with an NVMe re-read (it isn't).

**Lesson / stated rule**: before "fixing" a redundant read, check which
stager serves it. Redundant reads from a VRAM-resident tensor are free;
"amortizing" them through a different stager with a smaller residency slot
can convert VRAM hits into NVMe streams.

## 3. Numbers for the record (post-change baseline, 2026-08-28)

Math prompt, striped store, k4, tier 5120:

- 100 tok, batch verify (adaptive default after EMA warm): **401.4 ms/token**,
  3.70 accepted/round, draft 29.9 ms/round, fallback ~700 ms.
- 40 tok runs: 451.6–501.5 ms/token (first-round EMA + variance).

Compare pre-change reference points (same prompt, earlier sessions, unstirped
store / k4 / tier 5120): 843.8 ms/tok DFlash2-k4, 809 ms/tok plain. The striping
+ pools work from the intervening sessions accounts for the bulk of the
843.8 → 401.4 improvement; today's verify restructure is neutral-to-positive
on this prompt (adaptive picks batch) and preserves the low-acceptance win
path for real text.

### Remaining known costs

- `df_draft`'s 7 lm_head GEMVs: fine (VRAM-resident), no action.
- Empty-round fallback (`candidates[0] != truth0`) costs ~700 ms on the math
  prompt (a full target step after a wasted draft round); 1 empty round per
  12 rounds here, higher on real text.
- Cross-layer expert prediction (CCT / pre-attention predictor) is still the
  big unrealized win: per-layer union issue serialization caps decode I/O at
  ~6 GB/s vs the ~8.5 GB/s the striped pair can stream.

### Env surface added today

- `INSIGNIA_GLM53_DF_SEQ_VERIFY=1` — force row-sequential verify
- `INSIGNIA_GLM53_DF_BATCH_VERIFY=1` — force batched verify
- (default: adaptive by acceptance EMA, threshold 0.70·k)
- `build/bench-df-ab.sh` — A/B harness (seq vs batch, same prompt twice)
