# Optimization-wave A/B bench matrix (GLM-5.3-Flash, glm-box)

Serialized, resumable, Task-Scheduler-survivable harness. Files (this dir):

| file | role |
|---|---|
| `bench-matrix.sh` | the runner — runs INSIDE Arch WSL on glm-box, serialized cells, pgrep guard, per-run logs, resume by DONE marker, summary CSV |
| `bench-matrix-inner.sh` + `bench-matrix-task.cmd` | Windows Task Scheduler wrapper (pattern of `build/s6-task.cmd` + `build/s6-inner.sh`) |
| `deploy-matrix.sh` | dev-box helper: push → pull on glm-box → WSL build → stage harness → register scheduled task |
| `summarize-matrix.py` | medians + IQR + parity/acceptance columns -> `summary.csv` |

Everything lands under `/var/lib/insignia/bench-results/<date>-matrix/`
(survives WSL recycles; `/tmp` does not).

## 0. Preconditions (before any cell of that row can run)

1. **packed-on / combos**: the sidecar runtime blocker **F1** (scale-nibble
   3x overrun, `src/glm53_generate.cu:1024` — see `scratch/packed-runtime/
   runbook.md` section 0) must be fixed, committed, pushed, and the
   `packed experts: ... O_DIRECT` startup line verified. Cells auto-SKIP if
   `/var/lib/insignia/glm53-experts-nvfp4x.igx` is absent.
2. **pin-v2 / combos**: build the v2 pin list from the merged route trace:
   `wsl -d Arch -- /var/lib/insignia/bench-venv/bin/python
   tools/make_pinlist.py /var/lib/insignia/tracecampaign/route-merged.trace
   /var/lib/insignia/tracecampaign/pinlist-v2.txt`
   (v2 = long-trace list; v1 = the current short-trace list at
   `/var/lib/insignia/pinlist-v1.txt` — override paths with `PIN_V1=`/`PIN_V2=`).
3. **chunk128 / adaptk-v2**: code-gated (kMaxChunk is a constexpr in
   `src/glm53_generate.cu:2084`; adaptive-k v2 is the P1 rule from
   `audits/s6-open-problems.md`). Build those binaries into
   `/var/tmp/insignia-build-raptor-chunk128/` and `...-adaptv2/` — cells
   auto-SKIP until the binaries exist.
4. **cuda-graphs**: no knob exists in src yet (only the `src/cuda13_probe.cu`
   feasibility probe). The cell is stage `pending` and always SKIPs; when the
   feature lands, replace `__TBD__` in the cells table of `bench-matrix.sh`
   with the real knob and move the row to stage `singles`.

## 1. Common configuration (every cell unless overridden)

- Binary `/var/tmp/insignia-build-raptor/glm53-generate` (raptor-tuned).
- Driver `tools/benchmark_math.py` (unmodified): `--samples 2 --generate 32
  --verify-k 7 --q8-budget-mb 10240 --cache-mb 32768 --readers 4`, which
  internally sets `Q8_BUDGET_MB=10240`, `EXPERT_CACHE_MB=32768` (2425-slot
  host pin), `READERS=4`, `DFLASH2=1`,
  `DFLASH2_FP8=/var/lib/insignia/glm53-dflash2-fp8-fixed` (the `-fixed`
  cache — the plain default has the FC layout bug), `DF_VERIFY_K=7`, and
  strips `ALT_SHARD_DIR` (striping is a dev-box feature; glm-box has one NVMe).
- All other `INSIGNIA_GLM53_*` knobs are unset between cells by the runner
  (a leaked `PACKED_EXPERTS`/`DF_ADAPTIVE_K` would silently corrupt every
  later arm).
- Prompt set (canonical, deterministic): GSM8K
  `/var/lib/insignia/bench-data/gsm8k/main/test-00000-of-00001.parquet` +
  MATH-500 `/var/lib/insignia/bench-data/math500/test.jsonl`,
  `chat_prompt()` template, quantile-by-length pick, 2 samples per dataset =
  4 cases (prompt <= 225 tokens each). Exact per-case token counts:
  `bash scratch/bench/bench-matrix.sh listprompts` → `prompt-manifest.txt`.
  Cold process per run; scalar + DFlash2 arms per case; scalar-vs-dflash
  greedy-ID parity enforced inside the driver.

## 2. The matrix (each cell = 3 repeats of the driver + 1 parity pack)

| cell | env-knob vector on top of common | est. | acceptance match | gate |
|---|---|---|---|---|
| `baseline` | (none — engine defaults: auto VRAM tier, adaptive-k on, adaptive verify mode) | ~55 m | reference | reference |
| `baseline-end` | same, re-run last of stage | ~45 m | strict vs baseline | ids-only (drift control) |
| `packed-on` | `INSIGNIA_GLM53_PACKED_EXPERTS=/var/lib/insignia/glm53-experts-nvfp4x.igx` | ~55 m | strict | full |
| `vrm-576` | `INSIGNIA_GLM53_EXPERT_VRAM_MB=576` (old fixed tier; negative control) | ~55 m | strict | full |
| `vrm-max` | `INSIGNIA_GLM53_EXPERT_VRAM_MB=$BENCH_VRAM_MB` (default 3072; set from baseline's observed auto slot count) | ~55 m | strict | full |
| `pin-v1` | `INSIGNIA_GLM53_PIN_LIST=$PIN_V1` (current list; PIN_HOST/PIN_DEV defaults 8/2) | ~55 m | strict | full |
| `pin-v2` | `INSIGNIA_GLM53_PIN_LIST=$PIN_V2` (merged-trace list) | ~55 m | strict | full |
| `adaptk-off` | `INSIGNIA_GLM53_DF_ADAPTIVE_K=0` (fixed k7 all rounds) | ~55 m | ids-only | full |
| `seq-verify` | `INSIGNIA_GLM53_DF_SEQ_VERIFY=1` (forced sequential; tail-skip arm) | ~55 m | strict | full |
| `batch-verify` | `INSIGNIA_GLM53_DF_BATCH_VERIFY=1` (forced batch; adaptive default is cell 0) | ~55 m | strict | full |
| `chunk128` | kMaxChunk=64→128 build (`BIN_CHUNK128`) | ~55 m | strict | full |
| `adaptk-v2` | P1-optimal adaptive-k build (`BIN_ADAPTAV2`) | ~55 m | ids-only | full |
| `cuda-graphs` | TBD — knob does not exist in src yet | — | strict | full |
| `combo-packed-pin-vrm` | packed + pin-v2 + vrm-max | ~55 m | strict | full |
| `combo-packed-seq` | packed + forced seq-verify | ~55 m | strict | full |

Stage order enforced by the runner: `singles` first; `combos` only after the
singles are reviewed (edit/extend the cells table in `bench-matrix.sh` to
compose the actual winners — the two pre-wired combos are guesses).
Combos are meaningless until each single is known to pass parity and win.

### Expected duration model

~190-250 ms/token DFlash2 k7 and ~450-570 ms/token scalar on these prompt
sizes; per repeat = 4 cases x (scalar + dflash, cold process) = 12-16 min;
3 repeats + parity pack (12/40/30/100/240-token gens + 5 cold starts)
= 8-12 min -> **~55 min/cell**. Budgets: singles (9 cells) **~8.5 h**;
combos (2) +2 h; gated (2) +2 h plus ~15 min per extra build. Full matrix
**~13 h** — schedule overnight via Task Scheduler, resume-safe by design
(timings swing ~2x on WSL; these are planning numbers, medians only).

## 3. Parity gate (per cell, vs the `baseline` cell)

1. **In-driver**: `benchmark_math.py` raises on any scalar-vs-dflash greedy
   ID mismatch inside the cell.
2. **Parity pack** (direct engine runs, common env + cell knobs, forced
   `DF_VERIFY_K=7`, `DF_ADAPTIVE_K=0` so every cell shares one round
   structure — this is the acceptance-matching requirement):
   5 canonical runs, gens 12/40/30/100/240 on the bench-df.sh prompts
   (16-token math prompt etc.), compared to baseline by
   `diff` on `^position .* top10|^greedy IDs` lines (digit-identical
   top-10 logits + greedy IDs) and, on the 30-token case, by
   `INSIGNIA_GLM53_LOGITS_DUMP=<cell>/p2-logits.f32` +
   `tools/compare_logits.py --topk 10` (require `PASS`, top-1 100%, and
   `dmax max 0.000e+00`). Any divergence = determinism-law violation =
   cell rejected regardless of speed.
3. **Cross-cell acceptance matching** (`summarize-matrix.py`): dflash
   `rounds`/`verify_k`/`accepted_histogram` per case must equal baseline at
   the same k (`acceptance_match_baseline=yes`). Adaptive-k cells are
   exempt (round structure legitimately changes) but must keep
   `ids_match_baseline=yes`.
4. **Tier sanity**: the runner warns unless the host tier reports 2425
   slots; the summarizer records the slot count per cell (a halved pin
   invalidates the arm).

## 4. Statistical protocol

- N = 3 repeats per cell (env `BENCH_REPEATS`), report **median + IQR** of
  ms/token, speedup, prefill tok/s, accepted/round. Never single readings.
- Serialized cells, one engine process at a time, pgrep+nvidia-smi guard
  before every repeat (contention with a parallel session or the expert
  packer corrupts both the timings and the 2425-slot pin).
- Drift control: `baseline-end` re-runs the baseline last; >20% median
  shift vs `baseline` flags the whole night as drifted (re-run tight A/Bs
  interleaved instead).
- Perf claims additionally need the parity gate above (AGENTS.md law).

## 5. Deploy sequence (operator-run, from the dev box)

```bash
# day 0, once: fix F1/F2, commit, then
bash scratch/bench/deploy-matrix.sh --stage singles --time 23:30
#   = git push origin glm53-dflash2-4070ti-super (if unpushed commits)
#   + tar-stage scratch/bench -> C:\coding\Insignia-glm53-dflash2\scratch\bench
#   + git -C C:\coding\Insignia-glm53-dflash2 pull --ff-only
#   + wsl -d Arch build via build/glm53-gen.sh (INSIGNIA_BUILD_DIR=/var/tmp/insignia-build-raptor)
#   + echo singles > /var/lib/insignia/bench-matrix-args
#   + schtasks /Create ... /TR "C:\coding\...\bench-matrix-task.cmd"

# watch
ssh glm-box "wsl -d Arch -- tail -20 /var/lib/insignia/bench-matrix-task.log"
ssh glm-box "wsl -d Arch -- bash /mnt/c/coding/Insignia-glm53-dflash2/scratch/bench/bench-matrix.sh summarize"

# resume after a WSL recycle (done cells skip by DONE marker; partial cells
# redo only their missing reps/parity)
ssh glm-box "schtasks /Run /TN InsigniaBenchMatrix"

# stage 2 after reviewing singles
bash scratch/bench/deploy-matrix.sh --stage combos --run-now --no-push --no-build
```

Per-cell artifacts: `<root>/<cell>/rep{1,2,3}/` (driver output +
per-case engine logs), `<cell>/parity/` (5 logs + logits dump + VERDICT),
`<cell>/{DONE,SKIP,FAIL,WARN}`, `progress.tsv`, `summary.csv`.
