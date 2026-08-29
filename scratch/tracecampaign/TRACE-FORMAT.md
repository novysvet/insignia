# Merged route-trace format v1 (pin list v2 + CCT input)

## 1. Raw per-run trace (engine output, unchanged)

`INSIGNIA_GLM53_ROUTE_TRACE=<path>` makes `Runner::route_trace`
(src/glm53_generate.cu:2746) write one row per sparse-layer visit:

    <token> <layer> <e0> ... <e7> <s0> ... <s7>        (18 fields, text)

- `token`  — `token_index_`, incremented once per `Runner::step` (decode step);
  starts at 1 for the final prompt token. **Decode path only**: `route_trace`
  is called from `sparse_moe`, which only `step()` reaches. Prefill and verify
  rows go through `moe_multi` and are NOT traced. Run `glm53-generate` in
  scalar mode (no `INSIGNIA_GLM53_DFLASH2`) or verified-round tokens are
  silently missing from the trace.
- `layer`  — 3..44 (42 sparse layers; 0-2 are dense).
- `e0..e7` — selected experts in router slot order (descending `choice` =
  sigmoid(logit) + e_score_correction_bias; ties broken by partial_sort).
- `s0..s7` — raw sigmoid scores of those experts, `%.6e`.
- 42 rows per decode token, ffushed per row. Measured 137.7 B/row
  (route-realtext.txt: 28,914 B / 210 rows) -> 5.78 KB/token.

## 2. Merged trace (merge_traces.sh output)

`route-merged.trace`: identical 18-field rows with token ids renumbered so
they are globally unique across runs:

    run #k (order: legacy-realtext, legacy-campaign, then p00..p16 by manifest)
    gets token base k*100000; emitted token = base + original index.

Invariants:

1. Rows stay token-major and each token's 42 layer rows stay contiguous and
   in ascending layer order (per-run files are already in this order; the
   merger only rewrites field 1).
2. Lines starting with `#` are comments (allowed anywhere; both consumers
   skip short lines: `make_pinlist.py` needs >=10 fields, `dump_cct.py`
   needs >=11 — never emit score-less rows if CCT input matters).
3. Scores are passed through verbatim.

`route-merged.manifest.tsv` columns:

    run  file  rows  tokens  token_base  dataset  row

`dataset`/`row` identify the GSM8K/MATH-500 source row (`-` for legacy).
`token_base` is the provenance key: token // 100000 = run ordinal.

## 3. Consumer contract (verified against the tools)

- `tools/make_pinlist.py <trace> <out>` — counts `(layer, expert)` over
  fields[2:10]; token ids irrelevant; writes `layer expert hits` sorted by
  hits desc per layer (this IS pin list v2's input format; the loader takes
  the first N lines per layer).
- `tools/dump_cct.py <trace> <out>` — builds `by_token[token][layer] =
  experts[2:10]` and counts adjacent-sparse-layer co-activations **within a
  token**. This is why global token uniqueness is mandatory: colliding ids
  across runs would fuse layers from different prompts into garbage pairs.
  Binary output: `b"CCT0"`, u32 {45, 288, 8}, 41 x (288x8 u16 successor
  tables).

## 4. Split-sample protocol (P3 hot-set generalization)

- Exclude `legacy-*` and `p00` (the fixed 16-token campaign prompt — known
  atypically repetitive routing) from both arms; keep p00 as a separate
  continuity slice to A/B against route-campaign.txt.
- Train arm = odd real-text runs (p01, p03, ...), test arm = even runs
  (p02, p04, ...). Prompt-level splits (not token-level) because routing is
  autocorrelated within a prompt (adjacent-token overlap I/8 ~ 0.19).
- Learning-curve analysis: subsample tokens per layer (10/25/50/100/250/500+)
  from the train arm, rebuild pin lists, evaluate held-out hit rate to
  answer "how many trace tokens before the pin list is worth -10% ms/token".
- With 20k real-text tokens: 8 accesses/token -> 160k draws per layer, so
  even rank-57 experts (p ~ 0.2-0.5%) expect 320-800 observations and the
  whole 57-slot boundary is rank-stable.

## 5. Legacy / auxiliary data

- `/var/lib/insignia/route-realtext.txt` (5 tokens) and `route-campaign.txt`
  (60 tokens) merge in as runs 0/1 when present.
- `early-route-math.txt` (12 tokens) has a different 19-field shape
  (`token layer overlap p0..p7 a0..a7`); actual experts are fields 12..19.
  Convert by re-emitting `token layer a0..a7 0 0 0 0 0 0 0 0` if ever needed
  (both consumers ignore/junk-tolerate the zero scores; CCT requires the 18
  fields present).
- `early-multi-prompt.txt` (1x52 prefill rows) / `early-multi-df-k7.txt`
  (7 batches, verify rows) are batch-path routing (predicted+selected per
  row); prefill/verify routing != decode routing; use only for P4
  conditional-entropy studies, not hot-set frequencies.
