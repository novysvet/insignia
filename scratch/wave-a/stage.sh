#!/usr/bin/env bash
set -e
mkdir -p /var/lib/insignia/wave-a /var/lib/insignia/tracecampaign
cat > /var/lib/insignia/wave-a.sh <<'EOF_WAVEA'
#!/usr/bin/env bash
# Wave A pipeline: packed-sidecar parity gate + packed A/B bench + route-trace
# campaign. Runs on glm-box inside WSL Arch, serialized GPU use throughout.
# Logs to /var/lib/insignia/wave-a/run.log; touches DONE at the end.
set -uo pipefail
OUT=/var/lib/insignia/wave-a
BIN=/var/tmp/insignia-build-raptor/glm53-generate
MODEL=/var/lib/insignia/glm53-flash-text
INDEX=/var/lib/insignia/glm53-flash-text.index
FP8=/var/lib/insignia/glm53-fp8-g64
DFP8=/var/lib/insignia/glm53-dflash2-fp8-fixed
PACKED=/var/lib/insignia/glm53-experts-nvfp4x.igx
PY=/var/lib/insignia/bench-venv/bin/python
REPO=/mnt/c/coding/Insignia-glm53-dflash2
mkdir -p "$OUT"
exec >>"$OUT/run.log" 2>&1
echo "=== wave-a start $(date -Is) ==="

if pgrep -af 'glm53-(generate|stream-bench|ops-bench|q8-bench|expert-bench)' >/dev/null; then
    echo "guard: engine already running; aborting" >&2
    echo "=== wave-a DONE (aborted: engine busy) $(date -Is) ==="
    touch "$OUT/DONE"
    exit 1
fi

base_env() {
    export INSIGNIA_GLM53_Q8_BUDGET_MB=10240
    export INSIGNIA_GLM53_EXPERT_CACHE_MB=32768
    export INSIGNIA_GLM53_READERS=4
    unset INSIGNIA_GLM53_DFLASH2 INSIGNIA_GLM53_DFLASH2_FP8 INSIGNIA_GLM53_DF_VERIFY_K \
          INSIGNIA_GLM53_DF_ADAPTIVE_K INSIGNIA_GLM53_PACKED_EXPERTS
}

P1="154820,13,171,1496,2343"
P2="154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25"

run() { # run <name> <prompt> <gen> [packed]
    base_env
    [ "${4:-}" = packed ] && export INSIGNIA_GLM53_PACKED_EXPERTS="$PACKED"
    "$BIN" "$MODEL" "$INDEX" "$2" 0 "$3" "$FP8" < /dev/null > "$OUT/$1.log" 2>&1
    echo "[$1] exit=$?"
    grep -E '^greedy IDs' "$OUT/$1.log" > "$OUT/$1.ids" || true
    grep -E 'ms/token|NVFP4 cache|VRAM expert tier|packed experts:|packed expand' "$OUT/$1.log" \
        | head -6 | sed "s/^/[$1] /"
}

df_run() { # df_run <name> <prompt> <gen> [packed] [adaptive]
    base_env
    export INSIGNIA_GLM53_DFLASH2=1 INSIGNIA_GLM53_DFLASH2_FP8="$DFP8" \
           INSIGNIA_GLM53_DF_VERIFY_K=7
    [ "${5:-}" = adaptive ] || export INSIGNIA_GLM53_DF_ADAPTIVE_K=0
    [ "${4:-}" = packed ] && export INSIGNIA_GLM53_PACKED_EXPERTS="$PACKED"
    "$BIN" "$MODEL" "$INDEX" "$2" 0 "$3" "$FP8" < /dev/null > "$OUT/$1.log" 2>&1
    echo "[$1] exit=$?"
    grep -E '^greedy IDs' "$OUT/$1.log" > "$OUT/$1.ids" || true
    grep -E 'ms/token|accepted histogram|NVFP4 cache|packed experts:|packed expand' "$OUT/$1.log" \
        | head -6 | sed "s/^/[$1] /"
}

echo "--- parity: scalar reference ---"
run a1-scalar "$P1" 40
run a2-scalar "$P2" 100
echo "--- parity: scalar packed (codec gate) ---"
run b1-scalar-packed "$P1" 40 packed
run b2-scalar-packed "$P2" 100 packed
echo "--- parity: dflash k7 fixed (driver-change gate) ---"
df_run c2-dflash "$P2" 100
echo "--- parity: dflash k7 fixed packed ---"
df_run d2-dflash-packed "$P2" 100 packed
echo "--- parity: dflash k7 adaptive (EMA change smoke) ---"
df_run e2-dflash-adaptive "$P2" 100 "" adaptive

echo "--- parity diffs ---"
fail=0
for pair in "a1-scalar b1-scalar-packed" "a2-scalar b2-scalar-packed" \
            "a2-scalar c2-dflash" "a2-scalar d2-dflash-packed" "a2-scalar e2-dflash-adaptive"; do
    set -- $pair
    if diff -q "$OUT/$1.ids" "$OUT/$2.ids" >/dev/null 2>&1 && [ -s "$OUT/$1.ids" ]; then
        echo "PARITY OK: $1 == $2"
    else
        echo "PARITY FAIL: $1 != $2"
        diff "$OUT/$1.ids" "$OUT/$2.ids" 2>/dev/null | head -4
        fail=1
    fi
done

echo "--- quick A/B bench: packed off vs on ---"
cd "$REPO" || fail=1
if [ "$fail" -eq 0 ]; then
    base_env
    "$PY" tools/benchmark_math.py --binary "$BIN" --output "$OUT/bench-unpacked" \
        --dflash-fp8 "$DFP8" --samples 2 --generate 32 --verify-k 7 \
        --q8-budget-mb 10240 --cache-mb 32768 --readers 4 || echo "bench unpacked FAILED"
    INSIGNIA_GLM53_PACKED_EXPERTS="$PACKED" \
    "$PY" tools/benchmark_math.py --binary "$BIN" --output "$OUT/bench-packed" \
        --dflash-fp8 "$DFP8" --samples 2 --generate 32 --verify-k 7 \
        --q8-budget-mb 10240 --cache-mb 32768 --readers 4 || echo "bench packed FAILED"
    echo "--- bench-unpacked summary ---"; tail -12 "$OUT/bench-unpacked/summary.md" 2>/dev/null
    echo "--- bench-packed summary ---"; tail -12 "$OUT/bench-packed/summary.md" 2>/dev/null
fi
echo "=== wave-a bench complete $(date -Is) parity_fail=$fail ==="

if [ "$fail" -eq 0 ]; then
    echo "--- route-trace campaign (~3h) ---"
    bash /var/lib/insignia/tracecampaign/campaign.sh || echo "campaign exited nonzero"
    bash /var/lib/insignia/tracecampaign/merge_traces.sh || true
    echo "=== campaign complete $(date -Is) ==="
else
    echo "!!! parity failed; skipping campaign"
fi
echo "=== wave-a DONE $(date -Is) parity_fail=$fail ==="
touch "$OUT/DONE"
EOF_WAVEA
cat > /var/lib/insignia/tracecampaign/campaign.sh <<'EOF_CAMP'
#!/usr/bin/env bash
# Route-trace collection campaign for GLM-5.3-Flash hot-set estimation (P3/P4).
#
# RUN ON glm-box INSIDE WSL Arch (bash). The orchestrator schedules it: it must
# NOT run while any other GPU work (glm53-*) or the expert pack job
# (pack_glm53_experts.py) is active — scalar decode is itself ~70-90% NVMe
# expert reads and would contend with the pack for disk.
#
# What it does: for each staged prompt (16-token campaign prompt + 8 GSM8K +
# 8 MATH-500), run glm53-generate in SCALAR greedy mode with
# INSIGNIA_GLM53_ROUTE_TRACE per run. Scalar mode is mandatory for trace
# completeness: route_trace() fires only in Runner::step -> sparse_moe
# (src/glm53_generate.cu:2881); in DFlash2 mode committed tokens of verified
# rounds never pass through step(), so the trace would cover only empty-round
# fallbacks (a biased subset). 17 runs x ~1200 gen = ~20.5k decode tokens.
#
# Idempotent: a run whose log already contains the final summary line is
# skipped, so the campaign can be re-invoked after interruption.
set -uo pipefail

SDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"   # make_prompts.py lives next to this script
OUT="${OUT:-/var/lib/insignia/tracecampaign}"
BIN="${BIN:-/var/tmp/insignia-build-raptor/glm53-generate}"  # name verified from build/glm53-gen.sh
[ -x "$BIN" ] || BIN=/var/tmp/insignia-build/glm53-generate
MODEL=/var/lib/insignia/glm53-flash-text
INDEX=/var/lib/insignia/glm53-flash-text.index
FP8=/var/lib/insignia/glm53-fp8-g64
PY="${PY:-/var/lib/insignia/bench-venv/bin/python}"
[ -x "$PY" ] || PY=python3
RUN_TIMEOUT="${RUN_TIMEOUT:-5400}"   # 90 min per run; expected ~12-15 min

mkdir -p "$OUT/traces" "$OUT/logs"

guard () {
    local why=""
    pgrep -af 'glm53-(generate|stream-bench|ops-bench|q8-bench|expert-bench)' >/dev/null 2>&1 \
        && why="engine process running"
    [ -z "$why" ] && pgrep -af 'pack_glm53_experts' >/dev/null 2>&1 \
        && why="expert pack job running"
    if [ -z "$why" ] && command -v nvidia-smi >/dev/null 2>&1; then
        [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] \
            && why="GPU compute process visible"
    fi
    if [ -n "$why" ]; then
        echo "campaign: refusing to start/continue ($why). GPU+disk are reserved; the orchestrator must reschedule." >&2
        [ "${1:-}" = final ] || exit 1
    fi
}

[ -f "$OUT/prompts/manifest.tsv" ] || "$PY" "$SDIR/make_prompts.py" --outdir "$OUT"
cd /var/tmp 2>/dev/null || cd /tmp   # engine writes nothing to cwd; keep strays off the store

sha256sum "$BIN" > "$OUT/binary.sha256"
printf "run\twall_s\tstatus\n" > "$OUT/runs.tsv"

while IFS=$'\t' read -r id dataset row ptok gen; do
    [ "$id" = "id" ] && continue
    log="$OUT/logs/$id.log"
    if grep -q "greedy token" "$log" 2>/dev/null; then
        echo "[$id] already complete, skipping"
        printf '%s\t0\tskipped\n' "$id" >> "$OUT/runs.tsv"
        continue
    fi
    guard || exit 1
    unset INSIGNIA_GLM53_DFLASH2 INSIGNIA_GLM53_DFLASH2_FP8 INSIGNIA_GLM53_DFLASH2_INDEX \
          INSIGNIA_GLM53_MTP INSIGNIA_GLM53_MTP_VARIANT INSIGNIA_GLM53_DF_ADAPTIVE_K \
          INSIGNIA_GLM53_DF_SEQ_VERIFY INSIGNIA_GLM53_DF_BATCH_VERIFY INSIGNIA_GLM53_DF_VERIFY_K \
          INSIGNIA_GLM53_EARLY_ROUTE INSIGNIA_GLM53_EARLY_ROUTE_TRACE \
          INSIGNIA_GLM53_EARLY_MULTI_ROUTE INSIGNIA_GLM53_EARLY_MULTI_PREFETCH \
          INSIGNIA_GLM53_EARLY_MULTI_TRACE INSIGNIA_GLM53_EARLY_PREFETCH \
          INSIGNIA_GLM53_CCT INSIGNIA_GLM53_CCT_MAX INSIGNIA_GLM53_PIN_LIST \
          INSIGNIA_GLM53_PIN_HOST INSIGNIA_GLM53_PIN_DEV INSIGNIA_GLM53_PACKED_EXPERTS \
          INSIGNIA_GLM53_LAYER_DUMP INSIGNIA_GLM53_MLA_DUMP INSIGNIA_GLM53_MTP_DUMP \
          INSIGNIA_GLM53_DF_DUMP INSIGNIA_GLM53_LOGITS_DUMP INSIGNIA_GLM53_SEAM_DUMP \
          INSIGNIA_GLM53_PROFILE
    export INSIGNIA_GLM53_Q8_BUDGET_MB=10240        # dense FP8 resident (bench default)
    export INSIGNIA_GLM53_EXPERT_CACHE_MB=32768     # 2425-slot host tier (caching only; routing is deterministic)
    export INSIGNIA_GLM53_READERS=4                 # single-NVMe optimum on glm-box
    export INSIGNIA_GLM53_ROUTE_TRACE="$OUT/traces/route-$id.txt"
    echo "[$id] $dataset row=$row prompt=$ptok gen=$gen -> $(basename "$OUT")/traces/route-$id.txt"
    start=$SECONDS
    timeout "$RUN_TIMEOUT" "$BIN" "$MODEL" "$INDEX" "@$OUT/prompts/$id.csv" 0 "$gen" "$FP8" \
        < /dev/null > "$log" 2>&1
    status=$?
    unset INSIGNIA_GLM53_ROUTE_TRACE
    rows=$(wc -l < "$OUT/traces/route-$id.txt" 2>/dev/null || echo 0)
    echo "[$id] exit=$status trace_rows=$rows wall=$((SECONDS-start))s"
    printf '%s\t%d\t%d\n' "$id" "$((SECONDS-start))" "$status" >> "$OUT/runs.tsv"
    guard final   # log a warning if the box got busy mid-campaign, but keep going
done < "$OUT/prompts/manifest.tsv"

echo "campaign done. Merge with: $SDIR/merge_traces.sh $OUT"
EOF_CAMP
cat > /var/lib/insignia/tracecampaign/make_prompts.py <<'EOF_MKPY'
#!/usr/bin/env python3
"""Stage route-trace campaign prompts from GSM8K + MATH-500 (+ the campaign prompt).

Writes, under --outdir:
  prompts/p<NN>.csv        one line, comma-separated token ids (engine @file input)
  prompts/manifest.tsv     id  dataset  row  prompt_tokens  gen  file

Selection: chat-wrapped prompts (same template as tools/benchmark_math.py),
length-filtered to [--min-tokens, --max-tokens], then evenly spaced across the
eligible row INDEX range so topics stay diverse (adjacent dataset rows are
often near-duplicates in difficulty/style). The fixed 16-token campaign prompt
(bench-df.sh) is emitted first as p00 for continuity with route-campaign.txt;
it is a known-atypical (repetitive-routing) prompt and is excluded from the
train/test split by the merge spec.
"""

import argparse
import json
import pathlib

from tokenizers import Tokenizer

CAMPAIGN_TOKENS = [154820, 11, 301, 2745, 941, 1516, 87, 29871,
                   526, 1052, 374, 123, 77, 918, 1520, 25]


def chat_prompt(problem):
    return (
        "[gMASK]<sop><|system|>Reasoning Effort: Max"
        "<|user|>Solve this problem. Show your reasoning and end with "
        "\\boxed{answer}.\n\n" + problem.strip() + "\n<|assistant|><think>"
    )


def pick(rows, count, lo, hi):
    eligible = [(i, ids) for i, ids in rows if lo <= len(ids) <= hi]
    if len(eligible) < count:
        raise SystemExit(f"only {len(eligible)} prompts in [{lo},{hi}] tokens, need {count}")
    picked = []
    for ordinal in range(count):
        # benchmark_math.py's bounded spread: never exceeds len(eligible)-1
        rank = round((ordinal + 1) * (len(eligible) - 1) / (count + 1)) if count > 1 else 0
        picked.append(eligible[rank])
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k", type=pathlib.Path,
                    default=pathlib.Path("/var/lib/insignia/bench-data/gsm8k/main/test-00000-of-00001.parquet"))
    ap.add_argument("--math500", type=pathlib.Path,
                    default=pathlib.Path("/var/lib/insignia/bench-data/math500/test.jsonl"))
    ap.add_argument("--tokenizer", type=pathlib.Path,
                    default=pathlib.Path("/var/lib/insignia/glm53-flash-text/tokenizer.json"))
    ap.add_argument("--outdir", type=pathlib.Path,
                    default=pathlib.Path("/var/lib/insignia/tracecampaign"))
    ap.add_argument("--per-dataset", type=int, default=8)
    ap.add_argument("--min-tokens", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=110)
    ap.add_argument("--gen", type=int, default=1250,
                    help="decode tokens per real-text prompt")
    ap.add_argument("--campaign-gen", type=int, default=500)
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    import pyarrow.parquet as pq
    gsm = pq.read_table(args.gsm8k).to_pylist()
    math = [json.loads(line) for line in
            args.math500.read_text(encoding="utf-8").splitlines() if line.strip()]

    rows = {"gsm8k": [], "math500": []}
    for i, row in enumerate(gsm):
        rows["gsm8k"].append((i, tok.encode(chat_prompt(row["question"])).ids))
    for i, row in enumerate(math):
        rows["math500"].append((i, tok.encode(chat_prompt(row["problem"])).ids))

    prompts_dir = args.outdir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    manifest = [("p00", "campaign", "-", len(CAMPAIGN_TOKENS), args.campaign_gen)]

    selected = {"gsm8k": pick(rows["gsm8k"], args.per_dataset, args.min_tokens,
                              args.max_tokens),
                "math500": pick(rows["math500"], args.per_dataset, args.min_tokens,
                                args.max_tokens)}
    ordinal = 1
    for dataset in ("gsm8k", "math500"):
        for index, ids in selected[dataset]:
            manifest.append((f"p{ordinal:02d}", dataset, str(index), len(ids), args.gen))
            (prompts_dir / f"p{ordinal:02d}.csv").write_text(",".join(map(str, ids)) + "\n",
                                                             encoding="utf-8")
            ordinal += 1
    (prompts_dir / "p00.csv").write_text(",".join(map(str, CAMPAIGN_TOKENS)) + "\n",
                                         encoding="utf-8")

    with (prompts_dir / "manifest.tsv").open("w", encoding="utf-8") as handle:
        handle.write("id\tdataset\trow\tprompt_tokens\tgen\n")
        for entry in manifest:
            handle.write("\t".join(map(str, entry)) + "\n")
    total = sum(entry[4] for entry in manifest)
    print(f"{len(manifest)} prompts, {total} decode tokens -> {prompts_dir}")


if __name__ == "__main__":
    main()
EOF_MKPY
cat > /var/lib/insignia/tracecampaign/merge_traces.sh <<'EOF_MERGE'
#!/usr/bin/env bash
# Merge per-run ROUTE_TRACE files into one token-renumbered trace + manifest.
#
# Usage: merge_traces.sh [CAMPAIGN_DIR] [OUTPUT_PREFIX]
#   CAMPAIGN_DIR   default /var/lib/insignia/tracecampaign
#                  (reads traces/route-<id>.tt per prompts/manifest.tsv; legacy
#                   /var/lib/insignia/route-{realtext,campaign}.txt fold in first)
#   OUTPUT_PREFIX  default <CAMPAIGN_DIR>/merged/route-merged
#                  -> $PREFIX.trace (rows) + $PREFIX.manifest.tsv (run table)
#
# Format v1 (see TRACE-FORMAT.md): rows `<token> <layer> e0..e7 s0..s7`
# (18 fields), token-major, the 42 sparse-layer rows of one decode token
# contiguous, token ids globally unique: run #k gets token base k*100000, so
# ids encode provenance and dump_cct.py's by-token grouping stays correct.
# make_pinlist.py and dump_cct.py consume $PREFIX.trace unchanged.
set -uo pipefail

DIR="${1:-/var/lib/insignia/tracecampaign}"
PREFIX="${2:-$DIR/merged/route-merged}"
mkdir -p "$(dirname "$PREFIX")"
: > "$PREFIX.trace"
printf 'run\tfile\trows\ttokens\ttoken_base\tdataset\trow\n' > "$PREFIX.manifest.tsv"

K=0
emit () {  # emit <id> <file> [dataset row]
    local id="$1" file="$2" ds="${3:--}" row="${4:--}" base rows tokens
    [ -f "$file" ] || return 0
    base=$((K * 100000))
    rows=$(awk -v base="$base" 'NF >= 11 {
        printf "%d %s", $1 + base, $2
        for (i = 3; i <= NF; i++) printf " %s", $i
        printf "\n" }' "$file" >> "$PREFIX.trace"; wc -l < "$file")
    tokens=$(awk 'NF >= 11 { s[$1] = 1 } END { print length(s) }' "$file")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$file" "$rows" "$tokens" "$base" "$ds" "$row" \
        >> "$PREFIX.manifest.tsv"
    echo "  $id: $tokens tokens ($rows rows, base $base)" >&2
    K=$((K + 1))
}

for legacy in /var/lib/insignia/route-realtext.txt /var/lib/insignia/route-campaign.txt; do
    [ -f "$legacy" ] && emit "legacy-$(basename "$legacy" .txt)" "$legacy"
done
if [ -f "$DIR/prompts/manifest.tsv" ]; then
    while IFS=$'\t' read -r id dataset row ptok gen; do
        [ "$id" = "id" ] && continue
        emit "$id" "$DIR/traces/route-$id.txt" "$dataset" "$row"
    done < "$DIR/prompts/manifest.tsv"
fi

echo "merged -> $PREFIX.trace ($(wc -l < "$PREFIX.trace") rows)"
echo "split protocol: drop legacy + p00 rows, alternate runs -> train/test (TRACE-FORMAT.md)"
EOF_MERGE
chmod +x /var/lib/insignia/wave-a.sh /var/lib/insignia/tracecampaign/*.sh
printf "wsl -d Arch -- bash /var/lib/insignia/wave-a.sh\r\n" > /mnt/c/coding/wave-a-task.cmd
echo staged OK
