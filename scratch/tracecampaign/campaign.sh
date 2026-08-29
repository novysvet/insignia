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
