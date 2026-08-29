#!/usr/bin/env bash
# Serialized A/B bench matrix for the GLM-5.3 optimization wave (glm-box,
# inside Arch WSL). One cell = one engine configuration = 3 repeats of
# tools/benchmark_math.py (GSM8K + MATH-500, cold-process, scalar + DFlash2)
# + one parity pack (5 canonical direct-engine runs, top-10-logit and
# greedy-ID diff vs the baseline cell + LOGITS_DUMP compare).
#
# Survives WSL VM recycles by resume: a cell whose DONE marker exists is
# skipped, a repeat whose results.json is complete is skipped. Intended to
# be launched by Windows Task Scheduler via bench-matrix-task.cmd ->
# bench-matrix-inner.sh (pattern of build/s6-task.cmd + build/s6-inner.sh),
# but also runs fine by hand:
#
#   bash scratch/bench/bench-matrix.sh singles            # stage 1
#   bash scratch/bench/bench-matrix.sh combos             # stage 2 (after singles review)
#   bash scratch/bench/bench-matrix.sh all
#   bash scratch/bench/bench-matrix.sh listprompts        # dump canonical prompt manifest
#   bash scratch/bench/bench-matrix.sh summarize          # rebuild summary.csv only
#
# Output: /var/lib/insignia/bench-results/<date>-matrix/<cell>/...
set -uo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/var/lib/insignia/bench-venv/bin/python}"
[ -x "$PY" ] || PY=python3
BIN_MAIN="${BIN_MAIN:-/var/tmp/insignia-build-raptor/glm53-generate}"
BIN_CHUNK128="${BIN_CHUNK128:-/var/tmp/insignia-build-raptor-chunk128/glm53-generate}"
BIN_ADAPTAV2="${BIN_ADAPTAV2:-/var/tmp/insignia-build-raptor-adaptv2/glm53-generate}"
MODEL=/var/lib/insignia/glm53-flash-text
INDEX=/var/lib/insignia/glm53-flash-text.index
FP8=/var/lib/insignia/glm53-fp8-g64
DFP8=/var/lib/insignia/glm53-dflash2-fp8-fixed          # the -fixed cache, NEVER the plain default
SIDECAR=/var/lib/insignia/glm53-experts-nvfp4x.igx
PIN_V1="${PIN_V1:-/var/lib/insignia/pinlist-v1.txt}"     # current pin list (trace-derived, short)
PIN_V2="${PIN_V2:-/var/lib/insignia/tracecampaign/pinlist-v2.txt}"  # merged-trace pin list
BENCH_REPEATS="${BENCH_REPEATS:-3}"
BENCH_VRAM_MB="${BENCH_VRAM_MB:-3072}"                   # explicit "max" expert VRAM tier (MiB)
RUN_TIMEOUT="${RUN_TIMEOUT:-1200}"                       # benchmark_math.py per-run timeout (s)

STAGE="${1:-singles}"
ROOT="${2:-/var/lib/insignia/bench-results/$(date +%Y-%m-%d)-matrix}"

# ---------------------------------------------------------------- guard ----
guard () {
    local why=""
    pgrep -af 'glm53-(generate|stream-bench|ops-bench|q8-bench|expert-bench)' >/dev/null 2>&1 \
        && why="engine process running"
    [ -z "$why" ] && pgrep -af 'pack_glm53_experts' >/dev/null 2>&1 \
        && why="expert pack job running"
    [ -z "$why" ] && pgrep -af 'benchmark_math' >/dev/null 2>&1 \
        && why="another benchmark_math running"
    if [ -z "$why" ] && command -v nvidia-smi >/dev/null 2>&1; then
        [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] \
            && why="GPU compute process visible"
    fi
    if [ -n "$why" ]; then
        echo "$(date -Is) matrix: refusing to start/continue ($why). GPU+disk reserved; reschedule." >&2
        return 1
    fi
    return 0
}

sanity () {
    local fail=0
    [ -x "$BIN_MAIN" ] || { echo "missing binary $BIN_MAIN"; fail=1; }
    [ -f "$MODEL/tokenizer.json" ] || { echo "missing model $MODEL"; fail=1; }
    [ -f "$INDEX" ] || { echo "missing index $INDEX"; fail=1; }
    [ -d "$FP8" ] || { echo "missing fp8 cache $FP8"; fail=1; }
    [ -d "$DFP8" ] || { echo "missing drafter fp8 cache $DFP8"; fail=1; }
    [ -d /var/lib/insignia/bench-data/gsm8k/main ] || { echo "missing bench-data"; fail=1; }
    [ -f /var/lib/insignia/bench-data/math500/test.jsonl ] || { echo "missing math500"; fail=1; }
    return $fail
}

# ------------------------------------------------------------- env base ----
# Start every engine invocation from a known-clean env (mirrors the unset
# list of scratch/tracecampaign/campaign.sh), then export common + cell knobs.
# Called before EVERY cell and every parity pack: knobs must never leak from
# one cell into the next (a leaked DF_ADAPTIVE_K / PACKED_EXPERTS silently
# corrupts every later arm).
clear_knobs () {
    unset INSIGNIA_GLM53_DFLASH2_INDEX INSIGNIA_GLM53_MTP INSIGNIA_GLM53_MTP_VARIANT \
          INSIGNIA_GLM53_DF_SEQ_VERIFY INSIGNIA_GLM53_DF_BATCH_VERIFY \
          INSIGNIA_GLM53_EARLY_ROUTE INSIGNIA_GLM53_EARLY_ROUTE_TRACE \
          INSIGNIA_GLM53_EARLY_MULTI_ROUTE INSIGNIA_GLM53_EARLY_MULTI_PREFETCH \
          INSIGNIA_GLM53_EARLY_MULTI_TRACE INSIGNIA_GLM53_EARLY_MULTI_N \
          INSIGNIA_GLM53_EARLY_MULTI_MAX INSIGNIA_GLM53_EARLY_PREFETCH \
          INSIGNIA_GLM53_EARLY_PREFETCH_N INSIGNIA_GLM53_CCT INSIGNIA_GLM53_CCT_MAX \
          INSIGNIA_GLM53_PIN_LIST INSIGNIA_GLM53_PIN_HOST INSIGNIA_GLM53_PIN_DEV \
          INSIGNIA_GLM53_PACKED_EXPERTS INSIGNIA_GLM53_EXPERT_VRAM_MB \
          INSIGNIA_GLM53_LAYER_DUMP INSIGNIA_GLM53_MLA_DUMP INSIGNIA_GLM53_MTP_DUMP \
          INSIGNIA_GLM53_DF_DUMP INSIGNIA_GLM53_DF_ITRACE INSIGNIA_GLM53_DF_LTRACE \
          INSIGNIA_GLM53_LOGITS_DUMP INSIGNIA_GLM53_SEAM_DUMP INSIGNIA_GLM53_PROFILE \
          INSIGNIA_GLM53_ROUTE_TRACE INSIGNIA_GLM53_ALT_SHARD_DIR \
          INSIGNIA_GLM53_EAGER_EXPERT_JOIN INSIGNIA_GLM53_DF_ADAPTIVE_K
    # benchmark_math.py sets Q8_BUDGET_MB / EXPERT_CACHE_MB / READERS /
    # DFLASH2 / DFLASH2_FP8 / DF_VERIFY_K itself from its args; never inherit.
    unset INSIGNIA_GLM53_Q8_BUDGET_MB INSIGNIA_GLM53_EXPERT_CACHE_MB \
          INSIGNIA_GLM53_READERS INSIGNIA_GLM53_DFLASH2 INSIGNIA_GLM53_DFLASH2_FP8 \
          INSIGNIA_GLM53_DF_VERIFY_K
}
clear_knobs

apply_knobs () {   # "K=V;K=V" -> export, with format validation
    local kv name value
    local IFS=';'
    for kv in $1; do
        [ -z "$kv" ] && continue
        case "$kv" in
            [A-Z0-9_]+=*) ;;
            *) echo "bad knob spec '$kv'" >&2; return 1 ;;
        esac
        name="${kv%%=*}"; value="${kv#*=}"
        export "$name=$value"
    done
    return 0
}

BENCH_ARGS="--samples 2 --generate 32 --verify-k 7 \
--q8-budget-mb 10240 --cache-mb 32768 --readers 4 --timeout $RUN_TIMEOUT"

# ------------------------------------------------------------- parity ------
# Canonical parity prompts (build/bench-df.sh + scratch/packed-runtime
# runbook). Generations 12/40/30/100/240; prompt 2 carries LOGITS_DUMP.
PARITY_PROMPTS=(
    "154820"
    "154820,13,171,1496,2343"
    "154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25"
    "154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25"
    "154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25"
)
PARITY_GENS=(12 40 30 100 240)
PARITY_DUMP_CASE=2   # index into the arrays above

parity_pack () {   # cell_name binary "knobs" outdir
    local cell="$1" bin="$2" knobs="$3" out="$4" n rc
    mkdir -p "$out"
    clear_knobs
    apply_knobs "$knobs" || return 1
    export INSIGNIA_GLM53_Q8_BUDGET_MB=10240
    export INSIGNIA_GLM53_EXPERT_CACHE_MB=32768
    export INSIGNIA_GLM53_READERS=4
    export INSIGNIA_GLM53_DFLASH2=1
    export INSIGNIA_GLM53_DFLASH2_FP8="$DFP8"
    export INSIGNIA_GLM53_DF_VERIFY_K=7      # acceptance-matched: same k everywhere
    export INSIGNIA_GLM53_DF_ADAPTIVE_K=0    # fixed round structure for cross-cell diffing
    for n in "${!PARITY_PROMPTS[@]}"; do
        local log="$out/p$n.log" dump=""
        [ -s "$log" ] && { echo "  parity p$n already present"; continue; }
        [ "$n" -eq "$PARITY_DUMP_CASE" ] && dump="$out/p$n-logits.f32"
        [ -n "$dump" ] && export INSIGNIA_GLM53_LOGITS_DUMP="$dump" || unset INSIGNIA_GLM53_LOGITS_DUMP
        guard || return 1
        echo "  parity p$n gen=${PARITY_GENS[$n]}"
        timeout $((RUN_TIMEOUT * 2)) "$bin" "$MODEL" "$INDEX" "${PARITY_PROMPTS[$n]}" \
            0 "${PARITY_GENS[$n]}" "$FP8" >"$log" 2>&1 || {
            echo "  parity p$n exited $? (see $log)" >&2
            echo "p$n:EXIT" >> "$out/FAIL"
            return 1
        }
    done
    unset INSIGNIA_GLM53_LOGITS_DUMP
    : > "$out/DONE"
    return 0
}

diff_parity () {   # refdir celldir -> 0 ok, writes VERDICT
    local ref="$1" cell="$2" n rc=0 line
    : > "$cell/VERDICT"
    for n in 0 1 2 3 4; do
        line="p$n:"
        if diff <(grep -E "^position .* top10|^greedy IDs" "$ref/p$n.log" 2>/dev/null) \
                <(grep -E "^position .* top10|^greedy IDs" "$cell/p$n.log" 2>/dev/null) \
                >"$cell/p$n.diff" 2>&1; then
            line+="top10+ids-identical"
        else
            line+="DIVERGED($(wc -l <"$cell/p$n.diff") lines)"
            rc=1
        fi
        echo "$line" | tee -a "$cell/VERDICT"
    done
    if [ -f "$ref/p$PARITY_DUMP_CASE-logits.f32" ] && [ -f "$cell/p$PARITY_DUMP_CASE-logits.f32" ]; then
        local cmplog="$cell/compare_logits.log"
        if "$PY" "$REPO/tools/compare_logits.py" \
              "$ref/p$PARITY_DUMP_CASE-logits.f32" "$cell/p$PARITY_DUMP_CASE-logits.f32" \
              --topk 10 >"$cmplog" 2>&1 \
           && grep -q "^PASS:" "$cmplog" \
           && grep -E "^dmax" "$cmplog" | grep -q "max 0.000e+00"; then
            echo "logits: digit-identical (dmax 0.0, top-1 100%)" | tee -a "$cell/VERDICT"
        else
            echo "logits: FAIL (see $cmplog)" | tee -a "$cell/VERDICT"
            rc=1
        fi
    else
        echo "logits: dumps missing (run baseline parity first)" | tee -a "$cell/VERDICT"
        rc=1
    fi
    return $rc
}

# ------------------------------------------------------------- cells ------
# name|stage|binary|knobs (K=V;K=V)|acceptance-match-mode|requires(,files)
# Acceptance-match mode: strict = dflash rounds/histogram/verify_k must equal
# baseline at the same k; ids-only = round structure legitimately changes
# (adaptive-k variants), greedy IDs must still be identical.
matrix_cells () {
    cat <<'EOF'
baseline|singles|main||strict|
baseline-end|singles|main||strict|NO_PARITY
packed-on|singles|main|INSIGNIA_GLM53_PACKED_EXPERTS=/var/lib/insignia/glm53-experts-nvfp4x.igx|strict|REQUIRES:/var/lib/insignia/glm53-experts-nvfp4x.igx
vrm-576|singles|main|INSIGNIA_GLM53_EXPERT_VRAM_MB=576|strict|
vrm-max|singles|main|INSIGNIA_GLM53_EXPERT_VRAM_MB=__VRAMMB__|strict|
pin-v1|singles|main|INSIGNIA_GLM53_PIN_LIST=__PINV1__|strict|REQUIRES:__PINV1__
pin-v2|singles|main|INSIGNIA_GLM53_PIN_LIST=__PINV2__|strict|REQUIRES:__PINV2__
adaptk-off|singles|main|INSIGNIA_GLM53_DF_ADAPTIVE_K=0|ids-only|
seq-verify|singles|main|INSIGNIA_GLM53_DF_SEQ_VERIFY=1|strict|
batch-verify|singles|main|INSIGNIA_GLM53_DF_BATCH_VERIFY=1|strict|
chunk128|gated|chunk128||strict|REQUIRES-BIN:chunk128
adaptk-v2|gated|adaptv2||ids-only|REQUIRES-BIN:adaptv2
cuda-graphs|pending|main|__TBD__|strict|PENDING:cuda-graphs-knob-not-in-src
combo-packed-pin-vrm|combos|main|INSIGNIA_GLM53_PACKED_EXPERTS=/var/lib/insignia/glm53-experts-nvfp4x.igx;INSIGNIA_GLM53_PIN_LIST=__PINV2__;INSIGNIA_GLM53_EXPERT_VRAM_MB=__VRAMMB__|strict|REQUIRES:/var/lib/insignia/glm53-experts-nvfp4x.igx,REQUIRES:__PINV2__
combo-packed-seq|combos|main|INSIGNIA_GLM53_PACKED_EXPERTS=/var/lib/insignia/glm53-experts-nvfp4x.igx;INSIGNIA_GLM53_DF_SEQ_VERIFY=1|strict|REQUIRES:/var/lib/insignia/glm53-experts-nvfp4x.igx
EOF
}

cell_binary () {   # tag -> path
    case "$1" in
        main) echo "$BIN_MAIN" ;;
        chunk128) echo "$BIN_CHUNK128" ;;
        adaptv2) echo "$BIN_ADAPTAV2" ;;
        *) echo "" ;;
    esac
}

# ------------------------------------------------------------ helpers -----
rep_complete () {   # repdir -> 0 if results.json holds 4 parity-clean cases
    "$PY" - "$1" <<'PYEOF' 2>/dev/null
import json, sys
try:
    rows = json.load(open(sys.argv[1] + "/results.json"))
except Exception:
    sys.exit(1)
sys.exit(0 if len(rows) == 4 and all(r.get("parity") for r in rows) else 1)
PYEOF
}

tier_slots () {   # grep host-tier slot count out of a cell's rep logs
    grep -rhoE "NVFP4 cache [0-9]+/[0-9]+ hits \([^;]*; [0-9]+ slots\)" "$1" 2>/dev/null \
        | grep -oE "[0-9]+ slots\)" | grep -oE "^[0-9]+" | sort | uniq -c | sort -rn | head -1 | awk '{print $2}'
}

run_cell () {   # name stage bintag knobs accept extras
    local name="$1" stage="$2" bintag="$3" knobs="$4" accept="$5" extras="$6"
    local bin cell rep n
    bin="$(cell_binary "$bintag")"
    cell="$ROOT/$name"
    knobs="${knobs//__VRAMMB__/$BENCH_VRAM_MB}"
    knobs="${knobs//__PINV1__/$PIN_V1}"
    knobs="${knobs//__PINV2__/$PIN_V2}"
    extras="${extras//__VRAMMB__/$BENCH_VRAM_MB}"
    extras="${extras//__PINV1__/$PIN_V1}"
    extras="${extras//__PINV2__/$PIN_V2}"

    [ -f "$cell/DONE" ] && { echo "[$name] DONE marker present, skipping"; return 0; }

    # requirement gates -> SKIP (logged, resumable once the artifact appears)
    local IFS=','
    for token in $extras; do
        [ -z "$token" ] && continue
        case "$token" in
            REQUIRES:*)
                [ -f "${token#REQUIRES:}" ] || {
                    echo "[$name] SKIP: missing ${token#REQUIRES:}" | tee "$cell/SKIP"; return 0; }
                ;;
            REQUIRES-BIN:*)
                [ -x "$(cell_binary "${token#REQUIRES-BIN:}")" ] || {
                    echo "[$name] SKIP: missing binary $(cell_binary "${token#REQUIRES-BIN:}")" | tee "$cell/SKIP"; return 0; }
                ;;
            PENDING:*)
                echo "[$name] SKIP: ${token#PENDING:}" | tee "$cell/SKIP"; return 0 ;;
            NO_PARITY) : ;;
        esac
    done
    unset IFS

    if [ ! -x "$bin" ]; then
        echo "[$name] SKIP: binary $bin missing/not executable" | tee "$cell/SKIP"; return 0
    fi
    mkdir -p "$cell"
    clear_knobs

    # ---- repeats of the benchmark harness ----
    for rep in $(seq 1 "$BENCH_REPEATS"); do
        local rd="$cell/rep$rep"
        if [ -d "$rd" ] && rep_complete "$rd"; then
            echo "[$name] rep$rep already complete"
        else
            rm -rf "$rd"; mkdir -p "$rd"
            guard || { echo "[$name] interrupted by guard at rep$rep" >&2; return 1; }
            echo "[$name] rep$rep $(date -Is)"
            apply_knobs "$knobs" || return 1
            if "$PY" "$REPO/tools/benchmark_math.py" \
                    --binary "$bin" --model "$MODEL" --index "$INDEX" \
                    --fp8 "$FP8" --dflash-fp8 "$DFP8" \
                    --output "$rd" $BENCH_ARGS \
                    > "$cell/rep$rep.log" 2>&1; then
                rep_complete "$rd" || {
                    echo "[$name] rep$rep incomplete results" | tee -a "$cell/FAIL"; return 1; }
                echo "[$name] rep$rep ok: $(grep -o 'DFlash2 median:.*' "$cell/rep$rep.log" | tail -1)"
            else
                echo "[$name] rep$rep FAILED (see $cell/rep$rep.log)" | tee -a "$cell/FAIL"
                return 1
            fi
        fi
    done

    # ---- tier sanity: 32 GiB pin must not have halved ----
    local slots
    slots="$(tier_slots "$cell")"
    if [ -n "$slots" ] && [ "$slots" != "2425" ]; then
        echo "[$name] WARNING: host tier = $slots slots (expected 2425; pin halved?)" | tee -a "$cell/WARN"
    fi

    # ---- parity pack + cross-cell gate vs baseline ----
    case " $extras " in
        *" NO_PARITY "*)
            :  # drift-control cell reuses baseline's parity reference
            ;;
        *)
            if [ ! -f "$ROOT/parity-base/DONE" ]; then
                if [ "$name" != "baseline" ]; then
                    echo "[$name] baseline parity missing; run the baseline cell first" >&2; return 1
                fi
                echo "[$name] collecting baseline parity pack"
                parity_pack "$name" "$bin" "$knobs" "$ROOT/parity-base" || return 1
            fi
            if [ ! -f "$cell/parity/DONE" ]; then
                echo "[$name] collecting parity pack"
                parity_pack "$name" "$bin" "$knobs" "$cell/parity" || return 1
            fi
            if diff_parity "$ROOT/parity-base" "$cell/parity"; then
                echo "[$name] parity pack: PASS" | tee -a "$cell/VERDICT"
            else
                echo "[$name] PARITY FAIL - determinism law violated, reject this cell" | tee -a "$cell/FAIL"
                return 1
            fi
            ;;
    esac

    touch "$cell/DONE"
    echo "$(date -Is) $name accept=$accept slots=$slots DONE" >> "$ROOT/progress.tsv"
    echo "[$name] DONE"
    return 0
}

summarize () {
    "$PY" "$(dirname -- "${BASH_SOURCE[0]}")/summarize-matrix.py" "$ROOT" \
        > "$ROOT/summary.csv" 2>"$ROOT/summary.err" \
        && cat "$ROOT/summary.csv" \
        || { echo "summarize failed:"; cat "$ROOT/summary.err"; return 1; }
}

listprompts () {
    mkdir -p "$ROOT"
    apply_knobs "" || true
    "$PY" "$REPO/tools/benchmark_math.py" --binary "$BIN_MAIN" --model "$MODEL" \
        --index "$INDEX" --fp8 "$FP8" --dflash-fp8 "$DFP8" \
        --output "$ROOT" --samples 2 --generate 32 --list-only \
        | tee "$ROOT/prompt-manifest.txt"
}

main () {
    sanity || { echo "pre-flight failed"; exit 2; }
    mkdir -p "$ROOT"
    echo "$(date -Is) matrix stage=$STAGE root=$ROOT binary=$BIN_MAIN" | tee -a "$ROOT/progress.tsv"

    case "$STAGE" in
        listprompts) listprompts; exit $? ;;
        summarize)    summarize;   exit $? ;;
    esac

    local selected=0 status=0
    while IFS='|' read -r name stage bintag knobs accept extras; do
        [ -z "$name" ] && continue
        case "$STAGE" in
            all) [ "$stage" = "pending" ] && continue ;;   # everything except pending rows
            *)   [ "$stage" = "$STAGE" ] || continue ;;
        esac
        selected=$((selected + 1))
        run_cell "$name" "$stage" "$bintag" "$knobs" "$accept" "$extras" || {
            echo "[$name] cell aborted; fix and re-run for resume" >&2
            status=1
            # continue to next cell so one bad knob vector does not waste the night
        }
    done < <(matrix_cells)
    # Re-run baseline parity collection if it was skipped over due to failure.
    [ "$selected" -eq 0 ] && echo "no cells matched stage=$STAGE" >&2 && exit 2
    summarize || status=1
    exit $status
}

main
