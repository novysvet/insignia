#!/usr/bin/env bash
# Same-token quality campaign for approximate DFlash verification. The exact
# free-generation log supplies the fixed reference token stream; every arm is
# then teacher-forced through that stream and compared at full vocabulary.
set -euo pipefail

PROMPT=$1
REFERENCE_LOG=$2
LABEL=$3
shift 3
POLICIES=("$@")
(( ${#POLICIES[@]} )) || POLICIES=(6 4 3 2)

BIN=/var/tmp/insignia-build/glm53-generate
MODEL=/var/lib/insignia/glm53-flash-text
INDEX=/var/lib/insignia/glm53-flash-text.index
FP8=/var/lib/insignia/glm53-fp8-g64
REPO=/mnt/c/coding/Insignia-glm53-dflash2
PY=/var/lib/insignia/bench-venv/bin/python
OUT=/var/lib/insignia/bench-results/s10-approx-quality/$LABEL
FORCED=$OUT/forced.csv
FORCE_COUNT=${INSIGNIA_GLM53_FALSIFIER_TOKENS:-32}

test -x "$BIN"
test -f "$PROMPT"
test -f "$REFERENCE_LOG"
test ! -e "$OUT"
mkdir -p "$OUT"
(( FORCE_COUNT >= 2 && FORCE_COUNT <= 240 ))
sed -n 's/^greedy IDs //p' "$REFERENCE_LOG" | head -n 1 | \
  awk -v limit="$FORCE_COUNT" '{
    for (field = 1; field <= NF && field <= limit; ++field)
      printf "%s%s", field == 1 ? "" : ",", $field
    printf "\n"
  }' > "$FORCED"
test -s "$FORCED"
(( $(awk -F, '{print NF}' "$FORCED") >= 2 ))

COMMON=(
  INSIGNIA_GLM53_Q8_BUDGET_MB=10240
  INSIGNIA_GLM53_EXPERT_CACHE_MB=32768
  INSIGNIA_GLM53_READERS=4
  INSIGNIA_GLM53_DFLASH2=1
  INSIGNIA_GLM53_DFLASH2_FP8=/var/lib/insignia/glm53-dflash2-fp8-fixed
  INSIGNIA_GLM53_DF_VERIFY_K=4
  INSIGNIA_GLM53_DF_ADAPTIVE_K=0
  INSIGNIA_GLM53_DF_BATCH_VERIFY=1
  INSIGNIA_GLM53_FORCE_TOKENS="@$FORCED"
)

run() {
  local tag=$1
  shift
  local trace=()
  if [[ $tag != exact ]]; then
    trace=(INSIGNIA_GLM53_DF_FALSIFIER_FEATURE_TRACE="$OUT/$tag-features.bin")
  fi
  echo "=== $tag ==="
  env -u INSIGNIA_GLM53_DF_APPROX_TOPM \
      -u INSIGNIA_GLM53_DF_APPROX_RENORM \
      -u INSIGNIA_GLM53_DF_APPROX_MASS \
      -u INSIGNIA_GLM53_DF_APPROX_MIN_K \
      -u INSIGNIA_GLM53_DF_APPROX_MAX_K \
      -u INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN \
      -u INSIGNIA_GLM53_DF_LOGIT_GUARD_PREFIX \
      -u INSIGNIA_GLM53_DF_CALIBRATION_GUARD_JS \
      -u INSIGNIA_GLM53_DF_CACHE_ROUTE_K \
      -u INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN \
      -u INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET \
      -u INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS \
      -u INSIGNIA_GLM53_DF_CACHE_GUARD_RETAIN \
      -u INSIGNIA_GLM53_DF_MOE_METRICS \
      -u INSIGNIA_GLM53_DF_FALSIFIER_TRACE \
      -u INSIGNIA_GLM53_DF_FALSIFIER_FEATURE_TRACE \
      "${COMMON[@]}" "${trace[@]}" "$@" \
      INSIGNIA_GLM53_FORCE_LOGITS_DUMP="$OUT/$tag-logits.f32" \
      INSIGNIA_GLM53_FORCE_DF_LOGITS_DUMP="$OUT/$tag-draft-logits.f32" \
      "$BIN" "$MODEL" "$INDEX" "@$PROMPT" 0 1 "$FP8" \
      > "$OUT/$tag.log" 2>&1
  grep -E '^target-forced logits|^DFlash2 (approximate|adaptive) verify' "$OUT/$tag.log"
}

run exact INSIGNIA_GLM53_DF_MOE_METRICS="$OUT/moe-metrics.csv" \
    INSIGNIA_GLM53_DF_FALSIFIER_TRACE="$OUT/falsifier-events.bin"
"$PY" "$REPO/tools/summarize_moe_metrics.py" "$OUT/moe-metrics.csv" \
    > "$OUT/moe-summary.md"

for policy in "${POLICIES[@]}"; do
  if [[ $policy =~ ^top([1-8])-m([0-9]+)-cjs([0-9]+)$ ]]; then
    tag=$policy
    topm=${BASH_REMATCH[1]}
    margin_raw=${BASH_REMATCH[2]}
    calibration_raw=${BASH_REMATCH[3]}
    margin=$(awk -v raw="$margin_raw" 'BEGIN { printf "%.6f", raw / 100 }')
    calibration=$(awk -v raw="$calibration_raw" 'BEGIN { printf "%.6f", raw / 100 }')
    run "$policy" INSIGNIA_GLM53_DF_APPROX_TOPM="$topm" \
        INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN="$margin" \
        INSIGNIA_GLM53_DF_CALIBRATION_GUARD_JS="$calibration"
  elif [[ $policy =~ ^top([1-8])-m([0-9]+)(-row)?$ ]]; then
    tag=$policy
    topm=${BASH_REMATCH[1]}
    margin_raw=${BASH_REMATCH[2]}
    row_only=${BASH_REMATCH[3]}
    margin=$(awk -v raw="$margin_raw" 'BEGIN { printf "%.6f", raw / 100 }')
    guard_env=()
    [[ -z $row_only ]] || guard_env=(INSIGNIA_GLM53_DF_LOGIT_GUARD_PREFIX=0)
    run "$policy" INSIGNIA_GLM53_DF_APPROX_TOPM="$topm" \
        INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN="$margin" "${guard_env[@]}"
  elif [[ $policy =~ ^cache([0-9]+)-r([67])-e([0-9]+)-joint([2-8])-m([0-9]+)-gr([78])$ ]]; then
    tag=$policy
    cache_k=${BASH_REMATCH[1]}
    retain=${BASH_REMATCH[2]}
    regret=${BASH_REMATCH[3]}
    joint=${BASH_REMATCH[4]}
    margin_raw=${BASH_REMATCH[5]}
    guard_retain=${BASH_REMATCH[6]}
    margin=$(awk -v raw="$margin_raw" 'BEGIN { printf "%.6f", raw / 100 }')
    run "$policy" INSIGNIA_GLM53_DF_CACHE_ROUTE_K="$cache_k" \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN="$retain" \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET="0.$regret" \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS="$joint" \
        INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN="$margin" \
        INSIGNIA_GLM53_DF_CACHE_GUARD_RETAIN="$guard_retain"
  elif [[ $policy =~ ^cache([0-9]+)-r([67])-e([0-9]+)-joint([2-8])-m([0-9]+)$ ]]; then
    tag=$policy
    cache_k=${BASH_REMATCH[1]}
    retain=${BASH_REMATCH[2]}
    regret=${BASH_REMATCH[3]}
    joint=${BASH_REMATCH[4]}
    margin_raw=${BASH_REMATCH[5]}
    margin=$(awk -v raw="$margin_raw" 'BEGIN { printf "%.6f", raw / 100 }')
    run "$policy" INSIGNIA_GLM53_DF_CACHE_ROUTE_K="$cache_k" \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN="$retain" \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET="0.$regret" \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS="$joint" \
        INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN="$margin"
  elif [[ $policy =~ ^cache([0-9]+)-r([67])-e([0-9]+)-joint([2-8])$ ]]; then
    tag=$policy
    run "$policy" INSIGNIA_GLM53_DF_CACHE_ROUTE_K="${BASH_REMATCH[1]}" \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN="${BASH_REMATCH[2]}" \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET="0.${BASH_REMATCH[3]}" \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS="${BASH_REMATCH[4]}"
  elif [[ $policy =~ ^cache([0-9]+)-r([67])-e([0-9]+)-cjs([0-9]+)$ ]]; then
    tag=$policy
    run "$policy" INSIGNIA_GLM53_DF_CACHE_ROUTE_K="${BASH_REMATCH[1]}" \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN="${BASH_REMATCH[2]}" \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET="0.${BASH_REMATCH[3]}" \
        INSIGNIA_GLM53_DF_CALIBRATION_GUARD_JS="0.${BASH_REMATCH[4]}"
  elif [[ $policy =~ ^cache([0-9]+)-r([67])-e([0-9]+)$ ]]; then
    tag=$policy
    run "$policy" INSIGNIA_GLM53_DF_CACHE_ROUTE_K="${BASH_REMATCH[1]}" \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN="${BASH_REMATCH[2]}" \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET="0.${BASH_REMATCH[3]}"
  elif [[ $policy =~ ^mass([0-9]+)-guard([0-9]+)$ ]]; then
    tag=$policy
    threshold=0.${BASH_REMATCH[1]}
    guard=0.${BASH_REMATCH[2]}
    run "$tag" INSIGNIA_GLM53_DF_APPROX_MASS="$threshold" \
        INSIGNIA_GLM53_DF_APPROX_MIN_K=3 \
        INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN="$guard"
  elif [[ $policy == mass* ]]; then
    tag=$policy
    threshold=0.${policy#mass}
    run "$tag" INSIGNIA_GLM53_DF_APPROX_MASS="$threshold" \
        INSIGNIA_GLM53_DF_APPROX_MIN_K=3
  else
    tag=top$policy
    run "$tag" INSIGNIA_GLM53_DF_APPROX_TOPM="$policy"
  fi
  if ! "$PY" "$REPO/tools/compare_logits.py" \
      "$OUT/exact-logits.f32" "$OUT/$tag-logits.f32" \
      --tokens "$FORCED" --cos-threshold 0 --quiet > "$OUT/$tag-compare.txt"; then
    : # Divergence is measured output, not a harness failure.
  fi
  tail -n 10 "$OUT/$tag-compare.txt"
done

sha256sum "$BIN" "$PROMPT" "$REFERENCE_LOG" > "$OUT/SHA256SUMS"
echo "RESULTS=$OUT"
