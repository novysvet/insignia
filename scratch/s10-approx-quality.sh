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

test -x "$BIN"
test -f "$PROMPT"
test -f "$REFERENCE_LOG"
test ! -e "$OUT"
mkdir -p "$OUT"
sed -n 's/^greedy IDs //p' "$REFERENCE_LOG" | head -n 1 | tr ' ' ',' > "$FORCED"
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
  echo "=== $tag ==="
  env -u INSIGNIA_GLM53_DF_APPROX_TOPM \
      -u INSIGNIA_GLM53_DF_APPROX_RENORM \
      -u INSIGNIA_GLM53_DF_APPROX_MASS \
      -u INSIGNIA_GLM53_DF_APPROX_MIN_K \
      -u INSIGNIA_GLM53_DF_APPROX_MAX_K \
      -u INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN \
      -u INSIGNIA_GLM53_DF_MOE_METRICS \
      "${COMMON[@]}" "$@" \
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
  if [[ $policy =~ ^mass([0-9]+)-guard([0-9]+)$ ]]; then
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
