#!/usr/bin/env bash
# Short, targeted DFlash approximate-verification A/B. This is deliberately
# not the full math campaign: one supplied tokenized prompt, five cold
# processes, exact controls bracketing top-6/top-4 experiments.
set -euo pipefail

PROMPT=${1:-/var/lib/insignia/tracecampaign/prompts/p02.csv}
GENERATE=${2:-32}
LABEL=${3:-p02-k4-g32}
PLAN=${4:-full}
BIN=/var/tmp/insignia-build/glm53-generate
MODEL=/var/lib/insignia/glm53-flash-text
INDEX=/var/lib/insignia/glm53-flash-text.index
FP8=/var/lib/insignia/glm53-fp8-g64
OUT=/var/lib/insignia/bench-results/s10-approx-verify/$LABEL

test -x "$BIN"
test -f "$PROMPT"
test ! -e "$OUT"
mkdir -p "$OUT"

COMMON=(
  INSIGNIA_GLM53_Q8_BUDGET_MB=10240
  INSIGNIA_GLM53_EXPERT_CACHE_MB=32768
  INSIGNIA_GLM53_READERS=4
  INSIGNIA_GLM53_DFLASH2=1
  INSIGNIA_GLM53_DFLASH2_FP8=/var/lib/insignia/glm53-dflash2-fp8-fixed
  INSIGNIA_GLM53_DF_VERIFY_K=4
  INSIGNIA_GLM53_DF_ADAPTIVE_K=0
  INSIGNIA_GLM53_DF_BATCH_VERIFY=1
)

run() {
  local tag=$1
  shift
  echo "=== $tag ==="
  env -u INSIGNIA_GLM53_DF_APPROX_TOPM \
      -u INSIGNIA_GLM53_DF_APPROX_RENORM \
      "${COMMON[@]}" "$@" \
      "$BIN" "$MODEL" "$INDEX" "@$PROMPT" 0 "$GENERATE" "$FP8" \
      > "$OUT/$tag.log" 2>&1
  grep -E '^greedy IDs|greedy tokens in|^  accepted histogram|^  expert I/O' \
      "$OUT/$tag.log" | tee "$OUT/$tag.summary"
}

run exact-a
case "$PLAN" in
  full)
    run top6 INSIGNIA_GLM53_DF_APPROX_TOPM=6
    run top4-zero INSIGNIA_GLM53_DF_APPROX_TOPM=4
    run top4-renorm INSIGNIA_GLM53_DF_APPROX_TOPM=4 \
        INSIGNIA_GLM53_DF_APPROX_RENORM=1
    ;;
  frontier)
    run top5 INSIGNIA_GLM53_DF_APPROX_TOPM=5
    run top4-zero INSIGNIA_GLM53_DF_APPROX_TOPM=4
    ;;
  *)
    echo "PLAN must be full or frontier" >&2
    exit 64
    ;;
esac
run exact-b

printf '%s\n' "$BIN" "$PROMPT" "$GENERATE" "$PLAN" > "$OUT/config.txt"
sha256sum "$BIN" "$PROMPT" > "$OUT/SHA256SUMS"
echo "RESULTS=$OUT"
