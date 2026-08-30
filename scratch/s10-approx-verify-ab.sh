#!/usr/bin/env bash
# Short, targeted DFlash approximate-verification A/B. This is deliberately
# not the full math campaign: one supplied tokenized prompt, five cold
# processes, exact controls bracketing top-6/top-4 experiments.
set -euo pipefail

if [[ ${1:-} == --summarize ]]; then
  DIR=$2
  reference=$(sed -n '/^greedy IDs/{p;q;}' "$DIR/exact-a.log")
  read -ra reference_ids <<< "${reference#greedy IDs }"
  for log in "$DIR"/*.log; do
    line=$(sed -n '/^greedy IDs/{p;q;}' "$log")
    read -ra ids <<< "${line#greedy IDs }"
    first=none
    limit=${#reference_ids[@]}
    (( ${#ids[@]} < limit )) && limit=${#ids[@]}
    for ((index = 0; index < limit; ++index)); do
      if [[ ${ids[index]} != "${reference_ids[index]}" ]]; then
        first=$((index + 1))
        break
      fi
    done
    [[ $first == none && ${#ids[@]} -ne ${#reference_ids[@]} ]] && first=$((limit + 1))
    digest=$(printf '%s\n' "$line" | sha256sum)
    printf '%-14s first_divergence=%-4s ids_sha256=%s\n' \
        "$(basename "$log" .log)" "$first" "${digest%% *}"
  done
  exit 0
fi

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
  aggressive)
    run top3 INSIGNIA_GLM53_DF_APPROX_TOPM=3
    run top2 INSIGNIA_GLM53_DF_APPROX_TOPM=2
    ;;
  *)
    echo "PLAN must be full, frontier, or aggressive" >&2
    exit 64
    ;;
esac
run exact-b

printf '%s\n' "$BIN" "$PROMPT" "$GENERATE" "$PLAN" > "$OUT/config.txt"
sha256sum "$BIN" "$PROMPT" > "$OUT/SHA256SUMS"
echo "RESULTS=$OUT"
