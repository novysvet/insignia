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
PLAN_ENV=()
if [[ $PLAN == packedslots || $PLAN == packedjoint || $PLAN == packedjointreverse ]]; then
  PLAN_ENV=(
    INSIGNIA_GLM53_PACKED_EXPERTS=/var/lib/insignia/glm53-experts-nvfp4x-v2.igx
    INSIGNIA_GLM53_PACKED_V2=1
    INSIGNIA_GLM53_PACKED_KERNEL=2
    INSIGNIA_GLM53_F3=1
    INSIGNIA_GLM53_TIER_O1=1
    INSIGNIA_GLM53_TIER_SLRU=1
  )
fi

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
      -u INSIGNIA_GLM53_DF_LOGIT_GUARD_PREFIX \
      -u INSIGNIA_GLM53_DF_CALIBRATION_GUARD_JS \
      -u INSIGNIA_GLM53_DF_CACHE_ROUTE_K \
      -u INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN \
      -u INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET \
      -u INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS \
      -u INSIGNIA_GLM53_DF_CACHE_GUARD_RETAIN \
      -u INSIGNIA_GLM53_DEVICE_PACKED_SCALES \
      "${COMMON[@]}" "${PLAN_ENV[@]}" "$@" \
      "$BIN" "$MODEL" "$INDEX" "@$PROMPT" 0 "$GENERATE" "$FP8" \
      > "$OUT/$tag.log" 2>&1
  grep -E '^greedy IDs|greedy tokens in|^  accepted histogram|^  expert I/O|^  DFlash (approximate k|expert union|logit guard|cache route|joint cache union|calibration guard)' \
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
  ceiling)
    run top1 INSIGNIA_GLM53_DF_APPROX_TOPM=1
    ;;
  adaptive)
    run mass80 INSIGNIA_GLM53_DF_APPROX_MASS=.80 \
        INSIGNIA_GLM53_DF_APPROX_MIN_K=3
    run mass70 INSIGNIA_GLM53_DF_APPROX_MASS=.70 \
        INSIGNIA_GLM53_DF_APPROX_MIN_K=3
    ;;
  guard)
    run mass80-guard75 INSIGNIA_GLM53_DF_APPROX_MASS=.80 \
        INSIGNIA_GLM53_DF_APPROX_MIN_K=3 \
        INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN=.75
    ;;
  top4guard)
    run top4 INSIGNIA_GLM53_DF_APPROX_TOPM=4
    run top4-m75 INSIGNIA_GLM53_DF_APPROX_TOPM=4 \
        INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN=.75
    run top4-m75-row INSIGNIA_GLM53_DF_APPROX_TOPM=4 \
        INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN=.75 \
        INSIGNIA_GLM53_DF_LOGIT_GUARD_PREFIX=0
    ;;
  top4prefix)
    run top4-m75 INSIGNIA_GLM53_DF_APPROX_TOPM=4 \
        INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN=.75
    ;;
  top4context)
    run top4-m05-cjs60 INSIGNIA_GLM53_DF_APPROX_TOPM=4 \
        INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN=.05 \
        INSIGNIA_GLM53_DF_CALIBRATION_GUARD_JS=.60
    ;;
  top4cache)
    run top4 INSIGNIA_GLM53_DF_APPROX_TOPM=4
    run top4-cache32-e0025-joint6 INSIGNIA_GLM53_DF_APPROX_TOPM=4 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6
    ;;
  top4cachereverse)
    run top4-cache32-e0025-joint6 INSIGNIA_GLM53_DF_APPROX_TOPM=4 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6
    run top4 INSIGNIA_GLM53_DF_APPROX_TOPM=4
    ;;
  top4cachefrontier)
    run top4 INSIGNIA_GLM53_DF_APPROX_TOPM=4
    run top4-cache32-e0010-joint8 INSIGNIA_GLM53_DF_APPROX_TOPM=4 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0010 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=8
    run top4-cache32-e0025-joint8 INSIGNIA_GLM53_DF_APPROX_TOPM=4 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=8
    ;;
  cache)
    run cache32-r7-e0025 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025
    ;;
  cacheguard)
    run cache32-r7-e0025 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025
    run cache32-r7-e0025-cjs6000 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CALIBRATION_GUARD_JS=.6000
    ;;
  cachejoint)
    run cache32-r7-e0025 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025
    run cache32-r7-e0025-joint6 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6
    ;;
  cachejointreverse)
    run cache32-r7-e0025-joint6 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6
    run cache32-r7-e0025 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025
    ;;
  cachejointretain)
    run cache32-r7-e0025-joint6 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6
    run cache32-r6-e0025-joint6 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=6 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6
    ;;
  cachejointretainreverse)
    run cache32-r6-e0025-joint6 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=6 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6
    run cache32-r7-e0025-joint6 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6
    ;;
  cachejointguard)
    run cache32-r7-e0025-joint6 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6
    run cache32-r6-e0025-joint6-m75 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=6 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6 \
        INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN=.75
    run cache32-r6-e0025-joint6 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=6 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6
    ;;
  cachejointguardreverse)
    run cache32-r6-e0025-joint6-m75 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=6 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6 \
        INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN=.75
    run cache32-r7-e0025-joint6 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6
    ;;
  cachejointguardretain)
    run cache32-r6-e0025-joint6-m75-gr7 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=6 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6 \
        INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN=.75 \
        INSIGNIA_GLM53_DF_CACHE_GUARD_RETAIN=7
    run cache32-r7-e0025-joint6 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6
    ;;
  cachejointguardretainreverse)
    run cache32-r7-e0025-joint6 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6
    run cache32-r6-e0025-joint6-m75-gr7 INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=6 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6 \
        INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN=.75 \
        INSIGNIA_GLM53_DF_CACHE_GUARD_RETAIN=7
    ;;
  packedjoint)
    run joint-expanded INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6
    run joint-packed INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6 \
        INSIGNIA_GLM53_DEVICE_PACKED_SCALES=1
    ;;
  packedjointreverse)
    run joint-packed INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6 \
        INSIGNIA_GLM53_DEVICE_PACKED_SCALES=1
    run joint-expanded INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN=7 \
        INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0025 \
        INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6
    ;;
  packedslots)
    run device-packed INSIGNIA_GLM53_DEVICE_PACKED_SCALES=1
    ;;
  *)
    echo "PLAN must be full, frontier, aggressive, ceiling, adaptive, guard, top4guard, top4prefix, top4context, cache, cacheguard, cachejoint, cachejointreverse, cachejointretain, cachejointretainreverse, cachejointguard, cachejointguardreverse, cachejointguardretain, cachejointguardretainreverse, packedjoint, packedjointreverse, or packedslots" >&2
    exit 64
    ;;
esac
run exact-b

printf '%s\n' "$BIN" "$PROMPT" "$GENERATE" "$PLAN" > "$OUT/config.txt"
sha256sum "$BIN" "$PROMPT" > "$OUT/SHA256SUMS"
echo "RESULTS=$OUT"
