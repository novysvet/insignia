#!/usr/bin/env bash
# s9 O1 LRU gate: INSIGNIA_GLM53_TIER_O1 off vs on, unpacked + dflash.
# Checks: greedy IDs + top10 digit-identical AND tier hit counts equal-or-near.
set -u
exec >> /var/lib/insignia/s9-o1-parity-run.log 2>&1
echo "o1 gate start: $(date)"
G=/var/tmp/insignia-build/glm53-generate
M=/var/lib/insignia/glm53-flash-text
I=$M.index
P8=/var/lib/insignia/glm53-fp8-g64
O=/var/lib/insignia/s9-o1-parity
mkdir -p "$O"
COMMON="INSIGNIA_GLM53_Q8_BUDGET_MB=10240 INSIGNIA_GLM53_EXPERT_CACHE_MB=32768 INSIGNIA_GLM53_READERS=4"
DF="INSIGNIA_GLM53_DFLASH2=1 INSIGNIA_GLM53_DFLASH2_FP8=/var/lib/insignia/glm53-dflash2-fp8-fixed INSIGNIA_GLM53_DF_VERIFY_K=7 INSIGNIA_GLM53_DF_ADAPTIVE_K=0"
P2="154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25"

run() {
  local tag=$1 prompt=$2 gen=$3; shift 3
  if grep -q "greedy tokens" "$O/$tag.log" 2>/dev/null; then
    echo "$tag SKIP (complete)"; return 0
  fi
  while true; do pgrep -x glm53-generate >/dev/null && sleep 20 || break; done
  env $COMMON "$@" "$G" "$M" "$I" "$prompt" 0 "$gen" "$P8" > "$O/$tag.log" 2>&1
  echo "$tag exit=$?"
}

run s-x0 "$P2" 100
run s-x1 "$P2" 100 INSIGNIA_GLM53_TIER_O1=1
run d-x0 "$P2" 100 $DF
run d-x1 "$P2" 100 $DF INSIGNIA_GLM53_TIER_O1=1

gate_extract() {
  grep -E "^position .* top10|^greedy IDs|^  accepted histogram|greedy tokens in .* rounds" "$1" \
    | sed -E "s/^[0-9]+-token prompt [0-9.]+ s; //; s/\(.*//"
}
fail=0
for pair in "s-x0 s-x1" "d-x0 d-x1"; do
  set -- $pair
  if diff <(gate_extract "$O/$1.log") <(gate_extract "$O/$2.log") > "$O/diff-$1-vs-$2.txt" 2>&1; then
    echo "$1 vs $2: PARITY OK"
  else
    echo "$1 vs $2: PARITY FAIL (see diff)"; fail=1
  fi
  ta=$(grep "NVFP4 cache" "$O/$1.log" | sed 's/ hits.*//')
  tb=$(grep "NVFP4 cache" "$O/$2.log" | sed 's/ hits.*//')
  echo "host-tier hits: $1=$ta  $2=$tb"
done
echo "o1 gate fail=$fail"
grep -h "ms/token" "$O"/s-x0.log "$O"/s-x1.log "$O"/d-x0.log "$O"/d-x1.log
