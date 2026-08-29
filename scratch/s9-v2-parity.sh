#!/usr/bin/env bash
# s9 v2 sidecar parity gate: v1 file vs v2 file, same production knobs
# (PACKED_V2=1 merged transport + PACKED_KERNEL=2 uint32 kernel), scalar + DFlash2.
# Determinism law: greedy IDs + top10 + acceptance structure digit-identical.
set -u
exec >> /var/lib/insignia/s9-v2-parity-run.log 2>&1
echo "v2 gate start: $(date)"
G=/var/tmp/insignia-build/glm53-generate
M=/var/lib/insignia/glm53-flash-text
I=$M.index
P8=/var/lib/insignia/glm53-fp8-g64
O=/var/lib/insignia/s9-v2-parity
mkdir -p "$O"
COMMON="INSIGNIA_GLM53_Q8_BUDGET_MB=10240 INSIGNIA_GLM53_EXPERT_CACHE_MB=32768 INSIGNIA_GLM53_READERS=4 INSIGNIA_GLM53_PACKED_V2=1 INSIGNIA_GLM53_PACKED_KERNEL=2"
V1="INSIGNIA_GLM53_PACKED_EXPERTS=/var/lib/insignia/glm53-experts-nvfp4x.igx"
V2="INSIGNIA_GLM53_PACKED_EXPERTS=/var/lib/insignia/glm53-experts-nvfp4x-v2.igx"
DF="INSIGNIA_GLM53_DFLASH2=1 INSIGNIA_GLM53_DFLASH2_FP8=/var/lib/insignia/glm53-dflash2-fp8-fixed INSIGNIA_GLM53_DF_VERIFY_K=7 INSIGNIA_GLM53_DF_ADAPTIVE_K=0"
P0="154820"
P1="154820,13,171,1496,2343"
P2="154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25"

run() { # tag prompt gen extraenv
  local tag=$1 prompt=$2 gen=$3; shift 3
  if grep -q "greedy tokens" "$O/$tag.log" 2>/dev/null; then
    echo "$tag SKIP (complete)"; return 0
  fi
  while true; do pgrep -x glm53-generate >/dev/null && sleep 20 || break; done
  local start=$(date +%s)
  env $COMMON "$@" "$G" "$M" "$I" "$prompt" 0 "$gen" "$P8" > "$O/$tag.log" 2>&1
  echo "$tag exit=$? wall=$(($(date +%s) - start))s"
}

run s16-v1 "$P2" 30 $V1
run s16-v2 "$P2" 30 $V2
run s5-v1 "$P1" 40 $V1
run s5-v2 "$P1" 40 $V2
run d16-v1 "$P2" 12 $V1 $DF
run d16-v2 "$P2" 12 $V2 $DF
run d16L-v1 "$P2" 30 $V1 $DF
run d16L-v2 "$P2" 30 $V2 $DF

gate_extract() {
  grep -E "^position .* top10|^greedy IDs|^  accepted histogram|greedy tokens in .* rounds" "$1" \
    | sed -E "s/^[0-9]+-token prompt [0-9.]+ s; //; s/\(.*//"
}
echo "=== v2 gate ==="
fail=0
for pair in "s16-v1 s16-v2" "s5-v1 s5-v2" "d16-v1 d16-v2" "d16L-v1 d16L-v2"; do
  set -- $pair
  if diff <(gate_extract "$O/$1.log") <(gate_extract "$O/$2.log") > "$O/diff-$1-vs-$2.txt" 2>&1; then
    echo "$1 vs $2: PARITY OK"
  else
    echo "$1 vs $2: PARITY FAIL (see diff-$1-vs-$2.txt)"; fail=1
  fi
done
echo "v2 gate fail=$fail"
grep -h "format v" "$O"/*.log | sort | uniq -c || true
