#!/usr/bin/env bash
# s9 F3 parity gate: INSIGNIA_GLM53_F3=0 vs =1, unpacked + packed stores,
# scalar + DFlash2. Determinism law: greedy IDs + top10 lines + acceptance
# histograms must be digit-identical across knob arms of the same store.
set -u
# Task-Scheduler safe: redirect self (cmd.exe cannot understand bash paths).
exec >> /var/lib/insignia/s9-f3-parity-run.log 2>&1
echo "gate start: $(date)"
G=/var/tmp/insignia-build/glm53-generate
M=/var/lib/insignia/glm53-flash-text
I=$M.index
P8=/var/lib/insignia/glm53-fp8-g64
O=/var/lib/insignia/s9-f3-parity
mkdir -p "$O"
COMMON="INSIGNIA_GLM53_Q8_BUDGET_MB=10240 INSIGNIA_GLM53_EXPERT_CACHE_MB=32768 INSIGNIA_GLM53_READERS=4"
PK="INSIGNIA_GLM53_PACKED_EXPERTS=/var/lib/insignia/glm53-experts-nvfp4x.igx"
DF="INSIGNIA_GLM53_DFLASH2=1 INSIGNIA_GLM53_DFLASH2_FP8=/var/lib/insignia/glm53-dflash2-fp8-fixed INSIGNIA_GLM53_DF_VERIFY_K=7 INSIGNIA_GLM53_DF_ADAPTIVE_K=0"
P0="154820"
P1="154820,13,171,1496,2343"
P2="154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25"

run() { # tag prompt gen extraenv
  local tag=$1 prompt=$2 gen=$3; shift 3
  # Resume-safe: a prior completed run leaves a greedy-tokens summary line;
  # a WSL VM recycle mid-run leaves a partial log that we redo from scratch.
  if grep -q "greedy tokens" "$O/$tag.log" 2>/dev/null; then
    echo "$tag SKIP (complete)"
    return 0
  fi
  while true; do pgrep -x glm53-generate >/dev/null && sleep 20 || break; done
  env $COMMON "$@" "$G" "$M" "$I" "$prompt" 0 "$gen" "$P8" > "$O/$tag.log" 2>&1
  echo "$tag exit=$? $(grep -c '^position' "$O/$tag.log") positions"
}

# scalar, unpacked store (current production arm)
run s-u-f0 "$P0" 12
run s-u-f1 "$P0" 12 INSIGNIA_GLM53_F3=1
run s-u5-f0 "$P1" 40
run s-u5-f1 "$P1" 40 INSIGNIA_GLM53_F3=1
run s-u16-f0 "$P2" 30
run s-u16-f1 "$P2" 30 INSIGNIA_GLM53_F3=1
# scalar, packed store (smoke)
run s-p16-f0 "$P2" 30 $PK
run s-p16-f1 "$P2" 30 $PK INSIGNIA_GLM53_F3=1
# dflash, unpacked
run d-u16-f0 "$P2" 12 $DF
run d-u16-f1 "$P2" 12 $DF INSIGNIA_GLM53_F3=1
run d-u16L-f0 "$P2" 30 $DF
run d-u16L-f1 "$P2" 30 $DF INSIGNIA_GLM53_F3=1
# dflash, packed (smoke)
run d-p16-f0 "$P2" 12 $PK $DF
run d-p16-f1 "$P2" 12 $PK $DF INSIGNIA_GLM53_F3=1

gate_extract() {
  grep -E "^position .* top10|^greedy IDs|^  accepted histogram|greedy tokens in .* rounds" "$1" \
    | sed -E 's/\(.*//'
}
echo "=== gate ==="
fail=0
for pair in "s-u-f0 s-u-f1" "s-u5-f0 s-u5-f1" "s-u16-f0 s-u16-f1" "s-p16-f0 s-p16-f1" "d-u16-f0 d-u16-f1" "d-u16L-f0 d-u16L-f1" "d-p16-f0 d-p16-f1"; do
  set -- $pair
  if diff <(gate_extract "$O/$1.log") <(gate_extract "$O/$2.log") > "$O/diff-$1-vs-$2.txt" 2>&1; then
    echo "$1 vs $2: PARITY OK"
  else
    echo "$1 vs $2: PARITY FAIL (see diff-$1-vs-$2.txt)"; fail=1
  fi
done
echo "gate fail=$fail"
grep -h "F3 device-consult" "$O"/*-f1.log 2>/dev/null | sort | uniq -c || true
