#!/usr/bin/env bash
# Focused production-stack A/B for exact MLA weight compaction. Every arm
# retains the s9 winner:
# packed-v2 transport + F3 device consult + O(1) host LRU + host SLRU.
set -u
exec >> /var/lib/insignia/s9-reclaim-vram-ab-run.log 2>&1
exec 9>/var/lib/insignia/s9-reclaim-vram-ab.lock
flock -n 9 || { echo "reclaim/VRAM A/B already running"; exit 1; }

ROOT=/var/lib/insignia/bench-results/s9-reclaim-vram-ab
PY=/var/lib/insignia/bench-venv/bin/python
DRIVER=/mnt/c/coding/Insignia-glm53-dflash2/tools/benchmark_math.py
BIN=/var/tmp/insignia-build/glm53-generate
V2=/var/lib/insignia/glm53-experts-nvfp4x-v2.igx

run_arm() { # output offset [environment...]
  local out=$1 offset=$2
  shift 2
  while pgrep -x glm53-generate >/dev/null; do sleep 20; done
  if [ -f "$out/DONE" ]; then
    echo "SKIP $out (complete)"
    return 0
  fi
  if [ -d "$out" ]; then
    mv "$out" "$out.partial-$(date +%Y%m%d-%H%M%S)"
  fi
  mkdir -p "$out"
  local start
  start=$(date +%s)
  env -i PATH=/usr/bin:/bin HOME=/root \
    INSIGNIA_GLM53_PACKED_EXPERTS="$V2" \
    INSIGNIA_GLM53_PACKED_V2=1 \
    INSIGNIA_GLM53_PACKED_KERNEL=2 \
    INSIGNIA_GLM53_F3=1 \
    INSIGNIA_GLM53_TIER_O1=1 \
    INSIGNIA_GLM53_TIER_SLRU=1 \
    INSIGNIA_GLM53_DF_ADAPTIVE_K=0 \
    "$@" \
    "$PY" "$DRIVER" --binary "$BIN" --output "$out" \
    --samples 2 --offset "$offset" --generate 32 --verify-k 7 \
    --cache-mb 32768 --q8-budget-mb 10240 --readers 4 --timeout 1200 \
    > "$out/driver.log" 2>&1
  local rc=$?
  if [ "$rc" = 0 ]; then touch "$out/DONE"; fi
  echo "arm $out exit=$rc wall=$(($(date +%s) - start))s"
  return "$rc"
}

run_named() {
  local arm=$1 wave=$2 offset=$3
  case "$arm" in
    base)
      run_arm "$ROOT/a-base/w$wave" "$offset"
      ;;
    absorb)
      run_arm "$ROOT/b-absorb/w$wave" "$offset" \
        INSIGNIA_GLM53_MLA_FP8_ABSORB=1
      ;;
  esac
}

echo "reclaim/VRAM A/B start: $(date)"
for arm in base absorb; do run_named "$arm" 1 0; done

if find "$ROOT" -mindepth 3 -maxdepth 3 -type f -name DONE |
    awk 'END { exit(NR == 2 ? 0 : 1) }'; then
  touch "$ROOT/DONE"
  echo "reclaim/VRAM A/B done: $(date)"
else
  echo "reclaim/VRAM A/B incomplete: $(date)"
  exit 1
fi
