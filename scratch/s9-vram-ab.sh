#!/usr/bin/env bash
# Factorial A/B for the two trace-derived VRAM-tier fixes.  Wave 2 reverses
# arm order to counterbalance WSL/storage drift.
set -u
exec >> /var/lib/insignia/s9-vram-ab-run.log 2>&1
exec 9>/var/lib/insignia/s9-vram-ab.lock
flock -n 9 || { echo "VRAM A/B already running"; exit 1; }

ROOT=/var/lib/insignia/bench-results/s9-vram-ab
PY=/var/lib/insignia/bench-venv/bin/python
DRIVER=/mnt/c/coding/Insignia-glm53-dflash2/tools/benchmark_math.py
BIN=/var/tmp/insignia-build/glm53-generate

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
  env -i PATH=/usr/bin:/bin HOME=/root "$@" \
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
    legacy)
      run_arm "$ROOT/a-legacy/w$wave" "$offset"
      ;;
    compact)
      run_arm "$ROOT/b-compact/w$wave" "$offset" \
        INSIGNIA_GLM53_VRAM_COMPACT_SEGMENTS=1
      ;;
    batch)
      run_arm "$ROOT/c-batch-victim/w$wave" "$offset" \
        INSIGNIA_GLM53_VRAM_BATCH_VICTIM=1
      ;;
    both)
      run_arm "$ROOT/d-compact-batch/w$wave" "$offset" \
        INSIGNIA_GLM53_VRAM_COMPACT_SEGMENTS=1 \
        INSIGNIA_GLM53_VRAM_BATCH_VICTIM=1
      ;;
  esac
}

echo "VRAM A/B start: $(date)"
for arm in legacy compact batch both; do run_named "$arm" 1 0; done
for arm in both batch compact legacy; do run_named "$arm" 2 2; done

if find "$ROOT" -mindepth 3 -maxdepth 3 -type f -name DONE |
    awk 'END { exit(NR == 8 ? 0 : 1) }'; then
  touch "$ROOT/DONE"
  echo "VRAM A/B done: $(date)"
else
  echo "VRAM A/B incomplete: $(date)"
  exit 1
fi
