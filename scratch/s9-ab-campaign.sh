#!/usr/bin/env bash
# s9 production A/B campaign: benchmark_math.py driver, ABAB reps.
# Arms: (A) unpacked+F3 off = current production champion
#       (B) unpacked+F3 on
#       (C) packed v2 + F3 on + merged transport + uint32 kernel
set -u
exec >> /var/lib/insignia/s9-ab-run.log 2>&1
echo "AB campaign start: $(date)"
ROOT=/var/lib/insignia/bench-results/s9-ab
PY=/var/lib/insignia/bench-venv/bin/python
DRIVER=/mnt/c/coding/Insignia-glm53-dflash2/tools/benchmark_math.py
BIN=/var/tmp/insignia-build/glm53-generate
V2=/var/lib/insignia/glm53-experts-nvfp4x-v2.igx

run_arm() { # outdir offset env...
  local out=$1 offset=$2; shift 2
  while true; do pgrep -x glm53-generate >/dev/null && sleep 20 || break; done
  if [ -f "$out/summary.md" ]; then
    echo "SKIP $out (summary exists)"; return 0
  fi
  mkdir -p "$out"
  local start=$(date +%s)
  env -i PATH=/usr/bin:/bin HOME=/root "$@" \
    "$PY" "$DRIVER" --binary "$BIN" --output "$out" \
    --samples 2 --offset "$offset" --generate 32 --verify-k 7 \
    --cache-mb 32768 --q8-budget-mb 10240 --readers 4 --timeout 1200 \
    > "$out/driver.log" 2>&1
  echo "arm $out exit=$? wall=$(($(date +%s) - start))s"
}
# Design: two waves, arms A-D each wave; wave-2 slices are NEW prompts
# (--offset 2) so coverage is 8 distinct case-pairs per arm at 1 rep each.
# Between-prompt acceptance variance dominates cold-process run variance, so
# distinct prompts beat repetitions at fixed wall time.

for wave in 1 2; do
  if [ "$wave" = 1 ]; then OFFSET=0; else OFFSET=2; fi
  # A: current production champion (unpacked store, no experimental knobs)
  run_arm "$ROOT/a-unpacked/w$wave" "$OFFSET"
  # B: packed v2 sidecar alone (store + transport effect)
  run_arm "$ROOT/b-packedv2/w$wave" "$OFFSET" \
    INSIGNIA_GLM53_PACKED_EXPERTS="$V2" \
    INSIGNIA_GLM53_PACKED_V2=1 \
    INSIGNIA_GLM53_PACKED_KERNEL=2
  # C: full tier stack on the packed v2 store (B + F3 + O1 LRU + segment-LRU)
  run_arm "$ROOT/c-full-stack/w$wave" "$OFFSET" \
    INSIGNIA_GLM53_PACKED_EXPERTS="$V2" \
    INSIGNIA_GLM53_PACKED_V2=1 \
    INSIGNIA_GLM53_PACKED_KERNEL=2 \
    INSIGNIA_GLM53_F3=1 \
    INSIGNIA_GLM53_TIER_O1=1 \
    INSIGNIA_GLM53_TIER_SLRU=1
  # D: C plus adaptive draft length v2 with the online cost regression (k may
  # differ per round; acceptance histogram legitimately diverges from C)
  run_arm "$ROOT/d-adaptk-v2/w$wave" "$OFFSET" \
    INSIGNIA_GLM53_PACKED_EXPERTS="$V2" \
    INSIGNIA_GLM53_PACKED_V2=1 \
    INSIGNIA_GLM53_PACKED_KERNEL=2 \
    INSIGNIA_GLM53_F3=1 \
    INSIGNIA_GLM53_TIER_O1=1 \
    INSIGNIA_GLM53_TIER_SLRU=1 \
    INSIGNIA_GLM53_DF_ADAPTIVE_K=2 \
    INSIGNIA_GLM53_DF_COSTTRACE=1
done
echo "AB campaign done: $(date)"
