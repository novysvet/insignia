#!/usr/bin/env bash
set -u
BIN=/var/tmp/insignia-build/glm53-generate
ROOT=/var/lib/insignia/glm53-flash-text
IDX=/var/lib/insignia/glm53-flash-text.index
Q8=/var/lib/insignia/glm53-fp8-g64
TOKS=154820,13,171,1496,2343,200
run() {
  local name="$1"; shift
  local out
  out=$(env "$@" $BIN $ROOT $IDX $TOKS 0 60 $Q8 2>&1 | grep -E "NVFP4 cache|greedy tokens total|fallback|slot" | tail -4)
  echo "=== $name [$*]"
  echo "$out"
}
run t0.5g  INSIGNIA_GLM53_EXPERT_CACHE_MB=6600
run t13g   INSIGNIA_GLM53_EXPERT_CACHE_MB=13500
run t24g   INSIGNIA_GLM53_EXPERT_CACHE_MB=24576
run t32g   INSIGNIA_GLM53_EXPERT_CACHE_MB=32768
run t48g   INSIGNIA_GLM53_EXPERT_CACHE_MB=48000
run t54g   INSIGNIA_GLM53_EXPERT_CACHE_MB=54000
run t54r8  INSIGNIA_GLM53_EXPERT_CACHE_MB=54000 INSIGNIA_GLM53_READERS=8
run t54pc  INSIGNIA_GLM53_EXPERT_CACHE_MB=54000 INSIGNIA_GLM53_PAGECACHE_L2=1
run t54v   INSIGNIA_GLM53_EXPERT_CACHE_MB=54000 INSIGNIA_GLM53_VRAM_BUDGET_MB=1024
