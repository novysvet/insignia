#!/usr/bin/env bash
# DFlash2 decode benchmark with knobs. Usage:
#   bench-df.sh [tokens] [expert_cache_mb] [verify_k|-] [extra env as K=V,K=V] [index]
set -uo pipefail
G=/var/tmp/insignia-build/glm53-generate
M=/var/lib/insignia/glm53-flash-text
P=/var/lib/insignia/glm53-fp8-g64
TOK="${1:-100}"
CACHE="${2:-5120}"
VK="${3:--}"
EXTRA="${4:-}"
IDX="${5:-/var/lib/insignia/glm53-flash-text.index}"
PROMPT="154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25"
export INSIGNIA_GLM53_Q8_BUDGET_MB=10240
export INSIGNIA_GLM53_DFLASH2=1
export INSIGNIA_GLM53_DFLASH2_FP8=/var/lib/insignia/glm53-dflash2-fp8-fixed
export INSIGNIA_GLM53_EXPERT_CACHE_MB="$CACHE"
if [ "$VK" != "-" ]; then export INSIGNIA_GLM53_DF_VERIFY_K="$VK"; fi
if [ -n "$EXTRA" ]; then
  IFS=',' read -ra pairs <<< "$EXTRA"
  for kv in "${pairs[@]}"; do export "$kv"; done
fi
cd /var/tmp
"$G" "$M" "$IDX" "$PROMPT" 0 "$TOK" "$P" 2>&1 | grep -E "greedy IDs|DFLASH2|accepted histogram|NVFP4 cache|O_DIRECT|slots|prompt .* s|hierarchy|expert host cache" | tail -12
