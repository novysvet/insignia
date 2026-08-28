#!/usr/bin/env bash
# DFlash2 bench with an arbitrary prompt. Usage:
#   bench-df-prompt.sh "TOKEN_CSV" [gen_tokens] [verify_k|-] [extra env K=V,K=V]
set -uo pipefail
G=/var/tmp/insignia-build/glm53-generate
M=/var/lib/insignia/glm53-flash-text
I=/var/lib/insignia/glm53-flash-text.index
P=/var/lib/insignia/glm53-fp8-g64
TOKS="${1:?token csv required}"
GEN="${2:-30}"
VK="${3:--}"
EXTRA="${4:-}"
export INSIGNIA_GLM53_Q8_BUDGET_MB=10240
export INSIGNIA_GLM53_DFLASH2=1
export INSIGNIA_GLM53_DFLASH2_FP8=/var/lib/insignia/glm53-dflash2-fp8-fixed
export INSIGNIA_GLM53_EXPERT_CACHE_MB=32768
if [ "$VK" != "-" ]; then export INSIGNIA_GLM53_DF_VERIFY_K="$VK"; fi
if [ -n "$EXTRA" ]; then
  IFS=',' read -ra pairs <<< "$EXTRA"
  for kv in "${pairs[@]}"; do export "$kv"; done
fi
"$G" "$M" "$I" "$TOKS" 0 "$GEN" "$P" 2>&1 | grep -E "greedy IDs|DFLASH2|accepted|histogram|prompt .* s|MLA" | tail -8
