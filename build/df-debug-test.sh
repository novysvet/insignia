#!/usr/bin/env bash
# DFlash2 per-round trace on the realistic prompt. Usage: df-debug-test.sh [gen] [k] [extra-env]
P="14572,25,362,8632,18611,220,99590,22,31720,817,6460,13,576,8632,25976,220,99366,4115,817,1899,11,220,21,2849,817,2003,13,2585,1657,31720,1558,432,8192,304,220,19,5555,30,6928,697,32559,3019,553,3019,11,1221,2968,279,1590,4226,624,16127,25"
G=/var/tmp/insignia-build/glm53-generate
GEN="${1:-12}"
VK="${2:--}"
EXTRA="${3:-}"
export INSIGNIA_GLM53_Q8_BUDGET_MB=10240
export INSIGNIA_GLM53_DFLASH2=1
export INSIGNIA_GLM53_DFLASH2_FP8=/var/lib/insignia/glm53-dflash2-fp8-fixed
export INSIGNIA_GLM53_EXPERT_CACHE_MB=32768
export INSIGNIA_GLM53_DF_VERIFY_K="${VK/-/4}"
export INSIGNIA_GLM53_DF_DEBUG=1
if [ -n "$EXTRA" ]; then
  IFS=',' read -ra pairs <<< "$EXTRA"
  for kv in "${pairs[@]}"; do export "$kv"; done
fi
"$G" /var/lib/insignia/glm53-flash-text /var/lib/insignia/glm53-flash-text.index "$P" 0 "$GEN" /var/lib/insignia/glm53-fp8-g64 2>&1 | grep -E "df round|greedy IDs|DFLASH2-k|histogram|position .* top10"
