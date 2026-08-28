#!/usr/bin/env bash
# MTP variant acceptance probe: 20 tokens per variant, summary lines only.
set -euo pipefail
cd /var/tmp/insignia-build
M=/var/lib/insignia/glm53-flash-text
I=/var/lib/insignia/glm53-flash-text.index
P=/var/lib/insignia/glm53-fp8-g64
for V in "$@"; do
  echo "=== VARIANT ${V} ==="
  INSIGNIA_GLM53_MTP_VARIANT=${V} INSIGNIA_GLM53_MTP=2 INSIGNIA_GLM53_Q8_BUDGET_MB=10240 \
    ./glm53-generate ${M} ${I} 154820 0 20 ${P} 2>&1 \
    | grep -E "greedy IDs|accepted"
done
