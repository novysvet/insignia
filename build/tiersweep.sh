#!/usr/bin/env bash
# Sweep the pinned host-RAM expert tier size; print one summary line per config.
set -euo pipefail
cd /var/tmp/insignia-build
for MB in 6656 7168 7680 7936 8192; do
  echo "=== EXPERT_CACHE_MB=${MB} ==="
  INSIGNIA_GLM53_Q8_BUDGET_MB=10240 INSIGNIA_GLM53_EXPERT_CACHE_MB=${MB} \
    ./glm53-generate /var/lib/insignia/glm53-flash-text /var/lib/insignia/glm53-flash-text.index \
    154820 0 60 /var/lib/insignia/glm53-fp8-g64 2>&1 \
    | grep -E "expert cache:|NVFP4 cache|greedy IDs|total"
done
