#!/usr/bin/env bash
# Median decode timing for reader-count sweep (3 reps each).
set -uo pipefail
cd /mnt/e/coding/Insignia
G=/var/tmp/insignia-build/glm53-generate
M=/var/lib/insignia/glm53-flash-text
I=/var/lib/insignia/glm53-flash-text.index
P=/var/lib/insignia/glm53-fp8-g64
T=154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25

for R in 4 6 12; do
  echo "=== READERS=$R ==="
  for rep in 1 2 3; do
    INSIGNIA_GLM53_READERS=$R INSIGNIA_GLM53_Q8_BUDGET_MB=10240 $G $M $I $T 0 60 $P 2>&1 \
      | grep -E "greedy tokens total" | sed "s/^/  rep$rep /"
  done
done
