#!/usr/bin/env bash
# MTP speculative-decode parity + benchmark: the greedy ID sequence must be
# identical to the plain-greedy baseline, token for token.
set -euo pipefail
cd /var/tmp/insignia-build
M=/var/lib/insignia/glm53-flash-text
I=/var/lib/insignia/glm53-flash-text.index
P=/var/lib/insignia/glm53-fp8-g64
BASE12="2343 284 1373 364 2343 1248 3848 284 1373 364 2 2343"
BASE60="2343 284 1373 364 2343 1248 3848 284 1373 364 2 2343 1248 90 45809 11 3883 92 284 1373 364 3048 11300 1248 90 1556 21917 92 284 1373 364 12383 3876 90 5910 92 284 1373 364 7338 1248 90 3332 92 284 1373 364 4352 3876 90 3608 2227 92 284 1373 364 2427 25777 3876 90"

for K in "$@"; do
  echo "=== MTP K=$K (12-token parity) ==="
  INSIGNIA_GLM53_MTP=$K INSIGNIA_GLM53_Q8_BUDGET_MB=10240 \
    ./glm53-generate $M $I 154820 0 12 $P 2>&1 | grep -E "greedy IDs|accepted|top10|error|non-finite" | head -5
  echo "=== MTP K=$K (60-token benchmark) ==="
  INSIGNIA_GLM53_MTP=$K INSIGNIA_GLM53_Q8_BUDGET_MB=10240 \
    ./glm53-generate $M $I 154820 0 60 $P 2>&1 | grep -E "greedy IDs|accepted|streamed|NVFP4 cache" | head -6
done
echo "baseline 12: $BASE12"
