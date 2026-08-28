#!/usr/bin/env bash
# Long-context smoke test: prefill + greedy decode over a @file prompt.
# Usage: s6-longtest.sh <context> <ids-file> <gen> [binary]
set -euo pipefail
CTX="$1"; FILE="$2"; GEN="${3:-16}"; BIN="${4:-/var/tmp/insignia-build-c64/glm53-generate}"
OUT=/var/lib/insignia/longtest-ctx${CTX}-$(basename "$FILE").log
INSIGNIA_GLM53_CONTEXT="$CTX" \
INSIGNIA_GLM53_Q8_BUDGET_MB=10240 \
INSIGNIA_GLM53_EXPERT_CACHE_MB=32768 \
INSIGNIA_GLM53_PIN_LIST=/var/lib/insignia/pin-realtext.txt \
"$BIN" \
  /var/lib/insignia/glm53-flash-text \
  /var/lib/insignia/glm53-flash-text.index \
  "@$FILE" 0 "$GEN" /var/lib/insignia/glm53-fp8-g64 > "$OUT" 2>&1
grep -E "greedy IDs|prompt .* s|DFLASH2-k|NVFP4 cache|VRAM expert|expert cache" "$OUT" | tail -6
