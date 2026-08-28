#!/usr/bin/env bash
# Re-run both engine dense modes with routing traces + layer dumps.
set -euo pipefail
BIN=/var/tmp/insignia-build/glm53-generate
ROOT=/var/lib/insignia/glm53-flash-text
IDX=/var/lib/insignia/glm53-flash-text.index
cd /var/tmp
echo "=== FP8 mode ==="
INSIGNIA_GLM53_Q8_BUDGET_MB=10240 \
INSIGNIA_GLM53_ROUTE_TRACE=/var/tmp/route-fp8.txt \
INSIGNIA_GLM53_LAYER_DUMP=/var/tmp/layer-dump-fp8.bin \
  "$BIN" "$ROOT" "$IDX" 154820 0 1 /var/lib/insignia/glm53-fp8-g64 2>&1 | grep -E "greedy|top10" | head -3
echo "=== BF16 mode ==="
INSIGNIA_GLM53_VRAM_BUDGET_MB=1024 \
INSIGNIA_GLM53_ROUTE_TRACE=/var/tmp/route-bf16.txt \
INSIGNIA_GLM53_LAYER_DUMP=/var/tmp/layer-dump-bf16.bin \
  "$BIN" "$ROOT" "$IDX" 154820 0 1 2>&1 | grep -E "greedy|top10" | head -3
echo "=== reference ==="
/var/lib/insignia/oracle-venv/bin/python /mnt/e/coding/Insignia/tools/reference_glm53_flash.py \
  /var/lib/insignia/glm53-flash-text --token 154820 --layers 15 --routes 2>&1 | grep "ref route"
