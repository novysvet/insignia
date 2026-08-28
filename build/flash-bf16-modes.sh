#!/usr/bin/env bash
# BF16-mode engine: default resident budget vs 1024 MiB, plus determinism.
set -euo pipefail
BIN=/var/tmp/insignia-build/glm53-generate
ROOT=/var/lib/insignia/glm53-flash-text
IDX=/var/lib/insignia/glm53-flash-text.index
cd /var/tmp

echo "=== BF16 default budget (no VRAM env) ==="
INSIGNIA_GLM53_ROUTE_TRACE=/var/tmp/route-bf16def.txt \
INSIGNIA_GLM53_LAYER_DUMP=/var/tmp/layer-dump-bf16def.bin \
  "$BIN" "$ROOT" "$IDX" 154820 0 1 2>&1 | grep -E "greedy IDs"

echo "=== BF16 1024 MiB budget rerun (determinism) ==="
INSIGNIA_GLM53_VRAM_BUDGET_MB=1024 \
INSIGNIA_GLM53_ROUTE_TRACE=/var/tmp/route-bf16b.txt \
INSIGNIA_GLM53_LAYER_DUMP=/var/tmp/layer-dump-bf16b.bin \
  "$BIN" "$ROOT" "$IDX" 154820 0 1 2>&1 | grep -E "greedy IDs"

md5sum /var/tmp/layer-dump-bf16.bin /var/tmp/layer-dump-bf16b.bin
echo "--- route diffs (empty = identical routing)"
diff /var/tmp/route-bf16.txt /var/tmp/route-bf16b.txt || true
