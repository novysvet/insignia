#!/usr/bin/env bash
# Wider budget sweep with layer-0 attention-seam scoring.
set -euo pipefail
BIN=/var/tmp/insignia-build/glm53-generate
ROOT=/var/lib/insignia/glm53-flash-text
IDX=/var/lib/insignia/glm53-flash-text.index
PY=/var/lib/insignia/oracle-venv/bin/python
REF=/mnt/e/coding/Insignia/tools/reference_glm53_flash.py
cd /var/tmp
for B in 130 300 400 460 560 700 900 1024; do
  INSIGNIA_GLM53_VRAM_BUDGET_MB=$B \
  INSIGNIA_GLM53_SEAM_DUMP=/var/tmp/seam-b$B-L0.bin \
  INSIGNIA_GLM53_SEAM_LAYER=0 \
    "$BIN" "$ROOT" "$IDX" 154820 0 1 2>&1 | grep "greedy IDs" | sed "s/^/budget $B: /"
  "$PY" "$REF" "$ROOT" --token 154820 --layers 1 \
    --seam-layer 0 --seam-compare "/var/tmp/seam-b$B-L0.bin" 2>&1 | grep -E "attn-out|ffn-out" | sed "s/^/  b$B /"
done
