#!/usr/bin/env bash
# Budget sweep at layer 0: does mixing resident + streamed tensors inside one
# KDA block corrupt the attention output?
set -euo pipefail
BIN=/var/tmp/insignia-build/glm53-generate
ROOT=/var/lib/insignia/glm53-flash-text
IDX=/var/lib/insignia/glm53-flash-text.index
cd /var/tmp
for B in 8 63 65 67 100 129 200; do
  INSIGNIA_GLM53_VRAM_BUDGET_MB=$B \
  INSIGNIA_GLM53_SEAM_DUMP=/var/tmp/seam-b$B-L0.bin \
  INSIGNIA_GLM53_SEAM_LAYER=0 \
    "$BIN" "$ROOT" "$IDX" 154820 0 1 2>&1 | grep "greedy IDs" | sed "s/^/budget $B: /"
done
