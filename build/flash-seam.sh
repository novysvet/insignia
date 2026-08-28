#!/usr/bin/env bash
# Seam bisection: engine sub-op dumps vs reference seams at a given layer.
# Usage: flash-seam.sh <layer> [bf16def|bf16|fp8]
set -euo pipefail
BIN=/var/tmp/insignia-build/glm53-generate
ROOT=/var/lib/insignia/glm53-flash-text
IDX=/var/lib/insignia/glm53-flash-text.index
PY=/var/lib/insignia/oracle-venv/bin/python
REF=/mnt/e/coding/Insignia/tools/reference_glm53_flash.py
LAYER="${1:-0}"
MODE="${2:-bf16def}"
cd /var/tmp
case "$MODE" in
  fp8)    ENVV="INSIGNIA_GLM53_Q8_BUDGET_MB=10240"; EXTRA="/var/lib/insignia/glm53-fp8-g64";;
  bf16)   ENVV="INSIGNIA_GLM53_VRAM_BUDGET_MB=1024"; EXTRA="";;
  bf16def) ENVV="X=1"; EXTRA="";;
  *) echo "bad mode"; exit 64;;
esac
eval "INSIGNIA_GLM53_SEAM_DUMP=/var/tmp/seam-$MODE-L$LAYER.bin \
INSIGNIA_GLM53_SEAM_LAYER=$LAYER $ENVV \
  $BIN $ROOT $IDX 154820 0 1 $EXTRA" 2>&1 | grep -E "greedy IDs" || true
REFLAYER=$((LAYER + 1))
"$PY" "$REF" "$ROOT" --token 154820 --layers "$REFLAYER" \
  --seam-layer "$LAYER" --seam-compare "/var/tmp/seam-$MODE-L$LAYER.bin" 2>&1 | grep -A8 "seam comparison"
