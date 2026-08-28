#!/usr/bin/env bash
set -euo pipefail
PY=/var/lib/insignia/oracle-venv/bin/python
REF=/mnt/e/coding/Insignia/tools/reference_glm53_flash.py
MODEL=/var/lib/insignia/glm53-flash-text
"$PY" "$REF" "$MODEL" --token 154820 --layers 15 \
  --engine-dump /var/tmp/layer-dump-bf16def.bin \
  --save /var/tmp/ref-means.npz 2>&1 | tail -20
