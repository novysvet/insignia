#!/usr/bin/env bash
# Run the big-model NumPy reference and compare with engine layer dumps.
set -euo pipefail
PY=/var/lib/insignia/oracle-venv/bin/python
REF=/mnt/e/coding/Insignia/tools/reference_glm53_flash.py
MODEL=/var/lib/insignia/glm53-flash-text
exec "$PY" "$REF" "$MODEL" --token 154820 --layers "${1:-15}" \
  --engine-dump /var/tmp/layer-dump-fp8.bin \
  --engine-dump /var/tmp/layer-dump-bf16.bin
