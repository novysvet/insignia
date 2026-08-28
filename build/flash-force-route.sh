#!/usr/bin/env bash
# Layer-9 routing sensitivity test: force the reference to each engine's
# expert set and see how much of the layer-9/10 gap the route explains.
set -euo pipefail
PY=/var/lib/insignia/oracle-venv/bin/python
REF=/mnt/e/coding/Insignia/tools/reference_glm53_flash.py
MODEL=/var/lib/insignia/glm53-flash-text

echo "############ FP8 engine route at layer 9 forced into reference"
"$PY" "$REF" "$MODEL" --token 154820 --layers 11 --routes \
  --force-route 9:35,43,63,71,118,125,131,252 \
  --engine-dump /var/tmp/layer-dump-fp8.bin 2>&1 | grep -E "ref route layer 9|layer  9|layer 10|first layer"

echo "############ BF16 engine route at layer 9 forced into reference"
"$PY" "$REF" "$MODEL" --token 154820 --layers 11 --routes \
  --force-route 9:21,63,71,89,125,131,158,252 \
  --engine-dump /var/tmp/layer-dump-bf16.bin 2>&1 | grep -E "ref route layer 9|layer  9|layer 10|first layer"
