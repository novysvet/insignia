#!/usr/bin/env bash
# Targeted one-policy on-policy corpus expansion. This is not an ABCD sweep:
# each prompt gets one exact teacher pass and one cache-aware pass, then joins
# the feature trace and logits immediately so alignment failures stop the wave.
set -euo pipefail

REPO=/mnt/c/coding/Insignia-glm53-dflash2
ROOT=/var/lib/insignia/tracecampaign
RESULTS=/var/lib/insignia/bench-results/s10-approx-quality
PY=/var/lib/insignia/bench-venv/bin/python
MANIFEST=$ROOT/prompts/manifest.tsv
IDS=("$@")
(( ${#IDS[@]} )) || IDS=(p01 p03 p09 p11)

for prompt_id in "${IDS[@]}"; do
  prompt=$ROOT/prompts/$prompt_id.csv
  reference=$ROOT/logs/$prompt_id.log
  family=$(awk -F '\t' -v id="$prompt_id" '$1 == id {print $2}' "$MANIFEST")
  label=$prompt_id-cache-onpolicy-wave-20260830a
  out=$RESULTS/$label
  test -n "$family"
  test -f "$prompt"
  test -f "$reference"
  echo "=== $prompt_id ($family) ==="
  INSIGNIA_GLM53_FALSIFIER_TOKENS=32 \
    bash "$REPO/scratch/s10-approx-quality.sh" \
      "$prompt" "$reference" "$label" cache32-r7-e0025
  "$PY" "$REPO/tools/build_falsifier_dataset.py" \
    "$out/cache32-r7-e0025-features.bin" \
    "$out/exact-logits.f32" \
    "$out/cache32-r7-e0025-logits.f32" \
    "$out/cache32-r7-e0025-draft-logits.f32" \
    "$out/cache-onpolicy-dataset.npz" \
    --prompt-id "$prompt_id" --family "$family" --policy cache32-r7-e0025
done

echo "ON_POLICY_RESULTS=$RESULTS"
