#!/usr/bin/env bash
# Compare the two long-context dumps with aligned tokens + PPL.
set -uo pipefail
OUT=/var/lib/insignia/longctx
REPO=/mnt/c/coding/Insignia-glm53-dflash2
grep -oP '(?<=greedy IDs ).*' "$OUT/a8.log" | tr ' ' ',' > "$OUT/gen.csv"
/var/lib/insignia/bench-venv/bin/python - > "$OUT/tok17.csv" <<'EOF'
ids = open("/var/lib/insignia/longctx/ids.csv").read().strip().split(",")
gen = open("/var/lib/insignia/longctx/gen.csv").read().strip().split(",")
print(",".join([ids[-1]] + gen))
EOF
echo "tok17: $(cat "$OUT/tok17.csv")"
echo "== compare (FP8-latent vs FP32-latent) =="
python3 "$REPO/tools/compare_logits.py" "$OUT/a8.f32" "$OUT/b32.f32" --tokens "$OUT/tok17.csv" || true
echo "== PPL FP8-latent =="
python3 "$REPO/tools/ppl.py" "$OUT/a8.f32" "$OUT/tok17.csv"
echo "== PPL FP32-latent =="
python3 "$REPO/tools/ppl.py" "$OUT/b32.f32" "$OUT/tok17.csv"
