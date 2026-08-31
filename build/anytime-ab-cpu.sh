#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/scratch/anytime-ab}"
TRIALS="${TRIALS:-1000}"
SEED="${SEED:-20260831}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-399}"

mkdir -p "$OUT"
cd "$ROOT"

PYTHONPATH=tools python tools/test_anytime_ab.py -v
python tools/anytime_ab.py protocol \
  --candidate-id example-candidate \
  --output "$OUT/protocol-example.json"
python tools/anytime_ab.py log-template \
  --protocol "$OUT/protocol-example.json" \
  --output "$OUT/pair-log-template.csv" \
  > "$OUT/log-template-output.json"
python tools/anytime_ab.py dp-demo > "$OUT/dp-demo.json"
python tools/simulate_anytime_ab.py simulate \
  --trials "$TRIALS" \
  --seed "$SEED" \
  --bootstrap-reps "$BOOTSTRAP_REPS" \
  --output-dir "$OUT" > "$OUT/run-output.json"
python tools/simulate_anytime_ab.py find-replay-seed \
  --start 0 \
  --limit 1 \
  --true-ratio 1.05 \
  --max-pairs 80 \
  --output "$OUT/replay-three-median-seed-0.csv" \
  > "$OUT/replay-three-median-seed-0.json"

printf 'Wrote CPU artifacts to %s\n' "$OUT"
