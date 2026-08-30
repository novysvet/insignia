#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ARCHIVE=${1:-/var/lib/insignia/fp8-residency-sample-v1.tar.zst}
COLLECTION=${2:-/var/lib/insignia/fp8-residency-sample-v1.collection.json}
OUTPUT=${3:-/var/lib/insignia/fp8-residency-analysis-v1}
BOOTSTRAP=${FP8_ANALYSIS_BOOTSTRAP:-2000}
HOLDOUT=${FP8_ANALYSIS_HOLDOUT:-20}
SEED=${FP8_ANALYSIS_SEED:-5902584}
SLOT_MIB=${FP8_ANALYSIS_SLOT_MIB:-13.5}
MIN_DENSE_MIB=${FP8_ANALYSIS_MIN_DENSE_MIB:-256}

ARCHIVE=$(readlink -f -- "$ARCHIVE")
COLLECTION=$(readlink -f -- "$COLLECTION")
OUTPUT=$(readlink -m -- "$OUTPUT")
RECEIPT="${ARCHIVE}.SHA256SUMS"
BUNDLE="${OUTPUT}.tar.zst"
BUNDLE_RECEIPT="${BUNDLE}.sha256"

[[ -f "$ARCHIVE" ]] || { echo "missing archive: $ARCHIVE" >&2; exit 2; }
[[ -f "$COLLECTION" ]] || { echo "missing collection JSON: $COLLECTION" >&2; exit 2; }
[[ -f "$RECEIPT" ]] || { echo "missing archive receipt: $RECEIPT" >&2; exit 2; }
[[ "$BOOTSTRAP" =~ ^[1-9][0-9]*$ ]] || { echo "invalid FP8_ANALYSIS_BOOTSTRAP" >&2; exit 2; }
[[ "$HOLDOUT" =~ ^[1-9][0-9]*$ ]] || { echo "invalid FP8_ANALYSIS_HOLDOUT" >&2; exit 2; }
(( HOLDOUT < 100 )) || { echo "FP8_ANALYSIS_HOLDOUT must be below 100" >&2; exit 2; }
[[ "$MIN_DENSE_MIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "invalid FP8_ANALYSIS_MIN_DENSE_MIB" >&2; exit 2; }

cd -- "$(dirname -- "$ARCHIVE")"
sha256sum -c -- "$(basename -- "$RECEIPT")"

rm -rf -- "$OUTPUT"
mkdir -p -- "$OUTPUT"

python3 "$ROOT/tools/sample_fp8_cache.py" validate \
  "$ARCHIVE" \
  --minimum-dense-weight-mib "$MIN_DENSE_MIB" \
  --report-json "$OUTPUT/sample-validation.json"

python3 "$ROOT/tools/analyze_fp8_residency.py" \
  "$ARCHIVE" \
  --collection-json "$COLLECTION" \
  --output-dir "$OUTPUT/results" \
  --minimum-dense-weight-mib "$MIN_DENSE_MIB" \
  --holdout-percent "$HOLDOUT" \
  --bootstrap-replicates "$BOOTSTRAP" \
  --bootstrap-seed "$SEED" \
  --allocator-slot-mib "$SLOT_MIB"

cp -- "$COLLECTION" "$OUTPUT/collection.json"
cp -- "$RECEIPT" "$OUTPUT/archive.SHA256SUMS"

{
  printf 'schema\tfp8-residency-analysis-run-v1\n'
  printf 'utc_finished\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repository_head\t%s\n' "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
  printf 'archive\t%s\n' "$ARCHIVE"
  printf 'archive_sha256\t%s\n' "$(sha256sum "$ARCHIVE" | awk '{print $1}')"
  printf 'collection_sha256\t%s\n' "$(sha256sum "$COLLECTION" | awk '{print $1}')"
  printf 'collector_sha256\t%s\n' "$(sha256sum "$ROOT/tools/sample_fp8_cache.py" | awk '{print $1}')"
  printf 'codec_sha256\t%s\n' "$(sha256sum "$ROOT/tools/fp8_residency_codec.py" | awk '{print $1}')"
  printf 'analyzer_sha256\t%s\n' "$(sha256sum "$ROOT/tools/analyze_fp8_residency.py" | awk '{print $1}')"
  printf 'holdout_percent\t%s\n' "$HOLDOUT"
  printf 'bootstrap_replicates\t%s\n' "$BOOTSTRAP"
  printf 'bootstrap_seed\t%s\n' "$SEED"
  printf 'allocator_slot_mib\t%s\n' "$SLOT_MIB"
  printf 'minimum_dense_weight_mib\t%s\n' "$MIN_DENSE_MIB"
  printf 'python\t%s\n' "$(python3 --version 2>&1)"
  printf 'numpy\t%s\n' "$(python3 -c 'import numpy; print(numpy.__version__)')"
  printf 'kernel\t%s\n' "$(uname -srmo)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version \
      --format=csv,noheader | sed 's/^/gpu\t/'
  fi
} > "$OUTPUT/RUN.tsv"

(
  cd -- "$OUTPUT"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

rm -f -- "$BUNDLE" "$BUNDLE_RECEIPT"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
  -C "$(dirname -- "$OUTPUT")" -cf - "$(basename -- "$OUTPUT")" \
  | zstd -19 -T0 -q -o "$BUNDLE"
zstd -tq -- "$BUNDLE"
sha256sum "$BUNDLE" > "$BUNDLE_RECEIPT"

printf 'analysis directory: %s\n' "$OUTPUT"
printf 'result bundle:      %s\n' "$BUNDLE"
printf 'bundle receipt:     %s\n' "$BUNDLE_RECEIPT"
cat -- "$BUNDLE_RECEIPT"
