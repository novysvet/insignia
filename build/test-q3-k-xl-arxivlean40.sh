#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_root="${INSIGNIA_Q3_K_XL_ROOT:-/var/lib/insignia/glm53-q3-k-xl/UD-Q3_K_XL}"
case_dir="${INSIGNIA_Q3_K_XL_CASE_DIR:-/var/lib/insignia/bench-results/q3-k-xl-arxivlean40/arxivlean-40}"
reference_dir="${INSIGNIA_Q3_K_XL_REFERENCE_DIR:-/var/lib/insignia/bench-results/iq3-xxs-arxivlean40/arxivlean-40}"
llama_results="${INSIGNIA_LLAMA_RESULTS:-/mnt/e/coding/quant-lab/llama-build-glm5next/bin/llama-results}"
python="${INSIGNIA_ORACLE_PYTHON:-/var/lib/insignia/oracle-venv/bin/python}"

first_shard="$model_root/GLM-5.3-Flash-UNCENSORED-FP8-UD-Q3_K_XL-00001-of-00004.gguf"
for shard in "$model_root"/*.gguf; do
    test -s "$shard"
done
test "$(find "$model_root" -maxdepth 1 -type f -name '*.gguf' | wc -l)" -eq 4
test -x "$llama_results"
test -x "$python"
test -s "$reference_dir/exact-quality-logits.f32"

install -d "$case_dir"
cp "$reference_dir/prompt.csv" "$reference_dir/prompt.txt" \
   "$reference_dir/forced.csv" "$reference_dir/forced-input.csv" "$case_dir/"

dump="$case_dir/q3-k-xl-quality-logits.f32"
LLAMA_RESULTS_RAW_LOGITS_FILE="$dump" \
LLAMA_RESULTS_TOKEN_IDS_FILE="$case_dir/forced-input.csv" \
LLAMA_RESULTS_LOGITS_LAST=64 \
"$llama_results" \
    -m "$first_shard" -c 1024 -b 1024 -ub 128 \
    -t 12 -tb 12 -ngl 4 -fa on -lm mmap -lzm on -fit on \
    2>&1 | tee "$case_dir/q3-k-xl.log"

test "$(stat -c %s "$dump")" -eq $((64 * 154880 * 4))
"$python" "$repo/tools/compare_logits.py" \
    "$reference_dir/exact-quality-logits.f32" "$dump" \
    --steps 64 --tokens "$case_dir/forced.csv" --cos-threshold 0 \
    --max-ppl-delta 0.035 --allow-top1-mismatch --quiet \
    2>&1 | tee "$case_dir/q3-k-xl-vs-nvfp4-metrics.log"
