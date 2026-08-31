#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="${INSIGNIA_BUILD_DIR:-/var/tmp/insignia-build-raptor}"
mkdir -p "$out"

common=(/opt/cuda/bin/nvcc -ccbin /usr/bin/g++-15 -arch=sm_89 -O3
        --use_fast_math -lineinfo -Xptxas=-v -std=c++20 -I"$repo/include")

"${common[@]}" \
    "$repo/src/glm53_q3.cu" "$repo/src/glm53_q3_bench.cu" \
    -o "$out/glm53-q3-bench"

"${common[@]}" \
    "$repo/src/glm53_iq.cu" "$repo/src/glm53_iq_bench.cu" \
    -o "$out/glm53-iq-bench"

exec "${common[@]}" \
    "$repo/src/glm53_iq.cu" "$repo/src/glm53_q6_bench.cu" \
    -o "$out/glm53-q6-bench"
