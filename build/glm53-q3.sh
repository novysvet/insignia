#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="${INSIGNIA_BUILD_DIR:-/var/tmp/insignia-build-raptor}"
mkdir -p "$out"

exec /opt/cuda/bin/nvcc -ccbin /usr/bin/g++-15 -arch=sm_89 -O3 \
    --use_fast_math -lineinfo -Xptxas=-v -std=c++20 -I"$repo/include" \
    "$repo/src/glm53_q3.cu" "$repo/src/glm53_q3_bench.cu" \
    -o "$out/glm53-q3-bench"
