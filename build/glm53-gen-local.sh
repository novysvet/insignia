#!/usr/bin/env bash
# Rebuild glm53-generate for the local Ryzen 5 5600X + RTX 4070 SUPER box.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="${INSIGNIA_BUILD_DIR:-/var/tmp/insignia-build}"
mkdir -p "$out"
exec /opt/cuda/bin/nvcc -ccbin "${NVCC_CCBIN:-/usr/bin/g++-15}" \
    -arch=sm_89 -O3 --use_fast_math -lineinfo \
    -Xcompiler=-pthread -Xcompiler=-march=znver3 -Xcompiler=-mtune=znver3 \
    -std=c++20 -DINSIGNIA_GLM53_NO_MAIN -I"$repo/include" \
    "$repo/src/bf16.cu" "$repo/src/glm53_expert_bench.cu" "$repo/src/glm53_ops.cu" \
    "$repo/src/glm53_q8.cu" "$repo/src/glm53_fp8.cu" "$repo/src/glm53_dflash2.cu" \
    "$repo/src/glm53_logit_metrics.cu" "$repo/src/glm53_generate.cu" \
    "$repo/src/glm53_index.cpp" "$repo/src/glm53_q8_index.cpp" \
    -o "$out/glm53-generate"
