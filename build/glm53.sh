#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/cuda/bin:$PATH"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="${INSIGNIA_BUILD_DIR:-/var/tmp/insignia-build}"
fixture="${1:-/var/lib/insignia/l3e0-gate.ig53}"
mkdir -p "$out"

nvcc -ccbin "${NVCC_CCBIN:-/usr/bin/g++-15}" -arch=sm_89 -O3 --use_fast_math -lineinfo \
    -Xptxas=-v -std=c++20 "$repo/src/smoke.cu" -o "$out/smoke"
nvcc -ccbin "${NVCC_CCBIN:-/usr/bin/g++-15}" -arch=sm_89 -O3 --use_fast_math -lineinfo \
    -Xptxas=-v -std=c++20 -I"$repo/include" "$repo/src/glm53_expert_bench.cu" -o "$out/glm53-expert-bench"
nvcc -ccbin "${NVCC_CCBIN:-/usr/bin/g++-15}" -arch=sm_89 -O3 --use_fast_math -lineinfo \
    -Xptxas=-v -std=c++20 -I"$repo/include" \
    "$repo/src/glm53_ops.cu" "$repo/src/glm53_ops_bench.cu" -o "$out/glm53-ops-bench"
nvcc -ccbin "${NVCC_CCBIN:-/usr/bin/g++-15}" -arch=sm_89 -O3 --use_fast_math -lineinfo \
    -Xptxas=-v -std=c++20 -I"$repo/include" \
    "$repo/src/bf16.cu" "$repo/src/glm53_q8.cu" "$repo/src/glm53_fp8.cu" \
    "$repo/src/glm53_q8_bench.cu" \
    -o "$out/glm53-q8-bench"
g++-15 -O3 -march=znver3 -std=c++20 -I"$repo/include" \
    "$repo/src/glm53_index.cpp" "$repo/src/glm53_index_check.cpp" -o "$out/glm53-index-check"
nvcc -ccbin "${NVCC_CCBIN:-/usr/bin/g++-15}" -arch=sm_89 -O3 --use_fast_math -lineinfo \
    -Xptxas=-v -std=c++20 -DINSIGNIA_GLM53_NO_MAIN -I"$repo/include" \
    "$repo/src/glm53_expert_bench.cu" "$repo/src/glm53_stream_bench.cu" \
    "$repo/src/glm53_index.cpp" -o "$out/glm53-stream-bench"
nvcc -ccbin "${NVCC_CCBIN:-/usr/bin/g++-15}" -arch=sm_89 -O3 --use_fast_math -lineinfo \
    -Xptxas=-v -Xcompiler=-pthread -std=c++20 -DINSIGNIA_GLM53_NO_MAIN -I"$repo/include" \
    "$repo/src/bf16.cu" "$repo/src/glm53_expert_bench.cu" "$repo/src/glm53_ops.cu" \
    "$repo/src/glm53_q8.cu" "$repo/src/glm53_fp8.cu" "$repo/src/glm53_dflash2.cu" \
    "$repo/src/glm53_generate.cu" "$repo/src/glm53_index.cpp" \
    "$repo/src/glm53_q8_index.cpp" -o "$out/glm53-generate"

"$out/smoke"
"$out/glm53-expert-bench" "$fixture"
"$out/glm53-ops-bench"
"$out/glm53-q8-bench"
