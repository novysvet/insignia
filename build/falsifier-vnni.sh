#!/usr/bin/env bash
# Build the exact i7-14700KF AVX-VNNI controller ceilings.  Fast math is part
# of the runtime contract: normalization and activation approximations account
# for almost half of the full-pipeline cost.
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="${INSIGNIA_BUILD_DIR:-/var/tmp/insignia-build-raptor}"
mkdir -p "$out"

flags=(
    -O3 -ffast-math -fno-math-errno
    -march=raptorlake -mtune=raptorlake
    -std=c++20 -pthread
)

g++ "${flags[@]}" "$repo/tools/benchmark_falsifier_vnni.cpp" \
    -o "$out/benchmark-falsifier-vnni"
g++ "${flags[@]}" "$repo/tools/benchmark_falsifier_vnni_pipeline.cpp" \
    -o "$out/benchmark-falsifier-vnni-pipeline"
