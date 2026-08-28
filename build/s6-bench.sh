#!/usr/bin/env bash
# Session-6 benchmark wrapper. Run inside Arch WSL on glm-box:
#   wsl -d Arch -- bash /mnt/c/coding/Insignia-glm53-dflash2/build/s6-bench.sh <outdir> [extra args]
# Extra args go to tools/benchmark_math.py verbatim (--samples N --generate N --verify-k K ...).
set -euo pipefail
cd /mnt/c/coding/Insignia-glm53-dflash2
PY=/var/lib/insignia/bench-venv/bin/python
BIN=/var/tmp/insignia-build-raptor/glm53-generate
OUT="${1:?output dir required}"
shift
exec "$PY" tools/benchmark_math.py --binary "$BIN" --output "$OUT" "$@"
