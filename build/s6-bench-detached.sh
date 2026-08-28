#!/usr/bin/env bash
# Detached benchmark runner: survives ssh drops. Usage (inside Arch WSL):
#   bash build/s6-bench-detached.sh <outdir> <logfile> [extra benchmark_math.py args]
# Watch progress from the dev box:  ssh glm-box "wsl -d Arch -- tail -5 <logfile>"
set -euo pipefail
cd /mnt/c/coding/Insignia-glm53-dflash2
OUT="${1:?output dir required}"
LOG="${2:?log file required}"
shift 2
nohup bash build/s6-bench.sh "$OUT" "$@" >"$LOG" 2>&1 < /dev/null &
echo "started pid $! logging to $LOG"
