#!/usr/bin/env bash
# Task-Scheduler entry point (runs under wsl.exe from bench-matrix-task.cmd,
# independent of any ssh session so WSL VM recycles / ssh drops cannot kill
# it). Mirrors build/s6-inner.sh. Reads one line from
# /var/lib/insignia/bench-matrix-args:  <stage> [root]
#   stage: singles | combos | gated | all | listprompts | summarize
set -euo pipefail
read -ra ARGS < /var/lib/insignia/bench-matrix-args
exec bash /mnt/c/coding/Insignia-glm53-dflash2/scratch/bench/bench-matrix.sh "${ARGS[@]}" \
    > /var/lib/insignia/bench-matrix-task.log 2>&1
