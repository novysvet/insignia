#!/usr/bin/env bash
# Task-Scheduler entry point (runs under wsl.exe from s6-task.cmd, independent
# of any ssh session so WSL VM recycles / ssh drops cannot kill it).
# Reads one line of benchmark_math.py arguments from /var/lib/insignia/s6-args:
#   <output-dir> [--samples N --generate N --verify-k K ...]
set -euo pipefail
read -ra ARGS < /var/lib/insignia/s6-args
exec bash /mnt/c/coding/Insignia-glm53-dflash2/build/s6-bench.sh "${ARGS[@]}" \
    > /var/lib/insignia/s6-task.log 2>&1
