#!/usr/bin/env bash
# Sample per-device sectors-read (field 3 of /sys/block/<dev>/stat).
set -uo pipefail
DUR="${1:-30}"
IV="${2:-2}"
declare -A prev_r
for i in $(seq 1 $((DUR / IV))); do
  line=""
  for D in $(lsblk -dn -o NAME); do
    sr=$(awk '{print $3}' "/sys/block/$D/stat" 2>/dev/null) || continue
    [ -n "$sr" ] || continue
    bytes=$((sr * 512))
    if [ -n "${prev_r[$D]:-}" ]; then
      rate=$(( (bytes - prev_r[$D]) / IV / 1024 / 1024 ))
      [ "$rate" -gt 1 ] && line="$line  $D: ${rate} MB/s"
    fi
    prev_r[$D]=$bytes
  done
  [ -n "$line" ] && echo "$(date +%H:%M:%S)$line"
  sleep "$IV"
done
