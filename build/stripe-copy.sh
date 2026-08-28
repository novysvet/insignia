#!/usr/bin/env bash
# Copy even-ordinal shards (2,4,...,120) of the compact store to the E: vhdx
# for dual-channel expert striping. Byte sizes verified afterwards.
set -euo pipefail
SRC=/var/lib/insignia/glm53-flash-text
DST=/var/lib/insignia/e2store
mkdir -p "$DST"
START=$(date +%s)
for N in $(seq 2 2 120); do
  P=$(printf "model-%05d-of-00120.safetensors" "$N")
  cp -f "$SRC/$P" "$DST/$P" &
  # 4 concurrent writers
  [ $(( (N / 2) % 4 )) -eq 0 ] && wait
done
wait
sync
END=$(date +%s)
echo "copy done in $((END - START))s"
FAIL=0
for N in $(seq 2 2 120); do
  P=$(printf "model-%05d-of-00120.safetensors" "$N")
  S1=$(stat -c%s "$SRC/$P")
  S2=$(stat -c%s "$DST/$P" 2>/dev/null || echo 0)
  if [ "$S1" != "$S2" ]; then echo "SIZE MISMATCH $P: $S1 vs $S2"; FAIL=1; fi
done
[ $FAIL -eq 0 ] && echo "all 60 even shards copied, sizes match"
df -h /var/lib/insignia/e2store
