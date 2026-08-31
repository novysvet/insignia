#!/usr/bin/env bash
# Start the parity run in background, wait, dump thread stacks and states.
set -uo pipefail
bash /mnt/e/coding/Insignia/build/ensure-stripe.sh >/dev/null
cd /var/tmp
stdbuf -oL -eL env INSIGNIA_GLM53_Q8_BUDGET_MB=10240 INSIGNIA_GLM53_ALT_SHARD_DIR=/stripe \
  INSIGNIA_GLM53_STRIPE_INDEX=/var/lib/insignia/glm53-flash-text-striped.index \
  /var/tmp/insignia-build/glm53-generate /var/lib/insignia/glm53-flash-text \
  /var/lib/insignia/glm53-flash-text.index 154820 0 12 \
  /var/lib/insignia/glm53-fp8-g64 > /var/tmp/parity-hang.log 2>&1 &
P=$!
echo "engine pid $P"
sleep 45
if ! kill -0 $P 2>/dev/null; then
  echo "engine EXITED early; log:"; cat /var/tmp/parity-hang.log; exit 2
fi
echo "=== log so far ==="; tail -5 /var/tmp/parity-hang.log
echo "=== threads ==="
for T in $(ls "/proc/$P/task"); do
  S=$(awk '{print $3}' "/proc/$P/task/$T/stat")
  W=$(cat "/proc/$P/task/$T/wchan")
  echo "$T $S $W"
done | sort | uniq -c | sort -rn
echo "=== stacks (unique) ==="
for T in $(ls "/proc/$P/task"); do
  cat "/proc/$P/task/$T/stack" 2>/dev/null
done | sort | uniq -c | sort -rn | head -30
kill -9 $P 2>/dev/null
