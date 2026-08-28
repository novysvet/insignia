#!/usr/bin/env bash
# Hang the engine, attach gdb, dump all thread backtraces.
set -uo pipefail
cd /var/tmp
stdbuf -oL -eL env INSIGNIA_GLM53_Q8_BUDGET_MB=10240 INSIGNIA_GLM53_ALT_SHARD_DIR=/stripe \
  /var/tmp/insignia-build/glm53-generate /var/lib/insignia/glm53-flash-text \
  /var/lib/insignia/glm53-flash-text-striped.index 154820 0 12 \
  /var/lib/insignia/glm53-fp8-g64 > /var/tmp/parity-hang.log 2>&1 &
P=$!
echo "engine pid $P"
sleep 40
if ! kill -0 $P 2>/dev/null; then
  echo "engine EXITED; log:"; cat /var/tmp/parity-hang.log; exit 2
fi
gdb -p $P -batch -ex "bt 12" -ex "thread 2" -ex "bt 12" -ex "thread 1" -ex "bt 18" 2>/dev/null | grep -v "^\[" | head -70
kill -9 $P 2>/dev/null
