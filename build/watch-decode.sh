#!/usr/bin/env bash
# 100-token decode in background while sampling per-disk read rates.
set -uo pipefail
bash /mnt/e/coding/Insignia/build/ensure-stripe.sh >/dev/null
cd /var/tmp
INSIGNIA_GLM53_Q8_BUDGET_MB=10240 INSIGNIA_GLM53_EXPERT_CACHE_MB=5120 \
INSIGNIA_GLM53_ALT_SHARD_DIR=/stripe \
INSIGNIA_GLM53_STRIPE_INDEX=/var/lib/insignia/glm53-flash-text-striped.index \
/var/tmp/insignia-build/glm53-generate /var/lib/insignia/glm53-flash-text \
/var/lib/insignia/glm53-flash-text.index \
154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25 0 100 \
/var/lib/insignia/glm53-fp8-g64 > /var/tmp/decode-watch.log 2>&1 &
DPID=$!
sleep 12   # skip pin/prefill
bash /mnt/e/coding/Insignia/build/diskwatch.sh 55 3
wait $DPID
echo === decode tail ===
grep -e "tokens total" -e hits /var/tmp/decode-watch.log | tail -2
