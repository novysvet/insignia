#!/usr/bin/env bash
# A/B sequential vs batch verify on the math prompt, k4, striped store.
set -uo pipefail
bash /mnt/e/coding/Insignia/build/ensure-stripe.sh >/dev/null
echo ===SEQ===
INSIGNIA_GLM53_ALT_SHARD_DIR=/stripe \
  INSIGNIA_GLM53_STRIPE_INDEX=/var/lib/insignia/glm53-flash-text-striped.index \
  bash /mnt/e/coding/Insignia/build/bench-df.sh "${1:-100}" "${2:-5120}" 4 DUMMY=1 \
  /var/lib/insignia/glm53-flash-text.index 2>&1 | grep -E 'accepted/round|histogram'
echo ===BATCH===
INSIGNIA_GLM53_ALT_SHARD_DIR=/stripe \
  INSIGNIA_GLM53_STRIPE_INDEX=/var/lib/insignia/glm53-flash-text-striped.index \
  INSIGNIA_GLM53_DF_BATCH_VERIFY=1 \
  bash /mnt/e/coding/Insignia/build/bench-df.sh "${1:-100}" "${2:-5120}" 4 DUMMY=1 \
  /var/lib/insignia/glm53-flash-text.index 2>&1 | grep -E 'accepted/round|histogram'
