#!/usr/bin/env bash
# Final benchmark suite: parity (12 tok), decode 100 tok, prefill.
set -uo pipefail
cd /mnt/e/coding/Insignia
G=/var/tmp/insignia-build/glm53-generate
M=/var/lib/insignia/glm53-flash-text
I=/var/lib/insignia/glm53-flash-text.index
P=/var/lib/insignia/glm53-fp8-g64

echo "=== PARITY 12 (logits tail) ==="
INSIGNIA_GLM53_Q8_BUDGET_MB=10240 $G $M $I 154820 0 12 $P 2>&1 | tail -3
echo "=== DECODE 100 ==="
INSIGNIA_GLM53_Q8_BUDGET_MB=10240 $G $M $I 154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25 0 100 $P 2>&1 | grep -E "greedy IDs|prompt|NVFP4 cache|O_DIRECT" | tail -4
echo "=== PREFILL 16 ==="
INSIGNIA_GLM53_Q8_BUDGET_MB=10240 $G $M $I 154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25 0 1 $P 2>&1 | grep -E "greedy IDs|prompt"
