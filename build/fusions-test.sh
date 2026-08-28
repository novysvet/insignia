#!/usr/bin/env bash
# Parity gate for the mhc+rms and fp8-pair fusions, then 100-token timing,
# then the QuIP# lattice pilot.
set -uo pipefail
cd /mnt/e/coding/Insignia
G=/var/tmp/insignia-build/glm53-generate
M=/var/lib/insignia/glm53-flash-text
I=/var/lib/insignia/glm53-flash-text.index
P=/var/lib/insignia/glm53-fp8-g64

echo "=== FUSION PARITY (12 tok, full logits) ==="
INSIGNIA_GLM53_Q8_BUDGET_MB=10240 $G $M $I 154820 0 12 $P 2>&1 | tail -4
echo "=== 100-TOK TIMING ==="
INSIGNIA_GLM53_Q8_BUDGET_MB=10240 $G $M $I 154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25 0 100 $P 2>&1 | grep -E "greedy IDs|prompt|NVFP4 cache" | tail -3
echo "=== PREFILL TIMING ==="
INSIGNIA_GLM53_Q8_BUDGET_MB=10240 $G $M $I 154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25 0 1 $P 2>&1 | grep -E "greedy IDs|prompt"

echo "=== QUIP# PILOT (GPU) ==="
/var/lib/insignia/oracle-venv/bin/python tools/nvfp4_quip_pilot.py --experts 8 \
  > /var/lib/insignia/pilot-quip.txt 2>&1
tail -8 /var/lib/insignia/pilot-quip.txt
