#!/usr/bin/env bash
# One WSL session: traced 200-token run, routing analysis, 2-bit VQ pilot.
set -uo pipefail
cd /mnt/e/coding/Insignia

echo "=== TRACED 200-TOKEN RUN ==="
INSIGNIA_GLM53_Q8_BUDGET_MB=10240 INSIGNIA_GLM53_ROUTE_TRACE=/var/lib/insignia/route1.txt \
  /var/tmp/insignia-build/glm53-generate \
  /var/lib/insignia/glm53-flash-text /var/lib/insignia/glm53-flash-text.index \
  154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25 0 200 \
  /var/lib/insignia/glm53-fp8-g64 2>&1 | tail -5

echo "=== ROUTE ANALYSIS ==="
python3 tools/glm53_route_analysis.py /var/lib/insignia/route1.txt > /var/lib/insignia/route1-analysis.txt 2>&1
grep -vE "^ *[0-9]+ +[0-9]+ " /var/lib/insignia/route1-analysis.txt | head -40
echo "(per-layer table in /var/lib/insignia/route1-analysis.txt)"

echo "=== 2-BIT VQ PILOT (GPU) ==="
/var/lib/insignia/oracle-venv/bin/python tools/nvfp4_2bit_pilot.py --experts 16 --layers 8 24 \
  > /var/lib/insignia/pilot2bit.txt 2>&1
tail -12 /var/lib/insignia/pilot2bit.txt
