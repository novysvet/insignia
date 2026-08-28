#!/usr/bin/env bash
# Realistic-prompt DFlash2 acceptance probe (the tools/prompt_math.txt prompt).
P="14572,25,362,8632,18611,220,99590,22,31720,817,6460,13,576,8632,25976,220,99366,4115,817,1899,11,220,21,2849,817,2003,13,2585,1657,31720,1558,432,8192,304,220,19,5555,30,6928,697,32559,3019,553,3019,11,1221,2968,279,1590,4226,624,16127,25"
S=/mnt/c/coding/Insignia-glm53-dflash2/build/bench-df-prompt.sh
echo "==K4 realistic=="
bash "$S" "$P" "${1:-30}" -
echo "==K7 realistic=="
bash "$S" "$P" "${1:-30}" 7
echo "==SCALAR realistic (parity reference)=="
G=/var/tmp/insignia-build/glm53-generate
export INSIGNIA_GLM53_Q8_BUDGET_MB=10240 INSIGNIA_GLM53_EXPERT_CACHE_MB=32768
$G /var/lib/insignia/glm53-flash-text /var/lib/insignia/glm53-flash-text.index "$P" 0 "${1:-30}" /var/lib/insignia/glm53-fp8-g64 2>&1 | grep -E "greedy IDs|prompt .* s"
