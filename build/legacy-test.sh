#!/usr/bin/env bash
# Realistic-prompt DFlash2 probe under the LEGACY exact MLA path.
export INSIGNIA_GLM53_MLA_LEGACY=1
P="14572,25,362,8632,18611,220,99590,22,31720,817,6460,13,576,8632,25976,220,99366,4115,817,1899,11,220,21,2849,817,2003,13,2585,1657,31720,1558,432,8192,304,220,19,5555,30,6928,697,32559,3019,553,3019,11,1221,2968,279,1590,4226,624,16127,25"
S=/mnt/c/coding/Insignia-glm53-dflash2/build/bench-df-prompt.sh
echo "==K4 LEGACY=="
bash "$S" "$P" 30 -
echo "==K7 LEGACY=="
bash "$S" "$P" 30 7
