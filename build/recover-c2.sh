#!/usr/bin/env bash
# REAL compaction pass: tar-export streams filesystem contents (not the vhdx),
# so the reimported distro only occupies its used ~162 GB.
set -euo pipefail
DISTRO=Arch
TAR=/e/wsl2/arch.tar

echo "[$(date +%T)] shutting down WSL"
wsl.exe --shutdown >/dev/null 2>&1 || true
sleep 3

echo "[$(date +%T)] tar-exporting ${DISTRO} (~162 GB expected)"
wsl.exe --export "${DISTRO}" 'E:\wsl2\arch.tar'

SIZE=$(stat -c%s "${TAR}")
echo "[$(date +%T)] tar done: ${SIZE} bytes"
if [ "${SIZE}" -lt 140000000000 ] || [ "${SIZE}" -gt 220000000000 ]; then
  echo "TAR SIZE OUT OF EXPECTED RANGE - ABORTING (distro untouched)"; exit 1
fi

echo "[$(date +%T)] swapping distros"
mv /c/Users/Pufos/WSL/Arch/ext4.vhdx /c/Users/Pufos/WSL/arch-old2.vhdx
wsl.exe --unregister "${DISTRO}"
rm -f /c/Users/Pufos/WSL/arch-old2.vhdx

wsl.exe --import "${DISTRO}" 'C:\Users\Pufos\WSL\Arch' 'E:\wsl2\arch.tar'

echo "[$(date +%T)] verifying"
wsl.exe -d "${DISTRO}" -- bash -c 'df -h / | tail -1; ls /var/lib/insignia/glm53-flash-text | wc -l; ls -la /var/tmp/insignia-build/glm53-generate'
rm -f "${TAR}"
echo "[$(date +%T)] DONE"
df -h /c | tail -1
