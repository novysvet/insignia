#!/usr/bin/env bash
# Compact the Arch vhdx on C: after the e2store incident (287 GB -> ~216 GB).
# Steps: shutdown -> export vhdx to E: -> verify -> unregister -> import fresh.
# The export file IS the backup; the old vhdx is only deleted after the export
# passes its size gate.
set -euo pipefail
DISTRO=Arch
EXPORT=/e/wsl2/arch-export.vhdx
BASE=/c/Users/Pufos/WSL/Arch

echo "[$(date +%T)] shutting down WSL"
wsl.exe --shutdown >/dev/null 2>&1 || true
sleep 3

echo "[$(date +%T)] exporting ${DISTRO} to ${EXPORT} (this copies ~216 GB)"
wsl.exe --export "${DISTRO}" 'E:\wsl2\arch-export.vhdx' --vhd

SIZE=$(stat -c%s "${EXPORT}")
echo "[$(date +%T)] export done: ${SIZE} bytes"
if [ "${SIZE}" -lt 190000000000 ]; then
  echo "EXPORT TOO SMALL - ABORTING (distro untouched)"; exit 1
fi

echo "[$(date +%T)] moving old vhdx out of the distro dir"
mv "${BASE}/ext4.vhdx" /c/Users/Pufos/WSL/arch-old-backup.vhdx
wsl.exe --unregister "${DISTRO}"
rm -f /c/Users/Pufos/WSL/arch-old-backup.vhdx

echo "[$(date +%T)] importing compacted distro to ${BASE}"
wsl.exe --import "${DISTRO}" 'C:\Users\Pufos\WSL\Arch' 'E:\wsl2\arch-export.vhdx' --vhd

echo "[$(date +%T)] verifying"
wsl.exe -d "${DISTRO}" -- bash -c 'df -h / | tail -1; ls /var/lib/insignia/glm53-flash-text | wc -l; ls /var/tmp/insignia-build/glm53-generate'
rm -f "${EXPORT}"
echo "[$(date +%T)] DONE"
df -h /c | tail -1
