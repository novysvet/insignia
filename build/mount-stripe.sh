#!/usr/bin/env bash
# Robustly (re)mount the stripe vhdx by SIZE, not letter (letters shift per boot).
# Usage: mount-stripe.sh [expected_size_G]
set -euo pipefail
EXPECT_G="${1:-100}"
TARGET=""
for D in $(lsblk -dn -o NAME); do
  SZ=$(lsblk -dn -o SIZE "$D" 2>/dev/null | tr -d ' G')
  if [ "${SZ%.*}" = "$EXPECT_G" ]; then
    M=$(findmnt -n -o TARGET "/dev/$D" 2>/dev/null || true)
    if [ -n "$M" ]; then echo "/dev/$D already mounted at $M"; exit 0; fi
    TARGET="$D"
  fi
done
[ -n "$TARGET" ] || { echo "no ${EXPECT_G}G disk found"; lsblk -d -o NAME,SIZE,MODEL; exit 1; }
mkdir -p /stripe
mount "/dev/$TARGET" /stripe
echo "mounted /dev/$TARGET at /stripe"
df -h /stripe | tail -1
