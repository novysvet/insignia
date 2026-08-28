#!/usr/bin/env bash
# Ensure /stripe is in fstab (by filesystem LABEL) and mounted.
set -euo pipefail
if ! grep -q "^LABEL=stripe" /etc/fstab; then
  printf 'LABEL=stripe /stripe ext4 defaults,nofail 0 2\n' >> /etc/fstab
  echo "fstab entry added"
fi
mkdir -p /stripe
if ! findmnt -n /stripe >/dev/null; then
  mount /stripe
fi
findmnt -n -o SOURCE /stripe
