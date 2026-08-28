#!/usr/bin/env bash
# Build a minimal bash-only rootfs tar for wsl --import (scratch distro creation).
set -euo pipefail
rm -rf /tmp/rootfs
mkdir -p /tmp/rootfs/bin /tmp/rootfs/usr/bin /tmp/rootfs/etc /tmp/rootfs/dev /tmp/rootfs/proc /tmp/rootfs/sys /tmp/rootfs/tmp /tmp/rootfs/root
cp /bin/bash /tmp/rootfs/bin/sh
cp /bin/bash /tmp/rootfs/usr/bin/bash
# Resolve bash's shared libraries into the same layout the loader expects.
mkdir -p /tmp/rootfs/usr/lib /tmp/rootfs/lib64
while read -r lib; do
    case "$lib" in
        /*) ;;
        *) continue ;;
    esac
    real=$(readlink -f "$lib")
    dest="/tmp/rootfs${real}"
    mkdir -p "$(dirname "$dest")"
    cp -L "$real" "$dest"
done < <(ldd /bin/bash | awk '{print $3}' | grep '^/')
cp -L /lib64/ld-linux-x86-64.so.2 /tmp/rootfs/lib64/
echo "root:x:0:0::/root:/bin/sh" > /tmp/rootfs/etc/passwd
echo "scratch" > /tmp/rootfs/etc/hostname
cd /tmp/rootfs
tar -cf /mnt/e/wsl2/rootfs.tar .
ls -la /mnt/e/wsl2/rootfs.tar
