#!/usr/bin/env bash
# Measure E: write throughput: parallel buffered writers, then sync.
set -e
cd /var/lib/insignia/e2store
rm -f wtest.*
START=$(date +%s.%N)
for i in 1 2 3 4; do
  dd if=/dev/zero of=wtest.$i bs=32M count=128 status=none &
done
wait
MID=$(date +%s.%N)
sync
END=$(date +%s.%N)
python3 -c "print(f'16 GiB buffered parallel write: {16/($MID - $START):.2f} GB/s app, {16/($END - $START):.2f} GB/s with sync')"
rm -f wtest.*
