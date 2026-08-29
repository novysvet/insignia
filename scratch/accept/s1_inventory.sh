#!/bin/bash
# Inventory acceptance artifacts on glm-box (read-only, CPU only).
echo "== dfdump dir =="
ls -la /var/lib/insignia/dfdump/ 2>/dev/null | head -40
echo "== /var/lib/insignia top-level (early/s6/bench) =="
ls -la /var/lib/insignia/ 2>/dev/null | grep -iE "early|s6|bench|dfdump|analysis"
echo "== s6-args =="
ls -la /var/lib/insignia/s6-args/ 2>/dev/null | head -40
echo "== bench-results =="
ls -la /var/lib/insignia/bench-results/ 2>/dev/null | head -40
echo "== bench-data =="
ls -la /var/lib/insignia/bench-data/ 2>/dev/null | head -40
echo "== early-multi files =="
ls -la /var/lib/insignia/early-multi-*.txt 2>/dev/null
for f in /var/lib/insignia/early-multi-df-k7.txt /var/lib/insignia/early-multi-prompt.txt; do
  [ -f "$f" ] && { echo "-- $f (first 40 lines) --"; head -40 "$f"; echo "-- $f (last 30 lines) --"; tail -30 "$f"; }
done
echo "== s6 logs heads =="
for f in /var/lib/insignia/s6-task.log /var/lib/insignia/s6-vramtier.log; do
  [ -f "$f" ] && { echo "-- $f size $(stat -c%s "$f") --"; head -20 "$f"; }
done
