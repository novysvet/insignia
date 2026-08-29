#!/usr/bin/env bash
set -u
R=/var/lib/insignia
echo "=== dfdump ls ==="
ls -la "$R/dfdump" | sed -n '1,25p'
echo "=== bench-data tree ==="
ls -laR "$R/bench-data" | sed -n '1,40p'
echo "=== bench-results ==="
ls -la "$R/bench-results"
for d in "$R"/bench-results/*/; do echo "--- $d"; ls "$d" | head -12; done
echo "=== analysis dir ==="
ls -laR "$R/analysis" | sed -n '1,40p'
echo "=== longctx ==="
ls -la "$R/longctx" | head -12
echo "=== binaries ==="
ls -la /var/tmp/insignia-build/glm53-generate /var/tmp/insignia-build-raptor/glm53-generate 2>/dev/null
ls /var/tmp/insignia-build-raptor/ 2>/dev/null | head
echo "=== s6-args / s6-task.log ==="
cat "$R/s6-args" "$R/s6-task.log" 2>/dev/null
echo "=== p2k.csv head ==="
head -c 200 "$R/p2k.csv"; echo; wc -lc "$R/p2k.csv"
echo "=== longtest log head ==="
head -6 "$R/longtest-ctx8192-p2k.csv.log"
echo "=== early-route-math row shape (first row) ==="
head -1 "$R/early-route-math.txt" | cut -c1-140
echo "=== route-realtext first row ==="
head -1 "$R/route-realtext.txt" | cut -c1-140
echo "=== grep run headers in bench-results logs ==="
grep -l "greedy tokens" "$R"/bench-results/*/*.log 2>/dev/null | head
grep -h "^[0-9]*-token prompt" "$R"/bench-results/*/*.log 2>/dev/null | head -12
echo "=== bench-venv python? ==="
ls "$R/bench-venv/bin" 2>/dev/null | grep -E "^python" | head -3
echo "=== GPU procs ==="
pgrep -af glm53 | head -3 || echo none
