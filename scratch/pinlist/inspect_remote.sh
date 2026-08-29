#!/usr/bin/env bash
# inspect_remote.sh - one-shot CPU-only inspection of pin-list artifacts + traces
# on glm-box (Arch WSL). Reads each file once. NEVER runs the engine.
set -euo pipefail
echo "== /var/lib/insignia pin + trace inventory =="
ls -la /var/lib/insignia/ | grep -Ei "pin|route" || true
echo
echo "== pin-realtext.txt: size/lines/bytes =="
for f in /var/lib/insignia/pin-*.txt; do
  [ -f "$f" ] || continue
  wc -lc < "$f" | awk -v f="$f" '{printf "%s: %d lines, %d bytes (CRLF? %s)\n", f, $1, $2, "check-below"}'
  # line-ending check: count CR bytes
  printf 'CR bytes: '; tr -dc '\r' < "$f" | wc -c
done
echo
echo "== pin-realtext.txt: first 12 lines (cat -A to show exact bytes) =="
head -12 /var/lib/insignia/pin-realtext.txt | cat -A
echo
echo "== pin-realtext.txt: last 4 lines =="
tail -4 /var/lib/insignia/pin-realtext.txt | cat -A
echo
echo "== structure audit (single pass over each pin file) =="
for f in /var/lib/insignia/pin-*.txt; do
  [ -f "$f" ] || continue
  awk -v f="$f" '
  {
    nf[NF]++
    if (NF != 3) { bad++ ; next }
    if ($1 < prev) nonmono++
    if ($1 != prev) { if (prev != "") blocks++; cnt[prev]=taken; taken=0 }
    if (seen[$1 "_" $2]++) dup++
    taken++
    prev = $1
    if ($3 > maxhits) maxhits = $3
    if (minl == "" || $1 < minl) minl = $1
    if ($1 > maxl) maxl = $1
    if ($2 > maxe) maxe = $2
    if ($2 < mine || mine == "") mine = $2
  }
  END {
    blocks++
    cnt[prev] = taken
    printf "%s:\n", f
    printf "  field-count histogram:"; for (k in nf) printf " NF=%s x%d", k, nf[k]; printf "\n"
    printf "  layers [%d..%d], experts [%d..%d], max hits value %d\n", minl, maxl, mine, maxe, maxhits
    printf "  layer blocks (contiguous runs): %d, non-monotonic layer drops: %d, dup (layer,expert) lines: %d\n", blocks, nonmono+0, dup+0
    m = 0; M = 999
    for (l in cnt) { if (cnt[l]+0 > m) m = cnt[l]; if (cnt[l]+0 < M) M = cnt[l] }
    printf "  per-block line counts: min=%d max=%d\n", M, m
  }' "$f"
done
echo
echo "== trace files present (no content read) =="
ls -la /var/lib/insignia/route-*.txt /var/lib/insignia/early-route-*.txt 2>/dev/null || true
echo
echo "== analysis caches =="
ls -la /var/lib/insignia/analysis/hotset/ 2>/dev/null || true
