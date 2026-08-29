#!/usr/bin/env bash
# One-pass read-only inventory of trace artifacts on glm-box (WSL Arch).
set -u
R=/var/lib/insignia

echo "=== top-level ls ==="
ls -la "$R" | sed -n '1,40p'

echo "=== sizes of trace-like files ==="
for f in "$R"/route-*.txt "$R"/early-route-math.txt "$R"/pin-*.txt "$R"/cct-*.table; do
  [ -f "$f" ] && wc -lc "$f"
done 2>/dev/null

stat_one () {  # $1 = trace file: single awk pass -> rows, tokens, layers, expert coverage
  awk -v fn="$1" '
    BEGIN { rows=0; hdr=0; maxtok=-1; mintok=1e18 }
    {
      if (NF < 10) { hdr++; next }
      if ($1 == "token") { hdr++; next }
      rows++
      tok[$1]=1
      if ($1+0 > maxtok) maxtok=$1+0
      if ($1+0 < mintok) mintok=$1+0
      lay[$2]=1
      for (i=3; i<=10; i++) pair[$2 " " $i]=1
      sel[$2 " " $1]=1   # (layer,token) rows sanity
    }
    END {
      ntok=0; for (t in tok) ntok++
      nlay=0; for (l in lay) nlay++
      printf "%s: rows=%d hdr=%d tokens=%d (idx %d..%d) layers=%d distinct(layer,expert)=%d\n", \
             fn, rows, hdr, ntok, mintok, maxtok, nlay, length(pair)
      # per-layer distinct experts (only for the biggest layer list)
      for (l in lay) {
        cnt=0; for (k in pair) { split(k, a, " "); if (a[1]==l) cnt++ }
        printf "  layer %s: %d distinct experts\n", l, cnt
      }
    }' "$1"
}

for f in "$R"/route-realtext.txt "$R"/route-campaign.txt "$R"/early-route-math.txt; do
  [ -f "$f" ] && stat_one "$f"
done

echo "=== pin-realtext head/tail ==="
[ -f "$R/pin-realtext.txt" ] && { head -3 "$R/pin-realtext.txt"; echo ...; wc -l < "$R/pin-realtext.txt"; }

echo "=== dfdump ls ==="
ls -la "$R/dfdump" 2>/dev/null | sed -n '1,25p'

echo "=== bench-data ==="
ls -laR "$R/bench-data" 2>/dev/null | sed -n '1,30p'

echo "=== bench-results ls ==="
ls -la "$R/bench-results" 2>/dev/null | sed -n '1,30p'
for d in "$R"/bench-results/*/; do
  [ -d "$d" ] && { echo "--- $d"; ls "$d" | sed -n '1,15p'; }
done 2>/dev/null

echo "=== binaries ==="
ls -la /var/tmp/insignia-build/glm53-generate /var/tmp/insignia-build-raptor/glm53-generate 2>/dev/null
ls /var/tmp/insignia-build-raptor/ 2>/dev/null | head -20

echo "=== cct table magic ==="
[ -f "$R/cct-gsm8k.table" ] && head -c 16 "$R/cct-gsm8k.table" | od -An -tx1

echo "=== any prompt csv / header files ==="
find "$R" -maxdepth 2 \( -name "*.csv" -o -iname "*prompt*" -o -name "*.tok" \) -printf "%p %s\n" 2>/dev/null | head -15

echo "=== model + tokenizer ==="
ls -la "$R/glm53-flash-text/tokenizer.json" "$R/glm53-flash-text.index" 2>/dev/null

echo "=== running GPU procs (guard sanity) ==="
pgrep -af "glm53|pack_glm53" | head -5 || echo "(none visible from WSL side)"
