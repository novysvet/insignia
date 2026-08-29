#!/usr/bin/env bash
set -u
R=/var/lib/insignia
echo "=== early-multi-prompt stats (single pass) ==="
awk 'NR==1{print "first row fields:", NF; print "header-ish:", substr($0,1,80)}
     {rows++; toksum+=$3; if($3>maxr)maxr=$3; layers[$2]=1; nbatch[$1]=1}
     END{n=0; for(b in nbatch) n++; l=0; for(x in layers) l++;
          printf "rows=%d batches=%d layers=%d sum(tokens)=%d max_row_tokens=%d\n", rows,n,l,toksum,maxr}' "$R/early-multi-prompt.txt"
echo "=== early-multi-df-k7 stats (single pass) ==="
awk 'NR==1{print "first row fields:", NF}
     {rows++; toksum+=$3; if($3>maxr)maxr=$3; layers[$2]=1; nbatch[$1]=1}
     END{n=0; for(b in nbatch) n++; l=0; for(x in layers) l++;
          printf "rows=%d batches=%d layers=%d sum(tokens)=%d max_row_tokens=%d\n", rows,n,l,toksum,maxr}' "$R/early-multi-df-k7.txt"
echo "=== gsm8k main dir ==="
ls -la "$R/bench-data/gsm8k/main/" "$R/bench-data/math500/" "$R/bench-data/math500/test/" 2>/dev/null
echo "=== math500 head ==="
head -c 150 "$R/bench-data/math500/test.jsonl" 2>/dev/null; echo
echo "=== prompt headers from math logs ==="
grep -h "prompt .*s;" "$R"/bench-results/math/*.log 2>/dev/null | head -14
echo "=== s6-baseline/vramtier headers ==="
grep -h "prompt .*s;" "$R"/bench-results/s6-baseline/*.log "$R"/bench-results/s6-vramtier*/**.log 2>/dev/null | head -10
echo "=== which gen counts ==="
grep -h "greedy tokens" "$R"/bench-results/math/gsm8k-2-scalar.log 2>/dev/null | head -2
