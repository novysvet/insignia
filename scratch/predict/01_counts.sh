set -e
cd /var/lib/insignia
python3 - <<'PY'
for f in ["route-realtext.txt","route-campaign.txt","early-route-math.txt"]:
    toks = {}
    nlines = 0
    for line in open(f):
        p = line.split()
        if len(p) < 11: continue
        nlines += 1
        t, L = int(p[0]), int(p[1])
        toks.setdefault(t, set()).add(L)
    ts = sorted(toks)
    layers = toks[ts[0]] if ts else set()
    print(f"{f}: lines={nlines} tokens={len(ts)} token_range=[{ts[0] if ts else '-'},{ts[-1] if ts else '-'}] "
          f"layers_per_token(min/max)={min(len(v) for v in toks.values())}/{max(len(v) for v in toks.values())} "
          f"layer_range={min((min(v) for v in toks.values()), default='-')}..{max((max(v) for v in toks.values()), default='-')}")
PY
echo "=== cct table hexdump head ==="
xxd cct-gsm8k.table | head -3
ls -la /var/lib/insignia/analysis 2>/dev/null || echo "no analysis dir yet"
