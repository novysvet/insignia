python3 - <<'PYEOF'
rows = []
for l in open("/var/lib/insignia/early-route-math.txt"):
    f = l.split()
    if len(f) != 19:
        continue
    rows.append((int(f[0]), int(f[1]), int(f[2]),
                 [int(x) for x in f[3:11]], [int(x) for x in f[11:19]]))
print("rows =", len(rows))
R = len(rows)
hits = [0] * 9
tot_overlap = 0
mismatch = 0
for tok, L, ov, P, A in rows:
    act = set(A)
    cnt = 0
    for n in range(1, 9):
        h = 1 if P[n - 1] in act else 0
        hits[n] += h
        cnt += h
    tot_overlap += ov
    if cnt != ov:
        mismatch += 1
print("hits[1..8] =", hits[1:])
for n in range(2, 9):
    assert hits[n] >= hits[n - 1], (n, hits)
print("monotone OK")
print("sum computed overlap =", tot_overlap, " vs trace field: field check mismatches =", mismatch)
for n in (1, 2, 3, 4, 6, 8):
    print(f"first-{n}: coverage={100.0*hits[n]/8/R:.1f}%  overfetch={n/8.0:.2f}x  precision={100.0*hits[n]/n/R:.1f}%")
from collections import defaultdict
lay = defaultdict(lambda: [0, 0, 0.0])
for tok, L, ov, P, A in rows:
    act = set(A)
    lay[L][0] += 1 if P[0] in act else 0
    lay[L][1] += 1
    lay[L][2] += ov
Ls = sorted(lay)
top1 = [lay[L][0] / lay[L][1] for L in Ls]
ovm = [lay[L][2] / lay[L][1] / 8.0 for L in Ls]
import statistics as st
print(f"per-layer top-1 hit: mean={100*st.mean(top1):.1f}% min={100*min(top1):.1f}% max={100*max(top1):.1f}%")
print(f"per-layer set overlap/8: mean={100*st.mean(ovm):.1f}% min={100*min(ovm):.1f}% max={100*max(ovm):.1f}%")
print("layer 3..9 top1:", [round(x, 2) for x in top1[:7]])
print("layer 38..44 top1:", [round(x, 2) for x in top1[-7:]])
PYEOF
