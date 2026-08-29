python3 - <<'PYEOF'
import json
rows = []
for l in open("/var/lib/insignia/early-route-math.txt"):
    f = l.split()
    if len(f) != 19:
        continue
    rows.append((int(f[0]), int(f[1]), int(f[2]),
                 [int(x) for x in f[3:11]], [int(x) for x in f[11:19]]))
R = len(rows)
hits = [0] * 9
for tok, L, ov, P, A in rows:
    act = set(A)
    for n in range(1, 9):
        hits[n] += 1 if P[n - 1] in act else 0
pref = [0] * 9
for n in range(1, 9):
    pref[n] = pref[n - 1] + hits[n]
print("ENGINE-LIVE early_route (pre-attention router hint), MATH-500, 12 tokens, 504 rows:")
out = {}
for n in range(1, 9):
    cov = pref[n] / 8 / R
    pr = pref[n] / n / R
    print(f"  predict-{n}: coverage={100*cov:.1f}%  overfetch={n/8:.2f}x  precision={100*pr:.1f}%")
    out[n] = dict(cov=cov, prec=pr)
from collections import defaultdict
lay = defaultdict(lambda: [0, 0, 0])
for tok, L, ov, P, A in rows:
    act = set(A)
    lay[L][0] += 1 if P[0] in act else 0
    lay[L][1] += 1
    lay[L][2] += ov
Ls = sorted(lay)
top1 = {L: lay[L][0] / lay[L][1] for L in Ls}
ov8 = {L: lay[L][2] / lay[L][1] / 8 for L in Ls}
import statistics as st
print(f"per-layer top-1 hit: mean={100*st.mean(top1.values()):.1f}% "
      f"min={100*min(top1.values()):.1f}% (L{min(top1, key=top1.get)}) "
      f"max={100*max(top1.values()):.1f}% (L{max(top1, key=top1.get)})")
print(f"per-layer overlap/8: mean={100*st.mean(ov8.values()):.1f}% "
      f"min={100*min(ov8.values()):.1f}% (L{min(ov8, key=ov8.get)}) "
      f"max={100*max(ov8.values()):.1f}% (L{max(ov8, key=ov8.get)})")
mla = {L: (top1[L], ov8[L]) for L in Ls if ov8[L] < 0.4}
print("layers with set overlap < 40% (likely MLA full-attention):")
print("  " + ", ".join(f"L{L}: top1={100*a:.0f}% ov8={100*b:.0f}%" for L, (a, b) in sorted(mla.items())))
json.dump({"engine_live_early_route_MATH12": out,
           "per_layer_top1": {str(k): v for k, v in top1.items()},
           "per_layer_ov8": {str(k): v for k, v in ov8.items()}},
          open("/var/lib/insignia/analysis/predict/results_early.json", "w"), indent=1)
print("saved results_early.json")
PYEOF
