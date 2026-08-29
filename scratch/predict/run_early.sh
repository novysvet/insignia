set -e
cd /var/lib/insignia/analysis/predict
python3 - <<'PYEOF'
import numpy as np

rows = []
for l in open("/var/lib/insignia/early-route-math.txt"):
    f = l.split()
    if len(f) != 19:
        continue
    rows.append((int(f[0]), int(f[1]), int(f[2]),
                 [int(x) for x in f[3:11]], [int(x) for x in f[11:19]]))
print(f"rows={len(rows)}")
ov_field = np.array([r[2] for r in rows], float)
print(f"trace overlap field: mean={ov_field.mean():.3f}/8 -> coverage@8={ov_field.mean()/8:.3f}")

cum = np.zeros(9)
for r in rows:
    P, A = r[3], set(r[4])
    for n in range(1, 9):
        cum[n] += P[n - 1] in A
cum /= len(rows)
assert np.all(np.diff(cum[1:]) >= -1e-12), f"non-monotone cum: {cum}"
print("cumulative hits:", np.round(cum, 3))
for n in (1, 2, 3, 4, 6, 8):
    print(f"  first-{n}: coverage={100*cum[n]/8:.1f}%  overfetch={n/8:.2f}x  precision={100*cum[n]/n:.1f}%")

# per-layer breakdown of top-1 hit rate and set overlap
from collections import defaultdict
lay = defaultdict(lambda: [0, 0, 0])
for r in rows:
    lay[r[1]][0] += (r[3][0] in set(r[4]))
    lay[r[1]][1] += 1
    lay[r[1]][2] += r[2]
stats = {L: (v[0] / v[1], v[2] / v[1] / 8) for L, v in lay.items()}
Ls = sorted(stats)
top1 = np.array([stats[L][0] for L in Ls])
ov8 = np.array([stats[L][1] for L in Ls])
print(f"per-layer top-1 hit: mean={100*top1.mean():.1f}% min={100*top1.min():.1f}% max={100*top1.max():.1f}%")
print(f"per-layer overlap/8: mean={100*ov8.mean():.1f}% min={100*ov8.min():.1f}% max={100*ov8.max():.1f}%")
print("worst layers (top1):", [(int(L), round(float(stats[L][0]), 2)) for L in Ls[:8]])
print("best layers (top1):", [(int(L), round(float(stats[L][0]), 2)) for L in Ls[-8:]])
PYEOF
