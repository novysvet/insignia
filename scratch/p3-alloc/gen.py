# P3(b): generalization MC.
# For a layer profile p, B: draw counts ~ Mult(m, p):
#   IS(m)   = E[ (sum of top-B counts) / m ]          in-sample coverage (optimistic)
#   OOS(m)  = E[ sum_{e in empirical top-B} p_e ]      fresh-draw coverage
#   C(B)    = sum of true top-B                        oracle
# G = OOS/IS;  GapIs = IS - OOS;  GapO = C - OOS.
# Plus: cross-prompt shift model p' ~ Dirichlet(nu * pbar).
import numpy as np

rng = np.random.default_rng(11)
d = np.load("anchors.npz")
A_flat, A_geom, A_pack = d["flat"], d["geom"], d["pack"]


def Hent(p):
    q = np.maximum(p, 1e-300)
    return -np.sum(q * np.log2(q))


blend = lambda A, Bp, w: (1 - w) * A + w * Bp

# E2-style head-shifted profile (low-H layer: above-average head mass)
Mhead = np.array([0.150, 0.500, 0.310, 0.040])
blocks = [np.full(1, Mhead[0]), A_geom[1:8] * 0 + np.full(7, 0.05),
          None, None]
import numpy as _np
# build head-shifted with geometric shapes per block, anchored like A_geom
b1 = np.array([Mhead[0]])
b2 = _np.full(7, Mhead[1] / 7)
# geometric b3 anchored at p8
def geo(mass, n, top):
    lo, hi = 1e-12, 1 - 1e-12
    for _ in range(100):
        rho = 0.5 * (lo + hi)
        s = top * (1 - rho ** n) / (1 - rho)
        if s > mass:
            hi = rho
        else:
            lo = rho
    rho = 0.5 * (lo + hi)
    v = top * rho ** _np.arange(n)
    return v * (mass / v.sum())


b3 = geo(Mhead[2], 20, b2[-1])
b4 = geo(Mhead[3], 260, b3[-1])
A_low = np.concatenate([b1, b2, b3, b4])

profiles = {
    "flat H=5.40": A_flat,
    "hi  H=5.20": blend(A_flat, A_geom, 0.6),
    "mid H=5.01": A_geom,
    "lo  H=4.83": blend(A_geom, A_pack, 0.9),
    "low H=%.2f" % Hent(A_low): A_low,
}
for k, v in profiles.items():
    assert np.all(np.diff(v) <= 1e-12), k

MG = [48, 96, 200, 480, 960, 2000, 4800, 9600, 20000, 48000, 96000, 200000]


def run(p, B, m, R):
    IS = OOS = 0.0
    for _ in range(R):
        c = rng.multinomial(m, p)
        idx = np.argpartition(-c, B)[:B]
        IS += c[idx].sum() / m
        OOS += p[idx].sum()
    return IS / R, OOS / R


print("=== B=8 (the shipped pin list size) ===")
res8 = {}
for name, p in profiles.items():
    row = []
    for m in MG:
        R = 3000 if m <= 2000 else (600 if m <= 48000 else 200)
        ISv, OOSv = run(p, 8, m, R)
        row.append((m, ISv, OOSv))
    res8[name] = row
    C = np.sort(p)[::-1][:8].sum()
    print(f"{name}: C8={C:.4f}")
    for m, ISv, OOSv in row:
        print(f"   m={m:6d}  IS={ISv:.4f}  OOS={OOSv:.4f}  G={OOSv/max(ISv,1e-9):.3f}  GapIs={ISv-OOSv:.4f}")

print("\n=== B=57 (equal-split slot count at 2425) ===")
res57 = {}
for name, p in profiles.items():
    row = []
    for m in MG:
        R = 3000 if m <= 2000 else (600 if m <= 48000 else 200)
        ISv, OOSv = run(p, 57, m, R)
        row.append((m, ISv, OOSv))
    res57[name] = row
    C = np.sort(p)[::-1][:57].sum()
    print(f"{name}: C57={C:.4f}")
    for m, ISv, OOSv in row:
        print(f"   m={m:6d}  IS={ISv:.4f}  OOS={OOSv:.4f}  GapIs={ISv-OOSv:.4f}  GapO={C-OOSv:.4f}")

# ---- solve: m such that G = 8/15 (implied n from the -15/-8 observation), B=8 ----
print("\n=== implied n: G(m) = 0.5333 at B=8 ===")
for name, row in res8.items():
    prev = None
    for m, ISv, OOSv in row:
        G = OOSv / max(ISv, 1e-9)
        if G >= 8 / 15:
            break
        prev = (m, G)
    print(f"{name}: G crosses 0.5333 between m={prev[0] if prev else 0} and m={m} "
          f"(G at {m}: {G:.3f}) -> n ~ {m // 8} tokens")

# ---- cross-prompt shift ----
print("\n=== cross-prompt shift: p' ~ Dir(nu*p), B=8, mid profile ===")
p = profiles["mid H=5.01"]
for nu in [50, 200, 1000, 10 ** 9]:
    for m in [48, 48000]:
        R = 400
        OOS = 0.0
        conc = nu * p
        for _ in range(R):
            pA = rng.dirichlet(conc)
            pB = rng.dirichlet(conc)
            c = rng.multinomial(m, pA)
            idx = np.argpartition(-c, 8)[:8]
            OOS += pB[idx].sum()
        print(f"  nu={nu:>10}: m={m:6d}  cross-prompt OOS={OOS / R:.4f}  "
              f"(same-prompt OOS would be ~{res8['mid H=5.01'][0 if m == 48 else 7][2]:.4f})")

np.save("A_low.npy", A_low)
