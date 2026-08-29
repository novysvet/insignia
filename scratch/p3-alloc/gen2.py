# P3(b)(d) part 2: fine-grained G(m) at low m, convergence thresholds,
# Dirichlet/EB shrinkage study, LCB rule, record-level q transform for VRAM,
# mirror-vs-disjoint VRAM value.
import numpy as np

rng = np.random.default_rng(23)
d = np.load("anchors.npz")
A_flat, A_geom, A_pack = d["flat"], d["geom"], d["pack"]
blend = lambda A, Bp, w: (1 - w) * A + w * Bp
A_low = np.load("A_low.npy")
PROF = {"flat5.40": A_flat, "hi5.20": blend(A_flat, A_geom, 0.6), "mid5.01": A_geom,
        "lo4.83": blend(A_geom, A_pack, 0.9), "low4.36": A_low}


def run(p, B, m, R, prior=None, kappa=0.0):
    IS = OOS = 0.0
    for _ in range(R):
        c = rng.multinomial(m, p)
        if prior is None or kappa == 0:
            key = c
        else:
            key = c + kappa * prior          # posterior-mean ranking (m+ kappa const)
        idx = np.argpartition(-key, B)[:B]
        IS += c[idx].sum() / m
        OOS += p[idx].sum()
    return IS / R, OOS / R


print("=== fine G(m) at low m, B=8 ===")
fine = [6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80]
for name, p in PROF.items():
    out = []
    for m in fine:
        R = 6000 if m <= 32 else 4000
        ISv, OOSv = run(p, 8, m, R)
        out.append((m, ISv, OOSv, OOSv / ISv))
    print(name)
    for m, ISv, OOSv, G in out:
        mark = " <-- G=0.533" if abs(G - 8 / 15) < 0.03 else ""
        print(f"   m={m:3d} IS={ISv:.4f} OOS={OOSv:.4f} G={G:.3f}{mark}")

print("\n=== thresholds (B=8 and B=57): m where OOS >= 0.95*C, OOS >= 0.99*C (GapO<=1%), GapIs<=1% ===")
MG = [48, 96, 200, 480, 960, 2000, 4800, 9600, 20000, 48000, 96000, 200000]


def threshold(name, p, B):
    C = np.sort(p)[::-1][:B].sum()
    rows = []
    for m in MG:
        R = 3000 if m <= 2000 else (600 if m <= 48000 else 200)
        ISv, OOSv = run(p, B, m, R)
        rows.append((m, ISv, OOSv))
    def solve(pred, better):
        for i, (m, ISv, OOSv) in enumerate(rows):
            if pred(ISv, OOSv):
                if i == 0:
                    return m
                m0, I0, O0 = rows[i - 1]
                # linear interp in log m
                import math
                f = lambda x: better(O0, I0) + (x - math.log(m0)) / (math.log(m) - math.log(m0)) * (better(OOSv, ISv) - better(O0, I0))
                lo, hi = math.log(m0), math.log(m)
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if abs(f(mid)) < 1e-4:
                        break
                    if f(mid) > 0:
                        hi = mid
                    else:
                        lo = mid
                return int(round(math.exp(0.5 * (lo + hi))))
        return None
    t95 = solve(lambda I, O: O >= 0.95 * C, lambda O, I: O - 0.95 * C)
    t99 = solve(lambda I, O: C - O <= 0.01, lambda O, I: 0.01 - (C - O))
    tis = solve(lambda I, O: I - O <= 0.01, lambda O, I: 0.01 - (I - O))
    print(f"{name} B={B}: C={C:.4f}  OOS>=95%C at m={t95} (n={t95 and t95//8} tok)  "
          f"GapO<=1% at m={t99} (n={t99 and t99//8} tok)  GapIs<=1% at m={tis} (n={tis and tis//8} tok)")


for name, p in PROF.items():
    threshold(name, p, 8)
    threshold(name, p, 57)

print("\n=== EB shrinkage toward population-average profile (prior = mid profile), B=57 and B=8 ===")
for name in ["mid5.01", "lo4.83"]:
    p = PROF[name]
    prior = A_geom  # population average shape
    for m in [48, 480, 2000]:
        base = run(p, 57, m, 3000)
        line = [f"{name} m={m:5d} B=57: raw OOS={base[1]:.4f}"]
        for kappa in [4, 12, 40, 120, 400]:
            ISv, OOSv = run(p, 57, m, 3000, prior=prior, kappa=kappa)
            line.append(f"k={kappa}:{OOSv:.4f}")
        print("  " + "  ".join(line))
    for m in [48, 480]:
        base = run(p, 8, m, 3000)
        line = [f"{name} m={m:5d} B=8 : raw OOS={base[1]:.4f}"]
        for kappa in [4, 12, 40, 120]:
            ISv, OOSv = run(p, 8, m, 3000, prior=prior, kappa=kappa)
            line.append(f"k={kappa}:{OOSv:.4f}")
        print("  " + "  ".join(line))

print("\n=== symmetric-Dirichlet alpha demo (ranking unchanged) ===")
c = rng.multinomial(48, PROF["mid5.01"])
for alpha in [0.001, 0.1, 1.0, 10.0]:
    key = c + alpha
    assert np.array_equal(np.argsort(-key), np.argsort(-c))
print("  order identical for all alpha (adding a constant cannot reorder): verified.")

print("\n=== LCB ordering (c - z*sqrt(c)) vs raw at m=48, B=8/57, mid ===")
p = PROF["mid5.01"]
for B in [8, 57]:
    raw = run(p, B, 48, 4000)
    print(f"  B={B}: raw OOS={raw[1]:.4f}", end="")
    for z in [0.5, 1.0]:
        OOS = 0.0
        for _ in range(4000):
            c = rng.multinomial(48, p)
            key = c - z * np.sqrt(np.maximum(c, 0))
            idx = np.argpartition(-key, B)[:B]
            OOS += p[idx].sum()
        print(f"  z={z}: {OOS/4000:.4f}", end="")
    print()

print("\n=== record-level q transform (VRAM value): q_e = 1-(1-8p)^tau, tau=4 ===")
tau = 4.0
for name in ["mid5.01", "lo4.83", "low4.36"]:
    p = PROF[name]
    q = 1 - (1 - 8 * p) ** tau
    qs = np.sort(q)[::-1]
    print(f"  {name}: record-level coverage top-7/layer={qs[:7].sum():.4f} top-8={qs[:8].sum():.4f} "
          f"top-28={qs[:28].sum():.4f} top-57={qs[:57].sum():.4f}")

print("\n=== VRAM mirror value: PCIe saving vs NVMe saving ===")
pcie_ms, nvme_ms = 0.585, 3.67     # per 13.56 MiB record at 23.2 GB/s and 3.7 GB/s
p = PROF["mid5.01"]
q = 1 - (1 - 8 * p) ** tau
qs = np.sort(q)[::-1]
mirror_val = qs[:2].sum() * pcie_ms          # top-2 served from VRAM instead of pinned host
disjoint_val = qs[8:10].sum() * nvme_ms * 0.7  # ranks 9-10 now pinned in VRAM (assume 70% would have missed)
print(f"  mirror (top-2 in VRAM, host pins 1-8): saves {mirror_val:.4f} ms/record-slot-unit")
print(f"  disjoint (VRAM = ranks 9-10 equivalent): saves ~{disjoint_val:.4f} ms  -> {disjoint_val/mirror_val:.1f}x better")
print(f"  (per-layer per-round units; ratio is the point, not the absolute)")
