# P3 final ensemble + allocation.
# All profiles: monotone (concave cumulative), exact block sums .0946/.3161/.5042/.0851.
#   A_flat : block-constant (max-ent, H=5.3969)
#   A_geom : flat b2, geometric b3 anchored at p8, geometric b4 anchored at p28
#   A_pack : front-packed blocks with continuity chain (cap_k = bottom of block k-1)
# Blends are elementwise mixtures (concave CDF mixtures) -> stay valid.
import numpy as np

N, L = 288, 42
BLO = np.array([1, 2, 9, 29])
BHI = np.array([1, 8, 28, 288])
ANCH = np.array([0.0946, 0.3161, 0.5042, 0.0851])
NB = np.array([1, 7, 20, 260])


def Hent(p):
    q = np.maximum(p, 1e-300)
    return -np.sum(q * np.log2(q))


def geo_block(mass, n, top):
    lo, hi = 1e-12, 1.0 - 1e-12
    for _ in range(100):
        rho = 0.5 * (lo + hi)
        s = top * (1 - rho ** n) / (1 - rho)
        if s > mass:
            hi = rho
        else:
            lo = rho
    rho = 0.5 * (lo + hi)
    v = top * rho ** np.arange(n)
    return v * (mass / v.sum())


def pack(mass, n, cap, floor):
    """front-pack: monotone within block, first element <= cap, last >= floor."""
    v = np.zeros(n)
    rem = mass
    for i in range(n):
        need = rem - floor * (n - 1 - i)
        c = min(cap if i == 0 else v[i - 1], max(need, floor))
        c = max(c, 0.0)
        v[i] = c
        rem -= c
    v[-1] += max(rem, 0.0)
    return v


A_flat = np.concatenate([np.full(n, m / n) for n, m in zip(NB, ANCH)])

p8f = ANCH[1] / 7
b3g = geo_block(ANCH[2], 20, p8f)
b4g = geo_block(ANCH[3], 260, b3g[-1])
A_geom = np.concatenate([[ANCH[0]], np.full(7, p8f), b3g, b4g])


def make_pack(x, y):
    b2 = pack(ANCH[1], 7, 0.0946, x)
    b3 = pack(ANCH[2], 20, b2[-1], y)
    b4 = pack(ANCH[3], 260, b3[-1], 0.0002)
    return np.concatenate([[ANCH[0]], b2, b3, b4])


best = None
for x in np.linspace(0.030, 0.0451, 60):
    for y in np.linspace(0.002, 0.024, 60):
        p = make_pack(x, y)
        if np.all(np.diff(p) <= 1e-15) and abs(p[:8].sum() - 0.4107) < 1e-8:
            if best is None or abs(Hent(p) - 4.54) < abs(Hent(best) - 4.54):
                best = p
A_pack = best
for nm, A in [("flat", A_flat), ("geom", A_geom), ("pack", A_pack)]:
    print(f"A_{nm}: H={Hent(A):.4f} mono={np.all(np.diff(A) <= 1e-15)} "
          f"C8={A[:8].sum():.5f} C28={A[:28].sum():.5f} p2={A[1]:.5f} p8={A[7]:.5f} "
          f"p9={A[8]:.5f} p28={A[27]:.5f} p29={A[28]:.6f} p57={A[56]:.7f}")
np.savez("anchors.npz", flat=A_flat, geom=A_geom, pack=A_pack)


def blend(A, Bp, w):
    return (1 - w) * A + w * Bp          # concave-CDF mixture; block sums preserved


def Hblend(w):
    if w <= 0.5:
        return Hent(blend(A_flat, A_geom, 2 * w))
    return Hent(blend(A_geom, A_pack, 2 * w - 1))


def build_w(Ht):
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if Hblend(mid) > Ht:
            lo = mid
        else:
            hi = mid
    w = 0.5 * (lo + hi)
    return blend(A_flat, A_geom, 2 * w) if w <= 0.5 else blend(A_geom, A_pack, 2 * w - 1)


Hgrid = np.linspace(4.54, 5.22, L)
P = np.array([build_w(h) for h in Hgrid])
print(f"\nensemble: monotone={all(np.all(np.diff(p) <= 1e-12) for p in P)}, "
      f"H {min(map(Hent, P)):.3f}..{max(map(Hent, P)):.3f}; every layer C1/C8/C28 = "
      f"{P[0,0]:.4f}/{P[:,:8].sum(axis=1).mean():.4f}/{P[:,:28].sum(axis=1).mean():.4f}")
mid = P[21]
print("mid layer (H=4.88):")
for k in [1, 2, 4, 8, 9, 12, 20, 28, 29, 40, 57, 86, 120, 200, 288]:
    print(f"  r={k:3d} p={mid[k-1]:.6f} C={mid[:k].sum():.5f}")
np.save("ensemble_final.npy", P)


def equalsplit(B):
    base, extra = divmod(B, L)
    Bl = np.full(L, base)
    Bl[:extra] += 1
    return Bl, float(np.mean([P[l, :Bl[l]].sum() for l in range(L)]))


def waterfill(B):
    order = np.argsort(-P.ravel(), kind="stable")
    sel = order[:B]
    counts = np.bincount(np.repeat(np.arange(L), N)[sel], minlength=L)
    return counts, float(P.ravel()[sel].sum() / L)


print("\n----- allocation -----")
for B in [336, 1176, 2425, 3620]:
    blw, cw = waterfill(B)
    ble, ce = equalsplit(B)
    print(f"B={B:5d}: WF={cw:.5f} EQ={ce:.5f} gain=+{100*(cw-ce):.3f} pts; B_l {blw.min()}..{blw.max()}")
    if B in (2425, 3620):
        print(f"   WF B_l = {blw.tolist()}")
blw, cw = waterfill(321)
ble, ce = equalsplit(321)
print(f"VRAM D=321: global-WF={cw:.5f} per-layer-EQ={ce:.5f} (+{100*(cw-ce):.2f} pts); spread {blw.min()}..{blw.max()}")
print(f"   WF B_l = {blw.tolist()}")
order = np.argsort(-P.ravel(), kind="stable")
mask = np.zeros((L, N), bool)
mask.ravel()[order[:321]] = True
mask[:, :8] = True
print(f"joint(VRAM=global top-321, pins top-8) = {float((P*mask).sum(axis=1).mean()):.5f} "
      f"vs pins alone {float(P[:, :8].sum(axis=1).mean()):.5f}")
