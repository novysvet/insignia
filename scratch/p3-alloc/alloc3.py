# P3 ensemble v3 (feasible family).
# Blocks: 1 | 2..8 | 9..28 | 29..288 with anchor masses .0946/.3161/.5042/.0851.
# Profile: block1 = M1; block2,3 share a power-law tilt r^-lh (level per block);
# block4 gets its own tilt r^-lt. Monotonicity caps (lh*, lt*) computed first;
# layer knob t in [0,1] scales both. Max-ent (t=0) is the block-constant step
# (H=5.3969); t=1 is the steepest monotone profile in this family.
import numpy as np

N, L = 288, 42
BLO = np.array([1, 2, 9, 29])
BHI = np.array([1, 8, 28, 288])
ANCH = np.array([0.0946, 0.3161, 0.5042, 0.0851])
rr = np.arange(1, N + 1, dtype=np.float64)


def profile(lh, lt, Mb=None):
    if Mb is None:
        Mb = ANCH
    p = np.empty(N)
    p[0] = Mb[0]
    for b, (lo, hi) in enumerate(zip(BLO[1:], BHI[1:]), start=1):
        seg = rr[lo - 1:hi] ** (-lh if b < 3 else lt)
        p[lo - 1:hi] = Mb[b] * seg / seg.sum()
    return p


def mono_ok(p):
    return np.all(np.diff(p) <= 1e-15)


def Hent(p):
    q = np.maximum(p, 1e-300)
    return -np.sum(q * np.log2(q))


def cap_lam(which, other=0.0):
    lo, hi = 0.0, 6.0
    if which == "h":
        ok = lambda x: mono_ok(profile(x, other))
    else:
        ok = lambda x: mono_ok(profile(other, x))
    assert ok(0.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


lh_max = cap_lam("h", 0.0)
lt_max = cap_lam("lt", lh_max)
print(f"monotone caps: lh* = {lh_max:.4f}, lt* = {lt_max:.4f} (with lh=lh*)")
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    p = profile(t * lh_max, t * lt_max)
    print(f"  t={t:4.2f}: H={Hent(p):.4f}  mono={mono_ok(p)}  C8={p[:8].sum():.5f} C28={p[:28].sum():.5f} "
          f"p2={p[1]:.5f} p8={p[7]:.5f} p9={p[8]:.5f} p29={p[28]:.6f} p57={p[56]:.7f}")
p_min = profile(lh_max, lt_max)
H_FLOOR = Hent(p_min)
print(f"family entropy floor = {H_FLOOR:.4f} bits (band bottom 4.54)")
np.save("p_anchor.npy", profile(0, 0))
np.save("p_floor.npy", p_min)

Hgrid = np.linspace(4.54, 5.22, L)


def build_t(t):
    return profile(t * lh_max, t * lt_max)


# t for a target entropy (monotone decreasing H in t)
def solve_t(Ht):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if Hent(build_t(mid)) > Ht:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


ts = np.array([solve_t(h) for h in Hgrid])
P1 = np.array([build_t(t) for t in ts])
print(f"\nE1: entropy range {min(map(Hent, P1)):.3f}..{max(map(Hent, P1)):.3f}; "
      f"layers clamped at floor: {np.sum(Hgrid < H_FLOOR)}")
print(f"E1 mean C1={P1[:,0].mean():.5f} C8={P1[:,:8].sum(axis=1).mean():.5f} C28={P1[:,:28].sum(axis=1).mean():.5f}")

# ---- E2: block-mass heterogeneity, 21 symmetric pairs, mean = anchor ----
u = np.array([1.0, 1.0, -1.0, -0.33])
u = u / np.linalg.norm(u)


def feasible_amp(a):
    return np.all(ANCH + a * u > 0.002) and np.all(ANCH - a * u > 0.002)


lo, hi = 0.0, 1.0
while hi - lo > 1e-6:
    mid = 0.5 * (lo + hi)
    if feasible_amp(mid):
        lo = mid
    else:
        hi = mid
amp_max = lo
# amp such that head endmember with block-constant shape hits 4.54:
def H_bc(Mb):
    q = np.maximum(Mb, 1e-300)
    return float(np.sum(q * np.log2(np.array([1, 7, 20, 260]) / q)))


lo, hi = 0.0, amp_max
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if H_bc(ANCH + mid * u) > 4.54:
        lo = mid
    else:
        hi = mid
amp = 0.5 * (lo + hi)
print(f"E2 amplitude = {amp:.4f}; head endmember masses = {np.round(ANCH + amp * u, 4)} (H_bc={H_bc(ANCH + amp * u):.3f})")

# per-layer profiles: 21 symmetric pairs. Pair j has shift magnitude s_j rising
# with |H - 4.88| so that low-H layers are head-shifted, high-H tail-shifted,
# and the ensemble mean block masses stay at the anchor.
P2 = []
for j in range(21):
    frac = 1 - (j + 0.5) / 21.0          # 0.976 .. 0.024
    Hlow = 4.54 + frac * (5.22 - 4.54)   # head-shifted member of the pair
    Hhigh = 4.54 + (1 - frac) * (5.22 - 4.54)  # tail-shifted member
    for sgn, Ht in ((+1, Hlow), (-1, Hhigh)):
        scale = (0.5 + 0.5 * abs(Ht - 4.88) / 0.34)
        Mb = np.maximum(ANCH + sgn * amp * scale * u, 1e-4)
        Mb = Mb / Mb.sum()

        def prof2(lh, lt, Mb=Mb):
            p = np.empty(N)
            p[0] = Mb[0]
            for b, (lo_, hi_) in enumerate(zip(BLO[1:], BHI[1:]), start=1):
                seg = rr[lo_ - 1:hi_] ** (-lh if b < 3 else lt)
                p[lo_ - 1:hi_] = Mb[b] * seg / seg.sum()
            return p

        lo_, hi_ = 0.0, 6.0
        for _ in range(50):
            mid = 0.5 * (lo_ + hi_)
            if np.all(np.diff(prof2(mid, 0.0)) <= 1e-15):
                lo_ = mid
            else:
                hi_ = mid
        lhc = lo_
        lo_, hi_ = 0.0, 6.0
        for _ in range(50):
            mid = 0.5 * (lo_ + hi_)
            if np.all(np.diff(prof2(lhc, mid)) <= 1e-15):
                lo_ = mid
            else:
                hi_ = mid
        ltc = lo_
        a, b = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (a + b)
            if Hent(prof2(mid * lhc, mid * ltc)) > Ht:
                a = mid
            else:
                b = mid
        P2.append(prof2(0.5 * (a + b) * lhc, 0.5 * (a + b) * ltc))
P2 = np.array(P2)
H2 = np.array([Hent(p) for p in P2])
print(f"E2: H range {H2.min():.3f}..{H2.max():.3f}; mean block masses "
      f"{[round(float(P2[:, (lo_-1):hi_].sum(axis=1).mean()), 4) for lo_, hi_ in zip(BLO, BHI)]}")
print(f"E2 mean C1={P2[:,0].mean():.5f} C8={P2[:,:8].sum(axis=1).mean():.5f} C28={P2[:,:28].sum(axis=1).mean():.5f}")
np.save("ensemble_E1.npy", P1)
np.save("ensemble_E2.npy", P2)

# ================= allocations =================
def equalsplit(P, B):
    base, extra = divmod(B, L)
    Bl = np.full(L, base)
    Bl[:extra] += 1
    return Bl, float(np.mean(P[:, :base].sum(axis=1))) + (extra / L) * float(np.mean(P[:, base])) if extra else float(np.mean(P[:, :base].sum(axis=1)))


def waterfill(P, B):
    order = np.argsort(-P.ravel(), kind="stable")
    sel = order[:B]
    counts = np.bincount(np.repeat(np.arange(L), N)[sel], minlength=L)
    return counts, float(P.ravel()[sel].sum() / L)


for name, P in [("E1", P1), ("E2", P2)]:
    print(f"\n===== {name} =====")
    for B in [336, 1176, 2425, 3620]:
        blw, cw = waterfill(P, B)
        ble, ce = equalsplit(P, B)
        print(f"  B={B:5d}: WF={cw:.5f} EQ={ce:.5f} gain=+{100*(cw-ce):.3f}pts  B_l {blw.min()}..{blw.max()}")
        if B in (2425, 3620):
            print(f"    WF B_l = {blw.tolist()}")
    blw, cw = waterfill(P, 321)
    ble, ce = equalsplit(P, 321)
    print(f"  VRAM D=321: global-WF={cw:.5f} per-layer-EQ={ce:.5f} (+{100*(cw-ce):.2f} pts); B_l spread {blw.min()}..{blw.max()}")
    order = np.argsort(-P.ravel(), kind="stable")
    mask = np.zeros((L, N), bool)
    mask.ravel()[order[:321]] = True
    mask[:, :8] = True
    jwf = float((P * mask).sum(axis=1).mean())
    mask2 = np.zeros((L, N), bool)
    mask2[:, :8] = True                     # per-layer: VRAM inside top-8 (mirror-style) vs global
    for l in range(321 - 8 * L if 321 > 8 * L else 0):
        pass
    # per-layer VRAM = top-7 everywhere + 27 extra layers get rank 8 (already in pins -> mirror)
    print(f"  joint(VRAM=global-WF, pins=top-8) = {jwf:.5f}  vs per-layer pins alone = {float(P[:, :8].sum(axis=1).mean()):.5f}")
