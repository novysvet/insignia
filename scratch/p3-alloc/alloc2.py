# P3 ensemble v2: max-ent subject to {C1=.0946, C8=.4107, C28=.9149} plus an
# accelerating-decay moment E[v_r], v_r=r^2 (one knob lam per layer; lam=0 is the
# block-constant max-ent profile H=5.396). Solve lam per layer to hit target H.
# E1: every layer has the anchor block sums; heterogeneity = within-block decay.
# E2: layers also shift block masses along a head<->tail direction (mean = anchor,
#     21 symmetric pairs), decay knob re-solved; heterogeneity = both.
# Then: water-fill vs equal split at B=2425/3620; VRAM D=321 options.
import numpy as np

N, L = 288, 42
BLO = np.array([1, 2, 9, 29])
BHI = np.array([1, 8, 28, 288])
ANCH = np.array([0.0946, 0.4107 - 0.0946, 0.9149 - 0.4107, 1 - 0.9149])
rr = np.arange(1, N + 1, dtype=np.float64)


def profile(lam, Mb=None):
    """max-ent profile with block masses Mb and exp(-lam*r^2) within-block tilt."""
    if Mb is None:
        Mb = ANCH
    w = np.exp(-lam * (rr ** 2) / 288.0)
    p = np.empty(N)
    for lo, hi, m in zip(BLO, BHI, Mb):
        seg = w[lo - 1:hi]
        p[lo - 1:hi] = m * seg / seg.sum()
    return p


def fix_monotone(p):
    """isotonic (non-increasing) projection, then re-impose block sums, iterate."""
    for _ in range(60):
        q = -np.sort(-p)                       # sort descending = isotonic for order stats
        if np.all(np.diff(q) <= 1e-15) and np.all(q >= 0):
            p = q
            break
        p = q
        # rebalance block sums
        for lo, hi, m in zip(BLO, BHI, ANCH):
            s = p[lo - 1:hi].sum()
            if s > 0:
                p[lo - 1:hi] *= m / s
    return p


def Hent(p):
    return -np.sum(p * np.log2(np.maximum(p, 1e-300)))


def build(lam, Mb=None):
    p = profile(lam, Mb)
    if not np.all(np.diff(p) <= 1e-12):
        p = fix_monotone(p)
    return p


def solve_lam(Ht, Mb=None):
    lo, hi = 0.0, 20.0
    # entropy decreasing in lam
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if Hent(build(mid, Mb)) > Ht:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


print("lam sweep (anchor block masses):")
for lam in [0, 0.5, 1, 2, 4, 8]:
    p = build(lam)
    print(f"  lam={lam:5.2f} H={Hent(p):.4f} mono={np.all(np.diff(p) <= 1e-12)} "
          f"C8={p[:8].sum():.5f} C28={p[:28].sum():.5f} p2={p[1]:.5f} p8={p[7]:.5f} p9={p[8]:.5f} p57={p[56]:.6f}")

Hgrid = np.linspace(4.54, 5.22, L)

# ---------------- E1 ----------------
lam1 = np.array([solve_lam(h) for h in Hgrid])
P1 = np.array([build(l) for l in lam1])

# ---------------- E2 ----------------
# head direction: move mass from blocks 3,4 into 1,2 ; u normalized so Mb>=0 for both signs
u = np.array([0.7, 1.0, -1.0, -0.35])
u = u / np.linalg.norm(u)
# choose amplitude so the most head-shifted layer (t=+1) at its natural (low) lam hits ~4.54
# and the most tail-shifted (t=-1) can still reach 5.22 with some lam (decay lowers H).
amp_candidates = np.linspace(0.02, 0.14, 25)
for amp in amp_candidates:
    Mbh = ANCH + amp * u
    Mbtl = ANCH - amp * u
    if np.all(Mbh > 0) and np.all(Mbtl > 0):
        # entropy at lam=0 for head-shifted
        h_head0 = Hent(profile(0.0, Mbh))
        if h_head0 <= 4.54:
            break
amp = amp  # largest feasible head amp whose lam=0 entropy is <= 4.54... keep last
# Actually pick amp such that head-shifted lam=0 lands at 4.54 exactly-ish:
lo, hi = 0.0, 0.2
for _ in range(40):
    mid = 0.5 * (lo + hi)
    if Hent(profile(0.0, ANCH + mid * u)) > 4.54:
        lo = mid
    else:
        hi = mid
amp = 0.5 * (lo + hi)
print(f"\nE2 head-shift amplitude = {amp:.4f}; head endmember block masses = {np.round(ANCH + amp * u, 4)}")
ts = np.concatenate([np.linspace(-1, 1, 21), np.linspace(1, -1, 21)])[:L]  # 21 symmetric pairs (mean 0)
P2 = []
for i, t in enumerate(ts):
    Mb = ANCH + t * amp * u
    Mb = np.maximum(Mb, 1e-4)
    Mb = Mb / Mb.sum() * (ANCH.sum())
    l = solve_lam(Hgrid[i], Mb)
    P2.append(build(l, Mb))
P2 = np.array(P2)
print(f"E2 mean block masses = {np.round(np.array([P2[:, (lo-1):hi].sum(axis=1).mean() for lo, hi in zip(BLO, BHI)]), 5)} (anchor {ANCH})")
print(f"E2 entropy range achieved: {min(Hent(p) for p in P2):.3f}..{max(Hent(p) for p in P2):.3f} (targets 4.54..5.22)")

for name, P in [("E1", P1), ("E2", P2)]:
    print(f"\n===== ensemble {name}: mean C1={P[:,0].mean():.5f} C8={P[:,:8].sum(axis=1).mean():.5f} "
          f"C28={P[:,:28].sum(axis=1).mean():.5f}")
    np.save(f"ensemble_{name}.npy", P)
    order = np.argsort(-P.ravel(), kind="stable")
    lay = np.repeat(np.arange(L), N)
    allp = P.ravel()

    def waterfill(B):
        sel = order[:B]
        return np.bincount(lay[sel], minlength=L), allp[sel].sum() / L

    def equalsplit(B):
        base, extra = divmod(B, L)
        Bl = np.full(L, base)
        Bl[:extra] += 1
        cov = np.mean([P[l, :Bl[l]].sum() for l in range(L)])
        return Bl, cov

    for B in [42 * 8, 42 * 28, 2425, 3620]:
        bl_w, c_w = waterfill(B)
        bl_e, c_e = equalsplit(B)
        print(f"  B={B:5d}: WF cov={c_w:.5f}  EQ cov={c_e:.5f}  gain=+{100*(c_w-c_e):.3f} pts "
              f"({100*(c_w-c_e)/max(1-c_e,1e-9):.2f}% of residual miss)  B_l {bl_w.min()}..{bl_w.max()}")
        if B in (2425, 3620):
            print(f"     WF B_l = {bl_w.tolist()}")
    # correlation
    bl_w, _ = waterfill(2425)
    print(f"  corr(H_l, B_l^WF) = {np.corrcoef(Hgrid, bl_w)[0,1]:+.3f}")

    # ---- VRAM D=321 ----
    for D in [321]:
        bl_w, c_w = waterfill(D)
        bl_e, c_e = equalsplit(D)
        print(f"  VRAM D={D}: global-WF cov={c_w:.5f} vs per-layer-EQ cov={c_e:.5f} "
              f"(+{100*(c_w-c_e):.2f} pts); B_l spread {bl_w.min()}..{bl_w.max()}")
        print(f"     global B_l = {bl_w.tolist()}")

    # joint VRAM(321, global-WF) + host pins top-8/layer
    sel = order[:321]
    mask = np.zeros((L, N), bool)
    mask.ravel()[sel] = True
    mask[:, :8] = True
    joint_wf = (P * mask).sum(axis=1).mean()
    mask2 = np.zeros((L, N), bool)
    for l in range(L):
        mask2[l, :7] = True
    for l in range(321 - 7 * L):
        mask2[l, 7] = True
    mask2[:, :8] = True
    joint_eq = (P * mask2).sum(axis=1).mean()
    # mirror design: VRAM = top-2/layer duplicated from pins; VRAM-slot value wasted = redundancy
    print(f"  joint(V=global-WF 321, pins=top-8) = {joint_wf:.5f}; joint(V=per-layer 321, pins=top-8) = {joint_eq:.5f}")
    top2 = P[:, :2].sum(axis=1).mean()
    print(f"  top-2/layer mass = {top2:.5f}  (the mirrored slots' fresh mass; they displace nothing in pins)")

print("\nflat-tail robustness bound: C(57) <= C28 + 29*dust_avg = "
      f"{0.9149 + 29 * (1 - 0.9149) / 260:.5f}")
