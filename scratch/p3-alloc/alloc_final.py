# P3 final: block-heterogeneous ensemble (21 head/tail pairs, mean = anchor),
# allocation vectors, kappa* theory, honest-prior shrinkage.
import numpy as np

rng = np.random.default_rng(31)
N, L = 288, 42
ANCH = np.array([0.0946, 0.3161, 0.5042, 0.0851])
NB = np.array([1, 7, 20, 260])


def geo(mass, n, top):
    if top <= 0 or top * n <= mass:
        return np.full(n, mass / n)
    lo, hi = 1e-12, 1 - 1e-12
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


def build(Mb):
    b1 = np.array([Mb[0]])
    b2 = np.full(7, Mb[1] / 7)
    b3 = geo(Mb[2], 20, b2[-1])
    b4 = geo(Mb[3], 260, b3[-1])
    p = np.concatenate([b1, b2, b3, b4])
    assert np.all(np.diff(p) <= 1e-12)
    return p


def Hent(p):
    q = np.maximum(p, 1e-300)
    return -np.sum(q * np.log2(q))


# head direction: move mass from blocks 3,4 to 1,2; amplitude halved to the
# monotone-feasible limit (block2-flat must dominate block3-flat at s=1)
dh = np.array([0.055, 0.185, -0.20, -0.04]) * 0.5
P = []
for j in range(21):
    s = (j + 0.5) / 21.0
    Mh = ANCH + s * dh
    Mt = 2 * ANCH - Mh
    assert np.all(Mh > 0) and np.all(Mt > 0), (Mh, Mt)
    P.append(build(Mh))
    P.append(build(Mt))
P = np.array(P)
Hs = np.array([Hent(p) for p in P])
print(f"heterogeneous ensemble: H {Hs.min():.3f}..{Hs.max():.3f} (superseding the band on purpose)")
print(f"mean C1={P[:,0].mean():.5f} C8={P[:,:8].sum(axis=1).mean():.5f} C28={P[:,:28].sum(axis=1).mean():.5f}")
print(f"per-layer C8 spread: {P[:,:8].sum(axis=1).min():.3f}..{P[:,:8].sum(axis=1).max():.3f}; "
      f"C28 spread: {P[:,:28].sum(axis=1).min():.3f}..{P[:,:28].sum(axis=1).max():.3f}")
np.save("ensemble_het.npy", P)


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


for B in [336, 1176, 2425, 3620]:
    blw, cw = waterfill(B)
    ble, ce = equalsplit(B)
    print(f"B={B:5d}: WF={cw:.5f} EQ={ce:.5f} gain=+{100*(cw-ce):.3f} pts")
    if B in (2425, 3620):
        print(f"   WF B_l = {blw.tolist()}")
blw, cw = waterfill(321)
ble, ce = equalsplit(321)
print(f"D=321: global-WF={cw:.5f} per-layer-EQ={ce:.5f} (+{100*(cw-ce):.2f} pts); spread {blw.min()}..{blw.max()}")
print(f"   WF B_l = {blw.tolist()}")

# ---- kappa* theory on representative layers ----
print("\nkappa* = pbar / Var_e(p_e) per layer (MSE-optimal shrinkage pseudo-counts):")
for l in [0, 21, 41]:
    p = P[l]
    pbar = 1.0 / N
    tau2 = np.sum(p * p) / N - pbar ** 2
    print(f"  layer {l} (H={Hent(p):.2f}): sum p^2={np.sum(p*p):.4f}, tau^2={tau2:.2e}, kappa*={pbar/tau2:.1f}")

# ---- honest-prior shrinkage: prior = mean profile of the OTHER 41 layers ----
print("\nhonest EB shrinkage (prior = leave-one-out ensemble mean profile), B=8 and B=57:")
prior_all = P.mean(axis=0)
for l in [0, 21, 41]:
    p = P[l]
    prior = (prior_all * L - p) / (L - 1)
    prior = prior / prior.sum()
    for B in [8, 57]:
        for m in [48, 480, 2000]:
            R = 4000 if m <= 480 else 2000
            raw_IS = raw_OOS = shr_IS = shr_OOS = 0.0
            for _ in range(R):
                c = rng.multinomial(m, p)
                i1 = np.argpartition(-c, B)[:B]
                i2 = np.argpartition(-(c + 20 * prior), B)[:B]
                raw_OOS += p[i1].sum()
                shr_OOS += p[i2].sum()
            print(f"  L{l} B={B} m={m:4d}: raw={raw_OOS/R:.4f}  shr(k=20)={shr_OOS/R:.4f}  "
                  f"oracle={np.sort(p)[::-1][:B].sum():.4f}")
