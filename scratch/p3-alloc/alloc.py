# P3(a)+(d): 42-layer ensemble from the block-sum anchor + geometric decay knob,
# then water-filling vs equal split at B=2425, B=3620, and D=321 VRAM options.
#
# Anchor (per-layer, in-sample): blocks  r=1 | 2..8 | 9..28 | 29..288
#   masses 0.0946 | 0.3161 | 0.5042 | 0.0851   (sum = 1)
# Max-ent given the 3 coverage numbers = block-constant -> H = 5.396 bits.
# Per-layer knob: within-block centered geometric decay, ratio gamma, solved
# to hit each layer's target entropy in [4.54, 5.22].
import numpy as np

N, L = 288, 42
BLO = np.array([1, 2, 9, 29])
BHI = np.array([1, 8, 28, 288])
BMASS = np.array([0.0946, 0.4107 - 0.0946, 0.9149 - 0.4107, 1 - 0.9149])


def layer_p(gamma):
    p = np.empty(N)
    for lo, hi, m in zip(BLO, BHI, BMASS):
        n = hi - lo + 1
        i = np.arange(n) - (n - 1) / 2.0
        w = gamma ** (-i)
        p[lo - 1:hi] = m * w / w.sum()
    return p


def entropy(p):
    return -np.sum(p * np.log2(p))


gamma_max = 1.30
for gtest in [1.0, 1.01, 1.02, 1.03, 1.04, 1.06]:
    p = layer_p(gtest)
    mono = np.all(np.diff(p) <= 1e-12)
    print(f"gamma={gtest:.2f}: H={entropy(p):.4f} bits, monotone={mono}, "
          f"C8={p[:8].sum():.5f} C28={p[:28].sum():.5f} p57={p[56]:.6f} p29={p[28]:.6f}")
print()

Hgrid_lo, Hgrid_hi = 4.54, 5.22


def solve_gamma(H_target):
    lo, hi = 1.0, gamma_max
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if entropy(layer_p(mid)) > H_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


Htargets = np.linspace(Hgrid_lo, Hgrid_hi, L)
gammas = np.array([solve_gamma(h) for h in Htargets])
P = np.array([layer_p(g) for g in gammas])          # 42 x 288, sorted desc
print(f"gamma range: {gammas.min():.4f} (H=5.22 layer) .. {gammas.max():.4f} (H=4.54 layer)")
print(f"ensemble mean C1={P[:, 0].mean():.5f} C8={P[:, :8].sum(axis=1).mean():.5f} "
      f"C28={P[:, :28].sum(axis=1).mean():.5f}  (targets .0946/.4107/.9149)")
np.save("ensemble_P.npy", P)
np.save("ensemble_H.npy", Htargets)

# ---- water-filling allocation: pool all (layer, rank) marginals ----
allp = P.ravel()                                       # layer-major
layer_id = np.repeat(np.arange(L), N)
order = np.argsort(-allp, kind="stable")


def alloc(Btotal):
    sel = order[:Btotal]
    counts = np.bincount(layer_id[sel], minlength=L)
    cover = allp[sel].sum() / L                        # mean per-layer coverage
    return counts, cover


for Btot in [42 * 8, 42 * 28, 2425, 3620]:
    counts, cover = alloc(Btot)
    eq = Btot // L
    eqc = P[:, :eq].sum() if eq <= N else 1.0
    extra = Btot - eq * L
    eqcov = P[:, :eq].sum(axis=1).mean() + (extra / L) * P[:, extra].mean(axis=0) if extra else eqc.mean() if eq <= N else 1.0
    print(f"B={Btot:5d}: waterfill cov={cover:.5f}  equal-split cov={eqcov:.5f}  "
          f"gain=+{100 * (cover - eqcov):.3f} pts   B_l range {counts.min()}..{counts.max()}")
    if Btot in (2425, 3620):
        print("   B_l vector:", counts.tolist())

print()
# ---- where do water-fill slots come from? correlation with entropy ----
for Btot in [2425, 3620]:
    counts, _ = alloc(Btot)
    print(f"B={Btot}: corr(entropy, B_l) = {np.corrcoef(Htargets, counts)[0, 1]:+.3f}; "
          f"lowest-entropy layer B_l={counts[0]}, highest-entropy layer B_l={counts[-1]}")

print()
# ---- (d) VRAM tier D=321: global-hottest water-fill vs per-layer equal ----
for D in [294, 321, 336, 322]:
    counts, cover = alloc(D)
    per_layer_eq = D // L
    eqcov = P[:, :per_layer_eq].sum(axis=1).mean() + (D - per_layer_eq * L) / L * P[:, D - per_layer_eq * L - 1:].mean(axis=1).mean() if D > per_layer_eq * L else P[:, :per_layer_eq].sum(axis=1).mean()
    print(f"D={D}: global waterfill cov={cover:.5f} vs per-layer top-{per_layer_eq}+spread cov={eqcov:.5f} "
          f"(+{100 * (cover - eqcov):.3f} pts); B_l spread {counts.min()}..{counts.max()}")

print()
# ---- mirror cost: top-2/layer mirrored in VRAM vs disjoint use of those slots ----
top2 = P[:, :2].sum(axis=1).mean()
# disjoint: VRAM global-top-84 + host pins ranks 3..(3+84/42-...) -- simplest: compare
# (i) mirror: VRAM serves top-2 (redundant w/ host pins 1-8)  -> fresh mass in VRAM = top2,
#     host pins add ranks 3..8
# (ii) disjoint: VRAM serves global top-84 (~top-2/layer by rank), host pins serve ranks 3..8 ->
#     same total served = top8. The mirror wastes nothing iff VRAM = subset of pins.
# Real question: with D=321 and pins = top-8/layer, which 321 VRAM slots maximize joint (V ∪ P)?
pins8 = P[:, :8].sum(axis=1).mean()
c_counts, c_cover = alloc(321)
print(f"top-2/layer mass = {top2:.5f}; top-8/layer mass = {pins8:.5f}")
print(f"VRAM(global top-321) ∪ host(top-8): VRAM adds ranks beyond per-layer 8 where it can")
# joint coverage of V=global top-321 plus per-layer top-8 pins:
sel = order[:321]
mask = np.zeros((L, N), bool)
mask.ravel()[sel] = True
mask[:, :8] = True                                     # host pins cover top-8 per layer
joint = (P * mask).sum(axis=1).mean()
# alternative: VRAM = per-layer top-8 (336 > 321, so top-7 + 27 spread): approx via alloc(321) per-layer
mask2 = np.zeros((L, N), bool)
for l in range(L):
    k = 321 // L + (1 if l < 321 % L else 0)
    mask2[l, :k] = True
joint2 = (P * mask2).sum(axis=1).mean()
print(f"joint fresh coverage, V=global-WF(321) + pins(top-8): {joint:.5f}")
print(f"joint fresh coverage, V=per-layer(321) + pins(top-8): {joint2:.5f}")
