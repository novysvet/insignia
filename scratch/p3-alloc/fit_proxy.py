# P3(a): fit 3-param proxy  p_r = w*Zipf(q,s;r<=288) + (1-w)/288  to
#   C1=0.0946, C8=0.4107, C28=0.9149  (288 experts, per layer, in-sample)
# Decoupled solve: for fixed w, fit (q,s) to (C1,C8); bisect w to close C28.
import numpy as np

N = 288
C1, C8, C28 = 0.0946, 0.4107, 0.9149
r = np.arange(1, N + 1, dtype=np.float64)


def zipf(q, s):
    g = (r + q) ** (-s)
    return g / g.sum()


def fit_qs(c1, c8):
    """2D Newton on (log q, log s) so Zipf(q,s) has top1=c1, top8=c8."""
    th = np.array([np.log(2.0), np.log(1.0)])
    for _ in range(300):
        g = zipf(*np.exp(th))
        f = np.array([g[0] - c1, g[:8].sum() - c8])
        if np.max(np.abs(f)) < 1e-14:
            break
        J = np.zeros((2, 2))
        for k in range(2):
            th2 = th.copy()
            th2[k] += 1e-6
            g2 = zipf(*np.exp(th2))
            J[:, k] = np.array([g2[0] - g[0], g2[:8].sum() - g[:8].sum()]) / 1e-6
        step = np.linalg.solve(J, -f)

        def resid(tt):
            gg = zipf(*np.exp(tt))
            return np.array([gg[0] - c1, gg[:8].sum() - c8])

        lam = 1.0
        while lam > 1e-12 and np.max(np.abs(resid(th + lam * step))) > np.max(np.abs(f)):
            lam *= 0.5
        th = th + lam * step
    return np.exp(th)


def mix(w, q, s):
    return w * zipf(q, s) + (1 - w) / N


def c28_of_w(w):
    q, s = fit_qs((C1 - (1 - w) / N) / w, (C8 - 8 * (1 - w) / N) / w)
    p = mix(w, q, s)
    return p[:28].sum(), q, s, p


lo, hi = 0.5, 0.999999   # w must exceed C28-28/288... bracket the root
sol = None
for _ in range(80):
    mid = 0.5 * (lo + hi)
    val, q, s, p = c28_of_w(mid)
    if abs(val - C28) < 1e-13:
        sol = (mid, q, s)
        break
    if val < C28:
        lo = mid
    else:
        hi = mid
    sol = (mid, q, s)

w, q, s = sol
p = mix(w, q, s)
H = -np.sum(p * np.log2(p))
print(f"w = {w:.6f}  q = {q:.6f}  s = {s:.6f}   dust/expert = {(1 - w) / N:.6f}")
print(f"coverage: C1={p[0]:.6f} C8={p[:8].sum():.6f} C28={p[:28].sum():.6f}  (targets .0946/.4107/.9149)")
print(f"implied entropy H = {H:.4f} bits  (band 4.54-5.22)   2^H = {2 ** H:.2f}")
print(f"sum p^2 = {np.sum(p * p):.5f}   2^H2 = {2 ** (-np.log2(np.sum(p * p))):.2f}")
print()
print(" r      p_r        C(r)")
for k in [1, 2, 4, 8, 12, 16, 20, 28, 29, 32, 40, 48, 57, 64, 86, 100, 128, 160, 200, 288]:
    print(f"{k:4d}  {p[k-1]:.6f}  {p[:k].sum():.6f}")
np.save("proxy_p.npy", p)
print("saved proxy_p.npy")
