#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Follow-up computations for the P2/P8 report (second pass numbers)."""
import math
import numpy as np

rng = np.random.default_rng(7)

M, TOPK = 288, 8
Kd = np.array([2, 3, 4, 5], float)
Ud = np.array([14.45, 20.61, 26.40, 31.40])
RHO1 = (16 - 14.45) / 8
PI = TOPK / M
REC_MIB, REC_MIB_PACK = 13.56, 12.75
MIB_GB = 1.048576e-3
W_PER_BW = 42 * REC_MIB * MIB_GB * 1000.0   # ms per record at 1 GB/s

def R_urn(alpha, n):
    b = alpha / M
    return math.exp(math.lgamma(alpha - b + n) - math.lgamma(alpha - b)
                    - math.lgamma(alpha + n) + math.lgamma(alpha))

def dm_U(a, K): return M * (1 - R_urn(a, 8 * K))
def dm_Ux(a, K): return dm_U(a, K) * (8 / dm_U(a, 1))     # U(1)-renormalized

def birthday(N, m, K): return N * (1 - (1 - 1 / N) ** (m * K))

def c1_U(K, u=RHO1):
    v = PI * (1 - u) / (1 - PI)
    return M * (1 - (1 - PI) * (1 - v) ** (K - 1))

def c2_U(K, lam):
    U = 8.0
    for k in range(2, int(K) + 1):
        prod = 1.0
        for l in range(1, k):
            r = RHO1 if l == 1 else PI + (RHO1 - PI) * lam ** (l - 1)
            prod *= (1 - r)
        U += 8 * prod
    return U

def hybrid_flat(K, N, kap):
    q = TOPK / N
    u = q + kap * (1 - q)
    v = q * (1 - u) / (1 - q)
    return N * (1 - (1 - q) * (1 - v) ** (K - 1))

AL_DM, AL_DMX = 90.4, 56.0
LAM_C2, N_E1, KAP_E1 = 0.2766, 95.5, 0.0977
C_PL, B_PL = 8.051, 0.8509

print("=== U(K) per model at K = 6,7,8,16,32,64,128,256,512 ===")
Ks = [6, 7, 8, 16, 32, 64, 128, 256, 512]
rows = {
    'naive':   [birthday(288, 8, k) for k in Ks],
    'a1(76.7)': [birthday(76.7, 8, k) for k in Ks],
    'b DM90':  [dm_U(AL_DM, k) for k in Ks],
    'b* DM56x':[dm_Ux(AL_DMX, k) for k in Ks],
    'c1':      [c1_U(k) for k in Ks],
    'c2(0.277)':[c2_U(k, LAM_C2) for k in Ks],
    'e1(95.5,.098)':[hybrid_flat(k, N_E1, KAP_E1) for k in Ks],
    'pl capped':[min(C_PL * k ** B_PL, 288) for k in Ks],
}
print("K:        " + "".join(f"{k:>8}" for k in Ks))
for n, v in rows.items():
    print(f"{n:14s}" + "".join(f"{x:8.1f}" for x in v))

print()
print("=== typical entropy of persistent Dirichlet(beta per expert) realizations ===")
print("(does the Polya model ALSO reproduce the measured 4.54-5.22 bit access entropy?)")
for alpha in (40, 56, 70, 90, 120):
    beta = alpha / M
    # typical realized p of Dir(beta^288) -- the empirical access freq a long trace sees
    g = rng.gamma(beta, 1.0, size=(4000, M))
    p = g / g.sum(axis=1, keepdims=True)
    H = -(p * np.log2(p)).sum(axis=1)
    E2 = (p ** 2).sum(axis=1)
    print(f"  alpha={alpha:5.1f} beta={beta:.4f}: median H={np.median(H):5.2f} b "
          f"(5-95%: {np.percentile(H,5):.2f}..{np.percentile(H,95):.2f}) "
          f"median 1/sum p^2={np.median(1/E2):6.1f}")

print()
print("=== max U(5) subject to access entropy <= Hmax (two-level marginals) ===")
def u5_H(k, a, Hmax):
    n = M - k
    if n <= 0 or k <= 0: return -1
    b = (8 - k * a) / n
    if not (0 < b <= a <= 1): return -1
    p = np.concatenate([np.full(k, a / 8), np.full(n, b / 8)])
    Hp = float(-(p * np.log2(p)).sum())
    if Hp > Hmax: return -1
    q = np.concatenate([np.full(k, a), np.full(n, b)])
    return float((1 - (1 - q) ** 5).sum()), Hp
for Hmax in (4.54, 5.22, 5.37, 6.5, 7.0):
    best = (-1, None)
    for k in range(1, 120):
        for a in np.linspace(0.05, 1.0, 200):
            r = u5_H(k, a, Hmax)
            if r == -1: continue
            if r[0] > best[0]: best = (r[0], (k, a, r[1]))
    print(f"  Hmax={Hmax:5.2f}: max U(5)={best[0]:6.2f}  at k={best[1][0]}, a={best[1][1]:.3f}, H={best[1][2]:.2f}b"
          if best[1] else f"  Hmax={Hmax}: none")

print()
print("=== flat references ===")
for N in (41.4, 66.0):
    q = 8 / N
    print(f"  flat N={N}: U(5)={N*(1-(1-q)**5):.2f}  E[I]={64/N:.3f}  H={math.log2(N):.2f}b")

print()
print("=== moment-bound extremal (sum q=8, sum q^2=1.544) ===")
best = (-1, None)
for k in range(1, 60):
    n = M - k
    A = k * (1 + k / n); B = -16 * k / n; C = 64 / n - 1.544
    disc = B * B - 4 * A * C
    if disc < 0: continue
    for a in ((-B + math.sqrt(disc)) / (2 * A), (-B - math.sqrt(disc)) / (2 * A)):
        if not (0 <= a <= 1): continue
        b = (8 - k * a) / n
        if not (0 <= b <= 1): continue
        q = np.concatenate([np.full(k, a), np.full(n, b)])
        U5 = float((1 - (1 - q) ** 5).sum())
        p = q / 8
        Hp = float(-(p * np.log2(p)).sum())
        if U5 > best[0]: best = (U5, (k, a, b, Hp))
print(f"  max U(5)={best[0]:.2f} at k={best[1][0]} hot experts q={best[1][1]:.3f}, rest q={best[1][2]:.4f}, "
      f"access H={best[1][3]:.2f} bits")

print()
print("=== d(k) = 42*U(k) verify-union records (context for P1) ===")
for k in (1, 2, 4, 6, 8):
    print(f"  k={k}: DM90 {42*dm_U(AL_DM,k):6.0f}  DM56x {42*dm_Ux(AL_DMX,k):6.0f}  "
          f"c1 {42*c1_U(k):6.0f}  (k*336 naive {k*336})")

print()
print("=== P8: warm-anchored physical model ===")
# reads(T) = U(T) - h(T); chunk(T) = reads*42*REC/BW + f ; ms/token = chunk/T + a
print("implied hits h from the two warm points as function of (f, BW):")
U32, U64 = dm_Ux(AL_DMX, 32), dm_Ux(AL_DMX, 64)
for f in (1.0, 1.6, 2.2):
    for bw in (5.45, 5.8):
        r32 = (11.392 - f) * bw / (42 * REC_MIB * MIB_GB)
        r64 = (11.520 - f) * bw / (42 * REC_MIB * MIB_GB)
        print(f"  f={f:.1f}s BW={bw:.2f}: reads32={r32:6.1f} (h32={U32-r32:6.1f})  "
              f"reads64={r64:6.1f} (h64={U64-r64:6.1f})")

def predict(T, Uk, h, f, bw, a=0.374, pack=False):
    rec = REC_MIB_PACK if pack else REC_MIB
    reads = max(0.0, Uk - h)
    chunk = reads * 42 * rec * MIB_GB / bw + f
    return 1000 * chunk / T + a

U16x, U128x, U256x, U512x = (dm_Ux(AL_DMX, k) for k in (16, 128, 256, 512))
print()
print("banded predictions (ms/token), DM56x central U-model:")
scens = [
    ("optimistic h=82 sat, f=1.0", lambda T: 82, 1.0, 5.8),
    ("central  h(16..512)=0,22,50,65,75, f=1.6", None, 1.6, 5.8),
    ("pessimistic h=0 (cold-tier), f=1.6", lambda T: 0, 1.6, 5.8),
    ("c1-high U, h=82, f=1.6", lambda T: 82, 1.6, 5.8),
]
U_of_T = lambda T: dm_Ux(AL_DMX, T) if T != 16 else U16x
hmap = {16: 0, 32: 22, 64: 40, 128: 50, 256: 65, 512: 75}
print(f"{'scenario':44s} T=16w  T=64   T=128  T=256  T=512")
for name, hf, f, bw in scens:
    vals = []
    for T in (16, 64, 128, 256, 512):
        Uk = c1_U(T) if 'c1-high' in name else U_of_T(T)
        h = hmap[T] if hf is None else (hf(T) if callable(hf) else hf)
        vals.append(predict(T, Uk, h, f, bw))
    print(f"{name:44s}" + "".join(f"{v:7.1f}" for v in vals))
print()
print("central scenario with KDA-fusion (a=0) and packed records (12.75 MiB):")
for T in (64, 128, 256, 512):
    base = predict(T, U_of_T(T), hmap[T], 1.6, 5.8)
    fused = predict(T, U_of_T(T), hmap[T], 1.6, 5.8, a=0.0)
    packd = predict(T, U_of_T(T), hmap[T], 1.6, 5.8, pack=True)
    both = predict(T, U_of_T(T), hmap[T], 1.6, 5.8, a=0.0, pack=True)
    print(f"  T={T:3d}: base={base:6.1f} +fused={fused:6.1f} +packed={packd:6.1f} both={both:6.1f}")

print()
print("T* (model-interior): with m(T)=a+c/T+alpha*T/2, T*=sqrt(2c/alpha):")
for cname, c in (("optimistic c=6.5s", 6500), ("central c=10.4s", 10400), ("pessimistic c=13.2s", 13200)):
    Tstar = math.sqrt(2 * c / 0.00725)
    print(f"  {cname}: T* = {Tstar:,.0f} tokens (attention-quadratic only)")
print("  VRAM staging per layer = U(T)*13.56 MiB;  U(512) DM56x = "
      f"{U512x:.0f} -> {U512x*13.56/1024:.2f} GiB; c1 {c1_U(512):.0f} -> {c1_U(512)*13.56/1024:.2f} GiB")

print()
print("=== P2(2) prefill chunk IO (records read per layer per chunk, band) ===")
print(f"{'K':>5} {'DM56x':>7} {'DM90':>7} {'c2':>7} {'e1':>7} {'pl':>7} {'c1':>7} | {'GiB(42U)@13.56':>15} {'ms/tok@5.8full':>14} {'dedup':>7}")
for K in (8, 16, 32, 64, 128, 256):
    band = [dm_Ux(AL_DMX, K), dm_U(AL_DM, K), c2_U(K, LAM_C2), hybrid_flat(K, N_E1, KAP_E1),
            min(C_PL * K ** B_PL, 288), c1_U(K)]
    gib = max(band) * 42 * REC_MIB / 1024
    mst = W_PER_BW / 5.8 * max(band) / K
    print(f"{K:>5}" + "".join(f"{v:7.1f}" for v in band) + f" | {gib:15.1f} {mst:14.1f} {100*(1-max(band)/(8*K)):6.1f}%")
