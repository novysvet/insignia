#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solves P2 (expert-union curve U(K) first-principles model) and P8 (prefill
chunk-size optimum) from audits/s6-open-problems.md, using only the measured
numbers in section 0 of that file.  Pure math + fitting, no repo deps.

Run:  python3 p2_p8_analysis.py   (numpy required, scipy NOT required)
"""
import math
import numpy as np

rng = np.random.default_rng(20260829)

# ----------------------------------------------------------------------------
# Section 0 data
# ----------------------------------------------------------------------------
M    = 288                    # experts per sparse layer
TOPK = 8                      # experts per token
Ndense_layers = 42            # sparse MoE layers
REC_MIB      = 13.56          # NVFP4 expert record size (MiB)
REC_MIB_PACK = 12.75          # packed record size (MiB)
MIB_GB = 1.048576e-3          # MiB -> GB (decimal GB, bandwidths are GB/s)

Kd = np.array([2, 3, 4, 5], dtype=float)      # fit targets
Ud = np.array([14.45, 20.61, 26.40, 31.40])   # measured U(K)
I_ADJ = 0.193 * 8                             # measured |S_t n S_{t+1}| = 1.544
RHO1  = (16.0 - 14.45) / 8.0                  # = 0.19375, exact from U(2)
PI    = TOPK / M                              # 1/36 stationary per-expert rate

ENT_LO, ENT_HI = 4.54, 5.22                   # measured per-layer routing entropy
# static hot-set coverage (per layer, fraction of accesses)
COV1, COV8, COV28 = 0.0946, 0.4107, 0.9149

def rel(pred):
    pred = np.asarray(pred, dtype=float)
    return (pred - Ud) / Ud

def rmsr(pred):
    r = rel(pred)
    return float(np.sqrt(np.mean(r * r)))

# ----------------------------------------------------------------------------
# 1-D / 2-D fitters (golden section + pattern search; no scipy)
# ----------------------------------------------------------------------------
def fit1(f, lo, hi, iters=200):
    """minimize f over [lo,hi] (unimodal enough in practice)."""
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = f(d)
    x = 0.5 * (a + b)
    return x, f(x)

def fit2(f, x0, y0, span0=(1.0, 1.0), shrink=0.5, rounds=60, step0=None):
    """crude 2-D pattern search minimizing f(x,y)."""
    x, y = x0, y0
    best = f(x, y)
    sx = span0[0] if step0 is None else step0
    sy = span0[1] if step0 is None else step0
    for _ in range(rounds):
        improved = False
        for dx, dy in ((sx,0),(-sx,0),(0,sy),(0,-sy),(sx,sy),(sx,-sy),(-sx,sy),(-sx,-sy)):
            xn, yn = x + dx, y + dy
            v = f(xn, yn)
            if np.isfinite(v) and v < best:
                best, x, y = v, xn, yn
                improved = True
        if not improved:
            sx *= shrink; sy *= shrink
            if sx < 1e-9 and sy < 1e-9:
                break
    return (x, y), best

# ============================================================================
# MODEL (a)  birthday family
# ============================================================================
def birthday(N, m, K):
    """with-replacement birthday: E[U] = N(1-(1-1/N)^{mK})."""
    if N <= 1: return np.inf
    return N * (1.0 - (1.0 - 1.0 / N) ** (m * K))

def birthday_exact_uniform(K):
    """uniform marginal, 8 DISTINCT experts per token: E[U]=288(1-(1-8/288)^K)."""
    return M * (1.0 - (1.0 - TOPK / M) ** K)

naive  = [birthday(M, TOPK, k) for k in Kd]
naiveX = [birthday_exact_uniform(k) for k in Kd]

aN, _   = fit1(lambda N: rmsr([birthday(N, TOPK, k) for k in Kd]), 8.5, 5000)
aNm, _  = fit2(lambda N, m: rmsr([birthday(N, m, k) for k in Kd]),
               60.0, 7.0, span0=(20.0, 2.0), shrink=0.6, rounds=200)
# keep m<=8, N sane
a1 = [birthday(aN, TOPK, k) for k in Kd]
a2 = [birthday(aNm[0], aNm[1], k) for k in Kd]

# ============================================================================
# MODEL (b)  Polya urn / Dirichlet-multinomial (symmetric, persistent p)
#   P(expert never drawn in n=8K with-replacement urn draws):
#     R(n) = prod_{j<n} (a-b+j)/(a+j),  b = a/M
# ============================================================================
def R_urn(alpha, n):
    b = alpha / M
    return math.exp(math.lgamma(alpha - b + n) - math.lgamma(alpha - b)
                    - math.lgamma(alpha + n) + math.lgamma(alpha))

def dm_U(alpha, K):
    return M * (1.0 - R_urn(alpha, 8 * K))

# alpha that exactly matches the adjacent pairwise overlap E[I]=1.544:
# pairwise slot collision q = (alpha+M)/(M(alpha+1));  E[I] = 8*... derived:
# 16 - E[U(2)] = E[I]  -> solve numerically
al_pw, _ = fit1(lambda a: (M * (1 - R_urn(a, 16)) - (16 - I_ADJ)) ** 2, 0.05, 5000)
al_ls, _ = fit1(lambda a: rmsr([dm_U(a, k) for k in Kd]), 0.5, 20000)
b_pw = [dm_U(al_pw, k) for k in Kd]
b_ls = [dm_U(al_ls, k) for k in Kd]
# (b*) renormalized so U(1)=8 exactly (compensates with-replacement defect)
scale = 8.0 / dm_U(al_ls, 1)
al_ls2, _ = fit1(lambda a: rmsr([dm_U(a, k) * (8.0 / dm_U(a, 1)) for k in Kd]), 0.5, 20000)
b_lsx = [dm_U(al_ls2, k) * (8.0 / dm_U(al_ls2, 1)) for k in Kd]

# ============================================================================
# MODEL (c)  per-expert 2-state Markov chains (set-overlap process)
# ============================================================================
def c1_U(K, u=RHO1):
    v = PI * (1 - u) / (1 - PI)
    return M * (1 - (1 - PI) * (1 - v) ** (K - 1))

def c2_U(K, lam):
    """lag-decay chain: rho_1 fixed by U(2); rho_l = pi + (rho1-pi)*lam^(l-1)."""
    U = 8.0
    for k in range(2, int(K) + 1):
        prod = 1.0
        for l in range(1, k):
            r = RHO1 if l == 1 else PI + (RHO1 - PI) * lam ** (l - 1)
            prod *= (1.0 - r)
        U += 8.0 * prod            # increment = new experts at token k
    return U

c1 = [c1_U(k) for k in Kd]
clam, _ = fit1(lambda l: rmsr([c2_U(k, l) for k in Kd]), 1e-4, 0.999)
c2 = [c2_U(k, clam) for k in Kd]

# empirical lag-profile implied by the increments (diagnostic inversion):
# r(k) = 8(1-prod_{l<k}(1-rho_l))  ->  rho_{k-1} = 1 - prod_{l<k}(1-rho_l)/prod_{l<k-1}(1-rho_l)
inc = np.diff(np.concatenate([[8.0], Ud]))          # new experts at token k=2..5
r_meas = 8.0 - inc                                   # repeats at k=2..5
rho_emp = []
prev = 1.0
for k in range(2, 6):
    cur = 1.0 - r_meas[k - 2] / 8.0                  # prod_{l<k}(1-rho_l)
    rho_emp.append(1.0 - cur / prev)
    prev = cur

# ============================================================================
# MODEL (d)  exchangeable tokens, skewed marginal (no temporal correlation)
# ============================================================================
def H_bits(p):
    p = np.asarray(p, float); p = p / p.sum()
    return float(-(p * np.log2(p)).sum())

def zipf_p(s, n=M):
    i = np.arange(1, n + 1, dtype=float)
    w = i ** (-s)
    return w / w.sum()

s_lo, _ = fit1(lambda s: (H_bits(zipf_p(s)) - ENT_LO) ** 2, 1e-3, 2.0)
s_hi, _ = fit1(lambda s: (H_bits(zipf_p(s)) - ENT_HI) ** 2, 1e-3, 2.0)

# anchor marginal: piecewise-geometric through measured cumulative coverage
def anchor_marginal():
    cum = {0: 0.0, 1: COV1, 8: COV8, 28: COV28, 288: 1.0}
    edges = sorted(cum)
    q = np.zeros(M)
    for na, nb in zip(edges[:-1], edges[1:]):
        cnt = nb - na
        mass = cum[nb] - cum[na]
        # geometric decay over 0-indexed experts na..nb-1 carrying that mass;
        # ratio heuristic from neighboring segment densities (continuity)
        lo_d = cum[na] / na if na > 0 else 1.0
        hi_d = (1 - cum[nb]) / max(1, M - nb)
        rr = math.sqrt(max(lo_d, 1e-9) / max(hi_d, 1e-9)) if nb < M else 1.0
        rr = min(max(rr, 1e-3), 1e3)
        g = np.geomspace(rr, 1.0, cnt)
        q[na:nb] = g / g.sum() * mass
    return q / q.sum()

def gumbel_topk_tokens(p, ntok, k=TOPK):
    """boolean membership matrix (ntok, M): exact weighted sample w/o replacement."""
    out = np.zeros((ntok, M), dtype=bool)
    B = 20000
    lp = np.log(p)
    for s in range(0, ntok, B):
        e = rng.gumbel(size=(min(B, ntok - s), M))
        keys = lp[None, :] + e
        idx = np.argpartition(keys, -k, axis=1)[:, -k:]
        rows = np.arange(idx.shape[0])[:, None]
        out[s:s + idx.shape[0]][rows, idx] = True
    return out

def mc_union_stats(mem, Ks):
    ntok = mem.shape[0]
    res = {}
    for K in Ks:
        nw = ntok // K
        if nw < 50: continue
        w = mem[:nw * K].reshape(nw, K, M).any(axis=1)
        res[K] = float(w.sum(axis=1).mean())
    adj = mem[:-1] & mem[1:]
    res['I'] = float(adj.sum(axis=1).mean())
    res['q2'] = float((mem.mean(axis=0) ** 2).sum())
    res['H'] = H_bits(mem.mean(axis=0))
    return res

NTOK = 400_000
p_anchor = anchor_marginal()
marginals = {
    'flat H=4.54 (N=23.46)': None,   # analytic
    'flat H=5.22 (N=37.20)': None,   # analytic
    'Zipf H=4.54': zipf_p(s_lo),
    'Zipf H=5.22': zipf_p(s_hi),
    'anchor(hot-set)': p_anchor,
}
mc_stats = {}
mem_anchor = None
for name, p in marginals.items():
    if p is None: continue
    mem = gumbel_topk_tokens(p, NTOK)
    mc_stats[name] = mc_union_stats(mem, list(range(1, 9)) + [16, 32, 64, 128, 256])
    if name == 'anchor(hot-set)':
        mem_anchor = mem            # kept for hybrid (e2) only
    del mem

def flat_U(N, K):
    q = TOPK / N
    return N * (1.0 - (1.0 - q) ** K)

for N, name in ((2 ** ENT_LO, 'flat H=4.54 (N=23.46)'), (2 ** ENT_HI, 'flat H=5.22 (N=37.20)')):
    mc_stats[name] = {'U': {K: flat_U(N, K) for K in range(1, 9)},
                      'I': N * (TOPK / N) ** 2, 'q2': 64.0 / N, 'H': math.log2(N)}

# moment bound: max U(5) over ALL marginals with sum q=8, sum q^2 = 1.544
def u5_two_level(k, stotal=8.0, s2=I_ADJ):
    # k experts at level a, 288-k at level b
    n = M - k
    if n <= 0 or k <= 0: return -1
    # k a + n b = S ; k a^2 + n b^2 = S2
    # b = (S - k a)/n ; k a^2 + (S-k a)^2/n = S2
    # k a^2 (1 + k/n) - 2 S k a / n + S^2/n - S2 = 0
    A = k * (1 + k / n); B = -2 * stotal * k / n; C = stotal ** 2 / n - s2
    disc = B * B - 4 * A * C
    if disc < 0: return -1
    best = -1
    for a in ((-B + math.sqrt(disc)) / (2 * A), (-B - math.sqrt(disc)) / (2 * A)):
        if 0 <= a <= 1:
            b = (stotal - k * a) / n
            if 0 <= b <= 1:
                q = np.concatenate([np.full(k, a), np.full(n, b)])
                best = max(best, float((1 - (1 - q) ** 5).sum()))
    return best

mom_bound = max(u5_two_level(k) for k in range(0, 40))

# ============================================================================
# MODEL (e)  hybrid: wide marginal + adjacent stickiness (per-expert chains)
# ============================================================================
def hybrid_U(K, q, kappa):
    q = np.asarray(q, float)
    u = q + kappa * (1 - q)
    v = q * (1 - u) / (1 - q)
    return float((1 - (1 - q) * (1 - v) ** (K - 1)).sum())

# (e1) flat marginal N + kappa
def e1_err(N, kap):
    if N < 8.01 or not (0 <= kap < 1): return np.inf
    q = np.full(int(round(N)), TOPK / N)
    return rmsr([hybrid_U(k, q, kap) for k in Kd])
(Ne, ke), _ = fit2(lambda N, kap: e1_err(N, kap), 66.0, 0.05, span0=(16.0, 0.04), shrink=0.6, rounds=300)
q_flat = np.full(int(round(Ne)), TOPK / Ne)
e1 = [hybrid_U(k, q_flat, ke) for k in Kd]

# (e2) anchor marginal + kappa (q = MC inclusion probs)
q_anchor = mem_anchor.mean(axis=0)
ke2, _ = fit1(lambda kap: rmsr([hybrid_U(k, q_anchor, kap) for k in Kd]), -0.5, 0.99)
e2 = [hybrid_U(k, q_anchor, ke2) for k in Kd]

# ============================================================================
# EMPIRICAL power-law union
# ============================================================================
def powlaw_fit(Kfit, Ufit):
    x, y = np.log(Kfit), np.log(Ufit)
    b, a = np.polyfit(x, y, 1)
    return math.exp(a), b
c_pl, b_pl = powlaw_fit(Kd, Ud)
pl = [c_pl * k ** b_pl for k in Kd]
# out-of-sample version fit on K=2..4
c_pl3, b_pl3 = powlaw_fit(Kd[:3], Ud[:3])

# ============================================================================
# out-of-sample check: fit on K=2..4, predict K=5
# ============================================================================
oos = {}
oos['(a1) N_eff']   = fit1(lambda N: np.sqrt(np.mean(((np.array([birthday(N, TOPK, k) for k in Kd[:3]]) - Ud[:3]) / Ud[:3]) ** 2)), 8.5, 5000)[0]
oos['(a1) U(5)'] = birthday(oos['(a1) N_eff'], TOPK, 5)
oos['(b) DM'] = fit1(lambda a: np.sqrt(np.mean(((np.array([dm_U(a, k) for k in Kd[:3]]) - Ud[:3]) / Ud[:3]) ** 2)), 0.5, 20000)[0]
oos['(b) U(5)'] = dm_U(oos['(b) DM'], 5)
oos['(c2) lam'] = fit1(lambda l: np.sqrt(np.mean(((np.array([c2_U(k, l) for k in Kd[:3]]) - Ud[:3]) / Ud[:3]) ** 2)), 1e-4, 0.999)[0]
oos['(c2) U(5)'] = c2_U(5, oos['(c2) lam'])
oos['(e1) N,kap'] = fit2(lambda N, kap: e1_err(N, kap), 66.0, 0.05, span0=(16.0, 0.04), shrink=0.6, rounds=300)[0]
oos['(e1) U(5)'] = hybrid_U(5, np.full(int(round(oos['(e1) N,kap'][0])), TOPK / oos['(e1) N,kap'][0]), oos['(e1) N,kap'][1])
oos['(pl) U(5)'] = c_pl3 * 5 ** b_pl3
oos['(c1) U(5)'] = c1_U(5)

# ============================================================================
# extrapolations U(6..8) + bootstrap bands
# ============================================================================
def preds_for(name):
    if name == 'naive':   return lambda k: birthday(M, TOPK, k)
    if name == 'a1':      return lambda k: birthday(aN, TOPK, k)
    if name == 'a2':      return lambda k: birthday(aNm[0], aNm[1], k)
    if name == 'b_pw':    return lambda k: dm_U(al_pw, k)
    if name == 'b_ls':    return lambda k: dm_U(al_ls, k)
    if name == 'b_lsx':   return lambda k: dm_U(al_ls2, k) * (8.0 / dm_U(al_ls2, 1))
    if name == 'c1':      return lambda k: c1_U(k)
    if name == 'c2':      return lambda k: c2_U(k, clam)
    if name == 'e1':      return lambda k: hybrid_U(k, q_flat, ke)
    if name == 'e2':      return lambda k: hybrid_U(k, q_anchor, ke2)
    if name == 'pl':      return lambda k: min(c_pl * k ** b_pl, M)
    raise KeyError(name)

sig = np.array([0.06, 0.15, 0.20, 0.25])   # assumed CV of measured U(2..5)
NB = 400
boot = {}
for name in ('a1', 'b_ls', 'b_lsx', 'c2', 'e1', 'pl'):
    out = []
    for _ in range(NB):
        Ud_b = Ud * (1 + rng.normal(0, 1) * sig)
        if name == 'a1':
            N, _ = fit1(lambda N: np.sqrt(np.mean(((np.array([birthday(N, TOPK, k) for k in Kd]) - Ud_b) / Ud_b) ** 2)), 8.5, 5000)
            out.append(birthday(N, TOPK, 8))
        elif name == 'b_ls':
            a, _ = fit1(lambda a: np.sqrt(np.mean(((np.array([dm_U(a, k) for k in Kd]) - Ud_b) / Ud_b) ** 2)), 0.5, 20000)
            out.append(dm_U(a, 8))
        elif name == 'b_lsx':
            a, _ = fit1(lambda a: np.sqrt(np.mean(((np.array([dm_U(a, k) * (8 / dm_U(a, 1)) for k in Kd]) - Ud_b) / Ud_b) ** 2)), 0.5, 20000)
            out.append(dm_U(a, 8) * (8 / dm_U(a, 1)))
        elif name == 'c2':
            l, _ = fit1(lambda l: np.sqrt(np.mean(((np.array([c2_U(k, l) for k in Kd]) - Ud_b) / Ud_b) ** 2)), 1e-4, 0.999)
            out.append(c2_U(8, l))
        elif name == 'e1':
            (Nb, kb), _ = fit2(lambda N, kap: (lambda: np.inf)() if (N < 8.01 or not (0 <= kap < 1))
                               else np.sqrt(np.mean(((np.array([hybrid_U(k, np.full(int(round(N)), TOPK / N), kap) for k in Kd]) - Ud_b) / Ud_b) ** 2)),
                               66.0, 0.05, span0=(16.0, 0.04), shrink=0.7, rounds=120)
            out.append(hybrid_U(8, np.full(int(round(Nb)), TOPK / Nb), kb))
        else:
            cc, bb = powlaw_fit(Kd, Ud_b)
            out.append(min(cc * 8 ** bb, M))
    boot[name] = (np.percentile(out, 5), np.median(out), np.percentile(out, 95))

# ============================================================================
# P2(2): U(K) for K=6..256 and prefill chunk IO
# ============================================================================
Kbig = [6, 7, 8, 16, 32, 64, 128, 256]
models_big = {n: [preds_for(n)(k) for k in Kbig] for n in
              ('a1', 'b_ls', 'b_lsx', 'c1', 'c2', 'e1', 'pl')}

# ============================================================================
# P8: prefill chunk-size fit
# ============================================================================
T_MEAS = np.array([16.0, 32.0, 64.0])
MS_MEAS = np.array([700.0, 356.0, 180.0])     # ms/token (T=16 cold)
chunk_ms = T_MEAS * MS_MEAS                   # per-chunk wall

def ms_model(T, a, b, w, Uk):
    return a + b / T + w * Uk / T

# w <-> BW_eff:  IO ms/token = w*U/T with w = 42*REC GB / BW * 1000
W_PER_BW = Ndense_layers * REC_MIB * MIB_GB * 1000.0    # = 597.18 ms per record @1GB/s

def exact3(Ufun):
    """exact 3-parameter solve of the 3 measured points. returns a,b,w"""
    U3 = np.array([Ufun(T) for T in T_MEAS])
    A = np.column_stack([np.ones(3), 1.0 / T_MEAS, U3 / T_MEAS])
    sol, *_ = np.linalg.lstsq(A, MS_MEAS, rcond=None)
    return sol, U3

# anchored 2-point fits on the warm points (32, 64):
def anchored(a, U32, U64):
    w = (128.0 - 32.0 * a) / (U64 - U32) if U64 != U32 else np.inf
    b = 32.0 * (356.0 - a) - w * U32
    return b, w

# KDA launch constant: 2 launches/token/KDA-layer, 34 KDA layers, 3-8 us each
KDA_LO, KDA_HI = 34 * 2 * 3e-3, 34 * 2 * 8e-3    # ms/token

# attention (within-chunk quadratic) coefficient from P12 numbers:
# 1.9e14 FLOP/token at 256K ctx -> 7.24e8 FLOP per position; assume 100 TFLOPs
ALPHA_ATTN = 1.9e14 / 262144 / 1e14 * 1e3        # ms per token per position = 0.00724

# constant-reads calibration from the ~constant per-chunk wall:
BW_LIST = (5.8, 5.45, 4.75, 3.7)

# ----------------------------------------------------------------------------
# printing
# ----------------------------------------------------------------------------
def line(char='='):  print(char * 100)

line()
print("P2 / P8 analysis -- data: audits/s6-open-problems.md section 0")
print(f"measured U(K): K=1..5 -> 8, 14.45, 20.61, 26.40, 31.40;  E[I_adj]={I_ADJ}")
print(f"increments (new experts/token, k=2..5): {np.round(inc,3)}")
print(f"repeats r(k)=8-inc: {np.round(r_meas,3)}  -> empirical rho_l: {np.round(rho_emp,4)}")
line('-')
print(f"naive birthday 288(1-(1-1/288)^8K):      {np.round(naive,3)}   rmsRel={rmsr(naive):.4f}")
print(f"uniform distinct 288(1-(1-8/288)^K):     {np.round(naiveX,3)}   rmsRel={rmsr(naiveX):.4f}")
print(f"(a1) N_eff={aN:.1f}:                     {np.round(a1,3)}   rmsRel={rmsr(a1):.4f}")
print(f"(a2) N={aNm[0]:.1f}, m/token={aNm[1]:.3f}:      {np.round(a2,3)}   rmsRel={rmsr(a2):.4f}")
print(f"(b) DM alpha(pairwise, matches U(2))={al_pw:.3f}: {np.round(b_pw,3)}  rmsRel={rmsr(b_pw):.4f}  U(1)={dm_U(al_pw,1):.2f}")
print(f"(b) DM alpha(LS)={al_ls:.1f}:                    {np.round(b_ls,3)}  rmsRel={rmsr(b_ls):.4f}  U(1)={dm_U(al_ls,1):.2f}")
print(f"(b*) DM renorm alpha={al_ls2:.1f}:               {np.round(b_lsx,3)}  rmsRel={rmsr(b_lsx):.4f}  U(1)=8")
v_c1 = PI * (1 - RHO1) / (1 - PI)
print(f"(c1) chain (0 free params, lam={RHO1 - v_c1:.4f}):    {np.round(c1,3)}  rmsRel={rmsr(c1):.4f}")
print(f"(c2) chain lam={clam:.4f}:               {np.round(c2,3)}  rmsRel={rmsr(c2):.4f}")
print(f"(e1) hybrid flat N={Ne:.1f}, kappa={ke:.4f}:     {np.round(e1,3)}  rmsRel={rmsr(e1):.4f}")
print(f"(e2) hybrid anchor kappa={ke2:.4f}:              {np.round(e2,3)}  rmsRel={rmsr(e2):.4f}")
print(f"(pl) power law U={c_pl:.3f}*K^{b_pl:.4f}: {np.round(pl,3)}  rmsRel={rmsr(pl):.4f}")
line('-')
print("MODEL (d) exchangeable skewed marginal (MC, 400k tokens, exact w/o replacement):")
for name, st in mc_stats.items():
    u = st.get('U', st)
    if isinstance(u, dict):
        uv = [u[k] for k in range(2, 6)]
    else:
        uv = [mc_stats[name][k] if k in mc_stats[name] else None for k in range(2, 6)]
    print(f"  {name:24s} H={st['H']:.2f}b E[I]={st['I']:.3f} sum q^2={st['q2']:.3f} "
          f"U(2..5)={np.round(uv,2)} rmsRel={rmsr(uv):.4f}")
print(f"  Zipf exponents: s(H=4.54)={s_lo:.3f}  s(H=5.22)={s_hi:.3f}")
print(f"  moment bound: max U(5) over ALL marginals with sum q=8, sum q^2=1.544: {mom_bound:.2f}"
      f"  (measured U(5)=31.40 -> shortfall >= {31.40-mom_bound:.2f})")
line('-')
print("out-of-sample (fit K=2..4, predict K=5; measured 31.40):")
for k, v in oos.items():
    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
line('-')
print("extrapolations U(6), U(7), U(8):")
for name in ('a1','a2','b_pw','b_ls','b_lsx','c1','c2','e1','e2','pl'):
    f = preds_for(name)
    print(f"  {name:5s}: U(6)={f(6):6.2f}  U(7)={f(7):6.2f}  U(8)={f(8):6.2f}")
print("bootstrap 90% bands on U(8) (sigma=[0.06,0.15,0.20,0.25] on U(2..5)):")
for name, (lo5, med, hi5) in boot.items():
    print(f"  {name:5s}: U(8) 5%={lo5:.1f} med={med:.1f} 95%={hi5:.1f}")
line('-')
print("P2(2) PREFILL CHUNK IO TABLE (per sparse layer: records = U(K)):")
print(f"{'K':>5} | {'U range (models)':>22} | {'recs/chunk 42U':>13} | {'GiB @13.56':>10} | {'ms/tok @5.8GB/s':>15} | {'dedup saving':>12}")
for i, K in enumerate(Kbig):
    vals = [m[i] for m in models_big.values()]
    lo, hi = min(vals), max(vals)
    recs = Ndense_layers * hi
    gib = recs * REC_MIB / 1024
    mstok = W_PER_BW / 5.8 * hi / K
    ded = 1 - hi / (8 * K)
    print(f"{K:>5} | {lo:8.1f} .. {hi:8.1f} | {recs:13.0f} | {gib:10.1f} | {mstok:15.1f} | {ded*100:11.1f}%")
line()

line()
print("P8: prefill chunk-size optimum")
print(f"measured ms/token: T=16:700 (cold) 32:356 64:180 -> per-chunk wall ms: {chunk_ms}")
print(f"  (2048-token prompt at T=64: 32 chunks x 11.52s = 368.6s vs 369.4s measured)")
print(f"KDA per-token launch constant: {KDA_LO:.3f}..{KDA_HI:.3f} ms/token (34 KDA layers x2 launches x 3-8us)")
print(f"attention within-chunk coefficient alpha ~= {ALPHA_ATTN:.5f} ms/token per chunk-position")
line('-')
print("prescribed exact 3-param fit m(T)=a+b/T+w*U(T)/T under each U model:")
for name in ('a1', 'b_ls', 'c1', 'pl'):
    (a, b, w), U3 = exact3(lambda T, f=preds_for(name): f(T))
    bw = W_PER_BW / w if w > 0 else float('inf')
    print(f"  {name:4s}: U(16,32,64)={np.round(U3,1)}  a={a:8.2f} b={b:9.1f} w={w:6.2f} -> BW_eff={bw:7.1f} GB/s")
line('-')
print("anchored warm-2-point fit (T=32,64): w=(128-32a)/(U64-U32), b from eq1")
for name in ('b_ls', 'c1', 'pl'):
    U32, U64 = preds_for(name)(32), preds_for(name)(64)
    for a in (0.0, KDA_LO, 0.374, KDA_HI):
        b, w = anchored(a, U32, U64)
        bw = W_PER_BW / w if w > 0 and np.isfinite(w) else float('nan')
        print(f"  {name:4s} a={a:5.3f}: U32={U32:6.1f} U64={U64:6.1f} -> w={w:8.3f} BW_eff={bw:8.1f} GB/s b={b:9.1f}")
line('-')
print("constant-chunk-wall interpretation: chunk ~11.37 s at BW => records/layer read:")
for bw in BW_LIST:
    R = 11.37 * bw / (Ndense_layers * REC_MIB * MIB_GB)
    print(f"  BW={bw:4.2f} GB/s -> R={R:6.1f} recs/layer ({R*Ndense_layers:6.0f} recs, {R*Ndense_layers*REC_MIB/1024:6.1f} GiB/chunk)")
line('-')
print("predictions T=128/256/512 (ms/token):")
def predict(T, a, recs_per_layer, bw, rec_mib=REC_MIB):
    return a + W_PER_BW * (rec_mib / REC_MIB) / bw * recs_per_layer / T
scen = {}
scen['A const-reads (R=110,BW=5.8)'] = lambda T, a, pack: predict(T, a, 110.0, 5.8, 13.56 if not pack else 12.75)
mid = lambda K: preds_for('b_lsx')(K)
scen['B mid-U (b* DM), hits h=50']  = lambda T, a, pack: predict(T, a, max(0.0, mid(T) - 50), 5.8, 13.56 if not pack else 12.75)
hiU = lambda K: preds_for('c1')(K)
scen['C high-U (c1), hits h=81']    = lambda T, a, pack: predict(T, a, max(0.0, hiU(T) - 81), 5.8, 13.56 if not pack else 12.75)
for sname, sf in scen.items():
    row = []
    for T in (128.0, 256.0, 512.0):
        row.append((sf(T, 0.374, False), sf(T, 0.0, False), sf(T, 0.374, True), sf(T, 0.0, True)))
    print(f"  {sname}")
    for T, (base, fused, packed, both) in zip((128, 256, 512), row):
        print(f"    T={T:3d}: base={base:6.1f}  +KDA-fusion={fused:6.1f}  +packed={packed:6.1f}  both={both:6.1f}")
line('-')
print("T* derivation and VRAM table:")
for K in (32, 64, 128, 256, 512):
    for name in ('b_lsx', 'c1'):
        uK = preds_for(name)(K)
        print(f"  T={K:3d} U({name})={uK:6.1f} -> per-layer staging {uK*REC_MIB/1024:5.2f} GiB, "
              f"42-layer union {42*uK*REC_MIB/1024:6.1f} GiB, retention24 x42 = {24*42*REC_MIB/1024:5.2f} GiB")
line()
print("done.")
