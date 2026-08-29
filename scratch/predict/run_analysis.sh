set -e
cd /var/lib/insignia/analysis/predict
python3 - <<'PYEOF'
import json
import numpy as np

rng = np.random.default_rng(7)
E, K = 288, 8
LN2 = np.log(2.0)
TRACES = ("route-realtext", "route-campaign", "early-route-math")
data = {n: dict(np.load(f"{n}.npz")) for n in TRACES}

def entropy_bits(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())

results = {"task1": {}, "task2": {}, "task3": {}}

# =====================================================================
# TASK 1: set-valued marginal/conditional entropy, MI with bias control
# =====================================================================
print("=" * 78)
print("TASK 1: H(top8_L), H(top8_L | top8_{L-1}), MI  (sets as frozensets)")
print("=" * 78)
NBOOT, NSHUF = 500, 500
for name in TRACES:
    d = data[name]
    Ets, T = d["E"], d["E"].shape[0]
    L = Ets.shape[1]
    rows = []
    for j in range(1, L):
        prev = [frozenset(Ets[t, j - 1].tolist()) for t in range(T)]
        cur = [frozenset(Ets[t, j].tolist()) for t in range(T)]
        # marginal entropies
        from collections import Counter
        cx = Counter(cur); cy = Counter(prev); cxy = Counter(zip(prev, cur))
        Hx = entropy_bits(np.array(list(cx.values()), float))
        Hy = entropy_bits(np.array(list(cy.values()), float))
        Hxy = entropy_bits(np.array(list(cxy.values()), float))
        Hcond = Hxy - Hy
        MI = Hx - Hcond
        qx, qy, qxy = len(cx), len(cy), len(cxy)
        MI_MM = MI + (qxy - qx - qy + 1) / (2 * T * LN2)
        # bootstrap bias (resample tokens)
        idx_all = np.arange(T)
        bs = np.empty(NBOOT)
        pv = np.array([hash(s) for s in prev]); cu = np.array([hash(s) for s in cur])
        for b in range(NBOOT):
            ii = rng.integers(0, T, T)
            c = Counter(zip(pv[ii], cu[ii]))
            Hxyb = entropy_bits(np.array(list(c.values()), float))
            cb = Counter(pv[ii]); cxb = Counter(cu[ii])
            Hb = entropy_bits(np.array(list(cb.values()), float))
            Hxb = entropy_bits(np.array(list(cxb.values()), float))
            bs[b] = Hxb - Hb - (Hxyb - Hb)
        MI_boot = MI - (bs.mean() - MI)
        # shuffle control (pure finite-sample bias level)
        sh = np.empty(NSHUF)
        for s_ in range(NSHUF):
            ii = rng.permutation(T)
            c = Counter(zip(pv[ii], cu))
            Hxys = entropy_bits(np.array(list(c.values()), float))
            cs = Counter(pv[ii])
            Hs = entropy_bits(np.array(list(cs.values()), float))
            sh[s_] = Hx - (Hxys - Hs)
        rows.append(dict(j=j, Hx=Hx, Hcond=Hcond, MI=MI, MI_MM=MI_MM,
                         MI_boot=MI_boot, MI_shuf=float(sh.mean())))
    R = {k: float(np.mean([r[k] for r in rows])) for k in
         ("Hx", "Hcond", "MI", "MI_MM", "MI_boot", "MI_shuf")}
    # audit-style per-access bag entropy for reconciliation
    bags = [entropy_bits(np.bincount(Ets[:, j].ravel(), minlength=E).astype(float))
            for j in range(L)]
    # adjacent-token same-layer set overlap (temporal baseline)
    ov = float(np.mean([[len(set(Ets[t, j].tolist()) & set(Ets[t + 1, j].tolist()))
                         for t in range(T - 1)] for j in range(L)])) / K
    results["task1"][name] = dict(T=T, n_pairs=len(rows), **R,
                                  H_bag_mean=float(np.mean(bags)),
                                  H_bag_min=float(np.min(bags)), H_bag_max=float(np.max(bags)),
                                  adj_token_overlap=ov)
    print(f"\n{name}: T={T} tokens, {len(rows)} adjacent-layer pairs")
    print(f"  H_marg(set)      = {R['Hx']:.3f} bits   [max possible log2({T})={np.log2(T):.2f}]")
    print(f"  H_cond(set|set)  = {R['Hcond']:.3f} bits")
    print(f"  MI plugin        = {R['MI']:.3f} bits  ratio={R['MI']/max(R['Hx'],1e-9):.3f}")
    print(f"  MI Miller-Madow  = {R['MI_MM']:.3f} bits  ratio={max(R['MI_MM'],0)/max(R['Hx'],1e-9):.3f}")
    print(f"  MI bootstrap-bc  = {R['MI_boot']:.3f} bits  ratio={max(R['MI_boot'],0)/max(R['Hx'],1e-9):.3f}")
    print(f"  MI shuffled-ctrl = {R['MI_shuf']:.3f} bits  (bias floor; plugin minus this = "
          f"{R['MI']-R['MI_shuf']:.3f} bits)")
    print(f"  per-access bag entropy (audit-style) = {np.mean(bags):.2f} bits "
          f"(min {np.min(bags):.2f} max {np.max(bags):.2f})")
    print(f"  adjacent-token same-layer overlap I/8 = {ov:.3f}")

# =====================================================================
# TASK 2: CCT split-sample validation
# =====================================================================
print()
print("=" * 78)
print("TASK 2: CCT successor table, split-sample (rebuild on train, eval on test)")
print("=" * 78)

def onehot(Ex):
    T, k = Ex.shape
    M = np.zeros((T, E), dtype=np.int32)
    M[np.repeat(np.arange(T), k), Ex.ravel()] = 1
    return M

def cct_eval(M0tr, M1tr, M0ev, M1ev, N):
    C = M0tr.T @ M1tr
    tab = np.zeros_like(C, dtype=bool)
    topn = np.argsort(-C, axis=1, kind="stable")[:, :N]
    tab[np.repeat(np.arange(E), N), topn.ravel()] = True
    pred = (M0ev @ tab) > 0
    hits = (pred & (M1ev > 0)).sum(1)
    return hits / K, pred.sum(1) / K

def cct_curve(name, frac):
    d = data[name]
    Ets, T, L = d["E"], d["E"].shape[0], d["E"].shape[1]
    cut = int(frac * T)
    out = {}
    for N in range(1, 17):
        covs, ovs = [], []
        for j in range(1, L):
            M0 = onehot(Ets[:, j - 1]); M1 = onehot(Ets[:, j])
            c, o = cct_eval(M0[:cut], M1[:cut], M0[cut:], M1[cut:], N)
            covs.append(c); ovs.append(o)
        cov = float(np.concatenate(covs).mean()); ovf = float(np.concatenate(ovs).mean())
        out[N] = (cov, ovf)
    return out

def cov_at_overfetch(curve, target):
    xs = sorted((ov, c) for c, ov in curve.values())
    ovs = np.array([x[0] for x in xs]); cs = np.array([x[1] for x in xs])
    if target <= ovs[0]: return float(cs[0])
    if target >= ovs[-1]: return float(cs[-1])
    return float(np.interp(target, ovs, cs))

for name in TRACES:
    T = data[name]["E"].shape[0]
    for frac, tag in ((0.6, "60/40 (audit)"), (0.5, "50/50 (task)")):
        curve = cct_curve(name, frac)
        line = "  ".join(f"N={N}:{100*curve[N][0]:.1f}%@{curve[N][1]:.2f}x" for N in (4, 8, 16))
        print(f"\n{name} (T={T}) split {tag}: {line}")
        print(f"    coverage at fixed overfetch: 1.28x->{100*cov_at_overfetch(curve,1.28):.1f}%  "
              f"2.0x->{100*cov_at_overfetch(curve,2.0):.1f}%  2.45x->{100*cov_at_overfetch(curve,2.45):.1f}%  "
              f"4.58x->{100*cov_at_overfetch(curve,4.58):.1f}%")
        results["task2"][f"{name}|{tag}"] = {f"N{N}": curve[N] for N in curve}
        results["task2"][f"{name}|{tag}|fixed"] = dict(
            o128=cov_at_overfetch(curve, 1.28), o200=cov_at_overfetch(curve, 2.0),
            o245=cov_at_overfetch(curve, 2.45), o458=cov_at_overfetch(curve, 4.58))

# in-sample (all tokens) for reference
for name in TRACES:
    d = data[name]
    Ets, T, L = d["E"], d["E"].shape[0], d["E"].shape[1]
    out = {}
    for N in (4, 8, 16):
        covs, ovs = [], []
        for j in range(1, L):
            M0 = onehot(Ets[:, j - 1]); M1 = onehot(Ets[:, j])
            c, o = cct_eval(M0, M1, M0, M1, N)
            covs.append(c); ovs.append(o)
        out[N] = (float(np.concatenate(covs).mean()), float(np.concatenate(ovs).mean()))
    print(f"{name} in-sample: " + "  ".join(f"N={N}:{100*out[N][0]:.1f}%@{out[N][1]:.2f}x" for N in out))
    results["task2"][f"{name}|insample"] = {f"N{N}": out[N] for N in out}

# cross-trace: train on realtext (what cct-gsm8k.table was built from), eval elsewhere
print("\ncross-trace CCT (train realtext, eval other traces):")
for name in ("route-campaign", "early-route-math"):
    d0, d1 = data["route-realtext"], data[name]
    line = []
    for N in (4, 8, 16):
        covs, ovs = [], []
        for j in range(1, d0["E"].shape[1]):
            M0tr = onehot(d0["E"][:, j - 1]); M1tr = onehot(d0["E"][:, j])
            M0ev = onehot(d1["E"][:, j - 1]); M1ev = onehot(d1["E"][:, j])
            c, o = cct_eval(M0tr, M1tr, M0ev, M1ev, N)
            covs.append(c); ovs.append(o)
        line.append(f"N={N}:{100*np.concatenate(covs).mean():.1f}%@{np.concatenate(ovs).mean():.2f}x")
    print(f"  -> {name}: " + "  ".join(line))
    results["task2"][f"cross->{name}"] = line

# engine-live early-route trace (predicted sets from early_routing_, MATH prompt)
d = dict(np.load("early-math.npz"))
P, A = d["P"], d["A"]
print("\nengine live CCT (early-route-math.txt, predictor = engine early_routing_):")
cum = np.zeros(9)
for r in range(P.shape[0]):
    act = set(A[r].tolist())
    for n in range(1, 9):
        cum[n] += P[r, n - 1] in act
cum /= P.shape[0]
for n in range(1, 9):
    print(f"  first-{n} predicted: coverage={100*cum[n]/8:.1f}%  overfetch={n/8:.2f}x  "
          f"precision={100*cum[n]/n:.1f}%")
results["task2"]["engine-live"] = {f"n{n}": float(cum[n] / 8) for n in range(1, 9)}

json.dump(results, open("results_t12.json", "w"), indent=1)
print("\nOK task1+task2 cached")
PYEOF
