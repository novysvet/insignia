set -e
cd /var/lib/insignia/analysis/predict
python3 - <<'PYEOF'
import json
from collections import Counter, defaultdict
import numpy as np

rng = np.random.default_rng(7)
E, K = 288, 8
LN2 = np.log(2.0)
OUT = "/var/lib/insignia/analysis/predict"

# ---- correct re-parse of early-route-math (19 fields: tok L ov p0..7 a0..7) --
rows = []
for line in open("/var/lib/insignia/early-route-math.txt"):
    f = line.split()
    if len(f) != 19: continue
    rows.append((int(f[0]), int(f[1]), int(f[2]),
                 [int(x) for x in f[3:11]], [int(x) for x in f[11:19]]))
by_tok = defaultdict(dict)
for tok, L, ov, P, A in rows:
    by_tok[tok][L] = A          # ACTUAL routing sets are the analysis target
toks = sorted(by_tok); layers = sorted(by_tok[toks[0]])
Ets = np.zeros((len(toks), len(layers), K), dtype=np.int16)
Pm = np.zeros_like(Ets)
for i, t in enumerate(toks):
    for j, L in enumerate(layers):
        Ets[i, j] = by_tok[t][L]
np.savez_compressed(f"{OUT}/early-route-math.npz", toks=np.array(toks),
                    layers=np.array(layers), E=Ets,
                    S=np.zeros_like(Ets, dtype=np.float32))
print(f"re-parsed early-route-math: tokens={len(toks)} actual sets verified")

TRACES = ("route-realtext", "route-campaign", "early-route-math")
data = {n: dict(np.load(f"{n}.npz")) for n in TRACES}
results = {"task1": {}, "task2": {}, "task3": {}}

def entropy_bits(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())

# =====================================================================
# TASK 1
# =====================================================================
NBOOT, NSHUF = 300, 300
for name in TRACES:
    d = data[name]
    Ets, T, L = d["E"], d["E"].shape[0], d["E"].shape[1]
    acc = defaultdict(list)
    for j in range(1, L):
        prev = [tuple(sorted(Ets[t, j - 1].tolist())) for t in range(T)]
        cur = [tuple(sorted(Ets[t, j].tolist())) for t in range(T)]
        cx, cy, cxy = Counter(cur), Counter(prev), Counter(zip(prev, cur))
        Hx = entropy_bits(np.array(list(cx.values()), float))
        Hy = entropy_bits(np.array(list(cy.values()), float))
        Hxy = entropy_bits(np.array(list(cxy.values()), float))
        Hcond = Hxy - Hy
        MI = Hx - Hcond
        MI_MM = MI + (len(cxy) - len(cx) - len(cy) + 1) / (2 * T * LN2)
        pv = np.array([hash(s) for s in prev]); cu = np.array([hash(s) for s in cur])
        bs = np.empty(NBOOT)
        for b in range(NBOOT):
            ii = rng.integers(0, T, T)
            c = Counter(zip(pv[ii], cu[ii]))
            Hxyb = entropy_bits(np.array(list(c.values()), float))
            cb = Counter(pv[ii]); cxb = Counter(cu[ii])
            Hb = entropy_bits(np.array(list(cb.values()), float))
            Hxb = entropy_bits(np.array(list(cxb.values()), float))
            bs[b] = Hxb - Hb - (Hxyb - Hb)
        MI_boot = MI - (bs.mean() - MI)
        sh = np.empty(NSHUF)
        for s_ in range(NSHUF):
            ii = rng.permutation(T)
            c = Counter(zip(pv[ii], cu))
            Hxys = entropy_bits(np.array(list(c.values()), float))
            cs = Counter(pv[ii])
            Hs = entropy_bits(np.array(list(cs.values()), float))
            sh[s_] = Hx - (Hxys - Hs)
        for k, v in (("Hx", Hx), ("Hcond", Hcond), ("MI", MI), ("MI_MM", MI_MM),
                     ("MI_boot", MI_boot), ("MI_shuf", float(sh.mean()))):
            acc[k].append(v)
    R = {k: float(np.mean(v)) for k, v in acc.items()}
    bags = [entropy_bits(np.bincount(Ets[:, j].ravel(), minlength=E).astype(float)) for j in range(L)]
    ov = float(np.mean([[len(set(Ets[t, j].tolist()) & set(Ets[t + 1, j].tolist()))
                         for t in range(T - 1)] for j in range(L)])) / K
    results["task1"][name] = dict(T=T, **R, H_bag_mean=float(np.mean(bags)),
                                  H_bag_min=float(np.min(bags)), H_bag_max=float(np.max(bags)),
                                  adj_token_overlap=ov)
    print(f"\n{name}: T={T}")
    print(f"  H_marg(set)={R['Hx']:.3f}b [cap log2(T)={np.log2(T):.2f}]  H_cond={R['Hcond']:.3f}b")
    print(f"  MI plugin={R['MI']:.3f}b ({100*R['MI']/max(R['Hx'],1e-9):.1f}%)  "
          f"MM={R['MI_MM']:.3f}b  boot={R['MI_boot']:.3f}b  shuffled-ctrl={R['MI_shuf']:.3f}b")
    print(f"  plugin-minus-shuffled = {R['MI']-R['MI_shuf']:.3f}b "
          f"({100*(R['MI']-R['MI_shuf'])/max(R['Hx'],1e-9):.2f}% of H_marg)")
    print(f"  bag entropy={np.mean(bags):.2f}b [{np.min(bags):.2f},{np.max(bags):.2f}]  adj-tok overlap={ov:.3f}")

# =====================================================================
# TASK 2
# =====================================================================
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
    return ((pred & (M1ev > 0)).sum(1) / K, pred.sum(1) / K)

def cov_at(curve, target):
    xs = sorted((ov, c) for c, ov in curve.values())
    ovs = np.array([x[0] for x in xs]); cs = np.array([x[1] for x in xs])
    return float(np.interp(target, ovs, cs))

print("\nTASK 2 (CCT split-sample):")
for name in TRACES:
    d = data[name]
    Ets, T, L = d["E"], d["E"].shape[0], d["E"].shape[1]
    for frac, tag in ((0.6, "60/40"), (0.5, "50/50")):
        cut = int(frac * T)
        curve = {}
        for N in range(1, 17):
            cc, oo = [], []
            for j in range(1, L):
                M0, M1 = onehot(Ets[:, j - 1]), onehot(Ets[:, j])
                c, o = cct_eval(M0[:cut], M1[:cut], M0[cut:], M1[cut:], N)
                cc.append(c); oo.append(o)
            curve[N] = (float(np.concatenate(cc).mean()), float(np.concatenate(oo).mean()))
        print(f"  {name} T={T} {tag}: " + "  ".join(
            f"N={N}:{100*curve[N][0]:.1f}%@{curve[N][1]:.2f}x" for N in (4, 8, 16))
            + f" | fixed-of={cov_at(curve,1.28)*100:.1f}/{cov_at(curve,2.0)*100:.1f}/"
              f"{cov_at(curve,2.45)*100:.1f}/{cov_at(curve,4.58)*100:.1f}% @1.28/2.0/2.45/4.58x")
        results["task2"][f"{name}|{tag}"] = {f"N{N}": curve[N] for N in curve}
    ins = {}
    for N in (4, 8, 16):
        cc, oo = [], []
        for j in range(1, L):
            M0, M1 = onehot(Ets[:, j - 1]), onehot(Ets[:, j])
            c, o = cct_eval(M0, M1, M0, M1, N)
            cc.append(c); oo.append(o)
        ins[N] = (float(np.concatenate(cc).mean()), float(np.concatenate(oo).mean()))
    print(f"  {name} in-sample: " + "  ".join(f"N={N}:{100*ins[N][0]:.1f}%@{ins[N][1]:.2f}x" for N in ins))

# cross-trace
print("  cross-trace (train realtext -> eval):")
for name in ("route-campaign", "early-route-math"):
    d0, d1 = data["route-realtext"], data[name]
    line = []
    for N in (4, 8, 16):
        cc, oo = [], []
        for j in range(1, d0["E"].shape[1]):
            cc.append(cct_eval(onehot(d0["E"][:, j-1]), onehot(d0["E"][:, j]),
                               onehot(d1["E"][:, j-1]), onehot(d1["E"][:, j]), N)[0])
            oo.append(cct_eval(onehot(d0["E"][:, j-1]), onehot(d0["E"][:, j]),
                               onehot(d1["E"][:, j-1]), onehot(d1["E"][:, j]), N)[1])
        line.append(f"N={N}:{100*np.concatenate(cc).mean():.1f}%@{np.concatenate(oo).mean():.2f}x")
    print(f"    -> {name}: " + "  ".join(line))
    results["task2"][f"cross->{name}"] = line

# engine-live predictor from the raw trace (predicted = f[3:11])
rows = [([int(x) for x in l.split()[3:11]], [int(x) for x in l.split()[11:19]])
        for l in open("/var/lib/insignia/early-route-math.txt") if len(l.split()) == 19]
cum = np.zeros(9)
for P, A in rows:
    act = set(A)
    for n in range(1, 9):
        cum[n] += P[n - 1] in act
cum /= len(rows)
print("  engine-live early_routing_ on MATH: " + "  ".join(
    f"n={n}:{100*cum[n]/8:.1f}%cov/{100*cum[n]/n:.1f}%prec" for n in (1, 2, 4, 8)))
results["task2"]["engine_live"] = {f"n{n}": float(cum[n] / 8) for n in range(1, 9)}

json.dump(results, open(f"{OUT}/results_t12.json", "w"), indent=1)
print("saved results_t12.json")
PYEOF
