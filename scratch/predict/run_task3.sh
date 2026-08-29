set -e
cd /var/lib/insignia/analysis/predict
python3 - <<'PYEOF'
import json
import numpy as np

E, K = 288, 8
TRACES = ("route-realtext", "route-campaign", "early-route-math")
data = {n: dict(np.load(f"{n}.npz")) for n in TRACES}
results = {}
# note: realtext/campaign have real scores in S; early-route-math npz has S=0
# (that trace has no score fields) -> score-weighted variants skip it.

# =====================================================================
# TASK 3: stronger predictors, 50/50 split, coverage at <=2x overfetch
# =====================================================================
print()
print("=" * 78)
print("TASK 3: predictor probes (50/50 split; coverage = |pred & actual|/8)")
print("=" * 78)

def onehot(Ex):
    T, k = Ex.shape
    M = np.zeros((T, E), dtype=np.float64)
    M[np.repeat(np.arange(T), k), Ex.ravel()] = 1.0
    return M

def ridge_fit(X, Y, lam):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)

def topk_cov(scores, Yev, k):
    """scores: (n,E) predicted scores; Yev: (n,E) binary truth -> coverage@k, precision@k"""
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    hits = Yev[np.repeat(np.arange(len(idx)), k), idx.ravel()].reshape(idx.shape).sum(1)
    return hits / K, hits / k

def eval_variant(name, buildXY, Ets, S, split_j_range, lam_grid, ks=(4, 8, 16)):
    """buildXY(Ets, S, j, rows) -> (X, Y) for pair ending at layer-index j."""
    T = Ets.shape[0]
    cut = T // 2
    icut = cut // 2  # inner split for lambda selection
    out = {}
    for j in split_j_range:
        Xa, Ya = buildXY(Ets, S, j, np.arange(0, cut))
        if Xa is None or Xa.shape[0] < 2:
            return None
        Xte, Yte_b = buildXY(Ets, S, j, np.arange(cut, T))
        Yte = Yte_b > 0
        # lambda selection on inner split of train
        best = None
        for lam in lam_grid:
            Wt = ridge_fit(Xa[icut:], Ya[icut:], lam)
            c, _ = topk_cov(Xa[:icut] @ Wt, Ya[:icut] > 0, 16)
            if best is None or c.mean() > best[1]:
                best = (lam, c.mean())
        lam = best[0]
        W = ridge_fit(Xa, Ya, lam)
        sc = Xte @ W
        for k in ks:
            cov, prec = topk_cov(sc, Yte, k)
            out.setdefault(k, []).append((cov, prec))
        out.setdefault("lam", []).append(lam)
    res = {k: (float(np.concatenate([c for c, p in out[k]]).mean()),
               float(np.concatenate([p for c, p in out[k]]).mean())) for k in ks}
    res["lam"] = float(np.median(out["lam"]))
    return res

LAMBDAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

# --- feature builders -------------------------------------------------
def mk_onehot(Ets, S, j, rows):
    return onehot(Ets[rows, j - 1]), onehot(Ets[rows, j])

def mk_two_layer(Ets, S, j, rows):
    if j < 2: return None, None
    return (np.hstack([onehot(Ets[rows, j - 2]), onehot(Ets[rows, j - 1])]),
            onehot(Ets[rows, j]))

def mk_weighted(Ets, S, j, rows):
    T = len(rows)
    X = np.zeros((T, E)); Y = onehot(Ets[rows, j])
    w = S[rows, j - 1]
    X[np.repeat(np.arange(T), K), Ets[rows, j - 1].ravel()] = w.ravel()
    X *= K / np.maximum(X.sum(1, keepdims=True), 1e-9)  # row mean entry ~1 like one-hot
    return X, Y

def mk_weighted_y(Ets, S, j, rows):
    X, _ = mk_weighted(Ets, S, j, rows)
    T = len(rows)
    Y = np.zeros((T, E))
    Y[np.repeat(np.arange(T), K), Ets[rows, j].ravel()] = S[rows, j].ravel()
    return X, Y

# --- baselines ---------------------------------------------------------
def static_topk(Ets, j, cut, k):
    cnt = np.bincount(Ets[:cut, j].ravel(), minlength=E)
    order = np.argsort(-cnt, kind="stable")[:k]
    Yte = onehot(Ets[cut:, j])
    hits = Yte[:, order].sum(1)
    return float((hits / K).mean()), float((hits / k).mean())

def cct_curve_pair(Ets, j, cut, N):
    M0tr = onehot(Ets[:cut, j - 1]).astype(np.int32); M1tr = onehot(Ets[:cut, j]).astype(np.int32)
    M0ev = onehot(Ets[cut:, j - 1]).astype(np.int32); M1ev = onehot(Ets[cut:, j]).astype(np.int32)
    C = M0tr.T @ M1tr
    tab = np.zeros_like(C, dtype=bool)
    topn = np.argsort(-C, axis=1, kind="stable")[:, :N]
    tab[np.repeat(np.arange(E), N), topn.ravel()] = True
    pred = (M0ev @ tab) > 0
    hits = (pred & (M1ev > 0)).sum(1)
    return hits / K, pred.sum(1) / K

for name in TRACES:
    d = data[name]
    Ets, S, T, L = d["E"], d["S"], d["E"].shape[0], d["E"].shape[1]
    cut = T // 2
    print(f"\n{name} (T={T}, train={cut}/eval={T-cut}):")
    print(f"  {'predictor':<34}{'cov@4(0.5x)':>12}{'cov@8(1x)':>11}{'cov@16(2x)':>11}{'prec@16':>9}")
    # temporal baseline: next-token same-layer set
    if T - cut > 1:
        ov = np.mean([len(set(Ets[t-1, j].tolist()) & set(Ets[t, j].tolist())) / K
                      for t in range(max(cut, 1), T) for j in range(L)])
        print(f"  {'temporal (prev-token set)':<34}{ov:>11.1%}  [1.0x overfetch exactly]")

    # static top-k
    for k in (4, 8, 16):
        cs = [static_topk(Ets, j, cut, k) for j in range(1, L)]
        if k == 16:
            c16 = np.mean([x[0] for x in cs]); p16 = np.mean([x[1] for x in cs])
            row = [np.mean([x[0] for x in [static_topk(Ets, j, cut, kk) for j in range(1, L)]]) for kk in (4, 8, 16)]
            print(f"  {'static top-k (train marg)':<34}{row[0]:>11.1%}{row[1]:>11.1%}{row[2]:>11.1%}{p16:>9.1%}")
            results.setdefault(name, {})["static"] = dict(zip(("k4", "k8", "k16"), row))
    # CCT budget-matched: largest N with mean overfetch <= 2
    covs_by_N = {}
    for N in range(1, 17):
        cs = [cct_curve_pair(Ets, j, cut, N) for j in range(1, L)]
        cov = np.concatenate([c for c, o in cs]); ovf = np.concatenate([o for c, o in cs]).mean()
        covs_by_N[N] = (float(cov.mean()), float(ovf), float((cov * 8 / max(ovf, 1e-9)).mean()))
    bestN = max(N for N in covs_by_N if covs_by_N[N][1] <= 2.0)
    print(f"  {'CCT union (N<=2x budget)':<34}{'-':>12}{'-':>11}"
          f"{covs_by_N[bestN][0]:>11.1%}{covs_by_N[bestN][2]:>9.1%}"
          f"   [N={bestN}, overfetch={covs_by_N[bestN][1]:.2f}x]")
    results.setdefault(name, {})["cct_budget2"] = dict(N=bestN, **dict(zip(("cov", "ovf", "prec"), covs_by_N[bestN])))

    for vname, builder, jrange in (
            ("ridge one-hot 288d (a)", mk_onehot, range(1, L)),
            ("ridge 2-layer 576d (b)", mk_two_layer, range(2, L)),
            ("ridge score-w X (c)", mk_weighted, range(1, L)),
            ("ridge score-w X+Y", mk_weighted_y, range(1, L))):
        r = eval_variant(vname, builder, Ets, S, jrange, LAMBDAS)
        if r is None:
            print(f"  {vname:<34} (too few tokens)")
            continue
        print(f"  {vname:<34}{r[4][0]:>11.1%}{r[8][0]:>11.1%}{r[16][0]:>11.1%}{r[16][1]:>9.1%}"
              f"   [median lam={r['lam']:.1f}]")
        results[name][vname] = {str(k): (r[k][0], r[k][1]) for k in (4, 8, 16)}

json.dump(results, open("results_t3.json", "w"), indent=1)
print("\nOK cached results_t3.json + engine_live in it")
PYEOF
