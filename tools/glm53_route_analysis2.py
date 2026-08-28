#!/usr/bin/env python3
"""Second-pass routing analysis for INSIGNIA_GLM53_ROUTE_TRACE traces (numpy only).

Trace line: "token_index layer e0..e7 s0..s7" (one line per (token, sparse-layer);
e-order is the engine's execution order, s are the matching router scores).

Sections (all on post-warmup tokens, matching glm53_route_analysis.py):
  1. static coverage curves  - global-hottest-B pinning vs even per-layer split of B
  2. cross-layer correlation - lift distributions + CCT prefetch simulation (ST-MoE style)
  3. temporal union curves   - MTP dedup estimator, global + repetitive halves of trace
  4. per-layer skew          - entropy, top-1 access prob, static coverage at 4 slots
  5. ideal-LRU re-check      - hit rates at engine-relevant slot counts

Usage: python glm53_route_analysis2.py route1.txt [--warmup N] [--experts 288]
"""
import argparse
from collections import OrderedDict

import numpy as np

SLOT_MIB = 13.5
STATIC_BUDGETS = [42, 84, 126, 168, 252, 336, 504, 672, 840, 1176]  # all /42
LRU_SIZES = [379, 512, 591, 672, 840, 1024, 1176]
UNION_KS = [2, 3, 4, 5, 6, 8]
CCT_SIZES = [4, 8, 16]
TRAIN_FRAC = 0.6


def load(path):
    """-> ({layer: (toks(T,), E(T,k) int, S(T,k) float)}, exec-order stream [(tok,key)])"""
    layers, stream = {}, []
    for line in open(path, encoding="utf-8"):
        f = line.split()
        if len(f) < 4:
            continue
        tok, layer = int(f[0]), int(f[1])
        k = (len(f) - 2) // 2
        experts = [int(x) for x in f[2:2 + k]]
        scores = [float(x) for x in f[2 + k:2 + 2 * k]]
        layers.setdefault(layer, []).append((tok, experts, scores))
        stream.extend((tok, layer * 1024 + e) for e in experts)
    out = {}
    for layer, rows in layers.items():
        rows.sort(key=lambda r: r[0])
        out[layer] = (np.array([r[0] for r in rows]),
                      np.array([r[1] for r in rows], dtype=np.int64),
                      np.array([r[2] for r in rows], dtype=np.float64))
    return out, stream


def one_hot(E, experts):
    M = np.zeros((E.shape[0], experts), dtype=bool)
    M[np.repeat(np.arange(E.shape[0]), E.shape[1]), E.ravel()] = True
    return M


# ---------------------------------------------------------------- section 1
def sec1_static(layers, warm, experts):
    print("\n(1) STATIC COVERAGE - fraction of expert ACCESSES covered (equal-size records,")
    print("    so fraction of bytes too). global = B hottest (layer,expert) keys overall;")
    print("    per-layer = B/42 hottest experts within each layer.")
    ls = sorted(layers)
    stack = np.stack([np.bincount(layers[l][1][layers[l][0] >= warm].ravel(),
                                  minlength=experts) for l in ls])  # (L, experts)
    total = stack.sum()
    order = np.argsort(-stack.ravel(), kind="stable")
    gcum = np.cumsum(stack.ravel()[order])
    pl_cum = np.cumsum(-np.sort(-stack, axis=1), axis=1)  # per-layer desc cumulative
    print(f"    accesses={total} (of {stack.size} possible keys, {int((stack > 0).sum())} touched)")
    print(f"{'B':>5} {'MiB':>6} {'/layer':>6} {'global%':>8} {'perlayer%':>10}")
    for B in STATIC_BUDGETS:
        g = gcum[min(B, gcum.size) - 1] / total
        p = pl_cum[:, B // len(ls) - 1].sum() / total
        print(f"{B:>5} {B * SLOT_MIB:>6.0f} {B // len(ls):>6} {100 * g:>8.2f} {100 * p:>10.2f}")
    print("    top-k hottest experts PER LAYER, coverage averaged over layers (unweighted):")
    for k in (1, 2, 4, 8):
        v = pl_cum[:, k - 1] / stack.sum(1)
        print(f"      top-{k}: {100 * v.mean():6.2f}%   (min layer {100 * v.min():6.2f}%, "
              f"max layer {100 * v.max():6.2f}%)")


# ---------------------------------------------------------------- section 2
def sec2_cct(layers, warm, experts):
    print("\n(2) CROSS-LAYER CORRELATION (CCT, ST-MoE style), adjacent sparse-layer pairs (L,L+1)")
    print("    lift = P(f in top8_{L+1}(t) | e in top8_L(t)) / P(f in top8_{L+1})")
    ls = sorted(layers)
    ind = {}
    for l in ls:
        toks, E, S = layers[l]
        m = toks >= warm
        M = one_hot(E[m], experts)
        T1 = np.zeros_like(M)
        t1 = E[m][np.arange(m.sum()), np.argmax(S[m], axis=1)]  # top-1 = argmax score
        T1[np.arange(m.sum()), t1] = True
        ind[l] = (M, T1)
    lift_all, cnt_all, lift1, cnt1 = [], [], [], []
    for l, l1 in zip(ls[:-1], ls[1:]):
        M0, T1s = ind[l]
        M1, _ = ind[l1]
        T = M0.shape[0]
        marg = M1.sum(0) / T
        ok_marg = marg > 0
        C = M0.T.astype(np.int32) @ M1.astype(np.int32)  # joint counts e@L, f@L+1
        ne = M0.sum(0)
        ok = (C > 0) & (ne[:, None] > 0) & ok_marg[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = (C / ne[:, None]) / marg[None, :]
        lift_all.append(lift[ok]); cnt_all.append(C[ok])
        C1 = T1s.T.astype(np.int32) @ M1.astype(np.int32)  # f@L+1 given e is top-1 of L
        n1 = T1s.sum(0)
        ok1 = (C1 > 0) & (n1[:, None] > 0) & ok_marg[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            l1v = (C1 / n1[:, None]) / marg[None, :]
        lift1.append(l1v[ok1]); cnt1.append(C1[ok1])
    la, ca = np.concatenate(lift_all), np.concatenate(cnt_all)
    lb, cb = np.concatenate(lift1), np.concatenate(cnt1)

    def dist(v, c, tag):
        q = np.percentile(v, [10, 25, 50, 75, 90, 99])
        print(f"    {tag}: n={v.size} mean={v.mean():6.2f} cnt-weighted={np.average(v, weights=c):6.2f}")
        print(f"      p10={q[0]:5.2f} p25={q[1]:5.2f} med={q[2]:5.2f} p75={q[3]:5.2f} "
              f"p90={q[4]:5.2f} p99={q[5]:6.2f} max={v.max():7.2f}  "
              f"share>=2:{(v >= 2).mean():.2f} >=5:{(v >= 5).mean():.2f} >=10:{(v >= 10).mean():.2f}")

    dist(la, ca, "(a) all co-activating (e,f) entries, pooled over 41 pairs")
    dist(lb, cb, "(b) source e = top-1 (argmax score) of layer L           ")

    print("\n    CCT prefetch sim: CCT[e@L] = N most co-activated experts at L+1;")
    print("    pred = union over e in top8_L(t) of CCT[e]; evaluated per token per pair.")
    print("    (in-sample: CCT built+evaled on all warm tokens; split: first 60% / last 40%)")
    print(f"{'N':>3} {'insitu cov%':>11} {'insitu over':>11} {'split cov%':>10} {'split over':>10}")
    for N in CCT_SIZES:
        row = []
        for mode in ("in", "split"):
            covs, ovs = [], []
            for l, l1 in zip(ls[:-1], ls[1:]):
                M0, _ = ind[l]
                M1, _ = ind[l1]
                T = M0.shape[0]
                cut = int(TRAIN_FRAC * T) if mode == "split" else T
                Ct = M0[:cut].T.astype(np.int32) @ M1[:cut].astype(np.int32)
                CCTb = np.zeros_like(Ct, dtype=bool)
                topn = np.argsort(-Ct, axis=1)[:, :N]
                CCTb[np.repeat(np.arange(Ct.shape[0]), N), topn.ravel()] = True
                ev0 = M0 if mode == "in" else M0[cut:]
                ev1 = M1 if mode == "in" else M1[cut:]
                pred = (ev0.astype(np.int32) @ CCTb.astype(np.int32)) > 0
                covs.append((pred & ev1).sum(1) / 8)
                ovs.append(pred.sum(1) / 8)
            row += [np.concatenate(covs).mean(), np.concatenate(ovs).mean()]
        print(f"{N:>3} {100 * row[0]:>11.2f} {row[1]:>11.2f} {100 * row[2]:>10.2f} {row[3]:>10.2f}")
    print("    baseline: predicting the static top-8 hottest experts of L+1 covers the")
    print("    'per-layer%' value at B=336 in section 1 - compare before trusting CCT.")


# ---------------------------------------------------------------- section 3
def sec3_union(layers, warm, experts):
    print("\n(3) TEMPORAL UNION CURVES (MTP dedup estimator)")
    print("    U(K) = mean unique experts touched per layer over windows of K consecutive")
    print("    tokens; U(K)/8 vs 1.0 for K=1; U(K)/K/8 = bytes per accepted token vs 1.0.")
    ls = sorted(layers)
    toks = np.array(sorted({t for t in layers[ls[0]][0] if t >= warm}))
    T = len(toks)
    M = np.stack([one_hot(layers[l][1][layers[l][0] >= warm], experts) for l in ls])  # (L,T,E)
    assert all(np.array_equal(layers[l][0][layers[l][0] >= warm], toks) for l in ls)
    # adjacent-token overlap score per token (pair t,t+1) -> median split by rank
    ov = (M[:, :-1, :] & M[:, 1:, :]).sum(2).mean(0) / 8  # one score per pair (T-1,)
    rank = np.argsort(np.argsort(ov, kind="stable"))
    hi_pair = rank >= (T - 1) // 2  # most-repetitive half of pairs, by rank
    lo_pair = ~hi_pair
    hi = np.empty(T, bool); hi[:T - 1] = hi_pair; hi[-1] = hi_pair[-1]  # -> per-token mask
    lo = ~hi
    print(f"    adjacent-overlap I/8: mean={ov.mean():.3f} "
          f"(rep-half mean={ov[hi_pair].mean():.3f}, nonrep-half mean={ov[lo_pair].mean():.3f})")
    print(f"{'K':>2} {'| all U(K)':>9} {'U/8':>6} {'U/K/8':>7} {'| rep U(K)':>10} {'rep U/K/8':>9} "
          f"{'| nonrep U(K)':>12} {'nonrep U/K/8':>12}")

    def curve(sel):
        out = {}
        for K in UNION_KS:
            us = []
            for i in range(T - K + 1):
                if toks[i + K - 1] != toks[i] + K - 1 or not sel[i:i + K].all():
                    continue
                us.append(M[:, i:i + K, :].any(1).sum(1).mean())
            u = float(np.mean(us)) if us else float("nan")
            out[K] = (u, u / 8, u / K / 8)
        return out

    ca, ch, cl = curve(np.ones(T, bool)), curve(hi), curve(lo)
    for K in UNION_KS:
        print(f"{K:>2} {ca[K][0]:>9.2f} {ca[K][1]:>6.3f} {ca[K][2]:>7.3f} |"
              f" {ch[K][0]:>9.2f} {ch[K][2]:>9.3f} | {cl[K][0]:>11.2f} {cl[K][2]:>12.3f}")


# ---------------------------------------------------------------- section 4
def sec4_skew(layers, warm, experts):
    print(f"\n(4) PER-LAYER SKEW (max entropy = log2({experts}) = {np.log2(experts):.2f} bits)")
    print(f"{'layer':>5} {'H(bits)':>8} {'top1%':>7} {'cov@4%':>8}")
    rows = []
    for l in sorted(layers):
        toks, E, _ = layers[l]
        c = np.bincount(E[toks >= warm].ravel(), minlength=experts).astype(np.float64)
        n = c.sum()
        p = c / n
        H = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        cs = np.sort(c)[::-1]
        rows.append((l, H, cs[0] / n, cs[:4].sum() / n))
        print(f"{l:>5} {H:>8.3f} {100 * rows[-1][2]:>7.2f} {100 * rows[-1][3]:>8.2f}")
    by_h = sorted(rows, key=lambda r: r[1])
    print("    most skewed (lowest H):  " + ", ".join(f"L{r[0]}({r[1]:.2f}b,t1={100*r[2]:.0f}%) " for r in by_h[:5]))
    print("    least skewed (highest H): " + ", ".join(f"L{r[0]}({r[1]:.2f}b,t1={100*r[2]:.0f}%)" for r in by_h[-5:]))
    print(f"    mean H={np.mean([r[1] for r in rows]):.3f} bits "
          f"({100 * np.mean([r[1] for r in rows]) / np.log2(experts):.0f}% of max), "
          f"mean top1={100 * np.mean([r[2] for r in rows]):.1f}%, "
          f"mean cov@4={100 * np.mean([r[3] for r in rows]):.1f}%")


# ---------------------------------------------------------------- section 5
def sec5_lru(stream, warm, sizes):
    print(f"\n(5) IDEAL-LRU RE-CHECK on the same exec-order stream ({SLOT_MIB} MiB/slot)")
    accesses = [k for t, k in stream if t >= warm]
    n = len(accesses)
    print(f"{'slots':>6} {'MiB':>7} {'LRU hit%':>9}")
    for cap in sizes:
        cache, hits = OrderedDict(), 0
        for key in accesses:
            if key in cache:
                hits += 1
                cache.move_to_end(key)
            else:
                cache[key] = True
                if len(cache) > cap:
                    cache.popitem(last=False)
        print(f"{cap:>6} {cap * SLOT_MIB:>7.0f} {100 * hits / n:>9.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--warmup", type=int, default=0, help="cut token_index < N; 0 = 10%%")
    ap.add_argument("--experts", type=int, default=288)
    a = ap.parse_args()
    layers, stream = load(a.trace)
    max_tok = max(t for t, _ in stream)
    warm = a.warmup if a.warmup > 0 else max(1, max_tok // 10)
    print(f"trace={a.trace} sparse layers={len(layers)} topk={layers[next(iter(layers))][1].shape[1]} "
          f"tokens={max_tok + 1}; warmup cuts token_index < {warm}")
    sec1_static(layers, warm, a.experts)
    sec2_cct(layers, warm, a.experts)
    sec3_union(layers, warm, a.experts)
    sec4_skew(layers, warm, a.experts)
    sec5_lru(stream, warm, LRU_SIZES)


if __name__ == "__main__":
    main()
