#!/usr/bin/env bash
# 02_analyze_hotset.sh - hot-set coverage / allocation analysis on glm-box (Arch WSL),
# reading ONLY the parsed caches in /var/lib/insignia/analysis/hotset/*.npz.
# Produces: in-sample top-B curves, split-sample OOS curves, 2425/3620 allocation
# comparison (equal vs water-fill), VRAM ~330-slot comparison, marginal-slot sweep,
# Wilson + block-bootstrap CIs, power statement. Writes CSVs next to the caches.
# Usage (local Git Bash):
#   ssh glm-box "wsl -d Arch -- bash -s" < 02_analyze_hotset.sh
set -euo pipefail
python3 - <<'PYEOF'
import json, math
import numpy as np

rng = np.random.default_rng(1234)
OUT = "/var/lib/insignia/analysis/hotset"
TRACES = ["route-realtext", "route-campaign", "early-route-math"]
SHORT  = {"route-realtext": "realtext", "route-campaign": "campaign", "early-route-math": "math"}
L, NE, L0 = 42, 288, 3
B_OOS = [8, 16, 28, 48, 57, 96]
PC = lambda x: f"{100*x:6.2f}"

# ---------- load caches ----------
data = {}
for name in TRACES:
    z = np.load(f"{OUT}/{name}.npz")
    tok = z["token"].astype(np.int64)
    lidx = (z["layer"].astype(np.int64) - L0)
    E = z["experts"].astype(np.int64)
    utok = np.unique(tok)
    T = len(utok)
    d = dict(tok=tok, lidx=lidx, E=E, utok=utok, T=T,
             train=utok[:T//2], test=utok[T//2:])
    def _cnt(ts):
        sel = np.isin(tok, list(ts))
        idx = np.repeat(lidx[sel], 8)*NE + E[sel].reshape(-1)
        return np.bincount(idx, minlength=L*NE).reshape(L, NE).astype(np.int64)
    d["Cte_tok"] = {int(t): _cnt([t]) for t in d["test"]}
    d["C_full"] = _cnt(utok)
    d["C_tr"] = _cnt(d["train"])
    data[name] = d

def hdr(s):
    print("\n" + "="*78 + "\n== " + s + "\n" + "="*78)

# ---------- SECTION A: inventory ----------
hdr("A. TRACE INVENTORY (from caches)")
print(f"{'trace':10} {'tokens':>6} {'train':>5} {'test':>4} {'acts/layer':>10} {'distinct/layer':>14} {'H bits/layer':>22}")
for name in TRACES:
    d = data[name]
    dist = (d["C_full"] > 0).sum(1)
    p = d["C_full"] / d["C_full"].sum(1, keepdims=True)
    lp = np.zeros_like(p); m = p > 0
    lp[m] = np.log2(p[m])
    H = -(p * lp).sum(1)
    print(f"{SHORT[name]:10} {d['T']:>6} {len(d['train']):>5} {len(d['test']):>4} {8*d['T']:>10} "
          f"{dist.mean():>8.1f} [{dist.min():>3},{dist.max():>3}] "
          f"{H.mean():>8.2f} [{H.min():.2f},{H.max():.2f}]")
print("audit anchors: entropy 4.54-5.22 bits; adjacent-token I/8 mean 0.193")
for name in TRACES:
    d = data[name]
    ov_sum, n, prev, prev_t = 0, 0, None, -1
    for t in d["utok"]:
        sel = (d["tok"] == t)
        cur = np.repeat(d["lidx"][sel], 8)*NE + d["E"][sel].reshape(-1)
        if prev is not None and t == prev_t + 1:
            ov_sum += len(np.intersect1d(prev, cur)); n += 1
        prev, prev_t = cur, t
    print(f"  adjacent-token same-layer overlap {SHORT[name]:10}: mean I/8 = {ov_sum/max(n,1)/L/8:.3f}  (pairs={n})")
zov = np.load(f"{OUT}/early-route-math.npz")["overlap"]
print(f"  early-route-math built-in pair column (|e SET u|/8): mean = {zov.mean()/8:.3f}")

# ---------- SECTION B: in-sample top-B coverage ----------
hdr("B. IN-SAMPLE TOP-B COVERAGE (share of activations), avg over 42 layers")
Bgrid = [1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,36,40,44,48,52,56,57,60,64,68,72,76,80,84,88,92,96]
ins_rows = {}
for name in TRACES + ["POOL"]:
    C = sum(data[n]["C_full"] for n in TRACES) if name == "POOL" else data[name]["C_full"]
    o = np.argsort(-C, axis=1, kind="stable")
    ins_rows[name] = np.cumsum(np.take_along_axis(C, o, 1), 1) / C.sum(1, keepdims=True)
print(f"{'B':>3} | {'realtext':>8} | {'campaign':>8} | {'math':>8} | {'POOL avg':>8} {'POOL min':>8} {'POOL max':>8}")
csv = ["B," + ",".join(f"{s}_avg,{s}_min,{s}_max" for s in ["realtext","campaign","math","pool"])]
for B in Bgrid:
    r = [ins_rows[n][:, B-1] for n in TRACES]
    pl = ins_rows["POOL"][:, B-1]
    print(f"{B:>3} | {PC(r[0].mean())} | {PC(r[1].mean())} | {PC(r[2].mean())} | {PC(pl.mean())} {PC(pl.min())} {PC(pl.max())}")
    csv.append(f"{B}," + ",".join(f"{100*x.mean():.4f},{100*x.min():.4f},{100*x.max():.4f}" for x in r) +
               f",{100*pl.mean():.4f},{100*pl.min():.4f},{100*pl.max():.4f}")
open(f"{OUT}/insample_curve.csv","w").write("\n".join(csv)+"\n")
print("audit anchors: top-1 9.46%  top-8 41.07%  top-28 91.49%  (of accesses, per layer)")
print("full curve CSV -> /var/lib/insignia/analysis/hotset/insample_curve.csv")

# ---------- dataset wrapper for OOS work ----------
class DS:
    def __init__(self, label, names):
        self.label = label; self.names = names
        self.C_tr = sum(data[n]["C_tr"] for n in names)
        self.Cte = sum(data[n]["Cte_tok"][t] for n in names for t in data[n]["test"])
        self.order = np.argsort(-self.C_tr, axis=1, kind="stable")   # ties -> lower expert id
        self.flat_order = np.argsort(-self.C_tr.ravel(), kind="stable")
    def topB(self, Cte, B):
        cum = np.cumsum(np.take_along_axis(Cte, self.order, 1), 1)
        return (cum[:, B-1] / Cte.sum(1)).mean()
    def eq(self, Cte, total):
        base, rem = divmod(total, L)
        cum = np.cumsum(np.take_along_axis(Cte, self.order, 1), 1)
        Bs = base + (np.arange(L) < rem)
        return cum[np.arange(L), np.maximum(Bs - 1, 0)].sum() / max(Cte.sum(), 1)
    def wf(self, Cte, total):                 # global hottest (water-fill)
        flat_te = Cte.ravel()[self.flat_order]
        return flat_te[:total].sum() / max(Cte.sum(), 1)
    def wf_Bl(self, total):
        return np.bincount(self.flat_order[:total] // NE, minlength=L)
    def perlayer_k(self, Cte, k):
        cum = np.cumsum(np.take_along_axis(Cte, self.order, 1), 1)
        return (cum[:, k-1] / Cte.sum(1)).mean()

ds_list = [DS(SHORT[n], [n]) for n in TRACES] + [DS("pooled", TRACES)]
Ds = {d.label: d for d in ds_list}

# ---------- SECTION C: split-sample OOS ----------
hdr("C. SPLIT-SAMPLE OOS HIT RATE (top-B built on first half, hit on second half)")
print(f"{'trace':10} {'n_test_acts/layer':>17} | " + " | ".join(f"B={b:>2}" for b in B_OOS))
for d in ds_list:
    nte = 8 * sum(len(data[n]["test"]) for n in d.names)
    print(f"{d.label:10} {nte:>17} | " + " | ".join(f"{PC(d.topB(d.Cte, b)).strip()}" for b in B_OOS))
def wilson(k, n, z=1.959963984540054):
    if n == 0: return float('nan'), float('nan')
    p = k/n; den = 1+z*z/n; c = (p + z*z/(2*n))/den
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return c-h, c+h
print("\nWilson 95% CI on the pooled-activation OOS rate (iid-activation assumption; optimistic):")
print(f"{'trace':10} " + " | ".join(f"B={b:>2}" for b in B_OOS))
for d in ds_list:
    cells = []
    for b in B_OOS:
        cumv = np.cumsum(np.take_along_axis(d.Cte, d.order, 1), 1)[:, b-1]
        lo, hi = wilson(cumv.sum(), d.Cte.sum())
        cells.append(f"{100*lo:4.1f}-{100*hi:5.1f}")
    print(f"{d.label:10} " + " | ".join(cells))
rows = ["trace,layer," + ",".join(f"B{b}" for b in B_OOS)]
for d in ds_list:
    cumv = np.cumsum(np.take_along_axis(d.Cte, d.order, 1), 1)
    for l in range(L):
        rows.append(f"{d.label},{l+L0}," + ",".join(f"{100*cumv[l,b-1]/d.Cte[l].sum():.4f}" for b in B_OOS))
open(f"{OUT}/oos_perlayer.csv","w").write("\n".join(rows)+"\n")
print("per-layer OOS CSV -> /var/lib/insignia/analysis/hotset/oos_perlayer.csv")

# ---------- SECTION C2: cross-text transfer ----------
hdr("C2. CROSS-TEXT TRANSFER (train first halves on one text set, test second half of another)")
pairs = [("campaign+math+realtext -> each", None),
         ("campaign only -> realtext", ["route-campaign"]),
         ("campaign only -> math", ["route-campaign"]),
         ("realtext+math -> campaign", ["route-realtext", "early-route-math"])]
print(f"{'train set':>28} {'test set':>10} | " + " | ".join(f"B={b:>2}" for b in B_OOS))
DsX = {tuple(v): DS("+".join(SHORT[x] for x in v), v) for v in
       [["route-campaign"], ["route-realtext","early-route-math"], TRACES]}
def test_of(name): return sum(data[name]["Cte_tok"][t] for t in data[name]["test"])
for lab, tr in pairs:
    src = DsX[tuple(TRACES)] if tr is None else DsX[tuple(tr)]
    targets = TRACES if tr is None else (["route-realtext"] if "realtext" in lab.split("->")[1]
                                         else ["early-route-math"] if "math" in lab.split("->")[1]
                                         else ["route-campaign"])
    for tg in targets:
        Cte = test_of(tg)
        print(f"{lab:>28} {SHORT[tg]:>10} | " + " | ".join(f"{PC(src.topB(Cte, b)).strip()}" for b in B_OOS))

# ---------- SECTION D: allocation at 2425 / 3620 ----------
hdr("D. HOST-TIER ALLOCATION, budget split across 42 layers (OOS on second half)")
for total in (2425, 3620):
    print(f"\n--- total slots = {total} ({total/42:.1f}/layer equal) ---")
    print(f"{'trace':10} {'equal OOS':>9} {'waterfill OOS':>13} {'wf-eq (pp)':>10}   {'equal in-sample':>15} {'wf in-sample':>12}")
    for d in ds_list:
        eq, wf = d.eq(d.Cte, total), d.wf(d.Cte, total)
        print(f"{d.label:10} {PC(eq)} {PC(wf)} {100*(wf-eq):>+9.2f}   {PC(d.eq(d.C_tr, total)):>15} {PC(d.wf(d.C_tr, total)):>12}")
    Bl = Ds['pooled'].wf_Bl(total)
    q = np.percentile(Bl, [0,25,50,75,100]).astype(int)
    print(f"  pooled water-fill B_l spread: min={q[0]} p25={q[1]} med={q[2]} p75={q[3]} max={q[4]}; "
          f"layers pinned to 0: {(Bl==0).sum()}, layers at >=288: {(Bl>=288).sum()}")

# ---------- SECTION E: VRAM mirror ~330 slots ----------
hdr("E. VRAM MIRROR (~330 slots), OOS on second half")
print("per-layer uniform top-k:")
print(f"{'trace':10} " + " | ".join(f"k={k}(slots {k*L})" for k in (1,2,4,7,8)))
for d in ds_list:
    print(f"{d.label:10} " + " | ".join(f"{PC(d.perlayer_k(d.Cte, k)).strip():>12}" for k in (1,2,4,7,8)))
print("\nequal-split vs global-hottest at matched totals:")
print(f"{'trace':10} {'total':>5} {'equal':>7} {'global':>7} {'global-eq (pp)':>14}")
for d in ds_list:
    for total in (294, 321, 330, 336):
        print(f"{d.label:10} {total:>5} {PC(d.eq(d.Cte, total))} {PC(d.wf(d.Cte, total))} {100*(d.wf(d.Cte, total)-d.eq(d.Cte, total)):>+13.2f}")

# ---------- SECTION F: marginal value of slots ----------
hdr("F. MARGINAL OOS HIT PER +100 SLOTS (2425 -> 3620)")
MS_LO, MS_HI, MS_B = 13.56*2**20/3.7e9*1e3, 13.56*2**20/5.84e9*1e3, 0.60
print(f"cost per marginal missed record: {MS_LO:.1f} ms (3.7 GB/s) .. {MS_HI:.1f} ms (5.84 GB/s); audit avg b = 0.60 ms")
totals = sorted(set([2425 + 100*i for i in range(12)] + [3620]))
print(f"{'total':>5} | {'pool eq':>7} {'pool wf':>7} {'d_wf pp':>7} {'ms/tok(3.7)':>11} | {'cmp eq':>7} {'cmp wf':>7}")
sweep = ["total,pool_eq,pool_wf,cmp_eq,cmp_wf"]
prev = None
for t in totals:
    a, b = Ds['pooled'].eq(Ds['pooled'].Cte, t), Ds['pooled'].wf(Ds['pooled'].Cte, t)
    c, e = Ds['campaign'].eq(Ds['campaign'].Cte, t), Ds['campaign'].wf(Ds['campaign'].Cte, t)
    dpp = 100*(b - prev) if prev is not None else float('nan')
    ms = 336*(b - prev)*MS_LO if prev is not None else float('nan')
    print(f"{t:>5} | {PC(a)} {PC(b)} {dpp:>6.2f} {ms:>10.1f} | {PC(c)} {PC(e)}")
    sweep.append(f"{t},{100*a:.5f},{100*b:.5f},{100*c:.5f},{100*e:.5f}")
    prev = b
open(f"{OUT}/alloc_sweep.csv","w").write("\n".join(sweep)+"\n")
BASE = 516.7
for base, span in ((2425, 600), (2425, 1195)):
    for lab, d in (("pooled", Ds['pooled']), ("campaign", Ds['campaign'])):
        for pol, fn in (("equal", lambda x,t: x.eq(x.Cte, t)), ("waterfill", lambda x,t: x.wf(x.Cte, t))):
            dh = fn(d, base+span) - fn(d, base)
            rec = 336*dh
            print(f"+{span} slots from {base} ({lab:8} {pol:9}): dOOS = {100*dh:+.2f} pp = {rec:5.1f} rec/token "
                  f"-> {rec*MS_LO:5.1f}/{rec*MS_HI:4.1f} ms at NVMe, {rec*MS_B:4.1f} ms at b=0.6ms "
                  f"({100*rec*MS_LO/BASE:+.1f}% of {BASE} ms/token)")
# flatness of the frequency head at the pin boundary
C = Ds['campaign'].C_tr
o = np.argsort(-C, axis=1, kind="stable")
cs = np.take_along_axis(C, o, 1)
print("\ncampaign train counts at per-layer rank 1/4/8/12/16/28/57 (median over layers):")
for rk in (1, 4, 8, 12, 16, 28, 57):
    print(f"  rank {rk:>2}: median count {np.median(cs[:, rk-1]):>5.1f}  (of 240 train acts)")

# ---------- SECTION G: bootstrap CIs ----------
hdr("G. BLOCK-BOOTSTRAP 95% CIs (token blocks, 2000 reps)")
NB = 2000
def boot(d):
    out = {("topB", b): [] for b in (8, 28, 57)}
    out.update({("eq", t): [] for t in (2425, 3025, 3620)})
    out.update({("wf", t): [] for t in (2425, 3025, 3620)})
    out[("perlayer8",)] = []; out[("global336",)] = []
    tlists = [(data[n]["test"], n) for n in d.names]
    for _ in range(NB):
        Cte = np.zeros((L, NE), np.int64)
        for tl, nm in tlists:
            bs = max(1, len(tl)//10)
            picks = []
            while len(picks) < len(tl):
                s = rng.integers(0, len(tl)-bs+1)
                picks.extend(tl[s:s+bs])
            for t in picks[:len(tl)]:
                Cte += data[nm]["Cte_tok"][int(t)]
        for b in (8, 28, 57): out[("topB", b)].append(d.topB(Cte, b))
        for t in (2425, 3025, 3620):
            out[("eq", t)].append(d.eq(Cte, t)); out[("wf", t)].append(d.wf(Cte, t))
        out[("perlayer8",)].append(d.perlayer_k(Cte, 8)); out[("global336",)].append(d.wf(Cte, 336))
    return out
for d in (Ds['pooled'], Ds['campaign']):
    bs = boot(d)
    def ci(k):
        v = np.asarray(bs[k]); return 100*np.percentile(v, 2.5), 100*np.percentile(v, 97.5)
    print(f"\n{d.label}:")
    for b in (8, 28, 57): print(f"  OOS top-{b:>2}: {ci(('topB',b))[0]:.1f}..{ci(('topB',b))[1]:.1f}%")
    for t in (2425, 3620):
        lo, hi = ci(('wf', t)); lo2, hi2 = ci(('eq', t))
        print(f"  wf@{t}: {lo:.1f}..{hi:.1f}%   eq@{t}: {lo2:.1f}..{hi2:.1f}%")
    for t, lab in ((2425, "wf-eq @2425"), (3620, "wf-eq @3620")):
        dv = np.asarray(bs[('wf', t)]) - np.asarray(bs[('eq', t)])
        print(f"  {lab}: {100*np.percentile(dv,2.5):+.2f}..{100*np.percentile(dv,97.5):+.2f} pp")
    dv = np.asarray(bs[('wf', 3025)]) - np.asarray(bs[('wf', 2425)])
    print(f"  wf +600 slots (2425->3025): {100*np.percentile(dv,2.5):+.2f}..{100*np.percentile(dv,97.5):+.2f} pp")
    dv = np.asarray(bs[('eq', 3025)]) - np.asarray(bs[('eq', 2425)])
    print(f"  eq +600 slots (2425->3025): {100*np.percentile(dv,2.5):+.2f}..{100*np.percentile(dv,97.5):+.2f} pp")
    dv = np.asarray(bs[('perlayer8',)]) - np.asarray(bs[('global336',)])
    print(f"  per-layer top-8 (336 slots) minus global-336: {100*np.percentile(dv,2.5):+.2f}..{100*np.percentile(dv,97.5):+.2f} pp")

# ---------- SECTION H: power statement ----------
hdr("H. POWER / SAMPLE-SIZE STATEMENT")
za, zb = 1.959963984540054, 0.8416212335729143
def n_two_prop(p1, p2, power_z=zb):
    pb = (p1+p2)/2
    return ((za*math.sqrt(2*pb*(1-pb)) + power_z*math.sqrt(p1*(1-p1)+p2*(1-p2)))/(abs(p1-p2)))**2
print("two-proportion test (alpha=0.05 two-sided, 80% power), per LAYER, counts in activations;")
print("tokens/arm per layer = n/8 (8 activations per token per layer); one trace feeds all 42 layers at once.")
print(f"{'p1':>5} {'p2':>5} {'n/arm (acts)':>12} {'tokens/arm':>10} {'tokens, both arms':>18}")
for p1, p2 in ((0.45, 0.50), (0.41, 0.45), (0.28, 0.33), (0.80, 0.85)):
    n = n_two_prop(p1, p2)
    print(f"{p1:>5} {p2:>5} {math.ceil(n):>12} {math.ceil(n/8):>10} {2*math.ceil(n/8):>18}")
print("\nWilson half-width at p=0.5 vs per-layer test-half size (activations = 8*tokens):")
print(f"{'tokens in test half':>19} {'acts/layer':>10} {'half-width':>10}")
for T in (3, 6, 30, 39, 125, 250, 1000, 2000):
    print(f"{T:>19} {8*T:>10} {100*1.96*math.sqrt(0.25/(8*T)):>9.2f}pp")
print("\nn for +/-1pp (95%, p~0.5): 9604 acts/layer ~ 1200 tokens; +/-0.5pp: 38416 acts ~ 4800 tokens.")
print("Caveats: (i) Wilson assumes iid activations; adjacent-token routing overlap (0.19-0.30 here)")
print("clusters hits -> inflate 2-4x; (ii) prompt-to-prompt variance dominates (campaign vs realtext")
print("in-sample rates differ strongly) -> collect across >=8 independent prompts.")
print("\nanalysis complete; caches+CSVs in /var/lib/insignia/analysis/hotset/")
PYEOF
