#!/usr/bin/env python3
"""Acceptance analysis: P(m|k), E[len], d(k), calibrated (a,b), T(k), k*, break-even.

Inputs: parsed caches pulled from glm-box under accept/ (runs.csv, dfdump_events.csv,
early-multi-*_batches.csv / _d_emp.csv). Outputs: CSV tables under out/ (and pushed
back to /var/lib/insignia/analysis/accept/).
"""
import csv
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ACC = os.path.join(HERE, "accept")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

SCALAR_MS = 570.9          # documented scalar median (session 6)
REC_MIB = 13.56            # NVFP4 expert record size
PCIE_GBPS = 23.2           # measured pinned H2D
RATIO = {1: 1.0, 2: 0.903, 3: 0.859, 4: 0.825, 5: 0.785}
# linear extrapolation of the ratio decline (deltas -0.044,-0.034,-0.040 => -0.0393/step)
_r = 0.785
for k in (6, 7, 8):
    _r -= 0.0393
    RATIO[k] = round(_r, 4)
D_MODEL = {k: 336 * k * RATIO[k] for k in range(1, 9)}

# measured verify-union sizes (early-multi-df-k7, c64 build, batch verify)
D_MEAS = {7: 1506.2, 4: 1067.0}   # mean over rounds (n=5, n=1)
# blended d(k): use measured where available, scale model elsewhere by meas/model at
# the anchor points (0.7938 @k4 -> factor 0.962; 0.6405 @k7 -> factor 0.907)
def d_blend(k):
    if k == 1:
        return 336.0
    if k in D_MEAS:
        return D_MEAS[k]
    f = 0.962 if k < 7 else 0.907
    return D_MODEL[k] * f

rows = list(csv.DictReader(open(os.path.join(ACC, "runs.csv"))))
df = [r for r in rows if r["kind"] == "dflash"]
sc = [r for r in rows if r["kind"] == "scalar"]

# ---------- 1. per-prompt acceptance table ----------
per = []
for r in df:
    hist = json.loads(r["accepted_histogram"]) if r["accepted_histogram"] else {}
    n = sum(hist.values())
    if not n:
        continue
    em = sum(int(m) * c for m, c in hist.items()) / n
    per.append({
        "variant": r["variant"], "case": r["case"], "k_max": int(r["verify_k"]),
        "prompt_tokens": r["prompt_tokens"], "generated": r["generated"],
        "rounds": r["rounds"], "empty": r["empty_rounds"],
        "accepted_per_round": float(r["accepted_per_round"]),
        "E_m_hist": round(em, 3), "E_len": round(em + 1, 3),
        "ms_per_token": float(r["ms_per_token"]),
        "draft_ms": float(r["draft_ms_per_round"]),
        "verify_ms": float(r["verify_ms_per_verified_round"]),
        "verified_rounds": r["verified_rounds"],
        "fallback_ms_empty_mean": round(float(r["fallback_total_ms"]) / int(r["empty_rounds"]), 1)
                                  if int(r["empty_rounds"]) else "",
        "hist": r["accepted_histogram"],
        "gen_distinct_ratio": r["distinct_ratio"],
        "gen_bigram_repeat": r["bigram_repeat"],
        "gap_median": r.get("gap_median", ""),
    })
for p in per:
    p["T_tok_s"] = round(1000.0 / p["ms_per_token"], 3)
    p["speedup_vs_scalar_570_9"] = round(SCALAR_MS / p["ms_per_token"], 3)
    p["be_E_m_at_wall"] = round((p["draft_ms"] + p["verify_ms"]) / SCALAR_MS - 1, 2)
with open(os.path.join(OUT, "per_prompt.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(per[0]))
    w.writeheader(); w.writerows(per)

# ---------- 2. pooled P(m|k) ----------
pools = {"k4_gsm8k": [], "k7_gsm8k_math500": [], "k7_baseline_gen4": []}
for p in per:
    if p["variant"] == "math":
        pools["k4_gsm8k"].append(p)
    elif p["variant"].startswith(("s6-c64", "s6-vramtier")):
        pools["k7_gsm8k_math500"].append(p)
    else:
        pools["k7_baseline_gen4"].append(p)

pm_rows = []
pool_stats = {}
for name, ps in pools.items():
    if not ps:
        continue
    agg = defaultdict(int)
    for p in ps:
        for m, c in json.loads(p["hist"]).items():
            agg[int(m)] += c
    n = sum(agg.values())
    em = sum(m * c for m, c in agg.items()) / n
    pool_stats[name] = {"E_m": em, "n": n, "hist": agg}
    for m in range(0, max(agg) + 1):
        pm_rows.append({"pool": name, "k_label": name.split("_")[0],
                        "m": m, "count": agg.get(m, 0),
                        "P_m_given_k": round(agg.get(m, 0) / n, 4)})
with open(os.path.join(OUT, "pmk_pooled.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["pool", "k_label", "m", "count", "P_m_given_k"])
    w.writeheader(); w.writerows(pm_rows)

# fixed-k reference pools (no adaptive-k mixing):
#  - dfdump r12 (52-tok prompt, k7, adaptive off): commit counts per round;
#    count==1 rounds are ambiguous between empty (m=0) and verified m=1.
#  - campaign prompt (16 tok) from audits/dflash2-regression-artifact.md
fixed = {
    "r12_fixed_k7_m_if_verified": {1: 3, 4: 1, 3: 1, 2: 1},
    "r12_fixed_k7_m_if_empty": {0: 3, 4: 1, 3: 1, 2: 1},
    "campaign_fixed_k4": {0: 1, 1: 1, 2: 1, 4: 24},
    "campaign_fixed_k7": {0: 1, 1: 1, 2: 1, 5: 1, 7: 13},
}
fixed_rows = []
for name, hist in fixed.items():
    n = sum(hist.values())
    em = sum(m * c for m, c in hist.items()) / n
    pool_stats[name] = {"E_m": em, "n": n, "hist": hist}
    for m in range(0, 8):
        fixed_rows.append({"pool": name, "k_label": "k" + name.split("_")[-2].replace("fixed", ""),
                           "m": m, "count": hist.get(m, 0),
                           "P_m_given_k": round(hist.get(m, 0) / n, 4)})
with open(os.path.join(OUT, "pmk_fixed_references.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["pool", "k_label", "m", "count", "P_m_given_k"])
    w.writeheader(); w.writerows(fixed_rows)

# dfdump r12 exact per-round (52-tok prompt, k7, adaptive off implied)
ev = list(csv.DictReader(open(os.path.join(ACC, "dfdump_events.csv"))))
rounds_r12, i = [], 0
seq = [e for e in ev]
idx = 0
round_no = 0
while idx < len(seq):
    e = seq[idx]
    if e["tag"] == "forward":
        round_no += 1
        j = idx + 1
        while j < len(seq) and seq[j]["tag"] == "fwd_intermediates":
            j += 1
        c = seq[j] if j < len(seq) else None
        if c and c["tag"] == "commit":
            cnt = int(c["count"])
            rounds_r12.append({
                "round": round_no, "draft_k": 7,
                "commit_count": cnt,
                "m_if_verified": cnt,
                "m_if_empty": 0 if cnt == 1 else cnt,
                "ambiguity": "m=0(empty) or m=1(verified)" if cnt == 1 else "",
                "pos0": c["pos0"], "anchor": e["anchor"]})
            idx += 2
            continue
    idx += 1
with open(os.path.join(OUT, "rounds_dfdump_r12.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rounds_r12[0]))
    w.writeheader(); w.writerows(rounds_r12)

# ---------- 3. d(k) table ----------
d_rows = []
for k in range(1, 9):
    d_rows.append({
        "k": k, "ratio_model": RATIO.get(k, ""), "d_model": round(D_MODEL.get(k, 0), 1),
        "d_measured": D_MEAS.get(k, ""),
        "d_blend": round(d_blend(k), 1),
        "ratio_measured": round(D_MEAS[k] / (336 * k), 4) if k in D_MEAS else ""})
with open(os.path.join(OUT, "d_of_k.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(d_rows[0]))
    w.writeheader(); w.writerows(d_rows)

# ---------- 4. calibration ----------
# (a) drafter cost: direct
draft_ms = [float(r["draft_ms_per_round"]) for r in df if int(r["generated"]) >= 32]
a_draft = sum(draft_ms) / len(draft_ms)
# (b) fallback (empty-round) step: full scalar target step incl. 336 records
fb = [(float(r["fallback_total_ms"]) / int(r["empty_rounds"]))
      for r in df if int(r["empty_rounds"])]
fb.sort()
fb_med = fb[len(fb) // 2]
# (c) verify wall by k group (steady runs only)
v4 = [float(r["verify_ms_per_verified_round"]) for r in df
      if r["kind"] == "dflash" and r["verify_k"] == "4"]
v7 = [float(r["verify_ms_per_verified_round"]) for r in df
      if r["kind"] == "dflash" and r["verify_k"] == "7" and int(r["generated"]) >= 32]
v4m, v7m = sum(v4) / len(v4), sum(v7) / len(v7)
b_two_point = (v7m - v4m) / (D_MEAS[7] - D_MEAS[4])
a_two_point = v4m - b_two_point * D_MEAS[4]
b_constrained = (v7m - a_draft) / D_MEAS[7]   # force a = draft cost
b4_constrained = (v4m - a_draft) / D_MEAS[4]
b_pcie_floor = REC_MIB * 1.048576 / PCIE_GBPS  # MB/(GB/s) = ms
b_nvme = [REC_MIB * 1.048576 / float(r.split('","')[2]) * 1000
          for r in []]  # filled below from odirect
nvme_gbps = []
for r in df:
    od = r.get("odirect", "")
    if od:
        try:
            parts = json.loads(od)
            g = float(parts[2])
        except Exception:
            continue
        if g > 1.0:
            nvme_gbps.append(g)
calib = [
    {"quantity": "a_drafter_ms", "value": round(a_draft, 1),
     "source": "mean draft ms/round over gen>=32 dflash runs (n=%d)" % len(draft_ms)},
    {"quantity": "fallback_step_ms_median", "value": round(fb_med, 1),
     "source": "fallback_total_ms/empty_rounds, n=%d empty rounds" % len(fb)},
    {"quantity": "verify_ms_k4_mean", "value": round(v4m, 1),
     "source": "math variant k=4, n=%d verified rounds-pools (%d runs)" % (sum(int(r['verified_rounds']) for r in df if r['verify_k']=='4'), len(v4))},
    {"quantity": "verify_ms_k7_mean", "value": round(v7m, 1),
     "source": "c64+vramtier k=7 gen32, n=%d verified rounds (%d runs)" % (sum(int(r['verified_rounds']) for r in df if r['verify_k']=='7' and int(r['generated'])>=32), len(v7))},
    {"quantity": "b_two_point_ms_per_record", "value": round(b_two_point, 3),
     "source": "(V7-V4)/(d7-d4) = (%.0f-%.0f)/(1506-1067)" % (v7m, v4m)},
    {"quantity": "a_two_point_ms", "value": round(a_two_point, 1),
     "source": "intercept V4 - b*d4 (can be ~0; dense verify cost folds into b)"},
    {"quantity": "b_k7_constrained_ms_per_record", "value": round(b_constrained, 3),
     "source": "(V7 - a_drafter)/d7 with a=17.4"},
    {"quantity": "b_k4_constrained_ms_per_record", "value": round(b4_constrained, 3),
     "source": "(V4 - a_drafter)/d4 with a=17.4"},
    {"quantity": "b_pcie_floor_ms_per_record", "value": round(b_pcie_floor, 3),
     "source": "13.56 MiB / 23.2 GB/s pinned H2D (all-resident floor)"},
    {"quantity": "b_nvme_ms_per_record_range", "value": "2.93-3.88",
     "source": "13.56 MiB / measured 3.66-4.85 GB/s O_DIRECT (nvme_gbps n=%d: %.2f-%.2f)" % (len(nvme_gbps), min(nvme_gbps) if nvme_gbps else 0, max(nvme_gbps) if nvme_gbps else 0)},
    {"quantity": "b_effective_recommended", "value": 1.8,
     "source": "constrained fits 1.76-1.83 agree; fallback step 640ms/336rec=1.9 corroborates"},
    {"quantity": "scalar_ms_per_token", "value": SCALAR_MS,
     "source": "s6 documented; local re-parse median 578 (gen32 subset 566)"},
]
with open(os.path.join(OUT, "calibration.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["quantity", "value", "source"])
    w.writeheader(); w.writerows(calib)

# ---------- 5. per-position acceptance profile p_i = P(m >= i) ----------
# readable directly from each pool's histogram tail sums (positions beyond the
# drafted width cannot accept, so pooled profiles are mild underestimates).
def profile(hist):
    n = sum(hist.values())
    ps = {}
    for i in range(1, 8):
        tail = sum(c for m, c in hist.items() if m >= i)
        ps[i] = tail / n
    return ps

prof_rows = []
profiles = {}
for name in ("k4_gsm8k", "k7_gsm8k_math500", "campaign_fixed_k4", "campaign_fixed_k7",
             "r12_fixed_k7_m_if_verified"):
    st = pool_stats.get(name)
    if not st:
        continue
    profiles[name] = profile(st["hist"])
    for i, p in profiles[name].items():
        prof_rows.append({"pool": name, "position_i": i,
                          "p_i_accept": round(p, 4),
                          "E_m_prefix": round(sum(profiles[name][j] for j in range(1, i + 1)), 4)})
with open(os.path.join(OUT, "position_acceptance_profile.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["pool", "position_i", "p_i_accept", "E_m_prefix"])
    w.writeheader(); w.writerows(prof_rows)

# ---------- 5b. T(k), k*, break-even ----------
# E[m](k) = sum_{i<=k} p_i from the pool's own profile (self-consistent truncation).
def E_m_of(k, cls):
    src = {"gsm8k": "k4_gsm8k", "campaign": "campaign_fixed_k7"}[cls]
    return sum(profiles[src][j] for j in range(1, k + 1) if j in profiles[src])

# smoothed d curve: model shape x factor interpolated between measured anchors
def d_smooth(k):
    if k == 1:
        return 336.0
    if k in D_MEAS:
        return D_MEAS[k]
    f = 0.962 if k <= 4 else (0.907 if k >= 7 else 0.962 + (0.907 - 0.962) * (k - 4) / 3)
    return D_MODEL[k] * f

ONE_ROW_MS = 578.0   # measured scalar step (gen32 median); fallback median 643
tk_rows = []
for cls in ("gsm8k", "campaign"):
    for b in (0.613, 1.8, "measured_1row"):
        for k in range(1, 8):
            em = E_m_of(k, cls)
            elen = em + 1
            d = d_smooth(k)
            if b == "measured_1row":
                if k != 1:
                    continue
                cost = ONE_ROW_MS   # dense compute dominates; a+b*d(1) is unattainable
            elif k == 1:
                cost = ONE_ROW_MS   # override: real single-row forward is compute-bound
            else:
                cost = a_draft + b * d
            t = 1000.0 * elen / cost          # tokens/second
            tk_rows.append({"class": cls, "b_ms_per_record": b, "k": k,
                            "E_m": round(em, 3), "E_len": round(elen, 3),
                            "d_k": round(d, 1), "round_cost_ms": round(cost, 1),
                            "tokens_per_s": round(t, 3),
                            "speedup_vs_scalar": round(t / (1000.0 / SCALAR_MS), 3),
                            "break_even_E_m": round(cost / SCALAR_MS - 1, 2)})
with open(os.path.join(OUT, "tk_optimal_k.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(tk_rows[0]))
    w.writeheader(); w.writerows(tk_rows)

# marginal position table: include position i iff p_i/(b*dd) > T/1000
marg_rows = []
for cls in ("gsm8k", "campaign"):
    src = {"gsm8k": "k4_gsm8k", "campaign": "campaign_fixed_k7"}[cls]
    for b in (0.613, 1.8):
        for i in range(2, 8):
            dd = d_smooth(i) - d_smooth(i - 1)
            mc = b * dd
            p_i = profiles[src].get(i, 0.0)
            marg_rows.append({"class": cls, "b_ms_per_record": b, "position_i": i,
                              "marginal_cost_ms": round(mc, 1),
                              "p_i": round(p_i, 4),
                              "marginal_tok_per_s": round(1000.0 * p_i / mc, 3),
                              "avg_T_at_k": next((t["tokens_per_s"] for t in tk_rows
                                                  if t["class"] == cls and
                                                  t["b_ms_per_record"] == b and
                                                  t["k"] == i - 1), ""),
                              "include_if": ""})
with open(os.path.join(OUT, "marginal_position.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["class", "b_ms_per_record", "position_i",
                                       "marginal_cost_ms", "p_i", "marginal_tok_per_s",
                                       "avg_T_at_k", "include_if"])
    w.writeheader(); w.writerows(marg_rows)

# ---------- 6. IO counters per run ----------
io_rows = []
for r in df:
    h = r.get("host_cache", ""); v = r.get("vram_cache", ""); o = r.get("odirect", "")
    hc = json.loads(h) if h else []
    vc = json.loads(v) if v else []
    oc = json.loads(o) if o else []
    nvme_gib = float(oc[0]) if oc else 0.0
    rounds = int(r["rounds"]) or 1
    io_rows.append({
        "variant": r["variant"][:14], "case": r["case"], "k": r["verify_k"],
        "host_hits": hc[0] if hc else "", "host_lookups": hc[1] if hc else "",
        "host_hit_pct": hc[2] if hc else "", "avoided_gib": hc[3] if hc else "",
        "slots": hc[4] if hc else "",
        "vram_hits": vc[0] if vc else "", "vram_lookups": vc[1] if vc else "",
        "vram_slots": vc[4] if vc else "",
        "nvme_gib": oc[0] if oc else "", "nvme_s": oc[1] if oc else "",
        "nvme_gbps": oc[2] if oc else "",
        "nvme_records_total": round(nvme_gib * 1024 / REC_MIB, 1),
        "nvme_records_per_round": round(nvme_gib * 1024 / REC_MIB / rounds, 2),
        "union_hits_pct": "",  # not logged in these runs
    })
with open(os.path.join(OUT, "io_counters.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(io_rows[0]))
    w.writeheader(); w.writerows(io_rows)

# ---------- console digest ----------
print("== P(m|k) pools ==")
for name, st in pool_stats.items():
    print(f"{name:32s} n={st['n']:3d} E[m]={st['E_m']:.2f} E[len]={st['E_m']+1:.2f} "
          f"hist={dict(sorted(st['hist'].items()))}")

# mixed-round throughput (empty rounds pay the fallback step) per measured pool
print("\n== mixed-round T (empty rounds pay draft+fallback) ==")
vmap = {"k4_gsm8k": v4m, "k7_gsm8k_math500": v7m}
for name in ("k4_gsm8k", "k7_gsm8k_math500"):
    st = pool_stats[name]
    p0 = st["hist"].get(0, 0) / st["n"]
    w_round = (1 - p0) * (a_draft + vmap[name]) + p0 * (a_draft + fb_med)
    t_mixed = 1000.0 * (st["E_m"] + 1) / w_round
    print(f"{name:24s} P0={p0:.3f} W={w_round:7.1f} ms/round  T_mixed={t_mixed:.2f} tok/s "
          f"({1000/t_mixed:.0f} ms/tok) vs scalar {SCALAR_MS} -> "
          f"{'WINS' if 1000/t_mixed < SCALAR_MS else 'LOSES'}")
print("\n== dfdump r12 rounds (52-tok prompt, k7) ==")
for rr in rounds_r12:
    print(rr)
print("\n== d(k) ==")
for d in d_rows:
    print(d)
print("\n== calibration ==")
for c in calib:
    cv = str(c["value"])
    print(f"{c['quantity']:36s} {cv[:14]:>14s}  {c['source'][:70]}")
print("\n== T(k) highlights (b=1.8) ==")
best = {}
for t in tk_rows:
    if t["b_ms_per_record"] == 1.8:
        key = t["class"]
        if key not in best or t["tokens_per_s"] > best[key]["tokens_per_s"]:
            best[key] = t
for k_, v_ in best.items():
    print(f"optimal for {k_}: k*={v_['k']} T={v_['tokens_per_s']} tok/s "
          f"(speedup {v_['speedup_vs_scalar']}x, E[m]={v_['E_m']})")
print("\noutputs in", OUT)
