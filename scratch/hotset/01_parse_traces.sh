#!/usr/bin/env bash
# 01_parse_traces.sh - parse the three ROUTE_TRACE files ONCE on glm-box (Arch WSL)
# and cache parsed numpy arrays + meta under /var/lib/insignia/analysis/hotset/.
# Formats handled:
#   18 fields: token layer e0..e7 s0..s7          (scores are floats)
#   19 fields: token layer ov  e0..e7 u0..u7      (all ints; early format; ov = |e SET u|)
# Usage (local Git Bash):
#   ssh glm-box "wsl -d Arch -- bash -s" < 01_parse_traces.sh
set -euo pipefail
mkdir -p /var/lib/insignia/analysis/hotset
python3 - <<'PYEOF'
import json
import numpy as np

OUT = "/var/lib/insignia/analysis/hotset"
TRACES = ["route-realtext", "route-campaign", "early-route-math"]
meta = {}
for name in TRACES:
    p = f"/var/lib/insignia/{name}.txt"
    toks, lays, ex, sc, ov = [], [], [], [], []
    nfields = {}
    dup_in_line = 0
    bad = 0
    with open(p) as f:
        for line in f:
            fl = line.split()
            if not fl:
                continue
            nf = len(fl)
            nfields[nf] = nfields.get(nf, 0) + 1
            if nf == 18:
                toks.append(int(fl[0])); lays.append(int(fl[1]))
                e = [int(x) for x in fl[2:10]]
                ex.append(e)
                sc.append([float(x) for x in fl[10:18]])
                ov.append(-1)
                if len(set(e)) != 8:
                    dup_in_line += 1
            elif nf == 19:
                toks.append(int(fl[0])); lays.append(int(fl[1]))
                e = [int(x) for x in fl[3:11]]
                ex.append(e)
                sc.append([float(x) for x in fl[11:19]])  # second expert set, cast to float store
                ov.append(int(fl[2]))
                if len(set(e)) != 8:
                    dup_in_line += 1
            else:
                bad += 1
    tok = np.asarray(toks, np.int32)
    lay = np.asarray(lays, np.int16)
    E = np.asarray(ex, np.int16)
    # For 18-field traces sc holds router scores (floats); for the 19-field trace it
    # holds the second expert set (exact ints <= 287), stored losslessly via float32.
    is_pair = (np.asarray(ov) >= 0).any()
    if is_pair:
        S2 = np.asarray(sc, np.float32).astype(np.int16)
        SC = np.zeros((len(toks), 8), np.float32)
    else:
        S2 = np.full((len(toks), 8), -1, np.int16)
        SC = np.asarray(sc, np.float32)
    OV = np.asarray(ov, np.int16)
    utok = np.unique(tok)
    ulay = np.unique(lay)
    lpt = [int((tok == t).sum()) for t in utok.tolist()]
    m = {
        "file": p,
        "lines": len(toks),
        "field_counts": nfields,
        "bad_lines": bad,
        "dup_experts_in_line": dup_in_line,
        "n_tokens": int(utok.size),
        "token_min": int(utok[0]), "token_max": int(utok[-1]),
        "n_layers": int(ulay.size),
        "layer_min": int(ulay[0]), "layer_max": int(ulay[-1]),
        "lines_per_token_min": min(lpt), "lines_per_token_max": max(lpt),
        "expert_min": int(E.min()), "expert_max": int(E.max()),
        "overlap_col_present": bool((OV >= 0).any()),
        "overlap_col_check_sum": int(OV[OV >= 0].sum()) if (OV >= 0).any() else -1,
    }
    meta[name] = m
    np.savez_compressed(f"{OUT}/{name}.npz", token=tok, layer=lay, experts=E,
                        second_set=S2, scores=SC, overlap=OV)
    print(name, json.dumps(m))

with open(f"{OUT}/meta.json", "w") as f:
    json.dump(meta, f, indent=1)
print("parse done ->", OUT)
PYEOF
ls -la /var/lib/insignia/analysis/hotset/
