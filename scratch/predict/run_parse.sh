set -e
mkdir -p /var/lib/insignia/analysis/predict
cd /var/lib/insignia/analysis/predict
python3 - <<'PYEOF'
import json, os, struct
from collections import defaultdict
import numpy as np

BASE = "/var/lib/insignia"
OUT = BASE + "/analysis/predict"
K = 8

def load_route(path):
    by_tok = defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        f = line.split()
        if len(f) < 11: continue
        tok, layer = int(f[0]), int(f[1])
        k = (len(f) - 2) // 2
        by_tok[tok][layer] = ([int(x) for x in f[2:2+k]], [float(x) for x in f[2+k:2+2*k]])
    toks = sorted(by_tok)
    layers = sorted(next(iter(by_tok.values())))
    Ep = np.zeros((len(toks), len(layers), K), dtype=np.int16)
    Sp = np.zeros((len(toks), len(layers), K), dtype=np.float32)
    for i, t in enumerate(toks):
        for j, L in enumerate(layers):
            e, s = by_tok[t][L]
            Ep[i, j] = e; Sp[i, j] = s
    return toks, layers, Ep, Sp

def parse_dfdump(path):
    data = open(path, "rb").read()
    pos, recs, cap_all = 0, [], []
    tag_counts = defaultdict(int)
    while pos < len(data):
        tag = data[pos]; pos += 1
        tag_counts[tag] += 1
        if tag == 1:
            count, pos0 = struct.unpack_from("<ii", data, pos); pos += 8
            n = count * 5 * 4096
            cap_all.append((count, pos0, np.frombuffer(data, "<f4", n, pos).copy()))
            pos += 4 * n
        elif tag == 2:
            anchor, apos = struct.unpack_from("<ii", data, pos); pos += 8
            recs.append(("fwd", anchor, apos))
        elif tag == 3:
            apos, = struct.unpack_from("<i", data, pos); pos += 4
            n1, = struct.unpack_from("<i", data, pos); pos += 4 + 4*n1
            n2, = struct.unpack_from("<i", data, pos); pos += 4 + 4*n2
            recs.append(("fwdint", apos, n1, n2))
        elif tag == 4:
            layer = struct.unpack_from("<b", data, pos)[0]; pos += 1
            n, = struct.unpack_from("<i", data, pos); pos += 4 + 4*n
            recs.append(("ltrace", layer, n))
        elif tag == 5:
            layer, stage = struct.unpack_from("<bb", data, pos); pos += 2
            rows, cols = struct.unpack_from("<ii", data, pos); pos += 8 + 4*rows*cols
            recs.append(("itrace", layer, stage, rows, cols))
        else:
            raise RuntimeError(f"unknown tag {tag} at {pos-1}")
    return data, recs, cap_all, dict(tag_counts)

info = {}
for name in ("route-realtext", "route-campaign", "early-route-math"):
    toks, layers, Ep, Sp = load_route(f"{BASE}/{name}.txt")
    np.savez_compressed(f"{OUT}/{name}.npz", toks=np.array(toks), layers=np.array(layers), E=Ep, S=Sp)
    info[name] = dict(tokens=len(toks))
    print(f"{name}: tokens={len(toks)} layers={layers[0]}..{layers[-1]} tok_range=({toks[0]},{toks[-1]})")

data, recs, cap_all, tag_counts = parse_dfdump(f"{BASE}/dfdump/r12")
total_commit = sum(c for c, _, _ in cap_all)
print(f"dfdump r12: bytes={len(data)} tags={dict(sorted(tag_counts.items()))}")
print(f"  tag1 commits: n={len(cap_all)} counts={[c for c,_,_ in cap_all]}")
print(f"  pos0 list: {[p for _,p,_ in cap_all]}")
print(f"  total committed tokens = {total_commit}")
fwd = [r for r in recs if r[0]=="fwd"]; fwdint = [r for r in recs if r[0]=="fwdint"]
print(f"  tag2 fwd rounds: {len(fwd)} (anchor, anchor_pos) = {[(r[1],r[2]) for r in fwd]}")
print(f"  tag3 fwdint rounds: {len(fwdint)} shapes(n1,n2)={set((r[2],r[3]) for r in fwdint)}")
print(f"  tag4/5 (ltrace/itrace) records: {sum(1 for r in recs if r[0] in ('ltrace','itrace'))}")
if cap_all:
    caps = np.concatenate([x[2].reshape(-1, 5, 4096) for x in cap_all], axis=0)
    np.save(f"{OUT}/df_r12_captures.npy", caps)
    nrm = np.linalg.norm(caps, axis=2)
    print(f"  cached captures {caps.shape} row-norm mean/min/max = {nrm.mean():.2f}/{nrm.min():.2f}/{nrm.max():.2f}")
info["dfdump"] = dict(bytes=len(data), tags={str(k): v for k, v in tag_counts.items()},
                      commit_rounds=len(cap_all), committed_tokens=total_commit, fwd_rounds=len(fwd))
json.dump(info, open(f"{OUT}/parse_info.json", "w"), indent=1)
print("OK cached to " + OUT)
PYEOF
ls -la /var/lib/insignia/analysis/predict/
