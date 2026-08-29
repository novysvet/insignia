#!/usr/bin/env python3
"""Parse route traces + dfdump on glm-box; cache npz under analysis/predict/.

Trace formats (from src/glm53_generate.cu):
  route-*.txt:       "token layer e0..e7 s0..s7"
  early-route-*.txt: "token layer overlap p0..p7 a0..a7" (19 fields)
dfdump (src/glm53_dflash2.cu), stream of tagged records:
  tag1 commit:  i32 count, i32 pos0, f32[count*5*4096]   (target captures 5/14/24/33/42)
  tag2 forward: i32 anchor, i32 anchor_position
  tag3 fwd-int: i32 anchor_position, i32 n1, f32[n1], i32 n2, f32[n2] (x_block, hidden)
  tag4 ltrace:  i8 layer, i32 n, f32[n]
  tag5 itrace:  i8 layer, i8 stage, i32 rows, i32 cols, f32[rows*cols]
"""
import json
import os
import struct
import sys
from collections import defaultdict

import numpy as np

BASE = "/var/lib/insignia"
OUT = f"{BASE}/analysis/predict"
os.makedirs(OUT, exist_ok=True)
E = 288
K = 8


def load_route(path):
    by_tok = defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        f = line.split()
        if len(f) < 11:
            continue
        tok, layer = int(f[0]), int(f[1])
        k = (len(f) - 2) // 2
        by_tok[tok][layer] = ([int(x) for x in f[2:2 + k]],
                              [float(x) for x in f[2 + k:2 + 2 * k]])
    toks = sorted(by_tok)
    layers = sorted(next(iter(by_tok.values())))
    Ep = np.zeros((len(toks), len(layers), K), dtype=np.int16)
    Sp = np.zeros((len(toks), len(layers), K), dtype=np.float32)
    for i, t in enumerate(toks):
        for j, L in enumerate(layers):
            e, s = by_tok[t][L]
            Ep[i, j] = e
            Sp[i, j] = s
    return toks, layers, Ep, Sp


def load_early(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        f = [int(x) for x in line.split()]
        if len(f) != 19:
            continue
        tok, layer, ov = f[:3]
        rows.append((tok, layer, ov, f[3:11], f[11:19]))
    return rows


def parse_dfdump(path):
    data = open(path, "rb").read()
    pos, recs = 0, []
    cap_all = []          # tag1 payloads
    tag_counts = defaultdict(int)
    while pos < len(data):
        tag = data[pos]
        pos += 1
        tag_counts[tag] += 1
        if tag == 1:
            count, pos0 = struct.unpack_from("<ii", data, pos)
            pos += 8
            n = count * 5 * 4096
            cap_all.append((count, pos0, np.frombuffer(data, "<f4", n, pos).copy()))
            pos += 4 * n
        elif tag == 2:
            anchor, apos = struct.unpack_from("<ii", data, pos)
            pos += 8
            recs.append(("fwd", anchor, apos))
        elif tag == 3:
            apos, = struct.unpack_from("<i", data, pos)
            pos += 4
            n1, = struct.unpack_from("<i", data, pos)
            pos += 4 + 4 * n1
            n2, = struct.unpack_from("<i", data, pos)
            pos += 4 + 4 * n2
            recs.append(("fwdint", apos, n1, n2))
        elif tag == 4:
            layer = struct.unpack_from("<b", data, pos)[0]
            pos += 1
            n, = struct.unpack_from("<i", data, pos)
            pos += 4 + 4 * n
            recs.append(("ltrace", layer, n))
        elif tag == 5:
            layer, stage = struct.unpack_from("<bb", data, pos)
            pos += 2
            rows, cols = struct.unpack_from("<ii", data, pos)
            pos += 8 + 4 * rows * cols
            recs.append(("itrace", layer, stage, rows, cols))
        else:
            raise RuntimeError(f"unknown tag {tag} at {pos - 1}")
    return data, recs, cap_all, dict(tag_counts)


def main():
    info = {}
    for name in ("route-realtext", "route-campaign", "early-route-math"):
        toks, layers, Ep, Sp = load_route(f"{BASE}/{name}.txt")
        np.savez_compressed(f"{OUT}/{name}.npz", toks=toks, layers=layers, E=Ep, S=Sp)
        info[name] = dict(tokens=len(toks), layers=[int(layers[0]), int(layers[-1])],
                          tok_range=[int(toks[0]), int(toks[-1])])
        print(f"{name}: tokens={len(toks)} layers={layers[0]}..{layers[-1]}")

    rows = load_early(f"{BASE}/early-route-math.txt")
    P = np.array([r[3] for r in rows], dtype=np.int16)
    A = np.array([r[4] for r in rows], dtype=np.int16)
    O = np.array([r[2] for r in rows], dtype=np.int16)
    np.savez_compressed(f"{OUT}/early-math.npz", P=P, A=A, O=O)
    info["early-route-math rows"] = len(rows)

    data, recs, cap_all, tag_counts = parse_dfdump(f"{BASE}/dfdump/r12")
    total_commit = sum(c for c, _, _ in cap_all)
    pos0s = [p for _, p, _ in cap_all]
    print(f"dfdump r12: bytes={len(data)} tags={dict(sorted(tag_counts.items()))}")
    print(f"  commits: n={len(cap_all)} counts={[c for c, _, _ in cap_all]} "
          f"pos0={pos0s[:20]} total_committed_tokens={total_commit}")
    fwd = [r for r in recs if r[0] == "fwd"]
    fwdint = [r for r in recs if r[0] == "fwdint"]
    print(f"  fwd(tag2): n={len(fwd)} anchor_positions={[r[2] for r in fwd[:20]]}")
    print(f"  fwdint(tag3): n={len(fwdint)} shapes={[(r[2], r[3]) for r in fwdint[:5]]}")
    other = [r for r in recs if r[0] in ("ltrace", "itrace")]
    print(f"  ltrace/itrace records: {len(other)}")
    if cap_all:
        caps = np.concatenate([c.reshape(-1).reshape(c.shape[0], 5, 4096) if False else c
                               for c in [x[2] for x in cap_all]])
        caps = caps.reshape(-1, 5, 4096)
        np.save(f"{OUT}/df_r12_captures.npy", caps)
        print(f"  cached captures: shape={caps.shape} "
              f"norm_mean={float(np.linalg.norm(caps, axis=2).mean()):.3f}")
    info["dfdump"] = dict(bytes=len(data), tags={str(k): v for k, v in tag_counts.items()},
                          commit_rounds=len(cap_all), committed_tokens=total_commit,
                          fwd_rounds=len(fwd))
    json.dump(info, open(f"{OUT}/parse_info.json", "w"), indent=1)
    print("cached to", OUT)


if __name__ == "__main__":
    main()
