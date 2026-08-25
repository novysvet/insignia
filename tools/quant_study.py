#!/usr/bin/env python3
"""Quantization design study: where does MXFP4 lose quality on Qwen3.5-9B weights?

Variants per 32-wide group, all keeping the dp4a-compatible contract
(codes index a 16-entry signed-int8 table, scale applied per group):
  mxfp4      E2M1 codes x2, E8M0 (pow2) scale            [checkpoint baseline]
  e2m1+f16   E2M1 codes, MSE-optimal fp16 scale
  e2m1+f32   E2M1 codes, MSE-optimal fp32 scale           [scale headroom bound]
  lloyd8     per-tensor learned integer codebook (k-means, 8 mags), fp16 scales
  lloyd8+f32 same with fp32 scales                        [codebook+scale bound]
  e2m1+clip  E2M1 + E8M0 with optimal clip of the scale   [free E8M0 improvement]
"""
import json
import pathlib
import sys
import numpy as np

snap = pathlib.Path(sys.argv[1])
idx = json.loads((snap / 'model.safetensors.index.json').read_text())
files = sorted(set(snap / v for v in idx['weight_map'].values()))

hdrs = []
for f in files:
    with f.open('rb') as fh:
        n = int.from_bytes(fh.read(8), 'little')
        hdrs.append((f, json.loads(fh.read(n)), 8 + n))

def load(name):
    for f, hdr, start in hdrs:
        if name in hdr:
            v = hdr[name]
            fh = f.open('rb')
            fh.seek(start + v['data_offsets'][0])
            raw = fh.read(v['data_offsets'][1] - v['data_offsets'][0])
            a = np.frombuffer(raw, dtype=np.float32 if v['dtype'] == 'F32' else np.uint16).reshape(v['shape'])
            return (a.astype(np.float32) if v['dtype'] == 'F32' else (a.astype(np.uint32) << 16).view(np.float32))
    raise KeyError(name)

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)

def quant_rtn(w, levels, scale):
    out = np.empty_like(w)
    step = max(1, int(2e7) // np.prod(w.shape[1:]))
    for i in range(0, w.shape[0], step):
        q = w[i:i+step] / scale[i:i+step]
        idx = np.abs(q[..., None] - levels).argmin(-1)
        out[i:i+step] = levels[idx] * scale[i:i+step]
    return out

def sqnr(w, wq):
    err = np.mean((w - wq) ** 2)
    sig = np.mean(w ** 2)
    return 10 * np.log10(sig / max(err, 1e-30))

def group_scale_e8m0(w):
    amax = np.abs(w).max(-1, keepdims=True)
    e = np.floor(np.log2(np.maximum(amax, 1e-30) / 6.0))
    return np.exp2(e)

def group_scale_opt(w, levels):
    # MSE-optimal scale search around amax/level_max (ternary/golden section)
    tmax = float(np.abs(levels).max())
    def mse(wl, sl):
        step = max(1, int(2e7) // np.prod(w.shape[1:]))
        acc = np.empty((wl.shape[0], 1), np.float64)
        for i in range(0, wl.shape[0], step):
            q = wl[i:i+step] / sl[i:i+step]
            idx = np.abs(q[..., None] - levels).argmin(-1)
            acc[i:i+step, 0] = np.mean((wl[i:i+step] - levels[idx] * sl[i:i+step]) ** 2, -1)
        return acc
    amax = np.abs(w).max(-1, keepdims=True) / tmax
    lo, hi = amax * 0.5, amax * 1.5
    for _ in range(12):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        e1, e2 = mse(w, m1), mse(w, m2)
        lo = np.where(e1 < e2, lo, m1)
        hi = np.where(e1 < e2, m2, hi)
    return (lo + hi) / 2

def lloyd_int_codes(w, iters=40):
    """Learn a symmetric integer codebook (8 magnitudes incl. 0) fitted on normalized weights."""
    x = np.abs(w).ravel()
    x = x[x > 0]
    x = x / np.median(x)  # normalize
    # init: log-spaced magnitudes like E2M1
    mags = np.array([0.5, 1, 1.5, 2, 3, 4, 6], np.float64)
    for _ in range(iters):
        grid = np.concatenate([[0.0], mags])
        ass = np.abs(x[..., None] - grid).argmin(-1)
        new = []
        for j in range(1, 8):
            m = x[ass == j]
            new.append(m.mean() if len(m) else mags[j - 1])
        mags = np.array(new)
    # scale so max magnitude maps into int8-friendly range: round to integers
    scale = 127.0 / mags.max()
    ints = np.round(mags * scale)
    ints = np.maximum(ints, 1)
    ints = np.unique(ints)  # dedupe
    # pad to 8 magnitudes with multiples
    while len(ints) < 7:
        cand = np.arange(ints.max(), 0, -1)
        for c in cand:
            if c not in ints:
                ints = np.sort(np.append(ints, c)); break
    table = np.concatenate([[0.0], ints]).astype(np.float64) / ints.max() * mags.max()
    return np.array(table, np.float32)  # normalized codebook (max ~ mags.max())

def run(name):
    w = load(name).astype(np.float32)
    g64full = w.reshape(-1, 64)
    if g64full.shape[0] > 100000:   # sample 64-groups to bound memory; 32-view stays paired
        sel = np.random.default_rng(0).choice(g64full.shape[0], size=100000, replace=False)
        g64full = g64full[sel]
    g64 = g64full
    g = g64full.reshape(-1, 32)
    out = {}
    def to_e4m3(x):
        m = np.floor(np.log2(np.maximum(np.abs(x), 1e-30)))
        m = np.clip(m, -6, 8)
        frac = x / np.exp2(m)
        q = np.round(frac * 8) / 8
        return np.sign(x) * q * np.exp2(m)
    def best_pow2(w):
        amax = np.abs(w).max(-1, keepdims=True) / 6.0
        e = np.log2(np.maximum(amax, 1e-30))
        lo = np.exp2(np.floor(e)); hi = np.exp2(np.ceil(e))
        e1 = np.mean((w - quant_rtn(w, E2M1, lo)) ** 2, -1, keepdims=True)
        e2 = np.mean((w - quant_rtn(w, E2M1, hi)) ** 2, -1, keepdims=True)
        return np.where(e1 < e2, lo, hi)
    out['mxfp4(floor)'] = sqnr(g, quant_rtn(g, E2M1, group_scale_e8m0(g)))
    out['pow2opt'] = sqnr(g, quant_rtn(g, E2M1, best_pow2(g)))
    out['e4m3/32'] = sqnr(g, quant_rtn(g, E2M1, to_e4m3(group_scale_opt(g, E2M1))))
    s64 = group_scale_opt(g64, E2M1).astype(np.float16).astype(np.float32).repeat(2, 0)
    out['f16/64sh'] = sqnr(g, quant_rtn(g, E2M1, s64))
    out['f16/32'] = sqnr(g, quant_rtn(g, E2M1, group_scale_opt(g, E2M1).astype(np.float16).astype(np.float32)))
    print(f"{name.split('layers.')[-1]:32s} " + " ".join(f"{k}={v:6.2f}" for k, v in out.items()), flush=True)

tensors = [
    'model.language_model.layers.0.linear_attn.in_proj_qkv.weight',
    'model.language_model.layers.0.mlp.down_proj.weight',
    'model.language_model.layers.3.self_attn.q_proj.weight',
    'model.language_model.layers.16.mlp.up_proj.weight',
    'model.language_model.layers.31.linear_attn.out_proj.weight',
    'model.language_model.layers.7.self_attn.o_proj.weight',
    'lm_head.weight',
    'model.language_model.embed_tokens.weight',
]
for t in tensors:
    try:
        run(t)
    except Exception as e:
        print(t, 'ERR', e)
