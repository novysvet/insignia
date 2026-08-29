#!/usr/bin/env python3
"""Theoretical entropy ceiling for lossless coding of NVFP4 E2M1 nibbles.

Simulates group-scaled nearest-codeword E2M1 quantization of iid Gaussian
weights (GLM expert weights measured kurtosis ~3.1 => Gaussian) and reports
the per-symbol entropy of the resulting 16-symbol nibble stream, i.e. the
ceiling a per-record static rANS codec could reach. Pure CPU numpy; no
engine/checkpoint access.
"""
import math

import numpy as np

rng = np.random.default_rng(53)
G = 16
N_GROUPS = 2_000_000
W = rng.standard_normal((N_GROUPS, G))

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def entropy_bits(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def nearest_idx(mag):
    return np.abs(mag[..., None] - E2M1).argmin(axis=-1)


def report(name, xg, s, clip):
    y = xg / s
    mag = np.clip(np.abs(y), 0.0, 6.0)
    k = nearest_idx(mag)
    rec = np.sign(y) * E2M1[k] * s
    sym = (((np.sign(y) < 0).astype(np.int64) << 3) | k).reshape(-1)
    h = entropy_bits(np.bincount(sym, minlength=16))
    h_mag = entropy_bits(np.bincount(k.reshape(-1), minlength=8))
    mse = float(((rec - xg) ** 2).mean())
    var = float(xg.var())
    cos = 1.0 - 0.5 * mse / var
    print(f"{name:36s} H(nib)={h:6.4f}  H(mag)={h_mag:5.4f}+1sgn"
          f"  MSE/s2={mse:8.6f}  cos~{cos:6.5f}  saved={100*(1-h/4):5.2f}%")


print(f"groups={N_GROUPS:,} iid N(0,1); E2M1 levels {E2M1.tolist()}")

for gsz in (16, 32, 64):
    xg = W[:, :gsz] if gsz == G else W.reshape(-1, gsz)
    amax = np.abs(xg).max(axis=1, keepdims=True)
    report(f"E2M1 g{gsz:2d} scale=amax/6 (NVFP4 rule)", xg, amax / 6.0, True)
    # clipped MSE-optimal scale: search effective top m in [2,6]
    best = None
    for m in np.linspace(2.0, 6.0, 41):
        s = amax / m
        y = xg / s
        mag = np.clip(np.abs(y), 0.0, 6.0)
        k = nearest_idx(mag)
        rec = np.sign(y) * E2M1[k] * s
        mse = ((rec - xg) ** 2).mean(axis=1, keepdims=True)
        if best is None:
            best = (mse, s, y, k)
        else:
            upd = mse < best[0]
            best = (np.where(upd, mse, best[0]), np.where(upd, s, best[1]),
                    np.where(upd, y, best[2]),
                    np.where(np.broadcast_to(upd, k.shape), k, best[3]))
    _, s, y, k = best
    rec = np.sign(y) * E2M1[k] * s
    sym = (((np.sign(y) < 0).astype(np.int64) << 3) | k).reshape(-1)
    h = entropy_bits(np.bincount(sym, minlength=16))
    mse = float(((rec - xg) ** 2).mean())
    cos = 1.0 - 0.5 * mse / float(xg.var())
    print(f"{'E2M1 g%2d scale=MSE-opt (clipped)' % gsz:36s} H(nib)={h:6.4f}"
          f"  MSE/s2={mse:8.6f}  cos~{cos:6.5f}  saved={100*(1-h/4):5.2f}%")

for gsz in (16, 64):
    xg = W[:, :gsz] if gsz == G else W.reshape(-1, gsz)
    s = np.abs(xg).max(axis=1, keepdims=True) / 7.0
    q = np.clip(np.round(xg / s), -7, 7)
    sym = q.reshape(-1).astype(np.int64) + 8
    h = entropy_bits(np.bincount(sym, minlength=16))
    mse = float(((q * s - xg) ** 2).mean())
    print(f"{'int4 g%2d scale=amax/7 (uniform)' % gsz:36s} H(nib)={h:6.4f}"
          f"  MSE/s2={mse:8.6f}  cos~{1-0.5*mse:6.5f}  saved={100*(1-h/4):5.2f}%")

print()
for cos_t in (0.990, 0.995, 0.997, 0.998, 0.999):
    d = 2 * (1 - cos_t)
    print(f"Shannon Gaussian source: cos {cos_t} needs R = "
          f"{-0.5 * math.log2(d):6.3f} bits/weight (scalar ECQ bound)")
