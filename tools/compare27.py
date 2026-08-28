#!/usr/bin/env python3
"""Compare 27B engine seam dumps against the reference27 trajectory.

engine dump (generate27 dump): [layers=64][T][5120] f32 — post-layer residuals.
reference traj (reference27 seams): [65][T][5120] f32 — traj[0]=embed output,
traj[l]=layer-l output (l>=1); traj[64] = final-norm output.

Usage: python tools/compare27.py <engine_dump.f32> <ref_seams.npy> [T]
Prints per-layer cos/abs and the R6 verdict line (min/median cos, NaN count).
"""
import sys

import numpy as np

eng = np.fromfile(sys.argv[1], np.float32).reshape(64, -1, 5120)
ref = np.load(sys.argv[2])
T = int(sys.argv[3]) if len(sys.argv) > 3 else eng.shape[1]
eng = eng[:, :T]
ref = ref[0:64, :T]          # traj[l] = layer-l output (traj[64] = final norm)

print(f"engine {eng.shape} vs ref {ref.shape}")
cos = np.zeros(64)
abserr = np.zeros(64)
for l in range(64):
    a, b = eng[l].ravel(), ref[l].ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    cos[l] = float(a @ b / n) if n else 1.0
    abserr[l] = float(np.abs(a - b).max())
    flag = "  <-- DIVERGED" if cos[l] < 0.999 else ""
    print(f"layer {l:2d} ({'A' if l % 4 == 3 else 'D'}): cos={cos[l]:.7f} max_abs={abserr[l]:.3e}{flag}")
print(f"VERDICT: min cos={cos.min():.7f} median={np.median(cos):.7f} nan={int(np.isnan(eng).sum())} "
      f"{'PASS' if cos.min() > 0.9999 and not np.isnan(eng).any() else 'FAIL'}")
worst = np.argsort(cos)[:5]
print("worst layers:", [(int(l), round(float(cos[l]), 5)) for l in worst])
