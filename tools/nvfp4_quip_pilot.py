#!/usr/bin/env python3
"""nvfp4_quip_pilot.py -- QuIP#-style 2-bit E8 lattice quantization for NVFP4 MoE experts.

Reuses the verified mmap ShardStore + NVFP4 dequant from tools/nvfp4_2bit_pilot.py.
For each expert matrix W (fp32, NVFP4-dequant reference):
  1. Incoherence transform  W' = Hr (Dr W Dc) Hc,  H = explicit Sylvester
     Hadamard / sqrt(n) (no scipy); Dr/Dc are seeded random sign diagonals.
  2. Lattice coding per [row, 8-col] block: per-block scale, nearest of 256 E8
     codewords via argmin distances, sequential error feedback (EF=0.9).
  3. Back-transform  What = Dr Hr^T Qt Hc^T Dc, then score against W.

bpw accounting: one 8-bit codebook index per 8 weights (1.0 bpw code) plus a
per-block scale: e8m0 (8-bit pow2) -> 2.00 bpw; log12 (4b exp + 8b mantissa)
-> 2.50 bpw; fp16 -> 3.00 bpw diagnostic. NVFP4 source is 4.50 bpw.

Honest deviations from the QuIP# paper:
 1. Codebook is NOT bit-exact E8P12: 112 (+-1,+-1,0^6) + 128 even-parity
    (+-1)^8 (240 E8 roots) + 16 half points (+-1/2)^8 signed by the extended
    Hamming (8,4) code. Expect close-but-not-identical distortion.
 2. EF residual fp32 (paper: EF16); no interleaved block permutation.
 3. No Hessian weighting, no LDFT fine-tuning, no outlier buffer: pure
    weight-space pilot. The paper's headline 2-bit quality leans on LDFT;
    expect cos ~0.99 at 2.0 bpw, not 0.995+.
Usage: /var/lib/insignia/oracle-venv/bin/python tools/nvfp4_quip_pilot.py
"""
import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvfp4_2bit_pilot import DEFAULT_DIRS, ShardStore, dequant_nvfp4, sync

MODE_BITS = {"e8m0": 8, "log12": 12, "fp16": 16}
_H = {}

def hada(n, dev):
    if (n, dev) not in _H:
        h = torch.ones((1, 1), device=dev)
        while h.shape[0] < n:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        _H[(n, dev)] = h * float(n) ** -0.5
    return _H[(n, dev)]

def e8_codebook(dev):
    pts = torch.zeros((256, 8))
    k = 0
    for i in range(8):
        for j in range(i + 1, 8):
            for a in (1.0, -1.0):
                for b in (1.0, -1.0):
                    pts[k, i], pts[k, j] = a, b
                    k += 1
    s = 1.0 - 2.0 * ((torch.arange(256)[:, None] >> torch.arange(8)) & 1).float()
    pts[k:k + 128] = s[(s < 0).sum(1) % 2 == 0]
    k += 128
    g = torch.tensor([[1, 0, 0, 0, 0, 1, 1, 1], [0, 1, 0, 0, 1, 0, 1, 1],
                      [0, 0, 1, 0, 1, 1, 0, 1], [0, 0, 0, 1, 1, 1, 1, 0]], dtype=torch.int64)
    msg = (torch.arange(16)[:, None] >> torch.arange(4)) & 1
    pts[k:] = 0.5 - ((msg @ g) % 2).float()
    assert k == 240 and torch.unique(pts, dim=0).shape[0] == 256
    return pts.to(dev)

def quant_scale(s, mode):
    if mode == "fp16":
        return s.to(torch.float16).float().clamp_min(6e-8)
    s = s.clamp_min(2.0 ** -120)
    lo = torch.exp2(torch.floor(torch.log2(s)))
    if mode == "e8m0":
        return torch.where(s - lo < 2.0 * lo - s, lo, 2.0 * lo)
    m = (((s / lo - 1.0) * 255.0).round() / 255.0).clamp(0.0, 1.0)
    return (lo * (1.0 + m)).clamp_min(6e-8)

def quip_encode(Wp, cb, mode, ef):
    R, C = Wp.shape
    out = torch.empty_like(Wp)
    r = torch.zeros(R, 8, device=Wp.device)
    cb2 = cb.pow(2).sum(1)
    for t in range(C // 8):
        v = Wp[:, 8 * t:8 * t + 8] + ef * r
        s = quant_scale(v.abs().amax(1, keepdim=True), mode)
        x = v / s
        d2 = (x * x).sum(1, keepdim=True) - 2.0 * (x @ cb.T) + cb2
        q = cb[d2.argmin(1)] * s
        out[:, 8 * t:8 * t + 8] = q
        r = v - q
    return out

def cosv(a, b):
    return (torch.dot(a.flatten(), b.flatten()) / (a.norm() * b.norm()).clamp_min(1e-30)).item()

def kurt(x):
    x = x - x.mean()
    return ((x ** 4).mean() / x.pow(2).mean().pow(2)).item()

def main():
    ap = argparse.ArgumentParser(description="QuIP#-style 2-bit E8 lattice pilot for NVFP4 MoE")
    ap.add_argument("--model", default=None)
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 24])
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--modes", nargs="+", default=["e8m0", "log12"], choices=list(MODE_BITS))
    ap.add_argument("--ef", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    d = args.model or next((p for p in DEFAULT_DIRS if os.path.isdir(p)), None)
    if not d:
        sys.exit(f"no checkpoint in {DEFAULT_DIRS}; pass --model")
    store = ShardStore(d)
    cb = e8_codebook(dev)
    print(f"# model={d}\n# device={dev} layers={args.layers} experts={args.experts}/layer "
          f"modes={args.modes} ef={args.ef}")
    print("# source NVFP4 = 4.50 bpw; pilot: "
          + ", ".join(f"{m}={1 + MODE_BITS[m] / 8:.2f}" for m in args.modes) + " bpw")

    mats = ("gate_proj", "up_proj", "down_proj")
    agg, t_all, n = {}, time.perf_counter(), 0
    for L in args.layers:
        pref = f"model.language_model.layers.{L}.mlp.experts."
        nE = 1 + max(int(k[len(pref):].split(".")[0]) for k in store.wmap
                     if k.startswith(pref) and k.endswith(".gate_proj.weight"))
        ids = sorted(set(min(nE - 1, i * nE // args.experts) for i in range(args.experts)))
        for e in ids:
            for m in mats:
                W = dequant_nvfp4(store, f"{pref}{e}.{m}", dev)
                R, C = W.shape
                assert not (R & (R - 1)) and not (C & (C - 1)), f"pow2 dims needed, got {R}x{C}"
                torch.manual_seed(args.seed)
                dr = torch.where(torch.randn(R, device=dev) < 0, -1.0, 1.0)
                dc = torch.where(torch.randn(C, device=dev) < 0, -1.0, 1.0)
                x1 = torch.randn(C, device=dev)
                x2 = torch.randn(C, device=dev) * torch.exp(0.5 * torch.randn(C, device=dev))
                Hr, Hc = hada(R, dev), hada(C, dev)
                Wt = Hr @ (dr[:, None] * W * dc[None, :]) @ Hc
                sync(dev)
                print(f"[L{L:>2} E{e:>3} {m:<9}] {R}x{C}  kurt W={kurt(W):6.1f} -> W'={kurt(Wt):4.2f}")
                for mode in args.modes:
                    t0 = time.perf_counter()
                    Qt = quip_encode(Wt, cb, mode, args.ef)
                    sync(dev)
                    t_enc = time.perf_counter() - t0
                    What = dr[:, None] * (Hr.T @ Qt @ Hc.T) * dc[None, :]
                    sync(dev)
                    rmse = (What - W).pow(2).mean().sqrt().item()
                    rT = (Qt - Wt).pow(2).mean().sqrt().item()
                    c1, c2 = cosv(W @ x1, What @ x1), cosv(W @ x2, What @ x2)
                    cW, cT = cosv(W, What), cosv(Wt, Qt)
                    print(f"    {mode:<6} rmse={rmse:.5f} (W'-space {rT:.5f}) cos[x1]={c1:.6f} "
                          f"cos[x2]={c2:.6f} cos[W]={cW:.6f} bpw={1 + MODE_BITS[mode] / 8:.2f} "
                          f"enc={1e3 * t_enc:7.1f}ms")
                    agg.setdefault(mode, []).append((rmse, c1, c2, cW, cT, t_enc))
                    del Qt, What
                    n += 1
                del W, Wt
    print(f"\n# summary: {n} codings in {time.perf_counter() - t_all:.1f}s, EF={args.ef}")
    for mode, rows in agg.items():
        g = lambda i: sum(r[i] for r in rows) / len(rows)
        print(f"  {mode:<6} mean rmse={g(0):.5f} cos[x1]={g(1):.6f} cos[x2]={g(2):.6f} "
              f"cos[W]={g(3):.6f} cos[W']={g(4):.6f} bpw={1 + MODE_BITS[mode] / 8:.2f} "
              f"enc={1e3 * g(5):.0f}ms")

if __name__ == "__main__":
    main()
