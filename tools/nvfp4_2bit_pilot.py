#!/usr/bin/env python3
"""nvfp4_2bit_pilot.py -- 2-bit vector-quantization pilot for NVFP4 MoE experts.

Reads NVFP4 expert weights straight out of safetensors shards (mmap + manual
header parse; no safetensors lib), dequantizes to a float32 reference, then
scores candidate re-codings per matrix:
  (a) int4 uniform RTN, per row-block fp16 scale                    ~4.06 bpw (control)
  (b) 2-bit uniform on W - U V^T, U,V = int8 top-r SVD              ~2.2-2.5 bpw
  --deep swaps the 2-bit uniform grid for 4-centroid k-means per block-64
      (quantile init); codebook costs 1 extra bit -> base 3.0 bpw.

Verified against the checkpoint 2026-08-27: tensors may live in any of 120
shards (index weight_map); gate/up U8[2048,2048] + F8_E4M3 scale [2048,256] +
F32 scalar scale_2; down U8[4096,1024] + [4096,128]; 288 routed experts per
MoE layer (3..45); nibble = s<<3|e2m1_idx, element 2i in the LOW nibble
(matches src/mxfp4.cu: lut[(p >> (j*4)) & 15], sign bit 3).
Deps: torch only (CPU or CUDA)."""
import argparse, json, mmap, os, struct, sys, time
import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
LUT = torch.tensor(E2M1 + tuple(-v for v in E2M1), dtype=torch.float32)

DEFAULT_DIRS = (
    "/var/lib/insignia/glm53-flash-text",
    "/mnt/e/coding/Insignia/GLM-5.3-Flash-ABLITERATED-NVFP4",
    r"E:\coding\Insignia\GLM-5.3-Flash-ABLITERATED-NVFP4",
)

class ShardStore:
    def __init__(self, d):
        self.dir, self._mmaps, self._hdrs = d, {}, {}
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                self.wmap = json.load(f)["weight_map"]
        else:
            self.wmap = {}
            for fn in sorted(os.listdir(d)):
                if fn.endswith(".safetensors"):
                    for k in self._hdr(fn)[1]:
                        if not k.startswith("__"):
                            self.wmap.setdefault(k, fn)

    def _mm(self, fn):
        if fn not in self._mmaps:
            f = open(os.path.join(self.dir, fn), "rb")
            self._mmaps[fn] = (f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
        return self._mmaps[fn][1]

    def _hdr(self, fn):
        if fn not in self._hdrs:
            m = self._mm(fn)
            n = struct.unpack_from("<Q", m, 0)[0]
            self._hdrs[fn] = (n, json.loads(bytes(m[8:8 + n])))
        return self._hdrs[fn]

    def raw(self, name):
        if name not in self.wmap:
            sys.exit(f"tensor {name!r} not in weight_map ({len(self.wmap)} keys)")
        n, hdr = self._hdr(self.wmap[name])
        info = hdr[name]
        b0, b1 = info["data_offsets"]
        m = self._mm(self.wmap[name])
        return (torch.frombuffer(bytearray(m[8 + n + b0:8 + n + b1]), dtype=torch.uint8),
                info["dtype"], info["shape"])

def e4m3_to_f32(u):
    try:
        return u.view(torch.float8_e4m3fn).float()
    except (AttributeError, RuntimeError):
        i = u.to(torch.int32)
        e, mg = (i >> 3) & 15, i & 7
        v = torch.where(e == 0, mg.float() * 2.0 ** -9,
                        (8 + mg).float() * torch.pow(2.0, (e - 10).float()))
        return torch.where(i >= 128, -v, v)

def dequant_nvfp4(store, base, dev):
    wq, dt, shp = store.raw(base + ".weight")
    s8, sdt, sshp = store.raw(base + ".weight_scale")
    a8, adt, _ = store.raw(base + ".weight_scale_2")
    assert dt == "U8" and sdt == "F8_E4M3" and adt == "F32", (dt, sdt, adt)
    R, Cp = shp
    W = torch.empty((R, 2 * Cp), dtype=torch.float32, device=dev)
    wq, lut = wq.view(shp).to(dev), LUT.to(dev)
    W[:, 0::2] = lut[(wq & 0xF).long()]
    W[:, 1::2] = lut[(wq >> 4).long()]
    W *= e4m3_to_f32(s8.to(dev)).view(sshp).repeat_interleave(16, 1)
    a2 = a8.view(torch.float32).reshape(-1).to(dev)
    W *= a2[:, None] if a2.numel() == R and R > 1 else a2[0]
    return W

def rtn_uniform(W, bits, bs):
    R, C = W.shape
    B = W.view(R, C // bs, bs)
    mx = B.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    if bits == 4:
        s = (mx / 7.0).to(torch.float16).float().clamp_min(6e-8)
        Q = torch.clamp(torch.round(B / s), -7, 7) * s
    else:
        s = (mx / 1.5).to(torch.float16).float().clamp_min(6e-8)
        Q = (torch.round(B / s - 0.5) + 0.5).clamp(-1.5, 1.5) * s
    return Q.view(R, C)

def q_int8(X):
    mx = X.abs().amax().clamp_min(1e-12)
    return torch.clamp(torch.round(X * (127.0 / mx)), -127, 127) * (mx / 127.0)

def kmeans2(W, bs, iters):
    R, C = W.shape
    B = W.reshape(-1, bs)
    nb = B.shape[0]
    q = torch.sort(B, 1).values[:, [bs // 8, 3 * bs // 8, 5 * bs // 8, 7 * bs // 8]].t().contiguous()
    for _ in range(iters):
        a = (B.unsqueeze(0) - q.unsqueeze(2)).abs().argmin(0)
        for k in range(4):
            m = a == k
            q[k] = (B * m).sum(1) / m.sum(1).clamp_min(1)
    a = (B.unsqueeze(0) - q.unsqueeze(2)).abs().argmin(0)
    return q[a, torch.arange(nb, device=W.device)[:, None]].reshape(R, C)

def evaluate(W, What, xs):
    rmse = (What - W).pow(2).mean().sqrt().item()
    cs = []
    for x in xs:
        p, r = W @ x, What @ x
        cs.append((torch.dot(p, r) / (p.norm() * r.norm()).clamp_min(1e-30)).item())
    return rmse, cs

def sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()

def main():
    ap = argparse.ArgumentParser(description="2-bit VQ pilot for NVFP4 MoE experts")
    ap.add_argument("--model", default=None)
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 24])
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--kblock", type=int, default=64)
    ap.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--km-iters", type=int, default=8)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)
    if str(dev).startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    d = args.model or next((p for p in DEFAULT_DIRS if os.path.isdir(p)), None)
    if not d:
        sys.exit(f"no checkpoint found in {DEFAULT_DIRS}; pass --model")
    store = ShardStore(d)
    print(f"# model={d}\n# device={dev} experts={args.experts}/layer layers={args.layers} "
          f"block={args.block} ranks={args.ranks} deep={args.deep} tensors={len(store.wmap)}")

    mats = ("gate_proj", "up_proj", "down_proj")
    agg, t_all, n_mats = {}, time.perf_counter(), 0
    for L in args.layers:
        pref = f"model.language_model.layers.{L}.mlp.experts."
        nE = 1 + max(int(k[len(pref):].split(".")[0]) for k in store.wmap
                     if k.startswith(pref) and k.endswith(".gate_proj.weight"))
        ids = sorted(set(min(nE - 1, i * nE // args.experts) for i in range(args.experts)))
        for e in ids:
            for m in mats:
                base = f"{pref}{e}.{m}"
                t0 = time.perf_counter()
                W = dequant_nvfp4(store, base, dev)
                sync(dev)
                tdq = time.perf_counter() - t0
                R, C = W.shape
                assert C % args.block == 0 and C % args.kblock == 0
                xs = [torch.randn(C, device=dev), 0.1 * torch.randn(C, device=dev)]
                t0 = time.perf_counter()
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                sync(dev)
                tsvd = time.perf_counter() - t0
                print(f"[L{L:>2} E{e:>3} {m:<9}] {R}x{C}  dequant={tdq*1e3:7.1f}ms svd={tsvd*1e3:7.1f}ms")
                results = []
                t0 = time.perf_counter()
                What = rtn_uniform(W, 4, args.block)
                sync(dev)
                results.append(("int4_rtn", What, 4 + 16.0 / args.block, time.perf_counter() - t0))
                base_bpw = 3.0 if args.deep else 2 + 16.0 / args.block
                for r in args.ranks:
                    t0 = time.perf_counter()
                    Lr = q_int8(U[:, :r] * S[:r]) @ q_int8(Vh[:r].T).T
                    res = W - Lr
                    Q = kmeans2(res, args.kblock, args.km_iters) if args.deep \
                        else rtn_uniform(res, 2, args.block)
                    What = Lr + Q
                    sync(dev)
                    bpw = base_bpw + 8.0 * r * (R + C) / (R * C)
                    results.append((f"q2{'km' if args.deep else 'uni'}+lr{r}", What, bpw,
                                    time.perf_counter() - t0))
                    del What, Q, res
                for name, What, bpw, dt in results:
                    rmse, (c1, c2) = evaluate(W, What, xs)
                    print(f"    {name:<14} rmse={rmse:.5f} cos[N(0,1)]={c1:.6f} "
                          f"cos[act x0.1]={c2:.6f} bpw={bpw:.2f} enc={dt*1e3:7.1f}ms")
                    agg.setdefault(name, []).append((rmse, c1, c2, bpw))
                    n_mats += 1
                del W, U, S, Vh, results
    print(f"\n# summary: {n_mats} matrices in {time.perf_counter() - t_all:.1f}s")
    for name, rows in agg.items():
        n, g = len(rows), lambda i: sum(x[i] for x in rows) / len(rows)
        print(f"  {name:<14} mean rmse={g(0):.5f} cos[N(0,1)]={g(1):.6f} "
              f"cos[act]={g(2):.6f} bpw={g(3):.2f}")

if __name__ == "__main__":
    main()
