#!/usr/bin/env python3
"""INSIG4 quantizer: Qwen3.5-9B BF16 -> Insignia INSIG4 checkpoint.

Format: per 32-element group, E2M1 nibble codes (8 per u32, MLX packing);
per 64-element super-group, one MSE-optimal fp16 scale shared by both groups.
Identical byte size to MLX MXFP4 (4.125 bpw); +5.9dB SQNR over E8M0 MXFP4.

Output: single safetensors with the engine's MLX-style names
(language_model.* prefix, .weight U32 packed + .scales F16 [groups/2]).
Non-quantized tensors (norms, biases, conv, MTP small ones) stored BF16 as-is.
"""
import json
import pathlib
import sys
import numpy as np

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)

def quant_codes(w64, sc):
    """Snap to the true E2M1 grid (sign x 8 magnitudes); returns dequantized values."""
    q = w64 / sc[:, None]
    mag = np.abs(q)
    idx = np.abs(mag[..., None] - E2M1[None, None, :]).argmin(-1)
    return np.sign(q) * E2M1[idx] * sc[:, None]

def optimal_scale(w64, iters=10):
    """MSE-optimal fp16 scale per 64-group against the true E2M1 grid (golden section)."""
    amax = np.abs(w64).max(1) / 6.0
    amax = np.maximum(amax, 1e-30)
    lo, hi = amax * 0.6, amax * 1.6
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        e1 = np.mean((w64 - quant_codes(w64, m1)) ** 2, 1)
        e2 = np.mean((w64 - quant_codes(w64, m2)) ** 2, 1)
        lo = np.where(e1 < e2, lo, m1)
        hi = np.where(e1 < e2, m2, hi)
    return ((lo + hi) / 2).astype(np.float16)

def quantize_tensor(w):
    """w: 2D float32 [rows, cols] -> (packed u32 [rows, cols/8], scales f16 [rows*cols/64])."""
    rows, cols = w.shape
    assert cols % 64 == 0
    g64 = w.reshape(rows * cols // 64, 64)
    # chunked scale search to bound temporaries
    CH = 200000
    sparts = []
    for i in range(0, g64.shape[0], CH):
        sparts.append(optimal_scale(g64[i:i+CH]).astype(np.float32))
    s = np.concatenate(sparts)
    codes_parts = []
    for i in range(0, g64.shape[0], CH):
        q = g64[i:i+CH] / s[i:i+CH, None]
        mag = np.abs(q)
        idx = np.abs(mag[..., None] - E2M1[None, None, :]).argmin(-1)
        sign = (q < 0).astype(np.int8)
        codes_parts.append((idx | (sign << 3)).astype(np.uint8))
    codes = np.concatenate(codes_parts)
    nib = codes.reshape(-1, 4, 8)  # per 32-group: 4 words x 8 nibbles
    packed = np.zeros((nib.shape[0], 4), np.uint32)
    for j in range(8):
        packed[:, :] |= nib[:, :, j].astype(np.uint32) << (4 * j)
    packed = packed.reshape(rows, cols // 8)
    scales = s.reshape(rows, cols // 64)
    return packed, scales

def main():
    snap = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2])
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
                with f.open('rb') as fh:
                    fh.seek(start + v['data_offsets'][0])
                    raw = fh.read(v['data_offsets'][1] - v['data_offsets'][0])
                a = np.frombuffer(raw, dtype=np.float32 if v['dtype'] == 'F32' else np.uint16).reshape(v['shape'])
                return a.astype(np.float32) if v['dtype'] == 'F32' else (a.astype(np.uint32) << 16).view(np.float32)
        raise KeyError(name)

    out_hdr = {}
    blobs = []
    off = 0
    def emit(name, arr, dt):
        nonlocal off
        arr = np.ascontiguousarray(arr)
        out_hdr[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + arr.nbytes]}
        blobs.append(arr.tobytes())
        off += arr.nbytes

    names = sorted(idx['weight_map'].keys())
    def engine_name(n):
        # HF -> engine (language_model.model.*, top-level lm_head/mtp live under language_model.*)
        if n.startswith('model.language_model.'):
            return 'language_model.model.' + n[len('model.language_model.'):]
        if n.startswith('model.lm_head'):
            return 'language_model.lm_head' + n[len('model.lm_head'):]
        if n.startswith('mtp.'):
            return 'language_model.mtp.' + n[4:]
        return n

    for name in names:
        if name.startswith('visual') or '.visual.' in name:
            continue  # text-only engine
        w = load(name)
        eng = engine_name(name)
        base = eng[:-len('.weight')] if eng.endswith('.weight') else eng
        if w.ndim == 2 and w.shape[1] % 64 == 0 and ('proj' in name or 'fc.weight' in name or name.endswith('lm_head.weight') or 'embed_tokens' in name):
            packed, scales = quantize_tensor(w)
            emit(base + '.weight', packed, 'U32')
            emit(base + '.scales', scales.astype(np.float16), 'F16')
            print(f"{eng}: {w.shape} -> INSIG4", flush=True)
        else:
            emit(eng, w.astype(np.float16), 'BF16')
    header = json.dumps(out_hdr, separators=(',', ':')).encode()
    with out_path.open('wb') as f:
        f.write(len(header).to_bytes(8, 'little'))
        f.write(header)
        for b in blobs:
            f.write(b)
    print(f"wrote {out_path} ({off/2**30:.2f} GiB)")

if __name__ == '__main__':
    main()
