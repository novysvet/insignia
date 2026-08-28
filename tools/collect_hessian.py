#!/usr/bin/env python3
"""Collect per-projection input second moments h = E[x_j^2] for INSIG4 weighted scale fitting.

Method (validated in audits/w4/insig4-evolution.md): re-run each layer's block math in
numpy, seeded per token by ENGINE seam dumps (build/dump-multistep.dll output), so the
activations are real engine activations without engine drift. Layer weights are
dequantized from the given INSIG4 file ONCE per layer; all T tokens flow through
batched GEMMs with per-token stateful cores (conv roll, DeltaNet state, causal KV).

Usage: python tools/collect_hessian.py <insig4.safetensors> <seams.f32> <tokens.csv> <out.npz>
  seams.f32: [T,33,4096] f32 — rows 0..31 = layer outputs, row 32 = final norm output.
  tokens.csv: the teacher-forced token ids the seams were dumped with (T of them).

Output npz: one key per QUANTIZED tensor base name (engine naming, e.g.
  language_model.model.layers.0.linear_attn.in_proj_qkv) -> h[cols] f32 (mean over T).
Embed rows and mtp.fc are one-hot / concat-fed: no h (quantizer falls back to unweighted).
"""
import json
import pathlib
import struct
import sys

import numpy as np

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)

def main():
    insig4 = pathlib.Path(sys.argv[1])
    seams = np.fromfile(sys.argv[2], np.float32).reshape(-1, 33, 4096)
    tokens = [int(x) for x in open(sys.argv[3]).read().strip().split(',')]
    T = min(len(tokens), seams.shape[0])
    seams = seams[:T]
    out_path = pathlib.Path(sys.argv[4])

    f = insig4.open('rb')
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))
    start = 8 + n
    def get(k):
        v = hdr[k]
        f.seek(start + v['data_offsets'][0])
        raw = f.read(v['data_offsets'][1] - v['data_offsets'][0])
        a = np.frombuffer(raw, dtype={'U32': '<u4', 'U8': 'u1', 'BF16': '<u2', 'F32': '<f4', 'F16': '<f2'}[v['dtype']])
        if v['dtype'] == 'BF16':
            return (a.astype(np.uint32) << 16).view(np.float32).reshape(v['shape'])
        if v['dtype'] == 'F16':
            return a.astype(np.float32).reshape(v['shape'])
        return a.reshape(v['shape'])
    def dq(base):
        w = get(base + '.weight'); s = get(base + '.scales')
        rows = w.shape[0]; cols = w.shape[1] * 8
        u = w.reshape(rows, -1)
        q = ((u[:, :, None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15).reshape(rows, cols)
        lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
        return lut[q] * np.repeat(s, 64, axis=1)   # INSIG4: one f16 scale per 64-elt super-group
    def rms_rows(x, w):                      # x [T,C] -> x/rms * w  (engine Z=false on this file)
        return x / np.sqrt(np.mean(x * x, 1, keepdims=True) + 1e-6) * w
    def rms_1(x, w):                         # x [C]
        return x / np.sqrt(np.mean(x * x) + 1e-6) * w

    H = {}                                   # engine base name -> running sum of x^2
    def acc(name, x):
        x = np.asarray(x, np.float32)
        H[name] = H[name] + x * x if name in H else x * x

    # layer-0 inputs: engine embed rows (dequantized) for each token
    E = dq('language_model.model.embed_tokens')
    Xin = E[tokens].astype(np.float32)       # [T,4096]

    rope_cos = None; rope_sin = None
    inv = 1e7 ** (-np.arange(32, dtype=np.float64) / 32)
    for l in range(32):
        p = f'language_model.model.layers.{l}'
        prev = Xin if l == 0 else seams[:, l - 1]
        ln_w = get(p + '.input_layernorm.weight')
        nrm = rms_rows(prev, ln_w)
        if (l & 3) != 3:                     # ---- gated DeltaNet linear-attention layer
            a = p + '.linear_attn'
            Wq = dq(a + '.in_proj_qkv'); Wz = dq(a + '.in_proj_z')
            Wa = dq(a + '.in_proj_a');   Wb = dq(a + '.in_proj_b')
            Wo = dq(a + '.out_proj')
            acc(a + '.in_proj_qkv', nrm); acc(a + '.in_proj_z', nrm)
            acc(a + '.in_proj_a', nrm);   acc(a + '.in_proj_b', nrm)
            qkv = nrm @ Wq.T                # [T,8192]
            z = nrm @ Wz.T
            aa = nrm @ Wa.T                 # [T,32]
            bb = nrm @ Wb.T
            cw = get(a + '.conv1d.weight').reshape(8192, 4)
            A_log = get(a + '.A_log'); dt = get(a + '.dt_bias')  # F32 / BF16->f32 by get()
            nw = get(a + '.norm.weight')
            state = np.zeros((32, 128, 128), np.float32)
            st = np.zeros((8192, 3), np.float32)
            gated = np.empty((T, 4096), np.float32)
            for t in range(T):
                x = qkv[t]
                zc = cw[:, 3] * x + cw[:, 2] * st[:, 0] + cw[:, 1] * st[:, 1] + cw[:, 0] * st[:, 2]
                st[:, 0] = st[:, 1]; st[:, 1] = st[:, 2]; st[:, 2] = x
                zc = zc / (1 + np.exp(-zc))
                q = zc[:2048].reshape(16, 128); k = zc[2048:4096].reshape(16, 128); v = zc[4096:].reshape(32, 128)
                q = q / np.sqrt((q * q).sum(1, keepdims=True) + 1e-6) / np.sqrt(128)
                k = k / np.sqrt((k * k).sum(1, keepdims=True) + 1e-6)
                q = np.repeat(q, 2, 0); k = np.repeat(k, 2, 0)
                dec = np.exp(-np.exp(A_log) * np.logaddexp(0, aa[t] + dt))
                beta = 1 / (1 + np.exp(-bb[t]))
                state *= dec[:, None, None]
                mem = np.einsum('hvk,hk->hv', state, k)
                state += ((v - mem) * beta[:, None])[:, :, None] * k[:, None, :]
                o = np.einsum('hvk,hk->hv', state, q)
                o = o / np.sqrt((o * o).mean(1, keepdims=True) + 1e-6) * nw[None, :]
                o = o * (z[t].reshape(32, 128) / (1 + np.exp(-z[t].reshape(32, 128))))
                gated[t] = o.reshape(-1)
            del Wq, Wz, Wa, Wb
            acc(a + '.out_proj', gated)
            block = gated @ Wo.T
            del Wo
        else:                                # ---- full-attention layer (real causal KV) ----
            a = p + '.self_attn'
            Wq = dq(a + '.q_proj'); Wk = dq(a + '.k_proj'); Wv = dq(a + '.v_proj'); Wo = dq(a + '.o_proj')
            qn = get(a + '.q_norm.weight'); kn = get(a + '.k_norm.weight')
            acc(a + '.q_proj', nrm); acc(a + '.k_proj', nrm); acc(a + '.v_proj', nrm)
            raw = (nrm @ Wq.T).reshape(T, 16, 512)
            qh = raw[:, :, :256]; gate = raw[:, :, 256:]
            kk = (nrm @ Wk.T).reshape(T, 4, 256)
            vv = (nrm @ Wv.T).reshape(T, 4, 256)
            qh = qh / np.sqrt((qh * qh).sum(2, keepdims=True) + 1e-6) * qn
            kk = kk / np.sqrt((kk * kk).sum(2, keepdims=True) + 1e-6) * kn
            def rope(x, pos):
                y = x.copy()
                for i in range(32):
                    c, s = float(np.cos(inv[i] * pos)), float(np.sin(inv[i] * pos))
                    a0, a1 = x[..., i], x[..., i + 32]
                    y[..., i] = c * a0 - s * a1; y[..., i + 32] = s * a0 + c * a1
                return y
            qs = np.stack([rope(qh[t], t) for t in range(T)])
            ks = np.stack([rope(kk[t], t) for t in range(T)])
            o_in = np.empty((T, 4096), np.float32)
            for t in range(T):
                kh = np.repeat(ks[:t + 1], 4, axis=1)              # [t+1,16,256], head h <- kv h//4
                sc = np.einsum('hd,thd->ht', qs[t], kh) / 16
                sc = np.exp(sc - sc.max(1, keepdims=True)); sc /= sc.sum(1, keepdims=True)
                vh = np.repeat(vv[:t + 1], 4, axis=1)              # [t+1,16,256]
                ctx = np.einsum('ht,thd->hd', sc, vh)              # [16,256]
                o_in[t] = (ctx / (1 + np.exp(-gate[t]))).reshape(-1)
            acc(a + '.o_proj', o_in)
            del Wq, Wk, Wv
            block = o_in @ Wo.T
            del Wo
        x1 = prev + block                    # input to the MLP block
        post_w = get(p + '.post_attention_layernorm.weight')
        nrm2 = rms_rows(x1, post_w)
        Wg = dq(p + '.mlp.gate_proj'); Wu = dq(p + '.mlp.up_proj')
        acc(p + '.mlp.gate_proj', nrm2); acc(p + '.mlp.up_proj', nrm2)
        g = nrm2 @ Wg.T; u = nrm2 @ Wu.T
        del Wg, Wu
        act = g / (1 + np.exp(-g)) * u
        acc(p + '.mlp.down_proj', act)
        print(f'layer {l} done', flush=True)

    hfinal = rms_rows(seams[:, 31], get('language_model.model.norm.weight'))
    acc('language_model.lm_head', hfinal)

    np.savez(out_path, **{k: (v.sum(0) / T).astype(np.float32) for k, v in H.items()})
    print(f'wrote {out_path}: {len(H)} h vectors, T={T}')

if __name__ == '__main__':
    main()
