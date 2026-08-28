#!/usr/bin/env python3
"""Stage-by-stage layer-0 bisect: engine dumps (build/stage-*.f32) vs numpy reference."""
import json
import struct

import numpy as np

MODEL = 'Qwen3.8-27B-FP8/layers-0.safetensors'
OUTSIDE = 'Qwen3.8-27B-FP8/outside.safetensors'
EPS = 1e-6
TOK = 760

mm = np.memmap(MODEL, dtype=np.uint8, mode='r')
n = struct.unpack_from('<Q', mm, 0)[0]
hdr = json.loads(bytes(mm[8:8 + n]))
start = 8 + n
def get(name, dt=None):
    v = hdr[name]
    off = start + v['data_offsets'][0]
    nb = v['data_offsets'][1] - v['data_offsets'][0]
    if dt is None: dt = np.uint8 if v['dtype'] == 'F8_E4M3' else np.uint16
    return np.frombuffer(mm[off:off + nb].tobytes(), dtype=dt), v['shape']
def e4m3_dec(u8):
    u8 = u8.astype(np.uint32)
    sign = np.where(u8 & 0x80, -1.0, 1.0)
    exp = (u8 >> 3) & 0xf
    man = u8 & 7
    out = np.where(exp == 0, (man / 8.0) * 2.0 ** -6, 2.0 ** (exp.astype(np.float64) - 7) * (1 + man / 8.0))
    return (out * sign).astype(np.float32)
def deq(ckpt):
    w8, shape = get(ckpt + '.weight')
    s16, sshape = get(ckpt + '.weight_scale_inv')
    W = e4m3_dec(w8).reshape(shape)
    S = (s16.astype(np.uint32) << 16).view(np.float32).reshape(sshape)
    Wd = np.zeros_like(W)
    for i in range(0, shape[0], 128):
        for j in range(0, shape[1], 128):
            Wd[i:i + 128, j:j + 128] = W[i:i + 128, j:j + 128] * S[i // 128, j // 128]
    return Wd
def bf16(name, shard=mm):
    v = hdr[name]
    off = start + v['data_offsets'][0]
    nb = v['data_offsets'][1] - v['data_offsets'][0]
    a = np.frombuffer(shard[off:off + nb].tobytes(), dtype=np.uint16)
    return (a.astype(np.uint32) << 16).view(np.float32).reshape(v['shape'])
def rms(x, w):
    return x / np.sqrt(np.mean(x * x) + EPS) * w

mmo = np.memmap(OUTSIDE, dtype=np.uint8, mode='r')
no = struct.unpack_from('<Q', mmo, 0)[0]
ho = json.loads(bytes(mmo[8:8 + no]))
so = 8 + no
vo = ho['model.language_model.embed_tokens.weight']
emb = (np.frombuffer(mmo[so + vo['data_offsets'][0] + TOK * 5120 * 2: so + vo['data_offsets'][0] + TOK * 5120 * 2 + 5120 * 2].tobytes(), dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)

P = 'model.language_model.layers.0.'
A = P + 'linear_attn.'
ln = 1.0 + bf16(P + 'input_layernorm.weight')
Wqkv = deq(A + 'in_proj_qkv')
Wz = deq(A + 'in_proj_z')
Wa = bf16(A + 'in_proj_a.weight')
Wb = bf16(A + 'in_proj_b.weight')
cw = bf16(A + 'conv1d.weight').reshape(10240, 4)
A_log = bf16(A + 'A_log')
dtb = bf16(A + 'dt_bias')
la_norm = bf16(A + 'norm.weight')
Wo = deq(A + 'out_proj')

def cmp(stage, ref):
    eng = np.fromfile(f'build/stage-{stage}.f32', np.float32)
    r = np.asarray(ref, np.float32).ravel()
    if eng.shape != r.shape:
        print(f"{stage:8s} SHAPE MISMATCH eng {eng.shape} vs ref {r.shape}")
        return
    cos = float(eng @ r / np.linalg.norm(eng) / np.linalg.norm(r)) if np.linalg.norm(eng) and np.linalg.norm(r) else 1.0
    print(f"{stage:8s} cos={cos:.7f} max_abs={np.abs(eng - r).max():.3e} |ref|max={np.abs(r).max():.3f}")

# engine stages
cmp('pf_x0', emb)
nrm = rms(emb, ln)
cmp('norm0', nrm)
qkv = Wqkv @ nrm
z = Wz @ nrm
cmp('qkv0', qkv)
cmp('z0', z)
a = Wa @ nrm
b = Wb @ nrm
cmp('ab0', a)
cmp('ab1', b)
# conv silu (state zero)
conv = qkv * cw[:, 3]  # state 0
conv = conv / (1 + np.exp(-conv))
cmp('conv0', conv)
# params
soft = np.logaddexp(0, a + dtb)
pa = -np.exp(A_log) * soft
pb = 1 / (1 + np.exp(-b))
cmp('pa0', pa)
cmp('pb0', pb)
# deltanet core
q = conv[:2048].reshape(16, 128)
k = conv[2048:4096].reshape(16, 128)
v = conv[4096:].reshape(48, 128)
q = q / np.sqrt((q * q).sum(1, keepdims=True) + 1e-6) / np.sqrt(128)
k = k / np.sqrt((k * k).sum(1, keepdims=True) + 1e-6)
core = np.zeros(6144, np.float32)
for h in range(48):
    kh = h // 3
    S = np.zeros((128, 128), np.float32)
    dec = np.exp(pa[h])
    beta = pb[h]
    mem = (S * dec) @ k[kh]
    d = (v[h] - mem) * beta
    S = S * dec + np.outer(d, k[kh])
    core[h * 128:(h + 1) * 128] = S @ q[kh]
cmp('core0', core)
# gated rmsnorm
g = core.reshape(48, 128)
g = g / np.sqrt((g * g).mean(1, keepdims=True) + 1e-6) * la_norm[None, :]
zr = z.reshape(48, 128)
g = g * (zr / (1 + np.exp(-zr)))
cmp('gated0', g.reshape(-1))
block = Wo @ g.reshape(-1)
cmp('block0', block)
cmp('res0', emb + block)
