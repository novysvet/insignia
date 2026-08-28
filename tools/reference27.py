#!/usr/bin/env python3
"""NumPy ground-truth reference for Qwen3.8-27B-FP8 layers (W3 ladder R4-R9).

Model: 64 layers (48 gated-DeltaNet linear-attention + 16 full-attention at
(l & 3) == 3), hidden 5120, inter 17408, vocab 248320. F8_E4M3 weights with
BF16 weight_scale_inv [rows/128, cols/128]; W = F8 x scale_inv (multiply).
There is no torch locally, so THIS SCRIPT IS THE GROUND TRUTH (parity-ladder
SS0). All shard access is np.memmap views straight from the safetensors data
region (u64 header-len + json header parsed by hand, per the i4 reference
pattern). The model directory is only ever READ.

Subcommands:
  selftest                     e4m3 decoder enumeration vs a hand-built table
  layer <N> <ids> [--attn] [--out F.npy] [--no-save]
                               run layers 0..N over the ids (embed input),
                               print per-layer output norms, save trajectory
                               npy [N+1, T, 5120] (seam-major, f32)
  seams <ids> <out.npy>        embed -> layers 0..63 -> final norm; writes
                               65 seams [65, T, 5120] f32 for engine compare
  nll <ids>                    full-model NLL (f64 logsumexp over the bf16
                               lm_head, read in 8192-row chunks)
  greedy <ids> <n>             n greedy tokens (full-vocab chunked argmax)
  enc <text> / dec <ids>       tokenizer glue (model dir tokenizer.json)

Usage: python tools/reference27.py [--model DIR] <subcommand> ...
"""

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

MODEL_DIR_DEFAULT = r'E:\coding\Insignia\Qwen3.8-27B-FP8'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

H = 5120          # hidden
INTER = 17408     # mlp intermediate
LAYERS = 64       # (l & 3) == 3 -> full attention
VOCAB = 248320
QH, KVH, HD = 24, 4, 256          # full attention: q heads, kv heads, head dim
QKV_ROWS = 12288                  # q_proj rows = QH * (HD q + HD gate)
GQA = QH // KVH                   # 6 -> kv head of q head h is h // 6
LK, LV, LD = 16, 48, 128          # linear attn: k heads, v heads, dims
KSHARE = LV // LK                 # 3 -> v head j uses k head j // 3
CONV_C = 10240                    # conv channels = 2048 q | 2048 k | 6144 v
EPS = 1e-6
ATT_SCALE = np.float32(0.0625)    # 1/sqrt(256)
QSCALE = np.float32(0.08838834764831845)  # 1/sqrt(128), folded into q norm
ROPE_N = 32                       # partial rope: 64 dims = 32 pairs (i, i+32)
ROPE_THETA = 1e7

DT = {'F8_E4M3': np.dtype('u1'), 'BF16': np.dtype('<u2'), 'F32': np.dtype('<f4'),
      'F16': np.dtype('<f2'), 'U8': np.dtype('u1'), 'U32': np.dtype('<u4'), 'I32': np.dtype('<i4')}

# ---------------------------------------------------------------- e4m3 decoder
# safetensors F8_E4M3 = OCP 'fn' variant: bias 7, subnormals m*2^-9, NaN only
# 0x7F/0xFF, no infinity, max finite +-448 (parity-ladder SS2).


def e4m3_fn(code):
    mag = int(code) & 0x7F
    e, m = mag >> 3, mag & 7
    if mag == 0x7F:
        return float('nan')
    if e == 0:
        v = m * (2.0 ** -9)
    else:
        v = (2.0 ** (e - 7)) * (1.0 + m / 8.0)
    return -v if code & 0x80 else v


LUT8 = np.array([e4m3_fn(c) for c in range(256)], np.float32)


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def silu(v):
    return v / (1.0 + np.exp(-v))


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def rms(x, w):
    # zero-centered HF RMSNorm: weight is applied as (1 + w) by callers that
    # pre-shift; `w` here is whatever scale vector is passed in.
    return x / np.sqrt(np.mean(x * x) + EPS) * w


# ---------------------------------------------------------------- shard reader
class Shard:
    """Read-only np.memmap view of one safetensors file (header parsed by hand)."""

    def __init__(self, path):
        self.path = str(path)
        self.mm = np.memmap(self.path, dtype=np.uint8, mode='r')
        n = struct.unpack_from('<Q', self.mm, 0)[0]
        self.hdr = json.loads(bytes(self.mm[8:8 + n]))
        self.hdr.pop('__metadata__', None)
        self.start = 8 + n

    def info(self, name):
        return self.hdr[name]

    def raw(self, name):
        """Untyped view [shape] of the tensor bytes (u1 / <u2 / <f4 ...)."""
        v = self.hdr[name]
        o = self.start + v['data_offsets'][0]
        nbytes = v['data_offsets'][1] - v['data_offsets'][0]
        return self.mm[o:o + nbytes].view(DT[v['dtype']]).reshape(v['shape'])

    def bf16(self, name):
        return bf16_to_f32(self.raw(name))

    def close(self):
        try:
            self.mm._mmap.close()
        except (BufferError, AttributeError, ValueError):
            pass
        self.mm = None


def dq_f8(shard, base):
    """Dequant W [rows=out, cols=in] f32 = LUT8[F8] * bf16 scale_inv broadcast
    over 128x128 blocks. Built 128 rows at a time so transient temps stay tiny.
    All 27B weight dims are multiples of 128 -> np.repeat is exact."""
    w = shard.raw(base + '.weight')                 # u1 [R, C] mmap view
    s = shard.bf16(base + '.weight_scale_inv')      # f32 [R//128, C//128]
    R, C = w.shape
    out = np.empty((R, C), np.float32)
    for i in range(0, R, 128):
        out[i:i + 128] = LUT8[w[i:i + 128]] * np.repeat(s[i // 128], 128)
    return out


def lin(shard, base):
    """Load [out, in] matrix f32: F8 dequant if quantized, else bf16."""
    if shard.info(base + '.weight')['dtype'] == 'F8_E4M3':
        return dq_f8(shard, base)
    return shard.bf16(base + '.weight')


class Model27B:
    """outside.safetensors: bf16 embed/lm_head/final-norm. Never materialized."""

    def __init__(self, model_dir):
        self.dir = model_dir
        self.outside = Shard(os.path.join(model_dir, 'outside.safetensors'))
        self.emb = self.outside.raw('model.language_model.embed_tokens.weight')  # <u2 view
        self.lm = self.outside.raw('lm_head.weight')                             # <u2 view
        self.fnorm = 1.0 + self.outside.bf16('model.language_model.norm.weight')

    def embed(self, tok):
        return bf16_to_f32(np.asarray(self.emb[tok]))

    def lm_rows(self, r0, r1):
        return bf16_to_f32(np.asarray(self.lm[r0:r1]))


# ---------------------------------------------------------------- layer weights
class LayerSet:
    """One layer's dequantized weights, ~1.5 GB f32 peak (one shard at a time)."""

    def __init__(self, model_dir, l, attn):
        self.l, self.attn = l, attn
        sh = Shard(os.path.join(model_dir, f'layers-{l}.safetensors'))
        p = f'model.language_model.layers.{l}'
        # HF Qwen3_5RMSNorm weights are ZERO-centered: apply (1 + w).
        self.ln = 1.0 + sh.bf16(f'{p}.input_layernorm.weight')
        self.pln = 1.0 + sh.bf16(f'{p}.post_attention_layernorm.weight')
        if attn:
            a = f'{p}.self_attn'
            self.wq, self.qn = lin(sh, f'{a}.q_proj'), 1.0 + sh.bf16(f'{a}.q_norm.weight')
            self.wk, self.kn = lin(sh, f'{a}.k_proj'), 1.0 + sh.bf16(f'{a}.k_norm.weight')
            self.wv = lin(sh, f'{a}.v_proj')
            self.wo = lin(sh, f'{a}.o_proj')
        else:
            d = f'{p}.linear_attn'
            self.wqkv = lin(sh, f'{d}.in_proj_qkv')
            self.wz = lin(sh, f'{d}.in_proj_z')
            self.wa = sh.bf16(f'{d}.in_proj_a.weight')          # bf16 [48, 5120]
            self.wb = sh.bf16(f'{d}.in_proj_b.weight')          # bf16 [48, 5120]
            self.convw = sh.bf16(f'{d}.conv1d.weight').reshape(CONV_C, 4)
            self.A_log = sh.bf16(f'{d}.A_log')                  # bf16 -> f32 [48]
            self.dt = sh.bf16(f'{d}.dt_bias')                   # bf16 -> f32 [48]
            self.normw = sh.bf16(f'{d}.norm.weight')            # ONE-centered: RAW
            self.wo = lin(sh, f'{d}.out_proj')
        self.wg = lin(sh, f'{p}.mlp.gate_proj')
        self.wu = lin(sh, f'{p}.mlp.up_proj')
        self.wd = lin(sh, f'{p}.mlp.down_proj')
        sh.close()

    def free(self):
        for k in list(vars(self)):
            setattr(self, k, None)


class States:
    """Per-layer recurrent state. Delta state S[head] is [k][v] -- the exact
    orientation of src/deltanet.cu: state[i*128 + tid] with i = KEY index and
    tid = VALUE index. mem[v] = sum_k S[k][v]*khat[k] (dot over the KEY axis,
    einsum 'hkv,hk->hv'), update S[k][v] += khat[k]*delta[v] (outer k (x) delta),
    out[v] = sum_k S[k][v]*qhat[k]. The 9B i4 scripts stored the transpose
    ('hvk'); this reference matches the GPU layout literally."""

    def __init__(self):
        self.conv = {}   # l -> [CONV_C, 3] f32 raw pre-conv inputs (oldest first)
        self.sst = {}    # l -> [LV, LD(k), LD(v)] f32 delta state
        self.kc = {}     # l -> list of [KVH, HD] f32 keys (append per position)
        self.vc = {}     # l -> list of [KVH, HD] f32 values

    def ensure(self, l, attn):
        if attn:
            self.kc.setdefault(l, [])
            self.vc.setdefault(l, [])
        else:
            self.conv.setdefault(l, np.zeros((CONV_C, 3), np.float32))
            self.sst.setdefault(l, np.zeros((LV, LD, LD), np.float32))


# ---------------------------------------------------------------- layer math
def mlp(ls, x):
    nrm = rms(x, ls.pln)
    g = ls.wg @ nrm
    u = ls.wu @ nrm
    return x + ls.wd @ (silu(g) * u)


def delta_step(ls, st, x, l):
    nrm = rms(x, ls.ln)
    qkvp = ls.wqkv @ nrm
    cw, cs = ls.convw, st.conv[l]
    y = cs[:, 0] * cw[:, 0] + cs[:, 1] * cw[:, 1] + cs[:, 2] * cw[:, 2] + qkvp * cw[:, 3]
    cs[:, 0] = cs[:, 1]
    cs[:, 1] = cs[:, 2]
    cs[:, 2] = qkvp                      # conv state keeps RAW pre-SiLU inputs
    y = silu(y)                          # SiLU on the whole 10240 channels
    q = y[:2048].reshape(LK, LD)
    k = y[2048:4096].reshape(LK, LD)
    v = y[4096:].reshape(LV, LD)
    q = q / (np.sqrt((q * q).sum(1, keepdims=True) + EPS) * np.float32(np.sqrt(LD)))
    k = k / np.sqrt((k * k).sum(1, keepdims=True) + EPS)   # unit norm, no scale
    k48 = np.repeat(k, KSHARE, 0)        # v head j -> k head j // 3
    q48 = np.repeat(q, KSHARE, 0)
    beta = sigmoid(ls.wb @ nrm)
    alpha = np.exp(-np.exp(ls.A_log) * np.logaddexp(np.float32(0), ls.wa @ nrm + ls.dt))
    S = st.sst[l]
    S *= alpha[:, None, None]            # decay FIRST, delta un-decayed
    mem = np.einsum('hkv,hk->hv', S, k48)
    dd = (v - mem) * beta[:, None]
    S += k48[:, :, None] * dd[:, None, :]
    out = np.einsum('hkv,hk->hv', S, q48)
    out = out / np.sqrt(np.mean(out * out, 1, keepdims=True) + EPS) * ls.normw[None, :]
    out = out * silu((ls.wz @ nrm).reshape(LV, LD))
    return mlp(ls, x + ls.wo @ out.reshape(-1))


ROPE_INV = 10000000.0 ** (-np.arange(ROPE_N, dtype=np.float64) / float(ROPE_N))


def rope64(hh, pos):
    """In-place partial RoPE on dims 0..63 of [heads, 256]; pairs (i, i+32),
    rotate_half convention; dims 64..255 untouched. f64 angles -> f32."""
    ang = pos * ROPE_INV
    c = np.cos(ang).astype(np.float32)
    s = np.sin(ang).astype(np.float32)
    a, b = hh[:, :ROPE_N].copy(), hh[:, ROPE_N:2 * ROPE_N].copy()
    hh[:, :ROPE_N] = a * c - b * s
    hh[:, ROPE_N:2 * ROPE_N] = b * c + a * s


def attn_step(ls, st, x, l, pos):
    nrm = rms(x, ls.ln)
    raw = (ls.wq @ nrm).reshape(QH, 2 * HD)     # per head: [256 q | 256 gate]
    q, gate = raw[:, :HD].copy(), raw[:, HD:].copy()
    k = (ls.wk @ nrm).reshape(KVH, HD)
    v = (ls.wv @ nrm).reshape(KVH, HD)
    q = q / np.sqrt(np.mean(q * q, 1, keepdims=True) + EPS) * ls.qn  # (1 + w)
    k = k / np.sqrt(np.mean(k * k, 1, keepdims=True) + EPS) * ls.kn
    rope64(q, pos)
    rope64(k, pos)
    st.kc[l].append(k)
    st.vc[l].append(v)
    K, V = np.stack(st.kc[l]), np.stack(st.vc[l])   # [T+, KVH, HD]
    out = np.empty((QH, HD), np.float32)
    for h in range(QH):
        kvh = h // GQA                              # 24/4 = 6
        sc = (K[:, kvh, :] @ q[h]) * ATT_SCALE      # 1/sqrt(256)
        w = np.exp(sc - sc.max())
        w /= w.sum()
        out[h] = w @ V[:, kvh, :]
    out = out * sigmoid(gate)                       # gate AFTER value mix
    return mlp(ls, x + ls.wo @ out.reshape(-1))


def run_layers(model_dir, model, X, states, through, pos_base=0, traj=None):
    """Layer-major: for l in 0..through, run all T tokens of X [T, 5120] through
    layer l in place (token-sequential inside the layer). traj[l] = layer-l
    output. Prints per-layer output norms."""
    T = X.shape[0]
    for l in range(through + 1):
        attn = (l & 3) == 3
        states.ensure(l, attn)
        ls = LayerSet(model_dir, l, attn)
        for t in range(T):
            if attn:
                X[t] = attn_step(ls, states, X[t], l, pos_base + t)
            else:
                X[t] = delta_step(ls, states, X[t], l)
        if traj is not None:
            traj[l] = X
        norms = [float(np.linalg.norm(X[t])) for t in range(T)]
        if T <= 20:
            detail = '[' + ', '.join(f'{n:.4f}' for n in norms) + ']'
        else:
            a = sorted(norms)
            detail = f'min={a[0]:.4f} med={a[T // 2]:.4f} max={a[-1]:.4f}'
        print(f'layer {l:2d} {"A" if attn else "D"} out_norm {detail}', flush=True)
        ls.free()
    return X


# ---------------------------------------------------------------- lm_head sweep
LM_CHUNK = 8192


def lm_head_sweep(model, Hn, fn):
    """Hn [T, 5120] f32 final-normed hiddens. Calls fn(logits_chunk [rows, T], r0)
    per 8192 vocab rows; the bf16 lm_head is read once, never materialized f32."""
    for i in range(0, VOCAB, LM_CHUNK):
        j = min(i + LM_CHUNK, VOCAB)
        fn(model.lm_rows(i, j) @ Hn.T, i)


def nll_positions(model, Hn, targets):
    """Online f64 logsumexp over the full vocab for each row of Hn; targets are
    the correct next-token ids. Returns per-position NLL f64 [T]."""
    T = Hn.shape[0]
    m = np.full(T, -np.inf)
    s = np.zeros(T)

    def acc(lc, _):
        nonlocal m, s
        lc64 = lc.astype(np.float64)
        mn = np.maximum(m, lc64.max(0))
        s = s * np.exp(m - mn) + np.exp(lc64 - mn).sum(0)
        m = mn

    lm_head_sweep(model, Hn, acc)
    lse = np.log(s) + m
    tgt = np.array([float(model.lm_rows(int(tk), int(tk) + 1)[0] @ Hn[t])
                    for t, tk in enumerate(targets)], np.float64)
    return lse - tgt


def greedy_argmax(model, h):
    """Full-vocab argmax for one final-normed hidden (8192-row chunks)."""
    best = [-np.inf, -1]  # [logit, vocab id]; '>' keeps the lowest index on ties

    def acc(lc, r0):
        j = int(lc[:, 0].argmax())
        if float(lc[j, 0]) > best[0]:
            best[0], best[1] = float(lc[j, 0]), r0 + j

    lm_head_sweep(model, h[None, :], acc)
    return best[1], best[0]


# ---------------------------------------------------------------- tokenizer glue
_TOK = {}


def _tokenizer(model_dir):
    if model_dir not in _TOK:
        from tokenizers import Tokenizer
        _TOK[model_dir] = Tokenizer.from_file(os.path.join(model_dir, 'tokenizer.json'))
    return _TOK[model_dir]


def enc(text, model_dir=MODEL_DIR_DEFAULT):
    return _tokenizer(model_dir).encode(text).ids


def dec(ids, model_dir=MODEL_DIR_DEFAULT):
    return _tokenizer(model_dir).decode(list(ids), skip_special_tokens=False)


def parse_ids(s):
    ids = [int(x) for x in s.replace(' ', '').split(',') if x != '']
    if not ids:
        raise SystemExit('empty token list')
    if min(ids) < 0 or max(ids) >= VOCAB:
        raise SystemExit(f'token id out of range [0, {VOCAB})')
    return ids


# ---------------------------------------------------------------- subcommands
def cmd_selftest(_):
    hand = np.array([
        float('nan') if (c & 0x7F) == 0x7F else
        (-1.0 if c & 0x80 else 1.0) * math.ldexp(
            (c & 0x7F) & 7, -9) if ((c & 0x7F) >> 3) == 0 else
        (-1.0 if c & 0x80 else 1.0) * math.ldexp(1.0 + ((c & 0x7F) & 7) / 8.0,
                                                 ((c & 0x7F) >> 3) - 7)
        for c in range(256)], np.float32)
    nan_lut = np.isnan(LUT8)
    nan_hand = np.isnan(hand)
    assert list(np.flatnonzero(nan_lut)) == [0x7F, 0xFF], 'NaN codes'
    assert np.array_equal(nan_lut, nan_hand), 'NaN positions vs hand table'
    fin = ~nan_lut
    assert np.array_equal(LUT8.view(np.uint32)[fin], hand.view(np.uint32)[fin]), \
        'bitwise mismatch vs hand-built ldexp table'
    assert int(fin.sum()) == 254 and int(nan_lut.sum()) == 2, 'finite/NaN counts'
    assert LUT8[0x7E] == 448.0 and LUT8[0xFE] == -448.0, 'max finite'
    assert float(LUT8[fin][:128].max()) == 448.0, 'global max finite'
    assert LUT8[0x01] == 2.0 ** -9 and LUT8[0x07] == 7 * 2.0 ** -9, 'subnormals m*2^-9'
    assert LUT8[0x08] == 2.0 ** -6, 'smallest normal'
    anchors = {0x0F: 0.029296875, 0x38: 1.0, 0x3F: 1.875, 0x40: 2.0,
               0x47: 3.75, 0x50: 8.0, 0x77: 240.0}
    for c, v in anchors.items():
        assert LUT8[c] == v, f'anchor 0x{c:02X} != {v}'
    pos = LUT8[1:0x7F].astype(np.float64)
    assert np.all(np.diff(pos) > 0), 'strict monotonicity 0x01..0x7E'
    assert LUT8[0] == 0.0 and LUT8[0x80] == 0.0, '+-0'
    for c in range(0x80):
        if c != 0x7F:  # 0x7F/0xFF are the two NaN codes
            assert LUT8[c + 0x80] == -LUT8[c], f'sign symmetry 0x{c:02X}'
    print('selftest PASS: 256 codes, 254 finite + 2 NaN (0x7F/0xFF), max +-448,')
    print('  subnormals m*2^-9, bitwise-identical to hand-built ldexp table,')
    print('  strict monotone 0x01..0x7E, sign-symmetric, anchors 1.0/1.875/240/448 ok')
    return 0


def _print_stats(n, ids, X):
    for t in range(X.shape[0]):
        v = X[t]
        first5 = np.array2string(v[:5], precision=6, floatmode='fixed')
        print(f'layer {n} tok {ids[t]}: |y|={float(np.linalg.norm(v)):.6f} '
              f'first5={first5}', flush=True)


def cmd_layer(a):
    ids = parse_ids(a.tokens)
    model = Model27B(a.model)
    traj = np.zeros((a.n + 1, len(ids), H), np.float32)
    X = np.stack([model.embed(t) for t in ids])
    run_layers(a.model, model, X, States(), a.n, traj=traj)
    if not a.no_save:
        out = a.out or os.path.join(REPO, 'build', '27b', f'layer{a.n}-traj.npy')
        if not out.endswith('.npy'):
            out += '.npy'
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        np.save(out, traj)
        print(f'trajectory {traj.shape} -> {out}')
    _print_stats(a.n, ids, X)
    return 0


def cmd_seams(a):
    ids = parse_ids(a.tokens)
    model = Model27B(a.model)
    traj = np.zeros((LAYERS + 1, len(ids), H), np.float32)
    X = np.stack([model.embed(t) for t in ids])
    run_layers(a.model, model, X, States(), LAYERS - 1, traj=traj)
    Xn = np.stack([rms(X[t], model.fnorm) for t in range(X.shape[0])])
    traj[LAYERS] = Xn
    for t in range(X.shape[0]):
        print(f'final-norm tok {ids[t]}: |y|={float(np.linalg.norm(Xn[t])):.6f}', flush=True)
    out = a.outfile
    if not out.endswith('.npy'):
        out += '.npy'
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.save(out, traj)
    print(f'{LAYERS + 1} seams {traj.shape} -> {out}')
    return 0


def cmd_nll(a):
    ids = parse_ids(a.tokens)
    if len(ids) < 2:
        raise SystemExit('nll needs >= 2 tokens')
    model = Model27B(a.model)
    X = np.stack([model.embed(t) for t in ids])
    run_layers(a.model, model, X, States(), LAYERS - 1)
    Hn = np.stack([rms(X[t], model.fnorm) for t in range(len(ids) - 1)])
    nll = nll_positions(model, Hn, ids[1:])
    for t in range(len(nll)):
        print(f'pos {t} tok {ids[t]} -> {ids[t + 1]}: nll={nll[t]:.6f}')
    mean = float(nll.mean())
    print(f'NLL mean {mean:.6f} nat/token  ppl {math.exp(mean):.4f}  '
          f'({len(nll)} positions)')
    return 0


def cmd_greedy(a):
    ids = parse_ids(a.tokens)
    model = Model27B(a.model)
    st = States()
    X = np.stack([model.embed(t) for t in ids])
    run_layers(a.model, model, X, st, LAYERS - 1)
    gen = []
    for step in range(a.n):
        h = rms(X[-1], model.fnorm)
        nxt, best = greedy_argmax(model, h)
        gen.append(nxt)
        print(f'step {step}: argmax {nxt} logit {best:.6f}', flush=True)
        x = model.embed(nxt)[None, :]
        run_layers(a.model, model, x, st, LAYERS - 1, pos_base=len(ids) + step)
        X = x
    print('generated ids:', gen)
    print('text:', dec(gen, a.model))
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=MODEL_DIR_DEFAULT)
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('selftest').set_defaults(fn=cmd_selftest)

    sp = sub.add_parser('layer', help='run layers 0..N over ids from embed')
    sp.add_argument('n', type=int)
    sp.add_argument('tokens')
    sp.add_argument('--attn', action='store_true',
                    help='force the full-attention math for layer N (shard must be attention-shaped)')
    sp.add_argument('--out', default=None)
    sp.add_argument('--no-save', action='store_true')
    sp.set_defaults(fn=cmd_layer)

    sp = sub.add_parser('seams', help='65 seams (64 layer outputs + final norm)')
    sp.add_argument('tokens')
    sp.add_argument('outfile')
    sp.set_defaults(fn=cmd_seams)

    sp = sub.add_parser('nll', help='full-model NLL over the id list')
    sp.add_argument('tokens')
    sp.set_defaults(fn=cmd_nll)

    sp = sub.add_parser('greedy', help='n greedy continuation tokens')
    sp.add_argument('tokens')
    sp.add_argument('n', type=int)
    sp.set_defaults(fn=cmd_greedy)

    sp = sub.add_parser('enc', help='encode text with the model dir tokenizer')
    sp.add_argument('text')
    sp.set_defaults(fn=lambda a: (print(enc(a.text, a.model)), 0)[1])

    sp = sub.add_parser('dec', help='decode comma-separated ids')
    sp.add_argument('ids')
    sp.set_defaults(fn=lambda a: (print(dec(parse_ids(a.ids), a.model)), 0)[1])

    a = ap.parse_args(argv)
    if getattr(a, 'n', 0) and getattr(a, 'attn', False) and (a.n & 3) != 3:
        print('note: --attn forced on a linear-attention slot; the shard has '
              'no self_attn tensors and this will fail', file=sys.stderr)
    with np.errstate(over='ignore', invalid='ignore'):
        return a.fn(a)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
