import json, struct, pathlib, sys, numpy as np
# argv: model.safetensors tokens(760,6511,...) native-pf-dump.f32
# native dump layout: 33 seams x T x 4096 f32 (32 layer outputs + final model.norm output),
# seam-major (seam l block = T rows, one per token).
p = pathlib.Path(sys.argv[1]); f = p.open('rb'); n = struct.unpack('<Q', f.read(8))[0]
hdr = json.loads(f.read(n)); start = 8 + n
dtmap = {'U32': '<u4', 'U8': 'u1', 'BF16': '<u2', 'F32': '<f4', 'F16': '<f2'}
def get(k):
    v = hdr[k]; f.seek(start + v['data_offsets'][0]); raw = f.read(v['data_offsets'][1] - v['data_offsets'][0])
    a = np.frombuffer(raw, dtype=dtmap[v['dtype']]).reshape(v['shape'])
    return (a.astype(np.uint32) << 16).view(np.float32) if v['dtype'] == 'BF16' else a
LUT = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
SH = np.arange(8, dtype=np.uint32) * 4
_cache = {}
def dq(base):
    if base in _cache: return _cache[base]
    w = get(base + '.weight'); s = get(base + '.scales')
    rows, wc = w.shape; cols = wc * 8
    q = ((w.reshape(rows, -1)[:, :, None] >> SH) & 15).reshape(rows, cols)
    sc = np.repeat(s.astype(np.float32), 64, axis=1)
    r = np.ascontiguousarray(LUT[q] * sc)
    _cache[base] = r
    if len(_cache) > 14:  # keep the working set bounded
        for k in list(_cache)[:-14]: del _cache[k]
    return r
def rms(x, w): return x / np.sqrt(np.mean(x * x) + 1e-6) * w
def mlp(x, p):
    nrm = rms(x, get(p + '.post_attention_layernorm.weight'))
    g = dq(p + '.mlp.gate_proj') @ nrm; u = dq(p + '.mlp.up_proj') @ nrm
    return x + dq(p + '.mlp.down_proj') @ (g / (1 + np.exp(-g)) * u)
def delta(x, l):
    p = f'language_model.model.layers.{l}'; nrm = rms(x, get(p + '.input_layernorm.weight')); a = p + '.linear_attn'
    qkv = dq(a + '.in_proj_qkv') @ nrm; z = dq(a + '.in_proj_z') @ nrm; aa = dq(a + '.in_proj_a') @ nrm; bb = dq(a + '.in_proj_b') @ nrm
    cw = get(a + '.conv1d.weight').reshape(8192, 4)
    qkv = qkv * cw[:, 3]; qkv = qkv / (1 + np.exp(-qkv))
    q, k, v = np.split(qkv, [2048, 4096])
    q = q.reshape(16, 128); k = k.reshape(16, 128); v = v.reshape(32, 128)
    q = q / (np.sqrt((q * q).sum(1, keepdims=True) + 1e-6) * np.sqrt(128)); k = k / np.sqrt((k * k).sum(1, keepdims=True) + 1e-6)
    q = np.repeat(q, 2, 0); k = np.repeat(k, 2, 0)
    beta = 1 / (1 + np.exp(-bb)); di = l - l // 4; state = states[di]
    state *= np.exp(-np.exp(get(a + '.A_log')) * np.logaddexp(0, aa + get(a + '.dt_bias')))[:, None, None]
    mem = np.einsum('hvk,hk->hv', state, k); dd = (v - mem) * beta[:, None]
    state += dd[:, :, None] * k[:, None, :]
    out = np.einsum('hvk,hk->hv', state, q); nw = get(a + '.norm.weight')
    out = out / np.sqrt(np.mean(out * out, 1, keepdims=True) + 1e-6) * nw[None, :] * (z.reshape(32, 128) / (1 + np.exp(-z.reshape(32, 128))))
    return mlp(x + dq(a + '.out_proj') @ out.reshape(-1), p)

INV = 10000000.0 ** (-np.arange(32) / 32.0)  # partial rope 64 dims, pairs (i, i+32)
def rope64(h, pos):
    c, s = np.cos(pos * INV), np.sin(pos * INV)
    a, b = h[:, :32].copy(), h[:, 32:64].copy()
    h[:, :32] = a * c - b * s
    h[:, 32:64] = b * c + a * s

def attn(x, l, pos):
    p = f'language_model.model.layers.{l}'; a = p + '.self_attn'; nrm = rms(x, get(p + '.input_layernorm.weight'))
    raw = (dq(a + '.q_proj') @ nrm).reshape(16, 512); q = raw[:, :256].copy(); gate = raw[:, 256:]
    k = (dq(a + '.k_proj') @ nrm).reshape(4, 256).copy(); v = (dq(a + '.v_proj') @ nrm).reshape(4, 256)
    q = q / np.sqrt(np.mean(q * q, 1, keepdims=True) + 1e-6) * get(a + '.q_norm.weight')
    k = k / np.sqrt(np.mean(k * k, 1, keepdims=True) + 1e-6) * get(a + '.k_norm.weight')
    rope64(q, pos); rope64(k, pos)
    ai = l // 4
    kvk[ai].append(k); kvv[ai].append(v)
    K = np.stack(kvk[ai]); V = np.stack(kvv[ai])  # [T+,4,256]
    K2 = np.repeat(K, 4, axis=1); V2 = np.repeat(V, 4, axis=1)  # [T,16,256], kv head of q head h is h//4
    scores = np.einsum('hd,thd->ht', q, K2) * (1.0 / 16.0)
    w_att = np.exp(scores - scores.max(-1, keepdims=True)); w_att /= w_att.sum(-1, keepdims=True)
    o = np.einsum('ht,thd->hd', w_att, V2)
    out = o * (1 / (1 + np.exp(-gate)))
    return mlp(x + dq(a + '.o_proj') @ out.reshape(-1), p)

tokens = [int(t) for t in sys.argv[2].split(',')]
T = len(tokens)
native = np.fromfile(sys.argv[3], np.float32).reshape(33, T, 4096)
emb = dq('language_model.model.embed_tokens')
states = [np.zeros((32, 128, 128), np.float32) for _ in range(24)]
kvk = [[] for _ in range(8)]; kvv = [[] for _ in range(8)]
worst = []
for t in range(T):
    x = emb[tokens[t]]
    for l in range(32):
        x = attn(x, l, t) if l % 4 == 3 else delta(x, l)
        d = native[l, t]
        cos = float(x @ d / np.linalg.norm(x) / np.linalg.norm(d))
        worst.append((cos, t, l))
    xn = rms(x, get('language_model.model.norm.weight'))
    d = native[32, t]
    cos = float(xn @ d / np.linalg.norm(xn) / np.linalg.norm(d))
    worst.append((cos, t, 32))
worst.sort()
print('worst seams (cos, token, layer):', worst[:10])
print('overall worst cos:', worst[0][0])
