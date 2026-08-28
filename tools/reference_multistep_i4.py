import json, struct, pathlib, sys, numpy as np
# argv: model.safetensors tokens(760,6511,...) native-dump.f32
# native dump layout: (steps+1) x 33 rows of 4096 f32 (32 layer outputs + final model.norm output)
p = pathlib.Path(sys.argv[1]); f = p.open('rb'); n = struct.unpack('<Q', f.read(8))[0]
hdr = json.loads(f.read(n)); start = 8 + n
dtmap = {'U32': '<u4', 'U8': 'u1', 'BF16': '<u2', 'F32': '<f4', 'F16': '<f2'}
def get(k):
    v = hdr[k]; f.seek(start + v['data_offsets'][0]); raw = f.read(v['data_offsets'][1] - v['data_offsets'][0])
    a = np.frombuffer(raw, dtype=dtmap[v['dtype']]).reshape(v['shape'])
    return (a.astype(np.uint32) << 16).view(np.float32) if v['dtype'] == 'BF16' else a
LUT = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
SH = np.arange(8, dtype=np.uint32) * 4
def dq(base, sl=None):
    w = get(base + '.weight'); s = get(base + '.scales')
    rows = w.shape[0]; cols = w.shape[1] * 8
    if sl is not None: w = w[sl]; s = s[sl]; rows = w.shape[0]
    q = ((w.reshape(rows, -1)[:, :, None] >> SH) & 15).reshape(rows, cols)
    sc = np.repeat(s.astype(np.float32), 64, axis=1)
    return LUT[q] * sc
def rms(x, w): return x / np.sqrt(np.mean(x * x) + 1e-6) * w
def mlp(x, p):
    nrm = rms(x, get(p + '.post_attention_layernorm.weight'))
    g = dq(p + '.mlp.gate_proj') @ nrm; u = dq(p + '.mlp.up_proj') @ nrm
    return x + dq(p + '.mlp.down_proj') @ (g / (1 + np.exp(-g)) * u)
def rope64(v, pos):
    v = v.copy(); a = pos * (10000000.0 ** (-np.arange(32, dtype=np.float64) / 32.0))
    c, s = np.cos(a), np.sin(a)
    x0, x1 = v[:64].copy(), v[:64].copy()
    first, second = v[:32], v[32:64]
    v[:32] = first * c - second * s; v[32:64] = second * c + first * s
    return v

tokens = [int(t) for t in sys.argv[2].split(',')]
native = np.fromfile(sys.argv[3], np.float32).reshape(len(tokens) + 1, 33, 4096)
states = [np.zeros((32, 128, 128), np.float32) for _ in range(24)]
convs = [np.zeros((8192, 3), np.float32) for _ in range(24)]
kvc = [np.zeros((64, 4, 256), np.float32) for _ in range(8)]  # per attn layer: pos x kvhead x dim
vcc = [np.zeros((64, 4, 256), np.float32) for _ in range(8)]
embed = get('language_model.model.embed_tokens.weight')  # U8 MXFP4 packed, scales separate
embed_scales = get('language_model.model.embed_tokens.scales')
def embed_row(tok):
    row = dq('language_model.model.embed_tokens', slice(tok, tok + 1))
    return row[0]

def delta(x, l, step):
    p = f'language_model.model.layers.{l}'; nrm = rms(x, get(p + '.input_layernorm.weight'))
    a = p + '.linear_attn'
    qkv = dq(a + '.in_proj_qkv') @ nrm; z = dq(a + '.in_proj_z') @ nrm
    aa = dq(a + '.in_proj_a') @ nrm; bb = dq(a + '.in_proj_b') @ nrm
    cw = get(a + '.conv1d.weight').reshape(8192, 4)
    di = l - l // 4
    cv = convs[di]
    convout = cv[:, 0] * cw[:, 0] + cv[:, 1] * cw[:, 1] + cv[:, 2] * cw[:, 2] + qkv * cw[:, 3]
    cv[:, 0] = cv[:, 1]; cv[:, 1] = cv[:, 2]; cv[:, 2] = qkv
    convout = convout / (1 + np.exp(-convout))
    q, k, v = np.split(convout, [2048, 4096])
    q = q.reshape(16, 128); k = k.reshape(16, 128); v = v.reshape(32, 128)
    q = q / (np.sqrt((q * q).sum(1, keepdims=True) + 1e-6) * np.sqrt(128))
    k = k / np.sqrt((k * k).sum(1, keepdims=True) + 1e-6)
    q = np.repeat(q, 2, 0); k = np.repeat(k, 2, 0)
    beta = 1 / (1 + np.exp(-bb))
    st = states[di]
    st *= np.exp(-np.exp(get(a + '.A_log')) * np.logaddexp(0, aa + get(a + '.dt_bias')))[:, None, None]
    mem = np.einsum('hvk,hk->hv', st, k)
    dd = (v - mem) * beta[:, None]
    st += dd[:, :, None] * k[:, None, :]
    out = np.einsum('hvk,hk->hv', st, q)
    nw = get(a + '.norm.weight')
    out = out / np.sqrt(np.mean(out * out, 1, keepdims=True) + 1e-6) * nw[None, :] * (z.reshape(32, 128) / (1 + np.exp(-z.reshape(32, 128))))
    return mlp(x + dq(a + '.out_proj') @ out.reshape(-1), p)

def attn(x, l, pos):
    p = f'language_model.model.layers.{l}'; a = p + '.self_attn'
    nrm = rms(x, get(p + '.input_layernorm.weight'))
    raw = (dq(a + '.q_proj') @ nrm).reshape(16, 512)
    q = raw[:, :256].copy(); gate = raw[:, 256:].copy()
    k = (dq(a + '.k_proj') @ nrm).reshape(4, 256); v = (dq(a + '.v_proj') @ nrm).reshape(4, 256)
    q = q / np.sqrt(np.mean(q * q, 1, keepdims=True) + 1e-6) * get(a + '.q_norm.weight')
    k = k / np.sqrt(np.mean(k * k, 1, keepdims=True) + 1e-6) * get(a + '.k_norm.weight')
    for h in range(16): q[h] = rope64(q[h], pos)
    for h in range(4): k[h] = rope64(k[h], pos)
    ai = l // 4
    kvc[ai][pos] = k; vcc[ai][pos] = v
    K = kvc[ai][:pos + 1]; V = vcc[ai][:pos + 1]  # (p+1, 4, 256)
    out = np.zeros((16, 256), np.float32)
    for h in range(16):
        kvh = h >> 2
        s = (K[:, kvh, :] @ q[h]) / 16.0
        s = np.exp(s - s.max()); s /= s.sum()
        out[h] = s @ V[:, kvh, :]
    out = out * (1 / (1 + np.exp(-gate)))
    return mlp(x + dq(a + '.o_proj') @ out.reshape(-1), p)

def lm_head_argmax(nrm):
    best_v, best_i = -1e30, -1
    for i in range(0, 248320, 8192):
        sl = slice(i, min(i + 8192, 248320))
        lg = dq('language_model.lm_head', sl) @ nrm
        j = int(lg.argmax())
        if lg[j] > best_v: best_v, best_i = float(lg[j]), i + j
    return best_i


worst = []
for step in range(len(tokens) + 1):
    if step < len(tokens): tok = tokens[step]
    else: tok = next_tok
    x = embed_row(tok)
    for l in range(32):
        if l % 4 == 3: x = attn(x, l, step)
        else: x = delta(x, l, step)
        d = native[step, l]
        c = float(x @ d / np.linalg.norm(x) / np.linalg.norm(d))
        worst.append((c, step, l))
    fn = rms(x, get('language_model.model.norm.weight'))
    d = native[step, 32]
    cf = float(fn @ d / np.linalg.norm(fn) / np.linalg.norm(d))
    am = lm_head_argmax(fn)
    next_tok = am
    print(f'step {step} tok {tok}: worst_layer_cos={min(c for c, s, l2 in worst if s == step):.8f} final_cos={cf:.8f} ref_argmax={am} ')
worst.sort()
print('overall worst:', [(f'{c:.8f}', s, l) for c, s, l in worst[:5]])
