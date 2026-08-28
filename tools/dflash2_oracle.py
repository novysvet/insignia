#!/usr/bin/env python3
"""NumPy oracle for the DFlash2 drafter against engine commit/draft dumps.

Reads the binary dump written by DFlash2Drafter (INSIGNIA_GLM53_DF_DUMP):
  tag 1 (commit): i32 count, i32 pos0, f32[count*5*4096] captured features
  tag 2 (draft):  i32 anchor id, i32 anchor position
Replays the commits in exact BF16->fp32 numpy (no FP8), runs the block
forward at each draft anchor, and prints the top-5 draft logits against the
next committed token. Discriminates CUDA/fp8 bugs (oracle agrees with truth)
from capture-semantics/draft-target mismatch (oracle wrong too).

Usage: dflash2_oracle.py <dump> <drafter.safetensors> <target_store_dir>
       [rounds] [expected_ids] [real|zero] [capture_noise] [layers]

Set layers=1 for the fast layer-0 CUDA/oracle regression. That mode skips the
2.4 GiB lm_head expansion and all later drafter layers.
"""

import json
import mmap
import pathlib
import struct
import sys

import numpy as np

H = 4096
LAYERS = 5
QH, KVH, HD = 32, 8, 128
INTER = 12288
BLOCK = 8
MASK = 154856
EPS = 1e-5
THETA = 1e4
STAGE_NAMES = (
    "xn_att", "dyn_att", "att_in", "q_raw", "k_raw", "v_raw",
    "q_rope", "k_rope", "att_out", "o_proj", "att_finish", "residual_att",
    "xn_mlp", "dyn_mlp", "mlp_in", "gate", "up", "swiglu", "down",
    "mlp_finish", "residual_mlp",
    "ctx_k", "ctx_v",
)


def bf16(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
    rows = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        rows[name] = (path, base + begin, meta["shape"], meta["dtype"])
    return rows


def load(rows, name, _unused=None):
    path, off, shape, dtype = rows[name]
    count = 1
    for s in shape:
        count *= s
    assert dtype == "BF16", f"{name} is {dtype}"
    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(count * 2)
    bits = np.frombuffer(raw, dtype="<u2", count=count).astype(np.uint32) << 16
    return bits.view(np.float32).reshape(shape)


def rms(x, w, axis=-1):
    ss = (x.astype(np.float64) ** 2).mean(axis=axis, keepdims=True)
    return (x / np.sqrt(ss + EPS)).astype(np.float32) * w


def rope(x, positions):
    # x: [..., heads, 128]; neox pairs (i, i+64)
    n = x.shape[-1] // 2
    inv = THETA ** (-2 * np.arange(n, dtype=np.float64) / 128.0)
    ang = np.asarray(positions, dtype=np.float64)[:, None] * inv  # [T, n]
    cos = np.cos(ang).astype(np.float32)[:, None]  # broadcast over heads
    sin = np.sin(ang).astype(np.float32)[:, None]
    a, b = x[..., :n], x[..., n:]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)


def head_rms(x, w):
    # per-head RMSNorm over the last dim of [..., heads, 128]
    return rms(x, w)


def conv(x, dyn, base, side):
    # x: [T, 4096]; dyn: [T, 1024] -> [T, 2*256] per side; base: [2,2,4096]
    rows = x.shape[0]
    groups = np.arange(H) // 16
    out = np.zeros_like(x)
    for t in range(rows):
        for tap in (0, 1):
            k = base[side, tap] + dyn[t, (side * 2 + tap) * 256 + groups]
            xp = x[t - 1] if t > 0 and tap == 1 else (x[t] if tap == 0 else 0.0)
            if tap == 0:
                out[t] += k * x[t]
            else:
                out[t] += k * (x[t - 1] if t > 0 else 0.0)
    return out


def attention(q, k_ctx, v_ctx, k_blk, v_blk):
    # q: [8, 32, 128]; k_*: [keys, 8, 128]
    keys = np.concatenate([k_ctx, k_blk], axis=0)  # [C+8, 8, 128]
    vals = np.concatenate([v_ctx, v_blk], axis=0)
    out = np.zeros_like(q)
    for t in range(q.shape[0]):
        for h in range(QH):
            kvh = h // 4
            scores = keys[:, kvh, :] @ q[t, h] / np.sqrt(HD, dtype=np.float32)
            p = np.exp(scores - scores.max())
            p /= p.sum()
            out[t, h] = p @ vals[:, kvh, :]
    return out.reshape(q.shape[0], QH * HD)


def main():
    dump_path, drafter_path, store_dir = sys.argv[1:4]
    max_rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    # Expected committed sequence (greedy IDs from the engine run); truth for
    # round r (0-based) is expected[r+1] since round r anchors on expected[r].
    expected = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else []
    ctx_mode = sys.argv[6] if len(sys.argv) > 6 else "real"
    ctx_noise = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    compare_layers = int(sys.argv[8]) if len(sys.argv) > 8 else LAYERS
    assert 1 <= compare_layers <= LAYERS
    rng = np.random.default_rng(7)

    drows = bf16(drafter_path)
    W = {}
    for l in range(compare_layers):
        s = f"layers.{l}."
        W[f"iln{l}"] = load(drows, s + "input_layernorm.weight", None)
        W[f"pln{l}"] = load(drows, s + "post_attention_layernorm.weight", None)
        W[f"qw{l}"] = load(drows, s + "self_attn.q_norm.weight", None)
        W[f"kw{l}"] = load(drows, s + "self_attn.k_norm.weight", None)
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            W[f"{n}{l}"] = load(drows, s + f"self_attn.{n}.weight", None)
        for n in ("gate_proj", "up_proj", "down_proj"):
            W[f"{n}{l}"] = load(drows, s + f"mlp.{n}.weight", None)
        W[f"ab{l}"] = load(drows, s + "attention_conv.base_kernel", None)
        W[f"akp{l}"] = load(drows, s + "attention_conv.kernel_projection.weight", None)
        W[f"mb{l}"] = load(drows, s + "mlp_conv.base_kernel", None)
        W[f"mkp{l}"] = load(drows, s + "mlp_conv.kernel_projection.weight", None)
    W["fc"] = load(drows, "fc.weight", None)
    W["hn"] = load(drows, "hidden_norm.weight", None)
    if compare_layers == LAYERS:
        W["fn"] = load(drows, "norm.weight", None)

    # Target store: find embed_tokens + lm_head shards.
    store = pathlib.Path(store_dir)
    embed_lm = {}
    for shard in sorted(store.glob("*.safetensors")):
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        base = 8 + n
        for name, meta in header.items():
            if name.endswith("embed_tokens.weight") or \
                    (compare_layers == LAYERS and name == "lm_head.weight"):
                embed_lm[name] = (shard, base + meta["data_offsets"][0], meta["shape"])
    assert any("embed" in k for k in embed_lm)
    embed_key = [k for k in embed_lm if "embed" in k][0]

    def embed_row(token):
        shard, off, shape = embed_lm[embed_key]
        with open(shard, "rb") as f:
            f.seek(off + token * H * 2)
            bits = np.frombuffer(f.read(H * 2), dtype="<u2").astype(np.uint32) << 16
        return bits.view(np.float32)

    lm_file = lm_map = lm_head = None
    if compare_layers == LAYERS:
        assert "lm_head.weight" in embed_lm
        lm_shard, lm_off, _ = embed_lm["lm_head.weight"]
        lm_file = open(lm_shard, "rb")
        lm_map = mmap.mmap(lm_file.fileno(), 0, access=mmap.ACCESS_READ)
        lm_bits = np.frombuffer(lm_map, dtype="<u2", count=154880 * H,
                                offset=lm_off).astype(np.uint32) << 16
        lm_head = lm_bits.view(np.float32).reshape(154880, H)  # ~2.4 GB fp32

    def ctx_kv(features):
        # features: [5, 4096] -> per-layer k/v rows
        x = rms(features.reshape(-1) @ W["fc"].T, W["hn"])  # [4096]
        outs = []
        for l in range(compare_layers):
            k = x @ W[f"k_proj{l}"].T
            v = x @ W[f"v_proj{l}"].T
            outs.append((k, v))
        return x, outs

    def block_forward(anchor, position, ctx):
        S = position
        x = np.stack([embed_row(anchor)] + [embed_row(MASK)] * (BLOCK - 1))  # [8, 4096]
        pos_block = list(range(S, S + BLOCK))
        my_layer_trace = {}
        my_stage_trace = {}
        for l in range(compare_layers):
            xn = rms(x, W[f"iln{l}"])
            my_stage_trace[(l, 0)] = xn.copy()
            dyn = xn @ W[f"akp{l}"].T  # [8, 1024]
            my_stage_trace[(l, 1)] = dyn.copy()
            att_in = conv(xn, dyn, W[f"ab{l}"], 0)
            my_stage_trace[(l, 2)] = att_in.copy()
            if l == 0:
                print(f"  oracle stats: x0 max {np.abs(x).max():.3e} "
                      f"xn max {np.abs(xn).max():.3e} dyn max {np.abs(dyn).max():.3e} "
                      f"att_in max {np.abs(att_in).max():.3e}")
            q = att_in @ W[f"q_proj{l}"].T
            k_blk = att_in @ W[f"k_proj{l}"].T
            v_blk = att_in @ W[f"v_proj{l}"].T
            my_stage_trace[(l, 3)] = q.copy()
            my_stage_trace[(l, 4)] = k_blk.copy()
            my_stage_trace[(l, 5)] = v_blk.copy()
            k_ctx_l, v_ctx_l = ctx[l]
            q = head_rms(q.reshape(BLOCK, QH, HD), W[f"qw{l}"])
            q = rope(q, pos_block)
            k_blk = head_rms(k_blk.reshape(BLOCK, KVH, HD), W[f"kw{l}"])
            k_blk = rope(k_blk, pos_block)
            my_stage_trace[(l, 6)] = q.reshape(BLOCK, -1).copy()
            my_stage_trace[(l, 7)] = k_blk.reshape(BLOCK, -1).copy()
            k_ctx = rope(head_rms(k_ctx_l.reshape(-1, KVH, HD), W[f"kw{l}"]),
                         list(range(S + 1)))
            v_ctx = v_ctx_l.reshape(-1, KVH, HD)
            my_stage_trace[(l, 21)] = k_ctx.reshape(-1, KVH * HD).copy()
            my_stage_trace[(l, 22)] = v_ctx.reshape(-1, KVH * HD).copy()
            v_blk3 = v_blk.reshape(BLOCK, KVH, HD)
            att = attention(q, k_ctx, v_ctx, k_blk, v_blk3)
            my_stage_trace[(l, 8)] = att.copy()
            o = att @ W[f"o_proj{l}"].T
            my_stage_trace[(l, 9)] = o.copy()
            fin = conv(o, dyn, W[f"ab{l}"], 1)
            my_stage_trace[(l, 10)] = fin.copy()
            if l == 0:
                print(f"  oracle stats: q max {np.abs(q).max():.3e} "
                      f"k_blk max {np.abs(k_blk).max():.3e} "
                      f"k_ctx max {np.abs(k_ctx).max():.3e} att max {np.abs(att).max():.3e} "
                      f"o max {np.abs(o).max():.3e} attn_fin max {np.abs(fin).max():.3e}")
            fin = conv(o, dyn, W[f"ab{l}"], 1)
            x = x + fin
            my_stage_trace[(l, 11)] = x.copy()
            xn = rms(x, W[f"pln{l}"])
            my_stage_trace[(l, 12)] = xn.copy()
            dyn = xn @ W[f"mkp{l}"].T
            my_stage_trace[(l, 13)] = dyn.copy()
            mlp_in = conv(xn, dyn, W[f"mb{l}"], 0)
            my_stage_trace[(l, 14)] = mlp_in.copy()
            g = mlp_in @ W[f"gate_proj{l}"].T
            u = mlp_in @ W[f"up_proj{l}"].T
            my_stage_trace[(l, 15)] = g.copy()
            my_stage_trace[(l, 16)] = u.copy()
            h = (g / (1 + np.exp(-g))) * u
            my_stage_trace[(l, 17)] = h.copy()
            d = h @ W[f"down_proj{l}"].T
            my_stage_trace[(l, 18)] = d.copy()
            fin = conv(d, dyn, W[f"mb{l}"], 1)
            my_stage_trace[(l, 19)] = fin.copy()
            x = x + fin
            my_stage_trace[(l, 20)] = x.copy()
            my_layer_trace[l] = x
        hidden = rms(x[1:], W["fn"]) if compare_layers == LAYERS else None
        logits = hidden @ lm_head.T if hidden is not None else None
        last_forward_cache[0] = x
        last_forward_cache[1] = hidden
        last_layer_trace.clear()
        last_layer_trace.update(my_layer_trace)
        last_stage_trace.clear()
        last_stage_trace.update(my_stage_trace)
        return logits

    # Replay the dump.
    committed = []       # list of per-token ctx KV (list of 5 (k,v) pairs)
    tokens = []
    rounds = 0
    last_forward_cache = [None, None]
    layer_trace = {}     # engine per-layer x_block snapshots (tag 4)
    last_layer_trace = {}
    last_stage_trace = {}

    def compare_stage(layer, stage, engine):
        oracle = last_stage_trace.get((layer, stage))
        if oracle is None:
            return
        oracle = oracle.reshape(engine.shape)
        cos = float(np.sum(engine * oracle) /
                    (np.linalg.norm(engine) * np.linalg.norm(oracle) + 1e-30))
        name = STAGE_NAMES[stage] if 0 <= stage < len(STAGE_NAMES) else str(stage)
        print(f"  stage L{layer}.{name}: cos {cos:.6f} "
              f"max|d| {np.abs(engine - oracle).max():.4e} "
              f"(engine max {np.abs(engine).max():.3e}, oracle max {np.abs(oracle).max():.3e})")
    with open(dump_path, "rb") as f:
        while True:
            tag = f.read(1)
            if not tag:
                break
            tag = tag[0]
            if tag == 1:
                count, pos0 = struct.unpack("<ii", f.read(8))
                feats = np.frombuffer(f.read(4 * count * 5 * H), dtype=np.float32)
                feats = feats.reshape(count, 5, H).copy()
                if ctx_noise > 0:
                    scale = np.linalg.norm(feats, axis=2, keepdims=True) / np.sqrt(H)
                    feats += rng.standard_normal(feats.shape).astype(np.float32) * \
                        (ctx_noise * scale)
                for t in range(count):
                    _, outs = ctx_kv(feats[t])
                    committed.append(outs)
            elif tag == 2:
                anchor, position = struct.unpack("<ii", f.read(8))
                assert len(committed) == position + 1, \
                    f"ctx {len(committed)} != pos {position}+1"
                ctx = []
                for l in range(compare_layers):
                    k = np.concatenate([committed[p][l][0][None] for p in range(position + 1)])
                    v = np.concatenate([committed[p][l][1][None] for p in range(position + 1)])
                    if ctx_mode == "zero":
                        k = np.zeros_like(k)
                        v = np.zeros_like(v)
                    ctx.append((k, v))
                logits = block_forward(anchor, position, ctx)
                print(f"draft anchor={anchor} pos={position}")
                if logits is not None:
                    for t in range(min(4, logits.shape[0])):
                        row = logits[t]
                        top = np.argsort(-row)[:5]
                        extra = ""
                        if t == 0 and len(expected) > position + 1:
                            truth = expected[position + 1]
                            rank = int((row > row[truth]).sum())
                            extra = f"  [truth {truth} logit {row[truth]:.3f} rank {rank}]"
                        print(f"  t{t} top5 " + " ".join(
                            f"{int(i)}:{row[i]:.3f}" for i in top) + extra)
                rounds += 1
                if rounds >= max_rounds:
                    # Drain trailing trace records (tags 3/4) then stop.
                    while True:
                        nxt = f.read(1)
                        if not nxt:
                            break
                        if nxt[0] == 3:
                            pos3 = struct.unpack("<i", f.read(4))[0]
                            n = struct.unpack("<i", f.read(4))[0]
                            eng_block = np.frombuffer(f.read(4 * n), dtype=np.float32).reshape(8, H)
                            n = struct.unpack("<i", f.read(4))[0]
                            eng_hidden = np.frombuffer(f.read(4 * n), dtype=np.float32).reshape(-1, H)
                            my_block, my_hidden = last_forward_cache
                            pairs = [("x_block", eng_block, my_block)]
                            if my_hidden is not None:
                                pairs.append(("hidden", eng_hidden, my_hidden))
                            for name, a, b in pairs:
                                cos = float(np.sum(a * b) /
                                            (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
                                print(f"  engine-vs-oracle {name} (pos {pos3}): cos {cos:.6f} "
                                      f"max|d| {np.abs(a - b).max():.4e}")
                            for l, eng in sorted(layer_trace.items()):
                                mine = last_layer_trace.get(l)
                                if mine is None:
                                    continue
                                cos = float(np.sum(eng * mine) /
                                            (np.linalg.norm(eng) * np.linalg.norm(mine) + 1e-30))
                                print(f"  layer {l}: cos {cos:.6f} max|d| {np.abs(eng - mine).max():.4e} "
                                      f"(engine max {np.abs(eng).max():.3e}, oracle max {np.abs(mine).max():.3e})")
                            break
                        elif nxt[0] == 4:
                            layer = struct.unpack("<b", f.read(1))[0]
                            n = struct.unpack("<i", f.read(4))[0]
                            layer_trace[layer] = np.frombuffer(f.read(4 * n), dtype=np.float32).reshape(8, H)
                        elif nxt[0] == 5:
                            layer, stage = struct.unpack("<bb", f.read(2))
                            rows, cols = struct.unpack("<ii", f.read(8))
                            engine = np.frombuffer(f.read(4 * rows * cols), dtype=np.float32).reshape(rows, cols)
                            compare_stage(layer, stage, engine)
                        else:
                            raise ValueError(f"unexpected tag {nxt[0]} in drain")
                    break
            elif tag == 3:
                pos3 = struct.unpack("<i", f.read(4))[0]
                n = struct.unpack("<i", f.read(4))[0]
                eng_block = np.frombuffer(f.read(4 * n), dtype=np.float32).reshape(8, H)
                n = struct.unpack("<i", f.read(4))[0]
                eng_hidden = np.frombuffer(f.read(4 * n), dtype=np.float32).reshape(-1, H)
                my_block, my_hidden = last_forward_cache
                pairs = [("x_block", eng_block, my_block)]
                if my_hidden is not None:
                    pairs.append(("hidden", eng_hidden, my_hidden))
                for name, a, b in pairs:
                    cos = float(np.sum(a * b) /
                                (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
                    print(f"  engine-vs-oracle {name} (pos {pos3}): cos {cos:.6f} "
                          f"max|d| {np.abs(a - b).max():.4e}")
                for l, eng in sorted(layer_trace.items()):
                    mine = last_layer_trace.get(l)
                    if mine is None:
                        continue
                    cos = float(np.sum(eng * mine) /
                                (np.linalg.norm(eng) * np.linalg.norm(mine) + 1e-30))
                    print(f"  layer {l}: cos {cos:.6f} max|d| {np.abs(eng - mine).max():.4e} "
                          f"(engine max {np.abs(eng).max():.3e}, oracle max {np.abs(mine).max():.3e})")
            elif tag == 4:
                layer = struct.unpack("<b", f.read(1))[0]
                n = struct.unpack("<i", f.read(4))[0]
                layer_trace[layer] = np.frombuffer(f.read(4 * n), dtype=np.float32).reshape(8, H)
            elif tag == 5:
                layer, stage = struct.unpack("<bb", f.read(2))
                rows, cols = struct.unpack("<ii", f.read(8))
                engine = np.frombuffer(f.read(4 * rows * cols), dtype=np.float32).reshape(rows, cols)
                compare_stage(layer, stage, engine)
            else:
                raise ValueError(f"bad tag {tag}")
    print(f"replayed {len(committed)} commits, {rounds} drafts")


if __name__ == "__main__":
    main()
