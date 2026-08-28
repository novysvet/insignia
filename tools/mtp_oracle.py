#!/usr/bin/env python3
"""NumPy oracle for the GLM-5.3 MTP layer-45 forward at position 0.

Reads a binary dump record produced by the engine (INSIGNIA_GLM53_MTP_DUMP),
replays eh_proj -> (attention = 0 at position 0) -> MoE -> shared_head.norm ->
lm_head with weights straight from the compact store, and prints the top-5
logits next to the engine's answer. Decodes NVFP4 exactly as the engine does
(low nibble first, e4m3 per-16 scales, fp32 per-tensor global).
"""
import json
import struct
import sys

import numpy as np

STORE = "/var/lib/insignia/glm53-flash-text"

# ---- safetensors shard reading -------------------------------------------
_headers = {}

def _header(shard_name):
    if shard_name in _headers:
        return _headers[shard_name]
    with open(f"{STORE}/{shard_name}", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    _headers[shard_name] = (hdr, 8 + n)
    return _headers[shard_name]

_INDEX = json.load(open("/var/lib/insignia/glm53-flash-text.index")) if False else None

def locate(name):
    import glob
    for shard in sorted(glob.glob(f"{STORE}/model-*.safetensors")):
        base = shard.split("/")[-1]
        hdr, _ = _header(base)
        if name in hdr:
            return base, hdr[name]
    raise KeyError(name)

def read_tensor(name):
    shard, meta = locate(name)
    with open(f"{STORE}/{shard}", "rb") as f:
        f.seek(0)
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + n + meta["data_offsets"][0])
        raw = f.read(meta["data_offsets"][1] - meta["data_offsets"][0])
    dtype, shape = meta["dtype"], meta["shape"]
    if dtype == "BF16":
        a = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return a.view(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    if dtype == "U8":
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    if dtype == "F8_E4M3":
        b = np.frombuffer(raw, dtype=np.uint8)
        # e4m3: 1-4-3, no infinities, NaN only at 0x7F/0xFF
        sign = np.where(b >= 128, -1.0, 1.0)
        e = b & 0x7F
        frac = (e & 7).astype(np.float32)
        exp = (e >> 3).astype(np.int32)
        val = np.where(exp == 0, frac / 8.0 * 0.5,
                       (1.0 + frac / 8.0) * np.exp2(exp - 7)).astype(np.float32)
        val[e == 0x7F] = np.nan
        return (sign * val).reshape(shape)
    raise ValueError(dtype)

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6],
                dtype=np.float32)

def nvfp4(packed, scales, global_scale):
    rows, nbytes = packed.shape
    codes = np.empty((rows, nbytes * 2), dtype=np.float32)
    lo = E2M1[packed & 15]
    hi = E2M1[packed >> 4]
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    scaled = codes.reshape(rows, -1, 16) * scales[:, :, None] * global_scale
    return scaled.reshape(rows, -1)

def rms_norm(x, w, eps=1e-5):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w

def main(dump_path):
    with open(dump_path, "rb") as f:
        token, position = struct.unpack("ii", f.read(8))
        hidden = np.frombuffer(f.read(4096 * 4), dtype=np.float32).copy()
        engine_argmax, = struct.unpack("i", f.read(4))
    print(f"dump: token={token} position={position} engine_argmax={engine_argmax}")

    stem = "model.language_model.layers.45."
    enorm = read_tensor(stem + "enorm.weight")
    hnorm = read_tensor(stem + "hnorm.weight")
    eh = read_tensor(stem + "eh_proj.weight")            # [4096, 8192]
    post_ln = read_tensor(stem + "post_attention_layernorm.weight")
    gate = read_tensor(stem + "mlp.gate.weight")          # [288, 4096]
    bias = read_tensor(stem + "mlp.gate.e_score_correction_bias")
    shared_norm = read_tensor(stem + "shared_head.norm.weight")
    embed = locate("model.language_model.embed_tokens.weight")
    lm_head = read_tensor("lm_head.weight")               # [154880, 4096]

    def expert_moe_input(ln):
        logits = gate @ ln
        scores = 1.0 / (1.0 + np.exp(-logits))
        choice = scores + bias
        top = np.argsort(-choice)[:8]
        denom = scores[top].sum()
        out = np.zeros(4096, dtype=np.float32)
        for e in top:
            p = f"{stem}mlp.experts.{e}."
            gw = nvfp4(read_tensor(p + "gate_proj.weight"),
                       read_tensor(p + "gate_proj.weight_scale"),
                       float(read_tensor(p + "gate_proj.weight_scale_2")))
            uw = nvfp4(read_tensor(p + "up_proj.weight"),
                       read_tensor(p + "up_proj.weight_scale"),
                       float(read_tensor(p + "up_proj.weight_scale_2")))
            dw = nvfp4(read_tensor(p + "down_proj.weight"),
                       read_tensor(p + "down_proj.weight_scale"),
                       float(read_tensor(p + "down_proj.weight_scale_2")))
            g = np.minimum(gw @ ln, 10.0)
            u = np.clip(uw @ ln, -10.0, 10.0)
            act = (g / (1.0 + np.exp(-g))) * u
            out += (dw @ act) * (2.5 * scores[e] / denom)
        return out, top

    for zero_embed in (True, False):
        if zero_embed and position == 0:
            eh_in = np.zeros(8192, dtype=np.float32)
        else:
            base, meta = embed
            with open(f"{STORE}/{base}", "rb") as f:
                f.seek(0)
                n = struct.unpack("<Q", f.read(8))[0]
                f.seek(8 + n + meta["data_offsets"][0] + token * 4096 * 2)
                row = np.frombuffer(f.read(4096 * 2), dtype=np.uint16).astype(np.uint32) << 16
            eh_in = np.zeros(8192, dtype=np.float32)
            eh_in[:4096] = rms_norm(row.view(np.float32), enorm)
        eh_in[4096:] = rms_norm(hidden, hnorm)
        x = eh @ eh_in
        # Attention at position 0 contributes nothing (no history).
        ln = rms_norm(x, post_ln)
        moe_out, top = expert_moe_input(ln)
        h2 = x + moe_out
        logits = lm_head @ rms_norm(h2, shared_norm)
        order = np.argsort(-logits)[:5]
        tag = "zero-embed" if zero_embed else "raw-embed "
        print(f"oracle[{tag}] experts={top.tolist()} top5:", end="")
        for i in order:
            print(f" {int(i)}:{logits[i]:.3f}", end="")
        print()

if __name__ == "__main__":
    main(sys.argv[1])
