#!/usr/bin/env python3
"""Independent NumPy forward for the GLM-5.3-Flash (abliterated) big checkpoint.

Same math contract as tools/reference_glm53_numpy.py (the tiny oracle), but
reads the sharded NVFP4 production store directly: BF16 dense/side tensors,
ModelOpt NVFP4 routed experts (E2M1 codes, E4M3 scale per 16 columns, one F32
global multiplier per matrix).  No FP8/Q8 cache is consulted.

Per layer the mean of the 4 mHC streams (the residual state) is emitted; with
--engine-dump it is compared against the binary records written by the CUDA
engine's INSIGNIA_GLM53_LAYER_DUMP instrumentation:
    i32[3] {token_index, layer, hidden} then hidden f32 values.
"""

import argparse
import json
import math
import pathlib
import struct

import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)


def to_bf16(value):
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32).copy()
    bits += np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits >> np.uint32(16)) << np.uint32(16)).view(np.float32)


def sigmoid(value):
    value = np.asarray(value, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-value))


def softmax(value, axis=-1):
    value = np.asarray(value, dtype=np.float32)
    shifted = value - np.max(value, axis=axis, keepdims=True)
    result = np.exp(shifted)
    return result / np.sum(result, axis=axis, keepdims=True)


def decode_e4m3(raw, shape):
    u = np.frombuffer(raw, np.uint8).reshape(shape)
    sign = np.where(u & 0x80, -1.0, 1.0).astype(np.float32)
    exponent = ((u >> 3) & 15).astype(np.int16)
    mantissa = (u & 7).astype(np.float32)
    normal = np.ldexp(8.0 + mantissa, exponent - 10)
    subnormal = mantissa * np.float32(2.0**-9)
    out = sign * np.where(exponent == 0, subnormal, normal)
    if not np.isfinite(out).all():
        raise ValueError("non-finite E4M3 block scale")
    return out.astype(np.float32)


class Sharded:
    """Weight-map shard reader; tensors are materialized as requested."""

    def __init__(self, root: pathlib.Path):
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self.headers = {}
        self.handles = []

    def _header(self, shard_name):
        shard = self.root / shard_name
        if shard not in self.headers:
            with shard.open("rb") as file:
                size = struct.unpack("<Q", file.read(8))[0]
                self.headers[shard] = (json.loads(file.read(size)), 8 + size)
        return self.headers[shard]

    def meta(self, name):
        header, _ = self._header(self.weight_map[name])
        return header[name]

    def array(self, name):
        """Return the raw storage of a tensor as uint8 (U8/F8_E4M3) or widened
        BF16/F32 arrays; NVFP4 packed tensors stay packed."""
        shard = self.weight_map[name]
        header, data_start = self._header(shard)
        meta = header[name]
        begin, end = meta["data_offsets"]
        dtype, shape = meta["dtype"], meta["shape"]
        if dtype == "BF16":
            raw = np.memmap(self.root / shard, np.uint16, "r", data_start + begin, (end - begin) // 2)
            value = (np.asarray(raw, np.uint32) << np.uint32(16)).view(np.float32)
        elif dtype == "F32":
            value = np.array(np.memmap(self.root / shard, "<f4", "r", data_start + begin, (end - begin) // 4))
        elif dtype in ("U8", "F8_E4M3"):
            value = np.array(np.memmap(self.root / shard, np.uint8, "r", data_start + begin, end - begin))
        else:
            raise ValueError(f"unsupported dtype {dtype} for {name}")
        return value.reshape(shape)

    def nvfp4(self, stem):
        """Dequantize one NVFP4 matrix (stem without .weight suffix)."""
        packed = self.array(stem + ".weight")           # U8 [rows, cols/2]
        scales = decode_e4m3(self.array(stem + ".weight_scale").tobytes(),
                             self.meta(stem + ".weight_scale")["shape"])
        global_scale = float(np.asarray(self.array(stem + ".weight_scale_2")))
        rows, half = packed.shape
        codes = np.empty((rows, half * 2), np.uint8)
        codes[:, 0::2] = packed & 15
        codes[:, 1::2] = packed >> 4
        signs = np.where(codes & 8, -1.0, 1.0).astype(np.float32)
        values = E2M1[codes & 7] * signs
        if scales.shape != (rows, values.shape[1] // 16):
            raise ValueError(f"bad block-scale shape for {stem}: {scales.shape}")
        return (values * np.repeat(scales, 16, axis=1) * np.float32(global_scale)).astype(np.float32)


class Flash:
    def __init__(self, root: pathlib.Path):
        self.store = Sharded(root)
        config = json.loads((root / "config.json").read_text())["text_config"]
        self.config = config
        self.route_print = False
        self.force_routes = {}
        self.seam_layer = None
        self.seams = None
        self.hidden = config["hidden_size"]
        self.hc = config["hc_mult"]
        self.eps = config["rms_norm_eps"]
        self.hc_eps = config["hc_eps"]
        self.kda = config["linear_attn_config"]
        self.vocab = config["vocab_size"]

    def linear(self, x, name, weight=None):
        w = weight if weight is not None else self.store.array(name)
        return to_bf16(np.matmul(np.asarray(x, np.float32), w.T))

    def rms_norm(self, x, name):
        x = np.asarray(x, np.float32)
        normalized = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return to_bf16(self.store.array(name) * to_bf16(normalized))

    def hyper(self, streams, stem):
        flat = streams.reshape(streams.shape[0], -1).astype(np.float32)
        flat /= np.sqrt(np.mean(flat * flat, axis=-1, keepdims=True) + self.eps)
        mixed = np.matmul(flat, self.store.array(stem + "_fn").T.astype(np.float32))
        pre_w, post_w, comb_w = np.split(mixed, [self.hc, self.hc * 2], axis=-1)
        base = self.store.array(stem + "_base").astype(np.float32)
        scale = self.store.array(stem + "_scale").astype(np.float32)
        pre = sigmoid(pre_w * scale[0] + base[: self.hc]) + self.config["hc_eps"]
        post = 2.0 * sigmoid(post_w * scale[1] + base[self.hc : self.hc * 2])
        logits = comb_w.reshape(-1, self.hc, self.hc) * scale[2] + base[self.hc * 2 :].reshape(self.hc, self.hc)
        comb = softmax(logits, axis=-1) + self.config["hc_eps"]
        comb /= np.sum(comb, axis=-2, keepdims=True) + self.config["hc_eps"]
        for _ in range(self.config["hc_sinkhorn_iters"] - 1):
            comb /= np.sum(comb, axis=-1, keepdims=True) + self.config["hc_eps"]
            comb /= np.sum(comb, axis=-2, keepdims=True) + self.config["hc_eps"]
        collapsed = to_bf16(np.sum(pre[:, :, None] * streams, axis=1))
        return post, comb, collapsed

    @staticmethod
    def mix(streams, output, post, comb):
        placed = to_bf16(to_bf16(post)[:, :, None] * output[:, None, :])
        residual = to_bf16(np.matmul(to_bf16(np.swapaxes(comb, -1, -2)), streams))
        return to_bf16(placed + residual)

    def kda_block(self, x, layer):
        stem = f"model.language_model.layers.{layer}.self_attn."
        heads, head_dim = self.kda["num_heads"], self.kda["head_dim"]
        width = heads * head_dim
        q = self.linear(x, stem + "q_proj.weight")
        k = self.linear(x, stem + "k_proj.weight")
        v = self.linear(x, stem + "v_proj.weight")
        mixed = np.concatenate([q, k, v], axis=-1)
        weight = np.concatenate([
            self.store.array(stem + "q_conv1d.weight"),
            self.store.array(stem + "k_conv1d.weight"),
            self.store.array(stem + "v_conv1d.weight"),
        ], axis=0)[:, 0, :]
        convolved = np.zeros_like(mixed, np.float32)
        kernel = weight.shape[1]
        for token in range(x.shape[0]):
            for tap in range(kernel):
                source = token + tap - kernel + 1
                if source >= 0:
                    convolved[token] += mixed[source].astype(np.float32) * weight[:, tap]
        convolved = to_bf16(convolved * sigmoid(convolved))
        q, k, v = np.split(convolved, 3, axis=-1)
        q = q.reshape(x.shape[0], heads, head_dim).astype(np.float32)
        k = k.reshape(x.shape[0], heads, head_dim).astype(np.float32)
        v = v.reshape(x.shape[0], heads, head_dim).astype(np.float32)

        forget = self.linear(self.linear(x, stem + "f_a_proj.weight"), stem + "f_b_proj.weight")
        forget = forget.astype(np.float32) + self.store.array(stem + "dt_bias").astype(np.float32)
        decay = np.exp(self.store.array(stem + "A_log").astype(np.float32))[None, :, None]
        g = self.kda["gate_lower_bound"] * sigmoid(decay * forget.reshape(x.shape[0], heads, head_dim))
        beta = to_bf16(sigmoid(self.linear(x, stem + "b_proj.weight"))).astype(np.float32)

        q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        q *= 1.0 / math.sqrt(head_dim)
        state = np.zeros((heads, head_dim, head_dim), np.float32)
        core = np.zeros_like(q)
        for token in range(x.shape[0]):
            state *= np.exp(g[token])[:, :, None]
            memory = np.sum(state * k[token, :, :, None], axis=1)
            delta = (v[token] - memory) * beta[token, :, None]
            state += k[token, :, :, None] * delta[:, None, :]
            core[token] = np.sum(state * q[token, :, :, None], axis=1)
        core = to_bf16(core)
        gate = self.linear(self.linear(x, stem + "g_a_proj.weight"), stem + "g_b_proj.weight")
        gate = gate.reshape(x.shape[0], heads, head_dim).astype(np.float32)
        normalized = core.astype(np.float32)
        normalized /= np.sqrt(np.mean(normalized * normalized, axis=-1, keepdims=True) + self.eps)
        normalized *= self.store.array(stem + "o_norm.weight").astype(np.float32)
        normalized *= sigmoid(gate)
        return self.linear(to_bf16(normalized).reshape(x.shape[0], -1), stem + "o_proj.weight")

    def mla_block(self, x, layer):
        stem = f"model.language_model.layers.{layer}.self_attn."
        q_resid = self.rms_norm(self.linear(x, stem + "q_a_proj.weight"), stem + "q_a_layernorm.weight")
        query = self.linear(q_resid, stem + "q_b_proj.weight")
        qdim = self.config["qk_nope_head_dim"]
        vdim = self.config["v_head_dim"]
        query = query.reshape(x.shape[0], self.config["num_attention_heads"], qdim)
        kv = self.linear(x, stem + "kv_a_proj_with_mqa.weight")
        k_pass = self.rms_norm(kv, stem + "kv_a_layernorm.weight")
        kv = self.linear(k_pass, stem + "kv_b_proj.weight")
        kv = kv.reshape(x.shape[0], self.config["num_attention_heads"], qdim + vdim)
        key, value = np.split(kv, [qdim], axis=-1)
        output = np.zeros((x.shape[0], self.config["num_attention_heads"], vdim), np.float32)
        scale = 1.0 / math.sqrt(qdim)
        for token in range(x.shape[0]):
            logits = to_bf16(np.einsum("hd,thd->ht", query[token].astype(np.float32),
                                       key[: token + 1].astype(np.float32)))
            logits = to_bf16(logits * scale)
            probabilities = to_bf16(softmax(logits, axis=-1))
            output[token] = to_bf16(np.einsum("ht,thd->hd", probabilities, value[: token + 1].astype(np.float32)))
        return self.linear(to_bf16(output).reshape(x.shape[0], -1), stem + "o_proj.weight")

    def mlp(self, x, stem, weights=None):
        """Clamped SwiGLU; weights optionally pre-dequantized {gate,up,down}."""
        if weights is None:
            gate = np.minimum(self.linear(x, stem + "gate_proj.weight"), self.config["swiglu_limit"])
            up = np.clip(self.linear(x, stem + "up_proj.weight"),
                         -self.config["swiglu_limit"], self.config["swiglu_limit"])
        else:
            gate = np.minimum(self.linear(x, None, weights["gate"]), self.config["swiglu_limit"])
            up = np.clip(self.linear(x, None, weights["up"]),
                         -self.config["swiglu_limit"], self.config["swiglu_limit"])
        activated = to_bf16(gate.astype(np.float32) * sigmoid(gate.astype(np.float32)))
        mult = to_bf16(activated * up)
        if weights is None:
            return self.linear(mult, stem + "down_proj.weight")
        return self.linear(mult, None, weights["down"])

    def moe(self, x, layer):
        stem = f"model.language_model.layers.{layer}.mlp."
        router = np.matmul(x.astype(np.float32), self.store.array(stem + "gate.weight").T.astype(np.float32))
        scores = sigmoid(router)
        choice = scores + self.store.array(stem + "gate.e_score_correction_bias").astype(np.float32)
        count = self.config["num_experts_per_tok"]
        order = np.argsort(-choice[0], kind="stable")
        indices = order[:count]
        if layer in self.force_routes:
            indices = np.array(self.force_routes[layer], np.int64)
            if self.route_print:
                boundary = order[: count + 3]
                print(f"ref route layer {layer}: FORCED {' '.join(map(str, indices))}; "
                      f"natural top11 {' '.join(map(str, boundary))} "
                      f"choice {' '.join(f'{float(choice[0, e]):.6f}' for e in boundary)}", flush=True)
        elif self.route_print:
            boundary = order[: count + 3]
            print(f"ref route layer {layer}: experts {' '.join(map(str, sorted(indices)))} "
                  f"scores {' '.join(f'{float(scores[0, e]):.6f}' for e in sorted(indices))}; "
                  f"top11 {' '.join(map(str, boundary))} "
                  f"choice {' '.join(f'{float(choice[0, e]):.6f}' for e in boundary)}", flush=True)
        selected = scores[0, indices]
        denominator = float(np.sum(selected))
        routed = np.zeros_like(x, np.float32)
        for slot, expert in enumerate(indices):
            weight = self.config["routed_scaling_factor"] * float(scores[0, expert]) / denominator
            expert_stem = f"{stem}experts.{int(expert)}."
            weights = {
                "gate": self.store.nvfp4(expert_stem + "gate_proj"),
                "up": self.store.nvfp4(expert_stem + "up_proj"),
                "down": self.store.nvfp4(expert_stem + "down_proj"),
            }
            routed = to_bf16(routed + to_bf16(self.mlp(x, expert_stem, weights)[0].astype(np.float32) * weight))
            del weights
        shared = self.mlp(x, stem + "shared_experts.")
        return to_bf16(routed + shared)

    def forward(self, token_ids, layers, sink=None):
        means = {}
        seams = {} if self.seam_layer is not None else None
        embedding = self.store.array("model.language_model.embed_tokens.weight")[token_ids]
        embedding = to_bf16(embedding)
        streams = np.repeat(embedding[:, None, :], self.hc, axis=1)
        means[-1] = streams.mean(axis=1)
        for layer in range(layers):
            base = f"model.language_model.layers.{layer}"
            post, comb, collapsed = self.hyper(streams, base + ".hc_attn")
            normalized = self.rms_norm(collapsed, base + ".input_layernorm.weight")
            if seams is not None and layer == self.seam_layer:
                seams[1] = normalized[0].copy()
            if self.config["layer_types"][layer] == "linear_attention":
                attention = self.kda_block(normalized, layer)
            else:
                attention = self.mla_block(normalized, layer)
            if seams is not None and layer == self.seam_layer:
                seams[2] = attention[0].copy()
            streams = self.mix(streams, attention, post, comb)
            if seams is not None and layer == self.seam_layer:
                seams[3] = streams[0].copy()

            post, comb, collapsed = self.hyper(streams, base + ".hc_ffn")
            normalized = self.rms_norm(collapsed, base + ".post_attention_layernorm.weight")
            if seams is not None and layer == self.seam_layer:
                seams[4] = normalized[0].copy()
            if self.config["mlp_layer_types"][layer] == "sparse":
                feed_forward = self.moe(normalized, layer)
            else:
                feed_forward = self.mlp(normalized, base + ".mlp.")
            if seams is not None and layer == self.seam_layer:
                seams[5] = feed_forward[0].copy()
            streams = self.mix(streams, feed_forward, post, comb)
            if seams is not None and layer == self.seam_layer:
                seams[6] = streams[0].copy()
            means[layer] = streams.mean(axis=1)
            if sink:
                print(f"reference layer {layer:2d} done", flush=True)
        self.seams = seams
        return means


def read_engine_dump(path):
    import os
    size = os.path.getsize(path)
    records = {}
    with open(path, "rb") as file:
        while file.tell() < size:
            header = file.read(12)
            if len(header) < 12:
                break
            step, layer, hidden = struct.unpack("<iii", header)
            values = np.frombuffer(file.read(4 * hidden), np.float32)
            records.setdefault(step, {})[layer] = values
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--token", type=int, default=154820)
    parser.add_argument("--layers", type=int, default=15)
    parser.add_argument("--engine-dump", type=pathlib.Path, action="append", default=[],
                        help="compare against an INSIGNIA_GLM53_LAYER_DUMP file (repeatable)")
    parser.add_argument("--save", type=pathlib.Path, help="npz with per-layer reference means")
    parser.add_argument("--routes", action="store_true", help="print reference MoE routing per layer")
    parser.add_argument("--force-route", action="append", default=[],
                        help="layer:e1,e2,...,e8 to override the reference routing (repeatable)")
    parser.add_argument("--seam-layer", type=int, default=None,
                        help="capture sub-op seams for this layer (attn-norm, attn-out, "
                             "streams-after-attn-mix, ffn-norm, ffn-out, streams-after-ffn-mix)")
    parser.add_argument("--seam-compare", type=pathlib.Path,
                        help="engine INSIGNIA_GLM53_SEAM_DUMP file to compare against")
    args = parser.parse_args()

    model = Flash(args.model)
    model.route_print = args.routes
    model.seam_layer = args.seam_layer
    for spec in args.force_route:
        layer, experts = spec.split(":")
        model.force_routes[int(layer)] = [int(e) for e in experts.split(",")]
    means = model.forward(np.array([args.token], np.int64), args.layers, sink=True)
    if args.save:
        np.savez(args.save, **{f"layer{layer}": value for layer, value in means.items()})

    if args.seam_compare and model.seams:
        names = {1: "attn-norm", 2: "attn-out", 3: "streams-attn-mix",
                 4: "ffn-norm", 5: "ffn-out", 6: "streams-ffn-mix"}
        engine = {}
        with open(args.seam_compare, "rb") as file:
            while True:
                header = file.read(16)
                if len(header) < 16:
                    break
                _, _, tag, count = struct.unpack("<iiii", header)
                engine[tag] = np.frombuffer(file.read(4 * count), np.float32)
        print(f"\nseam comparison at layer {args.seam_layer} (engine vs reference):")
        for tag, name in names.items():
            if tag not in engine or model.seams is None or tag not in model.seams:
                continue
            got = engine[tag].astype(np.float64).ravel()
            want = model.seams[tag].astype(np.float64).ravel()
            delta = got - want
            cosine = float(np.vdot(got, want) /
                           max(np.linalg.norm(got) * np.linalg.norm(want), 1e-30))
            print(f"  {tag} {name:17s} cos={cosine:.9f} rel={np.linalg.norm(delta) / np.linalg.norm(want):.6f} "
                  f"max={np.abs(delta).max():.6g}")

    for dump in args.engine_dump:
        records = read_engine_dump(dump)
        step = min(records)
        print(f"\nengine dump {dump} step {step}: layer  cos        maxabs      rel_l2")
        worst = None
        for layer in sorted(means):
            if layer < 0 or layer not in records[step]:
                continue
            engine = records[step][layer].astype(np.float64)
            reference = means[layer][0].astype(np.float64)
            delta = engine - reference
            cosine = float(np.vdot(engine, reference) /
                           max(np.linalg.norm(engine) * np.linalg.norm(reference), 1e-30))
            rel = float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-30))
            maximum = float(np.max(np.abs(delta)))
            flag = "  <-- below 0.999" if cosine < 0.999 else ""
            if worst is None and cosine < 0.999:
                worst = layer
            print(f"  layer {layer:2d}  cos={cosine:.9f}  max={maximum:.6g}  rel={rel:.6f}{flag}")
        if worst is None:
            print("  no layer below 0.999")
        else:
            print(f"  first layer below 0.999: {worst}")


if __name__ == "__main__":
    main()
