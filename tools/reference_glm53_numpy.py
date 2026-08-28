#!/usr/bin/env python3
"""Independent NumPy GLM-5.3 text forward pass for the tiny architecture oracle.

This intentionally implements the math instead of importing Transformers.  It
is small-model validation code, not the production runtime.  The CUDA engine can
use its seam errors to find the first divergent operator.
"""

import argparse
import json
import math
import pathlib
import struct

import numpy as np


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


class SafeTensors:
    def __init__(self, path):
        self.path = path
        with path.open("rb") as file:
            header_size = struct.unpack("<Q", file.read(8))[0]
            self.header = json.loads(file.read(header_size))
        self.data_start = 8 + header_size
        self.cache = {}

    def get(self, name):
        if name in self.cache:
            return self.cache[name]
        meta = self.header[name]
        begin, end = meta["data_offsets"]
        if meta["dtype"] == "BF16":
            raw = np.memmap(self.path, np.uint16, "r", self.data_start + begin, (end - begin) // 2)
            value = (np.asarray(raw, np.uint32) << np.uint32(16)).view(np.float32)
        elif meta["dtype"] == "F32":
            value = np.asarray(np.memmap(self.path, "<f4", "r", self.data_start + begin, (end - begin) // 4))
        else:
            raise ValueError(f"unsupported tiny-oracle dtype {meta['dtype']} for {name}")
        value = value.reshape(meta["shape"])
        self.cache[name] = value
        return value


class Glm53Numpy:
    def __init__(self, root, oracle=None):
        self.root = root
        config = json.loads((root / "config.json").read_text())["text_config"]
        self.config = config
        self.weights = SafeTensors(root / "model.safetensors")
        self.hidden = config["hidden_size"]
        self.heads = config["linear_attn_config"]["num_heads"]
        self.head_dim = config["linear_attn_config"]["head_dim"]
        self.hc = config["hc_mult"]
        self.eps = config["rms_norm_eps"]
        self.oracle = np.load(oracle) if oracle else None
        self.worst = ("", 0.0)

    def tensor(self, name):
        return self.weights.get(name)

    def check(self, name, value):
        if self.oracle is None or name not in self.oracle:
            return
        reference = np.asarray(self.oracle[name], np.float32).reshape(value.shape)
        actual = np.asarray(value, np.float32)
        delta = actual - reference
        rel = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(reference.ravel()), 1e-30))
        cosine = float(np.vdot(actual.ravel(), reference.ravel()) / max(
            np.linalg.norm(actual.ravel()) * np.linalg.norm(reference.ravel()), 1e-30))
        maximum = float(np.max(np.abs(delta)))
        print(f"{name:58s} rel={rel:9.6f} cos={cosine:.9f} max={maximum:.6g}")
        if rel > self.worst[1]:
            self.worst = (name, rel)

    def linear(self, x, name):
        return to_bf16(np.matmul(np.asarray(x, np.float32), self.tensor(name).T))

    def rms_norm(self, x, name):
        x = np.asarray(x, np.float32)
        normalized = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return to_bf16(self.tensor(name) * to_bf16(normalized))

    def hyper(self, streams, stem, oracle_stem):
        flat = streams.reshape(streams.shape[0], -1).astype(np.float32)
        flat /= np.sqrt(np.mean(flat * flat, axis=-1, keepdims=True) + self.eps)
        mixed = np.matmul(flat, self.tensor(stem + "_fn").T.astype(np.float32))
        pre_w, post_w, comb_w = np.split(mixed, [self.hc, self.hc * 2], axis=-1)
        base = self.tensor(stem + "_base").astype(np.float32)
        scale = self.tensor(stem + "_scale").astype(np.float32)
        pre = sigmoid(pre_w * scale[0] + base[: self.hc]) + self.config["hc_eps"]
        post = 2.0 * sigmoid(post_w * scale[1] + base[self.hc : self.hc * 2])
        logits = comb_w.reshape(-1, self.hc, self.hc) * scale[2] + base[self.hc * 2 :].reshape(self.hc, self.hc)
        comb = softmax(logits, axis=-1) + self.config["hc_eps"]
        comb /= np.sum(comb, axis=-2, keepdims=True) + self.config["hc_eps"]
        for _ in range(self.config["hc_sinkhorn_iters"] - 1):
            comb /= np.sum(comb, axis=-1, keepdims=True) + self.config["hc_eps"]
            comb /= np.sum(comb, axis=-2, keepdims=True) + self.config["hc_eps"]
        collapsed = to_bf16(np.sum(pre[:, :, None] * streams, axis=1))
        self.check(oracle_stem + ".0", post[None])
        self.check(oracle_stem + ".1", comb[None])
        self.check(oracle_stem + ".2", collapsed[None])
        return post, comb, collapsed

    @staticmethod
    def mix(streams, output, post, comb):
        placed = to_bf16(to_bf16(post)[:, :, None] * output[:, None, :])
        residual = to_bf16(np.matmul(to_bf16(np.swapaxes(comb, -1, -2)), streams))
        return to_bf16(placed + residual)

    def kda(self, x, layer):
        stem = f"model.language_model.layers.{layer}.self_attn."
        q = self.linear(x, stem + "q_proj.weight")
        k = self.linear(x, stem + "k_proj.weight")
        v = self.linear(x, stem + "v_proj.weight")
        mixed = np.concatenate([q, k, v], axis=-1)
        weight = np.concatenate([
            self.tensor(stem + "q_conv1d.weight"),
            self.tensor(stem + "k_conv1d.weight"),
            self.tensor(stem + "v_conv1d.weight"),
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
        q = q.reshape(x.shape[0], self.heads, self.head_dim).astype(np.float32)
        k = k.reshape(x.shape[0], self.heads, self.head_dim).astype(np.float32)
        v = v.reshape(x.shape[0], self.heads, self.head_dim).astype(np.float32)

        forget = self.linear(self.linear(x, stem + "f_a_proj.weight"), stem + "f_b_proj.weight")
        forget = forget.astype(np.float32) + self.tensor(stem + "dt_bias").astype(np.float32)
        decay = np.exp(self.tensor(stem + "A_log").astype(np.float32))[None, :, None]
        g = self.config["linear_attn_config"]["gate_lower_bound"] * sigmoid(
            decay * forget.reshape(x.shape[0], self.heads, self.head_dim))
        beta = to_bf16(sigmoid(self.linear(x, stem + "b_proj.weight"))).astype(np.float32)

        q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        q *= 1.0 / math.sqrt(self.head_dim)
        state = np.zeros((self.heads, self.head_dim, self.head_dim), np.float32)
        core = np.zeros_like(q)
        for token in range(x.shape[0]):
            state *= np.exp(g[token])[:, :, None]
            memory = np.sum(state * k[token, :, :, None], axis=1)
            delta = (v[token] - memory) * beta[token, :, None]
            state += k[token, :, :, None] * delta[:, None, :]
            core[token] = np.sum(state * q[token, :, :, None], axis=1)
        core = to_bf16(core)
        gate = self.linear(self.linear(x, stem + "g_a_proj.weight"), stem + "g_b_proj.weight")
        gate = gate.reshape(x.shape[0], self.heads, self.head_dim).astype(np.float32)
        normalized = core.astype(np.float32)
        normalized /= np.sqrt(np.mean(normalized * normalized, axis=-1, keepdims=True) + self.eps)
        normalized *= self.tensor(stem + "o_norm.weight").astype(np.float32)
        normalized *= sigmoid(gate)
        return self.linear(to_bf16(normalized).reshape(x.shape[0], -1), stem + "o_proj.weight")

    def mla(self, x, layer):
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
            logits = to_bf16(np.einsum("hd,thd->ht", query[token].astype(np.float32), key[: token + 1].astype(np.float32)))
            logits = to_bf16(logits * scale)
            probabilities = to_bf16(softmax(logits, axis=-1))
            output[token] = to_bf16(np.einsum("ht,thd->hd", probabilities, value[: token + 1].astype(np.float32)))
        return self.linear(to_bf16(output).reshape(x.shape[0], -1), stem + "o_proj.weight")

    def mlp(self, x, stem):
        gate = np.minimum(self.linear(x, stem + "gate_proj.weight"), self.config["swiglu_limit"])
        up = np.clip(self.linear(x, stem + "up_proj.weight"), -self.config["swiglu_limit"], self.config["swiglu_limit"])
        activated = to_bf16(gate.astype(np.float32) * sigmoid(gate.astype(np.float32)))
        return self.linear(to_bf16(activated * up), stem + "down_proj.weight")

    def moe(self, x, layer):
        stem = f"model.language_model.layers.{layer}.mlp."
        router = np.matmul(x.astype(np.float32), self.tensor(stem + "gate.weight").T.astype(np.float32))
        scores = sigmoid(router)
        choice = scores + self.tensor(stem + "gate.e_score_correction_bias").astype(np.float32)
        count = self.config["num_experts_per_tok"]
        indices = np.argpartition(choice, -count, axis=-1)[:, -count:]
        selected = np.take_along_axis(scores, indices, axis=-1)
        selected /= np.sum(selected, axis=-1, keepdims=True) + 1e-20
        selected *= self.config["routed_scaling_factor"]
        routed = np.zeros_like(x)
        for token in range(x.shape[0]):
            for slot, expert in enumerate(indices[token]):
                expert_stem = stem + f"experts.{int(expert)}."
                routed[token] = to_bf16(routed[token] + to_bf16(
                    self.mlp(x[token : token + 1], expert_stem)[0].astype(np.float32) * selected[token, slot]))
        shared = self.mlp(x, stem + "shared_experts.")
        return to_bf16(routed + shared)

    def forward(self, token_ids):
        embedding = self.tensor("model.language_model.embed_tokens.weight")[token_ids]
        embedding = to_bf16(embedding)
        self.check("model.language_model.embed_tokens", embedding[None])
        streams = np.repeat(embedding[:, None, :], self.hc, axis=1)
        self.check("hidden_states.0", streams[None])
        for layer, block_type in enumerate(self.config["layer_types"]):
            base = f"model.language_model.layers.{layer}"
            post, comb, collapsed = self.hyper(streams, base + ".hc_attn", base + ".attn_hc")
            normalized = self.rms_norm(collapsed, base + ".input_layernorm.weight")
            self.check(base + ".input_layernorm", normalized[None])
            attention = self.kda(normalized, layer) if block_type == "linear_attention" else self.mla(normalized, layer)
            self.check(base + (".self_attn" if block_type == "linear_attention" else ".self_attn.0"), attention[None])
            streams = self.mix(streams, attention, post, comb)

            post, comb, collapsed = self.hyper(streams, base + ".hc_ffn", base + ".ffn_hc")
            normalized = self.rms_norm(collapsed, base + ".post_attention_layernorm.weight")
            self.check(base + ".post_attention_layernorm", normalized[None])
            feed_forward = self.moe(normalized, layer) if self.config["mlp_layer_types"][layer] == "sparse" else self.mlp(normalized, base + ".mlp.")
            self.check(base + ".mlp", feed_forward[None])
            streams = self.mix(streams, feed_forward, post, comb)
            self.check(base + ".0", streams[None])
            if layer + 1 < self.config["num_hidden_layers"]:
                self.check(f"hidden_states.{layer + 1}", streams[None])

        hidden = to_bf16(np.mean(streams, axis=1))
        hidden = self.rms_norm(hidden, "model.language_model.norm.weight")
        self.check("model.language_model.norm", hidden[None])
        logits = self.linear(hidden, "lm_head.weight")
        self.check("lm_head", logits[None])
        self.check("logits", logits[None])
        top = np.argpartition(logits[-1], -8)[-8:]
        top = top[np.argsort(logits[-1, top])[::-1]]
        print("top8", " ".join(f"{int(index)}:{float(logits[-1, index]):.6f}" for index in top))
        print(f"worst={self.worst[0]} rel={self.worst[1]:.6f}")
        return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--oracle", type=pathlib.Path)
    parser.add_argument("--tokens", default="1,2,3,4")
    args = parser.parse_args()
    model = Glm53Numpy(args.model, args.oracle)
    model.forward(np.array([int(token) for token in args.tokens.split(",")], np.int64))


if __name__ == "__main__":
    main()
