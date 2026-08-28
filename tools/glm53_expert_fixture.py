#!/usr/bin/env python3
"""Extract one real GLM-5.3 NVFP4 expert and build kernel test fixtures.

The source checkpoint uses NVIDIA ModelOpt NVFP4:
  U8 packed E2M1 weights, one E4M3 scale per 16 values, and one F32 global
  multiplier.  The fixture also carries two per-64 integer candidates:
  symmetric INT4 and re-fitted E2M1 codes, both with FP16 scales.
"""

import argparse
import json
import math
import pathlib
import struct

import numpy as np


E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
E2M1_CUTS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)


class Checkpoint:
    def __init__(self, root: pathlib.Path):
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self.headers = {}

    def _header(self, shard: pathlib.Path):
        if shard not in self.headers:
            with shard.open("rb") as f:
                n = int.from_bytes(f.read(8), "little")
                self.headers[shard] = (json.loads(f.read(n)), 8 + n)
        return self.headers[shard]

    def raw(self, name: str):
        shard = self.root / self.weight_map[name]
        header, data_start = self._header(shard)
        meta = header[name]
        begin, end = meta["data_offsets"]
        with shard.open("rb") as f:
            f.seek(data_start + begin)
            raw = f.read(end - begin)
        if len(raw) != end - begin:
            raise IOError(f"short read for {name}")
        return raw, meta


def decode_e4m3(raw: bytes, shape):
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


def decode_nvfp4(packed_raw, packed_shape, scale_raw, scale_shape, global_scale):
    packed = np.frombuffer(packed_raw, np.uint8).reshape(packed_shape)
    rows, half_cols = packed.shape
    codes = np.empty((rows, half_cols * 2), np.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    signs = np.where(codes & 8, -1.0, 1.0).astype(np.float32)
    values = E2M1[codes & 7] * signs
    block_scales = decode_e4m3(scale_raw, scale_shape)
    if block_scales.shape != (rows, values.shape[1] // 16):
        raise ValueError(f"unexpected block-scale shape {block_scales.shape}")
    return values * np.repeat(block_scales, 16, axis=1) * np.float32(global_scale)


def pack_nibbles(codes):
    if codes.shape[1] & 1:
        raise ValueError("odd input dimension")
    return ((codes[:, 0::2] & 15) | ((codes[:, 1::2] & 15) << 4)).astype(np.uint8)


def quant_uniform(w, group, bits):
    rows, cols = w.shape
    if cols % group:
        raise ValueError("group must divide the input dimension")
    qmax = (1 << (bits - 1)) - 1
    wr = w.reshape(rows, cols // group, group)
    scales = np.maximum(np.max(np.abs(wr), axis=2) / qmax, 2.0**-24).astype(np.float32)
    q = np.clip(np.rint(wr / scales[:, :, None]), -qmax, qmax).astype(np.int8)
    return q.reshape(rows, cols), scales


def quant_e2m1(w, group, iterations=4):
    rows, cols = w.shape
    wr = w.reshape(-1, group)
    scales = np.maximum(np.max(np.abs(wr), axis=1) / 6.0, 2.0**-24).astype(np.float32)
    codes = np.empty(wr.shape, np.uint8)
    signed_levels = np.empty(wr.shape, np.float32)
    for _ in range(iterations):
        normalized = wr / scales[:, None]
        magnitude_codes = np.zeros(wr.shape, np.uint8)
        magnitude = np.abs(normalized)
        for cut in E2M1_CUTS:
            magnitude_codes += magnitude > cut
        codes[:] = magnitude_codes | ((normalized < 0).astype(np.uint8) << 3)
        signed_levels[:] = E2M1[magnitude_codes] * np.where(normalized < 0, -1.0, 1.0)
        denominator = np.sum(signed_levels * signed_levels, axis=1)
        fitted = np.sum(wr * signed_levels, axis=1) / np.maximum(denominator, 2.0**-24)
        scales = np.where(np.isfinite(fitted) & (fitted > 0), fitted, scales)
    return codes.reshape(rows, cols), scales.reshape(rows, cols // group)


def quality(label, reference, candidate, x):
    err = candidate - reference
    sqnr = 10.0 * math.log10(float(np.sum(reference * reference)) / max(float(np.sum(err * err)), 1e-30))
    cosine = float(np.vdot(reference.ravel(), candidate.ravel()) / (
        np.linalg.norm(reference.ravel()) * np.linalg.norm(candidate.ravel())))
    yr = reference @ x
    yc = candidate @ x
    ycos = float(np.vdot(yr, yc) / (np.linalg.norm(yr) * np.linalg.norm(yc)))
    rel = float(np.linalg.norm(yc - yr) / np.linalg.norm(yr))
    print(f"{label:14s} weight_sqnr={sqnr:7.3f}dB weight_cos={cosine:.9f} y_cos={ycos:.9f} y_rel={rel:.6f}")
    return yc.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--tensor", default="model.language_model.layers.3.mlp.experts.0.gate_proj.weight")
    parser.add_argument("--group", type=int, default=64)
    args = parser.parse_args()

    checkpoint = Checkpoint(args.checkpoint)
    base = args.tensor.removesuffix(".weight")
    packed_raw, packed_meta = checkpoint.raw(args.tensor)
    scale_raw, scale_meta = checkpoint.raw(base + ".weight_scale")
    global_raw, global_meta = checkpoint.raw(base + ".weight_scale_2")
    if packed_meta["dtype"] != "U8" or scale_meta["dtype"] != "F8_E4M3" or global_meta["dtype"] != "F32":
        raise ValueError("tensor is not ModelOpt NVFP4")
    global_scale = struct.unpack("<f", global_raw)[0]
    if not 0.0 < global_scale < 1.0:
        raise ValueError(f"unexpected ModelOpt global scale {global_scale}")

    w = decode_nvfp4(packed_raw, packed_meta["shape"], scale_raw, scale_meta["shape"], global_scale)
    rows, cols = w.shape
    rng = np.random.default_rng(0x53F1A5)
    x = rng.standard_normal(cols).astype(np.float32)
    x /= np.sqrt(np.mean(x * x))
    y_nv = (w @ x).astype(np.float32)

    print(f"tensor={args.tensor} shape={w.shape} global_scale={global_scale:.9g}")
    for group in (32, 64, 128, 256):
        q, s = quant_uniform(w, group, 4)
        dq = q.astype(np.float32).reshape(rows, cols // group, group) * s[:, :, None]
        quality(f"int4-g{group}", w, dq.reshape(rows, cols), x)
    q3, s3 = quant_uniform(w, 64, 3)
    dq3 = q3.astype(np.float32).reshape(rows, cols // 64, 64) * s3[:, :, None]
    quality("int3-g64", w, dq3.reshape(rows, cols), x)

    q4, s4 = quant_uniform(w, args.group, 4)
    w4 = q4.astype(np.float32).reshape(rows, cols // args.group, args.group) * s4[:, :, None]
    w4 = w4.reshape(rows, cols)
    y_i4 = quality(f"fixture-i4-g{args.group}", w, w4, x)

    qe2, se2 = quant_e2m1(w, args.group)
    signs = np.where(qe2 & 8, -1.0, 1.0).astype(np.float32)
    we2 = E2M1[qe2 & 7] * signs
    we2 *= np.repeat(se2, args.group, axis=1)
    y_e2 = quality(f"fixture-e2-g{args.group}", w, we2, x)

    i4_packed = pack_nibbles(q4.astype(np.uint8) & 15)
    e2_packed = pack_nibbles(qe2)
    s4 = s4.astype("<f2")
    se2 = se2.astype("<f2")
    header = struct.pack(
        "<8sIIIIfI6Q",
        b"IG53X001", 1, rows, cols, args.group, global_scale, 0,
        len(packed_raw), len(scale_raw), i4_packed.nbytes, s4.nbytes,
        e2_packed.nbytes, se2.nbytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        f.write(header)
        f.write(packed_raw)
        f.write(scale_raw)
        f.write(i4_packed.tobytes())
        f.write(s4.tobytes())
        f.write(e2_packed.tobytes())
        f.write(se2.tobytes())
        f.write(x.astype("<f4").tobytes())
        f.write(y_nv.astype("<f4").tobytes())
        f.write(y_i4.astype("<f4").tobytes())
        f.write(y_e2.astype("<f4").tobytes())
    print(f"wrote {args.output} ({args.output.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
