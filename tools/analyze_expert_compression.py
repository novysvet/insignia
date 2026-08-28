#!/usr/bin/env python3
"""Measure lossless compression headroom in real GLM NVFP4 expert records."""

import argparse
import collections
import json
import math
import pathlib
import shutil
import statistics
import subprocess
import time
import zlib

from glm53_expert_fixture import Checkpoint


PROJECTIONS = ("down_proj", "gate_proj", "up_proj")


def entropy(counts):
    total = sum(counts)
    return -sum((n / total) * math.log2(n / total) for n in counts if n)


def cli_codec(name, encode, decode):
    if not shutil.which(encode[0]):
        return None

    def run(raw, repeat):
        packed = subprocess.run(encode, input=raw, stdout=subprocess.PIPE, check=True).stdout
        samples = []
        restored = b""
        for _ in range(repeat):
            begin = time.perf_counter()
            restored = subprocess.run(
                decode, input=packed, stdout=subprocess.PIPE, check=True
            ).stdout
            samples.append(time.perf_counter() - begin)
        if restored != raw:
            raise RuntimeError(f"{name} round trip failed")
        return len(packed) / len(raw), len(raw) / statistics.median(samples) / 2**30

    return name, run


def zlib_codec(raw, repeat):
    packed = zlib.compress(raw, 1)
    samples = []
    restored = b""
    for _ in range(repeat):
        begin = time.perf_counter()
        restored = zlib.decompress(packed)
        samples.append(time.perf_counter() - begin)
    if restored != raw:
        raise RuntimeError("zlib round trip failed")
    return len(packed) / len(raw), len(raw) / statistics.median(samples) / 2**30


def read_expert(checkpoint, layer, expert):
    stem = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
    record = bytearray()
    bodies = bytearray()
    scales = bytearray()
    for projection in PROJECTIONS:
        body, _ = checkpoint.raw(stem + projection + ".weight")
        scale, _ = checkpoint.raw(stem + projection + ".weight_scale")
        global_scale, _ = checkpoint.raw(stem + projection + ".weight_scale_2")
        record += body + scale + global_scale
        bodies += body
        scales += scale
    return bytes(record), bodies, scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 1 or args.repeat < 1:
        parser.error("samples and repeat must be positive")

    config = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
    sparse = [
        layer for layer, kind in enumerate(config["mlp_layer_types"])
        if kind == "sparse"
    ]
    experts = int(config["n_routed_experts"])
    checkpoint = Checkpoint(args.checkpoint)
    codecs = [("zlib-1", zlib_codec)]
    for codec in (
        cli_codec("zstd-fast1", ["zstd", "--fast=1", "-q", "-c"],
                  ["zstd", "-d", "-q", "-c"]),
        cli_codec("zstd-1", ["zstd", "-1", "-q", "-c"],
                  ["zstd", "-d", "-q", "-c"]),
        cli_codec("lz4-1", ["lz4", "-1", "-q", "-c"],
                  ["lz4", "-d", "-q", "-c"]),
    ):
        if codec:
            codecs.append(codec)

    body_counts = [0] * 16
    scale_counts = [0] * 256
    results = {name: [] for name, _ in codecs}
    body_results = {name: [] for name, _ in codecs}
    scale_results = {name: [] for name, _ in codecs}
    for sample in range(args.samples):
        at = 0 if args.samples == 1 else round(sample * (len(sparse) - 1) / (args.samples - 1))
        layer = sparse[at]
        expert = (17 + 73 * sample) % experts
        raw, bodies, scales = read_expert(checkpoint, layer, expert)
        for byte, count in collections.Counter(scales).items():
            scale_counts[byte] += count
        for byte in bodies:
            body_counts[byte & 15] += 1
            body_counts[byte >> 4] += 1
        fields = []
        for name, codec in codecs:
            fraction, gib_s = codec(raw, args.repeat)
            results[name].append((fraction, gib_s))
            body_results[name].append(codec(bytes(bodies), args.repeat))
            scale_results[name].append(codec(bytes(scales), args.repeat))
            fields.append(f"{name}={fraction:.3f}x/{gib_s:.2f}GiB/s")
        print(f"L{layer:02d} E{expert:03d} {len(raw) / 2**20:.2f}MiB " + " ".join(fields))

    print(f"body nibble entropy: {entropy(body_counts):.4f}/4 bits")
    print(f"scale byte entropy:  {entropy(scale_counts):.4f}/8 bits")
    used_scales = [(count, code) for code, count in enumerate(scale_counts) if count]
    used_scales.sort(reverse=True)
    total_scales = sum(scale_counts)
    print(f"scale alphabet: {len(used_scales)} codes, fixed-width "
          f"{math.ceil(math.log2(max(1, len(used_scales))))} bits")
    print("scale top codes: " + " ".join(
        f"0x{code:02x}:{100 * count / total_scales:.2f}%"
        for count, code in used_scales[:16]
    ))
    for name, samples in results.items():
        fractions = [item[0] for item in samples]
        speeds = [item[1] for item in samples]
        fraction = statistics.median(fractions)
        print(
            f"{name:10s} median stored={fraction:.3f}x "
            f"effective-NVMe={1 / fraction:.3f}x decode={statistics.median(speeds):.2f}GiB/s"
        )
        print(
            f"{'':10s} body={statistics.median(item[0] for item in body_results[name]):.3f}x "
            f"scales={statistics.median(item[0] for item in scale_results[name]):.3f}x"
        )


if __name__ == "__main__":
    main()
