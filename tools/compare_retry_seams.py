#!/usr/bin/env python3
"""Locate the first layer where an exact retry diverges from one-pass exact.

Prefill seam tag 16 is an ordinary verifier pass and tag 17 is the exact
second pass after rollback.  A tag-17 pass is paired with the most recent
tag-16 logical block, then compared with the same block in the exact run.
"""

from __future__ import annotations

import argparse
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HEADER = struct.Struct("<4i")


@dataclass
class Pass:
    tag: int
    layers: list[np.ndarray]


def load(path: Path) -> list[Pass]:
    records: list[tuple[int, int, np.ndarray]] = []
    with path.open("rb") as source:
        while header := source.read(HEADER.size):
            if len(header) != HEADER.size:
                raise SystemExit(f"{path}: partial seam header")
            _, layer, tag, count = HEADER.unpack(header)
            payload = source.read(count * 4)
            if len(payload) != count * 4:
                raise SystemExit(f"{path}: partial seam payload")
            records.append((layer, tag, np.frombuffer(payload, dtype="<f4").copy()))
    passes: list[Pass] = []
    for layer, tag, payload in records:
        if layer == 0:
            passes.append(Pass(tag, []))
        if not passes or passes[-1].tag != tag or layer != len(passes[-1].layers):
            raise SystemExit(f"{path}: non-contiguous layer sequence at layer {layer}/tag {tag}")
        passes[-1].layers.append(payload)
    if any(len(item.layers) != 45 for item in passes):
        raise SystemExit(f"{path}: incomplete 45-layer verifier pass")
    return passes


def compare(reference: Pass, retry: Pass, block: int) -> None:
    for layer, (exact, replay) in enumerate(zip(reference.layers, retry.layers, strict=True)):
        if exact.shape != replay.shape:
            raise SystemExit(f"block {block} layer {layer}: shape mismatch")
        if np.array_equal(exact, replay):
            continue
        delta = replay.astype(np.float64) - exact.astype(np.float64)
        denominator = math.sqrt(float(np.dot(exact, exact)) * float(np.dot(replay, replay)))
        cosine = float(np.dot(exact, replay) / denominator) if denominator else 1.0
        print(f"block={block} first_divergent_layer={layer} "
              f"max_abs={float(np.max(np.abs(delta))):.9g} "
              f"rms={math.sqrt(float(np.mean(delta * delta))):.9g} "
              f"cos={cosine:.9f}")
        return
    print(f"block={block} retry is bit-identical through layer 44")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exact", type=Path)
    parser.add_argument("retry", type=Path)
    args = parser.parse_args()
    exact = [item for item in load(args.exact) if item.tag == 16]
    retry_passes = load(args.retry)
    logical = -1
    compared = 0
    for item in retry_passes:
        if item.tag == 16:
            logical += 1
        elif item.tag == 17:
            if logical < 0 or logical >= len(exact):
                raise SystemExit("retry pass has no matching exact logical block")
            compare(exact[logical], item, logical)
            compared += 1
        else:
            raise SystemExit(f"unexpected prefill seam tag {item.tag}")
    if not compared:
        raise SystemExit("retry trace contains no tag-17 exact replay")


if __name__ == "__main__":
    main()
