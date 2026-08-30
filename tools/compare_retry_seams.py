#!/usr/bin/env python3
"""Locate the first layer/phase where exact retry diverges from one-pass exact.

Ordinary prefill seams use tags 11..16; an exact second pass after rollback
uses 21..26.  A replay pass is paired with the preceding ordinary logical
block, then compared with the same block in the one-pass exact run.
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
    replay: bool
    records: dict[tuple[int, int], np.ndarray]


def load(path: Path) -> list[Pass]:
    passes: list[Pass] = []
    current: Pass | None = None
    maximum_layer = -1
    with path.open("rb") as source:
        while header := source.read(HEADER.size):
            if len(header) != HEADER.size:
                raise SystemExit(f"{path}: partial seam header")
            _, layer, tag, count = HEADER.unpack(header)
            payload = source.read(count * 4)
            if len(payload) != count * 4:
                raise SystemExit(f"{path}: partial seam payload")
            if not (11 <= tag <= 16 or 21 <= tag <= 26):
                continue
            replay = tag >= 21
            phase = tag - 10 if replay else tag
            new_pass = (current is None or current.replay != replay or
                        (phase == 11 and layer <= maximum_layer))
            if new_pass:
                current = Pass(replay, {})
                passes.append(current)
                maximum_layer = -1
            key = (layer, phase)
            if key in current.records:
                raise SystemExit(f"{path}: duplicate seam layer {layer}/phase {phase}")
            current.records[key] = np.frombuffer(payload, dtype="<f4").copy()
            maximum_layer = max(maximum_layer, layer)
    for item in passes:
        phases = {phase for _, phase in item.records}
        if phases != set(range(11, 17)):
            raise SystemExit(f"{path}: incomplete verifier seam phases {sorted(phases)}")
    return passes


def compare(reference: Pass, retry: Pass, block: int) -> None:
    names = {
        11: "attention input norm", 12: "attention output",
        13: "post-attention streams", 14: "FFN input norm",
        15: "FFN output", 16: "post-FFN streams",
    }
    if reference.records.keys() != retry.records.keys():
        raise SystemExit(f"block {block}: seam key mismatch")
    for layer, phase in sorted(reference.records):
        exact = reference.records[(layer, phase)]
        replay = retry.records[(layer, phase)]
        if exact.shape != replay.shape:
            raise SystemExit(f"block {block} layer {layer} phase {phase}: shape mismatch")
        if np.array_equal(exact, replay):
            continue
        exact64 = exact.astype(np.float64)
        replay64 = replay.astype(np.float64)
        delta = replay64 - exact64
        denominator = math.sqrt(float(np.dot(exact64, exact64)) *
                                float(np.dot(replay64, replay64)))
        cosine = float(np.dot(exact64, replay64) / denominator) if denominator else 1.0
        print(f"block={block} first_divergent_layer={layer} phase={phase} "
              f"({names[phase]}) "
              f"max_abs={float(np.max(np.abs(delta))):.9g} "
              f"rms={math.sqrt(float(np.mean(delta * delta))):.9g} "
              f"cos={cosine:.9f}")
        return
    print(f"block={block} retry is bit-identical through every recorded seam")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exact", type=Path)
    parser.add_argument("retry", type=Path)
    args = parser.parse_args()
    exact = [item for item in load(args.exact) if not item.replay]
    retry_passes = load(args.retry)
    logical = -1
    compared = 0
    for item in retry_passes:
        if not item.replay:
            logical += 1
        else:
            if logical < 0 or logical >= len(exact):
                raise SystemExit("retry pass has no matching exact logical block")
            compare(exact[logical], item, logical)
            compared += 1
    if not compared:
        raise SystemExit("retry trace contains no exact replay seams")


if __name__ == "__main__":
    main()
