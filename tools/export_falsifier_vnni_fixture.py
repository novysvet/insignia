#!/usr/bin/env python3
"""Create a deterministic controller-core parity fixture for AVX-VNNI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch

from evaluate_falsifier_int8 import prepare_quantized_model
from export_falsifier_vnni import HEAD_PREFIXES, fnv1a
from falsifier_moe import (
    FalsifierIncrementalState,
    FalsifierMoE,
    FalsifierMoEConfig,
)


MAGIC = b"IFVFIX1\0"
HEADER = struct.Struct("<8sIIQ40s")
VERSION = 1


def export_fixture(checkpoint_path: Path | None, output: Path, events: int,
                   seed: int) -> dict[str, object]:
    if checkpoint_path is None:
        torch.manual_seed(53)
        config = FalsifierMoEConfig()
        model = FalsifierMoE(config)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False)
        config = FalsifierMoEConfig(**checkpoint.get("config", {}))
        if config != FalsifierMoEConfig():
            raise ValueError("native fixture requires the fixed v1 controller geometry")
        model = FalsifierMoE(config)
        model.load_state_dict(checkpoint["model"])
    quantization = prepare_quantized_model(model)
    model.eval()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    initial_streams = torch.randn(1, events, config.streams, config.d_model) * 0.25

    expected_hidden: list[torch.Tensor] = []
    expected_heads: list[torch.Tensor] = []
    expected_routes: list[torch.Tensor] = []
    state = FalsifierIncrementalState.empty(config.cell_repeats)
    with torch.inference_mode():
        for event in range(events):
            streams = initial_streams[:, event:event + 1].clone()
            history = [streams.mean(dim=-2)]
            next_latents: list[torch.Tensor] = []
            routes: list[torch.Tensor] = []
            for repeat in range(config.cell_repeats):
                streams, diagnostics, latent = model.cell.forward_incremental(
                    streams, history, state.latents[repeat], 336, False, False)
                history.append(streams.mean(dim=-2))
                next_latents.append(latent)
                routes.append(diagnostics["selected_experts"])
            hidden = model.final_norm(model.cell.depth(history))
            raw_heads = torch.cat(
                [getattr(model, name)(hidden) for name in HEAD_PREFIXES], dim=-1)
            expected_hidden.append(hidden.squeeze(0).squeeze(0).float().cpu())
            expected_heads.append(raw_heads.squeeze(0).squeeze(0).float().cpu())
            expected_routes.append(torch.stack(routes).squeeze(1).squeeze(1).int().cpu())
            state = FalsifierIncrementalState(next_latents)

    arrays = (
        initial_streams.squeeze(0).float().cpu().contiguous().numpy().astype("<f4"),
        torch.stack(expected_hidden).numpy().astype("<f4"),
        torch.stack(expected_heads).numpy().astype("<f4"),
        torch.stack(expected_routes).numpy().astype("<i4"),
    )
    payload = b"".join(array.tobytes(order="C") for array in arrays)
    checksum = fnv1a(payload)
    header = HEADER.pack(MAGIC, VERSION, events, checksum, bytes(40))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(header + payload)
    return {
        "schema": "insignia-falsifier-vnni-fixture-v1",
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "output": str(output),
        "events": events,
        "seed": seed,
        "file_bytes": output.stat().st_size,
        "file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "payload_checksum": checksum,
        "minimum_weight_cosine": quantization["minimum_weight_cosine"],
        "maximum_weight_mse": quantization["maximum_weight_mse"],
        "tensor_order": ["initial_streams", "hidden", "raw_heads", "routes"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--events", type=int, default=42)
    parser.add_argument("--seed", type=int, default=5303)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.events <= 336:
        raise SystemExit("--events must be in [1, 336]")
    report = export_fixture(args.checkpoint, args.output, args.events, args.seed)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
