#!/usr/bin/env python3
"""CUDA compute ceiling for full-sequence vs deployable incremental Falsifier-MoE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from falsifier_moe import FalsifierMoE, FalsifierMoEConfig
from train_falsifier_moe import PromptShard


def tensor_batch(shard: PromptShard, indices: np.ndarray, device: torch.device
                 ) -> dict[str, torch.Tensor]:
    result = {
        name: torch.from_numpy(value[indices]).unsqueeze(0).to(device)
        for name, value in shard.inputs.items()
    }
    result["valid"] = torch.ones(1, len(indices), device=device)
    return result


def first_round_groups(shard: PromptShard) -> list[np.ndarray]:
    meta = shard.inputs["event_meta"]
    first_epoch = int(meta[0, 0])
    round_indices = np.flatnonzero(meta[:, 0] == first_epoch)
    groups: list[np.ndarray] = []
    for layer in np.unique(meta[round_indices, 1]):
        group = round_indices[meta[round_indices, 1] == layer]
        groups.append(group)
    if not groups or sum(map(len, groups)) != len(round_indices):
        raise RuntimeError("failed to partition first round into layer groups")
    return groups


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    config = FalsifierMoEConfig()
    shard = PromptShard(args.data, config.trajectory_horizons)
    groups = first_round_groups(shard)
    all_indices = np.concatenate(groups)
    group_batches = [tensor_batch(shard, group, device) for group in groups]
    full_batch = tensor_batch(shard, all_indices, device)
    model = FalsifierMoE(config).to(device).eval()

    def incremental_once() -> torch.Tensor:
        state = None
        outputs = []
        for batch in group_batches:
            output, state = model.forward_incremental(
                batch, state, max_history=args.max_history)
            outputs.append(output["hidden"])
        return torch.cat(outputs, dim=1)

    def full_once() -> torch.Tensor:
        return model(full_batch)["hidden"]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(args.warmup):
            incremental_once()
            full_once()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    def measure(function) -> tuple[float, torch.Tensor]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        result = None
        start.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(args.iterations):
                result = function()
        end.record()
        end.synchronize()
        assert result is not None
        return start.elapsed_time(end) / args.iterations, result

    incremental_ms, incremental = measure(incremental_once)
    full_ms, full = measure(full_once)
    difference = incremental.float() - full.float()
    cosine = torch.nn.functional.cosine_similarity(
        incremental.float().flatten(), full.float().flatten(), dim=0)
    result = {
        "schema": "insignia-falsifier-moe-benchmark-v1",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "data": str(args.data),
        "events": int(len(all_indices)),
        "layer_groups": len(groups),
        "rows_per_group": [int(len(group)) for group in groups],
        "iterations": args.iterations,
        "precision": "bf16-autocast",
        "incremental_round_ms": incremental_ms,
        "incremental_layer_group_ms": incremental_ms / len(groups),
        "full_sequence_ms": full_ms,
        "eager_incremental_over_full": incremental_ms / full_ms,
        "incremental_full_max_abs": float(difference.abs().max()),
        "incremental_full_mse": float(difference.square().mean()),
        "incremental_full_cosine": float(cosine),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "parameter_ledger": model.parameter_ledger(),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=336)
    parser.add_argument("--seed", type=int, default=53)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

