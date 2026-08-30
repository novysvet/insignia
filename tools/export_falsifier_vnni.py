#!/usr/bin/env python3
"""Export Falsifier-MoE matrices for the fixed Raptor Lake AVX-VNNI runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Iterable

import numpy as np
import torch

from falsifier_moe import FalsifierMoE, FalsifierMoEConfig


MAGIC = b"IFVNNI1\0"
FILE_HEADER = struct.Struct("<8sIIQ40s")
MATRIX_HEADER = struct.Struct("<32sIIIIQQ")
MASK64 = (1 << 64) - 1


def fnv1a(payload: bytes) -> int:
    value = 14695981039346656037
    for byte in payload:
        value ^= byte
        value = (value * 1099511628211) & MASK64
    return value


def rotate_left_one(value: int) -> int:
    return ((value << 1) | (value >> 63)) & MASK64


def load_state(checkpoint: Path | None) -> dict[str, torch.Tensor]:
    if checkpoint is None:
        torch.manual_seed(53)
        return FalsifierMoE(FalsifierMoEConfig()).state_dict()
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = loaded.get("model", loaded) if isinstance(loaded, dict) else loaded
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a model state dictionary")
    return state


def tensor(state: dict[str, torch.Tensor], name: str) -> np.ndarray:
    if name not in state:
        raise KeyError(f"checkpoint is missing {name}")
    return state[name].detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)


def matrix_sources(state: dict[str, torch.Tensor]
                   ) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    entries = (
        ("encoder.logit", "encoder.logit.0"),
        ("encoder.hidden", "encoder.hidden.0"),
        ("encoder.cache", "encoder.cache.0"),
        ("encoder.router_tail", "encoder.router_tail.0"),
        ("encoder.candidate_a", "encoder.candidate.0"),
        ("encoder.candidate_b", "encoder.candidate.2"),
        ("cell.depth_q", "cell.depth.q"),
        ("cell.depth_k", "cell.depth.k"),
        ("cell.depth_v", "cell.depth.v"),
        ("cell.depth_out", "cell.depth.out"),
        ("cell.mhc_dynamic", "cell.mixer.dynamic"),
        ("cell.mla_q", "cell.attention.q_proj"),
        ("cell.mla_kv_down", "cell.attention.kv_down"),
        ("cell.mla_kv_up", "cell.attention.kv_up"),
        ("cell.mla_gate", "cell.attention.output_gate"),
        ("cell.mla_out", "cell.attention.out_proj"),
        ("cell.moe_router", "cell.moe.router"),
        ("cell.latent_down", "cell.moe.latent_down"),
        ("cell.latent_up", "cell.moe.latent_up"),
        ("cell.shared_gate_up", "cell.moe.shared.gate_up"),
        ("cell.shared_down", "cell.moe.shared.down"),
    )
    for export_name, state_prefix in entries:
        weight = tensor(state, state_prefix + ".weight")
        bias_name = state_prefix + ".bias"
        bias = (tensor(state, bias_name) if bias_name in state
                else np.zeros(weight.shape[0], dtype="<f4"))
        yield export_name, weight, bias

    for export_name, state_name in (
        ("cell.expert_gate_up", "cell.moe.expert_gate_up"),
        ("cell.expert_down", "cell.moe.expert_down"),
    ):
        weight = tensor(state, state_name)
        yield export_name, weight, np.zeros(weight.shape[0], dtype="<f4")

    head_prefixes = (
        "immediate_head",
        "trajectory_head",
        "free_trajectory_head",
        "collapse_head",
        "gram_factor",
        "gram_diagonal",
        "action_risk",
        "action_cost",
        "acceptance_head",
    )
    head_weight = np.concatenate(
        [tensor(state, name + ".weight") for name in head_prefixes], axis=0)
    head_bias = np.concatenate(
        [tensor(state, name + ".bias") for name in head_prefixes], axis=0)
    yield "heads", head_weight, head_bias


def encode_matrix(name: str, weight: np.ndarray, bias: np.ndarray
                  ) -> tuple[bytes, dict[str, object]]:
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise ValueError(f"invalid matrix shape for {name}: {weight.shape}, {bias.shape}")
    rows, logical_cols = weight.shape
    padded_cols = (logical_cols + 31) & ~31
    absmax = np.max(np.abs(weight), axis=1)
    scale = np.where(absmax > 0.0, absmax / 127.0, 1.0).astype("<f4")
    quantized_logical = np.clip(
        np.rint(weight / scale[:, None]), -127, 127).astype(np.int8)
    quantized = np.zeros((rows, padded_cols), dtype=np.int8)
    quantized[:, :logical_cols] = quantized_logical
    correction = (quantized.astype(np.int32).sum(axis=1) * 128).astype("<i4")
    dequantized = quantized_logical.astype(np.float32) * scale[:, None]
    error = dequantized - weight
    dot = float(np.sum(dequantized.astype(np.float64) * weight.astype(np.float64)))
    norm = float(np.sqrt(
        np.sum(dequantized.astype(np.float64) ** 2)
        * np.sum(weight.astype(np.float64) ** 2)))
    payload = b"".join((
        quantized.tobytes(order="C"), correction.tobytes(order="C"),
        scale.tobytes(order="C"), bias.astype("<f4", copy=False).tobytes(order="C"),
    ))
    checksum = fnv1a(payload)
    header = MATRIX_HEADER.pack(
        name.encode("ascii").ljust(32, b"\0"), rows, logical_cols,
        padded_cols, 0, len(payload), checksum)
    padding = b"\0" * ((-len(payload)) & 63)
    return header + payload + padding, {
        "name": name,
        "rows": rows,
        "logical_cols": logical_cols,
        "padded_cols": padded_cols,
        "payload_bytes": len(payload),
        "payload_checksum": checksum,
        "weight_cosine": dot / norm if norm else 1.0,
        "weight_mse": float(np.mean(error.astype(np.float64) ** 2)),
        "weight_max_abs_error": float(np.max(np.abs(error))),
    }


def export(checkpoint: Path | None, output: Path) -> dict[str, object]:
    state = load_state(checkpoint)
    encoded: list[bytes] = []
    matrices: list[dict[str, object]] = []
    manifest = 0
    for name, weight, bias in matrix_sources(state):
        blob, metadata = encode_matrix(name, weight, bias)
        encoded.append(blob)
        matrices.append(metadata)
        manifest = rotate_left_one(manifest) ^ int(metadata["payload_checksum"])
    header = FILE_HEADER.pack(MAGIC, 1, len(encoded), manifest, bytes(40))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        file.write(header)
        for blob in encoded:
            file.write(blob)
    file_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {
        "schema": "insignia-falsifier-vnni-export-v1",
        "checkpoint": str(checkpoint) if checkpoint else None,
        "output": str(output),
        "file_bytes": output.stat().st_size,
        "file_sha256": file_sha256,
        "manifest_checksum": manifest,
        "matrix_count": len(matrices),
        "minimum_weight_cosine": min(item["weight_cosine"] for item in matrices),
        "maximum_weight_mse": max(item["weight_mse"] for item in matrices),
        "maximum_weight_abs_error": max(
            item["weight_max_abs_error"] for item in matrices),
        "matrices": matrices,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export(args.checkpoint, args.output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
