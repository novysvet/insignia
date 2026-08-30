#!/usr/bin/env python3
"""Measure full-controller distortion from the AVX-VNNI INT8 policy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from falsifier_moe import FalsifierMoE, FalsifierMoEConfig, fake_dynamic_int8
from train_falsifier_moe import PromptShard, TraceChunkDataset, move_batch


EXPORTED_LINEAR_MODULES = {
    "encoder.logit.0", "encoder.hidden.0", "encoder.cache.0",
    "encoder.router_tail.0", "encoder.candidate.0", "encoder.candidate.2",
    "cell.depth.q", "cell.depth.k", "cell.depth.v", "cell.depth.out",
    "cell.mixer.dynamic", "cell.attention.q_proj", "cell.attention.kv_down",
    "cell.attention.kv_up", "cell.attention.output_gate",
    "cell.attention.out_proj", "cell.moe.router", "cell.moe.latent_down",
    "cell.moe.latent_up", "cell.moe.shared.gate_up", "cell.moe.shared.down",
    "immediate_head", "trajectory_head", "free_trajectory_head",
    "collapse_head", "gram_factor", "gram_diagonal", "action_risk",
    "action_cost", "acceptance_head",
}


def fake_weight_int8(weight: Tensor) -> tuple[Tensor, dict[str, float]]:
    value = weight.detach().float()
    scale = value.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    quantized = torch.round(value / scale).clamp(-127.0, 127.0)
    dequantized = quantized * scale
    error = dequantized - value
    value64 = value.double()
    dequantized64 = dequantized.double()
    denominator = (torch.linalg.vector_norm(value64)
                   * torch.linalg.vector_norm(dequantized64))
    cosine = ((value64 * dequantized64).sum()
              / denominator.clamp_min(1.0e-30)).item()
    return dequantized.to(weight.dtype), {
        "cosine": cosine,
        "mse": error.square().mean().item(),
        "max_abs": error.abs().max().item(),
    }


def prepare_quantized_model(model: FalsifierMoE) -> dict[str, object]:
    weight_metrics: dict[str, dict[str, float]] = {}
    activation_hooks = []
    for name, module in model.named_modules():
        if name not in EXPORTED_LINEAR_MODULES:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"expected Linear at {name}")
        dequantized, metrics = fake_weight_int8(module.weight)
        module.weight.data.copy_(dequantized)
        weight_metrics[name] = metrics
        # The absorbed MLA path consumes a float latent with a dequantized K/V
        # view, so it intentionally avoids a second activation quantization.
        if name != "cell.attention.kv_up":
            activation_hooks.append(module.register_forward_pre_hook(
                lambda _module, args: (fake_dynamic_int8(args[0]),)))

    for name in ("cell.moe.expert_gate_up", "cell.moe.expert_down"):
        parameter = dict(model.named_parameters())[name]
        dequantized, metrics = fake_weight_int8(parameter)
        parameter.data.copy_(dequantized)
        weight_metrics[name] = metrics
    model.cell.moe.fake_int8_expert_activations = True
    # Keep handles alive for the duration of evaluation.
    model._vnni_activation_hooks = activation_hooks  # type: ignore[attr-defined]
    return {
        "minimum_weight_cosine": min(item["cosine"] for item in weight_metrics.values()),
        "maximum_weight_mse": max(item["mse"] for item in weight_metrics.values()),
        "maximum_weight_abs_error": max(item["max_abs"] for item in weight_metrics.values()),
        "matrices": weight_metrics,
    }


@dataclass
class Metrics:
    count: int = 0
    squared_error: float = 0.0
    dot: float = 0.0
    reference_square: float = 0.0
    candidate_square: float = 0.0
    maximum_abs_error: float = 0.0
    decisions: int = 0
    matching_decisions: int = 0

    def add(self, reference: Tensor, candidate: Tensor, valid: Tensor) -> None:
        selected_reference = reference[valid].detach().float().reshape(-1, reference.shape[-1])
        selected_candidate = candidate[valid].detach().float().reshape(-1, candidate.shape[-1])
        if not selected_reference.numel():
            return
        difference = selected_candidate - selected_reference
        self.count += difference.numel()
        self.squared_error += difference.square().double().sum().item()
        self.dot += (selected_reference * selected_candidate).double().sum().item()
        self.reference_square += selected_reference.square().double().sum().item()
        self.candidate_square += selected_candidate.square().double().sum().item()
        self.maximum_abs_error = max(
            self.maximum_abs_error, difference.abs().max().item())
        if reference.shape[-1] > 1:
            self.decisions += selected_reference.shape[0]
            self.matching_decisions += (
                selected_reference.argmax(dim=-1) == selected_candidate.argmax(dim=-1)
            ).sum().item()

    def result(self) -> dict[str, float | int]:
        denominator = np.sqrt(self.reference_square * self.candidate_square)
        return {
            "elements": self.count,
            "mse": self.squared_error / max(1, self.count),
            "cosine": self.dot / max(1.0e-30, denominator),
            "max_abs_error": self.maximum_abs_error,
            "argmax_agreement": self.matching_decisions / max(1, self.decisions),
            "decisions": self.decisions,
        }


def expand(patterns: list[str]) -> list[Path]:
    paths = sorted({Path(value).resolve() for pattern in patterns
                    for value in glob.glob(pattern)})
    if not paths:
        raise SystemExit("no falsifier datasets matched")
    return paths


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = FalsifierMoEConfig(**checkpoint.get("config", {}))
    reference = FalsifierMoE(config)
    reference.load_state_dict(checkpoint["model"])
    candidate = FalsifierMoE(config)
    candidate.load_state_dict(checkpoint["model"])
    weight_metrics = prepare_quantized_model(candidate)
    device = torch.device(args.device)
    reference.to(device).eval()
    candidate.to(device).eval()

    shards = [PromptShard(path, config.trajectory_horizons) for path in expand(args.data)]
    dataset = TraceChunkDataset(shards, args.sequence_length, args.sequence_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    keys = (
        "hidden", "immediate", "forced_trajectory_hazard_logits",
        "forced_trajectory_peak", "free_trajectory_hazard_logits",
        "collapse_logits", "gram_upper", "action_risk", "action_cost",
        "acceptance_logits", "router_logits",
    )
    metrics = {key: Metrics() for key in keys}
    route_count = 0
    route_matches = 0
    route_events = 0
    route_set_matches = 0
    route_overlap = 0
    batches = 0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            reference_output = reference(batch["inputs"])
            candidate_output = candidate(batch["inputs"])
            valid = batch["inputs"]["valid"].bool()
            for key in keys:
                reference_value = reference_output[key]
                candidate_value = candidate_output[key]
                if key == "router_logits":
                    repeat_valid = valid.unsqueeze(0).expand(reference_value.shape[:3])
                    metrics[key].add(reference_value, candidate_value, repeat_valid)
                else:
                    metrics[key].add(reference_value, candidate_value, valid)
            reference_routes = reference_output["selected_experts"]
            candidate_routes = candidate_output["selected_experts"]
            route_valid = valid.unsqueeze(0).unsqueeze(-1).expand_as(reference_routes)
            route_count += route_valid.sum().item()
            route_matches += ((reference_routes == candidate_routes) & route_valid).sum().item()
            reference_set = reference_routes.sort(dim=-1).values
            candidate_set = candidate_routes.sort(dim=-1).values
            event_valid = valid.unsqueeze(0).expand(reference_routes.shape[:-1])
            route_events += event_valid.sum().item()
            route_set_matches += (
                (reference_set == candidate_set).all(dim=-1) & event_valid
            ).sum().item()
            per_event_overlap = (
                reference_routes.unsqueeze(-1) == candidate_routes.unsqueeze(-2)
            ).any(dim=-1).sum(dim=-1)
            route_overlap += per_event_overlap[event_valid].sum().item()
            batches += 1
            if args.max_batches and batches >= args.max_batches:
                break
    return {
        "schema": "insignia-falsifier-vnni-quality-v1",
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "datasets": [str(path) for path in expand(args.data)],
        "batches": batches,
        "weight_quantization": weight_metrics,
        "outputs": {key: value.result() for key, value in metrics.items()},
        "selected_expert_slot_agreement": route_matches / max(1, route_count),
        "selected_expert_slots": route_count,
        "selected_expert_set_agreement": route_set_matches / max(1, route_events),
        "selected_expert_mean_overlap": route_overlap / max(1, route_events),
        "selected_expert_events": route_events,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
