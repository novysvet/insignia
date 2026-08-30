#!/usr/bin/env python3
"""Prompt-held-out training harness for Falsifier-MoE v1.

The harness is intentionally strict about provenance:

* feature-only traces supervise downstream row/trajectory heads;
* exact teacher traces supervise contribution-Gram geometry;
* free-trajectory heads remain masked unless a future corpus explicitly
  supplies aligned autoregressive labels;
* prompt IDs, never rows, define train/validation membership.

The current seven-prompt corpus is suitable only for ``--smoke``.  A normal
run refuses to start below the configured on-policy row floor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from falsifier_moe import FalsifierMoE, FalsifierMoEConfig


SUPPORTED_SCHEMAS = {
    "insignia-falsifier-dataset-v2",
    "insignia-falsifier-dataset-v3",
}
ROW_LABELS = {
    "mse": 0,
    "cosine": 2,
    "kl": 4,
    "js": 5,
    "top1": 6,
}


def decode_residency(masks: np.ndarray, candidate_k: int = 32) -> np.ndarray:
    """Decode four uint32 candidate masks to [events,candidates,tiers]."""
    shifts = np.arange(candidate_k, dtype=np.uint32)
    decoded = ((masks.astype(np.uint32)[..., None] >> shifts) & 1).astype(np.float32)
    return np.swapaxes(decoded, -1, -2)


def row_risk(labels: np.ndarray) -> np.ndarray:
    """Monotone scale used only for multi-horizon peak supervision."""
    return (
        np.log1p(np.maximum(labels[:, ROW_LABELS["mse"]], 0.0))
        + 4.0 * np.maximum(1.0 - labels[:, ROW_LABELS["cosine"]], 0.0)
        + 0.10 * np.log1p(100.0 * np.maximum(labels[:, ROW_LABELS["kl"]], 0.0))
        + 0.10 * np.log1p(100.0 * np.maximum(labels[:, ROW_LABELS["js"]], 0.0))
        + 2.0 * labels[:, ROW_LABELS["top1"]]
    ).astype(np.float32)


def trajectory_targets(labels: np.ndarray, horizons: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    risk = row_risk(labels)
    top1 = labels[:, ROW_LABELS["top1"]] > 0.5
    hazard = np.zeros((len(labels), len(horizons)), dtype=np.float32)
    peak = np.zeros_like(hazard)
    for row in range(len(labels)):
        for column, horizon in enumerate(horizons):
            stop = min(len(labels), row + horizon)
            hazard[row, column] = float(np.any(top1[row:stop]))
            peak[row, column] = float(np.max(risk[row:stop]))
    return hazard, peak


def immediate_targets(labels: np.ndarray) -> np.ndarray:
    return np.stack((
        np.log1p(np.maximum(labels[:, ROW_LABELS["mse"]], 0.0)),
        np.maximum(1.0 - labels[:, ROW_LABELS["cosine"]], 0.0),
        np.log1p(100.0 * np.maximum(labels[:, ROW_LABELS["kl"]], 0.0)),
        np.log1p(100.0 * np.maximum(labels[:, ROW_LABELS["js"]], 0.0)),
        labels[:, ROW_LABELS["top1"]],
    ), axis=-1).astype(np.float32)


@dataclass
class CorpusSummary:
    prompts: int
    trajectories: int
    on_policy_rows: int
    on_policy_events: int
    exact_gram_events: int
    top1_failures: int
    mean_routed_activations_per_expert: float

    def to_dict(self) -> dict[str, int | float]:
        return self.__dict__.copy()


class PromptShard:
    def __init__(self, path: Path, horizons: tuple[int, ...]):
        self.path = path
        with np.load(path, allow_pickle=False) as archive:
            self.metadata = json.loads(str(archive["metadata"].item()))
            if self.metadata.get("schema") not in SUPPORTED_SCHEMAS:
                raise ValueError(
                    f"{path}: expected one of {sorted(SUPPORTED_SCHEMAS)}")
            if int(self.metadata["geometry"]["candidate_k"]) != 32:
                raise ValueError(f"{path}: unsupported candidate geometry")
            event_label_mask = (np.asarray(archive["event_label_mask"], dtype=np.bool_)
                                if "event_label_mask" in archive.files else
                                np.ones(len(archive["event_meta"]), dtype=np.bool_))
            # The first three exact-teacher v2 files predate the explicit
            # metadata field; their all-true event mask is authoritative.
            self.gram_present = bool(self.metadata.get(
                "gram_present", bool(np.all(event_label_mask))))
            self.prompt_id = str(self.metadata.get("prompt_id", path.stem))
            self.family = str(self.metadata.get("family", "unknown"))
            self.policy = str(self.metadata.get("policy", "unknown"))
            self.event_count = int(len(archive["event_meta"]))
            self.row_count = int(len(archive["row_meta"]))
            row_index = np.asarray(archive["event_row_index"], dtype=np.int64)
            if np.any(row_index < 0) or np.any(row_index >= self.row_count):
                raise ValueError(f"{path}: invalid event-to-row index")

            row_labels = np.asarray(archive["row_labels"], dtype=np.float32)
            forced_hazard, forced_peak = trajectory_targets(row_labels, horizons)
            self.inputs = {
                "event_meta": np.asarray(archive["event_meta"], dtype=np.int64),
                "candidate_ids": np.asarray(archive["candidate_ids"], dtype=np.int64),
                "candidate_logits": np.asarray(archive["candidate_logits"], dtype=np.float32),
                "candidate_choice": np.asarray(archive["candidate_choice"], dtype=np.float32),
                "candidate_residency": decode_residency(archive["candidate_residency"]),
                "expert_multiplicity": np.asarray(archive["expert_multiplicity"], dtype=np.float32),
                "router_features": np.asarray(archive["router_features"], dtype=np.float32),
                "hidden_countsketch": np.asarray(archive["hidden_countsketch"], dtype=np.float32),
                "event_derived": np.asarray(archive["event_derived"], dtype=np.float32),
                "row_scalars": np.asarray(archive["row_scalars"], dtype=np.float32)[row_index],
                "row_logit_sketch": np.asarray(
                    archive["row_logit_sketch"], dtype=np.float32)[row_index],
            }
            on_policy = not self.gram_present
            self.targets = {
                "immediate": immediate_targets(row_labels)[row_index],
                "forced_hazard": forced_hazard[row_index],
                "forced_peak": forced_peak[row_index],
                "row_mask": np.full(self.event_count, on_policy, dtype=np.bool_),
                "gram_upper": np.nan_to_num(
                    np.asarray(archive["contribution_gram"], dtype=np.float32), nan=0.0),
                "gram_mask": event_label_mask,
                # These heads must not silently train on same-prefix labels.  A
                # future v3 builder will populate explicitly aligned free-run
                # outcomes under these exact keys.
                "free_hazard": np.zeros((self.event_count, len(horizons)), dtype=np.float32),
                "free_hazard_mask": np.zeros(self.event_count, dtype=np.bool_),
                "collapse": np.zeros((self.event_count, 2), dtype=np.float32),
                "collapse_mask": np.zeros(self.event_count, dtype=np.bool_),
            }
            if "free_trajectory_hazard" in archive.files:
                free = np.asarray(archive["free_trajectory_hazard"], dtype=np.float32)
                if free.shape != (self.row_count, len(horizons)):
                    raise ValueError(f"{path}: invalid free_trajectory_hazard shape")
                self.targets["free_hazard"] = free[row_index]
                self.targets["free_hazard_mask"] = np.ones(self.event_count, dtype=np.bool_)
            if "collapse_labels" in archive.files:
                collapse = np.asarray(archive["collapse_labels"], dtype=np.float32)
                if collapse.shape != (self.row_count, 2):
                    raise ValueError(f"{path}: invalid collapse_labels shape")
                self.targets["collapse"] = collapse[row_index]
                self.targets["collapse_mask"] = np.ones(self.event_count, dtype=np.bool_)
            self.top1_failures = int(np.sum(row_labels[:, ROW_LABELS["top1"]] > 0.5))


class TraceChunkDataset(Dataset[dict[str, dict[str, Tensor]]]):
    def __init__(self, shards: list[PromptShard], sequence_length: int,
                 stride: int | None = None):
        self.shards = shards
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        self.chunks: list[tuple[int, int]] = []
        for shard_index, shard in enumerate(shards):
            starts = list(range(0, shard.event_count, self.stride))
            self.chunks.extend((shard_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self.chunks)

    @staticmethod
    def _pad(array: np.ndarray, length: int) -> np.ndarray:
        if len(array) == length:
            return array
        shape = (length,) + array.shape[1:]
        result = np.zeros(shape, dtype=array.dtype)
        result[:len(array)] = array
        return result

    def __getitem__(self, index: int) -> dict[str, dict[str, Tensor]]:
        shard_index, start = self.chunks[index]
        shard = self.shards[shard_index]
        stop = min(start + self.sequence_length, shard.event_count)
        actual = stop - start
        inputs = {
            key: torch.from_numpy(self._pad(value[start:stop], self.sequence_length))
            for key, value in shard.inputs.items()
        }
        inputs["valid"] = torch.cat((torch.ones(actual),
                                      torch.zeros(self.sequence_length - actual)))
        targets = {
            key: torch.from_numpy(self._pad(value[start:stop], self.sequence_length))
            for key, value in shard.targets.items()
        }
        return {"inputs": inputs, "targets": targets}


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern)]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        paths.update(path.resolve() for path in matches if path.suffix == ".npz")
    if not paths:
        raise SystemExit("no falsifier NPZ files matched --data")
    return sorted(paths)


def split_by_prompt(shards: list[PromptShard], validation_fraction: float,
                    seed: int) -> tuple[list[PromptShard], list[PromptShard], list[str]]:
    prompt_ids = sorted({shard.prompt_id for shard in shards})
    if len(prompt_ids) < 2:
        raise SystemExit("prompt-held-out training requires at least two prompt IDs")
    ranked = sorted(prompt_ids, key=lambda value: hashlib.sha256(
        f"{seed}:{value}".encode()).digest())
    count = max(1, min(len(ranked) - 1, math.ceil(len(ranked) * validation_fraction)))
    validation_ids = set(ranked[:count])
    train = [shard for shard in shards if shard.prompt_id not in validation_ids]
    validation = [shard for shard in shards if shard.prompt_id in validation_ids]
    return train, validation, sorted(validation_ids)


def summarize(shards: list[PromptShard], config: FalsifierMoEConfig) -> CorpusSummary:
    on_policy = [shard for shard in shards if not shard.gram_present]
    exact = [shard for shard in shards if shard.gram_present]
    events = sum(shard.event_count for shard in on_policy)
    activations = events * config.cell_repeats * config.routed_topk
    return CorpusSummary(
        prompts=len({shard.prompt_id for shard in shards}),
        trajectories=len(on_policy),
        on_policy_rows=sum(shard.row_count for shard in on_policy),
        on_policy_events=events,
        exact_gram_events=sum(shard.event_count for shard in exact),
        top1_failures=sum(shard.top1_failures for shard in on_policy),
        mean_routed_activations_per_expert=activations / config.routed_experts,
    )


def move_batch(batch: dict[str, dict[str, Tensor]], device: torch.device) -> dict[str, dict[str, Tensor]]:
    return {
        group: {name: value.to(device, non_blocking=True) for name, value in values.items()}
        for group, values in batch.items()
    }


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum() / weight.expand_as(value).sum().clamp_min(1.0)


def compute_loss(outputs: dict[str, Tensor], targets: dict[str, Tensor],
                 valid: Tensor) -> tuple[Tensor, dict[str, float]]:
    row_mask = targets["row_mask"].bool() & valid.bool()
    immediate = targets["immediate"]
    immediate_regression = masked_mean(
        F.smooth_l1_loss(outputs["immediate"][..., :4], immediate[..., :4], reduction="none"),
        row_mask)
    immediate_flip = masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["immediate"][..., 4], immediate[..., 4], reduction="none", pos_weight=torch.tensor(
                12.0, device=immediate.device)), row_mask)
    forced_hazard = masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["forced_trajectory_hazard_logits"], targets["forced_hazard"],
            reduction="none", pos_weight=torch.tensor(8.0, device=immediate.device)), row_mask)
    forced_peak = masked_mean(
        F.smooth_l1_loss(outputs["forced_trajectory_peak"], targets["forced_peak"], reduction="none"),
        row_mask)

    gram_mask = targets["gram_mask"].bool() & valid.bool()
    gram = masked_mean(F.smooth_l1_loss(
        outputs["gram_upper"], targets["gram_upper"], reduction="none"), gram_mask)

    free_mask = targets["free_hazard_mask"].bool() & valid.bool()
    free_hazard = masked_mean(F.binary_cross_entropy_with_logits(
        outputs["free_trajectory_hazard_logits"], targets["free_hazard"], reduction="none"), free_mask)
    collapse_mask = targets["collapse_mask"].bool() & valid.bool()
    collapse = masked_mean(F.binary_cross_entropy_with_logits(
        outputs["collapse_logits"], targets["collapse"], reduction="none"), collapse_mask)

    # This is a numerical stabilizer, not a load-balancing auxiliary loss;
    # Stable LatentMoE load is controlled by the one-step-late QB buffer.
    router_z = outputs["router_logits"].float().square().mean()
    total = (immediate_regression + 2.0 * immediate_flip + forced_hazard + forced_peak
             + gram + free_hazard + collapse + 1e-5 * router_z)
    metrics = {
        "loss": float(total.detach()),
        "immediate_regression": float(immediate_regression.detach()),
        "immediate_flip": float(immediate_flip.detach()),
        "forced_hazard": float(forced_hazard.detach()),
        "forced_peak": float(forced_peak.detach()),
        "gram": float(gram.detach()),
        "free_hazard": float(free_hazard.detach()),
        "collapse": float(collapse.detach()),
        "router_z": float(router_z.detach()),
    }
    return total, metrics


def make_optimizer(model: FalsifierMoE, name: str, learning_rate: float,
                   weight_decay: float, dion_fraction: float):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    if not torch.cuda.is_available():
        raise SystemExit(
            f"{name} smoke requires CUDA: the official Dion package torch.compiles its "
            "orthogonalization kernels and this recovered Windows install has no cl.exe")
    try:
        from dion import Dion3, Muon
    except ImportError as error:
        raise SystemExit(
            "Dion/Muon optimizer requested; install "
            "git+https://github.com/microsoft/dion.git") from error
    groups = model.dion_parameter_groups()
    # This controller intentionally contains many rectangular matrix shapes
    # (modality encoders, per-head MLA blocks, and per-expert blocks).  The
    # official fullgraph optimizer helpers specialize by shape, so their
    # default limit of eight recompiles is too small even for one valid step.
    torch._dynamo.config.recompile_limit = max(
        torch._dynamo.config.recompile_limit, 128)
    torch._dynamo.config.accumulated_recompile_limit = max(
        torch._dynamo.config.accumulated_recompile_limit, 1024)
    if name in ("dion3", "dion3-qkclip"):
        return Dion3(groups, lr=learning_rate, fraction=dion_fraction,
                     weight_decay=weight_decay, use_polar_express=False)
    if name == "muonclip":
        return Muon(groups, lr=learning_rate, weight_decay=weight_decay,
                    use_polar_express=False)
    raise ValueError(name)


@torch.no_grad()
def evaluate(model: FalsifierMoE, loader: DataLoader, device: torch.device,
             max_batches: int) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["inputs"])
        _, metrics = compute_loss(outputs, batch["targets"], batch["inputs"]["valid"])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        if count >= max_batches:
            break
    return {key: value / max(1, count) for key, value in totals.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = FalsifierMoEConfig()
    data_patterns = [value for group in args.data for value in group]
    paths = expand_paths(data_patterns)
    shards = [PromptShard(path, config.trajectory_horizons) for path in paths]
    summary = summarize(shards, config)
    if not args.smoke and summary.on_policy_rows < args.minimum_rows:
        raise SystemExit(
            f"refusing underfilled training corpus: {summary.on_policy_rows} on-policy rows "
            f"< --minimum-rows {args.minimum_rows}; use --smoke only for plumbing validation")
    train_shards, validation_shards, validation_ids = split_by_prompt(
        shards, args.validation_fraction, args.seed)
    sequence_length = min(args.sequence_length, 32) if args.smoke else args.sequence_length
    train_data = TraceChunkDataset(train_shards, sequence_length,
                                   max(1, sequence_length - args.context_overlap))
    validation_data = TraceChunkDataset(validation_shards, sequence_length, sequence_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size,
                                   shuffle=False, num_workers=0)

    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FalsifierMoE(config).to(device)
    optimizer = make_optimizer(model, args.optimizer, args.learning_rate,
                               args.weight_decay, args.dion_fraction)
    use_bf16 = device.type == "cuda" and args.precision == "bf16"
    steps = 1 if args.smoke else args.steps
    loader_iterator = iter(train_loader)
    started = time.monotonic()
    train_metrics: dict[str, float] = {}
    for step in range(steps):
        try:
            batch = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(train_loader)
            batch = next(loader_iterator)
        batch = move_batch(batch, device)
        model.train()
        model.clear_step_state()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=use_bf16):
            outputs = model(
                batch["inputs"], collect_quantiles=True,
                observe_qk=args.optimizer in ("muonclip", "dion3-qkclip"))
            loss, train_metrics = compute_loss(
                outputs, batch["targets"], batch["inputs"]["valid"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        post_step = model.finish_optimizer_step(
            quantile_balance=True,
            qk_clip_tau=(args.qk_clip_tau
                         if args.optimizer in ("muonclip", "dion3-qkclip") else None),
        )
        if not args.quiet:
            print(json.dumps({"step": step + 1, **train_metrics, **post_step}, sort_keys=True),
                  flush=True)

    validation = evaluate(model, validation_loader, device, args.validation_batches)
    elapsed = time.monotonic() - started
    result = {
        "schema": "insignia-falsifier-moe-training-v1",
        "smoke": args.smoke,
        "device": str(device),
        "precision": "bf16-autocast" if use_bf16 else "fp32",
        "optimizer": args.optimizer,
        "steps": steps,
        "seconds": elapsed,
        "corpus": summary.to_dict(),
        "validation_prompt_ids": validation_ids,
        "parameter_ledger": model.parameter_ledger(),
        "train_last": train_metrics,
        "validation": validation,
        "free_trajectory_supervision_present": any(
            np.any(shard.targets["free_hazard_mask"]) for shard in shards),
        "collapse_supervision_present": any(
            np.any(shard.targets["collapse_mask"]) for shard in shards),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        checkpoint = args.output.with_suffix(".pt")
        torch.save({
            "schema": result["schema"],
            "config": config.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "result": result,
        }, checkpoint)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="append", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--optimizer", choices=(
        "adamw", "dion3", "muonclip", "dion3-qkclip"), default="adamw")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--context-overlap", type=int, default=32)
    parser.add_argument("--minimum-rows", type=int, default=10_000)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dion-fraction", type=float, default=0.25)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--qk-clip-tau", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if not args.data:
        args.data = [["scratch/falsifier-data-20260830/*.npz"]]
    if args.sequence_length < 16 or args.sequence_length % 16:
        parser.error("--sequence-length must be a multiple of 16")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in (0,1)")
    if args.context_overlap < 0 or args.context_overlap >= args.sequence_length:
        parser.error("--context-overlap must be in [0,sequence-length)")
    return args


if __name__ == "__main__":
    run(parse_args())
