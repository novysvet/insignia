#!/usr/bin/env python3
"""Tiny sparse controller for DFlash2 expert-policy falsification.

This is deliberately a controller over Insignia runtime observations, not a
language model.  The same cell is reused over depth so the model can spend
compute on causal context without multiplying its resident parameter set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FalsifierMoEConfig:
    d_model: int = 192
    streams: int = 4
    cell_repeats: int = 3
    attention_heads: int = 4
    attention_head_dim: int = 32
    mla_latent: int = 64
    routed_experts: int = 256
    routed_topk: int = 2
    moe_latent: int = 96
    expert_hidden: int = 128
    shared_hidden: int = 384
    candidate_expert_embed: int = 16
    candidate_k: int = 32
    target_experts: int = 288
    target_layers: int = 45
    max_verify_rows: int = 8
    gram_rank: int = 4
    action_count: int = 6              # prefix k=3..8
    trajectory_horizons: tuple[int, ...] = (8, 16, 32)
    sinkhorn_iterations: int = 6

    def validate(self) -> None:
        if self.streams != 4:
            raise ValueError("the trace encoder has exactly four evidence streams")
        if self.d_model % 16 or self.mla_latent % 16 or self.moe_latent % 16:
            raise ValueError("FP8-facing dimensions must be divisible by 16")
        if self.attention_heads * self.attention_head_dim > self.d_model:
            raise ValueError("attention projection cannot exceed d_model")
        if not 0 < self.routed_topk < self.routed_experts:
            raise ValueError("invalid routed expert geometry")
        if self.candidate_k != 32:
            raise ValueError("falsifier trace v2 fixes candidate_k at 32")

    @property
    def routed_inactive_fraction(self) -> float:
        return 1.0 - self.routed_topk / self.routed_experts

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, value: Tensor) -> Tensor:
        scale = torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + self.epsilon)
        return (value.float() * scale).to(value.dtype) * self.weight.to(value.dtype)


def softcap(value: Tensor, beta: float) -> Tensor:
    return beta * torch.tanh(value / beta)


def situ_glu(gate: Tensor, up: Tensor, beta_gate: float = 4.0,
             beta_up: float = 25.0) -> Tensor:
    """Kimi K3 SiTU-GLU; every output coordinate is bounded by 100."""
    return softcap(gate, beta_gate) * torch.sigmoid(gate) * softcap(up, beta_up)


def sinkhorn(logits: Tensor, iterations: int = 6) -> Tensor:
    """Project the final two dimensions onto the Birkhoff polytope."""
    matrix = torch.exp(logits.float().clamp(-12.0, 12.0))
    for _ in range(iterations):
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(1e-12)
    return matrix.to(logits.dtype)


class MHCStreamMixer(nn.Module):
    """Dynamic, identity-biased 4x4 manifold-constrained stream mixing."""

    def __init__(self, config: FalsifierMoEConfig):
        super().__init__()
        base = torch.zeros(config.streams, config.streams)
        base.fill_diagonal_(4.0)
        self.base_logits = nn.Parameter(base)
        self.dynamic = nn.Linear(config.d_model, config.streams ** 2, bias=False)
        nn.init.zeros_(self.dynamic.weight)
        self.streams = config.streams
        self.iterations = config.sinkhorn_iterations
        self.last_matrix: Tensor | None = None

    def forward(self, streams: Tensor, context: Tensor) -> Tensor:
        dynamic = self.dynamic(context).view(*context.shape[:-1], self.streams, self.streams)
        matrix = sinkhorn(dynamic + self.base_logits, self.iterations)
        self.last_matrix = matrix.detach()
        return torch.einsum("...ij,...jd->...id", matrix, streams)


class BlockDepthAttention(nn.Module):
    """Attention over completed controller blocks at the same causal event."""

    def __init__(self, config: FalsifierMoEConfig):
        super().__init__()
        self.heads = config.attention_heads
        self.head_dim = config.attention_head_dim
        width = self.heads * self.head_dim
        self.q = nn.Linear(config.d_model, width, bias=False)
        self.k = nn.Linear(config.d_model, width, bias=False)
        self.v = nn.Linear(config.d_model, width, bias=False)
        self.out = nn.Linear(width, config.d_model, bias=False)

    def forward(self, history: list[Tensor], valid: Tensor | None = None) -> Tensor:
        current = history[-1]
        sources = torch.stack(history, dim=2)
        batch, tokens, source_count, _ = sources.shape
        query = self.q(current).view(batch, tokens, self.heads, self.head_dim)
        key = self.k(sources).view(batch, tokens, source_count, self.heads, self.head_dim)
        value = self.v(sources).view(batch, tokens, source_count, self.heads, self.head_dim)
        scores = torch.einsum("bthd,btshd->bths", query.float(), key.float())
        weights = torch.softmax(scores / math.sqrt(self.head_dim), dim=-1).to(value.dtype)
        result = torch.einsum("bths,btshd->bthd", weights, value)
        result = self.out(result.reshape(batch, tokens, -1))
        if valid is not None:
            result = result * valid.unsqueeze(-1)
        return result


class CausalMLA(nn.Module):
    """Compressed-latent causal attention with an absorbable 64-wide KV cache."""

    def __init__(self, config: FalsifierMoEConfig):
        super().__init__()
        self.heads = config.attention_heads
        self.head_dim = config.attention_head_dim
        self.projected = self.heads * self.head_dim
        self.q_proj = nn.Linear(config.d_model, self.projected, bias=False)
        self.kv_down = nn.Linear(config.d_model, config.mla_latent, bias=False)
        self.kv_norm = RMSNorm(config.mla_latent)
        self.kv_up = nn.Linear(config.mla_latent, 2 * self.projected, bias=False)
        self.output_gate = nn.Linear(config.d_model, self.projected, bias=False)
        self.out_proj = nn.Linear(self.projected, config.d_model, bias=False)
        self._qk_smax: Tensor | None = None

    def reset_qk_observation(self) -> None:
        self._qk_smax = None

    def forward(self, value: Tensor, valid: Tensor | None = None,
                observe_qk: bool = False) -> Tensor:
        batch, tokens, _ = value.shape
        query = self.q_proj(value).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        latent = self.kv_norm(self.kv_down(value))
        key_value = self.kv_up(latent).view(batch, tokens, 2, self.heads, self.head_dim)
        key = key_value[:, :, 0].transpose(1, 2)
        val = key_value[:, :, 1].transpose(1, 2)

        attention_mask = None
        causal = True
        if valid is not None:
            causal_mask = torch.ones(tokens, tokens, dtype=torch.bool, device=value.device).tril()
            attention_mask = causal_mask.view(1, 1, tokens, tokens) & valid[:, None, None, :].bool()
            causal = False
        result = F.scaled_dot_product_attention(
            query, key, val, attn_mask=attention_mask, is_causal=causal,
        )
        result = result.transpose(1, 2).reshape(batch, tokens, self.projected)
        result = result * torch.sigmoid(self.output_gate(value))
        result = self.out_proj(result)
        if valid is not None:
            result = result * valid.unsqueeze(-1)

        if observe_qk:
            with torch.no_grad():
                score = torch.einsum("bhtd,bhsd->bhts", query.float(), key.float())
                score /= math.sqrt(self.head_dim)
                mask = torch.ones(tokens, tokens, dtype=torch.bool, device=value.device).tril()
                if valid is not None:
                    mask = mask.view(1, 1, tokens, tokens) & valid[:, None, None, :].bool()
                else:
                    mask = mask.view(1, 1, tokens, tokens)
                score = score.masked_fill(~mask, -torch.inf)
                observed = score.amax(dim=(0, 2, 3))
                self._qk_smax = observed if self._qk_smax is None else torch.maximum(
                    self._qk_smax.to(observed.device), observed)
        return result

    @torch.no_grad()
    def qk_clip_(self, tau: float = 100.0) -> Tensor:
        """Post-step per-head Q/K projection used by MuonClip-style arms."""
        if self._qk_smax is None:
            return torch.ones(self.heads, device=self.q_proj.weight.device)
        gamma = (tau / self._qk_smax.clamp_min(tau)).clamp(max=1.0)
        scale = torch.sqrt(gamma).to(self.q_proj.weight.dtype)
        self.q_proj.weight.view(self.heads, self.head_dim, -1).mul_(scale[:, None, None])
        key_weight = self.kv_up.weight[:self.projected]
        key_weight.view(self.heads, self.head_dim, -1).mul_(scale[:, None, None])
        return gamma


class GatedMLP(nn.Module):
    def __init__(self, input_width: int, hidden_width: int, output_width: int):
        super().__init__()
        self.gate_up = nn.Linear(input_width, 2 * hidden_width, bias=False)
        self.down = nn.Linear(hidden_width, output_width, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        gate, up = self.gate_up(value).chunk(2, dim=-1)
        return self.down(situ_glu(gate, up))


class StableLatentMoE(nn.Module):
    """Full-width shared branch plus 99.21875%-inactive latent experts."""

    def __init__(self, config: FalsifierMoEConfig):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.d_model, config.routed_experts, bias=False)
        self.latent_down = nn.Linear(config.d_model, config.moe_latent, bias=False)
        self.latent_norm = RMSNorm(config.moe_latent)
        self.latent_up = nn.Linear(config.moe_latent, config.d_model, bias=False)
        self.shared = GatedMLP(config.d_model, config.shared_hidden, config.d_model)

        # Flat 2-D parameters let Dion3 treat each expert as one independent
        # block via num_heads=routed_experts instead of orthogonalizing a 3-D
        # tensor or blending experts together.
        self.expert_gate_up = nn.Parameter(torch.empty(
            config.routed_experts * 2 * config.expert_hidden, config.moe_latent))
        self.expert_down = nn.Parameter(torch.empty(
            config.routed_experts * config.moe_latent, config.expert_hidden))
        nn.init.normal_(self.expert_gate_up, std=0.02)
        nn.init.normal_(self.expert_down, std=0.02)
        self.register_buffer("routing_bias", torch.zeros(config.routed_experts))
        self._qb_margins: list[Tensor] = []
        self.last_routes: Tensor | None = None

    def clear_quantile_batch(self) -> None:
        self._qb_margins.clear()

    @torch.no_grad()
    def quantile_bias_candidate(self) -> Tensor | None:
        if not self._qb_margins:
            return None
        margins = torch.cat(self._qb_margins, dim=0).float()
        quantile = 1.0 - self.config.routed_topk / self.config.routed_experts
        candidate = -torch.quantile(margins, quantile, dim=0)
        return candidate - candidate.mean()

    @torch.no_grad()
    def commit_quantile_balance(self) -> bool:
        candidate = self.quantile_bias_candidate()
        self._qb_margins.clear()
        if candidate is None:
            return False
        self.routing_bias.copy_(candidate.to(self.routing_bias.dtype))
        return True

    def forward(self, value: Tensor, valid: Tensor | None = None,
                collect_quantiles: bool = False) -> tuple[Tensor, dict[str, Tensor]]:
        shape = value.shape
        flat = value.reshape(-1, shape[-1])
        flat_valid = (torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
                      if valid is None else valid.reshape(-1).bool())
        scores = torch.sigmoid(self.router(flat))
        biased = scores + self.routing_bias
        selected_plus = torch.topk(biased, self.config.routed_topk + 1, dim=-1)
        selected = selected_plus.indices[:, :self.config.routed_topk]
        selected_raw = scores.gather(1, selected)
        selected_weight = selected_raw / selected_raw.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        latent = self.latent_down(flat)
        gate_weight = self.expert_gate_up.view(
            self.config.routed_experts, 2 * self.config.expert_hidden, self.config.moe_latent)
        down_weight = self.expert_down.view(
            self.config.routed_experts, self.config.moe_latent, self.config.expert_hidden)
        gathered_gate = gate_weight[selected]
        gate_up = torch.einsum("nkol,nl->nko", gathered_gate, latent)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = situ_glu(gate, up)
        gathered_down = down_weight[selected]
        expert_value = torch.einsum("nkdh,nkh->nkd", gathered_down, activated)
        aggregate = (expert_value * selected_weight.unsqueeze(-1)).sum(dim=1)
        routed = self.latent_up(self.latent_norm(aggregate))
        result = routed + self.shared(flat)
        result = result * flat_valid.unsqueeze(-1)

        if collect_quantiles and torch.any(flat_valid):
            cutoff = selected_plus.values[:, self.config.routed_topk]
            margins = scores - cutoff.unsqueeze(-1)
            self._qb_margins.append(margins[flat_valid].detach())
        self.last_routes = selected.detach()
        diagnostics = {
            "router_logits": torch.logit(scores.clamp(1e-6, 1.0 - 1e-6)).view(
                *shape[:-1], self.config.routed_experts),
            "router_scores": scores.view(*shape[:-1], self.config.routed_experts),
            "selected_experts": selected.view(*shape[:-1], self.config.routed_topk),
            "selected_weights": selected_weight.view(*shape[:-1], self.config.routed_topk),
        }
        return result.view(shape), diagnostics


class TraceEncoder(nn.Module):
    """Maps dataset-v2 fields into the four semantically distinct streams."""

    def __init__(self, config: FalsifierMoEConfig):
        super().__init__()
        d = config.d_model
        self.config = config
        self.logit = nn.Sequential(nn.Linear(208, d), RMSNorm(d))
        self.expert_embedding = nn.Embedding(config.target_experts, config.candidate_expert_embed)
        self.candidate = nn.Sequential(
            nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, d), RMSNorm(d))
        self.router_tail = nn.Sequential(nn.Linear(32, d), RMSNorm(d))
        self.hidden = nn.Sequential(nn.Linear(64, d), RMSNorm(d))
        self.cache = nn.Sequential(nn.Linear(144, d), RMSNorm(d))
        self.position = nn.Linear(16, d, bias=False)
        self.layer_embedding = nn.Embedding(config.target_layers, d)
        self.row_embedding = nn.Embedding(config.max_verify_rows, d)

    @staticmethod
    def _pad(value: Tensor, width: int) -> Tensor:
        return F.pad(value, (0, width - value.shape[-1]))

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        scalars = batch["row_scalars"]
        sketches = batch["row_logit_sketch"].flatten(-2)
        logit = self.logit(self._pad(torch.cat((scalars, sketches), dim=-1), 208))

        candidate_ids = batch["candidate_ids"].long().clamp(0, self.config.target_experts - 1)
        candidate_embed = self.expert_embedding(candidate_ids)
        candidate_logits = batch["candidate_logits"]
        candidate_choice = batch["candidate_choice"]
        candidate_logits = (candidate_logits - candidate_logits.mean(dim=-1, keepdim=True)) / (
            candidate_logits.std(dim=-1, keepdim=True).clamp_min(1e-4))
        candidate_choice = (candidate_choice - candidate_choice.mean(dim=-1, keepdim=True)) / (
            candidate_choice.std(dim=-1, keepdim=True).clamp_min(1e-4))
        rank = torch.linspace(0.0, 1.0, self.config.candidate_k,
                              dtype=candidate_logits.dtype, device=candidate_logits.device)
        rank = rank.view(*((1,) * (candidate_logits.ndim - 1)), self.config.candidate_k, 1)
        rank = rank.expand(*candidate_logits.shape, 1)
        per_candidate = torch.cat((
            candidate_embed,
            candidate_logits.unsqueeze(-1),
            candidate_choice.unsqueeze(-1),
            batch["candidate_residency"],
            rank,
        ), dim=-1)
        per_candidate = self.candidate(self._pad(per_candidate, 32))
        pool_weight = torch.softmax(candidate_choice, dim=-1).unsqueeze(-1)
        candidate_pool = (per_candidate * pool_weight).sum(dim=-2)
        router_tail = torch.cat((
            batch["router_features"], batch["expert_multiplicity"],
            batch["event_meta"][..., 5:6].to(batch["router_features"].dtype) / 8.0,
        ), dim=-1)
        router = candidate_pool + self.router_tail(self._pad(router_tail, 32))

        meta = batch["event_meta"]
        layer = meta[..., 1].long().clamp(0, self.config.target_layers - 1)
        row = meta[..., 2].long().clamp(0, self.config.max_verify_rows - 1)
        position = torch.stack((
            meta[..., 1] / max(1, self.config.target_layers - 1),
            meta[..., 2] / max(1, self.config.max_verify_rows - 1),
            meta[..., 3] / 8.0,
            meta[..., 4] / 15.0,
            meta[..., 5] / 8.0,
        ), dim=-1).to(batch["hidden_countsketch"].dtype)
        hidden = (self.hidden(batch["hidden_countsketch"]) + self.layer_embedding(layer) +
                  self.row_embedding(row) + self.position(self._pad(position, 16)))

        cache_features = torch.cat((
            batch["candidate_residency"].flatten(-2), batch["event_derived"]
        ), dim=-1)
        cache = self.cache(self._pad(cache_features, 144))
        return torch.stack((logit, router, hidden, cache), dim=-2)


class ControllerCell(nn.Module):
    def __init__(self, config: FalsifierMoEConfig):
        super().__init__()
        self.depth = BlockDepthAttention(config)
        self.mixer = MHCStreamMixer(config)
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CausalMLA(config)
        self.moe_norm = RMSNorm(config.d_model)
        self.moe = StableLatentMoE(config)
        self.attention_stream_scale = nn.Parameter(torch.full((config.streams,), 0.10))
        self.moe_stream_scale = nn.Parameter(torch.full((config.streams,), 0.10))

    def forward(self, streams: Tensor, history: list[Tensor], valid: Tensor | None,
                collect_quantiles: bool, observe_qk: bool) -> tuple[Tensor, dict[str, Tensor]]:
        depth = self.depth(history, valid)
        streams = self.mixer(streams, depth)
        attention = self.attention(self.attention_norm(depth), valid, observe_qk)
        streams = streams + attention.unsqueeze(-2) * self.attention_stream_scale.view(1, 1, -1, 1)
        collapsed = streams.mean(dim=-2)
        moe, diagnostics = self.moe(self.moe_norm(collapsed), valid, collect_quantiles)
        streams = streams + moe.unsqueeze(-2) * self.moe_stream_scale.view(1, 1, -1, 1)
        streams = self.mixer(streams, streams.mean(dim=-2))
        if valid is not None:
            streams = streams * valid[..., None, None]
        return streams, diagnostics


def pack_upper(matrix: Tensor) -> Tensor:
    indices = torch.triu_indices(matrix.shape[-1], matrix.shape[-1], device=matrix.device)
    return matrix[..., indices[0], indices[1]]


class FalsifierMoE(nn.Module):
    def __init__(self, config: FalsifierMoEConfig = FalsifierMoEConfig()):
        super().__init__()
        config.validate()
        self.config = config
        self.encoder = TraceEncoder(config)
        self.cell = ControllerCell(config)  # deliberately weight-tied over depth
        self.final_norm = RMSNorm(config.d_model)
        self.immediate_head = nn.Linear(config.d_model, 5)
        self.trajectory_head = nn.Linear(config.d_model, 2 * len(config.trajectory_horizons))
        self.free_trajectory_head = nn.Linear(config.d_model, len(config.trajectory_horizons))
        self.collapse_head = nn.Linear(config.d_model, 2)  # repetition, entropy collapse
        self.gram_factor = nn.Linear(config.d_model, 8 * config.gram_rank)
        self.gram_diagonal = nn.Linear(config.d_model, 8)
        self.action_risk = nn.Linear(config.d_model, config.action_count * 4)
        self.action_cost = nn.Linear(config.d_model, config.action_count * 3)
        self.acceptance_head = nn.Linear(config.d_model, config.max_verify_rows + 1)

    def forward(self, batch: dict[str, Tensor], *, collect_quantiles: bool = False,
                observe_qk: bool = False) -> dict[str, Tensor]:
        valid = batch.get("valid")
        streams = self.encoder(batch)
        history = [streams.mean(dim=-2)]
        routing: list[dict[str, Tensor]] = []
        for _ in range(self.config.cell_repeats):
            streams, diagnostics = self.cell(
                streams, history, valid, collect_quantiles, observe_qk)
            history.append(streams.mean(dim=-2))
            routing.append(diagnostics)
        hidden = self.final_norm(self.cell.depth(history, valid))
        factor = self.gram_factor(hidden).view(*hidden.shape[:-1], 8, self.config.gram_rank)
        diagonal = F.softplus(self.gram_diagonal(hidden))
        gram = factor @ factor.transpose(-1, -2) + torch.diag_embed(diagonal)
        trajectory = self.trajectory_head(hidden)
        horizons = len(self.config.trajectory_horizons)
        return {
            "hidden": hidden,
            "immediate": self.immediate_head(hidden),
            "forced_trajectory_hazard_logits": trajectory[..., :horizons],
            "forced_trajectory_peak": F.softplus(trajectory[..., horizons:]),
            "free_trajectory_hazard_logits": self.free_trajectory_head(hidden),
            "collapse_logits": self.collapse_head(hidden),
            "gram_upper": pack_upper(gram),
            "action_risk": self.action_risk(hidden).view(
                *hidden.shape[:-1], self.config.action_count, 4),
            "action_cost": F.softplus(self.action_cost(hidden)).view(
                *hidden.shape[:-1], self.config.action_count, 3),
            "acceptance_logits": self.acceptance_head(hidden),
            "router_logits": torch.stack([item["router_logits"] for item in routing]),
            "selected_experts": torch.stack([item["selected_experts"] for item in routing]),
        }

    def parameter_ledger(self) -> dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        expert_pool = self.cell.moe.expert_gate_up.numel() + self.cell.moe.expert_down.numel()
        active_expert = expert_pool * self.config.routed_topk // self.config.routed_experts
        active = total - expert_pool + active_expert
        return {
            "total_parameters": total,
            "routed_expert_parameters": expert_pool,
            "active_parameter_estimate": active,
            "active_parameter_fraction": active / total,
            "routed_inactive_fraction": self.config.routed_inactive_fraction,
        }

    def clear_step_state(self) -> None:
        self.cell.moe.clear_quantile_batch()
        self.cell.attention.reset_qk_observation()

    @torch.no_grad()
    def finish_optimizer_step(self, *, quantile_balance: bool,
                              qk_clip_tau: float | None) -> dict[str, object]:
        result: dict[str, object] = {}
        if quantile_balance:
            result["quantile_balance_committed"] = self.cell.moe.commit_quantile_balance()
        else:
            self.cell.moe.clear_quantile_batch()
            result["quantile_balance_committed"] = False
        if qk_clip_tau is not None:
            result["qk_gamma"] = self.cell.attention.qk_clip_(qk_clip_tau).cpu().tolist()
        return result

    def dion_parameter_groups(self) -> list[dict[str, object]]:
        """Explicit grouping; embeddings and prediction heads never enter Dion3."""
        special = {
            id(self.cell.moe.expert_gate_up), id(self.cell.moe.expert_down),
            id(self.cell.attention.q_proj.weight), id(self.cell.attention.kv_up.weight),
        }
        scalar_prefixes = (
            "encoder.expert_embedding", "immediate_head", "trajectory_head",
            "free_trajectory_head", "collapse_head",
            "gram_factor", "gram_diagonal", "action_risk", "action_cost",
            "acceptance_head",
        )
        matrices: list[nn.Parameter] = []
        scalars: list[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            if id(parameter) in special:
                continue
            if parameter.ndim == 2 and not name.startswith(scalar_prefixes):
                matrices.append(parameter)
            else:
                scalars.append(parameter)
        return [
            {"params": [self.cell.moe.expert_gate_up],
             "num_heads": self.config.routed_experts},
            {"params": [self.cell.moe.expert_down],
             "num_heads": self.config.routed_experts},
            {"params": [self.cell.attention.q_proj.weight],
             "num_heads": self.config.attention_heads},
            {"params": [self.cell.attention.kv_up.weight],
             "num_heads": 2 * self.config.attention_heads},
            {"params": matrices},
            {"params": scalars, "algorithm": "adamw"},
        ]


def iter_trainable_matrices(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.ndim == 2:
            yield name, parameter
