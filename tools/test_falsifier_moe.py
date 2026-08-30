#!/usr/bin/env python3
"""Numerical and structural checks for the learned falsifier controller."""

from __future__ import annotations

import math

import torch

from falsifier_moe import (
    FalsifierMoE,
    FalsifierMoEConfig,
    fake_dynamic_int8,
    sinkhorn,
    situ_glu,
)


def synthetic_batch(batch: int, tokens: int, candidate_k: int = 32) -> dict[str, torch.Tensor]:
    event_meta = torch.zeros(batch, tokens, 6, dtype=torch.int64)
    event_meta[..., 1] = torch.arange(tokens).remainder(45)
    event_meta[..., 2] = torch.arange(tokens).remainder(4)
    event_meta[..., 3] = 4
    event_meta[..., 5] = 7
    return {
        "event_meta": event_meta,
        "candidate_ids": torch.randint(0, 288, (batch, tokens, candidate_k)),
        "candidate_logits": torch.randn(batch, tokens, candidate_k),
        "candidate_choice": torch.randn(batch, tokens, candidate_k),
        "candidate_residency": torch.randint(
            0, 2, (batch, tokens, candidate_k, 4), dtype=torch.int64).float(),
        "expert_multiplicity": torch.rand(batch, tokens, 8),
        "router_features": torch.randn(batch, tokens, 16),
        "hidden_countsketch": torch.randn(batch, tokens, 64),
        "event_derived": torch.rand(batch, tokens, 9),
        "row_scalars": torch.randn(batch, tokens, 13),
        "row_logit_sketch": torch.randn(batch, tokens, 3, 64),
        "valid": torch.ones(batch, tokens),
    }


def test_activations() -> None:
    gate = torch.linspace(-1000.0, 1000.0, 2001)
    up = torch.flip(gate, dims=(0,))
    result = situ_glu(gate, up)
    assert float(result.abs().max()) <= 100.0001
    small = torch.tensor([-1e-4, 1e-4])
    reference = F_silu(small) * small
    assert torch.allclose(situ_glu(small, small), reference, atol=1e-10, rtol=1e-4)

    vectors = torch.tensor([[0.0, 0.5, -1.0], [3.0, -2.0, 1.0]])
    quantized = fake_dynamic_int8(vectors)
    scales = vectors.abs().amax(dim=-1) / 127.0
    assert torch.all((quantized - vectors).abs().amax(dim=-1) <= scales * 0.50001)
    assert torch.equal(fake_dynamic_int8(torch.zeros_like(vectors)), torch.zeros_like(vectors))


def F_silu(value: torch.Tensor) -> torch.Tensor:
    return value * torch.sigmoid(value)


def test_sinkhorn() -> None:
    torch.manual_seed(3)
    matrix = sinkhorn(torch.randn(7, 4, 4), iterations=32)
    assert torch.allclose(matrix.sum(-1), torch.ones(7, 4), atol=2e-5)
    assert torch.allclose(matrix.sum(-2), torch.ones(7, 4), atol=2e-5)
    assert bool(torch.all(matrix >= 0))


def test_model() -> None:
    torch.manual_seed(7)
    config = FalsifierMoEConfig(
        d_model=64,
        cell_repeats=2,
        attention_heads=2,
        attention_head_dim=16,
        mla_latent=32,
        routed_experts=16,
        routed_topk=2,
        moe_latent=32,
        expert_hidden=32,
        shared_hidden=64,
        gram_rank=2,
    )
    model = FalsifierMoE(config)
    batch = synthetic_batch(2, 8)
    model.clear_step_state()
    outputs = model(batch, collect_quantiles=True, observe_qk=True)
    assert outputs["immediate"].shape == (2, 8, 5)
    assert outputs["forced_trajectory_hazard_logits"].shape == (2, 8, 3)
    assert outputs["forced_trajectory_peak"].shape == (2, 8, 3)
    assert outputs["free_trajectory_hazard_logits"].shape == (2, 8, 3)
    assert outputs["collapse_logits"].shape == (2, 8, 2)
    assert outputs["gram_upper"].shape == (2, 8, 36)
    assert outputs["action_risk"].shape == (2, 8, 6, 4)
    assert outputs["action_cost"].shape == (2, 8, 6, 3)
    assert outputs["selected_experts"].shape == (2, 2, 8, 2)
    assert bool(torch.all(outputs["forced_trajectory_peak"] >= 0))

    loss = (outputs["immediate"].square().mean() +
            outputs["forced_trajectory_peak"].mean() + outputs["gram_upper"].mean())
    loss.backward()
    assert model.cell.moe.expert_gate_up.grad is not None
    assert bool(torch.isfinite(model.cell.moe.expert_gate_up.grad).all())

    before = model.cell.moe.routing_bias.clone()
    step = model.finish_optimizer_step(quantile_balance=True, qk_clip_tau=0.01)
    assert step["quantile_balance_committed"] is True
    assert not torch.equal(before, model.cell.moe.routing_bias)
    assert abs(float(model.cell.moe.routing_bias.mean())) < 2e-6
    assert min(step["qk_gamma"]) < 1.0

    model.eval()
    sequence = {name: value[:1] for name, value in batch.items()}
    with torch.no_grad():
        full = model(sequence)
        state = None
        pieces = []
        for start, stop in ((0, 3), (3, 8)):
            chunk = {name: value[:, start:stop] for name, value in sequence.items()}
            output, state = model.forward_incremental(chunk, state, max_history=16)
            pieces.append(output["hidden"])
        incremental = torch.cat(pieces, dim=1)
    assert torch.allclose(full["hidden"], incremental, atol=2e-5, rtol=2e-5)
    assert all(latent is not None and latent.shape[1] == 8 for latent in state.latents)

    groups = model.dion_parameter_groups()
    grouped = [parameter for group in groups for parameter in group["params"]]
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad}


def test_default_ledger() -> None:
    config = FalsifierMoEConfig()
    model = FalsifierMoE(config)
    ledger = model.parameter_ledger()
    assert math.isclose(ledger["routed_inactive_fraction"], 0.9921875)
    assert 9_500_000 < ledger["total_parameters"] < 11_500_000
    assert ledger["routed_expert_parameters"] == 9_437_184
    assert 0.05 < ledger["active_parameter_fraction"] < 0.12


if __name__ == "__main__":
    test_activations()
    test_sinkhorn()
    test_model()
    test_default_ledger()
    print("falsifier MoE tests: PASS")
