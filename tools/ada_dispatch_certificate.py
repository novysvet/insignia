#!/usr/bin/env python3
"""CPU reference for certifiable dispatch over a finite CUDA kernel family.

The module has five independent pieces:

* an sm_89 resource-residency calculator with explicit allocation assumptions;
* paired, time-uniform best-arm certification that does not assume IID run noise;
* structured simulators whose model borrowing is either advisory or explicitly unsafe;
* minimax / distributionally robust static dispatch helpers;
* an exact dynamic program for a small kernel DAG under a workspace budget.

It intentionally does not predict speed from occupancy or roofline counts.  Those are
feasibility/lower-bound inputs.  A dispatch entry is promoted only by measurements.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import pathlib
import random
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("divisor must be positive")
    return -(-a // b)


def round_up(value: int, quantum: int) -> int:
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    return ceil_div(value, quantum) * quantum


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# ---------------------------------------------------------------------------
# 1. Ada sm_89 resource feasibility and residency
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sm89Limits:
    """Architectural limits for compute capability 8.9.

    Values are bytes, threads, warps, blocks, or 32-bit registers as named.
    The 99 KiB block limit is the user-addressable limit.  The total SM shared
    capacity is 100 KiB.
    """

    compute_capability: str = "8.9"
    warp_size: int = 32
    max_threads_per_block: int = 1024
    max_threads_per_sm: int = 1536
    max_warps_per_sm: int = 48
    max_blocks_per_sm: int = 24
    registers_per_sm: int = 65_536
    max_registers_per_block: int = 65_536
    max_registers_per_thread: int = 255
    shared_bytes_per_sm: int = 100 * 1024
    max_shared_bytes_per_block: int = 99 * 1024


@dataclass(frozen=True)
class AllocationAssumptions:
    """Allocation micro-constants that must be part of a certificate fingerprint.

    The defaults match the modern CUDA occupancy convention: registers are
    rounded in 256-register warp units and a block's register footprint rounds
    its warp count to a four-warp allocation granularity.  Shared memory rounds
    in 256-byte units.  Ada reserves 1 KiB of shared capacity per resident CTA.

    `block_barriers_per_sm=None` deliberately disables a guessed barrier limit.
    A production certificate should populate this from Nsight Compute's
    `ncu_occupancy.get_gpu_data(8, 9)` or validate the result against
    `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for the compiled function.
    The supplied 4/8-warp kernels are limited to at most 12/6 CTAs by warps, so
    their single CTA barrier is non-binding on ordinary Ada occupancy data.
    """

    register_allocation_unit: int = 256
    register_allocation_granularity: str = "warp"  # "warp" or "block"
    warp_register_allocation_granularity: int = 4
    shared_allocation_unit: int = 256
    reserved_shared_bytes_per_block: int = 1024
    block_barriers_per_sm: int | None = None
    provenance: str = (
        "defaults; validate against ncu_occupancy/cudaOccupancy for the exact toolkit+device"
    )


@dataclass(frozen=True)
class KernelMetadata:
    name: str
    threads_per_block: int
    registers_per_thread: int
    static_shared_bytes: int = 0
    dynamic_shared_bytes: int = 0
    block_barriers: int = 0
    spill_load_bytes: int = 0
    spill_store_bytes: int = 0
    stack_bytes: int = 0
    grid_blocks: int | None = None
    sm_count: int | None = None
    operation_count: int | None = None
    bytes_transferred: int | None = None
    instruction_count: int | None = None
    code_bytes: int = 0
    metadata_source: str = "supplied"


@dataclass(frozen=True)
class OccupancyResult:
    feasible: bool
    reason: str
    resident_blocks_per_sm: int
    resident_warps_per_sm: int
    theoretical_occupancy: float
    warps_per_block: int
    register_allocation_warps: int
    allocated_registers_per_warp: int
    allocated_registers_per_block: int
    requested_shared_bytes_per_block: int
    allocated_shared_bytes_per_block: int
    limits_by_resource: Mapping[str, int]
    limiting_resources: tuple[str, ...]
    resource_model_status: str
    unmodeled_resource_limits: tuple[str, ...]
    first_wave_occupancy_upper_bound: float | None
    tail_wave_occupancy_upper_bound: float | None
    achieved_occupancy: None = None
    issue_model: str = (
        "not inferred; collect eligible/issuing warp and stall metrics per SM subpartition"
    )


def calculate_occupancy(
    metadata: KernelMetadata,
    limits: Sm89Limits = Sm89Limits(),
    assumptions: AllocationAssumptions = AllocationAssumptions(),
) -> OccupancyResult:
    """Calculate resource-limited CTA residency for one compiled kernel.

    This is a static residency calculation.  `achieved_occupancy` stays `None`
    because elapsed active-warps-per-cycle requires a profile.  Grid underfill is
    reported only as an upper bound for the first and final waves.
    """

    t = metadata.threads_per_block
    r = metadata.registers_per_thread
    requested_smem = metadata.static_shared_bytes + metadata.dynamic_shared_bytes
    unmodeled_resource_limits = (
        ("barriers",)
        if metadata.block_barriers and assumptions.block_barriers_per_sm is None
        else ()
    )
    resource_model_status = (
        "conditional_barrier_capacity_unknown"
        if unmodeled_resource_limits
        else "exact_under_recorded_microconstants"
    )

    violations: list[str] = []
    if t <= 0:
        violations.append("threads_per_block must be positive")
    if t > limits.max_threads_per_block:
        violations.append("threads_per_block exceeds architectural limit")
    if r < 0 or r > limits.max_registers_per_thread:
        violations.append("registers_per_thread exceeds architectural limit")
    if requested_smem < 0 or requested_smem > limits.max_shared_bytes_per_block:
        violations.append("requested shared memory exceeds per-block limit")
    if metadata.block_barriers < 0:
        violations.append("block_barriers must be non-negative")

    if violations:
        return OccupancyResult(
            feasible=False,
            reason="; ".join(violations),
            resident_blocks_per_sm=0,
            resident_warps_per_sm=0,
            theoretical_occupancy=0.0,
            warps_per_block=0,
            register_allocation_warps=0,
            allocated_registers_per_warp=0,
            allocated_registers_per_block=0,
            requested_shared_bytes_per_block=max(requested_smem, 0),
            allocated_shared_bytes_per_block=0,
            limits_by_resource={},
            limiting_resources=(),
            resource_model_status=resource_model_status,
            unmodeled_resource_limits=unmodeled_resource_limits,
            first_wave_occupancy_upper_bound=0.0,
            tail_wave_occupancy_upper_bound=0.0,
        )

    warps = ceil_div(t, limits.warp_size)
    allocated_warps = round_up(
        warps, assumptions.warp_register_allocation_granularity
    )

    if assumptions.register_allocation_granularity == "warp":
        allocated_regs_per_warp = round_up(
            r * limits.warp_size, assumptions.register_allocation_unit
        )
        allocated_regs_per_block = allocated_warps * allocated_regs_per_warp
    elif assumptions.register_allocation_granularity == "block":
        allocated_regs_per_warp = 0
        allocated_regs_per_block = round_up(
            r * t, assumptions.register_allocation_unit
        )
    else:
        raise ValueError("register_allocation_granularity must be 'warp' or 'block'")

    allocated_smem = round_up(
        requested_smem + assumptions.reserved_shared_bytes_per_block,
        assumptions.shared_allocation_unit,
    )

    if allocated_regs_per_block > limits.max_registers_per_block:
        violations.append("allocated register footprint exceeds per-block limit")
    if allocated_smem > limits.shared_bytes_per_sm:
        violations.append("allocated shared footprint exceeds per-SM capacity")

    if violations:
        return OccupancyResult(
            feasible=False,
            reason="; ".join(violations),
            resident_blocks_per_sm=0,
            resident_warps_per_sm=0,
            theoretical_occupancy=0.0,
            warps_per_block=warps,
            register_allocation_warps=allocated_warps,
            allocated_registers_per_warp=allocated_regs_per_warp,
            allocated_registers_per_block=allocated_regs_per_block,
            requested_shared_bytes_per_block=requested_smem,
            allocated_shared_bytes_per_block=allocated_smem,
            limits_by_resource={},
            limiting_resources=(),
            resource_model_status=resource_model_status,
            unmodeled_resource_limits=unmodeled_resource_limits,
            first_wave_occupancy_upper_bound=0.0,
            tail_wave_occupancy_upper_bound=0.0,
        )

    limits_by_resource: dict[str, int] = {
        "architectural_blocks": limits.max_blocks_per_sm,
        "threads": limits.max_threads_per_sm // t,
        "warps": limits.max_warps_per_sm // warps,
        "registers": (
            limits.registers_per_sm // allocated_regs_per_block
            if allocated_regs_per_block
            else limits.max_blocks_per_sm
        ),
        "shared_memory": (
            limits.shared_bytes_per_sm // allocated_smem
            if allocated_smem
            else limits.max_blocks_per_sm
        ),
    }
    if assumptions.block_barriers_per_sm is not None and metadata.block_barriers:
        limits_by_resource["barriers"] = (
            assumptions.block_barriers_per_sm // metadata.block_barriers
        )

    resident_blocks = min(limits_by_resource.values())
    resident_warps = resident_blocks * warps
    occupancy = resident_warps / limits.max_warps_per_sm
    limiting = tuple(
        key for key, value in limits_by_resource.items() if value == resident_blocks
    )

    first_wave: float | None = None
    tail_wave: float | None = None
    if metadata.grid_blocks is not None and metadata.sm_count is not None:
        if metadata.grid_blocks < 0 or metadata.sm_count <= 0:
            raise ValueError("grid_blocks must be non-negative and sm_count positive")
        capacity = resident_blocks * metadata.sm_count
        if capacity == 0 or metadata.grid_blocks == 0:
            first_wave = 0.0
            tail_wave = 0.0
        else:
            first_blocks = min(metadata.grid_blocks, capacity)
            first_wave = min(
                1.0,
                (first_blocks * warps)
                / (metadata.sm_count * limits.max_warps_per_sm),
            )
            remainder = metadata.grid_blocks % capacity
            tail_blocks = capacity if remainder == 0 else remainder
            tail_wave = min(
                1.0,
                (tail_blocks * warps)
                / (metadata.sm_count * limits.max_warps_per_sm),
            )

    return OccupancyResult(
        feasible=resident_blocks >= 1,
        reason="ok" if resident_blocks >= 1 else "no CTA can reside",
        resident_blocks_per_sm=resident_blocks,
        resident_warps_per_sm=resident_warps,
        theoretical_occupancy=occupancy,
        warps_per_block=warps,
        register_allocation_warps=allocated_warps,
        allocated_registers_per_warp=allocated_regs_per_warp,
        allocated_registers_per_block=allocated_regs_per_block,
        requested_shared_bytes_per_block=requested_smem,
        allocated_shared_bytes_per_block=allocated_smem,
        limits_by_resource=limits_by_resource,
        limiting_resources=limiting,
        resource_model_status=resource_model_status,
        unmodeled_resource_limits=unmodeled_resource_limits,
        first_wave_occupancy_upper_bound=first_wave,
        tail_wave_occupancy_upper_bound=tail_wave,
    )


def validate_runtime_residency(
    metadata: KernelMetadata,
    runtime_active_blocks_per_sm: int,
    limits: Sm89Limits = Sm89Limits(),
    assumptions: AllocationAssumptions = AllocationAssumptions(),
) -> dict[str, Any]:
    """Cross-check the static model with CUDA's compiled-function occupancy API.

    `runtime_active_blocks_per_sm` is the value returned for the exact compiled
    function and dynamic shared-memory setting by
    `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.  A match closes any
    architecture micro-constant left unknown in the portable CPU fallback.
    """

    if runtime_active_blocks_per_sm < 0:
        raise ValueError("runtime_active_blocks_per_sm must be non-negative")
    calculated = calculate_occupancy(metadata, limits, assumptions)
    match = calculated.resident_blocks_per_sm == runtime_active_blocks_per_sm
    return {
        "kernel": metadata.name,
        "calculated_active_blocks_per_sm": calculated.resident_blocks_per_sm,
        "runtime_active_blocks_per_sm": runtime_active_blocks_per_sm,
        "match": match,
        "static_resource_model_status": calculated.resource_model_status,
        "unmodeled_resource_limits": calculated.unmodeled_resource_limits,
        "certificate_eligible": calculated.feasible and match,
        "status": (
            "validated_against_cuda_runtime"
            if match
            else "static_runtime_residency_mismatch"
        ),
    }


_PTXAS_KERNEL_RE = re.compile(r"Compiling entry function '([^']+)'")
_PTXAS_USED_REG_RE = re.compile(r"Used\s+(\d+)\s+registers")
_PTXAS_SMEM_RE = re.compile(r"(\d+)\s+bytes smem")
_PTXAS_STACK_RE = re.compile(r"(\d+)\s+bytes stack frame")
_PTXAS_SPILL_STORE_RE = re.compile(r"(\d+)\s+bytes spill stores")
_PTXAS_SPILL_LOAD_RE = re.compile(r"(\d+)\s+bytes spill loads")


def parse_ptxas_verbose(text: str) -> dict[str, dict[str, int]]:
    """Parse the stable resource fields from `ptxas -v` output.

    The parser keeps only fields used by the certificate.  Unknown wording is
    ignored rather than guessed.
    """

    current = "<unknown>"
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for line in text.splitlines():
        kernel_match = _PTXAS_KERNEL_RE.search(line)
        if kernel_match:
            current = kernel_match.group(1)
            result.setdefault(current, {})
            continue
        patterns = (
            ("registers_per_thread", _PTXAS_USED_REG_RE),
            ("static_shared_bytes", _PTXAS_SMEM_RE),
            ("stack_bytes", _PTXAS_STACK_RE),
            ("spill_store_bytes", _PTXAS_SPILL_STORE_RE),
            ("spill_load_bytes", _PTXAS_SPILL_LOAD_RE),
        )
        for key, pattern in patterns:
            match = pattern.search(line)
            if match:
                result[current][key] = int(match.group(1))
    return dict(result)


def realistic_resource_table() -> list[KernelMetadata]:
    """Known real resource points plus representative fixed-B variants.

    Entries marked `audit` are quoted from the repository's ptxas reports.
    Entries marked `synthetic` are deliberately labeled and exist only to drive
    the CPU simulator; they must not be copied into a production certificate.
    """

    rows = [
        KernelMetadata(
            "nvfp4_lut_dp4a_cta4",
            128,
            40,
            2048,
            block_barriers=1,
            metadata_source="audit:s11-nvfp4-direct-execution",
        ),
        KernelMetadata(
            "nvfp4_lut_dp4a_cta8",
            256,
            40,
            2048,
            block_barriers=1,
            metadata_source="audit:s11-nvfp4-direct-execution",
        ),
        KernelMetadata(
            "nvfp4_tablefree_cta8",
            256,
            40,
            0,
            block_barriers=0,
            metadata_source="audit:s11-nvfp4-direct-execution",
        ),
        KernelMetadata(
            "nvfp4_fp16_tc",
            256,
            48,
            46_080,
            block_barriers=1,
            metadata_source="audit:nvfp4-fp16-tc-frontier",
        ),
        KernelMetadata(
            "nvfp4_imma_single",
            256,
            64,
            4096,
            block_barriers=1,
            metadata_source="audit:nvfp4-fp16-tc-frontier",
        ),
        KernelMetadata(
            "nvfp4_imma_pair",
            256,
            62,
            4096,
            block_barriers=1,
            metadata_source="audit:nvfp4-fp16-tc-frontier",
        ),
    ]
    for role, base_regs, smem in (
        ("packed_gate_up", 44, 4096),
        ("packed_down_store", 38, 3072),
        ("packed_down_acc", 42, 3072),
    ):
        for multiplicity in range(1, 9):
            regs = base_regs + 2 * ((multiplicity - 1) // 2)
            for warps in (4, 8):
                rows.append(
                    KernelMetadata(
                        f"synthetic_{role}_b{multiplicity}_cta{warps}",
                        warps * 32,
                        regs,
                        smem,
                        block_barriers=1,
                        metadata_source="synthetic-realistic; replace with ptxas",
                    )
                )
    return rows


# ---------------------------------------------------------------------------
# 2. Roofline lower bounds and non-identifiability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RooflineCounts:
    operations: float
    bytes_transferred: float
    instruction_count: int
    theoretical_occupancy: float
    launch_overhead_us: float = 0.0


def roofline_lower_bound_us(
    counts: RooflineCounts,
    peak_operations_per_second: float,
    sustainable_bytes_per_second: float,
) -> float:
    if peak_operations_per_second <= 0 or sustainable_bytes_per_second <= 0:
        raise ValueError("roofline ceilings must be positive")
    compute = counts.operations / peak_operations_per_second * 1e6
    memory = counts.bytes_transferred / sustainable_bytes_per_second * 1e6
    return counts.launch_overhead_us + max(compute, memory)


def dependency_mlp_counterexample() -> dict[str, Any]:
    """Two indistinguishable roofline points whose ordering changes by cache state.

    Both kernels issue the same loads, arithmetic, and reductions and reserve the
    same registers.  `memory_parallel` schedules independent loads early but has
    one long arithmetic accumulator chain.  `compute_parallel` serializes load
    consumption yet carries four independent arithmetic accumulators.  Cold data
    rewards memory-level parallelism; hot data rewards arithmetic ILP.
    """

    counts = RooflineCounts(
        operations=8_388_608,
        bytes_transferred=4_718_592,
        instruction_count=23_040,
        theoretical_occupancy=1.0,
        launch_overhead_us=2.8,
    )
    measured = {
        "cold_dram": {"memory_parallel": 11.7, "compute_parallel": 14.2},
        "hot_l2": {"memory_parallel": 7.4, "compute_parallel": 6.6},
    }
    return {
        "counts": to_jsonable(counts),
        "roofline_tie": True,
        "measured_synthetic_us": measured,
        "cold_winner": min(measured["cold_dram"], key=measured["cold_dram"].get),
        "hot_winner": min(measured["hot_l2"], key=measured["hot_l2"].get),
        "explanation": (
            "aggregate counts omit dependency DAGs, outstanding-request depth, "
            "scoreboard stalls, and per-pipe issue readiness"
        ),
    }


def analytically_favored_but_slower_example() -> dict[str, Any]:
    """Real repository result: zero-smem table-free decode loses to the LUT path."""

    return {
        "analytical_preference": "tablefree",
        "reason": "same 40 registers/thread and no spills, but 0 vs 2048 shared bytes",
        "measured_median_us": {"tablefree": 15.443, "lut_dp4a": 12.332},
        "measured_winner": "lut_dp4a",
        "tablefree_slowdown_fraction": 15.443 / 12.332 - 1.0,
        "status": "measured historical evidence, not a synthetic roofline prediction",
    }


# ---------------------------------------------------------------------------
# 3. Paired sequential experiment and structured simulation
# ---------------------------------------------------------------------------


LOCAL_GAPS_US: dict[str, tuple[float, ...]] = {
    # gap is CTA8 - CTA4.  Negative selects CTA8.
    "gate_up_pair": (-0.378, -0.328, -0.702, -0.205, -0.267, -0.453, -0.271, -1.276),
    "down_store": (+0.561, +0.091, -0.323, +0.188, +1.008, +0.498, +0.411, +0.608),
    "down_weighted": (+0.001, +0.007, -0.049, -0.033, +0.651, +0.441, +0.479, +0.344),
}


@dataclass(frozen=True)
class WorkloadState:
    role: str
    multiplicity: int

    @property
    def key(self) -> str:
        return f"{self.role}:B{self.multiplicity}"


@dataclass
class StateSamples:
    contrasts_us: list[float] = field(default_factory=list)
    total_us: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.total_us = float(sum(self.contrasts_us))

    def add(self, contrast_us: float) -> None:
        self.contrasts_us.append(contrast_us)
        self.total_us += contrast_us

    @property
    def n(self) -> int:
        return len(self.contrasts_us)

    @property
    def mean(self) -> float:
        return self.total_us / self.n if self.n else 0.0


@dataclass(frozen=True)
class SequentialDesign:
    alpha: float = 0.05
    epsilon_us: float = 0.10
    contrast_bound_us: float = 2.0
    max_blocks: int = 20_000
    min_blocks_per_state: int = 2
    runs_per_superblock: int = 4

    def validate(self) -> None:
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must be in (0,1)")
        if self.epsilon_us < 0 or self.contrast_bound_us <= 0:
            raise ValueError("epsilon must be non-negative and bound positive")
        if self.max_blocks <= 0 or self.min_blocks_per_state <= 0:
            raise ValueError("sample limits must be positive")


def anytime_hoeffding_radius(
    n: int, contrast_bound_us: float, alpha_for_comparison: float
) -> float:
    """Two-sided time-uniform Azuma-Hoeffding radius.

    If each clipped paired contrast D_t is in [-b,b] and
    E[D_t | history] is the stable pair effect, then a union bound over all
    n >= 1 with alpha_n = 6 alpha / (pi^2 n^2) gives

        |mean_n - effect| <= b sqrt(2/n log(pi^2 n^2/(3 alpha))).

    No IID assumption appears.  Autocorrelation is permitted under the stated
    conditional-mean condition.  Clipping changes the estimand to clipped
    latency and must be declared in the certificate.
    """

    if n <= 0:
        return math.inf
    if not (0 < alpha_for_comparison < 1):
        raise ValueError("alpha_for_comparison must be in (0,1)")
    return contrast_bound_us * math.sqrt(
        2.0
        / n
        * math.log(math.pi**2 * n * n / (3.0 * alpha_for_comparison))
    )


def direct_decision(
    samples: StateSamples,
    design: SequentialDesign,
    alpha_for_state: float,
    incumbent: int = 4,
) -> tuple[int | None, tuple[float, float], str]:
    """Return an epsilon-optimal CTA width when directly certified.

    The contrast is T(CTA8)-T(CTA4).  CTA8 is epsilon-optimal when the upper
    confidence bound is <= epsilon.  CTA4 is epsilon-optimal when the lower
    bound is >= -epsilon.  If both hold, retain the incumbent.
    """

    if samples.n < design.min_blocks_per_state:
        return None, (-math.inf, math.inf), "insufficient_blocks"
    radius = anytime_hoeffding_radius(
        samples.n, design.contrast_bound_us, alpha_for_state
    )
    lower, upper = samples.mean - radius, samples.mean + radius
    eight_ok = upper <= design.epsilon_us
    four_ok = lower >= -design.epsilon_us
    if eight_ok and four_ok:
        return incumbent, (lower, upper), "both_epsilon_optimal"
    if eight_ok:
        return 8, (lower, upper), "cta8_certified"
    if four_ok:
        return 4, (lower, upper), "cta4_certified"
    return None, (lower, upper), "unresolved"


class HeavyTailedAutocorrelatedLatency:
    """Synthetic paired WSL-like timing process.

    Every call emits a randomized ABBA/BAAB superblock.  A global AR(1) process,
    Student-t shocks, slow within-block drift, and rare positive stalls make raw
    runs heavy-tailed and autocorrelated.  The returned contrast is clipped to
    the design's declared bound.
    """

    def __init__(
        self,
        gaps_us: Mapping[WorkloadState, float],
        seed: int,
        contrast_bound_us: float,
        rho: float = 0.88,
        common_sigma_us: float = 0.32,
        arm_sigma_us: float = 0.10,
        outlier_probability: float = 0.025,
        outlier_scale_us: float = 4.5,
    ) -> None:
        self.gaps = dict(gaps_us)
        self.rng = np.random.default_rng(seed)
        self.contrast_bound_us = contrast_bound_us
        self.rho = rho
        self.common_sigma_us = common_sigma_us
        self.arm_sigma_us = arm_sigma_us
        self.outlier_probability = outlier_probability
        self.outlier_scale_us = outlier_scale_us
        self.hidden = 0.0
        self.time_index = 0

    def _baseline(self, state: WorkloadState) -> float:
        role_offset = {
            "gate_up_pair": 12.5,
            "down_store": 10.8,
            "down_weighted": 11.2,
        }.get(state.role, 11.0)
        return role_offset + 0.55 * state.multiplicity

    def sample_superblock(self, state: WorkloadState) -> tuple[float, dict[str, Any]]:
        gap = self.gaps[state]
        means = {4: self._baseline(state), 8: self._baseline(state) + gap}
        order = (4, 8, 8, 4) if self.rng.random() < 0.5 else (8, 4, 4, 8)
        slope = self.rng.normal(0.0, 0.055)
        raw: list[tuple[int, float]] = []
        for position, arm in enumerate(order):
            self.hidden = (
                self.rho * self.hidden
                + self.common_sigma_us * self.rng.standard_t(df=3) / math.sqrt(3.0)
            )
            common = self.hidden + slope * (position - 1.5)
            arm_noise = self.arm_sigma_us * self.rng.standard_t(df=4)
            stall = 0.0
            if self.rng.random() < self.outlier_probability:
                stall = self.rng.exponential(self.outlier_scale_us)
            latency = max(0.01, means[arm] + common + arm_noise + stall)
            raw.append((arm, latency))
            self.time_index += 1
        mean4 = statistics.fmean(v for a, v in raw if a == 4)
        mean8 = statistics.fmean(v for a, v in raw if a == 8)
        unbounded = mean8 - mean4
        clipped = float(
            np.clip(unbounded, -self.contrast_bound_us, self.contrast_bound_us)
        )
        return clipped, {
            "order": order,
            "raw_us": raw,
            "unclipped_contrast_us": unbounded,
            "clipped": clipped != unbounded,
        }


# --- Structured estimators used by the simulator --------------------------------


def pava(values: Sequence[float], weights: Sequence[float], increasing: bool = True) -> np.ndarray:
    """Weighted pool-adjacent-violators algorithm."""

    y = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if y.shape != w.shape:
        raise ValueError("values and weights must have the same shape")
    if not increasing:
        y = -y
    blocks: list[list[float]] = []  # start, end, weight, weighted sum
    for i, (value, weight) in enumerate(zip(y, w, strict=True)):
        blocks.append([float(i), float(i), float(weight), float(weight * value)])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[3] / left[2] <= right[3] / right[2]:
                break
            merged = [left[0], right[1], left[2] + right[2], left[3] + right[3]]
            blocks[-2:] = [merged]
    out = np.empty_like(y)
    for start, end, weight, total in blocks:
        out[int(start) : int(end) + 1] = total / weight
    return out if increasing else -out


def fit_piecewise_constant(
    means: Sequence[float], weights: Sequence[float], max_segments: int = 3
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Exact weighted least-squares segmentation for a short 1-D sequence."""

    y = np.asarray(means, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(y)
    if n == 0:
        return np.array([]), []
    prefix_w = np.concatenate(([0.0], np.cumsum(w)))
    prefix_y = np.concatenate(([0.0], np.cumsum(w * y)))
    prefix_y2 = np.concatenate(([0.0], np.cumsum(w * y * y)))

    def segment_cost(i: int, j: int) -> float:
        sw = prefix_w[j] - prefix_w[i]
        if sw <= 0:
            return 0.0
        sy = prefix_y[j] - prefix_y[i]
        sy2 = prefix_y2[j] - prefix_y2[i]
        return max(0.0, sy2 - sy * sy / sw)

    segments = min(max_segments, n)
    dp = np.full((segments + 1, n + 1), np.inf)
    prev = np.full((segments + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for s in range(1, segments + 1):
        for j in range(s, n + 1):
            for i in range(s - 1, j):
                candidate = dp[s - 1, i] + segment_cost(i, j)
                if candidate < dp[s, j]:
                    dp[s, j] = candidate
                    prev[s, j] = i
    # BIC-like penalty avoids always taking max_segments.
    total_weight = max(float(np.sum(w)), 1.0)
    penalty = max(float(np.var(y)), 1e-6) * math.log(total_weight + 1.0)
    chosen_s = min(
        range(1, segments + 1), key=lambda s: dp[s, n] + penalty * s
    )
    spans: list[tuple[int, int]] = []
    j = n
    s = chosen_s
    while s > 0:
        i = int(prev[s, j])
        spans.append((i, j))
        j, s = i, s - 1
    spans.reverse()
    fitted = np.empty(n, dtype=float)
    for i, j in spans:
        sw = np.sum(w[i:j])
        fitted[i:j] = np.sum(w[i:j] * y[i:j]) / max(sw, 1e-12)
    return fitted, spans


def fit_low_rank_gap_matrix(
    states: Sequence[WorkloadState], samples: Mapping[WorkloadState, StateSamples], rank: int = 1
) -> dict[WorkloadState, float]:
    roles = sorted({s.role for s in states})
    max_b = max(s.multiplicity for s in states)
    matrix = np.full((len(roles), max_b), np.nan)
    role_index = {role: i for i, role in enumerate(roles)}
    for state in states:
        if samples[state].n:
            matrix[role_index[state.role], state.multiplicity - 1] = samples[state].mean
    global_mean = float(np.nanmean(matrix)) if np.isfinite(matrix).any() else 0.0
    for i in range(matrix.shape[0]):
        row_mean = float(np.nanmean(matrix[i])) if np.isfinite(matrix[i]).any() else global_mean
        matrix[i] = np.where(np.isnan(matrix[i]), row_mean, matrix[i])
    # The matrices here are tiny (roles x multiplicities).  A deterministic
    # power iteration avoids repeatedly entering a heavyweight BLAS/LAPACK SVD
    # inside the sequential loop and is sufficient for sampling guidance.
    residual = matrix.copy()
    fitted = np.zeros_like(matrix)
    components = min(rank, min(matrix.shape))
    for component_index in range(components):
        vector = np.arange(1, residual.shape[1] + 1, dtype=float)
        vector += component_index
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        left = np.ones(residual.shape[0], dtype=float)
        for _ in range(12):
            left = residual @ vector
            left_norm = float(np.linalg.norm(left))
            if left_norm <= 1e-12:
                break
            left /= left_norm
            vector = residual.T @ left
            vector_norm = float(np.linalg.norm(vector))
            if vector_norm <= 1e-12:
                break
            vector /= vector_norm
        sigma = float(left @ residual @ vector)
        component = sigma * np.outer(left, vector)
        fitted += component
        residual -= component
    return {
        state: float(fitted[role_index[state.role], state.multiplicity - 1])
        for state in states
    }


def structured_predictions(
    policy: str,
    states: Sequence[WorkloadState],
    samples: Mapping[WorkloadState, StateSamples],
) -> dict[WorkloadState, float]:
    if policy in {"low_rank", "safe_low_rank"}:
        return fit_low_rank_gap_matrix(states, samples, rank=1)
    predictions: dict[WorkloadState, float] = {}
    for role in sorted({s.role for s in states}):
        role_states = sorted(
            (s for s in states if s.role == role), key=lambda s: s.multiplicity
        )
        values = [samples[s].mean for s in role_states]
        weights = [max(samples[s].n, 1) for s in role_states]
        if policy == "monotone":
            inc = pava(values, weights, increasing=True)
            dec = pava(values, weights, increasing=False)
            raw = np.asarray(values)
            fitted = inc if np.sum((raw - inc) ** 2) <= np.sum((raw - dec) ** 2) else dec
        elif policy == "piecewise":
            fitted, _ = fit_piecewise_constant(values, weights, max_segments=3)
        else:
            fitted = np.asarray(values)
        predictions.update({s: float(v) for s, v in zip(role_states, fitted, strict=True)})
    return predictions


@dataclass(frozen=True)
class SimulationOutcome:
    policy: str
    blocks: int
    runs: int
    classified_states: int
    total_states: int
    exact_misselection_fraction: float
    any_exact_misselection: bool
    any_epsilon_violation: bool
    mean_simple_regret_us: float
    max_simple_regret_us: float
    completed_direct_certificate: bool


def run_structured_bai(
    true_gaps: Mapping[WorkloadState, float],
    policy: str,
    design: SequentialDesign,
    seed: int,
    incumbent: int = 4,
    update_batch: int = 12,
) -> SimulationOutcome:
    """Run one sequential dispatch experiment.

    `independent` and `safe_low_rank` stop only on direct time-uniform intervals.
    The other model policies use a deliberately optimistic model band so their
    misspecification cost is visible; they are never labeled certified.
    """

    design.validate()
    states = sorted(true_gaps, key=lambda state: (state.role, state.multiplicity))
    samples = {state: StateSamples() for state in states}
    decisions: dict[WorkloadState, int] = {}
    alpha_state = design.alpha / len(states)
    process = HeavyTailedAutocorrelatedLatency(
        true_gaps, seed=seed, contrast_bound_us=design.contrast_bound_us
    )
    rng = random.Random(seed ^ 0xA5A5_7F7F)
    blocks_used = 0

    # Balanced initialization prevents a data-dependent model from inventing an
    # entirely unobserved state.  These are still direct paired superblocks.
    for _ in range(design.min_blocks_per_state):
        for state in states:
            if blocks_used >= design.max_blocks:
                break
            contrast, _ = process.sample_superblock(state)
            samples[state].add(contrast)
            blocks_used += 1

    while blocks_used < design.max_blocks:
        if policy in {"independent", "safe_low_rank"}:
            for state in states:
                if state in decisions:
                    continue
                decision, _, _ = direct_decision(
                    samples[state], design, alpha_state, incumbent=incumbent
                )
                if decision is not None:
                    decisions[state] = decision
        else:
            predictions = structured_predictions(policy, states, samples)
            # Heuristic simultaneous normal-like band.  It is intentionally not
            # a certificate because the fitted constraints/rank and intervals use
            # the same data without a selective-inference correction.
            z = math.sqrt(2.0 * math.log(2.0 * len(states) / design.alpha))
            pooled_n = sum(state_samples.n for state_samples in samples.values())
            for state in states:
                if state in decisions:
                    continue
                if policy == "piecewise":
                    effective_n = max(
                        samples[state].n, pooled_n // max(len(states) // 3, 1)
                    )
                elif policy == "monotone":
                    effective_n = max(
                        samples[state].n, pooled_n // max(len(states) // 2, 1)
                    )
                else:  # low_rank
                    effective_n = max(samples[state].n, pooled_n // 3)
                radius = design.contrast_bound_us * z / math.sqrt(max(effective_n, 1))
                prediction = predictions[state]
                if prediction + radius <= design.epsilon_us:
                    decisions[state] = 8
                elif prediction - radius >= -design.epsilon_us:
                    decisions[state] = 4

        if len(decisions) == len(states):
            break

        unresolved = [state for state in states if state not in decisions]
        predictions = (
            structured_predictions(policy, states, samples)
            if policy != "independent"
            else {state: samples[state].mean for state in states}
        )

        batch: list[WorkloadState] = []
        for _ in range(min(update_batch, design.max_blocks - blocks_used)):
            if policy == "independent":
                minimum = min(samples[state].n + batch.count(state) for state in unresolved)
                candidates = [
                    state
                    for state in unresolved
                    if samples[state].n + batch.count(state) == minimum
                ]
                chosen = rng.choice(candidates)
            elif policy == "safe_low_rank":
                # Model-assisted ordering only.  A balance term prevents starvation,
                # and all stopping decisions remain direct.
                chosen = min(
                    unresolved,
                    key=lambda state: (
                        0.35 * (samples[state].n + batch.count(state))
                        + abs(abs(predictions[state]) - design.epsilon_us),
                        samples[state].n + batch.count(state),
                    ),
                )
            else:
                chosen = min(
                    unresolved,
                    key=lambda state: (
                        abs(abs(predictions[state]) - design.epsilon_us)
                        * math.sqrt(samples[state].n + batch.count(state) + 1),
                        samples[state].n + batch.count(state),
                    ),
                )
            batch.append(chosen)

        for state in batch:
            contrast, _ = process.sample_superblock(state)
            samples[state].add(contrast)
            blocks_used += 1

    completed_before_fallback = len(decisions) == len(states)
    # Unresolved entries retain the incumbent.  This is an explicit ROI/budget
    # fallback, not a statistical certificate for those states.
    for state in states:
        decisions.setdefault(state, incumbent)

    regrets: list[float] = []
    exact_mistakes = 0
    for state in states:
        gap = true_gaps[state]
        chosen = decisions[state]
        chosen_latency = gap if chosen == 8 else 0.0
        best_latency = min(0.0, gap)
        regret = max(0.0, chosen_latency - best_latency)
        regrets.append(regret)
        exact_best = 8 if gap < 0 else 4
        if chosen != exact_best:
            exact_mistakes += 1

    direct_classified = sum(
        1
        for state in states
        if direct_decision(samples[state], design, alpha_state, incumbent)[0] is not None
    )
    direct_policy = policy in {"independent", "safe_low_rank"}
    return SimulationOutcome(
        policy=policy,
        blocks=blocks_used,
        runs=blocks_used * design.runs_per_superblock,
        classified_states=direct_classified if direct_policy else len(decisions),
        total_states=len(states),
        exact_misselection_fraction=exact_mistakes / len(states),
        any_exact_misselection=exact_mistakes > 0,
        any_epsilon_violation=any(
            regret > design.epsilon_us + 1e-12 for regret in regrets
        ),
        mean_simple_regret_us=float(np.mean(regrets)),
        max_simple_regret_us=float(np.max(regrets)),
        completed_direct_certificate=direct_policy and completed_before_fallback,
    )


def synthetic_gap_scenario(name: str) -> dict[WorkloadState, float]:
    roles = ("gate_up_pair", "down_store", "down_weighted")
    result: dict[WorkloadState, float] = {}
    if name == "low_rank_piecewise":
        role_factor = {"gate_up_pair": -1.0, "down_store": 0.8, "down_weighted": 0.55}
        b_factor = (0.85, 0.85, -1.10, -1.10, 1.35, 1.35, 1.35, 1.35)
        for role in roles:
            for b, factor in enumerate(b_factor, start=1):
                result[WorkloadState(role, b)] = role_factor[role] * factor
    elif name == "monotone":
        offsets = {"gate_up_pair": -1.4, "down_store": -0.7, "down_weighted": -0.35}
        for role in roles:
            for b in range(1, 9):
                result[WorkloadState(role, b)] = offsets[role] + 0.24 * (b - 1)
    elif name == "misspecified":
        for role, gaps in LOCAL_GAPS_US.items():
            for b, gap in enumerate(gaps, start=1):
                # Enlarge gaps enough for a bounded CPU experiment while preserving
                # the non-monotone sign pattern and near-tie states.
                scaled = 2.2 * gap
                if abs(scaled) < 0.12:
                    scaled = math.copysign(0.08, scaled if scaled else 1.0)
                result[WorkloadState(role, b)] = scaled
        # Localized interaction breaks rank-1 and smoothness assumptions.
        result[WorkloadState("down_store", 6)] = -0.75
        result[WorkloadState("gate_up_pair", 5)] = +0.62
    else:
        raise ValueError(f"unknown scenario {name!r}")
    return result


def evaluate_bai_policies(
    trials: int = 12,
    design: SequentialDesign | None = None,
    scenarios: Sequence[str] = ("low_rank_piecewise", "monotone", "misspecified"),
    policies: Sequence[str] = (
        "independent",
        "safe_low_rank",
        "monotone",
        "piecewise",
        "low_rank",
    ),
    seed: int = 4070,
) -> list[dict[str, Any]]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    design = design or SequentialDesign(
        alpha=0.05,
        epsilon_us=0.12,
        contrast_bound_us=0.75,
        max_blocks=6500,
        min_blocks_per_state=3,
    )
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        gaps = synthetic_gap_scenario(scenario)
        for policy_index, policy in enumerate(policies):
            outcomes = [
                run_structured_bai(
                    gaps,
                    policy,
                    design,
                    seed + 100_000 * scenario_index + 1000 * policy_index + trial,
                )
                for trial in range(trials)
            ]
            rows.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "trials": trials,
                    "family_misselection_probability": statistics.fmean(
                        float(o.any_exact_misselection) for o in outcomes
                    ),
                    "mean_state_misselection_fraction": statistics.fmean(
                        o.exact_misselection_fraction for o in outcomes
                    ),
                    "mean_blocks": statistics.fmean(o.blocks for o in outcomes),
                    "median_blocks": statistics.median(o.blocks for o in outcomes),
                    "mean_runs": statistics.fmean(o.runs for o in outcomes),
                    "family_epsilon_violation_probability": statistics.fmean(
                        float(o.any_epsilon_violation) for o in outcomes
                    ),
                    "direct_certificate_completion_probability": statistics.fmean(
                        float(o.completed_direct_certificate) for o in outcomes
                    ),
                    "mean_simple_regret_us": statistics.fmean(
                        o.mean_simple_regret_us for o in outcomes
                    ),
                    "max_simple_regret_us": max(o.max_simple_regret_us for o in outcomes),
                    "formal_direct_certificate": policy
                    in {"independent", "safe_low_rank"},
                }
            )
    return rows


def evaluate_analytical_reversal_stress(
    trials: int = 100, seed: int = 8900
) -> dict[str, Any]:
    """Stress a resource-favored arm that is truly slower.

    CTA8 is labeled the analytical favorite (for example, from a lower static
    shared-memory footprint), but its true clipped-mean gap relative to CTA4 is
    +0.55 us.  The direct procedure must overturn that preference using the
    same heavy-tailed/autocorrelated paired process as the main simulator.
    """

    if trials <= 0:
        raise ValueError("trials must be positive")
    state = WorkloadState("analytical_favorite_cta8", 1)
    design = SequentialDesign(
        alpha=0.05,
        epsilon_us=0.10,
        contrast_bound_us=0.75,
        max_blocks=2_000,
        min_blocks_per_state=3,
    )
    outcomes = [
        run_structured_bai(
            {state: +0.55}, "independent", design, seed + trial, incumbent=8
        )
        for trial in range(trials)
    ]
    return {
        "analytical_favorite": "cta8",
        "true_winner": "cta4",
        "true_cta8_minus_cta4_gap_us": 0.55,
        "trials": trials,
        "empirical_family_misselection_probability": statistics.fmean(
            float(outcome.any_exact_misselection) for outcome in outcomes
        ),
        "empirical_epsilon_violation_probability": statistics.fmean(
            float(outcome.any_epsilon_violation) for outcome in outcomes
        ),
        "direct_certificate_completion_probability": statistics.fmean(
            float(outcome.completed_direct_certificate) for outcome in outcomes
        ),
        "mean_superblocks": statistics.fmean(outcome.blocks for outcome in outcomes),
        "mean_timed_runs": statistics.fmean(outcome.runs for outcome in outcomes),
        "status": (
            "empirical heavy-tail stress result; the formal error claim comes from "
            "the bounded martingale confidence sequence, not this finite simulation"
        ),
    }


# ---------------------------------------------------------------------------
# 4. Robust dispatch across machines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyInterval:
    lower_us: float
    upper_us: float

    def __post_init__(self) -> None:
        if self.lower_us > self.upper_us:
            raise ValueError("interval lower bound exceeds upper bound")


def interval_regret_upper(
    selected: str, intervals: Mapping[str, LatencyInterval]
) -> float:
    """Exact rectangular-set worst-case simple regret for one selected arm.

    The simultaneous intervals define a rectangular uncertainty set.  Regret
    compares the selected arm only with *other* arms; including the selected
    arm's own lower endpoint would add an artificial interval-width penalty.
    """

    if selected not in intervals:
        raise KeyError(selected)
    challengers = [
        interval.lower_us for name, interval in intervals.items() if name != selected
    ]
    if not challengers:
        return 0.0
    return max(0.0, intervals[selected].upper_us - min(challengers))


def minimax_kernel(intervals: Mapping[str, LatencyInterval]) -> tuple[str, dict[str, float]]:
    regrets = {k: interval_regret_upper(k, intervals) for k in intervals}
    return min(regrets, key=lambda k: (regrets[k], k)), regrets


def worst_case_expected_value_l1(
    values: Sequence[float], nominal: Sequence[float], l1_radius: float
) -> float:
    """Maximize q dot values over the simplex with ||q-p||_1 <= radius."""

    v = np.asarray(values, dtype=float)
    p = np.asarray(nominal, dtype=float).copy()
    if v.shape != p.shape or v.ndim != 1:
        raise ValueError("values and nominal must be one-dimensional with equal length")
    if np.any(p < 0) or not math.isclose(float(np.sum(p)), 1.0, abs_tol=1e-9):
        raise ValueError("nominal must be a probability vector")
    if not 0 <= l1_radius <= 2:
        raise ValueError("l1_radius must be in [0,2]")
    movable = l1_radius / 2.0
    low_order = list(np.argsort(v))
    high_order = list(np.argsort(-v))
    i = j = 0
    while movable > 1e-15 and i < len(v) and j < len(v):
        low = low_order[i]
        high = high_order[j]
        if v[high] <= v[low]:
            break
        take = min(p[low], 1.0 - p[high], movable)
        if take <= 1e-15:
            if p[low] <= 1e-15:
                i += 1
            if 1.0 - p[high] <= 1e-15:
                j += 1
            continue
        p[low] -= take
        p[high] += take
        movable -= take
    return float(np.dot(p, v))


def portable_minimax_kernel(
    machine_intervals: Mapping[str, Mapping[str, LatencyInterval]],
    nominal_machine_probabilities: Mapping[str, float] | None = None,
    distribution_l1_radius: float = 2.0,
    code_cost_us: Mapping[str, float] | None = None,
) -> tuple[str, dict[str, float]]:
    machines = sorted(machine_intervals)
    kernels = sorted(set.intersection(*(set(machine_intervals[m]) for m in machines)))
    if not kernels:
        raise ValueError("machines have no common kernel")
    code_cost_us = dict(code_cost_us or {})
    if nominal_machine_probabilities is None:
        nominal = np.full(len(machines), 1.0 / len(machines))
    else:
        nominal = np.array([nominal_machine_probabilities[m] for m in machines], dtype=float)
        nominal /= np.sum(nominal)
    objective: dict[str, float] = {}
    for kernel in kernels:
        regrets = [interval_regret_upper(kernel, machine_intervals[m]) for m in machines]
        objective[kernel] = worst_case_expected_value_l1(
            regrets, nominal, distribution_l1_radius
        ) + code_cost_us.get(kernel, 0.0)
    return min(objective, key=lambda k: (objective[k], k)), objective


def latency_saving_lower_bound(
    portable_interval: LatencyInterval,
    machine_specific_interval: LatencyInterval,
) -> float:
    """Conservative lower bound on T_portable - T_machine_specific.

    A paired confidence interval for the difference is usually sharper.  This
    helper is valid from simultaneous marginal intervals alone.
    """

    return portable_interval.lower_us - machine_specific_interval.upper_us


def separate_tables_dominate(
    certified_saving_lower_us_per_call: float,
    expected_calls: int,
    extra_tuning_cost_us: float,
    extra_code_and_icache_cost_us: float,
    extra_dispatch_cost_us: float = 0.0,
) -> dict[str, Any]:
    """Sufficient lifetime-ROI certificate for machine-specific tables.

    `certified_saving_lower_us_per_call` must itself be a simultaneous lower
    confidence bound, preferably from a direct paired contrast.  Comparing two
    regret upper bounds is useful for choosing a robust policy, but their
    difference is not a lower confidence bound on realized latency saving.
    """

    gross = expected_calls * max(0.0, certified_saving_lower_us_per_call)
    cost = (
        extra_tuning_cost_us
        + extra_code_and_icache_cost_us
        + extra_dispatch_cost_us
    )
    return {
        "certified_saving_lower_us_per_call": certified_saving_lower_us_per_call,
        "certified_lifetime_saving_lower_us": gross,
        "extra_cost_us": cost,
        "separate_tables_dominate": gross > cost,
        "margin_us": gross - cost,
        "criterion": (
            "simultaneous lower bound on realized latency saving exceeds all "
            "incremental tuning, code/i-cache, and dispatch costs"
        ),
    }


# ---------------------------------------------------------------------------
# 5. Finite compile/measure search and stopping certificate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompileCandidate:
    name: str
    static_latency_lower_bound_us: float
    estimated_compile_cost_us: float
    metadata: KernelMetadata | None = None
    measured_interval: LatencyInterval | None = None
    code_bytes: int = 0
    estimated_measurement_cost_us: float = 0.0
    estimated_deployment_cost_us: float = 0.0


def compile_measure_stopping_certificate(
    candidates: Sequence[CompileCandidate],
    incumbent: str,
    incumbent_upper_us: float,
    epsilon_us: float,
    expected_lifetime_calls: int,
    remaining_measurement_cost_us: float,
    limits: Sm89Limits = Sm89Limits(),
    assumptions: AllocationAssumptions = AllocationAssumptions(),
) -> dict[str, Any]:
    if incumbent not in {c.name for c in candidates}:
        raise ValueError("incumbent must be a candidate")
    classifications: dict[str, str] = {}
    optimistic_best = incumbent_upper_us
    unresolved: list[str] = []
    lower_bounds: dict[str, float] = {}
    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    for candidate in candidates:
        if candidate.metadata is not None:
            occ = calculate_occupancy(candidate.metadata, limits, assumptions)
            if not occ.feasible:
                classifications[candidate.name] = "infeasible_after_compile"
                continue
            if candidate.metadata.spill_load_bytes or candidate.metadata.spill_store_bytes:
                classifications[candidate.name] = "spill_gate_rejected"
                continue
        lower = candidate.static_latency_lower_bound_us
        if candidate.measured_interval is not None:
            lower = max(lower, candidate.measured_interval.lower_us)
        lower_bounds[candidate.name] = lower
        optimistic_best = min(optimistic_best, lower)
        if candidate.name == incumbent:
            classifications[candidate.name] = "incumbent"
        elif lower >= incumbent_upper_us - epsilon_us:
            classifications[candidate.name] = (
                "measured_eliminated"
                if candidate.measured_interval is not None
                else "static_lower_bound_eliminated"
            )
        else:
            classifications[candidate.name] = "unresolved"
            unresolved.append(candidate.name)

    unresolved_before_roi = list(unresolved)
    candidate_lifetime_value_upper_us: dict[str, float] = {}
    candidate_remaining_cost_us: dict[str, float] = {}
    survivors_after_roi: list[str] = []
    for name in unresolved:
        candidate = candidate_by_name[name]
        value_upper = expected_lifetime_calls * max(
            0.0, incumbent_upper_us - lower_bounds[name]
        )
        compile_cost = candidate.estimated_compile_cost_us if candidate.metadata is None else 0.0
        remaining_cost = (
            compile_cost
            + candidate.estimated_measurement_cost_us
            + candidate.estimated_deployment_cost_us
            + remaining_measurement_cost_us
        )
        candidate_lifetime_value_upper_us[name] = value_upper
        candidate_remaining_cost_us[name] = remaining_cost
        if value_upper <= remaining_cost:
            classifications[name] = "roi_abandoned"
        else:
            survivors_after_roi.append(name)
    unresolved = survivors_after_roi
    maximum_lifetime_value_us = max(candidate_lifetime_value_upper_us.values(), default=0.0)
    minimum_remaining_cost_us = min(candidate_remaining_cost_us.values(), default=0.0)
    roi_stop = bool(unresolved_before_roi) and not unresolved
    complete = not unresolved
    return {
        "incumbent": incumbent,
        "epsilon_us": epsilon_us,
        "classifications": classifications,
        "unresolved": unresolved,
        "unresolved_before_roi": unresolved_before_roi,
        "candidate_lifetime_value_upper_us": candidate_lifetime_value_upper_us,
        "candidate_remaining_cost_us": candidate_remaining_cost_us,
        "maximum_remaining_lifetime_value_us": maximum_lifetime_value_us,
        "minimum_remaining_cost_us": minimum_remaining_cost_us,
        "shared_remaining_measurement_cost_us": remaining_measurement_cost_us,
        "stopped_for_roi": roi_stop,
        "complete": complete,
        "certificate_rule": (
            "every finite candidate is infeasible, spill-rejected, statically lower-bound "
            "eliminated, statistically eliminated, selected, or abandoned because its "
            "maximum lifetime value cannot repay the remaining tuning cost"
        ),
    }


# ---------------------------------------------------------------------------
# Hard extension: exact small-DAG scheduler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DagAction:
    name: str
    covers: frozenset[str]
    latency_us: float
    transient_workspace_bytes: int = 0
    launches: int = 1
    synchronization_us: float = 0.0
    cache_tag_out: str = "cold"

    def __post_init__(self) -> None:
        if not self.covers:
            raise ValueError("an action must cover at least one DAG node")
        if self.latency_us < 0 or self.transient_workspace_bytes < 0:
            raise ValueError("latency and workspace must be non-negative")
        if self.launches < 0:
            raise ValueError("launches must be non-negative")


@dataclass(frozen=True)
class DagProblem:
    nodes: tuple[str, ...]
    predecessors: Mapping[str, frozenset[str]]
    output_bytes: Mapping[str, int]
    actions: tuple[DagAction, ...]
    workspace_budget_bytes: int
    launch_overhead_us: float
    transition_cost_us: Mapping[tuple[str, str], float] = field(default_factory=dict)
    initial_cache_tag: str = "cold"


@dataclass(frozen=True)
class DagSchedule:
    total_latency_us: float
    peak_workspace_bytes: int
    actions: tuple[str, ...]
    covered_nodes: tuple[str, ...]


def _validate_dag(problem: DagProblem) -> None:
    node_set = set(problem.nodes)
    if set(problem.predecessors) != node_set:
        raise ValueError("predecessors must define every node")
    if set(problem.output_bytes) != node_set:
        raise ValueError("output_bytes must define every node")
    for node, preds in problem.predecessors.items():
        if not set(preds) <= node_set or node in preds:
            raise ValueError(f"invalid predecessor set for {node}")
    for action in problem.actions:
        if not set(action.covers) <= node_set:
            raise ValueError(f"action {action.name} covers unknown nodes")
    # Kahn cycle check.
    done: set[str] = set()
    while len(done) < len(node_set):
        ready = [
            n for n in problem.nodes if n not in done and set(problem.predecessors[n]) <= done
        ]
        if not ready:
            raise ValueError("DAG contains a cycle")
        done.update(ready)


def exact_dag_schedule(problem: DagProblem) -> DagSchedule:
    """Topological DP over (completed-node mask, cache tag), exact for a small DAG.

    An action may cover one node or a legal fused subset.  External predecessors
    must already be complete; internal predecessors may be in the same action.
    Live outputs are retained until every successor is complete.  Peak workspace
    conservatively includes current live outputs, transient action workspace,
    and outputs escaping the fused subset.
    """

    _validate_dag(problem)
    index = {node: i for i, node in enumerate(problem.nodes)}
    all_mask = (1 << len(problem.nodes)) - 1
    successors: dict[str, set[str]] = {n: set() for n in problem.nodes}
    for node, preds in problem.predecessors.items():
        for pred in preds:
            successors[pred].add(node)

    def set_from_mask(mask: int) -> set[str]:
        return {n for n, i in index.items() if mask & (1 << i)}

    def live_bytes(done: set[str]) -> int:
        total = 0
        for node in done:
            if not successors[node] or any(s not in done for s in successors[node]):
                total += problem.output_bytes[node]
        return total

    start = (0, problem.initial_cache_tag)
    distance: dict[tuple[int, str], float] = {start: 0.0}
    peak_at: dict[tuple[int, str], int] = {start: 0}
    parent: dict[tuple[int, str], tuple[tuple[int, str], DagAction]] = {}
    for completed_count in range(len(problem.nodes) + 1):
        states = sorted(
            (
                state
                for state in distance
                if state[0].bit_count() == completed_count and state[0] != all_mask
            ),
            key=lambda state: (state[0], state[1]),
        )
        for state in states:
            mask, cache_tag = state
            cost = distance[state]
            done = set_from_mask(mask)
            for action in problem.actions:
                covers = set(action.covers)
                if covers & done:
                    continue
                legal = True
                for node in covers:
                    external_preds = set(problem.predecessors[node]) - covers
                    if not external_preds <= done:
                        legal = False
                        break
                if not legal:
                    continue
                # Internal subset must itself be acyclic/reachable.  The global DAG
                # cycle check and predecessor closure imply this, but explicitly walk it.
                internal_done: set[str] = set()
                while len(internal_done) < len(covers):
                    ready = [
                        n
                        for n in covers - internal_done
                        if (set(problem.predecessors[n]) & covers) <= internal_done
                    ]
                    if not ready:
                        legal = False
                        break
                    internal_done.update(ready)
                if not legal:
                    continue

                produced = sum(
                    problem.output_bytes[n]
                    for n in covers
                    if not successors[n] or any(s not in covers for s in successors[n])
                )
                peak = live_bytes(done) + action.transient_workspace_bytes + produced
                if peak > problem.workspace_budget_bytes:
                    continue
                new_mask = mask
                for node in covers:
                    new_mask |= 1 << index[node]
                transition = problem.transition_cost_us.get(
                    (cache_tag, action.cache_tag_out), 0.0
                )
                step = (
                    action.latency_us
                    + action.launches * problem.launch_overhead_us
                    + action.synchronization_us
                    + transition
                )
                new_state = (new_mask, action.cache_tag_out)
                new_cost = cost + step
                new_peak = max(peak_at[state], peak)
                incumbent_cost = distance.get(new_state, math.inf)
                incumbent_peak = peak_at.get(new_state, math.inf)
                if new_cost < incumbent_cost - 1e-12 or (
                    math.isclose(new_cost, incumbent_cost, abs_tol=1e-12)
                    and new_peak < incumbent_peak
                ):
                    distance[new_state] = new_cost
                    peak_at[new_state] = new_peak
                    parent[new_state] = (state, action)

    final_candidates = [state for state in distance if state[0] == all_mask]
    if not final_candidates:
        raise ValueError("no schedule satisfies the workspace budget")
    final_state = min(
        final_candidates,
        key=lambda state: (distance[state], peak_at[state], state[1]),
    )

    chosen: list[DagAction] = []
    state = final_state
    while state != start:
        prev, action = parent[state]
        chosen.append(action)
        state = prev
    chosen.reverse()
    covered = tuple(itertools.chain.from_iterable(sorted(a.covers) for a in chosen))
    return DagSchedule(
        total_latency_us=distance[final_state],
        peak_workspace_bytes=peak_at[final_state],
        actions=tuple(a.name for a in chosen),
        covered_nodes=covered,
    )


def greedy_dag_schedule(problem: DagProblem) -> DagSchedule:
    """Per-node fastest legal action, for comparison only; ignores fusion."""

    single = [a for a in problem.actions if len(a.covers) == 1]
    reduced = dataclasses.replace(problem, actions=tuple(single))
    return exact_dag_schedule(reduced)


def demo_dag_problem() -> DagProblem:
    mib = 1024 * 1024
    nodes = ("quantize", "gate", "activation", "down", "accumulate")
    predecessors = {
        "quantize": frozenset(),
        "gate": frozenset({"quantize"}),
        "activation": frozenset({"gate"}),
        "down": frozenset({"activation"}),
        "accumulate": frozenset({"down"}),
    }
    output = {
        "quantize": 4 * mib,
        "gate": 8 * mib,
        "activation": 8 * mib,
        "down": 4 * mib,
        "accumulate": 1 * mib,
    }
    actions = (
        DagAction("quantize_cta4", frozenset({"quantize"}), 2.00, 2 * mib, cache_tag_out="x4"),
        DagAction("quantize_cta8", frozenset({"quantize"}), 2.12, 3 * mib, cache_tag_out="x8"),
        DagAction("gate_cta4", frozenset({"gate"}), 5.15, 4 * mib, cache_tag_out="gate4"),
        DagAction("gate_cta8", frozenset({"gate"}), 4.88, 6 * mib, cache_tag_out="gate8"),
        DagAction("activation", frozenset({"activation"}), 1.05, 1 * mib, cache_tag_out="act"),
        DagAction("down_cta4", frozenset({"down"}), 5.00, 5 * mib, cache_tag_out="down4"),
        DagAction("down_cta8", frozenset({"down"}), 4.92, 7 * mib, cache_tag_out="down8"),
        DagAction("accumulate", frozenset({"accumulate"}), 0.92, 1 * mib, cache_tag_out="done"),
        DagAction(
            "fused_gate_activation_cta8",
            frozenset({"gate", "activation"}),
            5.28,
            8 * mib,
            launches=1,
            cache_tag_out="act8",
        ),
        DagAction(
            "fused_activation_down_cta4",
            frozenset({"activation", "down"}),
            5.62,
            9 * mib,
            launches=1,
            cache_tag_out="down4",
        ),
    )
    return DagProblem(
        nodes=nodes,
        predecessors=predecessors,
        output_bytes=output,
        actions=actions,
        workspace_budget_bytes=24 * mib,
        launch_overhead_us=2.75,
        transition_cost_us={
            ("x4", "gate8"): 0.18,
            ("x8", "gate4"): 0.22,
            ("gate4", "act"): 0.12,
            ("gate8", "act"): 0.08,
            ("act", "down8"): 0.16,
            ("act8", "down8"): -0.20,
        },
    )


# ---------------------------------------------------------------------------
# Static table, certificate template, and minimal remeasurement plan
# ---------------------------------------------------------------------------


def local_historical_dispatch() -> dict[str, list[int]]:
    return {
        "gate_up_pair": [8, 8, 8, 8, 8, 8, 8, 8],
        "down_store": [4, 4, 8, 4, 4, 4, 4, 4],
        "packed_down_store": [4, 4, 8, 4, 4, 4, 4, 4],
        "down_weighted": [4, 4, 8, 8, 4, 4, 4, 4],
    }


def minimal_remeasurement_plan() -> dict[str, Any]:
    return {
        "trigger_fingerprint_fields": [
            "GPU UUID and SM count",
            "driver/runtime version",
            "locked core and memory clocks or overclock profile",
            "source commit and benchmark hash",
            "nvcc/ptxas version and complete flags",
            "cubin hash",
            "register/shared/stack/spill metadata",
        ],
        "stage_0_compile": (
            "compile the incumbent and CTA4/CTA8 challenger for every affected state; "
            "parse ptxas -v; reject infeasible/spilling variants; compare occupancy with "
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor"
        ),
        "stage_1_sentinels": [
            "gate_up_pair:B4",
            "down_store:B2",
            "down_store:B4",
            "down_weighted:B1",
            "down_weighted:B2",
            "down_weighted:B3",
            "down_weighted:B4",
        ],
        "stage_2_full_family": (
            "one randomized ABBA/BAAB superblock for every state, then allocate only to "
            "unresolved states; a structure model may choose order but cannot certify an "
            "unmeasured state"
        ),
        "stopping": (
            "stop when every state has a simultaneous epsilon-optimal direct interval, "
            "or when the maximum possible lifetime saving is below remaining tuning cost"
        ),
        "remote_rule": (
            "the 4070 Ti SUPER starts unmeasured; no local entry is promoted remotely "
            "without a machine-specific interval"
        ),
    }


def build_dispatch_certificate() -> dict[str, Any]:
    table = local_historical_dispatch()
    return {
        "schema": "insignia.ada_dispatch_certificate.v1",
        "source_commit": "0740c63d1b7c24ff603bf81462f4d8430ad239a1",
        "dispatch": {
            "rtx_4070_super_sm89": {
                "status": "historical_evidence_requires_raw_recertification",
                "table": table,
                "evidence": {
                    "serialized_repetitions": 21,
                    "summary_statistic": "median within-run CTA8-CTA4 difference",
                    "paired_median_cta8_minus_cta4_us": LOCAL_GAPS_US,
                    "raw_path_recorded_in_audit": "/var/lib/insignia/bench-results/s11-down-repeats/",
                    "raw_data_in_supplied_bundle": False,
                    "familywise_error_claim": None,
                    "reason": (
                        "summarized medians do not reconstruct simultaneous confidence intervals"
                    ),
                },
            },
            "rtx_4070_ti_super_oc": {
                "status": "unmeasured",
                "table": None,
                "familywise_error_claim": None,
            },
        },
        "allocation_assumptions": to_jsonable(AllocationAssumptions()),
        "validity": {
            "estimand": "clipped paired mean latency under the named machine regime",
            "required_alpha": 0.05,
            "required_epsilon_us": 0.10,
            "iid_assumption": False,
            "invalidated_by": [
                "compiler/cubin/resource change",
                "clock or power profile change",
                "driver/runtime change",
                "GPU identity change",
                "operation-mode/cache-residency state outside the table",
            ],
        },
        "remeasurement_plan": minimal_remeasurement_plan(),
        "certificate_hash": None,
    }


def finalize_certificate(certificate: MutableMapping[str, Any]) -> dict[str, Any]:
    result = dict(certificate)
    result["certificate_hash"] = None
    result["certificate_hash"] = stable_hash(result)
    return result


def emit_cpp_dispatch_header(certificate: Mapping[str, Any]) -> str:
    table = certificate["dispatch"]["rtx_4070_super_sm89"]["table"]
    rows = []
    for role in ("gate_up_pair", "down_store", "packed_down_store", "down_weighted"):
        values = ", ".join(str(v) for v in table[role])
        symbol = "k" + "".join(part.capitalize() for part in role.split("_")) + "Warps"
        rows.append(f"constexpr std::array<std::uint8_t, 8> {symbol}{{{values}}};")
    return "\n".join(
        [
            "#pragma once",
            "#include <array>",
            "#include <cstdint>",
            "",
            "namespace insignia::ada_dispatch_certificate {",
            "// Historical 4070 SUPER evidence. Remote table is intentionally absent.",
            *rows,
            "constexpr bool kRtx4070SuperTableCertified = false;",
            "constexpr bool kRtx4070TiSuperTableCertified = false;",
            "}  // namespace insignia::ada_dispatch_certificate",
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Evaluation bundle
# ---------------------------------------------------------------------------


def occupancy_report() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata in realistic_resource_table():
        result = calculate_occupancy(metadata)
        rows.append(
            {
                "kernel": metadata.name,
                "metadata_source": metadata.metadata_source,
                "threads": metadata.threads_per_block,
                "registers_per_thread": metadata.registers_per_thread,
                "requested_shared_bytes": result.requested_shared_bytes_per_block,
                "resident_blocks_per_sm": result.resident_blocks_per_sm,
                "resident_warps_per_sm": result.resident_warps_per_sm,
                "theoretical_occupancy": result.theoretical_occupancy,
                "limiting_resources": ",".join(result.limiting_resources),
                "resource_model_status": result.resource_model_status,
                "unmodeled_resource_limits": ",".join(result.unmodeled_resource_limits),
                "feasible": result.feasible,
            }
        )
    return rows


def evaluation_bundle(trials: int = 12) -> dict[str, Any]:
    exact = exact_dag_schedule(demo_dag_problem())
    greedy = greedy_dag_schedule(demo_dag_problem())
    compile_demo = compile_measure_stopping_certificate(
        candidates=(
            CompileCandidate(
                "incumbent_lut",
                9.0,
                20_000,
                metadata=KernelMetadata("incumbent_lut", 256, 40, 2048, block_barriers=1),
                measured_interval=LatencyInterval(12.1, 12.6),
                code_bytes=4096,
            ),
            CompileCandidate(
                "tablefree",
                8.8,
                20_000,
                metadata=KernelMetadata("tablefree", 256, 40, 0),
                measured_interval=LatencyInterval(15.0, 15.9),
                code_bytes=4608,
            ),
            CompileCandidate("wide_uncompiled", 11.8, 35_000, metadata=None, code_bytes=8192),
            CompileCandidate(
                "spilling_variant",
                8.5,
                30_000,
                metadata=KernelMetadata(
                    "spilling_variant",
                    256,
                    80,
                    4096,
                    spill_load_bytes=128,
                    spill_store_bytes=128,
                ),
            ),
        ),
        incumbent="incumbent_lut",
        incumbent_upper_us=12.6,
        epsilon_us=0.10,
        expected_lifetime_calls=2_000_000,
        remaining_measurement_cost_us=5_000_000,
    )
    certificate = finalize_certificate(build_dispatch_certificate())
    return {
        "occupancy": occupancy_report(),
        "roofline_counterexample": dependency_mlp_counterexample(),
        "analytically_favored_but_slower": analytically_favored_but_slower_example(),
        "analytical_reversal_heavy_tail": evaluate_analytical_reversal_stress(),
        "best_arm_simulation": evaluate_bai_policies(trials=trials),
        "dag": {
            "exact": to_jsonable(exact),
            "greedy_no_fusion": to_jsonable(greedy),
            "latency_saving_us": greedy.total_latency_us - exact.total_latency_us,
        },
        "compile_search": compile_demo,
        "dispatch_certificate": certificate,
    }


def write_evaluation(output_dir: pathlib.Path, trials: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = evaluation_bundle(trials=trials)
    (output_dir / "evaluation.json").write_text(
        json.dumps(to_jsonable(bundle), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    certificate = bundle["dispatch_certificate"]
    (output_dir / "dispatch-certificate.json").write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "dispatch-table.hpp").write_text(
        emit_cpp_dispatch_header(certificate), encoding="utf-8"
    )
    # CSV without a pandas dependency.
    rows = bundle["best_arm_simulation"]
    headers = list(rows[0]) if rows else []
    with (output_dir / "best-arm-summary.csv").open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            values = []
            for header in headers:
                value = row[header]
                text = str(value)
                if "," in text or '"' in text:
                    text = '"' + text.replace('"', '""') + '"'
                values.append(text)
            f.write(",".join(values) + "\n")
    return bundle


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    occ = sub.add_parser("occupancy", help="calculate one sm_89 occupancy point")
    occ.add_argument("--threads", type=int, required=True)
    occ.add_argument("--registers", type=int, required=True)
    occ.add_argument("--static-shared", type=int, default=0)
    occ.add_argument("--dynamic-shared", type=int, default=0)
    occ.add_argument("--barriers", type=int, default=0)

    evaluate = sub.add_parser("evaluate", help="write the complete CPU artifact outputs")
    evaluate.add_argument("--output", type=pathlib.Path, required=True)
    evaluate.add_argument("--trials", type=int, default=12)

    ptxas = sub.add_parser("parse-ptxas", help="parse ptxas -v text")
    ptxas.add_argument("path", type=pathlib.Path)

    args = parser.parse_args()
    if args.command == "occupancy":
        metadata = KernelMetadata(
            "cli",
            args.threads,
            args.registers,
            args.static_shared,
            args.dynamic_shared,
            args.barriers,
        )
        print(json.dumps(to_jsonable(calculate_occupancy(metadata)), indent=2, sort_keys=True))
        return 0
    if args.command == "evaluate":
        bundle = write_evaluation(args.output, args.trials)
        print(json.dumps({
            "output": str(args.output),
            "certificate_hash": bundle["dispatch_certificate"]["certificate_hash"],
            "bai_rows": len(bundle["best_arm_simulation"]),
        }, indent=2))
        return 0
    if args.command == "parse-ptxas":
        print(json.dumps(parse_ptxas_verbose(args.path.read_text(encoding="utf-8")), indent=2))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(_cli())
