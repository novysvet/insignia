#!/usr/bin/env python3
"""CPU-only parity/structure gate for exact whole-layer sparse prefill."""

from __future__ import annotations

import random
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "glm53_generate.cu"
TOPK = 8
CHUNK = 128


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def first_seen(routes: list[tuple[int, ...]], begin: int, end: int) -> list[int]:
    result: list[int] = []
    for row in range(begin, end):
        for expert in routes[row]:
            if expert not in result:
                result.append(expert)
    return result


def legacy_replay(
    routes: list[tuple[int, ...]],
    values: list[float],
    weights: list[float],
) -> tuple[list[float], list[tuple[int, int]]]:
    output = [f32(0.0)] * len(routes)
    order: list[tuple[int, int]] = []
    for chunk0 in range(0, len(routes), CHUNK):
        chunk_end = min(chunk0 + CHUNK, len(routes))
        for block0 in range(chunk0, chunk_end, 64):
            block_end = min(block0 + 64, chunk_end)
            experts = first_seen(routes, block0, block_end)
            for expert in experts:
                for row in range(block0, block_end):
                    for pick, selected in enumerate(routes[row]):
                        if selected != expert:
                            continue
                        sidecar_id = row * TOPK + pick
                        order.append((row, pick))
                        # A deterministic FP32 non-associative stand-in.  The
                        # test gates operation order, not CUDA's fmaf result.
                        product = f32(values[sidecar_id] * weights[sidecar_id])
                        output[row] = f32(output[row] + product)
    return output, order


def whole_layer_replay(
    routes: list[tuple[int, ...]],
    values: list[float],
    weights: list[float],
) -> tuple[list[float], list[tuple[int, int]], set[int]]:
    """Model Phase-B global production and Phase-C legacy-order replay."""
    sidecar = [f32(0.0)] * (len(routes) * TOPK)
    written: set[int] = set()
    for expert in first_seen(routes, 0, len(routes)):
        for row, picks in enumerate(routes):
            for pick, selected in enumerate(picks):
                if selected != expert:
                    continue
                sidecar_id = row * TOPK + pick
                if sidecar_id in written:
                    raise AssertionError("sidecar output produced twice")
                sidecar[sidecar_id] = values[sidecar_id]
                written.add(sidecar_id)

    output = [f32(0.0)] * len(routes)
    order: list[tuple[int, int]] = []
    for chunk0 in range(0, len(routes), CHUNK):
        chunk_end = min(chunk0 + CHUNK, len(routes))
        for block0 in range(chunk0, chunk_end, 64):
            block_end = min(block0 + 64, chunk_end)
            for expert in first_seen(routes, block0, block_end):
                for row in range(block0, block_end):
                    for pick, selected in enumerate(routes[row]):
                        if selected != expert:
                            continue
                        sidecar_id = row * TOPK + pick
                        order.append((row, pick))
                        product = f32(sidecar[sidecar_id] * weights[sidecar_id])
                        output[row] = f32(output[row] + product)
    return output, order, written


def wrong_global_union_replay(
    routes: list[tuple[int, ...]],
    values: list[float],
    weights: list[float],
) -> tuple[list[float], list[tuple[int, int]]]:
    """Counterexample: accumulating in global-union order is not exact."""
    output = [f32(0.0)] * len(routes)
    order: list[tuple[int, int]] = []
    for chunk0 in range(0, len(routes), CHUNK):
        chunk_end = min(chunk0 + CHUNK, len(routes))
        for block0 in range(chunk0, chunk_end, 64):
            block_end = min(block0 + 64, chunk_end)
            for expert in first_seen(routes, 0, len(routes)):
                for row in range(block0, block_end):
                    for pick, selected in enumerate(routes[row]):
                        if selected != expert:
                            continue
                        sidecar_id = row * TOPK + pick
                        order.append((row, pick))
                        product = f32(values[sidecar_id] * weights[sidecar_id])
                        output[row] = f32(output[row] + product)
    return output, order


class WholeLayerPrefillTest(unittest.TestCase):
    def make_case(self) -> tuple[list[tuple[int, ...]], list[float], list[float]]:
        rng = random.Random(0x53F4)
        routes = [tuple(rng.sample(range(37), TOPK)) for _ in range(275)]
        magnitude = (1.0e20, 3.0, -1.0e20, 0.125, 8192.0, -7.0, 0.5, -4096.0)
        values = [f32(magnitude[pick] * (1.0 + (row % 5) * 0.03125))
                  for row in range(len(routes)) for pick in range(TOPK)]
        weights = [f32(0.125 + ((row * 11 + pick * 7) % 19) / 16.0)
                   for row in range(len(routes)) for pick in range(TOPK)]
        return routes, values, weights

    def test_sidecar_replay_matches_legacy_per_64_order(self) -> None:
        routes, values, weights = self.make_case()
        legacy, legacy_order = legacy_replay(routes, values, weights)
        whole, whole_order, written = whole_layer_replay(routes, values, weights)
        self.assertEqual(legacy_order, whole_order)
        self.assertEqual(written, set(range(len(routes) * TOPK)))
        self.assertEqual(
            b"".join(struct.pack("<f", value) for value in legacy),
            b"".join(struct.pack("<f", value) for value in whole),
        )

        wrong, wrong_order = wrong_global_union_replay(routes, values, weights)
        self.assertNotEqual(legacy_order, wrong_order)

    def test_global_union_accumulation_order_has_fp32_counterexample(self) -> None:
        routes = [(1, 2, 3, 4, 5, 6, 7, 8)] * 64
        routes.append((3, 2, 1, 9, 10, 11, 12, 13))
        values = [f32(0.0)] * (len(routes) * TOPK)
        weights = [f32(1.0)] * (len(routes) * TOPK)
        target = 64 * TOPK
        values[target + 0] = f32(1.0e20)
        values[target + 1] = f32(-1.0e20)
        values[target + 2] = f32(1.0)
        legacy, _ = legacy_replay(routes, values, weights)
        wrong, _ = wrong_global_union_replay(routes, values, weights)
        self.assertNotEqual(
            b"".join(struct.pack("<f", value) for value in legacy),
            b"".join(struct.pack("<f", value) for value in wrong),
        )

    def test_global_union_computes_every_sidecar_id_once(self) -> None:
        routes, _, _ = self.make_case()
        union = first_seen(routes, 0, len(routes))
        written: set[int] = set()
        for expert in union:
            for row, picks in enumerate(routes):
                for pick, selected in enumerate(picks):
                    if selected == expert:
                        sidecar_id = row * TOPK + pick
                        self.assertNotIn(sidecar_id, written)
                        written.add(sidecar_id)
        self.assertEqual(written, set(range(len(routes) * TOPK)))

        chunk_uploads = sum(
            len(first_seen(routes, start, min(start + CHUNK, len(routes))))
            for start in range(0, len(routes), CHUNK)
        )
        self.assertLessEqual(len(union), chunk_uploads)
        prev_routing = None
        for start in range(0, len(routes), CHUNK):
            prev_routing = routes[min(start + CHUNK, len(routes)) - 1]
        self.assertEqual(prev_routing, routes[-1])

    def test_source_keeps_fail_closed_exactness_seams(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        function = source[source.index("void Runner::prefill_prompt_full_layer_major"):]
        self.assertIn("bool prefill_whole_layer_moe_ = false;", source)
        self.assertIn("prompt exceeds the 8192-row sidecar cap", function)
        self.assertIn("approximate/cache-aware prompt routing is enabled", function)
        self.assertIn("MoE metrics/falsifier instrumentation is active", function)
        self.assertLess(function.index("prime_device_arena()"),
                        function.index("DeviceBuffer<float> prompt_device"))
        self.assertLess(function.index("whole_normalized.reset"),
                        function.index("struct ActiveScope"))
        self.assertIn("whole_users.data() + base_user", function)
        self.assertRegex(
            function,
            r"c_gateu_\.get\(\),\s*c_up_\.get\(\),\s*local_ids\.data\(\)",
        )
        self.assertIn("whole_out_ids.data() + base_user", function)
        self.assertIn("block0 += 64", function)
        self.assertIn("cache_slots() >= union_count", function)
        self.assertIn("persist whole-layer FFN result", function)
        self.assertIn("whole_moe_route_sink_ = nullptr;", function)


if __name__ == "__main__":
    unittest.main()
