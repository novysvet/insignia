#!/usr/bin/env python3
"""Exact full-vocabulary collision witness for the dataset-v3 logit features.

This intentionally documents a known v3 blind spot.  A future schema must turn
this into a separating/fail-safe test instead of deleting the witness.
"""

from __future__ import annotations

import numpy as np

from build_falsifier_dataset import (
    CountSketch,
    centered_cosine,
    js_divergence,
    softmax_stats,
    top_values,
)
from falsifier_online_features import OnlineLogitState


VOCAB = 154_880
TOP = 32


def collision() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Swap high/low values inside every public (bucket, sign) cell."""
    sketch = CountSketch(VOCAB, 64)
    cells: dict[tuple[int, int], list[int]] = {}
    for token in range(TOP, VOCAB):
        key = (int(sketch.bucket[token]), int(sketch.sign[token] > 0.0))
        cells.setdefault(key, []).append(token)

    safe = np.full(VOCAB, -20.0, dtype=np.float32)
    risky = np.full(VOCAB, -20.0, dtype=np.float32)
    safe[:TOP] = 11.0 - 0.03 * np.arange(TOP, dtype=np.float32)
    risky[:TOP] = safe[:TOP]

    dangerous: list[int] = []
    for tokens in cells.values():
        for offset in range(0, len(tokens) - 1, 2):
            left, right = tokens[offset], tokens[offset + 1]
            safe[left], safe[right] = 9.0, -20.0
            risky[left], risky[right] = -20.0, 9.0
            dangerous.append(right)
    return safe, risky, np.asarray(dangerous, dtype=np.int32)


def main() -> None:
    safe, risky, dangerous = collision()

    safe_scalars, safe_sketches = OnlineLogitState(VOCAB).begin_round(
        safe, np.stack((safe, safe)))
    risky_scalars, risky_sketches = OnlineLogitState(VOCAB).begin_round(
        safe, np.stack((safe, risky)))

    # The entire 16 + 3*64 row input is bit-identical in the two worlds.
    assert np.array_equal(safe_scalars, risky_scalars)
    assert np.array_equal(safe_sketches, risky_sketches)

    safe_ids, safe_values = top_values(safe, TOP)
    risky_ids, risky_values = top_values(risky, TOP)
    assert np.array_equal(safe_ids, risky_ids)
    assert np.array_equal(safe_values, risky_values)

    safe_probability, _ = softmax_stats(safe)
    risky_probability, _ = softmax_stats(risky)
    divergence = js_divergence(safe_probability, risky_probability)
    cosine = centered_cosine(safe, risky)
    safe_mass = float(np.sum(safe_probability[dangerous]))
    risky_mass = float(np.sum(risky_probability[dangerous]))

    assert divergence > 0.69
    assert cosine < -0.998
    assert safe_mass < 1e-12
    assert risky_mass > 0.998
    print(
        "falsifier v3 collision: PASS "
        f"JS={divergence:.9f} centered_cos={cosine:.9f} "
        f"dangerous_mass={safe_mass:.3e}->{risky_mass:.9f}"
    )


if __name__ == "__main__":
    main()
