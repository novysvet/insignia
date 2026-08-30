#!/usr/bin/env python3
"""Small numerical tests for train_falsifier_baseline.py."""

import numpy as np

from train_falsifier_baseline import (logistic_fit, logistic_predict, ridge_fit,
                                      ridge_predict, route_sketch, spearman)


def main() -> None:
    generator = np.random.default_rng(1234)
    features = generator.normal(size=(40, 6))
    target = features @ np.asarray((0.5, -1.0, 0.25, 2.0, -0.75, 0.1)) + 0.3
    prediction = ridge_predict(ridge_fit(features, target, 1e-6), features)
    assert float(np.max(np.abs(prediction - target))) < 1e-5
    assert abs(spearman(target, target) - 1.0) < 1e-12
    assert abs(spearman(target, -target) + 1.0) < 1e-12

    binary = (features[:, 0] + 0.5 * features[:, 1] > 0.0).astype(np.float64)
    probability = logistic_predict(
        logistic_fit(features, binary, regularization=0.001, steps=800), features)
    assert float(np.mean((probability > 0.5) == binary)) > 0.95

    layers = np.arange(4)
    experts = np.arange(32).reshape((4, 8))
    first = route_sketch(layers, experts)
    second = route_sketch(layers, experts)
    assert first.shape == (64,)
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))
    print("falsifier baseline numerical test: PASS")


if __name__ == "__main__":
    main()
