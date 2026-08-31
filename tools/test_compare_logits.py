#!/usr/bin/env python3
import math
import unittest

import numpy as np

from compare_logits import (
    centered_metrics,
    distribution_divergences,
    greedy_margin_slack,
    relative_l2,
)


class DistributionDivergenceTest(unittest.TestCase):
    def test_identical_logits_are_zero(self):
        logits = np.array([-3.0, 0.0, 2.0], dtype=np.float64)
        kl_ab, kl_ba, js, _, _ = distribution_divergences(logits, logits)
        self.assertAlmostEqual(kl_ab, 0.0, places=15)
        self.assertAlmostEqual(kl_ba, 0.0, places=15)
        self.assertAlmostEqual(js, 0.0, places=15)

    def test_known_binary_distributions(self):
        left = np.log(np.array([0.75, 0.25], dtype=np.float64))
        right = np.log(np.array([0.25, 0.75], dtype=np.float64))
        kl_ab, kl_ba, js, _, _ = distribution_divergences(left, right)
        self.assertAlmostEqual(kl_ab, 0.5 * math.log(3.0), places=14)
        self.assertAlmostEqual(kl_ba, 0.5 * math.log(3.0), places=14)
        self.assertAlmostEqual(
            js, 0.75 * math.log(1.5) + 0.25 * math.log(0.5), places=14)

    def test_reverse_kl_is_reported_separately(self):
        left = np.log(np.array([0.8, 0.2], dtype=np.float64))
        right = np.log(np.array([0.5, 0.5], dtype=np.float64))
        kl_ab, kl_ba, _, _, _ = distribution_divergences(left, right)
        self.assertAlmostEqual(kl_ab, 0.8 * math.log(1.6) + 0.2 * math.log(0.4))
        self.assertAlmostEqual(kl_ba, 0.5 * math.log(0.625) + 0.5 * math.log(2.5))

    def test_relative_l2_uses_first_argument_as_reference(self):
        reference = np.array([3.0, 4.0], dtype=np.float64)
        candidate = np.array([0.0, 4.0], dtype=np.float64)
        self.assertAlmostEqual(relative_l2(reference, candidate), 0.6)

    def test_centered_metrics_remove_additive_logit_shift(self):
        reference = np.array([-2.0, 1.0, 7.0], dtype=np.float64)
        candidate = reference + 123.5
        cosine, rel_l2, mse = centered_metrics(reference, candidate)
        self.assertAlmostEqual(cosine, 1.0, places=14)
        self.assertAlmostEqual(rel_l2, 0.0, places=14)
        self.assertAlmostEqual(mse, 0.0, places=14)
        self.assertGreater(relative_l2(reference, candidate), 0.0)

    def test_greedy_margin_slack_is_exact_pairwise_condition(self):
        reference = np.array([4.0, 3.0, -2.0], dtype=np.float64)
        candidate = np.array([3.5, 3.25, 8.0], dtype=np.float64)
        margin, slack = greedy_margin_slack(reference, candidate)
        self.assertEqual(margin, 1.0)
        self.assertEqual(slack, -4.5)


if __name__ == "__main__":
    unittest.main()
