#!/usr/bin/env python3
import math
import unittest

import numpy as np

from compare_logits import distribution_divergences


class DistributionDivergenceTest(unittest.TestCase):
    def test_identical_logits_are_zero(self):
        logits = np.array([-3.0, 0.0, 2.0], dtype=np.float64)
        kl, js, _, _ = distribution_divergences(logits, logits)
        self.assertAlmostEqual(kl, 0.0, places=15)
        self.assertAlmostEqual(js, 0.0, places=15)

    def test_known_binary_distributions(self):
        left = np.log(np.array([0.75, 0.25], dtype=np.float64))
        right = np.log(np.array([0.25, 0.75], dtype=np.float64))
        kl, js, _, _ = distribution_divergences(left, right)
        self.assertAlmostEqual(kl, 0.5 * math.log(3.0), places=14)
        self.assertAlmostEqual(
            js, 0.75 * math.log(1.5) + 0.25 * math.log(0.5), places=14)


if __name__ == "__main__":
    unittest.main()
