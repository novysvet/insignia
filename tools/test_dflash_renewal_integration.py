#!/usr/bin/env python3
"""Source-level guard for DFlash renewal-reward accounting.

The executable has no model-free seam for its main decode loop.  This test
therefore locks the three real call-site transitions until the counters are
factored out: scalar bypass (0 drafted, 1 committed), empty draft round
(0 drafted, 1 committed), and verified prefix (matched, matched).
"""

from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "src" / "glm53_generate.cu"


def between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    return text[begin : text.index(end, begin)]


class DFlashRenewalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        cls.block = between(
            source,
            "if (runner.dflash2_on() && generate > 1)",
            "} else if (mtp_k >= 2 && generate > 1)",
        )

    def test_counters_have_distinct_meanings(self) -> None:
        self.assertIn("double committed_total", self.block)
        self.assertIn("accepted_draft_total", self.block)
        self.assertIn("std::array<int, 9> accept_hist", self.block)

    def test_scalar_bypass_is_zero_draft_one_commit(self) -> None:
        scalar = between(
            self.block,
            "if (v2_k1_scalar",
            'if (std::getenv("INSIGNIA_GLM53_DF_DEBUG"))',
        )
        self.assertIn("committed_total += 1", scalar)
        self.assertIn("++accept_hist[0]", scalar)
        self.assertNotIn("accepted_draft_total +=", scalar)

    def test_empty_round_is_zero_draft_one_commit(self) -> None:
        empty = between(
            self.block,
            "if (candidates[0] != truth0)",
            "const auto verify_begin",
        )
        self.assertIn("committed_total += 1", empty)
        self.assertIn("++accept_hist[0]", empty)
        self.assertNotIn("accepted_draft_total +=", empty)

    def test_verified_prefix_rewards_both_counters(self) -> None:
        verified = between(
            self.block,
            "++accept_hist[size_t(matched)]",
            "std::fflush(stdout)",
        )
        self.assertIn("committed_total += matched", verified)
        self.assertIn("accepted_draft_total += matched", verified)

    def test_legacy_metric_reports_true_draft_acceptance(self) -> None:
        report = self.block[self.block.index('std::printf("greedy IDs"') :]
        self.assertIn("accepted_draft_total / std::max(1, rounds)", report)
        self.assertIn("committed_total / std::max(1, rounds)", report)
        self.assertIn("accepted_draft_total / std::max(1.0e-9, decode_seconds)", report)
        self.assertIn("committed_total / std::max(1.0e-9, decode_seconds)", report)
        self.assertIn("renewal rewards", report)

    def test_plain_greedy_tail_counts_only_committed_output(self) -> None:
        tail = between(
            self.block,
            "// Drafter window exhausted",
            "const double decode_seconds",
        )
        self.assertIn("committed_total += 1", tail)
        self.assertNotIn("accepted_draft_total +=", tail)


if __name__ == "__main__":
    unittest.main()
