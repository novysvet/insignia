#!/usr/bin/env python3

import unittest
from types import SimpleNamespace

import benchmark_matharena as benchmark


class FakeTokenizer:
    def token_to_id(self, _token):
        return None

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return " ".join(map(str, ids))


class BenchmarkMathArenaTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "problem_idx": 7,
            "problem": "Prove that one plus one is two.",
            "formal_statement": "theorem one_add_one : 1 + 1 = 2 := by\n  sorry",
        }

    def test_prompt_is_explicitly_one_shot_and_contains_both_statements(self):
        prompt = benchmark.chat_prompt(self.row)
        self.assertIn("one-shot run with no verifier", prompt)
        self.assertIn(self.row["problem"], prompt)
        self.assertIn(self.row["formal_statement"], prompt)
        self.assertTrue(prompt.endswith("<|assistant|><think>"))

    def test_structural_screen_rejects_sorry_and_accepts_named_theorem(self):
        rejected = benchmark.structural_quality(
            self.row, "```lean\ntheorem one_add_one : 1 + 1 = 2 := by\n  sorry\n```")
        accepted = benchmark.structural_quality(
            self.row, "```lean\ntheorem one_add_one : 1 + 1 = 2 := by\n  norm_num\n```")
        self.assertFalse(rejected["structural_pass"])
        self.assertEqual(rejected["forbidden_terms"], ["sorry"])
        self.assertTrue(accepted["structural_pass"])

    def test_parse_engine_timing(self):
        output = (
            "512-token prompt 10.000 s; 64 greedy tokens in 20 DFLASH2-k4 rounds "
            "(3.20 accepted/round, 1 empty; 250.0 ms/token; more timing)\n"
            "greedy IDs 10 11 12\n"
        )
        result = benchmark.parse_run(output, FakeTokenizer())
        self.assertEqual(result["ids"], [10, 11, 12])
        self.assertEqual(result["prefill_tokens_per_second"], 51.2)
        self.assertEqual(result["decode_tokens_per_second"], 4.0)

    def test_parse_full_vocab_quality_summary(self):
        output = """cos     mean 0.984966  median 0.99  max 1
mse     mean 2.265e-01  median 0.1  max 1
kl      mean 8.594e-03  median 0.1  max 1
js      mean 2.092e-03  median 0.1  max 1
top-1  agreement 97.92% (94/96; mismatches at steps [4, 9])
nll delta (B-A) total +2.0  ppl A 1.1587 -> B 1.1893
"""
        result = benchmark.parse_comparison(output)
        self.assertEqual(result["top1_matches"], 94)
        self.assertEqual(result["steps"], 96)
        self.assertAlmostEqual(result["cosine_mean"], 0.984966)
        self.assertAlmostEqual(result["ppl_delta_fraction"], 1.1893 / 1.1587 - 1)

    def test_first_divergence_is_one_based(self):
        self.assertIsNone(benchmark.first_divergence([1, 2], [1, 2]))
        self.assertEqual(benchmark.first_divergence([1, 2], [1, 3]), 2)
        self.assertEqual(benchmark.first_divergence([1], [1, 2]), 2)

    def test_prefill_scheduler_is_an_explicit_ab(self):
        common = dict(q8_budget_mb=10240, cache_mb=32768, readers=4,
                      dflash_fp8="/tmp/df", verify_k=4,
                      prefill_approx_moe=False)
        chunked = benchmark.base_environment(
            SimpleNamespace(**common, prefill_full_layer_major=False), "exact")
        layer_major = benchmark.base_environment(
            SimpleNamespace(**common, prefill_full_layer_major=True), "exact")
        self.assertEqual(chunked["INSIGNIA_GLM53_PREFILL_FULL_LAYER_MAJOR"], "0")
        self.assertEqual(layer_major["INSIGNIA_GLM53_PREFILL_FULL_LAYER_MAJOR"], "1")

    def test_prefill_approximation_never_touches_exact_policy(self):
        common = dict(q8_budget_mb=10240, cache_mb=32768, readers=4,
                      dflash_fp8="/tmp/df", verify_k=4,
                      prefill_full_layer_major=True, prefill_approx_moe=True)
        exact = benchmark.base_environment(SimpleNamespace(**common), "exact")
        candidate = benchmark.base_environment(SimpleNamespace(**common), "top6-cache")
        self.assertEqual(exact["INSIGNIA_GLM53_PREFILL_APPROX_MOE"], "0")
        self.assertEqual(candidate["INSIGNIA_GLM53_PREFILL_APPROX_MOE"], "1")


if __name__ == "__main__":
    unittest.main()
