#!/usr/bin/env python3

import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

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
            "_row": 0,
            "problem_idx": 7,
            "problem": "Prove that one plus one is two.",
            "formal_statement": "theorem one_add_one : 1 + 1 = 2 := by\n  sorry",
            "source": "synthetic",
            "title": "Synthetic",
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

    def test_parse_scalar_engine_timing(self):
        output = (
            "938-token prompt 69.000 s; 64 greedy tokens total 85.000 s\n"
            "greedy IDs 10 11 12\n"
        )
        result = benchmark.parse_run(output, FakeTokenizer())
        self.assertEqual(result["ids"], [10, 11, 12])
        self.assertEqual(result["generated"], 64)
        self.assertEqual(result["rounds"], 64)
        self.assertEqual(result["verify_k"], 1)
        self.assertEqual(result["accepted_per_round"], 1.0)
        self.assertEqual(result["empty_rounds"], 0)
        self.assertEqual(result["decode_ms_per_token"], 250.0)
        self.assertEqual(result["decode_tokens_per_second"], 4.0)

    def test_parse_full_vocab_quality_summary(self):
        output = """cos     mean 0.984966  median 0.99  max 1
rel_l2  mean 1.234e-01  median 0.1  max 1
mse     mean 2.265e-01  median 0.1  max 1
cos_ctr mean 0.995123  median 0.99  max 1
rel_l2_ctr mean 9.876e-02  median 0.1  max 1
mse_ctr mean 1.111e-01  median 0.1  max 1
kl      mean 8.594e-03  median 0.1  max 1
klrev   mean 9.321e-03  median 0.1  max 1
js      mean 2.092e-03  median 0.1  max 1
top-1  agreement 97.92% (94/96; mismatches at steps [4, 9])
nll delta (B-A) total +2.0  ppl A 1.1587 -> B 1.1893
"""
        result = benchmark.parse_comparison(output)
        self.assertEqual(result["top1_matches"], 94)
        self.assertEqual(result["steps"], 96)
        self.assertAlmostEqual(result["cosine_mean"], 0.984966)
        self.assertAlmostEqual(result["relative_l2_mean"], 0.1234)
        self.assertAlmostEqual(result["centered_cosine_mean"], 0.995123)
        self.assertAlmostEqual(result["centered_relative_l2_mean"], 0.09876)
        self.assertAlmostEqual(result["centered_mse_mean"], 0.1111)
        self.assertAlmostEqual(result["kl_reverse_mean"], 0.009321)
        self.assertAlmostEqual(result["ppl_ratio"], 1.1893 / 1.1587)
        self.assertAlmostEqual(result["ppl_delta_fraction"], 1.1893 / 1.1587 - 1)

    def test_first_divergence_is_one_based(self):
        self.assertIsNone(benchmark.first_divergence([1, 2], [1, 2]))
        self.assertEqual(benchmark.first_divergence([1, 2], [1, 3]), 2)
        self.assertEqual(benchmark.first_divergence([1], [1, 2]), 2)

    def test_prefill_scheduler_is_an_explicit_ab(self):
        common = dict(q8_budget_mb=10240, cache_mb=32768, readers=4,
                      dflash_fp8="/tmp/df", verify_k=4,
                      no_dflash=False,
                      prefill_approx_moe=False, prefill_approx_first_layer=0)
        chunked = benchmark.base_environment(
            SimpleNamespace(**common, prefill_full_layer_major=False), "exact")
        layer_major = benchmark.base_environment(
            SimpleNamespace(**common, prefill_full_layer_major=True), "exact")
        self.assertEqual(chunked["INSIGNIA_GLM53_PREFILL_FULL_LAYER_MAJOR"], "0")
        self.assertEqual(layer_major["INSIGNIA_GLM53_PREFILL_FULL_LAYER_MAJOR"], "1")

    def test_no_dflash_omits_drafter_cache(self):
        args = SimpleNamespace(
            q8_budget_mb=8192, cache_mb=8192, readers=4,
            dflash_fp8="/missing/df", verify_k=4, no_dflash=True,
            prefill_full_layer_major=True, prefill_approx_moe=False,
            prefill_approx_first_layer=0,
        )
        with mock.patch.dict(
                "os.environ", {"INSIGNIA_GLM53_DFLASH2_FP8": "/stale/df"}):
            environment = benchmark.base_environment(args, "exact")
        self.assertEqual(environment["INSIGNIA_GLM53_DFLASH2"], "0")
        self.assertNotIn("INSIGNIA_GLM53_DFLASH2_FP8", environment)

    def test_exact_free_run_reuses_logits_as_quality_reference(self):
        args = SimpleNamespace(
            q8_budget_mb=8192, cache_mb=8192, readers=4,
            dflash_fp8="/missing/df", verify_k=4, no_dflash=True,
            prefill_full_layer_major=True, prefill_approx_moe=False,
            prefill_approx_first_layer=0, quality_tokens=64, generate=64,
            timeout=1,
        )
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(benchmark, "engine_command", return_value=["engine"]), \
                mock.patch.object(benchmark, "run_process", return_value=("ignored", 1.0)) as run, \
                mock.patch.object(benchmark, "parse_run", return_value={"ids": []}):
            case_dir = pathlib.Path(directory)
            benchmark.run_free(args, case_dir / "prompt.csv", FakeTokenizer(),
                               "exact", case_dir)
        environment = run.call_args.args[1]
        self.assertEqual(
            environment["INSIGNIA_GLM53_LOGITS_DUMP"],
            str(case_dir / "exact-quality-logits.f32"),
        )

    def test_prefill_approximation_never_touches_exact_policy(self):
        common = dict(q8_budget_mb=10240, cache_mb=32768, readers=4,
                      dflash_fp8="/tmp/df", verify_k=4,
                      prefill_full_layer_major=True, prefill_approx_moe=True,
                      prefill_approx_first_layer=4)
        exact = benchmark.base_environment(SimpleNamespace(**common), "exact")
        candidate = benchmark.base_environment(SimpleNamespace(**common), "top6-cache")
        self.assertEqual(exact["INSIGNIA_GLM53_PREFILL_APPROX_MOE"], "0")
        self.assertEqual(candidate["INSIGNIA_GLM53_PREFILL_APPROX_MOE"], "1")
        self.assertEqual(candidate["INSIGNIA_GLM53_PREFILL_APPROX_FIRST_LAYER"], "4")

    def test_cache_route_regret_is_an_explicit_ab(self):
        args = SimpleNamespace(
            q8_budget_mb=10240, cache_mb=32768, readers=4,
            dflash_fp8="/tmp/df", verify_k=4,
            prefill_full_layer_major=True, prefill_approx_moe=True,
            prefill_approx_first_layer=0, cache_route_regret=.0005)
        exact = benchmark.base_environment(args, "exact")
        candidate = benchmark.base_environment(args, "top6-cache")
        self.assertNotIn("INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET", exact)
        self.assertEqual(candidate["INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET"], "0.0005")

    def test_candidate_only_report_does_not_require_exact(self):
        result = {
            "ids": [1, 2], "text": "candidate text",
            "prefill_tokens_per_second": 10.0,
            "decode_ms_per_token": 200.0,
            "decode_tokens_per_second": 5.0,
            "accepted_per_round": 2.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            report = pathlib.Path(directory) / "report.md"
            benchmark.write_report(report, self.row, ["top6-cache"],
                                   {"top6-cache": result}, {}, FakeTokenizer())
            rendered = report.read_text(encoding="utf-8")
        self.assertIn("candidate text", rendered)
        self.assertIn("| top6-cache | - |", rendered)


if __name__ == "__main__":
    unittest.main()
