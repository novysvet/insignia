#!/usr/bin/env python3
"""One-shot MathArena ArXivLean performance and approximation-quality gate.

MathArena's official ArXivLean runner gives an agent Lean 4.29, Mathlib, and
iterative verifier/search tools.  Insignia currently exposes only text
generation, so this harness deliberately runs a *one-shot* profile.  It is a
hard real-world prefill/decode and distribution-quality workload, not an
official MathArena leaderboard score.

For every selected theorem the harness:

* runs exact, Top-6, and Top-6+cache-aware DFlash verification;
* records cold-process prefill and decode throughput plus complete text;
* teacher-forces the exact continuation through every arm;
* reports full-vocabulary cosine, MSE, KL, JS, top-1 agreement, and PPL delta;
* marks candidates over the configured PPL budget (3.5% by default) rejected.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any

import pyarrow.parquet as pq
from tokenizers import Tokenizer


GREEDY_RE = re.compile(r"^greedy IDs(.*)$", re.MULTILINE)
DFLASH_RE = re.compile(
    r"(?P<generated>\d+) greedy tokens in (?P<rounds>\d+) DFLASH2-k(?P<verify_k>\d+) "
    r"rounds \((?P<accepted>[0-9.]+) accepted/round, (?P<empty>\d+) empty; "
    r"(?P<ms_token>[0-9.]+) ms/token;")
PROMPT_RE = re.compile(r"(?P<tokens>\d+)-token prompt (?P<seconds>[0-9.]+) s")
LEAN_BLOCK_RE = re.compile(r"```(?:lean)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
THEOREM_RE = re.compile(r"(?m)^\s*(?:theorem|lemma)\s+([A-Za-z_][\w.']*)")


POLICIES: dict[str, dict[str, str]] = {
    "exact": {},
    "top6": {
        "INSIGNIA_GLM53_DF_APPROX_TOPM": "6",
    },
    "top6-cache": {
        "INSIGNIA_GLM53_DF_APPROX_TOPM": "6",
        "INSIGNIA_GLM53_DF_CACHE_ROUTE_K": "32",
        "INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET": ".0010",
        "INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS": "8",
    },
}


POLICY_ENV = (
    "INSIGNIA_GLM53_DF_APPROX_TOPM",
    "INSIGNIA_GLM53_DF_APPROX_RENORM",
    "INSIGNIA_GLM53_DF_APPROX_MASS",
    "INSIGNIA_GLM53_DF_APPROX_MIN_K",
    "INSIGNIA_GLM53_DF_APPROX_MAX_K",
    "INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN",
    "INSIGNIA_GLM53_DF_LOGIT_GUARD_PREFIX",
    "INSIGNIA_GLM53_DF_CALIBRATION_GUARD_JS",
    "INSIGNIA_GLM53_DF_UNCERTAINTY_TOP1_P",
    "INSIGNIA_GLM53_DF_UNCERTAINTY_TOP1_DROP",
    "INSIGNIA_GLM53_DF_UNCERTAINTY_GUARD_K",
    "INSIGNIA_GLM53_DF_UNCERTAINTY_HOLD_ROUNDS",
    "INSIGNIA_GLM53_DF_EXACT_ROUNDS",
    "INSIGNIA_GLM53_DF_RETRY_TOP1_DROP",
    "INSIGNIA_GLM53_DF_CACHE_ROUTE_K",
    "INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN",
    "INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET",
    "INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS",
    "INSIGNIA_GLM53_DF_CACHE_GUARD_RETAIN",
    "INSIGNIA_GLM53_FORCE_TOKENS",
    "INSIGNIA_GLM53_FORCE_LOGITS_DUMP",
    "INSIGNIA_GLM53_FORCE_DF_LOGITS_DUMP",
)


ONE_SHOT_INSTRUCTION = """You are given a formal statement and its natural language description.
Generate a proof for the statement using Lean v4.29.0 and Mathlib. Under no
circumstance may you add axioms or assumptions, use `sorry` or `admit`, weaken
the statement, or change the formal statement. You may introduce helper lemmas
or definitions. This is a one-shot run with no verifier or search tools, so
check the proof mentally and return your best complete attempt. Output one Lean
code block ending with the exact theorem statement and its proof.

### Natural Language Problem Statement
{problem}

### Formal Statement
```lean
{formal_statement}
```"""


def chat_prompt(row: dict[str, Any]) -> str:
    instruction = ONE_SHOT_INSTRUCTION.format(
        problem=str(row["problem"]).strip(),
        formal_statement=str(row["formal_statement"]).strip(),
    )
    return (
        "[gMASK]<sop><|system|>Reasoning Effort: Max"
        "<|user|>" + instruction + "\n<|assistant|><think>"
    )


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    required = {"problem_idx", "problem", "formal_statement"}
    for ordinal, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise RuntimeError(f"ArXivLean row {ordinal} lacks {sorted(missing)}")
        row["_row"] = ordinal
    return rows


def theorem_name(row: dict[str, Any]) -> str:
    match = THEOREM_RE.search(str(row["formal_statement"]))
    return match.group(1) if match else "unknown"


def extract_lean(text: str) -> str:
    blocks = LEAN_BLOCK_RE.findall(text)
    return blocks[-1].strip() if blocks else ""


def structural_quality(row: dict[str, Any], text: str) -> dict[str, Any]:
    lean = extract_lean(text)
    expected = theorem_name(row)
    forbidden = sorted(set(re.findall(
        r"(?i)\b(?:sorry|admit|axiom|unsafe)\b", lean)))
    return {
        "has_lean_block": bool(lean),
        "theorem_name": expected,
        "has_expected_theorem": bool(lean and re.search(
            rf"(?m)^\s*(?:theorem|lemma)\s+{re.escape(expected)}\b", lean)),
        "forbidden_terms": forbidden,
        "structural_pass": bool(lean and expected != "unknown" and not forbidden and
                                re.search(r"(?m)^\s*(?:theorem|lemma)\s+" +
                                          re.escape(expected) + r"\b", lean)),
        "note": "Structural screen only; no Lean compiler was run.",
    }


def parse_run(output: str, tokenizer: Tokenizer) -> dict[str, Any]:
    ids_match = GREEDY_RE.search(output)
    timing = DFLASH_RE.search(output)
    prompt = PROMPT_RE.search(output)
    if not ids_match or not timing or not prompt:
        raise RuntimeError("engine output lacks greedy IDs, DFlash timing, or prompt timing")
    ids = [int(value) for value in ids_match.group(1).split()]
    display_ids = list(ids)
    for stop in (tokenizer.token_to_id("<|im_end|>"), 151643):
        if stop is not None and stop in display_ids:
            display_ids = display_ids[:display_ids.index(stop)]
    text = tokenizer.decode(display_ids, skip_special_tokens=False)
    prompt_tokens = int(prompt.group("tokens"))
    prompt_seconds = float(prompt.group("seconds"))
    return {
        "ids": ids,
        "text": text,
        "generated": int(timing.group("generated")),
        "rounds": int(timing.group("rounds")),
        "verify_k": int(timing.group("verify_k")),
        "accepted_per_round": float(timing.group("accepted")),
        "empty_rounds": int(timing.group("empty")),
        "decode_ms_per_token": float(timing.group("ms_token")),
        "decode_tokens_per_second": 1000.0 / float(timing.group("ms_token")),
        "prompt_tokens": prompt_tokens,
        "prompt_seconds": prompt_seconds,
        "prefill_tokens_per_second": prompt_tokens / prompt_seconds,
    }


def first_divergence(reference: list[int], candidate: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate), start=1):
        if left != right:
            return index
    return None if len(reference) == len(candidate) else min(len(reference), len(candidate)) + 1


def base_environment(args: argparse.Namespace, policy: str) -> dict[str, str]:
    environment = os.environ.copy()
    for key in POLICY_ENV:
        environment.pop(key, None)
    environment.update({
        "INSIGNIA_GLM53_Q8_BUDGET_MB": str(args.q8_budget_mb),
        "INSIGNIA_GLM53_EXPERT_CACHE_MB": str(args.cache_mb),
        "INSIGNIA_GLM53_READERS": str(args.readers),
        "INSIGNIA_GLM53_DFLASH2": "1",
        "INSIGNIA_GLM53_DFLASH2_FP8": str(args.dflash_fp8),
        "INSIGNIA_GLM53_DF_VERIFY_K": str(args.verify_k),
        "INSIGNIA_GLM53_DF_ADAPTIVE_K": "0",
        "INSIGNIA_GLM53_DF_BATCH_VERIFY": "1",
        **POLICIES[policy],
    })
    # Keep the harness switch a true A/B even when the engine automatically
    # selects full-prompt layer-major scheduling for multi-chunk prompts.
    environment["INSIGNIA_GLM53_PREFILL_FULL_LAYER_MAJOR"] = (
        "1" if args.prefill_full_layer_major else "0")
    environment["INSIGNIA_GLM53_PREFILL_APPROX_MOE"] = (
        "1" if args.prefill_approx_moe and policy != "exact" else "0")
    return environment


def engine_command(args: argparse.Namespace, prompt_file: pathlib.Path,
                   generate: int) -> list[str]:
    return [
        str(args.binary), str(args.model), str(args.index), f"@{prompt_file}",
        "0", str(generate), str(args.fp8),
    ]


def run_process(command: list[str], environment: dict[str, str], log: pathlib.Path,
                timeout: int) -> tuple[str, float]:
    started = time.monotonic()
    process = subprocess.run(command, env=environment, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             timeout=timeout)
    wall = time.monotonic() - started
    log.write_text(process.stdout, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"engine exited {process.returncode}; see {log}")
    return process.stdout, wall


def run_free(args: argparse.Namespace, prompt_file: pathlib.Path,
             tokenizer: Tokenizer, policy: str, case_dir: pathlib.Path) -> dict[str, Any]:
    output, wall = run_process(
        engine_command(args, prompt_file, args.generate),
        base_environment(args, policy), case_dir / f"{policy}.log", args.timeout)
    result = parse_run(output, tokenizer)
    result["wall_seconds"] = wall
    return result


def run_forced(args: argparse.Namespace, prompt_file: pathlib.Path,
               forced_file: pathlib.Path, policy: str, case_dir: pathlib.Path) -> pathlib.Path:
    dump = case_dir / f"{policy}-quality-logits.f32"
    environment = base_environment(args, policy)
    environment.update({
        "INSIGNIA_GLM53_FORCE_TOKENS": f"@{forced_file}",
        "INSIGNIA_GLM53_FORCE_LOGITS_DUMP": str(dump),
    })
    run_process(engine_command(args, prompt_file, 1), environment,
                case_dir / f"{policy}-quality.log", args.timeout)
    return dump


def parse_comparison(text: str) -> dict[str, Any]:
    def metric(name: str) -> float:
        match = re.search(rf"^{re.escape(name)}\s+mean\s+([0-9.eE+-]+)", text,
                          re.MULTILINE)
        if not match:
            raise RuntimeError(f"compare_logits output lacks {name} mean")
        return float(match.group(1))

    top1 = re.search(r"^top-1\s+agreement\s+([0-9.]+)%\s+\((\d+)/(\d+);", text,
                     re.MULTILINE)
    ppl = re.search(r"ppl A\s+([0-9.]+)\s+->\s+B\s+([0-9.]+)", text)
    if not top1 or not ppl:
        raise RuntimeError("compare_logits output lacks top-1 or PPL summary")
    ppl_a, ppl_b = float(ppl.group(1)), float(ppl.group(2))
    return {
        "cosine_mean": metric("cos"),
        "mse_mean": metric("mse"),
        "kl_mean": metric("kl"),
        "js_mean": metric("js"),
        "top1_percent": float(top1.group(1)),
        "top1_matches": int(top1.group(2)),
        "steps": int(top1.group(3)),
        "ppl_exact": ppl_a,
        "ppl_candidate": ppl_b,
        "ppl_delta_fraction": ppl_b / ppl_a - 1.0,
    }


def compare_quality(args: argparse.Namespace, forced_file: pathlib.Path,
                    exact_dump: pathlib.Path, candidate_dump: pathlib.Path,
                    policy: str, case_dir: pathlib.Path) -> dict[str, Any]:
    command = [
        sys.executable, str(args.compare_script), str(exact_dump), str(candidate_dump),
        "--tokens", str(forced_file), "--cos-threshold", "0", "--quiet",
    ]
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    comparison_log = case_dir / f"{policy}-compare.txt"
    comparison_log.write_text(process.stdout, encoding="utf-8")
    if process.returncode not in (0, 1):
        raise RuntimeError(f"compare_logits exited {process.returncode}; see {comparison_log}")
    result = parse_comparison(process.stdout)
    result["ppl_budget_fraction"] = args.max_ppl_delta
    result["ppl_gate_pass"] = result["ppl_delta_fraction"] <= args.max_ppl_delta
    return result


def write_report(path: pathlib.Path, row: dict[str, Any], policies: list[str],
                 results: dict[str, dict[str, Any]], quality: dict[str, dict[str, Any]],
                 tokenizer: Tokenizer) -> None:
    exact = results["exact"]
    lines = [
        f"# MathArena ArXivLean one-shot: problem {row['problem_idx']}", "",
        "This is an Insignia one-shot performance/quality workload, not an official "
        "MathArena score: no Lean verifier or search tools were exposed to the model.", "",
        f"- Parquet row: {row['_row']}",
        f"- Theorem: `{theorem_name(row)}`",
        f"- Source: `{row.get('source') or 'unlisted'}`",
        f"- Paper: {row.get('title') or 'unlisted'}", "",
        "## Performance", "",
        "| policy | first divergence | prefill tok/s | decode ms/tok | decode tok/s | accepted/round |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in policies:
        result = results[policy]
        divergence = first_divergence(exact["ids"], result["ids"])
        lines.append(
            f"| {policy} | {divergence if divergence is not None else '-'} | "
            f"{result['prefill_tokens_per_second']:.2f} | "
            f"{result['decode_ms_per_token']:.1f} | "
            f"{result['decode_tokens_per_second']:.3f} | "
            f"{result['accepted_per_round']:.2f} |"
        )
    if quality:
        lines += [
            "", "## Same-token full-vocabulary quality", "",
            "| policy | top-1 | cosine | MSE | KL | JS | PPL exact→candidate | delta | 3.5% gate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for policy in policies:
            if policy == "exact":
                continue
            item = quality[policy]
            lines.append(
                f"| {policy} | {item['top1_percent']:.2f}% | "
                f"{item['cosine_mean']:.6f} | {item['mse_mean']:.4g} | "
                f"{item['kl_mean']:.4g} | {item['js_mean']:.4g} | "
                f"{item['ppl_exact']:.4f}→{item['ppl_candidate']:.4f} | "
                f"{100.0 * item['ppl_delta_fraction']:+.2f}% | "
                f"{'PASS' if item['ppl_gate_pass'] else 'REJECT'} |"
            )
    lines += ["", "## Problem", "", str(row["problem"]), "",
              "## Formal statement", "", "```lean",
              str(row["formal_statement"]).rstrip(), "```"]
    for policy in policies:
        structure = structural_quality(row, results[policy]["text"])
        lines += [
            "", f"## {policy} output", "",
            f"Structural screen: **{'PASS' if structure['structural_pass'] else 'FAIL'}** "
            "(not compile verification).", "", "```text",
            results[policy]["text"], "```",
        ]
        if policy == "exact":
            continue
        divergence = first_divergence(exact["ids"], results[policy]["ids"])
        if divergence is None:
            continue
        begin = max(0, divergence - 13)
        stop = min(len(exact["ids"]), divergence + 12)
        lines += [
            "", f"### {policy} first-divergence window", "",
            f"Generated token {divergence}; decoded window {begin + 1}..{stop}.", "",
            "**Exact:**", "", "```text",
            tokenizer.decode(exact["ids"][begin:stop], skip_special_tokens=False), "```",
            "", f"**{policy}:**", "", "```text",
            tokenizer.decode(results[policy]["ids"][begin:stop], skip_special_tokens=False), "```",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=pathlib.Path,
                        default=pathlib.Path("/var/tmp/insignia-build/glm53-generate"))
    parser.add_argument("--model", type=pathlib.Path,
                        default=pathlib.Path("/var/lib/insignia/glm53-flash-text"))
    parser.add_argument("--index", type=pathlib.Path,
                        default=pathlib.Path("/var/lib/insignia/glm53-flash-text.index"))
    parser.add_argument("--fp8", type=pathlib.Path,
                        default=pathlib.Path("/var/lib/insignia/glm53-fp8-g64"))
    parser.add_argument("--dflash-fp8", type=pathlib.Path,
                        default=pathlib.Path("/var/lib/insignia/glm53-dflash2-fp8-fixed"))
    parser.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path(
        "/var/lib/insignia/bench-data/matharena/arxivlean-0326/data/"
        "train-00000-of-00001.parquet"))
    parser.add_argument("--compare-script", type=pathlib.Path,
                        default=pathlib.Path(__file__).with_name("compare_logits.py"))
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--row", type=int, action="append", default=[],
                        help="MathArena problem_idx (repeatable)")
    parser.add_argument("--policy", action="append", choices=sorted(POLICIES), default=[])
    parser.add_argument("--candidate-only", action="store_true",
                        help="run requested free-generation arms without inserting exact; requires quality=0")
    parser.add_argument("--generate", type=int, default=320)
    parser.add_argument("--quality-tokens", type=int, default=64,
                        help="same-token full-vocab comparison length; 0 disables")
    parser.add_argument("--max-ppl-delta", type=float, default=.035)
    parser.add_argument("--verify-k", type=int, default=4)
    parser.add_argument("--cache-mb", type=int, default=32768)
    parser.add_argument("--q8-budget-mb", type=int, default=10240)
    parser.add_argument("--readers", type=int, default=4)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--prefill-full-layer-major", action="store_true",
                        help="use the full-prompt layer-major prefill path")
    parser.add_argument("--prefill-approx-moe", action="store_true",
                        help="apply each non-exact policy's MoE pruning during layer-major prefill")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--show-problems", action="store_true")
    args = parser.parse_args()

    if args.generate < 2:
        parser.error("--generate must be at least 2 for DFlash timing")
    if args.quality_tokens < 0:
        parser.error("--quality-tokens cannot be negative")
    if args.max_ppl_delta < 0:
        parser.error("--max-ppl-delta cannot be negative")
    if args.candidate_only and (not args.policy or args.quality_tokens):
        parser.error("--candidate-only requires an explicit --policy and --quality-tokens 0")

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    rows = load_rows(args.dataset)
    by_problem = {int(row["problem_idx"]): row for row in rows}
    if len(by_problem) != len(rows):
        raise RuntimeError("ArXivLean problem_idx values are not unique")

    if args.list_only:
        ranked = []
        for row in rows:
            tokens = len(tokenizer.encode(chat_prompt(row)).ids)
            ranked.append((tokens, row))
        for tokens, row in sorted(ranked, reverse=True,
                                  key=lambda item: (item[0], item[1]["problem_idx"])):
            print(f"problem={row['problem_idx']:2d} parquet_row={row['_row']:2d} "
                  f"prompt_tokens={tokens:4d} theorem={theorem_name(row)} "
                  f"source={row.get('source') or '-'}")
            if args.show_problems:
                print("  " + re.sub(r"\s+", " ", str(row["problem"])).strip())
        return

    if not args.row or args.output is None:
        parser.error("generation requires --row and --output")
    missing = [problem for problem in args.row if problem not in by_problem]
    if missing:
        parser.error(f"unknown MathArena problem_idx values: {missing}")
    policies = args.policy or ["exact", "top6", "top6-cache"]
    if not args.candidate_only and "exact" not in policies:
        policies.insert(0, "exact")
    if len(set(policies)) != len(policies):
        parser.error("duplicate --policy")
    args.output.mkdir(parents=True, exist_ok=False)

    all_results = []
    any_rejected = False
    for problem_idx in args.row:
        row = by_problem[problem_idx]
        case_dir = args.output / f"arxivlean-{problem_idx}"
        case_dir.mkdir()
        prompt_ids = tokenizer.encode(chat_prompt(row)).ids
        if len(prompt_ids) + args.generate > args.max_context:
            raise RuntimeError(
                f"problem {problem_idx}: {len(prompt_ids)} prompt + {args.generate} generation "
                f"exceeds context {args.max_context}")
        prompt_file = case_dir / "prompt.csv"
        prompt_file.write_text(",".join(map(str, prompt_ids)) + "\n", encoding="utf-8")
        (case_dir / "prompt.txt").write_text(chat_prompt(row) + "\n", encoding="utf-8")

        results: dict[str, dict[str, Any]] = {}
        for policy in policies:
            print(f"ArXivLean problem {problem_idx}: {policy}", flush=True)
            results[policy] = run_free(args, prompt_file, tokenizer, policy, case_dir)
            results[policy]["structure"] = structural_quality(row, results[policy]["text"])
            print(f"  prefill {results[policy]['prefill_tokens_per_second']:.2f} tok/s; "
                  f"decode {results[policy]['decode_ms_per_token']:.1f} ms/token "
                  f"({results[policy]['decode_tokens_per_second']:.3f} tok/s)", flush=True)

        quality: dict[str, dict[str, Any]] = {}
        if args.quality_tokens:
            forced = results["exact"]["ids"][:args.quality_tokens]
            if len(forced) < 2:
                raise RuntimeError("exact run produced fewer than two quality tokens")
            forced_file = case_dir / "forced.csv"
            forced_file.write_text(",".join(map(str, forced)) + "\n", encoding="utf-8")
            dumps = {}
            for policy in policies:
                print(f"  same-token quality: {policy}", flush=True)
                dumps[policy] = run_forced(args, prompt_file, forced_file, policy, case_dir)
            for policy in policies:
                if policy == "exact":
                    continue
                quality[policy] = compare_quality(
                    args, forced_file, dumps["exact"], dumps[policy], policy, case_dir)
                gate = "PASS" if quality[policy]["ppl_gate_pass"] else "REJECT"
                print(f"    {policy}: cosine {quality[policy]['cosine_mean']:.6f}, "
                      f"MSE {quality[policy]['mse_mean']:.4g}, "
                      f"PPL {100.0 * quality[policy]['ppl_delta_fraction']:+.2f}% "
                      f"[{gate}]", flush=True)
                any_rejected |= not quality[policy]["ppl_gate_pass"]

        write_report(case_dir / "report.md", row, policies, results, quality, tokenizer)
        item = {
            "dataset": "MathArena/arxivlean-0326",
            "profile": "insignia-one-shot-v1",
            "problem_idx": problem_idx,
            "parquet_row": row["_row"],
            "source": row.get("source"),
            "title": row.get("title"),
            "theorem": theorem_name(row),
            "prompt_tokens": len(prompt_ids),
            "results": results,
            "quality": quality,
        }
        all_results.append(item)
        (args.output / "results.json").write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    (args.output / "campaign.json").write_text(json.dumps({
        "dataset": "MathArena/arxivlean-0326",
        "profile": "insignia-one-shot-v1",
        "official_matharena_score": False,
        "reason": "No iterative Lean verifier/search tools were exposed.",
        "policies": policies,
        "candidate_only": args.candidate_only,
        "generate": args.generate,
        "quality_tokens": args.quality_tokens,
        "max_ppl_delta": args.max_ppl_delta,
        "prefill_full_layer_major": args.prefill_full_layer_major,
        "prefill_approx_moe": args.prefill_approx_moe,
        "results": all_results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    if any_rejected:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
