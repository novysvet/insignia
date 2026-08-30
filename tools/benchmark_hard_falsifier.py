#!/usr/bin/env python3
"""Focused exact/approximate free-generation differential on hard MATH prompts.

This deliberately complements benchmark_math.py instead of replacing it.  The
older harness measures short performance/parity slices; this one runs enough
tokens to expose recurrent approximation drift and detokenizes the answer so a
human can inspect whether the reasoning remains useful.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import time
from typing import Any

from tokenizers import Tokenizer


GREEDY_RE = re.compile(r"^greedy IDs(.*)$", re.MULTILINE)
DFLASH_RE = re.compile(
    r"(?P<generated>\d+) greedy tokens in (?P<rounds>\d+) DFLASH2-k(?P<verify_k>\d+) "
    r"rounds \((?P<accepted>[0-9.]+) accepted/round, (?P<empty>\d+) empty; "
    r"(?P<ms_token>[0-9.]+) ms/token;")
PROMPT_RE = re.compile(r"(?P<tokens>\d+)-token prompt (?P<seconds>[0-9.]+) s")


POLICIES: dict[str, dict[str, str]] = {
    "exact": {},
    "top4-cache": {
        "INSIGNIA_GLM53_DF_APPROX_TOPM": "4",
        "INSIGNIA_GLM53_DF_CACHE_ROUTE_K": "32",
        "INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET": ".0010",
        "INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS": "8",
    },
    "top4-cache-context": {
        "INSIGNIA_GLM53_DF_APPROX_TOPM": "4",
        "INSIGNIA_GLM53_DF_CACHE_ROUTE_K": "32",
        "INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET": ".0010",
        "INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS": "8",
        "INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN": ".05",
        "INSIGNIA_GLM53_DF_CALIBRATION_GUARD_JS": ".60",
    },
    "top4-margin075": {
        "INSIGNIA_GLM53_DF_APPROX_TOPM": "4",
        "INSIGNIA_GLM53_DF_LOGIT_GUARD_MARGIN": ".75",
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
    "INSIGNIA_GLM53_DF_CACHE_ROUTE_K",
    "INSIGNIA_GLM53_DF_CACHE_ROUTE_RETAIN",
    "INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET",
    "INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS",
    "INSIGNIA_GLM53_DF_CACHE_GUARD_RETAIN",
)


def chat_prompt(problem: str) -> str:
    return (
        "[gMASK]<sop><|system|>Reasoning Effort: Max"
        "<|user|>Solve this problem rigorously. Do not narrate false starts. "
        "Give a concise proof in at most 140 words and end with \\boxed{answer}.\n\n"
        + problem.strip()
        + "\n<|assistant|><think>"
    )


def difficulty(row: dict[str, Any]) -> int:
    match = re.search(r"\d+", str(row.get("level", row.get("difficulty", ""))))
    return int(match.group()) if match else -1


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        row["_row"] = index
        rows.append(row)
    return rows


def reference_answer(row: dict[str, Any]) -> str:
    for key in ("answer", "target", "final_answer"):
        if key in row:
            return str(row[key])
    solution = str(row.get("solution", ""))
    boxes = extract_boxes(solution)
    return boxes[-1] if boxes else ""


def extract_boxes(text: str) -> list[str]:
    boxes: list[str] = []
    marker = "\\boxed{"
    start = 0
    while True:
        found = text.find(marker, start)
        if found < 0:
            break
        depth = 1
        cursor = found + len(marker)
        content = cursor
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            boxes.append(text[content:cursor - 1])
        start = max(cursor, found + len(marker))
    return boxes


def parse_run(output: str, tokenizer: Tokenizer) -> dict[str, Any]:
    ids_match = GREEDY_RE.search(output)
    timing = DFLASH_RE.search(output)
    prompt = PROMPT_RE.search(output)
    if not ids_match or not timing or not prompt:
        raise RuntimeError("engine output lacks greedy IDs or DFlash timing")
    ids = [int(value) for value in ids_match.group(1).split()]
    display_ids = list(ids)
    for stop in (tokenizer.token_to_id("<|im_end|>"), 151643):
        if stop is not None and stop in display_ids:
            display_ids = display_ids[:display_ids.index(stop)]
    text = tokenizer.decode(display_ids, skip_special_tokens=False)
    boxes = extract_boxes(text)
    return {
        "ids": ids,
        "text": text,
        "boxed": boxes[-1] if boxes else "",
        "generated": int(timing.group("generated")),
        "rounds": int(timing.group("rounds")),
        "verify_k": int(timing.group("verify_k")),
        "accepted_per_round": float(timing.group("accepted")),
        "empty_rounds": int(timing.group("empty")),
        "decode_ms_per_token": float(timing.group("ms_token")),
        "prompt_tokens": int(prompt.group("tokens")),
        "prompt_seconds": float(prompt.group("seconds")),
    }


def first_divergence(reference: list[int], candidate: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate), start=1):
        if left != right:
            return index
    return None if len(reference) == len(candidate) else min(len(reference), len(candidate)) + 1


def run_policy(args: argparse.Namespace, prompt_file: pathlib.Path, tokenizer: Tokenizer,
               name: str, output_dir: pathlib.Path) -> dict[str, Any]:
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
        **POLICIES[name],
    })
    command = [
        str(args.binary), str(args.model), str(args.index), f"@{prompt_file}",
        "0", str(args.generate), str(args.fp8),
    ]
    started = time.monotonic()
    process = subprocess.run(command, env=environment, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             timeout=args.timeout)
    wall = time.monotonic() - started
    (output_dir / f"{name}.log").write_text(process.stdout, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"{name} exited {process.returncode}; see {name}.log")
    result = parse_run(process.stdout, tokenizer)
    result["wall_seconds"] = wall
    return result


def write_report(path: pathlib.Path, case: dict[str, Any], policies: list[str],
                 results: dict[str, dict[str, Any]], tokenizer: Tokenizer) -> None:
    exact = results["exact"]
    lines = [
        f"# Hard-prompt falsifier differential: MATH-500 row {case['_row']}", "",
        f"- Level: {difficulty(case)}",
        f"- Subject: {case.get('subject', case.get('type', 'unknown'))}",
        f"- Reference answer: `{reference_answer(case)}`", "",
        "## Problem", "", str(case["problem"]), "",
        "## Differential", "",
        "| policy | first divergence | boxed answer | ms/token | tok/s | accepted/round |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for name in policies:
        result = results[name]
        divergence = first_divergence(exact["ids"], result["ids"])
        lines.append(
            f"| {name} | {divergence if divergence is not None else '-'} | "
            f"`{result['boxed']}` | {result['decode_ms_per_token']:.1f} | "
            f"{1000.0 / result['decode_ms_per_token']:.3f} | "
            f"{result['accepted_per_round']:.2f} |"
        )
    for name in policies:
        if name == "exact":
            continue
        divergence = first_divergence(exact["ids"], results[name]["ids"])
        if divergence is None:
            continue
        begin = max(0, divergence - 13)
        stop = min(len(exact["ids"]), divergence + 12)
        lines += [
            "", f"## {name} first-divergence context", "",
            f"Generated token {divergence}; decoded token window {begin + 1}..{stop}.", "",
            "**Exact:**", "", "```text",
            tokenizer.decode(exact["ids"][begin:stop], skip_special_tokens=False), "```",
            "", f"**{name}:**", "", "```text",
            tokenizer.decode(results[name]["ids"][begin:stop], skip_special_tokens=False), "```",
        ]
    for name in policies:
        lines += ["", f"## {name} output", "", "```text", results[name]["text"], "```"]
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
    parser.add_argument("--math500", type=pathlib.Path,
                        default=pathlib.Path("/var/lib/insignia/bench-data/math500/test.jsonl"))
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--row", type=int, action="append", default=[])
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--show-problems", action="store_true")
    parser.add_argument("--policy", action="append", choices=sorted(POLICIES), default=[])
    parser.add_argument("--generate", type=int, default=160)
    parser.add_argument("--verify-k", type=int, default=4)
    parser.add_argument("--cache-mb", type=int, default=32768)
    parser.add_argument("--q8-budget-mb", type=int, default=10240)
    parser.add_argument("--readers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--fail-on-divergence", action="store_true")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    rows = load_rows(args.math500)
    if args.list_only:
        candidates = [row for row in rows if difficulty(row) == args.level]
        for row in candidates:
            ids = tokenizer.encode(chat_prompt(str(row["problem"]))).ids
            problem = re.sub(r"\s+", " ", str(row["problem"]))
            print(f"row={row['_row']:3d} level={difficulty(row)} "
                  f"subject={row.get('subject', row.get('type', '?'))!s:20.20} "
                  f"tokens={len(ids):3d} answer={reference_answer(row)!r}")
            if args.show_problems:
                print(f"  {problem}")
        return

    if not args.row or args.output is None:
        parser.error("generation requires --row and --output")
    policies = args.policy or ["exact", "top4-cache"]
    if "exact" not in policies:
        policies.insert(0, "exact")
    if len(set(policies)) != len(policies):
        parser.error("duplicate --policy")
    args.output.mkdir(parents=True, exist_ok=False)
    all_results = []
    any_divergence = False
    for row_index in args.row:
        case = rows[row_index]
        case_dir = args.output / f"math500-{row_index}"
        case_dir.mkdir()
        prompt_ids = tokenizer.encode(chat_prompt(str(case["problem"]))).ids
        prompt_file = case_dir / "prompt.csv"
        prompt_file.write_text(",".join(map(str, prompt_ids)) + "\n", encoding="utf-8")
        results: dict[str, dict[str, Any]] = {}
        for name in policies:
            print(f"row {row_index} level {difficulty(case)}: {name}", flush=True)
            results[name] = run_policy(args, prompt_file, tokenizer, name, case_dir)
            print(f"  {results[name]['decode_ms_per_token']:.1f} ms/token, "
                  f"boxed={results[name]['boxed']!r}", flush=True)
        for name in policies:
            if name != "exact" and first_divergence(results["exact"]["ids"],
                                                      results[name]["ids"]) is not None:
                any_divergence = True
        write_report(case_dir / "report.md", case, policies, results, tokenizer)
        all_results.append({
            "row": row_index,
            "difficulty": difficulty(case),
            "subject": case.get("subject", case.get("type", "unknown")),
            "reference_answer": reference_answer(case),
            "problem": case["problem"],
            "results": results,
        })
    (args.output / "results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    if args.fail_on_divergence and any_divergence:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
