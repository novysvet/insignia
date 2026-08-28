#!/usr/bin/env python3
"""Cold-process scalar/DFlash2 speed and parity on GSM8K + MATH-500 prompts."""

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import time

import pyarrow.parquet as pq
from tokenizers import Tokenizer


SCALAR_RE = re.compile(
    r"(\d+)-token prompt ([0-9.]+) s; (\d+) greedy tokens total ([0-9.]+) s")
DFLASH_RE = re.compile(
    r"(\d+)-token prompt ([0-9.]+) s; (\d+) greedy tokens in (\d+) "
    r"DFLASH2-k(\d+) rounds \(([0-9.]+) accepted/round, .*?; ([0-9.]+) ms/token;")


def chat_prompt(problem):
    return (
        "[gMASK]<sop><|system|>Reasoning Effort: Max"
        "<|user|>Solve this problem. Show your reasoning and end with "
        "\\boxed{answer}.\n\n" + problem.strip() + "\n<|assistant|><think>"
    )


def load_cases(args, tokenizer):
    gsm = pq.read_table(args.gsm8k).to_pylist()
    math = [json.loads(line) for line in args.math500.read_text(encoding="utf-8").splitlines()]
    datasets = {
        "gsm8k": [(row["question"], i) for i, row in enumerate(gsm)],
        "math500": [(row["problem"], i) for i, row in enumerate(math)],
    }
    maximum = 257 - args.generate
    selected = []
    for dataset, rows in datasets.items():
        eligible = []
        for problem, index in rows:
            ids = tokenizer.encode(chat_prompt(problem)).ids
            if len(ids) <= maximum:
                eligible.append((len(ids), index, problem, ids))
        eligible.sort()
        if len(eligible) < args.samples:
            raise RuntimeError(f"{dataset} has only {len(eligible)} eligible prompts")
        for ordinal in range(args.samples):
            rank = round((ordinal + 1) * (len(eligible) - 1) / (args.samples + 1))
            length, index, problem, ids = eligible[rank]
            selected.append({
                "dataset": dataset,
                "index": index,
                "prompt_tokens": length,
                "problem": problem,
                "ids": ids,
            })
    return selected


def parse_run(mode, output, wall_seconds):
    match = (DFLASH_RE if mode == "dflash" else SCALAR_RE).search(output)
    ids_match = re.search(r"^greedy IDs(.*)$", output, re.MULTILINE)
    if not match or not ids_match:
        raise RuntimeError(f"could not parse {mode} output")
    ids = [int(value) for value in ids_match.group(1).split()]
    if mode == "dflash":
        prompt_tokens, prompt_s, generated, rounds, verify_k, accepted, ms_token = match.groups()
        histogram = re.search(r"^  accepted histogram(.*)$", output, re.MULTILINE)
        result = {
            "rounds": int(rounds),
            "verify_k": int(verify_k),
            "accepted_per_round": float(accepted),
            "accepted_histogram": histogram.group(1).strip() if histogram else "",
            "decode_ms_per_token": float(ms_token),
        }
    else:
        prompt_tokens, prompt_s, generated, total_s = match.groups()
        result = {
            "decode_ms_per_token":
                1000.0 * (float(total_s) - float(prompt_s)) / int(generated),
        }
    result.update({
        "prompt_tokens": int(prompt_tokens),
        "prompt_seconds": float(prompt_s),
        "prefill_tokens_per_second": int(prompt_tokens) / float(prompt_s),
        "generated_tokens": int(generated),
        "wall_seconds": wall_seconds,
        "ids": ids,
    })
    return result


def run_mode(args, case, mode, output_dir):
    environment = os.environ.copy()
    for name in (
        "INSIGNIA_GLM53_DFLASH2", "INSIGNIA_GLM53_DFLASH2_FP8",
        "INSIGNIA_GLM53_DF_VERIFY_K", "INSIGNIA_GLM53_ALT_SHARD_DIR",
    ):
        environment.pop(name, None)
    environment.update({
        "INSIGNIA_GLM53_Q8_BUDGET_MB": str(args.q8_budget_mb),
        "INSIGNIA_GLM53_EXPERT_CACHE_MB": str(args.cache_mb),
        "INSIGNIA_GLM53_READERS": str(args.readers),
    })
    if mode == "dflash":
        environment.update({
            "INSIGNIA_GLM53_DFLASH2": "1",
            "INSIGNIA_GLM53_DFLASH2_FP8": str(args.dflash_fp8),
            "INSIGNIA_GLM53_DF_VERIFY_K": str(args.verify_k),
        })
    command = [
        str(args.binary), str(args.model), str(args.index),
        ",".join(map(str, case["ids"])), "0", str(args.generate), str(args.fp8),
    ]
    started = time.monotonic()
    process = subprocess.run(command, env=environment, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             timeout=args.timeout)
    wall_seconds = time.monotonic() - started
    stem = f"{case['dataset']}-{case['index']}-{mode}"
    (output_dir / f"{stem}.log").write_text(process.stdout, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"{stem} exited {process.returncode}; see its log")
    return parse_run(mode, process.stdout, wall_seconds)


def write_summary(path, results):
    lines = [
        "# GSM8K + MATH-500 inference performance", "",
        "| dataset | row | prompt tok | scalar ms/tok | DFlash ms/tok | speedup | "
        "accept/round | parity |", "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        scalar, dflash = item["scalar"], item["dflash"]
        speedup = scalar["decode_ms_per_token"] / dflash["decode_ms_per_token"]
        lines.append(
            f"| {item['dataset']} | {item['index']} | {item['prompt_tokens']} | "
            f"{scalar['decode_ms_per_token']:.1f} | {dflash['decode_ms_per_token']:.1f} | "
            f"{speedup:.2f}x | {dflash['accepted_per_round']:.2f} | "
            f"{'yes' if item['parity'] else 'NO'} |"
        )
    scalar_ms = [item["scalar"]["decode_ms_per_token"] for item in results]
    dflash_ms = [item["dflash"]["decode_ms_per_token"] for item in results]
    lines += [
        "", f"Scalar median: {statistics.median(scalar_ms):.1f} ms/token.",
        f"DFlash2 median: {statistics.median(dflash_ms):.1f} ms/token.",
        f"Aggregate median speedup: {statistics.median(scalar_ms) / statistics.median(dflash_ms):.2f}x.",
        "",
        "These are deterministic cold-process performance samples, not official accuracy scores.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
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
    parser.add_argument("--gsm8k", type=pathlib.Path, default=pathlib.Path(
        "/var/lib/insignia/bench-data/gsm8k/main/test-00000-of-00001.parquet"))
    parser.add_argument("--math500", type=pathlib.Path,
                        default=pathlib.Path("/var/lib/insignia/bench-data/math500/test.jsonl"))
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("/var/lib/insignia/bench-results/math"))
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--generate", type=int, default=32)
    parser.add_argument("--verify-k", type=int, default=7)
    parser.add_argument("--cache-mb", type=int, default=32768)
    parser.add_argument("--q8-budget-mb", type=int, default=10240)
    parser.add_argument("--readers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    cases = load_cases(args, tokenizer)
    if args.list_only:
        for case in cases:
            print(f"{case['dataset']} row={case['index']} tokens={case['prompt_tokens']}")
        return
    results = []
    for ordinal, case in enumerate(cases):
        item = {key: value for key, value in case.items() if key != "ids"}
        order = ("scalar", "dflash") if ordinal % 2 == 0 else ("dflash", "scalar")
        for mode in order:
            print(f"[{ordinal + 1}/{len(cases)}] {case['dataset']} row {case['index']} "
                  f"{case['prompt_tokens']} prompt tokens: {mode}", flush=True)
            item[mode] = run_mode(args, case, mode, args.output)
            print(f"  {item[mode]['decode_ms_per_token']:.1f} ms/token", flush=True)
        item["parity"] = item["scalar"]["ids"] == item["dflash"]["ids"]
        results.append(item)
        (args.output / "results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_summary(args.output / "summary.md", results)
        if not item["parity"]:
            raise RuntimeError(f"scalar/DFlash token mismatch for {case['dataset']} {case['index']}")
    print(args.output / "summary.md")


if __name__ == "__main__":
    main()
