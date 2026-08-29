#!/usr/bin/env python3
"""Stage route-trace campaign prompts from GSM8K + MATH-500 (+ the campaign prompt).

Writes, under --outdir:
  prompts/p<NN>.csv        one line, comma-separated token ids (engine @file input)
  prompts/manifest.tsv     id  dataset  row  prompt_tokens  gen  file

Selection: chat-wrapped prompts (same template as tools/benchmark_math.py),
length-filtered to [--min-tokens, --max-tokens], then evenly spaced across the
eligible row INDEX range so topics stay diverse (adjacent dataset rows are
often near-duplicates in difficulty/style). The fixed 16-token campaign prompt
(bench-df.sh) is emitted first as p00 for continuity with route-campaign.txt;
it is a known-atypical (repetitive-routing) prompt and is excluded from the
train/test split by the merge spec.
"""

import argparse
import json
import pathlib

from tokenizers import Tokenizer

CAMPAIGN_TOKENS = [154820, 11, 301, 2745, 941, 1516, 87, 29871,
                   526, 1052, 374, 123, 77, 918, 1520, 25]


def chat_prompt(problem):
    return (
        "[gMASK]<sop><|system|>Reasoning Effort: Max"
        "<|user|>Solve this problem. Show your reasoning and end with "
        "\\boxed{answer}.\n\n" + problem.strip() + "\n<|assistant|><think>"
    )


def pick(rows, count, lo, hi):
    eligible = [(i, ids) for i, ids in rows if lo <= len(ids) <= hi]
    if len(eligible) < count:
        raise SystemExit(f"only {len(eligible)} prompts in [{lo},{hi}] tokens, need {count}")
    picked = []
    for ordinal in range(count):
        # benchmark_math.py's bounded spread: never exceeds len(eligible)-1
        rank = round((ordinal + 1) * (len(eligible) - 1) / (count + 1)) if count > 1 else 0
        picked.append(eligible[rank])
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k", type=pathlib.Path,
                    default=pathlib.Path("/var/lib/insignia/bench-data/gsm8k/main/test-00000-of-00001.parquet"))
    ap.add_argument("--math500", type=pathlib.Path,
                    default=pathlib.Path("/var/lib/insignia/bench-data/math500/test.jsonl"))
    ap.add_argument("--tokenizer", type=pathlib.Path,
                    default=pathlib.Path("/var/lib/insignia/glm53-flash-text/tokenizer.json"))
    ap.add_argument("--outdir", type=pathlib.Path,
                    default=pathlib.Path("/var/lib/insignia/tracecampaign"))
    ap.add_argument("--per-dataset", type=int, default=8)
    ap.add_argument("--min-tokens", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=110)
    ap.add_argument("--gen", type=int, default=1250,
                    help="decode tokens per real-text prompt")
    ap.add_argument("--campaign-gen", type=int, default=500)
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    import pyarrow.parquet as pq
    gsm = pq.read_table(args.gsm8k).to_pylist()
    math = [json.loads(line) for line in
            args.math500.read_text(encoding="utf-8").splitlines() if line.strip()]

    rows = {"gsm8k": [], "math500": []}
    for i, row in enumerate(gsm):
        rows["gsm8k"].append((i, tok.encode(chat_prompt(row["question"])).ids))
    for i, row in enumerate(math):
        rows["math500"].append((i, tok.encode(chat_prompt(row["problem"])).ids))

    prompts_dir = args.outdir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    manifest = [("p00", "campaign", "-", len(CAMPAIGN_TOKENS), args.campaign_gen)]

    selected = {"gsm8k": pick(rows["gsm8k"], args.per_dataset, args.min_tokens,
                              args.max_tokens),
                "math500": pick(rows["math500"], args.per_dataset, args.min_tokens,
                                args.max_tokens)}
    ordinal = 1
    for dataset in ("gsm8k", "math500"):
        for index, ids in selected[dataset]:
            manifest.append((f"p{ordinal:02d}", dataset, str(index), len(ids), args.gen))
            (prompts_dir / f"p{ordinal:02d}.csv").write_text(",".join(map(str, ids)) + "\n",
                                                             encoding="utf-8")
            ordinal += 1
    (prompts_dir / "p00.csv").write_text(",".join(map(str, CAMPAIGN_TOKENS)) + "\n",
                                         encoding="utf-8")

    with (prompts_dir / "manifest.tsv").open("w", encoding="utf-8") as handle:
        handle.write("id\tdataset\trow\tprompt_tokens\tgen\n")
        for entry in manifest:
            handle.write("\t".join(map(str, entry)) + "\n")
    total = sum(entry[4] for entry in manifest)
    print(f"{len(manifest)} prompts, {total} decode tokens -> {prompts_dir}")


if __name__ == "__main__":
    main()
