#!/usr/bin/env python3
"""Standalone perplexity scorer for GLM-5.3 logits dumps.

Dump on-disk format (identical to tools/compare_logits.py; reverse-
engineered from the writer in src/glm53_generate.cu, lines 3274-3281):

    std::vector<float> host_logits(model_.vocab_size());                 // :3274
    ...
    if (const char *dump_path = std::getenv("INSIGNIA_GLM53_LOGITS_DUMP")) {
        static std::FILE *dump = nullptr;
        if (!dump) dump = std::fopen(dump_path, "wb");                   // :3279
        if (dump) std::fwrite(host_logits.data(), sizeof(float),
                              host_logits.size(), dump);                 // :3280
    }

  * The file is a bare concatenation of records: no header, no record
    separator, no padding. Opened once with "wb" (truncated at the first
    dumped step); every step that produces logits appends one record.
  * Record k holds the full-vocabulary logits that predict the token
    following context position k (post final-RMSNorm lm_head output, no
    softmax applied).
  * Each record is `vocab_size` (= 154880 for GLM-5.3) little-endian
    float32 values in token-id order.
  * Record count = file_size / (vocab_size * 4); a trailing partial
    record (writer died mid-fwrite) is ignored with a warning.

Usage:
    python3 tools/ppl.py DUMP.f32 TOKENS.txt [--vocab V] [--steps N]
            [--per-step]

  DUMP.f32    logits dump in the format above
  TOKENS.txt  text file of comma-separated token ids, the same encoding
              the engine consumes on its command line (e.g.
              "154820,9707,11,..."); whitespace and newlines tolerated.
              Alignment: if the file holds records+1 ids, target(record k)
              = ids[k+1] (the natural engine alignment: record k was
              produced after consuming ids[k]); if it holds exactly
              `records` ids, they are taken as pre-aligned
              target(record k) = ids[k].
  --steps N   score at most the first N records (default: all)
  --per-step  also print the NLL of each record

NLL uses the log-sum-exp trick in float64; PPL = exp(total_NLL / n_scored).

Exit codes: 0 = scored, 2 = usage/IO error (including token/record count
mismatch).
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

import numpy as np

VOCAB_DEFAULT = 154880
DTYPE = np.dtype("<f4")  # std::fwrite of float on x86-64 -> little-endian f32


def die(msg: str):
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def load_token_ids(path: str, vocab: int) -> list[int]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        die(f"error: cannot read token file {path!r}: {exc}")
    ids = [int(tok) for tok in re.findall(r"-?\d+", text)]
    if not ids:
        die(f"error: token file {path!r} contains no ids")
    for tid in ids:
        if not 0 <= tid < vocab:
            die(f"error: token id {tid} outside vocabulary [0, {vocab})")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute NLL/PPL of a reference token sequence under a "
                    "GLM-5.3 logits dump (INSIGNIA_GLM53_LOGITS_DUMP format: "
                    "raw little-endian float32, 154880 floats per step, no "
                    "header). See module docstring for the full format.")
    parser.add_argument("dump", help="logits dump file (raw f32 records)")
    parser.add_argument("tokens", help="comma-separated token-id file")
    parser.add_argument("--vocab", type=int, default=VOCAB_DEFAULT, metavar="V",
                        help=f"floats per record (default {VOCAB_DEFAULT})")
    parser.add_argument("--steps", type=int, default=None, metavar="N",
                        help="score at most the first N records")
    parser.add_argument("--per-step", action="store_true",
                        help="print the NLL of each scored record")
    args = parser.parse_args()

    if args.vocab <= 1:
        die("error: --vocab must be > 1")
    if args.steps is not None and args.steps <= 0:
        die("error: --steps must be positive")
    if not os.path.isfile(args.dump):
        die(f"error: dump {args.dump!r} does not exist")

    records, tail = divmod(os.path.getsize(args.dump), args.vocab * DTYPE.itemsize)
    note = f" (+{tail} trailing partial-record bytes ignored)" if tail else ""
    print(f"dump: {args.dump} -> {records} records of {args.vocab} f32{note}")
    if records == 0:
        die("error: no complete records in dump")

    token_ids = load_token_ids(args.tokens, args.vocab)
    if len(token_ids) == records + 1:
        targets = token_ids[1:records + 1]
        alignment = "record k predicts ids[k+1]"
    elif len(token_ids) == records:
        targets = token_ids
        alignment = "ids pre-aligned: record k predicts ids[k]"
        print("note: token count equals record count; assuming pre-aligned ids")
    else:
        die(
            f"error: {len(token_ids)} token ids vs {records} records: expected "
            f"{records + 1} ids (record k predicts ids[k+1]) or {records} ids "
            f"(pre-aligned)")
    if args.steps is not None:
        targets = targets[:args.steps]
        records = min(records, args.steps)
    print(f"tokens: {args.tokens} -> {len(token_ids)} ids, scoring {records} "
          f"record(s) ({alignment})")

    rec_bytes = args.vocab * DTYPE.itemsize
    total_nll = 0.0
    with open(args.dump, "rb") as fh:
        for step in range(records):
            buf = fh.read(rec_bytes)
            if len(buf) < rec_bytes:
                print(f"warning: dump ended early at record {step}")
                records = step
                break
            logits = np.frombuffer(buf, dtype=DTYPE).astype(np.float64)
            m = float(np.max(logits))
            lse = m + math.log(float(np.sum(np.exp(logits - m))))
            nll = lse - float(logits[targets[step]])
            total_nll += nll
            if args.per_step:
                print(f"  record {step}: target {targets[step]:>6}  "
                      f"nll {nll:.6f}")

    mean_nll = total_nll / records
    ppl = math.exp(mean_nll)
    print(f"records scored : {records}")
    print(f"total NLL      : {total_nll:.6f}")
    print(f"mean NLL/token : {mean_nll:.6f}")
    print(f"PPL            : {ppl:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
