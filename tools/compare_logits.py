#!/usr/bin/env python3
"""A/B quality comparator for GLM-5.3 logits dumps (INSIGNIA_GLM53_LOGITS_DUMP).

Dump on-disk format (reverse-engineered from the writer in
src/glm53_generate.cu, lines 3274-3281):

    std::vector<float> host_logits(model_.vocab_size());                 // :3274
    ...
    if (const char *dump_path = std::getenv("INSIGNIA_GLM53_LOGITS_DUMP")) {
        static std::FILE *dump = nullptr;
        if (!dump) dump = std::fopen(dump_path, "wb");                   // :3279
        if (dump) std::fwrite(host_logits.data(), sizeof(float),
                              host_logits.size(), dump);                 // :3280
    }

  * The file is a bare concatenation of records: no header, no record
    separator, no padding. The file is opened once with "wb" (so it is
    truncated at the first dumped step) and each subsequent step appends.
  * One record = one decoding step that produced logits (the final prompt
    token plus every decode step; prefill-only chunk steps do not dump).
    Record k holds the full-vocabulary logits that predict the token
    following context position k.
  * Each record is `vocab_size` (= 154880 for GLM-5.3) little-endian
    float32 values in token-id order, exactly as downloaded from the
    device: post final-RMSNorm lm_head output, no softmax applied.
  * Record count = file_size / (vocab_size * 4). A trailing partial
    record means the writer died mid-fwrite; this tool ignores the tail
    with a warning.

Usage:
    python3 tools/compare_logits.py A.f32 B.f32 [--steps N] [--vocab V]
            [--tokens IDS.txt] [--topk K] [--cos-threshold T] [--quiet]

  A B              two dump files (raw float32 records, format above)
  --steps N        compare at most the first N records (default: all)
  --vocab V        floats per record (default 154880)
  --tokens FILE    text file of comma-separated token ids (same encoding
                   the engine consumes on its command line; whitespace and
                   newlines tolerated). Adds NLL/PPL of the reference
                   sequence under each dump. Alignment: if the file holds
                   records+1 ids, target(record k) = ids[k+1] (natural
                   engine alignment); if it holds exactly `records` ids,
                   they are taken as pre-aligned target(record k) = ids[k].
  --topk K         set-overlap / top-diff width (default 10)
  --cos-threshold  per-step full-vocab cosine floor (default 0.999)
  --quiet          suppress the per-step table, print summary only

Per step the tool reports: top-1 agreement, top-K set overlap, max/mean
abs logit difference over the union of the two dumps' top-K sets,
full-vocab cosine similarity and MSE. A summary (mean/median/max) is
printed at the end.

Exit codes: 0 = pass, 1 = quality failure (any compared step has cosine <
--cos-threshold or top-1 disagreement), 2 = usage/IO error.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import statistics
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


def record_count(path: str, vocab: int) -> tuple[int, int]:
    """Return (complete records, trailing partial bytes) for a dump."""
    size = os.path.getsize(path)
    rec = vocab * DTYPE.itemsize
    return divmod(size, rec)


def iter_records(path: str, vocab: int, limit: int):
    """Yield up to `limit` records as float64 arrays (streamed, O(1) memory)."""
    rec = vocab * DTYPE.itemsize
    with open(path, "rb") as fh:
        for index in range(limit):
            buf = fh.read(rec)
            if len(buf) < rec:
                break
            yield np.frombuffer(buf, dtype=DTYPE).astype(np.float64)


def logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


def distribution_divergences(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """Return KL(A||B), JS(A,B), logsumexp(A), and logsumexp(B)."""
    lse_a = logsumexp(a)
    lse_b = logsumexp(b)
    log_pa = a - lse_a
    log_pb = b - lse_b
    pa = np.exp(log_pa)
    pb = np.exp(log_pb)
    log_mix = np.logaddexp(log_pa, log_pb) - math.log(2.0)
    kl = float(np.sum(pa * (log_pa - log_pb)))
    js = 0.5 * float(np.sum(pa * (log_pa - log_mix)) +
                     np.sum(pb * (log_pb - log_mix)))
    return max(0.0, kl), max(0.0, js), lse_a, lse_b


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two GLM-5.3 logits dumps (INSIGNIA_GLM53_LOGITS_DUMP "
                    "format: raw little-endian float32, 154880 floats per step, "
                    "no header). See module docstring for the full format.")
    parser.add_argument("a", help="dump file A (reference)")
    parser.add_argument("b", help="dump file B (candidate)")
    parser.add_argument("--steps", type=int, default=None, metavar="N",
                        help="compare at most the first N records")
    parser.add_argument("--vocab", type=int, default=VOCAB_DEFAULT, metavar="V",
                        help=f"floats per record (default {VOCAB_DEFAULT})")
    parser.add_argument("--tokens", default=None, metavar="FILE",
                        help="comma-separated token-id file for NLL/PPL")
    parser.add_argument("--topk", type=int, default=10, metavar="K",
                        help="top-K overlap width (default 10)")
    parser.add_argument("--cos-threshold", type=float, default=0.999,
                        metavar="T", help="per-step cosine floor (default 0.999)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress the per-step table")
    args = parser.parse_args()

    if args.vocab <= 1:
        die("error: --vocab must be > 1")
    if args.topk < 1 or args.topk >= args.vocab:
        die("error: --topk must be in [1, vocab)")
    if args.steps is not None and args.steps <= 0:
        die("error: --steps must be positive")
    for label, path in (("A", args.a), ("B", args.b)):
        if not os.path.isfile(path):
            die(f"error: dump {label} {path!r} does not exist")

    steps_a, tail_a = record_count(args.a, args.vocab)
    steps_b, tail_b = record_count(args.b, args.vocab)
    for label, path, steps, tail in (("A", args.a, steps_a, tail_a),
                                     ("B", args.b, steps_b, tail_b)):
        note = f" (+{tail} trailing partial-record bytes ignored)" if tail else ""
        print(f"dump {label}: {path} -> {steps} records of {args.vocab} f32{note}")
    steps = min(steps_a, steps_b)
    if steps_a != steps_b:
        print(f"warning: record counts differ ({steps_a} vs {steps_b}); "
              f"comparing the common prefix of {steps} records")
    if args.steps is not None:
        steps = min(steps, args.steps)
    if steps == 0:
        die("error: no complete records to compare")

    k = args.topk
    token_ids = load_token_ids(args.tokens, args.vocab) if args.tokens else None
    targets: list[int] | None = None
    if token_ids is not None:
        if len(token_ids) == steps + 1:
            targets = token_ids[1:steps + 1]
        elif len(token_ids) == steps:
            print("note: token count equals record count; assuming ids are "
                  "already aligned as target(record k) = ids[k]")
            targets = token_ids
        else:
            die(
                f"error: {len(token_ids)} token ids vs {steps} records: expected "
                f"{steps + 1} ids (record k predicts ids[k+1]) or {steps} ids "
                f"(pre-aligned)")

    columns = (f"{'step':>5}  {'top1':>12}  {'ovlp':>7}  {'dmax':>10}  "
               f"{'dmean':>10}  {'cos':>9}  {'mse':>10}  {'kl':>10}  {'js':>10}")
    if targets is not None:
        columns += f"  {'nllA':>9}  {'nllB':>9}"
    if not args.quiet:
        print(columns)
        print("-" * len(columns))

    # Per-step metric series for the summary.
    cos_series, mse_series, dmax_series, dmean_series = [], [], [], []
    kl_series, js_series = [], []
    ovlp_series, top1_mismatches, cos_violations = [], [], []
    nll_a_sum = nll_b_sum = 0.0

    for step, (ra, rb) in enumerate(zip(iter_records(args.a, args.vocab, steps),
                                        iter_records(args.b, args.vocab, steps))):
        top1_a = int(np.argmax(ra))
        top1_b = int(np.argmax(rb))
        agree = top1_a == top1_b
        ka = np.argpartition(ra, -k)[-k:]
        kb = np.argpartition(rb, -k)[-k:]
        ovlp = len(np.intersect1d(ka, kb))
        union = np.union1d(ka, kb)
        d = ra[union] - rb[union]
        dmax = float(np.max(np.abs(d)))
        dmean = float(np.mean(np.abs(d)))
        na = float(np.linalg.norm(ra))
        nb = float(np.linalg.norm(rb))
        cos = 1.0 if na == 0.0 and nb == 0.0 else (
            0.0 if na == 0.0 or nb == 0.0 else float(np.dot(ra, rb)) / (na * nb))
        mse = float(np.mean((ra - rb) ** 2))
        kl, js, lse_a, lse_b = distribution_divergences(ra, rb)

        nll_a = nll_b = float("nan")
        if targets is not None:
            nll_a = lse_a - ra[targets[step]]
            nll_b = lse_b - rb[targets[step]]
            nll_a_sum += nll_a
            nll_b_sum += nll_b

        cos_series.append(cos)
        mse_series.append(mse)
        kl_series.append(kl)
        js_series.append(js)
        dmax_series.append(dmax)
        dmean_series.append(dmean)
        ovlp_series.append(ovlp)
        if not agree:
            top1_mismatches.append(step)
        if cos < args.cos_threshold:
            cos_violations.append(step)

        if not args.quiet:
            top1 = f"{top1_a}/{top1_b}" if agree else f"{top1_a}!{top1_b}"
            row = (f"{step:>5}  {top1:>12}  {ovlp:>3}/{k:<3}  {dmax:>10.3e}  "
                   f"{dmean:>10.3e}  {cos:>9.6f}  {mse:>10.3e}  "
                   f"{kl:>10.3e}  {js:>10.3e}")
            if targets is not None:
                row += f"  {nll_a:>9.4f}  {nll_b:>9.4f}"
            print(row)

    def summary(name, series, fmt):
        return (f"{name:<7} mean {fmt(statistics.fmean(series))}  "
                f"median {fmt(statistics.median(series))}  "
                f"max {fmt(max(series))}")

    print("-" * len(columns))
    print(summary("cos", cos_series, lambda v: f"{v:.6f}"))
    print(summary("mse", mse_series, lambda v: f"{v:.3e}"))
    print(summary("kl", kl_series, lambda v: f"{v:.3e}"))
    print(summary("js", js_series, lambda v: f"{v:.3e}"))
    print(summary("dmax", dmax_series, lambda v: f"{v:.3e}"))
    print(summary("dmean", dmean_series, lambda v: f"{v:.3e}"))
    print(f"ovlp   mean {statistics.fmean(ovlp_series):.2f}/{k}  "
          f"min {min(ovlp_series)}/{k}")
    agree_pct = 100.0 * (steps - len(top1_mismatches)) / steps
    print(f"top-1  agreement {agree_pct:.2f}% "
          f"({steps - len(top1_mismatches)}/{steps}; mismatches at steps "
          f"{top1_mismatches if top1_mismatches else 'none'})")
    print(f"cos    steps below {args.cos_threshold}: "
          f"{cos_violations if cos_violations else 'none'}")
    if targets is not None:
        for label, total in (("A", nll_a_sum), ("B", nll_b_sum)):
            print(f"nll[{label}] total {total:.4f}  "
                  f"mean {total / steps:.4f}  "
                  f"ppl {math.exp(total / steps):.4f}")
        print(f"nll delta (B-A) total {nll_b_sum - nll_a_sum:+.4f}  "
              f"ppl A {math.exp(nll_a_sum / steps):.4f} -> "
              f"B {math.exp(nll_b_sum / steps):.4f}")

    if top1_mismatches or cos_violations:
        reasons = []
        if top1_mismatches:
            reasons.append(f"top-1 disagreement on {len(top1_mismatches)} step(s)")
        if cos_violations:
            reasons.append(f"cosine < {args.cos_threshold} on "
                           f"{len(cos_violations)} step(s)")
        print(f"FAIL: {'; '.join(reasons)}")
        return 1
    print(f"PASS: {steps} step(s), top-1 100%, cosine >= {args.cos_threshold} "
          f"on every step")
    return 0


if __name__ == "__main__":
    sys.exit(main())
