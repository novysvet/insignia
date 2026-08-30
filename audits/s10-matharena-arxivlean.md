# Session 10: MathArena ArXivLean performance frontier

Date: 2026-08-30
Branch: `glm53-dflash2-4070ti-super`
Harness commit: `e9d2ee9`

## Benchmark contract

The new hard-prompt source is the Hugging Face dataset
`MathArena/arxivlean-0326`, not MATH-500. Its card reports 41 March-2026
problems under CC BY-SA 4.0. Each row contains an informal paper theorem and a
paired Lean 4.29 formal statement. The downloaded Parquet is 46,784 bytes and
is stored on glm-box at:

```text
/var/lib/insignia/bench-data/matharena/arxivlean-0326/data/train-00000-of-00001.parquet
```

The official MathArena configuration permits up to 200 calls to Lean
verification, submission checking, Loogle, LeanExplore, and persistent helper
files. Insignia currently exposes only text generation. Therefore
`tools/benchmark_matharena.py` labels its prompt profile
`insignia-one-shot-v1` and explicitly refuses to claim an official MathArena
score. Per the user's calibration (the best tool-enabled model scores only
37.50%), these are adversarial frontier prompts. Failure to finish a Lean proof
is not treated like failure on a standard math prompt.

The harness still gives three useful, reproducible gates:

1. Real long-prompt prefill and decode timing.
2. Decoded exact/candidate trajectories for qualitative comparison.
3. A fixed-prefix teacher-forced comparison of every vocabulary logit:
   top-1 agreement, cosine, MSE, KL, JS, and PPL delta.

Default accelerated arms are fixed Top-6 and Top-6 plus the compute-heavy
cache router (`K=32`, regret `0.0010`, eight joint actions). Candidate PPL may
increase at most 3.5% if the hard-prompt trajectory remains useful.

## Prompt sizing

The 41 one-shot prompts range from 272 to 938 GLM tokens. Problem 40 is the
largest at 938 tokens, followed by problem 1 at 848. Problem 40 formalizes an
integer GCD/LCM matrix factorization theorem and was chosen as the first
deliberately brutal screen, not as a representative expected solve.

Command:

```bash
/var/lib/insignia/bench-venv/bin/python \
  /mnt/c/coding/Insignia-glm53-dflash2/tools/benchmark_matharena.py \
  --row 40 \
  --output /var/lib/insignia/bench-results/matharena/arxivlean40-top6-screen-20260830a \
  --policy exact --policy top6 --policy top6-cache \
  --generate 320 --quality-tokens 64
```

No competing engine process was present. GPU clocks remained stable at about
2.94 GHz core and 12.251 GHz memory, 52 C, with no CUDA/Xid/token fault.

## Performance

| policy | first free divergence | prefill | prefill tok/s | decode ms/tok | decode tok/s | accepted/round |
|---|---:|---:|---:|---:|---:|---:|
| exact | - | 157.587 s | 5.95 | 595.8 | 1.678 | 2.52 |
| Top-6 | 14 | 157.939 s | 5.94 | 499.9 | 2.000 | 2.09 |
| Top-6 + cache | 39 | 158.556 s | 5.92 | 401.5 | 2.491 | 2.25 |

Cache-aware Top-6 is 32.6% lower latency and 48.4% higher throughput than
exact. It is 19.7% lower latency and 24.5% higher throughput than plain Top-6.
Prefill is unchanged because these policies act only in speculative target
verification.

The exact run reported 307.129 s of expert read-wait over 310.718 s expert
wall. The long, diverse trajectory reduced the 32 GiB host tier to 36.9% hits,
very different from the ~80% short-decode regime. Plain Top-6 reduced verified
expert-union records by 23.8% and read-wait to 270.746 s. Cache-aware Top-6:

- changed 9,518 of 14,448 routing rows with mean regret 0.000274 and maximum
  regret 0.001000;
- saved 4,692 immediate disk records and 2,084 immediate H2D records;
- removed a further 10.9% of joint layer-union records;
- reduced read-wait to 241.566 s, 21.3% below exact.

This is direct evidence that spending spare i7/GPU compute on resident-expert
selection manufactures effective storage bandwidth on high-entropy prompts.

## Full-vocabulary quality

All arms were teacher-forced through the same first 64 exact continuation
tokens. Each record compared all 154,880 float32 logits.

| policy | top-1 | cosine | MSE | KL | JS | PPL exact -> candidate | delta | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Top-6 | 62/64 | 0.978219 | 0.2989 | 0.01239 | 0.003081 | 1.1398 -> 1.1371 | -0.24% | PASS |
| Top-6 + cache | 60/64 | 0.976259 | 0.3301 | 0.01377 | 0.003417 | 1.1398 -> 1.1496 | +0.86% | PASS |

Top-6 mismatched top-1 at forced steps 29 and 38. Cache-aware Top-6 mismatched
at 6, 20, 29, and 38. The accepted speed/quality policy does not require
digit-identical approximate outputs; the exact mode remains available.

## Output reading

None of the three 320-token continuations reached a Lean code block. All three
correctly identified the task as a known GCD/LCM matrix divisibility result and
began reasoning about integer factorization through the GCD matrix and a
Mobius/meet-semilattice structure. None falsely declared success. Given the
benchmark's difficulty and the absence of iterative Lean tools, this run is a
relative trajectory/performance falsifier, not evidence about absolute solve
rate.

## Late deliverable triage

`session10-df-window-artifacts.tar.gz` reported `FAIL` on all of its own gates.
It had no CUDA patch, no oracle replay, no immutable reference extract, and no
compile result; its status explicitly forbids applying the blocked candidate.
Its useful residue is a proposed 2,048-slot DFlash ring semantics for contexts
beyond the current anchor-2040 cutoff. That design remains unimplemented and
must be rederived against current HEAD before use.

`handoff03-two-phase-falsifier.tar.gz` initially had 11/11 unit tests passing
but no campaign result. Its hash-locked inputs were already present locally.
Two isolated validator assumptions contradicted those exact inputs: the scale
archive intentionally has `body_bytes=0`, and the route manifest records
absolute source paths plus `-` null rows. After correcting only those schema
checks in the untracked review copy, both archives validated and the complete
CPU event simulation ran.

Two-phase result: KILL.

| design | favorable gain | favorable E2E | robust minimum | robust E2E | decision |
|---|---:|---:|---:|---:|---|
| blocking split body/tail | 19.591 ms/tok | 2.345% | -25.289 ms/tok | -2.422% | KILL |
| bounded io_uring CQE | 20.111 ms/tok | 2.407% | -24.645 ms/tok | -2.366% | KILL |
| impossible boundary oracle | 20.059 ms/tok | 2.401% | n/a | n/a | ceiling only |

One H2D engine and bursty concurrent body completions already hide most scale
tail latency. Splitting adds commands and can delay later bodies on the same
aggregate NVMe. No production code was changed.

## Disposition

- Replace MATH-500 with ArXivLean for new hard screens.
- Keep cache-aware Top-6 as the leading speed-first candidate; it survives the
  first ArXivLean PPL gate with substantial margin.
- Do not implement two-phase body/tail staging.
- Treat DFlash ring-window work as a separate future long-context project, not
  as a landed deliverable.
- Next prefill experiment: screen the existing full-prompt layer-major path on
  the 938-token ArXivLean prompt, because verifier pruning cannot improve the
  measured 157.6-second prefill.
