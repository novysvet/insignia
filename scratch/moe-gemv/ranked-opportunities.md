# Ranked expert-GEMV opportunities (arithmetic + determinism-law impact)

Wall model: T_round ~ d(k) x b, b ~ 610 us = 14.156 MB / 23.2 GB/s pinned H2D
(progress.md:150; audits/s6-open-problems.md:59). GPU share of the round is
3.3-5.1% (k=8) / 1.5-2.3% (scalar) — see cost-model.md. Determinism law
(AGENTS.md): expert accumulation order and every FP reassociation are part of
the effective model; parity gate = greedy IDs + digit-identical top-10 logits
+ 30/40/100/240-token sequence checks.

## Rank 0 — more rows per expert call across the 8 verify positions: ALREADY LANDED

`moe_multi` collects each expert's users (glm53_generate.cu:3608-3619) and
serves up to 8 rows per weight pass (3620-3648) with bit-exact per-row chains.
I/O is already per-distinct-record (stage_layer over the union, 3489). Average
rows/expert at k=8 ~ 1.4 (measured union curve; see cost-model.md §5).
Expected further saving: **0**. (Raising the 8-row cap only matters for the
64-token prefill chunks, where GPU time is again not the wall.)

## Rank 1 — ship packed scales over PCIe, expand on device: −5.5% of b

The packed sidecar (`INSIGNIA_GLM53_PACKED_EXPERTS`, glm53_generate.cu:591,
960-1094) already stores scales nibble-packed with a 16-entry codebook +
escapes: 3 x 256 KiB instead of 3 x 512 KiB. Today the reader thread
re-expands them on the CPU (AVX2, `expand_scale_nibbles` 1017-1051) and the
H2D still ships the full 14.156 MB. Shipping the packed payload
(12 MiB nibbles + 0.75 MiB packed scales + escapes/codebooks ~ 13.37 MB) and
expanding on the GPU right after the copy-wait:

- b: 610 us -> 577 us (−33 us/record, −5.5%)
- k=8 round (1930 rec): **−64 ms (−5.5%)** on 1.16 s
- scalar token (336 rec): **−11 ms (−2.3%)** on ~480 ms
- Determinism: byte-exact expansion (integer nibble/codebook/escape work, no
  FP anywhere); GEMV bytes identical; stream order copy -> expand -> GEMV.
  Parity gate expected to pass trivially; still must be run.
- Patch sketch: `patch-1-packed-scale-h2d.md`. Prerequisite: the packed
  sidecar built for the 180.2 GiB glm-box store.

## Rank 2 — completion-order upload in the verify (ordered) path

Uploads currently follow batch order; `upload(slot)` CPU-blocks in
`wait_window` (glm53_generate.cu:805-810) even when a later slot's read is
already resident, idling the copy engine and SMs. In the ordered-accumulation
verify path (`ordered_accumulation` = kda_archive_, 3543, 3636-3659) expert
visit order is provably free: each (token, pick_slot) output is written once
to a private slot (c_expert_out_, sized at 2008) and combined afterwards in a
fixed token/slot loop (3652-3659). Expected saving bounded by
wall − d(k) x 610 us ~ 0-180 ms/round; realistically 20-60 ms (2-5%).
Determinism: SAFE in ordered mode only — the acc path (`_acc_quantized_rows`,
prefill w/o archive) bakes expert order into c_routed_ and MUST keep batch
order. Env-gated. Patch sketch: `patch-2-verify-completion-order.md`.

## Rank 3 — warp-specialized producer/consumer (H2D of next record under GEMV of current): ALREADY EXISTS at stream level

`upload()` issues the record H2D on a dedicated copy stream and the default
stream waits on `copy_done` before the GEMVs (glm53_generate.cu:841-847; the
arena swaps in a truly non-blocking copy stream at 1143; s6 landed "async
multi-slot H2D"). Duty cycles: copy engine ~610 us per 610+ us record (>=96%),
SMs 20-31 us (3-5%). GEMV(N) cannot start before H2D(N) completes (data
dependency), so kernel-internal warp specialization adds nothing; the SMs are
data-starved, not latency-bound. Expected saving: **~0**. The only direction
that matters is more bytes in flight, and a second copy stream cannot exceed
the ~23-25 GB/s WSL2 pinned ceiling already measured.

## Rank 4 — cp.async / __ldg / 128-bit loads: determinism-blocked and ~0 anyway

- Any LDG.128 widening of the weight stream requires each lane to own TWO
  ADJACENT groups (uint4 = groups 2g,2g+1); today lane l owns groups
  {l, l+32, l+64, ...} (glm53_expert_bench.cu:187, 470). Re-partitioning
  changes each lane's fmaf chain and the `warp_sum` butterfly inputs
  (67-86) => different FP rounding => **parity gate rejects by law**.
- Mapping-preserving widening is impossible (a lane's groups are 32 groups
  = 256 B apart; a cross-lane LDG.128+shfl variant moves the same bytes and
  adds shuffles).
- `__ldcs` (evict-first) is already the right policy for read-once weights;
  `__ldg`/`__ldca` would only pollute L2 that nothing re-reads.
- cp.async stages weights to smem for reuse; here every byte is consumed
  exactly once from registers (multi-row reuses weights in registers across
  R<=8 rows — strictly better than smem).
Expected saving: <= 2-5 us/record GPU-side, hidden under 610 us I/O => **~0
wall**, and the effective variants violate the determinism law. Revisit only
in a hypothetical VRAM-resident regime, and only as order-preserving staging.

## Rank 5 — L2 persistence (cudaAccessPolicyWindow) on the VRAM expert tier: REJECT

- Persisting-window cap is ~75% of the 48 MiB AD103 L2 ~ 36 MiB
  (cudaDevAttrMaxPersistingL2CacheSize; query at runtime) = **2.5 records**;
  the arena is 576 MiB-4.6 GiB.
- Within a round every record byte is touched exactly once (weights read-once;
  scales register-reused) — there is no intra-round reuse to pin.
- Cross-round reuse is already served at higher bandwidth by the VRAM tier
  itself (`device_index_` hit skips PCIe entirely, glm53_generate.cu:828-832).
Expected saving: **~0**. Rejected under the no-superstition rule
(measurement or arithmetic must back an optimization).

## Rank 6 — smem-staged scales: REJECT

Scales are 1.573 MB/record (11% of bytes) but are read with perfect warp
coalescing (32 consecutive code bytes per lane-step) and are ALREADY
register-reused across all R rows via `base_scale`
(glm53_expert_bench.cu:475/527). Staging them to smem moves no fewer bytes
and saves no repeat loads. Expected saving: **~0-1 us/record**.

## Meta-conclusion

Every kernel-side item is bounded by the GPU's 3-5% share of the round; the
wall is bytes-per-record over PCIe/NVMe times d(k). The only levers that move
today's number are: fewer bytes per record (Rank 1: −5.5%), removing
order-induced stalls that sit above the H2D floor (Rank 2), and the d(k)/tier
coverage problems already formalized as P1/P3/P4 in audits/s6-open-problems.md.
