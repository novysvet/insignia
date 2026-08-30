# progress

### 2026-08-30 (session 10) - MathArena ArXivLean frontier; cache-aware Top-6 survives

New hard-prompt default: `MathArena/arxivlean-0326` replaces MATH-500 for new
campaigns. The 41-problem CC BY-SA 4.0 Parquet is staged persistently on
glm-box. `tools/benchmark_matharena.py` implements an explicitly non-official
one-shot Lean 4.29 profile (the canonical benchmark supplies iterative Lean and
search tools), cold prefill/decode measurement, decoded-output reports, and
same-token full-vocabulary cosine/MSE/KL/JS/PPL gates. On problem 40, the
hardest by GLM prompt length (938 tokens), exact was 157.6 s prefill / 595.8
ms-token decode (1.678 tok/s). Plain Top-6 reached 499.9 ms/token (2.000 tok/s,
+19.2% throughput); cache-aware Top-6 reached 401.5 ms/token (2.491 tok/s,
+48.4%). The latter changed 9,518/14,448 rows within 0.001 router regret and
cut expert read-wait 307.1->241.6 s. Its 64-token forced gate was 60/64 top-1,
cosine 0.976259, MSE 0.3301, KL 0.01377, JS 0.003417, and PPL
1.1398->1.1496 (+0.86%), safely inside the user's 3.5% budget. None of the
320-token one-shot arms reached Lean code; that is expected on this frontier
and is not treated as a normal-prompt failure. Full findings:
`audits/s10-matharena-arxivlean.md`.

Two late CPU deliverables were triaged in the same wave. The DFlash2 ring
archive failed every gate and contained no applicable patch, though its
2048-slot/window semantics are useful future design material. The two-phase
body/tail I/O falsifier was rescued after fixing two validator/schema mistakes
in an isolated review copy and then run against the exact hash-locked 504-scale
and 608,044-route inputs. Verdict KILL: favorable gain only 20.1 ms/token
(2.41%), robust minimum -24.6 ms/token (-2.37%), perfect-oracle primary ceiling
2.40%. No production patch was applied.

Exact full-prompt layer-major prefill is now automatic for prompts exceeding
one configured chunk (normally >128 rows), with `...PREFILL_FULL_LAYER_MAJOR=0`
as an opt-out. On the 938-token ArXivLean problem 40, two-pair medians were
157.845->69.187 s, 5.94->13.56 tok/s: 2.28x throughput and 56.2% lower
latency. A matched run cut expert O_DIRECT 706.852->157.083 GiB and read-wait
144.063->31.659 s while host hits rose 15.9%->81.8%; ~5.56 GB of bidirectional
host spill/restore displaced 549.8 GiB of NVMe traffic. On the shortest
272-token ArXivLean prompt it improved prefill by 1.50-1.73x across two pairs.
A 40-token pair reproduced every ID and DFlash decision; decode was neutral
within 1.1% (509.7 vs 515.2 ms/token). Rebuilt auto-selection also reproduced
digit-identical top-10 logits. Commit `7ec54f4`.

### 2026-08-30 (session 10) - compute-heavy cache-aware Top-4 verifier

Full findings: `audits/s10-learned-falsifier.md`. Fixed Top-4 now composes with
a widened Top-32 cache router: keep the strongest three experts, choose the
fourth under normalized router-regret .0010, retain eight actions per row, and
exhaustively search `8^4` layer-union assignments on the i7-14700KF. Original
Top-8 weights/denominator are preserved. On MATH p12 it measured 237.0 ms/token
(4.22 tok/s) versus plain Top-4 at 374.0 and exact controls at 563.4/615.3;
acceptance improved 2.91->3.56. Forced quality was 31/32 top-1, cosine
0.974237, MSE 0.4281, KL 0.03129, JS 0.008713, and PPL 1.1282 vs exact 1.0651.
On GSM p02 it measured 283.7 vs plain Top-4 297.8 ms/token, removed 6.9% of the
Top-4 union, and its 32-token free trajectory unexpectedly matched exact even
though forced quality was 31/32. This is the selected aggressive speed arm,
not an exact-model default. A previous-logit JS/margin guard improved quality
but still diverged autoregressively. Packed persistent scales remain kept and
default-off: with this residency-sensitive approximate router they changed the
chosen tail experts, reduced acceptance, and were 13-14% slower in matched
phases despite the real +13-slot/PCIe benefit. These verifier policies do not
improve prefill. No overclock instability was observed.

### 2026-08-30 (session 10) - cross-head FP8 MLA; 314.5 MiB exact reclaim

Full findings: `audits/s10-cross-head-mla.md`. The exact first-256 MLA bridge
can now retain 512-wide FP32 latents and incrementally reconstruct K/V into one
shared layer-major buffer: 352 MiB became 37.5 MiB (314.5 MiB reclaimed,
292->315 observed expert slots), with decode and two-chunk FA2 prefix tests bit
exact. An opt-in compute-for-bandwidth mode now uses H8 cross-head E4M3 MMA for
long decode (1.48x at 2K, 1.93x at 4K, 2.66x at 8K) and a persistent fused H4
x Q8 E4M3 prefill kernel (1.31-1.69x for 128-row chunks, 1.53x at the 8K
boundary). Worst focused numerical quality was rel-L2 0.0061 / cosine
0.9999817; a real 300-token smoke stayed coherent and kept the first four IDs,
then diverged at token five. The user accepts this speed/quality trade, but the
knob remains explicit: `INSIGNIA_GLM53_MLA_CROSS_HEAD_FP8=1`. A two-pair cold
whole-engine prefill check was I/O-confounded (10.4-12.9 s expert read waits,
3.73-4.69 GB/s) and supports no wall claim. No broad campaign was run. The
4070 Ti SUPER overclock remained stable with no Xid/CUDA fault.

Adaptive router-mass pruning is now closed as an offline reject; see
`audits/s10-router-mass-pruning.md`. A checksum-verified 608,044-row analysis
and independent rerun found that a noncausal oracle saves only 2.878% of
records at 1% mean omitted mass, while 15% record reduction necessarily drops
at least 7.425% mean routed mass. Fixed top-6 has only a ~45.1 ms/token
transfer-channel upper bound under the production 80%-hit/4.7-GiB/s model yet
removes 15.86% of routed coefficient mass. No implementation or glm-box run is
warranted, even under the speed-first quality policy.

### 2026-08-29 (session 9) — compute-for-bandwidth MLA; +81 DFlash expert slots

Full findings: `audits/s9-reclaim-session.md`. Exact on-consumption MLA absorb
now reconstructs `W_uk`/`W_uv` coefficients from resident E4M3 + FP16 scales,
trading idle Ada ALU for removal of the 704 MiB FP32 duplicate. Exhaustive GPU
comparisons at prefixes 8/520/4096 found zero mismatches; real-model top-10 and
IDs stayed digit-identical. With Task-4's lazy Q8/scratch allocations and
compact 34-row KDA state, expert slots rose 316->383 scalar and 211->281
DFlash. Forced sequential verification now omits 145.6 MiB of unreachable
rollback snapshots, reaching 292 DFlash slots (+81, +38.4%) with hard guards
against batch rollback. Final glm-box GSM smoke accepted a 3-token block and
reproduced IDs `1986 374 264 4285`. A focused four-prompt A/B was 4/4 parity;
paired median ratios improved ~0.7% scalar and ~0.8% DFlash. The long factorial
was stopped and the harness reduced to the necessary two arms. Task 9 staged
verification was rejected at its 2.4369% robust maximin ceiling. Full-prompt
layer-major prefill also landed behind an env gate; one exact two-chunk sample
improved 8.570->7.188 s (16.1%). No overclock instability was observed.

### 2026-08-29 (session 8) — packed transport v2 (2D copies + fused expansion); CCT builder bug; U1 settled

Full findings: `audits/s8-gpu-expand-session.md`. Follow-up to 8b3018e
(GPU scale expansion). A 13-agent analysis wave + implementation wave:

- Landed (parity-green, env-gated default-off): `INSIGNIA_GLM53_PACKED_V2=1`
  merges the per-record H2D into one pitched `cudaMemcpy2DAsync` for the
  three 4 MiB bodies + one linear blob copy + ONE fused expansion launch
  (6 memcpy + 3 launches → 2 + 1); `INSIGNIA_GLM53_PACKED_KERNEL=2` selects
  the warp uint32 expansion kernel (one thread per packed word, 64-bit
  store, shared pair-table with escape flags, PRMT assembly, 8x fewer
  blocks, ~12% faster than the byte worker, byte-identical). Escape-tail
  bounds hardened before window writes (corrupt-record overflow risk).
  `glm53-expert-bench --expand` microbench added (synthesis at 5 escape
  rates, byte-exactness gates, kernel/host/transport/record-mix timing).
- Gates: base parity (unpacked == packed-CPU == packed-GPU at 8b3018e) and
  v2-knob parity all green — greedy IDs + full-vocab logits byte-identical
  (140 tokens x 2 prompts x 3 knob combos). Cold A/B: all packed transports
  within noise of each other (scalar ~405, DFlash k7 ~223 ms/tok); the
  packed store's ~+5% vs unpacked is the staging window memcpy, not
  transport — direct-read staging is the next lever. GPU expand removes
  129 us/record of AVX2 decode from the readers and 5.38% of PCIe bytes.
- **CCT builder was broken**: `np.argsort(-co)` on uint32 wraps — every
  table ever built was an anti-signal ranking (shipped cct-gsm8k.table is
  100% garbage rows). Fixed; `/var/lib/insignia/cct-campaign-v2.table`
  regenerated from 10 campaign prompts (11,750 tokens), row-verified.
  Corrected CCT = 28.4% OOS coverage at 1.0x overfetch (cap 8) — keep as
  a capped reader-pool hint, never a router head.
- Campaign analytics (16k tokens): U1 settled — the 5.22-bit entropy was
  small-sample bias (true ~7.3 bits, support ~110-120/layer); exchangeable
  models now OVER-predict U(K) by 7-16% (one-step stickiness beta~0.165
  explains 99%); static top-330 VRAM tier covers only ~15% of real decode
  access; pin-list v3 built (+15-18pp over pin-v2); static-vs-LRU phase
  transition at B=336 slots. Adaptive-k T(k) accounting corrected (verified
  round yields `matched` tokens, not 1+Sigma S) — at b=1.8 real text NO k
  beats scalar (1.026x best); break-even b*=1.74 at p-bar 3.08; estimator
  needs mandatory k-probes (else deadlocks at k=1, 11-46% regret). Drafter
  checkpoint is block_size 8 / window 2048 — block-16 hypothesis dead;
  DFLASH2_FP8 default path was dead on disk, repointed to -fixed.
- Next queue: direct-read packed staging (kill the 12.8 MiB window memcpy),
  VRAM-tier residency ordering (20-40 ms/tok), host-tier segment-LRU +
  acceptance-prefix demote (offline falsifier first), O(1) LRU, pinned
  router D2H (graph prerequisite), adaptive-k v2 with DF_COSTTRACE, CCT
  cold-prompt A/B, pin-list v3 adoption.

### 2026-08-29 (session 7) — 27-agent optimization wave; packed sidecar validated; s6 problems resolved on paper

Full findings and the unsolved-problems list: `audits/s7-optimization-wave.md`.
Landed (23a041a): packed-expert scale-expansion fix (loop was 3x too large),
packed expand instrumentation, empty-round EMA update, prev_routing_ re-keyed
on the accepted anchor row, drafter window guard + commit clamp (OOB KV writes
past 264). Overnight on glm-box: packed sidecar parity 5/5 green (codec is
bit-exact; driver fixes greedy-exact); packed A/B within cold-process noise;
route-trace campaign 13/17 prompts (~16k trace tokens, replaces the 5-token
pin-list evidence). Established: d(k) measured 4-9% below the union curve;
P7 80%-vs-28% = LRU retention-horizon law (verify inserts 6.3x faster); the
pin list was built from 5 tokens (top-28 91.5% is a 5-token artifact, 84.3%
at 60); P6 bound: 20 tok/s unreachable on 16 GB (PCIe carries all off-VRAM
bytes, oracle cap 12.4 tok/s, realistic 3.5-4.7 real text); prefill is in a
chunk-constant regime (~11.4 s/chunk, T=128 ≈ 90-120 ms/tok); zstd on
records dead (bodies 1.0000x at zstd-19, entropy 3.968/4 bits); cross-layer
routing MI ≈ 0-0.14 bits; drafter 264-window is an engine artifact
(checkpoint trained at window 2047 — prompts >263 tokens run pure scalar).
P1/P2/P3/P5/P6/P7/P8/P11 resolved on paper with implementations drafted in
scratch/ (host-tier SLRU, VRAM static fill, adaptive-k argmax, drafter
window, prefill-128, unified MLA kernel, KDA fusion, graphs). Open math:
U1 routing-regime inconsistency (exchangeable-model impossibility), U2
online survival estimation under policy feedback, U3 cost-decomposition
identifiability, U4 quant→router-flip model (P9, untouched), U5 cross-prompt
shift ν, U6 sequential-mode revival threshold, U7 the 32 GB frontier.

### 2026-08-28 (session 6) — speed campaign: I/O engineering round; 256K unblocked at P0

Open problems handed to a math-focused follow-up: `audits/s6-open-problems.md`
(adaptive-K optimality, expert-union model, hot-set allocation +
generalization, routing predictability bounds, MLA tile-merge bit-exactness
proofs, throughput lower bound, admission policy, prefill optimum, latent
quantization risk, acceptance prediction, KDA transplant proof, DSA
trade-off). Landed this session, every step parity-gated (greedy IDs and
top-10 logits digit-identical in all A/Bs): VRAM expert LRU tier (per-layer
segments, async multi-slot H2D on a non-blocking copy stream, ~-8% verify
round), whole-layer demand read staging in moe_multi, prefill chunk 32->64
(128-tok prompt prefill 43.7->30.4 s, -30%) with verify scratch decoupled
(kMaxVerify=8) and DFlash2 capture/commit extended to 64 rows, multi-row
NVFP4 expert GEMV chain (one weight pass serves up to 8 verify rows,
bit-exact), adaptive draft length from the acceptance EMA, trace-derived
static hot-expert pin list (host + VRAM, eviction-excluded; in-sample -15%
ms/token, out-of-sample -8%), runtime context limit to 262144 with @file
prompt input and drafter cutoff at position 263. Real-text baselines
(cold-process, 4 cases): scalar 570.9 ms/tok median, DFlash2-k7 627.5
(0.91x) at session start, 608.0 (0.94x) after chunk-64, ~454-515 on 128-tok
prompt runs with pins+retention; acceptance on real text 2.9-4.6/round vs
5.88 on the parrot prompt. Routing truth from ROUTE_TRACE on real text:
per-layer entropy 4.5-5.2 bits (not uniform), top-8/layer static coverage
41%, adjacent overlap 0.193; CCT split-sample 14.5%@1.28x — weak; 40 GiB
tier -7%; 2048-token prefill 369.4 s (180 ms/token) with the new stack.
Decode is bounded by the single NVMe serving ~70% misses on near-uniform
routing; the 20 tok/s target needs the P6 proof run either way. Operational:
WSL VM recycles kill long runs — benchmarks run via Windows Task Scheduler
(build/s6-inner.sh + C:\coding\s6-task.cmd reading /var/lib/insignia/s6-args);
parallel-session GPU contention is the default hazard.

### 2026-08-28 (parallel session) — DFlash2 verdict byte-closed; latent past-256 validated; CCT repaired; pilot medians

Full findings in `audits/quality-cct-session.md`. Short form: (1) three
independent exonerations on top of session-5's prompt-artifact verdict —
LEGACY=1 bit-identical histograms on the realistic prompt, an all-batch
12-token DF_DEBUG trace (the seq-verify latch never engaged), and a BF16
NumPy oracle + byte-exact FP8-cache re-quantization proving drafter numerics
and the regenerated cache are clean; the drafter simply ranks truth0
2nd/3rd on `prompt_math.txt` openings ("Step"/"**" vs "Let"/"\n\n").
(2) Published calibration (DFlash arXiv:2602.06036, EAGLE-3, SPEED-Bench):
2.0–2.7 ADPR at ~35% empty rounds = EAGLE-3-small-tree floor, low-normal;
healthy is 2.5–3.5 with 15–25% empties; FP8 drafter weights cost only
single-digit %. (3) GSM8K pilot (10 cases, k4, parity 10/10): scalar
545 ms/tok vs DFlash 612 ms/tok = 0.89x — k4 loses on real prompts at this
acceptance; math500 half + k7 pending (VM recycle killed the run).
(4) Latent MLA past position 256 validated at last: 500-tok prompt, greedy
16/16 identical FP8-vs-FP32 latent, cos 0.9957, PPL 1.456 vs 1.414 (+3.0%);
tooling `tools/compare_logits.py` + `tools/ppl.py` (dump format documented).
(5) CCT was dead on arrival — builder/loader format mismatch
(`IGCCT1\0` vs `CCT0`), `prev_routing_ == -1` wild row read on the first
step, prefetch queued ahead of demand, ungated under PREFETCH=0 — repaired
with a rewritten `tools/dump_cct.py`, loader header validation,
`INSIGNIA_GLM53_CCT_MAX` (default 8), demand-first ordering, prefetch gate;
table regeneration + A/B pending. Ops: glm-box has no GitHub push creds
(pushes hang; use the bundle relay); the WSL VM recycled twice; parallel
engine sessions contend for the GPU — `pgrep -af glm53-generate` first.

### 2026-08-28 (session 5) — DFlash2 "regression" resolved: prompt artifact, engine healthy

`audits/dflash2-regression-artifact.md`. The session-4 alarm (1.43
accepted/round on the bridge) reproduces identically on a **pre-bridge
binary** and under `MLA_LEGACY=1` — same histogram to the round — so the
bridge is exonerated. The 1.43 number belongs to the 5-token oracle prompt,
which parrots `200 200 ...`; the drafter cannot anchor on it (15/21 rounds
die at the d1 short-circuit). On the 16-token campaign prompt HEAD holds k4
3.70/round 228.7 ms/tok and k7 5.88/round 227.8 ms/tok at 100 gen — campaign
levels, parity intact in every A/B. Rule: judge DFlash2 only on the campaign
prompt or real prompts; the oracle prompt is parity-gate-only. Also landed:
glm-box's two unpushed commits (`bf577e6`, `c295638` — logits comparator,
PPL scorer, parameterized bench) reached origin via bundle+scp. Open queue
unchanged: GSM8K/MATH-500 campaign, latent-MLA >256 validation, CCT
prefetch; prefill remains expert-I/O-bound.

### 2026-08-28 (session 4) — glm-box online: 5.3 tok/s peak; latent-MLA bridge; DFlash2 regression OPEN

Full findings in `audits/mla-latent-session.md`. Short form: the 4070 Ti
SUPER box (ssh `glm-box`, worktree `C:\coding\Insignia-glm53-dflash2`) is the
performance machine now — 32 GiB pinned expert tier (2425 slots, 80% hits,
new engine default with halve-and-retry), PyTorch-free FP8 quantizers, FA2
verify-width boundary bug fixed. Best sustained: **k7 DFlash2 187.7 ms/tok
(5.33 tok/s), 194.4 ms/tok over 240 tokens, 56.5% faster than the 447 ms/tok
scalar baseline, bit-exact output**. The latent-MLA rework (512-wide FP8
group-scaled latent + absorbed attention, 8192 context) was diagnosed to
death: kernels/formula correct, failure is router sensitivity to ~1e-6
attention perturbation; the coherent shipping path is the **shadow bridge**
(exact expanded K/V for the first 256 positions, 352 MiB, latent populated
beyond) which reproduces the oracle 12/12. Determinism law discovered:
expert-accumulation order and softmax operation order are part of the
effective model — canonical-order MoE probe rejected, exact 256-token oracle
restored (`INSIGNIA_GLM53_MLA_LEGACY=1`). **OPEN**: on the bridge, DFlash2
acceptance collapsed to 1.43/round (516.7 ms/tok) — drafter/verify alignment
under investigation. GSM8K/MATH-500 harness (`tools/benchmark_math.py`)
staged on glm-box, campaign not yet run.

### 2026-08-28 (session 3 continued) — DFlash2 root causes found: acceptance 0 -> 5.0 (backfilled audit)

`audits/dflash2-fixes-session.md` documents the arc this progress file
skipped: batch-1 paired-FP8 API, `df_gather` column-split, and the quantized
FC strided-slice bug (`glm53-dflash2-fp8-fixed` is the good cache) took
layer-0 cosine 0.664 -> 0.9995 and acceptance to 5.00/round on realistic
prompts; ordered MoE accumulation + the KDA archive scatter fix made every
block size greedy-exact (k4 628.2 ms/tok, first speculative win over plain
decode); empty-round short-circuit cut 30-token decode 32.8%.

### 2026-08-28 (session 3) — DFlash2 drafter wired, parity-exact, acceptance 0 (in progress; superseded by session 3-continued and 4)

Full findings in `audits/dflash2-session.md`; paper digests + links in
`audits/papers-session3.md`. Short form: DFlash2 block
drafter implemented end-to-end in CUDA (FP8 VRAM-resident, 1.07 GiB, target
embed/lm_head shared), verify machinery reuses the MTP flow, committed
output stays greedy-exact with it enabled — but all rounds reject (1.00
accepted/round). Independent NumPy oracle ALSO predicts wrong tokens (truth
rank 81-1729), while engine-vs-oracle diverge inside drafter layer 0 (cos
0.66) — so there is (a) a drafter-forward kernel bug to bisect and (b) a
suspected feed problem (engine deep-layer residual drift poisoning the
layer-5/14/24/33/42 captures — would also explain the parked MTP failure).
Drafter proven robust to +30% capture noise, weakening the
abliteration-only explanation. Zero-context ablation shows context K/V are
connected. Do not measure speculative speedups until acceptance > 1.5
(empty rounds cost ~4.7 s vs 0.69 plain). CCT cross-layer prefetch loader
also landed (INSIGNIA_GLM53_CCT) fixing the tree's compile break; baseline
parity gate re-verified after all edits.

### MTP outcome: machinery works, draft layer predicts wrong (PARKED)

Greedy-exact parity holds through every variant (committed sequence ==
plain greedy always). But acceptance is ~0.05-0.2 tokens/round. Root-cause
trail: (1) the FP8 cache contains FABRICATED layer-45 entries (shared-expert
tensors that don't exist in the MTP layer) — its layer-45 region is corrupt;
INSIGNIA_GLM53_MTP_BF16=1 bypasses it (engine then matches a fully
independent NumPy oracle, tools/mtp_oracle.py, to within fp8/activation-quant
noise: both put the same wrong token at #2 with sharp confidence). (2) With
true weights, the oracle itself predicts confidently WRONG tokens from the
exact inputs the engine sees — so the layer itself is the problem on this
ABLITERATED checkpoint (abliteration direction-edits likely shifted the
hidden manifold the MTP was trained on). Layer-45 semantics distilled:
eh_proj([enorm(embed)|hnorm(mean-of-4-streams)]), no pos-0 embed zeroing
(GLM-5.3 differs from GLM-4.5 there — raw embed is the confident variant),
NoPE MLA, noaux_tc MoE w/o shared expert, shared_head.norm + tied lm_head,
recycle pre-norm hidden. Verify = prefill machinery + KDA snapshot/replay
rollback (rollback proven exact by parity through rescue rounds).

## 2026-08-28 (session 2) — MTP speculative decode + hierarchy research; C: incident + recovery

### measured facts (this session)

- pinned H2D (256 MiB, 13.56 MiB chunks): **23.2 GB/s**; D2H 23.9; unpinned H2D 3.0.
  PCIe is NOT the wall: full 4.43 GiB/token H2D would cost only ~190 ms.
- VRAM: 10.79 GiB free of 11.99 at idle (before engine allocations).
- 64-slot/no-cache forced run (tier sweep quoting bug): pure-NVMe decode =
  930 ms/tok at 5.81-5.84 GB/s steady; the disk ceiling is confirmed stable.
- LRU cliff on the 200-token trace (ideal sim): 379 slots 26-29%, 512 26%,
  591 53%, **672 69%**, 840 76%, 1024 82%. Static per-layer top-k within ~2
  points of global-hottest at every budget. 8 GiB tier ≈ 591 slots ≈ 53% hits
  (the pinned ceiling measured 6.6-9.25 GiB = Windows' ~50%-of-RAM lockable
  law; cudaHostRegister is dead on WSL2; GDS unsupported on WSL2).
- Cross-layer CCT (ST-MoE style, honest 60/40 split): coverage 73.7% of next
  layer's top-8 at 2.36x overfetch (N=8 candidates per expert); lift median
  2.0. Worth building as a latency hider.
- MTP dedup (K-token unions on the trace): bytes/token vs 1.0 = K2: 0.87,
  K3: 0.68, **K4: 0.58**, K6: 0.46, K8: 0.39; robust-ish on the
  non-repetitive half (K4: 0.61).
- MTP reference semantics distilled (vLLM glm4_moe_mtp/deepseek_mtp +
  transformers Glm5Next): input = eh_proj([enorm(embed) | hnorm(mean of 4 mHC
  streams, pre-final-norm)]); plain pre-norm residual block; NoPE MLA;
  noaux_tc routing, NO shared expert in layer 45; shared_head.norm then tied
  lm_head; recycle the PRE-norm hidden. GLM-5.2 reports accept ~4.5-5.5 with
  7 draft steps; layer 45 in this checkpoint is complete (2617 tensors).

### MTP implementation (landed, unverified as of this entry)

`src/glm53_generate.cu`: INSIGNIA_GLM53_MTP=K (2..8) enables greedy-exact
speculative decoding — one target verify forward per round (the prefill
machinery processes the K candidates with expert dedup), drafts from layer
45 via mtp_forward(), KDA recurrent-state rollback by snapshot + replay from
archived pre-conv projections, per-row argmax verification
(rows_argmax_kernel), pending-candidate scheme (pending is always the
target's own argmax ⇒ committed sequence identical to plain greedy).
Layer-45 MLA gets mla_slot_ 11; stager resident budget raised to 448 MiB for
eh_proj + layer-45 projections. moe_multi now admits ALL distinct verify
records to the host LRU (verify_populate_) but only the first 8 per layer
during prompt prefill.

### E: striping attempt — FAILED (see memory note wsl2-mount-and-vhdx-traps)

wsl --mount does not survive VM recycle: the 60-shard copy ran with the
mount gone and wrote ~80 GB into the Arch root vhdx on C:, filling the
drive; ext4 aborted read-only mid-copy. All "E: bandwidth" numbers measured
after the mount session are invalid (they measured C:). Recovery: junk
deleted, vhdx tar-export/reimport compaction (287.5 → ~162 GB expected).
Engine support landed: INSIGNIA_GLM53_ALT_SHARD_DIR opens any complete-size
shard from an override dir; tools/stripe_copy.py rate-limits the copy
(300 MB/s, single stream, fsync per shard) for the redo.

## 2026-08-28 (final) — decode 908 -> 690 ms/tok (1.32x), prefill 49.3 -> 8.0 s (6.2x)

Everything is parity-gated: after every change, the 12-token greedy run must
reproduce the baseline's greedy IDs AND digit-identical top-10 logits
(`2343:13.681516 2740:13.608133 ...`); 60/100/200-token runs reproduce their
greedy sequences. That held through every landed change.

| config                                               | decode (median) | prefill 16-tok |
|------------------------------------------------------|-----------------|----------------|
| baseline (Aug 27 morning build)                      | 908 ms/tok      | 49.3 s         |
| + pin_all + host LRU + pool + async + finite gating  | 765 ms/tok      | 8.4 s          |
| + fusions batch 1 (scale_add fold, conv3)            | 736 ms/tok      | 8.2 s          |
| + fusions batch 2 (mhc+rms, fp8 pair) + audit fixes  | 733 ms/tok      | 8.1 s          |
| + reader pool 12 -> 4 (virtio sweet spot)            | 690 ms/tok      | 8.0 s          |

Expert reads 5.5-5.6 GB/s steady; host-tier hits 27.5% (379 slots).

### reader-count sweep (60-token medians, 3 reps)

4 readers: 690 ms/tok / prefill 8.0 s. 6 readers: 723 / 7.9. 12 readers:
739 / 8.1. fio + pread probe agree: 4-8 outstanding multi-MiB O_DIRECT reads
is the virtio-blk ceiling (~5.8 GB/s); more threads contend. Engine default
is now 4 (INSIGNIA_GLM53_READERS overrides).

### CUDA 13 feature probe (src/cuda13_probe.cu, on sm_89/WSL2)

- Graph replay vs stream launches for 400 trivial kernels: 311 vs 3254 µs —
  WSL launch overhead is ~8.1 µs/kernel; our ~1500-2700 launches/token cost
  ~12-22 ms, so graphs could reclaim only that (skipped: decode is NVMe-bound).
- Captured memcpys from pinned memory + exec-node updates: PASS.
- Cross-stream event fork/join inside capture: WORKS functionally (CUDA 13
  records events as implicit dependency edges, not event nodes).
- Conditional nodes (new CUDA 13 handle API): FAIL on this stack.
- Device-side graph launch: FAIL on WSL2.

## deferred with analysis on file
- CUDA graphs: ~1190 pointer-stable launches capturable = only 8-12 ms of
  733; MoE descriptor-table graphs not worth the ABI churn while decode is
  NVMe-bound.
- Context 256 -> 1024: +1.4 GiB KV VRAM; MLA decode ~1.2-2 ms/token at
  P=1024; prefill kernel's static smem caps the idea at 4096 (128 KiB).
- NanoQuant full encoder: ~28-45 GPU-hours for all 12,384 experts, and it
  needs xnor/popcount decode kernels to pay off (pilot infra ready in
  oracle-venv; design notes in session records).
- DSA indexer (topk-2048 sparse attention past position 2048): weights
  present in the checkpoint, engine untouched; needs paged KV first.

### admission-control saga (all reverted; plain LRU wins)

Three variants measured against plain LRU (26.3% hits, 736 ms/tok):
- second-sight admission: 26.1% hits, 760 ms/tok (repetition-loop text admits
  everything anyway).
- count>=3 threshold: 25.5%, 782 ms/tok (thousands of keys eventually cross any
  fixed threshold; still thrash).
- TinyLFU door (admit iff candidate lifetime count > victim hit count, evict
  min-hits): 9.4%, 1052 ms/tok — counts rise together under near-uniform
  routing, the door congeals and nothing new enters; min-hits eviction churns
  newcomers. Even after reverting the door, leaving min-hits EVICTION active
  cost 9.8% hits — eviction must be pure LRU (stamp).
Conclusion: with near-uniform routing and a tier below 2 working sets, plain
LRU is the right policy; the pinned ceiling (6.6-9.25 GiB) caps the tier below
the 672-slot cliff where hits would jump to ~73% (simulated).

### fusion batch 2 (bitwise-parity verified)

- mhc_finalize_rms_kernel: RMSNorm folded into the mHC finalize launch; the
  variance reduction tree is a verbatim transplant of rms_bf16_kernel's, and
  the collapsed value is recomputed with the identical fmaf chain instead of
  a store/reload. -2 launches/layer (90/token).
- fp8_tc_gemv2: paired FP8 tensor-core GEMV (gate+up in one launch, one
  activation quantize; blockIdx.y selects the matrix). Used by
  Runner::linear_pair/compute_mlp for all dense/shared MLPs. -90 launches/token.
- Both verified digit-identical (12-token greedy IDs + top-10 logits exact).

### sub-4-bit verdict (three independent methods, all rejected)

| method                        | bpw    | cos vs NVFP4-dequant |
|-------------------------------|--------|----------------------|
| 2-bit uniform + int8 lowrank  | 2.2-2.4| 0.870-0.876          |
| int4-g64 RTN control          | 4.06   | 0.9916               |
| E8-lattice VQ + Hadamard + EF | 2.0-2.5| 0.671-0.681          |

Notable: the NVFP4-dequantized expert weights are already Gaussian (kurtosis
3.1), so Hadamard incoherence is a no-op — QuIP#'s remaining machinery
(LDFT fine-tuning, calibrated Hessians) is mandatory, i.e. days of offline
compute for a format that then needs new xnor/lattice decode kernels.
NVFP4 at 4.5 bpw stands as the right operating point for this engine; the
I/O win the sub-4-bit path chased is better delivered by the host-RAM tier.

## 2026-08-27 (late) — pipelined expert streaming: decode 908 -> ~0.74 s/tok, prefill 49 -> 8.4 s

All changes parity-verified: greedy IDs and top-10 logits digit-identical to the
pre-change engine on every run (12-token and 100-token checks).

### profiled baseline (before tonight's work)

- decode 908 ms/token steady: ~780 ms is routed-expert O_DIRECT (4.43 GiB/token
  at 5.28 GB/s), ~128 ms sync/serialization slop (90 finite-check syncs, 42
  router D2Hs, sync H2D per expert).
- 16-token prefill 49.3 s, of which 39 s was the FP8 matrix cache pinning at
  0.22 GB/s: lazy per-tensor buffered preads interleaved with expert O_DIRECT
  collapse on the WSL virtio-blk stack.

### engine changes

- Q8Index: O_DIRECT fd + read_rows_direct (aligned-window pread) +
  for_each_by_offset. Q8Stager::pin_all(): pins the whole 8.13 GiB FP8 cache
  upfront in on-disk order (~2.6 s, 3.3 GB/s) before anything else touches the
  disk. Prefill 49.3 -> 8.2-8.4 s.
- ExpertStager v3: the 24 streaming windows became a pinned host-RAM LRU tier
  (default 5 GiB / 379 whole-record slots; INSIGNIA_GLM53_EXPERT_CACHE_MB).
  WSL pinned ceiling measured between 6.6 and 9.25 GiB (9200 MiB request falls
  back to 4.6 GiB). Completed records stay resident; hits skip NVMe entirely.
- Reader pool: 12 persistent workers with demand-priority queues (demand
  records always jump ahead of speculative ones; FIFO prefetch measurably
  delayed demand reads).
- Async expert H2D on a dedicated copy stream + per-window copy_done events;
  default-stream GEMVs wait on the event; eviction/reuse syncs the event.
- Per-layer finite checks + per-layer printf now gated (INSIGNIA_GLM53_
  FINITE_EVERY_LAYER / PROFILE); one drain per step instead of 90.
- Kernel fusions: nvfp4_gemv_dp4a_acc_quantized folds the routing-weight
  scale_add into the down-GEMV epilogue (fmaf, bitwise-identical); kda_conv_silu3
  merges the three KDA conv+SiLU launches into one (launch count -8/layer MoE,
  -2/layer KDA).
- Second-sight admission control (INSIGNIA_GLM53_ADMIT=1 default): a record
  seen for the first time streams through without entering the LRU (window
  releases after its async copy drains via cudaEventQuery reaping).
- Routing-trace instrumentation: INSIGNIA_GLM53_ROUTE_TRACE=path dumps
  "token layer e0..e7 s0..s7" per sparse layer; tools/glm53_route_analysis.py
  analyzes overlap/LRU curves/entropy.

### measured (100-200 token greedy runs, same prompt as baseline)

| config                          | decode          | prefill 16-tok |
|---------------------------------|-----------------|----------------|
| baseline (Sep 27 morning build) | 908 ms/tok      | 49.3 s         |
| + pin_all + LRU + pool + async  | 765 ms/tok      | 8.4 s          |
| + kernel fusions                | 736 ms/tok      | 8.2 s          |
| + 488-slot tier (6.6 GiB)       | 744 ms/tok      | (no gain)      |

Expert O_DIRECT bandwidth 5.28 -> 5.55-5.72 GB/s (reader pool, no thread churn).

### routing locality (200-token traced run, greedy repetition-loop text)

- adjacent-token same-layer intersection 2.19/8 (27%); p@1 0.39; entropy 7.98
  of 8.17 bits -> routing is near-uniform (load-balanced training), little Zipf.
- global (layer,expert) LRU simulation on the trace: cliff at 2 working sets —
  <=512 slots ~26%, 768 slots 72.7%, 1024 slots 81.7%. Static-by-frequency
  oracle: 55.7% at 384 slots (2x plain LRU) -> admission control is the
  cheap win, not more RAM.
- real-text traces will be less repetitive; treat 72% as an artifact ceiling.

### sub-4-bit pilot (tools/nvfp4_2bit_pilot.py, torch CUDA on 4070S)

- 2-bit uniform + int8 low-rank residual (r=16..64): cos 0.87-0.876 vs NVFP4
  dequant reference at 2.16-2.44 bpw — REJECTED (need >=0.995).
- int4-g64 RTN control: cos 0.9916 at 4.06 bpw (sanity check passes).
- conclusion: sub-4-bit requires real QuIP#-style E8P12 lattice VQ with
  Hadamard incoherence or NanoQuant LB-ADMM, not RTN. Infrastructure ready
  (oracle-venv now has torch 2.13+cu126; ShardStore reader verified against
  all 120 shard headers).

### environment notes

- WSL /tmp is wiped on VM recycle (systemd-tmpfiles): write traces/states to
  /var/lib/insignia, never /tmp.
- `wsl -- bash /mnt/e/...` gets MSYS-path-mangled from Git Bash; always use
  `wsl -d Arch -- bash -c 'bash /mnt/e/...'`.
- vhdx is at C:\Users\Pufos\WSL\Arch (moved off E: on 2026-08-27; the old
  memory note about E:\WSL\Arch is stale).

## 2026-08-27 — GLM-5.3-Flash big model runs end-to-end; storage fixed

### storage

- the Arch WSL distro's vhdx was on `E:\WSL\Arch\ext4.vhdx`, so all "ext4" writes
  physically hit E: — it grew to 390 GB. deleted the stale model copies inside the
  guest, exported/unregistered/imported the distro to `C:\Users\Pufos\WSL\Arch`
  (28 GB vhdx on the 980 PRO). E: back to 464 GB free.
- big-model store: `/var/lib/insignia/glm53-flash-text` (120 shards, 180.2 GiB,
  text-only, byte-verified against the E: original which stays the source of
  truth) + `/var/lib/insignia/glm53-flash-text.index`.

### compact_glm53.py fixes (it silently corrupted data before)

- safetensors `data_offsets` are data-relative; the writer now places each tensor
  exactly at its declared offset (the old version aligned the absolute file
  position → every tensor ~46 KB off from its own header).
- per-shard fsync + posix_fadvise(DONTNEED): without it 180 GiB of dirty pages
  exhausts the WSL VM and 9p reads die with ENOMEM.
- 9p reads retry with backoff (transient ENOMEM); resume path returns the full
  tensor mapping (empty mapping poisoned the sidecar index).
- throughput 399 MB/s (8 workers, 4 MiB chunks).

### engine

- FP8 cache (`glm53-fp8-g64`, 8.13 GiB, 699 dense matrices) is the default
  8-bit path: GEMV 24.8 µs vs 91.9 µs BF16 (3.7x, 698 GB/s, cos 0.9994).
  E2M1-Q4 measured slower than FP8-TC (165 GB/s, no tensor cores) — rejected.
- Q8Stager VRAM residency (new): `INSIGNIA_GLM53_Q8_BUDGET_MB` pins whole
  matrices + lm_head (dedicated try_pin path for the 620 MB head). After the
  first token only routed NVFP4 experts still stream (4.43 GiB/token at
  5.5 GB/s O_DIRECT). lm_head 529 ms → 3.5 ms.
- logits digit-identical across E:-original / compacted-streaming /
  compacted-resident runs.

### numbers (45 layers, greedy)

| config                        | decode         | 16-tok prefill |
|-------------------------------|----------------|----------------|
| E: original drvfs (contended) | ~194 s/tok     | (OOM crash)    |
| C: compacted, FP8 streaming   | 2.82 s/tok @ 3.0 GB/s | 24.0 s |
| C: + cache pinned (10 GiB)    | 1.33 s/tok     | 15.7 s         |

toy 84M (oracle parity): decode 105.6 → 4.3 ms/tok (25x, residency);
prefill ~105 → ~9 ms/tok (12x, chunked layer-major prefill).

### 2026-08-27 remote workstation

- SSH alias: `glm-box` -> `desktop-hlvh09q` over Tailscale; Windows OpenSSH and
  Tailscale are automatic services, and Tailscale unattended mode is enabled.
- working copy: `C:\coding\Insignia` at `92e1028`, including the dirty tracked
  and untracked source state. Large checkpoints and build artifacts were not
  mixed into the repository transfer.
- Arch WSL uses `C:\coding\ext4.vhdx` with a 62 GB memory limit. CUDA 13.3,
  GCC 15, CMake, Ninja, Git LFS, Nsight tools, Python, NumPy, safetensors, and
  the official `hf` CLI are installed.
- toy checkpoint: `C:\coding\GLM-5.3-Flash-0.1B-A0.1B` (verified).
- the original source checkpoint at
  `C:\coding\GLM-5.3-Flash-UNCENSORED-NVFP4` was removed after compact-store
  revalidation, recovering 362.6 GiB. Text-only compact store:
  `/var/lib/insignia/glm53-flash-text` in the VHDX,
  120 shards / 112,727 tensors / 180.227 GiB, plus
  `/var/lib/insignia/glm53-flash-text.index` (10.29 MiB). The source passed
  Git LFS fsck and the compact output passed full header/bounds indexing.
- fresh shallow reference clones: llama.cpp, ggml, exllamav3, colibri, MLX,
  vLLM, and TensorRT-LLM. TensorRT-LLM LFS payload smudging is intentionally
  skipped because only its source is needed for kernel research.

### next

- LRU expert cache in leftover VRAM (~2.5 GB) — experts are the only remaining
  per-token I/O; routing has locality.
