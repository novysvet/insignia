# Session 7 — optimization wave: progress and open problems (2026-08-29)

Session shape: a 27-agent parallel analysis wave (code maps, math proofs, remote
data analysis, patch drafting) followed by a first implementation wave ("wave A")
landed on glm-box, plus an overnight validation pipeline. This file records what
was established, what landed, and the unsolved problems that remain — the math
ones in particular. Section references to s6 problems (P1..P12) are
`audits/s6-open-problems.md`.

## 1. What landed (commit 23a041a + overnight run)

- **Packed expert sidecar validated end-to-end.** The .igx builder finished on
  glm-box: 12,096 records, 150.77 GiB, logical ratio 0.94532x (0.782% scale
  escape rate), matching the 1%-escape model to 0.006%. The runtime had a
  blocker — `expand_scale_nibbles` looped over the 3-projection scale total
  (1.5 MiB) instead of per-projection 512 KiB — fixed by passing the
  per-projection byte count. Overnight gate: **5/5 parity checks green**
  (scalar vs packed codec on two prompts; DFlash2 k7 fixed vs packed;
  adaptive-k smoke). All greedy IDs digit-identical.
- **Packed A/B (cold-process, N=2, noisy):** scalar 457.1 vs 518.5 ms/tok,
  DFlash2 579.6 vs 566.0 ms/tok unpacked-vs-packed — packed is inside noise,
  possibly ~2% better on DFlash. Needs repeated medians before any claim.
- **Driver bug fixes:** (B1) empty rounds now update the acceptance EMA
  (previously k stayed inflated through empty streaks); (B2) `prev_routing_`
  is re-keyed on the accepted anchor row (`adopt_anchor_routing`) instead of
  the chunk's last (rejected-tail) row; (B4) drafter window guard is now
  `position + 1 + kBlock > kMaxCtx` (anchors 256..262 previously overflowed
  the attention tile); (B3) `DFlash2Drafter::commit` clamps to the 264-row KV
  cache (prompts >264 tokens previously wrote past layer 4's cache).
- **Instrumentation:** packed expand cost counters printed in the stats block.
- **Route-trace campaign:** 13/17 prompts completed (~16k decode tokens of
  `token layer e0..e7 s0..s7` traces) before the box slept; resume-safe at
  `/var/lib/insignia/tracecampaign/`. This replaces the 5-token pin-list
  evidence base (see §3.3).
- **Infrastructure:** wave-a pipeline (parity → A/B → campaign) runs under
  Windows Task Scheduler (`InsigniaWaveA` → `C:\coding\wave-a-task.cmd`);
  bench-matrix harness with resume + drift control in `scratch/bench/`.

## 2. Established facts (measured or derived this session)

- **d(k) is smaller than the committed-text union curve:** verify-batch unions
  measured 1067 records at k=4 (model 1109) and 1506 at k=7 (model 1689) —
  drafted candidates route 4-9% more similarly than consecutive real tokens.
  Real-text per-position survival: p1..p7 = .71/.66/.55/.42/.34/.26/.13.
- **The verify-round economics are regime-coupled:** b (ms/record) is ~1.8 on
  cold real-text (host tier ~15-26% hits) and ~0.61 on hot repetitive text
  (2425-slot LRU stays warm). The "~3.6 accepted/round break-even" from s6 was
  an artifact of pricing verify records at scalar economics; at verify
  economics (b≈0.61) k=2 beats scalar unconditionally for a≤150 ms; at b≈1.8
  real text sits at break-even and k*=1-2 (measured: GSM8K DFlash 0.77-0.95x
  of scalar cold-process, acceptance 1.6-2.5/round).
- **The 80%-vs-28% host-tier mystery (P7) is solved:** a global LRU serving a
  stream with insert rate r_ins per layer-token has retention horizon
  T=(C/42)/r_ins; scalar inserts 1.58/layer-token (T=36.6 tokens) while k=4
  verify unions insert 10/layer-token (T=5.8). The verify bursts age the tier
  6.3x. The per-layer quota never binds (57 ≥ U(7)); the TinyLFU door is
  bypassed by verify traffic entirely. Fix (drafted, unlanded): soft
  segment-LRU + acceptance-prefix admission; predicted verify hits 28-31% →
  45-60%.
- **The pin list was built from 5 tokens.** `pin-realtext.txt` derives from
  `route-realtext.txt` (5 decode tokens). All static hot-set anchors carry
  that provenance: top-28 = 91.5% coverage is a 5-token artifact (84.3% at 60
  tokens); split-sample OOS coverage of a 57-slot list is 69% pooled, and
  cross-text transfer collapses to 22-49%. Generalization theory (P3): the
  measured -15%-in-sample/-8%-out-of-sample fingerprint is exactly G=0.53,
  what a 4-6-token trace predicts; ≤1% generalization gap at B=57 needs
  ~90-1200 tokens (conservative: 1-2k across ≥8 prompts). Water-filling ≈
  equal split (≤0.5pp at operating budgets); the VRAM mirror should be static
  global-hottest (~+2pp) rather than per-layer top-k.
- **20 tok/s decode is unreachable on this box (P6):** every off-VRAM expert
  byte crosses PCIe (23.2 GB/s) because the pinned host tier is the L2 — host
  hits and disk legs both. With records/token ≈ 237-264 at acceptance 5-8,
  PCIe caps at 6.9-7.3 tok/s if the host tier were perfect; the clairvoyant
  ceiling is 12.4 tok/s (needs f_v ≥ 0.633 = top-13.8/layer in-sample = 7.2
  GiB beside an irreducible ~10.6 GiB of dense/drafter/latent VRAM —
  impossible at 16 GiB). Realistic: ~3.5-4.7 tok/s real text (2x today's
  real-text), 5.1-5.7 repetitive. 20 tok/s needs ~24-32 GB VRAM + 2nd NVMe;
  compute never binds (7.5-17 ms/token floor) below ~11 tok/s.
- **Prefill is in a chunk-constant regime (P8):** per-chunk wall ≈ 11.4 s
  regardless of T∈[16,64] (union saturates ~70-90 experts/layer; IO ~85%).
  ms/token ≈ 11.4 s/T → T=128 ≈ 90-120 ms/tok, T=256 ≈ 45-50; no interior
  optimum (the attention term only binds at T≈1340-1910). ~+66 MiB VRAM per
  doubling; patch drafted (chunk env knob, kMaxChunkCap=128, DFlash2
  kMaxTokens 64→128).
- **Kernel-launch census:** scalar step ≈ 2.7k launches + 42 router D2H sync
  drains; verify round (k=4) ≈ 9.4k launches + 42 drains; KDA per-token storm
  is 6 launches/token/layer (13k launches per 64-chunk = 24%); mHC
  analyze/mix per-token = 32%. The 42 per-round router D2H syncs are the
  structural capture breakers (graphs can only cover fragments; drafted
  drafter-capture patch saves ~1-2 ms/round).
- **I/O path forensic audit:** the default path is already optimal in the big
  pieces (disk→pinned direct single-copy, per-record async fenced H2D on a
  non-blocking copy stream, zero-copy VRAM hits, no tier-to-tier copying).
  Real remaining warts: per-record memcpyAsync API overhead (~16-27 ms/round),
  O(2425) LRU victim scans (~10-12 ms/round), device-resident records
  re-read from NVMe after host eviction (bursty), thundering-herd notify_all,
  pageable D2H downloads, 9-string index lookups per read, no write-combining
  on the pinned arena. Top-5 patch sketches in `scratch/ioaudit/`.
- **zstd on expert records is dead (measured):** E2M1 bodies are exactly
  incompressible (zstd-19 = 1.0000x with/without dictionary; nibble entropy
  3.9683/4 bits — a perfect coder saves 0.79%); scales would win only 0.9%
  more than the landed nibble codec at a real CPU cost. The landed codec is
  within 1.7% of the whole-record lossless ceiling. Do not revisit.
- **Cross-layer routing is essentially unpredictable (P4, tiny-trace
  caveat):** set-valued MI between adjacent layers' top-8 ≈ 0.00-0.14 bits
  (≤2.6% of set entropy, permutation-controlled). The existing pre-attention
  router hint (`early_route`) achieves 59.9% coverage at 1.0x overfetch with
  95.4% top-1 precision on same-token routing — that is the bar any
  drafter-feature router head must beat; CCT stays justified only as a
  capped reader-pool hint. SP-MoE-style prediction not justified on current
  evidence.
- **Drafter mechanism map (complete):** 5-layer Qwen3-style GQA backbone with
  two-tap dynamic convs; target features enter ONLY as drafter-attention K/V
  via the shared fc projection of 5 captures; block-diffusion drafts 7
  positions bidirectionally; greedy top-16 lattice selector with rank-256
  bilinear rescore; no temperature/sampling anywhere. The 264-position window
  is an engine allocation artifact — the checkpoint was trained with
  window_left=2047, so extending to 2048 restores speculation for every
  prompt >263 tokens (~1.2-1.75x whole-run; patch drafted, no parity gate
  needed since drafter numerics are speed-only). Block-16 possibility: if the
  checkpoint was trained at block 16, a 2x union discount is a constants
  change — CHECK the drafter config.json.
- **P5 (MLA bit-exactness) is solved on paper:** the unified 8-row split-tile
  kernel with per-row merge trip counts (`row_tiles=(pos+row+512)/512`)
  makes neutral partials provably never-read; decode executes an
  instruction-identical sequence; prefill/verify rows become bit-identical to
  decode at the same position by induction (Theorems 1+2 in
  `scratch/p5-proofs/`). One residual documented: acc elements equal to -0
  canonicalize to +0 in the fmaf path (unreachable from real data; avoided
  by construction anyway). Load-bearing axiom: `__expf(+0)==1.0f` — needs a
  startup probe. This closes the only known parity seam AND may fix the
  DFlash2 acceptance seam (verify captures currently differ numerically from
  decode).
- **P11 (KDA fusion proofs) closed:** (a) FP32 smem round-trip is bit-exact
  (PTX ld/st are pure moves; no extended precision on NVIDIA GPUs); (b)
  contraction invariance holds for identical expression trees in one TU —
  the entire recurrence/conv path is already fmaf-pinned, one residual
  (ptxas-level contraction of one unpinned tree) closed by a SASS diff; (c)
  conv tap order is current-first and a register-ring rewrite preserving it
  is specified. Fused kernel design: 25-70x on the KDA prefill portion,
  ~1.2-1.5 ms/chunk for all 34 layers vs ~40-105 ms launch-storm today.

## 3. Landed-in-scratch, not yet in the engine (wave-B queue, ranked)

All patches are env-gated default-off and were drafted against 23a041a;
several are adaptations of pre-23a041a drafts still pending (host-tier, VRAM
tier, adaptive-k v2, df-window, prefill128, packed-gpu — wave-B agent run hit
a quota wall; drafts exist in scratch/ but need revalidation against current
HEAD before landing).

1. Host-tier soft segment-LRU + acceptance-prefix admission
   (`scratch/host-tier/`) — the largest single software lever (P7 fix).
2. VRAM expert tier: headroom knob, smooth retry, static global-hottest fill
   (~330 slots), dead-scratch reclaim, pin-starvation guard
   (`scratch/vram-tier/`).
3. Adaptive-k v2: survival-curve 8-point argmax with censoring-correct EMA
   estimators (`scratch/p1-policy/`, `scratch/accept/`); replaces the 1.3x
   heuristic; k* is bimodal (8 on campaign-like text, 2-3 where the survival
   tail collapses).
4. Drafter KV window 264→2048 (`scratch/drafter/` sketch) — long prompts
   currently run pure scalar.
5. Prefill chunk 128 (`scratch/prefill128/`) — ~180 → ~90 ms/tok.
6. Unified MLA kernel (P5) — closes the parity seam; then CUDA-graph
   fragments for the dense chains (`scratch/graphs/`), KDA fused chunk
   kernel (P11), verify trusted-prefix cut (`scratch/seqverify/`), tail-skip,
   GPU-side scale expand (`scratch/moe-gemv/patch-1`), ioaudit top-5.
7. Quality-gated (lossy, like the FP8 latent cache was): FP8-TC MLA score
   path + DSA indexer for 256K (`scratch/fp8score/`); QTIP-style 3.3-3.5 bpw
   trellis bodies for COLD experts only, hot set stays exact XPR1
   (`scratch/quant/` — X1 experiment; ~-15% on the timed miss stream).

## 4. The hardest unsolved problems (math-weighted)

### U1. The routing-regime inconsistency (P2's deep version) — OPEN, foundational

Hard bounds show the measured statistics are mutually inconsistent under
every exchangeable (token-independent) single-mechanism model: (i) the
entropy bound — no per-layer marginal with access entropy ≤5.22 bits can
produce U(5)=31.40 while matching U(2)=14.45 (max U(5)=25.79 at 5.22 bits;
the moment-matched maximum needs 7.40 bits of entropy); (ii) skewed marginals
alone under-predict U(5) by 5-9 at every measured entropy; (iii) the best
fits (Pólya/Dirichlet-multinomial α≈90, rms 0.25%; hybrid flat-N +
stickiness) imply effective support 41-95 experts, while the entropy
measurement implies 23-37. Three resolutions: (a) the traces sample different
routing regimes (5-token realtext vs 60-token campaign vs 12-token math) and
the "inconsistency" is a measurement artifact; (b) routing has genuine
non-exchangeable temporal structure (self-exciting / state-dependent
processes) that no single-mechanism model captures; (c) both. The campaign
traces now on disk are exactly the joint same-trace measurement needed:
per-layer q̂ᵢ, Σq̂ᵢ², the lag profile ρ̃(w) for w=1..7, and stratified
U(K) on one trace. Until this is settled, d(k) extrapolations (which drive
the adaptive-k policy and prefill economics) carry an unquantified model
risk. If (b) holds, the right object is a marked point process for expert
sets, and nothing in the literature supplies it for 288-way top-8 routing.

### U2. Online survival-curve estimation under policy feedback — OPEN

The adaptive-k policy changes what the drafter conditions on (committed
context length, tier temperature, mode switching), so q̂ drifts in response
to the policy — an average-reward MDP, not a stationary stream. The landed
design assumes two-timescale separation. Open: prove (or empirically
establish) stability of the censoring-correct EMA + 8-point argmax +
hysteresis loop; characterize when the argmax oscillates (the T(k) surface
is provably non-unimodal for correlated drafters — counterexamples with 9-12%
losses exist even for exchangeable laws); design exploration that is safe
(k_t±1 probes waste a round each). The honest formulation is a restless
bandit with delayed, censored feedback; a clean treatment would also decide
when per-prompt speculation on/off (P10) should latch.

### U3. Identifiability of the cost decomposition — OPEN, blocks the policy

The scalar cost a_s + 336·b_s = 570.9 ms admits b_s ∈ [0.66, 1.25] — the
dense-vs-I/O split is unidentifiable from wall times alone (the two-point
verify calibration returned a negative intercept). The adaptive policy needs
b̂ online to pick the right break-even surface. Open: a rank-sufficient
online estimator separating dense compute from overlapped record I/O —
candidates: per-layer timing instrumentation (read_wait_seconds already
exists per stage), regression on round walls across varying d(k), or
event-based GPU accounting. This is the difference between the policy
landing on the b=0.61 surface (speculation almost always pays) and the
b=1.8 surface (real text is break-even).

### U4. Quantization noise → router flip probability (P9) — OPEN, unchanged

Nothing was built this session. The question decides whether NVFP4 latents
(0.75 GiB at 256K) can replace FP8 (1.4 GiB) and whether the exact-256
prefix bridge can be dropped. Needs: latent perturbation (known: cos 0.9957)
→ attention output perturbation → next-layer hidden perturbation → router
logit perturbation, compared against the empirical top-1..top-10 logit-gap
distribution (0.79-2.03 measured on drafter failures — target-side gaps
unknown), through 42 discrete top-8 cascades. The determinism law makes this
a 0/1 cascade, not a PPL question.

### U5. Cross-prompt shift ν — OPEN, caps every static cache

With p′ ~ Dir(ν·p), a perfect static hot set delivers 86%/94%/98.4% of
same-distribution coverage at ν=50/200/1000. The actual workload-mix ν is
unmeasured (needs the per-prompt campaign traces + a held-out prompt
family). This is the permanent tax on static pinning and the reason the
dynamic admission fix (U-host-tier) may dominate. Formalizing the optimal
static/dynamic split as a function of measured ν is open.

### U6. Sequential-mode revival threshold — OPEN, becomes live after wave-B

Batch verify dominates sequential iff a_t ≥ a and q_j ≥ θ_j for all drafted
positions; sequential becomes interesting only if the per-position dense
cost a_t < ~100 ms (a ~3x dense-path speedup). Wave-B (graphs, KDA fusion,
fewer drains) pushes a_t down; when it crosses, the seq/batch policy and the
trusted-prefix cut economics must be re-derived with measured constants.

### U7. The 32 GB frontier (P6 extension) — OPEN

If the box ever gets a 24-32 GB card: in-sample f_v=0.63 needs top-13.8/layer
(7.2 GiB) — but out-of-sample generalization requires top-18-22/layer.
Whether ANY admission policy (static + dynamic, given U5's ν) pushes
out-of-sample f_v past 0.63 at 32 GB decides if 20 tok/s becomes reachable
on upgraded hardware. This is a concrete instance of the general question:
what is the maximum fresh-distribution coverage of a cache of size C over
288^42 expert distributions with entropy ~5 bits — the P3 theory gives the
per-layer answer; the 42-layer capacity allocation with record-level dedup
(q-transform flattening) is not yet optimized.

### Solved this session (for the record, implementations pending)

P1 (marginal rule disproved; argmax/Dinkelbach correct; bimodal k*; batch
dominates sequential), P2 (best-fit model family + U(6..8) extrapolation
36.2/40.7/44.8, modulo U1), P3 (allocation ≈ equal split; generalization
fingerprint solved; shrinkage κ* = 1/(Σp̂²−1/288) ≈ 25-35 closed form),
P5 (shared-tree construction, proofs), P6 (bound assembled), P7
(retention-horizon law), P8 (chunk-constant regime, monotone T*), P11
(proofs). P12 has a full design (FP8-TC + DSA indexer) but no math unknowns
beyond quality gating. P10 is partially answered (survival profiles
measured; bimodality discovered; the prompt-statistic predictor is unbuilt).
