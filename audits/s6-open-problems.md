# Session 6 open problems — mathematical handoff (2026-08-28)

This document is a handoff to a mathematically strong reader. It lists the
problems this session could not solve, mostly stochastic-optimization,
information-theory, and floating-point-exactness questions that determine the
next round of speedups for the Insignia GLM-5.3-Flash engine. Every problem
comes with the measured data that constrains it. Section 0 collects the
settled facts so nothing else is needed to work on these.

## 0. Context and settled facts

Engine: single RTX 4070 Ti SUPER (16 GiB), WSL2, one NVMe. Model: 42 sparse
MoE layers x 288 experts, top-8 per token, NVFP4 expert records of 13.56 MiB
(12,096 (layer,expert) keys total), DFlash2 block drafter (8 positions hard
cap, KV window 264), greedy-exact speculative decoding.

Measured this session (all on GSM8K/MATH-500-class text unless noted):

| quantity | value |
|---|---|
| scalar decode | 570.9 ms/token median (cold-process, 4 cases) |
| DFlash2 k7 decode, start of session | 627.5 ms/token (0.91x — loses to scalar) |
| DFlash2 k7 decode, after session's fixes | ~454-515 ms/token on 128-tok prompts |
| verify round wall | 0.99-1.23 s (of which >85% expert record I/O) |
| drafter forward + lm_head | ~17 ms/round |
| empty-round fallback step | 450-670 ms |
| accepted drafts/round, parrot "campaign" prompt | k4 3.70, k7 5.88 (7/7 in 76% of rounds) |
| accepted drafts/round, real GSM8K/MATH-500 | k4 mean 2.64; k7 2.91-4.57 per case |
| NVMe achieved (DFlash2 decode) | 3.6-3.8 GB/s; documented steady runs 5.45-5.84 GB/s |
| pinned host tier | 2425 slots; hits: 27.9-28.6% unpinned, 31.3% pinned (in-sample), ~42% (both arms) on a second prompt |
| VRAM expert tier | 321 slots; 0% global LRU, 3.2% per-layer segmented |
| PCIe pinned H2D raw | 23.2 GB/s |
| per-layer routing entropy (real text) | 4.54-5.22 bits (max = log2 288 = 8.17) |
| static hot-set coverage (per layer) | top-1 9.46%, top-8 41.07%, top-28 91.49% of accesses |
| adjacent-token same-layer overlap I/8 | mean 0.193 (repetitive half 0.266, non-repetitive 0.121) |
| union curve U(K) (distinct experts per layer over K consecutive tokens) | K=2: 14.45 (0.903 of 16), K=3: 20.61 (0.859), K=4: 26.40 (0.825), K=5: 31.40 (0.785) |
| CCT next-layer prediction, split-sample | 14.5% coverage at 1.28x overfetch (N=4), 24.1% at 2.45x (N=8), 31.3% at 4.58x (N=16); in-sample 64.9% at ~1x (overfit) |
| CCT co-activation lift | mean 3.28, median 2.50, max 5.0; share>=2 is 0.73 |
| FP8 latent cache quality | logit cos 0.9957, PPL +3.0%, zero greedy flips on one 500-token text |

What session 6 already landed (do not re-derive): VRAM expert LRU tier with
per-layer segments + async multi-slot H2D; whole-layer demand read staging in
prefill/verify; prefill chunk 32 -> 64 with verify scratch decoupled
(kMaxVerify=8); DFlash2 capture/commit extended to 64-row chunks; multi-row
(up to 8) NVFP4 expert GEMV chain, bit-exact; adaptive draft length heuristic
`k = clamp(int(1.3*accept_ema)+1, 2, k_max)`; trace-derived static pin list
(host tier, eviction-excluded, top-2-per-layer mirrored in VRAM); runtime
context limit to 262144 with @file prompt input and drafter cutoff at
position 263; prefill per-layer union retention (24/layer). Greedy IDs and
top-10 logits were digit-identical in every A/B (the determinism law held).

## P1. The optimal adaptive draft-length rule (the problem my heuristic fakes)

Maximize committed tokens/second

  T(k) = E[len(k)] / (a + b * d(k)),

where a ≈ fixed round cost (drafter 17 ms + verify dense path), b ≈ per-record
cost (≈ 0.60 ms/record at current bandwidth), d(k) = expected number of
distinct expert records the k-position verify pass touches, and E[len(k)] is
the expected accepted+bonus tokens from the DFlash2 block. Measured: d(1)=336,
and the union curve above gives d(k) ≈ 42 * 8 * k * ratio(k) with
ratio(2..5) = 0.903, 0.859, 0.825, 0.785 — but note these are single-layer
per-token-window numbers; the verify pass mixes 8 positions whose acceptance
correlates with rejection (the rejected tail's experts still get read in batch
mode). E[len] is NOT iid-geometric: the block drafter's per-position
acceptance within a round is strongly correlated (7/7 in 76% of rounds on
repetitive text; on real text 2.9-4.6 mean with 9/19 rounds empty on one
prompt).

Questions: (a) Given an arbitrary joint acceptance distribution P(m | k) and
the union curve d(k), is the optimal k characterized by a marginal
condition (stop when marginal expected tokens / marginal cost < current
rate)? Prove or disprove for non-iid P. (b) Fit P(m|k) and d(k) from the
measured histograms and solve for the optimal policy — including the choice
verify vs sequential mode, where sequential re-runs the target per position
but stops at first mismatch and skips the rejected tail's expert reads (its
cost is a random sum, not d(k)). (c) The break-even claim to check: with
scalar at 570.9 ms/token and a ≈ 17 + dense ms, speculation beats scalar only
if acceptance > ~3.6/round at current d(k); derive the exact threshold
surface.

What a solution buys: replaces my arbitrary 1.3x heuristic; worth an expected
5-15% on real text where acceptance straddles the break-even.

## P2. A first-principles model of the expert-union curve d(k)

The birthday model E[U] = 288(1-(1-1/288)^{8k}) badly under-predicts overlap
(measured U(2)=14.45 vs model 15.8 — fine — but the K-asymptotics with
adjacent overlap 0.19-0.27 and unknown higher-order correlation are not
modeled at all; the true process has (i) per-layer marginal skew ~5 bits,
(ii) adjacent-token overlap, (iii) layer-to-layer correlation). A correct
parametric model (Pólya urn? Markov chain over expert sets? empirical
process with exchangeable draws?) fit to U(2..5) would extrapolate the
K=6..8 tail needed for P1 and would also price the prefill chunk-size
trade (P8). The data (ROUTE_TRACE dumps, "token layer e0..e7 s0..s7" per
sparse layer) is on glm-box; more can be generated.

## P3. Optimal static hot-set allocation + generalization theory

Given per-layer empirical access frequencies (measured: entropy 4.5-5.2
bits; top-8 covers 41%, top-28 covers 91.5%), and a budget of B pinned
records (host) + D VRAM slots (device):

(a) Allocate B across 42 layers to maximize expected hit rate. Equal split
(8/layer) is what I shipped; is the optimum non-uniform (layers differ:
entropy 4.54 to 5.22)? Water-filling on the empirical tail?
(b) The generalization question, with the confusing evidence: pinning the
in-sample top-8 gave -15% ms/token in-sample but only -8% out-of-sample
(medians, n=2), and on the second prompt BOTH arms showed ~42% tier hits
(that prompt made the model parrot — 1.58 accepted/round, 9/19 empty — so
its routing was unusually repetitive). Formalize: traces of length n per
layer from an unknown distribution p_l, pick the empirical top-B_l; how does
E[hit rate on a fresh draw] converge, what smoothing (Dirichlet prior?)
maximizes it, and how many trace tokens are needed before the pin list is
worth -10% ms/token? Currently the list was built from ~6 warm tokens —
almost certainly too few; the theory should say how many to collect.

## P4. Is next-layer expert routing fundamentally unpredictable?

CCT (first-order Markov successor table over adjacent sparse-layer pairs)
measures split-sample coverage 14.5% at 1.28x overfetch. The lift
distribution (mean 3.28, capped at 5 by table construction) says co-activation
exists but the next top-8 SET is mostly not determined by the current set.
Questions: (a) estimate the conditional entropy H(top8_{L+1} | top8_L) vs
marginal H(top8_{L+1}) from the trace — is the mutual information so low that
NO routing-only predictor beats ~30% coverage at <=2x overfetch? (b) The
drafter reads target hidden states at 5 layers; SP-MoE (arXiv 2510.10302)
reports 88% top-1 expert prediction from draft features on other models. What
predicts the achievable recall given the hidden state's dimension (4096) and
the router logit gaps? An information-bottleneck-style bound or a concrete
fit would tell us whether to build a router-prediction head on drafter
features (the single biggest remaining I/O lever: converting the ~70% miss
stream into prefetch hits).

## P5. Bit-exactness proofs for the tiled MLA reduction tree (256K unblock)

The engine's determinism law requires decode and verify to share one FP
reduction tree. Plan (designed, not implemented): generalize the decode
split-tile online-softmax (512-key tiles + sequential merge,
glm53_ops.cu:1018-1134) to 8 query rows, route prefill AND decode through it,
with causally-inactive tiles writing neutral partials (max = -FLT_MAX,
den = 0, acc = 0). Needed proofs, under --use_fast_math (FTZ on, __expf):

(a) Neutral-partial no-op: show merge(acc, m, den; neutral) returns bit-wise
(m, den, acc) including sign-of-zero, denormals, and the fmaf(src, 0,
acc*correction) path.
(b) Causal-skip equivalence: online softmax state after processing keys
0..p by "skip keys > p" equals the decode path's state, by induction on
tiles — i.e. skipping issues no update, so the update SEQUENCE matches.
(c) The merge reassociation `acc = fmaf(tile_acc, tile_correction, acc*correction)`
is a different summation tree than single-pass; quantify the worst-case ULP
divergence as a function of context length (empirically it flipped nothing
at 500 tokens, but the parity gate needs a bound or a shared-tree
construction).

A clean write-up here unblocks the 256K prefill kernel (current prefill
kernel is O(position) sequential per block, ~100x too slow at 256K) and
removes the only known parity gap.

## P6. A defensible throughput lower bound (is 20 tok/s provably unreachable?)

Assemble a proof: minimize expected ms/token over ALL admission/prefetch/
speculation policies subject to (i) the measured marginal+overlap routing
statistics, (ii) NVMe 4.75-5.8 GB/s, (iii) 2425-slot pinned tier with the
measured hit curve, (iv) 23 GB/s PCIe, (v) the DFlash2 8-position block cap
(records/token floor ≈ 42*8*0.785 ≈ 264 even at perfect acceptance).
My session's arithmetic says the answer is ~6-10 tok/s decode on real text
(i.e. the user's 20 tok/s target is out of reach without a second drive or
a fundamentally different memory hierarchy), but the argument was corrected
twice mid-session (hit-rate and bandwidth figures were mis-mixed); it
deserves a rigorous treatment. If 20 IS reachable, the proof will show where
the slack is.

## P7. Two-tier admission policy under bursty verify unions (the 80% vs 28% mystery)

Scalar decode documents 80.3% hits on the same 2425-slot tier where DFlash2
verify measures 28-31%. Candidate causes: the verify union (~1,300-2,000
records/round) evicts the scalar working set; the per-layer quota admits
only ~8/layer; batch verify reads the whole union before the first mismatch
is known. Formalize as a cache serving a request stream = superposition of
(a) a high-locality token stream (adjacent overlap 0.19-0.27) and (b)
bursty 8x-correlated verify unions, and derive the admission/retention
policy (quota? TinyLFU door? union-prefix admission in sequential mode?)
that maximizes combined hits. The 40 GiB variant measured -7% ms/token;
explain why the marginal value of +600 slots is that large at 2425 but flat
in the documented scalar runs (pinning pressure was the recorded suspect).

## P8. Prefill chunk-size optimum

Prefill wall ≈ compute(T) + IO(T), IO(T) ∝ 42 * U(T) * 13.56 MiB at the
achieved bandwidth, plus a KDA per-token sequential term (2 launches/token/
layer until the fused kernel lands) and a dense-stream amortization term.
Measured: T=16 -> 700 ms/token cold, T=32 -> ~356, T=64 -> ~180 (2048-token
prompt, 369.4 s, with pins+retention). Given U(K) from P2 and the measured
component times, derive T* and say whether T=128 or 256 pays (VRAM cost of
chunk buffers is known: ~+30 MiB per doubling once verify scratch is
decoupled).

## P9. Latent-quantization -> routing-flip risk (NVFP4 latent viability)

FP8 (e4m3, group-64) latents cost cos 0.9957 / +3.0% PPL / zero flips at 500
tokens. To fit 256K context in VRAM (FP8 latent = 1.4 GiB; NVFP4 would be
~0.75 GiB) we need the probability that a latent perturbation of a given
cos-distance flips a router top-8 set (42 layers deep, each flip cascading —
the determinism law exists because of this) and/or flips greedy argmax over
154880 logits (typical top-1..top-10 logit gaps 0.79-2.03 measured on
drafter failures; target gaps unknown). Build the error-propagation model:
quantization noise -> attention output perturbation -> next-layer hidden
perturbation -> router logit perturbation vs the empirical logit-gap
distribution -> flip probability per token and per N tokens. This decides
whether the 256K tier can be NVFP4 (and whether the exact-256 prefix bridge
can be dropped).

## P10. Predicting DFlash2 acceptance from prompt statistics (credit assignment)

Acceptance spans 1.58-5.88 across prompts. Decompose the loss into drafter
capacity (5 layers), FP8 drafter numerics (ruled out at byte level for the
cache), capture staleness, and target-logit-gap structure; then find the
prompt statistic (repetition rate? token entropy? logit-margin distribution
of the target itself?) that predicts per-prompt acceptance well enough to
choose speculation-vs-scalar per prompt (break-even ~3.6/round, P1). The
parallel session's NumPy oracle replays + DF_DUMP machinery provide the
data-generating process.

## P11. KDA smem-transplant bit-exactness (compiler-level proof obligation)

The fused chunk-recurrence design (state in 64 KiB dynamic smem, T-loop
in-kernel, one launch/layer) claims bit-identity with the per-token decode
kernel because every expression tree is transplanted verbatim. Prove under
nvcc -O3 --use_fast_math -arch=sm_89: (a) an FP32 value round-tripped
through shared memory is bit-identical to the register-resident value across
kernel boundaries (no extended-precision carry), (b) FMA contraction cannot
differ between the two compilations for identical expression trees (what
invariant of the source guarantees it?), (c) the 4-tap conv's tap order
(current-first) preserved in a register-ring rewrite. A formal argument
here (or a falsifying counterexample) gates the fusion that removes
~6T launches per KDA layer per chunk.

## P12. Long-context attention cost and the DSA decision

Dense absorbed attention at 256K is ~1.9e14 FLOP/token (64 heads x 262144
keys x 1024 x 11 layers) — tens of seconds/token on the scalar split-tile
kernel; an FP8 tensor-core score path is ~2 s/token, still dominant. The
checkpoint ships a DSA indexer (index_topk = 2048) that the engine has not
implemented; beyond 2048 positions dense attention is off-model anyway.
Formalize the accuracy/speed trade: what does top-2048 sparse attention
cost in quality vs dense at 8K/32K/256K (the indexer's own scores decide
the selection, so this is a question about the index distribution), and
what is the optimal top-k as a function of context length given the
measured 3%-PPL sensitivity of the latent path?

## Non-math blockers hit this session (for completeness)

- WSL2 VM recycles SIGTERM everything inside the guest several times per
  hour; long benchmarks only survive via Windows Task Scheduler launching
  wsl.exe (build/s6-inner.sh + the C:\coding\s6-task.cmd wrapper).
- A parallel session works on the same box/GPU; 2x32 GiB pins contend and
  corrupt timings. Always `pgrep -af glm53-generate` first.
- glm-box cannot push to GitHub; commits flow local <- bundle <- box.
- 40 GiB pinned tier risks cudaHostAlloc failure when the parallel session
  pins 32 GiB; safe halving exists but changes the A/B.
