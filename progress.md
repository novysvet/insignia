# progress

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
