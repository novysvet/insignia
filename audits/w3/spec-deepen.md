# Spec decode deepening: D-draft chains, T-row verify — design spec (w3)

Scope: generalize the current MTP depth-1 / pair-verify spec decode to draft depth
D (chain of MTP invocations) and verify width T = D+1 rows. Read of `src/decode.cu`,
`src/generate.cu`, `src/prefill.cu`, `src/mxfp4.cu`, `src/gemm.cu`, `src/attention.cu`,
`src/qwen_kernels.cu`, `tools/reference_multistep.py`, audits/synthesis.md (item 3).

---

## 0. What the code does today (verified by trace)

`spec_step(t0)` / `capture_spec()` (decode.cu:213-248) run this sequence, all
device-state driven through `pos_dev` slots (prefill.cu:275-276):

| slot | name | role |
|---|---|---|
| pos[0] | position | next unprocessed main-model position P |
| pos[1] | token/pending | decided-but-unprocessed token occupying position P |
| pos[2] | next/after | mtp argmax during draft; verify row-1 argmax after |
| pos[3] | t2 | verify row-0 argmax (target token after pending) |
| pos[4] | draft | mtp proposal (copied from pos[2] by spec_setup) |
| pos[5] | committed count | cursor into `committed[]` |
| pos[6] | accept flag | draft==t2 |
| pos[7] | mtp position | MTP KV slot for this step's single invocation |

Step: `spec_prologue` (pos[7] = P-1) → `mtp_layer()` → `spec_setup`
(pf_tokens = [pending, draft]) → `prefill_chunk_device(pf_tokens, 2)` →
`spec_commit` → `spec_rollback`. Whole thing is graph-captured (decode.cu:232-243),
replayed 4x per host check (generate.cu:179-180), host scans `committed[]` for EOS.

Key traced facts:

- **F1. Every draft pays a FULL lm_head sweep + full-vocab argmax.**
  `mtp_layer()` (decode.cu:186-187) runs `linear("language_model.lm_head", …)`
  (full [248320,4096] MXFP4 = 540 MB pass for 9B; bf16 2.54 GB for 27B) then
  `argmax_fast` over 248320. This is the SAME lm_head the verify uses. Per-draft
  cost at 9B ≈ 1.3 ms lm_head + ~0.5 ms mtp(fc 67 MB + layer 111 MB) — the lm_head
  is ~72% of a draft. At 27B: 5.1 ms lm_head vs ~1.0 ms mtp layer (80%).
- **F2. Verify lm_head already amortizes both rows in ONE weight pass**: the T==2
  path (decode.cu:94-98) uses the pair kernel `mxfp4_gemv2_q8` (one weight stream
  feeds both rows' dp4a chains, mxfp4.cu:283-366) then two `argmax_fast` calls.
- **F3. Pair kernel = ONE pass for 2 tokens.** It is not "2 GEMV passes"; both
  activation rows are quantized to smem int8 and each loaded weight group feeds
  both dp4a chains. 213→430+ GB/s on lm_head shapes (commit 777f55f).
- **F4. GEMM v21 always computes a 64-row A tile.** `mxfp4_gemm_v21` (gemm.cu:210,
  launched from `linear_batch` decode.cu:36-40) zero-pads A to 64 rows in
  `pf_bf16` and the kernel hardcodes `As[2][64][…]` with wmma m16n16k16 —
  T∈[1,64] all do 64 rows of MMA. One weight pass per output tile (weights read
  once regardless of T), but compute is 64-row regardless of T.
- **F5. Rollback snapshots exist only for row 0.** `deltanet_prefill` snapshots
  state after t==0 (prefill.cu:258-261), `conv_roll_state` snapshots the pre-row-0
  conv window (prefill.cu:184-195). `spec_rollback` restores row-0 state + hidden
  = pf_x[0] only, and early-exits on accept (prefill.cu:305-313).
- **F6. Full-attn KV is never rewound** — it is append-only, position-indexed;
  stale draft slots beyond the position pointer are overwritten (store_kv precedes
  gqa within each layer call; gqa reads exactly slots 0..pos, attention.cu:7).
  Safe today because every reject is followed by a pair that rewrites the slot.
- **F7. BUG — MTP KV has holes; drafts attend garbage.** (a) The MTP KV region for
  the prompt (slots 0..N-2) is never written: after prefill, the first invocation
  at slot N-1 attends slots 0..N-2 of *uninitialized* cudaMalloc memory (probe
  mode zeroes the cache only AFTER probing, generate.cu:140-141; nothing zeroes it
  in the normal path). (b) Every ACCEPT skips a slot: step at P writes slot P-1;
  accept ends at pos P+2 → next invocation is at slot P+1, leaving slot P stale
  (reject is dense: P-1 → P). The NumPy reference (reference_multistep.py:104-119)
  zero-inits `mtp_kvc` and fills it densely at every position (teacher-forced), so
  engine drafts are NOT reference-equal in spec mode today. Greedy output
  exactness is unaffected (verify gates everything); the damage is a depressed
  acceptance rate. With D≥2 this worsens (holes accumulate).
- **F8. `spec_second`/`spec_accepted` are pair-specific host conveniences**
  (decode.cu:225-229); the real accept logic is `spec_commit_kernel`
  (prefill.cu:287-302): acc = (draft==t2); commit [pending] (+draft); pending ←
  t2-or-after; pos -= 1 on reject.
- **F9. Stale test API**: test_mtp.cu calls `d.mtp_draft(...)` which no longer
  exists (header has `mtp_layer()`), and `mtp_draft` writes to `x.logits`
  clobbering the main logits. Minor, but it means the MTP head currently has NO
  runnable unit test in the new device-state regime.

---

## 1. Cost model

### 1.1 Anchors (9B, all-VRAM, MLX MXFP4 ~4.99 GB)

Derived from shape math (24 DeltaNet ≈ 116 MB, 8 full-attn ≈ 111 MB, lm_head 540 MB,
mtp fc bf16 67 MB + mtp layer 111 MB) and measured bandwidths (379-430 GiB/s
streaming, internals.md; pair kernel 430 GB/s on lm_head shapes):

- plain decode step: body 3.67 GB ≈ 9.8 ms + lm_head 1.3 ms + state/KV/launch ≈ 1.0 ms
  → **12.1 ms/token = 83 tok/s** (matches pre-spec baseline).
- spec step today (D=1): verify body (same weight pass, T=2) 9.8 + verify lm_head 1.3
  + draft (mtp 0.5 + lm_head 1.3) + eps 0.2 → **13.1 ms/step**. At p=0.6 accept,
  E[tokens/step] = 1.6 → **122 tok/s ≈ measured 121**. Model calibrated.
- Verify is bandwidth-bound: extra verify rows are ~free (activation bytes
  T×4096×4B per GEMM « weight bytes; state traffic +50 MB/row for deltanet
  recurrence — the recurrence itself is required regardless).
- **Draft cost is dominated by its lm_head sweep** (F1): full-sweep draft = 1.8 ms
  (9B) / 6.2 ms (27B); sliced draft (argmax over a 64K-row compacted lm_head copy,
  143 MB @ 9B) = 0.83 ms. Slicing only affects the *proposal*; verify keeps
  exactness, so an out-of-slice true argmax merely causes a rejection
  (~×0.97 on p_i with a frequency-built slice).

General step cost (T = D+1):

```
C(D) = V + D_head·c_draft      V ≈ 11.3 ms (9B) / 1.57 s (27B, L21/M23/N21)
E(D) = 1 + p·(1-γ^D)/(1-γ)     p = p(draft1 accepted) ≈ 0.6, γ ≈ 0.7 marginal rate
```

### 1.2 (D,T) table — 9B, all-VRAM

| config | drafts lm_head | C(D) ms | E[tok] | tok/s | vs today |
|---|---|---|---|---|---|
| D=1, T=2 (today) | full | 13.1 | 1.60 | 122 | 1.00x (meas 121) |
| D=2, T=3 | full | 14.9 | 2.02 | 135 | 1.11x |
| D=3, T=4 | full | 16.7 | 2.31 | 138 | 1.13x |
| D=2, T=3 | slice d2 | 13.9 | 2.02 | **145** | 1.19x |
| D=3, T=4 | slice d2..d3 | 14.8 | 2.31 | **156** | 1.28x |
| D=4, T=5 | slice d2..d4 | 15.6 | 2.44 | **157** | 1.29x |
| D=5, T=6 | slice d2..d5 | 16.4 | 2.54 | 155 | 1.27x |

γ=0.6 pessimism: D=3 sliced → ~147-150. Fixing F7 should RAISE p (drafts stop
attending garbage): p=0.7, γ=0.7, D=3 → E=2.53 → ~170 tok/s upside.

**Recommendation 9B: v1 = D=2/T=3 with full-sweep drafts (no new weight copy,
+11%); v2 = D=3/T=4 with drafts 2..D sliced (+28%, ~156 tok/s).** Draft 1 always
keeps the full sweep (it is the highest-value proposal; also its full-vocab
argmax over the exact-input row is where p is highest).

### 1.3 (D,T) table — 27B tiered (L=21 VRAM / M=23 RAM / N=21 NVMe)

Verify cost is tier-bound (0.76/15.4/56.5 ms per layer → 1.56 s body + 5.1 ms
lm_head) and **independent of T** (weights stream once whether T=2 or T=8; GEMM
activation traffic negligible). Drafts touch only the VRAM-pinned MTP hot set
(fc 105 MB + mtp layer 372 MB + lm_head 2.54 GB ≈ 3.0 GB) ≈ 6.2 ms each.
No slicing needed: a draft is 0.4% of the step. Break-even for draft i:
γ^(i-1)·p·1570 ms > 6.2 ms → γ^(i-1) > 0.006 — every draft pays until acceptance
decay kills it; the real caps are snapshot VRAM and chain bookkeeping.

| config | C(D) s | E[tok] | tok/s |
|---|---|---|---|
| D=1, T=2 (today's scheme) | 1.58 | 1.60 | 1.01 |
| D=2, T=3 | 1.59 | 2.02 | 1.27 |
| D=3, T=4 | 1.59 | 2.31 | 1.45 |
| **D=4, T=5 (v1)** | **1.60** | **2.44** | **1.53** |
| D=6, T=7 (v2 cap) | 1.61 | 2.61 | 1.62 |

**Recommendation 27B: D=4/T=5 for v1 (×1.5 vs D=1), cap at D=6.** Since drafts
use only VRAM-resident weights, run them OVERLAPPED with the NVMe prefetch of
the next verify's layers (drafts ≈ 25 ms « the 1.19 s NVMe portion — effectively
free; that would push D=6 to ~1.65 tok/s).

---

## 2. Draft chain mechanics — exact slot/position bookkeeping

Semantics (from reference_multistep.py:103-130, DeepSeek-MTP style): an MTP
invocation at slot s consumes (embed(t_{s+1}), h_s) — h_0 = main-model hidden,
h_s = MTP's own residual stream for s≥1 — attends MTP KV slots 0..s, and
proposes a token for position s+2.

### 2.1 Timeline, D=3 / T=4 (T = D+1; a "D=2, T=4" reading is incoherent for
greedy-exact verify — the 4th row would have to process a token that is only
known after verify; D=2 ⇒ T=3 with identical structure)

Step start: pos = P (committed/processed through position P-1), pending = t_p
(will occupy position P), x.hidden = h_{P-1} (main residual after row P-1).

```
phase        token(s)          position   MTP slot   main-KV slot   state snapshot
─────────────────────────────────────────────────────────────────────────────────
draft d1     in: t_p, h_{P-1}     —         P-1        —              —
             out: a1 (→pos d1 slot), mtp residual R1 stays in x.hidden
draft d2     in: a1, R1           —         P          —              —
             out: a2, R2
draft d3     in: a2, R2           —         P+1        —              —
             out: a3, R3
verify row0  t_p                 P         —           P              snap[0] (delta+conv)
verify row1  a1                  P+1       —           P+1            snap[1]
verify row2  a2                  P+2       —           P+2            snap[2]
verify row3  a3                  P+3       —           P+3            (live state kept)
lm_head GEMM [4,248320] → ONE weight pass → argmax rows → t*_0..t*_3
             t*_i = exact greedy token for position P+1+i given prefix ≤ row i
commit       a = max k: a_i == t*_{i-1} ∀ i ≤ k          (chain rule; a ∈ 0..3)
```

Commit / rollback per accept length a (committed tokens = pending + a drafts;
pending is never "committed" — it becomes the carried token, matching today):

| a | committed += | new pending | new pos | delta/conv restore | hidden | MTP KV |
|---|---|---|---|---|---|---|
| 0 (reject at d1) | [t_p] | t*_0 | P+1 | snap[0] | pf_x[0] | keep slot P-1; next d1 rewrites slot P |
| 1 | [t_p, a1] | t*_1 | P+2 | snap[1] | pf_x[1] | keep P-1, P |
| 2 | [t_p, a1, a2] | t*_2 | P+3 | snap[2] | pf_x[2] | keep P-1..P+1 |
| 3 (full) | [t_p, a1..a3] | t*_3 (bonus, exact) | P+4 | none (live state = after row 3) | pf_x[3] | hole at P+2 (see below) |

Invariants that make the bookkeeping trivial:

- **Both KV caches stay append-only and position-indexed; no explicit rewind
  exists or is needed.** The "write pointer" IS pos. Stale slots ≥ pos are always
  overwritten before read, because store_kv(_batch) runs before gqa_* within each
  layer invocation, and both gqa kernels read exactly slots 0..own-position.
  This remains true for T>2: store_kv_batch writes all T rows before gqa_prefill,
  and gqa_prefill row t reads pos+t+1 ≥ own slot only.
- **Main hidden always ends at pf_x[a]** (copy after verify picks pf_x[T-1];
  the restore kernel corrects to pf_x[a] for a<T-1, exactly like today's
  reject path corrects to pf_x[0]).
- **Snapshots generalize from "row 0" to "rows 0..T-2"**: snapshot row i = delta
  + conv state immediately after processing verify row i. Restore picks snap[a]
  (a = T-1 → early-exit, live state is already correct). deltanet_prefill already
  snapshots inside the row loop (prefill.cu:258) — extend to `if (t <= T-2 &&
  snap) snap[t][…] = S`; conv_roll_state similarly emits one snapshot per row.
  Cost: (T-1)×50.3 MB extra writes at 9B ≈ 0.15 ms/step (1%).
- **MTP KV chain slots are dense within a step** (P-1, P, P+1, … for d1..dD);
  on accept length a the next step's d1 lands on slot (P+1+a)-1 = P+a — exactly
  the slot this step's draft a+2 would have used — stale, but rewritten by that
  d1 before its own attention reads it (own slot is included in the read window).
- **Two hole classes remain and must be fixed (F7):**
  1. *Prompt region*: fill slots 0..N-2 with a batched MTP pass folded into
     prefill — after each 64-row chunk's last layer, pf_x holds all rows' raw
     residuals; run embed+2×rmsnorm+concat+fc (batched GEMM) + qk-rope at
     positions (chunk_start + r − 1) + store_kv_batch into the MTP cache +
     gqa_prefill over it + o/mlp. ~15 launches/chunk, 178 MB/chunk ≈ 0.5 ms per
     64 tokens (~8 ms per 1k-token prompt), NO lm_head needed (no proposals).
     This makes draft 1 reference-equal.
  2. *Full-accept skip*: after a=D, slot P+D-1 (invocation (a_D, R_D)) is never
     written. v1: zero-fill that one slot at commit (8 KB memset — deterministic,
     bounded to a single slot). v2 (exact): run one extra MTP invocation without
     lm_head at commit time using (embed(a_D), R_D saved before the hidden copy),
     0.5 ms on ~p·γ^(D-1) of steps ≈ 0.1 ms amortized.

### 2.2 Draft-phase implementation shape

Refactor `mtp_layer()` into `mtp_draft_step(slot_source)`: identical body except
(i) the input token is `pos[chain_token]` (device copyi from the previous
proposal / pending), (ii) x.hidden carries the chain residual (draft 1 reads the
main hidden, drafts 2..D read the previous draft's residual — rmsnorm reads are
non-destructive, fc output overwrite is safe after its concat read, so the chain
works in-place with zero extra buffers), (iii) the proposal argmax writes
`pos[draft_i]` (draft 1 full-vocab; drafts 2..D optionally over the sliced
matrix), (iv) no main-hidden save needed: verify re-embeds pf_tokens and x.hidden
is reset from pf_x after the chunk.

---

## 3. Verify at T>2 — GEMM vs pair

The prompt's proposed syllogism ("GEMM reads weights once ⇒ strictly ≥ pair") is
**half right and needs correction against the actual kernels**:

- Weight bytes: pair (F3) and GEMM (F4) both read each weight byte exactly once
  for any T. Equal on memory.
- Compute: pair does 2·R·C·2 useful FLOPs and is hard memory-bound (430 GB/s on
  lm_head shapes). v21 as-written computes a **64-row A tile for any T** (zero
  padding + hardcoded `As[2][64][…]`). On the worst 9B shape (gate_proj
  12288×4096): memory floor 25.2 MB / 430 GB/s ≈ 59 µs, padded compute
  2·12288·4096·64 / ~82 TF sustained ≈ 78 µs → **today's GEMM is compute-bound
  from padding and would be ~15-30% SLOWER than the pair at T=2-4.** The pair
  kernel is not legacy; it is currently the memory-bound optimum at T=2.
- The win requires a **16-row-granularity A tile** (wmma m16 floor): 16-row
  compute = 2·R·C·16 ≈ 20 µs « 59 µs → memory-bound at every T ∈ [2,16], i.e.
  *equal to pair at T=2 and free beyond*. One kernel serves T=2..16, the dual
  pair/GEMM path collapses, and lm_head verify [T,248320] rides the same kernel
  (compute 2·248320·4096·16/82TF ≈ 0.4 ms « 1.3 ms memory).

**Verdict**: add `mxfp4_gemm_v21<MT=16>` (template the A-tile rows 16/64, pad T
to 16 in `pf_bf16` — the memset tail already exists at decode.cu:37, just 16 not
64), wire `linear_batch` to it for T≤16, keep the pair path only until
parity (multistep reference, cos ≥ 0.9999) + bench (≥ pair at T=2, spec step
13.1 ms not regressed) prove it. Then remove pair from the spec path (keep it in
eager diff-testing). Do NOT ship T>2 on the current 64-row GEMM — it donates the
padding loss on every one of ~200 GEMMs/step.

27B: same conclusion, simpler — weights stream once per layer from their tier at
any T (fp8 blockwise GEMM, m16n8k32 granularity pads T to 16 harmlessly; the
step is tier-bandwidth-bound, activation bytes irrelevant). T=5 verify ≡ T=2
verify in cost; GEMM path required anyway (no fp8 pair kernel exists).

lm_head verify bookkeeping: logits buffer becomes [T, 248320] f32 (T=4: 3.97 MB),
one GEMM, one **batched argmax_rows kernel** (T rows → t*[] in a single launch;
the current per-row pair of launches would cost 2T×~15 µs — 0.12 ms at T=4 —
fuse it: grid (T × 64 blocks), one u64 winner per row in am_scratch[T]).

---

## 4. Device-side accept logic, generalized

`spec_commit_kernel` generalized (1 thread, serial compare loop is fine at D≤8):

```
a = 0; while (a < D && pos[draft_{a+1}] == pos[tstar_a]) a++;
c = pos[count];
committed[c..c+a] = [pending, draft_1..draft_a];      // 1+a tokens
pos[pending] = pos[tstar_a];                           // correction (a<D) or bonus (a=D)
pos[count]   = c + 1 + a;
pos[posslot] = P + 1 + a;                              // addi added T; correct by T-1-a
pos[accept]  = a;                                      // restore selector; T-1 == full
```

Greedy-exactness argument (unchanged in kind): every emitted token is either a
verify-row argmax computed by the full model over the exact committed prefix
(t*_a), or a draft token confirmed equal to that argmax. Emitted stream ==
greedy stream, independent of D, T, slice quality, and MTP KV state.

`spec_rollback` generalized: read a = pos[accept]; if a == T-1 return (live state
kept); else restore delta/conv from `snap[a]` (grid-stride copy, early-exit
already proven at 9B scale: 50 MB ≈ 0.13 ms only on non-full-accept steps) and
hidden ← pf_x + a·4096 (block 0 extends to a runtime row offset).

`spec_second`/`spec_accepted` (F8) become `spec_last_second`/`spec_accept_len`,
populated in the eager path only; the graph path never reads them.

Host-side EOS/max_new checks (generate.cu:179-187) are unchanged — they read
`committed_count()`; the only host-visible change is the overshoot bound: with
4 graph replays between checks and ≤ D+1 commits each, committed can overshoot
want_total by ≤ 4(D+1); the existing trim logic (generate.cu:194-198) already
cuts at want_total/EOS. KV-full margin: require ctx − pos ≥ 4(D+1) + T at each
host check; below that, fall back to eager spec steps (guard today is bypassed
by replay — audit bug; keep the margin rule).

---

## 5. CUDA graph — whole step at fixed (D,T)

Capture `spec_prologue → D × mtp_draft_step → spec_setup_T → prefill_chunk(T)
→ argmax_rows → commit → restore` exactly as today (decode.cu:232-243) with
T,D compile-time constants of the capture. Device-side branching stays
predicated (restore early-exit; commit writes variable-length via count cursor).
No conditional graph nodes needed at 9B scale — the restore kernel's early-exit
costs ~5 µs on full-accept steps.

Node inventory (D=3/T=4): 3×~28 draft nodes + setup + embed + 32×~14 layer
nodes + lm_head GEMM + argmax_rows + copy/addi + commit + restore ≈ **545
nodes** (today ≈ 500). Per-step launch savings vs eager ≈ 1-2 ms — the reason
graphs stay mandatory at 9B (83→121 was partly this).

Buffers baked at capture (all device, stable pointers — see hazards):

- `logits[T][248320]` f32 (4 MB @ T=4)
- `pf_tokens[0..T-1]` (setup kernel writes [pending, d1..dD] from pos slots)
- `am_scratch[T]` u64 (per-row argmax winners)
- `snap_delta[T-1][24·32·128·128]` f32 (151 MB @ T=4), `snap_conv[T-1][…]`
- pos_dev widened 16 → 40 ints: [0]=pos [1]=pending [2..2+T-1]=t* [next D]=drafts,
  then count, accept-len, mtp_pos, chain-token scratch
- `committed[16384]`, pinned mirror, count — unchanged

Re-capture triggers (beyond today's single capture at start):
1. (D,T) change — new capture, explicit.
2. Context tail: pos within 4(D+1)+T of max_context → stop replaying, eager out.
3. **Weight-pointer stability**: everything touched during capture must be
   pinned against LRU eviction for the graph's lifetime (today's capture is safe
   only because warmup touches everything and the 6 GB budget holds the whole
   model; post-capture eviction would leave dangling pointers — audit bug #6,
   detailed in w3/graph-hazards.md). Minimal fix: a `capture_pin` flag on
   TieredStorage entries — acquires during capture increment it; pinned entries
   are exempt from make_room(); graph destroy clears the set.
4. Format switch (MLX vs INSIG4 selects different kernels) — capture per format.

---

## 6. 27B integration

- **No whole-step graph.** Launch overhead saved ~10-30 µs vs a 1.6 s step —
  0.002%. Tiered streaming (pinned staging ring rotation, NVMe completion
  re-issue, possible eviction between steps) makes capture hazardous and
  worthless. Plain stream execution with per-layer prefetch (colibri-style early
  issue, 1 sync/device); optionally small per-VRAM-layer-group graphs later —
  v1: none. Matches the w3/graph-hazards.md recommendation.
- VRAM budget with the MTP hot set pinned (3.02 GB): L=21 (8.06 GB) + hot set
  = 11.08 GB + KV + 151 MB delta state + snapshots — over 12.28 GB with fp32 KV
  at ctx 4096. Take bf16 KV (halves 537→268 MB; already backlog item 5) or run
  L=20. Snapshots at T=5 = 4×151 MB = 604 MB VRAM — if tight, stage row
  snapshots to pinned host memory (restore = 151 MB over ~13 GB/s PCIe ≈ 12 ms,
  0.8% of step) or cap T=4.
- Drafts overlap the next verify's NVMe prefetch (§1.3): issue the prefetch
  pipeline for step k+1's NVMe layers before running step k's drafts (drafts
  only touch VRAM-resident tensors, so no bandwidth contention on the load
  path).
- lm_head is bf16-dense [248320,5120]: needs a streaming bf16 GEMM/GEMV path
  (today's `bf16_gemv` is a naive rows-grid kernel); it must be VRAM-resident
  and graph/stream-stable. 27B deltas for kernels are in synthesis.md:44-45
  (GQA kvh = head/6, deltanet kh = head/3, hidden 5120, qkv row 10240, etc.).

---

## 7. Buffer diff summary (9B, D=3/T=4)

| buffer | today | after | Δ |
|---|---|---|---|
| logits | 2×248320 f32 (1.99 MB) | 4×248320 (3.97 MB) | +2 MB |
| snap_delta | 1×50.3 MB | 3×50.3 MB | +101 MB |
| snap_conv | 1×2.36 MB | 3×2.36 MB | +4.7 MB |
| am_scratch | 1 u64 | T u64 | nil |
| pos_dev | 16 int | 40 int | nil |
| slice lm_head copy (v2) | — | [65536,4096] MXFP4 | +143 MB |
| KV caches, pf_* (sized 64), committed | unchanged | unchanged | 0 |
| VRAM total | ~5.4 GB | ~5.65 GB (v2 ~5.8 GB) | fine |

---

## 8. Ordered implementation edits

1. **Determinism + MTP prefill fill (F7).** memset mtp_keys/values at workspace
   init (1 line). Add the batched MTP pass at the end of every prefill chunk
   (uses pf_tokens + pf_x rows; no lm_head). Validate via the probe path against
   reference_multistep.py's dense teacher-forced fills. Re-measure p — if it
   rises above 0.6, the (D,T) table shifts up for free.
2. **Per-row snapshots.** Extend deltanet_prefill's `snap` to per-row
   (`snap[t]`, t ≤ T-2) and conv_roll_state to emit T-1 row snapshots. Keep the
   row-0 layout as t=0 of the array so current code paths stay valid.
3. **Generalized spec kernels.** `spec_setup_T`, batched `argmax_rows`,
   `spec_commit_T` (chain rule, §4), `spec_restore_T` (row-a snapshot + hidden).
   Extend pos_dev layout + DecodeWorkspace buffers (§7). Eager `spec_step_T`
   first; extend the eager debug printout (generate.cu:152-167) to dump the
   whole accept vector for differential testing.
4. **Reference for depth ≥ 2.** Add `mtp_chain(depth)` to reference_multistep.py
   (module self-recursion with its own residual; confirm against the _mlx
   reference clone's MTP decode loop before trusting the recursion input
   convention). Freeze a greedy-chain regression: eager ids == graph ids ==
   teacher-forced greedy.
5. **T-flexible verify GEMM.** `mxfp4_gemm_v21<MT=16>` (both MLX and INSIG4
   scale paths), wire `linear_batch` + the T>2 lm_head branch to it; parity
   (host double reference, multistep cos) + bench vs pair at T=2 before
   deleting pair from the spec path.
6. **capture_spec(D,T)** with the graph pin set in TieredStorage (§5.3),
   context-tail fallback, host-check overshoot margin 4(D+1).
7. **9B tuning pass**: land D=2/T=3 full-sweep (no slice) → measure; then
   slice drafts 2..D (frequency- or corpus-argmax-built 64K compaction) →
   D=3/T=4 → measure. Keep whichever measured tok/s wins; re-run NLL + parity
   after each.
8. **27B v1** (after its loader/kernels land): plain stream, D=4/T=5, MTP hot
   set pinned, snapshots per §6, drafts overlapped with NVMe prefetch, no
   graphs.

---

## 9. Answers to the specific questions posed

- *How does the MTP draft produce its proposed token?* Through the SAME
  `language_model.lm_head`, full sweep + full-vocab argmax (decode.cu:186-187)
  — F1. This is why D>1 without slicing is lm_head-dominated at 9B.
- *Is the pair kernel legacy, does GEMM win outright?* No / not as-built: both
  do one weight pass; v21's hardcoded 64-row A tile makes it compute-bound at
  T≤4 (78 µs vs 59 µs memory floor on gate_proj). A 16-row-tile GEMM ties the
  pair at T=2 and is free beyond — unify on that, then drop pair (§3).
- *T positions × 5 ms lm_head?* No — the [T,248320] GEMM is one weight pass for
  all rows (weights read once per output tile regardless of T); only the logits
  writes scale with T (~1 MB/row, noise). The real per-draft sweep problem is
  the DRAFT side, fixed by slicing (9B) or tolerated (27B).
- *Is append-only KV rewind safe?* Yes (F6, §2.1 invariants): store-before-read
  within every layer call, read window never exceeds own position, stale slots
  beyond pos are dead until overwritten. Main model and MTP cache both.
* *Graphs for 27B?* No (§6) — 0.002% upside, real pointer-stability hazards.
