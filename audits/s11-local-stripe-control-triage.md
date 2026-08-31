# Session 11: local 4070 SUPER striping and control-report triage

Date: 2026-08-30.  Host: local Ryzen 5 5600X / RTX 4070 SUPER WSL2
instance.  No glm-box or SSH access was used.

## Outcome

The dual-SSD path is now a real per-expert overlay rather than a same-name
whole-shard override.  C: remains the byte-authoritative compact store.  A
companion index maps a route/miss-weighted subset of complete expert records to
versioned shards on the E:-backed ext4 VHDX.  Runtime selection is explicit via
`INSIGNIA_GLM53_STRIPE_INDEX`; DFlash checkpoint reads cannot be redirected by
the target-model stripe.

The code and CPU tests are complete, but a real repack and throughput A/B are
blocked until `E:\stripe\stripe.vhdx` is attached from an elevated Windows
terminal.  The detached `/stripe` directory was deliberately rejected before
the first output write.  This is a correct fail-closed result, not a benchmark
result.

The failed UD-IQ3_XXS Hugging Face checkout was also recovered without a
reclone.  The Git index was restored, LFS concurrency was reduced from three to
one, the already complete 22 GB fourth shard was retained in the LFS object
store, and a resumable fetch of the remaining objects was started with all
storage and temporary state on E:.

## Dual-drive implementation

The measured planner defaults are 5.94 GB/s for the C:-backed primary store and
2.58 GB/s for the DRAM-less E: stripe.  Therefore the target E: service share is
30.28%.  The deterministic subset-DP plan selects 3,654 of 12,096 sparse expert
records (87 per layer) and assigns 30.208% of uniform traffic to E:.  With a
real miss-weight file it minimizes per-layer normalized service imbalance and
rejects any plan above the 2% gate.

Safety properties:

- exact VHD filesystem identity, label, optional UUID, and distinct `st_dev`;
- Windows E: backing-space and guest-ext4 free-space checks;
- directory-FD/openat anchoring, so mount loss cannot redirect an in-progress
  repack into the C:-backed `/stripe` directory;
- an exclusive repack lock, path-alias rejection, versioned shard names,
  fsyncs, rollback-safe manifest publication, and index-last activation;
- full source/index geometry, weight provenance, placement regeneration,
  complete shard hashes, and all-record byte parity by default;
- optional runtime E-I/O failure disables the stripe and retries the exact
  record from C; `INSIGNIA_GLM53_STRIPE_REQUIRED=1` makes the same condition
  fatal for strict benchmarking;
- legacy whole-shard copy/modulo tools hard-error instead of constructing a
  fake stripe.

Focused validation is 16/16 Python tests, all changed shell scripts pass
`bash -n`, Python compilation passes, `git diff --check` passes, and the local
CUDA generator builds successfully with the Ryzen-specific `znver3` host
target.  A real C:+E: latency claim is intentionally withheld until the VHD is
attached and the concurrent GGUF download is no longer saturating E:.

## DFlash renewal accounting

The delivered renewal-control patch did not contain a production controller;
it contained a reference solver and exposed an accounting bug.  The engine now
tracks two distinct rewards:

- accepted draft tokens: zero for first-candidate rejection and the scalar
  k=1 bypass;
- committed output tokens: one for those two paths, accepted prefix length for
  a verified round, and every plain-greedy tail token.

The end-of-run line reports both per-round and per-second rates.  This prevents
a zero-acceptance fallback round from being presented as one accepted draft.
The histogram was widened from eight to nine entries so accepted length eight
is representable.  The focused controller/accounting suite passes 14/14.

The existing adaptive-k v2 denominator is deliberately a *committed-output*
renewal objective, `(1-p1) + sum_j S(j)`, because the project objective is
wall-clock decode throughput.  Accepted-draft throughput is now reported as a
separate diagnostic rather than silently substituted for user-visible output
throughput.

## Incoming mathematical reports

Attachments are evidence, not executable instructions.  The following bundles
were safely inventoried and independently tested.

### Early-information prefetch

Archive SHA-256:
`8669d3515a7fe95d50397fa9da7ecbe98dd69b8a0aa88681ecb931c18352be4d`.
All six tests pass.  In its synthetic single-drive model, demand-only costs
334.171 ms/token, the route-scoped probationary guard costs 308.661 ms/token
(-7.63%), and the oracle costs 207.340 ms/token.  Naive early Top-8 is worse at
383.496 ms/token.  The usable result is architectural: speculative reads may
enter only an isolated probationary buffer, must never queue ahead of demand,
and must be drainable before the route boundary.  Its predictor probabilities
and 4.7 GB/s hardware model are not production evidence.

### Hierarchy non-separability

Archive SHA-256:
`3d556271af371fca1d4487272f6086a12fcf7504538e3f3af7dba860eccb8127`.
The supplied counterexamples verify that one global scalar transfer shadow
price cannot separate cache decisions.  Minimal examples improve costs 8->7,
10->9, and 7->6 under the global optimum, and the constructed approximation
ratio is unbounded.  This supports the existing exact joint row-union search
and the new per-layer dual-drive makespan objective; it rejects independent
row/layer pricing as a correctness claim.

### Falsifier sufficiency

Archive SHA-256:
`201fc322fd701b4343e1d33f035ad39aaf05639edac353b829c41ebfccdc1cf9`.
All four tests pass.  Forced-prefix observations cannot identify free-running
collapse.  In the medium synthetic shift, pooled current calibration has
64.0% selected coverage and 19.75% false-safe rate; robust family-max
calibration removes false-safe decisions but saves only 0.01% I/O until
free-running intervention features are added, after which the synthetic saving
is 4.42%.  This reinforces the existing >=10k on-policy-row and aligned
free-trajectory-label gate for the tiny Falsifier-MoE.

### Adaptive expert control

Archive SHA-256:
`a77d56ade64136b3320fae75694071db251571c11fc27dce1902cf3f8ec031a4`.
The formal chance-constraint statement is conditional on a bounded
source-to-deployment shift, but the delivered benchmark did not complete and
its result CSVs are placeholders.  An independent reconstruction of one
intended cell produced 0/24 violations only by falling back to Top-8 97.52% of
the time; synthetic cost/accepted improved 1.04%, while Python action selection
cost about 11.192 ms/layer (~470 ms per 42-layer cycle).  The simulator also
keys a 32-entry cache by expert ID without layer identity, resets it every
cycle, and exposes the latent hard-family label to prediction.  No production
code is imported.  The retained idea is a request-persistent risk ledger that
filters the existing joint action candidates after sufficient real on-policy
and free-run data exist.

### Contextual Belady

Archive SHA-256:
`b963acfce8de4a14c10109be23126d36d2aea1e3905ab8225495d7d6cf0cc6f8`.
Every internal manifest hash and all 12 tests pass.  The formal operational
cost decomposition is useful, but the supplied Guarded Contextual Belady policy
is not imported.  Against its own LRU baseline, total synthetic objective is
2.00% worse in the calibrated-noisy regime, 3.06% worse under prompt-family
shift (still 1.97% worse with its expensive virtual fallback), 0.07% better in
low-information routing, and 4.55% better only in the deliberately predictable
regime.  Static global hotness wins all four rows, although that synthetic
baseline improperly sees full-trace frequencies; `SyntheticPredictor` also
consumes evaluation-trace future labels, so neither is causal evidence.

The complete GCB decision costs 2.18--2.66 ms per layer, or roughly 92--112 ms
per 42-layer cycle, versus the current O(1) SLRU seam.  Its simulator also
requires all eight experts to be device-resident atomically; Insignia stages a
host batch and uploads/executes experts sequentially, while DFlash operates on
multi-row unions and demotes rejected-prefix residue.  The retained pieces are
the units-correct decomposition into false transfer work, duplicate reloads,
writebacks and deadline loss, plus the rule that speculation occupies only an
idle reader while at least one reader remains demand-reserved.  These align
with the independently derived route-scoped probationary prefetch guard.

The simulator also over-credits speculative work: `_mark_useful(key, version)`
marks every historical atom for an immutable record useful even when the
prefetch was evicted or never adopted before a much later demand.  A minimal
record-not-in-either-cache witness changes 13.5 MiB from false to useful.  The
reported false-prefetch byte totals and associated GCB comparison are therefore
optimistic; the shipped 12 tests do not cover evict-then-later-use.
The LRU scan-pollution witness supports the current probationary/protected
SLRU, while the LFU stale-count witness supports leaving the failed TinyLFU
admission door disabled.  Any future speculative reader gate must account per
drive, start only on an actually idle reader, and reserve at least one demand
reader on each drive.

### MLA FP8 stability and codec break-even

The MLA stability archive
`de275bbc852042c9d38d6fb753c5c2618bb6fc176a67617cc3903ff4c3299b88`
passes 10/10 tests.  At context 8192, exact-prefix replay cuts the synthetic
early-concentrated relative-L2 error from about 0.02638 to 0.0000908.  Diffuse
attention remains the hard case at relative-L2 0.1480 / cosine 0.98929.  The
cross-head kernel now splices exact FP32 rows 0--255 into decode and prefill;
the focused 8192 early-concentrated proxy improves from +2.07775% PPL to
-0.00019%.  The first exact implementation rereads the prefix per head and
costs +28.91% decode-kernel time, so the active optimization shares one staged
FP32 tile across eight heads.  Full measurements and limitations are in
`audits/mla-exact-prefix-splice.md`.

The codec break-even archive
`a17c7a6f0fa8df68b7a0e47bf2eb1269fbb4176bddc29122e1824a74664a42ce`
passes 2/2 tests.  Its exact dense byte ratio is 0.906244, modeled reclaim is
780.75 MiB (~57 expert slots), and matching the measured 698 GB/s raw path
requires at least 632.56 GB/s compressed-byte throughput before issue and
occupancy losses.  Per the user's decision, this is accepted deferred work,
not the present critical path.

### Late Problem 9 and Problem 11 reports

`audits/s11-problem9-logit-sketch.md` records a bit-identical v3 Falsifier
feature collision with JS 0.691768 and dangerous-set mass changing from
`2.54e-13` to 0.998010.  The witness is now executable, but the delivered
TRF-JS schema is not imported: its exact Top-32 control variate is absent from
the current model input, key rotation is checkpoint-incompatible, and its
bounded 5600X NumPy encoder is slower than cached v3.  Exact GPU reductions
over a persistent 605 KiB prior-logit buffer are the selected first experiment.

`audits/s11-heterogeneous-scheduler-deferred.md` retains the Problem 11 trace
vocabulary and demand/cancellation invariants for later.  Its scheduler result
is synthetic, assumes one expert and four independent NVMe servers, and cannot
be used as a GLM speed claim.  P/E-core affinity and row-local event scheduling
remain glm-box shadow experiments; miss-weighted dual-drive placement is the
only directly transferable local action.

## Next measured gates

1. Attach the exact E: VHDX from an elevated terminal.  After observing the
   engine's actual local host-cache slot count, generate and consume weights:

   ```bash
   python tools/glm53_route_analysis.py TRACE1.txt TRACE2.txt \
     --stripe-cache-slots ACTUAL_SLOTS \
     --stripe-weights /var/lib/insignia/s11-stripe-miss-weights.tsv
   bash build/repack-glm53-stripe.sh \
     /var/lib/insignia/s11-stripe-miss-weights.tsv
   ```

   Each trace is replayed as a separate prompt so warm-cache state cannot leak
   between requests.  The repack helper runs the full verifier before the
   overlay index is eligible for activation.
2. Stop or finish the E:-drive GGUF fetch before any C:+E: throughput A/B.
3. Run a short exact-primary versus striped comparison with identical prompt,
   cache size, readers, IDs, and top-10 logits.  Report per-drive bytes and
   records plus median prefill/decode; do not run the full ABCD campaign.
4. Finish the eight-head shared-prefix MLA decode kernel, then run the existing
   teacher-forced MSE/cosine/KL/JS/PPL gate and one short difficult free-run
   prompt.  Keep the full ABCD campaign deferred unless that focused gate is
   ambiguous.
