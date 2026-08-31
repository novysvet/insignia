# Session 11: exact NVFP4 compute-for-bandwidth wave

Date: 2026-08-31  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware: local RTX 4070 SUPER (sm_89), Ryzen 5 5600X, WSL2; no SSH

## Status

This wave applies the useful parts of eight independent deliverables before
starting the downloaded IQ3 quality comparison.  The accepted work is exact:
it changes scheduling or losslessly encodes the existing E4M3 scale bytes.  It
does not requantize NVFP4 weights and does not relax the engine's arithmetic
order.

Current implementation status:

- fixed active-row multiplicity kernels are integrated and default-on;
- direct execution from XPR1-v2 packed scales is integrated and is now the
  packed-v2 default; `INSIGNIA_GLM53_PACKED_DIRECT=0` is the exact rollback;
- all direct store, pair, and weighted-accumulate fixtures are byte exact for
  active-row counts 1--8 and CTA widths 4/8;
- a dedicated 4096x2048 down-projection reinterpretation gate is exact for
  store and weighted accumulation at all counts and both CTA widths;
- the full 12,096-record sidecar completed its reverse in-place relayout and
  passed an all-record/all-prefix structural validation;
- the table-free E2M1 decoder is exact but rejected: seven-run medians made it
  25.2% slower than the shared-table DP4A kernel;
- packed-expanded versus packed-direct model logits are byte-identical for the
  first three generated tokens; three matched pairs show a 2.34% end-to-end
  median win and a fresh default-vs-rollback probe shows a 3.44% win.

## Deliverable triage

| Deliverable | Decision | Reason |
| --- | --- | --- |
| fixed-B multiplicity | Implement | Exact, small bounded specialization, measured store wins |
| packed-scale direct execution | Implement and model-gate | Removes scale expansion and spends Ada integer compute to reduce scale traffic |
| E2M1 compute embedding | Prototype table-free arm | Exact arithmetic candidate; tensor/MMA variants lose badly on Ada |
| certified residual INT8 | Reject | Certification overhead and residual traffic exceed the available saving |
| NVFP4 split-K | Reject | Small-M decode is not a split-K regime; extra reductions/launches lose |
| SMDP controller | Defer | Synthetic state/action model is not calibrated to this engine |
| router cascade | Defer | Requires real aligned labels and risks sensitivity-cascade divergence |
| selective falsifier | Reject for production | Synthetic quality labels cannot certify free-running behavior |

## 1. Exact fixed-multiplicity DP4A

The old row-count runtime loop is replaced by B-specialized kernels for
`B=1..8`.  The compiler sees the exact active-row count, fully unrolls the
activation loop, and removes live state for unreachable rows.  Twenty-one
serialized measurements on the real 4096x2048 down shape select separate
store and fused weighted-accumulate CTA schedules:

```text
B:                    1  2  3  4  5  6  7  8
expanded down store:  4  4  8  4  4  4  4  4
packed down store:    4  4  8  4  4  4  4  4
packed down acc:      4  4  8  8  4  4  4  4
packed gate/up pair:  8  8  8  8  8  8  8  8
```

`INSIGNIA_GLM53_NVFP4_FIXED_ROWS=0` is the diagnostic opt-out.  The exactness
gate compares every B and CTA-4/CTA-8 variant byte-for-byte against the old
runtime-row kernel.  Seven serialized post-relayout runs on the captured
2048x4096 gate fixture found exact fixed-store latency reductions of 3.2% to
26.3% across B=1..8.  Marginal medians were polluted by occasional 40--65 us
WSL scheduling spikes, so final choices use the median within-run difference
between CTA-8 and CTA-4.  Gate/up consistently prefers eight warps.  Down store
prefers four except B=3; down accumulation prefers eight only at B=3--4.
Decode-critical B=1 accumulation differs by just 0.001 us and remains at four
warps to avoid needless pressure.

Median paired `CTA8 - CTA4` latency in microseconds (negative selects CTA8):

```text
B:                  1       2       3       4       5       6       7       8
gate/up pair:   -0.378  -0.328  -0.702  -0.205  -0.267  -0.453  -0.271  -1.276
down store:     +0.561  +0.091  -0.323  +0.188  +1.008  +0.498  +0.411  +0.608
down weighted:  +0.001  +0.007  -0.049  -0.033  +0.651  +0.441  +0.479  +0.344
```

Every timed arm remained byte-exact.  Raw outputs for all 21 repetitions are
retained under `/var/lib/insignia/bench-results/s11-down-repeats/`.

## 2. Direct XPR1-v2 scale execution

Each projection has 524,288 E4M3 scale bytes.  XPR1-v2 represents them as:

```text
262,144 packed nibble bytes
+ ordered escape bytes
+ 16-byte record-local codebook
+ 1,025-entry uint32 escape-prefix directory
```

The real fixture has 4,351 escapes (0.830%) and occupies 264.3 KiB before the
region's page padding versus 512 KiB expanded.  Random access is exact: the
directory supplies the escape count before a 256-packed-byte block, a bounded
local popcount reaches the row, and a warp ballot supplies ranks inside each
32-symbol compute tile.

The new `Nvfp4PackedScaleView` is a 64-byte aligned, pointer-only launch view.
Direct fixed-B store, paired gate/up, and weighted down-accumulate kernels keep
the existing DP4A sequence and FP32 `fmaf` order.  The scale byte is decoded
once per output-row/group and reused by every active token.  No expanded-scale
scratch is read or written.

For an active XPR1-v2 `INSIGNIA_GLM53_PACKED_EXPERTS` sidecar, direct execution
is the packed-path default and intentionally acts as a complete switch: it
forces device-packed slots and the v2 GPU transport/kernel prerequisites.
`INSIGNIA_GLM53_PACKED_DIRECT=0` selects the expanded-scale rollback; v1
remains expanded. An active `INSIGNIA_GLM53_STRIPE_INDEX` overlay cannot yet
compose with packed experts and is hard-rejected, so these measurements are
single-store runs rather than C:+E: overlay results. Direct dispatch is wired
in all four production contexts:

1. scalar sparse decode;
2. MTP sparse decode;
3. batched verify/chunk prefill;
4. full-prompt whole-layer-major sparse prefill.

Host staging validates every directory endpoint, monotonic prefix, per-block
escape popcount, region bound, and declared escape count before any unchecked
device lookup.  Public wrappers reject every column count except the two real
GLM expert geometries, 2048 and 4096.

### Exactness coverage

The original captured fixture is 2048x4096 (gate/up).  Reinterpreting the same
8,388,608 weight values and 524,288 scale symbols as 4096x2048 creates a real-
byte down-shape transport fixture without claiming semantic equivalence.  This
exercises different row starts inside the prefix blocks.  Results on sm_89:

```text
gate/up 2048x4096: store, pair, accumulate exact for B=1..8, CTA=4/8
down    4096x2048: store and accumulate exact for B=1..8, CTA=4/8
benchmark exit status: nonzero on any exactness failure
```

The real XPR1-v1 sidecar completed at 12,096 records, 150.770 GiB, logical
ratio 0.9453x, and 0.782% scale escapes.  Its v2 target is 162,126,811,136
bytes (150.992 GiB), only 227.36 MiB larger.

## 3. Space-bounded v2 relayout

The ordinary relayout needs both the 150.770 GiB v1 and 150.992 GiB v2 files,
which no longer fit beside IQ3.  `tools/relayout_glm53_experts_v2.py` now has
an in-place reverse transform:

1. read the immutable v1 header/index and precompute every v2 record size;
2. prove every destination offset is at or above its source offset;
3. extend the file by only the 227.36 MiB final delta;
4. read and verify records from last to first before writing their v2 image;
5. publish the v2 header/index only after all records complete and `fsync`.

Recoverable mode journals the exact frontier and retains one source-record
backup.  Fast mode removes those extra NTFS writes for this reproducible
sidecar.  A three-record test produced a SHA-256-identical v2 file in normal
and in-place modes.  Resume preflight on the full partially converted file
proved all 12,096 offsets and the journal frontier before the fast pass.

The full fast pass completed with exit status zero in 2,787.6 seconds:

```text
records:       12,096
file bytes:    162,126,811,136 (150.992 GiB)
format:        IG53XPK1-v2
scale regions: 36,288/36,288 exact prefix recounts PASS
```

`tools/validate_glm53_experts_v2.py` checks the complete index coverage,
record keys, header-page padding, v2 span geometry, record/header codebooks,
prefix monotonicity/endpoints, and packed-nibble counts.  The first validator
version accidentally used an 8 MiB Python read buffer, turning sparse 4 KiB
probes into about 90 GiB of read-ahead; that read-only pass was stopped and the
tool changed to exact unbuffered reads before the successful validation.

The converter was hardened after the live pass with checked short-write loops,
source-index ordering/non-overlap proof, slot-to-record identity checks,
explicit index seeks, and durable journal/backup writes.  The hardened normal
path still reproduces the independent reference fixture byte-for-byte (SHA-256
`c284cac3fc56b5e0c07fe05a751a2b81eebc6079a04c6bd85a520af2e00ec851`).

During the first attempt E: reached zero free bytes.  No model bytes were
deleted.  The two independently verified, unreferenced Git-LFS incomplete
fragments (44,468,879,345 bytes) were removed, restoring about 41.4 GiB while
preserving the clean IQ3 checkout and its complete objects.  A disk-full patch
write truncated the tracked relayout script; it was reconstructed from HEAD
plus the tested changes, syntax-checked, and full-sidecar preflighted before
resuming.

## 4. Table-free E2M1 decode

The LUT path expands packed E2M1 codes through a 2 KiB shared table and a CTA
barrier.  The exact arithmetic prototype instead uses packed-u8 video
instructions (`VCMPGEU4`, `VCMPEQ4`, `VADD4`, `VSUB4`) plus two `PRMT`s.  An
exhaustive GPU test covers all 16 codewords, and the real expert output is
byte-identical to the LUT DP4A result.

Both kernels compile to 40 registers/thread with no spills.  The table-free
arm uses zero shared bytes versus 2,048 for the LUT arm.  A first loaded-system
sample misleadingly showed 11.141 versus 11.721 microseconds (4.95% faster).
Seven serialized post-relayout repetitions reversed that verdict: median LUT
DP4A was 12.332 microseconds and table-free was 15.443 microseconds, 25.2%
slower in latency.  The arithmetic arm remains benchmark-only and is rejected
for production.

## 5. Whole-model direct-execution gate

Both arms used the same v2 sidecar, 4 GiB pinned host tier, 8.132 GiB dense
FP8 cache, four readers, no DFlash, and prompt IDs
`154820,13,171,1496,2343`.  Three generated steps produced identical IDs and
digit-identical top-10 logits.  The complete 3x154,880 FP32 logit dumps are
byte-identical:

```text
greedy IDs: 220 98546 24
dump bytes: 1,858,560 each
SHA-256:    a19a80aad813e5ebdad78418c1b647ff48c2c64c77445cd7731e69d41de2978b
expanded:   78.225 s
direct:     76.765 s
```

Three matched runs per arm produced these medians:

```text
                         expanded       direct       direct win
prompt/prefill            52.613 s      50.669 s        3.84%
three-token wall          78.564 s      76.765 s        2.34%
```

The local DRAM-less E: drive delivers only about 0.39 GB/s for these O_DIRECT
records and dominates wall time, so this is already an end-to-end result under
the worst local hierarchy.  A post-promotion one-token cold probe, with no
`PACKED_DIRECT` variable, selected direct execution automatically and took
51.512 s; explicit `PACKED_DIRECT=0` took 53.285 s, a 3.44% win.  Both dumps
were byte-identical to the earlier direct oracle over all 154,880 logits.

XPR1-v2 direct execution is therefore promoted to the default within an active
packed-v2 sidecar run. It is not the engine-wide storage default. No
approximate quality gate is involved: the transformation losslessly decodes
the same E4M3 scale bytes and preserves the existing DP4A/FP32 operation order.

## Final verification and remaining remote gate

The final local build passes the sm_89 smoke test, the expert benchmark's
nonzero-on-any-failure gate, the full-vocabulary CUDA metric oracle, and 17/17
MathArena/comparator unit tests.  The downloaded 3.0074-bpw IQ2_S/IQ3_S GGUF
was then evaluated on 64 exact-prefix ArXivLean-40 rows and rejected at PPL
+4.303%, raw cosine 0.933472, centered cosine 0.889754, and Top-1 59/64; see
`audits/s11-iq3-quality.md`.  Its clone was deleted after the user-authorized
comparison, reclaiming 256.8 GiB from E: while all C:-backed evidence remains.

The remaining hardware-specific gate is to re-run the direct schedule on the
4070 Ti SUPER when the box returns.  Keep
   the local schedule as the 4070 SUPER specialization rather than assuming the
   two Ada cards have identical occupancy optima.
