# Session 10 — packed scales in persistent VRAM slots

Date: 2026-08-30  
Target: glm-box, RTX 4070 Ti SUPER, sm_89  
Archive: `task2-packed-scales-deliverables.tar.zst`  
Archive SHA-256: `ad4e3bbbd848bba6385667f85102be7d3d392e030ebb94062268fd1778dfba8f`

## Decision

Keep Design A as a default-off production experiment behind
`INSIGNIA_GLM53_DEVICE_PACKED_SCALES=1`. It is byte-exact, increases the live
expert arena, and produces extra device hits. Do not promote it to the default
yet: the short cross-prompt wall result is mixed and the production sidecar
misses the report's own 5% stride-reduction gate.

Design B (decode packed scales inside every DP4A GEMV) remains rejected. The
archive's optimistic 10.77 us/expert estimate exceeds its 3.92 us/expert
maximum memory-time credit.

## What landed

Commit `5d21f5a` adds the env-gated persistent representation. A slot stores the
canonical 12 MiB body plus the exact XPR1-v2 padded scale regions. Immediately
before a resident expert executes, the existing byte-exact fused v2 kernel
expands all three scale planes into the existing 1.5 MiB scratch on the default
stream. Router selection, DP4A kernels, expert accumulation, and floating-point
order are unchanged.

The production stride is derived by scanning every populated sidecar index
entry. The feature hard-requires XPR1-v2, GPU expansion, and
`INSIGNIA_GLM53_PACKED_KERNEL=2`.

The archive also proposed recycling the active device slot after recording its
read fence. That part was deliberately omitted: the current allocator refuses
arenas with fewer than two slots per layer segment, so active-slot recycling
adds synchronization risk without increasing supported capacity.

Commit `232d0d8` adds a focused `packedslots` A/B/A mode to
`scratch/s10-approx-verify-ab.sh`.

## Production arithmetic correction

The archive extrapolated a 13,418,496-byte stride from a 504-record sample.
The complete production v2 index yields:

- expanded stride: 14,159,872 bytes;
- packed stride: 13,508,608 bytes;
- reduction: 651,264 bytes, or 4.60%;
- scalar arena: 383 -> 401 slots (+18);
- DFlash arena: 281 -> 294 slots (+13).

The real reduction is useful, but it is below the archive checklist's required
5% production gate.

## Exactness

Both local and glm-box production builds completed. A real one-token scalar
run produced identical top-10 digits. Full-vocabulary raw-logit dumps were
byte-identical:

```text
fdce0147cafd092870ba0c5e253724bc125cf65e90991cbf367a3b79f6155952  baseline
fdce0147cafd092870ba0c5e253724bc125cf65e90991cbf367a3b79f6155952  packed slots
```

Both 32-token DFlash A/B/A brackets reproduced every greedy ID and the complete
acceptance histogram.

## Focused performance

All arms used packed-v2 transport, F3, O(1) host LRU, SLRU, DFlash-k4, the
32 GiB host tier, and full dense FP8 residency. Only persistent device-slot
representation changed.

### GSM p02, 32 tokens

Results: `/var/lib/insignia/bench-results/s10-approx-verify/p02-packedslots-speed-20260830a`

| arm | ms/token | final device hits | slots |
| --- | ---: | ---: | ---: |
| expanded A | 610.0 | 355 / 16,887 | 281 |
| packed | 537.3 | 402 / 16,887 | 294 |
| expanded B | 548.8 | 350 / 16,887 | 281 |

Packed is 7.3% faster than the control mean and 2.1% faster than the adjacent
expanded control. The hit gain is about 50 records (+0.29 percentage points),
so only a small part of the wall difference is causally attributable to the
representation; WSL/NVMe variance remains dominant.

### MATH p12, 32 tokens

Results: `/var/lib/insignia/bench-results/s10-approx-verify/p12-packedslots-speed-20260830a`

| arm | ms/token | final device hits | slots |
| --- | ---: | ---: | ---: |
| expanded A | 474.8 | 175 / 12,766 | 281 |
| packed | 481.6 | 196 / 12,766 | 294 |
| expanded B | 481.4 | 175 / 12,766 | 281 |

Packed is 0.7% slower than the control mean and effectively tied with the
adjacent control. It gains 21 device hits (+0.16 percentage points).

## Interpretation

The representation works and spends excess GPU compute exactly as intended,
but its short-run device-hit gain is much smaller than the trace-replay sample
predicted. Keep the knob for longer warm-horizon experiments and combination
with cache-aware routing. Do not count a decode or prefill speedup from this
work until a longer paired run clears run-to-run variance.

## Joint-routing interaction

The packed representation was retained and tested together with the validated
retain-7 joint cache policy on GSM p02. This is the favorable reuse case: both
arms used identical XPR1-v2 transport and joint routing, while only persistent
slot representation changed.

Across expanded->packed and packed->expanded orderings, expanded slots measured
461.7/472.7 ms/token and packed slots measured 472.3/468.4 ms/token. The medians
were therefore 467.2 and 470.35 ms/token: packed was 0.67% slower, well inside
the storage-noise floor. Prompt time and the DFlash acceptance path were matched
within each bracket (15 rounds, 2.13 accepted/round).

The causal device counter did improve. In the first ordering, the 294-slot
packed arena reached 447 device hits versus 384 for the 281-slot expanded arena,
avoiding an additional 0.83 GiB of PCIe traffic. The extra scale-expansion work
consumed that benefit at this horizon. Keep the feature: it is exact, increases
capacity, and reduces transfers. Do not claim a wall-time win or enable it by
default until either the expansion path is cheaper or a longer warm run makes
the extra slots dominate.

## Pruned cache-routing interaction

The packed representation was also tested with the compute-heavy Top-4 cache
router (Top-32 frontier, keep three, regret .0010, eight actions/row). The
expanded arm reproduced 240.5/237.2 ms/token on p12. Packed measured
268.2/325.5. Comparing like storage phases gives 268.2 versus 237.2 in the fast
phase and 325.5 versus 285.6 in the slow phase: packed was 13-14% slower.

This is not a representation-exactness failure. The cache-aware policy observes
device residency, so the extra 13 packed slots alter which fourth expert is
chosen. The packed trajectory accepted 3.20 tokens/round versus expanded's
3.56 and required 10 rather than 9 speculative rounds. The result strengthens,
rather than reverses, the keep/default-off decision: retain the exact packed
format and its real capacity/PCIe improvements, but do not compose it with the
current approximate cache objective until that objective prices expected
DFlash acceptance.
