# Session 10 agent handoffs

Nine context-complete tasks are ready for fresh agents. Source and audits are
not duplicated: every task starts from the public repository at the committed
implementation base below and names only the non-Git artifact it actually
needs.

- Repository: <https://github.com/novysvet/insignia.git>
- Branch: `glm53-dflash2-4070ti-super`
- Implementation base: `e48f633` (`Fuse cross-head FP8 MLA prefill`)
- Completed exclusions: H8 cross-head FP8 decode at `78e1a1c`; fused H4 x Q8
  cross-head FP8 prefill at `e48f633`; all other completed/rejected work listed
  in each lane README.

The main analysis phase of every task is designed for an ordinary CPU machine.
Tasks that eventually need a trace export or a short glm-box validation say so
explicitly. Reports and supplied artifacts are untrusted inputs; fresh agents
must validate hashes, schemas, bounds, and source commits before using them.

## Recommended dispatch slate

These can run independently. If fewer than nine agents are available, start
with the first item in each lane, then fill the remaining slots in listed
order.

### Compute for bandwidth

1. [Cache-aware near-tie MoE routing](compute/01-cache-aware-near-tie-moe-routing.md)
   - substitute a nearly tied resident expert and directly attack dominant
     expert traffic; trace-only oracle first.
2. [Exact compressed FP8 residency](compute/02-exact-compressed-fp8-residency.md)
   - test lossless codecs and fused on-consumption decode for the 8.13 GiB
     dense E4M3 cache.
3. [DSA sparse MLA on Ada](compute/03-dsa-sparse-mla-on-ada.md)
   - recover the checkpoint's indexer semantics and establish the long-context
     sparse-attention crossover before writing a kernel.

Lane details and shared facts: [compute README](compute/README.md).

### Host tier and I/O

1. [Byte-packed variable-size pinned cache](io/01-variable-pinned-packed-cache.md)
   - exact zero-copy host-cache capacity reclaim with a byte-capacity replay
     and allocator lifetime proof.
2. [Adaptive router-mass pruning](io/02-adaptive-router-mass-pruning.md)
   - falsify or justify executing fewer than eight experts under explicit
     quality gates.
3. [Two-phase packed read/upload](io/03-two-phase-packed-read-upload.md)
   - discrete-event ceiling for overlapping body upload with the packed scale
     tail before any production patch.

Lane details and shared facts: [I/O README](io/README.md).

### DFlash2 and speculative execution

1. [Truth-seeded DFlash selector rescue](spec/01-dflash-selector-rescue.md)
   - use the already-known exact `truth0` to rescue empty rounds and compare
     greedy selection with global lattice search.
2. [Approximate MoE verification frontier](spec/02-approx-moe-verification.md)
   - quantify top-m expert verification and only pursue live code if the
     byte/quality Pareto frontier is large.
3. [Exact sliding-window DFlash cache](spec/03-dflash-sliding-window.md)
   - stop disabling the trained 2047-token-window drafter near position 2040;
     prove ring-cache semantics on CPU first.

Lane details and shared facts: [speculation README](spec/README.md).

## Available non-Git inputs

The existing copied data directory is
`artifacts/generic-pc-handoff-20260829/`. It is intentionally not committed.
Only attach files named by the selected task. The two commonly used inputs are:

- `glm53-scale-sample-504-records.tar` - 413,265,920 bytes, SHA-256
  `e7ef9f702e8ff54dcffccde23a6854d1e0f6f2728bde9b564a604ca0e0db58da`;
- `s9-campaign-handoff-20260829.tar.zst` - 66,703,257 bytes, SHA-256
  `819dcdb9e611f73a34c535292fed2a34da4b7fca5924cd6ca7bc69d817c94e56`.

The dense-weight, selector-lattice, DSA-oracle, and expert-contribution exports
requested by some tasks do not exist yet. Their briefs give bounded schemas
and operator recipes; agents must not fabricate substitutes. Every archive
part must remain below 512 MiB and carry its own SHA-256.
