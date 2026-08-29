# Patch sketch 2 — completion-order expert upload in the verify (ordered) path

Rank 2 in ranked-opportunities.md. Expected: bounded by the gap between the
measured wall and the PCIe floor (d(k) x 610 us), i.e. 0-180 ms per k=8
round; realistic 20-60 ms (2-5%) on miss-heavy rounds. Instrument first,
enable behind an env knob, A/B with the parity gate.

## Problem

`Runner::moe_multi` processes the batch strictly in slot order
(src/glm53_generate.cu:3600-3649): `expert_stager_->upload(slot)` CPU-blocks
in `wait_window` (glm53_generate.cu:805-810) until THAT record's read
completes, even when a later slot in the same 8-record batch is already
resident in the pinned window. During the stall both the copy engine and the
SMs idle. With 4 readers returning out of order, the head-of-line wait per
batch is a fraction of one read (2.4-3.9 ms disk, ~0.6 ms host-hit copy) and
there are ~d(k)/8 = 241 batches per k=8 round.

## Why it is parity-safe ONLY in the ordered (verify) path

`ordered_accumulation` (= `kda_archive_`, set for verify passes; flag at
glm53_generate.cu:3543) already exists precisely because expert visit order
must not leak into the routed sums:

- Each expert's down output goes to a PRIVATE slot
  `c_expert_out_[(token*topk + pick_slot)*hidden]`
  (`nvfp4_gemv_dp4a_quantized_rows`, 3637-3641; buffer sized at 2008); every
  (token, pick_slot) is written exactly once.
- The routed sum is assembled AFTERWARDS in a fixed token/pick-slot loop by
  `scale_add_kernel` (3652-3659) — a deterministic fmaf sequence independent
  of expert processing order.
- All other per-expert state (nv workspaces, gate_/up_, c_gateu_/c_up_) is
  transient within one expert's chain.

So reordering experts changes no floating-point accumulation order. The
non-ordered path (`nvfp4_gemv_dp4a_acc_quantized_rows`, fmaf directly into
`c_routed_` in expert order, 3643-3647) MUST keep batch order — the knob is
ignored (assert) unless `ordered_accumulation`.

## Sketch (env `INSIGNIA_GLM53_UPLOAD_DONE_ORDER=1`)

Move the per-expert block inside a completion poll loop; everything else
(users collection, row grouping, kernel launches) is moved verbatim —
same kernels, same row sets, same per-row chains.

```cpp
// glm53_generate.cu, inside the base_slot loop (after load_batch at 3599):
if (ordered_accumulation && upload_done_order) {
    std::array<bool, 8> served{};
    int remaining = batch_count;
    while (remaining) {
        int pick = -1;
        for (int slot = 0; slot < batch_count; ++slot)
            if (!served[slot] && expert_stager_->window_ready(slot)) { pick = slot; break; }
        if (pick < 0) {           // nothing ready: brief yield, readers keep working
            std::this_thread::yield();
            continue;
        }
        served[pick] = true;
        --remaining;
        const int expert = distinct[base_slot + pick];
        // ... upload(pick) + the verbatim 3608-3649 block for `expert` ...
    }
} else {
    // existing in-order loop, untouched
}
```

`ExpertStager` needs one tiny accessor (read-only, no wait):

```cpp
// true once the window's disk read finished (success or error); the
// subsequent upload() call then takes the fast path through wait_window.
bool window_ready(int slot) const {
    const int window = batch_window_[slot];
    require(window >= 0, "expert slot has no record in flight");
    return window_done(window);
}
```

(`window_done()` already exists at glm53_generate.cu:1217-1220 and is the
same predicate `load_batch` uses at 731; it takes the pool mutex, so poll
with the 50-100 us sleep/yield cadence rather than a hot spin — that mutex
is also taken by reader completions.)

## Measurement plan (before/after numbers, medians per AGENTS.md)

- Existing counters: `read_wait_seconds_` (upload stalls, 806-810),
  `batch_read_ends_`, io_bytes_. Add: per-layer sum of head-of-line waits
  (time from "first window_ready" to "batch order reached that slot").
- A/B: k4/k7/k8 rounds on the GSM8K/MATH-500 driver
  (tools/benchmark_math.py), 3+ repeats, medians; parity gate
  (greedy IDs + top-10 logits + sequence checks) on both modes.
- If wall − d(k)·610 us is already <20 ms on the test prompts, the knob is
  not worth the branch — keep the instrumentation only.

## Risks / notes

- Busy-poll must not burn a core that a reader thread needs: yield loop with
  a short sleep cap (e.g. 50-100 us) is plenty; reads take ~ms.
- Determinism of TIMING changes (which slot wins a near-tie) is fine: outputs
  are order-independent by the argument above; the parity gate must still be
  run because that argument, like every other order claim here, is only
  trusted after the gate says so.
- Do NOT combine with the non-ordered prefill path or with
  `INSIGNIA_GLM53_VERIFY_CACHE_ALL` A/Bs in the same run — one variable at a
  time.
