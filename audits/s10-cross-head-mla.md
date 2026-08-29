# Session 10 - compute-for-bandwidth MLA wave

Date: 2026-08-30  
Target: glm-box, RTX 4070 Ti SUPER sm_89, i7-14700KF, single NVMe  
Branch: `glm53-dflash2-4070ti-super`  
Public repository: <https://github.com/novysvet/insignia.git>

## Outcome

This wave spent idle Ada compute in three places instead of moving or retaining
larger FP32 structures:

1. The exact 256-token MLA bridge now stores 512-wide FP32 latents and
   reconstructs expanded K/V into one shared layer-major work area. This
   replaces 352 MiB of per-layer expanded K/V with 37.5 MiB, a net 314.5 MiB
   reclaim (about 23 more routed-expert slots).
2. Long-context decode gained an opt-in H8 cross-head E4M3 tensor-core score
   kernel. It stages each latent tile once for eight heads and reaches 2.66x
   kernel speed at context 8192.
3. Long-context prefill gained an opt-in persistent H4 x Q8 E4M3 kernel. One
   CTA scans the full causal prefix and directly projects the normalized latent
   through compact `W_uv`; this removes the first prototype's 16 MiB partial
   scratch and all sequential partial/merge launch pairs. It reaches 1.53x at
   the 8192 boundary and wins 1.31-1.69x on useful 128-row chunks.

The exact bridge is suitable for the strict parity path. The cross-head kernels
are deliberately approximate and remain behind
`INSIGNIA_GLM53_MLA_CROSS_HEAD_FP8=1`. They are intended for the user's
speed-first mode: measured output quality is strong, but routing can diverge.

## Exact latent-prefix reconstruction

Commit `bcd2d47` stores each MLA layer's first 256 compressed latents as FP32
and reconstructs consecutive layer-major K/V rows into a shared 32 MiB buffer.
The reconstruction is incremental, so a later prompt chunk does not rebuild
rows already materialized for that layer.

GPU comparisons against the old expanded bridge were bit exact:

- 131,072/131,072 projection floats at 128 rows;
- 65,536/65,536 decode floats through position 255;
- 65,536/65,536 two-chunk FA2 prefill floats through position 255;
- production smoke: identical greedy IDs and top logit.

A focused 175/250-token boundary A/B reproduced IDs. Decode ranged from -6.9%
to +2.5% and prefill from -2% to +8% under large I/O variance, so no wall-time
claim is attached. The deterministic 314.5 MiB reclaim and the observed expert
tier increase from 292 to 315 slots are the result.

## H8 cross-head FP8 decode

Commit `78e1a1c` computes `q_eff = q W_uk` once per head from resident compact
E4M3 weights, quantizes it group-64, stages each 64-key latent microtile once
for eight heads, and evaluates scores with
`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`. The existing compact
`W_uv` merge is reused. `ptxas` reports 104 registers, 41,216 bytes shared
memory, two CTAs/SM possible, and zero spills.

| Context | scalar compact absorb | H8 cross-head | speedup | rel-L2 | cosine |
|---:|---:|---:|---:|---:|---:|
| 257 | 0.3129 ms | 0.3051 ms | 1.03x | 0.001518 | 0.9999989 |
| 512 | 0.4078 ms | 0.3971 ms | 1.03x | 0.005505 | 0.9999849 |
| 1024 | 0.4433 ms | 0.3975 ms | 1.12x | 0.005817 | 0.9999831 |
| 2048 | 0.5890 ms | 0.3979 ms | 1.48x | 0.005891 | 0.9999827 |
| 4096 | 0.9324 ms | 0.4832 ms | 1.93x | 0.005957 | 0.9999823 |
| 8192 | 1.5659 ms | 0.5880 ms | 2.66x | 0.006062 | 0.9999817 |

One synthetic 300-token production smoke reduced measured decode wall from
4.623 s to 3.646 s for eight scalar-generated rows (-21.1%). The first four
IDs matched and the fifth diverged, after which MoE routing also diverged. This
is evidence that the approximation is usable and non-degenerate, not a clean
end-to-end attribution or parity claim.

## H4 x Q8 fused FP8 prefill

The first implementation wrote one partial per 512-key tile and launched a
partial/merge pair for every eight queries. Quality was good, but it lost below
4K and only reached 1.23x at 8K. It was not accepted.

Commit `e48f633` replaces it with one persistent CTA per (four heads, eight
queries) tile. The CTA keeps 32 weighted-latent numerators in registers while
scanning the causal prefix, then reuses its shared score slab to perform all 32
compact `W_uv` projections. The 16 MiB global partial scratch disappears.
`ptxas` reports 118 registers, a 128-byte stack frame, 60,416 bytes dynamic
shared memory, one CTA/SM, and zero spills.

Focused local 4070 SUPER measurements (ten timed repeats):

| Position base | Rows | scalar prefill | fused H4 x Q8 | speedup | rel-L2 | cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 16 | 2.9037 ms | 2.4752 ms | 1.17x | 0.005393 | 0.9999855 |
| 1024 | 32 | 7.2902 ms | 6.0039 ms | 1.21x | 0.005558 | 0.9999846 |
| 1024 | 64 | 12.8511 ms | 9.2302 ms | 1.39x | 0.005461 | 0.9999851 |
| 256 | 128 | 20.4698 ms | 12.2060 ms | 1.68x | 0.001965 | 0.9999981 |
| 1024 | 128 | 21.0006 ms | 15.9732 ms | 1.31x | 0.005235 | 0.9999863 |
| 4096 | 128 | 44.6601 ms | 31.2294 ms | 1.43x | 0.005895 | 0.9999827 |
| 8064 | 128 | 78.4926 ms | 51.3082 ms | 1.53x | 0.006020 | 0.9999819 |

At base 256, one to eight rows are 0.74-0.90x and therefore retain the scalar
kernel. The runtime selects fused prefill for at least 16 rows, or for any tail
starting at position 4096 where the measured one/eight-row cases were
1.03-1.15x. All tested outputs were finite; the worst rel-L2 was 0.006052 and
the worst cosine was 0.99998175.

## Production boundary check

A deliberately narrow ABAB check used a 300-token repeated-token prompt,
full-prompt layer-major execution, packed-v2/F3/O(1)-LRU/SLRU, the exact prefix
reconstruction, 32 GiB pinned cache, and one generated token.

| Arm | Prompt walls | Median | Greedy ID |
|---|---|---:|---:|
| scalar cross-head off | 15.169 s, 15.667 s | 15.418 s | 220 |
| approximate cross-head on | 15.616 s, 18.473 s | 17.045 s | 220 |

This does **not** falsify the kernel microbenchmark. The second approximate run
fell to 3.73 GB/s expert O_DIRECT versus 4.68-4.69 GB/s in the scalar runs and
spent 12.933 s blocked on expert reads. The approximation also changed logits
and routed records (3708 versus 3718), so cold-process I/O dominates the few
milliseconds saved across 11 MLA layers. No whole-model prefill speedup is
claimed from this pair, and no longer campaign was run.

## Stability and operating decision

The remote build and all focused runs completed without CUDA failures, Xids,
PCIe faults, non-finite values, or token collapse. After load, glm-box reported
54 C, 12,501 MHz memory, P0, and normal idle-downclocked core state. The one
`PCI: Fatal: No config space access function found` line is the existing WSL
boot message, not an NVIDIA Xid.

- Exact prefix reconstruction: keep available for strict mode; promote only
  after a longer cache-warm wall test if desired.
- H8 decode and fused H4 x Q8 prefill: keep together behind
  `INSIGNIA_GLM53_MLA_CROSS_HEAD_FP8=1`.
- Quality contract for that knob: speed-first approximation, not digit parity;
  require finite coherent output plus an explicit quality sample before any
  future default-on decision.

## Focused benchmark reproduction

The synthetic quality/timing harness is
`build/test-mla-cross-head-prefill.cu`. Build and run it inside the Arch WSL
guest with:

```bash
/opt/cuda/bin/nvcc -ccbin /usr/bin/g++-15 -arch=sm_89 -O3 \
  --use_fast_math -lineinfo -Xptxas=-v -std=c++20 \
  -I/mnt/e/coding/Insignia/include \
  /mnt/e/coding/Insignia/src/glm53_ops.cu \
  /mnt/e/coding/Insignia/build/test-mla-cross-head-prefill.cu \
  -o /var/tmp/test-mla-cross-head-prefill
/var/tmp/test-mla-cross-head-prefill
```

The harness compares the shipping compact absorbed prefill against the fused
path at short tails and at 256/512/1024/2048/4096/8064 position bases, rejects
non-finite output, gates rel-L2/cosine, and times ten repetitions per arm.

## Commits

- `bcd2d47` exact compressed-prefix reconstruction
- `78e1a1c` H8 FP8 cross-head decode and incremental prefix reconstruction
- `e48f633` fused persistent H4 x Q8 FP8 prefill
