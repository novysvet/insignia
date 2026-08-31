# Session 11: UD-IQ3_XXS quality gate

Date: 2026-08-31  
Hardware: local RTX 4070 SUPER + Ryzen 5 5600X, 14 GiB WSL2 guest  
Reference: exact NVFP4/XPR1-v2 direct execution, DFlash disabled

## Verdict

Reject this downloaded GGUF for the current engine target.  It exceeds the
user's 3.5% teacher-forced PPL-regression ceiling on canonical ArXivLean
problem 40:

```text
NVFP4 PPL:       1.1317
IQ3-mix PPL:     1.1804
PPL ratio:       1.043030
PPL regression: +4.303%  (FAIL; ceiling +3.500%)
```

The repository calls the quant `UD-IQ3_XXS`, but GGUF tensor metadata shows a
3.0074-bpw mixture dominated by 61.761% IQ2_S and 30.881% IQ3_S tensors.  It is
not an all-IQ3_XXS payload.  The text model occupied 112.310 GiB (113.394 GiB
including mmproj).  This result evaluates that concrete artifact, not every
possible 3-bit quantizer.

## Alignment contract

Only MathArena `problem_idx=40` was run; this is parquet ordinal 39 and has a
938-token prompt.  It is the canonical hard p40 case used by prior Insignia
work, not the 431-token problem at literal parquet ordinal 40.

The exact NVFP4 free trajectory supplied 64 target token IDs.  IQ3 consumed
exactly `prompt + targets[:-1]`:

```text
prompt IDs:             938
teacher-forced targets:  64
IQ3 input IDs:         1001
logit rows per arm:      64
vocabulary:          154880
bytes per dump:     39649280
```

Both dumps are raw little-endian FP32 full-vocabulary rows.  No tokenizer text
round-trip, sampling, DFlash, approximate MoE policy, or Top-K logit sketch is
inside the measurement.  The free exact rows are reused as the teacher-forced
reference because they have the identical exact prefixes; this removed a
redundant second 64-token NVFP4 pass.

## Full-vocabulary result

`tools/compare_logits.py` compared all 64 rows with the exact target IDs:

| metric | mean | median | worst/max |
| --- | ---: | ---: | ---: |
| raw cosine | 0.933472 | 0.951492 | 0.978307 max |
| raw relative L2 | 0.3505 | 0.3163 | 0.7642 |
| raw MSE | 0.8970 | 0.7478 | 2.330 |
| centered cosine | 0.889754 | 0.909574 | 0.951348 max |
| centered relative L2 | 0.4552 | 0.4223 | 0.7747 |
| centered MSE | 0.8538 | 0.7200 | 2.231 |
| KL(NVFP4 || IQ3) | 0.06558 | 0.002003 | 0.6414 |
| KL(IQ3 || NVFP4) | 0.06583 | 0.001535 | 0.7428 |
| Jensen-Shannon | 0.01424 | 0.0004443 | 0.1374 |
| max absolute Top-10-union delta | 3.297 | 3.135 | 7.776 |
| mean absolute Top-10-union delta | 1.332 | 1.273 | 3.364 |

Additional discrete results:

```text
Top-1 agreement: 59/64 = 92.19%
Top-1 mismatches: steps 6, 13, 20, 38, 39
Top-10 overlap:   7.59/10 mean, 5/10 minimum
NVFP4 NLL:        7.9164 total, 0.1237 mean
IQ3 NLL:         10.6127 total, 0.1658 mean
```

No free-run Lean solve-rate claim is made.  The numerical gate already failed,
so spending another 27-minute GGUF fit plus slow autoregressive decode would
not rescue this artifact under the stated policy.

## Runtime observations

The NVFP4 reference measured 459.018 s for the 938-token prompt (2.0435 tok/s)
and 1,275.143 s total for 64 tokens.  Decode was 12,751.95 ms/token or 0.07842
tok/s because the local packed sidecar was read from E: through a 0.39 GB/s
path and the 8 GiB host-tier request fell back to 303 pinned slots.  This is a
quality/reference run, not the remote-box performance target.

The successful IQ3 teacher-forced pass took 39m53s.  Its 112 GiB GGUF lived on
the DRAM-less E: drvfs mount; model fitting/page faults left most CPU cores and
VRAM idle.  `-ngl 4` was not honored effectively by this llama build/model
combination (about 1.27 GiB VRAM resident).  Therefore this timing is a warning
about this GGUF/mmap placement, not a forecast for hypothetical native
Insignia IQ3 kernels.

Artifacts retained on the C:-backed WSL volume:

```text
/var/lib/insignia/bench-results/iq3-xxs-arxivlean40/arxivlean-40/
  exact.log
  exact-result.json
  exact-quality-logits.f32
  iq3-quality-logits.f32
  iq3-vs-nvfp4-metrics.log
  prompt.csv
  prompt.txt
  forced.csv
  forced-input.csv
```

The user explicitly authorized deleting the downloaded IQ3 clone after this
comparison.  The resolved target
`E:\coding\Insignia\GLM-5.3-Flash-UNCENSORED-GGUF` was verified to remain
inside the workspace and then permanently removed.  E: free space increased
from 44,405,731,328 to 320,137,748,480 bytes: 275,732,017,152 bytes (256.8
GiB) actually reclaimed because checkout/LFS storage consumed more physical
space than the 113.4 GiB logical model payload.  It can be re-fetched from
Hugging Face if a later kernel project needs the source artifact again.
