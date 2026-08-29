# Packed expert sidecar (IG53XPK1 / XPR1) runtime runbook

Scope: running the packed NVFP4 expert path in `src/glm53_generate.cu`
(commits 84ca16d / 2399210 / 1359c43) on **glm-box** against the finished
sidecar `/var/lib/insignia/glm53-experts-nvfp4x.igx` (~151 GiB, 12,096
records = 42 sparse layers x 288 experts). Model root/index/FPS caches stay
on the unpacked store; ONLY routed-expert records come from the sidecar.

## 0. Known state — read before running

Audit verdict (2026-08-29, working tree + uncommitted instrumentation diff):

- **F1 (blocker, must fix first):** `ExpertStager::expand_scale_nibbles`
  (src/glm53_generate.cu:1024) loops `index < kScaleBytes` (1.5 MiB = the
  TOTAL scale bytes of all 3 projections) but is called once PER projection,
  which owns only 512 KiB. 3x overrun: reads 768 KiB out of a 256 KiB packed
  buffer, writes 1.5 MiB per projection (4.5 MiB total) into a 13.5 MiB
  window, and consumes ~3x the escape budget. Deterministic failure on the
  first staged record: `packed expert scale escape underflow` (the loop
  nibble-decodes the next projection's E2M1 body as codes and burns the
  escape budget). Fix: bound the loop by `512ull << 10` (per-projection
  scale bytes). Everything else in the codec is byte-exact vs
  `tools/pack_glm53_experts.py` (verified: nibble interleave, pshufb
  decode, escape order, codebook placement, struct layouts, endianness).
- **F2 (instrumentation gap):** the uncommitted diff's counters
  (`packed_expanded_bytes` / `packed_expand_seconds`, accessors at
  src/glm53_generate.cu:915-919) are **never printed** — the stats block at
  src/glm53_generate.cu:3980-4008 has no line for them. Wire a printf there
  or the timing is invisible. The timing itself is correct and low-overhead
  (two `steady_clock::now()` calls ~50 ns vs a ~1-3 ms transform; relaxed
  atomics; pread excluded) but it measures header check + 3x 4 MiB body
  memcpy + scale expansion, i.e. the whole post-read transform, not just
  the AVX2 expansion.
- The small test sidecar `glm53-experts-pack-test.igx` (packer `--limit`)
  can only smoke-test startup parsing (`packed experts: ...` line); any
  decode that routes to an unpacked expert throws `expert is missing from
  packed sidecar`. End-to-end validation needs the full 12,096-record file.

## 1. Pre-flight (glm-box)

```bash
ssh glm-box
# 1. no parallel engine sessions (they contend for the 32 GiB host pin and
#    trigger the halve-retry; see scratch/host-tier/p7-host-tier-analysis.md)
wsl -d Arch -- pgrep -af glm53-generate || echo clear
# 2. sidecar complete: builder's last line said
#    "wrote ...: 12096 records, ~151 GiB, logical ratio ~0.944x"
wsl -d Arch -- ls -l /var/lib/insignia/glm53-experts-nvfp4x.igx
```

Header/index sanity (no engine run; paste into `wsl -d Arch -- python3 -`):

```python
import struct, pathlib
p = pathlib.Path("/var/lib/insignia/glm53-experts-nvfp4x.igx")
hdr = struct.unpack("<8sIIIIQQQQQ", p.open("rb").read(64))
magic, ver, layers, experts, records, ioff, doff, fbytes, src, stored = hdr
assert magic == b"IG53XPK1" and ver == 1 and layers == 45 and experts == 288
assert records == 12096 and fbytes == p.stat().st_size
idx = p.open("rb").read(16 * layers * experts)
pop = 0; maxpad = 0
for e in range(layers * experts):
    off, st, pad = struct.unpack_from("<QII", idx, 16 * e)
    if off:
        assert off % 4096 == 0 and pad % 4096 == 0 and st <= pad
        pop += 1; maxpad = max(maxpad, pad)
print("populated", pop, "max padded", maxpad)  # 12096, ~12.8 MiB
```

`maxpad` is exactly what the runtime sizes its per-reader scratch from.

## 2. Build (after F1 is fixed, committed, pushed)

```bash
# dev box: push the fix, then on glm-box pull the dedicated worktree
git -C C:\coding\Insignia-glm53-dflash2 pull --ff-only

# build inside Arch WSL (raptor-tuned host code = AVX2 expand path)
wsl -d Arch -- bash -c 'cd /mnt/c/coding/Insignia-glm53-dflash2 && \
  INSIGNIA_BUILD_DIR=/var/tmp/insignia-build-raptor bash build/glm53-gen.sh'
```

Binary: `/var/tmp/insignia-build-raptor/glm53-generate` (nvcc sm_89,
`-Xcompiler=-march=raptorlake`). `/var/tmp` persists across WSL recycles.

## 3. Enabling the packed path

One knob, read once at ExpertStager construction (src/glm53_generate.cu:591):

```bash
export INSIGNIA_GLM53_PACKED_EXPERTS=/var/lib/insignia/glm53-experts-nvfp4x.igx
```

- Startup confirmation line:
  `packed experts: 12096 records, ~140.9 GiB logical, ~5.6% smaller, O_DIRECT + AVX2 expand`
  (`O_DIRECT` must appear; if it says `buffered I/O`, the second open failed —
  investigate before benchmarking).
- Unset the knob = unpacked arm (the original `stage()` path is untouched).
- CLI is unchanged: `glm53-generate MODEL_ROOT MODEL.index TOKENS LAYERS GENERATE FP8PREFIX`
  with `MODEL_ROOT=/var/lib/insignia/glm53-flash-text`,
  `MODEL.index=/var/lib/insignia/glm53-flash-text.index`,
  `FP8PREFIX=/var/lib/insignia/glm53-fp8-g64`.
- Knob interactions (audited): striping is bypassed (all expert reads route to
  the drive-0 reader pool; `READERS_E` idles — fine, glm-box has one NVMe);
  `INSIGNIA_GLM53_PAGECACHE_L2` has NO effect on experts in packed mode;
  `PIN_LIST` still works (pins load through the packed path); `READERS`
  (default 4, optimal) sizes the drive-0 pool and each reader allocates its
  own ~12.8 MiB scratch.
- Recommended common env (documented defaults):
  `INSIGNIA_GLM53_EXPERT_CACHE_MB=32768`, `INSIGNIA_GLM53_Q8_BUDGET_MB=10240`,
  `INSIGNIA_GLM53_DFLASH2=1`,
  `INSIGNIA_GLM53_DFLASH2_FP8=/var/lib/insignia/glm53-dflash2-fp8-fixed`
  (the `-fixed` cache; the plain default has the FC layout bug).

## 4. Parity gate (packed vs unpacked) — before any perf claim

The determinism law (AGENTS.md, audits/s6-open-problems.md section 0): greedy
IDs AND digit-identical top-10 logits. Payloads are byte-identical by codec
construction, so ANY divergence means a real bug — reject regardless of speed.

```bash
wsl -d Arch
G=/var/tmp/insignia-build-raptor/glm53-generate
M=/var/lib/insignia/glm53-flash-text; I=$M.index; P=/var/lib/insignia/glm53-fp8-g64
O=/var/lib/insignia/packed-parity; mkdir -p $O
COMMON="INSIGNIA_GLM53_Q8_BUDGET_MB=10240 INSIGNIA_GLM53_EXPERT_CACHE_MB=32768"
PROMPTS=(154820 154820,13,171,1496,2343 \
  154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25)
GENS=(12 40 30 100 240)

parity() { # name  (env decides arm)
  for n in 0 1 2 3 4; do
    env $COMMON $G $M "${PROMPTS[n]}" 0 "${GENS[n]}" $P \
      > "$O/$1-$n.log" 2>&1 || true
  done
}
parity a                                   # unpacked baseline
INSIGNIA_GLM53_PACKED_EXPERTS=/var/lib/insignia/glm53-experts-nvfp4x.igx parity b

for n in 0 1 2 3 4; do
  diff <(grep -E "^position .* top10|^greedy IDs" "$O/a-$n.log") \
       <(grep -E "^position .* top10|^greedy IDs" "$O/b-$n.log") \
    && echo "case $n: parity OK" || echo "case $n: PARITY FAIL"
done
```

Pass = zero diff on every `position N top10 id:logit` and `greedy IDs` line
across the 12/40/30/100/240-token sequence checks. Also check both logs show
the same DFlash2 round counts / accepted histogram (built-in canary).

Stronger logit-level gate (mirror of build/longctx-ab.sh):

```bash
env $COMMON INSIGNIA_GLM53_LOGITS_DUMP=$O/a.f32 \
  $G $M "${PROMPTS[2]}" 0 30 $P > $O/a-ld.log 2>&1
env $COMMON INSIGNIA_GLM53_PACKED_EXPERTS=/var/lib/insignia/glm53-experts-nvfp4x.igx \
  INSIGNIA_GLM53_LOGITS_DUMP=$O/b.f32 \
  $G $M "${PROMPTS[2]}" 0 30 $P > $O/b-ld.log 2>&1
cd /mnt/c/coding/Insignia-glm53-dflash2
/var/lib/insignia/bench-venv/bin/python tools/compare_logits.py \
  $O/a.f32 $O/b.f32 --topk 10
# expect: max abs diff 0.0 on every step, cosine 1.000000
```

## 5. A/B performance bench (packed vs unpacked)

Harness: `tools/benchmark_math.py` (GSM8K + MATH-500, cold-process, scalar +
DFlash2 arms, greedy-ID parity enforced internally). It strips
`DFLASH2*`/`ALT_SHARD_DIR` but **passes `INSIGNIA_GLM53_PACKED_EXPERTS`
through**, so the packed/unpacked arm is chosen by the invoking shell:

```bash
# packed campaign (pilot)
wsl -d Arch -- bash -c 'cd /mnt/c/coding/Insignia-glm53-dflash2 && \
  INSIGNIA_GLM53_PACKED_EXPERTS=/var/lib/insignia/glm53-experts-nvfp4x.igx \
  /var/lib/insignia/bench-venv/bin/python tools/benchmark_math.py \
    --binary /var/tmp/insignia-build-raptor/glm53-generate \
    --output /var/lib/insignia/bench-results/packed-pilot \
    --samples 2 --generate 32'
# unpacked campaign: same command WITHOUT the packed env, --output ...-base
```

Protocol (per AGENTS.md / s6 conventions):

- ABAB-interleave the two campaigns, >= 3 repetitions, compare **medians**
  (WSL run-to-run swing is ~2x; never trust single readings).
- Acceptance-matched: only compare runs whose accepted/round histogram and
  round counts are IDENTICAL (they must be — divergence = parity failure).
- Record the tier slot count every run (`NVFP4 cache ... 2425 slots` for the
  32 GiB tier); discard runs where the pin halved.
- What to expect: ~5.5% fewer expert NVMe bytes per record (13.56 MiB ->
  ~12.79 MiB stored; window/slot footprint unchanged at 13.504 MiB), plus
  the added CPU expand cost — until F2 is wired, read it from the
  `QD8 expert O_DIRECT` GB/s delta and the total ms/token; after wiring,
  from the `packed expand` stats line.
- Quick single-prompt DFlash2 check (build/bench-df.sh pattern):
  `bash build/bench-df.sh 100 32768 4 INSIGNIA_GLM53_PACKED_EXPERTS=/var/lib/insignia/glm53-experts-nvfp4x.igx`

## 6. Failure-mode triage (exact engine messages)

| message | cause |
|---|---|
| `bad packed expert sidecar header` | magic != IG53XPK1 or version != 1 |
| `geometry does not match model` | header layers/experts vs config |
| `size does not match header` | file truncated / extra bytes vs file_bytes |
| `index record count mismatch` | populated entries != header.records (partial `--limit` sidecar) |
| `expert is missing from packed sidecar` | routed expert has no record (test sidecar) |
| `packed expert reader scratch is unavailable` | reader's posix_memalign failed |
| `packed expert scale escape underflow/overflow` | F1 (unfixed) or corrupt record |
| `packed expert record key mismatch` | wrong offset / corrupt index (magic/layer/expert check) |
| `truncated packed expert projection` / `trailing bytes` | stored_bytes vs layout mismatch |

All are fatal throws (main catches, prints `glm53-generate: <msg>`).

## 7. Rollback

`unset INSIGNIA_GLM53_PACKED_EXPERTS` (or run without exporting it): the
unpacked streaming path (`ExpertStager::stage`) is untouched by the feature;
`source_bytes()` falls back to the constant record size. No rebuild needed.
