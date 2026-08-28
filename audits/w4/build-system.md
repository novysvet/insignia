# Build-system audit (w4) — tools/mk.py, build/*.bat, flags, ENGINE27 closure

Audit performed read-only. No build was run. mk.py's logic was verified by
importing it as a module and simulating its pure file operations (dep closures,
collision scan, the flags-wipe path) with stub files in a temp dir — no
nvcc/cl/bat was ever invoked.

Scope: `tools/mk.py`, `tools/rundll.py`, `build/*.bat` (37 files), `src/`
(49 TUs), `include/` (13 headers).

---

## 1. tools/mk.py correctness

### 1.1 dep_scan include closure — TRANSITIVE, SOUND FOR THE CURRENT LAYOUT

- The closure is genuinely recursive. Real chains exist and are followed:
  `insignia_qwen35.hpp -> insignia_storage.hpp -> insignia_model.hpp` and
  `insignia_decode.hpp -> insignia_qwen35.hpp -> ...`. Measured closures:
  `src/generate.cu` = 7 files, `src/qwen35.cu` = 5, `src/attention.cu` = 2.
  A nested-header edit (e.g. `insignia_model.hpp`) invalidates every TU that
  reaches it transitively. Transitive header edits ARE caught.
- Resolution order `(dirname(src), "src", "include")` matches compiler
  behavior for the current layout. All 13 headers live in `include/`; `src/`
  contains ZERO headers, so the src/include shadowing ambiguity cannot arise
  today. Nested source dirs also work: `src/sub/x.cu` including `"y.h"`
  resolves via `dirname` first, then `src`, then `include` — as long as
  includes stay bare names (they all are).
- Verified zero missed includes: every quoted include in src resolves to a
  repo file; there are NO repo-local `<>` includes; no macro-expanded include
  names.
- Known gaps (all acceptable, one worth noting): (a) system/CUDA headers are
  not tracked — a CUDA toolkit upgrade invalidates nothing; `--clean` is the
  manual answer (the flags stamp does not cover toolchain version). (b)
  includes under `#if 0` are counted — over-approximation, safe direction.
  (c) A quoted include that resolved nowhere would be silently dropped rather
  than reported — none exist today; a one-line warning would harden it.

### 1.2 flags-hash wipe guard — TWO REAL DEFECTS (one crashes)

The stamp logic (`compile_changed`, mk.py:127-132):

**Defect 1 — guard runs too late (silent staleness).** The stamp is only
checked when at least one TU is already stale (`if not changed: return 0`
precedes it). Flip flags with a clean tree (e.g. set `INSIG_PTXAS_V=1`, or
edit BASE_FLAGS) and mk links OLD objects compiled under the previous flag
set; the stamp is never even rewritten. Flipping flags back and forth across
clean builds leaves the cache permanently describing the wrong flag set.

**Defect 2 — wipe + partial staleness = crash.** When the guard does fire it
`rmtree`s the whole OBJDIR but then recompiles ONLY the TUs that were already
stale. `link()` immediately calls `os.path.getmtime` on the other TUs' now-
deleted objs → `FileNotFoundError` traceback. Reproduced by pure-logic
simulation (2 fake TUs, flags A→B, touch one source: after the wipe only
`f1.obj` exists, link dies on `f2.obj`). Fix is two lines: hoist the stamp
check above the `if not changed` early-out, and on wipe set
`changed = [(s, obj_for(s)) for s in srcs]` (all of the current target's
sources), not just the previously-stale subset.

Minor: `sig` doesn't include the nvcc version — fine, `--clean` after toolkit
upgrades covers it. `open(stamp)` is never closed (CPython refcounting saves
it). Toggling `-Xptxas -v` (codegen-neutral, verbose-only) needlessly wipes
the whole cache — harmless, conservative.

### 1.3 OBJDIR basename collisions — NONE TODAY

Programmatic scan of every source referenced by TARGETS plus every file on
disk in `src/`: **zero stem collisions**. `generate.cu`/`generate27.cu`,
`nll.cu`/`nll27.cu`, `dump_layers.cu`/`dump_layers27.cu` all key to distinct
obj names. The design hazard is future `src/foo.cu` + `src/foo.cpp` pairs or
same-stem files in different dirs — flat basename keying has no guard against
that. A second, sneakier collision: if the cl path ever gains
`/Fo:build\obj`, cl's `model_file.obj` would overwrite nvcc's (see 1.5/§5).
mk also fixed the bat-era implib collision (`dump-i4-seams.bat` and
`dump-layers-i4.bat` both wrote `build\dli4x.lib`); mk's
`OBJDIR\{name}x.lib` is unique per target.

### 1.4 link freshness (newest_obj vs out mtime)

Rule: skip link iff `mtime(out) > max(mtime(objs))`; compile side uses
`newest(deps) >= mtime(obj)`. Properties, in the order asked:

- **Clock skew:** a future-dated source/header makes `newest(deps) >= obj`
  permanently true → that TU (or every TU including that header) recompiles
  on EVERY invocation, and each recompile makes the obj newer than the out →
  relink every time. Permanent rebuild loop until the timestamp is fixed.
  Checked the repo today: no future-dated files in src/include/tools (all
  mtimes <= now). No guard exists (a `min(mtime, now)` clamp would be a
  three-line fix) — acceptable until it actually bites.
- **Edited-source-identical-obj:** pure mtime comparison, no content hash —
  an edit that restores byte-identical content still recompiles + relinks.
  No ccache-style dedup. Accepted cost of the design.
- **Equal timestamps:** compile uses `>=` (conservative rebuild), link uses
  `>` (conservative relink) — a safe pairing; on NTFS (100 ns granularity)
  same-tick collisions are rare and both resolve after one extra pass.
- **Clock set backwards** after a build → one spurious relink, self-healing.
  Deleted `out` relinks (isfile check). Deleted sources leave orphan objs in
  the cache — harmless, `--clean` removes.

### 1.5 the cl host path (test-model / test-cpu)

- Hardcoded `if name in ("test-model", "test-cpu")` — a third host target
  needs a source edit; a `host=True` spec key would be cleaner.
- **Requires `cl` on PATH**: works via `build\mk.bat` (which vcvars64's
  first) but a bare `python tools\mk.py test-cpu` from a plain shell dies
  with a raw `FileNotFoundError` traceback instead of a clean message.
- **No `/Fo`** — cl drops `model_file.obj`, `test_cpu.obj` in CWD = repo
  ROOT. Evidence on disk: `hello.obj`, `nul.obj` (the Windows reserved-name
  footgun), `model_file.obj`, `test_cpu.obj` at root. Fix:
  `/Fo:build\obj\host\` — never `/Fo:build\obj` (see 1.3).
- **No `/utf-8`** — test_cpu.cpp has 563 non-ASCII bytes and
  insignia_cpu.hpp 1690 (UTF-8 em-dashes etc.). Today it links, but cl is
  interpreting them in the ANSI/OEM codepage; C4819 warnings now, real
  mis-tokenization risk later (a multibyte tail byte of 0x5C or 0x22).
  Add `/utf-8`. (See §3.3.)
- Behavioral delta vs the old `test-model.bat`: mk adds `/DNDEBUG` (zero
  `assert(` usages exist in those TUs — verified, so inert) and
  `/arch:AVX2` (matches the 5600X; keep).

### 1.6 rundll.py invocation — CORRECT

`[sys.executable, tools\rundll.py, build\<t>.dll, *args]`, `cwd=ROOT` —
identical protocol to the bat era (`generate.bat`, `test-pair*.bat`,
`test-prefill.bat`). rundll tries `dll_run` (wchar, SHIM_W) then falls back
to `dll_run_c` (narrow, SHIM_C) via AttributeError — covers both shims; the
argv[0]-as-dummy convention is preserved; the Smart-App-Control error-4551
retry loop (12 x 10 s) is intact. Only nit: `RUNDLL` is a relative path —
fine under `cwd=ROOT`.

### 1.7 Stale future flags + TARGETS coverage (the exact list)

Stale `future=True` (sources EXIST — flag is wrong):
| target | source | status |
|---|---|---|
| `test-cpu` | `src/test_cpu.cpp` (637 lines) | STALE — drop `future` |
| `io-bench` | `src/io_bench.cu` (447 lines) | STALE — drop `future` AND fix shim (below) |

`future=True` CORRECT (sources missing — expected, mark as such):
`generate27` (`src/generate27.cu` absent), `nll27` (`src/nll27.cu` absent),
`dump-layers27` (`src/dump_layers27.cu` absent).

Coverage scan (every file in src/ vs every TU referenced by TARGETS):
- **`src/streaming.cu` is the ONLY source on disk not in any target** —
  555-line host-only NVMe/pinned-ring/LayerFeeder TU (compiled as .cu to
  link cudart for `cudaHostRegister`). Nothing includes
  `insignia_streaming.hpp` yet — it is dangling, awaiting the 27B tools.
  Belongs in ENGINE27 (§4).
- `io-bench` target EXISTS — it is not missing; it is mis-flagged
  (`future`) and **mis-shimmed**: `io_bench.cu` defines narrow
  `main(int, char**)` (io_bench.cu:309) but `dll(CORE + ...)` defaults to
  SHIM_W (`dllshim.cu` forwards to `wmain`) → unresolved `wmain` the moment
  the future flag is removed and someone builds it. Must be `shim=SHIM_C`
  (rundll's `dll_run_c` fallback already supports it). CORE
  (model_file/storage) is also unneeded — io_bench.cu includes no repo
  headers at all (windows.h + std only).
- Every other src file is referenced by >= 1 target. `model_file.cpp` in the
  `test-cpu` src list is unnecessary (insignia_cpu.hpp is self-contained;
  test_cpu.cpp's dep closure is just cpu.hpp + itself) — 45 lines, harmless.
- Shim audit of all 16 dll targets: 15/16 OK. Only `io-bench` mismatched.
- One default-run regression vs bats: `generate-ids` mk default run is
  `[IDX]` only; `generate-ids.bat` ran IDX + a 12-token id list
  (`248045 846 198 9419 248046 ...`). If that tail is still the canonical
  smoke input, copy it into the run line.
- `IDX27` is defined but referenced by nothing — reserve it for the 27B run
  lines when they land (fine).

---

## 2. Bats inventory — per-file verdict

Legend: **DELETE** = superseded by an mk target (compile+link equivalent,
plus cache); **KEEP** = does something mk can't; **DEAD** = cannot serve its
original purpose anymore.

| bat | verdict | notes |
|---|---|---|
| `mk.bat` | KEEP | The driver entry (vcvars64 + `python tools\mk.py`). |
| `sani.bat` | KEEP | compute-sanitizer `--tool memcheck` wrapper; not a builder; mk has no equivalent. |
| `smoke.bat` | KEEP (for now) | Builds+RUNS 3 tools (insignia-smoke.exe, test-mxfp4, bench-mxfp4-mlx). All three builds exist as mk targets (`smoke`, `test-mxfp4`, `bench-mxfp4-mlx`) but mk cannot run multiple tools in one invocation, and the first stage's output name differs (`insignia-smoke.exe` vs mk's `smoke.exe`; bat also omits `-Iinclude` for it — smoke.cu is standalone). Delete once mk grows a group/run-all or the 3-cmd chain is documented. |
| `generate.bat` | DELETE | == `mk generate` + rundll `%*`. |
| `nll.bat` | DELETE | == `mk nll` (run via tools/nll_compare.py). |
| `dump-attention.bat` | DELETE | == `mk dump-attention` (bat closure is a subset of ENGINE; mk superset is cached). |
| `dump-layer0.bat` | DELETE | == `mk dump-layer0`. |
| `dump-layer3.bat` | DELETE | == `mk dump-layer3`. |
| `dump-layers.bat` | DELETE | == `mk dump-layers`. |
| `dump-layers-i4.bat` | DELETE | == `mk dump-layers-i4`; bat shared `dli4x.lib` with dump-i4-seams (mk fixed). |
| `dump-i4-chunk.bat` | DELETE | == `mk dump-i4-chunk`. |
| `dump-i4-seams.bat` | DELETE | == `mk dump-i4-seams`. |
| `dump-multistep.bat` | DELETE | == `mk dump-multistep`. |
| `dump-pf.bat` | DELETE | == `mk dump-pf`. |
| `generate-ids.bat` | DELETE (after copying the 12-token run tail into mk) | == `mk generate-ids` build. |
| `test-argmax.bat` | DELETE | == `mk test-argmax`. |
| `test-attention.bat` | DELETE | == `mk test-attention`. |
| `test-checkpoint.bat` | DELETE | == `mk test-checkpoint`. |
| `test-deltanet.bat` | DELETE — but see flag note | Bat builds WITHOUT `--use_fast_math`; mk builds WITH it (§3.1). Test passes either way (3e-3 rel tolerance); decide the canonical flags once, then delete. |
| `test-fp8.bat` | DELETE | == `mk test-fp8`. |
| `test-full-model.bat` | DELETE | == `mk test-full-model`. |
| `test-generate.bat` | DELETE | == `mk test-generate`. |
| `test-i4.bat` | DELETE | == `mk test-i4` minus an unnecessary `prefill.cu` (test_i4.cu includes the prefill header but calls only gemv functions — mk's shorter list is correct; the bat was overspecified). |
| `test-layer.bat` | DELETE | == `mk test-layer`. |
| `test-model.bat` | DELETE | == `mk test-model` modulo mk's extra `/DNDEBUG /arch:AVX2` (inert/keep — 1.5). |
| `test-mtp.bat` | DELETE | == `mk test-mtp`. |
| `test-ops.bat` | DELETE | == `mk test-ops`. |
| `test-pair.bat` | DELETE | == `mk test-pair` (+rundll `%*` == mk args-replace mode). |
| `test-pair-chain.bat` | DELETE | == `mk test-pair-chain`. |
| `test-prefill.bat` | DELETE | == `mk test-prefill`. |
| `test-qwen35.bat` | DELETE | == `mk test-qwen35` (bat omitted mxfp4_i4.cu; mk superset fine). |
| `bench-mxfp4.bat` | DELETE | == `mk bench-mxfp4` exactly (same srcs, SHIM_C). |
| `bench-gemm.bat` | DELETE | == `mk bench-gemm`; bat included a superfluous `mxfp4.cu` (bench_gemm.cu includes only insignia_layout.cuh; gemm.cu is self-contained) and wrote `build\bench-mxfp4.dll`, clobbering bench-mxfp4's output (mk's fixed names fix this). |
| `bench-gemm-blocked.bat` | DELETE NOW | DEAD DUPLICATE: byte-identical to `bench-gemm.bat` modulo CRLF/LF (md5 differs only from line endings); no `BLOCKED` macro exists anywhere in `bench_gemm.cu`; writes the same `bench-mxfp4.dll`+`bgx.lib`. The "blocked" experiment's distinguishing define was removed from the source at some point, leaving this a ghost. |
| `oldgen.bat` | DELETE | DEAD: forwards to narrow `main` via `dllshim_c.cu`, but `generate.cu` now defines `wmain` — it cannot link today. |
| `shim-only.bat` | DELETE | Stepping-stone: old `test-pair` closure without `mxfp4_i4.cu`. Superseded by `mk test-pair`. |
| `tiny.bat` | KEEP if used | Compiles an ad-hoc `%TMP%\tiny.cu` scratch DLL. mk deliberately compiles only fixed targets; this is the only bat that can compile an arbitrary scratch file. Personal-utility; delete if unused. |

Adjacent junk worth deleting while at it (bat-era forensics, not bats):
`build$t.dll` and `build\${t}x.exp`/`${t}x.lib` (unexpanded-variable quoting
bug from some old command), root `hello.obj`/`nul.obj`/`model_file.obj`/
`test_cpu.obj` (cl litter, see 1.5).

Bottom line: mk.bat + sani.bat (+ maybe smoke.bat, tiny.bat) survive; all 31
one-shot builders are superseded; 4 are outright dead
(`bench-gemm-blocked`, `oldgen`, `shim-only`, and `bench-gemm`'s collision).

---

## 3. Flags review

`BASE_FLAGS = -arch=sm_89 -O3 --use_fast_math -std=c++20 -Iinclude -DNDEBUG -lineinfo`

### 3.1 --use_fast_math — LOW RISK, TWO PARITY-CRITICAL SITES, ONE CONVENTION FIX

Full transcendental inventory of src/ + include/:

- **Explicit intrinsics everywhere (project convention):** `__expf` (11 sites:
  attention.cu:7 softmax, ops.cu:5/7 silu, prefill.cu:134/180/207/210,
  generate.cu:32, nll.cu:29, qwen_kernels.cu:5/7/9/11), `__logf`
  (generate.cu:44, nll.cu:41), `__powf/__cosf/__sinf` RoPE (ops.cu:9,
  prefill.cu:78), `rsqrtf` (all norms). For these `--use_fast_math` changes
  NOTHING — they are already the fast versions.
- **Plain device transcendentals subject to the rewrite — exactly TWO sites,
  both the DeltaNet decay path:**
  - `src/deltanet.cu:9` — `const float decay=expf(g[head]);` (decode kernel;
    feeds the layer-0 parity result, cosine 0.9999998)
  - `src/prefill.cu:244` — `const float decay=expf(a[t*32+head]);`
    (deltanet prefill kernel; feeds dump-multistep / i4 parity dumps)

  Under the flag both become `__expf`-class (~2^-21 max rel error). Note the
  historical parity numbers were themselves CAPTURED with fast math (the dump
  bats had `--use_fast_math`), so the numbers are self-consistent — but the
  flag is a hidden input to the ongoing full-attention parity chase.
  **Recommendation:** write `__expf` explicitly at both sites. Zero perf cost
  (the flag already lowers them to `__expf`), makes parity independent of
  the global flag, and matches the codebase convention.
- **Remaining fast-math effects:** `-ftz=true` (denormal flush — softmax/rms
  carry eps 1e-6 margins; low risk), `-prec-div=false` (approx reciprocal on
  the silu/sigmoid divisions in ops.cu:7, qwen_kernels.cu:7/9/11,
  attention's `1.f/den`, prefill — ~2^-22 rel; absorbed by existing
  tolerances: test-ops 2e-6, test-deltanet 3e-3 rel), `-prec-sqrt=false`
  (no device `sqrtf` in engine TUs — verified), `-fmad=true` (nvcc default
  anyway). Host code is untouched (`--use_fast_math` is device-side only;
  cl host passes run without `/fp:fast`), so the double-precision host
  reference loops in test/dump tools stay precise.
- **CPU parity path unaffected:** `insignia_cpu.hpp` compiles only via the
  cl path (no `/fp:fast`); its Remez-fitted exp is deliberately
  `__expf`-class (comment at :181) while its double reference sections stay
  precise. Pre-existing cross-path divergence to keep an eye on (NOT a fast
  math issue): CPU RoPE tables use double `std::pow` (insignia_cpu.hpp:740)
  while GPU uses float `__powf` (ops.cu:9, prefill.cu:78) — angle error
  grows with position.
- **Per-file flag overrides: DON'T.** mk.py's cache design assumes ONE flag
  set for the whole shared OBJDIR (the stamp IS the global flag set).
  Per-TU flags would need per-TU cache keys or a second OBJDIR — machinery
  for a problem that the explicit-intrinsic swap above solves for free.
  (AGENTS.md: measurement-backed, not superstition.)
- Related delta: `mk test-deltanet` builds WITH fast math while
  `test-deltanet.bat` did NOT. Test tolerance (3e-3 rel) absorbs expf
  differences; just be aware the mk-built binary is not bit-identical to the
  bat-era one.

### 3.2 --extended-lambda — NOT NEEDED, and here's why

All 7 gemm.cu lambdas (`stage` :109, `prefetch` :227/:387/:479, `dequant`
:244/:404/:496) are defined INSIDE `__global__` kernels
(`mxfp4_gemm_mlx_kernel`, `_v2_`, `_v21_`, `_mlx_i4_`, `_v21_i4_`).
Lambdas defined in device code are implicitly `__device__` closures and nvcc
compiles them with no flag. `--extended-lambda` is only required when a
lambda is declared in HOST code and used in device code (annotated
`__device__ [...]`, passed as a kernel argument, thrust/cub callables). The
repo has zero thrust/cub includes and zero `__device__ [...]` lambdas —
verified by grep. Empirical confirmation: every bat has compiled gemm.cu
without the flag since the beginning. Add the flag only if host-side device
lambda tables ever appear.

### 3.3 MSVC-specific flags worth adding (batch them — one flags change wipes the obj cache once, by design)

- **cl path: `/utf-8` — effectively REQUIRED.** 2253 non-ASCII bytes across
  the cl-compiled TUs (test_cpu.cpp: 563, insignia_cpu.hpp: 1690, UTF-8).
  Without it cl parses them in the ANSI codepage — C4819 warnings now,
  latent mis-tokenization later. (`model_file.cpp`/`test_model.cpp`/
  model.hpp/storage.hpp are pure ASCII; the .cu TUs' em-dashes go through
  nvcc, which handles UTF-8 fine.)
- **cl path: `/Fo:build\obj\host\`** — stops the repo-root obj litter
  (1.5). Never point it at `build\obj` itself (basename collision with
  nvcc's `model_file.obj`, §1.3).
- **cl path: do NOT add `/fp:fast`** — cpu.hpp's double-precision reference
  code must stay precise (its hand-fitted expf is already __expf-class by
  construction).
- Optional cl: `/Zc:__cplusplus` (cosmetic), `/Gy /Gw` (COMDAT/GDW — faster
  link, marginal), `/MP` (pointless at 2 TUs). `/arch:AVX2` already present
  and correct for the 5600X.
- Optional nvcc host side: `-Xcompiler /Zc:__cplusplus`; nothing else is
  load-bearing. `-lineinfo` is already there (free at runtime, Nsight-ready)
  — keep. `--threads 0` = one sub-compiler per core (documented nvcc
  semantics for 0 since CUDA 11.2) — correct on the 12-thread 5600X.

---

## 4. ENGINE27 closure — exact TARGETS dict diff

What 27B needs vs what ENGINE27 has: CORE is right (ModelFile + TieredStorage
read the `qwen38-27b-fp8.insignia-index`); fp8/ops/attention/deltanet/
qwen_kernels are right (test-fp8 links fp8.cu alone — it is self-contained,
so no gemm.cu/mxfp4 needed for the FP8 GEMV path; decode.cu/prefill.cu/
qwen35.cu are 9B-layout-specific and correctly absent). The gap is
**streaming.cu** — the 555-line host-only NVMe→pinned-RAM→LayerFeeder TU
that the 27B streaming execution model is built around, currently linked by
nothing (§1.7). Exact diff:

```python
 ENGINE27 = CORE + ["src/fp8.cu", "src/ops.cu", "src/attention.cu",
                    "src/deltanet.cu", "src/qwen_kernels.cu",
-                   ]
+                   "src/streaming.cu"]          # host NVMe/pinned-ring/feeder TU (w3); .cu only to link cudart
```

```python
-    "test-cpu":        exe(["src/model_file.cpp", "src/test_cpu.cpp"], future=True),
-    "io-bench":        dll(CORE + ["src/io_bench.cu"], future=True),
-    "generate27":      dll(ENGINE27 + ["src/generate27.cu"], future=True),
-    "nll27":           dll(ENGINE27 + ["src/nll27.cu"], future=True),
-    "dump-layers27":   dll(ENGINE27 + ["src/dump_layers27.cu"], future=True),
+    "test-cpu":        exe(["src/test_cpu.cpp"]),                      # src/test_cpu.cpp EXISTS; cpu.hpp is
+                                                                          # header-only & self-contained, so
+                                                                          # model_file.cpp is droppable (keep it
+                                                                          # only if test-cpu will grow a loader leg)
+    "io-bench":        dll(["src/io_bench.cu"], shim=SHIM_C),         # src/io_bench.cu EXISTS; SHIM_C MANDATORY
+                                                                          # (io_bench defines narrow main :309;
+                                                                          # current default SHIM_W -> unresolved
+                                                                          # wmain); CORE droppable (no repo includes)
+    "generate27":      dll(ENGINE27 + ["src/generate27.cu"], run=[IDX27], future=True),  # stays future: TU not landed
+    "nll27":           dll(ENGINE27 + ["src/nll27.cu"], future=True),                    # stays future; run via nll_compare.py
+    "dump-layers27":   dll(ENGINE27 + ["src/dump_layers27.cu"], future=True),            # stays future
```

Notes: generate27/nll27/dump_layers27 missing sources are EXPECTED (mark kept
`future=True` so mk reports them as not-yet-landed). When those TUs land the
closure may grow again (e.g. a 27B prefill or a gemm.cu reuse) — the
"ENGINE27 grows as TUs land" comment stays true. `run=[IDX27]` finally uses
the currently-dead `IDX27` constant. Optionally add `run=["all"]` to
test-cpu (its main defaults to mode "all" anyway, so no run line is also
fine). Two compile_changed fixes (§1.2) should land before any of this so
the first flags-touching edit doesn't crash.

---

## 5. Incremental soundness: nvcc TUs vs cl TUs

- The two cl targets bypass `compile_changed`/`link` entirely: always full
  compile+link, no cache, no dep graph, no flag stamp. **Cost: negligible.**
  model_file.cpp = 45 lines, test_model.cpp = 4, test_cpu.cpp = 637 +
  cpu.hpp = 994 → cl /O2 ≈1-2 s wall including process spawn and link.
  Correctness is trivially fresh (a header edit can never be missed — there
  is no cache to go stale). "test-cpu rebuilt every time" is a non-problem
  at today's sizes; it would only matter if cpu.hpp grew 100x.
- **No link ever mixes nvcc objs with cl objs** (cl targets compile all
  their own TUs), so there is no ABI/runtime-mixing hazard today. Keep it
  that way: nvcc's `model_file.obj` and cl's differ in flags (AVX2, /EHsc)
  and cudart linkage — do not "optimize" test-cpu into reusing nvcc's obj.
  model_file.cpp being compiled twice (once by nvcc into build\obj for CUDA
  targets, once fresh by cl for host tests) is intentional isolation and
  cheap.
- Real warts, restated: cl objs litter the repo root (fix `/Fo`, §1.5), cl
  depends on vcvars being active (use `build\mk.bat`, or mk.py should
  detect and message), and `--force`/`--clean`/`INSIG_PTXAS_V` are silently
  irrelevant to these targets (fine — they always rebuild).
- Multi-target invocations (`mk test-cpu test-model`) work — each rebuilt,
  nothing run; single-target `mk test-cpu` builds and runs nothing (no run
  line). Consistent with the mk arg convention (args after the target
  replace the run tail), though a stray second token intended as a target
  name would be silently eaten as a run arg — minor UX wart worth a
  `name in TARGETS` check on `argv[1:]`.

---

## Priority summary

1. **Fix compile_changed** (hoist stamp check; on wipe recompile the whole
   current target) — the wipe path crashes today (reproduced in simulation).
2. **io-bench: `shim=SHIM_C` + drop `future`** — it cannot link as written.
3. **ENGINE27 += streaming.cu; test-cpu de-future** — streaming.cu is the
   only orphaned source.
4. **cl line: add `/utf-8` and `/Fo:build\obj\host\`** — correctness hygiene
   + stop root litter.
5. **Write `__expf` explicitly at deltanet.cu:9 and prefill.cu:244** —
   removes the last hidden fast-math parity variable for free.
6. Delete the 31 superseded bats + 4 dead ones (keep mk.bat, sani.bat,
   tiny.bat if used, smoke.bat until a run-all exists); copy the
   generate-ids 12-token run tail into mk first.
