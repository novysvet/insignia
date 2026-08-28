# Build system audit (w3) — 2026-08-25

Scope: every `build/*.bat` + the sources they reference. Read-only audit; no nvcc runs,
no git changes. Method: dumped all 33 bats, extracted each TU's cross-TU calls
(`#include "insignia_*"` + called externs) and mapped them against the TU that defines
each symbol; verified against binaries/implibs on disk (timestamps prove which bat
wrote which artifact).

## 0. Symbol map (where things live)

Engine closure — every TU that includes `insignia_decode.hpp` needs ALL of these:

| symbol group | defined in |
|---|---|
| `mxfp4_gemv_v2_i4`, `mxfp4_gemv2_q8_i4`, `mxfp4_gemv_ab2_q8_i4`, `mxfp4_get_row_i4` | `src/mxfp4_i4.cu` |
| `mxfp4_gemm_mlx_i4`, `mxfp4_gemm_mlx`, `mxfp4_gemm_v2`, `mxfp4_gemm_v21`, `f32_to_bf16` | `src/gemm.cu` |
| `embed_gather_i4` **(lives in prefill.cu, NOT mxfp4_i4.cu)**, `embed_gather`, `gqa_prefill`, `conv_prefill_silu`, `deltanet_prefill`, `deltanet_params_batch`, `qk_norm_rope_batch`, `split_q_gate_batch`, `store_kv_batch`, `spec_*`, `addi_kernel_launch` | `src/prefill.cu` |
| u8 gemv family (`mxfp4_gemv_v2`, `mxfp4_gemv_ab2_q8`, `mxfp4_get_row_mlx`, `mxfp4_gemv_mlx`, quantize_*) | `src/mxfp4.cu` |
| `gqa_decode` | `src/attention.cu` |
| `deltanet_decode`, `deltanet_parameters` | `src/deltanet.cu` |
| `rmsnorm_bf16`, `gated_rmsnorm_bf16`, `causal_conv4_silu`, `store_kv`, `bf16_gemv`, `split_q_gate`, `expand_gate_heads`, `sigmoid_mul`, `concat`, `argmax_logits`, `argmax_fast` | `src/qwen_kernels.cu` |
| `rmsnorm_zero_centered`, `rmsnorm_gated_silu`, `silu_mul`, `residual_add`, `qwen35_qk_norm_rope_gate` | `src/ops.cu` |
| `Qwen35Weights` (calls `mxfp4_get_row_i4` itself!) | `src/qwen35.cu` |
| `Qwen35Decode`, `DecodeWorkspace` | `src/decode.cu` |
| `ModelFile` / `TieredStorage` | `src/model_file.cpp` / `src/storage.cu` |
| `fp8_gemv`, `fp8_gemv2`, `fp8_gemm`, `bf16_get_row` | `src/fp8.cu` (self-contained) |

ENGINE = `model_file.cpp storage.cu mxfp4.cu mxfp4_i4.cu gemm.cu prefill.cu ops.cu attention.cu deltanet.cu qwen_kernels.cu qwen35.cu decode.cu` (12 TUs).

Two traps found:
- **`qwen35.cu` itself calls `mxfp4_get_row_i4`** (`embed_dev`, line 13). So even a
  "no-decode" tool like test-qwen35.bat breaks without `mxfp4_i4.cu`.
- `decode.cu` references i4 symbols in **three** TUs (`mxfp4_i4.cu` + `gemm.cu::mxfp4_gemm_mlx_i4` +
  `prefill.cu::embed_gather_i4`). All three must be present.

Dead declaration: `bf16_gemv_rows` is declared in `include/insignia_fp8.cuh:40` but
defined nowhere and called nowhere — remove it before someone calls it and gets an
unresolved external.

## 1. Audit table

Common flags everywhere: `-arch=sm_89 -O3 --use_fast_math -std=c++20 -Iinclude` (+`-shared -Xlinker /IMPLIB:build\<x>.lib` for DLLs). "missing 3" = `mxfp4_i4.cu prefill.cu gemm.cu` (the decode.cu i4 closure).

| bat | sources (beyond engine core noted) | output | verdict |
|---|---|---|---|
| bench-gemm.bat | mxfp4.cu, gemm.cu, bench_gemm.cu, dllshim_c.cu | **build\bench-mxfp4.dll** (!) | **link-ok, OUTPUT NAME WRONG** — clobbers the real bench-mxfp4.dll; also `mxfp4.cu` is a wasted TU (bench_gemm.cu only calls `mxfp4_gemm_v2` from gemm.cu) |
| bench-gemm-blocked.bat | identical to bench-gemm.bat | same | **DUPE** — content byte-identical (only CRLF vs LF), same clobber bug; delete or make it actually build a blocked variant |
| bench-mxfp4.bat | bench_mxfp4_mlx.cu, mxfp4.cu, gemm.cu, dllshim_c.cu | build\bench-mxfp4.dll | OK (victim of the overwrite bug: run bench-gemm.bat and this dll silently becomes bench_gemm) |
| dump-attention.bat | model_file, storage, mxfp4, qwen35, qwen_kernels, ops, attention, deltanet, decode, dump_attention | dump-attention.exe | **LINK-BROKEN** — missing 3 (stale; exe predates i4 split) |
| dump-i4-chunk.bat | full 12-TU ENGINE + dump_i4_chunk + dllshim | dump-i4-chunk.dll | OK |
| dump-i4-seams.bat | full ENGINE + dump_i4_seams + dllshim | dump-i4-seams.dll | OK (IMPLIB `dli4x.lib` **collides** with dump-layers-i4.bat's) |
| dump-layer0.bat | same set as dump-attention + dump_layer0 | dump-layer0.exe | **LINK-BROKEN** — missing 3 |
| dump-layer3.bat | ditto + dump_layer3 | dump-layer3.exe | **LINK-BROKEN** — missing 3 |
| dump-layers-i4.bat | full ENGINE + dump_layers + dllshim | dump-layers-i4.dll | OK (dli4x.lib collision above) |
| dump-layers.bat | full ENGINE + dump_layers (exe) | dump-layers.exe | OK (this is the git-modified fixed version) |
| dump-multistep.bat | full ENGINE + dump_multistep + dllshim | dump-multistep.dll | OK |
| dump-pf.bat | full ENGINE + dump_pf + dllshim | dump-pf.dll | OK |
| generate-ids.bat | decode-closure minus 3 + generate_ids | generate-ids.exe | **LINK-BROKEN** — missing 3 |
| generate.bat | full ENGINE + generate.cu + dllshim, runs `python tools\rundll.py build\generate.dll %*` | generate.dll | OK |
| mk.bat | `cl /LD /O2 %TMP%\hello.c` | %TMP%\hello.dll | **leftover Smart App Control probe** — replace with the mk.py driver (see §4) |
| nll.bat | full ENGINE + nll.cu + dllshim | nll.dll | OK — note: synthesis.md item "nll.bat omits mxfp4_i4.cu" is **already fixed** in the working tree (uncommitted) |
| oldgen.bat | ENGINE minus mxfp4_i4 + generate.cu + **dllshim_c** | generate-old.dll | **LINK-BROKEN twice**: missing mxfp4_i4.cu, and dllshim_c forwards `main` while generate.cu defines `wmain` → unresolved `main`. (No `ogenx.lib` on disk = never linked.) |
| sani.bat | compute-sanitizer memcheck wrapper | — | OK; with the DLL pattern use `sani.bat python tools\rundll.py build\X.dll args` |
| shim-only.bat | ENGINE minus mxfp4_i4 + test_pair + dllshim | test-pair.dll | **LINK-BROKEN** (missing mxfp4_i4.cu) + redundant vs test-pair.bat + IMPLIB `tpx.lib` collides with test-pair.bat |
| smoke.bat | 3 stages | 3 exes | stage 1 (smoke.cu) OK; stage 2 (mxfp4+test_mxfp4) OK; **stage 3 (mxfp4+bench_mxfp4_mlx) LINK-BROKEN — needs gemm.cu** (`f32_to_bf16`, `mxfp4_gemm_v2`, `mxfp4_gemm_v21`). Also stray `if errorlevel` right after vcvars (cosmetic) |
| test-argmax.bat | qwen_kernels, test_argmax, dllshim | test-argmax.dll | OK |
| test-attention.bat | attention, test_attention | test-attention.exe | OK |
| test-checkpoint.bat | model_file, storage, mxfp4, test_checkpoint | test-checkpoint.exe | OK |
| test-deltanet.bat | deltanet, test_deltanet | test-deltanet.exe | OK |
| test-fp8.bat | test_fp8, fp8, dllshim | test-fp8.dll | OK — self-contained: test_fp8.cu defines its own static `f32_to_bf16_bits`; fp8.cu needs nothing else. **Only bat using fp8.cu today** |
| test-full-model.bat | decode-closure minus 3 + test_full_model | exe | **LINK-BROKEN** — missing 3 |
| test-generate.bat | ditto | exe | **LINK-BROKEN** — missing 3 |
| test-i4.bat | test_i4, mxfp4_i4, gemm, prefill, dllshim | test-i4.dll | OK but **prefill.cu is a wasted TU** (test_i4.cu calls no prefill symbols) |
| test-layer.bat | ditto pattern | exe | **LINK-BROKEN** — missing 3 |
| test-model.bat | cl: model_file.cpp + test_model.cpp | test-model.exe | OK (CPU-only, no nvcc) |
| test-mtp.bat | ditto pattern | exe | **LINK-BROKEN** — missing 3 |
| test-ops.bat | ops, test_ops | test-ops.exe | OK |
| test-pair-chain.bat | ENGINE minus mxfp4_i4 + test_pair_chain + dllshim | test-pair-chain.dll | **LINK-BROKEN** — missing mxfp4_i4.cu |
| test-pair.bat | full ENGINE + test_pair + dllshim + rundll | test-pair.dll | OK |
| test-prefill.bat | full ENGINE + test_prefill + dllshim + rundll | test-prefill.dll | OK |
| test-qwen35.bat | model_file, storage, mxfp4, qwen35, test_qwen35 | test-qwen35.exe | **LINK-BROKEN** — qwen35.cu needs `mxfp4_get_row_i4` (mxfp4_i4.cu missing) |
| tiny.bat | `nvcc -shared %TMP%\tiny.cu` | %TMP%\tiny.dll | leftover SAC probe; ignore |

**11 bats are link-broken** (all stale: their exes predate the i4 split of Aug 24 ~22:00):
dump-attention, dump-layer0, dump-layer3, generate-ids, oldgen, shim-only, smoke(stage3),
test-full-model, test-generate, test-layer, test-mtp, test-pair-chain, test-qwen35.

**fp8.cu users going forward**: today only test-fp8.bat. The 27B targets
(generate27, nll27, dump-layers27, io-bench) will need `fp8.cu` in their closure
(ENGINE27 = CORE + fp8 + ops + attention + deltanet + qwen_kernels + future 27B TUs).
fp8.cu depends on nothing (gemm-free), so it drops into any link cleanly.

## 2. Known issues (synthesis backlog item 7)

1. **bench-mxfp4.dll overwrite bug — found.** `bench-gemm.bat` and
   `bench-gemm-blocked.bat` both end `-o build\bench-mxfp4.dll` (and share
   `/IMPLIB:build\bgx.lib`). They were clearly forked from bench-mxfp4.bat and the
   `-o` was never renamed. Proof on disk: `bgx.lib` mtime Aug 25 00:06 (bench-gemm.bat
   ran then) while `bmfpx.lib` (real bench-mxfp4 build) is Aug 24 22:04 — the current
   `bench-mxfp4.dll` (00:32) contains the **bench_gemm** tool. Correct outputs:
   `build\bench-gemm.dll` / `build\bench-gemm-blocked.dll`. Also drop `mxfp4.cu` from
   bench-gemm.bat (bench_gemm.cu uses only gemm.cu).
2. **bench-gemm-blocked.bat is a dupe**: identical command list; only difference in the
   file is CRLF vs LF line endings. Either delete it or make it the "blocked/pipelined
   GEMM" variant it was presumably meant to be.
3. **IMPLIB name collisions** (benign today because rundll.py never uses the .lib, but
   a trap): `dli4x.lib` shared by dump-i4-seams.bat + dump-layers-i4.bat; `tpx.lib`
   shared by test-pair.bat + shim-only.bat; `bgx.lib` shared by both bench-gemm bats.
4. **Junk artifacts from shell-expansion bugs**: `build/${t}x.exp`, `build/${t}x.lib`
   (a *real* MSVC implib written to a literal `${t}x` filename by an unquoted-variable
   nvcc loop), repo-root `build$t.dll`, `hello.obj`, `nul.obj` (mk.bat's cl probe
   writing to CWD). All deletable.
5. Synthesis's "nll.bat omits mxfp4_i4.cu" is already fixed in the working tree
   (nll.bat modified, includes mxfp4_i4.cu + gemm.cu; uncommitted).

## 3. Flag recommendations

- **`-lineinfo` everywhere**: line tables only — zero runtime cost, no codegen change
  (do NOT confuse with `-G`). Gives per-source-line attribution in Nsight
  Systems/Compute and compute-sanitizer stack traces. Pure win; add to the shared flag set.
- **`-Xptxas -v` behind a debug switch**: per-kernel register/smem/spill report; too
  noisy for every build. In mk.py: env `INSIG_PTXAS_V=1` or `--ptxas-v`.
- **`--threads 0`** (verified against CUDA docs + forums): CUDA ≥ 11.3; `0` = one
  nvcc sub-compiler per CPU core; **only effective when multiple sources are passed to
  a single nvcc invocation** (e.g. `nvcc -c a.cu b.cu c.cu -odir build\obj --threads 0`).
  With the current one-command-per-bat pattern it does nothing. The 5600X has 12
  threads and the ENGINE is 12 TUs — batching compiles turns a full rebuild from the
  sum of TU times (~30–90 s) into roughly the slowest TU (~10–20 s).
- **`-DNDEBUG`**: strips host `assert()`; the engine's real checks are exceptions and
  are unaffected. Free, zero-risk on this codebase.
- **Per-TU object cache** (`build/obj/*.obj` via `nvcc -c`, then a single link):
  `model_file.cpp + storage.cu + ops.cu + qwen35.cu ...` are stable — only the touched
  tool TU recompiles, so warm builds drop to ~2–5 s. Design below keeps a flags-hash
  guard so toggling `-Xptxas -v` / experiments wipes the cache instead of producing
  mismatched objects.
- Do **not** add: `-G` (slow debug), `-dlink` (no relocatable device code anywhere).

## 4. tools/mk.py + mk.bat (drop-in replacement driver)

`mk.bat <target> [args...]` keeps the bat convention; args are forwarded to the tool
after building (DLLs via `python tools\rundll.py`; extra args *replace* the default
run line — mirrors the old bats' baked-in argv). mtime-based rebuild with recursive
`#include "..."` dependency scanning; `--threads 0` batch compile; DLL/shim/rundll
pattern preserved exactly. New targets included: test-fp8 (ported), test-cpu, io-bench,
generate27, nll27, dump-layers27 (marked FUTURE — mk.py reports missing sources
instead of failing other targets).

### build/mk.bat

```bat
@echo off
rem Insignia build driver: mk.bat <target> [args...]   (see: python tools\mk.py --list)
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
python tools\mk.py %*
```

### tools/mk.py

```python
#!/usr/bin/env python3
"""Insignia build driver: mtime-incremental, per-TU object cache in build/obj.

Usage (from repo root, or via build/mk.bat which vcvars first):
  python tools/mk.py --list                 list targets
  python tools/mk.py <target> [args...]     build; if the target has a run line,
                                            run it (extra args REPLACE the default tail;
                                            DLLs are run through tools/rundll.py)
  python tools/mk.py <t1> <t2> ...          build several targets (no run)
Options: --force (recompile all TUs)  --clean (wipe build/obj)  --ptxas-v
Env: INSIG_PTXAS_V=1, INSIG_THREADS=N (default 0 = one nvcc per CPU core).
"""
import os, re, shutil, subprocess, sys

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NVCC   = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
CCBIN  = r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64"
OBJDIR = os.path.join(ROOT, "build", "obj")
RUNDLL = os.path.join("tools", "rundll.py")
IDX    = r"build\qwen35.insignia-index"

CORE   = ["src/model_file.cpp", "src/storage.cu"]
ENGINE = CORE + ["src/mxfp4.cu", "src/mxfp4_i4.cu", "src/gemm.cu", "src/prefill.cu",
                 "src/ops.cu", "src/attention.cu", "src/deltanet.cu",
                 "src/qwen_kernels.cu", "src/qwen35.cu", "src/decode.cu"]
# ENGINE27 = FP8 (27B) closure once the 27B decode TUs exist; fp8.cu links clean alone.
ENGINE27 = CORE + ["src/fp8.cu", "src/ops.cu", "src/attention.cu",
                   "src/deltanet.cu", "src/qwen_kernels.cu"]
SHIM_W = "src/dllshim.cu"    # forwards wmain (wchar argv) — all wmain tools
SHIM_C = "src/dllshim_c.cu"  # forwards main (narrow argv) — bench_* tools

def dll(t, srcs, shim=SHIM_W, run=None, future=False):
    return dict(kind="dll", srcs=srcs + [shim], run=run, future=future)
def exe(t, srcs, run=None, future=False):
    return dict(kind="exe", srcs=srcs, run=run, future=future)

TARGETS = {
    # --- engine tools (DLL + rundll pattern) ---
    "generate":        dll("generate", ENGINE + ["src/generate.cu"], run=[IDX, "Hello!"]),
    "nll":             dll("nll", ENGINE + ["src/nll.cu"]),  # run via tools/nll_compare.py
    "test-pair":       dll("test-pair", ENGINE + ["src/test_pair.cu"], run=[IDX]),
    "test-pair-chain": dll("test-pair-chain", ENGINE + ["src/test_pair_chain.cu"], run=[IDX]),
    "test-prefill":    dll("test-prefill", ENGINE + ["src/test_prefill.cu"], run=[IDX]),
    "test-i4":         dll("test-i4", ["src/test_i4.cu", "src/mxfp4_i4.cu", "src/gemm.cu"]),
    "test-fp8":        dll("test-fp8", ["src/test_fp8.cu", "src/fp8.cu"]),
    "test-argmax":     dll("test-argmax", ["src/test_argmax.cu", "src/qwen_kernels.cu"]),
    "dump-layers-i4":  dll("dump-layers-i4", ENGINE + ["src/dump_layers.cu"]),
    "dump-i4-chunk":   dll("dump-i4-chunk", ENGINE + ["src/dump_i4_chunk.cu"]),
    "dump-i4-seams":   dll("dump-i4-seams", ENGINE + ["src/dump_i4_seams.cu"]),
    "dump-multistep":  dll("dump-multistep", ENGINE + ["src/dump_multistep.cu"]),
    "dump-pf":         dll("dump-pf", ENGINE + ["src/dump_pf.cu"]),
    # --- exe tools (auto-run lines mirror the old bats) ---
    "dump-layers":     exe("dump-layers", ENGINE + ["src/dump_layers.cu"], run=[IDX, r"build\layers-native.f32"]),
    "dump-attention":  exe("dump-attention", ENGINE + ["src/dump_attention.cu"], run=[IDX, r"build\attention-native.f32"]),
    "dump-layer0":     exe("dump-layer0", ENGINE + ["src/dump_layer0.cu"], run=[IDX, r"build\layer0-native.f32"]),
    "dump-layer3":     exe("dump-layer3", ENGINE + ["src/dump_layer3.cu"], run=[IDX, r"build\layer3-native.f32"]),
    "generate-ids":    exe("generate-ids", ENGINE + ["src/generate_ids.cu"], run=[IDX]),
    "test-full-model": exe("test-full-model", ENGINE + ["src/test_full_model.cu"], run=[IDX]),
    "test-generate":   exe("test-generate", ENGINE + ["src/test_generate.cu"], run=[IDX]),
    "test-layer":      exe("test-layer", ENGINE + ["src/test_layer.cu"], run=[IDX]),
    "test-mtp":        exe("test-mtp", ENGINE + ["src/test_mtp.cu"], run=[IDX]),
    "test-qwen35":     exe("test-qwen35", CORE + ["src/mxfp4.cu", "src/mxfp4_i4.cu", "src/qwen35.cu", "src/test_qwen35.cu"], run=[IDX]),
    "test-checkpoint": exe("test-checkpoint", CORE + ["src/mxfp4.cu", "src/test_checkpoint.cu"], run=[IDX]),
    "test-attention":  exe("test-attention", ["src/attention.cu", "src/test_attention.cu"]),
    "test-deltanet":   exe("test-deltanet", ["src/deltanet.cu", "src/test_deltanet.cu"]),
    "test-ops":        exe("test-ops", ["src/ops.cu", "src/test_ops.cu"]),
    "test-model":      exe("test-model", ["src/model_file.cpp", "src/test_model.cpp"], run=[IDX]),  # cl, see below
    "smoke":           exe("smoke", ["src/smoke.cu"]),
    "test-mxfp4":      exe("test-mxfp4", ["src/mxfp4.cu", "src/test_mxfp4.cu"]),
    "bench-mxfp4-mlx": exe("bench-mxfp4-mlx", ["src/mxfp4.cu", "src/gemm.cu", "src/bench_mxfp4_mlx.cu"]),
    # --- benches (DLL, narrow-main shim) — output names FIXED, no more clobbering ---
    "bench-mxfp4":     dll("bench-mxfp4", ["src/bench_mxfp4_mlx.cu", "src/mxfp4.cu", "src/gemm.cu"], shim=SHIM_C),
    "bench-gemm":      dll("bench-gemm", ["src/gemm.cu", "src/bench_gemm.cu"], shim=SHIM_C),
    # --- FUTURE (27B FP8 + CPU + IO): reported as missing until sources land ---
    "test-cpu":        exe("test-cpu", ["src/model_file.cpp", "src/test_cpu.cpp"], future=True),
    "io-bench":        dll("io-bench", CORE + ["src/io_bench.cu"], future=True),
    "generate27":      dll("generate27", ENGINE27 + ["src/generate27.cu"], future=True),
    "nll27":           dll("nll27", ENGINE27 + ["src/nll27.cu"], future=True),
    "dump-layers27":   dll("dump-layers27", ENGINE27 + ["src/dump_layers27.cu"], future=True),
}

BASE_FLAGS = ["-ccbin", CCBIN, "-arch=sm_89", "-O3", "--use_fast_math", "-std=c++20",
              "-Iinclude", "-DNDEBUG", "-lineinfo"]

def dep_scan(src, seen=None):
    """Recursive #include "..." closure (repo-local files only)."""
    seen = seen if seen is not None else set()
    if src in seen: return seen
    seen.add(src)
    try: text = open(os.path.join(ROOT, src), encoding="utf-8", errors="replace").read()
    except OSError: return seen
    for m in re.finditer(r'^\s*#\s*include\s+"([^"]+)"', text, re.M):
        inc = m.group(1)
        for base in (os.path.dirname(src), "src", "include"):
            cand = os.path.normpath(os.path.join(base, inc))
            if os.path.isfile(os.path.join(ROOT, cand)):
                dep_scan(cand, seen); break
    return seen

def newest(path_list):
    t = 0.0
    for p in path_list:
        f = os.path.join(ROOT, p)
        if os.path.isfile(f): t = max(t, os.path.getmtime(f))
    return t

def obj_for(src): return os.path.join(OBJDIR, os.path.splitext(os.path.basename(src))[0] + ".obj")

def build_flags():
    fl = list(BASE_FLAGS)
    if os.environ.get("INSIG_PTXAS_V", "") == "1" or "--ptxas-v" in sys.argv:
        fl += ["-Xptxas", "-v"]
    return fl

def compile_changed(srcs, flags, force):
    os.makedirs(OBJDIR, exist_ok=True)
    changed = []
    for s in srcs:
        if not os.path.isfile(os.path.join(ROOT, s)):
            raise SystemExit(f"mk: missing source {s}")
        obj = obj_for(s)
        deps = dep_scan(s)
        if force or not os.path.isfile(obj) or newest(deps) >= os.path.getmtime(obj):
            changed.append((s, obj))
    if not changed: return 0
    # flags-hash guard: wipe the cache when the flag set changes (avoids stale mismatches)
    stamp = os.path.join(OBJDIR, ".flags")
    sig = " ".join(flags)
    if os.path.isfile(stamp) and open(stamp).read() != sig:
        shutil.rmtree(OBJDIR); os.makedirs(OBJDIR); changed = [(s, obj_for(s)) for s, _ in changed]
    open(stamp, "w").write(sig)
    thr = os.environ.get("INSIG_THREADS", "0")
    # one nvcc invocation, one sub-compiler per CPU core (only helps with >1 TU)
    cmd = [NVCC, *flags, "-c", *[s for s, _ in changed], "-odir", OBJDIR, "--threads", thr]
    print("mk: compile", " ".join(s for s, _ in changed))
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc: raise SystemExit(rc)
    return len(changed)

def link(name, spec, flags):
    out = os.path.join("build", name + (".dll" if spec["kind"] == "dll" else ".exe"))
    objs = [obj_for(s) for s in spec["srcs"]]
    newest_obj = max(os.path.getmtime(os.path.join(ROOT, o)) for o in objs)
    if os.path.isfile(os.path.join(ROOT, out)) and os.path.getmtime(os.path.join(ROOT, out)) > newest_obj:
        return out  # up to date
    cmd = [NVCC, *flags, *objs]
    if spec["kind"] == "dll":
        cmd += ["-shared", "-Xlinker", f"/IMPLIB:{OBJDIR}\\{name}x.lib"]
    cmd += ["-o", out]
    print("mk: link  ", out)
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc: raise SystemExit(rc)
    return out

def run_tool(name, spec, out, args):
    run = args if args else spec.get("run")
    if not run: return 0
    if spec["kind"] == "dll":
        cmd = [sys.executable, RUNDLL, out, *run]
    else:
        cmd = [os.path.join(ROOT, out), *run]
    return subprocess.call(cmd, cwd=ROOT)

def main():
    argv = [a for a in sys.argv[1:] if a not in ("--ptxas-v",)]
    force = "--force" in argv;  argv = [a for a in argv if a != "--force"]
    if "--clean" in argv:
        shutil.rmtree(OBJDIR, ignore_errors=True); print("mk: cleaned", OBJDIR)
        argv = [a for a in argv if a != "--clean"]
    if "--list" in argv or not argv:
        for t, s in TARGETS.items():
            tag = " (FUTURE)" if s["future"] else ""
            print(f"{t:18s} {s['kind']}{tag}")
        return 0
    flags = build_flags()
    name, rest = argv[0], argv[1:]
    if name not in TARGETS: raise SystemExit(f"mk: unknown target {name} (--list)")
    spec = TARGETS[name]
    missing = [s for s in spec["srcs"] if not os.path.isfile(os.path.join(ROOT, s))]
    if missing:
        raise SystemExit(f"mk: {name}: missing sources {missing}" + (" (FUTURE target)" if spec["future"] else ""))
    if name == "test-model" or name == "test-cpu":   # host-only cl path (fast, no cache)
        cmd = ["cl", "/nologo", "/std:c++20", "/O2", "/EHsc", "/DNDEBUG", "/Iinclude",
               *spec["srcs"], f"/Fe:build\\{name}.exe"]
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc: raise SystemExit(rc)
        out = os.path.join("build", name + ".exe")
    else:
        compile_changed(spec["srcs"], flags, force)
        out = link(name, spec, flags)
    if rest:                            # explicit args: run with them (replace default tail)
        return run_tool(name, spec, out, rest)
    if len(argv) == 1:                  # single target, no args: use the default run line
        return run_tool(name, spec, out, [])
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

Notes on the design:
- Warm build of any ENGINE target after `model_file/storage/ops/...` are cached:
  1 TU compile + link ≈ 2–5 s (vs 30–90 s full recompile in every current bat).
- Cold full build: one nvcc call, `--threads 0` → 12 parallel sub-compilers on the
  5600X's 12 threads.
- IMPLIBs go to `build/obj/`, ending the `build/*x.lib` litter and all name collisions.
- `bench-gemm` gets its own output name; the clobber bug becomes impossible.
- Multi-target invocation (`mk.bat generate nll dump-pf`) builds without running.
- Everything still goes through vcvars + nvcc -ccbin sm_89 — the only known-good path.

## 5. DLL pattern (must keep working)

- `src/dllshim.cu` exports `dll_run(argc, wchar_t**)` → forwards to the tool's `wmain`
  (used by every wmain tool); `src/dllshim_c.cu` exports `dll_run_c(argc, char**)` →
  forwards to `main` (bench_gemm.cu, bench_mxfp4_mlx.cu only).
- `tools/rundll.py`: `ctypes.CDLL(path)`, tries `dll_run` first, falls back to
  `dll_run_c`; retries 12 × 10 s on Windows error 4551 (Smart App Control briefly
  blocks freshly written unsigned DLLs). Reason the DLL pattern exists at all:
  Smart App Control blocks unsigned *processes*, python.exe is signed, so tools ship
  as DLLs python loads.
- Consumers (grep of tools/):
  - `tools/chat.py` → **build/generate.dll** (rundll subprocess; tokenizes, feeds ids, detokenizes)
  - `tools/nll_compare.py` → **build/nll.dll** (rundll subprocess)
  - bats that auto-run rundll: generate.bat, test-pair.bat, test-pair-chain.bat, test-prefill.bat
  - build-only DLLs run manually via rundll: bench-mxfp4, dump-i4-chunk/seams,
    dump-layers-i4, dump-multistep, dump-pf, test-argmax, test-fp8, test-i4
- mk.py preserves all of this: dll targets append the right shim TU, run lines route
  through rundll.py, and `sani.bat python tools\rundll.py build\X.dll` still works for
  sanitizer runs.

## 6. gdb* / leftover artifacts in build/

There are **no gdb*.bat** files — the leftovers are logs + exes from one debugging
afternoon (Aug 24, 15:25–16:39), all gitignored (`*.exe`, `*.obj`), referenced by
nothing in the repo:

- `gdb12.log … gdb23c.log` (16:08–16:15): a dozen short gdb (mingw) console sessions —
  crash-hunting transcripts.
- `gdbX.exe` (16:37), `gdbP32.exe` / `gdbP128.exe` (16:39): repro payload exes built to
  be run under gdb (X = crash repro; P32/P128 = 32/128-thread variants).
- `wl_O0.exe / wl_O1.exe / wl_O2.exe / wlO0.exe` (16:31–16:34): the same workload at
  different MSVC optimization levels — compiler-behavior investigation.
- `mini.exe` (16:20): minimal repro. `wmma-s3.exe` (16:22): WMMA stage-3 test.
- `gemm_test.o` (16:16): stray intermediate. `mxfp4_v2.cubin` (15:25): cuobjdump/nvdisasm
  artifact from kernel inspection.
- Plus the shell-expansion junk from §2.4 (`${t}x.*`, `build$t.dll`, `hello.obj`, `nul.obj`)
  and mk.bat/tiny.bat `%TMP%` probes (`hello.c`/`tiny.c` don't even exist in the repo).

Verdict: **safe to ignore, safe to delete** — none are on any build path.

## 7. Recommended bat actions (after mk.py lands)

- Delete: `bench-gemm-blocked.bat` (dupe), `oldgen.bat`, `shim-only.bat`, `mk.bat`
  hello-probe (replaced), `tiny.bat`.
- Fix or let mk.py supersede: all 13 link-broken bats in §1.
- Keep as-is: `sani.bat` (sanitizer wrapper).
- Remove `prefill.cu` from test-i4.bat (wasted TU) and `mxfp4.cu` from bench-gemm.bat.
- Delete dead decl `bf16_gemv_rows` from `include/insignia_fp8.cuh` (declared, never
  defined).
