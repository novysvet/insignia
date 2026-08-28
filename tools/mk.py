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
IDX27  = r"build\qwen38-27b-fp8.insignia-index"
MAN27  = r"build\manifest-v1.txt"

CORE   = ["src/model_file.cpp", "src/storage.cu"]
ENGINE = CORE + ["src/mxfp4.cu", "src/mxfp4_i4.cu", "src/gemm.cu", "src/prefill.cu",
                 "src/ops.cu", "src/attention.cu", "src/deltanet.cu",
                 "src/qwen_kernels.cu", "src/qwen35.cu", "src/decode.cu"]
# ENGINE27 = FP8 (27B) closure; grows as the 27B TUs land.
ENGINE27 = CORE + ["src/fp8.cu", "src/bf16.cu", "src/gemm.cu", "src/ops.cu", "src/attention.cu",
                   "src/deltanet.cu", "src/qwen_kernels.cu", "src/prefill.cu",
                   "src/streaming.cu", "src/decode27.cu"]
SHIM_W = "src/dllshim.cu"    # forwards wmain (wchar argv) — all wmain tools
SHIM_C = "src/dllshim_c.cu"  # forwards main (narrow argv) — bench_* tools

def dll(srcs, shim=SHIM_W, run=None, future=False):
    return dict(kind="dll", srcs=srcs + [shim], run=run, future=future)
def exe(srcs, run=None, future=False):
    return dict(kind="exe", srcs=srcs, run=run, future=future)

TARGETS = {
    # --- engine tools (DLL + rundll pattern) ---
    "generate":        dll(ENGINE + ["src/generate.cu"]),
    "nll":             dll(ENGINE + ["src/nll.cu"]),  # run via tools/nll_compare.py
    "test-pair":       dll(ENGINE + ["src/test_pair.cu"], run=[IDX]),
    "test-pair-chain": dll(ENGINE + ["src/test_pair_chain.cu"], run=[IDX]),
    "test-prefill":    dll(ENGINE + ["src/test_prefill.cu"], run=[IDX]),
    "test-i4":         dll(["src/test_i4.cu", "src/mxfp4_i4.cu", "src/gemm.cu"]),
    "test-fp8":        dll(["src/test_fp8.cu", "src/fp8.cu"]),
    "test-argmax":     dll(["src/test_argmax.cu", "src/qwen_kernels.cu"]),
    "dump-layers-i4":  dll(ENGINE + ["src/dump_layers.cu"]),
    "dump-i4-chunk":   dll(ENGINE + ["src/dump_i4_chunk.cu"]),
    "dump-i4-seams":   dll(ENGINE + ["src/dump_i4_seams.cu"]),
    "dump-multistep":  dll(ENGINE + ["src/dump_multistep.cu"]),
    "dump-pf":         dll(ENGINE + ["src/dump_pf.cu"]),
    # --- exe tools (auto-run lines mirror the old bats) ---
    "dump-layers":     exe(ENGINE + ["src/dump_layers.cu"], run=[IDX, r"build\layers-native.f32"]),
    "dump-attention":  exe(ENGINE + ["src/dump_attention.cu"], run=[IDX, r"build\attention-native.f32"]),
    "dump-layer0":     exe(ENGINE + ["src/dump_layer0.cu"], run=[IDX, r"build\layer0-native.f32"]),
    "dump-layer3":     exe(ENGINE + ["src/dump_layer3.cu"], run=[IDX, r"build\layer3-native.f32"]),
    "generate-ids":    exe(ENGINE + ["src/generate_ids.cu"], run=[IDX]),
    "test-full-model": exe(ENGINE + ["src/test_full_model.cu"], run=[IDX]),
    "test-generate":   exe(ENGINE + ["src/test_generate.cu"], run=[IDX]),
    "test-layer":      exe(ENGINE + ["src/test_layer.cu"], run=[IDX]),
    "test-mtp":        exe(ENGINE + ["src/test_mtp.cu"], run=[IDX]),
    "test-qwen35":     exe(CORE + ["src/mxfp4.cu", "src/mxfp4_i4.cu", "src/qwen35.cu", "src/test_qwen35.cu"], run=[IDX]),
    "test-checkpoint": exe(CORE + ["src/mxfp4.cu", "src/test_checkpoint.cu"], run=[IDX]),
    "test-attention":  exe(["src/attention.cu", "src/test_attention.cu"]),
    "test-deltanet":   exe(["src/deltanet.cu", "src/test_deltanet.cu"]),
    "test-ops":        exe(["src/ops.cu", "src/test_ops.cu"]),
    "test-model":      exe(["src/model_file.cpp", "src/test_model.cpp"], run=[IDX]),  # host-only cl path
    "smoke":           exe(["src/smoke.cu"]),
    "test-mxfp4":      exe(["src/mxfp4.cu", "src/test_mxfp4.cu"]),
    "bench-mxfp4-mlx": exe(["src/mxfp4.cu", "src/gemm.cu", "src/bench_mxfp4_mlx.cu"]),
    # --- benches (DLL, narrow-main shim) — output names FIXED, no more clobbering ---
    "bench-mxfp4":     dll(["src/bench_mxfp4_mlx.cu", "src/mxfp4.cu", "src/gemm.cu"], shim=SHIM_C),
    "bench-gemm":      dll(["src/gemm.cu", "src/bench_gemm.cu"], shim=SHIM_C),
    # --- landed since the w3 plan: CPU tier + io bench ---
    "test-cpu":        exe(["src/test_cpu.cpp"], future=False),
    "io-bench":        exe(["src/io_bench.cu"], future=False),  # host-only pure Win32 (narrow main, cl path)
    "generate27":      dll(ENGINE27 + ["src/generate27.cu"]),
    "test-uva":        dll(["src/test_uva.cu"], shim=SHIM_C),
    "test27-layer0":  dll(ENGINE27 + ["src/test27_layer0.cu"], shim=SHIM_C),
    "test-gemm-t1":   dll(["src/test_gemm_t1.cu", "src/fp8.cu", "src/gemm.cu"], shim=SHIM_C),
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
    # flags-hash guard: wipe the cache when the flag set changes (avoids stale mismatches).
    # Must run BEFORE the early-out: a clean tree with new flags still needs a full rebuild,
    # and a wipe must rebuild ALL TUs (rebuilding only the stale subset orphans the rest).
    stamp = os.path.join(OBJDIR, ".flags")
    sig = " ".join(flags)
    if os.path.isfile(stamp) and open(stamp).read() != sig:
        shutil.rmtree(OBJDIR); os.makedirs(OBJDIR)
        changed = [(s, obj_for(s)) for s in srcs]
    open(stamp, "w").write(sig)
    if not changed: return 0
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
    if name in ("test-model", "test-cpu", "io-bench"):  # host-only cl path (fast, no cache)
        os.makedirs(os.path.join(OBJDIR, "host"), exist_ok=True)
        cmd = ["cl", "/nologo", "/TP", "/std:c++20", "/O2", "/EHsc", "/DNDEBUG", "/arch:AVX2",
               "/utf-8", "/Iinclude", *spec["srcs"], f"/Fe:build\\{name}.exe",
               f"/Fo:build\\obj\\host\\{name}.obj"]
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
