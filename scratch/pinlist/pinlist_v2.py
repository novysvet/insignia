#!/usr/bin/env python3
"""pinlist_v2.py -- build the ExpertStager static pin list from ROUTE_TRACE files.

Runtime contract (src/glm53_generate.cu, ExpertStager::load_pin_list, lines 646-694):

  env   INSIGNIA_GLM53_PIN_LIST   path to the pin file (unset -> feature off)
  env   INSIGNIA_GLM53_PIN_HOST   per-layer host pin quota  (default 8)
  env   INSIGNIA_GLM53_PIN_DEV    per-layer VRAM-key quota  (default 2)

  file  parsed with fscanf(file, "%d %d %d", &layer, &expert, &hits) -- plain
        whitespace-separated integer triples, no header, no comments, hits is
        parsed but UNUSED by the runtime (audit info only). LF or CRLF both
        parse (isspace); we emit LF-only, one triple per line, matching the
        existing artifact /var/lib/insignia/pin-realtext.txt.
        CRITICAL ordering rule: a change in the layer field re-arms the
        per-layer quotas, so the file must contain EXACTLY ONE contiguous
        block per layer, in strictly increasing layer order. Records beyond
        PIN_HOST in a block are skipped; duplicates are skipped via the
        in-flight index but still consume quota (never emit duplicates).
  VRAM  mirror is DERIVED, not a separate file: the first PIN_DEV records of
        each layer block get their route_key inserted into
        pinned_device_keys_ (eviction-protected device slots). Because the
        quota is a single uniform integer, only per-layer uniform top-k is
        expressible; --vram-mode global is provided for the REPORT only.
  cap   pinning stops when free_windows_.size() <= 16, so at most
        (window_count - 16) records ever pin; the report models this.

Trace input: one line per (token, sparse layer):
  "token layer e0..e7 s0..e7\n"   (18 fields, ROUTE_TRACE format)
  "token layer ov e0..e7 u0..u7\n" (19 fields, EARLY_ROUTE_TRACE format; the
                                     second expert set is ignored, matching
                                     the v1 hotset analysis)

Pipeline: per-layer activation counts -> Dirichlet(alpha) smoothed
frequencies -> marginal-greedy water-filling of B host slots across layers ->
uniform k = D//L device keys -> exact-format emit + coverage report with a
split-sample out-of-sample estimate (train = first half of tokens per file,
test = second half).

python3 + numpy only. CPU only. Never touches the engine.
"""

import argparse
import heapq
import itertools
import math
import os
import re
import sys
import tempfile

import numpy as np

LAYER_FIRST, LAYER_LAST = 3, 44          # absolute sparse-layer ids (45-layer GLM)
DEFAULT_EXPERTS = 288
DEFAULT_BUDGET = 2425                    # host-tier slots at 32 GiB / 13.5625 MiB
DEFAULT_VRAM = 321
DEFAULT_ALPHA = 0.5                      # Jeffreys; inert at equal acts/layer,
                                        # regularizes thin/partial layers
RUNTIME_WINDOW_RESERVE = 16              # load_pin_list stops at <=16 free windows


# ---------------------------------------------------------------- traces ----

def parse_trace(path, n_experts):
    """Parse one ROUTE_TRACE / EARLY_ROUTE_TRACE file -> (tok, layer, E[n,8])."""
    try:
        rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except Exception as exc:
        raise SystemExit(f"error: cannot parse trace {path}: {exc}")
    if rows.size == 0:
        raise SystemExit(f"error: trace {path} is empty")
    width = rows.shape[1]
    if width == 18:
        tok, layer, exps = rows[:, 0], rows[:, 1], rows[:, 2:10]
    elif width == 19:
        tok, layer, exps = rows[:, 0], rows[:, 1], rows[:, 3:11]
    else:
        raise SystemExit(f"error: {path}: {width} fields per line, expected 18 or 19")
    for name, col in (("token", tok), ("layer", layer)):
        if not np.all(np.mod(col, 1.0) == 0.0):
            raise SystemExit(f"error: {path}: non-integer {name} column")
    if not np.all(np.mod(exps, 1.0) == 0.0):
        raise SystemExit(f"error: {path}: non-integer expert column")
    tok, layer = tok.astype(np.int64), layer.astype(np.int64)
    E = exps.astype(np.int64)
    if E.min() < 0 or E.max() >= n_experts:
        raise SystemExit(f"error: {path}: expert ids outside [0,{n_experts})")
    if layer.min() < 0 or layer.max() > 45:
        raise SystemExit(f"error: {path}: layer ids outside [0,45]")
    if not np.all((layer >= LAYER_FIRST) & (layer <= LAYER_LAST)):
        print(f"warning: {path}: some layers outside the sparse range "
              f"[{LAYER_FIRST},{LAYER_LAST}]; proceeding anyway", file=sys.stderr)
    return tok, layer, E, (18 if width == 18 else 19)


def count_matrix(layer, E, layer_ids, n_experts):
    """(L,NE) activation counts for the given rows, on the compact layer axis."""
    lidx = np.searchsorted(layer_ids, layer)
    flat = lidx[:, None] * n_experts + E
    return np.bincount(flat.ravel(), minlength=len(layer_ids) * n_experts) \
             .reshape(len(layer_ids), n_experts).astype(np.int64)


# ------------------------------------------------------------- allocation ----

def build(C, alpha, weights, budget, vram, vram_mode, n_experts):
    """Counts -> (order, B_l, device_k, D_l, q_sorted). The whole construction
    procedure, factored so the split-sample estimate can rebuild on train."""
    L = C.shape[0]
    N = C.sum(1).astype(np.float64)                       # acts per layer
    order = np.argsort(-C, axis=1, kind="stable")         # ties -> lower expert id
    C_sorted = np.take_along_axis(C, order, 1)
    denom = N[:, None] + alpha * n_experts
    q_sorted = (C_sorted + alpha) / denom                 # posterior mean freq
    w = N / max(N.sum(), 1.0) if weights == "acts" else np.full(L, 1.0 / L)
    marg = q_sorted * w[:, None]

    B_l = waterfill(marg, budget, n_experts)
    active = int((B_l > 0).sum())
    k = vram // active if active else 0                   # uniform per-layer quota
    D_l = np.minimum(k, B_l)
    return dict(order=order, B_l=B_l, k=k, D_l=D_l, q=q_sorted, marg=marg, N=N)


def waterfill(marg_sorted, total, cap):
    """Marginal-greedy water-filling: repeatedly give the next slot to the
    layer whose next-ranked smoothed marginal is highest. Equal marginals are
    served FIFO (round-robin across layers), fully deterministic."""
    L, K = marg_sorted.shape
    B_l = np.zeros(L, np.int64)
    seq = itertools.count()
    heap = []
    for l in range(L):
        heapq.heappush(heap, (-float(marg_sorted[l, 0]), next(seq), l, 0))
    assigned = 0
    while assigned < total and heap:
        negm, _, l, r = heapq.heappop(heap)
        if marg_sorted[l, r] <= 0.0:
            break  # everything left has zero predicted value (alpha == 0)
        B_l[l] += 1
        assigned += 1
        if r + 1 < min(cap, K):
            heapq.heappush(heap, (-float(marg_sorted[l, r + 1]), next(seq), l, r + 1))
    return B_l


def coverage(C_test, order, B_l):
    """Per-layer and pooled coverage of a pin allocation against test counts."""
    L, NE = C_test.shape
    hit = np.zeros(L, np.int64)
    for l in range(L):
        hit[l] = C_test[l, order[l, :B_l[l]]].sum()
    tot = C_test.sum(1)
    per_layer = np.divide(hit, tot, out=np.zeros(L), where=tot > 0)
    pooled = hit.sum() / max(C_test.sum(), 1)
    return per_layer, float(pooled)


def global_coverage(C_test, C_train, alpha, D):
    """Report-only comparison: the D globally hottest (layer,expert) records,
    ranked by smoothed count (equivalent to posterior-mean ranking when the
    acts-per-layer are equal, which holds for well-formed traces)."""
    scores = (C_train.astype(np.float64) + alpha).ravel()
    flat = np.argsort(-scores, kind="stable")[:D]
    return float(C_test.ravel()[flat].sum()) / max(C_test.sum(), 1)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ------------------------------------------------------ runtime simulator ----
# Faithful model of ExpertStager::load_pin_list() (src/glm53_generate.cu
# lines 660-690): per-record semantics, block re-arming, quota skip, dedup via
# the in-flight index, the <=16 free-window stop, and head-of-block device
# keys. The runtime's taken_device counter is incremented but never read
# (dead); it is omitted here.

def simulate_runtime_pin(lines, per_layer, per_layer_device, window_count):
    free = window_count
    flight, pinned, dev_keys = set(), [], set()
    layer, taken = -1, 0
    for raw in lines:
        parts = raw.split()
        assert len(parts) == 3, f"line is not 3 fields: {raw!r}"
        pl, pe, _ph = (int(x) for x in parts)
        if pl != layer:                    # block change re-arms the quota
            layer, taken = pl, 0
        if taken >= per_layer:
            continue
        if free <= RUNTIME_WINDOW_RESERVE:
            break
        key = (pl, pe)
        if key in flight:                  # dedup: skipped, still eats quota
            taken += 1
            continue
        flight.add(key)
        free -= 1
        pinned.append(key)
        if taken < per_layer_device:       # VRAM mirror: head of each block
            dev_keys.add(key)
        taken += 1
    return pinned, dev_keys


def emit_pin_list(path, layer_ids, C, order, B_l):
    with open(path, "w", newline="\n") as f:
        for li, l in enumerate(layer_ids):
            for r in range(int(B_l[li])):
                e = int(order[li, r])
                f.write(f"{l} {e} {int(C[li, e])}\n")


# ------------------------------------------------------------------ main ----

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the ExpertStager static pin list from route traces.")
    ap.add_argument("traces", nargs="*", help="ROUTE_TRACE file(s), 18 or 19 fields")
    ap.add_argument("--out", help="output pin-list path (exact runtime format)")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="Dirichlet smoothing (default %(default)s)")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help="total host-tier pinned records B (default %(default)s)")
    ap.add_argument("--vram", type=int, default=DEFAULT_VRAM,
                    help="total VRAM mirror slots D (default %(default)s)")
    ap.add_argument("--vram-mode", choices=("perlayer", "global"),
                    default="perlayer",
                    help="perlayer is the only runtime-expressible mode "
                         "(device keys = head of each layer block); global is "
                         "a report-only comparison (default %(default)s)")
    ap.add_argument("--layer-weights", choices=("uniform", "acts"),
                    default="uniform",
                    help="uniform: every sparse layer sees equal traffic "
                         "(structurally true for this engine); acts: weight "
                         "by observed activations (default %(default)s)")
    ap.add_argument("--experts", type=int, default=DEFAULT_EXPERTS)
    ap.add_argument("--windows", type=int, default=DEFAULT_BUDGET,
                    help="runtime host-tier window count, for the "
                         f"{RUNTIME_WINDOW_RESERVE}-window reserve model "
                         "(default %(default)s)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        run_self_test()
        return

    if not args.traces:
        ap.error("at least one trace file is required (or --self-test)")

    files = []
    for path in args.traces:
        tok, layer, E, width = parse_trace(path, args.experts)
        utok = np.unique(tok)
        half = utok[:len(utok) // 2]
        train_mask = np.isin(tok, half)
        files.append(dict(path=path, tok=tok, layer=layer, E=E, width=width,
                          n_tok=len(utok), train_mask=train_mask))
    layer_ids = np.unique(np.concatenate([f["layer"] for f in files]))

    C = count_matrix(np.concatenate([f["layer"] for f in files]),
                     np.concatenate([f["E"] for f in files]), layer_ids, args.experts)
    C_tr = count_matrix(np.concatenate([f["layer"][f["train_mask"]] for f in files]),
                        np.concatenate([f["E"][f["train_mask"]] for f in files]),
                        layer_ids, args.experts)
    C_te = C - C_tr
    n_test_tok = sum(len(np.unique(f["tok"][~f["train_mask"]])) for f in files)

    alloc = build(C, args.alpha, args.layer_weights, args.budget, args.vram,
                  args.vram_mode, args.experts)

    # ---- report ----
    print("== input ==")
    for f in files:
        print(f"  {f['path']}: {len(f['tok'])} lines, {f['n_tok']} tokens, "
              f"{f['width']}-field format")
    dist = (C > 0).sum(1)
    p = C / np.maximum(C.sum(1, keepdims=True), 1)
    H = -np.where(p > 0, p * np.log2(np.maximum(p, 1e-300)), 0.0).sum(1)
    print(f"  pooled: {len(layer_ids)} layers [{layer_ids[0]}..{layer_ids[-1]}], "
          f"{int(C.sum())} activations, distinct/layer mean {dist.mean():.1f} "
          f"[{dist.min()}..{dist.max()}], entropy mean {H.mean():.2f} bits "
          f"[{H.min():.2f}..{H.max():.2f}]")

    B_l = alloc["B_l"]
    print(f"\n== allocation (alpha={args.alpha}, weights={args.layer_weights}) ==")
    unassigned = args.budget - int(B_l.sum())
    q = np.percentile(B_l, [0, 25, 50, 75, 100]).astype(int)
    print(f"  host budget {args.budget} -> assigned {int(B_l.sum())}"
          + (f" ({unassigned} unassigned: zero predicted value)" if unassigned else ""))
    print(f"  B_l spread min={q[0]} p25={q[1]} med={q[2]} p75={q[3]} max={q[4]}; "
          f"layers capped at {args.experts}: {(B_l >= args.experts).sum()}")
    print(f"  vram D={args.vram} -> uniform k={alloc['k']}/layer -> "
          f"{int(alloc['D_l'].sum())} device keys ({args.vram_mode} mode)")

    cov_is_pl, cov_is = coverage(C, alloc["order"], B_l)
    train_alloc = build(C_tr, args.alpha, args.layer_weights, args.budget,
                        args.vram, args.vram_mode, args.experts)
    cov_oos_pl, cov_oos = coverage(C_te, train_alloc["order"], train_alloc["B_l"])
    lo, hi = wilson(int((C_te * mask_of(train_alloc, layer_ids)).sum()),
                    int(C_te.sum()))

    print(f"\n== expected coverage ==")
    print(f"  in-sample  (final list, same data): pooled {100*cov_is:5.2f}%  "
          f"per-layer min/med/max {100*cov_is_pl.min():.1f}/"
          f"{100*np.median(cov_is_pl):.1f}/{100*cov_is_pl.max():.1f}%")
    if n_test_tok:
        print(f"  split-sample OOS (train {args.alpha}-smoothed first half -> "
              f"test second half, {n_test_tok} test tokens):")
        print(f"  pooled {100*cov_oos:5.2f}%   Wilson 95% [{100*lo:.1f}..{100*hi:.1f}]%")
        print(f"  {'layer':>5} {'B_l(train)':>10} {'OOS%':>6} {'B_l(final)':>10} {'D_l':>4}")
        for li, l in enumerate(layer_ids):
            print(f"  {l:>5} {int(train_alloc['B_l'][li]):>10} "
                  f"{100*cov_oos_pl[li]:>6.2f} {int(B_l[li]):>10} "
                  f"{int(alloc['D_l'][li]):>4}")
    else:
        print("  split-sample OOS: skipped (fewer than 2 tokens per trace)")

    sweep = []
    for a in (0.0, 0.25, 0.5, 1.0, 2.0):
        sa = build(C_tr, a, args.layer_weights, args.budget, args.vram,
                   args.vram_mode, args.experts)
        _, c = coverage(C_te, sa["order"], sa["B_l"])
        sweep.append((a, c))
    dmax = max(abs(c - cov_oos) for a, c in sweep if a != args.alpha) if sweep else 0.0
    print(f"  alpha stability probe (OOS pooled): " +
          "  ".join(f"a={a:g}:{100*c:.2f}%" for a, c in sweep) +
          f"   max|delta| vs default {100*dmax:.2f}pp")

    if n_test_tok:
        # device-mirror-only comparison (D records, both sides)
        d_slots = int(train_alloc["D_l"].sum())
        _, cov_dev_pl = coverage(C_te, train_alloc["order"], train_alloc["D_l"])
        g = global_coverage(C_te, C_tr, args.alpha, d_slots)
        print(f"  vram mirror only ({d_slots} slots): perlayer top-k "
              f"{100*cov_dev_pl:.2f}% vs global-hottest {100*g:.2f}% "
              f"(global NOT emittable: device keys are the head of each "
              f"layer block; per-layer is what the runtime requires)")

    recs_tok = 8 * len(layer_ids)
    print(f"  expected pinned hits per decode token: "
          f"{recs_tok * (cov_oos if n_test_tok else cov_is):.1f} / {recs_tok} records")

    if not args.out:
        print("\nno --out given: report only, nothing emitted")
        return

    emit_pin_list(args.out, layer_ids, C, alloc["order"], B_l)
    raw = open(args.out, "rb").read()
    n_lines = raw.count(b"\n")
    effective = min(n_lines, args.windows - RUNTIME_WINDOW_RESERVE)
    print(f"\n== emit ==")
    print(f"  {args.out}: {n_lines} lines, {len(raw)} bytes, "
          f"LF-only={b'\r' not in raw}, one block per layer, ascending")
    if n_lines > effective:
        print(f"  WARNING: runtime stops pinning at <=16 free windows; with "
              f"--windows {args.windows} only the first {effective} records "
              f"(the hottest) will pin -- consider --budget {effective}")
    print(f"  effective pinned records: {effective}")
    print(f"  recommended env:")
    print(f"    INSIGNIA_GLM53_PIN_LIST={os.path.abspath(args.out)}")
    print(f"    INSIGNIA_GLM53_PIN_HOST={int(B_l.max())}   "
          f"# >= max per-layer block size, else silent truncation")
    print(f"    INSIGNIA_GLM53_PIN_DEV={alloc['k']}        "
          f"# uniform per-layer device-key quota")


def mask_of(alloc, layer_ids):
    """(L,NE) bool mask of the allocation (for pooled Wilson input)."""
    L, NE = alloc["q"].shape
    m = np.zeros((L, NE), bool)
    for l in range(L):
        m[l, alloc["order"][l, :alloc["B_l"][l]]] = True
    return m


# -------------------------------------------------------------- self-test ----

def run_self_test():
    print("== pinlist_v2 self-test ==")
    rng = np.random.default_rng(7)
    NE, LAYERS = 288, [3, 4, 5, 6, 7, 8]
    T = 40

    lines_a, lines_b = [], []
    for t in range(1, T + 1):
        for li, l in enumerate(LAYERS):
            if l == 6 and t % 2:            # partial layer: half the tokens
                continue
            s = 0.3 + 0.18 * li             # skew varies by layer
            ranks = np.arange(NE)
            p = 1.0 / np.power(ranks + 1, s)
            if l == 6:
                p = np.full(NE, 1.0)        # partial layer is FLAT (noise-only)
            p = p[rng.permutation(NE)]      # rotate the head per layer
            p /= p.sum()
            e = rng.choice(NE, size=8, replace=False, p=p)
            scores = " ".join(f"{0.1 + 0.01 * (7 - j):.6e}" for j in range(8))
            line = f"{t} {l} " + " ".join(str(x) for x in e) + " " + scores
            (lines_a if t <= T // 2 else lines_b).append(line)

    with tempfile.TemporaryDirectory() as td:
        fa, fb = os.path.join(td, "a.txt"), os.path.join(td, "b.txt")
        for path, lines in ((fa, lines_a), (fb, lines_b)):
            with open(path, "w", newline="\n") as f:
                f.write("\n".join(lines) + "\n")

        budget, vram, windows = 240, 13, 2425
        argv = ["--out", os.path.join(td, "pin.txt"),
                "--budget", str(budget), "--vram", str(vram),
                "--windows", str(windows), fa, fb]
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(argv)
        out_path = os.path.join(td, "pin.txt")
        raw = open(out_path, "rb").read()
        text = raw.decode("ascii")

        # 1. byte-level format validity against the fscanf contract
        assert raw.endswith(b"\n") and b"\r" not in raw, "must be LF-only, newline-terminated"
        pat = re.compile(r"^([0-9]+) ([0-9]+) ([0-9]+)$")
        parsed = []
        for ln in text.split("\n")[:-1]:
            m = pat.match(ln)
            assert m, f"line violates 'layer expert hits' int triple format: {ln!r}"
            parsed.append(tuple(int(g) for g in m.groups()))
        print(f"  [PASS] format: {len(parsed)} lines, all match '^\\d+ \\d+ \\d+$', LF-only")

        # 2. exactly one contiguous block per layer, strictly increasing
        seen_layers = [l for l, e, h in parsed]
        changes = [i for i in range(1, len(seen_layers)) if seen_layers[i] != seen_layers[i - 1]]
        blocks = [seen_layers[0]] + [seen_layers[i] for i in changes]
        assert blocks == sorted(set(blocks)), "layer blocks must be strictly increasing, one per layer"
        assert set(blocks) <= set(LAYERS)
        print(f"  [PASS] ordering: one contiguous block per layer, strictly increasing {blocks}")

        # 3. dedup + budget + hits-field fidelity
        keys = [(l, e) for l, e, h in parsed]
        assert len(keys) == len(set(keys)), "duplicate (layer,expert) records"
        assert len(parsed) <= budget, "total records exceed budget"
        toks, layers, E, _ = parse_trace(fa, NE)
        toks2, layers2, E2, _ = parse_trace(fb, NE)
        layer_ids = np.unique(np.concatenate([layers, layers2]))
        Call = count_matrix(np.concatenate([layers, layers2]),
                            np.concatenate([E, E2]), layer_ids, NE)
        lidx = {int(l): i for i, l in enumerate(layer_ids)}
        for l, e, h in parsed:
            assert h == Call[lidx[l], e], f"hits field != raw activation count at ({l},{e})"
        B_l = np.array([[l for l, e, h in parsed].count(l) for l in blocks])
        assert B_l.max() <= NE, "per-layer allocation exceeds expert count"
        assert B_l.sum() == len(parsed)
        print(f"  [PASS] dedup + budgets: {len(parsed)} unique records = sum(B_l) <= {budget}; "
              f"max B_l {B_l.max()} <= {NE}; hits == raw counts")

        # 4. faithful C++ parser simulation: with the recommended env knobs the
        #    runtime pins EXACTLY the emitted set and derives the device mirror
        k = vram // len(blocks)
        per_layer, per_layer_dev = int(B_l.max()), k
        pinned, dev_keys = simulate_runtime_pin(text.split("\n")[:-1],
                                                per_layer, per_layer_dev, windows)
        assert pinned == keys, "simulated load_pin_list() deviates from emitted set"
        want_dev = set()
        pos = {}
        for idx, (l, e) in enumerate(keys):
            pos.setdefault(l, []).append(idx)
        for l, idxs in pos.items():
            for idx in idxs[:min(k, len(idxs))]:
                want_dev.add(keys[idx])
        assert dev_keys == want_dev, "device-key derivation mismatch"
        print(f"  [PASS] runtime sim: load_pin_list(PIN_HOST={per_layer}, "
              f"PIN_DEV={per_layer_dev}, windows={windows}) pins all "
              f"{len(pinned)} records; {len(dev_keys)} VRAM keys = head of each block")

        # 5. default runtime knobs (8/2) truncate: the knobs actually matter
        pinned8, dev8 = simulate_runtime_pin(text.split("\n")[:-1], 8, 2, windows)
        assert len(pinned8) == 8 * len(blocks) and len(dev8) == 2 * len(blocks)
        print(f"  [PASS] knob sensitivity: default PIN_HOST=8/PIN_DEV=2 would pin "
              f"only {len(pinned8)} records / {len(dev8)} device keys")

        # 6. window reserve: effective pins cap at windows-16
        pinned_c, _ = simulate_runtime_pin(text.split("\n")[:-1], per_layer,
                                           per_layer_dev, len(parsed) + 10)
        assert len(pinned_c) == len(parsed) - 6, "16-window reserve not modeled"
        print(f"  [PASS] window reserve: with windows={len(parsed)+10}, only "
              f"{len(pinned_c)}/{len(parsed)} pin (free<=16 stop)")

        # 7. alpha invariance at equal acts/layer; shrinkage on the partial layer
        Ctr = count_matrix(np.concatenate([layers[(toks <= T // 2)],
                                           layers2[(toks2 > T // 2)]]),
                           np.concatenate([E[(toks <= T // 2)], E2[(toks2 > T // 2)]]),
                           layer_ids, NE)
        equal_mask = np.ones(len(layer_ids), bool)
        eq_layers = [l for l in LAYERS if l != 6]
        sub = Ctr[[lidx[l] for l in eq_layers]]
        a0 = build(sub, 0.0, "uniform", 60, 7, "perlayer", NE)
        a5 = build(sub, 0.5, "uniform", 60, 7, "perlayer", NE)
        assert np.array_equal(a0["B_l"], a5["B_l"]), \
            "alpha must not change allocation at equal acts/layer"
        full0 = build(Ctr, 0.0, "uniform", 240, 13, "perlayer", NE)
        full5 = build(Ctr, 0.5, "uniform", 240, 13, "perlayer", NE)
        thin = lidx[6]
        print(f"  [PASS] alpha: invariant at equal N (B_l identical for a=0 vs 0.5); "
              f"flat partial layer B_l: a=0 -> {int(full0['B_l'][thin])}, "
              f"a=0.5 -> {int(full5['B_l'][thin])} (smoothing shrinks thin noisy layers)")
        assert full5["B_l"][thin] <= full0["B_l"][thin]

        # 8. OOS machinery sanity + determinism
        m1 = mask_of(build(Call, 0.5, "uniform", budget, vram, "perlayer", NE), layer_ids)
        m2 = mask_of(build(Call, 0.5, "uniform", budget, vram, "perlayer", NE), layer_ids)
        assert (m1 == m2).all(), "allocation is not deterministic"
        emit_pin_list(out_path + ".2", layer_ids, Call,
                      build(Call, 0.5, "uniform", budget, vram, "perlayer", NE)["order"],
                      build(Call, 0.5, "uniform", budget, vram, "perlayer", NE)["B_l"])
        assert open(out_path + ".2", "rb").read() == raw, "emit is not byte-deterministic"
        pl, pooled = coverage(Call, build(Call, 0.5, "uniform", budget, vram,
                                          "perlayer", NE)["order"],
                              build(Call, 0.5, "uniform", budget, vram,
                                    "perlayer", NE)["B_l"])
        assert 0.0 <= pooled <= 1.0 and pl.min() >= 0.0
        print(f"  [PASS] determinism (byte-identical re-emit) + coverage bounds "
              f"(in-sample pooled {100*pooled:.2f}%)")
    print("== ALL SELF-TEST CHECKS PASSED ==")


if __name__ == "__main__":
    main()
