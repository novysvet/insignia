#!/bin/bash
set -e
cat > /var/lib/insignia/analysis/accept/s3_parse.py <<'PYEOF'
#!/usr/bin/env python3
"""Pack-job parser for DFlash2 acceptance artifacts on glm-box.

Reads each artifact exactly once, caches parsed results as CSV/JSON under
/var/lib/insignia/analysis/accept/. CPU-only; never touches the GPU or the
engine binaries.
"""
import csv
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict

ROOT = "/var/lib/insignia"
OUT = os.path.join(ROOT, "analysis", "accept")
os.makedirs(OUT, exist_ok=True)

DFLASH_RE = re.compile(
    r"(\d+)-token prompt ([\d.]+) s; (\d+) greedy tokens in (\d+) DFLASH2-k(\d+) rounds "
    r"\(([\d.]+) accepted/round, (\d+) empty; ([\d.]+) ms/token; "
    r"draft ([\d.]+) ms/round \+ verify ([\d.]+) ms/verified round \((\d+) verified\), "
    r"fallback ([\d.]+) ms\)")
SCALAR_RE = re.compile(
    r"(\d+)-token prompt ([\d.]+) s; (\d+) greedy tokens? total ([\d.]+) s")
HIST_RE = re.compile(r"^  accepted histogram((?: \d+:\d+)*)$", re.MULTILINE)
IDS_RE = re.compile(r"^greedy IDs(.*)$", re.MULTILINE)
HOST_CACHE_RE = re.compile(r"NVFP4 cache (\d+)/(\d+) hits \(([\d.]+)%, ([\d.]+) GiB NVMe\+H2D avoided; (\d+) slots\)")
VRAM_CACHE_RE = re.compile(r"VRAM expert tier (\d+)/(\d+) hits \(([\d.]+)%, ([\d.]+) GiB PCIe avoided; (\d+) slots\)")
ODIRECT_RE = re.compile(r"QD8 expert O_DIRECT ([\d.]+) GiB / ([\d.]+) s \(([\d.]+) GB/s\)")
READWAIT_RE = re.compile(r"expert read-wait ([\d.]+) s of ([\d.]+) s expert wall")
PREFETCH_RE = re.compile(r"expert prefetch (\d+) started, (\d+) adopted, (\d+) wasted \(([\d.]+) GiB speculative\)")
HIER_RE = re.compile(r"streamed ([\d.]+) GiB in ([\d.]+) s \(([\d.]+) GB/s hierarchy aggregate\)")
PIN_RE = re.compile(r"pin list: (\d+) hot records pinned in host tier \((\d+) VRAM keys\)")
RECALL_RE = re.compile(r"pre-attention route recall (\d+)/(\d+) \(([\d.]+)%\)")
UNION_RE = re.compile(r"batched pre-attention union recall ([\d.]+)%, precision ([\d.]+)% "
                      r"\((\d+) useful / (\d+) predicted / (\d+) actual; (\d+)/(\d+) hints started\)")
PROMPTLINE_RE = re.compile(r"^position (\d+) top10(.*)$", re.MULTILINE)


def seq_stats(ids):
    if not ids:
        return {}
    bigram_same = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
    runs, best, cur = [], 1, 1
    for a, b in zip(ids, ids[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    cnt = Counter(ids)
    return {
        "gen_len": len(ids),
        "distinct": len(cnt),
        "distinct_ratio": round(len(cnt) / len(ids), 4),
        "bigram_repeat": round(bigram_same / max(1, len(ids) - 1), 4),
        "max_run": best,
        "top1_share": round(cnt.most_common(1)[0][1] / len(ids), 4),
        "top1_token": cnt.most_common(1)[0][0],
    }


def parse_log(path, variant, stem):
    text = open(path, encoding="utf-8", errors="replace").read()
    row = {"variant": variant, "case": stem, "path": path}
    dataset, index, mode = stem.rsplit("-", 2)
    row.update(dataset=dataset, row_index=index, mode=mode)
    m = DFLASH_RE.search(text)
    ids_m = IDS_RE.search(text)
    ids = [int(v) for v in ids_m.group(1).split()] if ids_m else []
    row.update(seq_stats(ids))
    if m:
        (ptok, ps, gen, rounds, k, acc, empty, mstok, dms, vms, nver, fbms) = m.groups()
        row.update(kind="dflash", prompt_tokens=int(ptok), prompt_s=float(ps),
                   generated=int(gen), rounds=int(rounds), verify_k=int(k),
                   accepted_per_round=float(acc), empty_rounds=int(empty),
                   ms_per_token=float(mstok), draft_ms_per_round=float(dms),
                   verify_ms_per_verified_round=float(vms), verified_rounds=int(nver),
                   fallback_total_ms=float(fbms))
        h = HIST_RE.search(text)
        hist = {}
        if h:
            for pair in h.group(1).split():
                a, b = pair.split(":")
                hist[int(a)] = int(b)
        row["accepted_histogram"] = json.dumps(hist)
        row["hist_total"] = sum(hist.values())
    else:
        s = SCALAR_RE.search(text)
        if s:
            ptok, ps, gen, tot = s.groups()
            row.update(kind="scalar", prompt_tokens=int(ptok), prompt_s=float(ps),
                       generated=int(gen),
                       ms_per_token=round(1000.0 * (float(tot) - float(ps)) / int(gen), 2))
        else:
            row.update(kind="other")
    for name, rx in (("host_cache", HOST_CACHE_RE), ("vram_cache", VRAM_CACHE_RE),
                     ("odirect", ODIRECT_RE), ("readwait", READWAIT_RE),
                     ("prefetch", PREFETCH_RE), ("hierarchy", HIER_RE),
                     ("pinlist", PIN_RE), ("recall", RECALL_RE), ("union", UNION_RE)):
        mm = rx.search(text)
        if mm:
            row[name] = mm.groups()
    row["n_prompt_lines"] = len(PROMPTLINE_RE.findall(text))
    gaps = []
    for pm in PROMPTLINE_RE.finditer(text):
        pairs = pm.group(2).split()
        logits = [float(p.split(":", 1)[1]) for p in pairs[:2] if ":" in p]
        if len(logits) == 2:
            gaps.append(round(logits[0] - logits[1], 4))
    if gaps:
        gaps.sort()
        row["gap_mean"] = round(sum(gaps) / len(gaps), 4)
        row["gap_median"] = gaps[len(gaps) // 2]
        row["gap_p25"] = gaps[len(gaps) // 4]
        row["gap_n"] = len(gaps)
    return row


def part_bench():
    rows = []
    base = os.path.join(ROOT, "bench-results")
    for dirpath, _dirs, files in os.walk(base):
        variant = os.path.relpath(dirpath, base)
        for fn in sorted(files):
            path = os.path.join(dirpath, fn)
            if fn.endswith(".log"):
                stem = fn[:-4]
                try:
                    rows.append(parse_log(path, variant, stem))
                except Exception as exc:
                    rows.append({"variant": variant, "case": stem, "path": path,
                                 "kind": "ERROR", "error": str(exc)})
            elif fn == "results.json":
                rel = os.path.relpath(path, base).replace("\\", "/")
                try:
                    data = json.load(open(path, encoding="utf-8"))
                    with open(os.path.join(OUT, "raw_" + rel.replace("/", "__") ), "w") as fh:
                        json.dump(data, fh, indent=1)
                except Exception as exc:
                    print("results.json parse fail", path, exc)
            elif fn == "summary.md":
                rel = os.path.relpath(path, base).replace("\\", "/")
                try:
                    with open(os.path.join(OUT, "raw_" + rel.replace("/", "__")), "w",
                              encoding="utf-8") as fh:
                        fh.write(open(path, encoding="utf-8").read())
                except Exception as exc:
                    print("summary copy fail", path, exc)
    flat = []
    for r in rows:
        flat.append({k: (json.dumps(v) if isinstance(v, tuple) else v)
                     for k, v in r.items()})
    keys = sorted({k for r in flat for k in r})
    with open(os.path.join(OUT, "runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    print("bench logs parsed:", len(rows))
    return rows


# ---------------- dfdump binary parse ----------------
KH = 4096          # drafter hidden
DLAYERS = 5        # drafter layers

def part_dfdump():
    path = os.path.join(ROOT, "dfdump", "r12")
    events = []
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        while True:
            b = f.read(1)
            if not b:
                break
            t = b[0]
            if t == 1:
                count, pos0 = struct.unpack("ii", f.read(8))
                f.seek(count * DLAYERS * KH * 4, 1)
                events.append({"tag": "commit", "count": count, "pos0": pos0})
            elif t == 2:
                anchor, apos = struct.unpack("ii", f.read(8))
                events.append({"tag": "forward", "anchor": anchor, "anchor_pos": apos})
            elif t == 3:
                apos = struct.unpack("i", f.read(4))[0]
                n1 = struct.unpack("i", f.read(4))[0]
                f.seek(4 * n1, 1)
                n2 = struct.unpack("i", f.read(4))[0]
                f.seek(4 * n2, 1)
                events.append({"tag": "fwd_intermediates", "anchor_pos": apos})
            elif t == 4:
                layer = struct.unpack("b", f.read(1))[0]
                n = struct.unpack("i", f.read(4))[0]
                f.seek(4 * n, 1)
                events.append({"tag": "layer_trace", "layer": layer})
            elif t == 5:
                f.read(2)
                rows, cols = struct.unpack("ii", f.read(8))
                f.seek(4 * rows * cols, 1)
                events.append({"tag": "stage_trace"})
            else:
                events.append({"tag": "UNKNOWN", "byte": t, "offset": f.tell() - 1})
                break
            if f.tell() > size:
                events.append({"tag": "OVERRUN"})
                break
    with open(os.path.join(OUT, "dfdump_events.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["i", "tag", "count", "pos0", "anchor",
                                           "anchor_pos", "byte", "offset"])
        w.writeheader()
        for i, e in enumerate(events):
            w.writerow(dict({"i": i}, **e))
    print("dfdump events:", len(events), Counter(e["tag"] for e in events))
    return events


# ---------------- early-multi trace parse ----------------
def part_early_multi():
    for name in ("early-multi-df-k7.txt", "early-multi-prompt.txt"):
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            continue
        batches = {}
        order = []
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 6:
                    continue
                batch = int(parts[0])
                layer, tokens = int(parts[1]), int(parts[2])
                overlap, predicted, distinct = int(parts[3]), int(parts[4]), int(parts[5])
                rec = batches.get(batch)
                if rec is None:
                    rec = batches[batch] = {"batch": batch, "layers": 0, "tokens": tokens,
                                            "sum_distinct": 0, "sum_predicted": 0,
                                            "sum_overlap": 0, "layer_set": set()}
                    order.append(batch)
                rec["layers"] += 1
                rec["layer_set"].add(layer)
                rec["sum_distinct"] += distinct
                rec["sum_predicted"] += predicted
                rec["sum_overlap"] += overlap
        out = os.path.join(OUT, name.replace(".txt", "_batches.csv"))
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["batch", "tokens", "layers", "sum_distinct",
                                               "sum_predicted", "sum_overlap"])
            w.writeheader()
            for b in order:
                r = batches[b]
                w.writerow({k: r[k] for k in w.fieldnames})
        byk = defaultdict(list)
        for r in batches.values():
            byk[(r["tokens"], r["layers"] == 42)].append(r["sum_distinct"])
        stats = []
        for (tokens, full), vals in sorted(byk.items()):
            vals2 = sorted(vals)
            stats.append({"tokens": tokens, "complete_42layers": full, "n": len(vals),
                          "distinct_mean": round(sum(vals) / len(vals), 2),
                          "distinct_median": vals2[len(vals2) // 2],
                          "distinct_min": min(vals), "distinct_max": max(vals)})
        sout = os.path.join(OUT, name.replace(".txt", "_d_emp.csv"))
        with open(sout, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["tokens", "complete_42layers", "n",
                                               "distinct_mean", "distinct_median",
                                               "distinct_min", "distinct_max"])
            w.writeheader()
            w.writerows(stats)
        print(name, "batches:", len(batches),
              Counter(r["tokens"] for r in batches.values()))


def main():
    part_bench()
    part_dfdump()
    part_early_multi()
    print("cached under", OUT)

if __name__ == "__main__":
    main()

PYEOF
python3 /var/lib/insignia/analysis/accept/s3_parse.py
