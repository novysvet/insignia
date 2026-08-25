#!/usr/bin/env python3
"""NLL comparison: MXFP4 vs INSIG4 checkpoints on a real text sample.

Tokenizes a text corpus with the BF16 model's tokenizer, runs the engine NLL tool
against both indexes, prints NLL/ppl side by side.
"""
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
MODELDIR = pathlib.Path(os.path.expanduser(
    r"~\.cache\huggingface\hub\models--Qwen--Qwen3.5-9B\snapshots"))


def run(index, ids):
    out = subprocess.run(
        [sys.executable, str(HERE / "rundll.py"), str(ROOT / "build" / "nll.dll"),
         str(index), ",".join(map(str, ids))],
        capture_output=True, text=True)
    return out.stdout.strip() or out.stderr.strip()


def main():
    snap = sorted(MODELDIR.glob("*"))[0]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    # a few paragraphs of ordinary prose (public-domain style) as the probe text
    text = (
        "The history of computing machinery is in part the history of automatic arithmetic. "
        "Charles Babbage conceived a difference engine that would tabulate polynomial functions "
        "through the method of finite differences, removing human error from the production of "
        "logarithmic tables. A century later, electronic valves replaced gears, and the stored "
        "program made the machine universal. Software became the new literature: programs were "
        "written, read, argued over, and revised like essays. Modern accelerators execute "
        "millions of instructions in the time a mechanical relay moved once, yet the essential "
        "problems of specification, correctness, and taste remain stubbornly human. "
        "The best engineers treat a compiler the way a writer treats a style guide: as an "
        "instrument for turning intention into artifact with the fewest possible accidents. "
        "Quantization, in this light, is compression with an editorial conscience: keep the "
        "meaning, discard the redundancy, and measure what was lost.")
    ids = tok.encode(text, add_special_tokens=False).ids
    print(f"{len(ids)} tokens")
    for name, idx in [("mxfp4", ROOT / "build" / "qwen35.insignia-index"),
                      ("insig4", ROOT / "build" / "qwen35-insig4.insignia-index")]:
        if not idx.exists():
            print(f"{name}: index missing ({idx})")
            continue
        print(f"{name}: {run(idx, ids)}")


if __name__ == "__main__":
    main()
