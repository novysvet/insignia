#!/usr/bin/env python3
"""Text-mode Insignia generation: chat.py <index> <model_dir> [prompt words...]

Tokenizes with the checkpoint's tokenizer.json (chat template), runs the engine DLL,
detokenizes the committed id stream. Greedy for now."""
import ctypes
import json
import subprocess
import sys
import os

MAX_NEW_DEFAULT = 256


def load_dll(path, tries=12):
    for i in range(tries):
        try:
            return ctypes.CDLL(path)
        except OSError as e:
            if "4551" not in str(e) or i == tries - 1:
                raise
            import time
            time.sleep(10)


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    index, model_dir = os.path.abspath(sys.argv[1]), sys.argv[2]
    prompt = " ".join(sys.argv[3:]) or "Hello!"
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    ids = tok.encode(prompt, add_special_tokens=False).ids
    eos = tok.token_to_id("<|im_end|>") or 151645
    max_new = MAX_NEW_DEFAULT

    out = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "rundll.py"),
         os.path.join(os.path.dirname(__file__), "..", "build", "generate.dll"),
         index, ",".join(map(str, ids)), str(max_new)],
        capture_output=True, text=True, check=False)
    lines = out.stdout.strip().splitlines()
    timing = next((l for l in lines if l.startswith("prefill")), "")
    ids_line = next((l for l in lines if l.startswith("ids:")), None)
    if ids_line is None:
        sys.stderr.write(out.stderr)
        raise SystemExit("engine failed")
    gen = [int(t) for t in ids_line.split()[1:]]
    # cut at any end-of-turn marker the host loop may have kept
    for stop in (eos, 151643):
        if stop in gen:
            gen = gen[:gen.index(stop)]
    text = tok.decode(gen, skip_special_tokens=False)
    print(timing)
    print(text)


if __name__ == "__main__":
    main()
