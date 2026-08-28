#!/usr/bin/env python3
"""Tokenize/detokenize with the GLM-5.3 tokenizer (fast tokenizers only).
Usage: tok.py 'prompt text'      -> prints comma-separated token ids
       tok.py --decode 1,2,3     -> prints text
"""
import sys
from tokenizers import Tokenizer

tok = Tokenizer.from_file("/mnt/e/coding/Insignia/GLM-5.3-Flash-ABLITERATED-NVFP4/tokenizer.json")
if sys.argv[1] in ("--decode", "-d"):
    ids = [int(x) for x in sys.argv[2].split(",") if x.strip()]
    print(tok.decode(ids))
elif sys.argv[1] in ("--file", "-f"):
    ids = tok.encode(open(sys.argv[2], encoding="utf-8").read()).ids
    print(",".join(map(str, ids)))
else:
    ids = tok.encode(sys.argv[1]).ids
    print(",".join(map(str, ids)))
