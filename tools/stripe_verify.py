import os
import struct
import sys

sys.path.insert(0, "/mnt/e/coding/Insignia/tools")
from stripe_repack import load_index

head, shards, entries = load_index(__import__("pathlib").Path("/var/lib/insignia/glm53-flash-text-striped.index"))
n_orig = 120
by_name = {e[0]: e for e in entries}

SRC = "/var/lib/insignia/glm53-flash-text"
DST = "/stripe"

# compare every 997th striped expert, all 9 tensors
import random
random.seed(7)
checked = 0
for layer in (3, 17, 33, 44):
    for expert in random.sample(range(1, 288, 2), 4):
        stem = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        for proj in ("down_proj", "gate_proj", "up_proj"):
            for suffix in (".weight", ".weight_scale", ".weight_scale_2"):
                name = stem + proj + suffix
                e = by_name[name]
                assert e[3] >= n_orig, f"{name} not striped"
                spath = f"{DST}/{shards[e[3]][0]}"
                want = e[6]
                got = os.pread(os.open(spath, os.O_RDONLY), want, e[5])
                # source bytes from ORIGINAL index
                _, oshards, oentries = load_index(__import__("pathlib").Path("/var/lib/insignia/glm53-flash-text.index"))
                oby = {x[0]: x for x in oentries}
                oe = oby[name]
                ref = os.pread(os.open(f"{SRC}/{oshards[oe[3]][0]}", os.O_RDONLY), oe[6], oe[5])
                assert got == ref, f"BYTE MISMATCH {name}"
        checked += 1
        print(f"layer {layer} expert {expert}: 9/9 tensors byte-identical", flush=True)
print(f"ALL {checked} sampled striped experts byte-identical")

# non-striped tensors must still reference original shards
probe = by_name["model.language_model.layers.10.mlp.experts.0.down_proj.weight"]
assert probe[3] < n_orig
print("even expert still on original shard ok")
