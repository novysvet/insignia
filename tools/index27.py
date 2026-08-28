#!/usr/bin/env python3
"""INSIDX02 multi-shard index builder for Qwen3.8-27B-FP8 (66 safetensors).

Emits build/qwen38-27b-fp8.insignia-index: magic "INSIDX02", version 2, a fixed
9-field shape header, a 66-entry shard table (relative paths + file bytes +
crc32, VERIFIED against <model>/crc32.txt — build fails on mismatch), then a
name-sorted tensor table under ENGINE names (src/qwen35.cu / src/decode.cu
lookup convention: matrix() acquires base+".weight" and base+".scales").

Name map (checkpoint -> engine, verified live against all 66 headers):
  model.language_model.layers.N.*        -> language_model.model.layers.N.*
  model.language_model.embed_tokens.weight-> language_model.model.embed_tokens.weight
  model.language_model.norm.weight        -> language_model.model.norm.weight
  lm_head.weight                          -> language_model.lm_head.weight
  mtp.*                                   -> language_model.mtp.*
  X.weight_scale_inv (BF16 [r/128,c/128]) -> <mapped stem>.scales
  model.visual.*                          -> SKIPPED (vision tower; counted + reported)

DType tags match include/insignia_model.hpp (+ loader-gaps §1 extension):
  F32=1, BF16=2, F8_E4M3=7. All integers little-endian, no padding anywhere.

Usage: python tools/index27.py <model_dir> <output_index> [--no-crc]
"""
import argparse
import json
import os
import pathlib
import random
import struct
import sys
import zlib

MAGIC = b"INSIDX02"
VERSION = 2
DTYPES = {"F32": 1, "BF16": 2, "F8_E4M3": 7}  # DType enum: f32=1, bf16=2, f8_e4m3=7
SCALE_INFIX, SCALE_SUFFIX = "_scale_inv", ".weight_scale_inv"  # X.weight -> X.weight_scale_inv

# Fixed shape header (verified against config.json text_config at build time).
SHAPE = {
    "hidden": 5120, "layers": 64, "vocab": 248320, "q_heads": 24, "kv_heads": 4,
    "delta_v": 48, "delta_k": 16, "inter": 17408, "full_attention_interval": 4,
}
SHAPE_ORDER = ("hidden", "layers", "vocab", "q_heads", "kv_heads",
               "delta_v", "delta_k", "inter", "full_attention_interval")
CONFIG_KEYS = {  # shape field -> config.json text_config key
    "hidden": "hidden_size", "layers": "num_hidden_layers", "vocab": "vocab_size",
    "q_heads": "num_attention_heads", "kv_heads": "num_key_value_heads",
    "delta_v": "linear_num_value_heads", "delta_k": "linear_num_key_heads",
    "inter": "intermediate_size", "full_attention_interval": "full_attention_interval",
}
CONFIG_EXPECT = {  # config key -> (derived value expected in config)
    "hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64,
    "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
    "vocab_size": 248320, "full_attention_interval": 4,
    "linear_num_value_heads": 48, "linear_num_key_heads": 16,
    "linear_value_head_dim": 128, "linear_key_head_dim": 128,
}
BYTES_PER_ELEM = {1: 4, 2: 2, 7: 1}  # dtype tag -> bytes/element


def fail(msg: str) -> None:
    print(f"index27: FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def engine_name(ckpt: str) -> str:
    """Checkpoint tensor name -> engine name (qwen35.cu/decode.cu convention)."""
    if ckpt.endswith(SCALE_SUFFIX):
        stem = ckpt[: -len(SCALE_SUFFIX)]
        return _map_stem(stem) + ".scales"
    if not ckpt.endswith(".weight"):
        return _map_stem(ckpt)  # A_log / dt_bias keep their leaf names verbatim
    return _map_stem(ckpt[: -len(".weight")]) + ".weight"


def _map_stem(stem: str) -> str:
    if stem.startswith("model.language_model."):
        return "language_model.model." + stem[len("model.language_model."):]
    if stem == "lm_head":
        return "language_model.lm_head"
    if stem.startswith("mtp"):
        return "language_model.mtp" + stem[3:]
    raise ValueError(f"unmappable checkpoint name: {stem!r}")


def read_header(path: pathlib.Path):
    with path.open("rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    return hlen, {k: v for k, v in hdr.items() if k != "__metadata__"}


def crc32_file(path: pathlib.Path) -> int:
    crc = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 24)  # 16 MiB
            if not chunk:
                break
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def load_crc_manifest(model_dir: pathlib.Path) -> dict:
    out = {}
    for line in (model_dir / "crc32.txt").read_text().splitlines():
        line = line.rstrip()
        if not line:
            continue
        crc_hex, name = line.split("  ", 1)
        out[name] = int(crc_hex, 16)
    return out


def verify_shape_header(model_dir: pathlib.Path) -> None:
    cfg = json.loads((model_dir / "config.json").read_text()).get("text_config", {})
    for key, expect in CONFIG_EXPECT.items():
        got = cfg.get(key)
        if got != expect:
            fail(f"config.json text_config {key}={got}, expected {expect}")
    layer_types = cfg.get("layer_types", [])
    if len(layer_types) != SHAPE["layers"] or sum(1 for t in layer_types if t == "full_attention") != 16:
        fail(f"config.json layer_types census unexpected ({len(layer_types)} entries)")
    print(f"shape header verified vs config.json: "
          f"{', '.join(f'{k}={SHAPE[k]}' for k in SHAPE_ORDER)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an INSIDX02 multi-shard Insignia index")
    ap.add_argument("model", type=pathlib.Path, help="Qwen3.8-27B-FP8 model directory")
    ap.add_argument("output", type=pathlib.Path, help="output .insignia-index path")
    ap.add_argument("--no-crc", action="store_true",
                    help="skip the crc32.txt verification pass (debug only)")
    args = ap.parse_args()
    model_dir = args.model.resolve()
    out_path = args.output.resolve()

    shards = sorted(model_dir.glob("layers-*.safetensors"),
                    key=lambda p: int(p.stem.split("-")[1]))
    shards += [model_dir / "mtp.safetensors", model_dir / "outside.safetensors"]
    if len(shards) != 66:
        fail(f"expected 66 shards, found {len(shards)}")
    verify_shape_header(model_dir)
    crc_manifest = load_crc_manifest(model_dir)

    # ---- pass 1: headers, name map, scale links, byte accounting ----------------
    shard_info, tensors, skipped_vision = [], [], 0
    total_f8 = total_bf16 = total_other = 0
    for idx, shard in enumerate(shards):
        hlen, hdr = read_header(shard)
        data_start = 8 + hlen
        f8_weights, indexed_bytes, skipped_bytes = set(), 0, 0
        f8_bytes = bf16_bytes = other_bytes = n_f8 = n_bf16 = 0
        for name, spec in hdr.items():
            begin, end = spec["data_offsets"]
            nbytes = end - begin
            if spec["dtype"] == "F8_E4M3" and name + SCALE_INFIX in hdr:
                f8_weights.add(name)
            if name.startswith("model.visual."):
                skipped_vision += 1
                skipped_bytes += nbytes
                continue
            ename = engine_name(name)
            if not ename.isascii():
                fail(f"non-ascii engine name {ename!r} (sort order would diverge from C++)")
            dtype = DTYPES.get(spec["dtype"])
            if dtype is None:
                fail(f"unsupported dtype {spec['dtype']} for {name}")
            shape = spec["shape"]
            if len(shape) > 255:
                fail(f"rank > 255 for {name}")
            expect = BYTES_PER_ELEM[dtype]
            for d in shape:
                expect *= d
            if expect != nbytes:
                fail(f"{name}: bytes {nbytes} != shape/dtype product {expect}")
            tensors.append({"name": ename, "ckpt": name, "dtype": dtype, "shape": shape,
                            "shard": idx, "off": data_start + begin, "bytes": nbytes})
            indexed_bytes += nbytes
            if spec["dtype"] == "F8_E4M3":
                f8_bytes += nbytes; n_f8 += 1
            elif spec["dtype"] == "BF16":
                bf16_bytes += nbytes; n_bf16 += 1
            else:
                other_bytes += nbytes
        # byte accounting: header + every tensor (indexed + skipped) == file size
        file_bytes = shard.stat().st_size
        if 8 + hlen + indexed_bytes + skipped_bytes != file_bytes:
            fail(f"{shard.name}: 8+header+payload != file size "
                 f"({8 + hlen + indexed_bytes + skipped_bytes} != {file_bytes})")
        # scale-link: every F8 weight has X.weight_scale_inv (BF16, ceil/128 shape) in-shard
        for name in f8_weights:
            sc = hdr[name + SCALE_INFIX]
            r, c = hdr[name]["shape"]
            if sc["dtype"] != "BF16" or sc["shape"] != [cdiv(r, 128), cdiv(c, 128)]:
                fail(f"bad companion scale for {name}: {sc['dtype']} {sc['shape']}")
        ckpt_indexed = {n for n in hdr if not n.startswith("model.visual.")}
        for name in ckpt_indexed:
            if name.endswith(SCALE_SUFFIX):
                stem = name[: -len(SCALE_SUFFIX)]
                if stem + ".weight" not in f8_weights:
                    fail(f"orphan scale {name} (no F8 {stem}.weight in shard)")
        shard_info.append({"path": shard, "file_bytes": file_bytes, "crc": crc_manifest.get(shard.name),
                           "hlen": hlen, "tensors": len(ckpt_indexed), "f8": n_f8, "bf16": n_bf16,
                           "f8_bytes": f8_bytes, "bf16_bytes": bf16_bytes, "other_bytes": other_bytes,
                           "skipped_bytes": skipped_bytes})
        total_f8 += f8_bytes; total_bf16 += bf16_bytes; total_other += other_bytes

    # every F8 tensor got a scale (407 expected); count via suffix scan
    n_scale = sum(1 for t in tensors if t["name"].endswith(".scales"))
    n_f8t = sum(1 for t in tensors if t["dtype"] == DTYPES["F8_E4M3"])
    if n_scale != n_f8t:
        fail(f"F8 tensors {n_f8t} != .scales entries {n_scale}")
    if len(tensors) != 1606 - skipped_vision:
        fail(f"indexed {len(tensors)} tensors, expected {1606 - skipped_vision} (1606 - {skipped_vision} vision)")

    # ---- pass 2: crc32 verification against crc32.txt (hard fail on mismatch) --
    if not args.no_crc:
        for info in shard_info:
            if info["crc"] is None:
                fail(f"{info['path'].name}: no crc32.txt entry")
            got = crc32_file(info["path"])
            if got != info["crc"]:
                fail(f"{info['path'].name}: crc32 {got:08x} != crc32.txt {info['crc']:08x}")
        print(f"crc32 verified: {len(shard_info)}/66 shards match crc32.txt")
    else:
        print("crc32 verification SKIPPED (--no-crc)")

    # ---- write INSIDX02 --------------------------------------------------------
    rel_base = out_path.parent
    blob = bytearray()
    blob += struct.pack("<8sI", MAGIC, VERSION)
    blob += struct.pack("<9I", *[SHAPE[k] for k in SHAPE_ORDER])
    blob += struct.pack("<I", len(shard_info))
    for info in shard_info:
        rel = os.path.relpath(info["path"], rel_base)  # relative to the INDEX dir, native separators
        pb = rel.encode("utf-8")
        blob += struct.pack("<IQI", len(pb), info["file_bytes"], info["crc"] or 0)
        blob += pb
    tensors.sort(key=lambda t: t["name"])
    blob += struct.pack("<I", len(tensors))
    for t in tensors:
        nb = t["name"].encode("utf-8")
        blob += struct.pack("<HBB", len(nb), t["dtype"], len(t["shape"]))
        blob += nb
        blob += struct.pack("<" + "Q" * len(t["shape"]), *t["shape"])
        blob += struct.pack("<HQQ", t["shard"], t["off"], t["bytes"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    print(f"wrote {out_path} ({len(blob)} bytes, {len(tensors)} tensors, {len(shard_info)} shards)")

    # ---- summary table ---------------------------------------------------------
    print(f"{'shard':<26}{'tensors':>8}{'F8':>4}{'BF16':>5}{'F8 MiB':>10}{'BF16 MiB':>10}{'skip MiB':>10}{'hdr':>7}")
    for info in shard_info:
        print(f"{info['path'].name:<26}{info['tensors']:>8}{info['f8']:>4}{info['bf16']:>5}"
              f"{info['f8_bytes'] / 2**20:>10.2f}{info['bf16_bytes'] / 2**20:>10.2f}"
              f"{info['skipped_bytes'] / 2**20:>10.2f}{8 + info['hlen']:>7}")
    grand = sum(i["file_bytes"] for i in shard_info)
    print(f"TOTAL: {len(tensors)} indexed (+{skipped_vision} vision skipped) | "
          f"F8 {total_f8 / 2**30:.3f} GiB, BF16 {(total_bf16 + sum(i['skipped_bytes'] for i in shard_info)) / 2**30:.3f} GiB, "
          f"other {total_other} B | shard bytes {grand} == sum(file sizes): "
          f"{grand == sum(i['file_bytes'] for i in shard_info)}")
    ok = all(8 + i["hlen"] + i["f8_bytes"] + i["bf16_bytes"] + i["other_bytes"] + i["skipped_bytes"] == i["file_bytes"]
             for i in shard_info)
    print(f"byte accounting: F8 + BF16 + header == file size for all 66 shards: {ok}")

    # ---- self-read: parse the index back and spot-check ------------------------
    self_read(out_path, tensors, shard_info, shards)


def self_read(out_path, tensors, shard_info, shards) -> None:
    blob = out_path.read_bytes()
    off = 0

    def take(fmt):
        nonlocal off
        vals = struct.unpack_from("<" + fmt, blob, off)
        off += struct.calcsize("<" + fmt)
        return vals if len(vals) > 1 else vals[0]

    magic = blob[:8]; off = 8
    if magic != MAGIC:
        fail("self-read: bad magic")
    version = take("I")
    if version != VERSION:
        fail(f"self-read: version {version}")
    shape_vals = take("9I")
    if list(shape_vals) != [SHAPE[k] for k in SHAPE_ORDER]:
        fail(f"self-read: shape header {shape_vals}")
    shard_count = take("I")
    r_shards = []
    for _ in range(shard_count):
        plen, file_bytes, crc = take("IQI")
        r_shards.append({"path": blob[off:off + plen].decode("utf-8"), "file_bytes": file_bytes, "crc": crc})
        off += plen
    tensor_count = take("I")
    r_tensors = {}
    for _ in range(tensor_count):
        name_len, dtype, rank = take("HBB")
        name = blob[off:off + name_len].decode("utf-8"); off += name_len
        shape = struct.unpack_from("<" + "Q" * rank, blob, off)  # always a tuple, never collapsed
        off += 8 * rank
        shard_idx, abs_off, nbytes = take("HQQ")
        r_tensors[name] = {"dtype": dtype, "shape": shape, "shard": shard_idx, "off": abs_off, "bytes": nbytes}
    if off != len(blob):
        fail(f"self-read: {len(blob) - off} trailing bytes")
    if len(r_tensors) != len(tensors):
        fail(f"self-read: tensor count {len(r_tensors)} != {len(tensors)}")
    for i, (a, b) in enumerate(zip(r_shards, shard_info)):
        if a["file_bytes"] != b["file_bytes"] or a["crc"] != b["crc"]:
            fail(f"self-read: shard table mismatch at {i}")
    rng = random.Random(27)
    probes = rng.sample(sorted(r_tensors), 10)
    by_engine = {t["name"]: t for t in tensors}
    for name in probes:
        rt = r_tensors[name]
        ck = by_engine[name]["ckpt"]
        hlen, hdr = read_header(shards[rt["shard"]])
        if ck not in hdr:
            fail(f"self-read: {name} -> {ck} not in shard {rt['shard']}")
        begin, end = hdr[ck]["data_offsets"]
        want = (8 + hlen + begin, end - begin, hdr[ck]["dtype"], hdr[ck]["shape"])
        got = (rt["off"], rt["bytes"], {v: k for k, v in DTYPES.items()}[rt["dtype"]], tuple(rt["shape"]))
        if got[:2] != want[:2] or got[2:] != (want[2], tuple(want[3])):
            fail(f"self-read mismatch {name}: index {got} vs header {want}")
        print(f"self-read OK {name}: shard={rt['shard']} off={rt['off']} bytes={rt['bytes']} "
              f"dtype={got[2]} shape={got[3]}")
    # matrix() convention probe: every F8 .weight has a .scales; bf16 matrices that don't
    f8_bases = {t["name"][: -len(".weight")] for t in tensors
                if t["dtype"] == DTYPES["F8_E4M3"] and t["name"].endswith(".weight")}
    missing = [b for b in f8_bases if b + ".scales" not in r_tensors]
    if missing:
        fail(f"matrix() convention broken: F8 without .scales: {missing[:3]}")
    bf16_no_scale = sorted(t["name"][: -len(".weight")] for t in tensors
                           if t["dtype"] == DTYPES["BF16"] and t["name"].endswith(".weight")
                           and len(t["shape"]) == 2 and (t["name"][: -len(".weight")] + ".scales") not in r_tensors)
    fams = {}
    for b in bf16_no_scale:  # group by leaf module (e.g. 'linear_attn.in_proj_a', 'mtp.fc')
        leaf = b.split(".")[-2] + "." + b.split(".")[-1] if "layers." in b else b
        fams[leaf] = fams.get(leaf, 0) + 1
    print(f"matrix() probe: {len(f8_bases)} F8 bases all have .scales companions; "
          f"bf16 2-D matrices WITHOUT .scales (matrix() must tolerate): "
          f"{len(bf16_no_scale)} tensors in {len(fams)} families -> {fams}")
    print("SELF-READ PASS")


if __name__ == "__main__":
    main()
