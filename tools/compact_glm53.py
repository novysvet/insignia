#!/usr/bin/env python3
"""Compact the text-only tensors of a sharded GLM-5.3 checkpoint.

The abliterated checkpoint carries a ~200 GiB vision tower the text runner
never touches; this streams only the indexed text tensors into fresh
safetensors shards with 4096-byte aligned offsets (clean O_DIRECT reads),
rewrites model.safetensors.index.json, and copies the config sidecars.

Shards copy in parallel: the source Samsung 980 (DRAM-less) starves at
~290 MB/s under a single QD1 stream but sustains >1 GB/s with queue depth,
and WSL's 9p bridge needs concurrency to aggregate as well.
"""

import argparse
import concurrent.futures
import json
import os
import pathlib
import shutil
import struct
import sys
import time

ALIGN = 8
CHUNK = 4 << 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    weight_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    headers = {}
    for shard_name in sorted(set(weight_map.values())):
        with (source / shard_name).open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            headers[shard_name] = (json.loads(handle.read(size)), 8 + size)

    # shard -> [(tensor, meta)] in original file order (sequential reads)
    per_shard = {}
    for name, shard_name in weight_map.items():
        if name.startswith("model.visual."):
            continue
        header, _ = headers[shard_name]
        per_shard.setdefault(shard_name, []).append((name, header[name]))
    shard_names = [name for name, tensors in sorted(per_shard.items()) if tensors]
    total_shards = len(shard_names)

    def copy_shard(index):
        shard_name = shard_names[index]
        tensors = per_shard[shard_name]
        out_name = f"model-{index + 1:05d}-of-{total_shards:05d}.safetensors"
        entries = {}
        layout = []  # (name, source_offset, declared_start, length)
        cursor = 0
        header, data_start = headers[shard_name]
        for name, meta in tensors:
            begin, end = meta["data_offsets"]
            length = end - begin
            cursor = (cursor + ALIGN - 1) & ~(ALIGN - 1)
            entries[name] = {"dtype": meta["dtype"], "shape": meta["shape"],
                             "data_offsets": [cursor, cursor + length]}
            layout.append((name, data_start + begin, cursor, length))
            cursor += length
        header_bytes = json.dumps(entries, separators=(",", ":")).encode()
        out_path = output / out_name
        if out_path.exists() and out_path.stat().st_size == 8 + len(header_bytes) + cursor:
            return index, out_name, cursor, 0.0, {name: out_name for name, _, _, _ in layout}
        began = time.time()
        moved = 0
        with (source / shard_name).open("rb", buffering=0) as reader, out_path.open("wb") as writer:
            writer.write(struct.pack("<Q", len(header_bytes)))
            writer.write(header_bytes)
            written = 0  # bytes since data start
            for _, source_offset, declared, length in layout:
                # Place each tensor exactly where its header declares it.
                if declared > written:
                    writer.write(b"\0" * (declared - written))
                    written = declared
                reader.seek(source_offset)
                remaining = length
                while remaining:
                    # 9p reads fail with ENOMEM under cumulative kernel memory
                    # pressure; back off and retry instead of losing the shard.
                    block = None
                    for attempt in range(6):
                        try:
                            block = reader.read(min(CHUNK, remaining))
                            break
                        except OSError:
                            if attempt == 5:
                                raise
                            time.sleep(3.0 * (attempt + 1))
                    if not block:
                        raise SystemExit(f"unexpected EOF reading {shard_name}")
                    writer.write(block)
                    remaining -= len(block)
                    moved += len(block)
                    written += len(block)
            # 180 GiB of dirty output pages otherwise exhaust the WSL VM's
            # ~8 GiB and the next 9p read fails with ENOMEM.
            writer.flush()
            os.fsync(writer.fileno())
            os.posix_fadvise(writer.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(reader.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        return index, out_name, cursor, time.time() - began, {name: out_name for name, _, _, _ in layout}

    new_map = {}
    written = 0
    began = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, out_name, cursor, seconds, mapping in pool.map(copy_shard, range(total_shards)):
            new_map.update(mapping)
            written += cursor
            print(f"{out_name}: {cursor / (1 << 20):.0f} MiB in {seconds:.1f}s, "
                  f"{written / (1 << 30):.1f} GiB total, {written / (time.time() - began) / 1e6:.0f} MB/s",
                  flush=True)

    (output / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": written}, "weight_map": new_map}))
    for sidecar in ("config.json", "generation_config.json", "tokenizer_config.json",
                    "tokenizer.json", "chat_template.jinja"):
        if (source / sidecar).exists():
            shutil.copy2(source / sidecar, output / sidecar)
    print(f"done: {total_shards} shards, {written / (1 << 30):.1f} GiB, "
          f"{written / (time.time() - began) / 1e6:.0f} MB/s average", flush=True)


if __name__ == "__main__":
    sys.exit(main())
