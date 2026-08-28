#!/usr/bin/env python3
"""Build the dual-drive expert stripe store for GLM-5.3-Flash.

Every routed expert of every sparse layer is a 9-tensor record
    {down,gate,up}_proj x {weight, weight_scale, weight_scale_2}
which the engine reads as one contiguous span (its packed fast path). This
script writes the ODD-numbered experts of each sparse layer as tight
4096-aligned records into new stripe shards on the second drive, and emits a
new index in which those tensors point at the stripe shards; every other
tensor keeps its original (shard, offset) so the main store stays untouched.

Usage:
  stripe_repack.py <src_store_dir> <src_index> <dst_dir> <new_index> [--parity 1]
Environment: run inside the WSL guest; src on the root vhdx, dst on /stripe.
"""
import os
import pathlib
import struct
import sys
import time

MAGIC = b"IGLMIDX1"
BODY = 4 << 20
SCALES = 512 << 10
GLOBAL = 4
PROJ_ORDER = ["down_proj", "gate_proj", "up_proj"]
MEMBER_SUFFIX = [".weight", ".weight_scale", ".weight_scale_2"]
SHARD_BYTES = (4 << 30) - (128 << 20)  # ~3.88 GiB per stripe shard
READ_CHUNK = 32 << 20
# Byte-proportional placement: the E: drive sustains ~2.58 GB/s where the C:
# root sustains ~5.94, so only ~30% of each layer's experts go to the stripe
# (shared demand FIFO gates every batch on the slower drive otherwise).
def stripe_expert(expert):
    return expert % 10 < 3


def load_index(path):
    data = path.read_bytes()
    assert data[:8] == MAGIC
    head = struct.unpack_from("<11IQ", data, 8)
    off = 8 + struct.calcsize("<11IQ")
    shards = []
    for _ in range(head[2]):
        nlen, size = struct.unpack_from("<HQ", data, off)
        off += struct.calcsize("<HQ")
        shards.append([data[off:off + nlen].decode(), size])
        off += nlen
    entries = []
    for _ in range(head[3]):
        nlen, dtype, ndim, shard, flags, absolute, length = struct.unpack_from("<HBBHHQQ", data, off)
        off += struct.calcsize("<HBBHHQQ")
        name = data[off:off + nlen].decode()
        off += nlen
        shape = struct.unpack_from("<" + "I" * ndim, data, off)
        off += 4 * ndim
        entries.append([name, dtype, shape, shard, flags, absolute, length])
    assert off == len(data)
    return head, shards, entries


def main():
    src_dir = pathlib.Path(sys.argv[1])
    src_index = pathlib.Path(sys.argv[2])
    dst_dir = pathlib.Path(sys.argv[3])
    out_index = pathlib.Path(sys.argv[4])
    parity = 1
    for arg in sys.argv[5:]:
        if arg.startswith("--parity"):
            parity = int(arg.split("=")[1])

    head, shards, entries = load_index(src_index)
    (version, flags0, nshards, nentries, hidden, layers, vocab,
     experts, topk, moe_inter, hc, total) = head

    # Group tensors per expert; geometry sanity.
    by_name = {e[0]: e for e in entries}
    sparse_layers = []
    for layer in range(layers):
        stem = f"model.language_model.layers.{layer}.mlp.experts.0.down_proj.weight"
        if stem in by_name:
            sparse_layers.append(layer)
    if not sparse_layers:
        raise SystemExit("no sparse layers found")
    print(f"sparse layers: {sparse_layers[0]}..{sparse_layers[-1]} ({len(sparse_layers)})")

    # Open source shard fds (direct twin for big aligned reads: records are
    # read exactly once, so bypassing the page cache avoids writeback stalls).
    fds = {}
    dfds = {}
    for sid, (name, size) in enumerate(shards):
        p = src_dir / name
        actual = p.stat().st_size
        if actual != size:
            raise SystemExit(f"size mismatch {p}: {actual} != {size}")
        fds[sid] = os.open(p, os.O_RDONLY)
        dfds[sid] = os.open(p, os.O_RDONLY | os.O_DIRECT)

    # Plan stripe records.
    stripe_tensors = {}   # name -> (stripe_sid, offset)
    stripe_sizes = []     # final size per stripe shard
    stripe_bytes = 0
    plan = []             # (layer, expert) in emission order
    for layer in sparse_layers:
        for expert in range(experts):
            if stripe_expert(expert):
                plan.append((layer, expert))
    print(f"striping {len(plan)} of {experts * len(sparse_layers)} expert records")

    # Writer state.
    dst_dir.mkdir(parents=True, exist_ok=True)
    cur_shard = -1
    cur_file = None
    cur_off = 0
    t_start = time.time()
    done_bytes = 0

    def open_next_shard():
        nonlocal cur_shard, cur_file, cur_off
        if cur_file is not None:
            os.fsync(cur_file.fileno())
            os.posix_fadvise(cur_file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            cur_file.close()
            stripe_sizes.append(cur_off)
        cur_shard += 1
        name = stripe_name(cur_shard)
        cur_file = open(dst_dir / name, "wb", buffering=0)
        cur_off = 0

    def stripe_name(i):
        return f"stripe-{i + 1:05d}.bin"

    open_next_shard()

    def write_record(parts):
        """parts: [(src_sid, src_off, length)] in engine order; returns (sid, off).

        Members pack tight as weight/scale/global per projection (engine ABI),
        but each projection body is kept at an offset divisible by 16 inside
        the record: the nvfp4 gate/up GEMV loads uint4 vectors and faults on
        alignment 4. The engine tolerates <=64B gaps BETWEEN projections."""
        nonlocal cur_off, cur_shard, cur_file, done_bytes
        assert cur_off % 4096 == 0
        record_bytes = sum(length for _, _, length in parts)
        record_bytes = 0
        for i, (_, _, length) in enumerate(parts):
            if i in (3, 6):  # next projection body: 16B align
                record_bytes += (-record_bytes) % 16
            record_bytes += length
        if cur_off + record_bytes > SHARD_BYTES:
            open_next_shard()
        where = (cur_shard, cur_off)
        member_offsets = []
        for i, (src_sid, src_off, length) in enumerate(parts):
            if i in (3, 6):
                pad16 = (-(cur_off)) % 16
                if pad16:
                    cur_file.write(b"\x00" * pad16)
                    cur_off += pad16
            member_offsets.append(cur_off)
            remaining = length
            src_pos = src_off
            while remaining:
                step = min(remaining, READ_CHUNK)
                block = os.pread(fds[src_sid], step, src_pos)
                assert len(block) == step, f"short read: {len(block)} != {step}"
                written = 0
                while written < step:
                    written += cur_file.write(block[written:])
                src_pos += step
                remaining -= step
                cur_off += step
                done_bytes += step
            # Records are read exactly once: drop the source pages immediately
            # so the cache stays available for the writeback path.
            if length % 4096 == 0 and src_off % 4096 == 0:
                os.posix_fadvise(fds[src_sid], src_off, length, os.POSIX_FADV_DONTNEED)
        pad = (-cur_off) % 4096
        if pad:
            cur_file.write(b"\x00" * pad)
            cur_off += pad
        return where, member_offsets

    for i, (layer, expert) in enumerate(plan):
        stem = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        parts = []
        names = []
        for proj in PROJ_ORDER:
            for suffix in MEMBER_SUFFIX:
                name = stem + proj + suffix
                e = by_name[name]
                parts.append((e[3], e[5], e[6]))
                names.append(name)
        where, member_offsets = write_record(parts)
        for name, moff in zip(names, member_offsets):
            stripe_tensors[name] = (where[0], moff)
        if (i + 1) % 200 == 0:
            dt = time.time() - t_start
            print(f"{i + 1}/{len(plan)} records, {done_bytes / 2**30:.1f} GiB, "
                  f"{done_bytes / 2**20 / max(dt, 1e-9):.0f} MiB/s", flush=True)

    os.fsync(cur_file.fileno())
    cur_file.close()
    stripe_sizes.append(cur_off)
    n_stripe = len(stripe_sizes)
    dt = time.time() - t_start
    print(f"wrote {n_stripe} stripe shards, {done_bytes / 2**30:.2f} GiB in {dt:.0f}s "
          f"({done_bytes / 2**20 / max(dt, 1e-9):.0f} MiB/s)", flush=True)

    # Emit the new index: original shards keep [name,size]; stripe shards appended.
    out_index.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_index.with_suffix(out_index.suffix + ".tmp")
    moved = 0
    with open(tmp, "wb") as out:
        out.write(struct.pack("<8s11IQ", MAGIC, version, flags0,
                              nshards + n_stripe, nentries, hidden, layers, vocab,
                              experts, topk, moe_inter, hc, total))
        for name, size in shards:
            enc = name.encode()
            out.write(struct.pack("<HQ", len(enc), size))
            out.write(enc)
        for i in range(n_stripe):
            enc = stripe_name(i).encode()
            out.write(struct.pack("<HQ", len(enc), stripe_sizes[i]))
            out.write(enc)
        for name, dtype, shape, shard, flags, absolute, length in entries:
            if name in stripe_tensors:
                ssid, soff = stripe_tensors[name]
                shard = nshards + ssid
                absolute = soff
                moved += 1
            enc = name.encode()
            out.write(struct.pack("<HBBHHQQ", len(enc), dtype, len(shape),
                                  shard, flags, absolute, length))
            out.write(enc)
            out.write(struct.pack("<" + "I" * len(shape), *shape))
    os.replace(tmp, out_index)
    print(f"index: {out_index} ({moved} tensors remapped to {n_stripe} stripe shards)")


if __name__ == "__main__":
    main()
