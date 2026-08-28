#!/usr/bin/env python3
"""Build an empty dynamic VHD (conectix/cxsparse) mountable via `wsl --mount --vhd`.

Usage: mkvhd.py <output> <virtual_size_gib>
Writes only the structural metadata (footer copy, dynamic header, BAT of
0xFFFFFFFF, trailing footer): the file is a few hundred KiB and grows as the
guest writes. No admin or Hyper-V module needed.
"""
import os
import struct
import sys
import time
import uuid

SECTOR = 512
BLOCK = 2 << 20  # 2 MiB data blocks


def checksum(buf: bytes) -> int:
    return (~sum(buf)) & 0xFFFFFFFF


def footer(virtual_size: int, disk_type: int = 3) -> bytes:
    total_sectors = virtual_size // SECTOR
    spt, heads = 63, 16
    cylinders = min(65535, total_sectors // (heads * spt))
    f = struct.pack(
        ">8sIIQI4sI4sQQHBBII",
        b"conectix",            # cookie
        0x00000002,            # features (no subsets, no temp)
        0x00010000,            # file format version 1.0
        0xFFFFFFFFFFFFFFFF,    # data offset (dynamic: unused)
        int(time.time() - 946684800),  # seconds since 2000-01-01
        b"win ",               # creator app
        0x00060001,            # creator version (WS2003)
        b"Wi2k",               # creator host OS
        virtual_size,          # original size
        virtual_size,          # current size
        cylinders, heads, spt, # geometry (2+1+1)
        disk_type,             # 3 = dynamic
        0,                     # checksum (patched below)
    )
    f += uuid.uuid4().bytes    # unique id
    f += b"\x00"               # saved state
    f += b"\x00" * 427         # reserved -> 512 bytes
    assert len(f) == SECTOR
    chk = checksum(f)
    f = f[:64] + struct.pack(">I", chk) + f[68:]
    assert checksum(f) == 0xFFFFFFFF
    return f


def dyn_header(num_blocks: int, table_offset: int) -> bytes:
    h = struct.pack(
        ">8sQQ I I I I",
        b"cxsparse",
        0xFFFFFFFFFFFFFFFF,
        table_offset,
        0x00010000,
        num_blocks,
        BLOCK,
        0,
    )
    h += b"\x00" * 16          # parent uuid
    h += struct.pack(">I", 0)   # parent timestamp
    h += b"\x00" * 4
    h += b"\x00" * 512          # parent unicode name
    h += b"\x00" * 192          # parent locators
    h += b"\x00" * 252          # reserved -> 1024 bytes
    assert len(h) == 1024
    chk = checksum(h)
    h = h[:36] + struct.pack(">I", chk) + h[40:]
    assert checksum(h) == 0xFFFFFFFF
    return h


def main():
    out, gib = sys.argv[1], int(sys.argv[2])
    virtual = gib << 30
    num_blocks = (virtual + BLOCK - 1) // BLOCK
    table_offset = SECTOR + 1024  # right after dyn header
    bat_bytes = num_blocks * 4
    bat_padded = (bat_bytes + SECTOR - 1) // SECTOR * SECTOR
    end_footer = table_offset + bat_padded
    with open(out, "wb") as f:
        f.write(footer(virtual))
        f.write(dyn_header(num_blocks, table_offset))
        f.write(b"\xFF" * bat_padded)
        f.write(footer(virtual))
    print(f"{out}: dynamic VHD {gib} GiB, {num_blocks} blocks, "
          f"structural size {end_footer + SECTOR} bytes")


if __name__ == "__main__":
    main()
