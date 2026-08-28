import sys
sys.path.insert(0, "/mnt/e/coding/Insignia/tools")
import struct, time, uuid

SECTOR = 512

def checksum(buf):
    return (~sum(buf)) & 0xFFFFFFFF

virtual = 24 << 30
total_sectors = virtual // SECTOR
spt, heads = 63, 16
cylinders = min(65535, total_sectors // (heads * spt))
f = struct.pack(
    ">8sIIQI4sI4sQQHBBII",
    b"conectix", 0x00000002, 0x00010000, 0xFFFFFFFFFFFFFFFF,
    int(time.time() - 946684800), b"win ", 0x00060001, b"Wi2k",
    virtual, virtual, cylinders, heads, spt, 3, 0,
)
print("struct len", len(f))
f += uuid.uuid4().bytes
f += b"\x00"
f += b"\x00" * 427
print("total len", len(f))
s0 = sum(f)
print("sum zeroed", hex(s0))
chk = checksum(f)
print("chk", hex(chk))
f2 = f[:64] + struct.pack(">I", chk) + f[68:]
print("verify (~sum)&mask:", hex(checksum(f2)))
print("sum mod 2^32:", hex(sum(f2) & 0xFFFFFFFF))
