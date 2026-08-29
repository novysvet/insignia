#!/bin/bash
# Focused inventory: bench-results contents, s6 logs, analysis dir, dfdump header.
D=/var/lib/insignia
echo "== files under bench-results (all subdirs) =="
find "$D/bench-results" -type f -printf "%s %p\n" 2>/dev/null | head -60
echo "== s6-args (file, 130B) =="
cat "$D/s6-args" 2>/dev/null; echo
echo "== s6-task.log (45B) =="
cat "$D/s6-task.log" 2>/dev/null; echo
echo "== s6-vramtier.log =="
cat "$D/s6-vramtier.log" 2>/dev/null; echo
echo "== analysis dir =="
find "$D/analysis" -maxdepth 3 -printf "%y %s %p\n" 2>/dev/null | head -30
echo "== dfdump/r12 first 64 bytes (hex) =="
xxd -l 64 "$D/dfdump/r12" 2>/dev/null || od -A d -t x1 -N 64 "$D/dfdump/r12"
echo "== dfdump/r12 tag scan (first 200 tags) =="
python3 - <<'EOF'
import struct, collections
f = open('/var/lib/insignia/dfdump/r12','rb')
KH = 4096
tags = collections.Counter()
seq = []
try:
    while True:
        b = f.read(1)
        if not b or len(b) < 1: break
        t = b[0]
        if t == 2:
            a, p = struct.unpack('ii', f.read(8)); seq.append(('F', a, p))
        elif t == 1:
            c, p0 = struct.unpack('ii', f.read(8))
            n = 5 * c * KH
            f.seek(n, 1); seq.append(('C', c, p0))
        elif t == 3:
            p = struct.unpack('i', f.read(4))[0]
            n1 = struct.unpack('i', f.read(4))[0]
            f.seek(4*n1, 1)
            n2 = struct.unpack('i', f.read(4))[0]
            f.seek(4*n2, 1); seq.append(('I', p, n1, n2))
        else:
            seq.append(('?', t, f.tell())); break
except Exception as e:
    seq.append(('ERR', str(e)))
print('events:', len(seq))
print('first 40:', seq[:40])
print('last 10:', seq[-10:])
EOF
echo "== early-route-math.txt head (truncated cols) =="
head -3 "$D/early-route-math.txt" | cut -c1-200
echo "== bench-data tree (2 levels) =="
find "$D/bench-data" -maxdepth 2 | head -20
