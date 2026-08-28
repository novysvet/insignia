#!/usr/bin/env python3
"""One-shot repair of build/qwen35-insig4-text.safetensors.

quantize_insig4.py v1 wrote non-quantized tensors as fp16 bytes labeled BF16
(engine reads them as bf16 -> garbage/NaN). This streams the file once and
rewrites every BF16 tensor as real bf16 (round-to-nearest-even via fp32),
leaving U32 weights and F16 scales untouched.
"""
import json, struct, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
with open(src, 'rb') as f:
    n = struct.unpack('<q', f.read(8))[0]
    hdr = json.loads(f.read(n))
    data_start = 8 + n

    def convert(name, raw, e):
        # A_log must stay F32 (engine casts it to const float*); everything else
        # non-quantized is real bf16. Input bytes are fp16 (the v1 quantizer bug).
        f32 = np.frombuffer(raw, dtype='<f2').astype(np.float32)
        if name.endswith('.A_log'):
            return f32.astype('<f4').tobytes(), 'F32'
        bits = f32.view(np.uint32).copy()
        bits += np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
        return (bits >> np.uint32(16)).astype('<u2').tobytes(), 'BF16'

    order = sorted(hdr, key=lambda k: hdr[k]['data_offsets'][0])
    new_hdr, off, new_offs, new_blobs = {}, 0, {}, {}
    for name in order:
        e = hdr[name]
        f.seek(data_start + e['data_offsets'][0])
        raw = f.read(e['data_offsets'][1] - e['data_offsets'][0])
        if e['dtype'] == 'BF16':
            raw, dt = convert(name, raw, e)
        else:
            dt = e['dtype']
        new_offs[name] = (off, off + len(raw))
        new_blobs[name] = raw
        new_hdr[name] = {"dtype": dt, "shape": e['shape'], "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    header = json.dumps(new_hdr, separators=(',', ':')).encode()
    with open(dst, 'wb') as o:
        o.write(struct.pack('<q', len(header)))
        o.write(header)
        for name in order:
            o.write(new_blobs[name])
print(f"rewrote {off/2**30:.2f} GiB -> {dst}")
