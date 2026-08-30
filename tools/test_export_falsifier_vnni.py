#!/usr/bin/env python3
"""Format and payload checks for the native Falsifier VNNI export."""

from __future__ import annotations

from pathlib import Path
import struct
import tempfile

import numpy as np

from export_falsifier_vnni import (
    ENTRY_FLOAT_TENSOR,
    ENTRY_HEADER,
    ENTRY_QUANTIZED_MATRIX,
    FILE_HEADER,
    FORMAT_VERSION,
    MAGIC,
    export,
    fnv1a,
    load_state,
    raw_sources,
    rotate_left_one,
)


def test_export() -> None:
    state = load_state(None)
    expected_raw = {name: value for name, value in raw_sources(state)}
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "controller.ifvnni"
        report = export(None, output)
        assert report["format_version"] == FORMAT_VERSION
        assert report["matrix_count"] == 24
        assert report["raw_tensor_count"] == len(expected_raw) == 18
        assert report["entry_count"] == 42

        payload = output.read_bytes()
        magic, version, count, expected_manifest, _ = FILE_HEADER.unpack_from(payload)
        assert magic == MAGIC
        assert version == FORMAT_VERSION
        assert count == report["entry_count"]
        offset = FILE_HEADER.size
        manifest = 0
        names: list[str] = []
        for index in range(count):
            fields = ENTRY_HEADER.unpack_from(payload, offset)
            offset += ENTRY_HEADER.size
            name = fields[0].split(b"\0", 1)[0].decode("ascii")
            rows, logical, padded, kind, byte_count, checksum = fields[1:]
            entry_payload = payload[offset:offset + byte_count]
            assert fnv1a(entry_payload) == checksum
            manifest = rotate_left_one(manifest) ^ checksum
            offset += byte_count + ((-byte_count) & 63)
            names.append(name)
            if index < 24:
                assert kind == ENTRY_QUANTIZED_MATRIX
                assert padded >= logical
            else:
                assert kind == ENTRY_FLOAT_TENSOR
                assert logical == padded == 1
                expected = expected_raw[name].astype("<f4", copy=False).reshape(-1)
                actual = np.frombuffer(entry_payload, dtype="<f4")
                assert rows == expected.size
                assert np.array_equal(actual, expected)
        assert len(names) == len(set(names))
        assert offset == len(payload)
        assert manifest == expected_manifest == report["manifest_checksum"]


if __name__ == "__main__":
    test_export()
    print("falsifier VNNI export tests passed")
