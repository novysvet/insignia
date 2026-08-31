#!/usr/bin/env python3

import pathlib
import struct
import tempfile
import unittest

from glm53_gguf import GGUFFile


def string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def make_fixture(path: pathlib.Path) -> tuple[int, bytes]:
    metadata = [
        ("general.alignment", 4, struct.pack("<I", 64)),
        ("general.architecture", 8, string("glm5next")),
        ("test.array", 9, struct.pack("<IQIII", 4, 3, 7, 8, 9)),
    ]
    name = "blk.3.ffn_gate_exps.weight"
    descriptor = (
        string(name) + struct.pack("<IQQQIQ", 3, 256, 2, 3, 18, 0)
    )
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata)))
    for key, type_id, value in metadata:
        header += string(key) + struct.pack("<I", type_id) + value
    header += descriptor
    data_offset = (len(header) + 63) & -64
    payload = bytes([index // 196 + 1 for index in range(588)])
    path.write_bytes(header + bytes(data_offset - len(header)) + payload)
    return data_offset, payload


class GGUFTest(unittest.TestCase):
    def test_header_and_expert_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "fixture.gguf"
            data_offset, payload = make_fixture(path)
            reader = GGUFFile(path)
            self.assertEqual(reader.data_offset, data_offset)
            self.assertEqual(reader.metadata["general.architecture"], "glm5next")
            self.assertEqual(reader.metadata["test.array"], [7, 8, 9])
            tensor = reader.tensor("blk.3.ffn_gate_exps.weight")
            self.assertEqual(tensor.type_name, "IQ3_XXS")
            self.assertEqual(tensor.dimensions, (256, 2, 3))
            self.assertEqual(tensor.bytes, len(payload))
            for expert in range(3):
                offset, size = tensor.expert_span(expert)
                self.assertEqual(size, 196)
                self.assertEqual(path.read_bytes()[offset:offset + size],
                                 bytes([expert + 1]) * size)

    def test_rejects_out_of_range_expert(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "fixture.gguf"
            make_fixture(path)
            tensor = GGUFFile(path).tensors[0]
            with self.assertRaises(ValueError):
                tensor.expert_span(3)

    def test_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "fixture.gguf"
            make_fixture(path)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                GGUFFile(path)


if __name__ == "__main__":
    unittest.main()

