#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import pathlib
import json
import random
import struct
import sys
import tempfile
import unittest

TOOLS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import fp8_residency_codec as codec


class BitPackingTest(unittest.TestCase):
    def test_round_trip_and_padding_rejection(self) -> None:
        rng = random.Random(0xF8C0DEC)
        for bits in range(1, 9):
            for count in range(0, 80):
                values = [rng.randrange(1 << bits) for _ in range(count)]
                packed = codec.pack_bits(values, bits)
                self.assertEqual(codec.unpack_bits(packed, count, bits), values)
                self.assertEqual(len(packed), (count * bits + 7) // 8)
                if count and (count * bits) & 7:
                    corrupt = bytearray(packed)
                    corrupt[-1] |= 1 << ((count * bits) & 7)
                    with self.assertRaises(codec.CodecError):
                        codec.unpack_bits(corrupt, count, bits)


class ModeTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = random.Random(0xE4F3)
        self.patterns = [
            bytes(range(128)),
            bytes(range(256)) * 4,
            bytes([0x00, 0x80, 0x7F, 0xFF, 0x7E, 0xFE, 0x01, 0x81] * 128),
            bytes([0] * 1024),
            bytes([0x80] * 1024),
            bytes(rng.randrange(256) for _ in range(1024)),
            bytes((index * 37 + (index >> 3)) & 0xFF for index in range(1024)),
        ]

    def test_every_mode_exact(self) -> None:
        for mode in codec.Mode:
            for raw in self.patterns:
                if len(raw) % 64:
                    continue
                with self.subTest(mode=mode.name, bytes=len(raw)):
                    encoded = codec.encode_mode(mode, raw)
                    decoded = codec.decode_mode(
                        encoded.mode, encoded.parameter, encoded.payload, len(raw)
                    )
                    self.assertEqual(decoded, raw)

    def test_all_256_values_signed_zero_and_nan(self) -> None:
        raw = bytes(range(256)) * 4
        source_sha = hashlib.sha256(raw).hexdigest()
        for mode in codec.Mode:
            encoded = codec.encode_mode(mode, raw)
            decoded = codec.decode_mode(mode, encoded.parameter, encoded.payload, len(raw))
            self.assertEqual(hashlib.sha256(decoded).hexdigest(), source_sha)
            self.assertEqual(decoded[0], 0x00)
            self.assertEqual(decoded[0x80], 0x80)
            self.assertEqual(decoded[0x7F], 0x7F)
            self.assertEqual(decoded[0xFF], 0xFF)

    def test_tile_raw_escape_wins_equal_sector(self) -> None:
        # No compressed mode may replace RAW unless it removes at least one
        # complete 16-byte payload sector.
        rng = random.Random(9)
        for _ in range(100):
            raw = bytes(rng.randrange(256) for _ in range(128))
            chosen = codec.encode_tile(raw)
            if chosen.padded_bytes == codec.align_up(len(raw)):
                self.assertEqual(chosen.mode, codec.Mode.RAW)

    def test_mode_payload_malformed_rejection(self) -> None:
        raw = bytes([1, 2, 1, 2, 3, 4, 1, 2] * 16)
        palette = codec.encode_mode(codec.Mode.BYTE_PALETTE4, raw)
        self.assertGreaterEqual(palette.parameter, 2)
        duplicate = bytearray(palette.payload)
        duplicate[1] = duplicate[0]
        with self.assertRaises(codec.CodecError):
            codec.decode_mode(palette.mode, palette.parameter, duplicate, len(raw))

        exp = codec.encode_mode(codec.Mode.EXP_PALETTE2, raw)
        with self.assertRaises(codec.CodecError):
            codec.decode_mode(exp.mode, exp.parameter, exp.payload[:-1], len(raw))

        zero = codec.encode_mode(codec.Mode.ZERO_SPARSE, bytes([0] * 64))
        corrupt = bytearray(zero.payload)
        # Mark one position non-zero without adding its required literal.
        corrupt[0] |= 1
        with self.assertRaises(codec.CodecError):
            codec.decode_mode(zero.mode, zero.parameter, corrupt, 64)

        with self.assertRaises(codec.CodecError):
            codec.decode_mode(255, 0, raw, len(raw))


class MatrixContainerTest(unittest.TestCase):
    @staticmethod
    def expected_tile(raw: bytes, rows: int, cols: int, tile_bytes: int, index: int) -> bytes:
        row_begin, group, valid_rows = codec.tile_coordinates(rows, cols, tile_bytes, index)
        result = bytearray()
        for row in range(row_begin, row_begin + valid_rows):
            begin = row * cols + group * 64
            result.extend(raw[begin : begin + 64])
        return bytes(result)

    def test_random_access_and_matrix_round_trip(self) -> None:
        rng = random.Random(12345)
        for rows in (1, 2, 3, 7, 16, 17, 31):
            for cols in (64, 128, 320):
                raw = bytes(rng.randrange(256) for _ in range(rows * cols))
                for tile_bytes in codec.ALLOWED_TILE_BYTES:
                    with self.subTest(rows=rows, cols=cols, tile=tile_bytes):
                        encoded = codec.encode_matrix(raw, rows, cols, tile_bytes)
                        decoder = codec.MatrixDecoder(encoded)
                        self.assertEqual(decoder.decode_matrix(), raw)
                        self.assertEqual(decoder.raw_bytes, len(raw))
                        for index in range(decoder.tile_count):
                            self.assertEqual(
                                decoder.decode_tile(index),
                                self.expected_tile(raw, rows, cols, tile_bytes, index),
                            )

    def test_deterministic_encoding(self) -> None:
        raw = bytes((index * 13 + index // 64) & 0xFF for index in range(19 * 256))
        first = codec.encode_matrix(raw, 19, 256, 1024)
        second = codec.encode_matrix(raw, 19, 256, 1024)
        self.assertEqual(first, second)
        self.assertEqual(
            hashlib.sha256(first).hexdigest(), hashlib.sha256(second).hexdigest()
        )
        stats = codec.MatrixDecoder(first).stats()
        self.assertEqual(stats.raw_bytes, len(raw))
        self.assertEqual(stats.encoded_bytes, len(first))
        self.assertEqual(sum(stats.mode_counts.values()), stats.tile_count)

    def test_header_directory_truncation_and_padding_rejection(self) -> None:
        raw = bytes([0] * (16 * 128))
        encoded = codec.encode_matrix(raw, 16, 128, 1024)
        for cut in (0, 1, 7, 63, 64, 71, len(encoded) - 1):
            with self.subTest(cut=cut):
                with self.assertRaises(codec.CodecError):
                    codec.MatrixDecoder(encoded[:cut])

        bad_magic = bytearray(encoded)
        bad_magic[0] ^= 1
        with self.assertRaises(codec.CodecError):
            codec.MatrixDecoder(bad_magic)

        bad_mode = bytearray(encoded)
        bad_mode[codec.HEADER_BYTES + 6] = 255
        with self.assertRaises(codec.CodecError):
            codec.MatrixDecoder(bad_mode)

        bad_offset = bytearray(encoded)
        struct.pack_into("<I", bad_offset, codec.HEADER_BYTES, 1)
        with self.assertRaises(codec.CodecError):
            codec.MatrixDecoder(bad_offset)

        decoder = codec.MatrixDecoder(encoded)
        descriptor = decoder.descriptors[0]
        self.assertLess(descriptor.stored_bytes, descriptor.padded_bytes)
        bad_padding = bytearray(encoded)
        padding_at = decoder.data_offset + descriptor.offset_bytes + descriptor.stored_bytes
        bad_padding[padding_at] = 1
        with self.assertRaises(codec.CodecError):
            codec.MatrixDecoder(bad_padding)

        trailing = encoded + b"\0"
        with self.assertRaises(codec.CodecError):
            codec.MatrixDecoder(trailing)

    def test_palette_container_corruption(self) -> None:
        raw = bytes([1, 2] * (16 * 64 // 2))
        encoded = codec.encode_matrix(
            raw, 16, 64, 1024, enabled_modes=(codec.Mode.BYTE_PALETTE4,)
        )
        decoder = codec.MatrixDecoder(encoded)
        descriptor = decoder.descriptors[0]
        self.assertEqual(descriptor.mode, codec.Mode.BYTE_PALETTE4)
        self.assertGreaterEqual(descriptor.parameter, 2)
        corrupt = bytearray(encoded)
        begin = decoder.data_offset + descriptor.offset_bytes
        corrupt[begin + 1] = corrupt[begin]
        with self.assertRaises(codec.CodecError):
            codec.MatrixDecoder(corrupt)


class PropertyFuzzTest(unittest.TestCase):
    def test_random_and_adversarial_properties(self) -> None:
        rng = random.Random(0x5A10F8)
        modes = list(codec.Mode)
        for case in range(600):
            count = rng.choice(codec.ALLOWED_TILE_BYTES)
            selector = case % 6
            if selector == 0:
                raw = bytes(rng.randrange(256) for _ in range(count))
            elif selector == 1:
                alphabet = [rng.randrange(256) for _ in range(rng.randrange(1, 17))]
                raw = bytes(rng.choice(alphabet) for _ in range(count))
            elif selector == 2:
                raw = bytes(rng.choice((0x00, 0x80, 0x7F, 0xFF)) for _ in range(count))
            elif selector == 3:
                raw = bytes((rng.randrange(16) << 3) | rng.randrange(8) for _ in range(count))
            elif selector == 4:
                base = rng.randrange(128)
                raw = bytes((base ^ rng.randrange(15)) | (rng.randrange(2) << 7) for _ in range(count))
            else:
                raw = bytes([rng.randrange(256)] * count)

            chosen = codec.encode_tile(raw)
            self.assertEqual(
                codec.decode_mode(chosen.mode, chosen.parameter, chosen.payload, count), raw
            )
            forced = codec.encode_mode(rng.choice(modes), raw)
            self.assertEqual(
                codec.decode_mode(forced.mode, forced.parameter, forced.payload, count), raw
            )

            rows = rng.randrange(1, 34)
            cols = rng.choice((64, 128, 256))
            matrix = bytes(rng.randrange(256) for _ in range(rows * cols))
            tile_bytes = rng.choice(codec.ALLOWED_TILE_BYTES)
            container = codec.encode_matrix(matrix, rows, cols, tile_bytes)
            self.assertEqual(codec.decode_matrix(container), matrix)


class CommandLineTest(unittest.TestCase):
    def test_encode_inspect_decode(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            root = pathlib.Path(text)
            rows, cols = 19, 128
            raw = bytes((index * 29 + index // 7) & 0xFF for index in range(rows * cols))
            source = root / "weights.bin"
            encoded = root / "weights.if8"
            decoded = root / "decoded.bin"
            encode_report = root / "encode.json"
            inspect_report = root / "inspect.json"
            decode_report = root / "decode.json"
            source.write_bytes(raw)
            self.assertEqual(
                codec.main(
                    [
                        "encode", str(source), str(encoded), "--rows", str(rows),
                        "--cols", str(cols), "--tile-bytes", "1024",
                        "--report-json", str(encode_report),
                    ]
                ),
                0,
            )
            self.assertEqual(
                codec.main(["inspect", str(encoded), "--report-json", str(inspect_report)]),
                0,
            )
            self.assertEqual(
                codec.main(
                    ["decode", str(encoded), str(decoded), "--report-json", str(decode_report)]
                ),
                0,
            )
            self.assertEqual(decoded.read_bytes(), raw)
            encode_json = json.loads(encode_report.read_text())
            inspect_json = json.loads(inspect_report.read_text())
            decode_json = json.loads(decode_report.read_text())
            self.assertEqual(encode_json["source_sha256"], decode_json["decoded_sha256"])
            self.assertEqual(encode_json["encoded_sha256"], inspect_json["encoded_sha256"])
            self.assertEqual(inspect_json["rows"], rows)
            self.assertEqual(inspect_json["cols"], cols)

    def test_require_saving_rejects_incompressible_container(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            root = pathlib.Path(text)
            raw = bytes(range(256)) * 4
            source = root / "raw.bin"
            output = root / "raw.if8"
            source.write_bytes(raw)
            with self.assertRaises(ValueError):
                codec.main(
                    [
                        "encode", str(source), str(output), "--rows", "16", "--cols", "64",
                        "--require-saving",
                    ]
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
