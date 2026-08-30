#!/usr/bin/env python3
"""Structural check for the deterministic native parity fixture."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from export_falsifier_vnni import fnv1a
from export_falsifier_vnni_fixture import (
    HEADER,
    MAGIC,
    VERSION,
    export_fixture,
)


def test_fixture() -> None:
    events = 2
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "core.ifvfix"
        report = export_fixture(None, path, events, 5303)
        data = path.read_bytes()
        magic, version, event_count, checksum, _ = HEADER.unpack_from(data)
        assert magic == MAGIC
        assert version == VERSION
        assert event_count == events == report["events"]
        payload = data[HEADER.size:]
        expected_bytes = events * (4 * 192 + 192 + 107 + 3 * 2) * 4
        assert len(payload) == expected_bytes
        assert fnv1a(payload) == checksum == report["payload_checksum"]
        floats = np.frombuffer(
            payload[:events * (4 * 192 + 192 + 107) * 4], dtype="<f4")
        routes = np.frombuffer(payload[floats.nbytes:], dtype="<i4")
        assert np.isfinite(floats).all()
        assert routes.shape == (events * 3 * 2,)
        assert ((routes >= 0) & (routes < 256)).all()


if __name__ == "__main__":
    test_fixture()
    print("falsifier VNNI fixture tests passed")
