#!/usr/bin/env python3
"""Fail-closed validation for Insignia's second-drive mount.

The stripe VHDX lives on Windows E:, but is consumed as an ext4 block device
inside WSL.  A plain `/stripe` directory is therefore dangerous: writes to it
land in the C:-hosted root VHDX.  This module deliberately rejects anything
other than an exact, distinct mount point.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import shutil
import subprocess
from typing import Iterable


class StripeMountError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class MountEntry:
    major_minor: str
    mount_point: pathlib.Path
    fs_type: str
    source: str


@dataclasses.dataclass(frozen=True)
class StripeMount:
    source_root: pathlib.Path
    destination: pathlib.Path
    device: str
    fs_type: str
    free_bytes: int


def _unescape_mount_field(value: str) -> str:
    # mountinfo uses octal escapes for whitespace and backslash.
    for encoded, decoded in (("\\040", " "), ("\\011", "\t"),
                             ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(encoded, decoded)
    return value


def parse_mountinfo(payload: str) -> list[MountEntry]:
    entries: list[MountEntry] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        left, separator, right = line.partition(" - ")
        if not separator:
            raise StripeMountError(f"malformed mountinfo line {line_number}")
        lhs = left.split()
        rhs = right.split()
        if len(lhs) < 5 or len(rhs) < 2:
            raise StripeMountError(f"truncated mountinfo line {line_number}")
        entries.append(MountEntry(
            major_minor=lhs[2],
            mount_point=pathlib.Path(_unescape_mount_field(lhs[4])),
            fs_type=rhs[0],
            source=_unescape_mount_field(rhs[1]),
        ))
    return entries


def read_mountinfo() -> list[MountEntry]:
    try:
        return parse_mountinfo(pathlib.Path("/proc/self/mountinfo").read_text())
    except OSError as error:
        raise StripeMountError(f"cannot read /proc/self/mountinfo: {error}") from error


def _containing_mount(path: pathlib.Path, entries: Iterable[MountEntry]) -> MountEntry | None:
    best: MountEntry | None = None
    for entry in entries:
        try:
            path.relative_to(entry.mount_point)
        except ValueError:
            continue
        if best is None or len(entry.mount_point.parts) > len(best.mount_point.parts):
            best = entry
    return best


def _blkid_device(option: str, value: str) -> str:
    try:
        result = subprocess.run(
            ["blkid", option, value], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise StripeMountError(f"cannot resolve block device for {value!r}: {error}") from error
    device = result.stdout.strip()
    if not device:
        raise StripeMountError(f"blkid returned no block device for {value!r}")
    return device


def _unique_label_device(label: str) -> str:
    try:
        result = subprocess.run(
            ["blkid", "-t", f"LABEL={label}", "-o", "device"],
            check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise StripeMountError(f"cannot resolve block device label {label!r}: {error}") from error
    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(devices) != 1:
        raise StripeMountError(
            f"block label {label!r} resolves to {len(devices)} devices; expected exactly one"
        )
    return devices[0]


def validate_stripe_mount(
    source_root: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
    *,
    expected_label: str | None = "stripe",
    expected_uuid: str | None = None,
    expected_fs: str = "ext4",
    min_free_bytes: int = 0,
    mount_entries: list[MountEntry] | None = None,
) -> StripeMount:
    source = pathlib.Path(source_root)
    target = pathlib.Path(destination)
    if not source.is_dir():
        raise StripeMountError(f"source root is not a directory: {source}")
    if target.is_symlink():
        raise StripeMountError(f"stripe destination must not be a symlink: {target}")
    if not target.is_dir():
        raise StripeMountError(
            f"stripe destination does not exist; refusing to create a mount-point directory: {target}"
        )

    source = source.resolve(strict=True)
    target = target.resolve(strict=True)
    entries = read_mountinfo() if mount_entries is None else mount_entries
    source_mount = _containing_mount(source, entries)
    target_mount = _containing_mount(target, entries)
    if source_mount is None or target_mount is None:
        raise StripeMountError("source or destination is absent from /proc/self/mountinfo")
    if target_mount.mount_point != target:
        raise StripeMountError(
            f"{target} is not an exact mount point (it resolves inside {target_mount.mount_point}); "
            "the E: stripe VHDX is probably detached"
        )
    if source_mount.major_minor == target_mount.major_minor:
        raise StripeMountError(
            f"stripe destination and source share device {target_mount.major_minor}; "
            "refusing to write both tiers to one physical mount"
        )
    if os.stat(source).st_dev == os.stat(target).st_dev:
        raise StripeMountError("stripe destination and source have the same st_dev")
    if expected_fs and target_mount.fs_type != expected_fs:
        raise StripeMountError(
            f"stripe filesystem is {target_mount.fs_type}, expected {expected_fs}"
        )

    expected_devices: list[str] = []
    if expected_label:
        expected_devices.append(_unique_label_device(expected_label))
    if expected_uuid:
        expected_devices.append(_blkid_device("-U", expected_uuid))
    mounted_device = os.path.realpath(target_mount.source)
    for expected in expected_devices:
        if os.path.realpath(expected) != mounted_device:
            raise StripeMountError(
                f"{target} is mounted from {target_mount.source}, expected {expected}"
            )

    free = shutil.disk_usage(target).free
    if free < min_free_bytes:
        raise StripeMountError(
            f"stripe mount has {free / 2**30:.2f} GiB free; "
            f"at least {min_free_bytes / 2**30:.2f} GiB is required"
        )
    return StripeMount(source, target, target_mount.source, target_mount.fs_type, free)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=pathlib.Path)
    parser.add_argument("destination", type=pathlib.Path)
    parser.add_argument("--expected-label", default="stripe")
    parser.add_argument("--expected-uuid")
    parser.add_argument("--expected-fs", default="ext4")
    parser.add_argument("--min-free-gib", type=float, default=0.0)
    args = parser.parse_args()
    try:
        result = validate_stripe_mount(
            args.source_root, args.destination,
            expected_label=args.expected_label or None,
            expected_uuid=args.expected_uuid,
            expected_fs=args.expected_fs,
            min_free_bytes=max(0, int(args.min_free_gib * 2**30)),
        )
    except StripeMountError as error:
        parser.exit(1, f"stripe mount check FAILED: {error}\n")
    print(
        f"stripe mount OK: {result.destination} <- {result.device} "
        f"({result.fs_type}, {result.free_bytes / 2**30:.1f} GiB free)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
