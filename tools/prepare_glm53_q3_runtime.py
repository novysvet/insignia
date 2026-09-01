#!/usr/bin/env python3
"""Build Insignia runtime metadata and a dense FP8 cache from Q3_K_XL GGUF.

Routed experts remain in their original IQ3_XXS/IQ4_XS/Q6_K tensors and are
referenced in-place by the generated IGLM index.  Only dense/shared linears are
dequantized and encoded into the engine's group-64 E4M3 cache.  Small tensors
needed verbatim by the recurrent/attention code are copied to one compact
metadata shard; FP32 norm vectors are rounded once to BF16 because the current
RMS/mHC kernels consume BF16 weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import dequantize, quantize

from quantize_dflash2 import encode_e4m3fn


ALIGNMENT = 4096
GROUP = 64
INDEX_MAGIC = b"IGLMIDX1"
FP8_MAGIC = b"IGLMF8A1"
META_NAME = "glm53-q3-meta.bin"
INDEX_NAME = "glm53-q3.index"
FP8_PREFIX = "glm53-q3-fp8-g64"

TYPE_F32 = 1
TYPE_BF16 = 2
TYPE_EXTERNAL_FP8 = 8
TYPE_IQ3_XXS = 9
TYPE_IQ4_XS = 10
TYPE_Q6_K = 11

QTYPE_TO_INDEX = {
    GGMLQuantizationType.IQ3_XXS: TYPE_IQ3_XXS,
    GGMLQuantizationType.IQ4_XS: TYPE_IQ4_XS,
    GGMLQuantizationType.Q6_K: TYPE_Q6_K,
}


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) & -ALIGNMENT


def layer_name(layer: int, suffix: str) -> str:
    return f"model.language_model.layers.{layer}.{suffix}"


@dataclass
class SourceTensor:
    shard: int
    tensor: object

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(reversed([int(value) for value in self.tensor.shape]))


@dataclass
class IndexEntry:
    name: str
    dtype: int
    shape: tuple[int, ...]
    shard: int
    offset: int
    length: int


@dataclass
class DenseRecord:
    name: str
    rows: int
    cols: int
    source: SourceTensor | None = None
    k_source: SourceTensor | None = None
    v_source: SourceTensor | None = None
    weight_offset: int = 0
    weight_bytes: int = 0
    scale_offset: int = 0
    scale_bytes: int = 0


def mapped_linear(layer: int, suffix: str) -> str | None:
    attention = {
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "ssm_beta.weight": "self_attn.b_proj.weight",
        "ssm_f_a.weight": "self_attn.f_a_proj.weight",
        "ssm_f_b.weight": "self_attn.f_b_proj.weight",
        "ssm_g_a.weight": "self_attn.g_a_proj.weight",
        "ssm_g_b.weight": "self_attn.g_b_proj.weight",
        "attn_q_a.weight": "self_attn.q_a_proj.weight",
        "attn_q_b.weight": "self_attn.q_b_proj.weight",
        "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    }
    if suffix in attention:
        return layer_name(layer, attention[suffix])
    ffn = {
        "ffn_down.weight": "mlp.down_proj.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down_shexp.weight": "mlp.shared_experts.down_proj.weight",
        "ffn_gate_shexp.weight": "mlp.shared_experts.gate_proj.weight",
        "ffn_up_shexp.weight": "mlp.shared_experts.up_proj.weight",
        "ffn_gate_inp.weight": "mlp.gate.weight",
    }
    return layer_name(layer, ffn[suffix]) if suffix in ffn else None


def mapped_actual(layer: int, suffix: str) -> tuple[str, str] | None:
    # value is (target name, action): raw preserves GGUF bytes; norm converts
    # the FP32 vector to the BF16 ABI consumed by existing kernels.
    mapping = {
        "attn_norm.weight": ("input_layernorm.weight", "norm"),
        "ffn_norm.weight": ("post_attention_layernorm.weight", "norm"),
        "attn_q_a_norm.weight": ("self_attn.q_a_layernorm.weight", "norm"),
        "attn_kv_a_norm.weight": ("self_attn.kv_a_layernorm.weight", "norm"),
        "ssm_norm.weight": ("self_attn.o_norm.weight", "norm"),
        "ssm_a": ("self_attn.A_log", "raw"),
        "ssm_dt.bias": ("self_attn.dt_bias", "raw"),
        "ssm_conv1d_q.weight": ("self_attn.q_conv1d.weight", "raw"),
        "ssm_conv1d_k.weight": ("self_attn.k_conv1d.weight", "raw"),
        "ssm_conv1d_v.weight": ("self_attn.v_conv1d.weight", "raw"),
        "exp_probs_b.bias": ("mlp.gate.e_score_correction_bias", "raw"),
        "hc_attn_base.weight": ("hc_attn_base", "raw"),
        "hc_attn_scale.weight": ("hc_attn_scale", "raw"),
        "hc_attn_fn.weight": ("hc_attn_fn", "raw"),
        "hc_ffn_base.weight": ("hc_ffn_base", "raw"),
        "hc_ffn_scale.weight": ("hc_ffn_scale", "raw"),
        "hc_ffn_fn.weight": ("hc_ffn_fn", "raw"),
    }
    if suffix not in mapping:
        return None
    target, action = mapping[suffix]
    return layer_name(layer, target), action


def load_sources(paths: list[Path]) -> tuple[list[GGUFReader], dict[str, SourceTensor]]:
    readers: list[GGUFReader] = []
    tensors: dict[str, SourceTensor] = {}
    for shard, path in enumerate(paths):
        reader = GGUFReader(path, "r")
        readers.append(reader)
        for tensor in reader.tensors:
            if tensor.name in tensors:
                raise RuntimeError(f"duplicate split tensor: {tensor.name}")
            tensors[tensor.name] = SourceTensor(shard, tensor)
    return readers, tensors


def source_values(source: SourceTensor, begin: int, count: int) -> np.ndarray:
    data = source.tensor.data
    if len(data.shape) != 2:
        raise RuntimeError(f"ordinary dense tensor is not rank two: {source.tensor.name}")
    return np.ascontiguousarray(
        dequantize(data[begin : begin + count], source.tensor.tensor_type),
        dtype=np.float32,
    )


def bf16_bytes(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(
        quantize(np.ascontiguousarray(values, dtype=np.float32),
                 GGMLQuantizationType.BF16)
    ).tobytes()


def raw_bytes(source: SourceTensor) -> bytes:
    return np.ascontiguousarray(source.tensor.data).view(np.uint8).tobytes()


def plan(
    tensors: dict[str, SourceTensor], meta_shard: int
) -> tuple[list[IndexEntry], list[DenseRecord], list[tuple[IndexEntry, SourceTensor, str]], int, list[str]]:
    entries: list[IndexEntry] = []
    dense: list[DenseRecord] = []
    actual: list[tuple[IndexEntry, SourceTensor, str]] = []
    cursor = 0
    layer_types: list[str] = []

    def add_actual(name: str, source: SourceTensor, action: str) -> None:
        nonlocal cursor
        shape = source.shape
        if action == "raw":
            if source.tensor.tensor_type == GGMLQuantizationType.F32:
                dtype, length = TYPE_F32, int(source.tensor.n_bytes)
            elif source.tensor.tensor_type == GGMLQuantizationType.BF16:
                dtype, length = TYPE_BF16, int(source.tensor.n_bytes)
            else:
                raise RuntimeError(f"unsupported raw metadata type for {source.tensor.name}")
        elif action in ("norm", "embed"):
            dtype, length = TYPE_BF16, math.prod(shape) * 2
        else:
            raise AssertionError(action)
        cursor = align(cursor)
        entry = IndexEntry(name, dtype, shape, meta_shard, cursor, length)
        entries.append(entry)
        actual.append((entry, source, action))
        cursor += length

    add_actual("model.language_model.embed_tokens.weight", tensors["token_embd.weight"], "embed")
    add_actual("model.language_model.norm.weight", tensors["output_norm.weight"], "norm")
    output = tensors["output.weight"]
    rows, cols = output.shape
    dense.append(DenseRecord("lm_head.weight", rows, cols, source=output))

    for layer in range(45):
        prefix = f"blk.{layer}."
        is_mla = prefix + "attn_q_a.weight" in tensors
        layer_types.append("full_attention" if is_mla else "linear_attention")
        for suffix in (
            "attn_norm.weight", "ffn_norm.weight",
            "hc_attn_base.weight", "hc_attn_scale.weight", "hc_attn_fn.weight",
            "hc_ffn_base.weight", "hc_ffn_scale.weight", "hc_ffn_fn.weight",
            "ssm_a", "ssm_dt.bias", "ssm_conv1d_q.weight",
            "ssm_conv1d_k.weight", "ssm_conv1d_v.weight", "ssm_norm.weight",
            "exp_probs_b.bias", "attn_q_a_norm.weight", "attn_kv_a_norm.weight",
        ):
            source = tensors.get(prefix + suffix)
            if source is None:
                continue
            target = mapped_actual(layer, suffix)
            if target is None:
                raise AssertionError(suffix)
            add_actual(target[0], source, target[1])

        linear_suffixes = (
            "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
            "ssm_beta.weight", "ssm_f_a.weight", "ssm_f_b.weight",
            "ssm_g_a.weight", "ssm_g_b.weight", "attn_q_a.weight",
            "attn_q_b.weight", "attn_kv_a_mqa.weight",
            "ffn_down.weight", "ffn_gate.weight", "ffn_up.weight",
            "ffn_down_shexp.weight", "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight", "ffn_gate_inp.weight",
        )
        for suffix in linear_suffixes:
            source = tensors.get(prefix + suffix)
            if source is None:
                continue
            target = mapped_linear(layer, suffix)
            if target is None:
                raise AssertionError(suffix)
            if len(source.shape) != 2:
                raise RuntimeError(f"linear is not rank two: {source.tensor.name}")
            rows, cols = source.shape
            if cols % GROUP:
                raise RuntimeError(f"linear width is not group-64: {source.tensor.name}")
            dense.append(DenseRecord(target, rows, cols, source=source))

        if is_mla:
            k_source = tensors[prefix + "attn_k_b.weight"]
            v_source = tensors[prefix + "attn_v_b.weight"]
            heads, latent, qk = k_source.shape
            v_heads, value, v_latent = v_source.shape
            if (heads, latent) != (v_heads, v_latent) or qk != value:
                raise RuntimeError(f"incompatible MLA K/V split in layer {layer}")
            dense.append(DenseRecord(
                layer_name(layer, "self_attn.kv_b_proj.weight"),
                heads * (qk + value), latent, k_source=k_source, v_source=v_source,
            ))

        if layer >= 3:
            for projection, suffix in (
                ("down_proj", "ffn_down_exps.weight"),
                ("gate_proj", "ffn_gate_exps.weight"),
                ("up_proj", "ffn_up_exps.weight"),
            ):
                source = tensors[prefix + suffix]
                try:
                    dtype = QTYPE_TO_INDEX[source.tensor.tensor_type]
                except KeyError as exc:
                    raise RuntimeError(
                        f"unsupported routed format {source.tensor.tensor_type.name} "
                        f"in {source.tensor.name}"
                    ) from exc
                entries.append(IndexEntry(
                    layer_name(layer, f"mlp.q3_experts.{projection}.weight"),
                    dtype, source.shape, source.shard,
                    int(source.tensor.data_offset), int(source.tensor.n_bytes),
                ))

    # Every dense matrix is represented by a geometry-only external entry.
    # The engine refuses to execute it unless the generated FP8 index has an
    # exact matching record, so a missing cache cannot silently read garbage.
    for record in dense:
        entries.append(IndexEntry(
            record.name, TYPE_EXTERNAL_FP8, (record.rows, record.cols),
            meta_shard, 0, 0,
        ))

    dense.sort(key=lambda record: record.name)
    fp8_cursor = 0
    for record in dense:
        record.weight_offset = align(fp8_cursor)
        record.weight_bytes = record.rows * record.cols
        record.scale_offset = align(record.weight_offset + record.weight_bytes)
        record.scale_bytes = record.rows * (record.cols // GROUP) * 2
        fp8_cursor = align(record.scale_offset + record.scale_bytes)
    entries.sort(key=lambda entry: entry.name)
    if len({entry.name for entry in entries}) != len(entries):
        raise RuntimeError("generated duplicate target tensor name")
    return entries, dense, actual, align(cursor), layer_types


def encode_fp8_rows(values: np.ndarray) -> tuple[bytes, bytes]:
    rows, cols = values.shape
    groups = cols // GROUP
    grouped = values.reshape(rows, groups, GROUP)
    maximum = np.max(np.abs(grouped), axis=2)
    scales = maximum * np.float32(1.0 / 448.0)
    divisors = np.where(scales > 0, scales, np.float32(1.0))
    normalized = np.ascontiguousarray(grouped / divisors[:, :, None])
    np.clip(normalized, -448.0, 448.0, out=normalized)
    encoded = encode_e4m3fn(normalized)
    if np.any((encoded & 0x7f) == 0x7f):
        raise RuntimeError("FP8 encoder emitted NaN")
    return np.ascontiguousarray(encoded).tobytes(), np.ascontiguousarray(scales.astype("<f2")).tobytes()


def write_meta(path: Path, actual: list[tuple[IndexEntry, SourceTensor, str]], size: int) -> None:
    temporary = Path(str(path) + ".tmp")
    descriptor = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(descriptor, size)
    try:
        for ordinal, (entry, source, action) in enumerate(actual, 1):
            if action == "raw":
                payload = raw_bytes(source)
                if len(payload) != entry.length:
                    raise RuntimeError(f"raw size mismatch for {entry.name}")
                os.pwrite(descriptor, payload, entry.offset)
            elif action == "norm":
                values = dequantize(source.tensor.data, source.tensor.tensor_type)
                payload = bf16_bytes(values)
                if len(payload) != entry.length:
                    raise RuntimeError(f"norm size mismatch for {entry.name}")
                os.pwrite(descriptor, payload, entry.offset)
            elif action == "embed":
                rows, cols = source.shape
                rows_per_chunk = max(1, (16 << 20) // (cols * 4))
                for row in range(0, rows, rows_per_chunk):
                    count = min(rows_per_chunk, rows - row)
                    payload = bf16_bytes(source_values(source, row, count))
                    os.pwrite(descriptor, payload, entry.offset + row * cols * 2)
            else:
                raise AssertionError(action)
            if ordinal == len(actual) or ordinal % 64 == 0:
                print(f"metadata {ordinal}/{len(actual)} {entry.name}", flush=True)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    temporary.replace(path)


def write_dense_cache(path: Path, records: list[DenseRecord], size: int) -> None:
    temporary = Path(str(path) + ".tmp")
    descriptor = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(descriptor, size)
    written = 0
    started = time.monotonic()
    try:
        for ordinal, record in enumerate(records, 1):
            if record.source is not None:
                rows_per_chunk = max(1, (16 << 20) // (record.cols * 4))
                for row in range(0, record.rows, rows_per_chunk):
                    count = min(rows_per_chunk, record.rows - row)
                    weights, scales = encode_fp8_rows(source_values(record.source, row, count))
                    os.pwrite(descriptor, weights, record.weight_offset + row * record.cols)
                    os.pwrite(descriptor, scales,
                              record.scale_offset + row * (record.cols // GROUP) * 2)
            else:
                assert record.k_source is not None and record.v_source is not None
                k = np.ascontiguousarray(
                    dequantize(record.k_source.tensor.data, record.k_source.tensor.tensor_type),
                    dtype=np.float32,
                )
                v = np.ascontiguousarray(
                    dequantize(record.v_source.tensor.data, record.v_source.tensor.tensor_type),
                    dtype=np.float32,
                )
                heads, latent, qk = k.shape
                for head in range(heads):
                    # GGUF stores K as [head,latent,qk] after conversion's
                    # transpose, and V as [head,value,latent]. Restore the
                    # source checkpoint's per-head [K rows,V rows] ordering.
                    values = np.concatenate((k[head].T, v[head]), axis=0)
                    weights, scales = encode_fp8_rows(values)
                    row = head * values.shape[0]
                    os.pwrite(descriptor, weights, record.weight_offset + row * record.cols)
                    os.pwrite(descriptor, scales,
                              record.scale_offset + row * (record.cols // GROUP) * 2)
                del k, v
            written += record.weight_bytes + record.scale_bytes
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"dense {ordinal:3d}/{len(records)} {written / 2**30:6.2f} GiB "
                f"{written / elapsed / 1e9:5.2f} GB/s {record.name}",
                flush=True,
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    temporary.replace(path)


def write_q8_index(path: Path, records: Iterable[DenseRecord], data_bytes: int) -> None:
    temporary = Path(str(path) + ".tmp")
    records = list(records)
    with temporary.open("wb") as output:
        output.write(struct.pack("<8sIIIQ", FP8_MAGIC, 1, GROUP, len(records), data_bytes))
        for record in records:
            name = record.name.encode()
            output.write(struct.pack(
                "<HIIQQQQ", len(name), record.rows, record.cols,
                record.weight_offset, record.weight_bytes,
                record.scale_offset, record.scale_bytes,
            ))
            output.write(name)
    temporary.replace(path)


def write_model_index(
    path: Path, shard_paths: list[Path], entries: list[IndexEntry]
) -> None:
    temporary = Path(str(path) + ".tmp")
    payload = sum(entry.length for entry in entries)
    with temporary.open("wb") as output:
        output.write(struct.pack(
            "<8s11IQ", INDEX_MAGIC, 1, 1, len(shard_paths), len(entries),
            4096, 45, 154880, 288, 8, 2048, 4, payload,
        ))
        for shard in shard_paths:
            encoded = shard.name.encode()
            output.write(struct.pack("<HQ", len(encoded), shard.stat().st_size))
            output.write(encoded)
        for entry in entries:
            encoded = entry.name.encode()
            output.write(struct.pack(
                "<HBBHHQQ", len(encoded), entry.dtype, len(entry.shape),
                entry.shard, 0, entry.offset, entry.length,
            ))
            output.write(encoded)
            output.write(struct.pack("<" + "I" * len(entry.shape), *entry.shape))
    temporary.replace(path)


def write_config(path: Path, layer_types: list[str]) -> None:
    payload = {
        "num_attention_heads": 64,
        "layer_types": layer_types,
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
    }
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.gguf_dir.glob("*UD-Q3_K_XL-*-of-00004.gguf"))
    if len(paths) != 4:
        raise RuntimeError(f"expected four Q3_K_XL GGUF shards, found {len(paths)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    readers, tensors = load_sources(paths)
    meta_path = args.output_dir / META_NAME
    shard_paths = paths + [meta_path]
    entries, dense, actual, meta_bytes, layer_types = plan(tensors, len(paths))
    dense_bytes = align(dense[-1].scale_offset + dense[-1].scale_bytes)
    expert_bytes = sum(
        entry.length for entry in entries
        if entry.dtype in (TYPE_IQ3_XXS, TYPE_IQ4_XS, TYPE_Q6_K)
    )
    print(
        f"plan: {len(entries)} model entries, {len(dense)} dense FP8 matrices, "
        f"metadata={meta_bytes / 2**30:.3f} GiB, dense={dense_bytes / 2**30:.3f} GiB, "
        f"in-place experts={expert_bytes / 2**30:.3f} GiB",
        flush=True,
    )
    if args.plan:
        return
    free = os.statvfs(args.output_dir)
    free_bytes = free.f_bavail * free.f_frsize
    required = meta_bytes + dense_bytes + (8 << 30)
    if free_bytes < required:
        raise RuntimeError(
            f"need {required / 2**30:.1f} GiB including headroom, "
            f"only {free_bytes / 2**30:.1f} GiB free"
        )
    # The IGLM index resolves every shard relative to MODEL_ROOT.  Keep the
    # 137 GiB GGUF single-copy by placing four relative symlinks beside the
    # generated metadata/cache instead of copying the source payload.
    for source in paths:
        link = args.output_dir / source.name
        if link.exists() or link.is_symlink():
            if link.resolve() != source.resolve():
                raise RuntimeError(f"wrong existing GGUF link: {link}")
        else:
            link.symlink_to(source.resolve())
    write_meta(meta_path, actual, meta_bytes)
    write_dense_cache(args.output_dir / f"{FP8_PREFIX}.bin", dense, dense_bytes)
    write_q8_index(args.output_dir / f"{FP8_PREFIX}.index", dense, dense_bytes)
    write_model_index(args.output_dir / INDEX_NAME, shard_paths, entries)
    write_config(args.output_dir / "config.json", layer_types)
    del readers  # retain all GGUF memmaps until every conversion has finished
    print(f"Q3 runtime ready under {args.output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
