# Session 13: UD-Q3_K_XL format and Ada kernel research

Date: 2026-08-31  
Scope: byte-level format research only; no engine code and no local-GPU measurements  
Artifact: `AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF`, revision
`0359efd18cfd7794b2faded6510452e0f9120ef4`, directory `UD-Q3_K_XL`

## Result in one sentence

The file called **UD-Q3_K_XL is not a Q3_K model in the kernel-dispatch sense**:
the live 42-layer MoE uses `IQ3_XXS` for almost every expert gate/up matrix and
`IQ4_XS` for almost every expert down matrix; literal `Q3_K` occurs only in the
parked layer-45 MTP gate/up tensors. Therefore the first useful kernels are
`IQ3_XXS x Q8_1` and `IQ4_XS x Q8_1`, not a port of the existing NVFP4 decoder
and not a `Q3_K` bit-plane kernel.

## Primary-source pins and inspection method

- The four files and their sizes came from the [official Hugging Face repository
  API](https://huggingface.co/api/models/AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF?blobs=true)
  and were pinned to the repository revision above. The pinned directory is
  [here](https://huggingface.co/AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF/tree/0359efd18cfd7794b2faded6510452e0f9120ef4/UD-Q3_K_XL).
- Shard 4 was fully present on `glm-box-wsl`. For shards 1--3 I fetched bytes
  `0..16777215` with an HTTP Range request against the pinned Hugging Face
  objects, placed each header in a sparse file of the declared LFS size, and
  parsed it on `glm-box-wsl`. This reads the real GGUF header without fetching
  the tensor payload. The GGUF tensor descriptors precede the aligned tensor
  data blob by specification; see [gguf.h lines 7--30](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/include/gguf.h#L7-L30).
- Parsing used the official `llama.cpp/gguf-py` reader from local reference
  revision `192067b72d1b7a3653b3f0c59190303b18596637`. Header data offsets were
  9,460,576, 29,472, 27,616, and 1,248 bytes for shards 1--4. All used the
  default 32-byte GGUF alignment.
- Format structs and CUDA paths below are from the same clean official
  `llama.cpp` revision. The separate official `ggml` reference at
  `36da57138425487184aa1da2eee2cde155909c6f` agrees.

The four LFS objects total **147,288,966,080 bytes (137.174 GiB)**. Tensor
payload is 147,279,445,368 bytes; headers and per-tensor padding account for the
remaining 9.08 MiB.

## Exact per-tensor type inventory

The scan found exactly the advertised `split.tensors.count = 1412` descriptors.
`general.architecture` is `glm5next`, `general.quantization_version` is 2, and
`general.file_type` is 7. That last value means `MOSTLY_Q8_0` in
[llama.h line 124](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/include/llama.h#L118-L130),
but plainly does **not** describe this dynamic mixture. Dispatch from every
tensor's `ggml_type`; never dispatch from `general.file_type` or the directory
name.

| Actual GGML type | Type id | Tensors | Elements | Payload bytes | Payload share |
|---|---:|---:|---:|---:|---:|
| `IQ3_XXS` | 18 | 82 | 198,105,366,528 | 75,837,210,624 | 51.4921% |
| `IQ4_XS` | 23 | 41 | 99,052,683,264 | 52,621,737,984 | 35.7292% |
| `Q6_K` | 14 | 50 | 9,828,302,848 | 8,062,279,680 | 5.4741% |
| `Q8_0` | 8 | 302 | 6,229,590,016 | 6,618,939,392 | 4.4941% |
| `Q3_K` | 11 | 2 | 4,831,838,208 | 2,076,180,480 | 1.4097% |
| `Q4_K` | 12 | 1 | 2,415,919,104 | 1,358,954,496 | 0.9227% |
| `BF16` | 30 | 296 | 239,337,472 | 478,674,944 | 0.3250% |
| `F32` | 0 | 638 | 56,366,942 | 225,467,768 | 0.1531% |

The stable type ids are defined in [ggml.h lines 394--431](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/include/ggml.h#L394-L431).

### Exact expert assignment

For the live sparse blocks 3--44:

- gate and up are `IQ3_XXS`, except both matrices in block 11 are `IQ4_XS`;
- down is `IQ4_XS`, except blocks 11, 12, and 44, whose down matrices are
  `Q6_K`.

For block 45, the parked MTP layer:

- gate and up are the **only two literal `Q3_K` tensors**;
- down is the only `Q4_K` tensor.

This accounts for all 82 `IQ3_XXS`, 41 `IQ4_XS`, two `Q3_K`, and one `Q4_K`
tensors. The many `Q8_0` tensors are predominantly shared experts and dense
projections; `Q6_K` contains attention outputs, the three high-precision expert
down exceptions, and the output matrix.

### Expert-record byte consequences

GGUF stores the quantization dimension as `ne[0]`. The parser rejects a tensor
whose `ne[0]` is not divisible by its block size and derives row stride directly
from `type_size * ne[0]/block_size`; see
[gguf.cpp lines 710--747](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/gguf.cpp#L710-L747).
All expert rows and every tensor offset in this artifact are 32-byte aligned.

| Matrix/type | Shape | Bytes/row | Bytes/expert |
|---|---:|---:|---:|
| gate or up, `IQ3_XXS` | `4096 x 2048 x 288` | 1,568 | 3,211,264 (3.0625 MiB) |
| down, `IQ4_XS` | `2048 x 4096 x 288` | 1,088 | 4,456,448 (4.25 MiB) |
| gate or up, `IQ4_XS` exception | `4096 x 2048 x 288` | 2,176 | 4,456,448 (4.25 MiB) |
| down, `Q6_K` exception | `2048 x 4096 x 288` | 1,680 | 6,881,280 (6.5625 MiB) |
| gate or up, MTP `Q3_K` | `4096 x 2048 x 288` | 1,760 | 3,604,480 (3.4375 MiB) |
| down, MTP `Q4_K` | `2048 x 4096 x 288` | 1,152 | 4,718,592 (4.5 MiB) |

A normal main-model routed expert is therefore **10.375 MiB** across
gate/up/down. Block 11 is 15.0625 MiB; blocks 12 and 44 are 12.6875 MiB. With
top-8 routing over all 42 sparse layers, the exact routed-expert payload is
**3,560.5 MiB (3.477 GiB) per token if nothing hits cache**, excluding shared
experts and non-MoE matrices. This is the correct byte budget for a Q3_K_XL
expert sidecar and cache simulation.

Tensor-major GGUF makes each individual expert slice contiguous inside a
matrix, but its gate/up/down slices live in three distant tensors. A streaming
Insignia store should therefore repack those three slices into one typed expert
record, retaining a per-component type/offset table. Treating every record as
one homogeneous "Q3" ABI would corrupt block 11, 12, and 44 immediately.

## The actual hot format: IQ3_XXS

`QK_K = 256`. The official ABI is
[ggml-common.h lines 404--411](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-common.h#L404-L411):

```c
typedef struct {
    ggml_half d;       // byte 0
    uint8_t qs[96];    // byte 2
} block_iq3_xxs;       // 98 bytes / 256 = 3.0625 bpw
```

This is **vector-quantized, not a packed linear three-bit integer**:

- `qs[0..63]` holds 64 byte indices into the 256-entry `iq3xxs_grid`.
  Every index selects four positive byte magnitudes; two indices describe eight
  weights.
- `qs[64..95]` holds eight little-endian 32-bit `scale_and_signs` words, one
  per 32 weights. Bits 0--27 are four 7-bit sign codes; bits 28--31 are a
  4-bit scale code. The eighth sign bit is parity-constrained and reconstructed
  with popcount.
- For a 32-weight subblock with scale nibble `s`, its scalar multiplier is
  `fp16(d) * (0.5 + s) * 0.5`.
- Each selected codebook magnitude is negated according to the reconstructed
  sign mask.

The exact scalar decode is
[ggml-quants.c lines 2575--2603](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-quants.c#L2575-L2603).
The 1 KiB grid and 128-entry sign mapping are generated directly in
[ggml-common.h](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-common.h#L513-L532)
and [the grid begins here](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-common.h#L1017-L1025).

Kernel consequence: an NVFP4 nibble decoder cannot decode this. The unavoidable
primitive is a random 8-bit index into a 256-entry, four-byte codebook followed
by sign application and group scaling. The cheap part is downstream signed-int8
dot product.

## The actual down format: IQ4_XS

The ABI is
[ggml-common.h lines 447--460](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-common.h#L447-L460):

```c
typedef struct {
    ggml_half d;                  // bytes 0..1
    uint16_t scales_h;            // bytes 2..3
    uint8_t scales_l[4];          // bytes 4..7
    uint8_t qs[128];              // bytes 8..135
} block_iq4_xs;                   // 136 bytes / 256 = 4.25 bpw
```

There are eight 32-weight subblocks. For subblock `b`:

```text
ls = ((scales_l[b/2] >> (4*(b%2))) & 15)
   | (((scales_h >> (2*b)) & 3) << 4)
subblock_scale = fp16(d) * (ls - 32)
```

Its 16 payload bytes carry low nibbles for weights 0--15 and high nibbles for
weights 16--31. A nibble is an index into the fixed nonlinear codebook

```text
[-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113]
```

and the decoded weight is `subblock_scale * codebook[nibble]`. There is no
per-group minimum. The exact scalar implementation is
[ggml-quants.c lines 2743--2763](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-quants.c#L2743-L2763),
and the codebook is defined at
[ggml-common.h lines 1120--1124](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-common.h#L1120-L1124).

This does share a useful *mechanism* with the current NVFP4 path—nibble lookup
into a 16-entry signed-byte table—but not its values, scale representation,
block geometry, or scaling frequency.

## Literal Q3_K, and why it is not the first target

The ABI is
[ggml-common.h lines 311--321](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-common.h#L311-L321):

```c
typedef struct {
    uint8_t hmask[32]; // byte 0: high/sign bit plane
    uint8_t qs[64];    // byte 32: low two bit planes
    uint8_t scales[12];// byte 96: sixteen packed 6-bit scales
    ggml_half d;       // byte 108: superblock scale
} block_q3_K;          // 110 bytes / 256 = 3.4375 bpw
```

For element `i`, the low two bits come from a strided field in `qs`; the bit in
`hmask[i % 32]` selects whether to subtract four. The result is a linear signed
integer in `[-4,3]`. Sixteen signed scale codes in `[-32,31]` are packed across
the 12 scale bytes, and the final value is `fp16(d) * scale * q`. The canonical
pack/decode is
[ggml-quants.c lines 1229--1353](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-quants.c#L1229-L1353).

Note the ABI trap: unlike `IQ3_XXS`, `Q3_K` puts `d` at the **end**. It is also
the only one of these formats that is genuinely a three-bit bit-plane decoder.
Since it is used only by the currently disabled MTP layer, a beautifully tuned
`Q3_K` kernel would not accelerate ordinary GLM decode or prefill.

## Do not conflate the IQ3 names

| Name | What it is | 256-value block | Decoder |
|---|---|---:|---|
| `Q3_K` | GGML tensor type 11 | 110 B | linear 3-bit bit planes + 16 signed scales |
| `IQ3_XXS` | GGML tensor type 18; hot format here | 98 B | 256-entry four-byte grid + parity signs |
| `IQ3_S` | GGML tensor type 21 | 110 B | 512-entry grid, explicit high index bits, explicit signs and scales |
| `IQ3_M` | llama file-level quantization recipe | none | a per-tensor mix, normally based on `IQ3_S`; no `GGML_TYPE_IQ3_M` exists |
| `UD-Q3_K_XL` | artifact/recipe label | none | the exact eight-type mixture in the inventory above |

`IQ3_S`'s separate struct is at
[ggml-common.h lines 413--422](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-common.h#L413-L422).
`IQ3_M` is declared as a file type, not a GGML tensor type, in
[llama.h](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/include/llama.h#L138-L147).
This is the same general rule as UD: inspect the descriptor of each tensor.

## Existing CUDA machinery worth adapting on sm_89

### Decode / very small token batches: packed MMVQ

The official CUDA path already avoids float materialization:

- `vec_dot_iq3_xxs_q8_1` loads two groups of four codebook bytes, reconstructs
  signs with `__popc`, converts signs with packed byte instructions, and performs
  signed `__dp4a` against `Q8_1` activations:
  [vecdotq.cuh lines 1152--1187](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-cuda/vecdotq.cuh#L1152-L1187).
- `vec_dot_iq4_xs_q8_1` uses `__byte_perm` to perform eight nonlinear nibble
  lookups at a time, then `__dp4a`:
  [vecdotq.cuh lines 1338--1362](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-cuda/vecdotq.cuh#L1338-L1362).
- The reusable 16-entry lookup helper is
  [vecdotq.cuh lines 29--92](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-cuda/vecdotq.cuh#L29-L92),
  and parity sign expansion is immediately below it.
- MMVQ dispatches these exact functions in
  [mmvq.cu lines 8--65](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-cuda/mmvq.cu#L8-L65).

Ada supports native DP4A, so this is the correct starting skeleton. Copying the
NVFP4 direct-FP32 nibble accumulation path would throw away the main compute
advantage of these integer-friendly formats.

### Prefill / multiple rows: unpack to shared int8, then integer MMA

Official MMQ does not have a native 3-bit or nonlinear-4-bit tensor-core
instruction. It expands the packed weights to signed int8 shared-memory tiles,
keeps group scales separately, and invokes int8 tensor cores:

- IQ3_XXS tile loader:
  [mmq-load-tiles.cuh lines 1287--1349](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-cuda/mmq-load-tiles.cuh#L1287-L1349)
- IQ4_XS tile loader:
  [mmq-load-tiles.cuh lines 1434--1500](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-cuda/mmq-load-tiles.cuh#L1434-L1500)
- Ada's downstream instruction is `mma.sync.aligned.m16n8k16` or
  `m16n8k32 ... s8.s8.s32`, not FP4 MMA:
  [mma.cuh lines 920--963](https://github.com/ggml-org/llama.cpp/blob/192067b72d1b7a3653b3f0c59190303b18596637/ggml/src/ggml-cuda/mma.cuh#L920-L963)
- Dedicated IQ3_XXS and IQ4_XS MMQ template instances already exist in
  `ggml/src/ggml-cuda/template-instances/`.

That loader/MMA split is the right prefill design to transplant into Insignia.
The upstream tile sizes and MMVQ/MMQ crossover were not tuned for this exact
4070 Ti SUPER, the GLM expert shapes, or Insignia's layer-major batches; they
are starting points, not accepted constants.

## Recommended implementation order

1. **Typed GGUF index and Q3 expert sidecar.** Parse each tensor descriptor,
   validate the exact block size, and repack gate/up/down expert slices into a
   record with three explicit component types. Unit-test byte offsets and a CPU
   scalar decoder before any CUDA timing.
2. **IQ3_XXS decode GEMV.** Quantize/reuse the input as Q8_1 and adapt the
   official codebook + parity-sign + DP4A path. This touches two of the three
   routed matrices and 51.49% of all artifact bytes.
3. **Fuse expert gate and up.** Both normally have identical IQ3 geometry and
   consume the same activation. A paired CTA can reuse Q8 activation loads and
   fuse the SiLU/product boundary, reducing launch/intermediate traffic. Block
   11 needs a separately compiled IQ4_XS/IQ4_XS specialization.
4. **IQ4_XS down GEMV.** Reuse the official `__byte_perm` nonlinear lookup and
   DP4A path. Accept the already gated expert intermediate in Q8_1 form if its
   quality gate passes.
5. **IQ3/IQ4 prefill MMQ.** Unpack to shared signed-int8 tiles and use Ada int8
   MMA. Sweep tiles and token-row crossover on `glm-box-wsl`; do not inherit the
   RTX 4090 thresholds blindly.
6. **Q6_K and Q8_0 exceptions/dense path.** Reuse existing GGML-style kernels or
   convert resident dense matrices to Insignia's accepted compute cache. The
   three Q6 expert-down exceptions require correct dispatch before full-model
   execution, even if they are not the first optimization target.
7. **Literal Q3_K last.** Implement the canonical bit-plane DP4A/MMQ path only
   when block-45 MTP is enabled or a future artifact actually assigns Q3_K to
   live layers.

## High-value experiments and unresolved questions

- **IQ3 codebook placement:** the grid is only 1 KiB but accesses are divergent.
  Compare ordinary device/read-only-cache loads, per-CTA shared staging, and a
  signed 16 KiB `(sign_nibble,index)` table. Constant-memory divergence may
  serialize; do not assume `__constant__` wins.
- **Table-free signs:** preserve the official parity `__popc` reconstruction.
  It already avoids the 128-byte sign table and is a good use of Ada's surplus
  integer compute to save memory traffic.
- **Gate/up fusion geometry:** measure one output row per warp versus paired
  output rows and multiple experts per CTA. Weight traffic dominates, but
  activation reuse, intermediate elimination, and fewer launches can still be
  material.
- **Repack versus on-disk compatibility:** a kernel may consume GGUF bytes
  directly, but an interleaved expert sidecar is much better for one-I/O expert
  misses. Any repack must be byte-decoder-equivalent to the GGUF source.
- **Prefill crossover:** upstream has generic/4090-derived MMVQ-MMQ heuristics.
  Insignia must sweep row count and the exact `4096x2048` / `2048x4096` shapes
  on the remote 4070 Ti SUPER.
- **Parity:** codebook decode equivalence is necessary but not sufficient.
  Integer accumulation grouping and post-scale order can change GLM routing.
  Establish per-block max error, matrix cosine/MSE, logits KL/JS/PPL, and then
  the project's sequence gate before promoting a kernel.
- **Artifact provenance:** the [Alice repository README](https://huggingface.co/AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF/blob/0359efd18cfd7794b2faded6510452e0f9120ef4/README.md)
  points users to an Unsloth `glm5next/upstream` llama.cpp branch, and the
  [official Unsloth model card](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/blob/main/README.md)
  labels its artifacts “Dynamic 3.0.” Neither label is an ABI. The pinned GGUF
  descriptor scan above remains authoritative for this exact download.

## Bottom line for the optimizer

Do not spend the first kernel wave optimizing `block_q3_K`. Build an
`IQ3_XXS` codebook/DP4A gate-up kernel, an `IQ4_XS` nonlinear-nibble/DP4A down
kernel, and their int8-MMA prefill variants. That attacks 87.22% of model
payload and every ordinary routed expert, while a literal Q3_K kernel attacks
only 1.41% of bytes in a layer that Insignia currently does not execute.
