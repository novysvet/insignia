# Audit: how vLLM runs DeepSeek-style FP8 block-quantized models on Ada/sm89 (Marlin path)

Audited clone: `E:\coding\Insignia\vllm\` (recent main, post `e239947777`; kernels live under
`csrc/libtorch_stable/quantization/marlin/` and `csrc/libtorch_stable/quantization/w8a8/fp8/`).
All paths below are relative to `E:\coding\Insignia\vllm\` unless absolute.

Scope: FP8 e4m3 weights + BF16 128x128 block scales (DeepSeek `weight_block_size=[128,128]`),
running **un-requantized** on sm89 which has no block-scaled FP8 MMA. Findings map directly to
the Insignia mission (Qwen3.8-27B-FP8 on RTX 4070 SUPER).

---

## 1. Executive summary

- On sm89, a 128x128-block FP8 dense linear **cannot** use FlashInfer/DeepGEMM/CUTLASS-block
  (all gated to sm90+/sm100+/sm120+), so vLLM falls through its priority list to
  `MarlinFP8ScaledMMLinearKernel` — a **W8A16** weight-only kernel: FP8 weights are kept as raw
  bytes, activations stay BF16, and the FP8 bytes are dequantized to BF16 *in registers* with two
  bit-mask/shift instructions, right before `mma.sync.m16n8k16.f32.bf16.bf16.f32`.
- The 128x128 block scales are converted to Marlin per-group scales (group_size=128 along K) by
  `repeat_interleave(128, dim=N)` — this is **exact** (no weight requantization). The only
  modification is the **2^120 exponent-bias fold**: scales are multiplied by `2^120` so the
  shift-only FP8→BF16 conversion (which produces `fp8_val * 2^-120`) comes out exact. For FP16
  compute the fold factor is `2^8`.
- Consequence for numerics: the folded scale is stored in **BF16** (8-bit mantissa), so the
  effective scale precision is ~0.4% relative — slightly worse than the FP32 scales in the
  checkpoint. The FP8 values themselves are converted exactly (3-bit mantissa fits).
- The per-token-group 1x128 activation quant kernel (`per_token_group_quant_8bit`) is only used by
  the W8A8 backends; the Marlin W8A16 path does not quantize activations at all
  (`apply_fp8_marlin_linear` raises `"Marlin W8A8 is not supported."`).

---

## 2. Kernel dispatch: when Marlin vs CUTLASS is chosen

### 2.1 Config keys derived from the checkpoint

`vllm/model_executor/layers/quantization/fp8.py:269-283` (`Fp8LinearMethod.__init__`):

```python
self.weight_block_size = self.quant_config.weight_block_size
self.block_quant = self.weight_block_size is not None
...
if self.block_quant:
    self.activation_quant_key = create_fp8_quant_key(
        static=self.act_q_static,
        group_shape=GroupShape(1, self.weight_block_size[0]))   # (1, 128) dynamic
    self.weight_quant_key = create_fp8_quant_key(
        static=True, group_shape=GroupShape(*self.weight_block_size))  # (128, 128) static
```

`init_fp8_linear_kernel` (`vllm/model_executor/kernels/linear/__init__.py:666-712`) checks
`activation_quant_key.scale.group_shape.is_per_group()`; for (1,128) it walks
`_POSSIBLE_FP8_BLOCK_KERNELS` in order:

`vllm/model_executor/kernels/linear/__init__.py:430-442`:

```python
_POSSIBLE_FP8_BLOCK_KERNELS: dict[...] = {
    PlatformEnum.CUDA: [
        FlashInferFp8DeepGEMMDynamicBlockScaledKernel,   # needs cc >= 100 (flashinfer.py:50)
        DeepGemmFp8BlockScaledMMKernel,                  # "only Hopper and Blackwell" (deep_gemm.py:51)
        CutlassFp8BlockScaledMMKernel,                   # CUTLASS_BLOCK_FP8_SUPPORTED (below)
        B12xFp8BlockScaledMMKernel,                      # sm120 family only (b12x.py:60-61)
        MarlinFP8ScaledMMLinearKernel,                   # cc >= 75  <-- WINNER on sm89
        HummingFP8ScaledMMLinearKernel,
        TritonFp8BlockScaledMMKernel,
        BlockWiseTorchFP8ScaledMMKernel,
    ],
    ...
```

First kernel whose `is_supported`+`can_implement` passes wins
(`choose_scaled_mm_linear_kernel`, `__init__.py:602-663`).

### 2.2 The sm89 gates (hard evidence)

`csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu:161-174`:

```cpp
bool cutlass_scaled_mm_supports_block_fp8(int64_t cuda_device_capability) {
  // CUTLASS block-quantized FP8 kernels need at least CUDA 12.0
  // and at least SM90 (Hopper)
#if defined CUDA_VERSION
  if (cuda_device_capability >= 100) {
    return CUDA_VERSION >= 12080;
  } else if (cuda_device_capability >= 90) {
    return CUDA_VERSION >= 12000;
  }
#endif
  return false;   // <-- sm89 lands here
}
```

Marlin's gate — `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py:31-32`:

```python
def is_fp8_marlin_supported():
    return current_platform.has_device_capability(75)
```

and `MarlinFP8ScaledMMLinearKernel.is_supported`
(`vllm/model_executor/kernels/linear/scaled_mm/marlin.py:35-46`) additionally rejects
`VLLM_BATCH_INVARIANT` mode. `can_implement` unconditionally returns True (marlin.py:48-50).

So on the 4070 SUPER (sm89, CUDA >= 12.4):
- **128x128 block quant (DeepSeek/Qwen3.8-FP8)** -> Marlin W8A16.
- **Per-tensor FP8 (w8a8)** -> `CutlassFP8ScaledMMLinearKernel` (2nd in
  `_POSSIBLE_FP8_KERNELS`, `__init__.py:397-406`); `cutlass_scaled_mm_supports_fp8` allows sm89
  with CUDA >= 12.4 (`scaled_mm_entry.cu:145-159`) because plain (non-block-scaled) FP8 MMA
  exists on Ada.
- Overrides: `--linear-backend` filter (`_resolve_backend_kernels`, `__init__.py:344-381`),
  `VLLM_DISABLED_KERNELS` (`__init__.py:580`).

### 2.3 What runs at load time vs forward

`MarlinFP8ScaledMMLinearKernel.process_weights_after_loading`
(`vllm/model_executor/kernels/linear/scaled_mm/marlin.py:66-81`) ->
`process_fp8_weight_block_strategy` (ROCm-only tweaks + no-op padding on CUDA,
`layers/quantization/utils/fp8_utils.py:1372-1406`) -> `prepare_fp8_layer_for_marlin`
(the repack, section 3). Forward: `apply_weights` -> `apply_fp8_marlin_linear`
(section 4.1).

MoE experts use a separate oracle (`vllm/model_executor/layers/fused_moe/oracle/fp8.py:80-95`);
`Fp8MoeBackend.MARLIN` sits below AITER/FlashInfer/DeepGEMM/CUTLASS/Triton — on sm89 it is
likewise the reachable FP8-block backend, using `prepare_fp8_moe_layer_for_marlin`
(`marlin_utils_fp8.py:241-374`, per-expert `gptq_marlin_repack` loops).

---

## 3. FP8 Marlin repacking: weights, scales, and the 2^120 fold

All in `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py`,
`prepare_fp8_layer_for_marlin` (lines 107-219).

### 3.1 Weights: FP8 bytes -> "GPTQ layout" -> Marlin tile interleave

```python
# marlin_utils_fp8.py:377-391
def pack_fp8_to_int32(fp8_tensor, size_k_first=True):
    """Repack FP8 weights to gptq format (packed int32 elements)"""
    fp8_tensor = fp8_tensor.T if size_k_first else fp8_tensor
    fp8_tensor = fp8_tensor.contiguous()
    # fp8_tensor is contiguous and have shape (N, K) now
    # with `.view(torch.int32)`, it become (N, K // 4)
    int32_tensor = fp8_tensor.view(torch.int32)
    return int32_tensor.T.contiguous() if size_k_first else int32_tensor
```

The weight is bitwise-reinterpreted as int32 (4 FP8 bytes per int; `pack_factor = 32/8 = 4`),
zero-padded to the Marlin tile family (`marlin_padded_nk`, `marlin_utils.py:221-243`: needs
`(n % 64, k % 128)` or `(n % 128, k % 64)`; K padded to `lcm(64, group_size)`), then shuffled by
the real repack kernel `ops.gptq_marlin_repack(num_bits=8)`
(marlin_utils_fp8.py:142-154 -> `csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu`).

Repack kernel structure (`gptq_marlin_repack.cu:117-236`): 16x64 tiles, 4 warps, 8-stage
cp.async pipeline from gmem to smem, then `repack_tile` writes each thread's values with the
tensor-core interleave. For 8-bit weights (W8A16, `!is_a_8bit`):

```cpp
// gptq_marlin_repack.cu:222-236
} else {
  constexpr int pack_idx[4] = {0, 2, 1, 3};
  uint32_t res1 = 0, res2 = 0;
  for (int i = 0; i < 4; i++) {
    const int ii = is_a_8bit ? i : pack_idx[i];
    res1 |= vals[ii] << (i * 8);
    res2 |= vals[4 + ii] << (i * 8);
  }
  out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
  out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
}
```

with `tc_offsets[4] = {0, 1, 8, 9}` (line 132) and `tc_row = (th_id % 4) * 2` (line 130), the net
byte layout of each output int is **k = [r, r+8, r+1, r+9]** — i.e. the exact byte placement the
GEMM's two-instruction dequant expects (below). Note bytes 1 and 3 hold k-rows {r+8, r+9} and
bytes 0/2 hold {r, r+1}; this mirrors the standard FP16 marlin ldmatrix layout so the FP8 kernel
reuses all of the FP16 machinery.

### 3.2 Scales: 128x128 blocks -> Marlin group scales (exact, no requant)

`prepare_fp8_layer_for_marlin` (marlin_utils_fp8.py:156-214):

```python
if "weight_scale" in dir(layer):
    scales = layer.weight_scale.to(layer.orig_dtype)
elif "weight_scale_inv" in dir(layer):
    scales = layer.weight_scale_inv.to(layer.orig_dtype)      # DeepSeek name; -> bf16

# marlin kernel only support channel-wise and group-wise quantization
# we need to convert the scales
...
else:
    # block-wise quantization -> group-wise quantization
    # (size_k // block_size[1], ceil(size_n / block_size[0]))
    #  =>(repeat)=> (size_k // block_size[1], size_n)
    if not size_k_first:
        scales = scales.T.contiguous()
    block_n = weight_block_size[0]
    scales = scales.repeat_interleave(block_n, 1)
    # size_n may not divisible by block_size[0]
    scales = scales[:, :part_size_n]

scales = marlin_pad_scales(scales, part_size_n, part_size_k, padded_n, padded_k, group_size)
marlin_scales = marlin_permute_scales(
    s=scales, size_k=padded_k, size_n=padded_n, group_size=group_size)
```

- `group_size = weight_block_size[1] = 128` (marlin_utils_fp8.py:124).
- Each 128x128 block scale becomes one Marlin group scale covering 128 consecutive K positions
  for 128 consecutive N columns — the per-element dequant scale is mathematically identical to
  the checkpoint's block scale. **The FP8 weight bytes are never touched** (un-requantized).
- `marlin_permute_scales` (`marlin_utils.py:470-480`) applies the 64-wide `scale_perm`
  interleave so that smem scale reads line up with the tensor-core N-fragment ownership:

```python
def get_scale_perms():                                  # marlin_utils.py:460-467
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single
```

### 3.3 The 2^120 exponent-bias fold

`marlin_utils_fp8.py:35-46`:

```python
def fp8_fused_exponent_bias_into_scales(scales):
    fp8_exponent = 4
    if scales.dtype == torch.half:
        target_exponent = 5
    elif scales.dtype == torch.bfloat16:
        target_exponent = 8
    # exponent_bias_fp16 = 2 ** 4 - 2 ** 3 = 8
    # exponent_bias_bf16 = 2 ** 7 - 2 ** 3 = 120
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp8_exponent - 1)
    s = torch.ones_like(scales) * 2
    s = s**exponent_bias
    return scales * s
```

Applied at marlin_utils_fp8.py:209-210 right after `marlin_permute_scales`:

```python
if input_dtype != torch.float8_e4m3fn:
    marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)
```

**Why 2^120 (derivation from the kernel side).** The GEMM dequantizes FP8->BF16 with pure
bit operations (`csrc/libtorch_stable/quantization/marlin/dequant.h:357-373`):

```cpp
template <>
__device__ inline void dequant<nv_bfloat162, vllm::kFE4M3fn.id(), true>(
    int q, nv_bfloat162* frag_b) {
  // Constants for FP8 (E4M3) and BF16 formats
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;
  constexpr int MASK = 0x7F007F00;
  // Extract and shift FP8 values to BF16 format
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}
```

The 7 non-sign FP8 bits `[E3..E0 M2..M0]` of each byte land at bits 10:4 of each BF16 lane:
BF16 exponent field (bits 14:7) becomes `0000_EEEE` = raw E, mantissa becomes `MMM0000`
(exact). So the constructed BF16 equals

```
(-1)^s * 2^(E-127) * (1+M/8)   =  fp8_value * 2^((E-127) - (E-7))  =  fp8_value * 2^-120
```

because E4M3 bias is 7 and BF16 bias is 127. Multiplying the scale by 2^120 cancels the factor
exactly. Same story for FP16 (bias 15): factor 2^-8, fold 2^8 (dequant.h:321-336).

The alternative in-kernel correction (multiply each element by 2^120 as a register constant)
exists as `dequant<..., false>` (dequant.h:376-395) but is **not used** for FP8-weights-with-
float-scales: `marlin_template.h:349-353` sets

```cpp
constexpr bool dequant_skip_flop =
    is_a_8bit || (b_type == vllm::kFE4M3fn && !(s_type == vllm::kFE8M0fnu)) || ...
```

i.e. for e4m3 weights + bf16 scales the "flop" (bias multiply) is skipped in-kernel and must be
pre-folded into the scales. One `and`, one `shr`, one `or` per 16-bit lane pair; FP8->BF16
conversion is essentially free.

**Numerics caveats of the fold:**
- Scale precision collapses to BF16's 8-bit mantissa (`scales.to(layer.orig_dtype)` at
  marlin_utils_fp8.py:159-161) — up to ~2^-9 (~0.2%) relative error per block vs the FP32
  checkpoint scales. (vLLM dequant accumulates in FP32 via `f32.bf16.bf16.f32` MMA, and
  `use_fp32_reduce=True` by default, `marlin_utils.py:41`, so the only extra rounding is the
  scale storage plus the `__hmul2` scale application.)
- Overflow headroom: BF16 max ~2^128, folded scale = scale * 2^120, so raw block scales must be
  <= 2^7 (fine for real checkpoints, which are ~1e-3). Tiny scales < 2^-120 would denormal
  (also unrealistic).
- FP8 subnormals (E=0) dequant to values <= 2^-126 * (M/8) — BF16 subnormal territory, so the
  tiniest FP8 subnormals lose mantissa bits. Under per-128-block absmax scaling these are
  rare and absolutely tiny; error is bounded and relative to an already-negligible magnitude.

### 3.4 Workspace and bias

- Workspace = `sms * max_blocks_per_sm` ints used as global-reduction locks
  (`marlin_make_workspace_new`, `marlin_utils.py:408-433`; 1 for dense, 4 for MoE).
- Bias is padded and permuted with the same `scale_perm_single`
  (marlin_utils_fp8.py:216-219).

---

## 4. The W8A16 Marlin GEMM kernel structure

Files: `csrc/libtorch_stable/quantization/marlin/marlin_template.h` (kernel),
`marlin.cu` (dispatch + `marlin_gemm` entry), `marlin_mma.h` (MMA wrappers),
`dequant.h` (dequant), `marlin.cuh` (cp.async helpers), `kernel_selector.h`
(build-generated instantiations from `generate_kernels.py`).

### 4.1 Python entry

`apply_fp8_marlin_linear` (marlin_utils_fp8.py:49-104):

```python
padded_n, padded_k = marlin_repacked_nk(weight, num_bits=8)
reshaped_x = marlin_pad_dim(reshaped_x, size_k, padded_k)
use_atomic_add = should_use_atomic_add_reduce(m, padded_n, padded_k, device, input.dtype)
...
output = ops.marlin_gemm(
    a=inputs, c=None, b_q_weight=weight, b_bias=bias, b_scales=weight_scale,
    a_scales=None, global_scale=None, b_zeros=None, g_idx=None, perm=None,
    workspace=workspace, b_q_type=scalar_types.float8_e4m3fn,
    size_m=reshaped_x.size(0), size_n=padded_n, size_k=padded_k,
    use_atomic_add=use_atomic_add, use_fp32_reduce=use_fp32_reduce)
output = marlin_unpad_output(output, size_n, padded_n)
```

- Activations: raw BF16 (`a_scales=None`); W8A8 raises RuntimeError
  (marlin_utils_fp8.py:79-81, 118-119).
- `should_use_atomic_add_reduce` (`marlin_utils.py:633-653`): needs
  `VLLM_MARLIN_USE_ATOMIC_ADD=1`, n < 2048, k >= 2048 — and **rejects bf16 on sm8x**
  ("sm8x doesn't support atomicAdd + bfloat16 natively"), so on Ada the default is the
  lock-based FP32 global reduce.

`marlin_gemm` (marlin.cu:545-) derives types from the tensors: bf16 A -> `a_type=c_type=bf16`,
and **`s_type_id = c_type_id`** (marlin.cu:599) — the scale dtype follows the compute dtype,
which is what makes the 2^120-folded bf16 scales consistent with the shift dequant.

### 4.2 Grid/block configuration (marlin.cu)

- `stages = 4` on sm80+ (marlin.cu:407); sm75 gets 2.
- Thread-tile candidates (marlin.cu:139-153):

```cpp
thread_config_t small_batch_thread_configs[] = {   // thread_m_blocks == 1
    {128, 128, 256}, {64, 128, 128}, {128, 64, 128}};
thread_config_t large_batch_thread_configs[] = {   // thread_m_blocks > 1
    {64, 256, 256}, {64, 128, 128}, {128, 64, 128}};
```

(units: thread_k, thread_n, threads; validity = divisibility + shared-mem budget,
`is_valid_config` marlin.cu:229-260; shared-mem sizing `get_kernel_cache_size`
marlin.cu:190-227.)

- `group_blocks = group_size / 16` (marlin.cu:355) = 8 for group_size 128; `prob_k % 8 == 0`
  required — trivially true for 128-multiple K.
- Decode specialization: `m_block_size_8 = prob_m_split <= 8 && a_type.size_bits() == 16`
  (marlin.cu:438) — **applies to BF16 W8A16 decode**; uses `mma_trans` (operand-swapped MMA,
  an 8-row tile from the same m16n8k16 instruction).
- Batch splitting: `prob_m` is chunked into up to `max_thread_m_blocks=4` x 16 = 64-row
  problems run as `parallel` stripes (marlin.cu:372-376, 423-427), `max_par = 16` (128 if
  n <= 4096), to keep all SMs busy during decode.
- Persistent-block striped partitioning over (k_tiles x n_tiles) with cross-block reduce in L2
  (comment marlin_template.h:271-281).

### 4.3 gmem -> smem pipeline (`fetch_to_shared`, marlin_template.h:842-909)

- A: `cp_async4_pred` (16B `cp.async.cg`) through a software-transposed smem write index
  (`a_sh_wr_trans`), so later `ldmatrix` sees the right layout.
- B: `cp_async4` of packed int4s — the FP8 payload is **never unpacked in smem**; it is stored
  exactly as repacked (4 FP8 bytes per int).
- Scales: fetched only when a pipeline stage starts a new group:

```cpp
// marlin_template.h:881-891
if constexpr (group_blocks != -1) {
  int4* sh_s_stage = sh_s + s_sh_stage * pipe;
  // Only fetch scales if this tile starts a new group
  if (pipe % div_ceil(group_blocks, thread_k_blocks) == 0) {
    if (s_sh_wr_pred) {
      cp_async4(&sh_s_stage[s_sh_wr], &scales_ptr[s_gl_rd]);
    }
    s_gl_rd += s_gl_rd_delta * s_tb_groups;
  }
}
```

For group_size 128 and thread_k 128 this fires every stage; scale smem footprint is tiny
(K/128 x N bf16).

- Double-buffered consumption with `cp_async_wait<stages - 2>` (marlin_template.h:924-929).

### 4.4 smem -> registers, dequant, scale, MMA (`matmul`, marlin_template.h:1177-1293)

- A fragments: `ldmatrix.sync.aligned.m8n8.x4.shared.b16` (marlin_template.h:82-96, used at
  939-940).
- B fragments: plain 16B smem reads into `frag_b_quant` (marlin_template.h:944-947).
- Scale fragments: `fetch_scales_to_registers` (marlin_template.h:968-1022) loads 16B of bf16
  scales per thread per group (`sh_s[s_sh_rd]`), for group_blocks >= thread_k_blocks every
  pipeline stage (986-992).
- Per k16 slice, for each of 4 n-sub-tiles j:

```cpp
// marlin_template.h:1237-1245 (8-bit weights)
int* frag_b_quant_ptr = reinterpret_cast<int*>(frag_b_quant[k2]);
b_quant_0 = frag_b_quant_ptr[j * 2 + 0];
b_quant_1 = frag_b_quant_ptr[j * 2 + 1];

dequant_data(b_quant_0, reinterpret_cast<scalar_32bit_t*>(&frag_b0));
dequant_data(b_quant_1, reinterpret_cast<scalar_32bit_t*>(&frag_b1));
...
// marlin_template.h:1275-1277 (group-quant, no act-order, no zp)
} else if constexpr (group_blocks != -1 && !is_a_8bit) {
  scale<a_type_id>(frag_b0, frag_s[k2][j], 0);
  scale<a_type_id>(frag_b1, frag_s[k2][j], 1);
}
```

`scale<>` (marlin_template.h:106-115) is a single `__hmul2` of each bf16x2 against the
already-2^120-folded scale held in `frag_s` (reverse-indexed lanes to match the dequant's
permutation, marlin_template.h:104-105).

- MMA: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` — marlin_mma.h:68-75:

```cpp
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
    : "r"(a[0]), "r"(a[1]), "=r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
      "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
```

FP32 accumulation; `thread_m_blocks x thread_n_blocks x thread_k_blocks` counted in 16x16x16
blocks per iteration; j-loop over 4 n-sub-tiles with two mmas each (frag_b0/frag_b1 cover the
two n-column halves of the 16-wide tile). The m-loop is deliberately innermost "in order to
encourage overlapping dequantization and matmul operations" (comment at 1223-1224).
Decode path (`m_block_size_8`) swaps operands via `mma_trans` (marlin_mma.h:137-267,
bf16 case 200-207).

### 4.5 Epilogue

- Warp/block reduce through smem (`sh_red`), then `global_reduce_fp16`
  (marlin_template.h:1467+): partial results are written as bf16 into C and serialized through
  the `locks` workspace with FP32 compute, "directly in the output buffer to maximize L2 cache
  utilization" (comment 1463-1466). atomicAdd reduce optional but bf16-blocked on sm8x.
- Bias added in smem during the epilogue (fetch around marlin_template.h:1915-1929).

---

## 5. Per-token-group 1x128 activation quant kernel (CUDA)

File: `csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu`
(entry `per_token_group_quant_fp8` -> `per_token_group_quant_8bit`, lines 613-622).
Used by the W8A8 backends (CUTLASS/DeepGEMM/Triton) — **not** by Marlin W8A16.

### 5.1 Generic kernel: `per_token_group_quant_8bit_kernel` (lines 100-164)

- Layout: 16 threads per group, `groups_per_block` groups per block (16/8/4/2/1 to fit,
  `GetGroupsPerBlock` 166-180); each group handles `group_size` (128) contiguous elements of
  one token row.
- Two-pass through **shared memory** to avoid double DRAM reads:

```cpp
// lines 42-74
template <typename T, bool SCALE_UE8M0>
__device__ __forceinline__ float ComputeGroupScale(...) {
  float local_absmax = eps;
  auto scalar_op_cache = [&] __device__ (T& dst, const T& src) {
    float abs_v = fabsf(static_cast<float>(src));
    local_absmax = fmaxf(local_absmax, abs_v);
    dst = src;                       // copy global -> shared while reducing
  };
  vllm::vectorize_with_alignment<vec_size>(   // vec_size = 16B / sizeof(T)
      group_input, smem_group, group_size, lane_id, threads_per_group,
      scalar_op_cache);
  local_absmax = GroupReduceMax(local_absmax);
  float y_s = local_absmax / max_8bit;        // max_8bit = 448 for e4m3
  if constexpr (SCALE_UE8M0) {
    y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));   // pow2-rounded scale
  }
  return y_s;
}
```

- Group max: 4-step `__shfl_xor_sync` butterfly (8/4/2/1) within each 16-thread group
  (`GroupReduceMax`, lines 21-40; ROCm variant handles packed 16-thread groups per wavefront).
- Scale write: `lane_id == 0` stores one fp32 (`*scale_output = y_s`, line 152-154), with
  optional column-major/transposed scale output for CUTLASS expectations (lines 126-139).
- Quantize pass re-reads from smem: `q = clamp(src / y_s, min, max); dst = fp8(q)`
  (`QuantizeGroup`, lines 76-96) — one fused-multiply-free div per element; writes vectorized.
- Launch uses `cudaLaunchKernelEx` with **PDL** (programmatic dependent launch,
  `cudaLaunchAttributeProgrammaticStreamSerialization`, lines 222-242) plus
  `cudaGridDependencySynchronize()`/`cudaTriggerProgrammaticLaunchCompletion()` on sm90+
  (lines 122-124, 161-163) to overlap with the preceding kernel.

### 5.2 Register-resident fast path for group=128 + UE8M0 (lines 289-611)

`per_token_group_quant_8bit_packed_register_kernel` (used with DeepGEMM-style packed e8m0
scales):

- 8 threads/group, each holds 16 elements (2x uint4 = 32B) in registers — **no smem**, single
  gmem read + single gmem write per element (lines 343-364).
- 3-step xor-shuffle reduce over the 8-lane subgroup (lines 366-379).
- e8m0 scale by pure bit math, bit-exact with `exp2f(ceilf(log2f()))`
  (lines 381-387): `exp_byte = exponent + (mantissa != 0)`, scales packed 4-per-int32 in a
  TMA-friendly transposed layout `[k_num_packed_sfk, tma_aligned_mn]` (lines 389-399).
- Quantize with reconstructed pow2 scale and its reciprocal (`inv_y`, lines 416-429), packs 16
  fp8 bytes into one uint4 store (lines 420-450); zero-fills TMA padding rows (lines 401-414).

### 5.3 Relevance on sm89

On Ada the marlin path never calls these (activations stay bf16). If Insignia ever runs a
mixed path (e.g., FP8 activations through `mma.m16n8k32.f32.e4m3.e4m3.f32` which **does**
exist on sm89 — see marlin_mma.h:94-101), the generic 16-thread/128-element smem kernel is the
reference design, and the register fast path is the design to beat.

---

## 6. Actionable implications for Insignia (Qwen3.8-27B-FP8 on RTX 4070 SUPER)

1. **Copy the dequant trick verbatim.** FP8-e4m3 -> BF16 is `(q & 0x80008000) | ((q &
   0x7F007F00) >> 4)` (+ a second word from `q << 8`): 3 logic ops per 4 values, zero FLOPs,
   zero conversion instructions. Fold `2^120` into the BF16 group scales at load time (or keep
   scales FP32 and fold at kernel launch). For FP16 compute the fold is `2^8`
   (`FP16_EXPONENT 5` shift = 1 bit).
2. **Block scales -> group scales is free and exact.** 128x128 block scale expansion is just
   broadcasting each scale over 128 N-columns with group_size 128 along K. Do it offline/at
   load; do NOT requantize weights — vLLM's path is bitwise-lossless on the weights.
3. **Improve on vLLM where it is weak**: vLLM stores folded scales in BF16 (8-bit mantissa).
   Insignia can keep per-block scales in FP32 in smem (footprint is K/128 x N — negligible)
   and apply via FFMA into the accumulator side, or convert weight to BF16 and scale in FP32,
   removing the ~0.2% per-block scale rounding error. This is the single easiest numerical
   advantage over vLLM marlin.
4. **Kernel skeleton worth mirroring** for decode GEMV/GEMM: persistent striped partitioning +
   locks-based L2 global reduce (bf16 atomicAdd unavailable on sm8x), cp.async 4-stage
   pipeline, packed-FP8 bytes stored raw in smem, ldmatrix for A only, dequant+`__hmul2`
   immediately before `mma.m16n8k16.f32.bf16.bf16.f32`, and the operand-swapped `mma_trans`
   trick for m<=8 decode batches.
5. **If Insignia wants W8A8-FP8 on Ada**: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
   is legal on sm89 (marlin_mma.h:94-101; guarded to sm89/sm12x only, marlin.cu:413-421,
   marlin_template.h:283-286). vLLM wires this only for W4A8; the activation quant kernel to
   pair with it is the audited `per_token_group_quant_8bit` (16 thr/group, smem-staged,
   xor-butterfly absmax, `absmax/448` scale).
6. **Padding rules to honor**: Marlin tiles need `(n%64 && k%128)` or `(n%128 && k%64)`
   (`marlin_padded_nk`); FP8 zero bytes decode to 0.0 so padding is harmless; K padding must
   keep `lcm(64, group_size)` alignment so scale groups stay integral.
7. **Watch the overflow/denormal envelope** of the 2^120 fold: block scales must be in
   (2^-120, 2^7) — fine for real checkpoints, but assert it at load time (vLLM does not).

---

## 7. Key file index (absolute paths)

| What | Where |
| --- | --- |
| Kernel priority lists (block fp8) | `E:\coding\Insignia\vllm\vllm\model_executor\kernels\linear\__init__.py:430-455` |
| `init_fp8_linear_kernel` | same file, `:666-736` |
| Marlin linear kernel class | `E:\coding\Insignia\vllm\vllm\model_executor\kernels\linear\scaled_mm\marlin.py:29-115` |
| CUTLASS block-fp8 gate (sm90+) | `E:\coding\Insignia\vllm\csrc\libtorch_stable\quantization\w8a8\cutlass\scaled_mm_entry.cu:161-174` |
| Fp8LinearMethod / quant keys | `E:\coding\Insignia\vllm\vllm\model_executor\layers\quantization\fp8.py:239-380` |
| FP8 marlin repack + 2^120 fold | `E:\coding\Insignia\vllm\vllm\model_executor\layers\quantization\utils\marlin_utils_fp8.py:35-46,107-219,377-391` |
| Scale permutation / padding helpers | `E:\coding\Insignia\vllm\vllm\model_executor\layers\quantization\utils\marlin_utils.py:221-302,460-487,633-653` |
| Marlin GEMM kernel template | `E:\coding\Insignia\vllm\csrc\libtorch_stable\quantization\marlin\marlin_template.h:225-1293 (dequant_skip_flop :349-353, fetch :842-909, scales :968-1022, matmul :1177-1293)` |
| Shift dequant FP8->BF16/FP16 | `E:\coding\Insignia\vllm\csrc\libtorch_stable\quantization\marlin\dequant.h:321-395` |
| MMA instructions | `E:\coding\Insignia\vllm\csrc\libtorch_stable\quantization\marlin\marlin_mma.h:68-101,200-207` |
| Dispatch/thread configs | `E:\coding\Insignia\vllm\csrc\libtorch_stable\quantization\marlin\marlin.cu:139-153,276-324,400-541,545-616` |
| Weight repack kernel | `E:\coding\Insignia\vllm\csrc\libtorch_stable\quantization\marlin\gptq_marlin_repack.cu:117-236` |
| Per-token-group 1x128 quant | `E:\coding\Insignia\vllm\csrc\libtorch_stable\quantization\w8a8\fp8\per_token_group_quant.cu:21-164 (generic), 289-611 (register fast path)` |
| MoE marlin prep | `E:\coding\Insignia\vllm\vllm\model_executor\layers\quantization\utils\marlin_utils_fp8.py:241-374` |
