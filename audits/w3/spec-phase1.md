# Spec decode phase 1 — patch-level designs (F7 determinism, D=2 / T=4 groundwork)

Read of: `src/decode.cu` (264 lines), `src/prefill.cu` (315), `src/generate.cu` (211),
`include/insignia_decode.hpp` (52), `include/insignia_prefill.cuh`, `include/insignia_ops.cuh`,
`include/insignia_qwen_kernels.cuh`, `include/insignia_layout.cuh`, `src/gemm.cu` (kernel inventory),
`src/mxfp4_i4.cu`, `tools/reference_multistep.py:90-135`, `audits/w3/spec-deepen.md`.
All line numbers below refer to the current working tree.

Reference convention (verified in `reference_multistep.py::mtp_draft`): an MTP invocation at
slot `s` consumes `(embed(t_{s+1}), h_s)`, ropes q/k at position `s`, stores k/v into
`mtp_kvc[s]`, attends slots `0..s` (scores ×1/16), proposes the token for position `s+2`.
`mtp_kvc` is `np.zeros((64,4,256))` — zero-initialized AND densely filled. The engine's
`mtp_layer()` already implements the invocation exactly right (rope pos = store slot = read
window = `mtp_pos`). **The only defects are the two hole classes (F7). The "attend with
base=+1" idea from the mission brief is semantically impossible: an invocation at slot `P`
would need `h_P`, which does not exist until after the verify row for position `P` runs.**
Keep `spec_prologue` (`pos[7] = pos[0]-1`) as-is; fix the holes, not the base.

---

## 1. F7 — MTP KV determinism + prefill fill

### 1.1 The hole map (exact trace)

Start: prompt of N tokens prefilled, `pos[0]=N`, `pending=t_N` (=`first`), `x_.hidden=h_{N-1}`.

| step | pos at entry | prologue mtp_pos | slot written by mtp_layer | slots read (gqa window) | accept? | pos[0] after commit | hole created |
|---|---|---|---|---|---|---|---|
| 1 | N   | N-1 | N-1 | 0..N-1 → **0..N-2 never written (F7a)** | yes | N+2 | slot N |
| 2 | N+2 | N+1 | N+1 | 0..N+1 → **slot N garbage** | yes | N+4 | slot N+2 |
| 3 | N+4 | N+3 | N+3 | 0..N+3 → **slots N, N+2 garbage** | yes | N+6 | slot N+4 |

Precisely: a step entering at `pos=P` writes slot `P-1`. On accept `pos` advances to `P+2`,
so the next write is `P+1` — **slot P is skipped forever** (its invocation
`(embed(t_{P+1}), h_P)` is exactly the *second* draft of a D=2 chain, which never runs at
D=1). After k consecutive accepts there are k garbage slots, one per accept, at each
`pos_after_commit - 2`. A reject is dense: reject at P leaves `pos=P+1` → next write is
slot P, overwriting nothing valid (the slot below was already written). A reject also
repairs the *previous* accept's hole: the hole from the prior accept sits at
`(P+1)-1-1+... ` — i.e. the next draft's own slot — so it is rewritten before its gqa
reads it (store_kv precedes gqa in `mtp_layer`, decode.cu:169-170).

Fix components:
- **F7a** — batched MTP pass folded into prompt prefill (section 1.3): fills slots
  `0..N-2` densely, reference-equal (teacher-forced, no lm_head).
- **F7b** — fill/patch the single accept-hole at commit (section 1.4): exact extra
  invocation (v2, default) or zero-patch (v1, fallback).
- **F7c** — determinism memset of both MTP caches at workspace init (section 1.2): no
  read slot is ever uninitialized once F7a+F7b land, but the memset turns any residual
  `cudaMalloc` garbage into deterministic zeros for probe/off-by-one insurance.

### 1.2 Patch 0 — workspace init memset + buffer widenings (decode.cu)

**Insertion: decode.cu line 27** (append to the existing memset chain):

```cpp
 cudaMemsetAsync(mtp_keys,0,size_t(ctx)*1024*4,stream);cudaMemsetAsync(mtp_values,0,size_t(ctx)*1024*4,stream);
```

**decode.cu line 14** — widen the logits buffer for T-row verify (item 5): change
`alloc(&logits,248320*2)` to

```cpp
alloc(&logits,248320*8)
```

(8 rows = 7.9 MB, covers T ≤ 8; the destructor needs no change — it frees the pointer.)

**decode.cu line 14 (end)** — add the D=2 fill-residual buffer:

```cpp
alloc(&mtp_fill_h,4096);
```

**decode.cu line 15** — add the extended pos-slot aliases after `mtp_pos_dev=pos_dev+7;`:

```cpp
tstar2_dev=pos_dev+8;tstar3_dev=pos_dev+9;draft2_dev=pos_dev+10;draft3_dev=pos_dev+11;
```

**decode.cu line 29** — add `cudaFree(mtp_fill_h);` to the destructor.

**include/insignia_decode.hpp** — in `DecodeWorkspace` (after line 11 `snap_conv`):

```cpp
 float *mtp_fill_h{};  // saved chain residual R_D for the full-accept MTP hole fill
```

and after `mtp_pos_dev{}` in line 9:

```cpp
 int *tstar2_dev{},*tstar3_dev{},*draft2_dev{},*draft3_dev{};  // pos_dev slots 8..11 (T=4/D=3 layout)
```

### 1.3 Patch 1 — `mtp_prefill_chunk`: the batched prompt fill (F7a)

Design facts that make this nearly free:

- After `prefill_chunk_device`'s layer loop, **`pf_x` rows 0..T-1 hold the raw main
  residuals h_{c0}..h_{c0+T-1}** (final rmsnorm at decode.cu:92 reads pf_x into pf_n,
  non-destructive; the copy at line 102 only exports the last row). So the per-chunk hidden
  states ARE retained inside the chunk — no recompute, no RAM bounce, no host round trip.
- The fill invocation for slot `s = c0+r` needs `embed(t_{s+1}) = embed(tokens_dev[r+1])` —
  all within the current chunk (`r+1 ≤ T-1`). Every chunk fills exactly its first **T-1**
  rows (slots `c0..c0+T-2`); the final chunk's last row (slot N-1) is deliberately left to
  the first draft's own store_kv. No cross-chunk dependency.
- Every batch kernel needed is parameterized by pointers and reused verbatim:
  `qk_norm_rope_batch(q,k,qw,kw,pos_dev,T)` (prefill.cu:84) reads `pos_dev[0]+t` — pass
  `x_.mtp_pos_dev` set to `c0`; `store_kv_batch`/`gqa_prefill` take `kc/vc` pointers —
  pass `x_.mtp_keys/x_.mtp_values`. The MTP layer is full-attention-only (16q/4kv/256d,
  partial rope 64 — identical geometry to main full-attn layers), so no conv/deltanet
  state is involved.
- Only two small new kernels are required: a batched `concat`, and (MLX format only) a
  small-T bf16 GEMM for `mtp.fc` ([4096,8192] bf16 = 67 MB). INSIG4 reuses the existing
  `mxfp4_gemm_mlx_i4` (T rows, gemm.cu:354). No bf16 GEMM exists today (`bf16_gemv`,
  qwen_kernels.cu:67-68, is a per-row GEMV — looping it T-1 times would re-stream the
  67 MB weight 63× ≈ 12 ms/chunk; rejected).

Buffer juggling inside the fill (all existing `pf_*` scratch, no new allocs; everything is
stream-ordered so read-before-overwrite is guaranteed):

| role | buffer | note |
|---|---|---|
| E = embed(tokens[1..T-1]) | `x_.pf_up` (compact [R,4096]) | `embed_gather(_i4)(..., tokens_dev+1, ..., R, ...)` |
| En = rmsnorm(E, pre_fc_norm_embedding) | `x_.pf_gate` | compact [R,4096] |
| Hn = rmsnorm(pf_x rows 0..R-1, pre_fc_norm_hidden) | `x_.pf_n` | pf_n is dead after the chunk; pf_x rows still raw |
| fc_in = [En | Hn] | `x_.pf_scratch` | [R,8192], exact concatenation order per mtp_layer:147 |
| fc_out = MTP residual stream | `x_.pf_x` | written only after concat consumed Hn → in-place residual, same pattern as mtp_layer's use of `x_.hidden` |
| layer scratch | pf_q/pf_g/pf_k/pf_v/pf_core/pf_down/pf_scratch/pf_gate/pf_up | E/En/Hn all dead by then |

**New kernel A — `concat_batch` (prefill.cu, insert after `embed_gather_i4`, ~line 40):**

```cpp
// Batched twin of concat() (qwen_kernels.cu): out[t] = [a[t] | b[t]], each row n floats.
__global__ void concat_batch_kernel(const float *__restrict__ a, const float *__restrict__ b,
                                    float *__restrict__ out, int n) {
    const size_t t = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[t * 2 * n + i] = a[t * n + i];
        out[t * 2 * n + n + i] = b[t * n + i];
    }
}
void concat_batch(const float *a, const float *b, float *out, int rows, int n, cudaStream_t stream) {
    concat_batch_kernel<<<rows, 256, 0, stream>>>(a, b, out, n);
}
```

**New kernel B — `bf16_gemm_tn` (gemm.cu, insert at end; declaration in
insignia_layout.cuh after `mxfp4_gemm_v21`, line 64):**

```cpp
// Y[t,r] = sum_k A[t,k]*W[r,k].  W bf16 row-major [rows,cols] (mtp.fc, MLX format);
// A f32 [T,cols], Y f32 [T,rows] row-major (linear_batch layout).  T <= 63 prefill fill path.
// Warp = one W row broadcast to 32 A rows (lane=t); A tile staged in smem to keep loads coalesced.
// W bytes are streamed ceil(T/32) times (67MB -> 134MB at T=63, ~0.35ms) — fine for prefill.
__global__ __launch_bounds__(256) void bf16_gemm_tn_kernel(const uint16_t *__restrict__ W,
                                                           const float *__restrict__ A,
                                                           float *__restrict__ Y, int cols) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5);    // 8 W rows per block, one per warp
    const int t = blockIdx.y * 32 + (threadIdx.x & 31);   // 32 A rows per block
    __shared__ float As[32][65];                          // padded to dodge bank conflicts
    float acc = 0.f;
    for (int k0 = 0; k0 < cols; k0 += 64) {
        __syncthreads();
        for (int i = threadIdx.x; i < 32 * 64; i += 256) {  // cooperative coalesced A load
            const int at = i >> 6;
            As[at][i & 63] = (blockIdx.y * 32 + at) < gridDim.y * 32 && (blockIdx.y * 32 + at) < gridDim.x * 0 + (1 << 30)
                             ? __ldg(&A[size_t(blockIdx.y * 32 + at) * cols + (i & 63) + k0]) : 0.f;
        }
        __syncthreads();
        #pragma unroll 8
        for (int k = 0; k < 64; k++)
            acc = fmaf(As[threadIdx.x & 31][k],
                       __bfloat162float(__ldg(&W[size_t(r) * cols + k0 + k])), acc);
    }
    if (t < gridDim.y * 32) Y[size_t(t) * (gridDim.x * 8) + r] = acc;
}
void bf16_gemm_tn(const uint16_t *W, const float *A, float *Y, int rows, int cols, int T, cudaStream_t stream) {
    bf16_gemm_tn_kernel<<<dim3(rows / 8, (T + 31) / 32), 256, 0, stream>>>(W, A, Y, cols);
}
```

Note: the `T` bound must reach the kernel — cleanest is to pass it and bound the smem load
with it (rows/8 requires rows%8==0; fc rows=4096 ✓). Simplify the guard in the real patch to:

```cpp
__global__ __launch_bounds__(256) void bf16_gemm_tn_kernel(const uint16_t *__restrict__ W,
                                                           const float *__restrict__ A,
                                                           float *__restrict__ Y, int cols, int T, int rows) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int t = blockIdx.y * 32 + (threadIdx.x & 31);
    __shared__ float As[32][65];
    float acc = 0.f;
    for (int k0 = 0; k0 < cols; k0 += 64) {
        __syncthreads();
        for (int i = threadIdx.x; i < 32 * 64; i += 256) {
            const int at = i >> 6, gt = blockIdx.y * 32 + at;
            As[at][i & 63] = gt < T ? __ldg(&A[size_t(gt) * cols + (i & 63) + k0]) : 0.f;
        }
        __syncthreads();
        #pragma unroll 8
        for (int k = 0; k < 64; k++)
            acc = fmaf(As[threadIdx.x & 31][k],
                       __bfloat162float(__ldg(&W[size_t(r) * cols + k0 + k])), acc);
    }
    if (t < T) Y[size_t(t) * rows + r] = acc;
}
```

**The fill function (decode.cu — insert after `prefill_chunk_seam`, ~line 117):**

```cpp
// Batched MTP pass over the chunk's first T-1 rows: fills mtp_keys/values slots
// c0..c0+T-2 with the teacher-forced invocations (embed(t_{s+1}), h_s) — no lm_head,
// no proposal.  Requires pf_x rows 0..T-1 = raw main residuals of the chunk (i.e. call
// immediately after prefill_chunk_device on the same stream) and mtp_pos_dev = c0.
void Qwen35Decode::mtp_prefill_chunk(const int *tokens_dev, int T) {
    const int R = T - 1;                     // fill rows; slot c0+r gets row r
    if (R <= 0) return;                      // T==1 chunk: nothing to fill
    {   // E = embed(tokens[1..T-1]) -> pf_up (compact [R,4096])
        const std::string base = "language_model.model.embed_tokens";
        auto m = w_.matrix(base);
        if (m.insig4) embed_gather_i4((const uint32_t *)m.weight.data, (const uint16_t *)m.scales.data, tokens_dev + 1, x_.pf_up, R, x_.stream);
        else embed_gather((const uint32_t *)m.weight.data, (const uint8_t *)m.scales.data, tokens_dev + 1, x_.pf_up, R, x_.stream);
        w_.release(base);
    }
    auto ew = tensor("language_model.mtp.pre_fc_norm_embedding.weight");
    auto hw = tensor("language_model.mtp.pre_fc_norm_hidden.weight");
    rmsnorm_bf16(x_.pf_up, (const uint16_t *)ew.data, x_.pf_gate, R, 4096, false, x_.stream);  // En
    rmsnorm_bf16(x_.pf_x,  (const uint16_t *)hw.data, x_.pf_n,    R, 4096, false, x_.stream);  // Hn (reads pf_x rows 0..R-1)
    w_.storage().release("language_model.mtp.pre_fc_norm_embedding.weight");
    w_.storage().release("language_model.mtp.pre_fc_norm_hidden.weight");
    concat_batch(x_.pf_gate, x_.pf_n, x_.pf_scratch, R, 4096, x_.stream);                       // fc_in [R,8192]
    {   // fc: [R,8192] -> [R,4096] into pf_x (becomes the MTP residual stream, in place)
        auto fc = w_.matrix("language_model.mtp.fc");
        if (fc.insig4) mxfp4_gemm_mlx_i4((const uint32_t *)fc.weight.data, (const uint16_t *)fc.scales.data, x_.pf_scratch, x_.pf_x, 4096, 8192, R, x_.stream);
        else bf16_gemm_tn((const uint16_t *)fc.weight.data, x_.pf_scratch, x_.pf_x, 4096, 8192, R, x_.stream);
        w_.release("language_model.mtp.fc");
    }
    // ---- the MTP layer, batch form: mirror of the full-attn branch of prefill_chunk_device,
    // ---- T=R, position source = mtp_pos_dev (row r at slot c0+r), kv = mtp_keys/mtp_values.
    const std::string p = "language_model.mtp.layers.0";
    const std::string a = p + ".self_attn";
    auto inw = tensor(p + ".input_layernorm.weight");
    rmsnorm_bf16(x_.pf_x, (const uint16_t *)inw.data, x_.pf_n, R, 4096, false, x_.stream);
    w_.storage().release(p + ".input_layernorm.weight");
    linear_batch(a + ".q_proj", x_.pf_n, x_.pf_scratch, R);
    split_q_gate_batch(x_.pf_scratch, x_.pf_q, x_.pf_g, R, x_.stream);
    linear_batch(a + ".k_proj", x_.pf_n, x_.pf_k, R);
    linear_batch(a + ".v_proj", x_.pf_n, x_.pf_v, R);
    auto qw = tensor(a + ".q_norm.weight"), kw = tensor(a + ".k_norm.weight");
    qk_norm_rope_batch(x_.pf_q, x_.pf_k, (const uint16_t *)qw.data, (const uint16_t *)kw.data, x_.mtp_pos_dev, R, x_.stream);
    w_.storage().release(a + ".q_norm.weight"); w_.storage().release(a + ".k_norm.weight");
    store_kv_batch(x_.pf_k, x_.pf_v, x_.mtp_keys, x_.mtp_values, x_.mtp_pos_dev, R, x_.max_context, x_.stream);
    gqa_prefill(x_.pf_q, x_.mtp_keys, x_.mtp_values, x_.pf_core, x_.mtp_pos_dev, R, x_.max_context, x_.stream);
    sigmoid_mul(x_.pf_core, x_.pf_g, size_t(R) * 4096, x_.stream);
    linear_batch(a + ".o_proj", x_.pf_core, x_.pf_down, R);
    residual_add(x_.pf_x, x_.pf_down, size_t(R) * 4096, x_.stream);
    auto post = tensor(p + ".post_attention_layernorm.weight");
    rmsnorm_bf16(x_.pf_x, (const uint16_t *)post.data, x_.pf_n, R, 4096, false, x_.stream);
    w_.storage().release(p + ".post_attention_layernorm.weight");
    linear_batch(p + ".mlp.gate_proj", x_.pf_n, x_.pf_gate, R);
    linear_batch(p + ".mlp.up_proj", x_.pf_n, x_.pf_up, R);
    silu_mul(x_.pf_gate, x_.pf_up, x_.pf_gate, size_t(R) * 12288, x_.stream);
    linear_batch(p + ".mlp.down_proj", x_.pf_gate, x_.pf_down, R);
    residual_add(x_.pf_x, x_.pf_down, size_t(R) * 4096, x_.stream);
    // no mtp.norm, no lm_head: the k/v side effects are the product
}

// Prompt-chunk driver: prefill the main model, then fill the MTP cache for the chunk.
int Qwen35Decode::prefill_chunk_mtp(const int *tokens, int T) {
    const int c0 = x_.position;                 // host mirror is exact before the chunk
    cudaMemcpyAsync(x_.pf_tokens, tokens, sizeof(int) * T, cudaMemcpyHostToDevice, x_.stream);
    set_mtp_position(c0);                       // async; ordered on the stream before the fill kernels
    prefill_chunk_device(x_.pf_tokens, T);
    mtp_prefill_chunk(x_.pf_tokens, T);         // same stream => pf_x/pf_tokens still live
    cudaMemcpyAsync(x_.next_host, x_.next_dev, sizeof(int), cudaMemcpyDeviceToHost, x_.stream);
    cudaStreamSynchronize(x_.stream);
    return *x_.next_host;
}
```

Header declarations (`insignia_decode.hpp`): `mtp_prefill_chunk` private,
`prefill_chunk_mtp` public (next to `prefill_chunk`, line 24). `concat_batch` declaration
goes in `insignia_prefill.cuh`; `bf16_gemm_tn` in `insignia_layout.cuh`.

Correctness notes:
- Row r ropes at position `mtp_pos_dev[0]+r = c0+r` (qk_norm_rope_batch, prefill.cu:73),
  stores at `c0+r` (store_kv_batch, :91), attends `c0+r+1` slots (gqa_prefill, :104) —
  exactly the reference's slot-s triad. Causality within the batch is free: store_kv_batch
  writes all R rows before gqa_prefill launches, and row r reads ≤ its own slot.
- `gqa_prefill`'s smem `score[4096]` bounds the window: max read = `c0+R = N-1 ≤ 4095` ✓
  (the same 4096 cap the workspace ctor enforces).
- `linear_batch` MLX path zero-pads `pf_bf16` to 64 rows (decode.cu:37) — the v21 kernel's
  64-row A-tile compute waste applies here too, but this is prefill; correctness unaffected.
  INSIG4 path streams `mxfp4_gemm_mlx_i4` unpipelined — correct; the i4-v21 port is a perf
  dependency only (same note as item 5).
- Cost/chunk(64): fc GEMM ≤ 134 MB + layer ≈ 106 MB + embed ≈ 0.25 MB ≈ **240 MB ≈ 0.6 ms**
  (~4-5% on top of a ~13 ms main chunk; ~10 ms per 1k prompt tokens), ~20 launches. The
  sequential-mtp_layer alternative (64 × 0.7 ms = 45 ms/chunk) is 75× worse — rejected.

**Wiring (generate.cu lines 125-129):** replace `d.prefill_chunk(...)` with
`d.prefill_chunk_mtp(...)` in the prompt loop:

```cpp
        while (done < tokens.size()) {
            int T = int(tokens.size() - done) >= 64 ? 64 : int(tokens.size() - done);
            first = d.prefill_chunk_mtp(tokens.data() + done, T);
            done += T;
        }
```

**Delete generate.cu lines 142-143** (the probe's post-probe `cudaMemsetAsync` of the MTP
caches): with F7a those two lines *destroy correctly filled state* right before the warm
spec step. The init memset (Patch 0) now owns determinism.

### 1.4 Patch 2 — the accept-skips-slot hole (F7b)

The next draft's read window always includes exactly one never-written slot after an
accept: slot `pos[0]-2` at commit time. Two fixes; ship v2 (exact) as default, keep v1 as
the emergency fallback.

**v1 — zero-patch (deterministic, ~free, NOT reference-equal).** New kernel in
prefill.cu (after `spec_commit`, ~line 302; declaration in insignia_prefill.cuh):

```cpp
// On accept, slot pos-2 of the MTP cache was skipped (its invocation is the never-run
// 2nd draft); the next draft would attend cudaMalloc garbage. Zero it: deterministic,
// bounded, and the zero-key's softmax weight is exp(0-max)/Z (typically negligible).
__global__ void spec_mtp_patch_kernel(const int *__restrict__ pos, float *__restrict__ mk, float *__restrict__ mv) {
    if (!pos[6]) return;                        // reject path rewrites this slot next step
    float *k = mk + size_t(pos[0] - 2) * 1024, *v = mv + size_t(pos[0] - 2) * 1024;
    for (int i = threadIdx.x; i < 1024; i += 32) { k[i] = 0.f; v[i] = 0.f; }
}
void spec_mtp_patch(int *pos, float *mk, float *mv, cudaStream_t stream) {
    spec_mtp_patch_kernel<<<1, 32, 0, stream>>>(pos, mk, mv);
}
```

**v2 — exact fill (default; reference-equal; unconditional is safe).** After commit, run
the missing invocation itself (no head): token = the last committed draft (`pos[4]`),
hidden = `h_P = pf_x[0]`. It requires the `mtp_layer` refactor of section 5.1 —
`mtp_layer_impl(token_src, hid, argmax_dst, with_head)`:

```cpp
// save h_P before anything scribbles pf_x (rollback already read it; fill is order-free after this)
cudaMemcpyAsync(x_.mtp_fill_h, x_.pf_x, 4096 * 4, cudaMemcpyDeviceToDevice, x_.stream);
bumpi_kernel<<<1, 1, 0, x_.stream>>>(x_.mtp_pos_dev);            // slot P-1 -> P = pos-2
mtp_layer_impl(x_.draft_dev, x_.mtp_fill_h, x_.draft_dev, false); // (embed(draft), h_P) @ slot P, no head
```

Why unconditional is safe (both branches traced):
- **Accept** (pos=P+2 after commit): bumpi → slot P = `pos-2` = exactly the hole; inputs
  (embed(a1), h_P) are the true invocation → k/v[P] reference-exact.
- **Reject** (pos=P+1): bumpi → slot P with (embed(rejected draft), h_P) — wrong token for
  that slot, but the next step's draft (pos=P+1 → slot P) store_kv-overwrites slot P
  before its gqa reads it. Nothing valid is clobbered (slot P-1, the good draft-1 entry,
  is untouched).
- Cost: ~0.55 ms/step unconditional (fc 67 MB + layer ~106 MB + embed + lm_head skipped);
  ≈ 0.33 ms amortized at p=0.6 (~2.5% of the 13.1 ms step). Conditional graph nodes
  (CUDA 12.4+) could gate it later; not worth it in phase 1.

**ORDERING HAZARD (documented):** the fill must run **after** `spec_rollback`, or — with
the `mtp_fill_h` staging copy above — it is order-free. Without the staging copy (reading
`pf_x[0]` directly as the residual stream), a fill placed *before* rollback would make
rollback's `hidden[i] = pf_x[i]` (prefill.cu:310) read the MTP chain residual instead of
the main h_P. The staging copy removes the hazard entirely; keep it.

**Wiring (decode.cu):** in `spec_step` (lines 215-233) and `capture_spec` (234-245),
insert between `spec_commit(...)` (line 221/240) and the `x_.position`/graph-end logic —
concretely right after `spec_rollback(...)` (line 222/241):

```cpp
    // F7b: fill the accept-hole slot (pos-2) with the exact skipped MTP invocation
    cudaMemcpyAsync(x_.mtp_fill_h, x_.pf_x, 4096 * 4, cudaMemcpyDeviceToDevice, x_.stream);
    bumpi_kernel<<<1, 1, 0, x_.stream>>>(x_.mtp_pos_dev);
    mtp_layer_impl(x_.draft_dev, x_.mtp_fill_h, x_.draft_dev, false);
```

(or the single `spec_mtp_patch` launch if v1 is preferred). `spec_prologue` rewrites
`pos[7]=pos[0]-1` every step, so the bumpi carries no state across steps. All ops
(kernels + D2D memcpy) are graph-capturable.

**Post-fix invariants:** every slot a draft's gqa can read (0..mtp_pos) is either
prefill-filled (F7a), written by a previous draft-1 of some step, written by this draft's
own store_kv, or accept-filled/zero-patched (F7b). MTP cache is append-only; no rewind.

---

## 2. Per-row snapshots at T=2 — verification only (no patch)

Confirmed correct as-is:

- `deltanet_prefill` snapshots state after t==0 (prefill.cu:258-261); `conv_roll_state`
  snapshots `[s1,s2,x0]` = the window rolled to just-after row 0 (prefill.cu:184-195).
- At T=2 the accept length a ∈ {0,1}. a=1 (full accept): `spec_rollback` early-exits
  (`if (pos[6]) return`, prefill.cu:306) — live state after row 1 is already the correct
  anchor. a=0: restore = row-0 snapshot + `hidden = pf_x[0]` — exactly the state after
  processing verify row 0, which is the correct next-step anchor when only the pending
  token was truly processed.
- Therefore the single row-0 snapshot is *exactly* the `snap[a]` selector at T=2; nothing
  to change. The generalization (`snap[t]` for t ≤ T-2 inside `deltanet_prefill`'s row
  loop at prefill.cu:258, and per-row conv snapshots in `conv_roll_state_kernel`) is
  spec-deepen list item 2 and is a **hard prerequisite for T=4 rollback** (a=1 and a=2
  restores would read nonexistent snapshots). Documented, deferred.

---

## 3. Host-loop position truth — verification (no patch)

Current state confirmed consistent:

- generate.cu:113-115 rejects `ctx > 4090` with the comment explaining that graph replay
  bypasses the eager KV-full guard (the clamp+overshoot OOB hazard).
- `committed_count()` (decode.cu:203-210) D2H-copies `pos_dev[0]` and refreshes
  `x_.position = pos` — the device-truth refresh is present.
- `spec_graph_step()` does `x_.position += 2` (decode.cu:249) — a pure guess, but nothing
  reads `x_.position` between replays: the graph loop (generate.cu:180-189) runs 4
  replays and then calls `committed_count()`, which corrects the mirror before any
  guard (`prefill_chunk_device`'s KV-full throw at decode.cu:45) can execute. The eager
  path refreshes via `tail[0]` in `spec_step` (decode.cu:226).
- Margin arithmetic: between host checks, 4 replays × ≤2 positions = ≤8 pos growth against
  the +16 ctx margin (`ctx = prompt + max_new + 16`, generate.cu:112) — the committed
  overshoot (≤ 4×2 = 8 ids past `want_total`) is trimmed by generate.cu:196-199. Safe.
  **Flag for T=4: 4 replays × 4 = 16 exactly exhausts the +16 margin** — no slack. When
  T>2 lands, either widen the margin to `4*(D+1)+T` (spec-deepen §4) or drop to 3
  replays/check. No change needed for phase 1 (T=2).
- `spec_second`/`spec_accepted` are populated only in the eager `spec_step`
  (decode.cu:227-231); the graph path never reads them (F8 state confirmed still true).

Conclusion: item 3 is already correct in the tree; the only forward-looking note is the
margin arithmetic above.

---

## 4. D=2 draft chain — groundwork (function + slot math; NOT wired into the graph yet)

### 4.1 mtp_layer refactor (decode.cu lines 133-188)

Rename the body to a parameterized impl; `mtp_layer()` becomes a thin wrapper. Exact
replacement (only the four marked lines differ from the current body — every
`x_.hidden`/`x_.token_dev`/`x_.next_dev` becomes a parameter):

```cpp
void Qwen35Decode::mtp_layer_impl(const int *token_src, float *hid, int *argmax_dst, bool with_head) {
    {   // embed the input token (device side) and rms-norm both inputs              // [was x_.token_dev]
        const std::string base = "language_model.model.embed_tokens";
        auto m = w_.matrix(base);
        if (m.insig4) embed_gather_i4((const uint32_t *)m.weight.data, (const uint16_t *)m.scales.data, token_src, x_.down, 1, x_.stream);
        else embed_gather((const uint32_t *)m.weight.data, (const uint8_t *)m.scales.data, token_src, x_.down, 1, x_.stream);
        w_.release(base);
    }
    auto ew = tensor("language_model.mtp.pre_fc_norm_embedding.weight");
    auto hw = tensor("language_model.mtp.pre_fc_norm_hidden.weight");
    rmsnorm_bf16(x_.down, (const uint16_t *)ew.data, x_.up, 1, 4096, false, x_.stream);
    rmsnorm_bf16(hid, (const uint16_t *)hw.data, x_.norm, 1, 4096, false, x_.stream);   // [was x_.hidden]
    w_.storage().release("language_model.mtp.pre_fc_norm_embedding.weight");
    w_.storage().release("language_model.mtp.pre_fc_norm_hidden.weight");
    concat(x_.up, x_.norm, x_.qkv, 4096, x_.stream);
    {
        auto fc = w_.matrix("language_model.mtp.fc");
        if (fc.insig4) mxfp4_gemv_v2_i4((const uint32_t *)fc.weight.data, (const uint16_t *)fc.scales.data, x_.qkv, hid, 4096, 8192, x_.stream);
        else bf16_gemv((const uint16_t *)fc.weight.data, x_.qkv, hid, 4096, 8192, x_.stream);  // [was x_.hidden]
        w_.release("language_model.mtp.fc");
    }
    const std::string p = "language_model.mtp.layers.0";
    const std::string a = p + ".self_attn";
    auto inw = tensor(p + ".input_layernorm.weight");
    rmsnorm_bf16(hid, (const uint16_t *)inw.data, x_.norm, 1, 4096, false, x_.stream);  // [was x_.hidden]
    w_.storage().release(p + ".input_layernorm.weight");
    linear(a + ".q_proj", x_.norm, x_.gate);
    split_q_gate(x_.gate, x_.qkv, x_.attn_gate, x_.stream);
    linear(a + ".k_proj", x_.norm, x_.key);
    linear(a + ".v_proj", x_.norm, x_.value);
    auto qw = tensor(a + ".q_norm.weight");
    auto kw = tensor(a + ".k_norm.weight");
    qwen35_qk_norm_rope_gate(x_.qkv, x_.key, (const uint16_t *)qw.data, (const uint16_t *)kw.data, x_.attn_gate, x_.mtp_pos_dev, 0, x_.stream);
    w_.storage().release(a + ".q_norm.weight");
    w_.storage().release(a + ".k_norm.weight");
    store_kv(x_.key, x_.value, x_.mtp_keys, x_.mtp_values, x_.mtp_pos_dev, 0, x_.stream);
    gqa_decode(x_.qkv, x_.mtp_keys, x_.mtp_values, x_.core, x_.mtp_pos_dev, 0, x_.max_context, x_.stream);
    expand_gate_heads(x_.attn_gate, x_.qkv, x_.stream);
    sigmoid_mul(x_.core, x_.qkv, 4096, x_.stream);
    linear(a + ".o_proj", x_.core, x_.down);
    residual_add(hid, x_.down, 4096, x_.stream);                                        // [was x_.hidden]
    auto post = tensor(p + ".post_attention_layernorm.weight");
    rmsnorm_bf16(hid, (const uint16_t *)post.data, x_.norm, 1, 4096, false, x_.stream); // [was x_.hidden]
    w_.storage().release(p + ".post_attention_layernorm.weight");
    linear(p + ".mlp.gate_proj", x_.norm, x_.gate);
    linear(p + ".mlp.up_proj", x_.norm, x_.up);
    silu_mul(x_.gate, x_.up, x_.gate, 12288, x_.stream);
    linear(p + ".mlp.down_proj", x_.gate, x_.down);
    residual_add(hid, x_.down, 4096, x_.stream);                                        // [was x_.hidden]
    if (with_head) {
        auto nw = tensor("language_model.mtp.norm.weight");
        rmsnorm_bf16(hid, (const uint16_t *)nw.data, x_.norm, 1, 4096, false, x_.stream);
        w_.storage().release("language_model.mtp.norm.weight");
        linear("language_model.lm_head", x_.norm, x_.logits);
        argmax_fast(x_.logits, Qwen35Shape::vocab, argmax_dst, x_.am_scratch, x_.stream); // [was x_.next_dev]
    }
}
void Qwen35Decode::mtp_layer() { mtp_layer_impl(x_.token_dev, x_.hidden, x_.next_dev, true); }
```

`mtp_layer_impl` is a private method (header, next to `linear2`). The F7b fill (section
1.4) and the chain below are the two consumers of the refactor; everything else is
byte-identical behavior.

### 4.2 The chain function

In-place residual chaining is safe (verified against the impl's dataflow): the second
draft reads R1 from `hid` via `rmsnorm_bf16` (non-destructive read → `x_.norm`), the
concat materializes both fc inputs into `x_.qkv` *before* fc overwrites `hid`, and every
downstream op writes scratch (`x_.gate/x_.key/x_.core/x_.down/...`), never `hid`. So
**no `mtp_hidden[2][4096]` ping-pong is needed for the chain itself**; the only new buffer
is `mtp_fill_h[4096]` (already added in Patch 0), which saves R_D for the full-accept
hole fill because the verify's `hidden <- pf_x[T-1]` copy (decode.cu:102) overwrites
`x_.hidden` before the fill runs.

```cpp
// D=2 draft chain at step pos=P: draft1 @ slot P-1 consumes (embed(pending), h_{P-1});
// draft2 @ slot P consumes (embed(a1), R1) where R1 is mtp's own residual (reference: h_s
// for s>=1 is the MTP residual).  Proposals land in pos slots: a1 -> pos[2] (next_dev,
// F8-compat), a2 -> pos[10] (draft2_dev).  Graph-capturable as-is.
void Qwen35Decode::mtp_chain2() {
    mtp_layer_impl(x_.token_dev, x_.hidden, x_.next_dev, true);    // slot P-1 (prologue set mtp_pos=P-1)
    bumpi_kernel<<<1, 1, 0, x_.stream>>>(x_.mtp_pos_dev);          // slot P
    mtp_layer_impl(x_.next_dev, x_.hidden, x_.draft2_dev, true);   // (embed(a1), R1) -> a2 in pos[10]
    cudaMemcpyAsync(x_.mtp_fill_h, x_.hidden, 4096 * 4, cudaMemcpyDeviceToDevice, x_.stream);  // save R2
}
```

(The mission's `mtp_chain(drafts[2])` host-array shape is deliberately not used: the
engine's device-state regime keeps drafts in `pos_dev` slots so the graph replays without
host iteration; `pos[4]`/`pos[10]` *are* the drafts[2].)

### 4.3 KV slot math + rollback story for D=2 (verified)

Step at pos=P writes slots P-1 (draft1) and P (draft2) — dense within the step. After
commit with accept length a (verify T=3, `pos` correction `-= (T-1-a)`):

| a | pos after | slots written this step | next draft-1 slot | hole? |
|---|---|---|---|---|
| 0 | P+1 | P-1, P (P has wrong-token input) | P (own store_kv rewrites it) | none |
| 1 | P+2 | P-1, P (P exact: a1 accepted, R1 true) | P+1 | none |
| 2 (full) | P+3 | P-1, P | P+2 | **slot P+1 = pos-2** |

- General rule: the hole exists only on full accept, at `pos_after_commit - 2`, and the
  fill invocation is always `bumpi(mtp_pos); mtp_layer_impl(last draft, R_D, -, false)`
  — for D=2, `mtp_pos` ends at P after the chain, bumpi → P+1 = pos-2 ✓ (same one-bump
  rule as D=1; generalizes to any D).
- **MTP KV rollback = append-only, position-pointer only — confirmed, no code needed.**
  `spec_prologue` re-derives `mtp_pos = pos[0]-1` from the committed position every step,
  and stale speculative slots (≥ next draft-1's own slot) are always overwritten by
  store_kv before any gqa read (store_kv precedes gqa_decode in the impl). On a=1,
  draft2's slot-P entry is *correct and kept* (a1 accepted ⇒ inputs true); on a=0 it is
  stale-but-soon-rewritten. The only mutable MTP state besides the cache is
  `mtp_pos_dev` (rewritten by prologue) and `mtp_fill_h` (scratch) — nothing to rewind.

---

## 5. Verify-side T=4 — what breaks today + the commit kernel

### 5.1 What breaks at T=4 in `prefill_chunk_device` today

1. **lm_head branch (decode.cu:93-101)**: only T==2 (pair, both rows) and the
   last-row-only GEMV else-branch exist. T=4 verify needs all 4 rows' logits + 4 argmaxes.
   Patch below.
2. **Rollback snapshots**: reject with a ∈ {1,2} needs `snap[1]`/`snap[2]` which don't
   exist (section 2). **Hard correctness blocker for T=4** — commit kernel below is
   correct independently, but T=4 cannot run before per-row snapshots land.
3. **`pair` flag**: `const bool pair = T==2` (decode.cu:50) — false at T=4, everything
   routes to `linear_batch` automatically. Correct today: MLX path pads A to 64 rows in
   the v21 kernel (compute-bound padding, spec-deepen §3 — perf loss on every GEMM, not
   a correctness issue); INSIG4 path uses `mxfp4_gemm_mlx_i4` (correct, unpipelined).
   The `<MT=16>` i4/v21 port is the perf dependency; correctness does not wait on it.
4. Everything else is T-generic already: `pf_*` buffers are 64 rows, `addi_kernel` adds
   T, `store_kv_batch`/`gqa_prefill`/conv/deltanet take T, am_scratch is reused
   sequentially by consecutive argmax calls (stream-ordered).

### 5.2 lm_head T-batch patch (decode.cu lines 93-101)

Replace the `else` branches (keep the T==2 pair branch verbatim):

```cpp
  } else if (T <= 8) {  // speculative verify width: all rows' logits in one weight pass
   linear_batch("language_model.lm_head", x_.pf_n, x_.logits, T);   // needs logits >= [8][vocab] (Patch 0)
   argmax_fast(x_.logits,                       Qwen35Shape::vocab, x_.next2_dev, x_.am_scratch, x_.stream);  // t*_0 -> pos[3]
   argmax_fast(x_.logits + Qwen35Shape::vocab,   Qwen35Shape::vocab, x_.next_dev,  x_.am_scratch, x_.stream);  // t*_1 -> pos[2]
   argmax_fast(x_.logits + 2 * size_t(Qwen35Shape::vocab), Qwen35Shape::vocab, x_.tstar2_dev, x_.am_scratch, x_.stream);  // t*_2 -> pos[8]
   if (T == 4)
    argmax_fast(x_.logits + 3 * size_t(Qwen35Shape::vocab), Qwen35Shape::vocab, x_.tstar3_dev, x_.am_scratch, x_.stream);  // t*_3 -> pos[9]
  } else if (m.insig4) { ... existing last-row gemv + argmax unchanged (prompt path) ... }
```

Guard `linear_batch` output size: lm_head rows 248320 × T ≤ 8 fits the widened logits
buffer; keep `T > 8` on the existing last-row path so 64-row prompt/NLL chunks never
touch it. (Later micro-opt, not phase 1: one batched `argmax_rows` kernel — 4 sequential
argmax launches cost ~60 µs.)

### 5.3 pos_dev slot layout for T=4 / D=3

**16 ints are enough** (13 used). Layout (existing slots 0-7 untouched — full backward
compatibility with eager printouts and F8):

| slot | alias | role |
|---|---|---|
| 0 | pos | next unprocessed main position P |
| 1 | pending | token occupying position P |
| 2 | next_dev | t*_1 (row-1 argmax; legacy "after") |
| 3 | next2_dev | t*_0 (row-0 argmax; legacy t2) |
| 4 | draft_dev | draft1 (= pos[2] copy by setup, F8 compat) |
| 5 | count_dev | committed cursor |
| 6 | accflag_dev | **accept length a** (0..3; T-1 == full) |
| 7 | mtp_pos_dev | MTP slot |
| 8 | tstar2_dev | t*_2 |
| 9 | tstar3_dev | t*_3 (bonus token on full accept) |
| 10 | draft2_dev | draft2 (chain argmax dst) |
| 11 | draft3_dev | draft3 |
| 12-15 | — | free (chain-token scratch; widen to 40 ints when D>3 — one-line alloc change) |

### 5.4 `spec_setup4` and `spec_commit4` (prefill.cu, after `spec_commit` ~line 302)

```cpp
__global__ void spec_setup4_kernel(int *__restrict__ pos, int *__restrict__ pf_tokens) {
    pos[4] = pos[2];                 // draft1 (compat with the eager printout / F8)
    pf_tokens[0] = pos[1];           // pending
    pf_tokens[1] = pos[4];           // a1
    pf_tokens[2] = pos[10];          // a2
    pf_tokens[3] = pos[11];          // a3
}
void spec_setup4(int *pos, int *pf_tokens, cudaStream_t stream) { spec_setup4_kernel<<<1, 1, 0, stream>>>(pos, pf_tokens); }

// Greedy chain accept over T=4 rows (D=3).  a = longest prefix with draft_{i+1} == t*_i.
// 1 thread, serial loop (D<=8 per spec-deepen §4).  pos[0] was advanced by T=4 in the
// chunk; the true advance is 1+a.  pos[6] carries a for the restore selector (T-1==full).
__global__ void spec_commit4_kernel(int *__restrict__ pos, int *__restrict__ committed) {
    const int ts[4] = {pos[3], pos[2], pos[8], pos[9]};   // t*_0..t*_3
    const int dr[3] = {pos[4], pos[10], pos[11]};         // a1..a3
    int a = 0;
    while (a < 3 && dr[a] == ts[a]) a++;
    const int c = pos[5];
    committed[c] = pos[1];                                // pending is committed first
    if (a >= 1) committed[c + 1] = dr[0];
    if (a >= 2) committed[c + 2] = dr[1];
    if (a >= 3) committed[c + 3] = dr[2];
    pos[1] = ts[a];                                       // correction (a<3) or bonus (a==3)
    pos[5] = c + 1 + a;
    pos[0] += 1 + a - 4;                                  // addi added T=4
    pos[6] = a;
}
void spec_commit4(int *pos, int *committed, cudaStream_t stream) { spec_commit4_kernel<<<1, 1, 0, stream>>>(pos, committed); }
```

Greedy-exactness (unchanged in kind): every emitted token is either a verify-row argmax
over the exact committed prefix (`ts[a]`) or a draft confirmed equal to that argmax; the
emitted stream equals the greedy stream independent of D, T, and MTP KV state.

`spec_rollback4` (design, **blocked on per-row snapshots** — do not land before
spec-deepen item 2):

```cpp
__global__ void spec_rollback4_kernel(const float *__restrict__ snap_delta, const float *__restrict__ snap_conv,  // [3][...] each
                                      float *__restrict__ delta_state, float *__restrict__ conv_state,
                                      const float *__restrict__ pf_x, float *__restrict__ hidden, const int *__restrict__ pos) {
    const int a = pos[6];
    if (a == 3) return;                                                     // full accept: live state correct
    const int n = 24 * 32 * 128 * 128;
    const float *sd = snap_delta + size_t(a) * n, *sc = snap_conv + size_t(a) * (24 * 8192 * 3);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) delta_state[i] = sd[i];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 24 * 8192 * 3; i += gridDim.x * blockDim.x) conv_state[i] = sc[i];
    const float *h = pf_x + size_t(a) * 4096;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 4096; i += gridDim.x * blockDim.x) hidden[i] = h[i];
}
```

(`deltanet_prefill_kernel`'s snapshot line prefill.cu:258 generalizes to
`if (t <= T-2 && snap)` writing `snap + t*n`; `conv_roll_state_kernel` emits one
`[s1,s2,x_t]` snapshot per row 0..T-2. Snap buffer cost at T=4: 3×50.3 MB + 3×2.36 MB.)

Host-side notes when T=4 goes live: ctx margin must grow (section 3 flag); the graph
capture becomes `prologue → mtp_chain(D) → spec_setup_T → prefill_chunk_device(T) →
commit_T → rollback_T → hole-fill`, T and D compile-time constants of the capture.

---

## 6. Validation plan (per AGENTS.md: measurement + parity before believing)

1. **F7a parity**: `generate.exe <idx> <prompt> 8 probe` after the fill lands — probe
   draft at slot N-1 now attends the filled cache; compare against
   `tools/reference_multistep.py` teacher-forced `mtp_draft` argmax at the same slot
   (ids should match; cosine on k/v optional via a seam-style dump).
2. **Multistep greedy equality**: eager mode (`argv[4]=eager`) committed ids == graph
   mode ids == reference greedy chain, before/after each patch. The F7b exact fill makes
   engine drafts reference-equal for the first time in spec mode.
3. **Acceptance rate**: re-measure p (tokens/step) pre/post F7. spec-deepen predicts p
   rises above 0.6 once drafts stop attending garbage; if so the (D,T) table shifts up.
4. **Perf gate**: spec step must not regress beyond the fill's ~0.33 ms amortized (13.1 →
   ~13.4 ms); prefill +~0.6 ms per 64-token chunk. Zero-patch v1 is the fallback if the
   exact fill's cost measures worse than its acceptance gain.

## 7. Ordered edit list

| # | file | edit | lines |
|---|---|---|---|
| 1 | decode.cu | mtp cache init memsets; logits 2→8 rows; `mtp_fill_h` alloc/free; pos aliases 8..11 | 14, 15, 27, 29 |
| 2 | insignia_decode.hpp | new members + method decls (`mtp_layer_impl`, `mtp_prefill_chunk`, `prefill_chunk_mtp`, `mtp_chain2`) | 9, 11, 42-49 |
| 3 | prefill.cu | `concat_batch` kernel | after 40 |
| 4 | gemm.cu + insignia_layout.cuh | `bf16_gemm_tn` (MLX fc batch path) | end / after 64 |
| 5 | decode.cu | `mtp_prefill_chunk` + `prefill_chunk_mtp` | after 117 |
| 6 | generate.cu | prompt loop → `prefill_chunk_mtp`; delete probe memsets | 125-129, 142-143 |
| 7 | decode.cu | `mtp_layer` → `mtp_layer_impl` refactor + wrapper | 133-188 |
| 8 | decode.cu | F7b fill wired into `spec_step` + `capture_spec` (after rollback) | after 222 / 241 |
| 9 | prefill.cu + .cuh | `spec_mtp_patch` (v1 fallback, optional) | after 302 |
| 10 | decode.cu | lm_head T≤8 batch branch | 93-101 |
| 11 | prefill.cu + .cuh | `spec_setup4`, `spec_commit4` (groundwork; unwired) | after 302 |
| 12 | — | per-row snapshots + `spec_rollback4` + chain wiring: **phase 2** (spec-deepen items 2-3) | — |

Edits 1-9 are the phase-1 land (F7 complete + refactor); 10-11 are compile-only
groundwork until snapshots arrive.
