# W3: gated-DeltaNet @ 27B — full kernel rewrite pack

Scope: every kernel touching the DeltaNet path for Qwen3.8-27B (48 linear layers),
paste-ready. Verified against: `src/deltanet.cu`, `src/prefill.cu`, `src/qwen_kernels.cu`,
`src/decode.cu`, `src/test_deltanet.cu`, `tools/reference_layer0.py`,
`tools/quantize_insig4.py`, `audits/w2/loader-27b-spec.md`, `audits/w2/shape-constants.md`.

## 0. Ground truth (verified this session)

| fact | value | source |
|---|---|---|
| v-heads / k-heads / q-heads | 48 / 16 / 16 | loader census `linear_num_value_heads 48`, key 16 |
| v:k ratio | **3:1** → `kh = head/3` | heads 0..47 → kh 0..15 |
| head dims | k=128, v=128 (unchanged) | config |
| qkv rows / sections | 10240 = q[0,2048) k[2048,4096) v[4096,10240) | in_proj_qkv [10240,5120]; q/k = 16x128 each, v = 48x128 |
| z rows | 6144 (= 48x128) | in_proj_z [6144,5120] |
| a/b rows | 48 each | in_proj_a/b [48,5120] BF16 |
| conv channels | 10240 (= qkv rows), width 4 | conv1d [10240,1,4] BF16 |
| state / layer | [48][128][128] f32 = 3,145,728 B = 3.146 MB | mamba_ssm_dtype float32 |
| **A_log dtype** | **BF16 [48]** (27B native FP8 ckpt) | loader census layers-0: `linear_attn.A_log BF16 [48] 96 B` |
| **dt_bias dtype** | **BF16 [48]** | census: `linear_attn.dt_bias BF16 [48] 96 B` |
| A_log dtype 9B INSIG4 | **F32** (quantizer special case) | `tools/quantize_insig4.py:131` "deltanet kernel reads A_log as float32"; dt_bias falls into the generic BF16 emit branch |
| delta layers | 48 (l%4!=3 of 64) | shape-constants |
| per-layer state/snap totals | delta 48*48*128*128 = 37,748,736 f32 = 150,994,944 B ≈ 151 MB; conv 48*10240*3 = 1,474,560 f32 ≈ 5.9 MB | — |

**The one dtype trap:** current `deltanet_parameters` / `deltanet_params_batch` read A_log
as `const float*` (F32, 9B INSIG4). The 27B checkpoint stores BF16. Reinterpreting BF16 as
F32 gives garbage decay (bf16 `0.02`-ish values as f32 are ~1e-40 → alpha ≈ 1, no forgetting
→ state saturates, silent). Fix in §4: kernels take `const void*` + `bool a_f32` and widen
at read (48 elements; in-kernel widen beats a load-time staging buffer — zero extra VRAM,
zero extra pass). `DeviceView` already carries `DType` (f32=1, bf16=2) so call sites
dispatch with one compare.

**State orientation (classic parity trap — stated explicitly).** Both GPU kernels index
`S[i*128 + tid]` where `i` is the **key** index and `tid` the **value** index
(src/deltanet.cu:10-12, src/prefill.cu:247-255). So:

- GPU/global state layout: **`S[k*128 + v]` — 128 rows of k, each row 128 contiguous v's.
  Row = key dim, column = value dim. Per head 64 KB.**
- Recurrence in this layout: `mem[v] = Σ_k S[k][v]·(α·k̂[k])`; `S[k][*] = α·S[k][*] + k̂[k]·δ[*]`
  with `δ = (v − mem)·β`; `out[v] = Σ_k S[k][v]·q̂[k]` (over the **updated** S).
- The NumPy reference (`tools/reference_layer0.py`) uses `state[h][v][k]` — the **transpose**
  (v-major). Fine standalone; fatal if mixed. `test_deltanet.cu` uses `s[(h*K+i)*V+j]` = the
  GPU layout, so the C test matches the kernel. The CPU variant (§6) must use the GPU layout.

**Alpha/softplus (mission item 7, verified).** Reference:
`g = exp(-exp(A_log) * logaddexp(0, a + dt_bias))` (reference_layer0.py). Kernel chain:
`params` kernel computes `b = sigmoid(b); z = a + bf16(dt_bias); soft = z>20 ? z : log1pf(expf(z)); a_out = -expf(A_log) * soft` — `logaddexp(0,x)` **is** softplus, and the `z>20`
linear branch has < 2.1e-9 absolute error (log1p(expf) would overflow past z≈88 anyway).
Then both scan kernels do `decay = expf(a_head)`. Composite:
`α = exp(−exp(A_log)·softplus(a+dt_bias))` — exact match, **f32 throughout** (a/b are f32
workspace buffers; bf16 inputs widened at load; `log1pf`/`expf`/`fmaf` all f32). No change
needed; formula kept verbatim in the new kernels below. (Note: shape-constants.md:8 calls
the q-norm constant "1/sqrt(128)/sqrt(2)" — it is exactly **1/sqrt(128) = 0.08838834764831845**;
verified against the reference's `q/(sqrt(mean(q²)+1e-6)·128)` == `q·rsqrt(Σq²+1e-6)/sqrt(128)`.)

---

## 1. `src/deltanet.cu` — decode kernel (full file)

Changes: `kh = head>>1` → `head/3`; launch `<<<32,128>>>` → `<<<48,128>>>`. Everything else
is shape-invariant: q/k reads `q16[kh*128+tid]`, `k16[kh*128+i]` (sections still 2048 rows =
16 k-heads), v read `v[head*128+tid]` (v base = `qkv+4096` passed by caller, head now 0..47
→ touches [4096,10240) of qkv = exactly the new v section), out `out[head*128+tid]`
(6144-long core buffer).

Per-block layout audit (mission 1): one block = one v-head, 128 threads (4 warps),
`__launch_bounds__(128,4)`. State lives in **global** memory (`state + head*128*128`, 64 KB
per head — NOT staged through smem in decode). Static smem = `sq[4]+sk[4]+delta[128]` =
544 B, no dynamic smem. Coalescing: threads read column `S[i*128+tid]` — a stride-128
gather per i (128 floats per row-hot pass), same as 9B; the fused single-pass update over
global state is the accepted decode design (state r/w 3.146+3.146 MB dominates; ~0.012 ms
at 504 GB/s).

```cuda
#include "insignia_deltanet.cuh"
#include <cuda_runtime.h>
namespace insignia {
// 27B: 48 v-heads over 16 k/q-heads -> kh = head/3 (9B was 2:1, head>>1 — WRONG at 27B).
// q16/k16 sections are still 2048 rows (16 heads x 128); v is 48*128 = 6144 rows.
// State per head: S[k*128+v] — 128 key rows x 128 value cols, f32, 64KB in GLOBAL memory.
// smem per block: 544 B static (sq/sk norm scratch + delta[128]); no dynamic smem.
__global__ __launch_bounds__(128,4) void deltanet_decode_kernel(float *__restrict__ state,const float *__restrict__ q16,const float *__restrict__ k16,const float *__restrict__ v,const float *__restrict__ g,const float *__restrict__ beta,float *__restrict__ out){
 const int head=blockIdx.x,tid=threadIdx.x,kh=head/3; float q=q16[kh*128+tid],k=k16[kh*128+tid];float qn=q*q,kn=k*k;
 for(int m=16;m;m>>=1){qn+=__shfl_xor_sync(0xffffffff,qn,m);kn+=__shfl_xor_sync(0xffffffff,kn,m);}
 __shared__ float sq[4],sk[4],delta[128];const int lane=tid&31,warp=tid>>5;if(lane==0){sq[warp]=qn;sk[warp]=kn;}__syncthreads();
 if(tid==0){float a=sq[0]+sq[1]+sq[2]+sq[3],b=sk[0]+sk[1]+sk[2]+sk[3];sq[0]=rsqrtf(a+1e-6f)*0.08838834764831845f;sk[0]=rsqrtf(b+1e-6f);}__syncthreads();q*=sq[0];k*=sk[0];
 float *S=state+head*128*128;const float decay=expf(g[head]);
 float dot=0;for(int i=0;i<128;i++)dot=fmaf(S[i*128+tid]*decay,k16[kh*128+i]*sk[0],dot);
 delta[tid]=(v[head*128+tid]-dot)*beta[head];__syncthreads();
 float acc=0;for(int i=0;i<128;i++){float &cell=S[i*128+tid];cell=fmaf(cell,decay,k16[kh*128+i]*sk[0]*delta[tid]);acc=fmaf(cell,q16[kh*128+i]*sq[0],acc);}out[head*128+tid]=acc;
}
void deltanet_decode(float*s,const float*q,const float*k,const float*v,const float*g,const float*b,float*o,cudaStream_t stream){deltanet_decode_kernel<<<48,128,0,stream>>>(s,q,k,v,g,b,o);}
}
```

Header `include/insignia_deltanet.cuh` (note: kernels hardcode 48 inline — the constant is
documentation/single-source material, not yet wired):

```cpp
#pragma once
#include <cuda_runtime.h>
namespace insignia {
constexpr int DELTA_HEADS=48, DELTA_QK_HEADS=16, DELTA_K=128, DELTA_V=128;  // 27B: 48 v / 16 k, kh=head/3
void deltanet_decode(float *state, const float *q16, const float *k16, const float *v48, const float *g48, const float *beta48, float *out48, cudaStream_t stream=nullptr);
}
```

---

## 2. `src/prefill.cu` — deltanet prefill scan (kernel + launcher)

Launch shape stays **one block per v-head, block scans T tokens sequentially**:
`<<<48,128,64*1024+512,stream>>>`. It is *not* `<<<T,…>>>` or `T*48` — T-parallelism is
impossible (recurrence). The `<<<T,48>>>` shape belongs to `params_batch_kernel` (§4).
Changes: `kh=head>>1`→`head/3`; qkv stride `t*8192`→`t*10240` (offsets 2048/4096 stay —
q/k rows unchanged, v section now 6144 long); a/b rows `t*32`→`t*48`; out
`(t*32+head)*128`→`(t*48+head)*128`.

**smem audit (mission 2): unchanged and still fits.** Dynamic request stays
`64*1024+512 = 66,048 B` (128x128 f32 state staging + 512 pad) + 544 B static
(`sq/sk/delta`). sm_89 opt-in per-block cap is 99 KB (`cudaFuncAttributeMaxDynamicSharedMemorySize`
already set in the launcher) → 66.6 KB total fits with margin. Per-head state row staging
is untouched because head dims stay 128/128. Occupancy: 66 KB/block → 1 block/SM (100 KB
smem/SM on AD104); 48 blocks ≤ 56 SMs = exactly 1 wave, same story as 9B's 32 blocks.

Max out index check: `(63*48+47)*128+127 = 393,215 < 64*6144` pf_core — exact fit.

```cuda
// Sequential gated DeltaNet over T tokens; one block per value head, state in shared.
// qkv layout per token (27B): q[16][128] | k[16][128] | v[48][128] = 10240; a/b [48]/token.
// kh = head/3: 48 v-heads share 16 k/q heads (27B 3:1; 9B was 2:1).
// After the first token the state is also written to `snap` (speculative rollback point).
__global__ __launch_bounds__(128) void deltanet_prefill_kernel(float *__restrict__ state, const float *__restrict__ qkv, const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ out, int T, float *__restrict__ snap) {
    extern __shared__ float sh[];  // [128*128] state staging + [512 B] slack
    const int head = blockIdx.x, tid = threadIdx.x, kh = head / 3;
    float *S = sh;
    float *gstate = state + static_cast<size_t>(head) * 128 * 128;
    for (int i = tid; i < 128 * 128; i += 128) S[i] = gstate[i];
    __syncthreads();
    for (int t = 0; t < T; t++) {
        const float *qt = qkv + static_cast<size_t>(t) * 10240 + kh * 128;
        const float *kt = qkv + static_cast<size_t>(t) * 10240 + 2048 + kh * 128;
        const float *vt = qkv + static_cast<size_t>(t) * 10240 + 4096 + head * 128;
        float q = qt[tid], k = kt[tid];
        float qn = q * q, kn = k * k;
        for (int m = 16; m; m >>= 1) { qn += __shfl_xor_sync(0xffffffff, qn, m); kn += __shfl_xor_sync(0xffffffff, kn, m); }
        __shared__ float sq[4], sk[4], delta[128];
        const int lane = tid & 31, warp = tid >> 5;
        if (lane == 0) { sq[warp] = qn; sk[warp] = kn; }
        __syncthreads();
        if (tid == 0) {
            const float qa = sq[0] + sq[1] + sq[2] + sq[3], ka = sk[0] + sk[1] + sk[2] + sk[3];
            sq[0] = rsqrtf(qa + 1e-6f) * 0.08838834764831845f;
            sk[0] = rsqrtf(ka + 1e-6f);
        }
        __syncthreads();
        q *= sq[0]; k *= sk[0];
        const float decay = expf(a[static_cast<size_t>(t) * 48 + head]);
        const float beta = b[static_cast<size_t>(t) * 48 + head];
        float dot = 0;
        for (int i = 0; i < 128; i++) dot = fmaf(S[i * 128 + tid] * decay, kt[i] * sk[0], dot);
        delta[tid] = (vt[tid] - dot) * beta;
        __syncthreads();
        float acc = 0;
        for (int i = 0; i < 128; i++) {
            float &cell = S[i * 128 + tid];
            cell = fmaf(cell, decay, kt[i] * sk[0] * delta[tid]);
            acc = fmaf(cell, qt[i] * sq[0], acc);
        }
        out[(static_cast<size_t>(t) * 48 + head) * 128 + tid] = acc;
        __syncthreads();
        if (t == 0 && snap) {  // committed row-0 checkpoint for speculative rollback
            float *gsnap = snap + static_cast<size_t>(head) * 128 * 128;
            for (int i = tid; i < 128 * 128; i += 128) gsnap[i] = S[i];
        }
    }
    for (int i = tid; i < 128 * 128; i += 128) gstate[i] = S[i];
}
void deltanet_prefill(float *state, const float *qkv, const float *a, const float *b, float *out, int T, cudaStream_t stream, float *row0_snap) {
    static const bool ok = [] { return cudaFuncSetAttribute(deltanet_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024 + 512) == cudaSuccess; }();
    (void)ok;
    deltanet_prefill_kernel<<<48, 128, 64 * 1024 + 512, stream>>>(state, qkv, a, b, out, T, row0_snap);
}
```

Header `include/insignia_prefill.cuh` line 13-14 becomes (params_batch signature change from §4):

```cpp
void deltanet_params_batch(float *a, float *b, const void *A_log, bool a_log_f32, const uint16_t *dt_bias, int T, cudaStream_t stream = nullptr);
void deltanet_prefill(float *state, const float *qkv, const float *a, const float *b, float *out, int T, cudaStream_t stream = nullptr, float *row0_snap = nullptr);
```

---

## 3. Conv path — prefill kernels + DECODE-side update

### 3a. `src/prefill.cu` conv prefill (both kernels + launcher, full)

Changes: every 8192 → 10240. Layout unchanged: conv_state per layer = `[10240][3]` f32,
`state[c*3+i]` = i-th history input of channel c; snap per layer `[10240][3]` at
`snap_conv + di*10240*3`. Rollback semantics preserved: snap = `[s1, s2, x_row0]` = the
conv state as of *after* consuming row 0 (correct restart point when the speculative row 1
is rejected). `int idx` max = 64*10240 = 655,359 — fits int; `t*10240+c` kept `size_t`.

```cuda
// Depthwise causal conv4 + SiLU over T tokens; state carries the 3 pre-chunk inputs.
// Output goes to a separate buffer: x must stay raw until every thread has read it.
// Also snapshots the state as of after the FIRST row (spec-decode rollback point).
__global__ void conv_prefill_kernel(const float *__restrict__ x, float *__restrict__ out, const float *__restrict__ state, const uint16_t *__restrict__ w) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // t*10240 + c
    const int t = idx / 10240, c = idx % 10240;
    float z = 0;
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        const float hist = t >= 3 - i ? x[static_cast<size_t>(t - (3 - i)) * 10240 + c] : state[c * 3 + i];
        z = fmaf(hist, __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(w + c * 4 + i)), z);
    }
    z = fmaf(x[static_cast<size_t>(t) * 10240 + c], __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(w + c * 4 + 3)), z);
    out[idx] = z / (1.f + __expf(-z));
}
// Roll the last 3 raw inputs of the combined [state, x] window into the state, and
// write the row-0 checkpoint [s1, s2, x0] for speculative rollback.
__global__ void conv_roll_state_kernel(const float *__restrict__ x, float *__restrict__ state, float *__restrict__ snap, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 10240) return;
    snap[c * 3 + 0] = state[c * 3 + 1];
    snap[c * 3 + 1] = state[c * 3 + 2];
    snap[c * 3 + 2] = x[c];  // row 0's raw input
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        const int j = T - 3 + i;
        state[c * 3 + i] = j >= 0 ? x[static_cast<size_t>(j) * 10240 + c] : state[c * 3 + 3 + j];
    }
}
void conv_prefill_silu(float *x, float *scratch, float *state, const uint16_t *w, int T, cudaStream_t stream, float *row0_snap) {
    const int n = T * 10240;
    conv_prefill_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, scratch, state, w);
    conv_roll_state_kernel<<<(10240 + 255) / 256, 256, 0, stream>>>(x, state, row0_snap ? row0_snap : state, T);
}
```

### 3b. Decode-side conv update — `causal_conv4_silu` (qwen_kernels.cu)

Found it: decode does **not** have its own conv kernel — `Qwen35Decode::delta_layer`
(decode.cu:124) calls `causal_conv4_silu(x_.qkv, x_.conv_state + di*8192*3, w, 8192, stream)`
which launches the generic `conv4` kernel in src/qwen_kernels.cu:7-8 (grid-stride over `n`,
state `[i][3]`, weight `[i*4+tap]`). **The kernel body is n-generic — no edit needed**;
pasted for verification. Only the call site literals change (8192 → 10240, both the channel
count and the per-layer state stride).

```cuda
// src/qwen_kernels.cu: UNCHANGED at 27B (n-generic); decode.cu now passes n=10240.
__global__ void conv4(float*x,float*state,const uint16_t*w,int n){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){float z=fmaf(state[i*3],bf(w+i*4),fmaf(state[i*3+1],bf(w+i*4+1),fmaf(state[i*3+2],bf(w+i*4+2),x[i]*bf(w+i*4+3))));state[i*3]=state[i*3+1];state[i*3+1]=state[i*3+2];state[i*3+2]=x[i];x[i]=z/(1+__expf(-z));}}
void causal_conv4_silu(float*x,float*st,const uint16_t*w,int n,cudaStream_t s){conv4<<<(n+255)/256,256,0,s>>>(x,st,w,n);}
```

decode.cu:124 delta_layer call-site diff (conv part only):

```cuda
// OLD: causal_conv4_silu(x_.qkv,x_.conv_state+size_t(di)*8192*3,(const uint16_t*)cw.data,8192,x_.stream);
      causal_conv4_silu(x_.qkv,x_.conv_state+size_t(di)*10240*3,(const uint16_t*)cw.data,10240,x_.stream);
```

State layout cross-check (prefill vs decode): both use `state[c*3+i]` channel-major with
the same roll order and the same weight indexing — conv snapshots taken by the prefill roll
are byte-compatible with the decode kernel. No 27B divergence beyond width.

---

## 4. `deltanet_parameters` (+ prefill `deltanet_params_batch`) — n=48 + A_log dtype fix

27B census: A_log **BF16** [48], dt_bias **BF16** [48]. 9B INSIG4: A_log **F32**
(quantize_insig4.py:131), dt_bias BF16. Fix: pass `const void*` + `bool a_log_f32`,
widen in-kernel (48 elements — free). Decode kernel launches `<<<1,48>>>` (48 threads =
1.5 warps, legal, guard `i<n` kept); prefill kernel launches `<<<T,48>>>` (one block per
token, one thread per head, guard `h>=48`).

`src/qwen_kernels.cu` — replace lines 9-10:

```cuda
// 27B: 48 heads; A_log dtype dispatch — 9B INSIG4 emitted F32, 27B native checkpoint is BF16.
// Widen in-kernel (48 elems): alpha = -exp(A_log)*softplus(a+dt_bias), beta = sigmoid(b), f32 throughout.
template<bool AF32>__global__ void params(float*a,float*b,const void*A,const uint16_t*dt,int n){int i=threadIdx.x;if(i<n){const float Alog=AF32?reinterpret_cast<const float*>(A)[i]:bf(reinterpret_cast<const uint16_t*>(A)+i);b[i]=1/(1+__expf(-b[i]));float z=a[i]+bf(dt+i);float soft=z>20?z:log1pf(__expf(z));a[i]=-__expf(Alog)*soft;}}
void deltanet_parameters(float*a,float*b,const void*A_log,bool a_log_f32,const uint16_t*dt,int n,cudaStream_t s){if(a_log_f32)params<true><<<1,48,0,s>>>(a,b,A_log,dt,n);else params<false><<<1,48,0,s>>>(a,b,A_log,dt,n);}
```

`src/prefill.cu` — replace `params_batch_kernel` + `deltanet_params_batch` (lines 203-214):

```cuda
// Per-token decay/beta: a[t][h] = -exp(A_log)*softplus(a+dt), b[t][h] = sigmoid(b). 48 heads.
// A_log BF16 in the 27B checkpoint, F32 in the 9B INSIG4 re-quant — dispatched by a_log_f32.
template<bool AF32>__global__ void params_batch_kernel(float*a,float*b,const void*A,const uint16_t*dt,int T){
    const int t=blockIdx.x,h=threadIdx.x;
    if(h>=48)return;
    float*ar=a+static_cast<size_t>(t)*48,*br=b+static_cast<size_t>(t)*48;
    br[h]=1.f/(1.f+__expf(-br[h]));
    const float Alog=AF32?reinterpret_cast<const float*>(A)[h]:__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint16_t*>(A)+h));
    const float z=ar[h]+__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(dt+h));
    const float soft=z>20?z:log1pf(__expf(z));
    ar[h]=-__expf(Alog)*soft;
}
void deltanet_params_batch(float*a,float*b,const void*A_log,bool a_log_f32,const uint16_t*dt_bias,int T,cudaStream_t stream){
    if(a_log_f32)params_batch_kernel<true><<<T,48,0,stream>>>(a,b,A_log,dt_bias,T);
    else params_batch_kernel<false><<<T,48,0,stream>>>(a,b,A_log,dt_bias,T);
}
```

Header `include/insignia_qwen_kernels.cuh` line 8:

```cpp
void deltanet_parameters(float*a,float*b,const void*A_log,bool a_log_f32,const uint16_t*dt_bias,int heads,cudaStream_t stream=nullptr);
```

(Alternative — rejected: convert A_log BF16→F32 once at load into a staging tensor. Costs
48 f32 * 48 layers = 9.2 KB VRAM + a loader branch per dtype; the in-kernel widen is free
and keeps mmap zero-copy. The dispatch bool also future-proofs a re-quantized 27B INSIG4
that would emit F32 again.)

---

## 5. Snapshot / rollback + DecodeWorkspace

### 5a. `src/prefill.cu` `spec_rollback_kernel` (full) — sizes + a latent 9B bug fixed

**Latent bug found:** old line 310 `if (blockIdx.x == 0 && threadIdx.x < 4096) hidden[threadIdx.x] = pf_x[threadIdx.x];`
runs in a 256-thread block, so `threadIdx.x < 4096` is always true and **only hidden[0..255]
of 4096 were restored**. Survived because a corrupted `hidden` only feeds the MTP draft
(next `mtp_layer`), which degrades acceptance rate, never correctness. Fixed with a
grid-stride copy (also the 27B 5120 width).

```cuda
// Undo the speculative row's recurrent state (accepted steps early-exit everywhere).
__global__ void spec_rollback_kernel(const float *__restrict__ snap_delta, const float *__restrict__ snap_conv, float *__restrict__ delta_state, float *__restrict__ conv_state, const float *__restrict__ pf_x, float *__restrict__ hidden, const int *__restrict__ pos) {
    if (pos[6]) return;
    const int n = 48 * 48 * 128 * 128;  // 37,748,736 f32 = 151 MB (was 24*32*128*128)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) delta_state[i] = snap_delta[i];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 48 * 10240 * 3; i += gridDim.x * blockDim.x) conv_state[i] = snap_conv[i];  // 1,474,560 (was 24*8192*3)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 5120; i += gridDim.x * blockDim.x) hidden[i] = pf_x[i];  // FULL row-0 hidden (was 256-of-4096 partial copy — latent 9B bug)
}
void spec_rollback(const float *snap_delta, const float *snap_conv, float *delta_state, float *conv_state, const float *pf_x, float *hidden, const int *pos, cudaStream_t stream) {
    spec_rollback_kernel<<<512, 256, 0, stream>>>(snap_delta, snap_conv, delta_state, conv_state, pf_x, hidden, pos);
}
```

(512 blocks x 256 = 131,072 threads; delta restore moves ~288 elements/thread — bandwidth
is the limit, 302 MB of copies on reject. Optionally raise to `<<<1024,256>>>`; frozen into
`capture_spec` either way — **graphs must be re-captured** after any of these edits.)

### 5b. `src/decode.cu` DecodeWorkspace ctor — alloc/memset lines (delta/conv-relevant; attention/MLP literals included for line consistency, they belong to the other W3 waves)

```cuda
// line 14
 alloc(&hidden,5120);alloc(&norm,5120);alloc(&qkv,10240);alloc(&attn_gate,6144);alloc(&key,1024);alloc(&value,1024);alloc(&z,6144);alloc(&a,48);alloc(&b,48);alloc(&core,6144);alloc(&gate,17408);alloc(&up,17408);alloc(&down,5120);alloc(&logits,248320*2);alloc(&delta_state,48*48*128*128);alloc(&conv_state,48*10240*3);alloc(&kv_keys,size_t(16)*ctx*1024);alloc(&kv_values,size_t(16)*ctx*1024);alloc(&mtp_keys,size_t(ctx)*1024);alloc(&mtp_values,size_t(ctx)*1024);
// lines 22-25
 alloc(&pf_x,64*5120);alloc(&pf_n,64*5120);alloc(&pf_qkv,64*10240);alloc(&pf_scratch,64*12288);alloc(&pf_z,64*6144);
 alloc(&pf_q,64*6144);alloc(&pf_g,64*6144);alloc(&pf_k,64*1024);alloc(&pf_v,64*1024);alloc(&pf_core,64*6144);
 alloc(&pf_down,64*5120);alloc(&pf_gate,64*17408);alloc(&pf_up,64*17408);alloc(&pf_a,64*48);alloc(&pf_b,64*48);
 alloc(&snap_delta,48*48*128*128);alloc(&snap_conv,48*10240*3);
// line 27 (memsets track the allocs)
 cudaMemsetAsync(pos_dev,0,16*sizeof(int),stream);cudaMemsetAsync(am_scratch,0,8,stream);cudaMemsetAsync(delta_state,0,48*48*128*128*4,stream);cudaMemsetAsync(conv_state,0,48*10240*3*4,stream);
```

DeltaNet-relevant traps called out: `z`/`core` are **6144** (=48*128, NOT hidden 5120);
`a`/`b`/`pf_a`/`pf_b` are 48; `snap_delta` = `snap_conv` sizes mirror `delta_state`/`conv_state`;
memset byte counts must track. VRAM cost of the snapshot pair: 151 MB (snap_delta) +
5.9 MB (snap_conv) on top of the live 151 MB state.

### 5c. All other decode.cu call-site diffs (deltanet path)

```cuda
// L72 (in_proj_a GEMV loop): x_.pf_n+size_t(t)*4096  -> x_.pf_n+size_t(t)*5120 ; x_.pf_a+size_t(t)*32 -> x_.pf_a+size_t(t)*48
// L73 (in_proj_b GEMV loop): same two substitutions (pf_b+size_t(t)*48)
// L75 (conv prefill):
// OLD: conv_prefill_silu(x_.pf_qkv,x_.pf_scratch,x_.conv_state+size_t(di)*8192*3,(const uint16_t*)cw.data,T,x_.stream,x_.snap_conv+size_t(di)*8192*3);
      conv_prefill_silu(x_.pf_qkv,x_.pf_scratch,x_.conv_state+size_t(di)*10240*3,(const uint16_t*)cw.data,T,x_.stream,x_.snap_conv+size_t(di)*10240*3);
// L77-78 (params batch, dtype dispatch):
      auto A=tensor(a+".A_log"),dt=tensor(a+".dt_bias");
      const bool a32=A.dtype==DType::f32;  // 27B native: BF16; 9B INSIG4: F32
      deltanet_params_batch(x_.pf_a,x_.pf_b,A.data,a32,(const uint16_t*)dt.data,T,x_.stream);
      w_.storage().release(a+".A_log");w_.storage().release(a+".dt_bias");
// L80 (scan): size_t(di)*32*128*128 -> size_t(di)*48*128*128  (BOTH delta_state and snap_delta)
      deltanet_prefill(x_.delta_state+size_t(di)*48*128*128,x_.pf_scratch,x_.pf_a,x_.pf_b,x_.pf_core,T,x_.stream,x_.snap_delta+size_t(di)*48*128*128);
// L81 (gated rmsnorm): size_t(T)*32 -> size_t(T)*48   (cols 128 stays)
      gated_rmsnorm_bf16(x_.pf_core,(const uint16_t*)nw.data,x_.pf_z,x_.pf_core,size_t(T)*48,128,x_.stream);
// L124 (delta_layer, decode path):
//  - conv:            causal_conv4_silu(...,x_.conv_state+size_t(di)*10240*3,...,10240,x_.stream);
//  - params:          auto A=tensor(a+".A_log"),dt=tensor(a+".dt_bias");const bool a32=A.dtype==DType::f32;
//                     deltanet_parameters(x_.a,x_.b,A.data,a32,(const uint16_t*)dt.data,48,x_.stream);
//  - scan:            deltanet_decode(x_.delta_state+size_t(di)*48*128*128,x_.qkv,x_.qkv+2048,x_.qkv+4096,x_.a,x_.b,x_.core,x_.stream);
//                     (offsets 2048/4096 STAY — q,k rows unchanged; v section is now 6144 long)
//  - gated rmsnorm:   gated_rmsnorm_bf16(x_.core,(const uint16_t*)nw.data,x_.z,x_.core,48,128,x_.stream);
```

Graph capture: `capture_step()` / `capture_spec()` freeze every launch config above
(grids 48, smem 66,048, buffer pointers) — destroy and re-capture both `graph_` and
`spec_graph_` after these edits (shape-constants G-note). `gated_rmsnorm_bf16` /
`rms_bf` kernels are rows-generic — no edits, call-site sizes only. `test_deltanet.cu`
needs `H=48, kh=h/3` (shape-constants instrumentation list).

---

## 6. CPU variant (CPU-tier delta layers) — spec + AVX2 sketch

Orientation contract: **CPU state must be `S[k*128 + v]` — row = key dim (128 rows),
contiguous 128-float lines along v — identical to the GPU kernels** (§0). The NumPy
reference's `state[h][v][k]` is the transpose; do not copy it. Per head, per token
(q̂/k̂ pre-normalized: `q̂ = q·rsqrt(Σq²+1e-6)·(1/√128)`, `k̂ = k·rsqrt(Σk²+1e-6)`; compute
once per **k-head** and reuse for its 3 v-heads):

1. `mem = α·(Sᵀ k̂)` — v-vector: for k in 0..127: `mem[0..127] += S_row_k · (α·k̂[k])` (axpy, contiguous)
2. `δ = (v − mem) · β`
3. fused update+out: for k in 0..127: `row = α·row + k̂[k]·δ` ; `out += row · q̂[k]` (both FMA over the row just produced)

Sketch (Zen 3 AVX2+FMA, one v-head; 16×`__m256` = one 512 B row = 8 cache lines):

```c
#include <immintrin.h>
// S: [128][128] f32 row-major (k rows, v cols) — MATCHES GPU S[k*128+v]. in-place.
static inline void dn_head(float *S, const float *qh, const float *kh_, const float *v_,
                           float alpha, float beta, float *out) {
    const __m256 a = _mm256_set1_ps(alpha);
    __m256 mem[16], del[16];
    for (int j = 0; j < 16; j++) mem[j] = _mm256_setzero_ps();
    for (int r = 0; r < 128; r++) {                      // pass 1: mem = alpha * S^T k
        const __m256 kk = _mm256_set1_ps(alpha * kh_[r]);
        const __m256 *row = (const __m256 *)(S + r * 128);
        for (int j = 0; j < 16; j++) mem[j] = _mm256_fmadd_ps(row[j], kk, mem[j]);
    }
    for (int j = 0; j < 16; j++) {
        del[j] = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(v_ + j * 8), mem[j]),
                               _mm256_set1_ps(beta));
    }
    __m256 acc[16];
    for (int j = 0; j < 16; j++) acc[j] = _mm256_setzero_ps();
    for (int r = 0; r < 128; r++) {                      // pass 2: S = a*S + k*del ; out += S*q
        __m256 *row = (__m256 *)(S + r * 128);
        const __m256 kr = _mm256_set1_ps(kh_[r]), qr = _mm256_set1_ps(qh[r]);
        for (int j = 0; j < 16; j++) {
            const __m256 up = _mm256_fmadd_ps(row[j], a, _mm256_mul_ps(kr, del[j]));
            row[j] = up;
            acc[j] = _mm256_fmadd_ps(up, qr, acc[j]);
        }
    }
    for (int j = 0; j < 16; j++) _mm256_storeu_ps(out + j * 8, acc[j]);
}
```

Cost model: per layer, state 3.146 MB read (pass 1) + read/write (pass 2) ≈ 9.4 MB touched,
but the whole layer state (48×64 KB) fits Zen 3's 32 MB L3 → effective DRAM ≈ 6.3 MB r/w
per token. Single thread ~0.25-0.3 ms/layer at ~35 GB/s; parallelize the 48 heads across
cores (state slices are disjoint) → DRAM-bound floor ~0.16 ms/layer. CPU-tier layers also
need host-side conv (channel-major `[10240][3]` roll, same order as `conv4`) and host-side
snapshot/rollback (memcpy of the layer's 64 KB/head slices + conv slice) mirroring §5.

---

## 7. Verification checklist for landing

1. `test_deltanet.cu`: H=48, KH=16, reference `kh=h/3` (ratio change!), state 48*128*128.
2. Parity: `tools/reference_layer0.py` deltanet section — rerun with 27B tensors; remember
   the numpy state is v-major (transposed) when comparing dumps against the engine buffer.
3. Alpha path: dump `a`/`b` after `deltanet_parameters` for one token; check
   `alpha ∈ (0,1)`, `beta ∈ (0,1)`, and `alpha != 1.0` (the BF16-as-F32 failure signature
   is alpha ≈ exp(-1e-40) ≈ 1.0).
4. Conv: `dump_i4_seams.cu` rows_probe {0, 2048, 4096, 10239} (q|k and k|v boundaries stay
   at 2048/4096; the row ceiling moves to 10239).
5. Spec: force one reject; verify `hidden` fully restored (5120), delta restored to row-0
   snapshot, conv restored to `[s1,s2,x0]`.
6. Re-capture both CUDA graphs; confirm no stale 32/8192 pointers survived in captures.
