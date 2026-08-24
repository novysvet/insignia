#include "insignia_layout.cuh"

namespace insignia {

__global__ __launch_bounds__(256, 4) void mxfp4_gemv_kernel(const MxFp4Block *__restrict__ weights, const float *__restrict__ x, float *__restrict__ y, int blocks_per_row) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    float sum = 0.0f;

    for (int block = warp; block < blocks_per_row; block += warps) {
        const MxFp4Block &w = weights[row * blocks_per_row + block];
        sum = fmaf(mxfp4_value(w, lane), x[block * 32 + lane], sum);
    }

    #pragma unroll
    for (int mask = 16; mask != 0; mask >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, mask);

    __shared__ float warp_sums[8];
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = lane < warps ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int mask = 16; mask != 0; mask >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, mask);
        if (lane == 0) y[row] = sum;
    }
}

void mxfp4_gemv(const MxFp4Block *weights, const float *x, float *y, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 31)) return;
    mxfp4_gemv_kernel<<<rows, 256, 0, stream>>>(weights, x, y, cols >> 5);
}

}

namespace insignia {

template<int WARPS>
__global__ void mxfp4_gemv_mlx_kernel(const uint32_t *__restrict__ weights, const uint8_t *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y, int groups) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float sum = 0.0f;
    const uint32_t *row_w = weights + static_cast<size_t>(row) * groups * 4;
    const uint8_t *row_s = scales + static_cast<size_t>(row) * groups;
    for (int group = warp; group < groups; group += WARPS) {
        const uint32_t packed = row_w[group * 4 + (lane >> 3)];
        const uint8_t code = uint8_t(packed >> ((lane & 7) * 4));
        const float scale = __int_as_float(uint32_t(row_s[group]) << 23);
        sum = fmaf(fp4_e2m1(code) * scale, x[group * 32 + lane], sum);
    }
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, mask);
    __shared__ float partial[WARPS];
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < WARPS ? partial[lane] : 0.0f;
        #pragma unroll
        for (int mask = 16; mask; mask >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, mask);
        if (lane == 0) y[row] = sum;
    }
}

void mxfp4_gemv_mlx(const uint32_t *weights, const uint8_t *scales, const float *x, float *y, int rows, int cols, int warps_per_row, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 31)) return;
    const int groups = cols >> 5;
    switch (warps_per_row) {
        case 1: mxfp4_gemv_mlx_kernel<1><<<rows,32,0,stream>>>(weights,scales,x,y,groups); break;
        case 2: mxfp4_gemv_mlx_kernel<2><<<rows,64,0,stream>>>(weights,scales,x,y,groups); break;
        case 8: mxfp4_gemv_mlx_kernel<8><<<rows,256,0,stream>>>(weights,scales,x,y,groups); break;
        default: mxfp4_gemv_mlx_kernel<4><<<rows,128,0,stream>>>(weights,scales,x,y,groups); break;
    }
}

}

namespace insignia {


// Decode GEMV tuned for sm_89: one warp per row; each lane owns whole 32-weight groups via
// uint4 loads (512B coalesced per warp per step). x is staged into shared memory transposed
// (slot k*groups+g holds x[g*32+k]) so lane-per-group reads are bank-conflict-free, and the
// E8M0 group scale is applied once per group after the raw FP32 dot.
__global__ __launch_bounds__(256) void mxfp4_gemv_v2_kernel(const uint32_t *__restrict__ weights, const uint8_t *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y, int rows, int groups) {
    constexpr int LANES = 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ float sx[];  // [groups*32] transposed x, then [16] e2m1 LUT
    float *lut = sx + groups * 32;
    for (int c0 = threadIdx.x * 32; c0 < groups * 32; c0 += blockDim.x * 32) {
        const float4 *xr = reinterpret_cast<const float4 *>(x + c0);
        float *sr = sx + (c0 >> 5);
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const float4 v = __ldg(xr + q);
            sr[(q * 4 + 0) * groups] = v.x;
            sr[(q * 4 + 1) * groups] = v.y;
            sr[(q * 4 + 2) * groups] = v.z;
            sr[(q * 4 + 3) * groups] = v.w;
        }
    }
    if (threadIdx.x < 16) lut[threadIdx.x] = decode4(threadIdx.x, 0);
    __syncthreads();
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_w = weights + static_cast<size_t>(row) * groups * 4;
    const uint8_t *row_s = scales + static_cast<size_t>(row) * groups;
    float acc = 0.f;
    #define V2_WORD(word, kb) { \
        const uint32_t w_ = (word); \
        _Pragma("unroll") \
        for (int j = 0; j < 8; j++) { \
            const float v = lut[(w_ >> (j * 4)) & 15u]; \
            const float xv = xg[(kb + j) * groups]; \
            const int which = j & 3; \
            if (which == 0) p0 = fmaf(v, xv, p0); \
            else if (which == 1) p1 = fmaf(v, xv, p1); \
            else if (which == 2) p2 = fmaf(v, xv, p2); \
            else p3 = fmaf(v, xv, p3); \
        } \
    }
    #pragma unroll 4
    for (int g0 = lane; g0 < groups; g0 += LANES) {
        const uint4 packed = __ldcs(reinterpret_cast<const uint4 *>(row_w + static_cast<size_t>(g0) * 4));
        const float scale = __int_as_float(static_cast<uint32_t>(row_s[g0]) << 23);
        const float *xg = sx + g0;  // xg[k*groups] == x[g0*32+k]
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
        V2_WORD(packed.x, 0)
        V2_WORD(packed.y, 8)
        V2_WORD(packed.z, 16)
        V2_WORD(packed.w, 24)
        acc = fmaf((p0 + p1) + (p2 + p3), scale, acc);
    }
    #undef V2_WORD
    float sum = acc;
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, mask);
    if (lane == 0) y[row] = sum;
}

void mxfp4_gemv_v2(const uint32_t *weights, const uint8_t *scales, const float *x, float *y, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 31)) return;
    if (cols & 1023) { mxfp4_gemv_mlx(weights, scales, x, y, rows, cols, 2, stream); return; }
    static const bool configured = [] {
        return cudaFuncSetAttribute(mxfp4_gemv_v2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024) == cudaSuccess;
    }();
    (void)configured;
    const int groups = cols >> 5;
    mxfp4_gemv_v2_kernel<<<(rows + 7) >> 3, 256, size_t(cols) * 4 + 64, stream>>>(weights, scales, x, y, rows, groups);
}

// Two activation rows through one weight pass (speculative-decode pair forward).
// x is [2, cols] and y is [2, rows], both row-major contiguous.
__global__ __launch_bounds__(256) void mxfp4_gemv2_v2_kernel(const uint32_t *__restrict__ weights, const uint8_t *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y, int rows, int groups) {
    constexpr int LANES = 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ float sx[];  // row0 transposed | row1 transposed | LUT
    float *lut = sx + groups * 64;
    for (int c0 = threadIdx.x * 32; c0 < groups * 32; c0 += blockDim.x * 32) {
        const float4 *xr0 = reinterpret_cast<const float4 *>(x + c0);
        const float4 *xr1 = reinterpret_cast<const float4 *>(x + groups * 32 + c0);
        float *sr0 = sx + (c0 >> 5);
        float *sr1 = sx + groups * 32 + (c0 >> 5);
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const float4 v0 = __ldg(xr0 + q), v1 = __ldg(xr1 + q);
            sr0[(q * 4 + 0) * groups] = v0.x; sr0[(q * 4 + 1) * groups] = v0.y;
            sr0[(q * 4 + 2) * groups] = v0.z; sr0[(q * 4 + 3) * groups] = v0.w;
            sr1[(q * 4 + 0) * groups] = v1.x; sr1[(q * 4 + 1) * groups] = v1.y;
            sr1[(q * 4 + 2) * groups] = v1.z; sr1[(q * 4 + 3) * groups] = v1.w;
        }
    }
    if (threadIdx.x < 16) lut[threadIdx.x] = decode4(threadIdx.x, 0);
    __syncthreads();
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *wr = weights + static_cast<size_t>(row) * groups * 4;
    const uint8_t *sr = scales + static_cast<size_t>(row) * groups;
    float a0 = 0.f, b0 = 0.f;  // dots of this weight row against both activation rows
    for (int g0 = lane; g0 < groups; g0 += LANES) {
        const uint4 p = __ldcs(reinterpret_cast<const uint4 *>(wr + static_cast<size_t>(g0) * 4));
        const float scale = __int_as_float(static_cast<uint32_t>(sr[g0]) << 23);
        const float *xg0 = sx + g0;
        const float *xg1 = sx + groups * 32 + g0;
        float p0 = 0.f, q0 = 0.f;
        p0 += lut[(p.x >> 0) & 15u] * xg0[0 * groups] + lut[(p.x >> 4) & 15u] * xg0[1 * groups] + lut[(p.x >> 8) & 15u] * xg0[2 * groups] + lut[(p.x >> 12) & 15u] * xg0[3 * groups];
        p0 += lut[(p.x >> 16) & 15u] * xg0[4 * groups] + lut[(p.x >> 20) & 15u] * xg0[5 * groups] + lut[(p.x >> 24) & 15u] * xg0[6 * groups] + lut[(p.x >> 28) & 15u] * xg0[7 * groups];
        p0 += lut[(p.y >> 0) & 15u] * xg0[8 * groups] + lut[(p.y >> 4) & 15u] * xg0[9 * groups] + lut[(p.y >> 8) & 15u] * xg0[10 * groups] + lut[(p.y >> 12) & 15u] * xg0[11 * groups];
        p0 += lut[(p.y >> 16) & 15u] * xg0[12 * groups] + lut[(p.y >> 20) & 15u] * xg0[13 * groups] + lut[(p.y >> 24) & 15u] * xg0[14 * groups] + lut[(p.y >> 28) & 15u] * xg0[15 * groups];
        p0 += lut[(p.z >> 0) & 15u] * xg0[16 * groups] + lut[(p.z >> 4) & 15u] * xg0[17 * groups] + lut[(p.z >> 8) & 15u] * xg0[18 * groups] + lut[(p.z >> 12) & 15u] * xg0[19 * groups];
        p0 += lut[(p.z >> 16) & 15u] * xg0[20 * groups] + lut[(p.z >> 20) & 15u] * xg0[21 * groups] + lut[(p.z >> 24) & 15u] * xg0[22 * groups] + lut[(p.z >> 28) & 15u] * xg0[23 * groups];
        p0 += lut[(p.w >> 0) & 15u] * xg0[24 * groups] + lut[(p.w >> 4) & 15u] * xg0[25 * groups] + lut[(p.w >> 8) & 15u] * xg0[26 * groups] + lut[(p.w >> 12) & 15u] * xg0[27 * groups];
        p0 += lut[(p.w >> 16) & 15u] * xg0[28 * groups] + lut[(p.w >> 20) & 15u] * xg0[29 * groups] + lut[(p.w >> 24) & 15u] * xg0[30 * groups] + lut[(p.w >> 28) & 15u] * xg0[31 * groups];
        q0 += lut[(p.x >> 0) & 15u] * xg1[0 * groups] + lut[(p.x >> 4) & 15u] * xg1[1 * groups] + lut[(p.x >> 8) & 15u] * xg1[2 * groups] + lut[(p.x >> 12) & 15u] * xg1[3 * groups];
        q0 += lut[(p.x >> 16) & 15u] * xg1[4 * groups] + lut[(p.x >> 20) & 15u] * xg1[5 * groups] + lut[(p.x >> 24) & 15u] * xg1[6 * groups] + lut[(p.x >> 28) & 15u] * xg1[7 * groups];
        q0 += lut[(p.y >> 0) & 15u] * xg1[8 * groups] + lut[(p.y >> 4) & 15u] * xg1[9 * groups] + lut[(p.y >> 8) & 15u] * xg1[10 * groups] + lut[(p.y >> 12) & 15u] * xg1[11 * groups];
        q0 += lut[(p.y >> 16) & 15u] * xg1[12 * groups] + lut[(p.y >> 20) & 15u] * xg1[13 * groups] + lut[(p.y >> 24) & 15u] * xg1[14 * groups] + lut[(p.y >> 28) & 15u] * xg1[15 * groups];
        q0 += lut[(p.z >> 0) & 15u] * xg1[16 * groups] + lut[(p.z >> 4) & 15u] * xg1[17 * groups] + lut[(p.z >> 8) & 15u] * xg1[18 * groups] + lut[(p.z >> 12) & 15u] * xg1[19 * groups];
        q0 += lut[(p.z >> 16) & 15u] * xg1[20 * groups] + lut[(p.z >> 20) & 15u] * xg1[21 * groups] + lut[(p.z >> 24) & 15u] * xg1[22 * groups] + lut[(p.z >> 28) & 15u] * xg1[23 * groups];
        q0 += lut[(p.w >> 0) & 15u] * xg1[24 * groups] + lut[(p.w >> 4) & 15u] * xg1[25 * groups] + lut[(p.w >> 8) & 15u] * xg1[26 * groups] + lut[(p.w >> 12) & 15u] * xg1[27 * groups];
        q0 += lut[(p.w >> 16) & 15u] * xg1[28 * groups] + lut[(p.w >> 20) & 15u] * xg1[29 * groups] + lut[(p.w >> 24) & 15u] * xg1[30 * groups] + lut[(p.w >> 28) & 15u] * xg1[31 * groups];
        a0 = fmaf(p0, scale, a0);
        b0 = fmaf(q0, scale, b0);
    }
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) { a0 += __shfl_xor_sync(0xffffffff, a0, mask); b0 += __shfl_xor_sync(0xffffffff, b0, mask); }
    if (lane == 0) { y[row] = a0; y[rows + row] = b0; }
}
void mxfp4_gemv2_v2(const uint32_t *weights, const uint8_t *scales, const float *x, float *y, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 1023)) return;
    static const bool configured = [] {
        return cudaFuncSetAttribute(mxfp4_gemv2_v2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024) == cudaSuccess;
    }();
    (void)configured;
    const int groups = cols >> 5;
    mxfp4_gemv2_v2_kernel<<<(rows + 7) >> 3, 256, size_t(cols) * 2 * 4 + 64, stream>>>(weights, scales, x, y, rows, groups);
}

}

namespace insignia {

__global__ void quantize_q8_groups_kernel(const float *__restrict__ x, int8_t *__restrict__ qx, float *__restrict__ qscale, int groups) {
    const int group = blockIdx.x;
    const int lane = threadIdx.x;
    if (group >= groups) return;
    float v = x[group * 32 + lane];
    float a = fabsf(v);
    #pragma unroll
    for (int mask=16;mask;mask>>=1) a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,mask));
    const float scale = a > 0.0f ? a * (1.0f/127.0f) : 1.0f;
    if (lane == 0) qscale[group] = scale;
    qx[group * 32 + lane] = static_cast<int8_t>(__float2int_rn(v / scale));
}

void quantize_q8_groups(const float *x, int8_t *qx, float *qscale, int cols, cudaStream_t stream) {
    if (cols <= 0 || (cols & 31)) return;
    quantize_q8_groups_kernel<<<cols>>5,32,0,stream>>>(x,qx,qscale,cols>>5);
}

template<int WARPS>
__global__ void mxfp4_gemv_dp4a_kernel(const uint32_t *__restrict__ weights, const uint8_t *__restrict__ scales, const int8_t *__restrict__ qx, const float *__restrict__ qscale, float *__restrict__ y, int groups) {
    const int row=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const uint32_t *row_w=weights+static_cast<size_t>(row)*groups*4;
    const uint8_t *row_s=scales+static_cast<size_t>(row)*groups;
    float sum=0.0f;
    for(int group=warp;group<groups;group+=WARPS){
        int dot=0;
        if(lane<4){
            const uint32_t packed=row_w[group*4+lane];
            const int vals[8]={0,1,2,3,4,6,8,12};
            #pragma unroll
            for(int half=0;half<2;half++){
                int8_t wb[4];
                #pragma unroll
                for(int j=0;j<4;j++){const uint8_t q=uint8_t(packed>>((half*4+j)*4))&15;wb[j]=static_cast<int8_t>((q&8)?-vals[q&7]:vals[q]);}
                uint32_t wi;memcpy(&wi,wb,4);const uint32_t xi=reinterpret_cast<const uint32_t*>(qx+group*32)[lane*2+half];dot=__dp4a(static_cast<int>(wi),static_cast<int>(xi),dot);
            }
        }
        sum=fmaf(float(dot),__int_as_float(uint32_t(row_s[group])<<23)*qscale[group]*0.5f,sum);
    }
    #pragma unroll
    for(int mask=16;mask;mask>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,mask);
    __shared__ float partial[WARPS];if(lane==0)partial[warp]=sum;__syncthreads();if(warp==0){sum=lane<WARPS?partial[lane]:0.0f;for(int mask=16;mask;mask>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,mask);if(lane==0)y[row]=sum;}
}

void mxfp4_gemv_dp4a(const uint32_t*w,const uint8_t*s,const int8_t*x,const float*xs,float*y,int rows,int cols,int warps,cudaStream_t stream){const int groups=cols>>5;if(warps==1)mxfp4_gemv_dp4a_kernel<1><<<rows,32,0,stream>>>(w,s,x,xs,y,groups);else if(warps==2)mxfp4_gemv_dp4a_kernel<2><<<rows,64,0,stream>>>(w,s,x,xs,y,groups);else if(warps==8)mxfp4_gemv_dp4a_kernel<8><<<rows,256,0,stream>>>(w,s,x,xs,y,groups);else mxfp4_gemv_dp4a_kernel<4><<<rows,128,0,stream>>>(w,s,x,xs,y,groups);}

}

namespace insignia {
__global__ void mxfp4_get_row_kernel(const uint32_t*weights,const uint8_t*scales,float*out,const int*row_dev,int groups){const int row=__ldg(row_dev);for(int group=blockIdx.x;group<groups;group+=gridDim.x){const int lane=threadIdx.x;const uint32_t packed=weights[(static_cast<size_t>(row)*groups+group)*4+(lane>>3)];const uint8_t q=uint8_t(packed>>((lane&7)*4));out[group*32+lane]=fp4_e2m1(q)*__int_as_float(uint32_t(scales[static_cast<size_t>(row)*groups+group])<<23);}}
void mxfp4_get_row_mlx(const uint32_t*w,const uint8_t*s,float*out,const int*row_dev,int cols,cudaStream_t stream){mxfp4_get_row_kernel<<<((cols>>5)<128?(cols>>5):128),32,0,stream>>>(w,s,out,row_dev,cols>>5);}
}

namespace insignia {

// Speculative pair GEMV, dp4a edition: both activation rows quantized once to per-group
// int8 (fp32 group scales), weights decoded to signed int8 (x2 E2M1 table, 0.5 folded
// into the group scale) via one 256-entry u64 broadcast table and __byte_perm interleave.
// One weight stream feeds both rows' dp4a chains. cols must be a multiple of 32.
__global__ __launch_bounds__(256) void mxfp4_gemv2_q8_kernel(const uint32_t *__restrict__ weights, const uint8_t *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y, int rows, int groups) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ char smem[];
    uint32_t *xq = reinterpret_cast<uint32_t *>(smem);                        // [2][groups][8]
    float *xs = reinterpret_cast<float *>(smem + size_t(16) * groups * 4);    // [2][groups]
    unsigned long long *btab = reinterpret_cast<unsigned long long *>(smem + size_t(16) * groups * 4 + 2 * groups * 4);
    for (int rg = threadIdx.x; rg < 2 * groups; rg += 256) {  // quantize one 32-group of one row
        const int r = rg / groups, g = rg % groups;
        const float *xg = x + r * (groups * 32) + g * 32;
        float v[32], m = 0.f;
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const float4 f4 = __ldg(reinterpret_cast<const float4 *>(xg) + q);
            v[q * 4 + 0] = f4.x; v[q * 4 + 1] = f4.y; v[q * 4 + 2] = f4.z; v[q * 4 + 3] = f4.w;
            m = fmaxf(m, fmaxf(fmaxf(fabsf(f4.x), fabsf(f4.y)), fmaxf(fabsf(f4.z), fabsf(f4.w))));
        }
        xs[r * groups + g] = m * (1.f / 127.f);
        const float inv = m > 0.f ? 127.f / m : 0.f;
        uint32_t *dst = xq + (r * groups + g) * 8;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            uint32_t packed = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) packed |= uint32_t(uint8_t(__float2int_rn(v[w * 4 + j] * inv))) << (8 * j);
            dst[w] = packed;
        }
    }
    if (threadIdx.x < 256) {  // btab[b] = { LO-nibble code broadcast | HI-nibble code broadcast }
        const signed char tbl[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        const int b = threadIdx.x;
        const int lo = b & 15, hi = b >> 4;
        const uint32_t clo = uint32_t(uint8_t((lo & 8) ? -tbl[lo & 7] : tbl[lo & 7])) * 0x01010101u;
        const uint32_t chi = uint32_t(uint8_t((hi & 8) ? -tbl[hi & 7] : tbl[hi & 7])) * 0x01010101u;
        btab[threadIdx.x] = unsigned long long(clo) | (unsigned long long(chi) << 32);
    }
    __syncthreads();
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_w = weights + static_cast<size_t>(row) * groups * 4;
    const uint8_t *row_s = scales + static_cast<size_t>(row) * groups;
    float acc0 = 0.f, acc1 = 0.f;
    #pragma unroll 4
    for (int g0 = lane; g0 < groups; g0 += 32) {
        const uint4 P = __ldcs(reinterpret_cast<const uint4 *>(row_w + static_cast<size_t>(g0) * 4));
        const uint4 *x0 = reinterpret_cast<const uint4 *>(xq + g0 * 8);
        const uint4 *x1 = reinterpret_cast<const uint4 *>(xq + (groups + g0) * 8);
        int d0a = 0, d0b = 0, d1a = 0, d1b = 0;
        const uint32_t pw[4] = {P.x, P.y, P.z, P.w};
        #pragma unroll
        for (int wi = 0; wi < 4; wi++) {
            const uint32_t w_ = pw[wi];
            const unsigned long long b0 = btab[w_ & 0xff], b1 = btab[(w_ >> 8) & 0xff];
            const unsigned long long b2 = btab[(w_ >> 16) & 0xff], b3 = btab[w_ >> 24];
            const uint32_t L0 = uint32_t(b0), H0 = uint32_t(b0 >> 32);
            const uint32_t L1 = uint32_t(b1), H1 = uint32_t(b1 >> 32);
            const uint32_t L2 = uint32_t(b2), H2 = uint32_t(b2 >> 32);
            const uint32_t L3 = uint32_t(b3), H3 = uint32_t(b3 >> 32);
            const uint32_t A = __byte_perm(L0, L1, 0xc480);       // [c0,0,c2,0]
            const uint32_t B = __byte_perm(H0, H1, 0x4c80);       // [0,c1,0,c3]
            const uint32_t w0 = __byte_perm(A, B, 0x6240);        // [c0,c1,c2,c3]
            const uint32_t A2 = __byte_perm(L2, L3, 0xc480);
            const uint32_t B2 = __byte_perm(H2, H3, 0x4c80);
            const uint32_t w1 = __byte_perm(A2, B2, 0x6240);      // [c4,c5,c6,c7]
            const uint32_t xx0 = reinterpret_cast<const uint32_t *>(x0)[wi * 2];
            const uint32_t xx1 = reinterpret_cast<const uint32_t *>(x0)[wi * 2 + 1];
            const uint32_t yy0 = reinterpret_cast<const uint32_t *>(x1)[wi * 2];
            const uint32_t yy1 = reinterpret_cast<const uint32_t *>(x1)[wi * 2 + 1];
            d0a = __dp4a(int(w0), int(xx0), d0a);
            d1a = __dp4a(int(w0), int(yy0), d1a);
            d0b = __dp4a(int(w1), int(xx1), d0b);
            d1b = __dp4a(int(w1), int(yy1), d1b);
        }
        const float ws = __int_as_float(uint32_t(row_s[g0]) << 23) * 0.5f;
        acc0 = fmaf(float(d0a + d0b), ws * xs[g0], acc0);
        acc1 = fmaf(float(d1a + d1b), ws * xs[groups + g0], acc1);
    }
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) { acc0 += __shfl_xor_sync(0xffffffff, acc0, mask); acc1 += __shfl_xor_sync(0xffffffff, acc1, mask); }
    if (lane == 0) { y[row] = acc0; y[rows + row] = acc1; }
}
void mxfp4_gemv2_q8(const uint32_t *weights, const uint8_t *scales, const float *x, float *y, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 31)) return;
    const int groups = cols >> 5;
    mxfp4_gemv2_q8_kernel<<<(rows + 7) >> 3, 256, size_t(16) * groups * 4 + 2 * groups * 4 + 2048, stream>>>(weights, scales, x, y, rows, groups);
}

}

namespace insignia {

// One thread quantizes one 32-wide group: int8 codes (group-major, 8 u32 words) + fp32 scale.
__global__ void quantize_x8_kernel(const float *__restrict__ x, uint32_t *__restrict__ xq, float *__restrict__ xs, int rows, int groups) {
    const int rg = blockIdx.x * blockDim.x + threadIdx.x;
    if (rg >= rows * groups) return;
    const int r = rg / groups, g = rg % groups;
    const float *xg = x + size_t(r) * groups * 32 + g * 32;
    float v[32], m = 0.f;
    #pragma unroll
    for (int q = 0; q < 8; q++) {
        const float4 f4 = __ldg(reinterpret_cast<const float4 *>(xg) + q);
        v[q * 4 + 0] = f4.x; v[q * 4 + 1] = f4.y; v[q * 4 + 2] = f4.z; v[q * 4 + 3] = f4.w;
        m = fmaxf(m, fmaxf(fmaxf(fabsf(f4.x), fabsf(f4.y)), fmaxf(fabsf(f4.z), fabsf(f4.w))));
    }
    xs[rg] = m * (1.f / 127.f);
    const float inv = m > 0.f ? 127.f / m : 0.f;
    uint32_t *dst = xq + size_t(rg) * 8;
    #pragma unroll
    for (int w = 0; w < 8; w++) {
        uint32_t packed = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) packed |= uint32_t(uint8_t(__float2int_rn(v[w * 4 + j] * inv))) << (8 * j);
        dst[w] = packed;
    }
}
void quantize_x8(const float *x, uint32_t *xq, float *xs, int rows, int cols, cudaStream_t stream) {
    const int groups = cols >> 5;
    quantize_x8_kernel<<<(rows * groups + 255) / 256, 256, 0, stream>>>(x, xq, xs, rows, groups);
}

// Prequantized pair GEMV: xq/xs live in global (written once per activation by quantize_x8),
// the nibble broadcast table stays in shared. Single weight stream feeds both rows.
__global__ __launch_bounds__(256) void mxfp4_gemv2_q8g_kernel(const uint32_t *__restrict__ weights, const uint8_t *__restrict__ scales, const uint32_t *__restrict__ xq, const float *__restrict__ xs, float *__restrict__ y, int rows, int groups) {
    __shared__ unsigned long long btab[256];
    {
        const signed char tbl[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        const int b = threadIdx.x;
        const int lo = b & 15, hi = b >> 4;
        const uint32_t clo = uint32_t(uint8_t((lo & 8) ? -tbl[lo & 7] : tbl[lo & 7])) * 0x01010101u;
        const uint32_t chi = uint32_t(uint8_t((hi & 8) ? -tbl[hi & 7] : tbl[hi & 7])) * 0x01010101u;
        btab[threadIdx.x] = unsigned long long(clo) | (unsigned long long(chi) << 32);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_w = weights + static_cast<size_t>(row) * groups * 4;
    const uint8_t *row_s = scales + static_cast<size_t>(row) * groups;
    const uint32_t *xq0 = xq, *xq1 = xq + size_t(groups) * 8;
    float acc0 = 0.f, acc1 = 0.f;
    #pragma unroll 4
    for (int g0 = lane; g0 < groups; g0 += 32) {
        const uint4 P = __ldcs(reinterpret_cast<const uint4 *>(row_w + static_cast<size_t>(g0) * 4));
        const uint4 *x0 = reinterpret_cast<const uint4 *>(xq0 + g0 * 8);
        const uint4 *x1 = reinterpret_cast<const uint4 *>(xq1 + g0 * 8);
        int d0a = 0, d0b = 0, d1a = 0, d1b = 0;
        const uint32_t pw[4] = {P.x, P.y, P.z, P.w};
        #pragma unroll
        for (int wi = 0; wi < 4; wi++) {
            const uint32_t w_ = pw[wi];
            const unsigned long long b0 = btab[w_ & 0xff], b1 = btab[(w_ >> 8) & 0xff];
            const unsigned long long b2 = btab[(w_ >> 16) & 0xff], b3 = btab[w_ >> 24];
            const uint32_t A = __byte_perm(uint32_t(b0), uint32_t(b1), 0xc480);
            const uint32_t B = __byte_perm(uint32_t(b0 >> 32), uint32_t(b1 >> 32), 0x4c80);
            const uint32_t w0 = __byte_perm(A, B, 0x6240);
            const uint32_t A2 = __byte_perm(uint32_t(b2), uint32_t(b3), 0xc480);
            const uint32_t B2 = __byte_perm(uint32_t(b2 >> 32), uint32_t(b3 >> 32), 0x4c80);
            const uint32_t w1 = __byte_perm(A2, B2, 0x6240);
            const uint32_t xx0 = reinterpret_cast<const uint32_t *>(x0)[wi * 2];
            const uint32_t xx1 = reinterpret_cast<const uint32_t *>(x0)[wi * 2 + 1];
            const uint32_t yy0 = reinterpret_cast<const uint32_t *>(x1)[wi * 2];
            const uint32_t yy1 = reinterpret_cast<const uint32_t *>(x1)[wi * 2 + 1];
            d0a = __dp4a(int(w0), int(xx0), d0a);
            d1a = __dp4a(int(w0), int(yy0), d1a);
            d0b = __dp4a(int(w1), int(xx1), d0b);
            d1b = __dp4a(int(w1), int(yy1), d1b);
        }
        const float ws = __int_as_float(uint32_t(row_s[g0]) << 23) * 0.5f;
        acc0 = fmaf(float(d0a + d0b), ws * __ldg(xs + g0), acc0);
        acc1 = fmaf(float(d1a + d1b), ws * __ldg(xs + groups + g0), acc1);
    }
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) { acc0 += __shfl_xor_sync(0xffffffff, acc0, mask); acc1 += __shfl_xor_sync(0xffffffff, acc1, mask); }
    if (lane == 0) { y[row] = acc0; y[rows + row] = acc1; }
}
void mxfp4_gemv2_q8g(const uint32_t *weights, const uint8_t *scales, const uint32_t *xq, const float *xs, float *y, int rows, int cols, cudaStream_t stream) {
    const int groups = cols >> 5;
    mxfp4_gemv2_q8g_kernel<<<(rows + 7) >> 3, 256, 0, stream>>>(weights, scales, xq, xs, y, rows, groups);
}

// Prequantized single-row GEMV (decode / MTP path): same decode as the pair kernel.
__global__ __launch_bounds__(256) void mxfp4_gemv_q8g_kernel(const uint32_t *__restrict__ weights, const uint8_t *__restrict__ scales, const uint32_t *__restrict__ xq, const float *__restrict__ xs, float *__restrict__ y, int rows, int groups) {
    __shared__ unsigned long long btab[256];
    {
        const signed char tbl[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        const int b = threadIdx.x;
        const int lo = b & 15, hi = b >> 4;
        const uint32_t clo = uint32_t(uint8_t((lo & 8) ? -tbl[lo & 7] : tbl[lo & 7])) * 0x01010101u;
        const uint32_t chi = uint32_t(uint8_t((hi & 8) ? -tbl[hi & 7] : tbl[hi & 7])) * 0x01010101u;
        btab[threadIdx.x] = unsigned long long(clo) | (unsigned long long(chi) << 32);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint32_t *row_w = weights + static_cast<size_t>(row) * groups * 4;
    const uint8_t *row_s = scales + static_cast<size_t>(row) * groups;
    float acc = 0.f;
    #pragma unroll 4
    for (int g0 = lane; g0 < groups; g0 += 32) {
        const uint4 P = __ldcs(reinterpret_cast<const uint4 *>(row_w + static_cast<size_t>(g0) * 4));
        const uint4 *x0 = reinterpret_cast<const uint4 *>(xq + g0 * 8);
        int da = 0, db = 0;
        const uint32_t pw[4] = {P.x, P.y, P.z, P.w};
        #pragma unroll
        for (int wi = 0; wi < 4; wi++) {
            const uint32_t w_ = pw[wi];
            const unsigned long long b0 = btab[w_ & 0xff], b1 = btab[(w_ >> 8) & 0xff];
            const unsigned long long b2 = btab[(w_ >> 16) & 0xff], b3 = btab[w_ >> 24];
            const uint32_t A = __byte_perm(uint32_t(b0), uint32_t(b1), 0xc480);
            const uint32_t B = __byte_perm(uint32_t(b0 >> 32), uint32_t(b1 >> 32), 0x4c80);
            const uint32_t w0 = __byte_perm(A, B, 0x6240);
            const uint32_t A2 = __byte_perm(uint32_t(b2), uint32_t(b3), 0xc480);
            const uint32_t B2 = __byte_perm(uint32_t(b2 >> 32), uint32_t(b3 >> 32), 0x4c80);
            const uint32_t w1 = __byte_perm(A2, B2, 0x6240);
            da = __dp4a(int(w0), int(reinterpret_cast<const uint32_t *>(x0)[wi * 2]), da);
            db = __dp4a(int(w1), int(reinterpret_cast<const uint32_t *>(x0)[wi * 2 + 1]), db);
        }
        acc = fmaf(float(da + db), __int_as_float(uint32_t(row_s[g0]) << 23) * 0.5f * __ldg(xs + g0), acc);
    }
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, mask);
    if (lane == 0) y[row] = acc;
}
void mxfp4_gemv_q8g(const uint32_t *weights, const uint8_t *scales, const uint32_t *xq, const float *xs, float *y, int rows, int cols, cudaStream_t stream) {
    const int groups = cols >> 5;
    mxfp4_gemv_q8g_kernel<<<(rows + 7) >> 3, 256, 0, stream>>>(weights, scales, xq, xs, y, rows, groups);
}

}

namespace insignia {

// Fused in_proj_a + in_proj_b for the pair path: 64 rows (32 a, then 32 b) in one launch,
// both activation rows through one weight stream, x prequantized (rows 0/1 of xq/xs).
__global__ __launch_bounds__(256) void mxfp4_gemv_ab2_q8g_kernel(const uint32_t *__restrict__ wa, const uint8_t *__restrict__ sa, const uint32_t *__restrict__ wb, const uint8_t *__restrict__ sb, const uint32_t *__restrict__ xq, const float *__restrict__ xs, float *__restrict__ ya, float *__restrict__ yb, int groups) {
    __shared__ unsigned long long btab[256];
    {
        const signed char tbl[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        const int b = threadIdx.x;
        const int lo = b & 15, hi = b >> 4;
        const uint32_t clo = uint32_t(uint8_t((lo & 8) ? -tbl[lo & 7] : tbl[lo & 7])) * 0x01010101u;
        const uint32_t chi = uint32_t(uint8_t((hi & 8) ? -tbl[hi & 7] : tbl[hi & 7])) * 0x01010101u;
        btab[threadIdx.x] = unsigned long long(clo) | (unsigned long long(chi) << 32);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint32_t *xq0 = xq, *xq1 = xq + size_t(groups) * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) {  // each of the 8 warps owns 8 concatenated rows
        const int rr = warp * 8 + i;
        const bool is_a = rr < 32;
        const int row = is_a ? rr : rr - 32;
        const uint32_t *row_w = (is_a ? wa : wb) + static_cast<size_t>(row) * groups * 4;
        const uint8_t *row_s = (is_a ? sa : sb) + static_cast<size_t>(row) * groups;
        float acc0 = 0.f, acc1 = 0.f;
        #pragma unroll 4
        for (int g0 = lane; g0 < groups; g0 += 32) {
            const uint4 P = __ldcs(reinterpret_cast<const uint4 *>(row_w + static_cast<size_t>(g0) * 4));
            const uint4 *x0 = reinterpret_cast<const uint4 *>(xq0 + g0 * 8);
            const uint4 *x1 = reinterpret_cast<const uint4 *>(xq1 + g0 * 8);
            int d0a = 0, d0b = 0, d1a = 0, d1b = 0;
            const uint32_t pw[4] = {P.x, P.y, P.z, P.w};
            #pragma unroll
            for (int wi = 0; wi < 4; wi++) {
                const uint32_t w_ = pw[wi];
                const unsigned long long b0 = btab[w_ & 0xff], b1 = btab[(w_ >> 8) & 0xff];
                const unsigned long long b2 = btab[(w_ >> 16) & 0xff], b3 = btab[w_ >> 24];
                const uint32_t A = __byte_perm(uint32_t(b0), uint32_t(b1), 0xc480);
                const uint32_t B = __byte_perm(uint32_t(b0 >> 32), uint32_t(b1 >> 32), 0x4c80);
                const uint32_t w0 = __byte_perm(A, B, 0x6240);
                const uint32_t A2 = __byte_perm(uint32_t(b2), uint32_t(b3), 0xc480);
                const uint32_t B2 = __byte_perm(uint32_t(b2 >> 32), uint32_t(b3 >> 32), 0x4c80);
                const uint32_t w1 = __byte_perm(A2, B2, 0x6240);
                d0a = __dp4a(int(w0), int(reinterpret_cast<const uint32_t *>(x0)[wi * 2]), d0a);
                d1a = __dp4a(int(w0), int(reinterpret_cast<const uint32_t *>(x1)[wi * 2]), d1a);
                d0b = __dp4a(int(w1), int(reinterpret_cast<const uint32_t *>(x0)[wi * 2 + 1]), d0b);
                d1b = __dp4a(int(w1), int(reinterpret_cast<const uint32_t *>(x1)[wi * 2 + 1]), d1b);
            }
            const float ws = __int_as_float(uint32_t(row_s[g0]) << 23) * 0.5f;
            acc0 = fmaf(float(d0a + d0b), ws * __ldg(xs + g0), acc0);
            acc1 = fmaf(float(d1a + d1b), ws * __ldg(xs + groups + g0), acc1);
        }
        #pragma unroll
        for (int mask = 16; mask; mask >>= 1) { acc0 += __shfl_xor_sync(0xffffffff, acc0, mask); acc1 += __shfl_xor_sync(0xffffffff, acc1, mask); }
        if (lane == 0) {
            if (is_a) { ya[row] = acc0; ya[32 + row] = acc1; }
            else { yb[row] = acc0; yb[32 + row] = acc1; }
        }
    }
}
void mxfp4_gemv_ab2_q8g(const uint32_t *wa, const uint8_t *sa, const uint32_t *wb, const uint8_t *sb, const uint32_t *xq, const float *xs, float *ya, float *yb, int cols, cudaStream_t stream) {
    mxfp4_gemv_ab2_q8g_kernel<<<1, 256, 0, stream>>>(wa, sa, wb, sb, xq, xs, ya, yb, cols >> 5);
}

}

namespace insignia {

// Fused in_proj_a + in_proj_b for the pair path: both rows staged+quantized in-block
// (256 threads = 2 rows x 128 groups), then 64 concatenated rows (32 a, 32 b) computed
// by the 8 warps. One launch per DeltaNet layer instead of four per-token GEMVs.
__global__ __launch_bounds__(256) void mxfp4_gemv_ab2_q8_kernel(const uint32_t *__restrict__ wa, const uint8_t *__restrict__ sa, const uint32_t *__restrict__ wb, const uint8_t *__restrict__ sb, const float *__restrict__ x, float *__restrict__ ya, float *__restrict__ yb, int groups) {
    extern __shared__ char smem[];
    uint32_t *xq = reinterpret_cast<uint32_t *>(smem);                        // [2][groups][8]
    float *xs = reinterpret_cast<float *>(smem + size_t(16) * groups * 4);
    unsigned long long *btab = reinterpret_cast<unsigned long long *>(smem + size_t(16) * groups * 4 + 2 * groups * 4);
    {
        const int r = threadIdx.x >> 7, g = threadIdx.x & 127;
        const float *xg = x + r * (groups * 32) + g * 32;
        float v[32], m = 0.f;
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const float4 f4 = __ldg(reinterpret_cast<const float4 *>(xg) + q);
            v[q * 4 + 0] = f4.x; v[q * 4 + 1] = f4.y; v[q * 4 + 2] = f4.z; v[q * 4 + 3] = f4.w;
            m = fmaxf(m, fmaxf(fmaxf(fabsf(f4.x), fabsf(f4.y)), fmaxf(fabsf(f4.z), fabsf(f4.w))));
        }
        xs[r * groups + g] = m * (1.f / 127.f);
        const float inv = m > 0.f ? 127.f / m : 0.f;
        uint32_t *dst = xq + (r * groups + g) * 8;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            uint32_t packed = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) packed |= uint32_t(uint8_t(__float2int_rn(v[w * 4 + j] * inv))) << (8 * j);
            dst[w] = packed;
        }
    }
    {
        const signed char tbl[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        const int b = threadIdx.x;
        const int lo = b & 15, hi = b >> 4;
        const uint32_t clo = uint32_t(uint8_t((lo & 8) ? -tbl[lo & 7] : tbl[lo & 7])) * 0x01010101u;
        const uint32_t chi = uint32_t(uint8_t((hi & 8) ? -tbl[hi & 7] : tbl[hi & 7])) * 0x01010101u;
        btab[threadIdx.x] = unsigned long long(clo) | (unsigned long long(chi) << 32);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint32_t *xq0 = xq, *xq1 = xq + groups * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int rr = warp * 8 + i;
        const bool is_a = rr < 32;
        const int row = is_a ? rr : rr - 32;
        const uint32_t *row_w = (is_a ? wa : wb) + static_cast<size_t>(row) * groups * 4;
        const uint8_t *row_s = (is_a ? sa : sb) + static_cast<size_t>(row) * groups;
        float acc0 = 0.f, acc1 = 0.f;
        #pragma unroll 4
        for (int g0 = lane; g0 < groups; g0 += 32) {
            const uint4 P = __ldcs(reinterpret_cast<const uint4 *>(row_w + static_cast<size_t>(g0) * 4));
            const uint4 *x0 = reinterpret_cast<const uint4 *>(xq0 + g0 * 8);
            const uint4 *x1 = reinterpret_cast<const uint4 *>(xq1 + g0 * 8);
            int d0a = 0, d0b = 0, d1a = 0, d1b = 0;
            const uint32_t pw[4] = {P.x, P.y, P.z, P.w};
            #pragma unroll
            for (int wi = 0; wi < 4; wi++) {
                const uint32_t w_ = pw[wi];
                const unsigned long long b0 = btab[w_ & 0xff], b1 = btab[(w_ >> 8) & 0xff];
                const unsigned long long b2 = btab[(w_ >> 16) & 0xff], b3 = btab[w_ >> 24];
                const uint32_t A = __byte_perm(uint32_t(b0), uint32_t(b1), 0xc480);
                const uint32_t B = __byte_perm(uint32_t(b0 >> 32), uint32_t(b1 >> 32), 0x4c80);
                const uint32_t w0 = __byte_perm(A, B, 0x6240);
                const uint32_t A2 = __byte_perm(uint32_t(b2), uint32_t(b3), 0xc480);
                const uint32_t B2 = __byte_perm(uint32_t(b2 >> 32), uint32_t(b3 >> 32), 0x4c80);
                const uint32_t w1 = __byte_perm(A2, B2, 0x6240);
                d0a = __dp4a(int(w0), int(reinterpret_cast<const uint32_t *>(x0)[wi * 2]), d0a);
                d1a = __dp4a(int(w0), int(reinterpret_cast<const uint32_t *>(x1)[wi * 2]), d1a);
                d0b = __dp4a(int(w1), int(reinterpret_cast<const uint32_t *>(x0)[wi * 2 + 1]), d0b);
                d1b = __dp4a(int(w1), int(reinterpret_cast<const uint32_t *>(x1)[wi * 2 + 1]), d1b);
            }
            const float ws = __int_as_float(uint32_t(row_s[g0]) << 23) * 0.5f;
            acc0 = fmaf(float(d0a + d0b), ws * xs[g0], acc0);
            acc1 = fmaf(float(d1a + d1b), ws * xs[groups + g0], acc1);
        }
        #pragma unroll
        for (int mask = 16; mask; mask >>= 1) { acc0 += __shfl_xor_sync(0xffffffff, acc0, mask); acc1 += __shfl_xor_sync(0xffffffff, acc1, mask); }
        if (lane == 0) {
            if (is_a) { ya[row] = acc0; ya[32 + row] = acc1; }
            else { yb[row] = acc0; yb[32 + row] = acc1; }
        }
    }
}
void mxfp4_gemv_ab2_q8(const uint32_t *wa, const uint8_t *sa, const uint32_t *wb, const uint8_t *sb, const float *x, float *ya, float *yb, int cols, cudaStream_t stream) {
    const int groups = cols >> 5;
    mxfp4_gemv_ab2_q8_kernel<<<1, 256, size_t(16) * groups * 4 + 2 * groups * 4 + 2048, stream>>>(wa, sa, wb, sb, x, ya, yb, groups);
}

}
