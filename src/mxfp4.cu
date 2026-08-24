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
__global__ void mxfp4_get_row_kernel(const uint32_t*weights,const uint8_t*scales,float*out,int row,int groups){for(int group=blockIdx.x;group<groups;group+=gridDim.x){const int lane=threadIdx.x;const uint32_t packed=weights[(static_cast<size_t>(row)*groups+group)*4+(lane>>3)];const uint8_t q=uint8_t(packed>>((lane&7)*4));out[group*32+lane]=fp4_e2m1(q)*__int_as_float(uint32_t(scales[static_cast<size_t>(row)*groups+group])<<23);}}
void mxfp4_get_row_mlx(const uint32_t*w,const uint8_t*s,float*out,int row,int cols,cudaStream_t stream){mxfp4_get_row_kernel<<<((cols>>5)<128?(cols>>5):128),32,0,stream>>>(w,s,out,row,cols>>5);}
}
