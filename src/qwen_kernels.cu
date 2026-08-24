#include "insignia_qwen_kernels.cuh"
#include <cuda_bf16.h>
namespace insignia {
__device__ __forceinline__ float bf(const uint16_t*p){return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(p));}
template<bool Z,bool G>__global__ void rms_bf(const float*x,const uint16_t*w,const float*g,float*y,int cols){int r=blockIdx.x;float ss=0;for(int i=threadIdx.x;i<cols;i+=256){float z=x[r*cols+i];ss=fmaf(z,z,ss);}for(int m=16;m;m>>=1)ss+=__shfl_xor_sync(0xffffffff,ss,m);__shared__ float p[8];int lane=threadIdx.x&31,warp=threadIdx.x>>5;if(!lane)p[warp]=ss;__syncthreads();if(!warp){ss=lane<8?p[lane]:0;for(int m=16;m;m>>=1)ss+=__shfl_xor_sync(0xffffffff,ss,m);if(!lane)p[0]=rsqrtf(ss/cols+1e-6f);}__syncthreads();for(int i=threadIdx.x;i<cols;i+=256){float z=x[r*cols+i]*p[0]*(Z?1+bf(w+i):bf(w+i));if constexpr(G){float v=g[r*cols+i];z*=v/(1+__expf(-v));}y[r*cols+i]=z;}}
void rmsnorm_bf16(const float*x,const uint16_t*w,float*y,int r,int c,bool z,cudaStream_t s){if(z)rms_bf<true,false><<<r,256,0,s>>>(x,w,nullptr,y,c);else rms_bf<false,false><<<r,256,0,s>>>(x,w,nullptr,y,c);}void gated_rmsnorm_bf16(const float*x,const uint16_t*w,const float*g,float*y,int r,int c,cudaStream_t s){rms_bf<false,true><<<r,256,0,s>>>(x,w,g,y,c);}
__global__ void conv4(float*x,float*state,const uint16_t*w,int n){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){float z=fmaf(state[i*3],bf(w+i*4),fmaf(state[i*3+1],bf(w+i*4+1),fmaf(state[i*3+2],bf(w+i*4+2),x[i]*bf(w+i*4+3))));state[i*3]=state[i*3+1];state[i*3+1]=state[i*3+2];state[i*3+2]=x[i];x[i]=z/(1+__expf(-z));}}
void causal_conv4_silu(float*x,float*st,const uint16_t*w,int n,cudaStream_t s){conv4<<<(n+255)/256,256,0,s>>>(x,st,w,n);}
__global__ void params(float*a,float*b,const float*A,const uint16_t*dt,int n){int i=threadIdx.x;if(i<n){b[i]=1/(1+__expf(-b[i]));float z=a[i]+bf(dt+i);float soft=z>20?z:log1pf(__expf(z));a[i]=-__expf(A[i])*soft;}}
void deltanet_parameters(float*a,float*b,const float*A,const uint16_t*dt,int n,cudaStream_t s){params<<<1,32,0,s>>>(a,b,A,dt,n);}
__global__ void sigmul(float*x,const float*g,int n){for(int i=blockIdx.x*256+threadIdx.x;i<n;i+=gridDim.x*256)x[i]*=1/(1+__expf(-g[i]));}void sigmoid_mul(float*x,const float*g,int n,cudaStream_t s){sigmul<<<(n+255)/256,256,0,s>>>(x,g,n);}
}

namespace insignia {
__global__ void store_kv_kernel(const float*k,const float*v,float*kc,float*vc,const int*pos_dev,int base){int i=blockIdx.x*blockDim.x+threadIdx.x;const size_t pos=size_t(__ldg(pos_dev)+base);if(i<1024){kc[pos*1024+i]=k[i];vc[pos*1024+i]=v[i];}}
void store_kv(const float*k,const float*v,float*kc,float*vc,const int*pos_dev,int base,cudaStream_t s){store_kv_kernel<<<4,256,0,s>>>(k,v,kc,vc,pos_dev,base);}
}

namespace insignia {
__global__ void argmax_kernel(const float*x,int n,int*out){float best=-3.402823466e38F;int idx=0;for(int i=threadIdx.x;i<n;i+=blockDim.x){float v=x[i];if(v>best){best=v;idx=i;}}for(int m=16;m;m>>=1){float v=__shfl_xor_sync(0xffffffff,best,m);int j=__shfl_xor_sync(0xffffffff,idx,m);if(v>best){best=v;idx=j;}}__shared__ float bv[8];__shared__ int bi[8];int lane=threadIdx.x&31,warp=threadIdx.x>>5;if(!lane){bv[warp]=best;bi[warp]=idx;}__syncthreads();if(!warp){best=lane<8?bv[lane]:-3.402823466e38F;idx=lane<8?bi[lane]:0;for(int m=16;m;m>>=1){float v=__shfl_xor_sync(0xffffffff,best,m);int j=__shfl_xor_sync(0xffffffff,idx,m);if(v>best){best=v;idx=j;}}if(!lane)*out=idx;}}
void argmax_logits(const float*x,int n,int*out,cudaStream_t s){argmax_kernel<<<1,256,0,s>>>(x,n,out);}

// Two-stage argmax for the 248K-vocab lm_head: stage 1 reduces 512-wide blocks into one
// monotonic u64 key (float order bits << 32 | index) via atomicMax, stage 2 unpacks.
__global__ void argmax_stage1_kernel(const float *__restrict__ x, int n, unsigned long long *__restrict__ best) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float best_v = -3.402823466e38F;
    int best_i = 0;
    for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
        const float v = __ldg(x + i);
        if (v > best_v) { best_v = v; best_i = i; }
    }
    for (int m = 16; m; m >>= 1) {
        const float v = __shfl_xor_sync(0xffffffff, best_v, m);
        const int j = __shfl_xor_sync(0xffffffff, best_i, m);
        if (v > best_v) { best_v = v; best_i = j; }
    }
    __shared__ float bv[16];
    __shared__ int bi[16];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (!lane) { bv[warp] = best_v; bi[warp] = best_i; }
    __syncthreads();
    if (!warp) {
        best_v = lane < 16 ? bv[lane] : -3.402823466e38F;
        best_i = lane < 16 ? bi[lane] : 0;
        for (int m = 16; m; m >>= 1) {
            const float v = __shfl_xor_sync(0xffffffff, best_v, m);
            const int j = __shfl_xor_sync(0xffffffff, best_i, m);
            if (v > best_v) { best_v = v; best_i = j; }
        }
        if (!lane) {
            uint32_t bits = __float_as_uint(best_v);
            bits ^= (uint32_t(int32_t(bits) >> 31)) | 0x80000000u;  // total order for +/- floats
            atomicMax(best, (unsigned long long(bits) << 32) | unsigned(best_i));
        }
    }
}
__global__ void argmax_stage2_kernel(const unsigned long long *__restrict__ best, int *__restrict__ out) { *out = int(*best & 0xffffffffu); }
void argmax_fast(const float *x, int n, int *out, unsigned long long *scratch, cudaStream_t s) {
    cudaMemsetAsync(scratch, 0, 8, s);
    argmax_stage1_kernel<<<64, 512, 0, s>>>(x, n, scratch);
    argmax_stage2_kernel<<<1, 1, 0, s>>>(scratch, out);
}
}

namespace insignia {
__global__ void bf16_gemv_kernel(const uint16_t*w,const float*x,float*y,int cols){int row=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x>>5;float z=0;for(int i=threadIdx.x;i<cols;i+=blockDim.x)z=fmaf(bf(w+size_t(row)*cols+i),x[i],z);for(int m=16;m;m>>=1)z+=__shfl_xor_sync(0xffffffff,z,m);__shared__ float p[8];if(!lane)p[warp]=z;__syncthreads();if(!warp){z=lane<8?p[lane]:0;for(int m=16;m;m>>=1)z+=__shfl_xor_sync(0xffffffff,z,m);if(!lane)y[row]=z;}}
void bf16_gemv(const uint16_t*w,const float*x,float*y,int rows,int cols,cudaStream_t s){bf16_gemv_kernel<<<rows,256,0,s>>>(w,x,y,cols);}
__global__ void concat_kernel(const float*a,const float*b,float*out,int n){int i=blockIdx.x*256+threadIdx.x;if(i<n){out[i]=a[i];out[n+i]=b[i];}}void concat(const float*a,const float*b,float*out,int n,cudaStream_t s){concat_kernel<<<(n+255)/256,256,0,s>>>(a,b,out,n);}
}

namespace insignia {
__global__ void split_q_gate_kernel(const float*src,float*q,float*g){int i=blockIdx.x*256+threadIdx.x;if(i<4096){int h=i>>8,d=i&255;q[i]=src[h*512+d];g[i]=src[h*512+256+d];}}
void split_q_gate(const float*s,float*q,float*g,cudaStream_t stream){split_q_gate_kernel<<<16,256,0,stream>>>(s,q,g);}
}

namespace insignia {
__global__ void expand_gate_kernel(const float*g,float*out){int i=blockIdx.x*256+threadIdx.x;if(i<4096)out[i]=g[i];}
void expand_gate_heads(const float*g,float*out,cudaStream_t s){expand_gate_kernel<<<16,256,0,s>>>(g,out);}
}
