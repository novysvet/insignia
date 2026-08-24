#include "insignia_deltanet.cuh"
#include <cuda_runtime.h>
namespace insignia {
__global__ __launch_bounds__(128,4) void deltanet_decode_kernel(float *__restrict__ state,const float *__restrict__ q16,const float *__restrict__ k16,const float *__restrict__ v,const float *__restrict__ g,const float *__restrict__ beta,float *__restrict__ out){
 const int head=blockIdx.x,tid=threadIdx.x,kh=head>>1; float q=q16[kh*128+tid],k=k16[kh*128+tid];float qn=q*q,kn=k*k;
 for(int m=16;m;m>>=1){qn+=__shfl_xor_sync(0xffffffff,qn,m);kn+=__shfl_xor_sync(0xffffffff,kn,m);}
 __shared__ float sq[4],sk[4],delta[128];const int lane=tid&31,warp=tid>>5;if(lane==0){sq[warp]=qn;sk[warp]=kn;}__syncthreads();
 if(tid==0){float a=sq[0]+sq[1]+sq[2]+sq[3],b=sk[0]+sk[1]+sk[2]+sk[3];sq[0]=rsqrtf(a+1e-6f)*0.08838834764831845f;sk[0]=rsqrtf(b+1e-6f);}__syncthreads();q*=sq[0];k*=sk[0];
 float *S=state+head*128*128;const float decay=expf(g[head]);
 float dot=0;for(int i=0;i<128;i++)dot=fmaf(S[i*128+tid]*decay,k16[kh*128+i]*sk[0],dot);
 delta[tid]=(v[head*128+tid]-dot)*beta[head];__syncthreads();
 float acc=0;for(int i=0;i<128;i++){float &cell=S[i*128+tid];cell=fmaf(cell,decay,k16[kh*128+i]*sk[0]*delta[tid]);acc=fmaf(cell,q16[kh*128+i]*sq[0],acc);}out[head*128+tid]=acc;
}
void deltanet_decode(float*s,const float*q,const float*k,const float*v,const float*g,const float*b,float*o,cudaStream_t stream){deltanet_decode_kernel<<<32,128,0,stream>>>(s,q,k,v,g,b,o);}
}
