#include "insignia_ops.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#define OK(x) do{auto e=(x);if(e){printf("%s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){constexpr int R=3,C=4096;std::vector<float>x(R*C),w(C),g(R*C),ref(R*C),y(R*C);for(int i=0;i<R*C;i++){x[i]=float(int(i*13%43)-21)*.03f;g[i]=float(int(i*7%29)-14)*.05f;}for(int i=0;i<C;i++)w[i]=float(int(i*5%17)-8)*.01f;for(int r=0;r<R;r++){double ss=0;for(int i=0;i<C;i++)ss+=double(x[r*C+i])*x[r*C+i];float inv=1.f/sqrtf(float(ss/C)+1e-6f);for(int i=0;i<C;i++)ref[r*C+i]=x[r*C+i]*inv*(1+w[i]);}float*dx,*dw,*dg,*dy;OK(cudaMalloc(&dx,x.size()*4));OK(cudaMalloc(&dw,w.size()*4));OK(cudaMalloc(&dg,g.size()*4));OK(cudaMalloc(&dy,y.size()*4));OK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));OK(cudaMemcpy(dw,w.data(),w.size()*4,cudaMemcpyHostToDevice));OK(cudaMemcpy(dg,g.data(),g.size()*4,cudaMemcpyHostToDevice));insignia::rmsnorm_zero_centered(dx,dw,dy,R,C);OK(cudaDeviceSynchronize());OK(cudaMemcpy(y.data(),dy,y.size()*4,cudaMemcpyDeviceToHost));float err=0;for(size_t i=0;i<y.size();i++)err=fmaxf(err,fabsf(y[i]-ref[i]));printf("RMSNorm max_abs=%g\n",err);return err<2e-6f?0:1;}
