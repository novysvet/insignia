#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
__global__ void smoke(uint32_t *out) { out[0] = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x); }
int main() { int n=0; cudaDeviceProp p{}; cudaGetDeviceCount(&n); if(n<1){std::fprintf(stderr,"no CUDA device\n");return 2;} cudaGetDeviceProperties(&p,0); if(p.major!=8 || p.minor!=9){std::fprintf(stderr,"expected sm_89, got sm_%d%d\n",p.major,p.minor);return 3;} uint32_t *d=nullptr; cudaMalloc(&d,sizeof(uint32_t)); smoke<<<1,32>>>(d); auto e=cudaDeviceSynchronize(); if(e!=cudaSuccess){std::fprintf(stderr,"kernel: %s\n",cudaGetErrorString(e));return 4;} uint32_t x=0; cudaMemcpy(&x,d,sizeof(x),cudaMemcpyDeviceToHost); cudaFree(d); std::printf("Insignia smoke OK: %s sm_%d%d, thread=%u\n",p.name,p.major,p.minor,x); return 0; }
