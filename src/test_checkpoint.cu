#include "insignia_model.hpp"
#include "insignia_storage.hpp"
#include "insignia_layout.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <exception>
#include <vector>
#include <stdexcept>
#include <cmath>
int wmain(int argc,wchar_t**argv){if(argc!=2)return 2;try{insignia::ModelFile model(argv[1]);insignia::TieredStorage store(model,32ull<<20);const char*wn="language_model.model.layers.0.linear_attn.in_proj_qkv.weight";const char*sn="language_model.model.layers.0.linear_attn.in_proj_qkv.scales";auto w=store.acquire(wn),s=store.acquire(sn);std::vector<float>x(4096),y(8192);for(int i=0;i<4096;i++)x[i]=float((i*17)%31-15)/16.f;float *dx,*dy;cudaMalloc(&dx,x.size()*4);cudaMalloc(&dy,y.size()*4);cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice);insignia::mxfp4_gemv_mlx((const uint32_t*)w.data,(const uint8_t*)s.data,dx,dy,8192,4096,2);auto e=cudaDeviceSynchronize();if(e)throw std::runtime_error(cudaGetErrorString(e));cudaMemcpy(y.data(),dy,y.size()*4,cudaMemcpyDeviceToHost);double checksum=0;float maxv=0;for(float z:y){checksum+=z;maxv=fmaxf(maxv,fabsf(z));}printf("actual checkpoint qkv: device=%llu MiB checksum=%.9g max_abs=%.9g first=[%.6g %.6g %.6g]\n",(unsigned long long)(store.device_bytes()>>20),checksum,maxv,y[0],y[1],y[2]);store.release(wn);store.release(sn);auto other=store.acquire("language_model.model.layers.0.mlp.gate_proj.weight");printf("LRU eviction/reload: device=%llu MiB gate=%llu bytes\n",(unsigned long long)(store.device_bytes()>>20),(unsigned long long)other.bytes);cudaFree(dx);cudaFree(dy);return std::isfinite(checksum)&&maxv>0?0:3;}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
