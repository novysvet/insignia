#include "insignia_qwen35.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
int wmain(int argc,wchar_t**argv){if(argc!=2)return 2;try{insignia::ModelFile m(argv[1]);insignia::Qwen35Weights q(m,1ull<<30);float*d;cudaMalloc(&d,4096*4);q.embed(42,d);std::vector<float>x(4096);cudaMemcpy(x.data(),d,4096*4,cudaMemcpyDeviceToHost);double sum=0;float mx=0;for(float z:x){sum+=z;mx=fmaxf(mx,fabsf(z));}auto a=q.matrix("language_model.model.layers.3.self_attn.q_proj");printf("embed token=42 sum=%g max=%g; layer3 q_proj=%dx%d device=%lluMiB\n",sum,mx,a.rows,a.cols,(unsigned long long)(q.storage().device_bytes()>>20));q.release("language_model.model.layers.3.self_attn.q_proj");cudaFree(d);return mx>0&&a.rows==8192&&a.cols==4096?0:1;}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 3;}}
