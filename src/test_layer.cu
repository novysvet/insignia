#include "insignia_decode.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
int wmain(int argc,wchar_t**argv){if(argc!=2)return 2;try{insignia::ModelFile m(argv[1]);insignia::DecodeWorkspace x;insignia::Qwen35Weights w(m,768ull<<20,x.stream);w.embed(42,x.hidden);insignia::Qwen35Decode d(w,x);cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a,x.stream);d.delta_layer(0);d.attention_layer(3);cudaEventRecord(b,x.stream);cudaEventSynchronize(b);float cold_ms;cudaEventElapsedTime(&cold_ms,a,b);cudaEventRecord(a,x.stream);d.delta_layer(0);d.attention_layer(3);cudaEventRecord(b,x.stream);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);std::vector<float>h(4096);cudaMemcpy(h.data(),x.hidden,h.size()*4,cudaMemcpyDeviceToHost);double sum=0;float mx=0;for(float z:h){sum+=z;mx=fmaxf(mx,fabsf(z));}printf("real layer0 decode: sum=%g max=%g cold=%.3f ms warm=%.3f ms resident=%lluMiB\n",sum,mx,cold_ms,ms,(unsigned long long)(w.storage().device_bytes()>>20));return std::isfinite(sum)&&mx>0?0:1;}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 3;}}
