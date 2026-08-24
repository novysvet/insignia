#include "insignia_decode.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
int wmain(int argc,wchar_t**argv){if(argc!=2)return 2;try{insignia::ModelFile m(argv[1]);insignia::Qwen35Weights w(m,6ull<<30);insignia::DecodeWorkspace x(512);w.embed(42,x.hidden);insignia::Qwen35Decode d(w,x);cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int l=0;l<32;l++)d.layer(l);cudaEventRecord(b);cudaEventSynchronize(b);float cold;cudaEventElapsedTime(&cold,a,b);x.position++;cudaEventRecord(a);for(int l=0;l<32;l++)d.layer(l);cudaEventRecord(b);cudaEventSynchronize(b);float warm;cudaEventElapsedTime(&warm,a,b);std::vector<float>h(4096);cudaMemcpy(h.data(),x.hidden,h.size()*4,cudaMemcpyDeviceToHost);double sum=0;float mx=0;for(float z:h){sum+=z;mx=fmaxf(mx,fabsf(z));}printf("Qwen3.5 all 32 layers: cold=%.3f ms warm=%.3f ms projected=%.2f tok/s resident=%lluMiB sum=%g max=%g\n",cold,warm,1000.0/warm,(unsigned long long)(w.storage().device_bytes()>>20),sum,mx);return std::isfinite(sum)&&mx>0?0:1;}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 3;}}
