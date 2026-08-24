#include "insignia_decode.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
int wmain(int argc,wchar_t**argv){if(argc!=3)return 2;try{insignia::ModelFile m(argv[1]);insignia::Qwen35Weights w(m,6ull<<30);insignia::DecodeWorkspace x(16);w.embed(42,x.hidden);insignia::Qwen35Decode d(w,x);FILE*f=_wfopen(argv[2],L"wb");for(int l=0;l<32;l++){d.layer(l);cudaDeviceSynchronize();std::vector<float>h(4096);cudaMemcpy(h.data(),x.hidden,h.size()*4,cudaMemcpyDeviceToHost);fwrite(h.data(),4,h.size(),f);}fclose(f);printf("dumped 32 layer seams\n");return 0;}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 3;}}
