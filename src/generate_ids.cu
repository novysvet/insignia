#include "insignia_decode.hpp"
#include <cuda_runtime.h>
#include <cstdio>
int wmain(int argc,wchar_t**argv){if(argc<3)return 2;try{insignia::ModelFile m(argv[1]);insignia::DecodeWorkspace x(512);insignia::Qwen35Weights w(m,6ull<<30,x.stream);insignia::Qwen35Decode d(w,x);for(int i=2;i<argc;i++){int token=_wtoi(argv[i]);d.forward_token(token);}int next=d.logits_argmax();printf("prompt_tokens=%d next=%d position=%d\n",argc-2,next,x.position);for(int i=0;i<8;i++){next=d.decode_token(next);printf(" %d",next);}printf("\n");return 0;}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 3;}}
