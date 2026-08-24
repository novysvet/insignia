#include "insignia_model.hpp"
#include <cstdio>
#include <exception>
int wmain(int argc,wchar_t**argv){if(argc!=2){std::fprintf(stderr,"usage: test-model INDEX\n");return 2;}try{insignia::ModelFile m(argv[1]);auto*w=m.find("language_model.model.layers.0.linear_attn.in_proj_qkv.weight");auto*s=m.find("language_model.model.layers.0.linear_attn.in_proj_qkv.scales");if(!w||!s)return 3;std::printf("mapped %zu tensors; qkv weight=%llu bytes scales=%llu bytes first=%08x scale=%u\n",m.tensors().size(),(unsigned long long)w->bytes,(unsigned long long)s->bytes,*reinterpret_cast<const unsigned*>(w->data),unsigned(*reinterpret_cast<const unsigned char*>(s->data)));return m.tensors().size()==699?0:4;}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
