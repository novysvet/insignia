#include "insignia_qwen35.hpp"
#include "insignia_layout.cuh"
#include <stdexcept>
namespace insignia {
Qwen35Weights::Qwen35Weights(const ModelFile&m,uint64_t b,cudaStream_t s):storage_(m,b,s),stream_(s){if(cudaMalloc(&scratch_int_,sizeof(int)))throw std::runtime_error("scratch alloc failed");}
Qwen35Weights::~Qwen35Weights(){cudaFree(scratch_int_);}
QuantMatrix Qwen35Weights::matrix(const std::string&base){auto w=storage_.acquire(base+".weight"),s=storage_.acquire(base+".scales");if(w.dtype!=DType::u32||s.dtype!=DType::u8||!w.shape||w.shape->size()!=2)throw std::runtime_error("not an MXFP4 matrix: "+base);const int rows=int((*w.shape)[0]),cols=int((*w.shape)[1])*8;if(s.bytes!=uint64_t(rows)*(cols/32))throw std::runtime_error("MXFP4 scale shape mismatch: "+base);return {w,s,rows,cols};}
void Qwen35Weights::release(const std::string&base)noexcept{storage_.release(base+".weight");storage_.release(base+".scales");}
void Qwen35Weights::embed(int token,float*out){if(token<0||token>=Qwen35Shape::vocab)throw std::runtime_error("token out of range");cudaMemcpyAsync(scratch_int_,&token,sizeof(int),cudaMemcpyHostToDevice,stream_);embed_dev(scratch_int_,out,stream_);}
void Qwen35Weights::embed_dev(const int*token_dev,float*out,cudaStream_t stream){const std::string base="language_model.model.embed_tokens";auto m=matrix(base);mxfp4_get_row_mlx(static_cast<const uint32_t*>(m.weight.data),static_cast<const uint8_t*>(m.scales.data),out,token_dev,Qwen35Shape::hidden,stream);release(base);}
}
