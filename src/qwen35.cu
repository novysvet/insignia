#include "insignia_qwen35.hpp"
#include "insignia_layout.cuh"
#include <stdexcept>
namespace insignia {
Qwen35Weights::Qwen35Weights(const ModelFile&m,uint64_t b,cudaStream_t s):storage_(m,b,s),stream_(s){if(cudaMalloc(&scratch_int_,sizeof(int)))throw std::runtime_error("scratch alloc failed");}
Qwen35Weights::~Qwen35Weights(){cudaFree(scratch_int_);}
QuantMatrix Qwen35Weights::matrix(const std::string&base){
 auto w=storage_.acquire(base+".weight");
 if(!w.shape||w.shape->size()!=2)throw std::runtime_error("not a 2-D matrix: "+base);
 const int rows=int((*w.shape)[0]);
 if(w.dtype==DType::u32){ // MXFP4/INSIG4 nibble family: shape[1] counts u32 words per row
  const int cols=int((*w.shape)[1])*8;
  auto s=storage_.acquire(base+".scales");
  const bool i4=s.dtype==DType::f16;
  if(!i4&&s.dtype!=DType::u8)throw std::runtime_error("unsupported scale dtype: "+base);
  if(s.bytes!=(i4?uint64_t(rows)*(cols/64)*2:uint64_t(rows)*(cols/32)))throw std::runtime_error("scale shape mismatch: "+base);
  return {w,s,rows,cols,i4,i4?WKind::mxfp4_i4:WKind::mxfp4_mlx,true};
 }
 if(w.dtype==DType::f8_e4m3){ // Qwen official FP8: bf16 weight_scale_inv per 128x128 block
  const int cols=int((*w.shape)[1]);
  auto s=storage_.acquire(base+".scales");
  if(s.dtype!=DType::bf16||s.bytes!=uint64_t((rows+127)/128)*((cols+127)/128)*2)throw std::runtime_error("bad fp8 scale tensor: "+base);
  return {w,s,rows,cols,false,WKind::fp8,true};
 }
 if(w.dtype==DType::bf16){ // raw bf16 linear (mtp.fc, 27B a/b, lm_head/embed of FP8 checkpoints)
  const int cols=int((*w.shape)[1]);
  return {w,DeviceView{},rows,cols,false,WKind::bf16,false};
 }
 throw std::runtime_error("unsupported weight dtype for "+base);
}
void Qwen35Weights::release(const std::string&base)noexcept{storage_.release(base+".weight");storage_.release(base+".scales");}
void Qwen35Weights::embed(int token,float*out){if(token<0||token>=Qwen35Shape::vocab)throw std::runtime_error("token out of range");cudaMemcpyAsync(scratch_int_,&token,sizeof(int),cudaMemcpyHostToDevice,stream_);embed_dev(scratch_int_,out,stream_);}
void Qwen35Weights::embed_dev(const int*token_dev,float*out,cudaStream_t stream){const std::string base="language_model.model.embed_tokens";auto m=matrix(base);if(m.insig4)mxfp4_get_row_i4(static_cast<const uint32_t*>(m.weight.data),static_cast<const uint16_t*>(m.scales.data),out,token_dev,Qwen35Shape::hidden,stream);else mxfp4_get_row_mlx(static_cast<const uint32_t*>(m.weight.data),static_cast<const uint8_t*>(m.scales.data),out,token_dev,Qwen35Shape::hidden,stream);release(base);}
}
