#include "insignia_decode.hpp"
#include "insignia_layout.cuh"
#include "insignia_deltanet.cuh"
#include "insignia_qwen_kernels.cuh"
#include "insignia_ops.cuh"
#include "insignia_attention.cuh"
#include "insignia_prefill.cuh"
#include <stdexcept>
namespace insignia {
static void alloc(float**p,size_t n){auto e=cudaMalloc(p,n*sizeof(float));if(e)throw std::runtime_error(cudaGetErrorString(e));}
DecodeWorkspace::DecodeWorkspace(int ctx,cudaStream_t s):max_context(ctx),stream(s){
 if(ctx<=0||ctx>4096)throw std::runtime_error("max_context must be in 1..4096 (gqa score buffer)");
 if(!stream){auto e=cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);if(e)throw std::runtime_error(cudaGetErrorString(e));}
 alloc(&hidden,4096);alloc(&norm,4096);alloc(&qkv,8192);alloc(&attn_gate,4096);alloc(&key,1024);alloc(&value,1024);alloc(&z,4096);alloc(&a,32);alloc(&b,32);alloc(&core,4096);alloc(&gate,12288);alloc(&up,12288);alloc(&down,4096);alloc(&logits,248320*2);alloc(&delta_state,24*32*128*128);alloc(&conv_state,24*8192*3);alloc(&kv_keys,size_t(8)*ctx*1024);alloc(&kv_values,size_t(8)*ctx*1024);alloc(&mtp_keys,size_t(ctx)*1024);alloc(&mtp_values,size_t(ctx)*1024);
 auto e=cudaMalloc(&pos_dev,16*sizeof(int));if(e)throw std::runtime_error(cudaGetErrorString(e));token_dev=pos_dev+1;next_dev=pos_dev+2;next2_dev=pos_dev+3;draft_dev=pos_dev+4;count_dev=pos_dev+5;accflag_dev=pos_dev+6;mtp_pos_dev=pos_dev+7;
 if(cudaHostAlloc(&next_host,sizeof(int),cudaHostAllocDefault))throw std::runtime_error("pinned alloc failed");
 if(cudaHostAlloc(&pos_host,sizeof(int),cudaHostAllocDefault))throw std::runtime_error("pinned alloc failed");
 if(cudaHostAlloc(&host_committed,16384*sizeof(int),cudaHostAllocDefault))throw std::runtime_error("pinned alloc failed");
 e=cudaMalloc(&am_scratch,8);if(e)throw std::runtime_error(cudaGetErrorString(e));
 e=cudaMalloc(&committed,16384*sizeof(int));if(e)throw std::runtime_error(cudaGetErrorString(e));
 e=cudaMalloc(&pf_tokens,64*sizeof(int));if(e)throw std::runtime_error(cudaGetErrorString(e));
 alloc(&pf_x,64*4096);alloc(&pf_n,64*4096);alloc(&pf_qkv,64*8192);alloc(&pf_scratch,64*8192);alloc(&pf_z,64*4096);
 alloc(&pf_q,64*4096);alloc(&pf_g,64*4096);alloc(&pf_k,64*1024);alloc(&pf_v,64*1024);alloc(&pf_core,64*4096);
 alloc(&pf_down,64*4096);alloc(&pf_gate,64*12288);alloc(&pf_up,64*12288);alloc(&pf_a,64*32);alloc(&pf_b,64*32);
 alloc(&snap_delta,24*32*128*128);alloc(&snap_conv,24*8192*3);
 {auto e2=cudaMalloc(&pf_xq8,6144*sizeof(unsigned));if(e2)throw std::runtime_error(cudaGetErrorString(e2));e2=cudaMalloc(&pf_xs8,768*sizeof(float));if(e2)throw std::runtime_error(cudaGetErrorString(e2));}
 cudaMemsetAsync(pos_dev,0,16*sizeof(int),stream);cudaMemsetAsync(am_scratch,0,8,stream);cudaMemsetAsync(delta_state,0,24*32*128*128*4,stream);cudaMemsetAsync(conv_state,0,24*8192*3*4,stream);
}
DecodeWorkspace::~DecodeWorkspace(){cudaFree(hidden);cudaFree(norm);cudaFree(qkv);cudaFree(attn_gate);cudaFree(key);cudaFree(value);cudaFree(z);cudaFree(a);cudaFree(b);cudaFree(core);cudaFree(gate);cudaFree(up);cudaFree(down);cudaFree(logits);cudaFree(delta_state);cudaFree(conv_state);cudaFree(kv_keys);cudaFree(kv_values);cudaFree(mtp_keys);cudaFree(mtp_values);cudaFree(pos_dev);cudaFreeHost(next_host);cudaFreeHost(pos_host);cudaFree(host_committed);cudaFree(am_scratch);cudaFree(committed);cudaFree(snap_delta);cudaFree(snap_conv);cudaFree(pf_tokens);cudaFree(pf_x);cudaFree(pf_n);cudaFree(pf_qkv);cudaFree(pf_scratch);cudaFree(pf_z);cudaFree(pf_q);cudaFree(pf_g);cudaFree(pf_k);cudaFree(pf_v);cudaFree(pf_core);cudaFree(pf_down);cudaFree(pf_gate);cudaFree(pf_up);cudaFree(pf_a);cudaFree(pf_b);cudaFree(pf_xq8);cudaFree(pf_xs8);}
DeviceView Qwen35Decode::tensor(const std::string&name){return w_.storage().acquire(name);}
void Qwen35Decode::linear(const std::string&base,const float*in,float*out){auto m=w_.matrix(base);mxfp4_gemv_v2((const uint32_t*)m.weight.data,(const uint8_t*)m.scales.data,in,out,m.rows,m.cols,x_.stream);w_.release(base);}
void Qwen35Decode::linear2(const std::string&base,const float*in,float*out){auto m=w_.matrix(base);mxfp4_gemv2_q8((const uint32_t*)m.weight.data,(const uint8_t*)m.scales.data,in,out,m.rows,m.cols,x_.stream);w_.release(base);}
void Qwen35Decode::linear_batch(const std::string&base,const float*in,float*out,int T){auto m=w_.matrix(base);mxfp4_gemm_mlx((const uint32_t*)m.weight.data,(const uint8_t*)m.scales.data,in,out,m.rows,m.cols,T,x_.stream);w_.release(base);}
void Qwen35Decode::prefill_chunk_device(const int *tokens_dev, int T) {
 if(T<=0||T>64)throw std::runtime_error("prefill chunk must be 1..64 tokens");
 if(x_.position+T>x_.max_context)throw std::runtime_error("KV cache full");
 {const std::string base="language_model.model.embed_tokens";auto m=w_.matrix(base);embed_gather((const uint32_t*)m.weight.data,(const uint8_t*)m.scales.data,tokens_dev,x_.pf_x,T,x_.stream);w_.release(base);}
 for(int l=0;l<32;l++){
  const std::string p="language_model.model.layers."+std::to_string(l);
  auto inw=tensor(p+".input_layernorm.weight");rmsnorm_bf16(x_.pf_x,(const uint16_t*)inw.data,x_.pf_n,T,4096,false,x_.stream);w_.storage().release(p+".input_layernorm.weight");
  const bool pair=T==2;
  if(Qwen35Shape::full_attention(l)){
   const std::string a=p+".self_attn";
   if(pair)linear2(a+".q_proj",x_.pf_n,x_.pf_scratch);else linear_batch(a+".q_proj",x_.pf_n,x_.pf_scratch,T);
   split_q_gate_batch(x_.pf_scratch,x_.pf_q,x_.pf_g,T,x_.stream);
   if(pair){linear2(a+".k_proj",x_.pf_n,x_.pf_k);linear2(a+".v_proj",x_.pf_n,x_.pf_v);}else{linear_batch(a+".k_proj",x_.pf_n,x_.pf_k,T);linear_batch(a+".v_proj",x_.pf_n,x_.pf_v,T);}
   auto qw=tensor(a+".q_norm.weight"),kw=tensor(a+".k_norm.weight");
   qk_norm_rope_batch(x_.pf_q,x_.pf_k,(const uint16_t*)qw.data,(const uint16_t*)kw.data,x_.pos_dev,T,x_.stream);
   w_.storage().release(a+".q_norm.weight");w_.storage().release(a+".k_norm.weight");
   const int ai=l/4;float*kc=x_.kv_keys+size_t(ai)*x_.max_context*1024,*vc=x_.kv_values+size_t(ai)*x_.max_context*1024;
   store_kv_batch(x_.pf_k,x_.pf_v,kc,vc,x_.pos_dev,T,x_.max_context,x_.stream);
   gqa_prefill(x_.pf_q,kc,vc,x_.pf_core,x_.pos_dev,T,x_.max_context,x_.stream);
   sigmoid_mul(x_.pf_core,x_.pf_g,size_t(T)*4096,x_.stream);
   if(pair)linear2(a+".o_proj",x_.pf_core,x_.pf_down);else linear_batch(a+".o_proj",x_.pf_core,x_.pf_down,T);
  }else{
   const std::string a=p+".linear_attn";
   if(pair){linear2(a+".in_proj_qkv",x_.pf_n,x_.pf_qkv);linear2(a+".in_proj_z",x_.pf_n,x_.pf_z);
    auto ma=w_.matrix(a+".in_proj_a"),mb=w_.matrix(a+".in_proj_b");
    mxfp4_gemv_ab2_q8((const uint32_t*)ma.weight.data,(const uint8_t*)ma.scales.data,(const uint32_t*)mb.weight.data,(const uint8_t*)mb.scales.data,x_.pf_n,x_.pf_a,x_.pf_b,ma.cols,x_.stream);
    w_.release(a+".in_proj_a");w_.release(a+".in_proj_b");}
   else{linear_batch(a+".in_proj_qkv",x_.pf_n,x_.pf_qkv,T);linear_batch(a+".in_proj_z",x_.pf_n,x_.pf_z,T);
    {auto m=w_.matrix(a+".in_proj_a");for(int t=0;t<T;t++)mxfp4_gemv_v2((const uint32_t*)m.weight.data,(const uint8_t*)m.scales.data,x_.pf_n+size_t(t)*4096,x_.pf_a+size_t(t)*32,m.rows,m.cols,x_.stream);w_.release(a+".in_proj_a");}
    {auto m=w_.matrix(a+".in_proj_b");for(int t=0;t<T;t++)mxfp4_gemv_v2((const uint32_t*)m.weight.data,(const uint8_t*)m.scales.data,x_.pf_n+size_t(t)*4096,x_.pf_b+size_t(t)*32,m.rows,m.cols,x_.stream);w_.release(a+".in_proj_b");}}
   auto cw=tensor(a+".conv1d.weight");const int di=l-l/4;
   conv_prefill_silu(x_.pf_qkv,x_.pf_scratch,x_.conv_state+size_t(di)*8192*3,(const uint16_t*)cw.data,T,x_.stream,x_.snap_conv+size_t(di)*8192*3);
   w_.storage().release(a+".conv1d.weight");
   auto A=tensor(a+".A_log"),dt=tensor(a+".dt_bias");
   deltanet_params_batch(x_.pf_a,x_.pf_b,(const float*)A.data,(const uint16_t*)dt.data,T,x_.stream);
   w_.storage().release(a+".A_log");w_.storage().release(a+".dt_bias");
   deltanet_prefill(x_.delta_state+size_t(di)*32*128*128,x_.pf_scratch,x_.pf_a,x_.pf_b,x_.pf_core,T,x_.stream,x_.snap_delta+size_t(di)*32*128*128);
   auto nw=tensor(a+".norm.weight");gated_rmsnorm_bf16(x_.pf_core,(const uint16_t*)nw.data,x_.pf_z,x_.pf_core,size_t(T)*32,128,x_.stream);w_.storage().release(a+".norm.weight");
   if(pair)linear2(a+".out_proj",x_.pf_core,x_.pf_down);else linear_batch(a+".out_proj",x_.pf_core,x_.pf_down,T);
  }
  residual_add(x_.pf_x,x_.pf_down,size_t(T)*4096,x_.stream);
  auto post=tensor(p+".post_attention_layernorm.weight");rmsnorm_bf16(x_.pf_x,(const uint16_t*)post.data,x_.pf_n,T,4096,false,x_.stream);w_.storage().release(p+".post_attention_layernorm.weight");
  if(pair){linear2(p+".mlp.gate_proj",x_.pf_n,x_.pf_gate);linear2(p+".mlp.up_proj",x_.pf_n,x_.pf_up);}else{linear_batch(p+".mlp.gate_proj",x_.pf_n,x_.pf_gate,T);linear_batch(p+".mlp.up_proj",x_.pf_n,x_.pf_up,T);}
  silu_mul(x_.pf_gate,x_.pf_up,x_.pf_gate,size_t(T)*12288,x_.stream);
  if(pair)linear2(p+".mlp.down_proj",x_.pf_gate,x_.pf_down);else linear_batch(p+".mlp.down_proj",x_.pf_gate,x_.pf_down,T);
  residual_add(x_.pf_x,x_.pf_down,size_t(T)*4096,x_.stream);
 }
 auto nw=tensor("language_model.model.norm.weight");rmsnorm_bf16(x_.pf_x,(const uint16_t*)nw.data,x_.pf_n,T,4096,false,x_.stream);w_.storage().release("language_model.model.norm.weight");
 {auto m=w_.matrix("language_model.lm_head");
  if(T==2){  // both rows through one lm_head weight pass; argmax each
   mxfp4_gemv2_q8((const uint32_t*)m.weight.data,(const uint8_t*)m.scales.data,x_.pf_n,x_.logits,m.rows,m.cols,x_.stream);
   argmax_fast(x_.logits,Qwen35Shape::vocab,x_.next2_dev,x_.am_scratch,x_.stream);
   argmax_fast(x_.logits+Qwen35Shape::vocab,Qwen35Shape::vocab,x_.next_dev,x_.am_scratch,x_.stream);
  } else {mxfp4_gemv_v2((const uint32_t*)m.weight.data,(const uint8_t*)m.scales.data,x_.pf_n+size_t(T-1)*4096,x_.logits,m.rows,m.cols,x_.stream);argmax_fast(x_.logits,Qwen35Shape::vocab,x_.next_dev,x_.am_scratch,x_.stream);}
  w_.release("language_model.lm_head");}
 cudaMemcpyAsync(x_.hidden,x_.pf_x+size_t(T-1)*4096,4096*4,cudaMemcpyDeviceToDevice,x_.stream);
 addi_kernel_launch(x_.pos_dev,T,x_.stream);
 x_.position+=T;
}
int Qwen35Decode::prefill_chunk(const int*tokens,int T){
 cudaMemcpyAsync(x_.pf_tokens,tokens,sizeof(int)*T,cudaMemcpyHostToDevice,x_.stream);
 prefill_chunk_device(x_.pf_tokens,T);
 cudaMemcpyAsync(x_.next_host,x_.next_dev,sizeof(int),cudaMemcpyDeviceToHost,x_.stream);
 cudaStreamSynchronize(x_.stream);
 return *x_.next_host;
}
__global__ void copyi_kernel(int*dst,const int*src){*dst=*src;}
__global__ void bumpi_kernel(int*p){(*p)++;}
void Qwen35Decode::set_position(int pos){*x_.pos_host=pos;cudaMemcpyAsync(x_.pos_dev,x_.pos_host,sizeof(int),cudaMemcpyHostToDevice,x_.stream);}
void Qwen35Decode::set_mtp_position(int pos){*x_.pos_host=pos;cudaMemcpyAsync(x_.mtp_pos_dev,x_.pos_host,sizeof(int),cudaMemcpyHostToDevice,x_.stream);}
void Qwen35Decode::delta_layer(int l){if(l<0||l>=32||Qwen35Shape::full_attention(l))throw std::runtime_error("not a DeltaNet layer");const std::string p="language_model.model.layers."+std::to_string(l);auto inw=tensor(p+".input_layernorm.weight");rmsnorm_bf16(x_.hidden,(const uint16_t*)inw.data,x_.norm,1,4096,false,x_.stream);w_.storage().release(p+".input_layernorm.weight");
 const std::string a=p+".linear_attn";linear(a+".in_proj_qkv",x_.norm,x_.qkv);linear(a+".in_proj_z",x_.norm,x_.z);linear(a+".in_proj_a",x_.norm,x_.a);linear(a+".in_proj_b",x_.norm,x_.b);
 auto cw=tensor(a+".conv1d.weight");const int di=l-l/4;causal_conv4_silu(x_.qkv,x_.conv_state+size_t(di)*8192*3,(const uint16_t*)cw.data,8192,x_.stream);w_.storage().release(a+".conv1d.weight");auto A=tensor(a+".A_log"),dt=tensor(a+".dt_bias");deltanet_parameters(x_.a,x_.b,(const float*)A.data,(const uint16_t*)dt.data,32,x_.stream);w_.storage().release(a+".A_log");w_.storage().release(a+".dt_bias");deltanet_decode(x_.delta_state+size_t(di)*32*128*128,x_.qkv,x_.qkv+2048,x_.qkv+4096,x_.a,x_.b,x_.core,x_.stream);auto nw=tensor(a+".norm.weight");gated_rmsnorm_bf16(x_.core,(const uint16_t*)nw.data,x_.z,x_.core,32,128,x_.stream);w_.storage().release(a+".norm.weight");linear(a+".out_proj",x_.core,x_.down);residual_add(x_.hidden,x_.down,4096,x_.stream);
 auto post=tensor(p+".post_attention_layernorm.weight");rmsnorm_bf16(x_.hidden,(const uint16_t*)post.data,x_.norm,1,4096,false,x_.stream);w_.storage().release(p+".post_attention_layernorm.weight");linear(p+".mlp.gate_proj",x_.norm,x_.gate);linear(p+".mlp.up_proj",x_.norm,x_.up);silu_mul(x_.gate,x_.up,x_.gate,12288,x_.stream);linear(p+".mlp.down_proj",x_.gate,x_.down);residual_add(x_.hidden,x_.down,4096,x_.stream);
}
void Qwen35Decode::attention_layer(int l){if(l<0||l>=32||!Qwen35Shape::full_attention(l))throw std::runtime_error("not a full attention layer");if(x_.position>=x_.max_context)throw std::runtime_error("KV cache full");const std::string p="language_model.model.layers."+std::to_string(l),a=p+".self_attn";auto inw=tensor(p+".input_layernorm.weight");rmsnorm_bf16(x_.hidden,(const uint16_t*)inw.data,x_.norm,1,4096,false,x_.stream);w_.storage().release(p+".input_layernorm.weight");linear(a+".q_proj",x_.norm,x_.gate);split_q_gate(x_.gate,x_.qkv,x_.attn_gate,x_.stream);linear(a+".k_proj",x_.norm,x_.key);linear(a+".v_proj",x_.norm,x_.value);auto qw=tensor(a+".q_norm.weight"),kw=tensor(a+".k_norm.weight");qwen35_qk_norm_rope_gate(x_.qkv,x_.key,(const uint16_t*)qw.data,(const uint16_t*)kw.data,x_.attn_gate,x_.pos_dev,0,x_.stream);w_.storage().release(a+".q_norm.weight");w_.storage().release(a+".k_norm.weight");const int ai=l/4;float*kc=x_.kv_keys+size_t(ai)*x_.max_context*1024,*vc=x_.kv_values+size_t(ai)*x_.max_context*1024;store_kv(x_.key,x_.value,kc,vc,x_.pos_dev,0,x_.stream);gqa_decode(x_.qkv,kc,vc,x_.core,x_.pos_dev,0,x_.max_context,x_.stream);expand_gate_heads(x_.attn_gate,x_.qkv,x_.stream);sigmoid_mul(x_.core,x_.qkv,4096,x_.stream);linear(a+".o_proj",x_.core,x_.down);residual_add(x_.hidden,x_.down,4096,x_.stream);auto post=tensor(p+".post_attention_layernorm.weight");rmsnorm_bf16(x_.hidden,(const uint16_t*)post.data,x_.norm,1,4096,false,x_.stream);w_.storage().release(p+".post_attention_layernorm.weight");linear(p+".mlp.gate_proj",x_.norm,x_.gate);linear(p+".mlp.up_proj",x_.norm,x_.up);silu_mul(x_.gate,x_.up,x_.gate,12288,x_.stream);linear(p+".mlp.down_proj",x_.gate,x_.down);residual_add(x_.hidden,x_.down,4096,x_.stream);}
void Qwen35Decode::layer(int l){if(Qwen35Shape::full_attention(l))attention_layer(l);else delta_layer(l);}
void Qwen35Decode::forward_body(){for(int l=0;l<32;l++)layer(l);auto nw=tensor("language_model.model.norm.weight");rmsnorm_bf16(x_.hidden,(const uint16_t*)nw.data,x_.norm,1,4096,false,x_.stream);w_.storage().release("language_model.model.norm.weight");linear("language_model.lm_head",x_.norm,x_.logits);}
void Qwen35Decode::forward_token(int token){if(token<0||token>=Qwen35Shape::vocab)throw std::runtime_error("token out of range");if(x_.position>=x_.max_context)throw std::runtime_error("KV cache full");cudaMemcpyAsync(x_.token_dev,&token,sizeof(int),cudaMemcpyHostToDevice,x_.stream);w_.embed_dev(x_.token_dev,x_.hidden,x_.stream);forward_body();bumpi_kernel<<<1,1,0,x_.stream>>>(x_.pos_dev);x_.position++;}
int Qwen35Decode::logits_argmax(){argmax_fast(x_.logits,Qwen35Shape::vocab,x_.next_dev,x_.am_scratch,x_.stream);cudaMemcpyAsync(x_.next_host,x_.next_dev,sizeof(int),cudaMemcpyDeviceToHost,x_.stream);cudaStreamSynchronize(x_.stream);return *x_.next_host;}
int Qwen35Decode::decode_token(int token){forward_token(token);return logits_argmax();}
void Qwen35Decode::mtp_layer() {
    {   // embed the pending token (device side) and rms-norm both inputs
        const std::string base = "language_model.model.embed_tokens";
        auto m = w_.matrix(base);
        embed_gather((const uint32_t *) m.weight.data, (const uint8_t *) m.scales.data, x_.token_dev, x_.down, 1, x_.stream);
        w_.release(base);
    }
    auto ew = tensor("language_model.mtp.pre_fc_norm_embedding.weight");
    auto hw = tensor("language_model.mtp.pre_fc_norm_hidden.weight");
    rmsnorm_bf16(x_.down, (const uint16_t *) ew.data, x_.up, 1, 4096, false, x_.stream);
    rmsnorm_bf16(x_.hidden, (const uint16_t *) hw.data, x_.norm, 1, 4096, false, x_.stream);
    w_.storage().release("language_model.mtp.pre_fc_norm_embedding.weight");
    w_.storage().release("language_model.mtp.pre_fc_norm_hidden.weight");
    concat(x_.up, x_.norm, x_.qkv, 4096, x_.stream);
    auto fc = tensor("language_model.mtp.fc.weight");
    bf16_gemv((const uint16_t *) fc.data, x_.qkv, x_.hidden, 4096, 8192, x_.stream);
    w_.storage().release("language_model.mtp.fc.weight");
    // the MTP layer attends at position-1 (slot 7); the main layers keep pos_dev[0]
    const std::string p = "language_model.mtp.layers.0";
    const std::string a = p + ".self_attn";
    auto inw = tensor(p + ".input_layernorm.weight");
    rmsnorm_bf16(x_.hidden, (const uint16_t *) inw.data, x_.norm, 1, 4096, false, x_.stream);
    w_.storage().release(p + ".input_layernorm.weight");
    linear(a + ".q_proj", x_.norm, x_.gate);
    split_q_gate(x_.gate, x_.qkv, x_.attn_gate, x_.stream);
    linear(a + ".k_proj", x_.norm, x_.key);
    linear(a + ".v_proj", x_.norm, x_.value);
    auto qw = tensor(a + ".q_norm.weight");
    auto kw = tensor(a + ".k_norm.weight");
    qwen35_qk_norm_rope_gate(x_.qkv, x_.key, (const uint16_t *) qw.data, (const uint16_t *) kw.data, x_.attn_gate, x_.mtp_pos_dev, 0, x_.stream);
    w_.storage().release(a + ".q_norm.weight");
    w_.storage().release(a + ".k_norm.weight");
    store_kv(x_.key, x_.value, x_.mtp_keys, x_.mtp_values, x_.mtp_pos_dev, 0, x_.stream);
    gqa_decode(x_.qkv, x_.mtp_keys, x_.mtp_values, x_.core, x_.mtp_pos_dev, 0, x_.max_context, x_.stream);
    expand_gate_heads(x_.attn_gate, x_.qkv, x_.stream);
    sigmoid_mul(x_.core, x_.qkv, 4096, x_.stream);
    linear(a + ".o_proj", x_.core, x_.down);
    residual_add(x_.hidden, x_.down, 4096, x_.stream);
    auto post = tensor(p + ".post_attention_layernorm.weight");
    rmsnorm_bf16(x_.hidden, (const uint16_t *) post.data, x_.norm, 1, 4096, false, x_.stream);
    w_.storage().release(p + ".post_attention_layernorm.weight");
    linear(p + ".mlp.gate_proj", x_.norm, x_.gate);
    linear(p + ".mlp.up_proj", x_.norm, x_.up);
    silu_mul(x_.gate, x_.up, x_.gate, 12288, x_.stream);
    linear(p + ".mlp.down_proj", x_.gate, x_.down);
    residual_add(x_.hidden, x_.down, 4096, x_.stream);
    auto nw = tensor("language_model.mtp.norm.weight");
    rmsnorm_bf16(x_.hidden, (const uint16_t *) nw.data, x_.norm, 1, 4096, false, x_.stream);
    w_.storage().release("language_model.mtp.norm.weight");
    linear("language_model.lm_head", x_.norm, x_.logits);
    argmax_fast(x_.logits, Qwen35Shape::vocab, x_.next_dev, x_.am_scratch, x_.stream);
}
void Qwen35Decode::prime_spec(int pending) {
    *x_.next_host = pending;
    cudaMemcpyAsync(x_.token_dev, x_.next_host, sizeof(int), cudaMemcpyHostToDevice, x_.stream);
}
void Qwen35Decode::append_committed_host(const int *ids, int n) {
    int c;
    cudaMemcpyAsync(&c, x_.count_dev, sizeof(int), cudaMemcpyDeviceToHost, x_.stream);
    cudaStreamSynchronize(x_.stream);
    for (int i = 0; i < n; i++) x_.host_committed[c + i] = ids[i];
    cudaMemcpyAsync(x_.committed + c, x_.host_committed + c, n * sizeof(int), cudaMemcpyHostToDevice, x_.stream);
    c += n;
    x_.host_committed[16383] = c;  // reuse the tail as the staging slot
    cudaMemcpyAsync(x_.count_dev, x_.host_committed + 16383, sizeof(int), cudaMemcpyHostToDevice, x_.stream);
}
int Qwen35Decode::committed_count() {
    int c;
    cudaMemcpyAsync(&c, x_.count_dev, sizeof(int), cudaMemcpyDeviceToHost, x_.stream);
    cudaStreamSynchronize(x_.stream);
    return c;
}
void Qwen35Decode::read_committed(int *host_dst, int n) {
    cudaMemcpyAsync(host_dst, x_.committed, n * sizeof(int), cudaMemcpyDeviceToHost, x_.stream);
    cudaStreamSynchronize(x_.stream);
}
int Qwen35Decode::spec_step(int t0) {
 prime_spec(t0);
 spec_prologue(x_.pos_dev, x_.stream);
 mtp_layer();
 spec_setup(x_.pos_dev, x_.pf_tokens, x_.stream);
 prefill_chunk_device(x_.pf_tokens,2);
 spec_commit(x_.pos_dev,x_.committed,x_.stream);
 spec_rollback(x_.snap_delta,x_.snap_conv,x_.delta_state,x_.conv_state,x_.pf_x,x_.hidden,x_.pos_dev,x_.stream);
 int tail[3];  // pos, pending, accept flag
 cudaMemcpyAsync(tail,x_.pos_dev,3*sizeof(int),cudaMemcpyDeviceToHost,x_.stream);
 cudaStreamSynchronize(x_.stream);
 x_.position=tail[0];
 spec_accepted=tail[2]!=0;
 int c;cudaMemcpyAsync(&c,x_.count_dev,sizeof(int),cudaMemcpyDeviceToHost,x_.stream);cudaStreamSynchronize(x_.stream);
 cudaMemcpyAsync(x_.next_host,x_.committed+(c-1),sizeof(int),cudaMemcpyDeviceToHost,x_.stream);cudaStreamSynchronize(x_.stream);
 spec_second=*x_.next_host;
 cudaMemcpyAsync(x_.next_host,x_.token_dev,sizeof(int),cudaMemcpyDeviceToHost,x_.stream);cudaStreamSynchronize(x_.stream);
 return *x_.next_host;
}
void Qwen35Decode::capture_spec(){
 cudaGraph_t graph;auto e=cudaStreamBeginCapture(x_.stream,cudaStreamCaptureModeThreadLocal);if(e)throw std::runtime_error(cudaGetErrorString(e));
 spec_prologue(x_.pos_dev,x_.stream);
 mtp_layer();
 spec_setup(x_.pos_dev,x_.pf_tokens,x_.stream);
 prefill_chunk_device(x_.pf_tokens,2);
 spec_commit(x_.pos_dev,x_.committed,x_.stream);
 spec_rollback(x_.snap_delta,x_.snap_conv,x_.delta_state,x_.conv_state,x_.pf_x,x_.hidden,x_.pos_dev,x_.stream);
 e=cudaStreamEndCapture(x_.stream,&graph);if(e)throw std::runtime_error(cudaGetErrorString(e));
 e=cudaGraphInstantiate(&spec_graph_,graph,0);cudaGraphDestroy(graph);
 if(e)throw std::runtime_error(cudaGetErrorString(e));
}
void Qwen35Decode::spec_graph_step(){
 auto e=cudaGraphLaunch(spec_graph_,x_.stream);
 if(e)throw std::runtime_error(cudaGetErrorString(e));
 x_.position+=2;  // host mirror is corrected to the device truth at each host read
}
void Qwen35Decode::capture_step(){
 cudaGraph_t graph;auto e=cudaStreamBeginCapture(x_.stream,cudaStreamCaptureModeThreadLocal);if(e)throw std::runtime_error(cudaGetErrorString(e));
 copyi_kernel<<<1,1,0,x_.stream>>>(x_.token_dev,x_.next_dev);
 w_.embed_dev(x_.token_dev,x_.hidden,x_.stream);
 forward_body();
 argmax_fast(x_.logits,Qwen35Shape::vocab,x_.next_dev,x_.am_scratch,x_.stream);
 bumpi_kernel<<<1,1,0,x_.stream>>>(x_.pos_dev);
 e=cudaStreamEndCapture(x_.stream,&graph);if(e)throw std::runtime_error(cudaGetErrorString(e));
 e=cudaGraphInstantiate(&graph_,graph,0);cudaGraphDestroy(graph);
 if(e)throw std::runtime_error(cudaGetErrorString(e));captured_=true;
}
void Qwen35Decode::step(int token){if(!captured_)throw std::runtime_error("step before capture_step");cudaMemcpyAsync(x_.next_dev,&token,sizeof(int),cudaMemcpyHostToDevice,x_.stream);cudaGraphLaunch(graph_,x_.stream);cudaMemcpyAsync(x_.next_host,x_.next_dev,sizeof(int),cudaMemcpyDeviceToHost,x_.stream);cudaStreamSynchronize(x_.stream);x_.position++;}

}
