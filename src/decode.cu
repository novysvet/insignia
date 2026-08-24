#include "insignia_decode.hpp"
#include "insignia_layout.cuh"
#include "insignia_deltanet.cuh"
#include "insignia_qwen_kernels.cuh"
#include "insignia_ops.cuh"
#include "insignia_attention.cuh"
#include <stdexcept>
namespace insignia {
static void alloc(float**p,size_t n){auto e=cudaMalloc(p,n*sizeof(float));if(e)throw std::runtime_error(cudaGetErrorString(e));}
DecodeWorkspace::DecodeWorkspace(int ctx,cudaStream_t s):max_context(ctx),stream(s){
 if(ctx<=0||ctx>4096)throw std::runtime_error("max_context must be in 1..4096 (gqa score buffer)");
 if(!stream){auto e=cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);if(e)throw std::runtime_error(cudaGetErrorString(e));}
 alloc(&hidden,4096);alloc(&norm,4096);alloc(&qkv,8192);alloc(&attn_gate,4096);alloc(&key,1024);alloc(&value,1024);alloc(&z,4096);alloc(&a,32);alloc(&b,32);alloc(&core,4096);alloc(&gate,12288);alloc(&up,12288);alloc(&down,4096);alloc(&logits,248320);alloc(&delta_state,24*32*128*128);alloc(&conv_state,24*8192*3);alloc(&kv_keys,size_t(8)*ctx*1024);alloc(&kv_values,size_t(8)*ctx*1024);alloc(&mtp_keys,size_t(ctx)*1024);alloc(&mtp_values,size_t(ctx)*1024);
 auto e=cudaMalloc(&pos_dev,3*sizeof(int));if(e)throw std::runtime_error(cudaGetErrorString(e));token_dev=pos_dev+1;next_dev=pos_dev+2;
 if(cudaHostAlloc(&next_host,sizeof(int),cudaHostAllocDefault))throw std::runtime_error("pinned alloc failed");
 cudaMemsetAsync(pos_dev,0,3*sizeof(int),stream);cudaMemsetAsync(delta_state,0,24*32*128*128*4,stream);cudaMemsetAsync(conv_state,0,24*8192*3*4,stream);
}
DecodeWorkspace::~DecodeWorkspace(){cudaFree(hidden);cudaFree(norm);cudaFree(qkv);cudaFree(attn_gate);cudaFree(key);cudaFree(value);cudaFree(z);cudaFree(a);cudaFree(b);cudaFree(core);cudaFree(gate);cudaFree(up);cudaFree(down);cudaFree(logits);cudaFree(delta_state);cudaFree(conv_state);cudaFree(kv_keys);cudaFree(kv_values);cudaFree(mtp_keys);cudaFree(mtp_values);cudaFree(pos_dev);cudaFreeHost(next_host);}
DeviceView Qwen35Decode::tensor(const std::string&name){return w_.storage().acquire(name);}
void Qwen35Decode::linear(const std::string&base,const float*in,float*out){auto m=w_.matrix(base);mxfp4_gemv_v2((const uint32_t*)m.weight.data,(const uint8_t*)m.scales.data,in,out,m.rows,m.cols,x_.stream);w_.release(base);}
__global__ void copyi_kernel(int*dst,const int*src){*dst=*src;}
__global__ void bumpi_kernel(int*p){(*p)++;}
void Qwen35Decode::set_position(int pos){cudaMemcpyAsync(x_.pos_dev,&pos,sizeof(int),cudaMemcpyHostToDevice,x_.stream);}
void Qwen35Decode::delta_layer(int l){if(l<0||l>=32||Qwen35Shape::full_attention(l))throw std::runtime_error("not a DeltaNet layer");const std::string p="language_model.model.layers."+std::to_string(l);auto inw=tensor(p+".input_layernorm.weight");rmsnorm_bf16(x_.hidden,(const uint16_t*)inw.data,x_.norm,1,4096,false,x_.stream);w_.storage().release(p+".input_layernorm.weight");
 const std::string a=p+".linear_attn";linear(a+".in_proj_qkv",x_.norm,x_.qkv);linear(a+".in_proj_z",x_.norm,x_.z);linear(a+".in_proj_a",x_.norm,x_.a);linear(a+".in_proj_b",x_.norm,x_.b);
 auto cw=tensor(a+".conv1d.weight");const int di=l-l/4;causal_conv4_silu(x_.qkv,x_.conv_state+size_t(di)*8192*3,(const uint16_t*)cw.data,8192,x_.stream);w_.storage().release(a+".conv1d.weight");auto A=tensor(a+".A_log"),dt=tensor(a+".dt_bias");deltanet_parameters(x_.a,x_.b,(const float*)A.data,(const uint16_t*)dt.data,32,x_.stream);w_.storage().release(a+".A_log");w_.storage().release(a+".dt_bias");deltanet_decode(x_.delta_state+size_t(di)*32*128*128,x_.qkv,x_.qkv+2048,x_.qkv+4096,x_.a,x_.b,x_.core,x_.stream);auto nw=tensor(a+".norm.weight");gated_rmsnorm_bf16(x_.core,(const uint16_t*)nw.data,x_.z,x_.core,32,128,x_.stream);w_.storage().release(a+".norm.weight");linear(a+".out_proj",x_.core,x_.down);residual_add(x_.hidden,x_.down,4096,x_.stream);
 auto post=tensor(p+".post_attention_layernorm.weight");rmsnorm_bf16(x_.hidden,(const uint16_t*)post.data,x_.norm,1,4096,false,x_.stream);w_.storage().release(p+".post_attention_layernorm.weight");linear(p+".mlp.gate_proj",x_.norm,x_.gate);linear(p+".mlp.up_proj",x_.norm,x_.up);silu_mul(x_.gate,x_.up,x_.gate,12288,x_.stream);linear(p+".mlp.down_proj",x_.gate,x_.down);residual_add(x_.hidden,x_.down,4096,x_.stream);
}
void Qwen35Decode::attention_layer(int l){if(l<0||l>=32||!Qwen35Shape::full_attention(l))throw std::runtime_error("not a full attention layer");if(x_.position>=x_.max_context)throw std::runtime_error("KV cache full");const std::string p="language_model.model.layers."+std::to_string(l),a=p+".self_attn";auto inw=tensor(p+".input_layernorm.weight");rmsnorm_bf16(x_.hidden,(const uint16_t*)inw.data,x_.norm,1,4096,false,x_.stream);w_.storage().release(p+".input_layernorm.weight");linear(a+".q_proj",x_.norm,x_.gate);split_q_gate(x_.gate,x_.qkv,x_.attn_gate,x_.stream);linear(a+".k_proj",x_.norm,x_.key);linear(a+".v_proj",x_.norm,x_.value);auto qw=tensor(a+".q_norm.weight"),kw=tensor(a+".k_norm.weight");qwen35_qk_norm_rope_gate(x_.qkv,x_.key,(const uint16_t*)qw.data,(const uint16_t*)kw.data,x_.attn_gate,x_.pos_dev,0,x_.stream);w_.storage().release(a+".q_norm.weight");w_.storage().release(a+".k_norm.weight");const int ai=l/4;float*kc=x_.kv_keys+size_t(ai)*x_.max_context*1024,*vc=x_.kv_values+size_t(ai)*x_.max_context*1024;store_kv(x_.key,x_.value,kc,vc,x_.pos_dev,0,x_.stream);gqa_decode(x_.qkv,kc,vc,x_.core,x_.pos_dev,0,x_.max_context,x_.stream);expand_gate_heads(x_.attn_gate,x_.qkv,x_.stream);sigmoid_mul(x_.core,x_.qkv,4096,x_.stream);linear(a+".o_proj",x_.core,x_.down);residual_add(x_.hidden,x_.down,4096,x_.stream);auto post=tensor(p+".post_attention_layernorm.weight");rmsnorm_bf16(x_.hidden,(const uint16_t*)post.data,x_.norm,1,4096,false,x_.stream);w_.storage().release(p+".post_attention_layernorm.weight");linear(p+".mlp.gate_proj",x_.norm,x_.gate);linear(p+".mlp.up_proj",x_.norm,x_.up);silu_mul(x_.gate,x_.up,x_.gate,12288,x_.stream);linear(p+".mlp.down_proj",x_.gate,x_.down);residual_add(x_.hidden,x_.down,4096,x_.stream);}
void Qwen35Decode::layer(int l){if(Qwen35Shape::full_attention(l))attention_layer(l);else delta_layer(l);}
void Qwen35Decode::forward_body(){for(int l=0;l<32;l++)layer(l);auto nw=tensor("language_model.model.norm.weight");rmsnorm_bf16(x_.hidden,(const uint16_t*)nw.data,x_.norm,1,4096,false,x_.stream);w_.storage().release("language_model.model.norm.weight");linear("language_model.lm_head",x_.norm,x_.logits);}
void Qwen35Decode::forward_token(int token){if(token<0||token>=Qwen35Shape::vocab)throw std::runtime_error("token out of range");if(x_.position>=x_.max_context)throw std::runtime_error("KV cache full");cudaMemcpyAsync(x_.token_dev,&token,sizeof(int),cudaMemcpyHostToDevice,x_.stream);w_.embed_dev(x_.token_dev,x_.hidden,x_.stream);forward_body();bumpi_kernel<<<1,1,0,x_.stream>>>(x_.pos_dev);x_.position++;}
int Qwen35Decode::logits_argmax(){argmax_logits(x_.logits,Qwen35Shape::vocab,x_.next_dev,x_.stream);cudaMemcpyAsync(x_.next_host,x_.next_dev,sizeof(int),cudaMemcpyDeviceToHost,x_.stream);cudaStreamSynchronize(x_.stream);return *x_.next_host;}
int Qwen35Decode::decode_token(int token){forward_token(token);return logits_argmax();}
void Qwen35Decode::capture_step(){
 cudaGraph_t graph;auto e=cudaStreamBeginCapture(x_.stream,cudaStreamCaptureModeThreadLocal);if(e)throw std::runtime_error(cudaGetErrorString(e));
 copyi_kernel<<<1,1,0,x_.stream>>>(x_.token_dev,x_.next_dev);
 w_.embed_dev(x_.token_dev,x_.hidden,x_.stream);
 forward_body();
 argmax_logits(x_.logits,Qwen35Shape::vocab,x_.next_dev,x_.stream);
 bumpi_kernel<<<1,1,0,x_.stream>>>(x_.pos_dev);
 e=cudaStreamEndCapture(x_.stream,&graph);if(e)throw std::runtime_error(cudaGetErrorString(e));
 e=cudaGraphInstantiate(&graph_,graph,0);cudaGraphDestroy(graph);
 if(e)throw std::runtime_error(cudaGetErrorString(e));captured_=true;
}
void Qwen35Decode::step(int token){if(!captured_)throw std::runtime_error("step before capture_step");cudaMemcpyAsync(x_.next_dev,&token,sizeof(int),cudaMemcpyHostToDevice,x_.stream);cudaGraphLaunch(graph_,x_.stream);cudaMemcpyAsync(x_.next_host,x_.next_dev,sizeof(int),cudaMemcpyDeviceToHost,x_.stream);cudaStreamSynchronize(x_.stream);x_.position++;}
int Qwen35Decode::mtp_draft(int token) {
    w_.embed(token, x_.down);
    auto ew = tensor("language_model.mtp.pre_fc_norm_embedding.weight");
    auto hw = tensor("language_model.mtp.pre_fc_norm_hidden.weight");
    rmsnorm_bf16(x_.down, (const uint16_t*) ew.data, x_.up, 1, 4096, false, x_.stream);
    rmsnorm_bf16(x_.hidden, (const uint16_t*) hw.data, x_.norm, 1, 4096, false, x_.stream);
    w_.storage().release("language_model.mtp.pre_fc_norm_embedding.weight");
    w_.storage().release("language_model.mtp.pre_fc_norm_hidden.weight");
    concat(x_.up, x_.norm, x_.qkv, 4096, x_.stream);
    auto fc = tensor("language_model.mtp.fc.weight");
    bf16_gemv((const uint16_t*) fc.data, x_.qkv, x_.hidden, 4096, 8192, x_.stream);
    w_.storage().release("language_model.mtp.fc.weight");
    const int mtp_pos = x_.position - 1;
    set_position(mtp_pos);  // borrow pos_dev for the MTP layer's own position (restored by next set_position)
    const std::string p = "language_model.mtp.layers.0";
    const std::string a = p + ".self_attn";
    auto inw = tensor(p + ".input_layernorm.weight");
    rmsnorm_bf16(x_.hidden, (const uint16_t*) inw.data, x_.norm, 1, 4096, false, x_.stream);
    w_.storage().release(p + ".input_layernorm.weight");
    linear(a + ".q_proj", x_.norm, x_.gate);
    split_q_gate(x_.gate, x_.qkv, x_.attn_gate, x_.stream);
    linear(a + ".k_proj", x_.norm, x_.key);
    linear(a + ".v_proj", x_.norm, x_.value);
    auto qw = tensor(a + ".q_norm.weight");
    auto kw = tensor(a + ".k_norm.weight");
    qwen35_qk_norm_rope_gate(x_.qkv, x_.key, (const uint16_t*) qw.data, (const uint16_t*) kw.data, x_.attn_gate, x_.pos_dev, 0, x_.stream);
    w_.storage().release(a + ".q_norm.weight");
    w_.storage().release(a + ".k_norm.weight");
    store_kv(x_.key, x_.value, x_.mtp_keys, x_.mtp_values, x_.pos_dev, 0, x_.stream);
    gqa_decode(x_.qkv, x_.mtp_keys, x_.mtp_values, x_.core, x_.pos_dev, 0, x_.max_context, x_.stream);
    expand_gate_heads(x_.attn_gate, x_.qkv, x_.stream);
    sigmoid_mul(x_.core, x_.qkv, 4096, x_.stream);
    linear(a + ".o_proj", x_.core, x_.down);
    residual_add(x_.hidden, x_.down, 4096, x_.stream);
    auto post = tensor(p + ".post_attention_layernorm.weight");
    rmsnorm_bf16(x_.hidden, (const uint16_t*) post.data, x_.norm, 1, 4096, false, x_.stream);
    w_.storage().release(p + ".post_attention_layernorm.weight");
    linear(p + ".mlp.gate_proj", x_.norm, x_.gate);
    linear(p + ".mlp.up_proj", x_.norm, x_.up);
    silu_mul(x_.gate, x_.up, x_.gate, 12288, x_.stream);
    linear(p + ".mlp.down_proj", x_.gate, x_.down);
    residual_add(x_.hidden, x_.down, 4096, x_.stream);
    auto nw = tensor("language_model.mtp.norm.weight");
    rmsnorm_bf16(x_.hidden, (const uint16_t*) nw.data, x_.norm, 1, 4096, false, x_.stream);
    w_.storage().release("language_model.mtp.norm.weight");
    linear("language_model.lm_head", x_.norm, x_.logits);
    set_position(x_.position);  // restore pos_dev for main-layer semantics
    return logits_argmax();
}
}
