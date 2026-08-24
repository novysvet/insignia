#pragma once
#include "insignia_qwen35.hpp"
#include <cuda_runtime.h>
namespace insignia {
class DecodeWorkspace final {
public:
 explicit DecodeWorkspace(int max_context=4096,cudaStream_t stream=nullptr);~DecodeWorkspace();DecodeWorkspace(const DecodeWorkspace&)=delete;
 float *hidden{},*norm{},*qkv{},*attn_gate{},*key{},*value{},*z{},*a{},*b{},*core{},*gate{},*up{},*down{},*logits{},*delta_state{},*conv_state{},*kv_keys{},*kv_values{},*mtp_keys{},*mtp_values{};
 int *pos_dev{},*token_dev{},*next_dev{};int *next_host{};  // device step state + pinned mirror
 int max_context{},position{};cudaStream_t stream{};
};
class Qwen35Decode final {
public:
 Qwen35Decode(Qwen35Weights &weights,DecodeWorkspace &workspace):w_(weights),x_(workspace){}
 void delta_layer(int layer);
 void attention_layer(int layer);
 void layer(int layer);
 void forward_token(int token);          // processes token at position x.position, then bumps
 int decode_token(int token);            // forward_token + argmax
 int logits_argmax();                    // argmax of x.logits into next_dev, returns it
 void set_position(int pos);             // async device write of the shared position
 int position() const noexcept{return x_.position;}
 void capture_step();                    // capture the self-feeding decode graph (needs one warmup token first)
 void step(int token);                   // prime next_dev with token, replay one graph step (position must already be set)
 int next_token() const noexcept{return *x_.next_host;}
 int mtp_draft(int input_token);
private:
 void linear(const std::string&base,const float*input,float*output);
 DeviceView tensor(const std::string&name);
 void forward_body();                    // embed..lm_head over device token/pos
 Qwen35Weights&w_;DecodeWorkspace&x_;cudaGraphExec_t graph_{};bool captured_{};
};
}
