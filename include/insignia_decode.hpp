#pragma once
#include "insignia_qwen35.hpp"
#include <cuda_runtime.h>
namespace insignia {
class DecodeWorkspace final {
public:
 explicit DecodeWorkspace(int max_context=4096,cudaStream_t stream=nullptr);~DecodeWorkspace();DecodeWorkspace(const DecodeWorkspace&)=delete;
 float *hidden{},*norm{},*qkv{},*attn_gate{},*key{},*value{},*z{},*a{},*b{},*core{},*gate{},*up{},*down{},*logits{},*delta_state{},*conv_state{},*kv_keys{},*kv_values{},*mtp_keys{},*mtp_values{};int max_context{},position{};cudaStream_t stream{};
};
class Qwen35Decode final {
public:
 Qwen35Decode(Qwen35Weights &weights,DecodeWorkspace &workspace):w_(weights),x_(workspace){}
 void delta_layer(int layer);
 void attention_layer(int layer);
 void layer(int layer);
 int decode_token(int token);
 void forward_token(int token);
 int logits_argmax();
 int mtp_draft(int input_token);
private:
 void linear(const std::string&base,const float*input,float*output);
 DeviceView tensor(const std::string&name);
 Qwen35Weights&w_;DecodeWorkspace&x_;
};
}
