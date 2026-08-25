#pragma once
#include "insignia_qwen35.hpp"
#include <cuda_runtime.h>
namespace insignia {
class DecodeWorkspace final {
public:
 explicit DecodeWorkspace(int max_context=4096,cudaStream_t stream=nullptr);~DecodeWorkspace();DecodeWorkspace(const DecodeWorkspace&)=delete;
 float *hidden{},*norm{},*qkv{},*attn_gate{},*key{},*value{},*z{},*a{},*b{},*core{},*gate{},*up{},*down{},*logits{},*delta_state{},*conv_state{},*kv_keys{},*kv_values{},*mtp_keys{},*mtp_values{};
 int *pos_dev{},*token_dev{},*next_dev{},*next2_dev{},*draft_dev{},*count_dev{},*accflag_dev{},*mtp_pos_dev{},*committed{},*pf_tokens{};int *next_host{},*pos_host{},*host_committed{};  // device step state + pinned mirror
 unsigned long long *am_scratch{};
 float *snap_delta{},*snap_conv{};  // speculative rollback snapshots
 float *pf_x{},*pf_n{},*pf_qkv{},*pf_scratch{},*pf_z{},*pf_q{},*pf_g{},*pf_k{},*pf_v{},*pf_core{},*pf_down{},*pf_gate{},*pf_up{},*pf_a{},*pf_b{};  // prefill chunk buffers
 unsigned *pf_xq8{};float *pf_xs8{};void *pf_bf16{};  // int8 pair staging + bf16 GEMM scratch (64 rows)
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
 int prefill_chunk(const int *tokens, int T);  // batched forward over T<=64 tokens, returns argmax after last
 int spec_step(int next_token);  // one speculative iteration: pair forward + MTP draft; commits 2 tokens, returns the next undecided token
 int spec_second{};bool spec_accepted{};  // second committed token of the last spec_step and whether the draft hit
 int logits_argmax();                    // argmax of x.logits into next_dev, returns it
 void set_position(int pos);             // async device write of the shared position
 void set_mtp_position(int pos);         // async device write of the MTP layer position slot
 int position() const noexcept{return x_.position;}
 void capture_step();                    // capture the self-feeding decode graph (needs one warmup token first)
 void step(int token);                   // prime next_dev with token, replay one graph step (position must already be set)
 int next_token() const noexcept{return *x_.next_host;}
 void mtp_layer();                       // device-state MTP draft: token_dev+hidden -> argmax into next_dev
 void capture_spec();                    // capture the full speculative step (MTP draft + pair verify + commit)
 void spec_graph_step();                 // replay one captured speculative step (no host sync)
 int committed_count();                  // device readback of the committed token count
 void read_committed(int *host_dst,int n);  // D2H of the committed id stream
 void prime_spec(int pending);           // host-side: token_dev <- pending (before eager/graph spec steps)
 void append_committed_host(const int *ids,int n);  // host-side commit of decided ids (pinned H2D)
private:
 void linear(const std::string&base,const float*input,float*output);
 void linear_batch(const std::string&base,const float*input,float*output,int T);
 void linear2(const std::string&base,const float*input,float*output);  // pair path: staged dp4a
 DeviceView tensor(const std::string&name);
 void forward_body();                    // embed..lm_head over device token/pos
 void prefill_chunk_device(const int*tokens_dev,int T);  // batched forward, argmax into next_dev (no sync)
 Qwen35Weights&w_;DecodeWorkspace&x_;cudaGraphExec_t graph_{},spec_graph_{};bool captured_{};
};
}
