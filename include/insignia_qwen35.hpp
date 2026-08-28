#pragma once
#include "insignia_model.hpp"
#include "insignia_storage.hpp"
#include <cuda_runtime.h>
#include <memory>
namespace insignia {
struct Qwen35Shape { static constexpr int hidden=4096,intermediate=12288,layers=32,vocab=248320; static constexpr bool full_attention(int i){return (i&3)==3;} };
enum class WKind : uint8_t { mxfp4_mlx, mxfp4_i4, fp8, bf16 };
struct QuantMatrix { DeviceView weight,scales; int rows{},cols{}; bool insig4{}; WKind kind{}; bool has_scales{}; };
class Qwen35Weights final {
public:
 Qwen35Weights(const ModelFile &model,uint64_t device_budget_bytes,cudaStream_t stream=nullptr);
 ~Qwen35Weights();
 void embed(int token,float *device_hidden);
 void embed_dev(const int *token_dev,float *device_hidden,cudaStream_t stream=nullptr);
 QuantMatrix matrix(const std::string &base);
 void release(const std::string &base) noexcept;
 TieredStorage &storage() noexcept{return storage_;}
private:
 TieredStorage storage_; cudaStream_t stream_{}; int *scratch_int_{};
};
}
