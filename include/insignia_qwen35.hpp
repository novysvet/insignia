#pragma once
#include "insignia_model.hpp"
#include "insignia_storage.hpp"
#include <cuda_runtime.h>
#include <memory>
namespace insignia {
struct Qwen35Shape { static constexpr int hidden=4096,intermediate=12288,layers=32,vocab=248320; static constexpr bool full_attention(int i){return (i&3)==3;} };
struct QuantMatrix { DeviceView weight,scales; int rows{},cols{}; };
class Qwen35Weights final {
public:
 Qwen35Weights(const ModelFile &model,uint64_t device_budget_bytes,cudaStream_t stream=nullptr);
 ~Qwen35Weights();
 void embed(int token,float *device_hidden);
 QuantMatrix matrix(const std::string &base);
 void release(const std::string &base) noexcept;
 TieredStorage &storage() noexcept{return storage_;}
private:
 TieredStorage storage_; cudaStream_t stream_{};
};
}
