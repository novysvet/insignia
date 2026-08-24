#pragma once
#include "insignia_model.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>

namespace insignia {

enum class MemoryTier : uint8_t { nvme_mapped, host_pinned, device };

struct DeviceView { void *data{}; uint64_t bytes{}; DType dtype{}; const std::vector<uint64_t> *shape{}; explicit operator bool() const noexcept{return data!=nullptr;} };

class TieredStorage final {
public:
    TieredStorage(const ModelFile &model,uint64_t device_budget_bytes,cudaStream_t stream=nullptr);
    ~TieredStorage();
    TieredStorage(const TieredStorage&)=delete;
    TieredStorage& operator=(const TieredStorage&)=delete;
    DeviceView acquire(std::string_view name);
    void release(std::string_view name) noexcept;
    void clear();
    uint64_t device_bytes() const noexcept{return used_;}
    uint64_t budget_bytes() const noexcept{return budget_;}
private:
    struct Entry { void *device{}; uint64_t bytes{}; uint32_t pins{}; uint64_t tick{}; const TensorView *host{}; };
    void make_room(uint64_t bytes);
    const ModelFile &model_; uint64_t budget_{},used_{},tick_{}; cudaStream_t stream_{};
    std::unordered_map<std::string,Entry> entries_;
};

}
