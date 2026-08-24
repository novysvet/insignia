#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace insignia {

enum class DType : uint8_t { f32=1, bf16=2, f16=3, u8=4, u32=5, i8=6 };

struct TensorView {
    std::string name;
    DType dtype{};
    std::vector<uint64_t> shape;
    const std::byte *data{};
    uint64_t bytes{};
};

class ModelFile final {
public:
    ModelFile() = default;
    explicit ModelFile(const wchar_t *index_path);
    ~ModelFile();
    ModelFile(const ModelFile &) = delete;
    ModelFile &operator=(const ModelFile &) = delete;
    ModelFile(ModelFile &&other) noexcept;
    ModelFile &operator=(ModelFile &&other) noexcept;
    const TensorView *find(std::string_view name) const noexcept;
    const std::vector<TensorView> &tensors() const noexcept { return tensors_; }
    uint64_t payload_offset() const noexcept { return payload_offset_; }
private:
    void close() noexcept;
    void *file_{};
    void *mapping_{};
    const std::byte *base_{};
    uint64_t mapped_bytes_{};
    uint64_t payload_offset_{};
    std::vector<TensorView> tensors_;
};

}
