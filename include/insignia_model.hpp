#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace insignia {

enum class DType : uint8_t { f32=1, bf16=2, f16=3, u8=4, u32=5, i8=6, f8_e4m3=7 };

struct TensorView {
    std::string name;
    DType dtype{};
    std::vector<uint64_t> shape;
    const std::byte *data{};   // INSIDX01: mapped payload pointer; INSIDX02: nullptr (use shard/off)
    uint64_t bytes{};
    uint16_t shard{};          // INSIDX02: shard table index
    uint64_t off{};            // INSIDX02: file-absolute byte offset inside that shard
};

struct ShardInfo {
    std::wstring path;         // absolute
    uint64_t bytes{};
    uint32_t crc{};
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
    // INSIDX02 multi-shard API. v1 indexes report v2()==false and empty shards().
    bool v2() const noexcept { return version_ == 2; }
    const std::vector<ShardInfo> &shards() const noexcept { return shards_; }
    // 9-field shape header order (tools/index27.py SHAPE_ORDER): hidden, layers, vocab,
    // q_heads, kv_heads, delta_v, delta_k, inter, full_attention_interval.
    const uint32_t *shape_hdr() const noexcept { return shape_hdr_; }
    // One-shot synchronous read of a tensor into dst (buffered; startup/small loads only —
    // the streaming tier uses NvmeReader direct handles). v1: memcpy from the mapping.
    void read_into(const TensorView &t, void *dst) const;
    // Chunked variant: reads [off, off+len) of the tensor (large tensors through a bounce).
    void read_into_of(const TensorView &t, uint64_t off, uint64_t len, void *dst) const;
private:
    void close() noexcept;
    void *shard_handle(uint16_t shard) const;   // buffered, cached per shard (v2)
    void *file_{};
    void *mapping_{};
    const std::byte *base_{};
    uint64_t mapped_bytes_{};
    uint64_t payload_offset_{};
    uint32_t version_{};
    uint32_t shape_hdr_[9]{};
    std::vector<ShardInfo> shards_;
    mutable std::vector<void *> shard_handles_;  // lazy buffered HANDLE per shard (v2)
    std::wstring index_dir_;
    std::vector<TensorView> tensors_;
};

}
