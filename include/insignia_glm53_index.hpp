#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace insignia::glm53 {

enum class TensorType : uint8_t {
    f32 = 1,
    bf16 = 2,
    f16 = 3,
    u8 = 4,
    u32 = 5,
    i8 = 6,
    f8_e4m3 = 7,
};

struct TensorLocation {
    TensorType type;
    uint16_t shard;
    uint64_t offset;
    uint64_t bytes;
    std::vector<uint32_t> shape;
};

class ShardedIndex {
public:
    explicit ShardedIndex(const std::filesystem::path &index_path,
                          const std::filesystem::path &model_root);
    ~ShardedIndex();
    ShardedIndex(const ShardedIndex &) = delete;
    ShardedIndex &operator=(const ShardedIndex &) = delete;

    const TensorLocation &tensor(std::string_view name) const;
    bool has(std::string_view name) const { return tensors_.count(std::string(name)) != 0; }
    void read(const TensorLocation &tensor, void *destination) const;
    void read_span(uint16_t shard, uint64_t offset, uint64_t bytes, void *destination) const;
    void read_span_direct(uint16_t shard, uint64_t offset, uint64_t bytes, void *destination) const;
    uint64_t read_span_direct_window(uint16_t shard, uint64_t offset, uint64_t bytes,
                                     void *aligned_destination, uint64_t capacity) const;
    // Buffered-twin read of the same span (page cache retained on purpose):
    // lets the kernel page cache act as an L2 tier behind the pinned LRU.
    uint64_t read_span_cached_window(uint16_t shard, uint64_t offset, uint64_t bytes,
                                     void *destination, uint64_t capacity) const;
    // Drop the page-cache copy of a span (used after a record is admitted to
    // the pinned tier so its pages stop shadowing the pinned window).
    void evict_span_cache(uint16_t shard, uint64_t offset, uint64_t bytes) const;
    bool direct_io_supported(uint16_t shard) const;
    // True when the shard file resolved to INSIGNIA_GLM53_ALT_SHARD_DIR
    // (second physical drive): lets the reader pool thread per-drive queues.
    bool shard_is_alt(uint16_t shard) const {
        return shard < alt_shard_.size() && alt_shard_[shard];
    }

    uint32_t hidden_size() const { return hidden_size_; }
    uint32_t layers() const { return layers_; }
    uint32_t vocab_size() const { return vocab_size_; }
    uint32_t experts() const { return experts_; }
    uint32_t active_experts() const { return active_experts_; }
    uint32_t moe_intermediate() const { return moe_intermediate_; }
    uint32_t hc_mult() const { return hc_mult_; }
    uint64_t payload_bytes() const { return payload_bytes_; }
    size_t tensor_count() const { return tensors_.size(); }
    size_t shard_count() const { return shard_names_.size(); }
    const std::string &shard_name(uint16_t shard) const { return shard_names_.at(shard); }

private:
    std::vector<std::string> shard_names_;
    std::vector<uint64_t> shard_sizes_;
    std::vector<int> shard_fds_;
    std::vector<int> direct_fds_;
    std::vector<uint8_t> alt_shard_;
    std::unordered_map<std::string, TensorLocation> tensors_;
    uint32_t hidden_size_ = 0;
    uint32_t layers_ = 0;
    uint32_t vocab_size_ = 0;
    uint32_t experts_ = 0;
    uint32_t active_experts_ = 0;
    uint32_t moe_intermediate_ = 0;
    uint32_t hc_mult_ = 0;
    uint64_t payload_bytes_ = 0;
};

}  // namespace insignia::glm53
