#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace insignia::glm53 {

enum class Cache8Format : uint8_t {
    q8 = 1,
    fp8_e4m3 = 2,
};

struct Q8TensorLocation {
    uint32_t rows;
    uint32_t cols;
    uint64_t weight_offset;
    uint64_t weight_bytes;
    uint64_t scale_offset;
    uint64_t scale_bytes;
};

class Q8Index {
public:
    explicit Q8Index(const std::filesystem::path &prefix);
    ~Q8Index();
    Q8Index(const Q8Index &) = delete;
    Q8Index &operator=(const Q8Index &) = delete;

    const Q8TensorLocation *find(std::string_view name) const;
    void read_rows(const Q8TensorLocation &tensor, uint32_t row, uint32_t rows,
                   void *weights, void *scales) const;
    // O_DIRECT variant for whole-tensor pinning. Both destinations must be
    // 4096-aligned and have 4096 bytes of scratch before and after the range
    // so unaligned head/tail of the on-disk span can land in the same buffer.
    void read_rows_direct(const Q8TensorLocation &tensor, uint8_t *weights,
                          uint8_t *scales) const;
    // Visits tensors in on-disk weight order so bulk pinning reads mostly
    // sequentially.
    void for_each_by_offset(
        const std::function<void(const std::string &, const Q8TensorLocation &)> &visit) const;
    uint64_t data_bytes() const { return data_bytes_; }
    size_t tensor_count() const { return tensors_.size(); }
    Cache8Format format() const { return format_; }

private:
    void pread_direct(uint64_t offset, uint64_t bytes, uint8_t *destination) const;

    int data_fd_ = -1;
    int direct_fd_ = -1;
    uint64_t data_bytes_ = 0;
    Cache8Format format_ = Cache8Format::q8;
    std::unordered_map<std::string, Q8TensorLocation> tensors_;
};

}  // namespace insignia::glm53
