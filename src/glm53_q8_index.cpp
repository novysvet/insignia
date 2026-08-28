#include "insignia_glm53_q8_index.hpp"
#include "insignia_glm53_q8.cuh"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace insignia::glm53 {
namespace {

struct Reader {
    const uint8_t *cursor;
    const uint8_t *end;

    void require(size_t bytes) const {
        if (bytes > size_t(end - cursor)) throw std::runtime_error("truncated GLM-5.3 Q8 index");
    }

    template <typename T>
    T scalar() {
        require(sizeof(T));
        T value;
        std::memcpy(&value, cursor, sizeof(value));
        cursor += sizeof(value);
        return value;
    }

    std::string string(size_t bytes) {
        require(bytes);
        std::string value(reinterpret_cast<const char *>(cursor), bytes);
        cursor += bytes;
        return value;
    }
};

std::vector<uint8_t> load_file(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::system_error(errno, std::generic_category(), "open Q8 index");
    const std::streamsize size = input.tellg();
    if (size < 0) throw std::runtime_error("cannot size GLM-5.3 Q8 index");
    input.seekg(0);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!input.read(reinterpret_cast<char *>(bytes.data()), size))
        throw std::runtime_error("short Q8 index read");
    return bytes;
}

void pread_all(int fd, uint64_t offset, uint64_t bytes, void *destination, const char *what) {
    auto *output = static_cast<uint8_t *>(destination);
    uint64_t done = 0;
    while (done < bytes) {
        const ssize_t count = ::pread(fd, output + done, size_t(bytes - done), off_t(offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0)
            throw std::system_error(count < 0 ? errno : EIO, std::generic_category(), what);
        done += uint64_t(count);
    }
}

}  // namespace

Q8Index::Q8Index(const std::filesystem::path &prefix) {
    std::vector<uint8_t> bytes = load_file(prefix.string() + ".index");
    Reader reader{bytes.data(), bytes.data() + bytes.size()};
    const std::string magic = reader.string(8);
    if (magic == "IGLMQ8A1")
        format_ = Cache8Format::q8;
    else if (magic == "IGLMF8A1")
        format_ = Cache8Format::fp8_e4m3;
    else
        throw std::runtime_error("bad GLM-5.3 8-bit index magic");
    if (reader.scalar<uint32_t>() != 1) throw std::runtime_error("unsupported GLM-5.3 Q8 index version");
    if (reader.scalar<uint32_t>() != kQ8GroupSize) throw std::runtime_error("wrong GLM-5.3 Q8 group size");
    const uint32_t count = reader.scalar<uint32_t>();
    data_bytes_ = reader.scalar<uint64_t>();
    tensors_.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        const uint16_t length = reader.scalar<uint16_t>();
        const uint32_t rows = reader.scalar<uint32_t>();
        const uint32_t cols = reader.scalar<uint32_t>();
        const uint64_t weight_offset = reader.scalar<uint64_t>();
        const uint64_t weight_bytes = reader.scalar<uint64_t>();
        const uint64_t scale_offset = reader.scalar<uint64_t>();
        const uint64_t scale_bytes = reader.scalar<uint64_t>();
        std::string name = reader.string(length);
        const uint64_t expected_weights = uint64_t(rows) * cols;
        const uint64_t expected_scales = uint64_t(rows) * (cols / kQ8GroupSize) * 2;
        if (!rows || !cols || (cols % kQ8GroupSize) || weight_bytes != expected_weights ||
            scale_bytes != expected_scales || weight_offset > data_bytes_ ||
            weight_bytes > data_bytes_ - weight_offset || scale_offset > data_bytes_ ||
            scale_bytes > data_bytes_ - scale_offset)
            throw std::runtime_error("invalid GLM-5.3 Q8 tensor: " + name);
        if (!tensors_.emplace(std::move(name), Q8TensorLocation{
                rows, cols, weight_offset, weight_bytes, scale_offset, scale_bytes}).second)
            throw std::runtime_error("duplicate GLM-5.3 Q8 tensor");
    }
    if (reader.cursor != reader.end) throw std::runtime_error("trailing bytes in GLM-5.3 Q8 index");
    const std::filesystem::path data_path = prefix.string() + ".bin";
    std::error_code error;
    if (std::filesystem::file_size(data_path, error) != data_bytes_ || error)
        throw std::runtime_error("missing or wrong-size GLM-5.3 Q8 data file");
    data_fd_ = ::open(data_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (data_fd_ < 0)
        throw std::system_error(errno, std::generic_category(), "open GLM-5.3 Q8 data");
    direct_fd_ = ::open(data_path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (direct_fd_ < 0) direct_fd_ = data_fd_;  // filesystem without O_DIRECT
}

Q8Index::~Q8Index() {
    if (direct_fd_ >= 0 && direct_fd_ != data_fd_) ::close(direct_fd_);
    if (data_fd_ >= 0) ::close(data_fd_);
}

const Q8TensorLocation *Q8Index::find(std::string_view name) const {
    const auto found = tensors_.find(std::string(name));
    return found == tensors_.end() ? nullptr : &found->second;
}

void Q8Index::read_rows(const Q8TensorLocation &tensor, uint32_t row, uint32_t rows,
                        void *weights, void *scales) const {
    if (row > tensor.rows || rows > tensor.rows - row)
        throw std::runtime_error("GLM-5.3 Q8 row range exceeds tensor");
    const uint64_t weight_stride = tensor.cols;
    const uint64_t scale_stride = uint64_t(tensor.cols / kQ8GroupSize) * 2;
    pread_all(data_fd_, tensor.weight_offset + uint64_t(row) * weight_stride,
              uint64_t(rows) * weight_stride, weights, "read GLM-5.3 Q8 weights");
    pread_all(data_fd_, tensor.scale_offset + uint64_t(row) * scale_stride,
              uint64_t(rows) * scale_stride, scales, "read GLM-5.3 Q8 scales");
}

void Q8Index::pread_direct(uint64_t offset, uint64_t bytes, uint8_t *destination) const {
    // destination is 4096-aligned with 4096 bytes of slack on both sides; the
    // logical data begins at destination but the disk transfer is padded to
    // full blocks around it.
    const uint64_t head = offset & 4095;
    const uint64_t tail = (4096 - ((offset + bytes) & 4095)) & 4095;
    pread_all(direct_fd_, offset - head, bytes + head + tail, destination - head,
              "read GLM-5.3 Q8 direct");
}

void Q8Index::read_rows_direct(const Q8TensorLocation &tensor, uint8_t *weights,
                               uint8_t *scales) const {
    pread_direct(tensor.weight_offset, tensor.weight_bytes, weights);
    pread_direct(tensor.scale_offset, tensor.scale_bytes, scales);
}

void Q8Index::for_each_by_offset(
    const std::function<void(const std::string &, const Q8TensorLocation &)> &visit) const {
    std::vector<const std::pair<const std::string, Q8TensorLocation> *> order;
    order.reserve(tensors_.size());
    for (const auto &entry : tensors_) order.push_back(&entry);
    std::sort(order.begin(), order.end(),
              [](const auto *left, const auto *right) {
                  return left->second.weight_offset < right->second.weight_offset;
              });
    for (const auto *entry : order) visit(entry->first, entry->second);
}

}  // namespace insignia::glm53
