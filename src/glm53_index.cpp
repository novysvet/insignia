#include "insignia_glm53_index.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <system_error>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>

namespace insignia::glm53 {
namespace {

struct Reader {
    const uint8_t *cursor;
    const uint8_t *end;

    void require(size_t bytes) const {
        if (bytes > size_t(end - cursor)) throw std::runtime_error("truncated GLM-5.3 index");
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
    if (!input) throw std::system_error(errno, std::generic_category(), "open index");
    const std::streamsize size = input.tellg();
    if (size < 0) throw std::runtime_error("cannot size GLM-5.3 index");
    input.seekg(0);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!input.read(reinterpret_cast<char *>(bytes.data()), size)) throw std::runtime_error("short index read");
    return bytes;
}

}  // namespace

ShardedIndex::ShardedIndex(const std::filesystem::path &index_path,
                           const std::filesystem::path &model_root,
                           AlternateShardPolicy alternate_policy) {
    std::vector<uint8_t> bytes = load_file(index_path);
    Reader reader{bytes.data(), bytes.data() + bytes.size()};
    if (reader.string(8) != "IGLMIDX1") throw std::runtime_error("bad GLM-5.3 index magic");
    const uint32_t version = reader.scalar<uint32_t>();
    if (version != 1) throw std::runtime_error("unsupported GLM-5.3 index version");
    (void)reader.scalar<uint32_t>();  // flags
    const uint32_t shard_count = reader.scalar<uint32_t>();
    const uint32_t tensor_count = reader.scalar<uint32_t>();
    hidden_size_ = reader.scalar<uint32_t>();
    layers_ = reader.scalar<uint32_t>();
    vocab_size_ = reader.scalar<uint32_t>();
    experts_ = reader.scalar<uint32_t>();
    active_experts_ = reader.scalar<uint32_t>();
    moe_intermediate_ = reader.scalar<uint32_t>();
    hc_mult_ = reader.scalar<uint32_t>();
    payload_bytes_ = reader.scalar<uint64_t>();

    shard_names_.reserve(shard_count);
    shard_sizes_.reserve(shard_count);
    shard_fds_.reserve(shard_count);
    direct_fds_.reserve(shard_count);
    try {
    // Optional alternate directory serving individual shards (dual-vhdx
    // striping): a shard present under the override opens from there instead
    // of the model root, spreading expert reads across two host drives.
    const char *alt_raw = alternate_policy == AlternateShardPolicy::disabled
                              ? nullptr
                              : std::getenv("INSIGNIA_GLM53_ALT_SHARD_DIR");
    const std::filesystem::path alt_dir = alt_raw ? std::filesystem::path(alt_raw) : std::filesystem::path();
    if (alternate_policy == AlternateShardPolicy::strict_overlay && alt_dir.empty())
        throw std::runtime_error(
            "INSIGNIA_GLM53_STRIPE_INDEX requires INSIGNIA_GLM53_ALT_SHARD_DIR");
    if (!alt_dir.empty()) {
        struct stat primary_stat{}, alternate_stat{};
        if (::stat(model_root.c_str(), &primary_stat) != 0)
            throw std::system_error(errno, std::generic_category(),
                                    "stat primary model root " + model_root.string());
        if (::stat(alt_dir.c_str(), &alternate_stat) != 0 || !S_ISDIR(alternate_stat.st_mode))
            throw std::runtime_error("alternate shard directory is missing: " + alt_dir.string());
        if (primary_stat.st_dev == alternate_stat.st_dev)
            throw std::runtime_error(
                "alternate shard directory is on the primary filesystem; refusing fake striping: " +
                alt_dir.string());
    }
    for (uint32_t index = 0; index < shard_count; ++index) {
        const uint16_t length = reader.scalar<uint16_t>();
        const uint64_t expected_size = reader.scalar<uint64_t>();
        std::string name = reader.string(length);
        const std::filesystem::path primary = model_root / name;
        std::filesystem::path path = primary;
        std::error_code primary_probe;
        const bool primary_valid =
            std::filesystem::file_size(primary, primary_probe) == expected_size && !primary_probe;
        bool use_alt = false;
        if (!alt_dir.empty()) {
            const std::filesystem::path alternate = alt_dir / name;
            std::error_code probe;
            const bool alternate_valid =
                std::filesystem::file_size(alternate, probe) == expected_size && !probe;
            use_alt = alternate_policy == AlternateShardPolicy::strict_overlay
                          ? !primary_valid && alternate_valid
                          : alternate_valid;
            if (use_alt) {
                path = alternate;
                alt_shard_.push_back(1);
                ++alt_shard_count_;
                alt_shard_bytes_ += expected_size;
            } else {
                alt_shard_.push_back(0);
            }
        } else {
            alt_shard_.push_back(0);
        }
        std::error_code error;
        const uint64_t actual_size = std::filesystem::file_size(path, error);
        if (error || actual_size != expected_size)
            throw std::runtime_error("missing or wrong-size shard: " + path.string());
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::system_error(errno, std::generic_category(), "open shard " + path.string());
        shard_names_.push_back(std::move(name));
        shard_sizes_.push_back(expected_size);
        shard_fds_.push_back(fd);
#ifdef O_DIRECT
        direct_fds_.push_back(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT));
        if (use_alt && direct_fds_.back() < 0)
            throw std::system_error(errno, std::generic_category(),
                                    "alternate shard lacks O_DIRECT " + path.string());
#else
        direct_fds_.push_back(-1);
#endif
    }
    if (!alt_dir.empty() && alt_shard_count_ == 0)
        throw std::runtime_error("alternate shard directory resolved zero indexed shards: " +
                                 alt_dir.string());

    tensors_.reserve(tensor_count);
    for (uint32_t index = 0; index < tensor_count; ++index) {
        const uint16_t name_length = reader.scalar<uint16_t>();
        const TensorType type = static_cast<TensorType>(reader.scalar<uint8_t>());
        const uint8_t rank = reader.scalar<uint8_t>();
        const uint16_t shard = reader.scalar<uint16_t>();
        (void)reader.scalar<uint16_t>();
        const uint64_t offset = reader.scalar<uint64_t>();
        const uint64_t length = reader.scalar<uint64_t>();
        std::string name = reader.string(name_length);
        std::vector<uint32_t> shape(rank);
        for (uint8_t dimension = 0; dimension < rank; ++dimension)
            shape[dimension] = reader.scalar<uint32_t>();
        if (shard >= shard_count || offset > shard_sizes_[shard] || length > shard_sizes_[shard] - offset)
            throw std::runtime_error("out-of-range tensor in GLM-5.3 index: " + name);
        if (!tensors_.emplace(std::move(name), TensorLocation{type, shard, offset, length, std::move(shape)}).second)
            throw std::runtime_error("duplicate tensor in GLM-5.3 index");
    }
    if (reader.cursor != reader.end) throw std::runtime_error("trailing bytes in GLM-5.3 index");
    } catch (...) {
        for (const int fd : shard_fds_)
            if (fd >= 0) ::close(fd);
        for (const int fd : direct_fds_)
            if (fd >= 0) ::close(fd);
        shard_fds_.clear();
        direct_fds_.clear();
        throw;
    }
}

ShardedIndex::~ShardedIndex() {
    for (const int fd : shard_fds_)
        if (fd >= 0) ::close(fd);
    for (const int fd : direct_fds_)
        if (fd >= 0) ::close(fd);
}

const TensorLocation &ShardedIndex::tensor(std::string_view name) const {
    const auto found = tensors_.find(std::string(name));
    if (found == tensors_.end()) throw std::runtime_error("missing GLM-5.3 tensor: " + std::string(name));
    return found->second;
}

void ShardedIndex::read(const TensorLocation &location, void *destination) const {
    read_span(location.shard, location.offset, location.bytes, destination);
}

void ShardedIndex::read_span(uint16_t shard, uint64_t offset, uint64_t bytes, void *destination) const {
    if (shard >= shard_fds_.size() || offset > shard_sizes_[shard] || bytes > shard_sizes_[shard] - offset)
        throw std::runtime_error("invalid GLM-5.3 shard span");
    auto *output = static_cast<uint8_t *>(destination);
    uint64_t done = 0;
    while (done < bytes) {
        const ssize_t count = ::pread(shard_fds_[shard], output + done, size_t(bytes - done), off_t(offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::system_error(count < 0 ? errno : EIO, std::generic_category(), "read shard");
        done += static_cast<uint64_t>(count);
    }
}

bool ShardedIndex::direct_io_supported(uint16_t shard) const {
    return shard < direct_fds_.size() && direct_fds_[shard] >= 0;
}

void ShardedIndex::read_span_direct(
    uint16_t shard, uint64_t offset, uint64_t bytes, void *destination) const {
    if (shard >= direct_fds_.size() || offset > shard_sizes_[shard] || bytes > shard_sizes_[shard] - offset)
        throw std::runtime_error("invalid GLM-5.3 direct-I/O shard span");
    if (direct_fds_[shard] < 0) {
        read_span(shard, offset, bytes, destination);
        return;
    }
    constexpr uint64_t alignment = 4096;
    const uint64_t aligned_offset = offset & ~(alignment - 1);
    const uint64_t delta = offset - aligned_offset;
    const uint64_t needed = delta + bytes;
    const uint64_t request = (needed + alignment - 1) & ~(alignment - 1);
    void *bounce = nullptr;
    if (::posix_memalign(&bounce, alignment, size_t(request))) throw std::bad_alloc();
    uint64_t done = 0;
    while (done < request) {
        const ssize_t count = ::pread(direct_fds_[shard], static_cast<uint8_t *>(bounce) + done,
                                      size_t(request - done), off_t(aligned_offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) {
            const int error = errno;
            std::free(bounce);
            throw std::system_error(error, std::generic_category(), "direct read shard");
        }
        if (count == 0) break;
        done += static_cast<uint64_t>(count);
    }
    if (done < needed) {
        std::free(bounce);
        throw std::runtime_error("short direct shard read");
    }
    std::memcpy(destination, static_cast<uint8_t *>(bounce) + delta, size_t(bytes));
    std::free(bounce);
}

// Reads an expanded O_DIRECT window into caller-owned aligned storage and
// returns the payload offset within that window. Expert staging uses this to
// skip a record-sized allocation and CPU memcpy on every routed-expert miss.
uint64_t ShardedIndex::read_span_direct_window(
    uint16_t shard, uint64_t offset, uint64_t bytes,
    void *aligned_destination, uint64_t capacity) const {
    if (shard >= direct_fds_.size() || offset > shard_sizes_[shard] ||
        bytes > shard_sizes_[shard] - offset)
        throw std::runtime_error("invalid GLM-5.3 direct-I/O shard window");
    constexpr uint64_t alignment = 4096;
    if (reinterpret_cast<uintptr_t>(aligned_destination) & (alignment - 1))
        throw std::runtime_error("direct-I/O destination is not page aligned");
    const uint64_t aligned_offset = offset & ~(alignment - 1);
    const uint64_t delta = offset - aligned_offset;
    const uint64_t needed = delta + bytes;
    const uint64_t request = (needed + alignment - 1) & ~(alignment - 1);
    if (request > capacity) throw std::runtime_error("direct-I/O window exceeds destination");
    auto *output = static_cast<uint8_t *>(aligned_destination);
    if (direct_fds_[shard] < 0) {
        read_span(shard, offset, bytes, output + delta);
        return delta;
    }
    const uint64_t available = shard_sizes_[shard] - aligned_offset;
    const uint64_t direct_bytes = std::min(request, available & ~(alignment - 1));
    uint64_t done = 0;
    while (done < direct_bytes) {
        const ssize_t count = ::pread(direct_fds_[shard], output + done,
                                      size_t(direct_bytes - done), off_t(aligned_offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0)
            throw std::system_error(count < 0 ? errno : EIO, std::generic_category(),
                                    "direct read shard window");
        done += static_cast<uint64_t>(count);
    }
    if (done < needed) {
        uint64_t tail = done;
        while (tail < needed) {
            const ssize_t count = ::pread(shard_fds_[shard], output + tail,
                                          size_t(needed - tail), off_t(aligned_offset + tail));
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0)
                throw std::system_error(count < 0 ? errno : EIO, std::generic_category(),
                                        "buffered read shard-window tail");
            tail += static_cast<uint64_t>(count);
        }
    }
    return delta;
}

// Buffered-twin read of read_span_direct_window: identical destination ABI
// (payload at +delta, returns delta) but reads through the page cache, so a
// repeat read of an evicted pinned record is served from RAM. The caller
// decides per record whether to evict_span_cache() (pinned-tier admission)
// or keep the pages (pass-through transients = the L2 tier).
uint64_t ShardedIndex::read_span_cached_window(
    uint16_t shard, uint64_t offset, uint64_t bytes,
    void *destination, uint64_t capacity) const {
    if (shard >= shard_fds_.size() || offset > shard_sizes_[shard] ||
        bytes > shard_sizes_[shard] - offset)
        throw std::runtime_error("invalid GLM-5.3 cached shard window");
    constexpr uint64_t alignment = 4096;
    const uint64_t aligned_offset = offset & ~(alignment - 1);
    const uint64_t delta = offset - aligned_offset;
    const uint64_t needed = delta + bytes;
    if (needed > capacity) throw std::runtime_error("cached shard window exceeds destination");
    auto *output = static_cast<uint8_t *>(destination);
    uint64_t done = 0;
    while (done < needed) {
        const ssize_t count = ::pread(shard_fds_[shard], output + done,
                                      size_t(needed - done), off_t(aligned_offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0)
            throw std::system_error(count < 0 ? errno : EIO, std::generic_category(),
                                    "cached shard window read");
        done += static_cast<uint64_t>(count);
    }
    return delta;
}

void ShardedIndex::evict_span_cache(uint16_t shard, uint64_t offset, uint64_t bytes) const {
    if (shard >= shard_fds_.size()) return;
    constexpr uint64_t alignment = 4096;
    const uint64_t aligned_offset = offset & ~(alignment - 1);
    const uint64_t end = (offset + bytes + alignment - 1) & ~(alignment - 1);
    // Best-effort: DONTNEED is advisory; failures are not actionable.
    (void)::posix_fadvise(shard_fds_[shard], off_t(aligned_offset), off_t(end - aligned_offset),
                          POSIX_FADV_DONTNEED);
}

}  // namespace insignia::glm53
