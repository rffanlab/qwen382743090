#include "q38/q38pack.hpp"

#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace q38 {
namespace {

template <class T>
T read_le(const std::byte* p) {
    T value{};
    std::memcpy(&value, p, sizeof(T));
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if constexpr (sizeof(T) == 4) value = __builtin_bswap32(value);
    if constexpr (sizeof(T) == 8) value = __builtin_bswap64(value);
#endif
    return value;
}

void ensure_range(std::uint64_t off, std::uint64_t len, std::uint64_t file_size, const char* what) {
    if (off > file_size || len > file_size - off) {
        throw std::runtime_error(std::string("q38pack: out-of-range ") + what);
    }
}

} // namespace

PackFile::~PackFile() { close(); }

PackFile::PackFile(PackFile&& other) noexcept { *this = std::move(other); }

PackFile& PackFile::operator=(PackFile&& other) noexcept {
    if (this == &other) return *this;
    close();
    fd_ = other.fd_;
    file_size_ = other.file_size_;
    mapped_ = other.mapped_;
    header_ = other.header_;
    tensors_ = std::move(other.tensors_);
    other.fd_ = -1;
    other.file_size_ = 0;
    other.mapped_ = nullptr;
    manifest_ = {};
    if (mapped_ && header_.manifest_bytes) {
        const auto* base = static_cast<const char*>(mapped_);
        manifest_ = std::string_view(base + header_.manifest_offset, header_.manifest_bytes);
    }
    return *this;
}

void PackFile::open(const std::filesystem::path& path) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::runtime_error("q38pack: unable to open " + path.string());

    struct stat st {};
    if (::fstat(fd_, &st) != 0 || st.st_size < static_cast<off_t>(kPackHeaderBytes)) {
        close();
        throw std::runtime_error("q38pack: invalid or too-small file");
    }
    file_size_ = static_cast<std::size_t>(st.st_size);
    mapped_ = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_ == MAP_FAILED) {
        mapped_ = nullptr;
        close();
        throw std::runtime_error("q38pack: mmap failed");
    }
    parse();
}

void PackFile::close() noexcept {
    manifest_ = {};
    tensors_.clear();
    header_ = {};
    if (mapped_) {
        ::munmap(mapped_, file_size_);
        mapped_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    file_size_ = 0;
}

void PackFile::parse() {
    const auto* p = static_cast<const std::byte*>(mapped_);
    std::memcpy(header_.magic.data(), p, 8);
    if (header_.magic != kPackMagic) throw std::runtime_error("q38pack: bad magic");

    header_.version = read_le<std::uint32_t>(p + 8);
    header_.header_bytes = read_le<std::uint32_t>(p + 12);
    header_.flags = read_le<std::uint32_t>(p + 16);
    header_.tensor_count = read_le<std::uint32_t>(p + 20);
    header_.alignment = read_le<std::uint32_t>(p + 24);
    header_.reserved0 = read_le<std::uint32_t>(p + 28);
    header_.directory_offset = read_le<std::uint64_t>(p + 32);
    header_.directory_bytes = read_le<std::uint64_t>(p + 40);
    header_.manifest_offset = read_le<std::uint64_t>(p + 48);
    header_.manifest_bytes = read_le<std::uint64_t>(p + 56);
    header_.raw_gguf_meta_offset = read_le<std::uint64_t>(p + 64);
    header_.raw_gguf_meta_bytes = read_le<std::uint64_t>(p + 72);
    header_.data_offset = read_le<std::uint64_t>(p + 80);
    header_.data_bytes = read_le<std::uint64_t>(p + 88);
    header_.source_file_size = read_le<std::uint64_t>(p + 96);
    header_.source_data_offset = read_le<std::uint64_t>(p + 104);

    if (header_.version != kPackVersion || header_.header_bytes != kPackHeaderBytes) {
        throw std::runtime_error("q38pack: unsupported version");
    }
    if (header_.directory_bytes != static_cast<std::uint64_t>(header_.tensor_count) * kTensorEntryBytes) {
        throw std::runtime_error("q38pack: malformed directory size");
    }

    ensure_range(header_.directory_offset, header_.directory_bytes, file_size_, "directory");
    ensure_range(header_.manifest_offset, header_.manifest_bytes, file_size_, "manifest");
    ensure_range(header_.raw_gguf_meta_offset, header_.raw_gguf_meta_bytes, file_size_, "raw GGUF metadata");
    ensure_range(header_.data_offset, header_.data_bytes, file_size_, "tensor data");

    manifest_ = std::string_view(
        static_cast<const char*>(mapped_) + header_.manifest_offset,
        static_cast<std::size_t>(header_.manifest_bytes));

    tensors_.reserve(header_.tensor_count);
    const auto* dir = p + header_.directory_offset;
    for (std::uint32_t i = 0; i < header_.tensor_count; ++i) {
        const auto* e = dir + static_cast<std::size_t>(i) * kTensorEntryBytes;
        std::size_t name_len = 0;
        while (name_len < 160 && reinterpret_cast<const char*>(e)[name_len] != '\0') ++name_len;

        TensorRecord t;
        t.name.assign(reinterpret_cast<const char*>(e), name_len);
        t.ndim = read_le<std::uint32_t>(e + 160);
        t.ggml_type = read_le<std::uint32_t>(e + 164);
        if (t.ndim > 4) throw std::runtime_error("q38pack: tensor has more than 4 dimensions");
        for (int d = 0; d < 4; ++d) t.dims[d] = read_le<std::uint64_t>(e + 168 + d * 8);
        t.data_offset = read_le<std::uint64_t>(e + 200);
        t.stored_bytes = read_le<std::uint64_t>(e + 208);
        t.source_offset = read_le<std::uint64_t>(e + 216);
        t.role = static_cast<TensorRole>(read_le<std::uint32_t>(e + 224));
        t.flags = read_le<std::uint32_t>(e + 228);
        ensure_range(t.data_offset, t.stored_bytes, file_size_, "tensor");
        if (t.data_offset < header_.data_offset) throw std::runtime_error("q38pack: tensor points outside data area");
        tensors_.push_back(std::move(t));
    }
}

const TensorRecord* PackFile::find_tensor(std::string_view name) const noexcept {
    for (const auto& t : tensors_) if (t.name == name) return &t;
    return nullptr;
}

const std::byte* PackFile::tensor_data(const TensorRecord& tensor) const {
    if (!mapped_) throw std::runtime_error("q38pack: no file open");
    ensure_range(tensor.data_offset, tensor.stored_bytes, file_size_, "tensor");
    return static_cast<const std::byte*>(mapped_) + tensor.data_offset;
}

} // namespace q38
