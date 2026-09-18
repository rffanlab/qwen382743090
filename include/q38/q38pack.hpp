#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace q38 {

inline constexpr std::array<char, 8> kPackMagic{'Q','3','8','P','A','C','K','\0'};
inline constexpr std::uint32_t kPackVersion = 1;
inline constexpr std::uint32_t kPackHeaderBytes = 256;
inline constexpr std::uint32_t kTensorEntryBytes = 256;

enum class TensorRole : std::uint32_t {
    Unknown = 0,
    TokenEmbedding = 1,
    OutputNorm = 2,
    Output = 3,
    Attention = 4,
    DeltaNet = 5,
    FeedForward = 6,
    Norm = 7,
    Mtp = 8,
};

struct PackHeader {
    std::array<char, 8> magic{};
    std::uint32_t version{};
    std::uint32_t header_bytes{};
    std::uint32_t flags{};
    std::uint32_t tensor_count{};
    std::uint32_t alignment{};
    std::uint32_t reserved0{};
    std::uint64_t directory_offset{};
    std::uint64_t directory_bytes{};
    std::uint64_t manifest_offset{};
    std::uint64_t manifest_bytes{};
    std::uint64_t raw_gguf_meta_offset{};
    std::uint64_t raw_gguf_meta_bytes{};
    std::uint64_t data_offset{};
    std::uint64_t data_bytes{};
    std::uint64_t source_file_size{};
    std::uint64_t source_data_offset{};
};

struct TensorRecord {
    std::string name;
    std::uint32_t ndim{};
    std::uint32_t ggml_type{};
    std::array<std::uint64_t, 4> dims{};
    std::uint64_t data_offset{};
    std::uint64_t stored_bytes{};
    std::uint64_t source_offset{};
    TensorRole role{TensorRole::Unknown};
    std::uint32_t flags{};
};

class PackFile {
public:
    PackFile() = default;
    ~PackFile();
    PackFile(const PackFile&) = delete;
    PackFile& operator=(const PackFile&) = delete;
    PackFile(PackFile&& other) noexcept;
    PackFile& operator=(PackFile&& other) noexcept;

    void open(const std::filesystem::path& path);
    void close() noexcept;

    [[nodiscard]] bool is_open() const noexcept { return mapped_ != nullptr; }
    [[nodiscard]] const PackHeader& header() const noexcept { return header_; }
    [[nodiscard]] const std::vector<TensorRecord>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] std::string_view manifest_json() const noexcept { return manifest_; }
    [[nodiscard]] const TensorRecord* find_tensor(std::string_view name) const noexcept;
    [[nodiscard]] const std::byte* tensor_data(const TensorRecord& tensor) const;

private:
    int fd_{-1};
    std::size_t file_size_{0};
    void* mapped_{nullptr};
    PackHeader header_{};
    std::vector<TensorRecord> tensors_;
    std::string_view manifest_{};

    void parse();
};

} // namespace q38
