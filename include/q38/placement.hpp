#pragma once

#include "q38/q38pack.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace q38 {

struct TensorPlacement {
    std::string name;
    TensorRole role{TensorRole::Unknown};
    std::uint64_t payload_bytes{};
    std::uint64_t va_offset{};
    std::uint64_t page_offset{};
};

struct PlacementStats {
    std::uint64_t tensor_payload_bytes{};
    std::uint64_t reserved_va_bytes{};
    std::uint64_t page_padding_bytes{};
    std::uint64_t page_count{};
};

class PlacementPlan {
public:
    static PlacementPlan build(const PackFile& pack, std::size_t gpu_page_bytes);

    [[nodiscard]] std::size_t gpu_page_bytes() const noexcept { return gpu_page_bytes_; }
    [[nodiscard]] const std::vector<TensorPlacement>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const PlacementStats& stats() const noexcept { return stats_; }

private:
    std::size_t gpu_page_bytes_{};
    std::vector<TensorPlacement> tensors_;
    PlacementStats stats_{};
};

} // namespace q38
