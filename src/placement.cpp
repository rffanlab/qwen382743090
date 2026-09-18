#include "q38/placement.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace q38 {
namespace {

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("GPU page size must be a power of two");
    }
    if (value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1)) {
        throw std::overflow_error("placement size overflow");
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace

PlacementPlan PlacementPlan::build(const PackFile& pack, std::size_t gpu_page_bytes) {
    if (!pack.is_open()) throw std::invalid_argument("cannot build placement plan for a closed Q38PACK");
    if (gpu_page_bytes == 0 || (gpu_page_bytes & (gpu_page_bytes - 1)) != 0) {
        throw std::invalid_argument("GPU page size must be a positive power of two");
    }

    PlacementPlan plan;
    plan.gpu_page_bytes_ = gpu_page_bytes;
    plan.tensors_.reserve(pack.tensors().size());

    // Q38PACK v1 keeps source GGUF storage spans. Multiple small tensors are
    // packed into the same GPU VMM page whenever possible; we intentionally do
    // not round every tensor to the 2 MiB GA102 allocation granularity.
    std::uint64_t cursor = 0;
    for (const auto& tensor : pack.tensors()) {
        constexpr std::uint64_t kTensorAlignment = 256;
        cursor = align_up(cursor, kTensorAlignment);

        TensorPlacement placement;
        placement.name = tensor.name;
        placement.role = tensor.role;
        placement.payload_bytes = tensor.stored_bytes;
        placement.va_offset = cursor;
        placement.page_offset = cursor % gpu_page_bytes;
        plan.tensors_.push_back(std::move(placement));

        plan.stats_.tensor_payload_bytes += tensor.stored_bytes;
        cursor += tensor.stored_bytes;
    }

    plan.stats_.reserved_va_bytes = align_up(cursor, gpu_page_bytes);
    plan.stats_.page_padding_bytes =
        plan.stats_.reserved_va_bytes >= plan.stats_.tensor_payload_bytes
            ? plan.stats_.reserved_va_bytes - plan.stats_.tensor_payload_bytes
            : 0;
    plan.stats_.page_count = plan.stats_.reserved_va_bytes / gpu_page_bytes;
    return plan;
}

} // namespace q38
