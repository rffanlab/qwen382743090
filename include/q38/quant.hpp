#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace q38 {

inline constexpr std::size_t kQ4KValuesPerBlock = 256;
inline constexpr std::size_t kQ4KBytesPerBlock = 144;
inline constexpr std::size_t kQ5KBytesPerBlock = 176;

float fp16_to_fp32(std::uint16_t bits) noexcept;
void dequantize_q4_k_block_cpu(const std::byte* block, std::array<float, kQ4KValuesPerBlock>& out);
void dequantize_q5_k_block_cpu(const std::byte* block, std::array<float, kQ4KValuesPerBlock>& out);

} // namespace q38
