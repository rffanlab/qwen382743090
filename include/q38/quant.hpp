#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace q38 {

inline constexpr std::size_t kQ4KValuesPerBlock = 256;
inline constexpr std::size_t kQ4KBytesPerBlock = 144;
inline constexpr std::size_t kQ5KBytesPerBlock = 176;
inline constexpr std::size_t kQ8KBytesPerBlock = 292;

float fp16_to_fp32(std::uint16_t bits) noexcept;
void dequantize_q4_k_block_cpu(const std::byte* block, std::array<float, kQ4KValuesPerBlock>& out);
void dequantize_q5_k_block_cpu(const std::byte* block, std::array<float, kQ4KValuesPerBlock>& out);
std::vector<std::byte> quantize_q8_k_cpu(const float* values, std::size_t count);
void dequantize_q8_k_block_cpu(const std::byte* block, std::array<float, kQ4KValuesPerBlock>& out);

} // namespace q38
