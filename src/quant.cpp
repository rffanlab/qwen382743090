#include "q38/quant.hpp"

#include <bit>
#include <cstring>
#include <stdexcept>

namespace q38 {

float fp16_to_fp32(std::uint16_t h) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1fu;
    std::uint32_t mant = h & 0x03ffu;
    std::uint32_t f{};

    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x03ffu;
            const std::uint32_t exp32 = static_cast<std::uint32_t>(e + 127);
            f = sign | (exp32 << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        const std::uint32_t exp32 = exp + (127 - 15);
        f = sign | (exp32 << 23) | (mant << 13);
    }
    return std::bit_cast<float>(f);
}

static void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t& d, std::uint8_t& m) noexcept {
    if (j < 4) {
        d = q[j] & 63u;
        m = q[j + 4] & 63u;
    } else {
        d = static_cast<std::uint8_t>((q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4));
        m = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

void dequantize_q4_k_block_cpu(const std::byte* block, std::array<float, kQ4KValuesPerBlock>& out) {
    if (!block) throw std::invalid_argument("Q4_K block is null");

    std::uint16_t d_bits{};
    std::uint16_t dmin_bits{};
    std::memcpy(&d_bits, block + 0, sizeof(d_bits));
    std::memcpy(&dmin_bits, block + 2, sizeof(dmin_bits));
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    d_bits = __builtin_bswap16(d_bits);
    dmin_bits = __builtin_bswap16(dmin_bits);
#endif

    const float d = fp16_to_fp32(d_bits);
    const float dmin = fp16_to_fp32(dmin_bits);
    const auto* scales = reinterpret_cast<const std::uint8_t*>(block + 4);
    const auto* qs = reinterpret_cast<const std::uint8_t*>(block + 16);

    std::size_t out_pos = 0;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        std::uint8_t sc{}, m{};
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * static_cast<float>(sc);
        const float m1 = dmin * static_cast<float>(m);
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * static_cast<float>(sc);
        const float m2 = dmin * static_cast<float>(m);

        for (int l = 0; l < 32; ++l) {
            out[out_pos++] = d1 * static_cast<float>(qs[l] & 0x0fu) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            out[out_pos++] = d2 * static_cast<float>(qs[l] >> 4) - m2;
        }
        qs += 32;
        is += 2;
    }
}

void dequantize_q5_k_block_cpu(const std::byte* block, std::array<float, kQ4KValuesPerBlock>& out) {
    if (!block) throw std::invalid_argument("Q5_K block is null");

    std::uint16_t d_bits{};
    std::uint16_t dmin_bits{};
    std::memcpy(&d_bits, block + 0, sizeof(d_bits));
    std::memcpy(&dmin_bits, block + 2, sizeof(dmin_bits));
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    d_bits = __builtin_bswap16(d_bits);
    dmin_bits = __builtin_bswap16(dmin_bits);
#endif

    const float d = fp16_to_fp32(d_bits);
    const float dmin = fp16_to_fp32(dmin_bits);
    const auto* scales = reinterpret_cast<const std::uint8_t*>(block + 4);
    const auto* qh = reinterpret_cast<const std::uint8_t*>(block + 16);
    const auto* ql = reinterpret_cast<const std::uint8_t*>(block + 48);

    std::size_t out_pos = 0;
    int is = 0;
    std::uint8_t u1 = 1;
    std::uint8_t u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        std::uint8_t sc{}, m{};
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * static_cast<float>(sc);
        const float m1 = dmin * static_cast<float>(m);
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * static_cast<float>(sc);
        const float m2 = dmin * static_cast<float>(m);

        for (int l = 0; l < 32; ++l) {
            const int qv = static_cast<int>(ql[l] & 0x0fu) + ((qh[l] & u1) ? 16 : 0);
            out[out_pos++] = d1 * static_cast<float>(qv) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            const int qv = static_cast<int>(ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
            out[out_pos++] = d2 * static_cast<float>(qv) - m2;
        }

        ql += 32;
        is += 2;
        u1 = static_cast<std::uint8_t>(u1 << 2);
        u2 = static_cast<std::uint8_t>(u2 << 2);
    }
}

} // namespace q38
