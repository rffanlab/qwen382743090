#include "q38/quant.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <cmath>
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

namespace {

int nearest_int_q38(float fval) noexcept {
    // Matches ggml's nearest_int helper for the small magnitudes used by Q8_K.
    float val = fval + 12582912.0f;
    int i{};
    std::memcpy(&i, &val, sizeof(i));
    return (i & 0x007fffff) - 0x00400000;
}

} // namespace

std::vector<std::byte> quantize_q8_k_cpu(const float* values, std::size_t count) {
    if (!values) throw std::invalid_argument("Q8_K source is null");
    if (count == 0 || count % kQ4KValuesPerBlock != 0) {
        throw std::invalid_argument("Q8_K count must be a positive multiple of 256");
    }

    const std::size_t blocks = count / kQ4KValuesPerBlock;
    std::vector<std::byte> out(blocks * kQ8KBytesPerBlock);

    for (std::size_t ib = 0; ib < blocks; ++ib) {
        const float* x = values + ib * kQ4KValuesPerBlock;
        std::byte* block = out.data() + ib * kQ8KBytesPerBlock;

        float max_value = 0.0f;
        float amax = 0.0f;
        for (std::size_t j = 0; j < kQ4KValuesPerBlock; ++j) {
            const float ax = std::fabs(x[j]);
            if (ax > amax) {
                amax = ax;
                max_value = x[j];
            }
        }

        float d = 0.0f;
        auto* qs = reinterpret_cast<std::int8_t*>(block + 4);
        auto* bsums = reinterpret_cast<std::int16_t*>(block + 260);

        if (amax == 0.0f) {
            std::memcpy(block, &d, sizeof(d));
            std::memset(qs, 0, kQ4KValuesPerBlock);
            std::memset(bsums, 0, 16 * sizeof(std::int16_t));
            continue;
        }

        const float iscale = -127.0f / max_value;
        d = 1.0f / iscale;
        std::memcpy(block, &d, sizeof(d));

        for (std::size_t j = 0; j < kQ4KValuesPerBlock; ++j) {
            const int v = nearest_int_q38(iscale * x[j]);
            qs[j] = static_cast<std::int8_t>(std::min(127, v));
        }

        for (std::size_t j = 0; j < 16; ++j) {
            int sum = 0;
            for (std::size_t k = 0; k < 16; ++k) {
                sum += static_cast<int>(qs[j * 16 + k]);
            }
            bsums[j] = static_cast<std::int16_t>(sum);
        }
    }

    return out;
}

void dequantize_q8_k_block_cpu(const std::byte* block, std::array<float, kQ4KValuesPerBlock>& out) {
    if (!block) throw std::invalid_argument("Q8_K block is null");
    float d{};
    std::memcpy(&d, block, sizeof(d));
    const auto* qs = reinterpret_cast<const std::int8_t*>(block + 4);
    for (std::size_t i = 0; i < kQ4KValuesPerBlock; ++i) {
        out[i] = d * static_cast<float>(qs[i]);
    }
}

void repack_q5_k_sm86_block(const std::byte* source, std::byte* destination) {
    if (!source || !destination) throw std::invalid_argument("Q5_K SM86 repack received null pointer");

    // Layout:
    //   0..3   : d, dmin (original fp16 bit-patterns)
    //   4..19  : 8 x uint16 { low8 = scale, high8 = min }
    //   20..51 : qh[32] (unchanged)
    //   52..179: qs[128] (unchanged)
    std::memcpy(destination + 0, source + 0, 4);

    const auto* scales = reinterpret_cast<const std::uint8_t*>(source + 4);
    for (int group = 0; group < 8; ++group) {
        std::uint8_t sc{}, m{};
        get_scale_min_k4(group, scales, sc, m);
        const std::uint16_t packed =
            static_cast<std::uint16_t>(sc) |
            (static_cast<std::uint16_t>(m) << 8);
        std::memcpy(destination + 4 + group * 2, &packed, sizeof(packed));
    }

    std::memcpy(destination + 20, source + 16, 32);
    std::memcpy(destination + 52, source + 48, 128);
}

void dequantize_q5_k_sm86_block_cpu(const std::byte* block, std::array<float, kQ4KValuesPerBlock>& out) {
    if (!block) throw std::invalid_argument("Q5_K SM86 block is null");

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
    const auto* qh = reinterpret_cast<const std::uint8_t*>(block + 20);
    const auto* ql = reinterpret_cast<const std::uint8_t*>(block + 52);

    std::size_t out_pos = 0;
    for (int group = 0; group < 8; ++group) {
        std::uint16_t sm{};
        std::memcpy(&sm, block + 4 + group * 2, sizeof(sm));
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        sm = __builtin_bswap16(sm);
#endif
        const float ds = d * static_cast<float>(sm & 0xffu);
        const float dm = dmin * static_cast<float>(sm >> 8);

        const int pair = group >> 1;
        const bool high = (group & 1) != 0;
        const std::uint8_t high_mask = static_cast<std::uint8_t>(1u << group);
        const auto* qlp = ql + pair * 32;

        for (int lane = 0; lane < 32; ++lane) {
            const int low4 = high ? (qlp[lane] >> 4) : (qlp[lane] & 0x0f);
            const int qv = low4 + ((qh[lane] & high_mask) ? 16 : 0);
            out[out_pos++] = ds * static_cast<float>(qv) - dm;
        }
    }
}

} // namespace q38
