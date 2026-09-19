#include "q38/driver.hpp"
#include "q38/q38pack.hpp"
#include "q38/quant.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

int main(int argc, char** argv) {
    std::string model;
    std::string tensor_name = "blk.0.attn_gate.weight";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model = argv[++i];
        else if (arg == "--tensor" && i + 1 < argc) tensor_name = argv[++i];
        else if (arg == "--help") {
            std::cout << "usage: q38-q5k-sm86-gemv --model MODEL.q38pack [--tensor NAME]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (model.empty()) {
        std::cerr << "q38-q5k-sm86-gemv: --model is required\n";
        return 2;
    }

    try {
        q38::PackFile pack;
        pack.open(model);

        const auto* tensor = pack.find_tensor(tensor_name);
        if (!tensor) {
            std::cerr << "q38-q5k-sm86-gemv: tensor not found: " << tensor_name << "\n";
            return 3;
        }
        if (tensor->ggml_type != 13) {
            std::cerr << "q38-q5k-sm86-gemv: tensor is not GGML_TYPE_Q5_K (13), type="
                      << tensor->ggml_type << "\n";
            return 4;
        }
        if (tensor->ndim < 2 || tensor->dims[0] == 0 || tensor->dims[1] == 0) {
            std::cerr << "q38-q5k-sm86-gemv: tensor is not a 2-D matrix\n";
            return 5;
        }

        const auto cols = static_cast<std::uint32_t>(tensor->dims[0]);
        const auto rows = static_cast<std::uint32_t>(tensor->dims[1]);
        if (cols % q38::kQ4KValuesPerBlock != 0) {
            std::cerr << "q38-q5k-sm86-gemv: cols must be divisible by 256\n";
            return 6;
        }

        const std::size_t blocks_per_row = cols / q38::kQ4KValuesPerBlock;
        const std::size_t total_blocks = static_cast<std::size_t>(rows) * blocks_per_row;
        const std::size_t original_bytes = total_blocks * q38::kQ5KBytesPerBlock;

        std::vector<std::byte> temporary_repack;
        const std::byte* repacked_ptr = nullptr;
        std::size_t qh_offset = 0;
        std::size_t qs_offset = 0;
        std::size_t repacked_bytes = 0;
        double repack_ms = 0.0;
        double repack_decode_max_abs = 0.0;
        bool persisted_v2 = tensor->layout == q38::TensorLayout::Sm86Q5KSoA;

        if (persisted_v2) {
            if (pack.header().version < 2) {
                throw std::runtime_error("SM86 Q5_K layout requires Q38PACK v2");
            }
            repacked_ptr = pack.tensor_data(*tensor);
            qh_offset = static_cast<std::size_t>(tensor->aux0_offset - tensor->data_offset);
            qs_offset = static_cast<std::size_t>(tensor->aux1_offset - tensor->data_offset);
            repacked_bytes = tensor->stored_bytes;
        } else if (tensor->layout == q38::TensorLayout::GgufNative) {
            if (tensor->stored_bytes < original_bytes) {
                std::cerr << "q38-q5k-sm86-gemv: native tensor storage is smaller than expected Q5_K payload\n";
                return 7;
            }

            const std::size_t meta_bytes = total_blocks * 20;
            qh_offset = align_up(meta_bytes, 128);
            const std::size_t qh_bytes = total_blocks * 32;
            qs_offset = align_up(qh_offset + qh_bytes, 128);
            const std::size_t qs_bytes = total_blocks * 128;
            repacked_bytes = qs_offset + qs_bytes;

            temporary_repack.resize(repacked_bytes);
            std::array<std::byte, q38::kQ5KSm86BytesPerBlock> temp{};
            const auto* source = pack.tensor_data(*tensor);

            const auto repack_t0 = std::chrono::steady_clock::now();
            for (std::size_t ib = 0; ib < total_blocks; ++ib) {
                q38::repack_q5_k_sm86_block(
                    source + ib * q38::kQ5KBytesPerBlock, temp.data());
                std::memcpy(temporary_repack.data() + ib * 20, temp.data() + 0, 20);
                std::memcpy(temporary_repack.data() + qh_offset + ib * 32, temp.data() + 20, 32);
                std::memcpy(temporary_repack.data() + qs_offset + ib * 128, temp.data() + 52, 128);
            }
            const auto repack_t1 = std::chrono::steady_clock::now();
            repack_ms =
                std::chrono::duration<double, std::milli>(repack_t1 - repack_t0).count();

            std::array<float, q38::kQ4KValuesPerBlock> original{};
            std::array<float, q38::kQ4KValuesPerBlock> transformed{};
            std::array<std::byte, q38::kQ5KSm86BytesPerBlock> check_block{};
            const std::size_t checked_blocks = std::min<std::size_t>(total_blocks, 64);
            for (std::size_t ib = 0; ib < checked_blocks; ++ib) {
                q38::dequantize_q5_k_block_cpu(
                    source + ib * q38::kQ5KBytesPerBlock, original);
                std::memcpy(check_block.data() + 0, temporary_repack.data() + ib * 20, 20);
                std::memcpy(check_block.data() + 20, temporary_repack.data() + qh_offset + ib * 32, 32);
                std::memcpy(check_block.data() + 52, temporary_repack.data() + qs_offset + ib * 128, 128);
                q38::dequantize_q5_k_sm86_block_cpu(check_block.data(), transformed);
                for (std::size_t j = 0; j < q38::kQ4KValuesPerBlock; ++j) {
                    repack_decode_max_abs = std::max(
                        repack_decode_max_abs,
                        std::abs(static_cast<double>(original[j]) - static_cast<double>(transformed[j])));
                }
            }
            if (repack_decode_max_abs != 0.0) {
                std::cerr << "q38-q5k-sm86-gemv: repack changed decoded weights, max_abs="
                          << repack_decode_max_abs << "\n";
                return 9;
            }

            repacked_ptr = temporary_repack.data();
        } else {
            std::cerr << "q38-q5k-sm86-gemv: unsupported tensor layout "
                      << static_cast<std::uint32_t>(tensor->layout) << "\n";
            return 10;
        }

        std::cout << "Q38RT SM86 Q5_K SoA GEMV vectorized\n";
        std::cout << "q38pack_version: " << pack.header().version << "\n";
        std::cout << "source_layout: " << (persisted_v2 ? "persisted_v2" : "temporary_from_native") << "\n";
        std::cout << "tensor: " << tensor->name << "\n";
        std::cout << "shape: [" << cols << "," << rows << "]\n";
        std::cout << "original_Q5K_bytes: " << original_bytes << "\n";
        std::cout << "repacked_bytes: " << repacked_bytes << "\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "repack_size_overhead_percent: "
                  << (100.0 * static_cast<double>(repacked_bytes - original_bytes) /
                      static_cast<double>(original_bytes))
                  << "\n";
        std::cout << "offline_repack_ms: " << repack_ms << "\n";
        if (!persisted_v2) {
            std::cout << std::scientific << std::setprecision(6);
            std::cout << "repack_decode_max_abs_error: " << repack_decode_max_abs << "\n";
            std::cout << std::fixed << std::setprecision(3);
        } else {
            std::cout << "repack_decode_max_abs_error: persisted_at_conversion\n";
        }
        std::cout << "layout: META20_SOA + QH32_SOA + QS128_SOA\n";

        q38::NvidiaDriver driver;
        driver.open();
        std::cout << "device: " << driver.device_name() << "\n";

        std::string error;
        double max_abs = 0.0;
        double max_rel = 0.0;
        double ms = 0.0;
        double original_gbps = 0.0;
        double physical_gbps = 0.0;

        if (!driver.run_q5k_sm86_gemv_smoke(
                repacked_ptr,
                qh_offset,
                qs_offset,
                cols,
                rows,
                &error,
                &max_abs,
                &max_rel,
                &ms,
                &original_gbps,
                &physical_gbps)) {
            std::cerr << "q5_k_sm86: FAILED: " << error << "\n";
            return 8;
        }

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "max_abs_error: " << max_abs << "\n";
        std::cout << "max_rel_error: " << max_rel << "\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "kernel_ms: " << ms << "\n";
        std::cout << "original_equiv_weight_bandwidth_GBps: " << original_gbps << "\n";
        std::cout << "physical_repacked_bandwidth_GBps: " << physical_gbps << "\n";
        std::cout << "q5_k_sm86: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-q5k-sm86-gemv: " << e.what() << "\n";
        return 1;
    }
}
