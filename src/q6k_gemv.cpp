#include "q38/driver.hpp"
#include "q38/q38pack.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    std::string model;
    std::string tensor_name = "blk.3.attn_v.weight";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (arg == "--tensor" && i + 1 < argc) {
            tensor_name = argv[++i];
        } else if (arg == "--help") {
            std::cout
                << "usage: q38-q6k-gemv --model MODEL.q38pack "
                   "[--tensor NAME]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (model.empty()) {
        std::cerr << "q38-q6k-gemv: --model is required\n";
        return 2;
    }

    try {
        q38::PackFile pack;
        pack.open(model);

        const auto* tensor = pack.find_tensor(tensor_name);
        if (!tensor) {
            std::cerr << "q38-q6k-gemv: tensor not found: "
                      << tensor_name << "\n";
            return 3;
        }
        if (tensor->ggml_type != 14 ||
            tensor->layout != q38::TensorLayout::GgufNative) {
            std::cerr
                << "q38-q6k-gemv: tensor must be Q6_K/GGUF_NATIVE, type="
                << tensor->ggml_type
                << " layout="
                << static_cast<std::uint32_t>(tensor->layout)
                << "\n";
            return 4;
        }
        if (tensor->ndim < 2 ||
            tensor->dims[0] == 0 ||
            tensor->dims[1] == 0) {
            std::cerr
                << "q38-q6k-gemv: tensor is not a 2-D matrix\n";
            return 5;
        }

        const auto cols =
            static_cast<std::uint32_t>(tensor->dims[0]);
        const auto rows =
            static_cast<std::uint32_t>(tensor->dims[1]);

        std::cout << "Q38RT Q6_K GEMV golden scalar-stride\n";
        std::cout << "tensor: " << tensor->name << "\n";
        std::cout << "shape: [" << cols << "," << rows << "]\n";
        std::cout << "stored bytes: " << tensor->stored_bytes << "\n";

        q38::NvidiaDriver driver;
        driver.open();
        std::cout << "device: " << driver.device_name() << "\n";

        std::string block_error;
        double block_abs = 0.0;
        double block_rel = 0.0;
        std::uint32_t first_bad = 256;
        const bool block_ok = driver.run_q6k_dequant_smoke(
            pack.tensor_data(*tensor),
            &block_error,
            &block_abs,
            &block_rel,
            &first_bad);

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "block_dequant_max_abs_error: "
                  << block_abs << "\n";
        std::cout << "block_dequant_max_rel_error: "
                  << block_rel << "\n";
        if (!block_ok) {
            std::cout << "block_dequant_first_bad_index: "
                      << first_bad << "\n";
            std::cerr << "q6_k_block_dequant: FAILED: "
                      << block_error << "\n";
            return 6;
        }
        std::cout << "q6_k_block_dequant: PASS\n";

        std::string error;
        double max_abs = 0.0;
        double max_rel = 0.0;
        double ms = 0.0;
        double gbps = 0.0;

        if (!driver.run_q6k_gemv_smoke(
                pack.tensor_data(*tensor),
                cols,
                rows,
                &error,
                &max_abs,
                &max_rel,
                &ms,
                &gbps)) {
            std::cerr << "q6_k_gemv: FAILED: "
                      << error << "\n";
            return 7;
        }

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "max_abs_error: " << max_abs << "\n";
        std::cout << "max_rel_error: " << max_rel << "\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "kernel_ms: " << ms << "\n";
        std::cout << "effective_weight_bandwidth_GBps: "
                  << gbps << "\n";
        std::cout << "kernel_mapping: 1warp_per_row_lane_stride_32\n";
        std::cout << "decode: exact_element_index_reference\n";
        std::cout << "q6_k_gemv: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-q6k-gemv: " << e.what() << "\n";
        return 1;
    }
}
