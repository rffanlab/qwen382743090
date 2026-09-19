#include "q38/driver.hpp"
#include "q38/q38pack.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    std::string model;
    std::string tensor_name = "blk.0.ffn_gate.weight";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model = argv[++i];
        else if (arg == "--tensor" && i + 1 < argc) tensor_name = argv[++i];
        else if (arg == "--help") {
            std::cout << "usage: q38-iq4xs-gemv --model MODEL.q38pack [--tensor NAME]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (model.empty()) {
        std::cerr << "q38-iq4xs-gemv: --model is required\n";
        return 2;
    }

    try {
        q38::PackFile pack;
        pack.open(model);

        const auto* tensor = pack.find_tensor(tensor_name);
        if (!tensor) {
            std::cerr << "q38-iq4xs-gemv: tensor not found: " << tensor_name << "\n";
            return 3;
        }
        if (tensor->ggml_type != 23 ||
            tensor->layout != q38::TensorLayout::GgufNative) {
            std::cerr << "q38-iq4xs-gemv: tensor must be IQ4_XS/GGUF_NATIVE, type="
                      << tensor->ggml_type << " layout="
                      << static_cast<std::uint32_t>(tensor->layout) << "\n";
            return 4;
        }
        if (tensor->ndim < 2 || tensor->dims[0] == 0 || tensor->dims[1] == 0) {
            std::cerr << "q38-iq4xs-gemv: tensor is not a 2-D matrix\n";
            return 5;
        }

        const auto cols = static_cast<std::uint32_t>(tensor->dims[0]);
        const auto rows = static_cast<std::uint32_t>(tensor->dims[1]);

        std::cout << "Q38RT fused IQ4_XS GEMV 4-warp CTA\n";
        std::cout << "tensor: " << tensor->name << "\n";
        std::cout << "shape: [" << cols << "," << rows << "]\n";
        std::cout << "stored bytes: " << tensor->stored_bytes << "\n";

        q38::NvidiaDriver driver;
        driver.open();
        std::cout << "device: " << driver.device_name() << "\n";

        std::string error;
        double max_abs = 0.0;
        double max_rel = 0.0;
        double ms = 0.0;
        double gbps = 0.0;

        if (!driver.run_iq4xs_gemv_smoke(
                pack.tensor_data(*tensor),
                cols,
                rows,
                &error,
                &max_abs,
                &max_rel,
                &ms,
                &gbps)) {
            std::cerr << "iq4_xs_gemv: FAILED: " << error << "\n";
            return 6;
        }

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "max_abs_error: " << max_abs << "\n";
        std::cout << "max_rel_error: " << max_rel << "\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "kernel_ms: " << ms << "\n";
        std::cout << "effective_weight_bandwidth_GBps: " << gbps << "\n";
        std::cout << "iq4_xs_gemv: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-iq4xs-gemv: " << e.what() << "\n";
        return 1;
    }
}
