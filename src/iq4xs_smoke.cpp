#include "q38/driver.hpp"
#include "q38/q38pack.hpp"
#include "q38/quant.hpp"

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
            std::cout << "usage: q38-iq4xs-smoke --model MODEL.q38pack [--tensor NAME]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (model.empty()) {
        std::cerr << "q38-iq4xs-smoke: --model is required\n";
        return 2;
    }

    try {
        q38::PackFile pack;
        pack.open(model);

        const auto* tensor = pack.find_tensor(tensor_name);
        if (!tensor) {
            std::cerr << "q38-iq4xs-smoke: tensor not found: " << tensor_name << "\n";
            return 3;
        }
        if (tensor->ggml_type != 23 ||
            tensor->layout != q38::TensorLayout::GgufNative) {
            std::cerr << "q38-iq4xs-smoke: tensor must be IQ4_XS/GGUF_NATIVE, type="
                      << tensor->ggml_type << " layout="
                      << static_cast<std::uint32_t>(tensor->layout) << "\n";
            return 4;
        }
        if (tensor->stored_bytes < q38::kIQ4XSBytesPerBlock) {
            std::cerr << "q38-iq4xs-smoke: tensor is too small for one IQ4_XS block\n";
            return 5;
        }

        std::cout << "Q38RT real-model IQ4_XS dequant smoke\n";
        std::cout << "tensor: " << tensor->name << "\n";
        std::cout << "stored bytes: " << tensor->stored_bytes << "\n";
        std::cout << "dims: [";
        for (std::uint32_t i = 0; i < tensor->ndim; ++i) {
            if (i) std::cout << ",";
            std::cout << tensor->dims[i];
        }
        std::cout << "]\n";

        q38::NvidiaDriver driver;
        driver.open();
        std::cout << "device: " << driver.device_name() << "\n";

        std::string error;
        double max_abs = 0.0;
        double max_rel = 0.0;

        if (!driver.run_iq4xs_dequant_smoke(
                pack.tensor_data(*tensor),
                &error,
                &max_abs,
                &max_rel)) {
            std::cerr << "iq4_xs: FAILED: " << error << "\n";
            return 6;
        }

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "max_abs_error: " << max_abs << "\n";
        std::cout << "max_rel_error: " << max_rel << "\n";
        std::cout << "iq4_xs: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-iq4xs-smoke: " << e.what() << "\n";
        return 1;
    }
}
