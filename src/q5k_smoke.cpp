#include "q38/driver.hpp"
#include "q38/q38pack.hpp"
#include "q38/quant.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    std::string model;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model = argv[++i];
        else if (arg == "--help") {
            std::cout << "usage: q38-q5k-smoke --model MODEL.q38pack\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (model.empty()) {
        std::cerr << "q38-q5k-smoke: --model is required\n";
        return 2;
    }

    try {
        q38::PackFile pack;
        pack.open(model);

        const q38::TensorRecord* target = nullptr;
        for (const auto& tensor : pack.tensors()) {
            if (tensor.ggml_type == 13 && tensor.stored_bytes >= q38::kQ5KBytesPerBlock) {
                target = &tensor;
                break;
            }
        }
        if (!target) {
            std::cerr << "q38-q5k-smoke: no GGML_TYPE_Q5_K (13) tensor found\n";
            return 3;
        }

        std::cout << "Q38RT real-model Q5_K dequant smoke\n";
        std::cout << "tensor: " << target->name << "\n";
        std::cout << "stored bytes: " << target->stored_bytes << "\n";
        std::cout << "dims: [";
        for (std::uint32_t i = 0; i < target->ndim; ++i) {
            if (i) std::cout << ",";
            std::cout << target->dims[i];
        }
        std::cout << "]\n";

        q38::NvidiaDriver driver;
        driver.open();
        std::cout << "device: " << driver.device_name() << "\n";

        std::string error;
        double max_abs = 0.0;
        double max_rel = 0.0;
        const auto* block = pack.tensor_data(*target);
        if (!driver.run_q5k_dequant_smoke(block, &error, &max_abs, &max_rel)) {
            std::cerr << "q5_k: FAILED: " << error << "\n";
            return 4;
        }

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "max_abs_error: " << max_abs << "\n";
        std::cout << "max_rel_error: " << max_rel << "\n";
        std::cout << "q5_k: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-q5k-smoke: " << e.what() << "\n";
        return 1;
    }
}
