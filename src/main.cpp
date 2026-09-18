#include "q38/runtime.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

static double gib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / 1024.0 / 1024.0 / 1024.0;
}

int main(int argc, char** argv) {
    std::string model;
    bool no_gpu = false;
    bool list_tensors = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model = argv[++i];
        else if (arg == "--no-gpu") no_gpu = true;
        else if (arg == "--list-tensors") list_tensors = true;
        else if (arg == "--help") {
            std::cout << "usage: q38-runtime --model MODEL.q38pack [--no-gpu] [--list-tensors]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (model.empty()) {
        std::cerr << "q38-runtime: --model is required\n";
        return 2;
    }

    try {
        q38::Runtime runtime;
        runtime.load(model, !no_gpu);
        const auto info = runtime.info();

        std::cout << "Q38 Runtime probe\n";
        std::cout << "model: " << info.model_path << "\n";
        std::cout << "tensors: " << info.tensor_count << "\n";
        std::cout << std::fixed << std::setprecision(2)
                  << "packed tensor bytes: " << gib(info.packed_bytes) << " GiB\n";
        std::cout << "driver: " << (info.driver_available ? "available" : "not available") << "\n";
        if (info.driver_available) {
            std::cout << "driver version: " << info.driver_version << "\n";
            std::cout << "device: " << info.device_name << "\n";
            std::cout << "VRAM: " << gib(info.device_memory) << " GiB\n";
        }
        std::cout << "native decode: " << (info.native_decode_ready ? "ready" : "not implemented yet") << "\n";

        if (list_tensors) {
            for (const auto& t : runtime.pack().tensors()) {
                std::cout << t.name << " type=" << t.ggml_type << " dims=[";
                for (std::uint32_t d = 0; d < t.ndim; ++d) {
                    if (d) std::cout << ",";
                    std::cout << t.dims[d];
                }
                std::cout << "] bytes=" << t.stored_bytes << "\n";
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-runtime: " << e.what() << "\n";
        return 1;
    }
}
