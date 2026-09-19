#include "q38/driver.hpp"
#include "q38/q38pack.hpp"

#include <cstdint>
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
            std::cout << "usage: q38-recurrent-prep --model MODEL.q38pack\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (model.empty()) {
        std::cerr << "q38-recurrent-prep: --model is required\n";
        return 2;
    }

    try {
        q38::PackFile pack;
        pack.open(model);

        const auto* conv = pack.find_tensor("blk.0.ssm_conv1d.weight");
        const auto* dt   = pack.find_tensor("blk.0.ssm_dt.bias");
        const auto* a    = pack.find_tensor("blk.0.ssm_a");

        if (!conv || !dt || !a) {
            std::cerr << "q38-recurrent-prep: missing recurrent F32 tensor\n";
            return 3;
        }
        if (conv->ggml_type != 0 || dt->ggml_type != 0 || a->ggml_type != 0) {
            std::cerr << "q38-recurrent-prep: recurrent prep tensors must be F32\n";
            return 4;
        }
        if (conv->ndim < 2 || conv->dims[0] != 4 || conv->dims[1] != 10240 ||
            dt->dims[0] != 48 || a->dims[0] != 48) {
            std::cerr << "q38-recurrent-prep: unexpected tensor shape\n";
            return 5;
        }

        std::cout << "Q38RT Qwen3.8 recurrent prep\n";
        std::cout << "q38pack_version: " << pack.header().version << "\n";
        std::cout << "conv_tensor: " << conv->name << " [4,10240]\n";
        std::cout << "dt_tensor: " << dt->name << " [48]\n";
        std::cout << "a_tensor: " << a->name << " [48]\n";
        std::cout << "qkv_split: q=2048 k=2048 v=6144\n";
        std::cout << "conv_state: 3x10240 + current qkv\n";

        q38::NvidiaDriver driver;
        driver.open();
        std::cout << "device: " << driver.device_name() << "\n";

        q38::RecurrentPrepSmokeStats stats{};
        std::string error;
        if (!driver.run_recurrent_prep_smoke(
                reinterpret_cast<const float*>(pack.tensor_data(*conv)),
                reinterpret_cast<const float*>(pack.tensor_data(*dt)),
                reinterpret_cast<const float*>(pack.tensor_data(*a)),
                &stats,
                &error)) {
            std::cerr << "recurrent_prep: FAILED: " << error << "\n";
            return 6;
        }

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "conv_max_abs_error: " << stats.conv_max_abs << "\n";
        std::cout << "q_max_abs_error: " << stats.q_max_abs << "\n";
        std::cout << "k_max_abs_error: " << stats.k_max_abs << "\n";
        std::cout << "beta_max_abs_error: " << stats.beta_max_abs << "\n";
        std::cout << "gate_max_abs_error: " << stats.gate_max_abs << "\n";
        std::cout << "conv_state_max_abs_error: " << stats.conv_state_max_abs << "\n";

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "conv_silu_ms: " << stats.conv_silu_ms << "\n";
        std::cout << "qk_norm_ms: " << stats.qk_norm_ms << "\n";
        std::cout << "beta_gate_ms: " << stats.beta_gate_ms << "\n";
        std::cout << "recurrent_prep_chain_ms: " << stats.chain_ms << "\n";
        std::cout << "recurrent_prep: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-recurrent-prep: " << e.what() << "\n";
        return 1;
    }
}
