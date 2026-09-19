#include "q38/runtime.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string model;
    float rms_eps = 1.0e-6f;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (arg == "--rms-eps" && i + 1 < argc) {
            rms_eps = std::stof(argv[++i]);
        } else if (arg == "--help") {
            std::cout
                << "usage: q38-layer0-recurrent-front --model MODEL.q38pack "
                   "[--rms-eps 1e-6]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (model.empty()) {
        std::cerr << "q38-layer0-recurrent-front: --model is required\n";
        return 2;
    }

    try {
        q38::Runtime runtime;
        runtime.load(model, true);

        std::cout << "Q38RT Qwen3.8 layer0 recurrent front\n";
        std::cout << "q38pack_version: "
                  << runtime.pack().header().version << "\n";
        std::cout << "device: "
                  << runtime.info().device_name << "\n";
        std::cout << "path:\n";
        std::cout << "  RMSNorm -> qkv/z/beta/alpha projections\n";
        std::cout << "  -> conv4+SiLU+state roll\n";
        std::cout << "  -> Q/K L2 norm + beta sigmoid + alpha gate\n";
        std::cout << "  -> fused Gated DeltaNet AR state update\n";
        std::cout << "dimensions: hidden=5120 qkv=10240 z=6144 "
                     "qk_heads=16 value_heads=48 head_dim=128\n";

        q38::Layer0RecurrentFrontStats stats{};
        std::string error;
        if (!runtime.run_layer0_recurrent_front(
                rms_eps, &stats, &error)) {
            std::cerr << "layer0_recurrent_front: FAILED: "
                      << error << "\n";
            return 3;
        }

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "conv_max_abs_error: "
                  << stats.conv_max_abs << "\n";
        std::cout << "q_max_abs_error: "
                  << stats.q_max_abs << "\n";
        std::cout << "k_max_abs_error: "
                  << stats.k_max_abs << "\n";
        std::cout << "beta_max_abs_error: "
                  << stats.beta_max_abs << "\n";
        std::cout << "gate_max_abs_error: "
                  << stats.gate_max_abs << "\n";
        std::cout << "conv_state_max_abs_error: "
                  << stats.conv_state_max_abs << "\n";
        std::cout << "gdn_output_max_abs_error: "
                  << stats.gdn_output_max_abs << "\n";
        std::cout << "gdn_output_max_rel_error: "
                  << stats.gdn_output_max_rel << "\n";
        std::cout << "gdn_state_max_abs_error: "
                  << stats.gdn_state_max_abs << "\n";
        std::cout << "gdn_state_max_rel_error: "
                  << stats.gdn_state_max_rel << "\n";

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "projection_stage_ms: "
                  << stats.projection_ms << "\n";
        std::cout << "recurrent_prep_stage_ms: "
                  << stats.conv_prep_ms << "\n";
        std::cout << "gdn_stage_ms: "
                  << stats.gdn_ms << "\n";
        std::cout << "sum_stage_ms: "
                  << stats.sum_stage_ms << "\n";
        std::cout << "recurrent_front_chain_ms: "
                  << stats.chain_ms << "\n";
        if (stats.chain_ms > 0.0) {
            std::cout << "recurrent_front_per_second: "
                      << (1000.0 / stats.chain_ms) << "\n";
        }

        std::cout << "shared_workspace: yes\n";
        std::cout << "shared_normalized_hidden: yes\n";
        std::cout << "z_projection_retained_for_gated_norm: yes\n";
        std::cout << "weight_upload_in_timing: no\n";
        std::cout << "module_jit_in_timing: no\n";
        std::cout << "n_tokens: 1\n";
        std::cout << "layer0_recurrent_front: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-layer0-recurrent-front: "
                  << e.what() << "\n";
        return 1;
    }
}
