#include "q38/runtime.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

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
                << "usage: q38-layer0-full --model MODEL.q38pack "
                   "[--rms-eps 1e-6]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (model.empty()) {
        std::cerr << "q38-layer0-full: --model is required\n";
        return 2;
    }

    try {
        q38::Runtime runtime;
        runtime.load(model, true);

        std::cout << "Q38RT Qwen3.8 complete recurrent layer0\n";
        std::cout << "q38pack_version: "
                  << runtime.pack().header().version << "\n";
        std::cout << "device: "
                  << runtime.info().device_name << "\n";
        std::cout << "attention:\n";
        std::cout << "  RMSNorm -> qkv/z/beta/alpha -> conv/GDN\n";
        std::cout << "  -> gated norm -> ssm_out -> residual\n";
        std::cout << "ffn:\n";
        std::cout << "  post_attention_norm\n";
        std::cout << "  -> gate IQ4XS_PRMT + up Q5K_SM86_VEC\n";
        std::cout << "  -> SiLU(gate)*up\n";
        std::cout << "  -> down Q5K_SM86_VEC -> residual\n";

        q38::Layer0FullStats stats{};
        std::string error;
        if (!runtime.run_layer0_full(rms_eps, &stats, &error)) {
            std::cerr << "layer0_full: FAILED: " << error << "\n";
            return 3;
        }

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "ffn_norm_max_abs_error: "
                  << stats.ffn_norm_max_abs << "\n";
        std::cout << "ffn_gate_up_max_abs_error: "
                  << stats.ffn_gate_up_max_abs << "\n";
        std::cout << "ffn_down_max_abs_error: "
                  << stats.ffn_down_max_abs << "\n";
        std::cout << "layer_output_max_abs_error: "
                  << stats.layer_output_max_abs << "\n";

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "attention_ms: "
                  << stats.attention_ms << "\n";
        std::cout << "post_norm_ms: "
                  << stats.post_norm_ms << "\n";
        std::cout << "ffn_gate_ms: "
                  << stats.ffn_gate_ms << "\n";
        std::cout << "ffn_up_ms: "
                  << stats.ffn_up_ms << "\n";
        std::cout << "ffn_pointwise_ms: "
                  << stats.ffn_pointwise_ms << "\n";
        std::cout << "ffn_down_ms: "
                  << stats.ffn_down_ms << "\n";
        std::cout << "ffn_ms: "
                  << stats.ffn_ms << "\n";
        std::cout << "sum_stage_ms: "
                  << stats.sum_stage_ms << "\n";
        std::cout << "layer_chain_ms: "
                  << stats.chain_ms << "\n";
        if (stats.chain_ms > 0.0) {
            std::cout << "layer_per_second: "
                      << (1000.0 / stats.chain_ms) << "\n";
        }

        std::cout << "shared_workspace: yes\n";
        std::cout << "weight_upload_in_timing: no\n";
        std::cout << "module_jit_in_timing: no\n";
        std::cout << "n_tokens: 1\n";
        std::cout << "layer0_full: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-layer0-full: " << e.what() << "\n";
        return 1;
    }
}
