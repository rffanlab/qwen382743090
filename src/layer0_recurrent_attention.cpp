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
                << "usage: q38-layer0-recurrent-attention --model MODEL.q38pack "
                   "[--rms-eps 1e-6]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (model.empty()) {
        std::cerr
            << "q38-layer0-recurrent-attention: --model is required\n";
        return 2;
    }

    try {
        q38::Runtime runtime;
        runtime.load(model, true);

        std::cout << "Q38RT Qwen3.8 layer0 recurrent attention\n";
        std::cout << "q38pack_version: "
                  << runtime.pack().header().version << "\n";
        std::cout << "device: "
                  << runtime.info().device_name << "\n";
        std::cout << "path:\n";
        std::cout << "  hidden -> RMSNorm\n";
        std::cout << "  -> qkv/z/beta/alpha projections\n";
        std::cout << "  -> conv4+SiLU+QK norm+beta/gate\n";
        std::cout << "  -> fused Gated DeltaNet AR\n";
        std::cout << "  -> gated RMSNorm x SiLU(z)\n";
        std::cout << "  -> ssm_out Q5K_SM86_VEC\n";
        std::cout << "  -> attention residual\n";

        q38::Layer0RecurrentAttentionStats stats{};
        std::string error;
        if (!runtime.run_layer0_recurrent_attention(
                rms_eps, &stats, &error)) {
            std::cerr
                << "layer0_recurrent_attention: FAILED: "
                << error << "\n";
            return 3;
        }

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "gated_norm_max_abs_error: "
                  << stats.gated_norm_max_abs << "\n";
        std::cout << "ssm_out_max_abs_error: "
                  << stats.ssm_out_max_abs << "\n";
        std::cout << "residual_max_abs_error: "
                  << stats.residual_max_abs << "\n";

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "recurrent_front_ms: "
                  << stats.front_ms << "\n";
        std::cout << "gated_norm_ms: "
                  << stats.gated_norm_ms << "\n";
        std::cout << "ssm_out_ms: "
                  << stats.ssm_out_ms << "\n";
        std::cout << "attention_tail_ms: "
                  << stats.tail_ms << "\n";
        std::cout << "sum_stage_ms: "
                  << stats.sum_stage_ms << "\n";
        std::cout << "recurrent_attention_chain_ms: "
                  << stats.chain_ms << "\n";
        if (stats.chain_ms > 0.0) {
            std::cout << "recurrent_attention_per_second: "
                      << (1000.0 / stats.chain_ms) << "\n";
        }

        std::cout << "shared_workspace: yes\n";
        std::cout << "weight_upload_in_timing: no\n";
        std::cout << "module_jit_in_timing: no\n";
        std::cout << "n_tokens: 1\n";
        std::cout << "layer0_recurrent_attention: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr
            << "q38-layer0-recurrent-attention: "
            << e.what() << "\n";
        return 1;
    }
}
