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
        if (arg == "--model" && i + 1 < argc) model = argv[++i];
        else if (arg == "--rms-eps" && i + 1 < argc) rms_eps = std::stof(argv[++i]);
        else if (arg == "--help") {
            std::cout
                << "usage: q38-layer0-projections --model MODEL.q38pack [--rms-eps 1e-6]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (model.empty()) {
        std::cerr << "q38-layer0-projections: --model is required\n";
        return 2;
    }

    try {
        q38::Runtime runtime;
        runtime.load(model, true);

        const std::vector<std::string> names = {
            "blk.0.attn_norm.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.attn_gate.weight",
            "blk.0.ssm_beta.weight",
            "blk.0.ssm_alpha.weight",
        };

        std::cout << "Q38RT Qwen3.8 layer0 projection pack\n";
        std::cout << "q38pack_version: " << runtime.pack().header().version << "\n";
        std::cout << "device: " << runtime.info().device_name << "\n";
        std::cout << "dispatcher:\n";
        for (const auto& name : names) {
            const auto* t = runtime.pack().find_tensor(name);
            if (!t) {
                std::cout << "  " << name << " -> MISSING\n";
                continue;
            }
            std::cout << "  " << name << " -> "
                      << q38::projection_kernel_name(
                             q38::select_projection_kernel(*t))
                      << "\n";
        }

        q38::Layer0ProjectionPackStats stats{};
        std::string error;
        if (!runtime.run_layer0_projection_pack(
                rms_eps, &stats, &error)) {
            std::cerr << "layer0_projections: FAILED: " << error << "\n";
            return 3;
        }

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "qkv_max_abs_error: " << stats.qkv.max_abs << "\n";
        std::cout << "qkv_max_rel_error: " << stats.qkv.max_rel << "\n";
        std::cout << "gate_max_abs_error: " << stats.gate.max_abs << "\n";
        std::cout << "gate_max_rel_error: " << stats.gate.max_rel << "\n";
        std::cout << "beta_max_abs_error: " << stats.beta.max_abs << "\n";
        std::cout << "beta_max_rel_error: " << stats.beta.max_rel << "\n";
        std::cout << "alpha_max_abs_error: " << stats.alpha.max_abs << "\n";
        std::cout << "alpha_max_rel_error: " << stats.alpha.max_rel << "\n";

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "rmsnorm_ms: " << stats.rmsnorm_ms << "\n";
        std::cout << "attn_qkv_ms: " << stats.qkv_ms << "\n";
        std::cout << "attn_gate_ms: " << stats.gate_ms << "\n";
        std::cout << "ssm_beta_ms: " << stats.beta_ms << "\n";
        std::cout << "ssm_alpha_ms: " << stats.alpha_ms << "\n";
        std::cout << "sum_individual_ms: "
                  << (stats.rmsnorm_ms + stats.qkv_ms + stats.gate_ms +
                      stats.beta_ms + stats.alpha_ms)
                  << "\n";
        std::cout << "projection_pack_chain_ms: " << stats.chain_ms << "\n";
        if (stats.chain_ms > 0.0) {
            std::cout << "projection_pack_per_second: "
                      << (1000.0 / stats.chain_ms) << "\n";
        }
        std::cout << "shared_normalized_hidden: yes\n";
        std::cout << "weight_upload_in_timing: no\n";
        std::cout << "module_jit_in_timing: no\n";
        std::cout << "layer0_projections: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-layer0-projections: " << e.what() << "\n";
        return 1;
    }
}
