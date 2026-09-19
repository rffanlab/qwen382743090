#include "q38/runtime.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    std::string model;
    std::uint32_t layer = 0;
    bool have_layer = false;
    float rms_eps = 1.0e-6f;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (arg == "--layer" && i + 1 < argc) {
            layer = static_cast<std::uint32_t>(
                std::stoul(argv[++i]));
            have_layer = true;
        } else if (arg == "--rms-eps" && i + 1 < argc) {
            rms_eps = std::stof(argv[++i]);
        } else if (arg == "--help") {
            std::cout
                << "usage: q38-recurrent-layer --model MODEL.q38pack "
                   "--layer N [--rms-eps 1e-6]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (model.empty() || !have_layer) {
        std::cerr
            << "q38-recurrent-layer: --model and --layer are required\n";
        return 2;
    }

    try {
        q38::Runtime runtime;
        runtime.load(model, true);

        const auto kind =
            q38::detect_qwen35_layer_kind(
                runtime.pack(), layer);

        std::cout << "Q38RT Qwen3.8 recurrent layer\n";
        std::cout << "layer: " << layer << "\n";
        std::cout << "kind: "
                  << q38::qwen35_layer_kind_name(kind)
                  << "\n";
        std::cout << "device: "
                  << runtime.info().device_name << "\n";

        q38::Layer0FullStats stats{};
        std::string error;
        if (!runtime.run_recurrent_layer(
                layer, rms_eps, &stats, &error)) {
            std::cerr
                << "recurrent_layer: FAILED: "
                << error << "\n";
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
        std::cout << "ffn_ms: "
                  << stats.ffn_ms << "\n";
        std::cout << "sum_stage_ms: "
                  << stats.sum_stage_ms << "\n";
        std::cout << "layer_chain_ms: "
                  << stats.chain_ms << "\n";
        std::cout << "layer_per_second: "
                  << (stats.chain_ms > 0.0
                        ? 1000.0 / stats.chain_ms : 0.0)
                  << "\n";
        std::cout << "recurrent_layer: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr
            << "q38-recurrent-layer: "
            << e.what() << "\n";
        return 1;
    }
}
