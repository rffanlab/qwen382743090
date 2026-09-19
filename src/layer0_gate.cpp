#include "q38/driver.hpp"
#include "q38/q38pack.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    std::string model;
    float rms_eps = 1.0e-6f;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model = argv[++i];
        else if (arg == "--rms-eps" && i + 1 < argc) rms_eps = std::stof(argv[++i]);
        else if (arg == "--help") {
            std::cout << "usage: q38-layer0-gate --model MODEL.q38pack [--rms-eps 1e-6]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (model.empty()) {
        std::cerr << "q38-layer0-gate: --model is required\n";
        return 2;
    }

    try {
        q38::PackFile pack;
        pack.open(model);

        const auto* norm = pack.find_tensor("blk.0.attn_norm.weight");
        const auto* gate = pack.find_tensor("blk.0.attn_gate.weight");

        if (!norm) {
            std::cerr << "q38-layer0-gate: missing blk.0.attn_norm.weight\n";
            return 3;
        }
        if (!gate) {
            std::cerr << "q38-layer0-gate: missing blk.0.attn_gate.weight\n";
            return 4;
        }

        if (norm->ggml_type != 0 || norm->layout != q38::TensorLayout::GgufNative) {
            std::cerr << "q38-layer0-gate: attn_norm must be F32/GGUF_NATIVE, type="
                      << norm->ggml_type << " layout="
                      << static_cast<std::uint32_t>(norm->layout) << "\n";
            return 5;
        }
        if (norm->ndim != 1 || norm->dims[0] == 0) {
            std::cerr << "q38-layer0-gate: invalid attn_norm shape\n";
            return 6;
        }

        if (gate->ggml_type != 13 ||
            gate->layout != q38::TensorLayout::Sm86Q5KSoA) {
            std::cerr << "q38-layer0-gate: attn_gate must be Q5_K/SM86_Q5K_SOA, type="
                      << gate->ggml_type << " layout="
                      << static_cast<std::uint32_t>(gate->layout) << "\n";
            return 7;
        }
        if (gate->ndim < 2 || gate->dims[0] == 0 || gate->dims[1] == 0) {
            std::cerr << "q38-layer0-gate: invalid attn_gate shape\n";
            return 8;
        }

        const auto cols = static_cast<std::uint32_t>(gate->dims[0]);
        const auto rows = static_cast<std::uint32_t>(gate->dims[1]);
        if (norm->dims[0] != cols) {
            std::cerr << "q38-layer0-gate: norm/gate hidden size mismatch\n";
            return 9;
        }
        if (norm->stored_bytes < static_cast<std::uint64_t>(cols) * sizeof(float)) {
            std::cerr << "q38-layer0-gate: attn_norm storage is too small\n";
            return 10;
        }

        const auto* norm_weight =
            reinterpret_cast<const float*>(pack.tensor_data(*norm));
        const auto* gate_base = pack.tensor_data(*gate);
        const auto qh_offset =
            static_cast<std::size_t>(gate->aux0_offset - gate->data_offset);
        const auto qs_offset =
            static_cast<std::size_t>(gate->aux1_offset - gate->data_offset);

        std::cout << "Q38RT Qwen3.8 layer0 RMSNorm -> attn_gate\n";
        std::cout << "q38pack_version: " << pack.header().version << "\n";
        std::cout << "norm_tensor: " << norm->name << " [" << norm->dims[0] << "]\n";
        std::cout << "gate_tensor: " << gate->name
                  << " [" << cols << "," << rows << "]\n";
        std::cout << "gate_layout: SM86_Q5K_SOA\n";
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "rms_eps: " << rms_eps << "\n";

        q38::NvidiaDriver driver;
        driver.open();
        std::cout << "device: " << driver.device_name() << "\n";

        std::string error;
        double max_abs = 0.0;
        double max_rel = 0.0;
        double rms_ms = 0.0;
        double proj_ms = 0.0;
        double chain_ms = 0.0;
        double proj_gbps = 0.0;

        if (!driver.run_qwen35_layer0_gate_smoke(
                norm_weight,
                gate_base,
                qh_offset,
                qs_offset,
                cols,
                rows,
                rms_eps,
                &error,
                &max_abs,
                &max_rel,
                &rms_ms,
                &proj_ms,
                &chain_ms,
                &proj_gbps)) {
            std::cerr << "layer0_gate: FAILED: " << error << "\n";
            return 11;
        }

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "max_abs_error: " << max_abs << "\n";
        std::cout << "max_rel_error: " << max_rel << "\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "rmsnorm_ms: " << rms_ms << "\n";
        std::cout << "attn_gate_projection_ms: " << proj_ms << "\n";
        std::cout << "attn_gate_original_equiv_GBps: " << proj_gbps << "\n";
        std::cout << "chain_ms: " << chain_ms << "\n";
        if (chain_ms > 0.0) {
            std::cout << "chain_equiv_calls_per_second: " << (1000.0 / chain_ms) << "\n";
        }
        std::cout << "layer0_gate: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-layer0-gate: " << e.what() << "\n";
        return 1;
    }
}
