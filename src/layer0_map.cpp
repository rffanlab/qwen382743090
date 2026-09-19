#include "q38/q38pack.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static const char* type_name(std::uint32_t type) {
    switch (type) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 6: return "Q5_0";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        case 9: return "Q8_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 15: return "Q8_K";
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 19: return "IQ1_S";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        default: return "UNKNOWN";
    }
}

static const char* layout_name(q38::TensorLayout layout) {
    switch (layout) {
        case q38::TensorLayout::GgufNative: return "GGUF_NATIVE";
        case q38::TensorLayout::Sm86Q5KSoA: return "SM86_Q5K_SOA";
        default: return "UNKNOWN";
    }
}

static const char* support_name(const q38::TensorRecord& t) {
    if (t.ggml_type == 0 && t.layout == q38::TensorLayout::GgufNative) {
        return "F32_DIRECT";
    }
    if (t.ggml_type == 13 && t.layout == q38::TensorLayout::Sm86Q5KSoA) {
        return "Q5K_SM86_READY";
    }
    if (t.ggml_type == 12) return "NEED_Q4K_GEMV";
    if (t.ggml_type == 14) return "NEED_Q6K_GEMV";
    if (t.ggml_type == 23) return "NEED_IQ4_XS_GEMV";
    if (t.ggml_type == 8) return "NEED_Q8_0_GEMV";
    return "NEED_KERNEL";
}

static void print_tensor(const q38::PackFile& pack, const std::string& name) {
    const auto* t = pack.find_tensor(name);
    if (!t) {
        std::cout << std::left << std::setw(32) << name << " MISSING\n";
        return;
    }

    std::cout << std::left << std::setw(32) << t->name
              << " type=" << std::setw(8) << type_name(t->ggml_type)
              << " layout=" << std::setw(16) << layout_name(t->layout)
              << " dims=[";
    for (std::uint32_t i = 0; i < t->ndim; ++i) {
        if (i) std::cout << ",";
        std::cout << t->dims[i];
    }
    std::cout << "]"
              << " bytes=" << t->stored_bytes
              << " support=" << support_name(*t)
              << "\n";
}

int main(int argc, char** argv) {
    std::string model;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model = argv[++i];
        else if (arg == "--help") {
            std::cout << "usage: q38-layer0-map --model MODEL.q38pack\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return 2;
        }
    }

    if (model.empty()) {
        std::cerr << "q38-layer0-map: --model is required\n";
        return 2;
    }

    try {
        q38::PackFile pack;
        pack.open(model);

        std::cout << "Q38RT Qwen3.8 layer0 tensor map\n";
        std::cout << "q38pack_version: " << pack.header().version << "\n";
        std::cout << "architecture: qwen35\n\n";

        const std::vector<std::string> tensors = {
            "blk.0.attn_norm.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.attn_gate.weight",
            "blk.0.ssm_conv1d.weight",
            "blk.0.ssm_dt.bias",
            "blk.0.ssm_a",
            "blk.0.ssm_beta.weight",
            "blk.0.ssm_alpha.weight",
            "blk.0.ssm_norm.weight",
            "blk.0.ssm_out.weight",
            "blk.0.attn_post_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        };

        for (const auto& name : tensors) print_tensor(pack, name);

        std::cout << "\nready projection tensors:\n";
        for (const auto& name : tensors) {
            const auto* t = pack.find_tensor(name);
            if (t && t->ggml_type == 13 &&
                t->layout == q38::TensorLayout::Sm86Q5KSoA &&
                t->ndim >= 2) {
                std::cout << "  " << t->name << " [" << t->dims[0] << "," << t->dims[1] << "]\n";
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-layer0-map: " << e.what() << "\n";
        return 1;
    }
}
