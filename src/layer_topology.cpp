#include "q38/runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static const char* type_name(std::uint32_t t) {
    switch (t) {
        case 0: return "F32";
        case 8: return "Q8_0";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 23: return "IQ4_XS";
        default: return "OTHER";
    }
}

static const char* layout_name(q38::TensorLayout l) {
    switch (l) {
        case q38::TensorLayout::GgufNative: return "GGUF_NATIVE";
        case q38::TensorLayout::Sm86Q5KSoA: return "SM86_Q5K_SOA";
        default: return "UNKNOWN";
    }
}

static std::string dims_string(const q38::TensorRecord& t) {
    std::ostringstream oss;
    oss << "[";
    for (std::uint32_t i = 0; i < t.ndim; ++i) {
        if (i) oss << ",";
        oss << t.dims[i];
    }
    oss << "]";
    return oss.str();
}

static std::string tensor_signature(
    const q38::PackFile& pack,
    std::uint32_t layer) {
    const std::string prefix =
        "blk." + std::to_string(layer) + ".";
    std::vector<std::string> parts;
    for (const auto& t : pack.tensors()) {
        if (t.name.rfind(prefix, 0) != 0) continue;
        std::ostringstream oss;
        oss << t.name.substr(prefix.size())
            << ":" << t.ggml_type
            << ":" << static_cast<std::uint32_t>(t.layout)
            << ":" << dims_string(t);
        parts.push_back(oss.str());
    }
    std::sort(parts.begin(), parts.end());
    std::ostringstream out;
    for (const auto& p : parts) out << p << "\n";
    return out.str();
}

static bool recurrent_ready(
    const q38::PackFile& pack,
    std::uint32_t layer,
    std::string& why) {
    const std::string p =
        "blk." + std::to_string(layer) + ".";

    auto get = [&](const char* suffix) -> const q38::TensorRecord* {
        return pack.find_tensor(p + suffix);
    };

    struct Req {
        const char* name;
        q38::ProjectionKernelKind kind;
    };
    const Req projections[] = {
        {"attn_norm.weight", q38::ProjectionKernelKind::F32Direct},
        {"attn_qkv.weight", q38::ProjectionKernelKind::Q5KSm86Vectorized},
        {"attn_gate.weight", q38::ProjectionKernelKind::Q5KSm86Vectorized},
        {"ssm_beta.weight", q38::ProjectionKernelKind::Q4KNative},
        {"ssm_alpha.weight", q38::ProjectionKernelKind::Q4KNative},
        {"ssm_out.weight", q38::ProjectionKernelKind::Q5KSm86Vectorized},
        {"post_attention_norm.weight", q38::ProjectionKernelKind::F32Direct},
        {"ffn_gate.weight", q38::ProjectionKernelKind::IQ4XSPrmt},
        {"ffn_up.weight", q38::ProjectionKernelKind::Q5KSm86Vectorized},
        {"ffn_down.weight", q38::ProjectionKernelKind::Q5KSm86Vectorized},
    };

    for (const auto& r : projections) {
        const auto* t = get(r.name);
        if (!t) {
            why = std::string("missing ") + r.name;
            return false;
        }
        if (q38::select_projection_kernel(*t) != r.kind) {
            why = std::string("unsupported ") + r.name;
            return false;
        }
    }

    for (const char* name : {
            "ssm_conv1d.weight", "ssm_dt.bias",
            "ssm_a", "ssm_norm.weight"}) {
        const auto* t = get(name);
        if (!t || t->ggml_type != 0 ||
            t->layout != q38::TensorLayout::GgufNative) {
            why = std::string("unsupported ") + name;
            return false;
        }
    }

    why = "ready";
    return true;
}

int main(int argc, char** argv) {
    std::string model;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (arg == "--help") {
            std::cout
                << "usage: q38-layer-topology --model MODEL.q38pack\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (model.empty()) {
        std::cerr << "q38-layer-topology: --model is required\n";
        return 2;
    }

    try {
        q38::Runtime runtime;
        runtime.load(model, false);
        const auto& pack = runtime.pack();

        std::vector<std::uint32_t> recurrent;
        std::vector<std::uint32_t> full;
        std::vector<std::uint32_t> missing;

        std::cout << "Q38RT Qwen3.8 layer topology\n";
        std::cout << "q38pack_version: "
                  << pack.header().version << "\n";

        for (std::uint32_t layer = 0; layer < 64; ++layer) {
            const auto kind =
                q38::detect_qwen35_layer_kind(pack, layer);
            std::cout << "layer[" << layer << "]: "
                      << q38::qwen35_layer_kind_name(kind);

            if (kind == q38::Qwen35LayerKind::Recurrent) {
                recurrent.push_back(layer);
                std::string why;
                const bool ready =
                    recurrent_ready(pack, layer, why);
                std::cout << " runtime="
                          << (ready ? "READY" : "BLOCKED")
                          << " (" << why << ")";
            } else if (kind ==
                       q38::Qwen35LayerKind::FullAttention) {
                full.push_back(layer);
            } else {
                missing.push_back(layer);
            }
            std::cout << "\n";
        }

        std::cout << "\nsummary:\n";
        std::cout << "  recurrent_layers: "
                  << recurrent.size() << "\n";
        std::cout << "  full_attention_layers: "
                  << full.size() << "\n";
        std::cout << "  missing_layers: "
                  << missing.size() << "\n";

        std::cout << "  recurrent_indices:";
        for (auto x : recurrent) std::cout << " " << x;
        std::cout << "\n";

        std::cout << "  full_attention_indices:";
        for (auto x : full) std::cout << " " << x;
        std::cout << "\n";

        if (!full.empty()) {
            const auto first = full.front();
            const std::string ref_sig =
                tensor_signature(pack, first);
            bool uniform = true;
            for (auto layer : full) {
                if (tensor_signature(pack, layer) != ref_sig) {
                    uniform = false;
                    break;
                }
            }

            std::cout << "\nfull_attention_signature:\n";
            std::cout << "  reference_layer: " << first << "\n";
            std::cout << "  uniform_across_full_layers: "
                      << (uniform ? "yes" : "no") << "\n";

            const std::string prefix =
                "blk." + std::to_string(first) + ".";
            for (const auto& t : pack.tensors()) {
                if (t.name.rfind(prefix, 0) != 0) continue;
                std::cout << "  "
                          << t.name.substr(prefix.size())
                          << " type=" << type_name(t.ggml_type)
                          << "(" << t.ggml_type << ")"
                          << " layout=" << layout_name(t.layout)
                          << " dims=" << dims_string(t)
                          << " bytes=" << t.stored_bytes
                          << "\n";
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-layer-topology: "
                  << e.what() << "\n";
        return 1;
    }
}
