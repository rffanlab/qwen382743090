#include "q38/placement.hpp"
#include "q38/q38pack.hpp"

#include <array>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

static double gib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / 1024.0 / 1024.0 / 1024.0;
}

static const char* ggml_type_name(std::uint32_t type) {
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

static const char* role_name(q38::TensorRole role) {
    switch (role) {
        case q38::TensorRole::TokenEmbedding: return "embedding";
        case q38::TensorRole::OutputNorm: return "output_norm";
        case q38::TensorRole::Output: return "output";
        case q38::TensorRole::Attention: return "attention";
        case q38::TensorRole::DeltaNet: return "deltanet";
        case q38::TensorRole::FeedForward: return "ffn";
        case q38::TensorRole::Norm: return "norm";
        case q38::TensorRole::Mtp: return "mtp";
        default: return "unknown";
    }
}

int main(int argc, char** argv) {
    std::string model;
    std::size_t page_bytes = 2ull * 1024 * 1024;
    bool list = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model = argv[++i];
        else if (arg == "--page-bytes" && i + 1 < argc) page_bytes = std::stoull(argv[++i]);
        else if (arg == "--list") list = true;
        else if (arg == "--help") {
            std::cout << "usage: q38-plan --model MODEL.q38pack [--page-bytes 2097152] [--list]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (model.empty()) {
        std::cerr << "q38-plan: --model is required\n";
        return 2;
    }

    try {
        q38::PackFile pack;
        pack.open(model);
        const auto plan = q38::PlacementPlan::build(pack, page_bytes);
        const auto& stats = plan.stats();

        std::array<std::uint64_t, 9> by_role{};
        std::array<std::uint64_t, 256> type_bytes{};
        std::array<std::uint64_t, 256> type_count{};
        const auto& source_tensors = pack.tensors();
        for (std::size_t i = 0; i < plan.tensors().size(); ++i) {
            const auto& t = plan.tensors()[i];
            const auto idx = static_cast<std::size_t>(t.role);
            if (idx < by_role.size()) by_role[idx] += t.payload_bytes;

            const auto type = source_tensors[i].ggml_type;
            if (type < type_bytes.size()) {
                type_bytes[type] += t.payload_bytes;
                type_count[type] += 1;
            }
        }

        std::cout << "Q38 GPU placement plan\n";
        std::cout << "GPU page: " << plan.gpu_page_bytes() << " bytes\n";
        std::cout << "tensors: " << plan.tensors().size() << "\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "payload: " << gib(stats.tensor_payload_bytes) << " GiB\n";
        std::cout << "reserved VA: " << gib(stats.reserved_va_bytes) << " GiB\n";
        std::cout << "page padding: " << gib(stats.page_padding_bytes) << " GiB\n";
        std::cout << "pages: " << stats.page_count << "\n";
        for (std::size_t i = 0; i < by_role.size(); ++i) {
            if (!by_role[i]) continue;
            std::cout << "role[" << role_name(static_cast<q38::TensorRole>(i)) << "]: "
                      << gib(by_role[i]) << " GiB\n";
        }

        std::cout << "tensor types:\n";
        for (std::size_t i = 0; i < type_bytes.size(); ++i) {
            if (!type_count[i]) continue;
            std::cout << "  type[" << i << "/" << ggml_type_name(static_cast<std::uint32_t>(i)) << "]: "
                      << type_count[i] << " tensors, " << gib(type_bytes[i]) << " GiB\n";
        }

        if (list) {
            for (const auto& t : plan.tensors()) {
                std::cout << t.name
                          << " role=" << role_name(t.role)
                          << " va=0x" << std::hex << t.va_offset << std::dec
                          << " page_off=" << t.page_offset
                          << " bytes=" << t.payload_bytes << "\n";
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-plan: " << e.what() << "\n";
        return 1;
    }
}
