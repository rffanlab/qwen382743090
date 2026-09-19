#include "q38/driver.hpp"

#include <iomanip>
#include <iostream>
#include <string>

int main() {
    try {
        q38::NvidiaDriver driver;
        driver.open();

        constexpr std::uint32_t state_dim = 128;
        constexpr std::uint32_t qk_heads = 16;
        constexpr std::uint32_t value_heads = 48;

        std::cout << "Q38RT Qwen3.8 fused Gated DeltaNet decode core\n";
        std::cout << "device: " << driver.device_name() << "\n";
        std::cout << "state_dim: " << state_dim << "\n";
        std::cout << "qk_heads: " << qk_heads << "\n";
        std::cout << "value_heads: " << value_heads << "\n";
        std::cout << "qk_broadcast: value_head_mod_qk_heads\n";
        std::cout << "state_layout: [H=48][col=128][row=128] transposed-column-major\n";
        std::cout << "kernel_mapping: grid[head,32] x block[128]=4warps\n";

        q38::GdnArSmokeStats stats{};
        std::string error;
        if (!driver.run_gdn_ar_smoke(
                state_dim,
                qk_heads,
                value_heads,
                &stats,
                &error)) {
            std::cerr << "gdn_ar: FAILED: " << error << "\n";
            return 2;
        }

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "output_max_abs_error: " << stats.output_max_abs << "\n";
        std::cout << "output_max_rel_error: " << stats.output_max_rel << "\n";
        std::cout << "state_max_abs_error: " << stats.state_max_abs << "\n";
        std::cout << "state_max_rel_error: " << stats.state_max_rel << "\n";

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "kernel_ms: " << stats.kernel_ms << "\n";
        std::cout << "state_read_write_bandwidth_GBps: "
                  << stats.state_bandwidth_gbps << "\n";
        std::cout << "n_tokens: 1\n";
        std::cout << "gdn_mode: GDA_scalar_gate_per_value_head\n";
        std::cout << "gdn_ar: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-gdn-ar-smoke: " << e.what() << "\n";
        return 1;
    }
}
