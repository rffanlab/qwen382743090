#include "q38/driver.hpp"

#include <iomanip>
#include <iostream>
#include <string>

int main() {
    try {
        q38::NvidiaDriver driver;
        driver.open();

        std::cout << "Q38RT RMSNorm smoke\n";
        std::cout << "device: " << driver.device_name() << "\n";
        std::cout << "compute capability: sm_" << driver.sm_major() << driver.sm_minor() << "\n";

        std::string error;
        double max_abs = 0.0;
        double max_rel = 0.0;
        if (!driver.run_rmsnorm_smoke(&error, &max_abs, &max_rel)) {
            std::cerr << "rmsnorm: FAILED: " << error << "\n";
            return 2;
        }

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "max_abs_error: " << max_abs << "\n";
        std::cout << "max_rel_error: " << max_rel << "\n";
        std::cout << "rmsnorm: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-rmsnorm-smoke: " << e.what() << "\n";
        return 1;
    }
}
