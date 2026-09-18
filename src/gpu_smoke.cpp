#include "q38/driver.hpp"

#include <iomanip>
#include <iostream>
#include <string>

static double gib(std::size_t bytes) {
    return static_cast<double>(bytes) / 1024.0 / 1024.0 / 1024.0;
}

int main() {
    try {
        q38::NvidiaDriver driver;
        driver.open();

        std::cout << "Q38RT GPU smoke\n";
        std::cout << "device: " << driver.device_name() << "\n";
        std::cout << "driver version: " << driver.driver_version() << "\n";
        std::cout << "compute capability: sm_" << driver.sm_major() << driver.sm_minor() << "\n";
        std::cout << std::fixed << std::setprecision(2)
                  << "VRAM: " << gib(driver.total_memory()) << " GiB\n";
        std::cout << "VMM: " << (driver.vmm_available() ? "available" : "not available");
        if (driver.vmm_available()) std::cout << " (granularity " << driver.vmm_granularity() << " bytes)";
        std::cout << "\n";

        std::string error;
        if (!driver.run_sm86_smoke(&error)) {
            std::cerr << "smoke: FAILED: " << error << "\n";
            return 2;
        }

        std::cout << "smoke: PASS (Driver API + VMM + PTX kernel)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "q38-gpu-smoke: " << e.what() << "\n";
        return 1;
    }
}
