#include "q38/placement.hpp"

#include <cstdint>
#include <iostream>

int main() {
    // PlacementPlan's algorithm is exercised with real Q38PACK files by the
    // Python conversion tests and q38-plan CLI. Keep this host smoke tiny so CI
    // does not need a generated fixture.
    static_assert(static_cast<std::uint32_t>(q38::TensorRole::Mtp) == 8);
    std::cout << "placement ABI constants OK\n";
    return 0;
}
