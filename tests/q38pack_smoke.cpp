#include "q38/q38pack.hpp"

#include <iostream>

int main() {
    static_assert(q38::kPackHeaderBytes == 256);
    static_assert(q38::kTensorEntryBytes == 256);
    static_assert(q38::kPackMinVersion == 1);
    static_assert(q38::kPackVersion == 2);
    static_assert(static_cast<std::uint32_t>(q38::TensorLayout::Sm86Q5KSoA) == 1);
    if (q38::kPackMagic[0] != 'Q') return 1;
    std::cout << "q38pack v1/v2 ABI constants OK\n";
    return 0;
}
