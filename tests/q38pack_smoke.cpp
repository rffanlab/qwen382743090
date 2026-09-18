#include "q38/q38pack.hpp"

#include <iostream>

int main() {
    static_assert(q38::kPackHeaderBytes == 256);
    static_assert(q38::kTensorEntryBytes == 256);
    if (q38::kPackMagic[0] != 'Q' || q38::kPackVersion != 1) return 1;
    std::cout << "q38pack ABI constants OK\n";
    return 0;
}
