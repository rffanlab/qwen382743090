#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace q38 {

class NvidiaDriver {
public:
    NvidiaDriver() = default;
    ~NvidiaDriver();
    NvidiaDriver(const NvidiaDriver&) = delete;
    NvidiaDriver& operator=(const NvidiaDriver&) = delete;

    void open();
    void close() noexcept;

    [[nodiscard]] bool available() const noexcept { return handle_ != nullptr && initialized_; }
    [[nodiscard]] int driver_version() const noexcept { return driver_version_; }
    [[nodiscard]] int device_ordinal() const noexcept { return device_ordinal_; }
    [[nodiscard]] const std::string& device_name() const noexcept { return device_name_; }
    [[nodiscard]] std::size_t total_memory() const noexcept { return total_memory_; }

private:
    void* handle_{nullptr};
    bool initialized_{false};
    int driver_version_{0};
    int device_ordinal_{0};
    std::string device_name_;
    std::size_t total_memory_{0};
};

} // namespace q38
