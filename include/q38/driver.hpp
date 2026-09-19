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
    [[nodiscard]] int sm_major() const noexcept { return sm_major_; }
    [[nodiscard]] int sm_minor() const noexcept { return sm_minor_; }
    [[nodiscard]] bool is_sm86() const noexcept { return sm_major_ == 8 && sm_minor_ == 6; }
    [[nodiscard]] std::size_t vmm_granularity() const noexcept { return vmm_granularity_; }
    [[nodiscard]] bool vmm_available() const noexcept { return vmm_available_; }

    bool probe_vmm(std::size_t bytes, std::string* error = nullptr);
    bool run_sm86_smoke(std::string* error = nullptr);
    bool run_rmsnorm_smoke(std::string* error = nullptr, double* max_abs_error = nullptr, double* max_rel_error = nullptr);
    bool run_q4k_dequant_smoke(const std::byte* block, std::string* error = nullptr, double* max_abs_error = nullptr, double* max_rel_error = nullptr);
    bool run_q5k_dequant_smoke(const std::byte* block, std::string* error = nullptr, double* max_abs_error = nullptr, double* max_rel_error = nullptr);
    bool run_q5k_gemv_smoke(const std::byte* matrix, std::uint32_t cols, std::uint32_t rows, std::string* error = nullptr, double* max_abs_error = nullptr, double* max_rel_error = nullptr, double* milliseconds = nullptr, double* bandwidth_gbps = nullptr);
    bool run_q5k_q8k_gemv_smoke(const std::byte* matrix, std::uint32_t cols, std::uint32_t rows, std::string* error = nullptr, double* max_abs_error = nullptr, double* max_rel_error = nullptr, double* milliseconds = nullptr, double* bandwidth_gbps = nullptr);
    bool run_q5k_sm86_gemv_smoke(const std::byte* repacked_matrix, std::size_t qh_offset, std::size_t qs_offset, std::uint32_t cols, std::uint32_t rows, std::string* error = nullptr, double* max_abs_error = nullptr, double* max_rel_error = nullptr, double* milliseconds = nullptr, double* original_equiv_gbps = nullptr, double* physical_gbps = nullptr);

private:
    void* handle_{nullptr};
    void* context_{nullptr};
    bool initialized_{false};
    int driver_version_{0};
    int device_ordinal_{0};
    int sm_major_{0};
    int sm_minor_{0};
    std::string device_name_;
    std::size_t total_memory_{0};
    std::size_t vmm_granularity_{0};
    bool vmm_available_{false};
};

} // namespace q38
