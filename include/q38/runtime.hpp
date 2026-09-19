#pragma once

#include "q38/driver.hpp"
#include "q38/q38pack.hpp"

#include <filesystem>
#include <string>

namespace q38 {

enum class ProjectionKernelKind {
    Unsupported = 0,
    F32Direct,
    Q4KNative,
    Q5KSm86Vectorized,
    IQ4XSPrmt,
};

const char* projection_kernel_name(ProjectionKernelKind kind) noexcept;
ProjectionKernelKind select_projection_kernel(const TensorRecord& tensor) noexcept;

struct RuntimeInfo {
    std::string model_path;
    std::uint32_t tensor_count{};
    std::uint64_t packed_bytes{};
    bool driver_available{};
    int driver_version{};
    std::string device_name;
    std::size_t device_memory{};
    int sm_major{};
    int sm_minor{};
    bool vmm_available{};
    std::size_t vmm_granularity{};
    bool native_decode_ready{};
};

class Runtime {
public:
    void load(const std::filesystem::path& model_path, bool require_gpu = true);
    [[nodiscard]] RuntimeInfo info() const;
    [[nodiscard]] const PackFile& pack() const noexcept { return pack_; }
    bool gpu_smoke(std::string* error = nullptr) { return driver_.run_sm86_smoke(error); }
    bool run_layer0_projection_pack(
        float rms_eps = 1.0e-6f,
        Layer0ProjectionPackStats* stats = nullptr,
        std::string* error = nullptr);

private:
    std::filesystem::path model_path_;
    PackFile pack_;
    NvidiaDriver driver_;
    bool native_decode_ready_{false};
};

} // namespace q38
