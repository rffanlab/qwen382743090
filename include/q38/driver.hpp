#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace q38 {

struct ProjectionErrorStats {
    double max_abs{};
    double max_rel{};
};

struct RecurrentPrepSmokeStats {
    double conv_max_abs{};
    double q_max_abs{};
    double k_max_abs{};
    double beta_max_abs{};
    double gate_max_abs{};
    double conv_state_max_abs{};
    double conv_silu_ms{};
    double qk_norm_ms{};
    double beta_gate_ms{};
    double chain_ms{};
};

struct GdnArSmokeStats {
    double output_max_abs{};
    double output_max_rel{};
    double state_max_abs{};
    double state_max_rel{};
    double kernel_ms{};
    double state_bandwidth_gbps{};
};

struct Layer0FullStats {
    double ffn_norm_max_abs{};
    double ffn_gate_up_max_abs{};
    double ffn_down_max_abs{};
    double layer_output_max_abs{};
    double attention_ms{};
    double post_norm_ms{};
    double ffn_gate_ms{};
    double ffn_up_ms{};
    double ffn_pointwise_ms{};
    double ffn_down_ms{};
    double ffn_ms{};
    double sum_stage_ms{};
    double chain_ms{};
};

struct Layer0RecurrentAttentionStats {
    double gated_norm_max_abs{};
    double ssm_out_max_abs{};
    double residual_max_abs{};
    double front_ms{};
    double gated_norm_ms{};
    double ssm_out_ms{};
    double tail_ms{};
    double sum_stage_ms{};
    double chain_ms{};
};

struct Layer0RecurrentFrontStats {
    double conv_max_abs{};
    double q_max_abs{};
    double k_max_abs{};
    double beta_max_abs{};
    double gate_max_abs{};
    double conv_state_max_abs{};
    double gdn_output_max_abs{};
    double gdn_output_max_rel{};
    double gdn_state_max_abs{};
    double gdn_state_max_rel{};
    double projection_ms{};
    double conv_prep_ms{};
    double gdn_ms{};
    double sum_stage_ms{};
    double chain_ms{};
};

struct Layer0ProjectionPackStats {
    ProjectionErrorStats qkv;
    ProjectionErrorStats gate;
    ProjectionErrorStats beta;
    ProjectionErrorStats alpha;
    double rmsnorm_ms{};
    double qkv_ms{};
    double gate_ms{};
    double beta_ms{};
    double alpha_ms{};
    double chain_ms{};
};

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
    bool run_q4k_gemv_smoke(const std::byte* matrix, std::uint32_t cols, std::uint32_t rows, std::string* error = nullptr, double* max_abs_error = nullptr, double* max_rel_error = nullptr, double* milliseconds = nullptr, double* bandwidth_gbps = nullptr);
    bool run_q5k_dequant_smoke(const std::byte* block, std::string* error = nullptr, double* max_abs_error = nullptr, double* max_rel_error = nullptr);
    bool run_iq4xs_dequant_smoke(const std::byte* block, std::string* error = nullptr, double* max_abs_error = nullptr, double* max_rel_error = nullptr);
    bool run_iq4xs_gemv_smoke(const std::byte* matrix, std::uint32_t cols, std::uint32_t rows, std::string* error = nullptr, double* max_abs_error = nullptr, double* max_rel_error = nullptr, double* milliseconds = nullptr, double* bandwidth_gbps = nullptr);
    bool run_iq4xs_sm86_gemv_smoke(const std::byte* repacked_matrix, std::size_t scale_offset, std::size_t qs_offset, std::uint32_t cols, std::uint32_t rows, std::string* error = nullptr, double* max_abs_error = nullptr, double* max_rel_error = nullptr, double* milliseconds = nullptr, double* original_equiv_gbps = nullptr, double* physical_gbps = nullptr);
    bool run_iq4xs_q8_1_gemv_smoke(const std::byte* matrix, std::uint32_t cols, std::uint32_t rows, std::string* error = nullptr, double* max_abs_error = nullptr, double* max_rel_error = nullptr, double* milliseconds = nullptr, double* bandwidth_gbps = nullptr);
    bool run_q5k_gemv_smoke(const std::byte* matrix, std::uint32_t cols, std::uint32_t rows, std::string* error = nullptr, double* max_abs_error = nullptr, double* max_rel_error = nullptr, double* milliseconds = nullptr, double* bandwidth_gbps = nullptr);
    bool run_q5k_q8k_gemv_smoke(const std::byte* matrix, std::uint32_t cols, std::uint32_t rows, std::string* error = nullptr, double* max_abs_error = nullptr, double* max_rel_error = nullptr, double* milliseconds = nullptr, double* bandwidth_gbps = nullptr);
    bool run_q5k_sm86_gemv_smoke(const std::byte* repacked_matrix, std::size_t qh_offset, std::size_t qs_offset, std::uint32_t cols, std::uint32_t rows, std::string* error = nullptr, double* max_abs_error = nullptr, double* max_rel_error = nullptr, double* milliseconds = nullptr, double* original_equiv_gbps = nullptr, double* physical_gbps = nullptr);
    bool run_qwen35_layer0_full(
        const float* norm_weight,
        const std::byte* qkv_matrix,
        std::size_t qkv_qh_offset,
        std::size_t qkv_qs_offset,
        const std::byte* z_matrix,
        std::size_t z_qh_offset,
        std::size_t z_qs_offset,
        const std::byte* beta_matrix,
        const std::byte* alpha_matrix,
        const float* conv_weight,
        const float* dt_bias,
        const float* ssm_a,
        const float* ssm_norm_weight,
        const std::byte* ssm_out_matrix,
        std::size_t ssm_out_qh_offset,
        std::size_t ssm_out_qs_offset,
        const float* post_norm_weight,
        const std::byte* ffn_gate_matrix,
        const std::byte* ffn_up_matrix,
        std::size_t ffn_up_qh_offset,
        std::size_t ffn_up_qs_offset,
        const std::byte* ffn_down_matrix,
        std::size_t ffn_down_qh_offset,
        std::size_t ffn_down_qs_offset,
        float rms_eps,
        Layer0FullStats* stats = nullptr,
        std::string* error = nullptr);

    bool run_qwen35_layer0_recurrent_attention(
        const float* norm_weight,
        const std::byte* qkv_matrix,
        std::size_t qkv_qh_offset,
        std::size_t qkv_qs_offset,
        const std::byte* z_matrix,
        std::size_t z_qh_offset,
        std::size_t z_qs_offset,
        const std::byte* beta_matrix,
        const std::byte* alpha_matrix,
        const float* conv_weight,
        const float* dt_bias,
        const float* ssm_a,
        const float* ssm_norm_weight,
        const std::byte* ssm_out_matrix,
        std::size_t ssm_out_qh_offset,
        std::size_t ssm_out_qs_offset,
        float rms_eps,
        Layer0RecurrentAttentionStats* stats = nullptr,
        std::string* error = nullptr);

    bool run_qwen35_layer0_recurrent_front(
        const float* norm_weight,
        const std::byte* qkv_matrix,
        std::size_t qkv_qh_offset,
        std::size_t qkv_qs_offset,
        const std::byte* z_matrix,
        std::size_t z_qh_offset,
        std::size_t z_qs_offset,
        const std::byte* beta_matrix,
        const std::byte* alpha_matrix,
        const float* conv_weight,
        const float* dt_bias,
        const float* ssm_a,
        float rms_eps,
        Layer0RecurrentFrontStats* stats = nullptr,
        std::string* error = nullptr);

    bool run_recurrent_prep_smoke(
        const float* conv_weight,
        const float* dt_bias,
        const float* ssm_a,
        RecurrentPrepSmokeStats* stats = nullptr,
        std::string* error = nullptr);

    bool run_gdn_ar_smoke(
        std::uint32_t state_dim = 128,
        std::uint32_t qk_heads = 16,
        std::uint32_t value_heads = 48,
        GdnArSmokeStats* stats = nullptr,
        std::string* error = nullptr);

    bool run_qwen35_layer0_projection_pack(
        const float* norm_weight,
        const std::byte* qkv_matrix,
        std::size_t qkv_qh_offset,
        std::size_t qkv_qs_offset,
        std::uint32_t qkv_rows,
        const std::byte* gate_matrix,
        std::size_t gate_qh_offset,
        std::size_t gate_qs_offset,
        std::uint32_t gate_rows,
        const std::byte* beta_matrix,
        const std::byte* alpha_matrix,
        std::uint32_t cols,
        std::uint32_t small_rows,
        float rms_eps,
        Layer0ProjectionPackStats* stats = nullptr,
        std::string* error = nullptr);

    bool run_qwen35_layer0_gate_smoke(
        const float* norm_weight,
        const std::byte* repacked_matrix,
        std::size_t qh_offset,
        std::size_t qs_offset,
        std::uint32_t cols,
        std::uint32_t rows,
        float rms_eps,
        std::string* error = nullptr,
        double* max_abs_error = nullptr,
        double* max_rel_error = nullptr,
        double* rmsnorm_ms = nullptr,
        double* projection_ms = nullptr,
        double* chain_ms = nullptr,
        double* projection_original_equiv_gbps = nullptr);

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
