#include "q38/runtime.hpp"

#include <stdexcept>

namespace q38 {

const char* projection_kernel_name(ProjectionKernelKind kind) noexcept {
    switch (kind) {
        case ProjectionKernelKind::F32Direct: return "F32_DIRECT";
        case ProjectionKernelKind::Q4KNative: return "Q4K_NATIVE";
        case ProjectionKernelKind::Q5KSm86Vectorized: return "Q5K_SM86_VEC";
        case ProjectionKernelKind::IQ4XSPrmt: return "IQ4XS_PRMT";
        default: return "UNSUPPORTED";
    }
}

ProjectionKernelKind select_projection_kernel(const TensorRecord& tensor) noexcept {
    if (tensor.ggml_type == 0 &&
        tensor.layout == TensorLayout::GgufNative) {
        return ProjectionKernelKind::F32Direct;
    }
    if (tensor.ggml_type == 12 &&
        tensor.layout == TensorLayout::GgufNative) {
        return ProjectionKernelKind::Q4KNative;
    }
    if (tensor.ggml_type == 13 &&
        tensor.layout == TensorLayout::Sm86Q5KSoA) {
        return ProjectionKernelKind::Q5KSm86Vectorized;
    }
    if (tensor.ggml_type == 23 &&
        tensor.layout == TensorLayout::GgufNative) {
        return ProjectionKernelKind::IQ4XSPrmt;
    }
    return ProjectionKernelKind::Unsupported;
}

void Runtime::load(const std::filesystem::path& model_path, bool require_gpu) {
    model_path_ = model_path;
    pack_.open(model_path);
    if (require_gpu) driver_.open();
    else {
        try { driver_.open(); } catch (...) {}
    }

    // Native graph execution is promoted only after the real model path has
    // layer-by-layer correctness coverage.
    native_decode_ready_ = false;
}

bool Runtime::run_layer0_projection_pack(
    float rms_eps,
    Layer0ProjectionPackStats* stats,
    std::string* error) {
    try {
        if (!driver_.available()) {
            throw std::runtime_error("layer0 projection pack requires an initialized GPU driver");
        }

        auto require = [&](const char* name) -> const TensorRecord& {
            const auto* tensor = pack_.find_tensor(name);
            if (!tensor) {
                throw std::runtime_error(std::string("missing tensor: ") + name);
            }
            return *tensor;
        };

        const auto& norm = require("blk.0.attn_norm.weight");
        const auto& qkv = require("blk.0.attn_qkv.weight");
        const auto& gate = require("blk.0.attn_gate.weight");
        const auto& beta = require("blk.0.ssm_beta.weight");
        const auto& alpha = require("blk.0.ssm_alpha.weight");

        if (select_projection_kernel(norm) != ProjectionKernelKind::F32Direct) {
            throw std::runtime_error("blk.0.attn_norm.weight must use F32_DIRECT");
        }
        if (select_projection_kernel(qkv) != ProjectionKernelKind::Q5KSm86Vectorized) {
            throw std::runtime_error("blk.0.attn_qkv.weight must use Q5K_SM86_VEC");
        }
        if (select_projection_kernel(gate) != ProjectionKernelKind::Q5KSm86Vectorized) {
            throw std::runtime_error("blk.0.attn_gate.weight must use Q5K_SM86_VEC");
        }
        if (select_projection_kernel(beta) != ProjectionKernelKind::Q4KNative ||
            select_projection_kernel(alpha) != ProjectionKernelKind::Q4KNative) {
            throw std::runtime_error("layer0 alpha/beta must use Q4K_NATIVE");
        }

        if (norm.ndim != 1 || qkv.ndim < 2 || gate.ndim < 2 ||
            beta.ndim < 2 || alpha.ndim < 2) {
            throw std::runtime_error("layer0 projection pack has unexpected tensor rank");
        }

        const auto cols = static_cast<std::uint32_t>(norm.dims[0]);
        if (cols == 0 ||
            qkv.dims[0] != cols ||
            gate.dims[0] != cols ||
            beta.dims[0] != cols ||
            alpha.dims[0] != cols) {
            throw std::runtime_error("layer0 projection pack input dimensions do not match");
        }
        if (beta.dims[1] != alpha.dims[1]) {
            throw std::runtime_error("layer0 alpha/beta output dimensions do not match");
        }

        const auto qkv_qh_offset =
            static_cast<std::size_t>(qkv.aux0_offset - qkv.data_offset);
        const auto qkv_qs_offset =
            static_cast<std::size_t>(qkv.aux1_offset - qkv.data_offset);
        const auto gate_qh_offset =
            static_cast<std::size_t>(gate.aux0_offset - gate.data_offset);
        const auto gate_qs_offset =
            static_cast<std::size_t>(gate.aux1_offset - gate.data_offset);

        return driver_.run_qwen35_layer0_projection_pack(
            reinterpret_cast<const float*>(pack_.tensor_data(norm)),
            pack_.tensor_data(qkv),
            qkv_qh_offset,
            qkv_qs_offset,
            static_cast<std::uint32_t>(qkv.dims[1]),
            pack_.tensor_data(gate),
            gate_qh_offset,
            gate_qs_offset,
            static_cast<std::uint32_t>(gate.dims[1]),
            pack_.tensor_data(beta),
            pack_.tensor_data(alpha),
            cols,
            static_cast<std::uint32_t>(beta.dims[1]),
            rms_eps,
            stats,
            error);
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}


bool Runtime::run_layer0_recurrent_front(
    float rms_eps,
    Layer0RecurrentFrontStats* stats,
    std::string* error) {
    try {
        if (!driver_.available()) {
            throw std::runtime_error(
                "layer0 recurrent front requires an initialized GPU driver");
        }

        auto require = [&](const char* name) -> const TensorRecord& {
            const auto* tensor = pack_.find_tensor(name);
            if (!tensor) {
                throw std::runtime_error(
                    std::string("missing tensor: ") + name);
            }
            return *tensor;
        };

        const auto& norm  = require("blk.0.attn_norm.weight");
        const auto& qkv   = require("blk.0.attn_qkv.weight");
        const auto& z     = require("blk.0.attn_gate.weight");
        const auto& beta  = require("blk.0.ssm_beta.weight");
        const auto& alpha = require("blk.0.ssm_alpha.weight");
        const auto& conv  = require("blk.0.ssm_conv1d.weight");
        const auto& dt    = require("blk.0.ssm_dt.bias");
        const auto& a     = require("blk.0.ssm_a");

        if (select_projection_kernel(norm) !=
                ProjectionKernelKind::F32Direct ||
            select_projection_kernel(qkv) !=
                ProjectionKernelKind::Q5KSm86Vectorized ||
            select_projection_kernel(z) !=
                ProjectionKernelKind::Q5KSm86Vectorized ||
            select_projection_kernel(beta) !=
                ProjectionKernelKind::Q4KNative ||
            select_projection_kernel(alpha) !=
                ProjectionKernelKind::Q4KNative) {
            throw std::runtime_error(
                "layer0 recurrent-front projection kernel dispatch mismatch");
        }

        if (conv.ggml_type != 0 ||
            dt.ggml_type != 0 ||
            a.ggml_type != 0 ||
            conv.layout != TensorLayout::GgufNative ||
            dt.layout != TensorLayout::GgufNative ||
            a.layout != TensorLayout::GgufNative) {
            throw std::runtime_error(
                "layer0 recurrent-front conv/dt/a must be F32 GGUF_NATIVE");
        }

        if (norm.ndim != 1 || norm.dims[0] != 5120 ||
            qkv.ndim < 2 || qkv.dims[0] != 5120 || qkv.dims[1] != 10240 ||
            z.ndim < 2 || z.dims[0] != 5120 || z.dims[1] != 6144 ||
            beta.ndim < 2 || beta.dims[0] != 5120 || beta.dims[1] != 48 ||
            alpha.ndim < 2 || alpha.dims[0] != 5120 || alpha.dims[1] != 48 ||
            conv.ndim < 2 || conv.dims[0] != 4 || conv.dims[1] != 10240 ||
            dt.dims[0] != 48 || a.dims[0] != 48) {
            throw std::runtime_error(
                "layer0 recurrent-front tensor shapes do not match Qwen3.8-27B");
        }

        const auto qkv_qh_offset =
            static_cast<std::size_t>(
                qkv.aux0_offset - qkv.data_offset);
        const auto qkv_qs_offset =
            static_cast<std::size_t>(
                qkv.aux1_offset - qkv.data_offset);
        const auto z_qh_offset =
            static_cast<std::size_t>(
                z.aux0_offset - z.data_offset);
        const auto z_qs_offset =
            static_cast<std::size_t>(
                z.aux1_offset - z.data_offset);

        return driver_.run_qwen35_layer0_recurrent_front(
            reinterpret_cast<const float*>(pack_.tensor_data(norm)),
            pack_.tensor_data(qkv),
            qkv_qh_offset,
            qkv_qs_offset,
            pack_.tensor_data(z),
            z_qh_offset,
            z_qs_offset,
            pack_.tensor_data(beta),
            pack_.tensor_data(alpha),
            reinterpret_cast<const float*>(pack_.tensor_data(conv)),
            reinterpret_cast<const float*>(pack_.tensor_data(dt)),
            reinterpret_cast<const float*>(pack_.tensor_data(a)),
            rms_eps,
            stats,
            error);
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}


bool Runtime::run_layer0_recurrent_attention(
    float rms_eps,
    Layer0RecurrentAttentionStats* stats,
    std::string* error) {
    try {
        if (!driver_.available()) {
            throw std::runtime_error(
                "layer0 recurrent attention requires an initialized GPU driver");
        }

        auto require = [&](const char* name) -> const TensorRecord& {
            const auto* tensor = pack_.find_tensor(name);
            if (!tensor) {
                throw std::runtime_error(
                    std::string("missing tensor: ") + name);
            }
            return *tensor;
        };

        const auto& norm      = require("blk.0.attn_norm.weight");
        const auto& qkv       = require("blk.0.attn_qkv.weight");
        const auto& z         = require("blk.0.attn_gate.weight");
        const auto& beta      = require("blk.0.ssm_beta.weight");
        const auto& alpha     = require("blk.0.ssm_alpha.weight");
        const auto& conv      = require("blk.0.ssm_conv1d.weight");
        const auto& dt        = require("blk.0.ssm_dt.bias");
        const auto& a         = require("blk.0.ssm_a");
        const auto& ssm_norm  = require("blk.0.ssm_norm.weight");
        const auto& ssm_out   = require("blk.0.ssm_out.weight");

        if (select_projection_kernel(norm) != ProjectionKernelKind::F32Direct ||
            select_projection_kernel(qkv) != ProjectionKernelKind::Q5KSm86Vectorized ||
            select_projection_kernel(z) != ProjectionKernelKind::Q5KSm86Vectorized ||
            select_projection_kernel(beta) != ProjectionKernelKind::Q4KNative ||
            select_projection_kernel(alpha) != ProjectionKernelKind::Q4KNative ||
            select_projection_kernel(ssm_out) != ProjectionKernelKind::Q5KSm86Vectorized) {
            throw std::runtime_error(
                "layer0 recurrent-attention projection dispatch mismatch");
        }

        if (conv.ggml_type != 0 || dt.ggml_type != 0 || a.ggml_type != 0 ||
            ssm_norm.ggml_type != 0 ||
            conv.layout != TensorLayout::GgufNative ||
            dt.layout != TensorLayout::GgufNative ||
            a.layout != TensorLayout::GgufNative ||
            ssm_norm.layout != TensorLayout::GgufNative) {
            throw std::runtime_error(
                "layer0 recurrent-attention F32 tensor layout mismatch");
        }

        if (norm.ndim != 1 || norm.dims[0] != 5120 ||
            qkv.ndim < 2 || qkv.dims[0] != 5120 || qkv.dims[1] != 10240 ||
            z.ndim < 2 || z.dims[0] != 5120 || z.dims[1] != 6144 ||
            beta.ndim < 2 || beta.dims[0] != 5120 || beta.dims[1] != 48 ||
            alpha.ndim < 2 || alpha.dims[0] != 5120 || alpha.dims[1] != 48 ||
            conv.ndim < 2 || conv.dims[0] != 4 || conv.dims[1] != 10240 ||
            dt.dims[0] != 48 || a.dims[0] != 48 ||
            ssm_norm.dims[0] != 128 ||
            ssm_out.ndim < 2 || ssm_out.dims[0] != 6144 || ssm_out.dims[1] != 5120) {
            throw std::runtime_error(
                "layer0 recurrent-attention tensor shapes do not match Qwen3.8-27B");
        }

        const auto qkv_qh_offset =
            static_cast<std::size_t>(qkv.aux0_offset - qkv.data_offset);
        const auto qkv_qs_offset =
            static_cast<std::size_t>(qkv.aux1_offset - qkv.data_offset);
        const auto z_qh_offset =
            static_cast<std::size_t>(z.aux0_offset - z.data_offset);
        const auto z_qs_offset =
            static_cast<std::size_t>(z.aux1_offset - z.data_offset);
        const auto ssm_out_qh_offset =
            static_cast<std::size_t>(ssm_out.aux0_offset - ssm_out.data_offset);
        const auto ssm_out_qs_offset =
            static_cast<std::size_t>(ssm_out.aux1_offset - ssm_out.data_offset);

        return driver_.run_qwen35_layer0_recurrent_attention(
            reinterpret_cast<const float*>(pack_.tensor_data(norm)),
            pack_.tensor_data(qkv),
            qkv_qh_offset,
            qkv_qs_offset,
            pack_.tensor_data(z),
            z_qh_offset,
            z_qs_offset,
            pack_.tensor_data(beta),
            pack_.tensor_data(alpha),
            reinterpret_cast<const float*>(pack_.tensor_data(conv)),
            reinterpret_cast<const float*>(pack_.tensor_data(dt)),
            reinterpret_cast<const float*>(pack_.tensor_data(a)),
            reinterpret_cast<const float*>(pack_.tensor_data(ssm_norm)),
            pack_.tensor_data(ssm_out),
            ssm_out_qh_offset,
            ssm_out_qs_offset,
            rms_eps,
            stats,
            error);
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

RuntimeInfo Runtime::info() const {
    RuntimeInfo out;
    out.model_path = model_path_.string();
    out.tensor_count = pack_.header().tensor_count;
    out.packed_bytes = pack_.header().data_bytes;
    out.driver_available = driver_.available();
    out.driver_version = driver_.driver_version();
    out.device_name = driver_.device_name();
    out.device_memory = driver_.total_memory();
    out.sm_major = driver_.sm_major();
    out.sm_minor = driver_.sm_minor();
    out.vmm_available = driver_.vmm_available();
    out.vmm_granularity = driver_.vmm_granularity();
    out.native_decode_ready = native_decode_ready_;
    return out;
}

} // namespace q38
