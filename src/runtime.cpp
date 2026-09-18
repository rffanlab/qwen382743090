#include "q38/runtime.hpp"

namespace q38 {

void Runtime::load(const std::filesystem::path& model_path, bool require_gpu) {
    model_path_ = model_path;
    pack_.open(model_path);
    if (require_gpu) driver_.open();
    else {
        try { driver_.open(); } catch (...) {}
    }

    // Phase 1 intentionally stops at a verified native container + Driver API probe.
    // Native Qwen3.8 graph execution is added kernel-by-kernel with a correctness oracle.
    native_decode_ready_ = false;
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
