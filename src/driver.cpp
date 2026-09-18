#include "q38/driver.hpp"

#include <dlfcn.h>
#include <stdexcept>

namespace q38 {
namespace {

using CUresult = int;
using CUdevice = int;
inline constexpr CUresult CUDA_SUCCESS = 0;

template <typename T>
T sym(void* handle, const char* name) {
    auto p = reinterpret_cast<T>(::dlsym(handle, name));
    if (!p) throw std::runtime_error(std::string("libcuda missing symbol: ") + name);
    return p;
}

} // namespace

NvidiaDriver::~NvidiaDriver() { close(); }

void NvidiaDriver::open() {
    close();
    handle_ = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!handle_) throw std::runtime_error("unable to load libcuda.so.1; NVIDIA driver is not installed");

    try {
        const auto cuInit = sym<CUresult(*)(unsigned int)>(handle_, "cuInit");
        const auto cuDriverGetVersion = sym<CUresult(*)(int*)>(handle_, "cuDriverGetVersion");
        const auto cuDeviceGet = sym<CUresult(*)(CUdevice*, int)>(handle_, "cuDeviceGet");
        const auto cuDeviceGetName = sym<CUresult(*)(char*, int, CUdevice)>(handle_, "cuDeviceGetName");
        const auto cuDeviceTotalMem = sym<CUresult(*)(std::size_t*, CUdevice)>(handle_, "cuDeviceTotalMem_v2");

        if (cuInit(0) != CUDA_SUCCESS) throw std::runtime_error("cuInit failed");
        if (cuDriverGetVersion(&driver_version_) != CUDA_SUCCESS) throw std::runtime_error("cuDriverGetVersion failed");

        CUdevice dev{};
        if (cuDeviceGet(&dev, device_ordinal_) != CUDA_SUCCESS) throw std::runtime_error("cuDeviceGet(0) failed");
        char name[256]{};
        if (cuDeviceGetName(name, sizeof(name), dev) != CUDA_SUCCESS) throw std::runtime_error("cuDeviceGetName failed");
        device_name_ = name;
        if (cuDeviceTotalMem(&total_memory_, dev) != CUDA_SUCCESS) throw std::runtime_error("cuDeviceTotalMem failed");
        initialized_ = true;
    } catch (...) {
        close();
        throw;
    }
}

void NvidiaDriver::close() noexcept {
    initialized_ = false;
    driver_version_ = 0;
    device_name_.clear();
    total_memory_ = 0;
    if (handle_) {
        ::dlclose(handle_);
        handle_ = nullptr;
    }
}

} // namespace q38
