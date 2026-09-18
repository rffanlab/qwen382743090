#include "q38/driver.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <dlfcn.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace q38 {
namespace {

// Minimal CUDA Driver ABI subset. These declarations mirror the public CUDA
// driver ABI so Q38RT does not require CUDA Toolkit headers at build time.
using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
using CUmemGenericAllocationHandle = unsigned long long;
using CUcontext = struct CUctx_st*;
using CUmodule = struct CUmod_st*;
using CUfunction = struct CUfunc_st*;
using CUstream = struct CUstream_st*;

inline constexpr CUresult CUDA_SUCCESS = 0;
inline constexpr int CU_MEM_ALLOCATION_TYPE_PINNED = 0x1;
inline constexpr int CU_MEM_HANDLE_TYPE_NONE = 0x0;
inline constexpr int CU_MEM_LOCATION_TYPE_DEVICE = 0x1;
inline constexpr int CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3;
inline constexpr int CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0;

struct CUmemLocation {
    int type;
    union {
        int id;
        struct {
            unsigned char deviceId;
            unsigned char localityDomainId;
        } localized;
    };
};

struct CUmemAllocationProp {
    int type;
    int requestedHandleTypes;
    CUmemLocation location;
    void* win32HandleMetaData;
    struct {
        unsigned char compressionType;
        unsigned char gpuDirectRDMACapable;
        unsigned short usage;
        unsigned char reserved[4];
    } allocFlags;
};

struct CUmemAccessDesc {
    CUmemLocation location;
    int flags;
};

template <typename T>
T sym(void* handle, const char* name) {
    auto p = reinterpret_cast<T>(::dlsym(handle, name));
    if (!p) throw std::runtime_error(std::string("libcuda missing symbol: ") + name);
    return p;
}

template <typename T>
T sym_optional(void* handle, const char* name) noexcept {
    return reinterpret_cast<T>(::dlsym(handle, name));
}

using PFN_cuGetErrorName = CUresult(*)(CUresult, const char**);
using PFN_cuGetErrorString = CUresult(*)(CUresult, const char**);

std::string cuda_error(void* handle, CUresult rc, const char* where) {
    const char* name = nullptr;
    const char* text = nullptr;
    if (auto f = sym_optional<PFN_cuGetErrorName>(handle, "cuGetErrorName")) f(rc, &name);
    if (auto f = sym_optional<PFN_cuGetErrorString>(handle, "cuGetErrorString")) f(rc, &text);
    std::ostringstream oss;
    oss << where << " failed with CUDA error " << rc;
    if (name) oss << " (" << name << ")";
    if (text) oss << ": " << text;
    return oss.str();
}

void check(void* handle, CUresult rc, const char* where) {
    if (rc != CUDA_SUCCESS) throw std::runtime_error(cuda_error(handle, rc, where));
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct VmmApi {
    using AddressReserve = CUresult(*)(CUdeviceptr*, std::size_t, std::size_t, CUdeviceptr, unsigned long long);
    using AddressFree = CUresult(*)(CUdeviceptr, std::size_t);
    using MemCreate = CUresult(*)(CUmemGenericAllocationHandle*, std::size_t, const CUmemAllocationProp*, unsigned long long);
    using MemRelease = CUresult(*)(CUmemGenericAllocationHandle);
    using MemMap = CUresult(*)(CUdeviceptr, std::size_t, std::size_t, CUmemGenericAllocationHandle, unsigned long long);
    using MemUnmap = CUresult(*)(CUdeviceptr, std::size_t);
    using MemSetAccess = CUresult(*)(CUdeviceptr, std::size_t, const CUmemAccessDesc*, std::size_t);
    using GetGranularity = CUresult(*)(std::size_t*, const CUmemAllocationProp*, int);

    AddressReserve address_reserve{};
    AddressFree address_free{};
    MemCreate mem_create{};
    MemRelease mem_release{};
    MemMap mem_map{};
    MemUnmap mem_unmap{};
    MemSetAccess mem_set_access{};
    GetGranularity get_granularity{};

    explicit operator bool() const noexcept {
        return address_reserve && address_free && mem_create && mem_release &&
               mem_map && mem_unmap && mem_set_access && get_granularity;
    }
};

VmmApi load_vmm(void* handle) {
    return {
        sym_optional<VmmApi::AddressReserve>(handle, "cuMemAddressReserve"),
        sym_optional<VmmApi::AddressFree>(handle, "cuMemAddressFree"),
        sym_optional<VmmApi::MemCreate>(handle, "cuMemCreate"),
        sym_optional<VmmApi::MemRelease>(handle, "cuMemRelease"),
        sym_optional<VmmApi::MemMap>(handle, "cuMemMap"),
        sym_optional<VmmApi::MemUnmap>(handle, "cuMemUnmap"),
        sym_optional<VmmApi::MemSetAccess>(handle, "cuMemSetAccess"),
        sym_optional<VmmApi::GetGranularity>(handle, "cuMemGetAllocationGranularity"),
    };
}

CUmemAllocationProp device_prop(int ordinal) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = ordinal;
    return prop;
}

class ScopedVmm {
public:
    ScopedVmm(void* driver_handle, int ordinal, std::size_t requested)
        : h_(driver_handle), api_(load_vmm(driver_handle)) {
        if (!api_) throw std::runtime_error("CUDA VMM API is not available in this driver");
        const auto prop = device_prop(ordinal);
        check(h_, api_.get_granularity(&granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "cuMemGetAllocationGranularity");
        size_ = align_up(std::max<std::size_t>(requested, 1), granularity_);
        check(h_, api_.address_reserve(&ptr_, size_, 0, 0, 0), "cuMemAddressReserve");
        try {
            check(h_, api_.mem_create(&allocation_, size_, &prop, 0), "cuMemCreate");
            allocated_ = true;
            check(h_, api_.mem_map(ptr_, size_, 0, allocation_, 0), "cuMemMap");
            mapped_ = true;
            CUmemAccessDesc access{};
            access.location = prop.location;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            check(h_, api_.mem_set_access(ptr_, size_, &access, 1), "cuMemSetAccess");
        } catch (...) {
            release();
            throw;
        }
    }

    ~ScopedVmm() { release(); }
    ScopedVmm(const ScopedVmm&) = delete;
    ScopedVmm& operator=(const ScopedVmm&) = delete;

    [[nodiscard]] CUdeviceptr ptr() const noexcept { return ptr_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t granularity() const noexcept { return granularity_; }

private:
    void release() noexcept {
        if (mapped_) {
            api_.mem_unmap(ptr_, size_);
            mapped_ = false;
        }
        if (allocated_) {
            api_.mem_release(allocation_);
            allocated_ = false;
        }
        if (ptr_) {
            api_.address_free(ptr_, size_);
            ptr_ = 0;
        }
    }

    void* h_{};
    VmmApi api_;
    CUdeviceptr ptr_{};
    CUmemGenericAllocationHandle allocation_{};
    std::size_t size_{};
    std::size_t granularity_{};
    bool allocated_{false};
    bool mapped_{false};
};

constexpr const char* kSmokePtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

.visible .entry q38_add_one(
    .param .u64 p_data,
    .param .u32 p_count
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd1, [p_data];
    ld.param.u32 %r1, [p_count];
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.u32 %r2, [%rd3];
    add.u32 %r2, %r2, 1;
    st.global.u32 [%rd3], %r2;

DONE:
    ret;
}
)ptx";

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
        const auto cuDeviceComputeCapability = sym<CUresult(*)(int*, int*, CUdevice)>(handle_, "cuDeviceComputeCapability");
        const auto cuCtxCreate = sym<CUresult(*)(CUcontext*, unsigned int, CUdevice)>(handle_, "cuCtxCreate_v2");

        check(handle_, cuInit(0), "cuInit");
        check(handle_, cuDriverGetVersion(&driver_version_), "cuDriverGetVersion");

        CUdevice dev{};
        check(handle_, cuDeviceGet(&dev, device_ordinal_), "cuDeviceGet");

        char name[256]{};
        check(handle_, cuDeviceGetName(name, sizeof(name), dev), "cuDeviceGetName");
        device_name_ = name;
        check(handle_, cuDeviceTotalMem(&total_memory_, dev), "cuDeviceTotalMem");
        check(handle_, cuDeviceComputeCapability(&sm_major_, &sm_minor_, dev), "cuDeviceComputeCapability");

        CUcontext ctx{};
        check(handle_, cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");
        context_ = ctx;

        const auto vmm = load_vmm(handle_);
        vmm_available_ = static_cast<bool>(vmm);
        if (vmm_available_) {
            const auto prop = device_prop(device_ordinal_);
            if (vmm.get_granularity(&vmm_granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS) {
                vmm_available_ = false;
                vmm_granularity_ = 0;
            }
        }
        initialized_ = true;
    } catch (...) {
        close();
        throw;
    }
}

void NvidiaDriver::close() noexcept {
    initialized_ = false;
    vmm_available_ = false;
    vmm_granularity_ = 0;
    driver_version_ = 0;
    sm_major_ = 0;
    sm_minor_ = 0;
    device_name_.clear();
    total_memory_ = 0;

    if (handle_ && context_) {
        if (auto cuCtxDestroy = sym_optional<CUresult(*)(CUcontext)>(handle_, "cuCtxDestroy_v2")) {
            cuCtxDestroy(static_cast<CUcontext>(context_));
        }
        context_ = nullptr;
    }
    if (handle_) {
        ::dlclose(handle_);
        handle_ = nullptr;
    }
}

bool NvidiaDriver::probe_vmm(std::size_t bytes, std::string* error) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        ScopedVmm allocation(handle_, device_ordinal_, bytes);
        vmm_granularity_ = allocation.granularity();
        vmm_available_ = true;
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool NvidiaDriver::run_sm86_smoke(std::string* error) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error("q38 SM86 smoke requires compute capability 8.6; detected sm_" +
                                     std::to_string(sm_major_) + std::to_string(sm_minor_));
        }

        ScopedVmm memory(handle_, device_ordinal_, 4096);
        using MemcpyHtoD = CUresult(*)(CUdeviceptr, const void*, std::size_t);
        using MemcpyDtoH = CUresult(*)(void*, CUdeviceptr, std::size_t);
        using ModuleLoadDataEx = CUresult(*)(CUmodule*, const void*, unsigned int, int*, void**);
        using ModuleUnload = CUresult(*)(CUmodule);
        using ModuleGetFunction = CUresult(*)(CUfunction*, CUmodule, const char*);
        using LaunchKernel = CUresult(*)(CUfunction,
                                         unsigned int, unsigned int, unsigned int,
                                         unsigned int, unsigned int, unsigned int,
                                         unsigned int, CUstream, void**, void**);
        using CtxSynchronize = CUresult(*)();

        const auto memcpy_htod = sym<MemcpyHtoD>(handle_, "cuMemcpyHtoD_v2");
        const auto memcpy_dtoh = sym<MemcpyDtoH>(handle_, "cuMemcpyDtoH_v2");
        const auto module_load_ex = sym<ModuleLoadDataEx>(handle_, "cuModuleLoadDataEx");
        const auto module_unload = sym<ModuleUnload>(handle_, "cuModuleUnload");
        const auto module_get_function = sym<ModuleGetFunction>(handle_, "cuModuleGetFunction");
        const auto launch = sym<LaunchKernel>(handle_, "cuLaunchKernel");
        const auto sync = sym<CtxSynchronize>(handle_, "cuCtxSynchronize");

        std::array<std::uint32_t, 256> values{};
        for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<std::uint32_t>(i * 3 + 7);
        const auto before = values;
        check(handle_, memcpy_htod(memory.ptr(), values.data(), sizeof(values)), "cuMemcpyHtoD");

        // sm_86 was introduced in PTX ISA 7.1. Keep the smoke PTX at the
        // minimum compatible ISA so it works on old and new Ampere drivers.
        // Always ask the driver for the JIT logs: CUDA_ERROR_INVALID_PTX alone
        // is too opaque for bring-up/debugging.
        constexpr int CU_JIT_INFO_LOG_BUFFER = 3;
        constexpr int CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4;
        constexpr int CU_JIT_ERROR_LOG_BUFFER = 5;
        constexpr int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;
        constexpr int CU_JIT_LOG_VERBOSE = 12;

        std::array<char, 8192> jit_info{};
        std::array<char, 8192> jit_error{};
        int jit_options[] = {
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_LOG_VERBOSE,
        };
        void* jit_values[] = {
            jit_info.data(),
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(jit_info.size())),
            jit_error.data(),
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(jit_error.size())),
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(1)),
        };

        CUmodule module{};
        const auto module_rc = module_load_ex(
            &module,
            kSmokePtx,
            static_cast<unsigned int>(sizeof(jit_options) / sizeof(jit_options[0])),
            jit_options,
            jit_values);
        if (module_rc != CUDA_SUCCESS) {
            std::string detail = cuda_error(handle_, module_rc, "cuModuleLoadDataEx");
            if (jit_error[0] != '\0') detail += std::string("\nPTX JIT error log:\n") + jit_error.data();
            if (jit_info[0] != '\0') detail += std::string("\nPTX JIT info log:\n") + jit_info.data();
            throw std::runtime_error(detail);
        }
        try {
            CUfunction fn{};
            check(handle_, module_get_function(&fn, module, "q38_add_one"), "cuModuleGetFunction");
            CUdeviceptr ptr = memory.ptr();
            std::uint32_t count = static_cast<std::uint32_t>(values.size());
            void* params[] = {&ptr, &count};
            check(handle_, launch(fn, 1, 1, 1, 256, 1, 1, 0, nullptr, params, nullptr), "cuLaunchKernel");
            check(handle_, sync(), "cuCtxSynchronize");
            check(handle_, memcpy_dtoh(values.data(), memory.ptr(), sizeof(values)), "cuMemcpyDtoH");
        } catch (...) {
            module_unload(module);
            throw;
        }
        check(handle_, module_unload(module), "cuModuleUnload");

        for (std::size_t i = 0; i < values.size(); ++i) {
            if (values[i] != before[i] + 1) {
                throw std::runtime_error("SM86 smoke produced incorrect output at index " + std::to_string(i));
            }
        }
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

} // namespace q38
