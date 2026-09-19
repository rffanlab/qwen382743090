#include "q38/driver.hpp"
#include "q38/quant.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <dlfcn.h>
#include <sstream>
#include <stdexcept>
#include <vector>
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

constexpr const char* kRmsNormPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

.visible .entry q38_sumsq(
    .param .u64 p_x,
    .param .u64 p_sumsq,
    .param .u32 p_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [p_x];
    ld.param.u64 %rd2, [p_sumsq];
    ld.param.u32 %r1, [p_n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra SUM_DONE;

    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    mul.rn.f32 %f2, %f1, %f1;
    atom.global.add.f32 %f3, [%rd2], %f2;

SUM_DONE:
    ret;
}

.visible .entry q38_rmsnorm_apply(
    .param .u64 p_x,
    .param .u64 p_weight,
    .param .u64 p_out,
    .param .u64 p_sumsq,
    .param .u32 p_n,
    .param .f32 p_eps
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<12>;

    ld.param.u64 %rd1, [p_x];
    ld.param.u64 %rd2, [p_weight];
    ld.param.u64 %rd3, [p_out];
    ld.param.u64 %rd4, [p_sumsq];
    ld.param.u32 %r1, [p_n];
    ld.param.f32 %f1, [p_eps];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra APPLY_DONE;

    ld.global.f32 %f2, [%rd4];
    cvt.rn.f32.u32 %f3, %r1;
    div.rn.f32 %f4, %f2, %f3;
    add.rn.f32 %f4, %f4, %f1;
    rsqrt.approx.f32 %f5, %f4;

    mul.wide.u32 %rd5, %r5, 4;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;
    ld.global.f32 %f6, [%rd6];
    ld.global.f32 %f7, [%rd7];
    mul.rn.f32 %f8, %f6, %f5;
    mul.rn.f32 %f9, %f8, %f7;
    st.global.f32 [%rd8], %f9;

APPLY_DONE:
    ret;
}
)ptx";

constexpr const char* kQ4KDequantPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

.visible .entry q38_dequant_q4k_block(
    .param .u64 p_block,
    .param .u64 p_out
)
{
    .reg .pred %p<8>;
    .reg .f16 %h<4>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<12>;

    ld.param.u64 %rd1, [p_block];
    ld.param.u64 %rd2, [p_out];

    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p1, %r1, 256;
    @%p1 bra Q4K_DONE;

    // d and dmin
    ld.global.f16 %h1, [%rd1+0];
    ld.global.f16 %h2, [%rd1+2];
    cvt.f32.f16 %f1, %h1;
    cvt.f32.f16 %f2, %h2;

    // group = tid / 32, lane = tid % 32
    shr.u32 %r2, %r1, 5;
    and.b32 %r3, %r1, 31;

    // scales base = block + 4
    add.s64 %rd3, %rd1, 4;

    setp.lt.u32 %p2, %r2, 4;
    @%p2 bra SCALE_LOW;

    // group 4..7:
    // sc = (q[g+4] & 0x0f) | ((q[g-4] >> 6) << 4)
    // m  = (q[g+4] >> 4)   | ((q[g]   >> 6) << 4)
    add.u32 %r4, %r2, 4;
    cvt.u64.u32 %rd4, %r4;
    add.s64 %rd5, %rd3, %rd4;
    ld.global.u8 %r5, [%rd5];

    sub.u32 %r6, %r2, 4;
    cvt.u64.u32 %rd6, %r6;
    add.s64 %rd7, %rd3, %rd6;
    ld.global.u8 %r7, [%rd7];

    cvt.u64.u32 %rd8, %r2;
    add.s64 %rd9, %rd3, %rd8;
    ld.global.u8 %r8, [%rd9];

    and.b32 %r9, %r5, 15;
    shr.u32 %r10, %r7, 6;
    shl.b32 %r10, %r10, 4;
    or.b32 %r11, %r9, %r10;

    shr.u32 %r12, %r5, 4;
    shr.u32 %r13, %r8, 6;
    shl.b32 %r13, %r13, 4;
    or.b32 %r14, %r12, %r13;
    bra SCALE_READY;

SCALE_LOW:
    // group 0..3:
    // sc = q[g] & 63, m = q[g+4] & 63
    cvt.u64.u32 %rd4, %r2;
    add.s64 %rd5, %rd3, %rd4;
    ld.global.u8 %r5, [%rd5];
    and.b32 %r11, %r5, 63;

    add.u32 %r6, %r2, 4;
    cvt.u64.u32 %rd6, %r6;
    add.s64 %rd7, %rd3, %rd6;
    ld.global.u8 %r7, [%rd7];
    and.b32 %r14, %r7, 63;

SCALE_READY:
    cvt.rn.f32.u32 %f3, %r11;
    cvt.rn.f32.u32 %f4, %r14;
    mul.rn.f32 %f5, %f1, %f3;
    mul.rn.f32 %f6, %f2, %f4;

    // qs base = block + 16
    // byte index = (group/2)*32 + lane
    shr.u32 %r15, %r2, 1;
    shl.b32 %r15, %r15, 5;
    add.u32 %r15, %r15, %r3;
    cvt.u64.u32 %rd10, %r15;
    add.s64 %rd11, %rd1, 16;
    add.s64 %rd12, %rd11, %rd10;
    ld.global.u8 %r16, [%rd12];

    and.b32 %r17, %r2, 1;
    setp.eq.u32 %p3, %r17, 0;
    @%p3 bra LOW_NIBBLE;
    shr.u32 %r18, %r16, 4;
    bra NIBBLE_READY;

LOW_NIBBLE:
    and.b32 %r18, %r16, 15;

NIBBLE_READY:
    cvt.rn.f32.u32 %f7, %r18;
    mul.rn.f32 %f8, %f5, %f7;
    sub.rn.f32 %f9, %f8, %f6;

    mul.wide.u32 %rd13, %r1, 4;
    add.s64 %rd14, %rd2, %rd13;
    st.global.f32 [%rd14], %f9;

Q4K_DONE:
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

bool NvidiaDriver::run_rmsnorm_smoke(std::string* error, double* max_abs_error, double* max_rel_error) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error("q38 RMSNorm smoke requires compute capability 8.6; detected sm_" +
                                     std::to_string(sm_major_) + std::to_string(sm_minor_));
        }

        constexpr std::uint32_t n = 5120;
        constexpr float eps = 1.0e-6f;
        constexpr unsigned int block = 256;
        constexpr unsigned int grid = (n + block - 1) / block;
        const std::size_t vec_bytes = static_cast<std::size_t>(n) * sizeof(float);
        const std::size_t x_off = 0;
        const std::size_t w_off = x_off + vec_bytes;
        const std::size_t y_off = w_off + vec_bytes;
        const std::size_t s_off = y_off + vec_bytes;
        const std::size_t total_bytes = s_off + sizeof(float);

        ScopedVmm memory(handle_, device_ordinal_, total_bytes);

        using MemcpyHtoD = CUresult(*)(CUdeviceptr, const void*, std::size_t);
        using MemcpyDtoH = CUresult(*)(void*, CUdeviceptr, std::size_t);
        using MemsetD32 = CUresult(*)(CUdeviceptr, unsigned int, std::size_t);
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
        const auto memset_d32 = sym<MemsetD32>(handle_, "cuMemsetD32_v2");
        const auto module_load_ex = sym<ModuleLoadDataEx>(handle_, "cuModuleLoadDataEx");
        const auto module_unload = sym<ModuleUnload>(handle_, "cuModuleUnload");
        const auto module_get_function = sym<ModuleGetFunction>(handle_, "cuModuleGetFunction");
        const auto launch = sym<LaunchKernel>(handle_, "cuLaunchKernel");
        const auto sync = sym<CtxSynchronize>(handle_, "cuCtxSynchronize");

        std::vector<float> x(n), weight(n), out(n), reference(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            const float fi = static_cast<float>(i);
            x[i] = std::sin(fi * 0.013f) * 0.75f + std::cos(fi * 0.007f) * 0.25f;
            weight[i] = 0.8f + static_cast<float>(i % 37) * 0.01f;
        }

        double sumsq = 0.0;
        for (float v : x) sumsq += static_cast<double>(v) * static_cast<double>(v);
        const double inv_rms = 1.0 / std::sqrt(sumsq / static_cast<double>(n) + static_cast<double>(eps));
        for (std::uint32_t i = 0; i < n; ++i) {
            reference[i] = static_cast<float>(static_cast<double>(x[i]) * inv_rms * static_cast<double>(weight[i]));
        }

        const CUdeviceptr base = memory.ptr();
        const CUdeviceptr x_ptr = base + x_off;
        const CUdeviceptr w_ptr = base + w_off;
        const CUdeviceptr y_ptr = base + y_off;
        const CUdeviceptr s_ptr = base + s_off;

        check(handle_, memcpy_htod(x_ptr, x.data(), vec_bytes), "cuMemcpyHtoD(x)");
        check(handle_, memcpy_htod(w_ptr, weight.data(), vec_bytes), "cuMemcpyHtoD(weight)");
        check(handle_, memset_d32(s_ptr, 0, 1), "cuMemsetD32(sumsq)");

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
            kRmsNormPtx,
            static_cast<unsigned int>(sizeof(jit_options) / sizeof(jit_options[0])),
            jit_options,
            jit_values);
        if (module_rc != CUDA_SUCCESS) {
            std::string detail = cuda_error(handle_, module_rc, "cuModuleLoadDataEx(RMSNorm)");
            if (jit_error[0] != '\0') detail += std::string("\nPTX JIT error log:\n") + jit_error.data();
            if (jit_info[0] != '\0') detail += std::string("\nPTX JIT info log:\n") + jit_info.data();
            throw std::runtime_error(detail);
        }

        try {
            CUfunction sum_fn{}, apply_fn{};
            check(handle_, module_get_function(&sum_fn, module, "q38_sumsq"), "cuModuleGetFunction(q38_sumsq)");
            check(handle_, module_get_function(&apply_fn, module, "q38_rmsnorm_apply"), "cuModuleGetFunction(q38_rmsnorm_apply)");

            CUdeviceptr sx = x_ptr;
            CUdeviceptr ss = s_ptr;
            std::uint32_t count = n;
            void* sum_params[] = {&sx, &ss, &count};
            check(handle_, launch(sum_fn, grid, 1, 1, block, 1, 1, 0, nullptr, sum_params, nullptr),
                  "cuLaunchKernel(q38_sumsq)");

            CUdeviceptr ax = x_ptr;
            CUdeviceptr aw = w_ptr;
            CUdeviceptr ay = y_ptr;
            CUdeviceptr as = s_ptr;
            float kernel_eps = eps;
            void* apply_params[] = {&ax, &aw, &ay, &as, &count, &kernel_eps};
            check(handle_, launch(apply_fn, grid, 1, 1, block, 1, 1, 0, nullptr, apply_params, nullptr),
                  "cuLaunchKernel(q38_rmsnorm_apply)");
            check(handle_, sync(), "cuCtxSynchronize(RMSNorm)");
            check(handle_, memcpy_dtoh(out.data(), y_ptr, vec_bytes), "cuMemcpyDtoH(RMSNorm)");
        } catch (...) {
            module_unload(module);
            throw;
        }
        check(handle_, module_unload(module), "cuModuleUnload(RMSNorm)");

        double abs_max = 0.0;
        double rel_max = 0.0;
        for (std::uint32_t i = 0; i < n; ++i) {
            const double got = static_cast<double>(out[i]);
            const double ref = static_cast<double>(reference[i]);
            const double abs_err = std::abs(got - ref);
            const double rel_err = abs_err / std::max(1.0e-6, std::abs(ref));
            abs_max = std::max(abs_max, abs_err);
            rel_max = std::max(rel_max, rel_err);
        }
        if (max_abs_error) *max_abs_error = abs_max;
        if (max_rel_error) *max_rel_error = rel_max;

        if (!(abs_max <= 5.0e-4 && rel_max <= 5.0e-4)) {
            std::ostringstream oss;
            oss << "RMSNorm numeric mismatch: max_abs=" << abs_max << " max_rel=" << rel_max;
            throw std::runtime_error(oss.str());
        }
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool NvidiaDriver::run_q4k_dequant_smoke(const std::byte* block, std::string* error, double* max_abs_error, double* max_rel_error) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error("q38 Q4_K smoke requires compute capability 8.6; detected sm_" +
                                     std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (!block) throw std::invalid_argument("Q4_K block is null");

        std::array<float, kQ4KValuesPerBlock> reference{};
        dequantize_q4_k_block_cpu(block, reference);
        std::array<float, kQ4KValuesPerBlock> gpu{};

        const std::size_t block_off = 0;
        const std::size_t out_off = 256;
        const std::size_t total_bytes = out_off + gpu.size() * sizeof(float);
        ScopedVmm memory(handle_, device_ordinal_, total_bytes);

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

        const CUdeviceptr block_ptr = memory.ptr() + block_off;
        const CUdeviceptr out_ptr = memory.ptr() + out_off;
        check(handle_, memcpy_htod(block_ptr, block, kQ4KBytesPerBlock), "cuMemcpyHtoD(Q4_K block)");

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
            kQ4KDequantPtx,
            static_cast<unsigned int>(sizeof(jit_options) / sizeof(jit_options[0])),
            jit_options,
            jit_values);
        if (module_rc != CUDA_SUCCESS) {
            std::string detail = cuda_error(handle_, module_rc, "cuModuleLoadDataEx(Q4_K)");
            if (jit_error[0] != '\0') detail += std::string("\nPTX JIT error log:\n") + jit_error.data();
            if (jit_info[0] != '\0') detail += std::string("\nPTX JIT info log:\n") + jit_info.data();
            throw std::runtime_error(detail);
        }

        try {
            CUfunction fn{};
            check(handle_, module_get_function(&fn, module, "q38_dequant_q4k_block"),
                  "cuModuleGetFunction(q38_dequant_q4k_block)");

            CUdeviceptr arg_block = block_ptr;
            CUdeviceptr arg_out = out_ptr;
            void* params[] = {&arg_block, &arg_out};
            check(handle_, launch(fn, 1, 1, 1, 256, 1, 1, 0, nullptr, params, nullptr),
                  "cuLaunchKernel(q38_dequant_q4k_block)");
            check(handle_, sync(), "cuCtxSynchronize(Q4_K)");
            check(handle_, memcpy_dtoh(gpu.data(), out_ptr, gpu.size() * sizeof(float)),
                  "cuMemcpyDtoH(Q4_K)");
        } catch (...) {
            module_unload(module);
            throw;
        }
        check(handle_, module_unload(module), "cuModuleUnload(Q4_K)");

        double abs_max = 0.0;
        double rel_max = 0.0;
        for (std::size_t i = 0; i < gpu.size(); ++i) {
            const double got = static_cast<double>(gpu[i]);
            const double ref = static_cast<double>(reference[i]);
            const double abs_err = std::abs(got - ref);
            const double rel_err = abs_err / std::max(1.0e-6, std::abs(ref));
            abs_max = std::max(abs_max, abs_err);
            rel_max = std::max(rel_max, rel_err);
        }
        if (max_abs_error) *max_abs_error = abs_max;
        if (max_rel_error) *max_rel_error = rel_max;

        if (!(abs_max <= 1.0e-5 && rel_max <= 1.0e-5)) {
            std::ostringstream oss;
            oss << "Q4_K dequant mismatch: max_abs=" << abs_max << " max_rel=" << rel_max;
            throw std::runtime_error(oss.str());
        }
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

} // namespace q38
