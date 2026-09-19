#include "q38/driver.hpp"
#include "q38/quant.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <chrono>
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

constexpr const char* kQwen35RecurrentPrepPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

.visible .entry q38_qwen35_conv4_silu_update(
    .param .u64 p_qkv,
    .param .u64 p_weight,
    .param .u64 p_state_in,
    .param .u64 p_conv_out,
    .param .u64 p_state_out,
    .param .u32 p_channels,
    .param .f32 p_log2e
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<20>;

    ld.param.u64 %rd1, [p_qkv];
    ld.param.u64 %rd2, [p_weight];
    ld.param.u64 %rd3, [p_state_in];
    ld.param.u64 %rd4, [p_conv_out];
    ld.param.u64 %rd5, [p_state_out];
    ld.param.u32 %r1, [p_channels];
    ld.param.f32 %f1, [p_log2e];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra C4_DONE;

    mul.wide.u32 %rd6, %r5, 4;
    add.s64 %rd7, %rd1, %rd6;
    ld.global.f32 %f2, [%rd7];

    mul.lo.u32 %r6, %r5, 3;
    mul.wide.u32 %rd8, %r6, 4;
    add.s64 %rd9, %rd3, %rd8;
    add.s64 %rd10, %rd5, %rd8;
    ld.global.f32 %f3, [%rd9+0];
    ld.global.f32 %f4, [%rd9+4];
    ld.global.f32 %f5, [%rd9+8];

    shl.b32 %r7, %r5, 2;
    mul.wide.u32 %rd11, %r7, 4;
    add.s64 %rd12, %rd2, %rd11;
    ld.global.v4.f32 {%f6,%f7,%f8,%f9}, [%rd12];

    mul.rn.f32 %f10, %f3, %f6;
    fma.rn.f32 %f10, %f4, %f7, %f10;
    fma.rn.f32 %f10, %f5, %f8, %f10;
    fma.rn.f32 %f10, %f2, %f9, %f10;

    // SiLU(x) = x / (1 + exp(-x)).
    neg.f32 %f11, %f10;
    mul.rn.f32 %f11, %f11, %f1;
    ex2.approx.f32 %f12, %f11;
    add.rn.f32 %f13, %f12, 0f3f800000;
    div.rn.f32 %f14, %f10, %f13;

    add.s64 %rd13, %rd4, %rd6;
    st.global.f32 [%rd13], %f14;

    // Shift decode-time conv state: [s0,s1,s2] -> [s1,s2,current].
    st.global.f32 [%rd10+0], %f4;
    st.global.f32 [%rd10+4], %f5;
    st.global.f32 [%rd10+8], %f2;

C4_DONE:
    ret;
}

.visible .entry q38_qwen35_qk_sumsq(
    .param .u64 p_conv,
    .param .u64 p_sums,
    .param .u32 p_count
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<6>;

    ld.param.u64 %rd1, [p_conv];
    ld.param.u64 %rd2, [p_sums];
    ld.param.u32 %r1, [p_count];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra QKS_DONE;

    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    mul.rn.f32 %f2, %f1, %f1;

    shr.u32 %r6, %r5, 7;
    mul.wide.u32 %rd5, %r6, 4;
    add.s64 %rd6, %rd2, %rd5;
    atom.global.add.f32 %f3, [%rd6], %f2;

QKS_DONE:
    ret;
}

.visible .entry q38_qwen35_qk_norm(
    .param .u64 p_conv,
    .param .u64 p_sums,
    .param .u64 p_qk_out,
    .param .u32 p_count,
    .param .f32 p_eps
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<10>;

    ld.param.u64 %rd1, [p_conv];
    ld.param.u64 %rd2, [p_sums];
    ld.param.u64 %rd3, [p_qk_out];
    ld.param.u32 %r1, [p_count];
    ld.param.f32 %f1, [p_eps];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra QKN_DONE;

    shr.u32 %r6, %r5, 7;
    mul.wide.u32 %rd4, %r6, 4;
    add.s64 %rd5, %rd2, %rd4;
    ld.global.f32 %f2, [%rd5];
    add.rn.f32 %f3, %f2, %f1;
    rsqrt.approx.f32 %f4, %f3;

    mul.wide.u32 %rd6, %r5, 4;
    add.s64 %rd7, %rd1, %rd6;
    add.s64 %rd8, %rd3, %rd6;
    ld.global.f32 %f5, [%rd7];
    mul.rn.f32 %f6, %f5, %f4;
    st.global.f32 [%rd8], %f6;

QKN_DONE:
    ret;
}

.visible .entry q38_qwen35_gate_beta(
    .param .u64 p_beta_raw,
    .param .u64 p_alpha_raw,
    .param .u64 p_dt,
    .param .u64 p_a,
    .param .u64 p_beta_out,
    .param .u64 p_gate_out,
    .param .u32 p_heads,
    .param .f32 p_log2e,
    .param .f32 p_ln2
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<24>;

    ld.param.u64 %rd1, [p_beta_raw];
    ld.param.u64 %rd2, [p_alpha_raw];
    ld.param.u64 %rd3, [p_dt];
    ld.param.u64 %rd4, [p_a];
    ld.param.u64 %rd5, [p_beta_out];
    ld.param.u64 %rd6, [p_gate_out];
    ld.param.u32 %r1, [p_heads];
    ld.param.f32 %f1, [p_log2e];
    ld.param.f32 %f2, [p_ln2];

    mov.u32 %r2, %tid.x;
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra GB_DONE;

    mul.wide.u32 %rd7, %r2, 4;
    add.s64 %rd8, %rd1, %rd7;
    add.s64 %rd9, %rd2, %rd7;
    add.s64 %rd10, %rd3, %rd7;
    add.s64 %rd11, %rd4, %rd7;
    add.s64 %rd12, %rd5, %rd7;
    add.s64 %rd13, %rd6, %rd7;

    ld.global.f32 %f3, [%rd8];
    ld.global.f32 %f4, [%rd9];
    ld.global.f32 %f5, [%rd10];
    ld.global.f32 %f6, [%rd11];

    // sigmoid(beta_raw)
    neg.f32 %f7, %f3;
    mul.rn.f32 %f7, %f7, %f1;
    ex2.approx.f32 %f8, %f7;
    add.rn.f32 %f9, %f8, 0f3f800000;
    rcp.approx.f32 %f10, %f9;
    st.global.f32 [%rd12], %f10;

    // softplus(alpha + dt), stable for large positive values.
    add.rn.f32 %f11, %f4, %f5;
    setp.gt.f32 %p2, %f11, 0f41a00000; // 20.0
    @%p2 mov.f32 %f16, %f11;
    @%p2 bra GB_SOFTPLUS_READY;

    mul.rn.f32 %f12, %f11, %f1;
    ex2.approx.f32 %f13, %f12;
    add.rn.f32 %f14, %f13, 0f3f800000;
    lg2.approx.f32 %f15, %f14;
    mul.rn.f32 %f16, %f15, %f2;

GB_SOFTPLUS_READY:
    mul.rn.f32 %f17, %f16, %f6;
    st.global.f32 [%rd13], %f17;

GB_DONE:
    ret;
}
)ptx";

constexpr const char* kGdnAr128Ptx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

// Qwen3.8/Qwen35 decode-time Gated DeltaNet core.
// Fixed S_v=128. Grid: x=value head, y=state-column tile (4 columns/CTA).
// Block: 128 threads = 4 warps, one state column per warp.
// q/k have Hq heads; value/g/beta/state have Hv heads. value head h reuses
// q/k head h % Hq, matching llama.cpp fused GDN.
.visible .entry q38_gdn_ar_128(
    .param .u64 p_q,
    .param .u64 p_k,
    .param .u64 p_v,
    .param .u64 p_g,
    .param .u64 p_beta,
    .param .u64 p_state_in,
    .param .u64 p_out,
    .param .u64 p_state_out,
    .param .u32 p_qk_heads,
    .param .u32 p_value_heads,
    .param .f32 p_scale,
    .param .f32 p_log2e
)
{
    .reg .pred %p<16>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<40>;

    ld.param.u64 %rd1, [p_q];
    ld.param.u64 %rd2, [p_k];
    ld.param.u64 %rd3, [p_v];
    ld.param.u64 %rd4, [p_g];
    ld.param.u64 %rd5, [p_beta];
    ld.param.u64 %rd6, [p_state_in];
    ld.param.u64 %rd7, [p_out];
    ld.param.u64 %rd8, [p_state_out];
    ld.param.u32 %r1, [p_qk_heads];
    ld.param.u32 %r2, [p_value_heads];
    ld.param.f32 %f1, [p_scale];
    ld.param.f32 %f2, [p_log2e];

    mov.u32 %r3, %ctaid.x;      // value head
    mov.u32 %r4, %ctaid.y;      // column tile
    mov.u32 %r5, %tid.x;
    shr.u32 %r6, %r5, 5;        // warp 0..3
    and.b32 %r7, %r5, 31;       // lane 0..31

    setp.ge.u32 %p1, %r3, %r2;
    @%p1 bra GDN_DONE;

    shl.b32 %r8, %r4, 2;
    add.u32 %r8, %r8, %r6;      // state/output column 0..127
    setp.ge.u32 %p2, %r8, 128;
    @%p2 bra GDN_DONE;

    rem.u32 %r9, %r3, %r1;      // q/k head

    // q/k base indices in floats.
    shl.b32 %r10, %r9, 7;       // qhead * 128

    // State base in floats: head*128*128 + col*128.
    shl.b32 %r11, %r3, 14;      // head * 16384
    shl.b32 %r12, %r8, 7;       // col * 128
    add.u32 %r13, %r11, %r12;

    // Lane owns rows lane + {0,32,64,96}.
    add.u32 %r14, %r13, %r7;
    add.u32 %r15, %r10, %r7;

    // Byte addresses state rows.
    mul.wide.u32 %rd9, %r14, 4;
    add.s64 %rd10, %rd6, %rd9;
    add.s64 %rd11, %rd8, %rd9;

    ld.global.f32 %f3,  [%rd10+0];
    ld.global.f32 %f4,  [%rd10+128];
    ld.global.f32 %f5,  [%rd10+256];
    ld.global.f32 %f6,  [%rd10+384];

    // q/k rows.
    mul.wide.u32 %rd12, %r15, 4;
    add.s64 %rd13, %rd1, %rd12;
    add.s64 %rd14, %rd2, %rd12;

    ld.global.f32 %f7,  [%rd13+0];
    ld.global.f32 %f8,  [%rd13+128];
    ld.global.f32 %f9,  [%rd13+256];
    ld.global.f32 %f10, [%rd13+384];

    ld.global.f32 %f11, [%rd14+0];
    ld.global.f32 %f12, [%rd14+128];
    ld.global.f32 %f13, [%rd14+256];
    ld.global.f32 %f14, [%rd14+384];

    // g=head scalar, beta=head scalar, v=head*128+col.
    mul.wide.u32 %rd15, %r3, 4;
    add.s64 %rd16, %rd4, %rd15;
    add.s64 %rd17, %rd5, %rd15;
    ld.global.f32 %f15, [%rd16];
    ld.global.f32 %f16, [%rd17];

    shl.b32 %r16, %r3, 7;
    add.u32 %r16, %r16, %r8;
    mul.wide.u32 %rd18, %r16, 4;
    add.s64 %rd19, %rd3, %rd18;
    ld.global.f32 %f17, [%rd19];

    // g_val = exp(g) using exp2(g * log2(e)).
    mul.rn.f32 %f18, %f15, %f2;
    ex2.approx.f32 %f19, %f18;

    // kv partial = dot(state_col, k).
    mul.rn.f32 %f20, %f3, %f11;
    fma.rn.f32 %f20, %f4, %f12, %f20;
    fma.rn.f32 %f20, %f5, %f13, %f20;
    fma.rn.f32 %f20, %f6, %f14, %f20;

    mov.b32 %r20, %f20;
    shfl.sync.bfly.b32 %r21, %r20, 16, 31, 0xffffffff;
    mov.b32 %f21, %r21;
    add.rn.f32 %f20, %f20, %f21;
    mov.b32 %r20, %f20;
    shfl.sync.bfly.b32 %r21, %r20, 8, 31, 0xffffffff;
    mov.b32 %f21, %r21;
    add.rn.f32 %f20, %f20, %f21;
    mov.b32 %r20, %f20;
    shfl.sync.bfly.b32 %r21, %r20, 4, 31, 0xffffffff;
    mov.b32 %f21, %r21;
    add.rn.f32 %f20, %f20, %f21;
    mov.b32 %r20, %f20;
    shfl.sync.bfly.b32 %r21, %r20, 2, 31, 0xffffffff;
    mov.b32 %f21, %r21;
    add.rn.f32 %f20, %f20, %f21;
    mov.b32 %r20, %f20;
    shfl.sync.bfly.b32 %r21, %r20, 1, 31, 0xffffffff;
    mov.b32 %f21, %r21;
    add.rn.f32 %f20, %f20, %f21;

    // delta = (v - g*kv) * beta.
    mul.rn.f32 %f22, %f19, %f20;
    sub.rn.f32 %f22, %f17, %f22;
    mul.rn.f32 %f22, %f22, %f16;

    // Update state rows: g*S + k*delta.
    mul.rn.f32 %f23, %f19, %f3;
    fma.rn.f32 %f23, %f11, %f22, %f23;
    mul.rn.f32 %f24, %f19, %f4;
    fma.rn.f32 %f24, %f12, %f22, %f24;
    mul.rn.f32 %f25, %f19, %f5;
    fma.rn.f32 %f25, %f13, %f22, %f25;
    mul.rn.f32 %f26, %f19, %f6;
    fma.rn.f32 %f26, %f14, %f22, %f26;

    st.global.f32 [%rd11+0],   %f23;
    st.global.f32 [%rd11+128], %f24;
    st.global.f32 [%rd11+256], %f25;
    st.global.f32 [%rd11+384], %f26;

    // attn partial = dot(updated_state_col, q).
    mul.rn.f32 %f27, %f23, %f7;
    fma.rn.f32 %f27, %f24, %f8, %f27;
    fma.rn.f32 %f27, %f25, %f9, %f27;
    fma.rn.f32 %f27, %f26, %f10, %f27;

    mov.b32 %r22, %f27;
    shfl.sync.bfly.b32 %r23, %r22, 16, 31, 0xffffffff;
    mov.b32 %f28, %r23;
    add.rn.f32 %f27, %f27, %f28;
    mov.b32 %r22, %f27;
    shfl.sync.bfly.b32 %r23, %r22, 8, 31, 0xffffffff;
    mov.b32 %f28, %r23;
    add.rn.f32 %f27, %f27, %f28;
    mov.b32 %r22, %f27;
    shfl.sync.bfly.b32 %r23, %r22, 4, 31, 0xffffffff;
    mov.b32 %f28, %r23;
    add.rn.f32 %f27, %f27, %f28;
    mov.b32 %r22, %f27;
    shfl.sync.bfly.b32 %r23, %r22, 2, 31, 0xffffffff;
    mov.b32 %f28, %r23;
    add.rn.f32 %f27, %f27, %f28;
    mov.b32 %r22, %f27;
    shfl.sync.bfly.b32 %r23, %r22, 1, 31, 0xffffffff;
    mov.b32 %f28, %r23;
    add.rn.f32 %f27, %f27, %f28;

    setp.ne.u32 %p3, %r7, 0;
    @%p3 bra GDN_DONE;

    mul.rn.f32 %f29, %f27, %f1;
    add.s64 %rd20, %rd7, %rd18;
    st.global.f32 [%rd20], %f29;

GDN_DONE:
    ret;
}
)ptx";

constexpr const char* kRecurrentPrepPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

.visible .entry q38_conv4_silu_roll(
    .param .u64 p_qkv,
    .param .u64 p_weight,
    .param .u64 p_state_in,
    .param .u64 p_conv_out,
    .param .u64 p_state_out,
    .param .u32 p_channels,
    .param .f32 p_log2e
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<20>;
    .reg .f32 %f<24>;

    ld.param.u64 %rd1, [p_qkv];
    ld.param.u64 %rd2, [p_weight];
    ld.param.u64 %rd3, [p_state_in];
    ld.param.u64 %rd4, [p_conv_out];
    ld.param.u64 %rd5, [p_state_out];
    ld.param.u32 %r1, [p_channels];
    ld.param.f32 %f1, [p_log2e];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra C4_DONE;

    mul.wide.u32 %rd6, %r5, 4;

    // state layout [3][channels]
    add.s64 %rd7, %rd3, %rd6;
    ld.global.f32 %f2, [%rd7];

    mul.wide.u32 %rd8, %r1, 4;
    add.s64 %rd9, %rd7, %rd8;
    ld.global.f32 %f3, [%rd9];
    add.s64 %rd10, %rd9, %rd8;
    ld.global.f32 %f4, [%rd10];

    add.s64 %rd11, %rd1, %rd6;
    ld.global.f32 %f5, [%rd11];

    // weight layout [4,channels] with conv step contiguous:
    // address = (channel*4 + step)*4
    shl.b32 %r6, %r5, 2;
    mul.wide.u32 %rd12, %r6, 4;
    add.s64 %rd13, %rd2, %rd12;
    ld.global.v4.f32 {%f6,%f7,%f8,%f9}, [%rd13];

    mul.rn.f32 %f10, %f2, %f6;
    fma.rn.f32 %f10, %f3, %f7, %f10;
    fma.rn.f32 %f10, %f4, %f8, %f10;
    fma.rn.f32 %f10, %f5, %f9, %f10;

    // SiLU(x)=x*sigmoid(x), sigmoid via exp2.
    neg.f32 %f11, %f10;
    mul.rn.f32 %f11, %f11, %f1;
    ex2.approx.f32 %f12, %f11;
    add.rn.f32 %f12, %f12, 0f3F800000;
    rcp.approx.f32 %f13, %f12;
    mul.rn.f32 %f14, %f10, %f13;

    add.s64 %rd14, %rd4, %rd6;
    st.global.f32 [%rd14], %f14;

    // roll state: [x1,x2,x3]
    add.s64 %rd15, %rd5, %rd6;
    st.global.f32 [%rd15], %f3;
    add.s64 %rd16, %rd15, %rd8;
    st.global.f32 [%rd16], %f4;
    add.s64 %rd17, %rd16, %rd8;
    st.global.f32 [%rd17], %f5;

C4_DONE:
    ret;
}

.visible .entry q38_qk_l2norm_128(
    .param .u64 p_conv,
    .param .u64 p_q,
    .param .u64 p_k,
    .param .f32 p_eps
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<16>;
    .shared .align 4 .b8 smem[512];

    ld.param.u64 %rd1, [p_conv];
    ld.param.u64 %rd2, [p_q];
    ld.param.u64 %rd3, [p_k];
    ld.param.f32 %f1, [p_eps];

    mov.u32 %r1, %ctaid.x; // 0..31
    mov.u32 %r2, %tid.x;   // 0..127
    setp.ge.u32 %p1, %r2, 128;
    @%p1 bra QKN_DONE;

    setp.lt.u32 %p2, %r1, 16;
    @%p2 mov.u32 %r3, %r1;
    @!%p2 sub.u32 %r3, %r1, 16;

    shl.b32 %r4, %r3, 7;   // head*128
    add.u32 %r4, %r4, %r2;
    @!%p2 add.u32 %r4, %r4, 2048;

    mul.wide.u32 %rd4, %r4, 4;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    mul.rn.f32 %f3, %f2, %f2;

    mov.u64 %rd6, smem;
    mul.wide.u32 %rd7, %r2, 4;
    add.s64 %rd8, %rd6, %rd7;
    st.shared.f32 [%rd8], %f3;
    bar.sync 0;

    // binary shared reduction 128 -> 1
    setp.ge.u32 %p3, %r2, 64;
    @%p3 bra R64_END;
    ld.shared.f32 %f4, [%rd8];
    ld.shared.f32 %f5, [%rd8+256];
    add.rn.f32 %f4, %f4, %f5;
    st.shared.f32 [%rd8], %f4;
R64_END:
    bar.sync 0;

    setp.ge.u32 %p4, %r2, 32;
    @%p4 bra R32_END;
    ld.shared.f32 %f4, [%rd8];
    ld.shared.f32 %f5, [%rd8+128];
    add.rn.f32 %f4, %f4, %f5;
    st.shared.f32 [%rd8], %f4;
R32_END:
    bar.sync 0;

    // warp shuffle reduction for first 32
    setp.ge.u32 %p5, %r2, 32;
    @%p5 bra QKN_SCALE;
    ld.shared.f32 %f6, [%rd8];
    mov.b32 %r8, %f6;
    shfl.sync.down.b32 %r9, %r8, 16, 31, 0xffffffff;
    mov.b32 %f7, %r9;
    add.rn.f32 %f6, %f6, %f7;
    mov.b32 %r8, %f6;
    shfl.sync.down.b32 %r9, %r8, 8, 31, 0xffffffff;
    mov.b32 %f7, %r9;
    add.rn.f32 %f6, %f6, %f7;
    mov.b32 %r8, %f6;
    shfl.sync.down.b32 %r9, %r8, 4, 31, 0xffffffff;
    mov.b32 %f7, %r9;
    add.rn.f32 %f6, %f6, %f7;
    mov.b32 %r8, %f6;
    shfl.sync.down.b32 %r9, %r8, 2, 31, 0xffffffff;
    mov.b32 %f7, %r9;
    add.rn.f32 %f6, %f6, %f7;
    mov.b32 %r8, %f6;
    shfl.sync.down.b32 %r9, %r8, 1, 31, 0xffffffff;
    mov.b32 %f7, %r9;
    add.rn.f32 %f6, %f6, %f7;

    setp.ne.u32 %p6, %r2, 0;
    @%p6 bra QKN_SCALE;
    add.rn.f32 %f6, %f6, %f1;
    rsqrt.approx.f32 %f8, %f6;
    st.shared.f32 [smem], %f8;

QKN_SCALE:
    bar.sync 0;
    ld.shared.f32 %f9, [smem];
    mul.rn.f32 %f10, %f2, %f9;

    // output head-contiguous without k offset
    shl.b32 %r10, %r3, 7;
    add.u32 %r10, %r10, %r2;
    mul.wide.u32 %rd9, %r10, 4;
    @%p2 add.s64 %rd10, %rd2, %rd9;
    @!%p2 add.s64 %rd10, %rd3, %rd9;
    st.global.f32 [%rd10], %f10;

QKN_DONE:
    ret;
}

.visible .entry q38_beta_gate_48(
    .param .u64 p_beta_raw,
    .param .u64 p_alpha_raw,
    .param .u64 p_dt,
    .param .u64 p_a,
    .param .u64 p_beta_out,
    .param .u64 p_gate_out,
    .param .f32 p_log2e,
    .param .f32 p_inv_log2e
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<24>;

    ld.param.u64 %rd1, [p_beta_raw];
    ld.param.u64 %rd2, [p_alpha_raw];
    ld.param.u64 %rd3, [p_dt];
    ld.param.u64 %rd4, [p_a];
    ld.param.u64 %rd5, [p_beta_out];
    ld.param.u64 %rd6, [p_gate_out];
    ld.param.f32 %f1, [p_log2e];
    ld.param.f32 %f2, [p_inv_log2e];

    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p1, %r1, 48;
    @%p1 bra BG_DONE;

    mul.wide.u32 %rd7, %r1, 4;
    add.s64 %rd8, %rd1, %rd7;
    add.s64 %rd9, %rd2, %rd7;
    add.s64 %rd10, %rd3, %rd7;
    add.s64 %rd11, %rd4, %rd7;

    ld.global.f32 %f3, [%rd8];
    ld.global.f32 %f4, [%rd9];
    ld.global.f32 %f5, [%rd10];
    ld.global.f32 %f6, [%rd11];

    // beta sigmoid
    neg.f32 %f7, %f3;
    mul.rn.f32 %f7, %f7, %f1;
    ex2.approx.f32 %f8, %f7;
    add.rn.f32 %f8, %f8, 0f3F800000;
    rcp.approx.f32 %f9, %f8;

    add.s64 %rd12, %rd5, %rd7;
    st.global.f32 [%rd12], %f9;

    // softplus(alpha + dt), stable branch for large positive values.
    add.rn.f32 %f10, %f4, %f5;
    setp.gt.f32 %p2, %f10, 0f41A00000;
    @%p2 mov.f32 %f14, %f10;
    @%p2 bra BG_SP_DONE;

    mul.rn.f32 %f11, %f10, %f1;
    ex2.approx.f32 %f12, %f11;
    add.rn.f32 %f12, %f12, 0f3F800000;
    lg2.approx.f32 %f13, %f12;
    mul.rn.f32 %f14, %f13, %f2;

BG_SP_DONE:
    mul.rn.f32 %f15, %f14, %f6;
    add.s64 %rd13, %rd6, %rd7;
    st.global.f32 [%rd13], %f15;

BG_DONE:
    ret;
}
)ptx";

constexpr const char* kRecurrentTailPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

.visible .entry q38_gated_rmsnorm_silu_128(
    .param .u64 p_input,
    .param .u64 p_weight,
    .param .u64 p_z,
    .param .u64 p_out,
    .param .f32 p_eps,
    .param .f32 p_log2e
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<20>;
    .reg .f32 %f<24>;
    .shared .align 4 .b8 smem[512];

    ld.param.u64 %rd1, [p_input];
    ld.param.u64 %rd2, [p_weight];
    ld.param.u64 %rd3, [p_z];
    ld.param.u64 %rd4, [p_out];
    ld.param.f32 %f1, [p_eps];
    ld.param.f32 %f2, [p_log2e];

    mov.u32 %r1, %ctaid.x;   // head 0..47
    mov.u32 %r2, %tid.x;     // element 0..127
    setp.ge.u32 %p1, %r2, 128;
    @%p1 bra GRN_DONE;

    shl.b32 %r3, %r1, 7;
    add.u32 %r4, %r3, %r2;
    mul.wide.u32 %rd5, %r4, 4;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd3, %rd5;
    ld.global.f32 %f3, [%rd6];
    ld.global.f32 %f4, [%rd7];

    mul.wide.u32 %rd8, %r2, 4;
    add.s64 %rd9, %rd2, %rd8;
    ld.global.f32 %f5, [%rd9];

    mul.rn.f32 %f6, %f3, %f3;
    mov.u64 %rd10, smem;
    add.s64 %rd11, %rd10, %rd8;
    st.shared.f32 [%rd11], %f6;
    bar.sync 0;

    setp.ge.u32 %p2, %r2, 64;
    @%p2 bra GRN_R64_END;
    ld.shared.f32 %f7, [%rd11];
    ld.shared.f32 %f8, [%rd11+256];
    add.rn.f32 %f7, %f7, %f8;
    st.shared.f32 [%rd11], %f7;
GRN_R64_END:
    bar.sync 0;

    setp.ge.u32 %p3, %r2, 32;
    @%p3 bra GRN_R32_END;
    ld.shared.f32 %f7, [%rd11];
    ld.shared.f32 %f8, [%rd11+128];
    add.rn.f32 %f7, %f7, %f8;
    st.shared.f32 [%rd11], %f7;
GRN_R32_END:
    bar.sync 0;

    setp.ge.u32 %p4, %r2, 32;
    @%p4 bra GRN_SCALE;
    ld.shared.f32 %f9, [%rd11];
    mov.b32 %r8, %f9;
    shfl.sync.down.b32 %r9, %r8, 16, 31, 0xffffffff;
    mov.b32 %f10, %r9;
    add.rn.f32 %f9, %f9, %f10;
    mov.b32 %r8, %f9;
    shfl.sync.down.b32 %r9, %r8, 8, 31, 0xffffffff;
    mov.b32 %f10, %r9;
    add.rn.f32 %f9, %f9, %f10;
    mov.b32 %r8, %f9;
    shfl.sync.down.b32 %r9, %r8, 4, 31, 0xffffffff;
    mov.b32 %f10, %r9;
    add.rn.f32 %f9, %f9, %f10;
    mov.b32 %r8, %f9;
    shfl.sync.down.b32 %r9, %r8, 2, 31, 0xffffffff;
    mov.b32 %f10, %r9;
    add.rn.f32 %f9, %f9, %f10;
    mov.b32 %r8, %f9;
    shfl.sync.down.b32 %r9, %r8, 1, 31, 0xffffffff;
    mov.b32 %f10, %r9;
    add.rn.f32 %f9, %f9, %f10;

    setp.ne.u32 %p5, %r2, 0;
    @%p5 bra GRN_SCALE;
    mov.u32 %r10, 128;
    cvt.rn.f32.u32 %f11, %r10;
    div.rn.f32 %f12, %f9, %f11;
    add.rn.f32 %f12, %f12, %f1;
    rsqrt.approx.f32 %f13, %f12;
    st.shared.f32 [smem], %f13;

GRN_SCALE:
    bar.sync 0;
    ld.shared.f32 %f14, [smem];

    // normalized = input * inv_rms * weight
    mul.rn.f32 %f15, %f3, %f14;
    mul.rn.f32 %f15, %f15, %f5;

    // silu(z)
    neg.f32 %f16, %f4;
    mul.rn.f32 %f16, %f16, %f2;
    ex2.approx.f32 %f17, %f16;
    add.rn.f32 %f17, %f17, 0f3F800000;
    rcp.approx.f32 %f18, %f17;
    mul.rn.f32 %f19, %f4, %f18;

    mul.rn.f32 %f20, %f15, %f19;
    add.s64 %rd12, %rd4, %rd5;
    st.global.f32 [%rd12], %f20;

GRN_DONE:
    ret;
}

.visible .entry q38_add_residual_f32(
    .param .u64 p_x,
    .param .u64 p_residual,
    .param .u64 p_out,
    .param .u32 p_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [p_x];
    ld.param.u64 %rd2, [p_residual];
    ld.param.u64 %rd3, [p_out];
    ld.param.u32 %r1, [p_n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra AR_DONE;

    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.rn.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;

AR_DONE:
    ret;
}

.visible .entry q38_ffn_silu_mul_f32(
    .param .u64 p_gate,
    .param .u64 p_up,
    .param .u64 p_out,
    .param .u32 p_n,
    .param .f32 p_log2e
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<12>;

    ld.param.u64 %rd1, [p_gate];
    ld.param.u64 %rd2, [p_up];
    ld.param.u64 %rd3, [p_out];
    ld.param.u32 %r1, [p_n];
    ld.param.f32 %f1, [p_log2e];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra FSM_DONE;

    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;

    ld.global.f32 %f2, [%rd5];
    ld.global.f32 %f3, [%rd6];

    neg.f32 %f4, %f2;
    mul.rn.f32 %f4, %f4, %f1;
    ex2.approx.f32 %f5, %f4;
    add.rn.f32 %f5, %f5, 0f3F800000;
    rcp.approx.f32 %f6, %f5;
    mul.rn.f32 %f7, %f2, %f6;
    mul.rn.f32 %f8, %f7, %f3;

    st.global.f32 [%rd7], %f8;

FSM_DONE:
    ret;
}
)ptx";

constexpr const char* kQ6KDequantPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

.visible .entry q38_dequant_q6k_block(
    .param .u64 p_block,
    .param .u64 p_out
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [p_block];
    ld.param.u64 %rd2, [p_out];

    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p1, %r1, 256;
    @%p1 bra Q6D_DONE;

    shr.u32 %r2, %r1, 7;       // half 0..1
    and.b32 %r3, %r1, 127;     // within half
    shr.u32 %r4, %r3, 5;       // quarter 0..3
    and.b32 %r5, %r3, 31;      // l 0..31
    shr.u32 %r6, %r5, 4;       // is 0..1

    // ql index = half*64 + (quarter&1)*32 + l
    shl.b32 %r7, %r2, 6;
    and.b32 %r8, %r4, 1;
    shl.b32 %r8, %r8, 5;
    add.u32 %r9, %r7, %r8;
    add.u32 %r9, %r9, %r5;
    cvt.u64.u32 %rd3, %r9;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.u8 %r10, [%rd4];

    // qh index = half*32 + l
    shl.b32 %r11, %r2, 5;
    add.u32 %r11, %r11, %r5;
    cvt.u64.u32 %rd5, %r11;
    add.s64 %rd6, %rd1, 128;
    add.s64 %rd7, %rd6, %rd5;
    ld.global.u8 %r12, [%rd7];

    setp.ge.u32 %p2, %r4, 2;
    @%p2 shr.u32 %r13, %r10, 4;
    @!%p2 and.b32 %r13, %r10, 15;
    and.b32 %r13, %r13, 15;

    shl.b32 %r14, %r4, 1;
    shr.u32 %r15, %r12, %r14;
    and.b32 %r15, %r15, 3;
    shl.b32 %r15, %r15, 4;
    or.b32 %r16, %r13, %r15;
    sub.s32 %r16, %r16, 32;

    // scale index = half*8 + quarter*2 + is
    shl.b32 %r17, %r2, 3;
    shl.b32 %r18, %r4, 1;
    add.u32 %r17, %r17, %r18;
    add.u32 %r17, %r17, %r6;
    cvt.u64.u32 %rd8, %r17;
    add.s64 %rd9, %rd1, 192;
    add.s64 %rd10, %rd9, %rd8;
    ld.global.s8 %r19, [%rd10];

    ld.global.b16 %r20, [%rd1+208];
    cvt.f32.f16 %f1, %r20;
    cvt.rn.f32.s32 %f2, %r19;
    cvt.rn.f32.s32 %f3, %r16;
    mul.rn.f32 %f4, %f1, %f2;
    mul.rn.f32 %f5, %f4, %f3;

    mul.wide.u32 %rd11, %r1, 4;
    add.s64 %rd12, %rd2, %rd11;
    st.global.f32 [%rd12], %f5;

Q6D_DONE:
    ret;
}
)ptx";

constexpr const char* kQ6KGemvPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

// Native Q6_K x F32 GEMV.
// 4 warps/CTA, one output row per warp.
// 16 x 2-lane subgroups map 1:1 to the 16 signed-scale groups.
// Each lane handles 8 consecutive weights; scale*d is applied only after the
// 2-lane group reduction.
.visible .entry q38_q6k_gemv_f32(
    .param .u64 p_weights,
    .param .u64 p_x,
    .param .u64 p_y,
    .param .u32 p_cols,
    .param .u32 p_rows
)
{
    .reg .pred %p<16>;
    .reg .b32 %r<128>;
    .reg .b64 %rd<36>;
    .reg .f32 %f<40>;

    ld.param.u64 %rd1, [p_weights];
    ld.param.u64 %rd2, [p_x];
    ld.param.u64 %rd3, [p_y];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_rows];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    shr.u32 %r5, %r4, 5;       // warp id 0..3
    and.b32 %r6, %r4, 31;      // lane 0..31
    shl.b32 %r7, %r3, 2;
    add.u32 %r7, %r7, %r5;     // output row
    setp.ge.u32 %p1, %r7, %r2;
    @%p1 bra Q6_DONE;

    shr.u32 %r8, %r6, 1;       // group 0..15
    and.b32 %r9, %r6, 1;       // sublane 0..1

    shr.u32 %r10, %r1, 8;      // blocks_per_row
    mul.lo.u32 %r11, %r10, 210;
    mul.wide.u32 %rd4, %r7, %r11;
    add.s64 %rd5, %rd1, %rd4;

    mov.u32 %r12, 0;            // block index
    mov.f32 %f30, 0f00000000;   // valid in even lanes

Q6_BLOCK_LOOP:
    setp.ge.u32 %p2, %r12, %r10;
    @%p2 bra Q6_BLOCKS_DONE;

    mul.wide.u32 %rd6, %r12, 210;
    add.s64 %rd7, %rd5, %rd6;

    // group decomposition.
    shr.u32 %r13, %r8, 3;      // half 0..1
    and.b32 %r14, %r8, 7;      // group in half 0..7
    shr.u32 %r15, %r14, 1;     // 0..3 -> qh shift/2
    and.b32 %r16, %r15, 1;     // ql segment 0 or 1
    shl.b32 %r17, %r16, 5;     // ql +0 / +32
    shl.b32 %r18, %r13, 6;     // half ql +0 / +64
    add.u32 %r17, %r17, %r18;
    shl.b32 %r19, %r9, 3;      // sublane * 8
    and.b32 %r22, %r14, 1;      // which 16-value scale group in pair
    shl.b32 %r22, %r22, 4;      // +0 / +16
    add.u32 %r17, %r17, %r22;
    add.u32 %r17, %r17, %r19;

    // ql: 8 bytes. Q6_K blocks are 210 bytes, so odd blocks are only
    // 2-byte aligned. Use b16 loads and assemble b32 values in registers.
    cvt.u64.u32 %rd8, %r17;
    add.s64 %rd9, %rd7, %rd8;
    ld.global.b16 %r40, [%rd9+0];
    ld.global.b16 %r41, [%rd9+2];
    ld.global.b16 %r42, [%rd9+4];
    ld.global.b16 %r43, [%rd9+6];
    and.b32 %r40, %r40, 0x0000ffff;
    and.b32 %r41, %r41, 0x0000ffff;
    and.b32 %r42, %r42, 0x0000ffff;
    and.b32 %r43, %r43, 0x0000ffff;
    shl.b32 %r84, %r41, 16;
    or.b32 %r40, %r40, %r84;
    shl.b32 %r84, %r43, 16;
    or.b32 %r41, %r42, %r84;

    // qh: half*32 + sublane*8 + group-local 16 offset.
    shl.b32 %r20, %r13, 5;
    add.u32 %r20, %r20, %r22;
    add.u32 %r20, %r20, %r19;
    // qh byte position follows the same l=0..31 split as ql.
    cvt.u64.u32 %rd10, %r20;
    add.s64 %rd11, %rd7, 128;
    add.s64 %rd12, %rd11, %rd10;
    ld.global.b16 %r42, [%rd12+0];
    ld.global.b16 %r43, [%rd12+2];
    ld.global.b16 %r85, [%rd12+4];
    ld.global.b16 %r86, [%rd12+6];
    and.b32 %r42, %r42, 0x0000ffff;
    and.b32 %r43, %r43, 0x0000ffff;
    and.b32 %r85, %r85, 0x0000ffff;
    and.b32 %r86, %r86, 0x0000ffff;
    shl.b32 %r84, %r43, 16;
    or.b32 %r42, %r42, %r84;
    shl.b32 %r84, %r86, 16;
    or.b32 %r43, %r85, %r84;

    // nibble select: groups 4..7 use ql high nibble.
    setp.ge.u32 %p3, %r14, 4;
    @!%p3 and.b32 %r44, %r40, 0x0f0f0f0f;
    @!%p3 and.b32 %r45, %r41, 0x0f0f0f0f;
    @%p3 shr.u32 %r44, %r40, 4;
    @%p3 shr.u32 %r45, %r41, 4;
    @%p3 and.b32 %r44, %r44, 0x0f0f0f0f;
    @%p3 and.b32 %r45, %r45, 0x0f0f0f0f;

    // Upper 2 bits: shift 0,2,4,6 according to group pair.
    shl.b32 %r21, %r15, 1;
    shr.u32 %r46, %r42, %r21;
    shr.u32 %r47, %r43, %r21;
    and.b32 %r46, %r46, 0x03030303;
    and.b32 %r47, %r47, 0x03030303;
    shl.b32 %r46, %r46, 4;
    shl.b32 %r47, %r47, 4;
    or.b32 %r48, %r44, %r46;
    or.b32 %r49, %r45, %r47;   // packed unsigned q 0..63

    // Convert four packed unsigned 6-bit bytes to signed (q-32) bytes.
    not.b32 %r50, %r48;
    and.b32 %r50, %r50, 0x20202020;
    shl.b32 %r51, %r50, 1;
    shl.b32 %r52, %r50, 2;
    or.b32 %r50, %r50, %r51;
    or.b32 %r50, %r50, %r52;
    and.b32 %r53, %r48, 0x1f1f1f1f;
    or.b32 %r54, %r53, %r50;

    not.b32 %r55, %r49;
    and.b32 %r55, %r55, 0x20202020;
    shl.b32 %r56, %r55, 1;
    shl.b32 %r57, %r55, 2;
    or.b32 %r55, %r55, %r56;
    or.b32 %r55, %r55, %r57;
    and.b32 %r58, %r49, 0x1f1f1f1f;
    or.b32 %r59, %r58, %r55;

    // Activation base = block*256 + group*16 + sublane*8.
    shl.b32 %r60, %r12, 8;
    shl.b32 %r61, %r8, 4;
    add.u32 %r62, %r60, %r61;
    add.u32 %r62, %r62, %r19;
    mul.wide.u32 %rd13, %r62, 4;
    add.s64 %rd14, %rd2, %rd13;

    ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [%rd14+0];
    ld.global.v4.f32 {%f5,%f6,%f7,%f8}, [%rd14+16];

    mov.f32 %f20, 0f00000000;

    // packed byte q0..q7 -> F32 dot.
    shl.b32 %r70, %r54, 24;
    shr.s32 %r70, %r70, 24;
    cvt.rn.f32.s32 %f10, %r70;
    fma.rn.f32 %f20, %f10, %f1, %f20;

    shr.u32 %r71, %r54, 8;
    shl.b32 %r71, %r71, 24;
    shr.s32 %r71, %r71, 24;
    cvt.rn.f32.s32 %f11, %r71;
    fma.rn.f32 %f20, %f11, %f2, %f20;

    shr.u32 %r72, %r54, 16;
    shl.b32 %r72, %r72, 24;
    shr.s32 %r72, %r72, 24;
    cvt.rn.f32.s32 %f12, %r72;
    fma.rn.f32 %f20, %f12, %f3, %f20;

    shr.u32 %r73, %r54, 24;
    cvt.rn.f32.s32 %f13, %r73;
    fma.rn.f32 %f20, %f13, %f4, %f20;

    shl.b32 %r74, %r59, 24;
    shr.s32 %r74, %r74, 24;
    cvt.rn.f32.s32 %f14, %r74;
    fma.rn.f32 %f20, %f14, %f5, %f20;

    shr.u32 %r75, %r59, 8;
    shl.b32 %r75, %r75, 24;
    shr.s32 %r75, %r75, 24;
    cvt.rn.f32.s32 %f15, %r75;
    fma.rn.f32 %f20, %f15, %f6, %f20;

    shr.u32 %r76, %r59, 16;
    shl.b32 %r76, %r76, 24;
    shr.s32 %r76, %r76, 24;
    cvt.rn.f32.s32 %f16, %r76;
    fma.rn.f32 %f20, %f16, %f7, %f20;

    shr.u32 %r77, %r59, 24;
    cvt.rn.f32.s32 %f17, %r77;
    fma.rn.f32 %f20, %f17, %f8, %f20;

    // 2-lane subgroup reduction.
    mov.b32 %r80, %f20;
    shfl.sync.bfly.b32 %r81, %r80, 1, 31, 0xffffffff;
    mov.b32 %f21, %r81;
    add.rn.f32 %f20, %f20, %f21;

    setp.ne.u32 %p4, %r9, 0;
    @%p4 bra Q6_NEXT_BLOCK;

    // d at +208, signed scale[group] at +192.
    ld.global.b16 %r82, [%rd7+208];
    cvt.f32.f16 %f22, %r82;
    cvt.u64.u32 %rd15, %r8;
    add.s64 %rd16, %rd7, 192;
    add.s64 %rd17, %rd16, %rd15;
    ld.global.s8 %r83, [%rd17];
    cvt.rn.f32.s32 %f23, %r83;
    mul.rn.f32 %f24, %f22, %f23;
    fma.rn.f32 %f30, %f20, %f24, %f30;

Q6_NEXT_BLOCK:
    add.u32 %r12, %r12, 1;
    bra Q6_BLOCK_LOOP;

Q6_BLOCKS_DONE:
    // subgroup leaders are even lanes: reduce offsets 16,8,4,2.
    mov.b32 %r90, %f30;
    shfl.sync.down.b32 %r91, %r90, 16, 31, 0xffffffff;
    mov.b32 %f31, %r91;
    add.rn.f32 %f30, %f30, %f31;
    mov.b32 %r90, %f30;
    shfl.sync.down.b32 %r91, %r90, 8, 31, 0xffffffff;
    mov.b32 %f31, %r91;
    add.rn.f32 %f30, %f30, %f31;
    mov.b32 %r90, %f30;
    shfl.sync.down.b32 %r91, %r90, 4, 31, 0xffffffff;
    mov.b32 %f31, %r91;
    add.rn.f32 %f30, %f30, %f31;
    mov.b32 %r90, %f30;
    shfl.sync.down.b32 %r91, %r90, 2, 31, 0xffffffff;
    mov.b32 %f31, %r91;
    add.rn.f32 %f30, %f30, %f31;

    setp.ne.u32 %p5, %r6, 0;
    @%p5 bra Q6_DONE;

    mul.wide.u32 %rd18, %r7, 4;
    add.s64 %rd19, %rd3, %rd18;
    st.global.f32 [%rd19], %f30;

Q6_DONE:
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
    .reg .b32 %r<28>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<12>;

    ld.param.u64 %rd1, [p_block];
    ld.param.u64 %rd2, [p_out];

    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p1, %r1, 256;
    @%p1 bra Q4K_DONE;

    // d and dmin
    ld.global.b16 %r24, [%rd1+0];
    ld.global.b16 %r25, [%rd1+2];
    cvt.f32.f16 %f1, %r24;
    cvt.f32.f16 %f2, %r25;

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

constexpr const char* kQ4KGemvPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

.visible .entry q38_q4k_gemv_f32(
    .param .u64 p_weights,
    .param .u64 p_x,
    .param .u64 p_y,
    .param .u32 p_cols,
    .param .u32 p_rows
)
{
    .reg .pred %p<12>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<24>;
    .reg .f32 %f<16>;

    ld.param.u64 %rd1, [p_weights];
    ld.param.u64 %rd2, [p_x];
    ld.param.u64 %rd3, [p_y];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_rows];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    setp.ge.u32 %p1, %r3, %r2;
    @%p1 bra Q4G_DONE;
    setp.ge.u32 %p2, %r4, 32;
    @%p2 bra Q4G_DONE;

    shr.u32 %r5, %r1, 8;       // blocks_per_row = cols/256
    mul.lo.u32 %r6, %r5, 144;  // bytes_per_row
    mul.wide.u32 %rd4, %r3, %r6;
    add.s64 %rd5, %rd1, %rd4;

    mov.u32 %r7, 0;
    mov.f32 %f10, 0f00000000;

Q4G_BLOCK_LOOP:
    setp.ge.u32 %p3, %r7, %r5;
    @%p3 bra Q4G_BLOCKS_DONE;

    mul.wide.u32 %rd6, %r7, 144;
    add.s64 %rd7, %rd5, %rd6;

    ld.global.b16 %r40, [%rd7+0];
    ld.global.b16 %r41, [%rd7+2];
    cvt.f32.f16 %f1, %r40;
    cvt.f32.f16 %f2, %r41;

    mov.u32 %r8, 0;

Q4G_GROUP_LOOP:
    setp.ge.u32 %p4, %r8, 8;
    @%p4 bra Q4G_GROUPS_DONE;

    add.s64 %rd8, %rd7, 4;
    setp.lt.u32 %p5, %r8, 4;
    @%p5 bra Q4G_SCALE_LOW;

    add.u32 %r9, %r8, 4;
    cvt.u64.u32 %rd9, %r9;
    add.s64 %rd10, %rd8, %rd9;
    ld.global.u8 %r10, [%rd10];

    sub.u32 %r11, %r8, 4;
    cvt.u64.u32 %rd11, %r11;
    add.s64 %rd12, %rd8, %rd11;
    ld.global.u8 %r12, [%rd12];

    cvt.u64.u32 %rd13, %r8;
    add.s64 %rd14, %rd8, %rd13;
    ld.global.u8 %r13, [%rd14];

    and.b32 %r14, %r10, 15;
    shr.u32 %r15, %r12, 6;
    shl.b32 %r15, %r15, 4;
    or.b32 %r16, %r14, %r15;

    shr.u32 %r17, %r10, 4;
    shr.u32 %r18, %r13, 6;
    shl.b32 %r18, %r18, 4;
    or.b32 %r19, %r17, %r18;
    bra Q4G_SCALE_READY;

Q4G_SCALE_LOW:
    cvt.u64.u32 %rd9, %r8;
    add.s64 %rd10, %rd8, %rd9;
    ld.global.u8 %r10, [%rd10];
    and.b32 %r16, %r10, 63;

    add.u32 %r11, %r8, 4;
    cvt.u64.u32 %rd11, %r11;
    add.s64 %rd12, %rd8, %rd11;
    ld.global.u8 %r12, [%rd12];
    and.b32 %r19, %r12, 63;

Q4G_SCALE_READY:
    cvt.rn.f32.u32 %f3, %r16;
    cvt.rn.f32.u32 %f4, %r19;
    mul.rn.f32 %f5, %f1, %f3;
    mul.rn.f32 %f6, %f2, %f4;

    // One qs byte serves low/high adjacent groups. Load on even, reuse on odd.
    shr.u32 %r20, %r8, 1;
    shl.b32 %r20, %r20, 5;
    add.u32 %r20, %r20, %r4;
    cvt.u64.u32 %rd15, %r20;
    add.s64 %rd16, %rd7, 16;
    add.s64 %rd17, %rd16, %rd15;

    and.b32 %r21, %r8, 1;
    setp.eq.u32 %p6, %r21, 0;
    @%p6 ld.global.u8 %r22, [%rd17];
    @%p6 and.b32 %r23, %r22, 15;
    @!%p6 shr.u32 %r23, %r22, 4;

    cvt.rn.f32.u32 %f7, %r23;
    mul.rn.f32 %f8, %f5, %f7;
    sub.rn.f32 %f9, %f8, %f6;

    shl.b32 %r24, %r7, 8;
    shl.b32 %r25, %r8, 5;
    add.u32 %r26, %r24, %r25;
    add.u32 %r26, %r26, %r4;
    mul.wide.u32 %rd18, %r26, 4;
    add.s64 %rd19, %rd2, %rd18;
    ld.global.f32 %f11, [%rd19];

    fma.rn.f32 %f10, %f9, %f11, %f10;

    add.u32 %r8, %r8, 1;
    bra Q4G_GROUP_LOOP;

Q4G_GROUPS_DONE:
    add.u32 %r7, %r7, 1;
    bra Q4G_BLOCK_LOOP;

Q4G_BLOCKS_DONE:
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 16, 31, 0xffffffff;
    mov.b32 %f12, %r31;
    add.rn.f32 %f10, %f10, %f12;
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 8, 31, 0xffffffff;
    mov.b32 %f12, %r31;
    add.rn.f32 %f10, %f10, %f12;
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 4, 31, 0xffffffff;
    mov.b32 %f12, %r31;
    add.rn.f32 %f10, %f10, %f12;
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 2, 31, 0xffffffff;
    mov.b32 %f12, %r31;
    add.rn.f32 %f10, %f10, %f12;
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 1, 31, 0xffffffff;
    mov.b32 %f12, %r31;
    add.rn.f32 %f10, %f10, %f12;

    setp.eq.u32 %p7, %r4, 0;
    mul.wide.u32 %rd20, %r3, 4;
    add.s64 %rd21, %rd3, %rd20;
    @%p7 st.global.f32 [%rd21], %f10;

Q4G_DONE:
    ret;
}
)ptx";

constexpr const char* kIQ4XSDequantPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

.visible .entry q38_dequant_iq4xs_block(
    .param .u64 p_block,
    .param .u64 p_out
)
{
    .reg .pred %p<10>;
    .reg .b32 %r<40>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<10>;

    ld.param.u64 %rd1, [p_block];
    ld.param.u64 %rd2, [p_out];

    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p1, %r1, 256;
    @%p1 bra IQ4_DONE;

    // group 0..7, lane 0..31
    shr.u32 %r2, %r1, 5;
    and.b32 %r3, %r1, 31;

    // d (fp16)
    ld.global.b16 %r30, [%rd1+0];
    cvt.f32.f16 %f1, %r30;

    // 6-bit group scale:
    // low 4 bits packed two-per-byte at +4
    // high 2 bits packed in uint16 at +2
    ld.global.b16 %r31, [%rd1+2];

    shr.u32 %r4, %r2, 1;
    cvt.u64.u32 %rd3, %r4;
    add.s64 %rd4, %rd1, 4;
    add.s64 %rd5, %rd4, %rd3;
    ld.global.u8 %r5, [%rd5];

    and.b32 %r6, %r2, 1;
    shl.b32 %r6, %r6, 2;
    shr.u32 %r7, %r5, %r6;
    and.b32 %r7, %r7, 15;

    shl.b32 %r8, %r2, 1;
    shr.u32 %r9, %r31, %r8;
    and.b32 %r9, %r9, 3;
    shl.b32 %r9, %r9, 4;
    or.b32 %r10, %r7, %r9;
    sub.s32 %r11, %r10, 32;

    cvt.rn.f32.s32 %f2, %r11;
    mul.rn.f32 %f3, %f1, %f2;

    // qs starts at +8, 16 bytes per 32-value group.
    and.b32 %r12, %r3, 15;
    shl.b32 %r13, %r2, 4;
    add.u32 %r13, %r13, %r12;
    cvt.u64.u32 %rd6, %r13;
    add.s64 %rd7, %rd1, 8;
    add.s64 %rd8, %rd7, %rd6;
    ld.global.u8 %r14, [%rd8];

    setp.lt.u32 %p2, %r3, 16;
    @%p2 and.b32 %r15, %r14, 15;
    @!%p2 shr.u32 %r15, %r14, 4;

    // Non-linear IQ4 codebook.
    mov.s32 %r16, 0;
    setp.eq.u32 %p3, %r15, 0;  @%p3 mov.s32 %r16, -127;
    setp.eq.u32 %p3, %r15, 1;  @%p3 mov.s32 %r16, -104;
    setp.eq.u32 %p3, %r15, 2;  @%p3 mov.s32 %r16, -83;
    setp.eq.u32 %p3, %r15, 3;  @%p3 mov.s32 %r16, -65;
    setp.eq.u32 %p3, %r15, 4;  @%p3 mov.s32 %r16, -49;
    setp.eq.u32 %p3, %r15, 5;  @%p3 mov.s32 %r16, -35;
    setp.eq.u32 %p3, %r15, 6;  @%p3 mov.s32 %r16, -22;
    setp.eq.u32 %p3, %r15, 7;  @%p3 mov.s32 %r16, -10;
    setp.eq.u32 %p3, %r15, 8;  @%p3 mov.s32 %r16, 1;
    setp.eq.u32 %p3, %r15, 9;  @%p3 mov.s32 %r16, 13;
    setp.eq.u32 %p3, %r15, 10; @%p3 mov.s32 %r16, 25;
    setp.eq.u32 %p3, %r15, 11; @%p3 mov.s32 %r16, 38;
    setp.eq.u32 %p3, %r15, 12; @%p3 mov.s32 %r16, 53;
    setp.eq.u32 %p3, %r15, 13; @%p3 mov.s32 %r16, 69;
    setp.eq.u32 %p3, %r15, 14; @%p3 mov.s32 %r16, 89;
    setp.eq.u32 %p3, %r15, 15; @%p3 mov.s32 %r16, 113;

    cvt.rn.f32.s32 %f4, %r16;
    mul.rn.f32 %f5, %f3, %f4;

    mul.wide.u32 %rd9, %r1, 4;
    add.s64 %rd10, %rd2, %rd9;
    st.global.f32 [%rd10], %f5;

IQ4_DONE:
    ret;
}
)ptx";

constexpr const char* kIQ4XSGemvPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

.visible .entry q38_iq4xs_gemv_f32(
    .param .u64 p_weights,
    .param .u64 p_x,
    .param .u64 p_y,
    .param .u32 p_cols,
    .param .u32 p_rows
)
{
    .reg .pred %p<16>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<24>;
    .reg .f32 %f<16>;

    ld.param.u64 %rd1, [p_weights];
    ld.param.u64 %rd2, [p_x];
    ld.param.u64 %rd3, [p_y];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_rows];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    setp.ge.u32 %p1, %r3, %r2;
    @%p1 bra IQG_DONE;
    setp.ge.u32 %p2, %r4, 32;
    @%p2 bra IQG_DONE;

    // 256 weights per 136-byte IQ4_XS block.
    shr.u32 %r5, %r1, 8;
    mul.lo.u32 %r6, %r5, 136;
    mul.wide.u32 %rd4, %r3, %r6;
    add.s64 %rd5, %rd1, %rd4;

    // Packed codebook bytes:
    // [-127,-104,-83,-65], [-49,-35,-22,-10], [1,13,25,38], [53,69,89,113]
    mov.u32 %r50, 0xBFAD9881;
    mov.u32 %r51, 0xF6EADDCF;
    mov.u32 %r52, 0x26190D01;
    mov.u32 %r53, 0x71594535;

    mov.u32 %r7, 0;
    mov.f32 %f10, 0f00000000;

IQG_BLOCK_LOOP:
    setp.ge.u32 %p3, %r7, %r5;
    @%p3 bra IQG_BLOCKS_DONE;

    mul.wide.u32 %rd6, %r7, 136;
    add.s64 %rd7, %rd5, %rd6;

    // Cache d and scale high bits once per block.
    ld.global.b16 %r40, [%rd7+0];
    ld.global.b16 %r41, [%rd7+2];
    cvt.f32.f16 %f1, %r40;

    mov.u32 %r8, 0;

IQG_GROUP_LOOP:
    setp.ge.u32 %p4, %r8, 8;
    @%p4 bra IQG_GROUPS_DONE;

    // 6-bit scale = 4 low bits from scales_l + 2 high bits from scales_h.
    shr.u32 %r9, %r8, 1;
    cvt.u64.u32 %rd8, %r9;
    add.s64 %rd9, %rd7, 4;
    add.s64 %rd10, %rd9, %rd8;

    and.b32 %r10, %r8, 1;
    setp.eq.u32 %p5, %r10, 0;
    @%p5 ld.global.u8 %r42, [%rd10];

    shl.b32 %r11, %r10, 2;
    shr.u32 %r12, %r42, %r11;
    and.b32 %r12, %r12, 15;

    shl.b32 %r13, %r8, 1;
    shr.u32 %r14, %r41, %r13;
    and.b32 %r14, %r14, 3;
    shl.b32 %r14, %r14, 4;
    or.b32 %r15, %r12, %r14;
    sub.s32 %r16, %r15, 32;

    cvt.rn.f32.s32 %f2, %r16;
    mul.rn.f32 %f3, %f1, %f2;

    // One byte encodes two weights in each 32-value group.
    and.b32 %r17, %r4, 15;
    shl.b32 %r18, %r8, 4;
    add.u32 %r18, %r18, %r17;
    cvt.u64.u32 %rd11, %r18;
    add.s64 %rd12, %rd7, 8;
    add.s64 %rd13, %rd12, %rd11;
    ld.global.u8 %r19, [%rd13];

    setp.lt.u32 %p6, %r4, 16;
    @%p6 and.b32 %r20, %r19, 15;
    @!%p6 shr.u32 %r20, %r19, 4;

    // Register-only IQ4 nonlinear table lookup.
    shr.u32 %r21, %r20, 2;
    mov.u32 %r22, %r50;
    setp.eq.u32 %p7, %r21, 1;
    @%p7 mov.u32 %r22, %r51;
    setp.eq.u32 %p8, %r21, 2;
    @%p8 mov.u32 %r22, %r52;
    setp.eq.u32 %p9, %r21, 3;
    @%p9 mov.u32 %r22, %r53;

    and.b32 %r23, %r20, 3;
    shl.b32 %r23, %r23, 3;
    shr.u32 %r24, %r22, %r23;
    and.b32 %r24, %r24, 255;
    shl.b32 %r24, %r24, 24;
    shr.s32 %r24, %r24, 24;

    cvt.rn.f32.s32 %f4, %r24;
    mul.rn.f32 %f5, %f3, %f4;

    shl.b32 %r25, %r7, 8;
    shl.b32 %r26, %r8, 5;
    add.u32 %r27, %r25, %r26;
    add.u32 %r27, %r27, %r4;
    mul.wide.u32 %rd14, %r27, 4;
    add.s64 %rd15, %rd2, %rd14;
    ld.global.f32 %f6, [%rd15];

    fma.rn.f32 %f10, %f5, %f6, %f10;

    add.u32 %r8, %r8, 1;
    bra IQG_GROUP_LOOP;

IQG_GROUPS_DONE:
    add.u32 %r7, %r7, 1;
    bra IQG_BLOCK_LOOP;

IQG_BLOCKS_DONE:
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 16, 31, 0xffffffff;
    mov.b32 %f11, %r31;
    add.rn.f32 %f10, %f10, %f11;

    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 8, 31, 0xffffffff;
    mov.b32 %f11, %r31;
    add.rn.f32 %f10, %f10, %f11;

    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 4, 31, 0xffffffff;
    mov.b32 %f11, %r31;
    add.rn.f32 %f10, %f10, %f11;

    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 2, 31, 0xffffffff;
    mov.b32 %f11, %r31;
    add.rn.f32 %f10, %f10, %f11;

    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 1, 31, 0xffffffff;
    mov.b32 %f11, %r31;
    add.rn.f32 %f10, %f10, %f11;

    setp.eq.u32 %p10, %r4, 0;
    mul.wide.u32 %rd16, %r3, 4;
    add.s64 %rd17, %rd3, %rd16;
    @%p10 st.global.f32 [%rd17], %f10;

IQG_DONE:
    ret;
}
)ptx";


constexpr const char* kIQ4XSGemv4WarpPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

.visible .entry q38_iq4xs_gemv_f32_4warp(
    .param .u64 p_weights,
    .param .u64 p_x,
    .param .u64 p_y,
    .param .u32 p_cols,
    .param .u32 p_rows
)
{
    .reg .pred %p<16>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<24>;
    .reg .f32 %f<16>;

    ld.param.u64 %rd1, [p_weights];
    ld.param.u64 %rd2, [p_x];
    ld.param.u64 %rd3, [p_y];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_rows];

    // Four warps per CTA, one independent output row per warp.
    // SM86 is limited to 16 resident CTAs but 48 resident warps, so
    // 1-warp CTAs can cap occupancy at 16/48 before register limits.
    mov.u32 %r32, %ctaid.x;
    mov.u32 %r33, %tid.x;
    shr.u32 %r34, %r33, 5;      // warp id 0..3
    and.b32 %r4, %r33, 31;      // lane 0..31
    shl.b32 %r3, %r32, 2;
    add.u32 %r3, %r3, %r34;     // row = cta*4 + warp
    setp.ge.u32 %p1, %r3, %r2;
    @%p1 bra IQG_DONE;

    // 256 weights per 136-byte IQ4_XS block.
    shr.u32 %r5, %r1, 8;
    mul.lo.u32 %r6, %r5, 136;
    mul.wide.u32 %rd4, %r3, %r6;
    add.s64 %rd5, %rd1, %rd4;

    // Packed codebook bytes:
    // [-127,-104,-83,-65], [-49,-35,-22,-10], [1,13,25,38], [53,69,89,113]
    mov.u32 %r50, 0xBFAD9881;
    mov.u32 %r51, 0xF6EADDCF;
    mov.u32 %r52, 0x26190D01;
    mov.u32 %r53, 0x71594535;

    mov.u32 %r7, 0;
    mov.f32 %f10, 0f00000000;

IQG_BLOCK_LOOP:
    setp.ge.u32 %p3, %r7, %r5;
    @%p3 bra IQG_BLOCKS_DONE;

    mul.wide.u32 %rd6, %r7, 136;
    add.s64 %rd7, %rd5, %rd6;

    // Cache d and scale high bits once per block.
    ld.global.b16 %r40, [%rd7+0];
    ld.global.b16 %r41, [%rd7+2];
    cvt.f32.f16 %f1, %r40;

    mov.u32 %r8, 0;

IQG_GROUP_LOOP:
    setp.ge.u32 %p4, %r8, 8;
    @%p4 bra IQG_GROUPS_DONE;

    // 6-bit scale = 4 low bits from scales_l + 2 high bits from scales_h.
    shr.u32 %r9, %r8, 1;
    cvt.u64.u32 %rd8, %r9;
    add.s64 %rd9, %rd7, 4;
    add.s64 %rd10, %rd9, %rd8;

    and.b32 %r10, %r8, 1;
    setp.eq.u32 %p5, %r10, 0;
    @%p5 ld.global.u8 %r42, [%rd10];

    shl.b32 %r11, %r10, 2;
    shr.u32 %r12, %r42, %r11;
    and.b32 %r12, %r12, 15;

    shl.b32 %r13, %r8, 1;
    shr.u32 %r14, %r41, %r13;
    and.b32 %r14, %r14, 3;
    shl.b32 %r14, %r14, 4;
    or.b32 %r15, %r12, %r14;
    sub.s32 %r16, %r15, 32;

    cvt.rn.f32.s32 %f2, %r16;
    mul.rn.f32 %f3, %f1, %f2;

    // One byte encodes two weights in each 32-value group.
    and.b32 %r17, %r4, 15;
    shl.b32 %r18, %r8, 4;
    add.u32 %r18, %r18, %r17;
    cvt.u64.u32 %rd11, %r18;
    add.s64 %rd12, %rd7, 8;
    add.s64 %rd13, %rd12, %rd11;
    ld.global.u8 %r19, [%rd13];

    setp.lt.u32 %p6, %r4, 16;
    @%p6 and.b32 %r20, %r19, 15;
    @!%p6 shr.u32 %r20, %r19, 4;

    // Register-only IQ4 nonlinear table lookup.
    shr.u32 %r21, %r20, 2;
    mov.u32 %r22, %r50;
    setp.eq.u32 %p7, %r21, 1;
    @%p7 mov.u32 %r22, %r51;
    setp.eq.u32 %p8, %r21, 2;
    @%p8 mov.u32 %r22, %r52;
    setp.eq.u32 %p9, %r21, 3;
    @%p9 mov.u32 %r22, %r53;

    and.b32 %r23, %r20, 3;
    shl.b32 %r23, %r23, 3;
    shr.u32 %r24, %r22, %r23;
    and.b32 %r24, %r24, 255;
    shl.b32 %r24, %r24, 24;
    shr.s32 %r24, %r24, 24;

    cvt.rn.f32.s32 %f4, %r24;
    mul.rn.f32 %f5, %f3, %f4;

    shl.b32 %r25, %r7, 8;
    shl.b32 %r26, %r8, 5;
    add.u32 %r27, %r25, %r26;
    add.u32 %r27, %r27, %r4;
    mul.wide.u32 %rd14, %r27, 4;
    add.s64 %rd15, %rd2, %rd14;
    ld.global.f32 %f6, [%rd15];

    fma.rn.f32 %f10, %f5, %f6, %f10;

    add.u32 %r8, %r8, 1;
    bra IQG_GROUP_LOOP;

IQG_GROUPS_DONE:
    add.u32 %r7, %r7, 1;
    bra IQG_BLOCK_LOOP;

IQG_BLOCKS_DONE:
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 16, 31, 0xffffffff;
    mov.b32 %f11, %r31;
    add.rn.f32 %f10, %f10, %f11;

    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 8, 31, 0xffffffff;
    mov.b32 %f11, %r31;
    add.rn.f32 %f10, %f10, %f11;

    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 4, 31, 0xffffffff;
    mov.b32 %f11, %r31;
    add.rn.f32 %f10, %f10, %f11;

    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 2, 31, 0xffffffff;
    mov.b32 %f11, %r31;
    add.rn.f32 %f10, %f10, %f11;

    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 1, 31, 0xffffffff;
    mov.b32 %f11, %r31;
    add.rn.f32 %f10, %f10, %f11;

    setp.eq.u32 %p10, %r4, 0;
    mul.wide.u32 %rd16, %r3, 4;
    add.s64 %rd17, %rd3, %rd16;
    @%p10 st.global.f32 [%rd17], %f10;

IQG_DONE:
    ret;
}
)ptx";

constexpr const char* kIQ4XSSm86SoAGemvPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

.visible .entry q38_iq4xs_sm86_soa_gemv(
    .param .u64 p_d,
    .param .u64 p_scale,
    .param .u64 p_qs,
    .param .u64 p_x,
    .param .u64 p_y,
    .param .u32 p_cols,
    .param .u32 p_rows
)
{
    .reg .pred %p<16>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<28>;
    .reg .f32 %f<16>;

    ld.param.u64 %rd1, [p_d];
    ld.param.u64 %rd2, [p_scale];
    ld.param.u64 %rd3, [p_qs];
    ld.param.u64 %rd4, [p_x];
    ld.param.u64 %rd5, [p_y];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_rows];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    setp.ge.u32 %p1, %r3, %r2;
    @%p1 bra IQS_DONE;
    setp.ge.u32 %p2, %r4, 32;
    @%p2 bra IQS_DONE;

    shr.u32 %r5, %r1, 8;        // blocks_per_row
    mul.lo.u32 %r6, %r3, %r5;  // first block index for row

    // Packed codebook bytes.
    mov.u32 %r50, 0xBFAD9881;
    mov.u32 %r51, 0xF6EADDCF;
    mov.u32 %r52, 0x26190D01;
    mov.u32 %r53, 0x71594535;

    mov.u32 %r7, 0;
    mov.f32 %f10, 0f00000000;

IQS_BLOCK_LOOP:
    setp.ge.u32 %p3, %r7, %r5;
    @%p3 bra IQS_BLOCKS_DONE;

    add.u32 %r8, %r6, %r7;      // global block index

    // d plane: fp16 stride 2.
    shl.b32 %r9, %r8, 1;
    cvt.u64.u32 %rd6, %r9;
    add.s64 %rd7, %rd1, %rd6;
    ld.global.b16 %r40, [%rd7];
    cvt.f32.f16 %f1, %r40;

    // qs block base: 128-byte stride.
    shl.b32 %r10, %r8, 7;

    mov.u32 %r11, 0;

IQS_GROUP_LOOP:
    setp.ge.u32 %p4, %r11, 8;
    @%p4 bra IQS_GROUPS_DONE;

    // Pre-expanded signed group scale: 8-byte stride per block.
    shl.b32 %r12, %r8, 3;
    add.u32 %r12, %r12, %r11;
    cvt.u64.u32 %rd8, %r12;
    add.s64 %rd9, %rd2, %rd8;
    ld.global.s8 %r13, [%rd9];

    cvt.rn.f32.s32 %f2, %r13;
    mul.rn.f32 %f3, %f1, %f2;

    // Same compact 4-bit weights, now from a 128-byte-aligned SoA plane.
    and.b32 %r14, %r4, 15;
    shl.b32 %r15, %r11, 4;
    add.u32 %r15, %r15, %r14;
    add.u32 %r15, %r15, %r10;
    cvt.u64.u32 %rd10, %r15;
    add.s64 %rd11, %rd3, %rd10;
    ld.global.u8 %r16, [%rd11];

    setp.lt.u32 %p5, %r4, 16;
    @%p5 and.b32 %r17, %r16, 15;
    @!%p5 shr.u32 %r17, %r16, 4;

    // Four-register nonlinear codebook lookup.
    shr.u32 %r18, %r17, 2;
    mov.u32 %r19, %r50;
    setp.eq.u32 %p6, %r18, 1;
    @%p6 mov.u32 %r19, %r51;
    setp.eq.u32 %p7, %r18, 2;
    @%p7 mov.u32 %r19, %r52;
    setp.eq.u32 %p8, %r18, 3;
    @%p8 mov.u32 %r19, %r53;

    and.b32 %r20, %r17, 3;
    shl.b32 %r20, %r20, 3;
    shr.u32 %r21, %r19, %r20;
    and.b32 %r21, %r21, 255;
    shl.b32 %r21, %r21, 24;
    shr.s32 %r21, %r21, 24;

    cvt.rn.f32.s32 %f4, %r21;
    mul.rn.f32 %f5, %f3, %f4;

    // x index = block*256 + group*32 + lane
    shl.b32 %r22, %r7, 8;
    shl.b32 %r23, %r11, 5;
    add.u32 %r24, %r22, %r23;
    add.u32 %r24, %r24, %r4;
    mul.wide.u32 %rd12, %r24, 4;
    add.s64 %rd13, %rd4, %rd12;
    ld.global.f32 %f6, [%rd13];

    fma.rn.f32 %f10, %f5, %f6, %f10;

    add.u32 %r11, %r11, 1;
    bra IQS_GROUP_LOOP;

IQS_GROUPS_DONE:
    add.u32 %r7, %r7, 1;
    bra IQS_BLOCK_LOOP;

IQS_BLOCKS_DONE:
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 16, 31, 0xffffffff;
    mov.b32 %f11, %r31;
    add.rn.f32 %f10, %f10, %f11;
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 8, 31, 0xffffffff;
    mov.b32 %f11, %r31;
    add.rn.f32 %f10, %f10, %f11;
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 4, 31, 0xffffffff;
    mov.b32 %f11, %r31;
    add.rn.f32 %f10, %f10, %f11;
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 2, 31, 0xffffffff;
    mov.b32 %f11, %r31;
    add.rn.f32 %f10, %f10, %f11;
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 1, 31, 0xffffffff;
    mov.b32 %f11, %r31;
    add.rn.f32 %f10, %f10, %f11;

    setp.eq.u32 %p9, %r4, 0;
    mul.wide.u32 %rd14, %r3, 4;
    add.s64 %rd15, %rd5, %rd14;
    @%p9 st.global.f32 [%rd15], %f10;

IQS_DONE:
    ret;
}
)ptx";

constexpr const char* kIQ4XSQ81GemvPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

// Native IQ4_XS weights x standard Q8_1 activation.
// One warp computes one output row. Four 8-lane subgroups process four
// 32-value IQ4_XS groups in parallel; two passes cover all eight groups.
.visible .entry q38_iq4xs_q81_gemv(
    .param .u64 p_weights,
    .param .u64 p_q8,
    .param .u64 p_y,
    .param .u32 p_cols,
    .param .u32 p_rows
)
{
    .reg .pred %p<32>;
    .reg .b32 %r<96>;
    .reg .b64 %rd<36>;
    .reg .f32 %f<24>;

    ld.param.u64 %rd1, [p_weights];
    ld.param.u64 %rd2, [p_q8];
    ld.param.u64 %rd3, [p_y];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_rows];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    shr.u32 %r5, %r4, 3;       // subgroup 0..3
    and.b32 %r6, %r4, 7;       // lane in subgroup 0..7

    setp.ge.u32 %p1, %r3, %r2;
    @%p1 bra IQQ_DONE;
    setp.ge.u32 %p2, %r4, 32;
    @%p2 bra IQQ_DONE;

    shr.u32 %r7, %r1, 8;       // IQ4 blocks per row
    mul.lo.u32 %r8, %r7, 136;
    mul.wide.u32 %rd4, %r3, %r8;
    add.s64 %rd5, %rd1, %rd4;

    // IQ4 nonlinear codebook packed into four registers.
    mov.u32 %r50, 0xBFAD9881;
    mov.u32 %r51, 0xF6EADDCF;
    mov.u32 %r52, 0x26190D01;
    mov.u32 %r53, 0x71594535;

    mov.u32 %r9, 0;             // block index
    mov.f32 %f15, 0f00000000;   // subgroup-leader accumulator

IQQ_BLOCK_LOOP:
    setp.ge.u32 %p3, %r9, %r7;
    @%p3 bra IQQ_BLOCKS_DONE;

    mul.wide.u32 %rd6, %r9, 136;
    add.s64 %rd7, %rd5, %rd6;

    mov.u32 %r10, 0;            // pass 0..1

IQQ_PASS_LOOP:
    setp.ge.u32 %p4, %r10, 2;
    @%p4 bra IQQ_PASSES_DONE;

    shl.b32 %r11, %r10, 2;
    add.u32 %r11, %r11, %r5;   // group = subgroup + pass*4

    // Two IQ4 bytes encode four weights:
    // byte0 low/high -> positions lane*2 / 16+lane*2
    // byte1 low/high -> positions lane*2+1 / 16+lane*2+1
    shl.b32 %r12, %r11, 4;     // group * 16
    shl.b32 %r13, %r6, 1;      // subgroup lane * 2
    add.u32 %r14, %r12, %r13;
    cvt.u64.u32 %rd8, %r14;
    add.s64 %rd9, %rd7, 8;
    add.s64 %rd10, %rd9, %rd8;
    ld.global.b16 %r20, [%rd10];

    // q0
    and.b32 %r21, %r20, 15;
    shr.u32 %r22, %r21, 2;
    mov.u32 %r23, %r50;
    setp.eq.u32 %p5, %r22, 1;  @%p5 mov.u32 %r23, %r51;
    setp.eq.u32 %p6, %r22, 2;  @%p6 mov.u32 %r23, %r52;
    setp.eq.u32 %p7, %r22, 3;  @%p7 mov.u32 %r23, %r53;
    and.b32 %r24, %r21, 3;
    shl.b32 %r24, %r24, 3;
    shr.u32 %r25, %r23, %r24;
    and.b32 %r25, %r25, 255;

    // q1 = low nibble of second byte
    shr.u32 %r26, %r20, 8;
    and.b32 %r26, %r26, 15;
    shr.u32 %r27, %r26, 2;
    mov.u32 %r28, %r50;
    setp.eq.u32 %p8, %r27, 1;  @%p8 mov.u32 %r28, %r51;
    setp.eq.u32 %p9, %r27, 2;  @%p9 mov.u32 %r28, %r52;
    setp.eq.u32 %p10, %r27, 3; @%p10 mov.u32 %r28, %r53;
    and.b32 %r29, %r26, 3;
    shl.b32 %r29, %r29, 3;
    shr.u32 %r30, %r28, %r29;
    and.b32 %r30, %r30, 255;
    shl.b32 %r30, %r30, 8;
    or.b32 %r25, %r25, %r30;

    // q2 = high nibble of first byte
    shr.u32 %r31, %r20, 4;
    and.b32 %r31, %r31, 15;
    shr.u32 %r32, %r31, 2;
    mov.u32 %r33, %r50;
    setp.eq.u32 %p11, %r32, 1; @%p11 mov.u32 %r33, %r51;
    setp.eq.u32 %p12, %r32, 2; @%p12 mov.u32 %r33, %r52;
    setp.eq.u32 %p13, %r32, 3; @%p13 mov.u32 %r33, %r53;
    and.b32 %r34, %r31, 3;
    shl.b32 %r34, %r34, 3;
    shr.u32 %r35, %r33, %r34;
    and.b32 %r35, %r35, 255;
    shl.b32 %r35, %r35, 16;
    or.b32 %r25, %r25, %r35;

    // q3 = high nibble of second byte
    shr.u32 %r36, %r20, 12;
    and.b32 %r36, %r36, 15;
    shr.u32 %r37, %r36, 2;
    mov.u32 %r38, %r50;
    setp.eq.u32 %p14, %r37, 1; @%p14 mov.u32 %r38, %r51;
    setp.eq.u32 %p15, %r37, 2; @%p15 mov.u32 %r38, %r52;
    setp.eq.u32 %p16, %r37, 3; @%p16 mov.u32 %r38, %r53;
    and.b32 %r39, %r36, 3;
    shl.b32 %r39, %r39, 3;
    shr.u32 %r40, %r38, %r39;
    and.b32 %r40, %r40, 255;
    shl.b32 %r40, %r40, 24;
    or.b32 %r25, %r25, %r40;   // packed four signed IQ4 values

    // Q8_1 block index = IQ4 block*8 + group, 36 bytes/block.
    shl.b32 %r41, %r9, 3;
    add.u32 %r41, %r41, %r11;
    mul.wide.u32 %rd11, %r41, 36;
    add.s64 %rd12, %rd2, %rd11;

    // Pack q8 positions [2l,2l+1,16+2l,17+2l+1].
    shl.b32 %r42, %r6, 1;
    cvt.u64.u32 %rd13, %r42;
    add.s64 %rd14, %rd12, 4;
    add.s64 %rd15, %rd14, %rd13;
    ld.global.b16 %r43, [%rd15];

    add.u32 %r44, %r42, 16;
    cvt.u64.u32 %rd16, %r44;
    add.s64 %rd17, %rd14, %rd16;
    ld.global.b16 %r45, [%rd17];
    shl.b32 %r45, %r45, 16;
    or.b32 %r46, %r43, %r45;

    mov.s32 %r47, 0;
    dp4a.s32.s32 %r47, %r25, %r46, %r47;

    // 8-lane subgroup reduction.
    shfl.sync.bfly.b32 %r48, %r47, 4, 31, 0xffffffff;
    add.s32 %r47, %r47, %r48;
    shfl.sync.bfly.b32 %r48, %r47, 2, 31, 0xffffffff;
    add.s32 %r47, %r47, %r48;
    shfl.sync.bfly.b32 %r48, %r47, 1, 31, 0xffffffff;
    add.s32 %r47, %r47, %r48;

    setp.ne.u32 %p17, %r6, 0;
    @%p17 bra IQQ_NEXT_PASS;

    // Native IQ4_XS group scale.
    ld.global.b16 %r54, [%rd7+0];
    ld.global.b16 %r55, [%rd7+2];
    cvt.f32.f16 %f1, %r54;

    shr.u32 %r56, %r11, 1;
    cvt.u64.u32 %rd18, %r56;
    add.s64 %rd19, %rd7, 4;
    add.s64 %rd20, %rd19, %rd18;
    ld.global.u8 %r57, [%rd20];

    and.b32 %r58, %r11, 1;
    shl.b32 %r58, %r58, 2;
    shr.u32 %r59, %r57, %r58;
    and.b32 %r59, %r59, 15;

    shl.b32 %r60, %r11, 1;
    shr.u32 %r61, %r55, %r60;
    and.b32 %r61, %r61, 3;
    shl.b32 %r61, %r61, 4;
    or.b32 %r62, %r59, %r61;
    sub.s32 %r62, %r62, 32;

    // Q8_1 d is fp16 at +0.
    ld.global.b16 %r63, [%rd12+0];
    cvt.f32.f16 %f2, %r63;

    cvt.rn.f32.s32 %f3, %r47;
    cvt.rn.f32.s32 %f4, %r62;
    mul.rn.f32 %f5, %f3, %f4;
    mul.rn.f32 %f6, %f1, %f2;
    fma.rn.f32 %f15, %f5, %f6, %f15;

IQQ_NEXT_PASS:
    add.u32 %r10, %r10, 1;
    bra IQQ_PASS_LOOP;

IQQ_PASSES_DONE:
    add.u32 %r9, %r9, 1;
    bra IQQ_BLOCK_LOOP;

IQQ_BLOCKS_DONE:
    // Gather subgroup leader partials from lanes 0/8/16/24.
    mov.b32 %r70, %f15;
    shfl.sync.idx.b32 %r71, %r70, 8, 31, 0xffffffff;
    shfl.sync.idx.b32 %r72, %r70, 16, 31, 0xffffffff;
    shfl.sync.idx.b32 %r73, %r70, 24, 31, 0xffffffff;

    setp.ne.u32 %p18, %r4, 0;
    @%p18 bra IQQ_DONE;

    mov.b32 %f16, %r71;
    mov.b32 %f17, %r72;
    mov.b32 %f18, %r73;
    add.rn.f32 %f19, %f15, %f16;
    add.rn.f32 %f19, %f19, %f17;
    add.rn.f32 %f19, %f19, %f18;

    mul.wide.u32 %rd21, %r3, 4;
    add.s64 %rd22, %rd3, %rd21;
    st.global.f32 [%rd22], %f19;

IQQ_DONE:
    ret;
}
)ptx";

constexpr const char* kIQ4XSGemvPrmtPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

// Native IQ4_XS x F32, vectorized nonlinear decode.
// One CTA has four warps; each warp computes one output row.
// Within a warp, eight 4-lane subgroups map 1:1 to the eight 32-value
// groups of each IQ4_XS superblock. Every lane loads four packed bytes
// (8 weights), resolves all eight codebook values with prmt.b32, and
// performs eight F32 FMAs. Group scale is applied only after the 4-lane
// subgroup reduction.
.visible .entry q38_iq4xs_gemv_f32_prmt(
    .param .u64 p_weights,
    .param .u64 p_x,
    .param .u64 p_y,
    .param .u32 p_cols,
    .param .u32 p_rows
)
{
    .reg .pred %p<24>;
    .reg .b32 %r<112>;
    .reg .b64 %rd<40>;
    .reg .f32 %f<40>;

    ld.param.u64 %rd1, [p_weights];
    ld.param.u64 %rd2, [p_x];
    ld.param.u64 %rd3, [p_y];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_rows];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    shr.u32 %r5, %r4, 5;       // warp id 0..3
    and.b32 %r6, %r4, 31;      // lane 0..31
    shl.b32 %r7, %r3, 2;
    add.u32 %r7, %r7, %r5;     // output row

    setp.ge.u32 %p1, %r7, %r2;
    @%p1 bra IQP_DONE;

    shr.u32 %r8, %r6, 2;       // group 0..7
    and.b32 %r9, %r6, 3;       // sublane 0..3

    // blocks_per_row and row base
    shr.u32 %r10, %r1, 8;
    mul.lo.u32 %r11, %r10, 136;
    mul.wide.u32 %rd4, %r7, %r11;
    add.s64 %rd5, %rd1, %rd4;

    // Nonlinear IQ4 codebook bytes:
    // [-127,-104,-83,-65], [-49,-35,-22,-10],
    // [1,13,25,38], [53,69,89,113].
    mov.u32 %r80, 0xBFAD9881;
    mov.u32 %r81, 0xF6EADDCF;
    mov.u32 %r82, 0x26190D01;
    mov.u32 %r83, 0x71594535;

    mov.u32 %r12, 0;            // block index
    mov.f32 %f20, 0f00000000;   // valid in subgroup leaders

IQP_BLOCK_LOOP:
    setp.ge.u32 %p2, %r12, %r10;
    @%p2 bra IQP_BLOCKS_DONE;

    mul.wide.u32 %rd6, %r12, 136;
    add.s64 %rd7, %rd5, %rd6;

    // q4 = 4 consecutive qs bytes for this group/sublane.
    shl.b32 %r13, %r8, 4;      // group*16
    shl.b32 %r14, %r9, 2;      // sublane*4
    add.u32 %r15, %r13, %r14;
    cvt.u64.u32 %rd8, %r15;
    add.s64 %rd9, %rd7, 8;
    add.s64 %rd10, %rd9, %rd8;
    ld.global.u32 %r16, [%rd10];

    // Emulate CUDA __byte_perm for 8 nonlinear table lookups.
    // First four nibbles (low 16 bits of q4).
    and.b32 %r17, %r16, 0x00007777;
    prmt.b32 %r18, %r80, %r81, %r17;
    prmt.b32 %r19, %r82, %r83, %r17;
    and.b32 %r20, %r16, 0x00008888;
    shr.u32 %r20, %r20, 1;
    or.b32 %r20, %r20, 0x00003210;
    prmt.b32 %r21, %r18, %r19, %r20;

    // Second four nibbles (high 16 bits of q4).
    shr.u32 %r22, %r16, 16;
    and.b32 %r23, %r22, 0x00007777;
    prmt.b32 %r24, %r80, %r81, %r23;
    prmt.b32 %r25, %r82, %r83, %r23;
    and.b32 %r26, %r22, 0x00008888;
    shr.u32 %r26, %r26, 1;
    or.b32 %r26, %r26, 0x00003210;
    prmt.b32 %r27, %r24, %r25, %r26;

    // Gather low nibbles of the 4 source bytes into r28 and
    // high nibbles into r29. Each register contains four signed int8 values.
    prmt.b32 %r28, %r21, %r27, 0x00006420;
    prmt.b32 %r29, %r21, %r27, 0x00007531;

    // Activation indices:
    // low nibble weights -> group*32 + sublane*4 + [0..3]
    // high nibble weights -> +16.
    shl.b32 %r30, %r12, 8;     // block*256
    shl.b32 %r31, %r8, 5;      // group*32
    add.u32 %r32, %r30, %r31;
    add.u32 %r32, %r32, %r14;
    mul.wide.u32 %rd11, %r32, 4;
    add.s64 %rd12, %rd2, %rd11;

    ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [%rd12];
    add.s64 %rd13, %rd12, 64;
    ld.global.v4.f32 {%f5,%f6,%f7,%f8}, [%rd13];

    mov.f32 %f9, 0f00000000;

    // r28 byte 0
    shl.b32 %r40, %r28, 24;
    shr.s32 %r40, %r40, 24;
    cvt.rn.f32.s32 %f10, %r40;
    fma.rn.f32 %f9, %f10, %f1, %f9;

    // r28 byte 1
    shr.u32 %r41, %r28, 8;
    shl.b32 %r41, %r41, 24;
    shr.s32 %r41, %r41, 24;
    cvt.rn.f32.s32 %f11, %r41;
    fma.rn.f32 %f9, %f11, %f2, %f9;

    // r28 byte 2
    shr.u32 %r42, %r28, 16;
    shl.b32 %r42, %r42, 24;
    shr.s32 %r42, %r42, 24;
    cvt.rn.f32.s32 %f12, %r42;
    fma.rn.f32 %f9, %f12, %f3, %f9;

    // r28 byte 3
    shr.u32 %r43, %r28, 24;
    shl.b32 %r43, %r43, 24;
    shr.s32 %r43, %r43, 24;
    cvt.rn.f32.s32 %f13, %r43;
    fma.rn.f32 %f9, %f13, %f4, %f9;

    // r29 byte 0
    shl.b32 %r44, %r29, 24;
    shr.s32 %r44, %r44, 24;
    cvt.rn.f32.s32 %f14, %r44;
    fma.rn.f32 %f9, %f14, %f5, %f9;

    // r29 byte 1
    shr.u32 %r45, %r29, 8;
    shl.b32 %r45, %r45, 24;
    shr.s32 %r45, %r45, 24;
    cvt.rn.f32.s32 %f15, %r45;
    fma.rn.f32 %f9, %f15, %f6, %f9;

    // r29 byte 2
    shr.u32 %r46, %r29, 16;
    shl.b32 %r46, %r46, 24;
    shr.s32 %r46, %r46, 24;
    cvt.rn.f32.s32 %f16, %r46;
    fma.rn.f32 %f9, %f16, %f7, %f9;

    // r29 byte 3
    shr.u32 %r47, %r29, 24;
    shl.b32 %r47, %r47, 24;
    shr.s32 %r47, %r47, 24;
    cvt.rn.f32.s32 %f17, %r47;
    fma.rn.f32 %f9, %f17, %f8, %f9;

    // Reduce 4 lanes inside each group.
    mov.b32 %r48, %f9;
    shfl.sync.bfly.b32 %r49, %r48, 2, 31, 0xffffffff;
    mov.b32 %f18, %r49;
    add.rn.f32 %f9, %f9, %f18;
    mov.b32 %r48, %f9;
    shfl.sync.bfly.b32 %r49, %r48, 1, 31, 0xffffffff;
    mov.b32 %f18, %r49;
    add.rn.f32 %f9, %f9, %f18;

    // Only subgroup leader loads/applies block d and signed group scale.
    setp.ne.u32 %p3, %r9, 0;
    @%p3 bra IQP_NEXT_BLOCK;

    ld.global.b16 %r50, [%rd7+0];
    ld.global.b16 %r51, [%rd7+2];
    cvt.f32.f16 %f21, %r50;

    shr.u32 %r52, %r8, 1;
    cvt.u64.u32 %rd14, %r52;
    add.s64 %rd15, %rd7, 4;
    add.s64 %rd16, %rd15, %rd14;
    ld.global.u8 %r53, [%rd16];

    and.b32 %r54, %r8, 1;
    shl.b32 %r54, %r54, 2;
    shr.u32 %r55, %r53, %r54;
    and.b32 %r55, %r55, 15;

    shl.b32 %r56, %r8, 1;
    shr.u32 %r57, %r51, %r56;
    and.b32 %r57, %r57, 3;
    shl.b32 %r57, %r57, 4;
    or.b32 %r58, %r55, %r57;
    sub.s32 %r58, %r58, 32;

    cvt.rn.f32.s32 %f22, %r58;
    mul.rn.f32 %f23, %f21, %f22;
    fma.rn.f32 %f20, %f9, %f23, %f20;

IQP_NEXT_BLOCK:
    add.u32 %r12, %r12, 1;
    bra IQP_BLOCK_LOOP;

IQP_BLOCKS_DONE:
    // f20 is valid in lanes 0,4,8,...28. Reduce those 8 group leaders.
    mov.b32 %r60, %f20;
    shfl.sync.down.b32 %r61, %r60, 16, 31, 0xffffffff;
    mov.b32 %f24, %r61;
    add.rn.f32 %f20, %f20, %f24;

    mov.b32 %r60, %f20;
    shfl.sync.down.b32 %r61, %r60, 8, 31, 0xffffffff;
    mov.b32 %f24, %r61;
    add.rn.f32 %f20, %f20, %f24;

    mov.b32 %r60, %f20;
    shfl.sync.down.b32 %r61, %r60, 4, 31, 0xffffffff;
    mov.b32 %f24, %r61;
    add.rn.f32 %f20, %f20, %f24;

    setp.ne.u32 %p4, %r6, 0;
    @%p4 bra IQP_DONE;

    mul.wide.u32 %rd17, %r7, 4;
    add.s64 %rd18, %rd3, %rd17;
    st.global.f32 [%rd18], %f20;

IQP_DONE:
    ret;
}
)ptx";

constexpr const char* kQ5KDequantPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

.visible .entry q38_dequant_q5k_block(
    .param .u64 p_block,
    .param .u64 p_out
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<18>;
    .reg .f32 %f<12>;

    ld.param.u64 %rd1, [p_block];
    ld.param.u64 %rd2, [p_out];

    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p1, %r1, 256;
    @%p1 bra Q5K_DONE;

    // PTX 7.1: load f16 payload via b16 then convert.
    ld.global.b16 %r24, [%rd1+0];
    ld.global.b16 %r25, [%rd1+2];
    cvt.f32.f16 %f1, %r24;
    cvt.f32.f16 %f2, %r25;

    // group = tid / 32, lane = tid % 32
    shr.u32 %r2, %r1, 5;
    and.b32 %r3, %r1, 31;

    // scales base = block + 4
    add.s64 %rd3, %rd1, 4;
    setp.lt.u32 %p2, %r2, 4;
    @%p2 bra Q5_SCALE_LOW;

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
    bra Q5_SCALE_READY;

Q5_SCALE_LOW:
    cvt.u64.u32 %rd4, %r2;
    add.s64 %rd5, %rd3, %rd4;
    ld.global.u8 %r5, [%rd5];
    and.b32 %r11, %r5, 63;

    add.u32 %r6, %r2, 4;
    cvt.u64.u32 %rd6, %r6;
    add.s64 %rd7, %rd3, %rd6;
    ld.global.u8 %r7, [%rd7];
    and.b32 %r14, %r7, 63;

Q5_SCALE_READY:
    cvt.rn.f32.u32 %f3, %r11;
    cvt.rn.f32.u32 %f4, %r14;
    mul.rn.f32 %f5, %f1, %f3;
    mul.rn.f32 %f6, %f2, %f4;

    // ql starts at +48. byte index = (group/2)*32 + lane.
    shr.u32 %r15, %r2, 1;
    shl.b32 %r15, %r15, 5;
    add.u32 %r15, %r15, %r3;
    cvt.u64.u32 %rd10, %r15;
    add.s64 %rd11, %rd1, 48;
    add.s64 %rd12, %rd11, %rd10;
    ld.global.u8 %r16, [%rd12];

    and.b32 %r17, %r2, 1;
    setp.eq.u32 %p3, %r17, 0;
    @%p3 bra Q5_LOW_NIBBLE;
    shr.u32 %r18, %r16, 4;
    bra Q5_NIBBLE_READY;

Q5_LOW_NIBBLE:
    and.b32 %r18, %r16, 15;

Q5_NIBBLE_READY:
    // qh starts at +16. One byte per lane; bit[group] is the fifth bit.
    cvt.u64.u32 %rd13, %r3;
    add.s64 %rd14, %rd1, 16;
    add.s64 %rd15, %rd14, %rd13;
    ld.global.u8 %r19, [%rd15];

    mov.u32 %r20, 1;
    shl.b32 %r20, %r20, %r2;
    and.b32 %r21, %r19, %r20;
    setp.ne.u32 %p4, %r21, 0;
    mov.u32 %r22, 0;
    @%p4 mov.u32 %r22, 16;
    add.u32 %r23, %r18, %r22;

    cvt.rn.f32.u32 %f7, %r23;
    mul.rn.f32 %f8, %f5, %f7;
    sub.rn.f32 %f9, %f8, %f6;

    mul.wide.u32 %rd16, %r1, 4;
    add.s64 %rd17, %rd2, %rd16;
    st.global.f32 [%rd17], %f9;

Q5K_DONE:
    ret;
}
)ptx";

constexpr const char* kQ5KGemvPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

.visible .entry q38_q5k_gemv_f32(
    .param .u64 p_weights,
    .param .u64 p_x,
    .param .u64 p_y,
    .param .u32 p_cols,
    .param .u32 p_rows
)
{
    .reg .pred %p<12>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<24>;
    .reg .f32 %f<16>;

    ld.param.u64 %rd1, [p_weights];
    ld.param.u64 %rd2, [p_x];
    ld.param.u64 %rd3, [p_y];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_rows];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    setp.ge.u32 %p1, %r3, %r2;
    @%p1 bra GEMV_DONE;
    setp.ge.u32 %p2, %r4, 32;
    @%p2 bra GEMV_DONE;

    // 256 values per Q5_K block.
    shr.u32 %r5, %r1, 8;
    mov.u32 %r6, 0;       // block index
    mov.f32 %f10, 0f00000000;

    // bytes_per_row = nblocks * 176
    mul.lo.u32 %r7, %r5, 176;
    mul.wide.u32 %rd4, %r3, %r7;
    add.s64 %rd5, %rd1, %rd4;

BLOCK_LOOP:
    setp.ge.u32 %p3, %r6, %r5;
    @%p3 bra BLOCKS_DONE;

    mul.wide.u32 %rd6, %r6, 176;
    add.s64 %rd7, %rd5, %rd6;

    // Identical-address warp loads are efficiently serviced by Ampere's
    // memory hierarchy. Keep every lane active; lane-0 serialization was
    // measured to regress bandwidth badly on RTX 3090.
    ld.global.b16 %r40, [%rd7+0];
    ld.global.b16 %r41, [%rd7+2];
    cvt.f32.f16 %f1, %r40;
    cvt.f32.f16 %f2, %r41;

    // qh has one byte per lane and its 8 bits serve all 8 groups.
    // Cache it once per Q5_K super-block instead of reloading it per group.
    cvt.u64.u32 %rd18, %r4;
    add.s64 %rd19, %rd7, 16;
    add.s64 %rd20, %rd19, %rd18;
    ld.global.u8 %r39, [%rd20];

    mov.u32 %r8, 0; // group 0..7

GROUP_LOOP:
    setp.ge.u32 %p4, %r8, 8;
    @%p4 bra GROUPS_DONE;

    // Decode 6-bit scale/min for this 32-value group.
    add.s64 %rd8, %rd7, 4;
    setp.lt.u32 %p5, %r8, 4;
    @%p5 bra GEMV_SCALE_LOW;

    add.u32 %r9, %r8, 4;
    cvt.u64.u32 %rd9, %r9;
    add.s64 %rd10, %rd8, %rd9;
    ld.global.u8 %r10, [%rd10];

    sub.u32 %r11, %r8, 4;
    cvt.u64.u32 %rd11, %r11;
    add.s64 %rd12, %rd8, %rd11;
    ld.global.u8 %r12, [%rd12];

    cvt.u64.u32 %rd13, %r8;
    add.s64 %rd14, %rd8, %rd13;
    ld.global.u8 %r13, [%rd14];

    and.b32 %r14, %r10, 15;
    shr.u32 %r15, %r12, 6;
    shl.b32 %r15, %r15, 4;
    or.b32 %r16, %r14, %r15;

    shr.u32 %r17, %r10, 4;
    shr.u32 %r18, %r13, 6;
    shl.b32 %r18, %r18, 4;
    or.b32 %r19, %r17, %r18;
    bra GEMV_SCALE_READY;

GEMV_SCALE_LOW:
    cvt.u64.u32 %rd9, %r8;
    add.s64 %rd10, %rd8, %rd9;
    ld.global.u8 %r10, [%rd10];
    and.b32 %r16, %r10, 63;

    add.u32 %r11, %r8, 4;
    cvt.u64.u32 %rd11, %r11;
    add.s64 %rd12, %rd8, %rd11;
    ld.global.u8 %r12, [%rd12];
    and.b32 %r19, %r12, 63;

GEMV_SCALE_READY:
    cvt.rn.f32.u32 %f3, %r16;
    cvt.rn.f32.u32 %f4, %r19;
    mul.rn.f32 %f5, %f1, %f3;
    mul.rn.f32 %f6, %f2, %f4;

    // ql byte index = (group/2)*32 + lane, ql base = +48.
    // One ql byte serves two adjacent groups (low and high nibble), so only
    // fetch it for even groups and retain r21 for the odd group.
    shr.u32 %r20, %r8, 1;
    shl.b32 %r20, %r20, 5;
    add.u32 %r20, %r20, %r4;
    cvt.u64.u32 %rd15, %r20;
    add.s64 %rd16, %rd7, 48;
    add.s64 %rd17, %rd16, %rd15;

    and.b32 %r22, %r8, 1;
    setp.eq.u32 %p6, %r22, 0;
    @%p6 ld.global.u8 %r21, [%rd17];
    @%p6 bra GEMV_LOW_NIBBLE;
    shr.u32 %r23, %r21, 4;
    bra GEMV_NIBBLE_READY;

GEMV_LOW_NIBBLE:
    and.b32 %r23, %r21, 15;

GEMV_NIBBLE_READY:
    // High bit: cached qh[lane] bit[group].
    mov.u32 %r25, 1;
    shl.b32 %r25, %r25, %r8;
    and.b32 %r26, %r39, %r25;
    setp.ne.u32 %p7, %r26, 0;
    mov.u32 %r27, 0;
    @%p7 mov.u32 %r27, 16;
    add.u32 %r28, %r23, %r27;

    cvt.rn.f32.u32 %f7, %r28;
    mul.rn.f32 %f8, %f5, %f7;
    sub.rn.f32 %f9, %f8, %f6;

    // x index = block*256 + group*32 + lane
    shl.b32 %r29, %r6, 8;
    shl.b32 %r30, %r8, 5;
    add.u32 %r31, %r29, %r30;
    add.u32 %r31, %r31, %r4;
    mul.wide.u32 %rd21, %r31, 4;
    add.s64 %rd22, %rd2, %rd21;
    ld.global.f32 %f11, [%rd22];

    fma.rn.f32 %f10, %f9, %f11, %f10;

    add.u32 %r8, %r8, 1;
    bra GROUP_LOOP;

GROUPS_DONE:
    add.u32 %r6, %r6, 1;
    bra BLOCK_LOOP;

BLOCKS_DONE:
    // Warp reduction. Shuffle moves raw 32-bit float payloads; only lane 0
    // writes the final row result. This replaces 32 contended global atomics.
    mov.b32 %r42, %f10;
    shfl.sync.down.b32 %r43, %r42, 16, 31, 0xffffffff;
    mov.b32 %f12, %r43;
    add.rn.f32 %f10, %f10, %f12;

    mov.b32 %r42, %f10;
    shfl.sync.down.b32 %r43, %r42, 8, 31, 0xffffffff;
    mov.b32 %f12, %r43;
    add.rn.f32 %f10, %f10, %f12;

    mov.b32 %r42, %f10;
    shfl.sync.down.b32 %r43, %r42, 4, 31, 0xffffffff;
    mov.b32 %f12, %r43;
    add.rn.f32 %f10, %f10, %f12;

    mov.b32 %r42, %f10;
    shfl.sync.down.b32 %r43, %r42, 2, 31, 0xffffffff;
    mov.b32 %f12, %r43;
    add.rn.f32 %f10, %f10, %f12;

    mov.b32 %r42, %f10;
    shfl.sync.down.b32 %r43, %r42, 1, 31, 0xffffffff;
    mov.b32 %f12, %r43;
    add.rn.f32 %f10, %f10, %f12;

    setp.eq.u32 %p8, %r4, 0;
    mul.wide.u32 %rd23, %r3, 4;
    add.s64 %rd23, %rd3, %rd23;
    @%p8 st.global.f32 [%rd23], %f10;

GEMV_DONE:
    ret;
}
)ptx";

constexpr const char* kQ5KQ8KGemvPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

.visible .entry q38_q5k_q8k_gemv(
    .param .u64 p_weights,
    .param .u64 p_q8,
    .param .u64 p_y,
    .param .u32 p_cols,
    .param .u32 p_rows
)
{
    .reg .pred %p<16>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<20>;

    ld.param.u64 %rd1, [p_weights];
    ld.param.u64 %rd2, [p_q8];
    ld.param.u64 %rd3, [p_y];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_rows];

    mov.u32 %r3, %ctaid.x;      // row
    mov.u32 %r4, %tid.x;
    shr.u32 %r5, %r4, 5;       // warp/group 0..7
    and.b32 %r6, %r4, 31;      // lane

    setp.ge.u32 %p1, %r3, %r2;
    @%p1 bra Q8_GEMV_DONE;
    setp.ge.u32 %p2, %r5, 8;
    @%p2 bra Q8_GEMV_DONE;

    shr.u32 %r7, %r1, 8;       // Q5/Q8 blocks per row (cols/256)
    mul.lo.u32 %r8, %r7, 176;  // Q5 bytes per row
    mul.wide.u32 %rd4, %r3, %r8;
    add.s64 %rd5, %rd1, %rd4;  // row weights

    mov.u32 %r9, 0;            // block index
    mov.f32 %f15, 0f00000000;  // per-warp accumulated contribution (lane0)

Q8_BLOCK_LOOP:
    setp.ge.u32 %p3, %r9, %r7;
    @%p3 bra Q8_BLOCKS_DONE;

    // Q5 block pointer.
    mul.wide.u32 %rd6, %r9, 176;
    add.s64 %rd7, %rd5, %rd6;

    // Q8 activation block pointer.
    mul.wide.u32 %rd8, %r9, 292;
    add.s64 %rd9, %rd2, %rd8;

    // ql byte: ((group/2)*32 + lane), base +48.
    shr.u32 %r10, %r5, 1;
    shl.b32 %r10, %r10, 5;
    add.u32 %r10, %r10, %r6;
    cvt.u64.u32 %rd10, %r10;
    add.s64 %rd11, %rd7, 48;
    add.s64 %rd12, %rd11, %rd10;
    ld.global.u8 %r11, [%rd12];

    and.b32 %r12, %r5, 1;
    setp.eq.u32 %p4, %r12, 0;
    @%p4 bra Q8_LOW_NIBBLE;
    shr.u32 %r13, %r11, 4;
    bra Q8_NIBBLE_READY;

Q8_LOW_NIBBLE:
    and.b32 %r13, %r11, 15;

Q8_NIBBLE_READY:
    // High fifth bit: qh[lane] bit[group], qh base +16.
    cvt.u64.u32 %rd13, %r6;
    add.s64 %rd14, %rd7, 16;
    add.s64 %rd15, %rd14, %rd13;
    ld.global.u8 %r14, [%rd15];

    mov.u32 %r15, 1;
    shl.b32 %r15, %r15, %r5;
    and.b32 %r16, %r14, %r15;
    setp.ne.u32 %p5, %r16, 0;
    mov.u32 %r17, 0;
    @%p5 mov.u32 %r17, 16;
    add.u32 %r18, %r13, %r17;  // q5 value 0..31

    // Signed Q8 activation for this group/lane, qs base +4.
    shl.b32 %r19, %r5, 5;
    add.u32 %r19, %r19, %r6;
    cvt.u64.u32 %rd16, %r19;
    add.s64 %rd17, %rd9, 4;
    add.s64 %rd18, %rd17, %rd16;
    ld.global.s8 %r20, [%rd18];

    mul.lo.s32 %r21, %r18, %r20;

    // Integer warp reduction of q5*q8 for this 32-value group.
    shfl.sync.down.b32 %r22, %r21, 16, 31, 0xffffffff;
    add.s32 %r21, %r21, %r22;
    shfl.sync.down.b32 %r22, %r21, 8, 31, 0xffffffff;
    add.s32 %r21, %r21, %r22;
    shfl.sync.down.b32 %r22, %r21, 4, 31, 0xffffffff;
    add.s32 %r21, %r21, %r22;
    shfl.sync.down.b32 %r22, %r21, 2, 31, 0xffffffff;
    add.s32 %r21, %r21, %r22;
    shfl.sync.down.b32 %r22, %r21, 1, 31, 0xffffffff;
    add.s32 %r21, %r21, %r22;

    // Only lane0 needs the group metadata and correction.
    setp.ne.u32 %p6, %r6, 0;
    @%p6 bra Q8_NEXT_BLOCK;

    // Q5 super-block d/dmin.
    ld.global.b16 %r40, [%rd7+0];
    ld.global.b16 %r41, [%rd7+2];
    cvt.f32.f16 %f1, %r40;
    cvt.f32.f16 %f2, %r41;

    // Decode 6-bit scale/min for group r5.
    add.s64 %rd19, %rd7, 4;
    setp.lt.u32 %p7, %r5, 4;
    @%p7 bra Q8_SCALE_LOW;

    add.u32 %r23, %r5, 4;
    cvt.u64.u32 %rd20, %r23;
    add.s64 %rd21, %rd19, %rd20;
    ld.global.u8 %r24, [%rd21];

    sub.u32 %r25, %r5, 4;
    cvt.u64.u32 %rd22, %r25;
    add.s64 %rd23, %rd19, %rd22;
    ld.global.u8 %r26, [%rd23];

    cvt.u64.u32 %rd24, %r5;
    add.s64 %rd25, %rd19, %rd24;
    ld.global.u8 %r27, [%rd25];

    and.b32 %r28, %r24, 15;
    shr.u32 %r29, %r26, 6;
    shl.b32 %r29, %r29, 4;
    or.b32 %r30, %r28, %r29;

    shr.u32 %r31, %r24, 4;
    shr.u32 %r32, %r27, 6;
    shl.b32 %r32, %r32, 4;
    or.b32 %r33, %r31, %r32;
    bra Q8_SCALE_READY;

Q8_SCALE_LOW:
    cvt.u64.u32 %rd20, %r5;
    add.s64 %rd21, %rd19, %rd20;
    ld.global.u8 %r24, [%rd21];
    and.b32 %r30, %r24, 63;

    add.u32 %r25, %r5, 4;
    cvt.u64.u32 %rd22, %r25;
    add.s64 %rd23, %rd19, %rd22;
    ld.global.u8 %r26, [%rd23];
    and.b32 %r33, %r26, 63;

Q8_SCALE_READY:
    // Q8 block scale.
    ld.global.f32 %f3, [%rd9+0];

    // Q8 bsums[2*g] + bsums[2*g+1], base +260.
    shl.b32 %r34, %r5, 2;      // group * 4 bytes
    cvt.u64.u32 %rd26, %r34;
    add.s64 %rd27, %rd9, 260;
    add.s64 %rd28, %rd27, %rd26;
    ld.global.s16 %r35, [%rd28+0];
    ld.global.s16 %r36, [%rd28+2];
    add.s32 %r37, %r35, %r36;

    cvt.rn.f32.s32 %f4, %r21;  // dot(q5,q8)
    cvt.rn.f32.u32 %f5, %r30;  // Q5 scale
    cvt.rn.f32.u32 %f6, %r33;  // Q5 min scale
    cvt.rn.f32.s32 %f7, %r37;  // sum(q8)

    mul.rn.f32 %f8, %f1, %f5;
    mul.rn.f32 %f9, %f8, %f4;
    mul.rn.f32 %f10, %f2, %f6;
    mul.rn.f32 %f11, %f10, %f7;
    sub.rn.f32 %f12, %f9, %f11;
    mul.rn.f32 %f13, %f3, %f12;
    add.rn.f32 %f15, %f15, %f13;

Q8_NEXT_BLOCK:
    add.u32 %r9, %r9, 1;
    bra Q8_BLOCK_LOOP;

Q8_BLOCKS_DONE:
    // Eight warp partials per output row. Baseline uses 8 atomics; once the
    // integer path is validated we can switch this to one CTA-level reduction.
    setp.ne.u32 %p8, %r6, 0;
    @%p8 bra Q8_GEMV_DONE;
    mul.wide.u32 %rd29, %r3, 4;
    add.s64 %rd30, %rd3, %rd29;
    atom.global.add.f32 %f16, [%rd30], %f15;

Q8_GEMV_DONE:
    ret;
}
)ptx";

constexpr const char* kQ5KQ8KDp4aGemvPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

// One warp computes one output row.
// The warp is split into four 8-lane subgroups. In pass 0 they process Q5_K
// groups 0..3; in pass 1 they process groups 4..7. Each subgroup lane packs
// four Q5 bytes and four signed Q8 bytes and uses one dp4a instruction.
.visible .entry q38_q5k_q8k_dp4a_gemv(
    .param .u64 p_weights,
    .param .u64 p_q8,
    .param .u64 p_y,
    .param .u32 p_cols,
    .param .u32 p_rows
)
{
    .reg .pred %p<24>;
    .reg .b32 %r<88>;
    .reg .b64 %rd<40>;
    .reg .f32 %f<28>;

    ld.param.u64 %rd1, [p_weights];
    ld.param.u64 %rd2, [p_q8];
    ld.param.u64 %rd3, [p_y];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_rows];

    mov.u32 %r3, %ctaid.x;       // output row
    mov.u32 %r4, %tid.x;
    shr.u32 %r5, %r4, 3;        // subgroup 0..3
    and.b32 %r6, %r4, 7;        // lane in subgroup 0..7

    setp.ge.u32 %p1, %r3, %r2;
    @%p1 bra DP4A_DONE;

    shr.u32 %r7, %r1, 8;        // blocks_per_row = cols / 256
    mul.lo.u32 %r8, %r7, 176;
    mul.wide.u32 %rd4, %r3, %r8;
    add.s64 %rd5, %rd1, %rd4;  // row weight base

    mov.u32 %r9, 0;             // block index
    mov.f32 %f15, 0f00000000;   // subgroup-leader accumulator

DP4A_BLOCK_LOOP:
    setp.ge.u32 %p2, %r9, %r7;
    @%p2 bra DP4A_BLOCKS_DONE;

    mul.wide.u32 %rd6, %r9, 176;
    add.s64 %rd7, %rd5, %rd6;  // Q5 block

    mul.wide.u32 %rd8, %r9, 292;
    add.s64 %rd9, %rd2, %rd8;  // Q8 block

    // Four qh bytes per subgroup lane. All four subgroups stay on the same
    // Q5 row/block, preserving row-local/coalesced access.
    shl.b32 %r10, %r6, 2;       // sublane * 4
    cvt.u64.u32 %rd10, %r10;
    add.s64 %rd11, %rd7, 16;
    add.s64 %rd12, %rd11, %rd10;
    ld.global.b32 %r40, [%rd12];

    mov.u32 %r11, 0;            // pass 0..1

DP4A_PASS_LOOP:
    setp.ge.u32 %p3, %r11, 2;
    @%p3 bra DP4A_PASSES_DONE;

    // group = subgroup + pass*4
    shl.b32 %r12, %r11, 2;
    add.u32 %r12, %r12, %r5;

    // ql pair storage, four bytes per subgroup lane.
    shr.u32 %r13, %r12, 1;
    shl.b32 %r13, %r13, 5;      // pair * 32
    add.u32 %r13, %r13, %r10;   // + sublane * 4
    cvt.u64.u32 %rd13, %r13;
    add.s64 %rd14, %rd7, 48;
    add.s64 %rd15, %rd14, %rd13;
    ld.global.b32 %r41, [%rd15];

    and.b32 %r14, %r12, 1;
    setp.eq.u32 %p4, %r14, 0;
    @%p4 and.b32 %r42, %r41, 0x0f0f0f0f;
    @!%p4 shr.u32 %r42, %r41, 4;
    @!%p4 and.b32 %r42, %r42, 0x0f0f0f0f;

    // Add the Q5 fifth bit into bit4 of each packed byte.
    mov.u32 %r43, 0x01010101;
    shl.b32 %r43, %r43, %r12;
    and.b32 %r44, %r40, %r43;

    setp.lt.u32 %p5, %r12, 4;
    @%p5 sub.u32 %r45, 4, %r12;
    @%p5 shl.b32 %r44, %r44, %r45;
    @!%p5 sub.u32 %r45, %r12, 4;
    @!%p5 shr.u32 %r44, %r44, %r45;
    or.b32 %r42, %r42, %r44;

    // Four signed Q8 values. Across the whole warp pass 0 reads groups 0..3
    // (128 contiguous bytes); pass 1 reads groups 4..7 (next 128 bytes).
    shl.b32 %r46, %r12, 5;
    add.u32 %r46, %r46, %r10;
    cvt.u64.u32 %rd16, %r46;
    add.s64 %rd17, %rd9, 4;
    add.s64 %rd18, %rd17, %rd16;
    ld.global.b32 %r47, [%rd18];

    mov.s32 %r48, 0;
    dp4a.u32.s32 %r48, %r42, %r47, %r48;

    // Independent 8-lane subgroup reduction.
    shfl.sync.bfly.b32 %r49, %r48, 4, 31, 0xffffffff;
    add.s32 %r48, %r48, %r49;
    shfl.sync.bfly.b32 %r49, %r48, 2, 31, 0xffffffff;
    add.s32 %r48, %r48, %r49;
    shfl.sync.bfly.b32 %r49, %r48, 1, 31, 0xffffffff;
    add.s32 %r48, %r48, %r49;

    setp.ne.u32 %p6, %r6, 0;
    @%p6 bra DP4A_NEXT_PASS;

    // Only subgroup leaders need floating metadata/correction.
    ld.global.b16 %r50, [%rd7+0];
    ld.global.b16 %r51, [%rd7+2];
    cvt.f32.f16 %f1, %r50;
    cvt.f32.f16 %f2, %r51;

    add.s64 %rd19, %rd7, 4;
    setp.lt.u32 %p7, %r12, 4;
    @%p7 bra DP4A_SCALE_LOW;

    add.u32 %r52, %r12, 4;
    cvt.u64.u32 %rd20, %r52;
    add.s64 %rd21, %rd19, %rd20;
    ld.global.u8 %r53, [%rd21];

    sub.u32 %r54, %r12, 4;
    cvt.u64.u32 %rd22, %r54;
    add.s64 %rd23, %rd19, %rd22;
    ld.global.u8 %r55, [%rd23];

    cvt.u64.u32 %rd24, %r12;
    add.s64 %rd25, %rd19, %rd24;
    ld.global.u8 %r56, [%rd25];

    and.b32 %r57, %r53, 15;
    shr.u32 %r58, %r55, 6;
    shl.b32 %r58, %r58, 4;
    or.b32 %r59, %r57, %r58;

    shr.u32 %r60, %r53, 4;
    shr.u32 %r61, %r56, 6;
    shl.b32 %r61, %r61, 4;
    or.b32 %r62, %r60, %r61;
    bra DP4A_SCALE_READY;

DP4A_SCALE_LOW:
    cvt.u64.u32 %rd20, %r12;
    add.s64 %rd21, %rd19, %rd20;
    ld.global.u8 %r53, [%rd21];
    and.b32 %r59, %r53, 63;

    add.u32 %r54, %r12, 4;
    cvt.u64.u32 %rd22, %r54;
    add.s64 %rd23, %rd19, %rd22;
    ld.global.u8 %r55, [%rd23];
    and.b32 %r62, %r55, 63;

DP4A_SCALE_READY:
    ld.global.f32 %f3, [%rd9+0];

    shl.b32 %r63, %r12, 2;
    cvt.u64.u32 %rd26, %r63;
    add.s64 %rd27, %rd9, 260;
    add.s64 %rd28, %rd27, %rd26;
    ld.global.s16 %r64, [%rd28+0];
    ld.global.s16 %r65, [%rd28+2];
    add.s32 %r66, %r64, %r65;

    cvt.rn.f32.s32 %f4, %r48;
    cvt.rn.f32.u32 %f5, %r59;
    cvt.rn.f32.u32 %f6, %r62;
    cvt.rn.f32.s32 %f7, %r66;

    mul.rn.f32 %f8, %f1, %f5;
    mul.rn.f32 %f9, %f8, %f4;
    mul.rn.f32 %f10, %f2, %f6;
    mul.rn.f32 %f11, %f10, %f7;
    sub.rn.f32 %f12, %f9, %f11;
    mul.rn.f32 %f13, %f3, %f12;
    add.rn.f32 %f15, %f15, %f13;

DP4A_NEXT_PASS:
    add.u32 %r11, %r11, 1;
    bra DP4A_PASS_LOOP;

DP4A_PASSES_DONE:
    add.u32 %r9, %r9, 1;
    bra DP4A_BLOCK_LOOP;

DP4A_BLOCKS_DONE:
    // Subgroup leaders are lanes 0,8,16,24. Gather their two-group partial
    // sums to lane 0 with register shuffles; no global/shared atomics.
    mov.b32 %r70, %f15;
    shfl.sync.idx.b32 %r71, %r70, 8, 31, 0xffffffff;
    shfl.sync.idx.b32 %r72, %r70, 16, 31, 0xffffffff;
    shfl.sync.idx.b32 %r73, %r70, 24, 31, 0xffffffff;

    setp.ne.u32 %p8, %r4, 0;
    @%p8 bra DP4A_DONE;

    mov.b32 %f16, %r71;
    mov.b32 %f17, %r72;
    mov.b32 %f18, %r73;
    add.rn.f32 %f19, %f15, %f16;
    add.rn.f32 %f19, %f19, %f17;
    add.rn.f32 %f19, %f19, %f18;

    mul.wide.u32 %rd29, %r3, 4;
    add.s64 %rd30, %rd3, %rd29;
    st.global.f32 [%rd30], %f19;

DP4A_DONE:
    ret;
}
)ptx";
constexpr const char* kQ5KSm86SoAGemvVecPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

// Vectorized SM86 Q5_K SoA x F32.
// Four warps per CTA, one output row per warp.
// Eight 4-lane subgroups map 1:1 to the eight 32-value Q5_K groups.
// Each lane handles eight weights as two packed groups of four bytes.
// Scale/min correction is applied only after the 4-lane group reduction.
.visible .entry q38_q5k_sm86_soa_gemv_vec(
    .param .u64 p_meta,
    .param .u64 p_qh,
    .param .u64 p_qs,
    .param .u64 p_x,
    .param .u64 p_y,
    .param .u32 p_cols,
    .param .u32 p_rows
)
{
    .reg .pred %p<24>;
    .reg .b32 %r<128>;
    .reg .b64 %rd<44>;
    .reg .f32 %f<48>;

    ld.param.u64 %rd1, [p_meta];
    ld.param.u64 %rd2, [p_qh];
    ld.param.u64 %rd3, [p_qs];
    ld.param.u64 %rd4, [p_x];
    ld.param.u64 %rd5, [p_y];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_rows];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    shr.u32 %r5, %r4, 5;       // warp id 0..3
    and.b32 %r6, %r4, 31;      // lane 0..31
    shl.b32 %r7, %r3, 2;
    add.u32 %r7, %r7, %r5;     // output row

    setp.ge.u32 %p1, %r7, %r2;
    @%p1 bra Q5V_DONE;

    shr.u32 %r8, %r6, 2;       // Q5 group 0..7
    and.b32 %r9, %r6, 3;       // sublane 0..3

    shr.u32 %r10, %r1, 8;      // blocks_per_row
    mul.lo.u32 %r11, %r7, %r10;
    mov.u32 %r12, 0;            // block index
    mov.f32 %f30, 0f00000000;   // valid in subgroup leaders

Q5V_BLOCK_LOOP:
    setp.ge.u32 %p2, %r12, %r10;
    @%p2 bra Q5V_BLOCKS_DONE;

    add.u32 %r13, %r11, %r12;   // global block index

    // meta stride 20
    mul.wide.u32 %rd6, %r13, 20;
    add.s64 %rd7, %rd1, %rd6;

    // Each sublane owns eight consecutive qh bytes.
    shl.b32 %r14, %r13, 5;      // block*32
    shl.b32 %r15, %r9, 3;       // sublane*8
    add.u32 %r16, %r14, %r15;
    cvt.u64.u32 %rd8, %r16;
    add.s64 %rd9, %rd2, %rd8;
    ld.global.b32 %r40, [%rd9+0];
    ld.global.b32 %r41, [%rd9+4];

    // qs pair region: 4 x 32-byte regions per block.
    shr.u32 %r17, %r8, 1;       // pair 0..3
    shl.b32 %r18, %r13, 7;      // block*128
    shl.b32 %r19, %r17, 5;      // pair*32
    add.u32 %r20, %r18, %r19;
    add.u32 %r20, %r20, %r15;   // + sublane*8
    cvt.u64.u32 %rd10, %r20;
    add.s64 %rd11, %rd3, %rd10;
    ld.global.b32 %r42, [%rd11+0];
    ld.global.b32 %r43, [%rd11+4];

    // Select low/high nibble for this group.
    and.b32 %r21, %r8, 1;
    setp.eq.u32 %p3, %r21, 0;
    @%p3 and.b32 %r44, %r42, 0x0f0f0f0f;
    @%p3 and.b32 %r45, %r43, 0x0f0f0f0f;
    @!%p3 shr.u32 %r44, %r42, 4;
    @!%p3 shr.u32 %r45, %r43, 4;
    @!%p3 and.b32 %r44, %r44, 0x0f0f0f0f;
    @!%p3 and.b32 %r45, %r45, 0x0f0f0f0f;

    // Add Q5 fifth bit into bit4 of every packed byte.
    mov.u32 %r46, 0x01010101;
    shl.b32 %r46, %r46, %r8;

    and.b32 %r47, %r40, %r46;
    and.b32 %r48, %r41, %r46;

    setp.lt.u32 %p4, %r8, 4;
    @%p4 sub.u32 %r49, 4, %r8;
    @%p4 shl.b32 %r47, %r47, %r49;
    @%p4 shl.b32 %r48, %r48, %r49;
    @!%p4 sub.u32 %r49, %r8, 4;
    @!%p4 shr.u32 %r47, %r47, %r49;
    @!%p4 shr.u32 %r48, %r48, %r49;

    or.b32 %r50, %r44, %r47;    // first four q values
    or.b32 %r51, %r45, %r48;    // next four q values

    // Activation base: block*256 + group*32 + sublane*8.
    shl.b32 %r52, %r12, 8;
    shl.b32 %r53, %r8, 5;
    add.u32 %r54, %r52, %r53;
    add.u32 %r54, %r54, %r15;
    mul.wide.u32 %rd12, %r54, 4;
    add.s64 %rd13, %rd4, %rd12;

    ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [%rd13+0];
    ld.global.v4.f32 {%f5,%f6,%f7,%f8}, [%rd13+16];

    mov.f32 %f20, 0f00000000;   // sum(q*x)
    mov.f32 %f21, 0f00000000;   // sum(x)

    // q0
    and.b32 %r60, %r50, 255;
    cvt.rn.f32.u32 %f10, %r60;
    fma.rn.f32 %f20, %f10, %f1, %f20;
    add.rn.f32 %f21, %f21, %f1;

    // q1
    shr.u32 %r61, %r50, 8;
    and.b32 %r61, %r61, 255;
    cvt.rn.f32.u32 %f11, %r61;
    fma.rn.f32 %f20, %f11, %f2, %f20;
    add.rn.f32 %f21, %f21, %f2;

    // q2
    shr.u32 %r62, %r50, 16;
    and.b32 %r62, %r62, 255;
    cvt.rn.f32.u32 %f12, %r62;
    fma.rn.f32 %f20, %f12, %f3, %f20;
    add.rn.f32 %f21, %f21, %f3;

    // q3
    shr.u32 %r63, %r50, 24;
    cvt.rn.f32.u32 %f13, %r63;
    fma.rn.f32 %f20, %f13, %f4, %f20;
    add.rn.f32 %f21, %f21, %f4;

    // q4
    and.b32 %r64, %r51, 255;
    cvt.rn.f32.u32 %f14, %r64;
    fma.rn.f32 %f20, %f14, %f5, %f20;
    add.rn.f32 %f21, %f21, %f5;

    // q5
    shr.u32 %r65, %r51, 8;
    and.b32 %r65, %r65, 255;
    cvt.rn.f32.u32 %f15, %r65;
    fma.rn.f32 %f20, %f15, %f6, %f20;
    add.rn.f32 %f21, %f21, %f6;

    // q6
    shr.u32 %r66, %r51, 16;
    and.b32 %r66, %r66, 255;
    cvt.rn.f32.u32 %f16, %r66;
    fma.rn.f32 %f20, %f16, %f7, %f20;
    add.rn.f32 %f21, %f21, %f7;

    // q7
    shr.u32 %r67, %r51, 24;
    cvt.rn.f32.u32 %f17, %r67;
    fma.rn.f32 %f20, %f17, %f8, %f20;
    add.rn.f32 %f21, %f21, %f8;

    // Reduce q*x and x across the 4-lane group.
    mov.b32 %r70, %f20;
    shfl.sync.bfly.b32 %r71, %r70, 2, 31, 0xffffffff;
    mov.b32 %f22, %r71;
    add.rn.f32 %f20, %f20, %f22;
    mov.b32 %r70, %f20;
    shfl.sync.bfly.b32 %r71, %r70, 1, 31, 0xffffffff;
    mov.b32 %f22, %r71;
    add.rn.f32 %f20, %f20, %f22;

    mov.b32 %r72, %f21;
    shfl.sync.bfly.b32 %r73, %r72, 2, 31, 0xffffffff;
    mov.b32 %f23, %r73;
    add.rn.f32 %f21, %f21, %f23;
    mov.b32 %r72, %f21;
    shfl.sync.bfly.b32 %r73, %r72, 1, 31, 0xffffffff;
    mov.b32 %f23, %r73;
    add.rn.f32 %f21, %f21, %f23;

    setp.ne.u32 %p5, %r9, 0;
    @%p5 bra Q5V_NEXT_BLOCK;

    // Only subgroup leader applies d*scale and dmin*min.
    ld.global.b16 %r74, [%rd7+0];
    ld.global.b16 %r75, [%rd7+2];
    cvt.f32.f16 %f24, %r74;
    cvt.f32.f16 %f25, %r75;

    shl.b32 %r76, %r8, 1;
    cvt.u64.u32 %rd14, %r76;
    add.s64 %rd15, %rd7, 4;
    add.s64 %rd16, %rd15, %rd14;
    ld.global.b16 %r77, [%rd16];
    and.b32 %r78, %r77, 255;
    shr.u32 %r79, %r77, 8;

    cvt.rn.f32.u32 %f26, %r78;
    cvt.rn.f32.u32 %f27, %r79;
    mul.rn.f32 %f28, %f24, %f26;
    mul.rn.f32 %f29, %f25, %f27;
    mul.rn.f32 %f31, %f28, %f20;
    neg.f32 %f33, %f29;
    fma.rn.f32 %f31, %f33, %f21, %f31;
    add.rn.f32 %f30, %f30, %f31;

Q5V_NEXT_BLOCK:
    add.u32 %r12, %r12, 1;
    bra Q5V_BLOCK_LOOP;

Q5V_BLOCKS_DONE:
    // f30 is valid in lanes 0,4,8,...28.
    mov.b32 %r90, %f30;
    shfl.sync.down.b32 %r91, %r90, 16, 31, 0xffffffff;
    mov.b32 %f32, %r91;
    add.rn.f32 %f30, %f30, %f32;

    mov.b32 %r90, %f30;
    shfl.sync.down.b32 %r91, %r90, 8, 31, 0xffffffff;
    mov.b32 %f32, %r91;
    add.rn.f32 %f30, %f30, %f32;

    mov.b32 %r90, %f30;
    shfl.sync.down.b32 %r91, %r90, 4, 31, 0xffffffff;
    mov.b32 %f32, %r91;
    add.rn.f32 %f30, %f30, %f32;

    setp.ne.u32 %p6, %r6, 0;
    @%p6 bra Q5V_DONE;

    mul.wide.u32 %rd17, %r7, 4;
    add.s64 %rd18, %rd5, %rd17;
    st.global.f32 [%rd18], %f30;

Q5V_DONE:
    ret;
}
)ptx";

constexpr const char* kQ5KSm86SoAGemvPtx = R"ptx(
.version 7.1
.target sm_86
.address_size 64

.visible .entry q38_q5k_sm86_soa_gemv(
    .param .u64 p_meta,
    .param .u64 p_qh,
    .param .u64 p_qs,
    .param .u64 p_x,
    .param .u64 p_y,
    .param .u32 p_cols,
    .param .u32 p_rows
)
{
    .reg .pred %p<12>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<28>;
    .reg .f32 %f<18>;

    ld.param.u64 %rd1, [p_meta];
    ld.param.u64 %rd2, [p_qh];
    ld.param.u64 %rd3, [p_qs];
    ld.param.u64 %rd4, [p_x];
    ld.param.u64 %rd5, [p_y];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_rows];

    mov.u32 %r3, %ctaid.x;      // row
    mov.u32 %r4, %tid.x;        // lane
    setp.ge.u32 %p1, %r3, %r2;
    @%p1 bra SOA_DONE;
    setp.ge.u32 %p2, %r4, 32;
    @%p2 bra SOA_DONE;

    shr.u32 %r5, %r1, 8;        // blocks_per_row
    mul.lo.u32 %r6, %r3, %r5;  // first block index for row
    mov.u32 %r7, 0;             // block in row
    mov.f32 %f10, 0f00000000;

SOA_BLOCK_LOOP:
    setp.ge.u32 %p3, %r7, %r5;
    @%p3 bra SOA_BLOCKS_DONE;

    add.u32 %r8, %r6, %r7;      // global block index

    // meta stride 20
    mul.wide.u32 %rd6, %r8, 20;
    add.s64 %rd7, %rd1, %rd6;

    // qh stride 32; lane-local byte cached once for all 8 groups
    shl.b32 %r9, %r8, 5;
    add.u32 %r9, %r9, %r4;
    cvt.u64.u32 %rd8, %r9;
    add.s64 %rd9, %rd2, %rd8;
    ld.global.u8 %r39, [%rd9];

    // qs stride 128, pair area selected inside group loop
    shl.b32 %r10, %r8, 7;

    ld.global.b16 %r40, [%rd7+0];
    ld.global.b16 %r41, [%rd7+2];
    cvt.f32.f16 %f1, %r40;
    cvt.f32.f16 %f2, %r41;

    mov.u32 %r11, 0;            // group 0..7

SOA_GROUP_LOOP:
    setp.ge.u32 %p4, %r11, 8;
    @%p4 bra SOA_GROUPS_DONE;

    // Pre-expanded metadata: uint16 low8=scale, high8=min.
    shl.b32 %r12, %r11, 1;
    cvt.u64.u32 %rd10, %r12;
    add.s64 %rd11, %rd7, 4;
    add.s64 %rd12, %rd11, %rd10;
    ld.global.b16 %r13, [%rd12];
    and.b32 %r14, %r13, 255;
    shr.u32 %r15, %r13, 8;

    cvt.rn.f32.u32 %f3, %r14;
    cvt.rn.f32.u32 %f4, %r15;
    mul.rn.f32 %f5, %f1, %f3;
    mul.rn.f32 %f6, %f2, %f4;

    // qs: 4 pair regions x 32 bytes. One byte serves two adjacent groups.
    shr.u32 %r16, %r11, 1;
    shl.b32 %r16, %r16, 5;
    add.u32 %r16, %r16, %r10;
    add.u32 %r16, %r16, %r4;
    cvt.u64.u32 %rd13, %r16;
    add.s64 %rd14, %rd3, %rd13;

    and.b32 %r17, %r11, 1;
    setp.eq.u32 %p5, %r17, 0;
    @%p5 ld.global.u8 %r18, [%rd14];
    @%p5 and.b32 %r19, %r18, 15;
    @!%p5 shr.u32 %r19, %r18, 4;

    mov.u32 %r20, 1;
    shl.b32 %r20, %r20, %r11;
    and.b32 %r21, %r39, %r20;
    setp.ne.u32 %p6, %r21, 0;
    mov.u32 %r22, 0;
    @%p6 mov.u32 %r22, 16;
    add.u32 %r23, %r19, %r22;

    cvt.rn.f32.u32 %f7, %r23;
    mul.rn.f32 %f8, %f5, %f7;
    sub.rn.f32 %f9, %f8, %f6;

    // activation index = block*256 + group*32 + lane
    shl.b32 %r24, %r7, 8;
    shl.b32 %r25, %r11, 5;
    add.u32 %r26, %r24, %r25;
    add.u32 %r26, %r26, %r4;
    mul.wide.u32 %rd15, %r26, 4;
    add.s64 %rd16, %rd4, %rd15;
    ld.global.f32 %f11, [%rd16];

    fma.rn.f32 %f10, %f9, %f11, %f10;

    add.u32 %r11, %r11, 1;
    bra SOA_GROUP_LOOP;

SOA_GROUPS_DONE:
    add.u32 %r7, %r7, 1;
    bra SOA_BLOCK_LOOP;

SOA_BLOCKS_DONE:
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 16, 31, 0xffffffff;
    mov.b32 %f12, %r31;
    add.rn.f32 %f10, %f10, %f12;
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 8, 31, 0xffffffff;
    mov.b32 %f12, %r31;
    add.rn.f32 %f10, %f10, %f12;
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 4, 31, 0xffffffff;
    mov.b32 %f12, %r31;
    add.rn.f32 %f10, %f10, %f12;
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 2, 31, 0xffffffff;
    mov.b32 %f12, %r31;
    add.rn.f32 %f10, %f10, %f12;
    mov.b32 %r30, %f10;
    shfl.sync.down.b32 %r31, %r30, 1, 31, 0xffffffff;
    mov.b32 %f12, %r31;
    add.rn.f32 %f10, %f10, %f12;

    setp.eq.u32 %p7, %r4, 0;
    mul.wide.u32 %rd17, %r3, 4;
    add.s64 %rd18, %rd5, %rd17;
    @%p7 st.global.f32 [%rd18], %f10;

SOA_DONE:
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


bool NvidiaDriver::run_q4k_gemv_smoke(
    const std::byte* matrix,
    std::uint32_t cols,
    std::uint32_t rows,
    std::string* error,
    double* max_abs_error,
    double* max_rel_error,
    double* milliseconds,
    double* bandwidth_gbps) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error("q38 Q4_K GEMV requires sm_86; detected sm_" +
                                     std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (!matrix) throw std::invalid_argument("Q4_K matrix is null");
        if (cols == 0 || rows == 0 || (cols % kQ4KValuesPerBlock) != 0) {
            throw std::invalid_argument("Q4_K GEMV requires non-zero rows and cols divisible by 256");
        }

        const std::size_t blocks_per_row = cols / kQ4KValuesPerBlock;
        const std::size_t matrix_bytes =
            static_cast<std::size_t>(rows) * blocks_per_row * kQ4KBytesPerBlock;
        const std::size_t x_bytes = static_cast<std::size_t>(cols) * sizeof(float);
        const std::size_t y_bytes = static_cast<std::size_t>(rows) * sizeof(float);

        auto align256 = [](std::size_t v) { return (v + 255u) & ~std::size_t(255u); };
        const std::size_t matrix_off = 0;
        const std::size_t x_off = align256(matrix_bytes);
        const std::size_t y_off = align256(x_off + x_bytes);
        const std::size_t total_bytes = y_off + y_bytes;

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

        std::vector<float> x(cols), y(rows);
        for (std::uint32_t i = 0; i < cols; ++i) {
            const float fi = static_cast<float>(i);
            x[i] = std::sin(fi * 0.017f) * 0.65f + std::cos(fi * 0.011f) * 0.35f;
        }

        const CUdeviceptr matrix_ptr = memory.ptr() + matrix_off;
        const CUdeviceptr x_ptr = memory.ptr() + x_off;
        const CUdeviceptr y_ptr = memory.ptr() + y_off;
        check(handle_, memcpy_htod(matrix_ptr, matrix, matrix_bytes), "cuMemcpyHtoD(Q4_K matrix)");
        check(handle_, memcpy_htod(x_ptr, x.data(), x_bytes), "cuMemcpyHtoD(Q4_K GEMV x)");

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
            kQ4KGemvPtx,
            static_cast<unsigned int>(sizeof(jit_options) / sizeof(jit_options[0])),
            jit_options,
            jit_values);
        if (module_rc != CUDA_SUCCESS) {
            std::string detail = cuda_error(handle_, module_rc, "cuModuleLoadDataEx(Q4_K GEMV)");
            if (jit_error[0] != '\0') detail += std::string("\nPTX JIT error log:\n") + jit_error.data();
            if (jit_info[0] != '\0') detail += std::string("\nPTX JIT info log:\n") + jit_info.data();
            throw std::runtime_error(detail);
        }

        try {
            CUfunction fn{};
            check(handle_, module_get_function(&fn, module, "q38_q4k_gemv_f32"),
                  "cuModuleGetFunction(q38_q4k_gemv_f32)");

            CUdeviceptr arg_w = matrix_ptr;
            CUdeviceptr arg_x = x_ptr;
            CUdeviceptr arg_y = y_ptr;
            std::uint32_t arg_cols = cols;
            std::uint32_t arg_rows = rows;
            void* params[] = {&arg_w, &arg_x, &arg_y, &arg_cols, &arg_rows};

            check(handle_, launch(fn, rows, 1, 1, 32, 1, 1, 0, nullptr, params, nullptr),
                  "cuLaunchKernel(q38_q4k_gemv_f32)");
            check(handle_, sync(), "cuCtxSynchronize(Q4_K GEMV correctness)");
            check(handle_, memcpy_dtoh(y.data(), y_ptr, y_bytes), "cuMemcpyDtoH(Q4_K GEMV)");

            const std::size_t checked_rows = std::min<std::size_t>(rows, 8);
            std::array<float, kQ4KValuesPerBlock> deq{};
            double abs_max = 0.0;
            double rel_max = 0.0;
            for (std::size_t row = 0; row < checked_rows; ++row) {
                double ref = 0.0;
                const auto* row_ptr = matrix + row * blocks_per_row * kQ4KBytesPerBlock;
                for (std::size_t ib = 0; ib < blocks_per_row; ++ib) {
                    dequantize_q4_k_block_cpu(row_ptr + ib * kQ4KBytesPerBlock, deq);
                    const std::size_t base = ib * kQ4KValuesPerBlock;
                    for (std::size_t j = 0; j < kQ4KValuesPerBlock; ++j) {
                        ref += static_cast<double>(deq[j]) *
                               static_cast<double>(x[base + j]);
                    }
                }
                const double got = static_cast<double>(y[row]);
                const double abs_err = std::abs(got - ref);
                const double rel_err = abs_err / std::max(1.0e-5, std::abs(ref));
                abs_max = std::max(abs_max, abs_err);
                rel_max = std::max(rel_max, rel_err);
            }
            if (max_abs_error) *max_abs_error = abs_max;
            if (max_rel_error) *max_rel_error = rel_max;
            if (!(abs_max <= 2.0e-3 && rel_max <= 2.0e-3)) {
                std::ostringstream oss;
                oss << "Q4_K GEMV mismatch: max_abs=" << abs_max << " max_rel=" << rel_max;
                throw std::runtime_error(oss.str());
            }

            constexpr int kIters = 1000;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) {
                check(handle_, launch(fn, rows, 1, 1, 32, 1, 1, 0, nullptr, params, nullptr),
                      "cuLaunchKernel(q38_q4k_gemv_f32 benchmark)");
            }
            check(handle_, sync(), "cuCtxSynchronize(Q4_K GEMV benchmark)");
            const auto t1 = std::chrono::steady_clock::now();

            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count() /
                static_cast<double>(kIters);
            const double gbps =
                static_cast<double>(matrix_bytes) / (elapsed_ms / 1000.0) / 1.0e9;

            if (milliseconds) *milliseconds = elapsed_ms;
            if (bandwidth_gbps) *bandwidth_gbps = gbps;
        } catch (...) {
            module_unload(module);
            throw;
        }

        check(handle_, module_unload(module), "cuModuleUnload(Q4_K GEMV)");
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}



bool NvidiaDriver::run_q6k_gemv_smoke(
    const std::byte* matrix,
    std::uint32_t cols,
    std::uint32_t rows,
    std::string* error,
    double* max_abs_error,
    double* max_rel_error,
    double* milliseconds,
    double* bandwidth_gbps) {
    try {
        if (!available()) {
            throw std::runtime_error("driver is not initialized");
        }
        if (!is_sm86()) {
            throw std::runtime_error(
                "q38 Q6_K GEMV requires sm_86; detected sm_" +
                std::to_string(sm_major_) +
                std::to_string(sm_minor_));
        }
        if (!matrix) {
            throw std::invalid_argument("Q6_K matrix is null");
        }
        if (cols == 0 || rows == 0 ||
            (cols % kQ4KValuesPerBlock) != 0) {
            throw std::invalid_argument(
                "Q6_K GEMV requires non-zero rows and cols divisible by 256");
        }

        const std::size_t blocks_per_row =
            cols / kQ4KValuesPerBlock;
        const std::size_t matrix_bytes =
            static_cast<std::size_t>(rows) *
            blocks_per_row * kQ6KBytesPerBlock;
        const std::size_t x_bytes =
            static_cast<std::size_t>(cols) * sizeof(float);
        const std::size_t y_bytes =
            static_cast<std::size_t>(rows) * sizeof(float);

        auto align256 = [](std::size_t v) {
            return (v + 255u) & ~std::size_t(255u);
        };
        const std::size_t matrix_off = 0;
        const std::size_t x_off = align256(matrix_bytes);
        const std::size_t y_off = align256(x_off + x_bytes);
        const std::size_t total_bytes = y_off + y_bytes;

        ScopedVmm memory(handle_, device_ordinal_, total_bytes);

        using MemcpyHtoD =
            CUresult(*)(CUdeviceptr, const void*, std::size_t);
        using MemcpyDtoH =
            CUresult(*)(void*, CUdeviceptr, std::size_t);
        using ModuleLoadDataEx =
            CUresult(*)(CUmodule*, const void*, unsigned int, int*, void**);
        using ModuleUnload = CUresult(*)(CUmodule);
        using ModuleGetFunction =
            CUresult(*)(CUfunction*, CUmodule, const char*);
        using LaunchKernel =
            CUresult(*)(CUfunction,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, CUstream, void**, void**);
        using CtxSynchronize = CUresult(*)();

        const auto memcpy_htod =
            sym<MemcpyHtoD>(handle_, "cuMemcpyHtoD_v2");
        const auto memcpy_dtoh =
            sym<MemcpyDtoH>(handle_, "cuMemcpyDtoH_v2");
        const auto module_load_ex =
            sym<ModuleLoadDataEx>(handle_, "cuModuleLoadDataEx");
        const auto module_unload =
            sym<ModuleUnload>(handle_, "cuModuleUnload");
        const auto module_get_function =
            sym<ModuleGetFunction>(handle_, "cuModuleGetFunction");
        const auto launch =
            sym<LaunchKernel>(handle_, "cuLaunchKernel");
        const auto sync =
            sym<CtxSynchronize>(handle_, "cuCtxSynchronize");

        std::vector<float> x(cols), y(rows);
        for (std::uint32_t i = 0; i < cols; ++i) {
            const float fi = static_cast<float>(i);
            x[i] =
                std::sin(fi * 0.017f) * 0.65f +
                std::cos(fi * 0.011f) * 0.35f;
        }

        const CUdeviceptr matrix_ptr =
            memory.ptr() + matrix_off;
        const CUdeviceptr x_ptr =
            memory.ptr() + x_off;
        const CUdeviceptr y_ptr =
            memory.ptr() + y_off;

        check(handle_, memcpy_htod(
            matrix_ptr, matrix, matrix_bytes),
            "cuMemcpyHtoD(Q6_K matrix)");
        check(handle_, memcpy_htod(
            x_ptr, x.data(), x_bytes),
            "cuMemcpyHtoD(Q6_K x)");

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
            reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(jit_info.size())),
            jit_error.data(),
            reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(jit_error.size())),
            reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(1)),
        };

        CUmodule module{};
        const auto rc = module_load_ex(
            &module,
            kQ6KGemvPtx,
            static_cast<unsigned int>(
                sizeof(jit_options) / sizeof(jit_options[0])),
            jit_options,
            jit_values);
        if (rc != CUDA_SUCCESS) {
            std::string detail =
                cuda_error(
                    handle_, rc,
                    "cuModuleLoadDataEx(Q6_K GEMV)");
            if (jit_error[0] != '\0') {
                detail +=
                    std::string("\nPTX JIT error log:\n") +
                    jit_error.data();
            }
            if (jit_info[0] != '\0') {
                detail +=
                    std::string("\nPTX JIT info log:\n") +
                    jit_info.data();
            }
            throw std::runtime_error(detail);
        }

        try {
            CUfunction fn{};
            check(handle_, module_get_function(
                &fn, module, "q38_q6k_gemv_f32"),
                "cuModuleGetFunction(q38_q6k_gemv_f32)");

            CUdeviceptr arg_w = matrix_ptr;
            CUdeviceptr arg_x = x_ptr;
            CUdeviceptr arg_y = y_ptr;
            std::uint32_t arg_cols = cols;
            std::uint32_t arg_rows = rows;
            void* params[] = {
                &arg_w, &arg_x, &arg_y,
                &arg_cols, &arg_rows
            };

            const unsigned int grid_rows4 =
                (rows + 3u) / 4u;
            check(handle_, launch(
                fn,
                grid_rows4, 1, 1,
                128, 1, 1,
                0, nullptr, params, nullptr),
                "cuLaunchKernel(q38_q6k_gemv_f32)");
            check(handle_, sync(),
                  "cuCtxSynchronize(Q6_K correctness)");
            check(handle_, memcpy_dtoh(
                y.data(), y_ptr, y_bytes),
                "cuMemcpyDtoH(Q6_K)");

            const std::size_t checked_rows =
                std::min<std::size_t>(rows, 8);
            std::array<float, kQ4KValuesPerBlock> deq{};
            double abs_max = 0.0;
            double rel_max = 0.0;
            bool bad = false;

            for (std::size_t row = 0;
                 row < checked_rows; ++row) {
                double ref = 0.0;
                const auto* row_ptr =
                    matrix +
                    row * blocks_per_row *
                    kQ6KBytesPerBlock;

                for (std::size_t ib = 0;
                     ib < blocks_per_row; ++ib) {
                    dequantize_q6_k_block_cpu(
                        row_ptr +
                            ib * kQ6KBytesPerBlock,
                        deq);
                    const std::size_t base =
                        ib * kQ4KValuesPerBlock;
                    for (std::size_t j = 0;
                         j < kQ4KValuesPerBlock; ++j) {
                        ref +=
                            static_cast<double>(deq[j]) *
                            static_cast<double>(x[base + j]);
                    }
                }

                const double got =
                    static_cast<double>(y[row]);
                const double abs_err =
                    std::abs(got - ref);
                const double rel_err =
                    abs_err /
                    std::max(1.0e-5, std::abs(ref));
                abs_max = std::max(abs_max, abs_err);
                rel_max = std::max(rel_max, rel_err);
                if (abs_err > 2.0e-3 &&
                    rel_err > 2.0e-3) {
                    bad = true;
                }
            }

            if (max_abs_error) *max_abs_error = abs_max;
            if (max_rel_error) *max_rel_error = rel_max;
            if (bad) {
                std::ostringstream oss;
                oss << "Q6_K GEMV mismatch: max_abs="
                    << abs_max
                    << " max_rel=" << rel_max;
                throw std::runtime_error(oss.str());
            }

            const int iters =
                matrix_bytes >= (256ull << 20) ? 10 :
                matrix_bytes >= (64ull << 20) ? 30 : 200;

            const auto t0 =
                std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                check(handle_, launch(
                    fn,
                    grid_rows4, 1, 1,
                    128, 1, 1,
                    0, nullptr, params, nullptr),
                    "cuLaunchKernel(q38_q6k_gemv_f32 benchmark)");
            }
            check(handle_, sync(),
                  "cuCtxSynchronize(Q6_K benchmark)");
            const auto t1 =
                std::chrono::steady_clock::now();

            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    t1 - t0).count() /
                static_cast<double>(iters);
            const double gbps =
                static_cast<double>(matrix_bytes) /
                (elapsed_ms / 1000.0) /
                1.0e9;

            if (milliseconds) *milliseconds = elapsed_ms;
            if (bandwidth_gbps) *bandwidth_gbps = gbps;
        } catch (...) {
            module_unload(module);
            throw;
        }

        check(handle_, module_unload(module),
              "cuModuleUnload(Q6_K GEMV)");
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool NvidiaDriver::run_iq4xs_dequant_smoke(
    const std::byte* block,
    std::string* error,
    double* max_abs_error,
    double* max_rel_error) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error("q38 IQ4_XS smoke requires sm_86; detected sm_" +
                                     std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (!block) throw std::invalid_argument("IQ4_XS block is null");

        std::array<float, kQ4KValuesPerBlock> reference{};
        std::array<float, kQ4KValuesPerBlock> gpu{};
        dequantize_iq4_xs_block_cpu(block, reference);

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
        check(handle_, memcpy_htod(block_ptr, block, kIQ4XSBytesPerBlock),
              "cuMemcpyHtoD(IQ4_XS block)");

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
        const auto rc = module_load_ex(
            &module,
            kIQ4XSDequantPtx,
            static_cast<unsigned int>(sizeof(jit_options) / sizeof(jit_options[0])),
            jit_options,
            jit_values);
        if (rc != CUDA_SUCCESS) {
            std::string detail = cuda_error(handle_, rc, "cuModuleLoadDataEx(IQ4_XS)");
            if (jit_error[0] != '\0') detail += std::string("\nPTX JIT error log:\n") + jit_error.data();
            if (jit_info[0] != '\0') detail += std::string("\nPTX JIT info log:\n") + jit_info.data();
            throw std::runtime_error(detail);
        }

        try {
            CUfunction fn{};
            check(handle_, module_get_function(&fn, module, "q38_dequant_iq4xs_block"),
                  "cuModuleGetFunction(q38_dequant_iq4xs_block)");

            CUdeviceptr arg_block = block_ptr;
            CUdeviceptr arg_out = out_ptr;
            void* params[] = {&arg_block, &arg_out};
            check(handle_, launch(fn, 1, 1, 1, 256, 1, 1, 0, nullptr, params, nullptr),
                  "cuLaunchKernel(q38_dequant_iq4xs_block)");
            check(handle_, sync(), "cuCtxSynchronize(IQ4_XS)");
            check(handle_, memcpy_dtoh(gpu.data(), out_ptr, gpu.size() * sizeof(float)),
                  "cuMemcpyDtoH(IQ4_XS)");
        } catch (...) {
            module_unload(module);
            throw;
        }
        check(handle_, module_unload(module), "cuModuleUnload(IQ4_XS)");

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
            oss << "IQ4_XS dequant mismatch: max_abs=" << abs_max
                << " max_rel=" << rel_max;
            throw std::runtime_error(oss.str());
        }
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool NvidiaDriver::run_q5k_dequant_smoke(const std::byte* block, std::string* error, double* max_abs_error, double* max_rel_error) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error("q38 Q5_K smoke requires compute capability 8.6; detected sm_" +
                                     std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (!block) throw std::invalid_argument("Q5_K block is null");

        std::array<float, kQ4KValuesPerBlock> reference{};
        dequantize_q5_k_block_cpu(block, reference);
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
        check(handle_, memcpy_htod(block_ptr, block, kQ5KBytesPerBlock), "cuMemcpyHtoD(Q5_K block)");

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
            kQ5KDequantPtx,
            static_cast<unsigned int>(sizeof(jit_options) / sizeof(jit_options[0])),
            jit_options,
            jit_values);
        if (module_rc != CUDA_SUCCESS) {
            std::string detail = cuda_error(handle_, module_rc, "cuModuleLoadDataEx(Q5_K)");
            if (jit_error[0] != '\0') detail += std::string("\nPTX JIT error log:\n") + jit_error.data();
            if (jit_info[0] != '\0') detail += std::string("\nPTX JIT info log:\n") + jit_info.data();
            throw std::runtime_error(detail);
        }

        try {
            CUfunction fn{};
            check(handle_, module_get_function(&fn, module, "q38_dequant_q5k_block"),
                  "cuModuleGetFunction(q38_dequant_q5k_block)");

            CUdeviceptr arg_block = block_ptr;
            CUdeviceptr arg_out = out_ptr;
            void* params[] = {&arg_block, &arg_out};
            check(handle_, launch(fn, 1, 1, 1, 256, 1, 1, 0, nullptr, params, nullptr),
                  "cuLaunchKernel(q38_dequant_q5k_block)");
            check(handle_, sync(), "cuCtxSynchronize(Q5_K)");
            check(handle_, memcpy_dtoh(gpu.data(), out_ptr, gpu.size() * sizeof(float)),
                  "cuMemcpyDtoH(Q5_K)");
        } catch (...) {
            module_unload(module);
            throw;
        }
        check(handle_, module_unload(module), "cuModuleUnload(Q5_K)");

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
            oss << "Q5_K dequant mismatch: max_abs=" << abs_max << " max_rel=" << rel_max;
            throw std::runtime_error(oss.str());
        }
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}


bool NvidiaDriver::run_iq4xs_gemv_smoke(
    const std::byte* matrix,
    std::uint32_t cols,
    std::uint32_t rows,
    std::string* error,
    double* max_abs_error,
    double* max_rel_error,
    double* milliseconds,
    double* bandwidth_gbps) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error("q38 IQ4_XS GEMV requires sm_86; detected sm_" +
                                     std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (!matrix) throw std::invalid_argument("IQ4_XS matrix is null");
        if (cols == 0 || rows == 0 || (cols % kQ4KValuesPerBlock) != 0) {
            throw std::invalid_argument("IQ4_XS GEMV requires non-zero rows and cols divisible by 256");
        }

        const std::size_t blocks_per_row = cols / kQ4KValuesPerBlock;
        const std::size_t matrix_bytes =
            static_cast<std::size_t>(rows) * blocks_per_row * kIQ4XSBytesPerBlock;
        const std::size_t x_bytes = static_cast<std::size_t>(cols) * sizeof(float);
        const std::size_t y_bytes = static_cast<std::size_t>(rows) * sizeof(float);

        auto align256 = [](std::size_t v) { return (v + 255u) & ~std::size_t(255u); };
        const std::size_t matrix_off = 0;
        const std::size_t x_off = align256(matrix_bytes);
        const std::size_t y_off = align256(x_off + x_bytes);
        const std::size_t total_bytes = y_off + y_bytes;

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

        std::vector<float> x(cols), y(rows);
        for (std::uint32_t i = 0; i < cols; ++i) {
            const float fi = static_cast<float>(i);
            x[i] = std::sin(fi * 0.017f) * 0.65f + std::cos(fi * 0.011f) * 0.35f;
        }

        const CUdeviceptr matrix_ptr = memory.ptr() + matrix_off;
        const CUdeviceptr x_ptr = memory.ptr() + x_off;
        const CUdeviceptr y_ptr = memory.ptr() + y_off;

        check(handle_, memcpy_htod(matrix_ptr, matrix, matrix_bytes), "cuMemcpyHtoD(IQ4_XS matrix)");
        check(handle_, memcpy_htod(x_ptr, x.data(), x_bytes), "cuMemcpyHtoD(IQ4_XS x)");

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
        const auto rc = module_load_ex(
            &module,
            kIQ4XSGemvPrmtPtx,
            static_cast<unsigned int>(sizeof(jit_options) / sizeof(jit_options[0])),
            jit_options,
            jit_values);
        if (rc != CUDA_SUCCESS) {
            std::string detail = cuda_error(handle_, rc, "cuModuleLoadDataEx(IQ4_XS GEMV)");
            if (jit_error[0] != '\0') detail += std::string("\nPTX JIT error log:\n") + jit_error.data();
            if (jit_info[0] != '\0') detail += std::string("\nPTX JIT info log:\n") + jit_info.data();
            throw std::runtime_error(detail);
        }

        try {
            CUfunction fn{};
            check(handle_, module_get_function(&fn, module, "q38_iq4xs_gemv_f32_prmt"),
                  "cuModuleGetFunction(q38_iq4xs_gemv_f32_prmt)");

            CUdeviceptr arg_w = matrix_ptr;
            CUdeviceptr arg_x = x_ptr;
            CUdeviceptr arg_y = y_ptr;
            std::uint32_t arg_cols = cols;
            std::uint32_t arg_rows = rows;
            void* params[] = {&arg_w, &arg_x, &arg_y, &arg_cols, &arg_rows};

            const unsigned int grid_rows4 = (rows + 3u) / 4u;
            check(handle_, launch(fn, grid_rows4, 1, 1, 128, 1, 1, 0, nullptr, params, nullptr),
                  "cuLaunchKernel(q38_iq4xs_gemv_f32_prmt)");
            check(handle_, sync(), "cuCtxSynchronize(IQ4_XS GEMV correctness)");
            check(handle_, memcpy_dtoh(y.data(), y_ptr, y_bytes), "cuMemcpyDtoH(IQ4_XS GEMV)");

            const std::size_t checked_rows = std::min<std::size_t>(rows, 8);
            std::array<float, kQ4KValuesPerBlock> deq{};
            double abs_max = 0.0;
            double rel_max = 0.0;

            for (std::size_t row = 0; row < checked_rows; ++row) {
                double ref = 0.0;
                const auto* row_ptr =
                    matrix + row * blocks_per_row * kIQ4XSBytesPerBlock;
                for (std::size_t ib = 0; ib < blocks_per_row; ++ib) {
                    dequantize_iq4_xs_block_cpu(
                        row_ptr + ib * kIQ4XSBytesPerBlock, deq);
                    const std::size_t base = ib * kQ4KValuesPerBlock;
                    for (std::size_t j = 0; j < kQ4KValuesPerBlock; ++j) {
                        ref += static_cast<double>(deq[j]) *
                               static_cast<double>(x[base + j]);
                    }
                }

                const double got = static_cast<double>(y[row]);
                const double abs_err = std::abs(got - ref);
                const double rel_err = abs_err / std::max(1.0e-5, std::abs(ref));
                abs_max = std::max(abs_max, abs_err);
                rel_max = std::max(rel_max, rel_err);
            }

            if (max_abs_error) *max_abs_error = abs_max;
            if (max_rel_error) *max_rel_error = rel_max;
            if (!(abs_max <= 3.0e-3 && rel_max <= 3.0e-3)) {
                std::ostringstream oss;
                oss << "IQ4_XS GEMV mismatch: max_abs=" << abs_max
                    << " max_rel=" << rel_max;
                throw std::runtime_error(oss.str());
            }

            constexpr int kIters = 50;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) {
                check(handle_, launch(fn, grid_rows4, 1, 1, 128, 1, 1, 0, nullptr, params, nullptr),
                      "cuLaunchKernel(q38_iq4xs_gemv_f32_prmt benchmark)");
            }
            check(handle_, sync(), "cuCtxSynchronize(IQ4_XS GEMV benchmark)");
            const auto t1 = std::chrono::steady_clock::now();

            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count() /
                static_cast<double>(kIters);
            const double gbps =
                static_cast<double>(matrix_bytes) / (elapsed_ms / 1000.0) / 1.0e9;

            if (milliseconds) *milliseconds = elapsed_ms;
            if (bandwidth_gbps) *bandwidth_gbps = gbps;
        } catch (...) {
            module_unload(module);
            throw;
        }

        check(handle_, module_unload(module), "cuModuleUnload(IQ4_XS GEMV)");
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}


bool NvidiaDriver::run_iq4xs_sm86_gemv_smoke(
    const std::byte* repacked_matrix,
    std::size_t scale_offset,
    std::size_t qs_offset,
    std::uint32_t cols,
    std::uint32_t rows,
    std::string* error,
    double* max_abs_error,
    double* max_rel_error,
    double* milliseconds,
    double* original_equiv_gbps,
    double* physical_gbps) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error("q38 SM86 IQ4_XS GEMV requires sm_86; detected sm_" +
                                     std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (!repacked_matrix) throw std::invalid_argument("SM86 IQ4_XS matrix is null");
        if (cols == 0 || rows == 0 || (cols % kQ4KValuesPerBlock) != 0) {
            throw std::invalid_argument("SM86 IQ4_XS GEMV requires non-zero rows and cols divisible by 256");
        }

        const std::size_t blocks_per_row = cols / kQ4KValuesPerBlock;
        const std::size_t total_blocks =
            static_cast<std::size_t>(rows) * blocks_per_row;
        const std::size_t d_bytes = total_blocks * 2;
        const std::size_t scale_bytes = total_blocks * 8;
        const std::size_t qs_bytes = total_blocks * 128;
        if (scale_offset < d_bytes || qs_offset < scale_offset + scale_bytes) {
            throw std::invalid_argument("invalid SM86 IQ4_XS SoA offsets");
        }

        const std::size_t repacked_bytes = qs_offset + qs_bytes;
        const std::size_t original_bytes =
            total_blocks * kIQ4XSBytesPerBlock;
        const std::size_t x_bytes =
            static_cast<std::size_t>(cols) * sizeof(float);
        const std::size_t y_bytes =
            static_cast<std::size_t>(rows) * sizeof(float);

        auto align256 = [](std::size_t v) {
            return (v + 255u) & ~std::size_t(255u);
        };
        const std::size_t model_off = 0;
        const std::size_t x_off = align256(repacked_bytes);
        const std::size_t y_off = align256(x_off + x_bytes);
        const std::size_t total_bytes = y_off + y_bytes;

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

        const auto memcpy_htod =
            sym<MemcpyHtoD>(handle_, "cuMemcpyHtoD_v2");
        const auto memcpy_dtoh =
            sym<MemcpyDtoH>(handle_, "cuMemcpyDtoH_v2");
        const auto module_load_ex =
            sym<ModuleLoadDataEx>(handle_, "cuModuleLoadDataEx");
        const auto module_unload =
            sym<ModuleUnload>(handle_, "cuModuleUnload");
        const auto module_get_function =
            sym<ModuleGetFunction>(handle_, "cuModuleGetFunction");
        const auto launch =
            sym<LaunchKernel>(handle_, "cuLaunchKernel");
        const auto sync =
            sym<CtxSynchronize>(handle_, "cuCtxSynchronize");

        std::vector<float> x(cols), y(rows);
        for (std::uint32_t i = 0; i < cols; ++i) {
            const float fi = static_cast<float>(i);
            x[i] = std::sin(fi * 0.017f) * 0.65f +
                   std::cos(fi * 0.011f) * 0.35f;
        }

        const CUdeviceptr model_ptr = memory.ptr() + model_off;
        const CUdeviceptr d_ptr = model_ptr;
        const CUdeviceptr scale_ptr = model_ptr + scale_offset;
        const CUdeviceptr qs_ptr = model_ptr + qs_offset;
        const CUdeviceptr x_ptr = memory.ptr() + x_off;
        const CUdeviceptr y_ptr = memory.ptr() + y_off;

        check(handle_, memcpy_htod(
            model_ptr, repacked_matrix, repacked_bytes),
            "cuMemcpyHtoD(SM86 IQ4_XS SoA)");
        check(handle_, memcpy_htod(
            x_ptr, x.data(), x_bytes),
            "cuMemcpyHtoD(SM86 IQ4_XS x)");

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
        const auto rc = module_load_ex(
            &module,
            kIQ4XSSm86SoAGemvPtx,
            static_cast<unsigned int>(
                sizeof(jit_options) / sizeof(jit_options[0])),
            jit_options,
            jit_values);
        if (rc != CUDA_SUCCESS) {
            std::string detail =
                cuda_error(handle_, rc, "cuModuleLoadDataEx(SM86 IQ4_XS GEMV)");
            if (jit_error[0] != '\0') {
                detail += std::string("\nPTX JIT error log:\n") + jit_error.data();
            }
            if (jit_info[0] != '\0') {
                detail += std::string("\nPTX JIT info log:\n") + jit_info.data();
            }
            throw std::runtime_error(detail);
        }

        try {
            CUfunction fn{};
            check(handle_, module_get_function(
                &fn, module, "q38_iq4xs_sm86_soa_gemv"),
                "cuModuleGetFunction(q38_iq4xs_sm86_soa_gemv)");

            CUdeviceptr arg_d = d_ptr;
            CUdeviceptr arg_scale = scale_ptr;
            CUdeviceptr arg_qs = qs_ptr;
            CUdeviceptr arg_x = x_ptr;
            CUdeviceptr arg_y = y_ptr;
            std::uint32_t arg_cols = cols;
            std::uint32_t arg_rows = rows;
            void* params[] = {
                &arg_d, &arg_scale, &arg_qs, &arg_x, &arg_y,
                &arg_cols, &arg_rows
            };

            check(handle_, launch(
                fn, rows, 1, 1, 32, 1, 1, 0, nullptr, params, nullptr),
                "cuLaunchKernel(q38_iq4xs_sm86_soa_gemv)");
            check(handle_, sync(),
                  "cuCtxSynchronize(SM86 IQ4_XS correctness)");
            check(handle_, memcpy_dtoh(
                y.data(), y_ptr, y_bytes),
                "cuMemcpyDtoH(SM86 IQ4_XS)");

            const std::size_t checked_rows =
                std::min<std::size_t>(rows, 8);
            std::array<float, kQ4KValuesPerBlock> deq{};
            std::array<std::byte, kIQ4XSSm86BytesPerBlock> block{};
            double abs_max = 0.0;
            double rel_max = 0.0;

            for (std::size_t row = 0; row < checked_rows; ++row) {
                double ref = 0.0;
                for (std::size_t ib = 0; ib < blocks_per_row; ++ib) {
                    const std::size_t block_index =
                        row * blocks_per_row + ib;
                    std::memcpy(
                        block.data() + 0,
                        repacked_matrix + block_index * 2,
                        2);
                    std::memcpy(
                        block.data() + 2,
                        repacked_matrix + scale_offset + block_index * 8,
                        8);
                    std::memcpy(
                        block.data() + 10,
                        repacked_matrix + qs_offset + block_index * 128,
                        128);

                    dequantize_iq4_xs_sm86_block_cpu(
                        block.data(), deq);

                    const std::size_t base =
                        ib * kQ4KValuesPerBlock;
                    for (std::size_t j = 0;
                         j < kQ4KValuesPerBlock; ++j) {
                        ref += static_cast<double>(deq[j]) *
                               static_cast<double>(x[base + j]);
                    }
                }

                const double got = static_cast<double>(y[row]);
                const double abs_err = std::abs(got - ref);
                const double rel_err =
                    abs_err / std::max(1.0e-5, std::abs(ref));
                abs_max = std::max(abs_max, abs_err);
                rel_max = std::max(rel_max, rel_err);
            }

            if (max_abs_error) *max_abs_error = abs_max;
            if (max_rel_error) *max_rel_error = rel_max;
            if (!(abs_max <= 3.0e-3 && rel_max <= 3.0e-3)) {
                std::ostringstream oss;
                oss << "SM86 IQ4_XS GEMV mismatch: max_abs="
                    << abs_max << " max_rel=" << rel_max;
                throw std::runtime_error(oss.str());
            }

            constexpr int kIters = 50;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) {
                check(handle_, launch(
                    fn, rows, 1, 1, 32, 1, 1, 0, nullptr,
                    params, nullptr),
                    "cuLaunchKernel(q38_iq4xs_sm86_soa_gemv benchmark)");
            }
            check(handle_, sync(),
                  "cuCtxSynchronize(SM86 IQ4_XS benchmark)");
            const auto t1 = std::chrono::steady_clock::now();

            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    t1 - t0).count() /
                static_cast<double>(kIters);
            const double original_gbps =
                static_cast<double>(original_bytes) /
                (elapsed_ms / 1000.0) / 1.0e9;
            const double physical =
                static_cast<double>(repacked_bytes) /
                (elapsed_ms / 1000.0) / 1.0e9;

            if (milliseconds) *milliseconds = elapsed_ms;
            if (original_equiv_gbps) {
                *original_equiv_gbps = original_gbps;
            }
            if (physical_gbps) *physical_gbps = physical;
        } catch (...) {
            module_unload(module);
            throw;
        }

        check(handle_, module_unload(module),
              "cuModuleUnload(SM86 IQ4_XS GEMV)");
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}


bool NvidiaDriver::run_iq4xs_q8_1_gemv_smoke(
    const std::byte* matrix,
    std::uint32_t cols,
    std::uint32_t rows,
    std::string* error,
    double* max_abs_error,
    double* max_rel_error,
    double* milliseconds,
    double* bandwidth_gbps) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error(
                "q38 IQ4_XS x Q8_1 GEMV requires sm_86; detected sm_" +
                std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (!matrix) throw std::invalid_argument("IQ4_XS matrix is null");
        if (cols == 0 || rows == 0 ||
            (cols % kQ4KValuesPerBlock) != 0 ||
            (cols % kQ8_1ValuesPerBlock) != 0) {
            throw std::invalid_argument(
                "IQ4_XS x Q8_1 GEMV requires non-zero rows and cols divisible by 256");
        }

        const std::size_t iq_blocks_per_row =
            cols / kQ4KValuesPerBlock;
        const std::size_t q8_blocks =
            cols / kQ8_1ValuesPerBlock;
        const std::size_t matrix_bytes =
            static_cast<std::size_t>(rows) *
            iq_blocks_per_row * kIQ4XSBytesPerBlock;
        const std::size_t q8_bytes =
            q8_blocks * kQ8_1BytesPerBlock;
        const std::size_t y_bytes =
            static_cast<std::size_t>(rows) * sizeof(float);

        std::vector<float> x(cols);
        for (std::uint32_t i = 0; i < cols; ++i) {
            const float fi = static_cast<float>(i);
            x[i] =
                std::sin(fi * 0.017f) * 0.65f +
                std::cos(fi * 0.011f) * 0.35f;
        }
        const auto q8 =
            quantize_q8_1_cpu(x.data(), x.size());

        auto align256 = [](std::size_t v) {
            return (v + 255u) & ~std::size_t(255u);
        };
        const std::size_t matrix_off = 0;
        const std::size_t q8_off =
            align256(matrix_bytes);
        const std::size_t y_off =
            align256(q8_off + q8_bytes);
        const std::size_t total_bytes =
            y_off + y_bytes;

        ScopedVmm memory(
            handle_, device_ordinal_, total_bytes);

        using MemcpyHtoD =
            CUresult(*)(CUdeviceptr, const void*, std::size_t);
        using MemcpyDtoH =
            CUresult(*)(void*, CUdeviceptr, std::size_t);
        using ModuleLoadDataEx =
            CUresult(*)(CUmodule*, const void*, unsigned int, int*, void**);
        using ModuleUnload = CUresult(*)(CUmodule);
        using ModuleGetFunction =
            CUresult(*)(CUfunction*, CUmodule, const char*);
        using LaunchKernel =
            CUresult(*)(CUfunction,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, CUstream, void**, void**);
        using CtxSynchronize = CUresult(*)();

        const auto memcpy_htod =
            sym<MemcpyHtoD>(handle_, "cuMemcpyHtoD_v2");
        const auto memcpy_dtoh =
            sym<MemcpyDtoH>(handle_, "cuMemcpyDtoH_v2");
        const auto module_load_ex =
            sym<ModuleLoadDataEx>(handle_, "cuModuleLoadDataEx");
        const auto module_unload =
            sym<ModuleUnload>(handle_, "cuModuleUnload");
        const auto module_get_function =
            sym<ModuleGetFunction>(handle_, "cuModuleGetFunction");
        const auto launch =
            sym<LaunchKernel>(handle_, "cuLaunchKernel");
        const auto sync =
            sym<CtxSynchronize>(handle_, "cuCtxSynchronize");

        std::vector<float> y(rows);
        const CUdeviceptr matrix_ptr =
            memory.ptr() + matrix_off;
        const CUdeviceptr q8_ptr =
            memory.ptr() + q8_off;
        const CUdeviceptr y_ptr =
            memory.ptr() + y_off;

        check(handle_, memcpy_htod(
            matrix_ptr, matrix, matrix_bytes),
            "cuMemcpyHtoD(IQ4_XS matrix)");
        check(handle_, memcpy_htod(
            q8_ptr, q8.data(), q8_bytes),
            "cuMemcpyHtoD(Q8_1 activation)");

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
            reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(jit_info.size())),
            jit_error.data(),
            reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(jit_error.size())),
            reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(1)),
        };

        CUmodule module{};
        const auto rc = module_load_ex(
            &module,
            kIQ4XSQ81GemvPtx,
            static_cast<unsigned int>(
                sizeof(jit_options) / sizeof(jit_options[0])),
            jit_options,
            jit_values);
        if (rc != CUDA_SUCCESS) {
            std::string detail =
                cuda_error(
                    handle_, rc,
                    "cuModuleLoadDataEx(IQ4_XS x Q8_1 GEMV)");
            if (jit_error[0] != '\0') {
                detail +=
                    std::string("\nPTX JIT error log:\n") +
                    jit_error.data();
            }
            if (jit_info[0] != '\0') {
                detail +=
                    std::string("\nPTX JIT info log:\n") +
                    jit_info.data();
            }
            throw std::runtime_error(detail);
        }

        try {
            CUfunction fn{};
            check(handle_, module_get_function(
                &fn, module, "q38_iq4xs_q81_gemv"),
                "cuModuleGetFunction(q38_iq4xs_q81_gemv)");

            CUdeviceptr arg_w = matrix_ptr;
            CUdeviceptr arg_q8 = q8_ptr;
            CUdeviceptr arg_y = y_ptr;
            std::uint32_t arg_cols = cols;
            std::uint32_t arg_rows = rows;
            void* params[] = {
                &arg_w, &arg_q8, &arg_y,
                &arg_cols, &arg_rows
            };

            check(handle_, launch(
                fn, rows, 1, 1, 32, 1, 1,
                0, nullptr, params, nullptr),
                "cuLaunchKernel(q38_iq4xs_q81_gemv)");
            check(handle_, sync(),
                  "cuCtxSynchronize(IQ4_XS x Q8_1 correctness)");
            check(handle_, memcpy_dtoh(
                y.data(), y_ptr, y_bytes),
                "cuMemcpyDtoH(IQ4_XS x Q8_1)");

            std::vector<float> xq(cols);
            std::array<float, kQ8_1ValuesPerBlock> q8_deq{};
            for (std::size_t ib = 0;
                 ib < q8_blocks; ++ib) {
                dequantize_q8_1_block_cpu(
                    q8.data() + ib * kQ8_1BytesPerBlock,
                    q8_deq);
                std::copy(
                    q8_deq.begin(),
                    q8_deq.end(),
                    xq.begin() +
                        ib * kQ8_1ValuesPerBlock);
            }

            const std::size_t checked_rows =
                std::min<std::size_t>(rows, 8);
            std::array<float, kQ4KValuesPerBlock> deq{};
            double abs_max = 0.0;
            double rel_max = 0.0;

            for (std::size_t row = 0;
                 row < checked_rows; ++row) {
                double ref = 0.0;
                const auto* row_ptr =
                    matrix +
                    row * iq_blocks_per_row *
                        kIQ4XSBytesPerBlock;

                for (std::size_t ib = 0;
                     ib < iq_blocks_per_row; ++ib) {
                    dequantize_iq4_xs_block_cpu(
                        row_ptr +
                            ib * kIQ4XSBytesPerBlock,
                        deq);
                    const std::size_t base =
                        ib * kQ4KValuesPerBlock;
                    for (std::size_t j = 0;
                         j < kQ4KValuesPerBlock; ++j) {
                        ref +=
                            static_cast<double>(deq[j]) *
                            static_cast<double>(
                                xq[base + j]);
                    }
                }

                const double got =
                    static_cast<double>(y[row]);
                const double abs_err =
                    std::abs(got - ref);
                const double rel_err =
                    abs_err /
                    std::max(1.0e-5, std::abs(ref));

                abs_max =
                    std::max(abs_max, abs_err);
                rel_max =
                    std::max(rel_max, rel_err);
            }

            if (max_abs_error) *max_abs_error = abs_max;
            if (max_rel_error) *max_rel_error = rel_max;

            if (!(abs_max <= 3.0e-3 &&
                  rel_max <= 3.0e-3)) {
                std::ostringstream oss;
                oss
                    << "IQ4_XS x Q8_1 GEMV mismatch: max_abs="
                    << abs_max
                    << " max_rel=" << rel_max;
                throw std::runtime_error(oss.str());
            }

            constexpr int kIters = 50;
            const auto t0 =
                std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) {
                check(handle_, launch(
                    fn, rows, 1, 1, 32, 1, 1,
                    0, nullptr, params, nullptr),
                    "cuLaunchKernel(q38_iq4xs_q81_gemv benchmark)");
            }
            check(handle_, sync(),
                  "cuCtxSynchronize(IQ4_XS x Q8_1 benchmark)");
            const auto t1 =
                std::chrono::steady_clock::now();

            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    t1 - t0).count() /
                static_cast<double>(kIters);
            const double gbps =
                static_cast<double>(matrix_bytes) /
                (elapsed_ms / 1000.0) / 1.0e9;

            if (milliseconds) *milliseconds = elapsed_ms;
            if (bandwidth_gbps) *bandwidth_gbps = gbps;
        } catch (...) {
            module_unload(module);
            throw;
        }

        check(handle_, module_unload(module),
              "cuModuleUnload(IQ4_XS x Q8_1)");
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool NvidiaDriver::run_q5k_gemv_smoke(
    const std::byte* matrix,
    std::uint32_t cols,
    std::uint32_t rows,
    std::string* error,
    double* max_abs_error,
    double* max_rel_error,
    double* milliseconds,
    double* bandwidth_gbps) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error("q38 Q5_K GEMV requires sm_86; detected sm_" +
                                     std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (!matrix) throw std::invalid_argument("Q5_K matrix is null");
        if (cols == 0 || rows == 0 || (cols % kQ4KValuesPerBlock) != 0) {
            throw std::invalid_argument("Q5_K GEMV requires non-zero rows and cols divisible by 256");
        }

        const std::size_t blocks_per_row = cols / kQ4KValuesPerBlock;
        const std::size_t matrix_bytes =
            static_cast<std::size_t>(rows) * blocks_per_row * kQ5KBytesPerBlock;
        const std::size_t x_bytes = static_cast<std::size_t>(cols) * sizeof(float);
        const std::size_t y_bytes = static_cast<std::size_t>(rows) * sizeof(float);

        auto align256 = [](std::size_t v) { return (v + 255u) & ~std::size_t(255u); };
        const std::size_t matrix_off = 0;
        const std::size_t x_off = align256(matrix_bytes);
        const std::size_t y_off = align256(x_off + x_bytes);
        const std::size_t total_bytes = y_off + y_bytes;

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

        std::vector<float> x(cols);
        std::vector<float> y(rows);
        for (std::uint32_t i = 0; i < cols; ++i) {
            const float fi = static_cast<float>(i);
            x[i] = std::sin(fi * 0.017f) * 0.65f + std::cos(fi * 0.011f) * 0.35f;
        }

        const CUdeviceptr matrix_ptr = memory.ptr() + matrix_off;
        const CUdeviceptr x_ptr = memory.ptr() + x_off;
        const CUdeviceptr y_ptr = memory.ptr() + y_off;
        check(handle_, memcpy_htod(matrix_ptr, matrix, matrix_bytes), "cuMemcpyHtoD(Q5_K matrix)");
        check(handle_, memcpy_htod(x_ptr, x.data(), x_bytes), "cuMemcpyHtoD(GEMV x)");
        check(handle_, memset_d32(y_ptr, 0, rows), "cuMemsetD32(GEMV y)");

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
            kQ5KGemvPtx,
            static_cast<unsigned int>(sizeof(jit_options) / sizeof(jit_options[0])),
            jit_options,
            jit_values);
        if (module_rc != CUDA_SUCCESS) {
            std::string detail = cuda_error(handle_, module_rc, "cuModuleLoadDataEx(Q5_K GEMV)");
            if (jit_error[0] != '\0') detail += std::string("\nPTX JIT error log:\n") + jit_error.data();
            if (jit_info[0] != '\0') detail += std::string("\nPTX JIT info log:\n") + jit_info.data();
            throw std::runtime_error(detail);
        }

        CUfunction fn{};
        try {
            check(handle_, module_get_function(&fn, module, "q38_q5k_gemv_f32"),
                  "cuModuleGetFunction(q38_q5k_gemv_f32)");

            CUdeviceptr arg_w = matrix_ptr;
            CUdeviceptr arg_x = x_ptr;
            CUdeviceptr arg_y = y_ptr;
            std::uint32_t arg_cols = cols;
            std::uint32_t arg_rows = rows;
            void* params[] = {&arg_w, &arg_x, &arg_y, &arg_cols, &arg_rows};

            check(handle_, launch(fn, rows, 1, 1, 32, 1, 1, 0, nullptr, params, nullptr),
                  "cuLaunchKernel(q38_q5k_gemv_f32)");
            check(handle_, sync(), "cuCtxSynchronize(Q5_K GEMV correctness)");
            check(handle_, memcpy_dtoh(y.data(), y_ptr, y_bytes), "cuMemcpyDtoH(Q5_K GEMV)");

            const std::size_t checked_rows = std::min<std::size_t>(rows, 8);
            std::array<float, kQ4KValuesPerBlock> deq{};
            double abs_max = 0.0;
            double rel_max = 0.0;
            for (std::size_t row = 0; row < checked_rows; ++row) {
                double ref = 0.0;
                const auto* row_ptr =
                    matrix + row * blocks_per_row * kQ5KBytesPerBlock;
                for (std::size_t ib = 0; ib < blocks_per_row; ++ib) {
                    dequantize_q5_k_block_cpu(
                        row_ptr + ib * kQ5KBytesPerBlock, deq);
                    const std::size_t base = ib * kQ4KValuesPerBlock;
                    for (std::size_t j = 0; j < kQ4KValuesPerBlock; ++j) {
                        ref += static_cast<double>(deq[j]) *
                               static_cast<double>(x[base + j]);
                    }
                }
                const double got = static_cast<double>(y[row]);
                const double abs_err = std::abs(got - ref);
                const double rel_err = abs_err / std::max(1.0e-5, std::abs(ref));
                abs_max = std::max(abs_max, abs_err);
                rel_max = std::max(rel_max, rel_err);
            }
            if (max_abs_error) *max_abs_error = abs_max;
            if (max_rel_error) *max_rel_error = rel_max;
            if (!(abs_max <= 2.0e-3 && rel_max <= 2.0e-3)) {
                std::ostringstream oss;
                oss << "Q5_K GEMV mismatch: max_abs=" << abs_max
                    << " max_rel=" << rel_max;
                throw std::runtime_error(oss.str());
            }

            constexpr int kIters = 50;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) {
                check(handle_, launch(fn, rows, 1, 1, 32, 1, 1, 0, nullptr, params, nullptr),
                      "cuLaunchKernel(q38_q5k_gemv_f32 benchmark)");
            }
            check(handle_, sync(), "cuCtxSynchronize(Q5_K GEMV benchmark)");
            const auto t1 = std::chrono::steady_clock::now();
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count() /
                static_cast<double>(kIters);
            const double gbps =
                static_cast<double>(matrix_bytes) / (elapsed_ms / 1000.0) / 1.0e9;
            if (milliseconds) *milliseconds = elapsed_ms;
            if (bandwidth_gbps) *bandwidth_gbps = gbps;
        } catch (...) {
            module_unload(module);
            throw;
        }

        check(handle_, module_unload(module), "cuModuleUnload(Q5_K GEMV)");
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool NvidiaDriver::run_q5k_q8k_gemv_smoke(
    const std::byte* matrix,
    std::uint32_t cols,
    std::uint32_t rows,
    std::string* error,
    double* max_abs_error,
    double* max_rel_error,
    double* milliseconds,
    double* bandwidth_gbps) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error("q38 Q5_K x Q8_K GEMV requires sm_86; detected sm_" +
                                     std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (!matrix) throw std::invalid_argument("Q5_K matrix is null");
        if (cols == 0 || rows == 0 || (cols % kQ4KValuesPerBlock) != 0) {
            throw std::invalid_argument("Q5_K x Q8_K GEMV requires non-zero rows and cols divisible by 256");
        }

        const std::size_t blocks_per_row = cols / kQ4KValuesPerBlock;
        const std::size_t matrix_bytes =
            static_cast<std::size_t>(rows) * blocks_per_row * kQ5KBytesPerBlock;
        const std::size_t q8_bytes = blocks_per_row * kQ8KBytesPerBlock;
        const std::size_t y_bytes = static_cast<std::size_t>(rows) * sizeof(float);

        std::vector<float> x(cols);
        for (std::uint32_t i = 0; i < cols; ++i) {
            const float fi = static_cast<float>(i);
            x[i] = std::sin(fi * 0.017f) * 0.65f + std::cos(fi * 0.011f) * 0.35f;
        }
        const auto q8 = quantize_q8_k_cpu(x.data(), x.size());

        auto align256 = [](std::size_t v) { return (v + 255u) & ~std::size_t(255u); };
        const std::size_t matrix_off = 0;
        const std::size_t q8_off = align256(matrix_bytes);
        const std::size_t y_off = align256(q8_off + q8_bytes);
        const std::size_t total_bytes = y_off + y_bytes;

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

        std::vector<float> y(rows);
        const CUdeviceptr matrix_ptr = memory.ptr() + matrix_off;
        const CUdeviceptr q8_ptr = memory.ptr() + q8_off;
        const CUdeviceptr y_ptr = memory.ptr() + y_off;

        check(handle_, memcpy_htod(matrix_ptr, matrix, matrix_bytes), "cuMemcpyHtoD(Q5_K matrix)");
        check(handle_, memcpy_htod(q8_ptr, q8.data(), q8_bytes), "cuMemcpyHtoD(Q8_K activation)");
        check(handle_, memset_d32(y_ptr, 0, rows), "cuMemsetD32(Q5_K x Q8_K y)");

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
            kQ5KQ8KDp4aGemvPtx,
            static_cast<unsigned int>(sizeof(jit_options) / sizeof(jit_options[0])),
            jit_options,
            jit_values);
        if (module_rc != CUDA_SUCCESS) {
            std::string detail = cuda_error(handle_, module_rc, "cuModuleLoadDataEx(Q5_K x Q8_K GEMV)");
            if (jit_error[0] != '\0') detail += std::string("\nPTX JIT error log:\n") + jit_error.data();
            if (jit_info[0] != '\0') detail += std::string("\nPTX JIT info log:\n") + jit_info.data();
            throw std::runtime_error(detail);
        }

        CUfunction fn{};
        try {
            check(handle_, module_get_function(&fn, module, "q38_q5k_q8k_dp4a_gemv"),
                  "cuModuleGetFunction(q38_q5k_q8k_dp4a_gemv)");

            CUdeviceptr arg_w = matrix_ptr;
            CUdeviceptr arg_q8 = q8_ptr;
            CUdeviceptr arg_y = y_ptr;
            std::uint32_t arg_cols = cols;
            std::uint32_t arg_rows = rows;
            void* params[] = {&arg_w, &arg_q8, &arg_y, &arg_cols, &arg_rows};

            check(handle_, launch(fn, rows, 1, 1, 32, 1, 1, 0, nullptr, params, nullptr),
                  "cuLaunchKernel(q38_q5k_q8k_dp4a_gemv)");
            check(handle_, sync(), "cuCtxSynchronize(Q5_K x Q8_K correctness)");
            check(handle_, memcpy_dtoh(y.data(), y_ptr, y_bytes), "cuMemcpyDtoH(Q5_K x Q8_K)");

            const std::size_t checked_rows = std::min<std::size_t>(rows, 8);
            std::array<float, kQ4KValuesPerBlock> w_deq{};
            std::array<float, kQ4KValuesPerBlock> x_deq{};
            double abs_max = 0.0;
            double rel_max = 0.0;
            for (std::size_t row = 0; row < checked_rows; ++row) {
                double ref = 0.0;
                const auto* row_ptr =
                    matrix + row * blocks_per_row * kQ5KBytesPerBlock;
                for (std::size_t ib = 0; ib < blocks_per_row; ++ib) {
                    dequantize_q5_k_block_cpu(
                        row_ptr + ib * kQ5KBytesPerBlock, w_deq);
                    dequantize_q8_k_block_cpu(
                        q8.data() + ib * kQ8KBytesPerBlock, x_deq);
                    for (std::size_t j = 0; j < kQ4KValuesPerBlock; ++j) {
                        ref += static_cast<double>(w_deq[j]) *
                               static_cast<double>(x_deq[j]);
                    }
                }
                const double got = static_cast<double>(y[row]);
                const double abs_err = std::abs(got - ref);
                const double rel_err = abs_err / std::max(1.0e-5, std::abs(ref));
                abs_max = std::max(abs_max, abs_err);
                rel_max = std::max(rel_max, rel_err);
            }
            if (max_abs_error) *max_abs_error = abs_max;
            if (max_rel_error) *max_rel_error = rel_max;
            if (!(abs_max <= 2.0e-3 && rel_max <= 2.0e-3)) {
                std::ostringstream oss;
                oss << "Q5_K x Q8_K GEMV mismatch: max_abs=" << abs_max
                    << " max_rel=" << rel_max;
                throw std::runtime_error(oss.str());
            }

            // Clear once before timing. Atomic accumulation changes y across
            // iterations but not the amount of kernel work or contention.
            check(handle_, memset_d32(y_ptr, 0, rows), "cuMemsetD32(Q5_K x Q8_K benchmark)");
            constexpr int kIters = 50;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) {
                check(handle_, launch(fn, rows, 1, 1, 32, 1, 1, 0, nullptr, params, nullptr),
                      "cuLaunchKernel(q38_q5k_q8k_dp4a_gemv benchmark)");
            }
            check(handle_, sync(), "cuCtxSynchronize(Q5_K x Q8_K benchmark)");
            const auto t1 = std::chrono::steady_clock::now();

            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count() /
                static_cast<double>(kIters);
            const double gbps =
                static_cast<double>(matrix_bytes) / (elapsed_ms / 1000.0) / 1.0e9;
            if (milliseconds) *milliseconds = elapsed_ms;
            if (bandwidth_gbps) *bandwidth_gbps = gbps;
        } catch (...) {
            module_unload(module);
            throw;
        }

        check(handle_, module_unload(module), "cuModuleUnload(Q5_K x Q8_K GEMV)");
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool NvidiaDriver::run_q5k_sm86_gemv_smoke(
    const std::byte* repacked_matrix,
    std::size_t qh_offset,
    std::size_t qs_offset,
    std::uint32_t cols,
    std::uint32_t rows,
    std::string* error,
    double* max_abs_error,
    double* max_rel_error,
    double* milliseconds,
    double* original_equiv_gbps,
    double* physical_gbps) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error("q38 SM86 Q5_K GEMV requires sm_86; detected sm_" +
                                     std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (!repacked_matrix) throw std::invalid_argument("SM86 Q5_K matrix is null");
        if (cols == 0 || rows == 0 || (cols % kQ4KValuesPerBlock) != 0) {
            throw std::invalid_argument("SM86 Q5_K GEMV requires non-zero rows and cols divisible by 256");
        }

        const std::size_t blocks_per_row = cols / kQ4KValuesPerBlock;
        const std::size_t total_blocks = static_cast<std::size_t>(rows) * blocks_per_row;
        const std::size_t meta_bytes = total_blocks * 20;
        const std::size_t qh_bytes = total_blocks * 32;
        const std::size_t qs_bytes = total_blocks * 128;
        if (qh_offset < meta_bytes || qs_offset < qh_offset + qh_bytes) {
            throw std::invalid_argument("invalid SM86 Q5_K SoA offsets");
        }
        const std::size_t repacked_bytes = qs_offset + qs_bytes;
        const std::size_t original_bytes = total_blocks * kQ5KBytesPerBlock;
        const std::size_t x_bytes = static_cast<std::size_t>(cols) * sizeof(float);
        const std::size_t y_bytes = static_cast<std::size_t>(rows) * sizeof(float);

        auto align256 = [](std::size_t v) { return (v + 255u) & ~std::size_t(255u); };
        const std::size_t repacked_off = 0;
        const std::size_t x_off = align256(repacked_bytes);
        const std::size_t y_off = align256(x_off + x_bytes);
        const std::size_t total_bytes = y_off + y_bytes;

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

        std::vector<float> x(cols), y(rows);
        for (std::uint32_t i = 0; i < cols; ++i) {
            const float fi = static_cast<float>(i);
            x[i] = std::sin(fi * 0.017f) * 0.65f + std::cos(fi * 0.011f) * 0.35f;
        }

        const CUdeviceptr base_ptr = memory.ptr() + repacked_off;
        const CUdeviceptr meta_ptr = base_ptr;
        const CUdeviceptr qh_ptr = base_ptr + qh_offset;
        const CUdeviceptr qs_ptr = base_ptr + qs_offset;
        const CUdeviceptr x_ptr = memory.ptr() + x_off;
        const CUdeviceptr y_ptr = memory.ptr() + y_off;

        check(handle_, memcpy_htod(base_ptr, repacked_matrix, repacked_bytes), "cuMemcpyHtoD(SM86 Q5_K SoA)");
        check(handle_, memcpy_htod(x_ptr, x.data(), x_bytes), "cuMemcpyHtoD(SM86 GEMV x)");

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
            kQ5KSm86SoAGemvVecPtx,
            static_cast<unsigned int>(sizeof(jit_options) / sizeof(jit_options[0])),
            jit_options,
            jit_values);
        if (module_rc != CUDA_SUCCESS) {
            std::string detail = cuda_error(handle_, module_rc, "cuModuleLoadDataEx(SM86 Q5_K GEMV)");
            if (jit_error[0] != '\0') detail += std::string("\nPTX JIT error log:\n") + jit_error.data();
            if (jit_info[0] != '\0') detail += std::string("\nPTX JIT info log:\n") + jit_info.data();
            throw std::runtime_error(detail);
        }

        CUfunction fn{};
        try {
            check(handle_, module_get_function(&fn, module, "q38_q5k_sm86_soa_gemv_vec"),
                  "cuModuleGetFunction(q38_q5k_sm86_soa_gemv_vec)");

            CUdeviceptr arg_meta = meta_ptr;
            CUdeviceptr arg_qh = qh_ptr;
            CUdeviceptr arg_qs = qs_ptr;
            CUdeviceptr arg_x = x_ptr;
            CUdeviceptr arg_y = y_ptr;
            std::uint32_t arg_cols = cols;
            std::uint32_t arg_rows = rows;
            void* params[] = {
                &arg_meta, &arg_qh, &arg_qs, &arg_x, &arg_y, &arg_cols, &arg_rows
            };

            const unsigned int grid_rows4 = (rows + 3u) / 4u;
            check(handle_, launch(fn, grid_rows4, 1, 1, 128, 1, 1, 0, nullptr, params, nullptr),
                  "cuLaunchKernel(q38_q5k_sm86_soa_gemv_vec)");
            check(handle_, sync(), "cuCtxSynchronize(SM86 Q5_K correctness)");
            check(handle_, memcpy_dtoh(y.data(), y_ptr, y_bytes), "cuMemcpyDtoH(SM86 Q5_K GEMV)");

            const std::size_t checked_rows = std::min<std::size_t>(rows, 8);
            std::array<float, kQ4KValuesPerBlock> deq{};
            std::array<std::byte, kQ5KSm86BytesPerBlock> block{};
            double abs_max = 0.0;
            double rel_max = 0.0;
            bool tolerance_violation = false;
            constexpr double kAbsTol = 2.0e-3;
            constexpr double kRelTol = 2.0e-3;

            for (std::size_t row = 0; row < checked_rows; ++row) {
                double ref = 0.0;
                for (std::size_t ib = 0; ib < blocks_per_row; ++ib) {
                    const std::size_t block_index = row * blocks_per_row + ib;
                    std::memcpy(block.data() + 0,
                                repacked_matrix + block_index * 20,
                                20);
                    std::memcpy(block.data() + 20,
                                repacked_matrix + qh_offset + block_index * 32,
                                32);
                    std::memcpy(block.data() + 52,
                                repacked_matrix + qs_offset + block_index * 128,
                                128);
                    dequantize_q5_k_sm86_block_cpu(block.data(), deq);
                    const std::size_t base = ib * kQ4KValuesPerBlock;
                    for (std::size_t j = 0; j < kQ4KValuesPerBlock; ++j) {
                        ref += static_cast<double>(deq[j]) *
                               static_cast<double>(x[base + j]);
                    }
                }
                const double got = static_cast<double>(y[row]);
                const double abs_err = std::abs(got - ref);
                const double rel_err = abs_err / std::max(1.0e-5, std::abs(ref));
                abs_max = std::max(abs_max, abs_err);
                rel_max = std::max(rel_max, rel_err);

                // Relative error is ill-conditioned when the reference dot is
                // close to zero. Treat a row as wrong only when both its
                // absolute and relative errors exceed tolerance.
                if (abs_err > kAbsTol && rel_err > kRelTol) {
                    tolerance_violation = true;
                }
            }

            if (max_abs_error) *max_abs_error = abs_max;
            if (max_rel_error) *max_rel_error = rel_max;
            if (tolerance_violation) {
                std::ostringstream oss;
                oss << "SM86 Q5_K GEMV mismatch: max_abs=" << abs_max
                    << " max_rel=" << rel_max;
                throw std::runtime_error(oss.str());
            }

            constexpr int kIters = 50;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) {
                check(handle_, launch(fn, grid_rows4, 1, 1, 128, 1, 1, 0, nullptr, params, nullptr),
                      "cuLaunchKernel(q38_q5k_sm86_soa_gemv_vec benchmark)");
            }
            check(handle_, sync(), "cuCtxSynchronize(SM86 Q5_K benchmark)");
            const auto t1 = std::chrono::steady_clock::now();
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count() /
                static_cast<double>(kIters);

            const double original_gbps =
                static_cast<double>(original_bytes) / (elapsed_ms / 1000.0) / 1.0e9;
            const double physical =
                static_cast<double>(repacked_bytes) / (elapsed_ms / 1000.0) / 1.0e9;
            if (milliseconds) *milliseconds = elapsed_ms;
            if (original_equiv_gbps) *original_equiv_gbps = original_gbps;
            if (physical_gbps) *physical_gbps = physical;
        } catch (...) {
            module_unload(module);
            throw;
        }

        check(handle_, module_unload(module), "cuModuleUnload(SM86 Q5_K GEMV)");
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}





bool NvidiaDriver::run_qwen35_layer0_full(
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
    Layer0FullStats* stats,
    std::string* error) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error(
                "q38 layer0 recurrent front requires sm_86; detected sm_" +
                std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (!norm_weight || !qkv_matrix || !z_matrix ||
            !beta_matrix || !alpha_matrix ||
            !conv_weight || !dt_bias || !ssm_a ||
            !ssm_norm_weight || !ssm_out_matrix ||
            !post_norm_weight || !ffn_gate_matrix ||
            !ffn_up_matrix || !ffn_down_matrix) {
            throw std::invalid_argument(
                "layer0 full received null model tensor");
        }

        constexpr std::uint32_t kCols = 5120;
        constexpr std::uint32_t kQkvRows = 10240;
        constexpr std::uint32_t kZRows = 6144;
        constexpr std::uint32_t kSmallRows = 48;
        constexpr std::uint32_t kHeadDim = 128;
        constexpr std::uint32_t kQkHeads = 16;
        constexpr std::uint32_t kValueHeads = 48;
        constexpr std::uint32_t kFfnDim = 17408;
        constexpr std::uint32_t kKeyDim = kQkHeads * kHeadDim;
        constexpr std::uint32_t kValueDim = kValueHeads * kHeadDim;
        constexpr std::uint32_t kConvStateSteps = 3;
        constexpr std::size_t kGdnStateValues =
            static_cast<std::size_t>(kValueHeads) * kHeadDim * kHeadDim;

        const std::size_t q5_blocks_per_row = kCols / kQ4KValuesPerBlock;

        const std::size_t qkv_blocks =
            static_cast<std::size_t>(kQkvRows) * q5_blocks_per_row;
        const std::size_t qkv_meta_bytes = qkv_blocks * 20;
        const std::size_t qkv_qh_bytes = qkv_blocks * 32;
        const std::size_t qkv_qs_bytes = qkv_blocks * 128;
        if (qkv_qh_offset < qkv_meta_bytes ||
            qkv_qs_offset < qkv_qh_offset + qkv_qh_bytes) {
            throw std::invalid_argument("invalid qkv Q5_K plane offsets");
        }
        const std::size_t qkv_bytes = qkv_qs_offset + qkv_qs_bytes;

        const std::size_t z_blocks =
            static_cast<std::size_t>(kZRows) * q5_blocks_per_row;
        const std::size_t z_meta_bytes = z_blocks * 20;
        const std::size_t z_qh_bytes = z_blocks * 32;
        const std::size_t z_qs_bytes = z_blocks * 128;
        if (z_qh_offset < z_meta_bytes ||
            z_qs_offset < z_qh_offset + z_qh_bytes) {
            throw std::invalid_argument("invalid z Q5_K plane offsets");
        }
        const std::size_t z_bytes = z_qs_offset + z_qs_bytes;

        const std::size_t ssm_out_blocks_per_row =
            kValueDim / kQ4KValuesPerBlock;
        const std::size_t ssm_out_blocks =
            static_cast<std::size_t>(kCols) * ssm_out_blocks_per_row;
        const std::size_t ssm_out_meta_bytes = ssm_out_blocks * 20;
        const std::size_t ssm_out_qh_bytes = ssm_out_blocks * 32;
        const std::size_t ssm_out_qs_bytes = ssm_out_blocks * 128;
        if (ssm_out_qh_offset < ssm_out_meta_bytes ||
            ssm_out_qs_offset < ssm_out_qh_offset + ssm_out_qh_bytes) {
            throw std::invalid_argument("invalid ssm_out Q5_K plane offsets");
        }
        const std::size_t ssm_out_bytes =
            ssm_out_qs_offset + ssm_out_qs_bytes;
        const std::size_t ssm_norm_bytes =
            static_cast<std::size_t>(kHeadDim) * sizeof(float);
        const std::size_t post_norm_bytes =
            static_cast<std::size_t>(kCols) * sizeof(float);

        const std::size_t ffn_gate_blocks_per_row =
            kCols / kQ4KValuesPerBlock;
        const std::size_t ffn_gate_bytes =
            static_cast<std::size_t>(kFfnDim) *
            ffn_gate_blocks_per_row * kIQ4XSBytesPerBlock;

        const std::size_t ffn_up_blocks_per_row =
            kCols / kQ4KValuesPerBlock;
        const std::size_t ffn_up_blocks =
            static_cast<std::size_t>(kFfnDim) *
            ffn_up_blocks_per_row;
        const std::size_t ffn_up_meta_bytes = ffn_up_blocks * 20;
        const std::size_t ffn_up_qh_bytes = ffn_up_blocks * 32;
        const std::size_t ffn_up_qs_bytes = ffn_up_blocks * 128;
        if (ffn_up_qh_offset < ffn_up_meta_bytes ||
            ffn_up_qs_offset < ffn_up_qh_offset + ffn_up_qh_bytes) {
            throw std::invalid_argument("invalid ffn_up Q5_K plane offsets");
        }
        const std::size_t ffn_up_bytes =
            ffn_up_qs_offset + ffn_up_qs_bytes;

        const std::size_t ffn_down_blocks_per_row =
            kFfnDim / kQ4KValuesPerBlock;
        const std::size_t ffn_down_blocks =
            static_cast<std::size_t>(kCols) *
            ffn_down_blocks_per_row;
        const std::size_t ffn_down_meta_bytes = ffn_down_blocks * 20;
        const std::size_t ffn_down_qh_bytes = ffn_down_blocks * 32;
        const std::size_t ffn_down_qs_bytes = ffn_down_blocks * 128;
        if (ffn_down_qh_offset < ffn_down_meta_bytes ||
            ffn_down_qs_offset < ffn_down_qh_offset + ffn_down_qh_bytes) {
            throw std::invalid_argument("invalid ffn_down Q5_K plane offsets");
        }
        const std::size_t ffn_down_bytes =
            ffn_down_qs_offset + ffn_down_qs_bytes;

        const std::size_t q4_blocks_per_row =
            kCols / kQ4KValuesPerBlock;
        const std::size_t small_matrix_bytes =
            static_cast<std::size_t>(kSmallRows) *
            q4_blocks_per_row * kQ4KBytesPerBlock;

        const std::size_t hidden_bytes =
            static_cast<std::size_t>(kCols) * sizeof(float);
        const std::size_t ffn_bytes =
            static_cast<std::size_t>(kFfnDim) * sizeof(float);
        const std::size_t qkv_out_bytes =
            static_cast<std::size_t>(kQkvRows) * sizeof(float);
        const std::size_t z_out_bytes =
            static_cast<std::size_t>(kZRows) * sizeof(float);
        const std::size_t small_bytes =
            static_cast<std::size_t>(kSmallRows) * sizeof(float);
        const std::size_t conv_weight_bytes =
            static_cast<std::size_t>(4) * kQkvRows * sizeof(float);
        const std::size_t conv_state_bytes =
            static_cast<std::size_t>(kConvStateSteps) *
            kQkvRows * sizeof(float);
        const std::size_t qk_bytes =
            static_cast<std::size_t>(kKeyDim) * sizeof(float);
        const std::size_t gdn_out_bytes =
            static_cast<std::size_t>(kValueDim) * sizeof(float);
        const std::size_t gdn_state_bytes =
            kGdnStateValues * sizeof(float);

        auto align256 = [](std::size_t n) {
            return (n + 255u) & ~std::size_t(255u);
        };

        // Model tensors and persistent test-state inputs.
        const std::size_t qkv_w_off = 0;
        const std::size_t z_w_off = align256(qkv_w_off + qkv_bytes);
        const std::size_t beta_w_off = align256(z_w_off + z_bytes);
        const std::size_t alpha_w_off =
            align256(beta_w_off + small_matrix_bytes);
        const std::size_t norm_w_off =
            align256(alpha_w_off + small_matrix_bytes);
        const std::size_t conv_w_off =
            align256(norm_w_off + hidden_bytes);
        const std::size_t dt_off =
            align256(conv_w_off + conv_weight_bytes);
        const std::size_t a_off =
            align256(dt_off + small_bytes);
        const std::size_t ssm_norm_w_off =
            align256(a_off + small_bytes);
        const std::size_t ssm_out_w_off =
            align256(ssm_norm_w_off + ssm_norm_bytes);
        const std::size_t post_norm_w_off =
            align256(ssm_out_w_off + ssm_out_bytes);
        const std::size_t ffn_gate_w_off =
            align256(post_norm_w_off + post_norm_bytes);
        const std::size_t ffn_up_w_off =
            align256(ffn_gate_w_off + ffn_gate_bytes);
        const std::size_t ffn_down_w_off =
            align256(ffn_up_w_off + ffn_up_bytes);
        const std::size_t hidden_off =
            align256(ffn_down_w_off + ffn_down_bytes);
        const std::size_t conv_state_in_off =
            align256(hidden_off + hidden_bytes);
        const std::size_t gdn_state_in_off =
            align256(conv_state_in_off + conv_state_bytes);

        // Runtime workspace.
        const std::size_t sumsq_off =
            align256(gdn_state_in_off + gdn_state_bytes);
        const std::size_t normed_off =
            align256(sumsq_off + sizeof(float));
        const std::size_t qkv_out_off =
            align256(normed_off + hidden_bytes);
        const std::size_t z_out_off =
            align256(qkv_out_off + qkv_out_bytes);
        const std::size_t beta_raw_off =
            align256(z_out_off + z_out_bytes);
        const std::size_t alpha_raw_off =
            align256(beta_raw_off + small_bytes);
        const std::size_t conv_out_off =
            align256(alpha_raw_off + small_bytes);
        const std::size_t conv_state_out_off =
            align256(conv_out_off + qkv_out_bytes);
        const std::size_t q_out_off =
            align256(conv_state_out_off + conv_state_bytes);
        const std::size_t k_out_off =
            align256(q_out_off + qk_bytes);
        const std::size_t beta_out_off =
            align256(k_out_off + qk_bytes);
        const std::size_t gate_out_off =
            align256(beta_out_off + small_bytes);
        const std::size_t gdn_out_off =
            align256(gate_out_off + small_bytes);
        const std::size_t gdn_state_out_off =
            align256(gdn_out_off + gdn_out_bytes);
        const std::size_t gated_norm_out_off =
            align256(gdn_state_out_off + gdn_state_bytes);
        const std::size_t ssm_out_out_off =
            align256(gated_norm_out_off + gdn_out_bytes);
        const std::size_t residual_out_off =
            align256(ssm_out_out_off + hidden_bytes);
        const std::size_t ffn_normed_off =
            align256(residual_out_off + hidden_bytes);
        const std::size_t ffn_gate_out_off =
            align256(ffn_normed_off + hidden_bytes);
        const std::size_t ffn_up_out_off =
            align256(ffn_gate_out_off + ffn_bytes);
        const std::size_t ffn_mul_out_off =
            align256(ffn_up_out_off + ffn_bytes);
        const std::size_t ffn_down_out_off =
            align256(ffn_mul_out_off + ffn_bytes);
        const std::size_t layer_out_off =
            align256(ffn_down_out_off + hidden_bytes);
        const std::size_t total_bytes =
            layer_out_off + hidden_bytes;

        ScopedVmm memory(handle_, device_ordinal_, total_bytes);

        using MemcpyHtoD =
            CUresult(*)(CUdeviceptr, const void*, std::size_t);
        using MemcpyDtoH =
            CUresult(*)(void*, CUdeviceptr, std::size_t);
        using MemsetD32 =
            CUresult(*)(CUdeviceptr, unsigned int, std::size_t);
        using ModuleLoadDataEx =
            CUresult(*)(CUmodule*, const void*, unsigned int, int*, void**);
        using ModuleUnload = CUresult(*)(CUmodule);
        using ModuleGetFunction =
            CUresult(*)(CUfunction*, CUmodule, const char*);
        using LaunchKernel =
            CUresult(*)(CUfunction,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, CUstream, void**, void**);
        using CtxSynchronize = CUresult(*)();

        const auto memcpy_htod =
            sym<MemcpyHtoD>(handle_, "cuMemcpyHtoD_v2");
        const auto memcpy_dtoh =
            sym<MemcpyDtoH>(handle_, "cuMemcpyDtoH_v2");
        const auto memset_d32 =
            sym<MemsetD32>(handle_, "cuMemsetD32_v2");
        const auto module_load_ex =
            sym<ModuleLoadDataEx>(handle_, "cuModuleLoadDataEx");
        const auto module_unload =
            sym<ModuleUnload>(handle_, "cuModuleUnload");
        const auto module_get_function =
            sym<ModuleGetFunction>(handle_, "cuModuleGetFunction");
        const auto launch =
            sym<LaunchKernel>(handle_, "cuLaunchKernel");
        const auto sync =
            sym<CtxSynchronize>(handle_, "cuCtxSynchronize");

        // Deterministic decode-time input and recurrent states.
        std::vector<float> hidden(kCols);
        std::vector<float> conv_state_in(
            static_cast<std::size_t>(kConvStateSteps) * kQkvRows);
        std::vector<float> gdn_state_in(kGdnStateValues);

        for (std::uint32_t i = 0; i < kCols; ++i) {
            const float fi = static_cast<float>(i);
            hidden[i] =
                std::sin(fi * 0.017f) * 0.65f +
                std::cos(fi * 0.011f) * 0.35f;
        }
        for (std::uint32_t s = 0; s < kConvStateSteps; ++s) {
            for (std::uint32_t i = 0; i < kQkvRows; ++i) {
                const float fi = static_cast<float>(i);
                const float fs = static_cast<float>(s + 1);
                conv_state_in[
                    static_cast<std::size_t>(s) * kQkvRows + i] =
                    0.08f *
                        std::sin(fi * (0.0029f + 0.0006f * fs)) +
                    0.03f *
                        std::cos(fi * (0.0017f + 0.0004f * fs));
            }
        }
        for (std::size_t i = 0; i < gdn_state_in.size(); ++i) {
            const float fi = static_cast<float>(i);
            gdn_state_in[i] =
                0.011f * std::sin(fi * 0.0011f) +
                0.005f * std::cos(fi * 0.00073f);
        }

        const CUdeviceptr base = memory.ptr();
        const CUdeviceptr qkv_w_ptr = base + qkv_w_off;
        const CUdeviceptr z_w_ptr = base + z_w_off;
        const CUdeviceptr beta_w_ptr = base + beta_w_off;
        const CUdeviceptr alpha_w_ptr = base + alpha_w_off;
        const CUdeviceptr norm_w_ptr = base + norm_w_off;
        const CUdeviceptr conv_w_ptr = base + conv_w_off;
        const CUdeviceptr dt_ptr = base + dt_off;
        const CUdeviceptr a_ptr = base + a_off;
        const CUdeviceptr ssm_norm_w_ptr = base + ssm_norm_w_off;
        const CUdeviceptr ssm_out_w_ptr = base + ssm_out_w_off;
        const CUdeviceptr post_norm_w_ptr = base + post_norm_w_off;
        const CUdeviceptr ffn_gate_w_ptr = base + ffn_gate_w_off;
        const CUdeviceptr ffn_up_w_ptr = base + ffn_up_w_off;
        const CUdeviceptr ffn_down_w_ptr = base + ffn_down_w_off;
        const CUdeviceptr hidden_ptr = base + hidden_off;
        const CUdeviceptr conv_state_in_ptr = base + conv_state_in_off;
        const CUdeviceptr gdn_state_in_ptr = base + gdn_state_in_off;

        const CUdeviceptr sumsq_ptr = base + sumsq_off;
        const CUdeviceptr normed_ptr = base + normed_off;
        const CUdeviceptr qkv_out_ptr = base + qkv_out_off;
        const CUdeviceptr z_out_ptr = base + z_out_off;
        const CUdeviceptr beta_raw_ptr = base + beta_raw_off;
        const CUdeviceptr alpha_raw_ptr = base + alpha_raw_off;
        const CUdeviceptr conv_out_ptr = base + conv_out_off;
        const CUdeviceptr conv_state_out_ptr = base + conv_state_out_off;
        const CUdeviceptr q_out_ptr = base + q_out_off;
        const CUdeviceptr k_out_ptr = base + k_out_off;
        const CUdeviceptr beta_out_ptr = base + beta_out_off;
        const CUdeviceptr gate_out_ptr = base + gate_out_off;
        const CUdeviceptr gdn_out_ptr = base + gdn_out_off;
        const CUdeviceptr gdn_state_out_ptr = base + gdn_state_out_off;
        const CUdeviceptr gated_norm_out_ptr = base + gated_norm_out_off;
        const CUdeviceptr ssm_out_out_ptr = base + ssm_out_out_off;
        const CUdeviceptr residual_out_ptr = base + residual_out_off;
        const CUdeviceptr ffn_normed_ptr = base + ffn_normed_off;
        const CUdeviceptr ffn_gate_out_ptr = base + ffn_gate_out_off;
        const CUdeviceptr ffn_up_out_ptr = base + ffn_up_out_off;
        const CUdeviceptr ffn_mul_out_ptr = base + ffn_mul_out_off;
        const CUdeviceptr ffn_down_out_ptr = base + ffn_down_out_off;
        const CUdeviceptr layer_out_ptr = base + layer_out_off;

        check(handle_, memcpy_htod(
            qkv_w_ptr, qkv_matrix, qkv_bytes),
            "cuMemcpyHtoD(front qkv weight)");
        check(handle_, memcpy_htod(
            z_w_ptr, z_matrix, z_bytes),
            "cuMemcpyHtoD(front z weight)");
        check(handle_, memcpy_htod(
            beta_w_ptr, beta_matrix, small_matrix_bytes),
            "cuMemcpyHtoD(front beta weight)");
        check(handle_, memcpy_htod(
            alpha_w_ptr, alpha_matrix, small_matrix_bytes),
            "cuMemcpyHtoD(front alpha weight)");
        check(handle_, memcpy_htod(
            norm_w_ptr, norm_weight, hidden_bytes),
            "cuMemcpyHtoD(front norm weight)");
        check(handle_, memcpy_htod(
            conv_w_ptr, conv_weight, conv_weight_bytes),
            "cuMemcpyHtoD(front conv weight)");
        check(handle_, memcpy_htod(
            dt_ptr, dt_bias, small_bytes),
            "cuMemcpyHtoD(front dt bias)");
        check(handle_, memcpy_htod(
            a_ptr, ssm_a, small_bytes),
            "cuMemcpyHtoD(front ssm a)");
        check(handle_, memcpy_htod(
            ssm_norm_w_ptr, ssm_norm_weight, ssm_norm_bytes),
            "cuMemcpyHtoD(attention ssm norm)");
        check(handle_, memcpy_htod(
            ssm_out_w_ptr, ssm_out_matrix, ssm_out_bytes),
            "cuMemcpyHtoD(attention ssm out weight)");
        check(handle_, memcpy_htod(
            post_norm_w_ptr, post_norm_weight, post_norm_bytes),
            "cuMemcpyHtoD(ffn post norm)");
        check(handle_, memcpy_htod(
            ffn_gate_w_ptr, ffn_gate_matrix, ffn_gate_bytes),
            "cuMemcpyHtoD(ffn gate weight)");
        check(handle_, memcpy_htod(
            ffn_up_w_ptr, ffn_up_matrix, ffn_up_bytes),
            "cuMemcpyHtoD(ffn up weight)");
        check(handle_, memcpy_htod(
            ffn_down_w_ptr, ffn_down_matrix, ffn_down_bytes),
            "cuMemcpyHtoD(ffn down weight)");
        check(handle_, memcpy_htod(
            hidden_ptr, hidden.data(), hidden_bytes),
            "cuMemcpyHtoD(front hidden)");
        check(handle_, memcpy_htod(
            conv_state_in_ptr, conv_state_in.data(), conv_state_bytes),
            "cuMemcpyHtoD(front conv state)");
        check(handle_, memcpy_htod(
            gdn_state_in_ptr, gdn_state_in.data(), gdn_state_bytes),
            "cuMemcpyHtoD(front gdn state)");

        constexpr int CU_JIT_INFO_LOG_BUFFER = 3;
        constexpr int CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4;
        constexpr int CU_JIT_ERROR_LOG_BUFFER = 5;
        constexpr int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;
        constexpr int CU_JIT_LOG_VERBOSE = 12;

        auto load_module = [&](const char* ptx, const char* label) {
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
                reinterpret_cast<void*>(
                    static_cast<std::uintptr_t>(jit_info.size())),
                jit_error.data(),
                reinterpret_cast<void*>(
                    static_cast<std::uintptr_t>(jit_error.size())),
                reinterpret_cast<void*>(
                    static_cast<std::uintptr_t>(1)),
            };
            CUmodule module{};
            const auto rc = module_load_ex(
                &module,
                ptx,
                static_cast<unsigned int>(
                    sizeof(jit_options) / sizeof(jit_options[0])),
                jit_options,
                jit_values);
            if (rc != CUDA_SUCCESS) {
                std::string detail = cuda_error(handle_, rc, label);
                if (jit_error[0] != '\0') {
                    detail +=
                        std::string("\nPTX JIT error log:\n") +
                        jit_error.data();
                }
                if (jit_info[0] != '\0') {
                    detail +=
                        std::string("\nPTX JIT info log:\n") +
                        jit_info.data();
                }
                throw std::runtime_error(detail);
            }
            return module;
        };

        CUmodule norm_module =
            load_module(kRmsNormPtx, "cuModuleLoadDataEx(front RMSNorm)");
        CUmodule q5_module{};
        CUmodule q4_module{};
        CUmodule prep_module{};
        CUmodule gdn_module{};
        CUmodule tail_module{};
        CUmodule iq4_module{};

        try {
            q5_module = load_module(
                kQ5KSm86SoAGemvVecPtx,
                "cuModuleLoadDataEx(front Q5)");
            q4_module = load_module(
                kQ4KGemvPtx,
                "cuModuleLoadDataEx(front Q4)");
            prep_module = load_module(
                kRecurrentPrepPtx,
                "cuModuleLoadDataEx(front prep)");
            gdn_module = load_module(
                kGdnAr128Ptx,
                "cuModuleLoadDataEx(front GDN)");
            tail_module = load_module(
                kRecurrentTailPtx,
                "cuModuleLoadDataEx(attention tail)");
            iq4_module = load_module(
                kIQ4XSGemvPrmtPtx,
                "cuModuleLoadDataEx(ffn IQ4_XS)");

            CUfunction sum_fn{}, norm_fn{}, q5_fn{}, q4_fn{};
            CUfunction conv_fn{}, qk_fn{}, bg_fn{}, gdn_fn{};
            CUfunction gated_norm_fn{}, residual_fn{};
            CUfunction iq4_fn{}, ffn_mul_fn{};

            check(handle_, module_get_function(
                &sum_fn, norm_module, "q38_sumsq"),
                "cuModuleGetFunction(front sumsq)");
            check(handle_, module_get_function(
                &norm_fn, norm_module, "q38_rmsnorm_apply"),
                "cuModuleGetFunction(front rmsnorm)");
            check(handle_, module_get_function(
                &q5_fn, q5_module, "q38_q5k_sm86_soa_gemv_vec"),
                "cuModuleGetFunction(front q5)");
            check(handle_, module_get_function(
                &q4_fn, q4_module, "q38_q4k_gemv_f32"),
                "cuModuleGetFunction(front q4)");
            check(handle_, module_get_function(
                &conv_fn, prep_module, "q38_conv4_silu_roll"),
                "cuModuleGetFunction(front conv)");
            check(handle_, module_get_function(
                &qk_fn, prep_module, "q38_qk_l2norm_128"),
                "cuModuleGetFunction(front qk norm)");
            check(handle_, module_get_function(
                &bg_fn, prep_module, "q38_beta_gate_48"),
                "cuModuleGetFunction(front beta gate)");
            check(handle_, module_get_function(
                &gdn_fn, gdn_module, "q38_gdn_ar_128"),
                "cuModuleGetFunction(front gdn)");
            check(handle_, module_get_function(
                &gated_norm_fn, tail_module, "q38_gated_rmsnorm_silu_128"),
                "cuModuleGetFunction(attention gated norm)");
            check(handle_, module_get_function(
                &residual_fn, tail_module, "q38_add_residual_f32"),
                "cuModuleGetFunction(attention residual)");
            check(handle_, module_get_function(
                &ffn_mul_fn, tail_module, "q38_ffn_silu_mul_f32"),
                "cuModuleGetFunction(ffn silu mul)");
            check(handle_, module_get_function(
                &iq4_fn, iq4_module, "q38_iq4xs_gemv_f32_prmt"),
                "cuModuleGetFunction(ffn IQ4_XS)");

            // RMSNorm args.
            constexpr unsigned int norm_block = 256;
            const unsigned int norm_grid =
                (kCols + norm_block - 1) / norm_block;

            CUdeviceptr sum_x = hidden_ptr;
            CUdeviceptr sum_out = sumsq_ptr;
            std::uint32_t norm_count = kCols;
            void* sum_params[] = {
                &sum_x, &sum_out, &norm_count
            };

            CUdeviceptr norm_x = hidden_ptr;
            CUdeviceptr norm_w = norm_w_ptr;
            CUdeviceptr norm_y = normed_ptr;
            CUdeviceptr norm_sumsq = sumsq_ptr;
            float norm_eps = rms_eps;
            void* norm_params[] = {
                &norm_x, &norm_w, &norm_y,
                &norm_sumsq, &norm_count, &norm_eps
            };

            // Q5 qkv args.
            CUdeviceptr qkv_meta = qkv_w_ptr;
            CUdeviceptr qkv_qh = qkv_w_ptr + qkv_qh_offset;
            CUdeviceptr qkv_qs = qkv_w_ptr + qkv_qs_offset;
            CUdeviceptr qkv_x = normed_ptr;
            CUdeviceptr qkv_y = qkv_out_ptr;
            std::uint32_t qkv_cols = kCols;
            std::uint32_t qkv_rows = kQkvRows;
            void* qkv_params[] = {
                &qkv_meta, &qkv_qh, &qkv_qs, &qkv_x, &qkv_y,
                &qkv_cols, &qkv_rows
            };

            // Q5 z args.
            CUdeviceptr z_meta = z_w_ptr;
            CUdeviceptr z_qh = z_w_ptr + z_qh_offset;
            CUdeviceptr z_qs = z_w_ptr + z_qs_offset;
            CUdeviceptr z_x = normed_ptr;
            CUdeviceptr z_y = z_out_ptr;
            std::uint32_t z_cols = kCols;
            std::uint32_t z_rows = kZRows;
            void* z_params[] = {
                &z_meta, &z_qh, &z_qs, &z_x, &z_y,
                &z_cols, &z_rows
            };

            // Q4 beta/alpha args.
            CUdeviceptr beta_w = beta_w_ptr;
            CUdeviceptr beta_x = normed_ptr;
            CUdeviceptr beta_y = beta_raw_ptr;
            std::uint32_t beta_cols = kCols;
            std::uint32_t beta_rows = kSmallRows;
            void* beta_params[] = {
                &beta_w, &beta_x, &beta_y,
                &beta_cols, &beta_rows
            };

            CUdeviceptr alpha_w = alpha_w_ptr;
            CUdeviceptr alpha_x = normed_ptr;
            CUdeviceptr alpha_y = alpha_raw_ptr;
            std::uint32_t alpha_cols = kCols;
            std::uint32_t alpha_rows = kSmallRows;
            void* alpha_params[] = {
                &alpha_w, &alpha_x, &alpha_y,
                &alpha_cols, &alpha_rows
            };

            // Conv prep args.
            CUdeviceptr c_qkv = qkv_out_ptr;
            CUdeviceptr c_w = conv_w_ptr;
            CUdeviceptr c_si = conv_state_in_ptr;
            CUdeviceptr c_co = conv_out_ptr;
            CUdeviceptr c_so = conv_state_out_ptr;
            std::uint32_t c_channels = kQkvRows;
            float log2e = 1.4426950408889634f;
            void* conv_params[] = {
                &c_qkv, &c_w, &c_si, &c_co, &c_so,
                &c_channels, &log2e
            };

            CUdeviceptr n_conv = conv_out_ptr;
            CUdeviceptr n_q = q_out_ptr;
            CUdeviceptr n_k = k_out_ptr;
            float qk_eps = rms_eps;
            void* qk_params[] = {
                &n_conv, &n_q, &n_k, &qk_eps
            };

            CUdeviceptr b_br = beta_raw_ptr;
            CUdeviceptr b_ar = alpha_raw_ptr;
            CUdeviceptr b_dt = dt_ptr;
            CUdeviceptr b_a = a_ptr;
            CUdeviceptr b_bo = beta_out_ptr;
            CUdeviceptr b_go = gate_out_ptr;
            float inv_log2e = 0.6931471805599453f;
            void* bg_params[] = {
                &b_br, &b_ar, &b_dt, &b_a, &b_bo, &b_go,
                &log2e, &inv_log2e
            };

            // GDN args. v is the last 6144 values of conv_out.
            CUdeviceptr g_q = q_out_ptr;
            CUdeviceptr g_k = k_out_ptr;
            CUdeviceptr g_v =
                conv_out_ptr +
                static_cast<CUdeviceptr>(2 * kKeyDim * sizeof(float));
            CUdeviceptr g_gate = gate_out_ptr;
            CUdeviceptr g_beta = beta_out_ptr;
            CUdeviceptr g_state_in = gdn_state_in_ptr;
            CUdeviceptr g_out = gdn_out_ptr;
            CUdeviceptr g_state_out = gdn_state_out_ptr;
            std::uint32_t g_qk_heads = kQkHeads;
            std::uint32_t g_value_heads = kValueHeads;
            float g_scale =
                1.0f / std::sqrt(static_cast<float>(kHeadDim));
            void* gdn_params[] = {
                &g_q, &g_k, &g_v, &g_gate, &g_beta,
                &g_state_in, &g_out, &g_state_out,
                &g_qk_heads, &g_value_heads,
                &g_scale, &log2e
            };

            CUdeviceptr gn_input = gdn_out_ptr;
            CUdeviceptr gn_weight = ssm_norm_w_ptr;
            CUdeviceptr gn_z = z_out_ptr;
            CUdeviceptr gn_out = gated_norm_out_ptr;
            float gn_eps = rms_eps;
            void* gated_norm_params[] = {
                &gn_input, &gn_weight, &gn_z, &gn_out,
                &gn_eps, &log2e
            };

            CUdeviceptr so_meta = ssm_out_w_ptr;
            CUdeviceptr so_qh = ssm_out_w_ptr + ssm_out_qh_offset;
            CUdeviceptr so_qs = ssm_out_w_ptr + ssm_out_qs_offset;
            CUdeviceptr so_x = gated_norm_out_ptr;
            CUdeviceptr so_y = ssm_out_out_ptr;
            std::uint32_t so_cols = kValueDim;
            std::uint32_t so_rows = kCols;
            void* ssm_out_params[] = {
                &so_meta, &so_qh, &so_qs, &so_x, &so_y,
                &so_cols, &so_rows
            };

            CUdeviceptr r_x = ssm_out_out_ptr;
            CUdeviceptr r_residual = hidden_ptr;
            CUdeviceptr r_out = residual_out_ptr;
            std::uint32_t r_n = kCols;
            void* residual_params[] = {
                &r_x, &r_residual, &r_out, &r_n
            };

            CUdeviceptr fsum_x = residual_out_ptr;
            CUdeviceptr fsum_out = sumsq_ptr;
            std::uint32_t ffn_norm_count = kCols;
            void* ffn_sum_params[] = {
                &fsum_x, &fsum_out, &ffn_norm_count
            };

            CUdeviceptr fn_x = residual_out_ptr;
            CUdeviceptr fn_w = post_norm_w_ptr;
            CUdeviceptr fn_y = ffn_normed_ptr;
            CUdeviceptr fn_sumsq = sumsq_ptr;
            float ffn_norm_eps = rms_eps;
            void* ffn_norm_params[] = {
                &fn_x, &fn_w, &fn_y,
                &fn_sumsq, &ffn_norm_count, &ffn_norm_eps
            };

            CUdeviceptr fg_w = ffn_gate_w_ptr;
            CUdeviceptr fg_x = ffn_normed_ptr;
            CUdeviceptr fg_y = ffn_gate_out_ptr;
            std::uint32_t fg_cols = kCols;
            std::uint32_t fg_rows = kFfnDim;
            void* ffn_gate_params[] = {
                &fg_w, &fg_x, &fg_y, &fg_cols, &fg_rows
            };

            CUdeviceptr fu_meta = ffn_up_w_ptr;
            CUdeviceptr fu_qh = ffn_up_w_ptr + ffn_up_qh_offset;
            CUdeviceptr fu_qs = ffn_up_w_ptr + ffn_up_qs_offset;
            CUdeviceptr fu_x = ffn_normed_ptr;
            CUdeviceptr fu_y = ffn_up_out_ptr;
            std::uint32_t fu_cols = kCols;
            std::uint32_t fu_rows = kFfnDim;
            void* ffn_up_params[] = {
                &fu_meta, &fu_qh, &fu_qs, &fu_x, &fu_y,
                &fu_cols, &fu_rows
            };

            CUdeviceptr fm_gate = ffn_gate_out_ptr;
            CUdeviceptr fm_up = ffn_up_out_ptr;
            CUdeviceptr fm_out = ffn_mul_out_ptr;
            std::uint32_t fm_n = kFfnDim;
            void* ffn_mul_params[] = {
                &fm_gate, &fm_up, &fm_out, &fm_n, &log2e
            };

            CUdeviceptr fd_meta = ffn_down_w_ptr;
            CUdeviceptr fd_qh = ffn_down_w_ptr + ffn_down_qh_offset;
            CUdeviceptr fd_qs = ffn_down_w_ptr + ffn_down_qs_offset;
            CUdeviceptr fd_x = ffn_mul_out_ptr;
            CUdeviceptr fd_y = ffn_down_out_ptr;
            std::uint32_t fd_cols = kFfnDim;
            std::uint32_t fd_rows = kCols;
            void* ffn_down_params[] = {
                &fd_meta, &fd_qh, &fd_qs, &fd_x, &fd_y,
                &fd_cols, &fd_rows
            };

            CUdeviceptr lr_x = ffn_down_out_ptr;
            CUdeviceptr lr_residual = residual_out_ptr;
            CUdeviceptr lr_out = layer_out_ptr;
            std::uint32_t lr_n = kCols;
            void* layer_residual_params[] = {
                &lr_x, &lr_residual, &lr_out, &lr_n
            };

            const unsigned int qkv_grid =
                (kQkvRows + 3u) / 4u;
            const unsigned int z_grid =
                (kZRows + 3u) / 4u;
            const unsigned int conv_grid =
                (kQkvRows + 255u) / 256u;
            const unsigned int gdn_grid_y =
                (kHeadDim + 3u) / 4u;
            const unsigned int ssm_out_grid =
                (kCols + 3u) / 4u;
            const unsigned int residual_grid =
                (kCols + 255u) / 256u;
            const unsigned int ffn_gate_grid =
                (kFfnDim + 3u) / 4u;
            const unsigned int ffn_up_grid =
                (kFfnDim + 3u) / 4u;
            const unsigned int ffn_pointwise_grid =
                (kFfnDim + 255u) / 256u;
            const unsigned int ffn_down_grid =
                (kCols + 3u) / 4u;

            auto launch_projection = [&]() {
                check(handle_, memset_d32(
                    sumsq_ptr, 0, 1),
                    "cuMemsetD32(front sumsq)");
                check(handle_, launch(
                    sum_fn, norm_grid, 1, 1,
                    norm_block, 1, 1,
                    0, nullptr, sum_params, nullptr),
                    "cuLaunchKernel(front sumsq)");
                check(handle_, launch(
                    norm_fn, norm_grid, 1, 1,
                    norm_block, 1, 1,
                    0, nullptr, norm_params, nullptr),
                    "cuLaunchKernel(front rmsnorm)");
                check(handle_, launch(
                    q5_fn, qkv_grid, 1, 1,
                    128, 1, 1,
                    0, nullptr, qkv_params, nullptr),
                    "cuLaunchKernel(front qkv)");
                check(handle_, launch(
                    q5_fn, z_grid, 1, 1,
                    128, 1, 1,
                    0, nullptr, z_params, nullptr),
                    "cuLaunchKernel(front z)");
                check(handle_, launch(
                    q4_fn, kSmallRows, 1, 1,
                    32, 1, 1,
                    0, nullptr, beta_params, nullptr),
                    "cuLaunchKernel(front beta)");
                check(handle_, launch(
                    q4_fn, kSmallRows, 1, 1,
                    32, 1, 1,
                    0, nullptr, alpha_params, nullptr),
                    "cuLaunchKernel(front alpha)");
            };

            auto launch_prep = [&]() {
                check(handle_, launch(
                    conv_fn, conv_grid, 1, 1,
                    256, 1, 1,
                    0, nullptr, conv_params, nullptr),
                    "cuLaunchKernel(front conv)");
                check(handle_, launch(
                    qk_fn, 32, 1, 1,
                    128, 1, 1,
                    0, nullptr, qk_params, nullptr),
                    "cuLaunchKernel(front qk norm)");
                check(handle_, launch(
                    bg_fn, 1, 1, 1,
                    64, 1, 1,
                    0, nullptr, bg_params, nullptr),
                    "cuLaunchKernel(front beta gate)");
            };

            auto launch_gdn = [&]() {
                check(handle_, launch(
                    gdn_fn,
                    kValueHeads, gdn_grid_y, 1,
                    128, 1, 1,
                    0, nullptr, gdn_params, nullptr),
                    "cuLaunchKernel(front gdn)");
            };

            auto launch_gated_norm = [&]() {
                check(handle_, launch(
                    gated_norm_fn,
                    kValueHeads, 1, 1,
                    128, 1, 1,
                    0, nullptr, gated_norm_params, nullptr),
                    "cuLaunchKernel(attention gated norm)");
            };

            auto launch_ssm_out = [&]() {
                check(handle_, launch(
                    q5_fn,
                    ssm_out_grid, 1, 1,
                    128, 1, 1,
                    0, nullptr, ssm_out_params, nullptr),
                    "cuLaunchKernel(attention ssm_out)");
            };

            auto launch_residual = [&]() {
                check(handle_, launch(
                    residual_fn,
                    residual_grid, 1, 1,
                    256, 1, 1,
                    0, nullptr, residual_params, nullptr),
                    "cuLaunchKernel(attention residual)");
            };

            auto launch_front = [&]() {
                launch_projection();
                launch_prep();
                launch_gdn();
            };

            auto launch_tail = [&]() {
                launch_gated_norm();
                launch_ssm_out();
                launch_residual();
            };

            auto launch_attention = [&]() {
                launch_front();
                launch_tail();
            };

            auto launch_post_norm = [&]() {
                check(handle_, memset_d32(
                    sumsq_ptr, 0, 1),
                    "cuMemsetD32(ffn sumsq)");
                check(handle_, launch(
                    sum_fn, norm_grid, 1, 1,
                    norm_block, 1, 1,
                    0, nullptr, ffn_sum_params, nullptr),
                    "cuLaunchKernel(ffn sumsq)");
                check(handle_, launch(
                    norm_fn, norm_grid, 1, 1,
                    norm_block, 1, 1,
                    0, nullptr, ffn_norm_params, nullptr),
                    "cuLaunchKernel(ffn post norm)");
            };

            auto launch_ffn_gate = [&]() {
                check(handle_, launch(
                    iq4_fn,
                    ffn_gate_grid, 1, 1,
                    128, 1, 1,
                    0, nullptr, ffn_gate_params, nullptr),
                    "cuLaunchKernel(ffn gate IQ4_XS)");
            };

            auto launch_ffn_up = [&]() {
                check(handle_, launch(
                    q5_fn,
                    ffn_up_grid, 1, 1,
                    128, 1, 1,
                    0, nullptr, ffn_up_params, nullptr),
                    "cuLaunchKernel(ffn up Q5)");
            };

            auto launch_ffn_pointwise = [&]() {
                check(handle_, launch(
                    ffn_mul_fn,
                    ffn_pointwise_grid, 1, 1,
                    256, 1, 1,
                    0, nullptr, ffn_mul_params, nullptr),
                    "cuLaunchKernel(ffn silu mul)");
            };

            auto launch_ffn_down = [&]() {
                check(handle_, launch(
                    q5_fn,
                    ffn_down_grid, 1, 1,
                    128, 1, 1,
                    0, nullptr, ffn_down_params, nullptr),
                    "cuLaunchKernel(ffn down Q5)");
            };

            auto launch_layer_residual = [&]() {
                check(handle_, launch(
                    residual_fn,
                    residual_grid, 1, 1,
                    256, 1, 1,
                    0, nullptr, layer_residual_params, nullptr),
                    "cuLaunchKernel(layer residual)");
            };

            auto launch_ffn = [&]() {
                launch_post_norm();
                launch_ffn_gate();
                launch_ffn_up();
                launch_ffn_pointwise();
                launch_ffn_down();
                launch_layer_residual();
            };

            auto launch_chain = [&]() {
                launch_attention();
                launch_ffn();
            };

            // One full layer for correctness.
            launch_chain();
            check(handle_, sync(),
                  "cuCtxSynchronize(layer0 recurrent front correctness)");

            // Read GPU projection products used as the CPU downstream oracle.
            // The projection kernels are already independently validated by
            // q38-layer0-projections; this check focuses on composition.
            std::vector<float> qkv_gpu(kQkvRows);
            std::vector<float> beta_raw_gpu(kSmallRows);
            std::vector<float> alpha_raw_gpu(kSmallRows);
            std::vector<float> conv_gpu(kQkvRows);
            std::vector<float> conv_state_gpu(
                static_cast<std::size_t>(kConvStateSteps) * kQkvRows);
            std::vector<float> q_gpu(kKeyDim);
            std::vector<float> k_gpu(kKeyDim);
            std::vector<float> beta_gpu(kSmallRows);
            std::vector<float> gate_gpu(kSmallRows);
            std::vector<float> gdn_out_gpu(kValueDim);
            std::vector<float> gdn_state_gpu(kGdnStateValues);
            std::vector<float> z_gpu(kZRows);
            std::vector<float> gated_norm_gpu(kValueDim);
            std::vector<float> ssm_out_gpu(kCols);
            std::vector<float> residual_gpu(kCols);
            std::vector<float> ffn_normed_gpu(kCols);
            std::vector<float> ffn_gate_gpu(kFfnDim);
            std::vector<float> ffn_up_gpu(kFfnDim);
            std::vector<float> ffn_mul_gpu(kFfnDim);
            std::vector<float> ffn_down_gpu(kCols);
            std::vector<float> layer_out_gpu(kCols);

            check(handle_, memcpy_dtoh(
                qkv_gpu.data(), qkv_out_ptr, qkv_out_bytes),
                "cuMemcpyDtoH(front qkv)");
            check(handle_, memcpy_dtoh(
                beta_raw_gpu.data(), beta_raw_ptr, small_bytes),
                "cuMemcpyDtoH(front beta raw)");
            check(handle_, memcpy_dtoh(
                alpha_raw_gpu.data(), alpha_raw_ptr, small_bytes),
                "cuMemcpyDtoH(front alpha raw)");
            check(handle_, memcpy_dtoh(
                conv_gpu.data(), conv_out_ptr, qkv_out_bytes),
                "cuMemcpyDtoH(front conv)");
            check(handle_, memcpy_dtoh(
                conv_state_gpu.data(), conv_state_out_ptr, conv_state_bytes),
                "cuMemcpyDtoH(front conv state)");
            check(handle_, memcpy_dtoh(
                q_gpu.data(), q_out_ptr, qk_bytes),
                "cuMemcpyDtoH(front q)");
            check(handle_, memcpy_dtoh(
                k_gpu.data(), k_out_ptr, qk_bytes),
                "cuMemcpyDtoH(front k)");
            check(handle_, memcpy_dtoh(
                beta_gpu.data(), beta_out_ptr, small_bytes),
                "cuMemcpyDtoH(front beta)");
            check(handle_, memcpy_dtoh(
                gate_gpu.data(), gate_out_ptr, small_bytes),
                "cuMemcpyDtoH(front gate)");
            check(handle_, memcpy_dtoh(
                gdn_out_gpu.data(), gdn_out_ptr, gdn_out_bytes),
                "cuMemcpyDtoH(front gdn output)");
            check(handle_, memcpy_dtoh(
                gdn_state_gpu.data(), gdn_state_out_ptr, gdn_state_bytes),
                "cuMemcpyDtoH(front gdn state)");
            check(handle_, memcpy_dtoh(
                z_gpu.data(), z_out_ptr, z_out_bytes),
                "cuMemcpyDtoH(attention z)");
            check(handle_, memcpy_dtoh(
                gated_norm_gpu.data(), gated_norm_out_ptr, gdn_out_bytes),
                "cuMemcpyDtoH(attention gated norm)");
            check(handle_, memcpy_dtoh(
                ssm_out_gpu.data(), ssm_out_out_ptr, hidden_bytes),
                "cuMemcpyDtoH(attention ssm out)");
            check(handle_, memcpy_dtoh(
                residual_gpu.data(), residual_out_ptr, hidden_bytes),
                "cuMemcpyDtoH(attention residual)");
            check(handle_, memcpy_dtoh(
                ffn_normed_gpu.data(), ffn_normed_ptr, hidden_bytes),
                "cuMemcpyDtoH(ffn normed)");
            check(handle_, memcpy_dtoh(
                ffn_gate_gpu.data(), ffn_gate_out_ptr, ffn_bytes),
                "cuMemcpyDtoH(ffn gate)");
            check(handle_, memcpy_dtoh(
                ffn_up_gpu.data(), ffn_up_out_ptr, ffn_bytes),
                "cuMemcpyDtoH(ffn up)");
            check(handle_, memcpy_dtoh(
                ffn_mul_gpu.data(), ffn_mul_out_ptr, ffn_bytes),
                "cuMemcpyDtoH(ffn mul)");
            check(handle_, memcpy_dtoh(
                ffn_down_gpu.data(), ffn_down_out_ptr, hidden_bytes),
                "cuMemcpyDtoH(ffn down)");
            check(handle_, memcpy_dtoh(
                layer_out_gpu.data(), layer_out_ptr, hidden_bytes),
                "cuMemcpyDtoH(layer output)");

            std::vector<float> conv_ref(kQkvRows);
            std::vector<float> conv_state_ref(
                static_cast<std::size_t>(kConvStateSteps) * kQkvRows);
            std::vector<float> q_ref(kKeyDim);
            std::vector<float> k_ref(kKeyDim);
            std::vector<float> beta_ref(kSmallRows);
            std::vector<float> gate_ref(kSmallRows);
            std::vector<float> gdn_out_ref(kValueDim);
            std::vector<float> gdn_state_ref(kGdnStateValues);

            for (std::uint32_t ch = 0; ch < kQkvRows; ++ch) {
                const float x0 = conv_state_in[ch];
                const float x1 = conv_state_in[kQkvRows + ch];
                const float x2 = conv_state_in[2 * kQkvRows + ch];
                const float x3 = qkv_gpu[ch];
                const float* w =
                    conv_weight + static_cast<std::size_t>(ch) * 4;
                const double raw =
                    static_cast<double>(x0) * w[0] +
                    static_cast<double>(x1) * w[1] +
                    static_cast<double>(x2) * w[2] +
                    static_cast<double>(x3) * w[3];
                conv_ref[ch] =
                    static_cast<float>(
                        raw / (1.0 + std::exp(-raw)));
                conv_state_ref[ch] = x1;
                conv_state_ref[kQkvRows + ch] = x2;
                conv_state_ref[2 * kQkvRows + ch] = x3;
            }

            for (std::uint32_t h = 0; h < kQkHeads; ++h) {
                const std::size_t q_base =
                    static_cast<std::size_t>(h) * kHeadDim;
                const std::size_t k_base =
                    static_cast<std::size_t>(kKeyDim) +
                    static_cast<std::size_t>(h) * kHeadDim;

                double q_ss = 0.0;
                double k_ss = 0.0;
                for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                    const double qv = conv_ref[q_base + i];
                    const double kv = conv_ref[k_base + i];
                    q_ss += qv * qv;
                    k_ss += kv * kv;
                }
                const double q_inv =
                    1.0 / std::sqrt(q_ss + static_cast<double>(rms_eps));
                const double k_inv =
                    1.0 / std::sqrt(k_ss + static_cast<double>(rms_eps));

                for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                    q_ref[q_base + i] =
                        static_cast<float>(
                            static_cast<double>(conv_ref[q_base + i]) *
                            q_inv);
                    k_ref[q_base + i] =
                        static_cast<float>(
                            static_cast<double>(conv_ref[k_base + i]) *
                            k_inv);
                }
            }

            auto softplus = [](double x) {
                if (x > 20.0) return x;
                if (x < -20.0) return std::exp(x);
                return std::log1p(std::exp(x));
            };
            for (std::uint32_t h = 0; h < kSmallRows; ++h) {
                const double br = beta_raw_gpu[h];
                beta_ref[h] =
                    static_cast<float>(
                        1.0 / (1.0 + std::exp(-br)));
                const double biased =
                    static_cast<double>(alpha_raw_gpu[h]) +
                    static_cast<double>(dt_bias[h]);
                gate_ref[h] =
                    static_cast<float>(
                        softplus(biased) *
                        static_cast<double>(ssm_a[h]));
            }

            const double gdn_scale =
                1.0 / std::sqrt(static_cast<double>(kHeadDim));
            for (std::uint32_t h = 0; h < kValueHeads; ++h) {
                const std::uint32_t qh = h % kQkHeads;
                const float* qh_ptr =
                    q_ref.data() +
                    static_cast<std::size_t>(qh) * kHeadDim;
                const float* kh_ptr =
                    k_ref.data() +
                    static_cast<std::size_t>(qh) * kHeadDim;
                const double g_val =
                    std::exp(static_cast<double>(gate_ref[h]));
                const double beta_val =
                    static_cast<double>(beta_ref[h]);

                for (std::uint32_t col = 0; col < kHeadDim; ++col) {
                    const std::size_t state_base =
                        (static_cast<std::size_t>(h) * kHeadDim + col) *
                        kHeadDim;

                    double kv = 0.0;
                    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                        kv +=
                            static_cast<double>(
                                gdn_state_in[state_base + i]) *
                            static_cast<double>(kh_ptr[i]);
                    }

                    const std::size_t v_idx =
                        static_cast<std::size_t>(2 * kKeyDim) +
                        static_cast<std::size_t>(h) * kHeadDim + col;
                    const double delta =
                        (static_cast<double>(conv_ref[v_idx]) -
                         g_val * kv) *
                        beta_val;

                    double attn = 0.0;
                    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                        const double updated =
                            g_val *
                                static_cast<double>(
                                    gdn_state_in[state_base + i]) +
                            static_cast<double>(kh_ptr[i]) * delta;
                        gdn_state_ref[state_base + i] =
                            static_cast<float>(updated);
                        attn +=
                            updated * static_cast<double>(qh_ptr[i]);
                    }

                    gdn_out_ref[
                        static_cast<std::size_t>(h) * kHeadDim + col] =
                        static_cast<float>(attn * gdn_scale);
                }
            }

            std::vector<float> gated_norm_ref(kValueDim);
            for (std::uint32_t h = 0; h < kValueHeads; ++h) {
                const std::size_t base_h =
                    static_cast<std::size_t>(h) * kHeadDim;
                double ss = 0.0;
                for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                    const double x =
                        static_cast<double>(gdn_out_ref[base_h + i]);
                    ss += x * x;
                }
                const double inv_rms =
                    1.0 / std::sqrt(
                        ss / static_cast<double>(kHeadDim) +
                        static_cast<double>(rms_eps));
                for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                    const std::size_t idx = base_h + i;
                    const double zv = static_cast<double>(z_gpu[idx]);
                    const double silu_z =
                        zv / (1.0 + std::exp(-zv));
                    gated_norm_ref[idx] =
                        static_cast<float>(
                            static_cast<double>(gdn_out_ref[idx]) *
                            inv_rms *
                            static_cast<double>(ssm_norm_weight[i]) *
                            silu_z);
                }
            }

            constexpr std::size_t kTailCheckedRows = 8;
            std::array<float, kQ4KValuesPerBlock> tail_deq{};
            std::array<std::byte, kQ5KSm86BytesPerBlock> tail_block{};
            std::array<double, kTailCheckedRows> ssm_out_ref{};
            std::array<double, kTailCheckedRows> residual_ref{};

            for (std::size_t row = 0; row < kTailCheckedRows; ++row) {
                double dot = 0.0;
                for (std::size_t ib = 0;
                     ib < ssm_out_blocks_per_row; ++ib) {
                    const std::size_t block_index =
                        row * ssm_out_blocks_per_row + ib;
                    std::memcpy(
                        tail_block.data() + 0,
                        ssm_out_matrix + block_index * 20,
                        20);
                    std::memcpy(
                        tail_block.data() + 20,
                        ssm_out_matrix +
                            ssm_out_qh_offset + block_index * 32,
                        32);
                    std::memcpy(
                        tail_block.data() + 52,
                        ssm_out_matrix +
                            ssm_out_qs_offset + block_index * 128,
                        128);
                    dequantize_q5_k_sm86_block_cpu(
                        tail_block.data(), tail_deq);

                    const std::size_t base_x =
                        ib * kQ4KValuesPerBlock;
                    for (std::size_t j = 0;
                         j < kQ4KValuesPerBlock; ++j) {
                        dot +=
                            static_cast<double>(tail_deq[j]) *
                            static_cast<double>(
                                gated_norm_ref[base_x + j]);
                    }
                }
                ssm_out_ref[row] = dot;
                residual_ref[row] =
                    dot + static_cast<double>(hidden[row]);
            }

            Layer0RecurrentFrontStats front_local{};

            auto calc_max_abs = [](const std::vector<float>& a,
                                   const std::vector<float>& b) {
                double m = 0.0;
                for (std::size_t i = 0; i < a.size(); ++i) {
                    m = std::max(
                        m,
                        std::abs(
                            static_cast<double>(a[i]) -
                            static_cast<double>(b[i])));
                }
                return m;
            };

            auto calc_max_rel = [](const std::vector<float>& a,
                                   const std::vector<float>& b) {
                double m = 0.0;
                for (std::size_t i = 0; i < a.size(); ++i) {
                    const double ref = static_cast<double>(b[i]);
                    const double abs_err =
                        std::abs(
                            static_cast<double>(a[i]) - ref);
                    m = std::max(
                        m,
                        abs_err /
                            std::max(1.0e-6, std::abs(ref)));
                }
                return m;
            };

            front_local.conv_max_abs =
                calc_max_abs(conv_gpu, conv_ref);
            front_local.q_max_abs =
                calc_max_abs(q_gpu, q_ref);
            front_local.k_max_abs =
                calc_max_abs(k_gpu, k_ref);
            front_local.beta_max_abs =
                calc_max_abs(beta_gpu, beta_ref);
            front_local.gate_max_abs =
                calc_max_abs(gate_gpu, gate_ref);
            front_local.conv_state_max_abs =
                calc_max_abs(conv_state_gpu, conv_state_ref);
            front_local.gdn_output_max_abs =
                calc_max_abs(gdn_out_gpu, gdn_out_ref);
            front_local.gdn_output_max_rel =
                calc_max_rel(gdn_out_gpu, gdn_out_ref);
            front_local.gdn_state_max_abs =
                calc_max_abs(gdn_state_gpu, gdn_state_ref);
            front_local.gdn_state_max_rel =
                calc_max_rel(gdn_state_gpu, gdn_state_ref);

            if (front_local.conv_max_abs > 1.0e-3 ||
                front_local.q_max_abs > 1.0e-3 ||
                front_local.k_max_abs > 1.0e-3 ||
                front_local.beta_max_abs > 1.0e-4 ||
                front_local.gate_max_abs > 1.0e-3 ||
                front_local.conv_state_max_abs > 1.0e-6 ||
                (front_local.gdn_output_max_abs > 2.0e-3 &&
                 front_local.gdn_output_max_rel > 2.0e-3) ||
                (front_local.gdn_state_max_abs > 1.0e-3 &&
                 front_local.gdn_state_max_rel > 1.0e-3)) {
                std::ostringstream oss;
                oss << "layer0 recurrent front mismatch:"
                    << " conv=" << front_local.conv_max_abs
                    << " q=" << front_local.q_max_abs
                    << " k=" << front_local.k_max_abs
                    << " beta=" << front_local.beta_max_abs
                    << " gate=" << front_local.gate_max_abs
                    << " conv_state=" << front_local.conv_state_max_abs
                    << " gdn_out_abs=" << front_local.gdn_output_max_abs
                    << " gdn_out_rel=" << front_local.gdn_output_max_rel
                    << " gdn_state_abs=" << front_local.gdn_state_max_abs
                    << " gdn_state_rel=" << front_local.gdn_state_max_rel;
                throw std::runtime_error(oss.str());
            }

            std::vector<float> ffn_normed_ref(kCols);
            double ffn_ss = 0.0;
            for (float x : residual_gpu) {
                ffn_ss += static_cast<double>(x) *
                          static_cast<double>(x);
            }
            const double ffn_inv_rms =
                1.0 / std::sqrt(
                    ffn_ss / static_cast<double>(kCols) +
                    static_cast<double>(rms_eps));
            for (std::uint32_t i = 0; i < kCols; ++i) {
                ffn_normed_ref[i] =
                    static_cast<float>(
                        static_cast<double>(residual_gpu[i]) *
                        ffn_inv_rms *
                        static_cast<double>(post_norm_weight[i]));
            }

            std::vector<float> ffn_mul_ref(kFfnDim);
            for (std::uint32_t i = 0; i < kFfnDim; ++i) {
                const double g =
                    static_cast<double>(ffn_gate_gpu[i]);
                const double silu_g =
                    g / (1.0 + std::exp(-g));
                ffn_mul_ref[i] =
                    static_cast<float>(
                        silu_g *
                        static_cast<double>(ffn_up_gpu[i]));
            }

            constexpr std::size_t kFfnCheckedRows = 8;
            std::array<float, kQ4KValuesPerBlock> ffn_down_deq{};
            std::array<std::byte, kQ5KSm86BytesPerBlock> ffn_down_block{};
            std::array<double, kFfnCheckedRows> ffn_down_ref{};
            std::array<double, kFfnCheckedRows> layer_ref{};

            for (std::size_t row = 0; row < kFfnCheckedRows; ++row) {
                double dot = 0.0;
                for (std::size_t ib = 0;
                     ib < ffn_down_blocks_per_row; ++ib) {
                    const std::size_t block_index =
                        row * ffn_down_blocks_per_row + ib;
                    std::memcpy(
                        ffn_down_block.data() + 0,
                        ffn_down_matrix + block_index * 20,
                        20);
                    std::memcpy(
                        ffn_down_block.data() + 20,
                        ffn_down_matrix +
                            ffn_down_qh_offset + block_index * 32,
                        32);
                    std::memcpy(
                        ffn_down_block.data() + 52,
                        ffn_down_matrix +
                            ffn_down_qs_offset + block_index * 128,
                        128);
                    dequantize_q5_k_sm86_block_cpu(
                        ffn_down_block.data(), ffn_down_deq);
                    const std::size_t base_x =
                        ib * kQ4KValuesPerBlock;
                    for (std::size_t j = 0;
                         j < kQ4KValuesPerBlock; ++j) {
                        dot +=
                            static_cast<double>(ffn_down_deq[j]) *
                            static_cast<double>(
                                ffn_mul_ref[base_x + j]);
                    }
                }
                ffn_down_ref[row] = dot;
                layer_ref[row] =
                    dot + static_cast<double>(residual_gpu[row]);
            }

            Layer0RecurrentAttentionStats attention_local{};

            for (std::size_t i = 0; i < gated_norm_gpu.size(); ++i) {
                attention_local.gated_norm_max_abs =
                    std::max(
                        attention_local.gated_norm_max_abs,
                        std::abs(
                            static_cast<double>(gated_norm_gpu[i]) -
                            static_cast<double>(gated_norm_ref[i])));
            }

            for (std::size_t row = 0;
                 row < kTailCheckedRows; ++row) {
                attention_local.ssm_out_max_abs =
                    std::max(
                        attention_local.ssm_out_max_abs,
                        std::abs(
                            static_cast<double>(ssm_out_gpu[row]) -
                            ssm_out_ref[row]));
                attention_local.residual_max_abs =
                    std::max(
                        attention_local.residual_max_abs,
                        std::abs(
                            static_cast<double>(residual_gpu[row]) -
                            residual_ref[row]));
            }

            if (attention_local.gated_norm_max_abs > 2.0e-4 ||
                attention_local.ssm_out_max_abs > 3.0e-3 ||
                attention_local.residual_max_abs > 3.0e-3) {
                std::ostringstream oss;
                oss << "layer0 recurrent attention tail mismatch:"
                    << " gated_norm="
                    << attention_local.gated_norm_max_abs
                    << " ssm_out="
                    << attention_local.ssm_out_max_abs
                    << " residual="
                    << attention_local.residual_max_abs;
                throw std::runtime_error(oss.str());
            }

            Layer0FullStats full_local{};

            for (std::size_t i = 0; i < ffn_normed_gpu.size(); ++i) {
                full_local.ffn_norm_max_abs =
                    std::max(
                        full_local.ffn_norm_max_abs,
                        std::abs(
                            static_cast<double>(ffn_normed_gpu[i]) -
                            static_cast<double>(ffn_normed_ref[i])));
            }

            for (std::size_t i = 0; i < ffn_mul_gpu.size(); ++i) {
                full_local.ffn_gate_up_max_abs =
                    std::max(
                        full_local.ffn_gate_up_max_abs,
                        std::abs(
                            static_cast<double>(ffn_mul_gpu[i]) -
                            static_cast<double>(ffn_mul_ref[i])));
            }

            for (std::size_t row = 0;
                 row < kFfnCheckedRows; ++row) {
                full_local.ffn_down_max_abs =
                    std::max(
                        full_local.ffn_down_max_abs,
                        std::abs(
                            static_cast<double>(ffn_down_gpu[row]) -
                            ffn_down_ref[row]));
                full_local.layer_output_max_abs =
                    std::max(
                        full_local.layer_output_max_abs,
                        std::abs(
                            static_cast<double>(layer_out_gpu[row]) -
                            layer_ref[row]));
            }

            if (full_local.ffn_norm_max_abs > 3.0e-4 ||
                full_local.ffn_gate_up_max_abs > 3.0e-4 ||
                full_local.ffn_down_max_abs > 4.0e-3 ||
                full_local.layer_output_max_abs > 4.0e-3) {
                std::ostringstream oss;
                oss << "layer0 FFN mismatch:"
                    << " norm=" << full_local.ffn_norm_max_abs
                    << " gate_up=" << full_local.ffn_gate_up_max_abs
                    << " down=" << full_local.ffn_down_max_abs
                    << " output=" << full_local.layer_output_max_abs;
                throw std::runtime_error(oss.str());
            }

            auto bench = [&](int iters, auto&& fn, const char* label) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < iters; ++i) fn();
                check(handle_, sync(), label);
                const auto t1 = std::chrono::steady_clock::now();
                return std::chrono::duration<double, std::milli>(
                    t1 - t0).count() /
                    static_cast<double>(iters);
            };

            // Keep attention timing comparable to the standalone target.
            front_local.projection_ms =
                bench(50, launch_projection,
                      "cuCtxSynchronize(front projection benchmark)");
            front_local.conv_prep_ms =
                bench(200, launch_prep,
                      "cuCtxSynchronize(front prep benchmark)");
            front_local.gdn_ms =
                bench(200, launch_gdn,
                      "cuCtxSynchronize(front gdn benchmark)");
            front_local.sum_stage_ms =
                front_local.projection_ms +
                front_local.conv_prep_ms +
                front_local.gdn_ms;
            front_local.chain_ms =
                bench(50, launch_front,
                      "cuCtxSynchronize(front chain benchmark)");

            attention_local.front_ms = front_local.chain_ms;
            attention_local.gated_norm_ms =
                bench(200, launch_gated_norm,
                      "cuCtxSynchronize(attention gated norm benchmark)");
            attention_local.ssm_out_ms =
                bench(50, launch_ssm_out,
                      "cuCtxSynchronize(attention ssm_out benchmark)");
            attention_local.tail_ms =
                bench(50, launch_tail,
                      "cuCtxSynchronize(attention tail benchmark)");
            attention_local.sum_stage_ms =
                attention_local.front_ms +
                attention_local.tail_ms;
            attention_local.chain_ms =
                bench(50, launch_attention,
                      "cuCtxSynchronize(attention full chain benchmark)");

            full_local.attention_ms = attention_local.chain_ms;
            full_local.post_norm_ms =
                bench(100, launch_post_norm,
                      "cuCtxSynchronize(ffn post norm benchmark)");
            full_local.ffn_gate_ms =
                bench(50, launch_ffn_gate,
                      "cuCtxSynchronize(ffn gate benchmark)");
            full_local.ffn_up_ms =
                bench(50, launch_ffn_up,
                      "cuCtxSynchronize(ffn up benchmark)");
            full_local.ffn_pointwise_ms =
                bench(200, launch_ffn_pointwise,
                      "cuCtxSynchronize(ffn pointwise benchmark)");
            full_local.ffn_down_ms =
                bench(50, launch_ffn_down,
                      "cuCtxSynchronize(ffn down benchmark)");
            full_local.ffn_ms =
                bench(50, launch_ffn,
                      "cuCtxSynchronize(ffn full benchmark)");
            full_local.sum_stage_ms =
                full_local.attention_ms +
                full_local.ffn_ms;
            full_local.chain_ms =
                bench(50, launch_chain,
                      "cuCtxSynchronize(full layer benchmark)");

            if (stats) *stats = full_local;
        } catch (...) {
            if (iq4_module) module_unload(iq4_module);
            if (tail_module) module_unload(tail_module);
            if (gdn_module) module_unload(gdn_module);
            if (prep_module) module_unload(prep_module);
            if (q4_module) module_unload(q4_module);
            if (q5_module) module_unload(q5_module);
            module_unload(norm_module);
            throw;
        }

        check(handle_, module_unload(iq4_module),
              "cuModuleUnload(ffn IQ4_XS)");
        check(handle_, module_unload(tail_module),
              "cuModuleUnload(attention tail)");
        check(handle_, module_unload(gdn_module),
              "cuModuleUnload(front GDN)");
        check(handle_, module_unload(prep_module),
              "cuModuleUnload(front prep)");
        check(handle_, module_unload(q4_module),
              "cuModuleUnload(front Q4)");
        check(handle_, module_unload(q5_module),
              "cuModuleUnload(front Q5)");
        check(handle_, module_unload(norm_module),
              "cuModuleUnload(front RMSNorm)");

        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool NvidiaDriver::run_qwen35_layer0_recurrent_attention(
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
    Layer0RecurrentAttentionStats* stats,
    std::string* error) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error(
                "q38 layer0 recurrent front requires sm_86; detected sm_" +
                std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (!norm_weight || !qkv_matrix || !z_matrix ||
            !beta_matrix || !alpha_matrix ||
            !conv_weight || !dt_bias || !ssm_a ||
            !ssm_norm_weight || !ssm_out_matrix) {
            throw std::invalid_argument(
                "layer0 recurrent attention received null model tensor");
        }

        constexpr std::uint32_t kCols = 5120;
        constexpr std::uint32_t kQkvRows = 10240;
        constexpr std::uint32_t kZRows = 6144;
        constexpr std::uint32_t kSmallRows = 48;
        constexpr std::uint32_t kHeadDim = 128;
        constexpr std::uint32_t kQkHeads = 16;
        constexpr std::uint32_t kValueHeads = 48;
        constexpr std::uint32_t kKeyDim = kQkHeads * kHeadDim;
        constexpr std::uint32_t kValueDim = kValueHeads * kHeadDim;
        constexpr std::uint32_t kConvStateSteps = 3;
        constexpr std::size_t kGdnStateValues =
            static_cast<std::size_t>(kValueHeads) * kHeadDim * kHeadDim;

        const std::size_t q5_blocks_per_row = kCols / kQ4KValuesPerBlock;

        const std::size_t qkv_blocks =
            static_cast<std::size_t>(kQkvRows) * q5_blocks_per_row;
        const std::size_t qkv_meta_bytes = qkv_blocks * 20;
        const std::size_t qkv_qh_bytes = qkv_blocks * 32;
        const std::size_t qkv_qs_bytes = qkv_blocks * 128;
        if (qkv_qh_offset < qkv_meta_bytes ||
            qkv_qs_offset < qkv_qh_offset + qkv_qh_bytes) {
            throw std::invalid_argument("invalid qkv Q5_K plane offsets");
        }
        const std::size_t qkv_bytes = qkv_qs_offset + qkv_qs_bytes;

        const std::size_t z_blocks =
            static_cast<std::size_t>(kZRows) * q5_blocks_per_row;
        const std::size_t z_meta_bytes = z_blocks * 20;
        const std::size_t z_qh_bytes = z_blocks * 32;
        const std::size_t z_qs_bytes = z_blocks * 128;
        if (z_qh_offset < z_meta_bytes ||
            z_qs_offset < z_qh_offset + z_qh_bytes) {
            throw std::invalid_argument("invalid z Q5_K plane offsets");
        }
        const std::size_t z_bytes = z_qs_offset + z_qs_bytes;

        const std::size_t ssm_out_blocks_per_row =
            kValueDim / kQ4KValuesPerBlock;
        const std::size_t ssm_out_blocks =
            static_cast<std::size_t>(kCols) * ssm_out_blocks_per_row;
        const std::size_t ssm_out_meta_bytes = ssm_out_blocks * 20;
        const std::size_t ssm_out_qh_bytes = ssm_out_blocks * 32;
        const std::size_t ssm_out_qs_bytes = ssm_out_blocks * 128;
        if (ssm_out_qh_offset < ssm_out_meta_bytes ||
            ssm_out_qs_offset < ssm_out_qh_offset + ssm_out_qh_bytes) {
            throw std::invalid_argument("invalid ssm_out Q5_K plane offsets");
        }
        const std::size_t ssm_out_bytes =
            ssm_out_qs_offset + ssm_out_qs_bytes;
        const std::size_t ssm_norm_bytes =
            static_cast<std::size_t>(kHeadDim) * sizeof(float);

        const std::size_t q4_blocks_per_row =
            kCols / kQ4KValuesPerBlock;
        const std::size_t small_matrix_bytes =
            static_cast<std::size_t>(kSmallRows) *
            q4_blocks_per_row * kQ4KBytesPerBlock;

        const std::size_t hidden_bytes =
            static_cast<std::size_t>(kCols) * sizeof(float);
        const std::size_t qkv_out_bytes =
            static_cast<std::size_t>(kQkvRows) * sizeof(float);
        const std::size_t z_out_bytes =
            static_cast<std::size_t>(kZRows) * sizeof(float);
        const std::size_t small_bytes =
            static_cast<std::size_t>(kSmallRows) * sizeof(float);
        const std::size_t conv_weight_bytes =
            static_cast<std::size_t>(4) * kQkvRows * sizeof(float);
        const std::size_t conv_state_bytes =
            static_cast<std::size_t>(kConvStateSteps) *
            kQkvRows * sizeof(float);
        const std::size_t qk_bytes =
            static_cast<std::size_t>(kKeyDim) * sizeof(float);
        const std::size_t gdn_out_bytes =
            static_cast<std::size_t>(kValueDim) * sizeof(float);
        const std::size_t gdn_state_bytes =
            kGdnStateValues * sizeof(float);

        auto align256 = [](std::size_t n) {
            return (n + 255u) & ~std::size_t(255u);
        };

        // Model tensors and persistent test-state inputs.
        const std::size_t qkv_w_off = 0;
        const std::size_t z_w_off = align256(qkv_w_off + qkv_bytes);
        const std::size_t beta_w_off = align256(z_w_off + z_bytes);
        const std::size_t alpha_w_off =
            align256(beta_w_off + small_matrix_bytes);
        const std::size_t norm_w_off =
            align256(alpha_w_off + small_matrix_bytes);
        const std::size_t conv_w_off =
            align256(norm_w_off + hidden_bytes);
        const std::size_t dt_off =
            align256(conv_w_off + conv_weight_bytes);
        const std::size_t a_off =
            align256(dt_off + small_bytes);
        const std::size_t ssm_norm_w_off =
            align256(a_off + small_bytes);
        const std::size_t ssm_out_w_off =
            align256(ssm_norm_w_off + ssm_norm_bytes);
        const std::size_t hidden_off =
            align256(ssm_out_w_off + ssm_out_bytes);
        const std::size_t conv_state_in_off =
            align256(hidden_off + hidden_bytes);
        const std::size_t gdn_state_in_off =
            align256(conv_state_in_off + conv_state_bytes);

        // Runtime workspace.
        const std::size_t sumsq_off =
            align256(gdn_state_in_off + gdn_state_bytes);
        const std::size_t normed_off =
            align256(sumsq_off + sizeof(float));
        const std::size_t qkv_out_off =
            align256(normed_off + hidden_bytes);
        const std::size_t z_out_off =
            align256(qkv_out_off + qkv_out_bytes);
        const std::size_t beta_raw_off =
            align256(z_out_off + z_out_bytes);
        const std::size_t alpha_raw_off =
            align256(beta_raw_off + small_bytes);
        const std::size_t conv_out_off =
            align256(alpha_raw_off + small_bytes);
        const std::size_t conv_state_out_off =
            align256(conv_out_off + qkv_out_bytes);
        const std::size_t q_out_off =
            align256(conv_state_out_off + conv_state_bytes);
        const std::size_t k_out_off =
            align256(q_out_off + qk_bytes);
        const std::size_t beta_out_off =
            align256(k_out_off + qk_bytes);
        const std::size_t gate_out_off =
            align256(beta_out_off + small_bytes);
        const std::size_t gdn_out_off =
            align256(gate_out_off + small_bytes);
        const std::size_t gdn_state_out_off =
            align256(gdn_out_off + gdn_out_bytes);
        const std::size_t gated_norm_out_off =
            align256(gdn_state_out_off + gdn_state_bytes);
        const std::size_t ssm_out_out_off =
            align256(gated_norm_out_off + gdn_out_bytes);
        const std::size_t residual_out_off =
            align256(ssm_out_out_off + hidden_bytes);
        const std::size_t total_bytes =
            residual_out_off + hidden_bytes;

        ScopedVmm memory(handle_, device_ordinal_, total_bytes);

        using MemcpyHtoD =
            CUresult(*)(CUdeviceptr, const void*, std::size_t);
        using MemcpyDtoH =
            CUresult(*)(void*, CUdeviceptr, std::size_t);
        using MemsetD32 =
            CUresult(*)(CUdeviceptr, unsigned int, std::size_t);
        using ModuleLoadDataEx =
            CUresult(*)(CUmodule*, const void*, unsigned int, int*, void**);
        using ModuleUnload = CUresult(*)(CUmodule);
        using ModuleGetFunction =
            CUresult(*)(CUfunction*, CUmodule, const char*);
        using LaunchKernel =
            CUresult(*)(CUfunction,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, CUstream, void**, void**);
        using CtxSynchronize = CUresult(*)();

        const auto memcpy_htod =
            sym<MemcpyHtoD>(handle_, "cuMemcpyHtoD_v2");
        const auto memcpy_dtoh =
            sym<MemcpyDtoH>(handle_, "cuMemcpyDtoH_v2");
        const auto memset_d32 =
            sym<MemsetD32>(handle_, "cuMemsetD32_v2");
        const auto module_load_ex =
            sym<ModuleLoadDataEx>(handle_, "cuModuleLoadDataEx");
        const auto module_unload =
            sym<ModuleUnload>(handle_, "cuModuleUnload");
        const auto module_get_function =
            sym<ModuleGetFunction>(handle_, "cuModuleGetFunction");
        const auto launch =
            sym<LaunchKernel>(handle_, "cuLaunchKernel");
        const auto sync =
            sym<CtxSynchronize>(handle_, "cuCtxSynchronize");

        // Deterministic decode-time input and recurrent states.
        std::vector<float> hidden(kCols);
        std::vector<float> conv_state_in(
            static_cast<std::size_t>(kConvStateSteps) * kQkvRows);
        std::vector<float> gdn_state_in(kGdnStateValues);

        for (std::uint32_t i = 0; i < kCols; ++i) {
            const float fi = static_cast<float>(i);
            hidden[i] =
                std::sin(fi * 0.017f) * 0.65f +
                std::cos(fi * 0.011f) * 0.35f;
        }
        for (std::uint32_t s = 0; s < kConvStateSteps; ++s) {
            for (std::uint32_t i = 0; i < kQkvRows; ++i) {
                const float fi = static_cast<float>(i);
                const float fs = static_cast<float>(s + 1);
                conv_state_in[
                    static_cast<std::size_t>(s) * kQkvRows + i] =
                    0.08f *
                        std::sin(fi * (0.0029f + 0.0006f * fs)) +
                    0.03f *
                        std::cos(fi * (0.0017f + 0.0004f * fs));
            }
        }
        for (std::size_t i = 0; i < gdn_state_in.size(); ++i) {
            const float fi = static_cast<float>(i);
            gdn_state_in[i] =
                0.011f * std::sin(fi * 0.0011f) +
                0.005f * std::cos(fi * 0.00073f);
        }

        const CUdeviceptr base = memory.ptr();
        const CUdeviceptr qkv_w_ptr = base + qkv_w_off;
        const CUdeviceptr z_w_ptr = base + z_w_off;
        const CUdeviceptr beta_w_ptr = base + beta_w_off;
        const CUdeviceptr alpha_w_ptr = base + alpha_w_off;
        const CUdeviceptr norm_w_ptr = base + norm_w_off;
        const CUdeviceptr conv_w_ptr = base + conv_w_off;
        const CUdeviceptr dt_ptr = base + dt_off;
        const CUdeviceptr a_ptr = base + a_off;
        const CUdeviceptr ssm_norm_w_ptr = base + ssm_norm_w_off;
        const CUdeviceptr ssm_out_w_ptr = base + ssm_out_w_off;
        const CUdeviceptr hidden_ptr = base + hidden_off;
        const CUdeviceptr conv_state_in_ptr = base + conv_state_in_off;
        const CUdeviceptr gdn_state_in_ptr = base + gdn_state_in_off;

        const CUdeviceptr sumsq_ptr = base + sumsq_off;
        const CUdeviceptr normed_ptr = base + normed_off;
        const CUdeviceptr qkv_out_ptr = base + qkv_out_off;
        const CUdeviceptr z_out_ptr = base + z_out_off;
        const CUdeviceptr beta_raw_ptr = base + beta_raw_off;
        const CUdeviceptr alpha_raw_ptr = base + alpha_raw_off;
        const CUdeviceptr conv_out_ptr = base + conv_out_off;
        const CUdeviceptr conv_state_out_ptr = base + conv_state_out_off;
        const CUdeviceptr q_out_ptr = base + q_out_off;
        const CUdeviceptr k_out_ptr = base + k_out_off;
        const CUdeviceptr beta_out_ptr = base + beta_out_off;
        const CUdeviceptr gate_out_ptr = base + gate_out_off;
        const CUdeviceptr gdn_out_ptr = base + gdn_out_off;
        const CUdeviceptr gdn_state_out_ptr = base + gdn_state_out_off;
        const CUdeviceptr gated_norm_out_ptr = base + gated_norm_out_off;
        const CUdeviceptr ssm_out_out_ptr = base + ssm_out_out_off;
        const CUdeviceptr residual_out_ptr = base + residual_out_off;

        check(handle_, memcpy_htod(
            qkv_w_ptr, qkv_matrix, qkv_bytes),
            "cuMemcpyHtoD(front qkv weight)");
        check(handle_, memcpy_htod(
            z_w_ptr, z_matrix, z_bytes),
            "cuMemcpyHtoD(front z weight)");
        check(handle_, memcpy_htod(
            beta_w_ptr, beta_matrix, small_matrix_bytes),
            "cuMemcpyHtoD(front beta weight)");
        check(handle_, memcpy_htod(
            alpha_w_ptr, alpha_matrix, small_matrix_bytes),
            "cuMemcpyHtoD(front alpha weight)");
        check(handle_, memcpy_htod(
            norm_w_ptr, norm_weight, hidden_bytes),
            "cuMemcpyHtoD(front norm weight)");
        check(handle_, memcpy_htod(
            conv_w_ptr, conv_weight, conv_weight_bytes),
            "cuMemcpyHtoD(front conv weight)");
        check(handle_, memcpy_htod(
            dt_ptr, dt_bias, small_bytes),
            "cuMemcpyHtoD(front dt bias)");
        check(handle_, memcpy_htod(
            a_ptr, ssm_a, small_bytes),
            "cuMemcpyHtoD(front ssm a)");
        check(handle_, memcpy_htod(
            ssm_norm_w_ptr, ssm_norm_weight, ssm_norm_bytes),
            "cuMemcpyHtoD(attention ssm norm)");
        check(handle_, memcpy_htod(
            ssm_out_w_ptr, ssm_out_matrix, ssm_out_bytes),
            "cuMemcpyHtoD(attention ssm out weight)");
        check(handle_, memcpy_htod(
            hidden_ptr, hidden.data(), hidden_bytes),
            "cuMemcpyHtoD(front hidden)");
        check(handle_, memcpy_htod(
            conv_state_in_ptr, conv_state_in.data(), conv_state_bytes),
            "cuMemcpyHtoD(front conv state)");
        check(handle_, memcpy_htod(
            gdn_state_in_ptr, gdn_state_in.data(), gdn_state_bytes),
            "cuMemcpyHtoD(front gdn state)");

        constexpr int CU_JIT_INFO_LOG_BUFFER = 3;
        constexpr int CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4;
        constexpr int CU_JIT_ERROR_LOG_BUFFER = 5;
        constexpr int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;
        constexpr int CU_JIT_LOG_VERBOSE = 12;

        auto load_module = [&](const char* ptx, const char* label) {
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
                reinterpret_cast<void*>(
                    static_cast<std::uintptr_t>(jit_info.size())),
                jit_error.data(),
                reinterpret_cast<void*>(
                    static_cast<std::uintptr_t>(jit_error.size())),
                reinterpret_cast<void*>(
                    static_cast<std::uintptr_t>(1)),
            };
            CUmodule module{};
            const auto rc = module_load_ex(
                &module,
                ptx,
                static_cast<unsigned int>(
                    sizeof(jit_options) / sizeof(jit_options[0])),
                jit_options,
                jit_values);
            if (rc != CUDA_SUCCESS) {
                std::string detail = cuda_error(handle_, rc, label);
                if (jit_error[0] != '\0') {
                    detail +=
                        std::string("\nPTX JIT error log:\n") +
                        jit_error.data();
                }
                if (jit_info[0] != '\0') {
                    detail +=
                        std::string("\nPTX JIT info log:\n") +
                        jit_info.data();
                }
                throw std::runtime_error(detail);
            }
            return module;
        };

        CUmodule norm_module =
            load_module(kRmsNormPtx, "cuModuleLoadDataEx(front RMSNorm)");
        CUmodule q5_module{};
        CUmodule q4_module{};
        CUmodule prep_module{};
        CUmodule gdn_module{};
        CUmodule tail_module{};

        try {
            q5_module = load_module(
                kQ5KSm86SoAGemvVecPtx,
                "cuModuleLoadDataEx(front Q5)");
            q4_module = load_module(
                kQ4KGemvPtx,
                "cuModuleLoadDataEx(front Q4)");
            prep_module = load_module(
                kRecurrentPrepPtx,
                "cuModuleLoadDataEx(front prep)");
            gdn_module = load_module(
                kGdnAr128Ptx,
                "cuModuleLoadDataEx(front GDN)");
            tail_module = load_module(
                kRecurrentTailPtx,
                "cuModuleLoadDataEx(attention tail)");

            CUfunction sum_fn{}, norm_fn{}, q5_fn{}, q4_fn{};
            CUfunction conv_fn{}, qk_fn{}, bg_fn{}, gdn_fn{};
            CUfunction gated_norm_fn{}, residual_fn{};

            check(handle_, module_get_function(
                &sum_fn, norm_module, "q38_sumsq"),
                "cuModuleGetFunction(front sumsq)");
            check(handle_, module_get_function(
                &norm_fn, norm_module, "q38_rmsnorm_apply"),
                "cuModuleGetFunction(front rmsnorm)");
            check(handle_, module_get_function(
                &q5_fn, q5_module, "q38_q5k_sm86_soa_gemv_vec"),
                "cuModuleGetFunction(front q5)");
            check(handle_, module_get_function(
                &q4_fn, q4_module, "q38_q4k_gemv_f32"),
                "cuModuleGetFunction(front q4)");
            check(handle_, module_get_function(
                &conv_fn, prep_module, "q38_conv4_silu_roll"),
                "cuModuleGetFunction(front conv)");
            check(handle_, module_get_function(
                &qk_fn, prep_module, "q38_qk_l2norm_128"),
                "cuModuleGetFunction(front qk norm)");
            check(handle_, module_get_function(
                &bg_fn, prep_module, "q38_beta_gate_48"),
                "cuModuleGetFunction(front beta gate)");
            check(handle_, module_get_function(
                &gdn_fn, gdn_module, "q38_gdn_ar_128"),
                "cuModuleGetFunction(front gdn)");
            check(handle_, module_get_function(
                &gated_norm_fn, tail_module, "q38_gated_rmsnorm_silu_128"),
                "cuModuleGetFunction(attention gated norm)");
            check(handle_, module_get_function(
                &residual_fn, tail_module, "q38_add_residual_f32"),
                "cuModuleGetFunction(attention residual)");

            // RMSNorm args.
            constexpr unsigned int norm_block = 256;
            const unsigned int norm_grid =
                (kCols + norm_block - 1) / norm_block;

            CUdeviceptr sum_x = hidden_ptr;
            CUdeviceptr sum_out = sumsq_ptr;
            std::uint32_t norm_count = kCols;
            void* sum_params[] = {
                &sum_x, &sum_out, &norm_count
            };

            CUdeviceptr norm_x = hidden_ptr;
            CUdeviceptr norm_w = norm_w_ptr;
            CUdeviceptr norm_y = normed_ptr;
            CUdeviceptr norm_sumsq = sumsq_ptr;
            float norm_eps = rms_eps;
            void* norm_params[] = {
                &norm_x, &norm_w, &norm_y,
                &norm_sumsq, &norm_count, &norm_eps
            };

            // Q5 qkv args.
            CUdeviceptr qkv_meta = qkv_w_ptr;
            CUdeviceptr qkv_qh = qkv_w_ptr + qkv_qh_offset;
            CUdeviceptr qkv_qs = qkv_w_ptr + qkv_qs_offset;
            CUdeviceptr qkv_x = normed_ptr;
            CUdeviceptr qkv_y = qkv_out_ptr;
            std::uint32_t qkv_cols = kCols;
            std::uint32_t qkv_rows = kQkvRows;
            void* qkv_params[] = {
                &qkv_meta, &qkv_qh, &qkv_qs, &qkv_x, &qkv_y,
                &qkv_cols, &qkv_rows
            };

            // Q5 z args.
            CUdeviceptr z_meta = z_w_ptr;
            CUdeviceptr z_qh = z_w_ptr + z_qh_offset;
            CUdeviceptr z_qs = z_w_ptr + z_qs_offset;
            CUdeviceptr z_x = normed_ptr;
            CUdeviceptr z_y = z_out_ptr;
            std::uint32_t z_cols = kCols;
            std::uint32_t z_rows = kZRows;
            void* z_params[] = {
                &z_meta, &z_qh, &z_qs, &z_x, &z_y,
                &z_cols, &z_rows
            };

            // Q4 beta/alpha args.
            CUdeviceptr beta_w = beta_w_ptr;
            CUdeviceptr beta_x = normed_ptr;
            CUdeviceptr beta_y = beta_raw_ptr;
            std::uint32_t beta_cols = kCols;
            std::uint32_t beta_rows = kSmallRows;
            void* beta_params[] = {
                &beta_w, &beta_x, &beta_y,
                &beta_cols, &beta_rows
            };

            CUdeviceptr alpha_w = alpha_w_ptr;
            CUdeviceptr alpha_x = normed_ptr;
            CUdeviceptr alpha_y = alpha_raw_ptr;
            std::uint32_t alpha_cols = kCols;
            std::uint32_t alpha_rows = kSmallRows;
            void* alpha_params[] = {
                &alpha_w, &alpha_x, &alpha_y,
                &alpha_cols, &alpha_rows
            };

            // Conv prep args.
            CUdeviceptr c_qkv = qkv_out_ptr;
            CUdeviceptr c_w = conv_w_ptr;
            CUdeviceptr c_si = conv_state_in_ptr;
            CUdeviceptr c_co = conv_out_ptr;
            CUdeviceptr c_so = conv_state_out_ptr;
            std::uint32_t c_channels = kQkvRows;
            float log2e = 1.4426950408889634f;
            void* conv_params[] = {
                &c_qkv, &c_w, &c_si, &c_co, &c_so,
                &c_channels, &log2e
            };

            CUdeviceptr n_conv = conv_out_ptr;
            CUdeviceptr n_q = q_out_ptr;
            CUdeviceptr n_k = k_out_ptr;
            float qk_eps = rms_eps;
            void* qk_params[] = {
                &n_conv, &n_q, &n_k, &qk_eps
            };

            CUdeviceptr b_br = beta_raw_ptr;
            CUdeviceptr b_ar = alpha_raw_ptr;
            CUdeviceptr b_dt = dt_ptr;
            CUdeviceptr b_a = a_ptr;
            CUdeviceptr b_bo = beta_out_ptr;
            CUdeviceptr b_go = gate_out_ptr;
            float inv_log2e = 0.6931471805599453f;
            void* bg_params[] = {
                &b_br, &b_ar, &b_dt, &b_a, &b_bo, &b_go,
                &log2e, &inv_log2e
            };

            // GDN args. v is the last 6144 values of conv_out.
            CUdeviceptr g_q = q_out_ptr;
            CUdeviceptr g_k = k_out_ptr;
            CUdeviceptr g_v =
                conv_out_ptr +
                static_cast<CUdeviceptr>(2 * kKeyDim * sizeof(float));
            CUdeviceptr g_gate = gate_out_ptr;
            CUdeviceptr g_beta = beta_out_ptr;
            CUdeviceptr g_state_in = gdn_state_in_ptr;
            CUdeviceptr g_out = gdn_out_ptr;
            CUdeviceptr g_state_out = gdn_state_out_ptr;
            std::uint32_t g_qk_heads = kQkHeads;
            std::uint32_t g_value_heads = kValueHeads;
            float g_scale =
                1.0f / std::sqrt(static_cast<float>(kHeadDim));
            void* gdn_params[] = {
                &g_q, &g_k, &g_v, &g_gate, &g_beta,
                &g_state_in, &g_out, &g_state_out,
                &g_qk_heads, &g_value_heads,
                &g_scale, &log2e
            };

            CUdeviceptr gn_input = gdn_out_ptr;
            CUdeviceptr gn_weight = ssm_norm_w_ptr;
            CUdeviceptr gn_z = z_out_ptr;
            CUdeviceptr gn_out = gated_norm_out_ptr;
            float gn_eps = rms_eps;
            void* gated_norm_params[] = {
                &gn_input, &gn_weight, &gn_z, &gn_out,
                &gn_eps, &log2e
            };

            CUdeviceptr so_meta = ssm_out_w_ptr;
            CUdeviceptr so_qh = ssm_out_w_ptr + ssm_out_qh_offset;
            CUdeviceptr so_qs = ssm_out_w_ptr + ssm_out_qs_offset;
            CUdeviceptr so_x = gated_norm_out_ptr;
            CUdeviceptr so_y = ssm_out_out_ptr;
            std::uint32_t so_cols = kValueDim;
            std::uint32_t so_rows = kCols;
            void* ssm_out_params[] = {
                &so_meta, &so_qh, &so_qs, &so_x, &so_y,
                &so_cols, &so_rows
            };

            CUdeviceptr r_x = ssm_out_out_ptr;
            CUdeviceptr r_residual = hidden_ptr;
            CUdeviceptr r_out = residual_out_ptr;
            std::uint32_t r_n = kCols;
            void* residual_params[] = {
                &r_x, &r_residual, &r_out, &r_n
            };

            const unsigned int qkv_grid =
                (kQkvRows + 3u) / 4u;
            const unsigned int z_grid =
                (kZRows + 3u) / 4u;
            const unsigned int conv_grid =
                (kQkvRows + 255u) / 256u;
            const unsigned int gdn_grid_y =
                (kHeadDim + 3u) / 4u;
            const unsigned int ssm_out_grid =
                (kCols + 3u) / 4u;
            const unsigned int residual_grid =
                (kCols + 255u) / 256u;

            auto launch_projection = [&]() {
                check(handle_, memset_d32(
                    sumsq_ptr, 0, 1),
                    "cuMemsetD32(front sumsq)");
                check(handle_, launch(
                    sum_fn, norm_grid, 1, 1,
                    norm_block, 1, 1,
                    0, nullptr, sum_params, nullptr),
                    "cuLaunchKernel(front sumsq)");
                check(handle_, launch(
                    norm_fn, norm_grid, 1, 1,
                    norm_block, 1, 1,
                    0, nullptr, norm_params, nullptr),
                    "cuLaunchKernel(front rmsnorm)");
                check(handle_, launch(
                    q5_fn, qkv_grid, 1, 1,
                    128, 1, 1,
                    0, nullptr, qkv_params, nullptr),
                    "cuLaunchKernel(front qkv)");
                check(handle_, launch(
                    q5_fn, z_grid, 1, 1,
                    128, 1, 1,
                    0, nullptr, z_params, nullptr),
                    "cuLaunchKernel(front z)");
                check(handle_, launch(
                    q4_fn, kSmallRows, 1, 1,
                    32, 1, 1,
                    0, nullptr, beta_params, nullptr),
                    "cuLaunchKernel(front beta)");
                check(handle_, launch(
                    q4_fn, kSmallRows, 1, 1,
                    32, 1, 1,
                    0, nullptr, alpha_params, nullptr),
                    "cuLaunchKernel(front alpha)");
            };

            auto launch_prep = [&]() {
                check(handle_, launch(
                    conv_fn, conv_grid, 1, 1,
                    256, 1, 1,
                    0, nullptr, conv_params, nullptr),
                    "cuLaunchKernel(front conv)");
                check(handle_, launch(
                    qk_fn, 32, 1, 1,
                    128, 1, 1,
                    0, nullptr, qk_params, nullptr),
                    "cuLaunchKernel(front qk norm)");
                check(handle_, launch(
                    bg_fn, 1, 1, 1,
                    64, 1, 1,
                    0, nullptr, bg_params, nullptr),
                    "cuLaunchKernel(front beta gate)");
            };

            auto launch_gdn = [&]() {
                check(handle_, launch(
                    gdn_fn,
                    kValueHeads, gdn_grid_y, 1,
                    128, 1, 1,
                    0, nullptr, gdn_params, nullptr),
                    "cuLaunchKernel(front gdn)");
            };

            auto launch_gated_norm = [&]() {
                check(handle_, launch(
                    gated_norm_fn,
                    kValueHeads, 1, 1,
                    128, 1, 1,
                    0, nullptr, gated_norm_params, nullptr),
                    "cuLaunchKernel(attention gated norm)");
            };

            auto launch_ssm_out = [&]() {
                check(handle_, launch(
                    q5_fn,
                    ssm_out_grid, 1, 1,
                    128, 1, 1,
                    0, nullptr, ssm_out_params, nullptr),
                    "cuLaunchKernel(attention ssm_out)");
            };

            auto launch_residual = [&]() {
                check(handle_, launch(
                    residual_fn,
                    residual_grid, 1, 1,
                    256, 1, 1,
                    0, nullptr, residual_params, nullptr),
                    "cuLaunchKernel(attention residual)");
            };

            auto launch_front = [&]() {
                launch_projection();
                launch_prep();
                launch_gdn();
            };

            auto launch_tail = [&]() {
                launch_gated_norm();
                launch_ssm_out();
                launch_residual();
            };

            auto launch_chain = [&]() {
                launch_front();
                launch_tail();
            };

            // One full chain for correctness.
            launch_chain();
            check(handle_, sync(),
                  "cuCtxSynchronize(layer0 recurrent front correctness)");

            // Read GPU projection products used as the CPU downstream oracle.
            // The projection kernels are already independently validated by
            // q38-layer0-projections; this check focuses on composition.
            std::vector<float> qkv_gpu(kQkvRows);
            std::vector<float> beta_raw_gpu(kSmallRows);
            std::vector<float> alpha_raw_gpu(kSmallRows);
            std::vector<float> conv_gpu(kQkvRows);
            std::vector<float> conv_state_gpu(
                static_cast<std::size_t>(kConvStateSteps) * kQkvRows);
            std::vector<float> q_gpu(kKeyDim);
            std::vector<float> k_gpu(kKeyDim);
            std::vector<float> beta_gpu(kSmallRows);
            std::vector<float> gate_gpu(kSmallRows);
            std::vector<float> gdn_out_gpu(kValueDim);
            std::vector<float> gdn_state_gpu(kGdnStateValues);
            std::vector<float> z_gpu(kZRows);
            std::vector<float> gated_norm_gpu(kValueDim);
            std::vector<float> ssm_out_gpu(kCols);
            std::vector<float> residual_gpu(kCols);

            check(handle_, memcpy_dtoh(
                qkv_gpu.data(), qkv_out_ptr, qkv_out_bytes),
                "cuMemcpyDtoH(front qkv)");
            check(handle_, memcpy_dtoh(
                beta_raw_gpu.data(), beta_raw_ptr, small_bytes),
                "cuMemcpyDtoH(front beta raw)");
            check(handle_, memcpy_dtoh(
                alpha_raw_gpu.data(), alpha_raw_ptr, small_bytes),
                "cuMemcpyDtoH(front alpha raw)");
            check(handle_, memcpy_dtoh(
                conv_gpu.data(), conv_out_ptr, qkv_out_bytes),
                "cuMemcpyDtoH(front conv)");
            check(handle_, memcpy_dtoh(
                conv_state_gpu.data(), conv_state_out_ptr, conv_state_bytes),
                "cuMemcpyDtoH(front conv state)");
            check(handle_, memcpy_dtoh(
                q_gpu.data(), q_out_ptr, qk_bytes),
                "cuMemcpyDtoH(front q)");
            check(handle_, memcpy_dtoh(
                k_gpu.data(), k_out_ptr, qk_bytes),
                "cuMemcpyDtoH(front k)");
            check(handle_, memcpy_dtoh(
                beta_gpu.data(), beta_out_ptr, small_bytes),
                "cuMemcpyDtoH(front beta)");
            check(handle_, memcpy_dtoh(
                gate_gpu.data(), gate_out_ptr, small_bytes),
                "cuMemcpyDtoH(front gate)");
            check(handle_, memcpy_dtoh(
                gdn_out_gpu.data(), gdn_out_ptr, gdn_out_bytes),
                "cuMemcpyDtoH(front gdn output)");
            check(handle_, memcpy_dtoh(
                gdn_state_gpu.data(), gdn_state_out_ptr, gdn_state_bytes),
                "cuMemcpyDtoH(front gdn state)");
            check(handle_, memcpy_dtoh(
                z_gpu.data(), z_out_ptr, z_out_bytes),
                "cuMemcpyDtoH(attention z)");
            check(handle_, memcpy_dtoh(
                gated_norm_gpu.data(), gated_norm_out_ptr, gdn_out_bytes),
                "cuMemcpyDtoH(attention gated norm)");
            check(handle_, memcpy_dtoh(
                ssm_out_gpu.data(), ssm_out_out_ptr, hidden_bytes),
                "cuMemcpyDtoH(attention ssm out)");
            check(handle_, memcpy_dtoh(
                residual_gpu.data(), residual_out_ptr, hidden_bytes),
                "cuMemcpyDtoH(attention residual)");

            std::vector<float> conv_ref(kQkvRows);
            std::vector<float> conv_state_ref(
                static_cast<std::size_t>(kConvStateSteps) * kQkvRows);
            std::vector<float> q_ref(kKeyDim);
            std::vector<float> k_ref(kKeyDim);
            std::vector<float> beta_ref(kSmallRows);
            std::vector<float> gate_ref(kSmallRows);
            std::vector<float> gdn_out_ref(kValueDim);
            std::vector<float> gdn_state_ref(kGdnStateValues);

            for (std::uint32_t ch = 0; ch < kQkvRows; ++ch) {
                const float x0 = conv_state_in[ch];
                const float x1 = conv_state_in[kQkvRows + ch];
                const float x2 = conv_state_in[2 * kQkvRows + ch];
                const float x3 = qkv_gpu[ch];
                const float* w =
                    conv_weight + static_cast<std::size_t>(ch) * 4;
                const double raw =
                    static_cast<double>(x0) * w[0] +
                    static_cast<double>(x1) * w[1] +
                    static_cast<double>(x2) * w[2] +
                    static_cast<double>(x3) * w[3];
                conv_ref[ch] =
                    static_cast<float>(
                        raw / (1.0 + std::exp(-raw)));
                conv_state_ref[ch] = x1;
                conv_state_ref[kQkvRows + ch] = x2;
                conv_state_ref[2 * kQkvRows + ch] = x3;
            }

            for (std::uint32_t h = 0; h < kQkHeads; ++h) {
                const std::size_t q_base =
                    static_cast<std::size_t>(h) * kHeadDim;
                const std::size_t k_base =
                    static_cast<std::size_t>(kKeyDim) +
                    static_cast<std::size_t>(h) * kHeadDim;

                double q_ss = 0.0;
                double k_ss = 0.0;
                for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                    const double qv = conv_ref[q_base + i];
                    const double kv = conv_ref[k_base + i];
                    q_ss += qv * qv;
                    k_ss += kv * kv;
                }
                const double q_inv =
                    1.0 / std::sqrt(q_ss + static_cast<double>(rms_eps));
                const double k_inv =
                    1.0 / std::sqrt(k_ss + static_cast<double>(rms_eps));

                for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                    q_ref[q_base + i] =
                        static_cast<float>(
                            static_cast<double>(conv_ref[q_base + i]) *
                            q_inv);
                    k_ref[q_base + i] =
                        static_cast<float>(
                            static_cast<double>(conv_ref[k_base + i]) *
                            k_inv);
                }
            }

            auto softplus = [](double x) {
                if (x > 20.0) return x;
                if (x < -20.0) return std::exp(x);
                return std::log1p(std::exp(x));
            };
            for (std::uint32_t h = 0; h < kSmallRows; ++h) {
                const double br = beta_raw_gpu[h];
                beta_ref[h] =
                    static_cast<float>(
                        1.0 / (1.0 + std::exp(-br)));
                const double biased =
                    static_cast<double>(alpha_raw_gpu[h]) +
                    static_cast<double>(dt_bias[h]);
                gate_ref[h] =
                    static_cast<float>(
                        softplus(biased) *
                        static_cast<double>(ssm_a[h]));
            }

            const double gdn_scale =
                1.0 / std::sqrt(static_cast<double>(kHeadDim));
            for (std::uint32_t h = 0; h < kValueHeads; ++h) {
                const std::uint32_t qh = h % kQkHeads;
                const float* qh_ptr =
                    q_ref.data() +
                    static_cast<std::size_t>(qh) * kHeadDim;
                const float* kh_ptr =
                    k_ref.data() +
                    static_cast<std::size_t>(qh) * kHeadDim;
                const double g_val =
                    std::exp(static_cast<double>(gate_ref[h]));
                const double beta_val =
                    static_cast<double>(beta_ref[h]);

                for (std::uint32_t col = 0; col < kHeadDim; ++col) {
                    const std::size_t state_base =
                        (static_cast<std::size_t>(h) * kHeadDim + col) *
                        kHeadDim;

                    double kv = 0.0;
                    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                        kv +=
                            static_cast<double>(
                                gdn_state_in[state_base + i]) *
                            static_cast<double>(kh_ptr[i]);
                    }

                    const std::size_t v_idx =
                        static_cast<std::size_t>(2 * kKeyDim) +
                        static_cast<std::size_t>(h) * kHeadDim + col;
                    const double delta =
                        (static_cast<double>(conv_ref[v_idx]) -
                         g_val * kv) *
                        beta_val;

                    double attn = 0.0;
                    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                        const double updated =
                            g_val *
                                static_cast<double>(
                                    gdn_state_in[state_base + i]) +
                            static_cast<double>(kh_ptr[i]) * delta;
                        gdn_state_ref[state_base + i] =
                            static_cast<float>(updated);
                        attn +=
                            updated * static_cast<double>(qh_ptr[i]);
                    }

                    gdn_out_ref[
                        static_cast<std::size_t>(h) * kHeadDim + col] =
                        static_cast<float>(attn * gdn_scale);
                }
            }

            std::vector<float> gated_norm_ref(kValueDim);
            for (std::uint32_t h = 0; h < kValueHeads; ++h) {
                const std::size_t base_h =
                    static_cast<std::size_t>(h) * kHeadDim;
                double ss = 0.0;
                for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                    const double x =
                        static_cast<double>(gdn_out_ref[base_h + i]);
                    ss += x * x;
                }
                const double inv_rms =
                    1.0 / std::sqrt(
                        ss / static_cast<double>(kHeadDim) +
                        static_cast<double>(rms_eps));
                for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                    const std::size_t idx = base_h + i;
                    const double zv = static_cast<double>(z_gpu[idx]);
                    const double silu_z =
                        zv / (1.0 + std::exp(-zv));
                    gated_norm_ref[idx] =
                        static_cast<float>(
                            static_cast<double>(gdn_out_ref[idx]) *
                            inv_rms *
                            static_cast<double>(ssm_norm_weight[i]) *
                            silu_z);
                }
            }

            constexpr std::size_t kTailCheckedRows = 8;
            std::array<float, kQ4KValuesPerBlock> tail_deq{};
            std::array<std::byte, kQ5KSm86BytesPerBlock> tail_block{};
            std::array<double, kTailCheckedRows> ssm_out_ref{};
            std::array<double, kTailCheckedRows> residual_ref{};

            for (std::size_t row = 0; row < kTailCheckedRows; ++row) {
                double dot = 0.0;
                for (std::size_t ib = 0;
                     ib < ssm_out_blocks_per_row; ++ib) {
                    const std::size_t block_index =
                        row * ssm_out_blocks_per_row + ib;
                    std::memcpy(
                        tail_block.data() + 0,
                        ssm_out_matrix + block_index * 20,
                        20);
                    std::memcpy(
                        tail_block.data() + 20,
                        ssm_out_matrix +
                            ssm_out_qh_offset + block_index * 32,
                        32);
                    std::memcpy(
                        tail_block.data() + 52,
                        ssm_out_matrix +
                            ssm_out_qs_offset + block_index * 128,
                        128);
                    dequantize_q5_k_sm86_block_cpu(
                        tail_block.data(), tail_deq);

                    const std::size_t base_x =
                        ib * kQ4KValuesPerBlock;
                    for (std::size_t j = 0;
                         j < kQ4KValuesPerBlock; ++j) {
                        dot +=
                            static_cast<double>(tail_deq[j]) *
                            static_cast<double>(
                                gated_norm_ref[base_x + j]);
                    }
                }
                ssm_out_ref[row] = dot;
                residual_ref[row] =
                    dot + static_cast<double>(hidden[row]);
            }

            Layer0RecurrentFrontStats front_local{};

            auto calc_max_abs = [](const std::vector<float>& a,
                                   const std::vector<float>& b) {
                double m = 0.0;
                for (std::size_t i = 0; i < a.size(); ++i) {
                    m = std::max(
                        m,
                        std::abs(
                            static_cast<double>(a[i]) -
                            static_cast<double>(b[i])));
                }
                return m;
            };

            auto calc_max_rel = [](const std::vector<float>& a,
                                   const std::vector<float>& b) {
                double m = 0.0;
                for (std::size_t i = 0; i < a.size(); ++i) {
                    const double ref = static_cast<double>(b[i]);
                    const double abs_err =
                        std::abs(
                            static_cast<double>(a[i]) - ref);
                    m = std::max(
                        m,
                        abs_err /
                            std::max(1.0e-6, std::abs(ref)));
                }
                return m;
            };

            front_local.conv_max_abs =
                calc_max_abs(conv_gpu, conv_ref);
            front_local.q_max_abs =
                calc_max_abs(q_gpu, q_ref);
            front_local.k_max_abs =
                calc_max_abs(k_gpu, k_ref);
            front_local.beta_max_abs =
                calc_max_abs(beta_gpu, beta_ref);
            front_local.gate_max_abs =
                calc_max_abs(gate_gpu, gate_ref);
            front_local.conv_state_max_abs =
                calc_max_abs(conv_state_gpu, conv_state_ref);
            front_local.gdn_output_max_abs =
                calc_max_abs(gdn_out_gpu, gdn_out_ref);
            front_local.gdn_output_max_rel =
                calc_max_rel(gdn_out_gpu, gdn_out_ref);
            front_local.gdn_state_max_abs =
                calc_max_abs(gdn_state_gpu, gdn_state_ref);
            front_local.gdn_state_max_rel =
                calc_max_rel(gdn_state_gpu, gdn_state_ref);

            if (front_local.conv_max_abs > 1.0e-3 ||
                front_local.q_max_abs > 1.0e-3 ||
                front_local.k_max_abs > 1.0e-3 ||
                front_local.beta_max_abs > 1.0e-4 ||
                front_local.gate_max_abs > 1.0e-3 ||
                front_local.conv_state_max_abs > 1.0e-6 ||
                (front_local.gdn_output_max_abs > 2.0e-3 &&
                 front_local.gdn_output_max_rel > 2.0e-3) ||
                (front_local.gdn_state_max_abs > 1.0e-3 &&
                 front_local.gdn_state_max_rel > 1.0e-3)) {
                std::ostringstream oss;
                oss << "layer0 recurrent front mismatch:"
                    << " conv=" << front_local.conv_max_abs
                    << " q=" << front_local.q_max_abs
                    << " k=" << front_local.k_max_abs
                    << " beta=" << front_local.beta_max_abs
                    << " gate=" << front_local.gate_max_abs
                    << " conv_state=" << front_local.conv_state_max_abs
                    << " gdn_out_abs=" << front_local.gdn_output_max_abs
                    << " gdn_out_rel=" << front_local.gdn_output_max_rel
                    << " gdn_state_abs=" << front_local.gdn_state_max_abs
                    << " gdn_state_rel=" << front_local.gdn_state_max_rel;
                throw std::runtime_error(oss.str());
            }

            Layer0RecurrentAttentionStats attention_local{};

            for (std::size_t i = 0; i < gated_norm_gpu.size(); ++i) {
                attention_local.gated_norm_max_abs =
                    std::max(
                        attention_local.gated_norm_max_abs,
                        std::abs(
                            static_cast<double>(gated_norm_gpu[i]) -
                            static_cast<double>(gated_norm_ref[i])));
            }

            for (std::size_t row = 0;
                 row < kTailCheckedRows; ++row) {
                attention_local.ssm_out_max_abs =
                    std::max(
                        attention_local.ssm_out_max_abs,
                        std::abs(
                            static_cast<double>(ssm_out_gpu[row]) -
                            ssm_out_ref[row]));
                attention_local.residual_max_abs =
                    std::max(
                        attention_local.residual_max_abs,
                        std::abs(
                            static_cast<double>(residual_gpu[row]) -
                            residual_ref[row]));
            }

            if (attention_local.gated_norm_max_abs > 2.0e-4 ||
                attention_local.ssm_out_max_abs > 3.0e-3 ||
                attention_local.residual_max_abs > 3.0e-3) {
                std::ostringstream oss;
                oss << "layer0 recurrent attention tail mismatch:"
                    << " gated_norm="
                    << attention_local.gated_norm_max_abs
                    << " ssm_out="
                    << attention_local.ssm_out_max_abs
                    << " residual="
                    << attention_local.residual_max_abs;
                throw std::runtime_error(oss.str());
            }

            auto bench = [&](int iters, auto&& fn, const char* label) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < iters; ++i) fn();
                check(handle_, sync(), label);
                const auto t1 = std::chrono::steady_clock::now();
                return std::chrono::duration<double, std::milli>(
                    t1 - t0).count() /
                    static_cast<double>(iters);
            };

            front_local.projection_ms =
                bench(50, launch_projection,
                      "cuCtxSynchronize(front projection benchmark)");
            front_local.conv_prep_ms =
                bench(200, launch_prep,
                      "cuCtxSynchronize(front prep benchmark)");
            front_local.gdn_ms =
                bench(200, launch_gdn,
                      "cuCtxSynchronize(front gdn benchmark)");
            front_local.sum_stage_ms =
                front_local.projection_ms +
                front_local.conv_prep_ms +
                front_local.gdn_ms;
            front_local.chain_ms =
                bench(50, launch_front,
                      "cuCtxSynchronize(front chain benchmark)");

            attention_local.front_ms = front_local.chain_ms;
            attention_local.gated_norm_ms =
                bench(200, launch_gated_norm,
                      "cuCtxSynchronize(attention gated norm benchmark)");
            attention_local.ssm_out_ms =
                bench(50, launch_ssm_out,
                      "cuCtxSynchronize(attention ssm_out benchmark)");
            attention_local.tail_ms =
                bench(50, launch_tail,
                      "cuCtxSynchronize(attention tail benchmark)");
            attention_local.sum_stage_ms =
                attention_local.front_ms +
                attention_local.tail_ms;
            attention_local.chain_ms =
                bench(50, launch_chain,
                      "cuCtxSynchronize(attention full chain benchmark)");

            if (stats) *stats = attention_local;
        } catch (...) {
            if (tail_module) module_unload(tail_module);
            if (gdn_module) module_unload(gdn_module);
            if (prep_module) module_unload(prep_module);
            if (q4_module) module_unload(q4_module);
            if (q5_module) module_unload(q5_module);
            module_unload(norm_module);
            throw;
        }

        check(handle_, module_unload(tail_module),
              "cuModuleUnload(attention tail)");
        check(handle_, module_unload(gdn_module),
              "cuModuleUnload(front GDN)");
        check(handle_, module_unload(prep_module),
              "cuModuleUnload(front prep)");
        check(handle_, module_unload(q4_module),
              "cuModuleUnload(front Q4)");
        check(handle_, module_unload(q5_module),
              "cuModuleUnload(front Q5)");
        check(handle_, module_unload(norm_module),
              "cuModuleUnload(front RMSNorm)");

        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool NvidiaDriver::run_qwen35_layer0_recurrent_front(
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
    Layer0RecurrentFrontStats* stats,
    std::string* error) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error(
                "q38 layer0 recurrent front requires sm_86; detected sm_" +
                std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (!norm_weight || !qkv_matrix || !z_matrix ||
            !beta_matrix || !alpha_matrix ||
            !conv_weight || !dt_bias || !ssm_a) {
            throw std::invalid_argument(
                "layer0 recurrent front received null model tensor");
        }

        constexpr std::uint32_t kCols = 5120;
        constexpr std::uint32_t kQkvRows = 10240;
        constexpr std::uint32_t kZRows = 6144;
        constexpr std::uint32_t kSmallRows = 48;
        constexpr std::uint32_t kHeadDim = 128;
        constexpr std::uint32_t kQkHeads = 16;
        constexpr std::uint32_t kValueHeads = 48;
        constexpr std::uint32_t kKeyDim = kQkHeads * kHeadDim;
        constexpr std::uint32_t kValueDim = kValueHeads * kHeadDim;
        constexpr std::uint32_t kConvStateSteps = 3;
        constexpr std::size_t kGdnStateValues =
            static_cast<std::size_t>(kValueHeads) * kHeadDim * kHeadDim;

        const std::size_t q5_blocks_per_row = kCols / kQ4KValuesPerBlock;

        const std::size_t qkv_blocks =
            static_cast<std::size_t>(kQkvRows) * q5_blocks_per_row;
        const std::size_t qkv_meta_bytes = qkv_blocks * 20;
        const std::size_t qkv_qh_bytes = qkv_blocks * 32;
        const std::size_t qkv_qs_bytes = qkv_blocks * 128;
        if (qkv_qh_offset < qkv_meta_bytes ||
            qkv_qs_offset < qkv_qh_offset + qkv_qh_bytes) {
            throw std::invalid_argument("invalid qkv Q5_K plane offsets");
        }
        const std::size_t qkv_bytes = qkv_qs_offset + qkv_qs_bytes;

        const std::size_t z_blocks =
            static_cast<std::size_t>(kZRows) * q5_blocks_per_row;
        const std::size_t z_meta_bytes = z_blocks * 20;
        const std::size_t z_qh_bytes = z_blocks * 32;
        const std::size_t z_qs_bytes = z_blocks * 128;
        if (z_qh_offset < z_meta_bytes ||
            z_qs_offset < z_qh_offset + z_qh_bytes) {
            throw std::invalid_argument("invalid z Q5_K plane offsets");
        }
        const std::size_t z_bytes = z_qs_offset + z_qs_bytes;

        const std::size_t q4_blocks_per_row =
            kCols / kQ4KValuesPerBlock;
        const std::size_t small_matrix_bytes =
            static_cast<std::size_t>(kSmallRows) *
            q4_blocks_per_row * kQ4KBytesPerBlock;

        const std::size_t hidden_bytes =
            static_cast<std::size_t>(kCols) * sizeof(float);
        const std::size_t qkv_out_bytes =
            static_cast<std::size_t>(kQkvRows) * sizeof(float);
        const std::size_t z_out_bytes =
            static_cast<std::size_t>(kZRows) * sizeof(float);
        const std::size_t small_bytes =
            static_cast<std::size_t>(kSmallRows) * sizeof(float);
        const std::size_t conv_weight_bytes =
            static_cast<std::size_t>(4) * kQkvRows * sizeof(float);
        const std::size_t conv_state_bytes =
            static_cast<std::size_t>(kConvStateSteps) *
            kQkvRows * sizeof(float);
        const std::size_t qk_bytes =
            static_cast<std::size_t>(kKeyDim) * sizeof(float);
        const std::size_t gdn_out_bytes =
            static_cast<std::size_t>(kValueDim) * sizeof(float);
        const std::size_t gdn_state_bytes =
            kGdnStateValues * sizeof(float);

        auto align256 = [](std::size_t n) {
            return (n + 255u) & ~std::size_t(255u);
        };

        // Model tensors and persistent test-state inputs.
        const std::size_t qkv_w_off = 0;
        const std::size_t z_w_off = align256(qkv_w_off + qkv_bytes);
        const std::size_t beta_w_off = align256(z_w_off + z_bytes);
        const std::size_t alpha_w_off =
            align256(beta_w_off + small_matrix_bytes);
        const std::size_t norm_w_off =
            align256(alpha_w_off + small_matrix_bytes);
        const std::size_t conv_w_off =
            align256(norm_w_off + hidden_bytes);
        const std::size_t dt_off =
            align256(conv_w_off + conv_weight_bytes);
        const std::size_t a_off =
            align256(dt_off + small_bytes);
        const std::size_t hidden_off =
            align256(a_off + small_bytes);
        const std::size_t conv_state_in_off =
            align256(hidden_off + hidden_bytes);
        const std::size_t gdn_state_in_off =
            align256(conv_state_in_off + conv_state_bytes);

        // Runtime workspace.
        const std::size_t sumsq_off =
            align256(gdn_state_in_off + gdn_state_bytes);
        const std::size_t normed_off =
            align256(sumsq_off + sizeof(float));
        const std::size_t qkv_out_off =
            align256(normed_off + hidden_bytes);
        const std::size_t z_out_off =
            align256(qkv_out_off + qkv_out_bytes);
        const std::size_t beta_raw_off =
            align256(z_out_off + z_out_bytes);
        const std::size_t alpha_raw_off =
            align256(beta_raw_off + small_bytes);
        const std::size_t conv_out_off =
            align256(alpha_raw_off + small_bytes);
        const std::size_t conv_state_out_off =
            align256(conv_out_off + qkv_out_bytes);
        const std::size_t q_out_off =
            align256(conv_state_out_off + conv_state_bytes);
        const std::size_t k_out_off =
            align256(q_out_off + qk_bytes);
        const std::size_t beta_out_off =
            align256(k_out_off + qk_bytes);
        const std::size_t gate_out_off =
            align256(beta_out_off + small_bytes);
        const std::size_t gdn_out_off =
            align256(gate_out_off + small_bytes);
        const std::size_t gdn_state_out_off =
            align256(gdn_out_off + gdn_out_bytes);
        const std::size_t total_bytes =
            gdn_state_out_off + gdn_state_bytes;

        ScopedVmm memory(handle_, device_ordinal_, total_bytes);

        using MemcpyHtoD =
            CUresult(*)(CUdeviceptr, const void*, std::size_t);
        using MemcpyDtoH =
            CUresult(*)(void*, CUdeviceptr, std::size_t);
        using MemsetD32 =
            CUresult(*)(CUdeviceptr, unsigned int, std::size_t);
        using ModuleLoadDataEx =
            CUresult(*)(CUmodule*, const void*, unsigned int, int*, void**);
        using ModuleUnload = CUresult(*)(CUmodule);
        using ModuleGetFunction =
            CUresult(*)(CUfunction*, CUmodule, const char*);
        using LaunchKernel =
            CUresult(*)(CUfunction,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, CUstream, void**, void**);
        using CtxSynchronize = CUresult(*)();

        const auto memcpy_htod =
            sym<MemcpyHtoD>(handle_, "cuMemcpyHtoD_v2");
        const auto memcpy_dtoh =
            sym<MemcpyDtoH>(handle_, "cuMemcpyDtoH_v2");
        const auto memset_d32 =
            sym<MemsetD32>(handle_, "cuMemsetD32_v2");
        const auto module_load_ex =
            sym<ModuleLoadDataEx>(handle_, "cuModuleLoadDataEx");
        const auto module_unload =
            sym<ModuleUnload>(handle_, "cuModuleUnload");
        const auto module_get_function =
            sym<ModuleGetFunction>(handle_, "cuModuleGetFunction");
        const auto launch =
            sym<LaunchKernel>(handle_, "cuLaunchKernel");
        const auto sync =
            sym<CtxSynchronize>(handle_, "cuCtxSynchronize");

        // Deterministic decode-time input and recurrent states.
        std::vector<float> hidden(kCols);
        std::vector<float> conv_state_in(
            static_cast<std::size_t>(kConvStateSteps) * kQkvRows);
        std::vector<float> gdn_state_in(kGdnStateValues);

        for (std::uint32_t i = 0; i < kCols; ++i) {
            const float fi = static_cast<float>(i);
            hidden[i] =
                std::sin(fi * 0.017f) * 0.65f +
                std::cos(fi * 0.011f) * 0.35f;
        }
        for (std::uint32_t s = 0; s < kConvStateSteps; ++s) {
            for (std::uint32_t i = 0; i < kQkvRows; ++i) {
                const float fi = static_cast<float>(i);
                const float fs = static_cast<float>(s + 1);
                conv_state_in[
                    static_cast<std::size_t>(s) * kQkvRows + i] =
                    0.08f *
                        std::sin(fi * (0.0029f + 0.0006f * fs)) +
                    0.03f *
                        std::cos(fi * (0.0017f + 0.0004f * fs));
            }
        }
        for (std::size_t i = 0; i < gdn_state_in.size(); ++i) {
            const float fi = static_cast<float>(i);
            gdn_state_in[i] =
                0.011f * std::sin(fi * 0.0011f) +
                0.005f * std::cos(fi * 0.00073f);
        }

        const CUdeviceptr base = memory.ptr();
        const CUdeviceptr qkv_w_ptr = base + qkv_w_off;
        const CUdeviceptr z_w_ptr = base + z_w_off;
        const CUdeviceptr beta_w_ptr = base + beta_w_off;
        const CUdeviceptr alpha_w_ptr = base + alpha_w_off;
        const CUdeviceptr norm_w_ptr = base + norm_w_off;
        const CUdeviceptr conv_w_ptr = base + conv_w_off;
        const CUdeviceptr dt_ptr = base + dt_off;
        const CUdeviceptr a_ptr = base + a_off;
        const CUdeviceptr hidden_ptr = base + hidden_off;
        const CUdeviceptr conv_state_in_ptr = base + conv_state_in_off;
        const CUdeviceptr gdn_state_in_ptr = base + gdn_state_in_off;

        const CUdeviceptr sumsq_ptr = base + sumsq_off;
        const CUdeviceptr normed_ptr = base + normed_off;
        const CUdeviceptr qkv_out_ptr = base + qkv_out_off;
        const CUdeviceptr z_out_ptr = base + z_out_off;
        const CUdeviceptr beta_raw_ptr = base + beta_raw_off;
        const CUdeviceptr alpha_raw_ptr = base + alpha_raw_off;
        const CUdeviceptr conv_out_ptr = base + conv_out_off;
        const CUdeviceptr conv_state_out_ptr = base + conv_state_out_off;
        const CUdeviceptr q_out_ptr = base + q_out_off;
        const CUdeviceptr k_out_ptr = base + k_out_off;
        const CUdeviceptr beta_out_ptr = base + beta_out_off;
        const CUdeviceptr gate_out_ptr = base + gate_out_off;
        const CUdeviceptr gdn_out_ptr = base + gdn_out_off;
        const CUdeviceptr gdn_state_out_ptr = base + gdn_state_out_off;

        check(handle_, memcpy_htod(
            qkv_w_ptr, qkv_matrix, qkv_bytes),
            "cuMemcpyHtoD(front qkv weight)");
        check(handle_, memcpy_htod(
            z_w_ptr, z_matrix, z_bytes),
            "cuMemcpyHtoD(front z weight)");
        check(handle_, memcpy_htod(
            beta_w_ptr, beta_matrix, small_matrix_bytes),
            "cuMemcpyHtoD(front beta weight)");
        check(handle_, memcpy_htod(
            alpha_w_ptr, alpha_matrix, small_matrix_bytes),
            "cuMemcpyHtoD(front alpha weight)");
        check(handle_, memcpy_htod(
            norm_w_ptr, norm_weight, hidden_bytes),
            "cuMemcpyHtoD(front norm weight)");
        check(handle_, memcpy_htod(
            conv_w_ptr, conv_weight, conv_weight_bytes),
            "cuMemcpyHtoD(front conv weight)");
        check(handle_, memcpy_htod(
            dt_ptr, dt_bias, small_bytes),
            "cuMemcpyHtoD(front dt bias)");
        check(handle_, memcpy_htod(
            a_ptr, ssm_a, small_bytes),
            "cuMemcpyHtoD(front ssm a)");
        check(handle_, memcpy_htod(
            hidden_ptr, hidden.data(), hidden_bytes),
            "cuMemcpyHtoD(front hidden)");
        check(handle_, memcpy_htod(
            conv_state_in_ptr, conv_state_in.data(), conv_state_bytes),
            "cuMemcpyHtoD(front conv state)");
        check(handle_, memcpy_htod(
            gdn_state_in_ptr, gdn_state_in.data(), gdn_state_bytes),
            "cuMemcpyHtoD(front gdn state)");

        constexpr int CU_JIT_INFO_LOG_BUFFER = 3;
        constexpr int CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4;
        constexpr int CU_JIT_ERROR_LOG_BUFFER = 5;
        constexpr int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;
        constexpr int CU_JIT_LOG_VERBOSE = 12;

        auto load_module = [&](const char* ptx, const char* label) {
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
                reinterpret_cast<void*>(
                    static_cast<std::uintptr_t>(jit_info.size())),
                jit_error.data(),
                reinterpret_cast<void*>(
                    static_cast<std::uintptr_t>(jit_error.size())),
                reinterpret_cast<void*>(
                    static_cast<std::uintptr_t>(1)),
            };
            CUmodule module{};
            const auto rc = module_load_ex(
                &module,
                ptx,
                static_cast<unsigned int>(
                    sizeof(jit_options) / sizeof(jit_options[0])),
                jit_options,
                jit_values);
            if (rc != CUDA_SUCCESS) {
                std::string detail = cuda_error(handle_, rc, label);
                if (jit_error[0] != '\0') {
                    detail +=
                        std::string("\nPTX JIT error log:\n") +
                        jit_error.data();
                }
                if (jit_info[0] != '\0') {
                    detail +=
                        std::string("\nPTX JIT info log:\n") +
                        jit_info.data();
                }
                throw std::runtime_error(detail);
            }
            return module;
        };

        CUmodule norm_module =
            load_module(kRmsNormPtx, "cuModuleLoadDataEx(front RMSNorm)");
        CUmodule q5_module{};
        CUmodule q4_module{};
        CUmodule prep_module{};
        CUmodule gdn_module{};

        try {
            q5_module = load_module(
                kQ5KSm86SoAGemvVecPtx,
                "cuModuleLoadDataEx(front Q5)");
            q4_module = load_module(
                kQ4KGemvPtx,
                "cuModuleLoadDataEx(front Q4)");
            prep_module = load_module(
                kRecurrentPrepPtx,
                "cuModuleLoadDataEx(front prep)");
            gdn_module = load_module(
                kGdnAr128Ptx,
                "cuModuleLoadDataEx(front GDN)");

            CUfunction sum_fn{}, norm_fn{}, q5_fn{}, q4_fn{};
            CUfunction conv_fn{}, qk_fn{}, bg_fn{}, gdn_fn{};

            check(handle_, module_get_function(
                &sum_fn, norm_module, "q38_sumsq"),
                "cuModuleGetFunction(front sumsq)");
            check(handle_, module_get_function(
                &norm_fn, norm_module, "q38_rmsnorm_apply"),
                "cuModuleGetFunction(front rmsnorm)");
            check(handle_, module_get_function(
                &q5_fn, q5_module, "q38_q5k_sm86_soa_gemv_vec"),
                "cuModuleGetFunction(front q5)");
            check(handle_, module_get_function(
                &q4_fn, q4_module, "q38_q4k_gemv_f32"),
                "cuModuleGetFunction(front q4)");
            check(handle_, module_get_function(
                &conv_fn, prep_module, "q38_conv4_silu_roll"),
                "cuModuleGetFunction(front conv)");
            check(handle_, module_get_function(
                &qk_fn, prep_module, "q38_qk_l2norm_128"),
                "cuModuleGetFunction(front qk norm)");
            check(handle_, module_get_function(
                &bg_fn, prep_module, "q38_beta_gate_48"),
                "cuModuleGetFunction(front beta gate)");
            check(handle_, module_get_function(
                &gdn_fn, gdn_module, "q38_gdn_ar_128"),
                "cuModuleGetFunction(front gdn)");

            // RMSNorm args.
            constexpr unsigned int norm_block = 256;
            const unsigned int norm_grid =
                (kCols + norm_block - 1) / norm_block;

            CUdeviceptr sum_x = hidden_ptr;
            CUdeviceptr sum_out = sumsq_ptr;
            std::uint32_t norm_count = kCols;
            void* sum_params[] = {
                &sum_x, &sum_out, &norm_count
            };

            CUdeviceptr norm_x = hidden_ptr;
            CUdeviceptr norm_w = norm_w_ptr;
            CUdeviceptr norm_y = normed_ptr;
            CUdeviceptr norm_sumsq = sumsq_ptr;
            float norm_eps = rms_eps;
            void* norm_params[] = {
                &norm_x, &norm_w, &norm_y,
                &norm_sumsq, &norm_count, &norm_eps
            };

            // Q5 qkv args.
            CUdeviceptr qkv_meta = qkv_w_ptr;
            CUdeviceptr qkv_qh = qkv_w_ptr + qkv_qh_offset;
            CUdeviceptr qkv_qs = qkv_w_ptr + qkv_qs_offset;
            CUdeviceptr qkv_x = normed_ptr;
            CUdeviceptr qkv_y = qkv_out_ptr;
            std::uint32_t qkv_cols = kCols;
            std::uint32_t qkv_rows = kQkvRows;
            void* qkv_params[] = {
                &qkv_meta, &qkv_qh, &qkv_qs, &qkv_x, &qkv_y,
                &qkv_cols, &qkv_rows
            };

            // Q5 z args.
            CUdeviceptr z_meta = z_w_ptr;
            CUdeviceptr z_qh = z_w_ptr + z_qh_offset;
            CUdeviceptr z_qs = z_w_ptr + z_qs_offset;
            CUdeviceptr z_x = normed_ptr;
            CUdeviceptr z_y = z_out_ptr;
            std::uint32_t z_cols = kCols;
            std::uint32_t z_rows = kZRows;
            void* z_params[] = {
                &z_meta, &z_qh, &z_qs, &z_x, &z_y,
                &z_cols, &z_rows
            };

            // Q4 beta/alpha args.
            CUdeviceptr beta_w = beta_w_ptr;
            CUdeviceptr beta_x = normed_ptr;
            CUdeviceptr beta_y = beta_raw_ptr;
            std::uint32_t beta_cols = kCols;
            std::uint32_t beta_rows = kSmallRows;
            void* beta_params[] = {
                &beta_w, &beta_x, &beta_y,
                &beta_cols, &beta_rows
            };

            CUdeviceptr alpha_w = alpha_w_ptr;
            CUdeviceptr alpha_x = normed_ptr;
            CUdeviceptr alpha_y = alpha_raw_ptr;
            std::uint32_t alpha_cols = kCols;
            std::uint32_t alpha_rows = kSmallRows;
            void* alpha_params[] = {
                &alpha_w, &alpha_x, &alpha_y,
                &alpha_cols, &alpha_rows
            };

            // Conv prep args.
            CUdeviceptr c_qkv = qkv_out_ptr;
            CUdeviceptr c_w = conv_w_ptr;
            CUdeviceptr c_si = conv_state_in_ptr;
            CUdeviceptr c_co = conv_out_ptr;
            CUdeviceptr c_so = conv_state_out_ptr;
            std::uint32_t c_channels = kQkvRows;
            float log2e = 1.4426950408889634f;
            void* conv_params[] = {
                &c_qkv, &c_w, &c_si, &c_co, &c_so,
                &c_channels, &log2e
            };

            CUdeviceptr n_conv = conv_out_ptr;
            CUdeviceptr n_q = q_out_ptr;
            CUdeviceptr n_k = k_out_ptr;
            float qk_eps = rms_eps;
            void* qk_params[] = {
                &n_conv, &n_q, &n_k, &qk_eps
            };

            CUdeviceptr b_br = beta_raw_ptr;
            CUdeviceptr b_ar = alpha_raw_ptr;
            CUdeviceptr b_dt = dt_ptr;
            CUdeviceptr b_a = a_ptr;
            CUdeviceptr b_bo = beta_out_ptr;
            CUdeviceptr b_go = gate_out_ptr;
            float inv_log2e = 0.6931471805599453f;
            void* bg_params[] = {
                &b_br, &b_ar, &b_dt, &b_a, &b_bo, &b_go,
                &log2e, &inv_log2e
            };

            // GDN args. v is the last 6144 values of conv_out.
            CUdeviceptr g_q = q_out_ptr;
            CUdeviceptr g_k = k_out_ptr;
            CUdeviceptr g_v =
                conv_out_ptr +
                static_cast<CUdeviceptr>(2 * kKeyDim * sizeof(float));
            CUdeviceptr g_gate = gate_out_ptr;
            CUdeviceptr g_beta = beta_out_ptr;
            CUdeviceptr g_state_in = gdn_state_in_ptr;
            CUdeviceptr g_out = gdn_out_ptr;
            CUdeviceptr g_state_out = gdn_state_out_ptr;
            std::uint32_t g_qk_heads = kQkHeads;
            std::uint32_t g_value_heads = kValueHeads;
            float g_scale =
                1.0f / std::sqrt(static_cast<float>(kHeadDim));
            void* gdn_params[] = {
                &g_q, &g_k, &g_v, &g_gate, &g_beta,
                &g_state_in, &g_out, &g_state_out,
                &g_qk_heads, &g_value_heads,
                &g_scale, &log2e
            };

            const unsigned int qkv_grid =
                (kQkvRows + 3u) / 4u;
            const unsigned int z_grid =
                (kZRows + 3u) / 4u;
            const unsigned int conv_grid =
                (kQkvRows + 255u) / 256u;
            const unsigned int gdn_grid_y =
                (kHeadDim + 3u) / 4u;

            auto launch_projection = [&]() {
                check(handle_, memset_d32(
                    sumsq_ptr, 0, 1),
                    "cuMemsetD32(front sumsq)");
                check(handle_, launch(
                    sum_fn, norm_grid, 1, 1,
                    norm_block, 1, 1,
                    0, nullptr, sum_params, nullptr),
                    "cuLaunchKernel(front sumsq)");
                check(handle_, launch(
                    norm_fn, norm_grid, 1, 1,
                    norm_block, 1, 1,
                    0, nullptr, norm_params, nullptr),
                    "cuLaunchKernel(front rmsnorm)");
                check(handle_, launch(
                    q5_fn, qkv_grid, 1, 1,
                    128, 1, 1,
                    0, nullptr, qkv_params, nullptr),
                    "cuLaunchKernel(front qkv)");
                check(handle_, launch(
                    q5_fn, z_grid, 1, 1,
                    128, 1, 1,
                    0, nullptr, z_params, nullptr),
                    "cuLaunchKernel(front z)");
                check(handle_, launch(
                    q4_fn, kSmallRows, 1, 1,
                    32, 1, 1,
                    0, nullptr, beta_params, nullptr),
                    "cuLaunchKernel(front beta)");
                check(handle_, launch(
                    q4_fn, kSmallRows, 1, 1,
                    32, 1, 1,
                    0, nullptr, alpha_params, nullptr),
                    "cuLaunchKernel(front alpha)");
            };

            auto launch_prep = [&]() {
                check(handle_, launch(
                    conv_fn, conv_grid, 1, 1,
                    256, 1, 1,
                    0, nullptr, conv_params, nullptr),
                    "cuLaunchKernel(front conv)");
                check(handle_, launch(
                    qk_fn, 32, 1, 1,
                    128, 1, 1,
                    0, nullptr, qk_params, nullptr),
                    "cuLaunchKernel(front qk norm)");
                check(handle_, launch(
                    bg_fn, 1, 1, 1,
                    64, 1, 1,
                    0, nullptr, bg_params, nullptr),
                    "cuLaunchKernel(front beta gate)");
            };

            auto launch_gdn = [&]() {
                check(handle_, launch(
                    gdn_fn,
                    kValueHeads, gdn_grid_y, 1,
                    128, 1, 1,
                    0, nullptr, gdn_params, nullptr),
                    "cuLaunchKernel(front gdn)");
            };

            auto launch_chain = [&]() {
                launch_projection();
                launch_prep();
                launch_gdn();
            };

            // One full chain for correctness.
            launch_chain();
            check(handle_, sync(),
                  "cuCtxSynchronize(layer0 recurrent front correctness)");

            // Read GPU projection products used as the CPU downstream oracle.
            // The projection kernels are already independently validated by
            // q38-layer0-projections; this check focuses on composition.
            std::vector<float> qkv_gpu(kQkvRows);
            std::vector<float> beta_raw_gpu(kSmallRows);
            std::vector<float> alpha_raw_gpu(kSmallRows);
            std::vector<float> conv_gpu(kQkvRows);
            std::vector<float> conv_state_gpu(
                static_cast<std::size_t>(kConvStateSteps) * kQkvRows);
            std::vector<float> q_gpu(kKeyDim);
            std::vector<float> k_gpu(kKeyDim);
            std::vector<float> beta_gpu(kSmallRows);
            std::vector<float> gate_gpu(kSmallRows);
            std::vector<float> gdn_out_gpu(kValueDim);
            std::vector<float> gdn_state_gpu(kGdnStateValues);

            check(handle_, memcpy_dtoh(
                qkv_gpu.data(), qkv_out_ptr, qkv_out_bytes),
                "cuMemcpyDtoH(front qkv)");
            check(handle_, memcpy_dtoh(
                beta_raw_gpu.data(), beta_raw_ptr, small_bytes),
                "cuMemcpyDtoH(front beta raw)");
            check(handle_, memcpy_dtoh(
                alpha_raw_gpu.data(), alpha_raw_ptr, small_bytes),
                "cuMemcpyDtoH(front alpha raw)");
            check(handle_, memcpy_dtoh(
                conv_gpu.data(), conv_out_ptr, qkv_out_bytes),
                "cuMemcpyDtoH(front conv)");
            check(handle_, memcpy_dtoh(
                conv_state_gpu.data(), conv_state_out_ptr, conv_state_bytes),
                "cuMemcpyDtoH(front conv state)");
            check(handle_, memcpy_dtoh(
                q_gpu.data(), q_out_ptr, qk_bytes),
                "cuMemcpyDtoH(front q)");
            check(handle_, memcpy_dtoh(
                k_gpu.data(), k_out_ptr, qk_bytes),
                "cuMemcpyDtoH(front k)");
            check(handle_, memcpy_dtoh(
                beta_gpu.data(), beta_out_ptr, small_bytes),
                "cuMemcpyDtoH(front beta)");
            check(handle_, memcpy_dtoh(
                gate_gpu.data(), gate_out_ptr, small_bytes),
                "cuMemcpyDtoH(front gate)");
            check(handle_, memcpy_dtoh(
                gdn_out_gpu.data(), gdn_out_ptr, gdn_out_bytes),
                "cuMemcpyDtoH(front gdn output)");
            check(handle_, memcpy_dtoh(
                gdn_state_gpu.data(), gdn_state_out_ptr, gdn_state_bytes),
                "cuMemcpyDtoH(front gdn state)");

            std::vector<float> conv_ref(kQkvRows);
            std::vector<float> conv_state_ref(
                static_cast<std::size_t>(kConvStateSteps) * kQkvRows);
            std::vector<float> q_ref(kKeyDim);
            std::vector<float> k_ref(kKeyDim);
            std::vector<float> beta_ref(kSmallRows);
            std::vector<float> gate_ref(kSmallRows);
            std::vector<float> gdn_out_ref(kValueDim);
            std::vector<float> gdn_state_ref(kGdnStateValues);

            for (std::uint32_t ch = 0; ch < kQkvRows; ++ch) {
                const float x0 = conv_state_in[ch];
                const float x1 = conv_state_in[kQkvRows + ch];
                const float x2 = conv_state_in[2 * kQkvRows + ch];
                const float x3 = qkv_gpu[ch];
                const float* w =
                    conv_weight + static_cast<std::size_t>(ch) * 4;
                const double raw =
                    static_cast<double>(x0) * w[0] +
                    static_cast<double>(x1) * w[1] +
                    static_cast<double>(x2) * w[2] +
                    static_cast<double>(x3) * w[3];
                conv_ref[ch] =
                    static_cast<float>(
                        raw / (1.0 + std::exp(-raw)));
                conv_state_ref[ch] = x1;
                conv_state_ref[kQkvRows + ch] = x2;
                conv_state_ref[2 * kQkvRows + ch] = x3;
            }

            for (std::uint32_t h = 0; h < kQkHeads; ++h) {
                const std::size_t q_base =
                    static_cast<std::size_t>(h) * kHeadDim;
                const std::size_t k_base =
                    static_cast<std::size_t>(kKeyDim) +
                    static_cast<std::size_t>(h) * kHeadDim;

                double q_ss = 0.0;
                double k_ss = 0.0;
                for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                    const double qv = conv_ref[q_base + i];
                    const double kv = conv_ref[k_base + i];
                    q_ss += qv * qv;
                    k_ss += kv * kv;
                }
                const double q_inv =
                    1.0 / std::sqrt(q_ss + static_cast<double>(rms_eps));
                const double k_inv =
                    1.0 / std::sqrt(k_ss + static_cast<double>(rms_eps));

                for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                    q_ref[q_base + i] =
                        static_cast<float>(
                            static_cast<double>(conv_ref[q_base + i]) *
                            q_inv);
                    k_ref[q_base + i] =
                        static_cast<float>(
                            static_cast<double>(conv_ref[k_base + i]) *
                            k_inv);
                }
            }

            auto softplus = [](double x) {
                if (x > 20.0) return x;
                if (x < -20.0) return std::exp(x);
                return std::log1p(std::exp(x));
            };
            for (std::uint32_t h = 0; h < kSmallRows; ++h) {
                const double br = beta_raw_gpu[h];
                beta_ref[h] =
                    static_cast<float>(
                        1.0 / (1.0 + std::exp(-br)));
                const double biased =
                    static_cast<double>(alpha_raw_gpu[h]) +
                    static_cast<double>(dt_bias[h]);
                gate_ref[h] =
                    static_cast<float>(
                        softplus(biased) *
                        static_cast<double>(ssm_a[h]));
            }

            const double gdn_scale =
                1.0 / std::sqrt(static_cast<double>(kHeadDim));
            for (std::uint32_t h = 0; h < kValueHeads; ++h) {
                const std::uint32_t qh = h % kQkHeads;
                const float* qh_ptr =
                    q_ref.data() +
                    static_cast<std::size_t>(qh) * kHeadDim;
                const float* kh_ptr =
                    k_ref.data() +
                    static_cast<std::size_t>(qh) * kHeadDim;
                const double g_val =
                    std::exp(static_cast<double>(gate_ref[h]));
                const double beta_val =
                    static_cast<double>(beta_ref[h]);

                for (std::uint32_t col = 0; col < kHeadDim; ++col) {
                    const std::size_t state_base =
                        (static_cast<std::size_t>(h) * kHeadDim + col) *
                        kHeadDim;

                    double kv = 0.0;
                    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                        kv +=
                            static_cast<double>(
                                gdn_state_in[state_base + i]) *
                            static_cast<double>(kh_ptr[i]);
                    }

                    const std::size_t v_idx =
                        static_cast<std::size_t>(2 * kKeyDim) +
                        static_cast<std::size_t>(h) * kHeadDim + col;
                    const double delta =
                        (static_cast<double>(conv_ref[v_idx]) -
                         g_val * kv) *
                        beta_val;

                    double attn = 0.0;
                    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                        const double updated =
                            g_val *
                                static_cast<double>(
                                    gdn_state_in[state_base + i]) +
                            static_cast<double>(kh_ptr[i]) * delta;
                        gdn_state_ref[state_base + i] =
                            static_cast<float>(updated);
                        attn +=
                            updated * static_cast<double>(qh_ptr[i]);
                    }

                    gdn_out_ref[
                        static_cast<std::size_t>(h) * kHeadDim + col] =
                        static_cast<float>(attn * gdn_scale);
                }
            }

            Layer0RecurrentFrontStats local{};

            auto calc_max_abs = [](const std::vector<float>& a,
                                   const std::vector<float>& b) {
                double m = 0.0;
                for (std::size_t i = 0; i < a.size(); ++i) {
                    m = std::max(
                        m,
                        std::abs(
                            static_cast<double>(a[i]) -
                            static_cast<double>(b[i])));
                }
                return m;
            };

            auto calc_max_rel = [](const std::vector<float>& a,
                                   const std::vector<float>& b) {
                double m = 0.0;
                for (std::size_t i = 0; i < a.size(); ++i) {
                    const double ref = static_cast<double>(b[i]);
                    const double abs_err =
                        std::abs(
                            static_cast<double>(a[i]) - ref);
                    m = std::max(
                        m,
                        abs_err /
                            std::max(1.0e-6, std::abs(ref)));
                }
                return m;
            };

            local.conv_max_abs =
                calc_max_abs(conv_gpu, conv_ref);
            local.q_max_abs =
                calc_max_abs(q_gpu, q_ref);
            local.k_max_abs =
                calc_max_abs(k_gpu, k_ref);
            local.beta_max_abs =
                calc_max_abs(beta_gpu, beta_ref);
            local.gate_max_abs =
                calc_max_abs(gate_gpu, gate_ref);
            local.conv_state_max_abs =
                calc_max_abs(conv_state_gpu, conv_state_ref);
            local.gdn_output_max_abs =
                calc_max_abs(gdn_out_gpu, gdn_out_ref);
            local.gdn_output_max_rel =
                calc_max_rel(gdn_out_gpu, gdn_out_ref);
            local.gdn_state_max_abs =
                calc_max_abs(gdn_state_gpu, gdn_state_ref);
            local.gdn_state_max_rel =
                calc_max_rel(gdn_state_gpu, gdn_state_ref);

            if (local.conv_max_abs > 1.0e-3 ||
                local.q_max_abs > 1.0e-3 ||
                local.k_max_abs > 1.0e-3 ||
                local.beta_max_abs > 1.0e-4 ||
                local.gate_max_abs > 1.0e-3 ||
                local.conv_state_max_abs > 1.0e-6 ||
                (local.gdn_output_max_abs > 2.0e-3 &&
                 local.gdn_output_max_rel > 2.0e-3) ||
                (local.gdn_state_max_abs > 1.0e-3 &&
                 local.gdn_state_max_rel > 1.0e-3)) {
                std::ostringstream oss;
                oss << "layer0 recurrent front mismatch:"
                    << " conv=" << local.conv_max_abs
                    << " q=" << local.q_max_abs
                    << " k=" << local.k_max_abs
                    << " beta=" << local.beta_max_abs
                    << " gate=" << local.gate_max_abs
                    << " conv_state=" << local.conv_state_max_abs
                    << " gdn_out_abs=" << local.gdn_output_max_abs
                    << " gdn_out_rel=" << local.gdn_output_max_rel
                    << " gdn_state_abs=" << local.gdn_state_max_abs
                    << " gdn_state_rel=" << local.gdn_state_max_rel;
                throw std::runtime_error(oss.str());
            }

            auto bench = [&](int iters, auto&& fn, const char* label) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < iters; ++i) fn();
                check(handle_, sync(), label);
                const auto t1 = std::chrono::steady_clock::now();
                return std::chrono::duration<double, std::milli>(
                    t1 - t0).count() /
                    static_cast<double>(iters);
            };

            local.projection_ms =
                bench(50, launch_projection,
                      "cuCtxSynchronize(front projection benchmark)");
            local.conv_prep_ms =
                bench(200, launch_prep,
                      "cuCtxSynchronize(front prep benchmark)");
            local.gdn_ms =
                bench(200, launch_gdn,
                      "cuCtxSynchronize(front gdn benchmark)");
            local.sum_stage_ms =
                local.projection_ms +
                local.conv_prep_ms +
                local.gdn_ms;
            local.chain_ms =
                bench(50, launch_chain,
                      "cuCtxSynchronize(front chain benchmark)");

            if (stats) *stats = local;
        } catch (...) {
            if (gdn_module) module_unload(gdn_module);
            if (prep_module) module_unload(prep_module);
            if (q4_module) module_unload(q4_module);
            if (q5_module) module_unload(q5_module);
            module_unload(norm_module);
            throw;
        }

        check(handle_, module_unload(gdn_module),
              "cuModuleUnload(front GDN)");
        check(handle_, module_unload(prep_module),
              "cuModuleUnload(front prep)");
        check(handle_, module_unload(q4_module),
              "cuModuleUnload(front Q4)");
        check(handle_, module_unload(q5_module),
              "cuModuleUnload(front Q5)");
        check(handle_, module_unload(norm_module),
              "cuModuleUnload(front RMSNorm)");

        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool NvidiaDriver::run_recurrent_prep_smoke(
    const float* conv_weight,
    const float* dt_bias,
    const float* ssm_a,
    RecurrentPrepSmokeStats* stats,
    std::string* error) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error(
                "q38 recurrent prep smoke requires sm_86; detected sm_" +
                std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (!conv_weight || !dt_bias || !ssm_a) {
            throw std::invalid_argument("recurrent prep received null real-model tensor");
        }

        constexpr std::uint32_t kChannels = 10240;
        constexpr std::uint32_t kKeyDim = 2048;
        constexpr std::uint32_t kHeads = 16;
        constexpr std::uint32_t kHeadDim = 128;
        constexpr std::uint32_t kValueHeads = 48;
        constexpr std::uint32_t kConvStateSteps = 3;
        constexpr float kNormEps = 1.0e-6f;

        std::vector<float> qkv(kChannels);
        std::vector<float> beta_raw(kValueHeads);
        std::vector<float> alpha_raw(kValueHeads);
        std::vector<float> state_in(
            static_cast<std::size_t>(kConvStateSteps) * kChannels);

        for (std::uint32_t i = 0; i < kChannels; ++i) {
            const float fi = static_cast<float>(i);
            qkv[i] =
                0.55f * std::sin(fi * 0.013f) +
                0.31f * std::cos(fi * 0.007f);
            for (std::uint32_t s = 0; s < kConvStateSteps; ++s) {
                const float fs = static_cast<float>(s + 1);
                state_in[static_cast<std::size_t>(s) * kChannels + i] =
                    0.10f * std::sin(fi * (0.003f + 0.001f * fs)) +
                    0.04f * std::cos(fi * (0.002f + 0.0005f * fs));
            }
        }
        for (std::uint32_t h = 0; h < kValueHeads; ++h) {
            const float fh = static_cast<float>(h);
            beta_raw[h] = 0.2f * std::sin(fh * 0.21f) - 0.05f;
            alpha_raw[h] = 0.3f * std::cos(fh * 0.17f) - 0.1f;
        }

        std::vector<float> conv_ref(kChannels);
        std::vector<float> q_ref(kKeyDim);
        std::vector<float> k_ref(kKeyDim);
        std::vector<float> beta_ref(kValueHeads);
        std::vector<float> gate_ref(kValueHeads);
        std::vector<float> state_ref(
            static_cast<std::size_t>(kConvStateSteps) * kChannels);

        for (std::uint32_t ch = 0; ch < kChannels; ++ch) {
            const float x0 = state_in[ch];
            const float x1 = state_in[kChannels + ch];
            const float x2 = state_in[2 * kChannels + ch];
            const float x3 = qkv[ch];
            const float* w = conv_weight + static_cast<std::size_t>(ch) * 4;
            const double raw =
                static_cast<double>(x0) * w[0] +
                static_cast<double>(x1) * w[1] +
                static_cast<double>(x2) * w[2] +
                static_cast<double>(x3) * w[3];
            const double silu = raw / (1.0 + std::exp(-raw));
            conv_ref[ch] = static_cast<float>(silu);
            state_ref[ch] = x1;
            state_ref[kChannels + ch] = x2;
            state_ref[2 * kChannels + ch] = x3;
        }

        for (std::uint32_t hk = 0; hk < kHeads; ++hk) {
            double q_ss = 0.0;
            double k_ss = 0.0;
            const std::size_t q_base =
                static_cast<std::size_t>(hk) * kHeadDim;
            const std::size_t k_base =
                kKeyDim + static_cast<std::size_t>(hk) * kHeadDim;
            for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                const double qv = conv_ref[q_base + i];
                const double kv = conv_ref[k_base + i];
                q_ss += qv * qv;
                k_ss += kv * kv;
            }
            const double q_inv = 1.0 / std::sqrt(q_ss + kNormEps);
            const double k_inv = 1.0 / std::sqrt(k_ss + kNormEps);
            for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                q_ref[q_base + i] =
                    static_cast<float>(
                        static_cast<double>(conv_ref[q_base + i]) * q_inv);
                k_ref[q_base + i] =
                    static_cast<float>(
                        static_cast<double>(conv_ref[k_base + i]) * k_inv);
            }
        }

        auto softplus = [](double x) {
            if (x > 20.0) return x;
            if (x < -20.0) return std::exp(x);
            return std::log1p(std::exp(x));
        };
        for (std::uint32_t h = 0; h < kValueHeads; ++h) {
            const double br = beta_raw[h];
            beta_ref[h] =
                static_cast<float>(1.0 / (1.0 + std::exp(-br)));
            const double biased =
                static_cast<double>(alpha_raw[h]) +
                static_cast<double>(dt_bias[h]);
            gate_ref[h] =
                static_cast<float>(
                    softplus(biased) *
                    static_cast<double>(ssm_a[h]));
        }

        const std::size_t qkv_bytes = qkv.size() * sizeof(float);
        const std::size_t conv_w_bytes =
            static_cast<std::size_t>(kChannels) * 4 * sizeof(float);
        const std::size_t state_bytes =
            state_in.size() * sizeof(float);
        const std::size_t qk_bytes =
            static_cast<std::size_t>(kKeyDim) * sizeof(float);
        const std::size_t small_bytes =
            static_cast<std::size_t>(kValueHeads) * sizeof(float);

        auto align256 = [](std::size_t n) {
            return (n + 255u) & ~std::size_t(255u);
        };

        const std::size_t qkv_off = 0;
        const std::size_t conv_w_off = align256(qkv_off + qkv_bytes);
        const std::size_t state_in_off = align256(conv_w_off + conv_w_bytes);
        const std::size_t conv_out_off = align256(state_in_off + state_bytes);
        const std::size_t state_out_off = align256(conv_out_off + qkv_bytes);
        const std::size_t q_out_off = align256(state_out_off + state_bytes);
        const std::size_t k_out_off = align256(q_out_off + qk_bytes);
        const std::size_t beta_raw_off = align256(k_out_off + qk_bytes);
        const std::size_t alpha_raw_off = align256(beta_raw_off + small_bytes);
        const std::size_t dt_off = align256(alpha_raw_off + small_bytes);
        const std::size_t a_off = align256(dt_off + small_bytes);
        const std::size_t beta_out_off = align256(a_off + small_bytes);
        const std::size_t gate_out_off = align256(beta_out_off + small_bytes);
        const std::size_t total_bytes = gate_out_off + small_bytes;

        ScopedVmm memory(handle_, device_ordinal_, total_bytes);

        using MemcpyHtoD =
            CUresult(*)(CUdeviceptr, const void*, std::size_t);
        using MemcpyDtoH =
            CUresult(*)(void*, CUdeviceptr, std::size_t);
        using ModuleLoadDataEx =
            CUresult(*)(CUmodule*, const void*, unsigned int, int*, void**);
        using ModuleUnload = CUresult(*)(CUmodule);
        using ModuleGetFunction =
            CUresult(*)(CUfunction*, CUmodule, const char*);
        using LaunchKernel =
            CUresult(*)(CUfunction,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, CUstream, void**, void**);
        using CtxSynchronize = CUresult(*)();

        const auto memcpy_htod =
            sym<MemcpyHtoD>(handle_, "cuMemcpyHtoD_v2");
        const auto memcpy_dtoh =
            sym<MemcpyDtoH>(handle_, "cuMemcpyDtoH_v2");
        const auto module_load_ex =
            sym<ModuleLoadDataEx>(handle_, "cuModuleLoadDataEx");
        const auto module_unload =
            sym<ModuleUnload>(handle_, "cuModuleUnload");
        const auto module_get_function =
            sym<ModuleGetFunction>(handle_, "cuModuleGetFunction");
        const auto launch =
            sym<LaunchKernel>(handle_, "cuLaunchKernel");
        const auto sync =
            sym<CtxSynchronize>(handle_, "cuCtxSynchronize");

        const CUdeviceptr base = memory.ptr();
        const CUdeviceptr qkv_ptr = base + qkv_off;
        const CUdeviceptr conv_w_ptr = base + conv_w_off;
        const CUdeviceptr state_in_ptr = base + state_in_off;
        const CUdeviceptr conv_out_ptr = base + conv_out_off;
        const CUdeviceptr state_out_ptr = base + state_out_off;
        const CUdeviceptr q_out_ptr = base + q_out_off;
        const CUdeviceptr k_out_ptr = base + k_out_off;
        const CUdeviceptr beta_raw_ptr = base + beta_raw_off;
        const CUdeviceptr alpha_raw_ptr = base + alpha_raw_off;
        const CUdeviceptr dt_ptr = base + dt_off;
        const CUdeviceptr a_ptr = base + a_off;
        const CUdeviceptr beta_out_ptr = base + beta_out_off;
        const CUdeviceptr gate_out_ptr = base + gate_out_off;

        check(handle_, memcpy_htod(qkv_ptr, qkv.data(), qkv_bytes),
              "cuMemcpyHtoD(recurrent qkv)");
        check(handle_, memcpy_htod(conv_w_ptr, conv_weight, conv_w_bytes),
              "cuMemcpyHtoD(recurrent conv weight)");
        check(handle_, memcpy_htod(state_in_ptr, state_in.data(), state_bytes),
              "cuMemcpyHtoD(recurrent conv state)");
        check(handle_, memcpy_htod(beta_raw_ptr, beta_raw.data(), small_bytes),
              "cuMemcpyHtoD(recurrent beta raw)");
        check(handle_, memcpy_htod(alpha_raw_ptr, alpha_raw.data(), small_bytes),
              "cuMemcpyHtoD(recurrent alpha raw)");
        check(handle_, memcpy_htod(dt_ptr, dt_bias, small_bytes),
              "cuMemcpyHtoD(recurrent dt)");
        check(handle_, memcpy_htod(a_ptr, ssm_a, small_bytes),
              "cuMemcpyHtoD(recurrent a)");

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
            reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(jit_info.size())),
            jit_error.data(),
            reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(jit_error.size())),
            reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(1)),
        };

        CUmodule module{};
        const auto rc = module_load_ex(
            &module,
            kRecurrentPrepPtx,
            static_cast<unsigned int>(
                sizeof(jit_options) / sizeof(jit_options[0])),
            jit_options,
            jit_values);
        if (rc != CUDA_SUCCESS) {
            std::string detail =
                cuda_error(handle_, rc, "cuModuleLoadDataEx(recurrent prep)");
            if (jit_error[0] != '\0') {
                detail +=
                    std::string("\nPTX JIT error log:\n") +
                    jit_error.data();
            }
            if (jit_info[0] != '\0') {
                detail +=
                    std::string("\nPTX JIT info log:\n") +
                    jit_info.data();
            }
            throw std::runtime_error(detail);
        }

        try {
            CUfunction conv_fn{}, qk_fn{}, bg_fn{};
            check(handle_, module_get_function(
                &conv_fn, module, "q38_conv4_silu_roll"),
                "cuModuleGetFunction(q38_conv4_silu_roll)");
            check(handle_, module_get_function(
                &qk_fn, module, "q38_qk_l2norm_128"),
                "cuModuleGetFunction(q38_qk_l2norm_128)");
            check(handle_, module_get_function(
                &bg_fn, module, "q38_beta_gate_48"),
                "cuModuleGetFunction(q38_beta_gate_48)");

            CUdeviceptr c_qkv = qkv_ptr;
            CUdeviceptr c_w = conv_w_ptr;
            CUdeviceptr c_si = state_in_ptr;
            CUdeviceptr c_co = conv_out_ptr;
            CUdeviceptr c_so = state_out_ptr;
            std::uint32_t c_channels = kChannels;
            float log2e = 1.4426950408889634f;
            void* conv_params[] = {
                &c_qkv, &c_w, &c_si, &c_co, &c_so, &c_channels, &log2e
            };

            CUdeviceptr n_conv = conv_out_ptr;
            CUdeviceptr n_q = q_out_ptr;
            CUdeviceptr n_k = k_out_ptr;
            float norm_eps = kNormEps;
            void* qk_params[] = {
                &n_conv, &n_q, &n_k, &norm_eps
            };

            CUdeviceptr b_br = beta_raw_ptr;
            CUdeviceptr b_ar = alpha_raw_ptr;
            CUdeviceptr b_dt = dt_ptr;
            CUdeviceptr b_a = a_ptr;
            CUdeviceptr b_bo = beta_out_ptr;
            CUdeviceptr b_go = gate_out_ptr;
            float inv_log2e = 0.6931471805599453f;
            void* bg_params[] = {
                &b_br, &b_ar, &b_dt, &b_a, &b_bo, &b_go,
                &log2e, &inv_log2e
            };

            const unsigned int conv_grid =
                (kChannels + 255u) / 256u;

            auto launch_conv = [&]() {
                check(handle_, launch(
                    conv_fn, conv_grid, 1, 1,
                    256, 1, 1,
                    0, nullptr, conv_params, nullptr),
                    "cuLaunchKernel(recurrent conv+silu)");
            };
            auto launch_qk = [&]() {
                check(handle_, launch(
                    qk_fn, 32, 1, 1,
                    128, 1, 1,
                    0, nullptr, qk_params, nullptr),
                    "cuLaunchKernel(recurrent qk norm)");
            };
            auto launch_bg = [&]() {
                check(handle_, launch(
                    bg_fn, 1, 1, 1,
                    64, 1, 1,
                    0, nullptr, bg_params, nullptr),
                    "cuLaunchKernel(recurrent beta/gate)");
            };

            launch_conv();
            launch_qk();
            launch_bg();
            check(handle_, sync(),
                  "cuCtxSynchronize(recurrent prep correctness)");

            std::vector<float> conv_gpu(kChannels);
            std::vector<float> q_gpu(kKeyDim);
            std::vector<float> k_gpu(kKeyDim);
            std::vector<float> beta_gpu(kValueHeads);
            std::vector<float> gate_gpu(kValueHeads);
            std::vector<float> state_gpu(state_in.size());

            check(handle_, memcpy_dtoh(
                conv_gpu.data(), conv_out_ptr, qkv_bytes),
                "cuMemcpyDtoH(recurrent conv)");
            check(handle_, memcpy_dtoh(
                q_gpu.data(), q_out_ptr, qk_bytes),
                "cuMemcpyDtoH(recurrent q)");
            check(handle_, memcpy_dtoh(
                k_gpu.data(), k_out_ptr, qk_bytes),
                "cuMemcpyDtoH(recurrent k)");
            check(handle_, memcpy_dtoh(
                beta_gpu.data(), beta_out_ptr, small_bytes),
                "cuMemcpyDtoH(recurrent beta)");
            check(handle_, memcpy_dtoh(
                gate_gpu.data(), gate_out_ptr, small_bytes),
                "cuMemcpyDtoH(recurrent gate)");
            check(handle_, memcpy_dtoh(
                state_gpu.data(), state_out_ptr, state_bytes),
                "cuMemcpyDtoH(recurrent conv state)");

            RecurrentPrepSmokeStats local{};
            auto max_abs = [](const std::vector<float>& a,
                              const std::vector<float>& b) {
                double m = 0.0;
                for (std::size_t i = 0; i < a.size(); ++i) {
                    m = std::max(
                        m,
                        std::abs(
                            static_cast<double>(a[i]) -
                            static_cast<double>(b[i])));
                }
                return m;
            };

            local.conv_max_abs = max_abs(conv_gpu, conv_ref);
            local.q_max_abs = max_abs(q_gpu, q_ref);
            local.k_max_abs = max_abs(k_gpu, k_ref);
            local.beta_max_abs = max_abs(beta_gpu, beta_ref);
            local.gate_max_abs = max_abs(gate_gpu, gate_ref);
            local.conv_state_max_abs = max_abs(state_gpu, state_ref);

            if (local.conv_max_abs > 5.0e-4 ||
                local.q_max_abs > 5.0e-4 ||
                local.k_max_abs > 5.0e-4 ||
                local.beta_max_abs > 5.0e-5 ||
                local.gate_max_abs > 5.0e-4 ||
                local.conv_state_max_abs > 1.0e-7) {
                std::ostringstream oss;
                oss << "recurrent prep mismatch: conv=" << local.conv_max_abs
                    << " q=" << local.q_max_abs
                    << " k=" << local.k_max_abs
                    << " beta=" << local.beta_max_abs
                    << " gate=" << local.gate_max_abs
                    << " state=" << local.conv_state_max_abs;
                throw std::runtime_error(oss.str());
            }

            auto bench = [&](int iters, auto&& fn, const char* label) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < iters; ++i) fn();
                check(handle_, sync(), label);
                const auto t1 = std::chrono::steady_clock::now();
                return std::chrono::duration<double, std::milli>(
                    t1 - t0).count() /
                    static_cast<double>(iters);
            };

            local.conv_silu_ms =
                bench(200, launch_conv,
                      "cuCtxSynchronize(recurrent conv benchmark)");
            local.qk_norm_ms =
                bench(500, launch_qk,
                      "cuCtxSynchronize(recurrent qk benchmark)");
            local.beta_gate_ms =
                bench(1000, launch_bg,
                      "cuCtxSynchronize(recurrent beta/gate benchmark)");

            auto chain = [&]() {
                launch_conv();
                launch_qk();
                launch_bg();
            };
            local.chain_ms =
                bench(200, chain,
                      "cuCtxSynchronize(recurrent prep chain benchmark)");

            if (stats) *stats = local;
        } catch (...) {
            module_unload(module);
            throw;
        }

        check(handle_, module_unload(module),
              "cuModuleUnload(recurrent prep)");
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool NvidiaDriver::run_gdn_ar_smoke(
    std::uint32_t state_dim,
    std::uint32_t qk_heads,
    std::uint32_t value_heads,
    GdnArSmokeStats* stats,
    std::string* error) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error(
                "q38 GDN AR smoke requires sm_86; detected sm_" +
                std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (state_dim != 128) {
            throw std::invalid_argument("current GDN AR PTX is specialized for state_dim=128");
        }
        if (qk_heads == 0 || value_heads == 0 || value_heads % qk_heads != 0) {
            throw std::invalid_argument("GDN AR requires non-zero heads and value_heads divisible by qk_heads");
        }

        constexpr float kEps = 1.0e-6f;
        const std::size_t qk_values =
            static_cast<std::size_t>(qk_heads) * state_dim;
        const std::size_t v_values =
            static_cast<std::size_t>(value_heads) * state_dim;
        const std::size_t state_values =
            static_cast<std::size_t>(value_heads) *
            state_dim * state_dim;

        std::vector<float> q(qk_values);
        std::vector<float> k(qk_values);
        std::vector<float> v(v_values);
        std::vector<float> g(value_heads);
        std::vector<float> beta(value_heads);
        std::vector<float> state_in(state_values);
        std::vector<float> out_gpu(v_values);
        std::vector<float> state_gpu(state_values);
        std::vector<float> out_ref(v_values);
        std::vector<float> state_ref(state_values);

        for (std::uint32_t h = 0; h < qk_heads; ++h) {
            double q_ss = 0.0;
            double k_ss = 0.0;
            for (std::uint32_t i = 0; i < state_dim; ++i) {
                const float fi = static_cast<float>(i);
                const float fh = static_cast<float>(h);
                const float qv =
                    std::sin(fi * 0.037f + fh * 0.11f) * 0.8f +
                    std::cos(fi * 0.013f - fh * 0.07f) * 0.2f;
                const float kv =
                    std::cos(fi * 0.029f + fh * 0.09f) * 0.75f -
                    std::sin(fi * 0.017f + fh * 0.05f) * 0.25f;
                q[static_cast<std::size_t>(h) * state_dim + i] = qv;
                k[static_cast<std::size_t>(h) * state_dim + i] = kv;
                q_ss += static_cast<double>(qv) * qv;
                k_ss += static_cast<double>(kv) * kv;
            }
            const double q_inv = 1.0 / std::sqrt(q_ss + kEps);
            const double k_inv = 1.0 / std::sqrt(k_ss + kEps);
            for (std::uint32_t i = 0; i < state_dim; ++i) {
                q[static_cast<std::size_t>(h) * state_dim + i] =
                    static_cast<float>(
                        static_cast<double>(
                            q[static_cast<std::size_t>(h) * state_dim + i]) *
                        q_inv);
                k[static_cast<std::size_t>(h) * state_dim + i] =
                    static_cast<float>(
                        static_cast<double>(
                            k[static_cast<std::size_t>(h) * state_dim + i]) *
                        k_inv);
            }
        }

        for (std::uint32_t h = 0; h < value_heads; ++h) {
            g[h] = -0.025f - 0.0015f * static_cast<float>(h % 19);
            beta[h] = 0.20f + 0.012f * static_cast<float>(h % 37);
            for (std::uint32_t col = 0; col < state_dim; ++col) {
                const std::size_t out_idx =
                    static_cast<std::size_t>(h) * state_dim + col;
                const float fidx = static_cast<float>(out_idx);
                v[out_idx] =
                    std::sin(fidx * 0.009f) * 0.7f +
                    std::cos(fidx * 0.004f) * 0.3f;

                const std::size_t state_base =
                    (static_cast<std::size_t>(h) * state_dim + col) *
                    state_dim;
                for (std::uint32_t i = 0; i < state_dim; ++i) {
                    const float fs =
                        static_cast<float>(state_base + i);
                    state_in[state_base + i] =
                        0.012f * std::sin(fs * 0.0013f) +
                        0.006f * std::cos(fs * 0.0007f);
                }
            }
        }

        const double scale =
            1.0 / std::sqrt(static_cast<double>(state_dim));

        for (std::uint32_t h = 0; h < value_heads; ++h) {
            const std::uint32_t qh = h % qk_heads;
            const float* qh_ptr =
                q.data() + static_cast<std::size_t>(qh) * state_dim;
            const float* kh_ptr =
                k.data() + static_cast<std::size_t>(qh) * state_dim;
            const double g_val = std::exp(static_cast<double>(g[h]));
            const double beta_val = static_cast<double>(beta[h]);

            for (std::uint32_t col = 0; col < state_dim; ++col) {
                const std::size_t state_base =
                    (static_cast<std::size_t>(h) * state_dim + col) *
                    state_dim;

                double kv = 0.0;
                for (std::uint32_t i = 0; i < state_dim; ++i) {
                    kv +=
                        static_cast<double>(state_in[state_base + i]) *
                        static_cast<double>(kh_ptr[i]);
                }

                const double delta =
                    (static_cast<double>(
                        v[static_cast<std::size_t>(h) * state_dim + col]) -
                     g_val * kv) *
                    beta_val;

                double attn = 0.0;
                for (std::uint32_t i = 0; i < state_dim; ++i) {
                    const double updated =
                        g_val *
                            static_cast<double>(state_in[state_base + i]) +
                        static_cast<double>(kh_ptr[i]) * delta;
                    state_ref[state_base + i] =
                        static_cast<float>(updated);
                    attn +=
                        updated * static_cast<double>(qh_ptr[i]);
                }

                out_ref[
                    static_cast<std::size_t>(h) * state_dim + col] =
                    static_cast<float>(attn * scale);
            }
        }

        const std::size_t q_bytes = q.size() * sizeof(float);
        const std::size_t k_bytes = k.size() * sizeof(float);
        const std::size_t v_bytes = v.size() * sizeof(float);
        const std::size_t g_bytes = g.size() * sizeof(float);
        const std::size_t beta_bytes = beta.size() * sizeof(float);
        const std::size_t state_bytes = state_in.size() * sizeof(float);
        const std::size_t out_bytes = out_gpu.size() * sizeof(float);

        auto align256 = [](std::size_t n) {
            return (n + 255u) & ~std::size_t(255u);
        };

        const std::size_t q_off = 0;
        const std::size_t k_off = align256(q_off + q_bytes);
        const std::size_t v_off = align256(k_off + k_bytes);
        const std::size_t g_off = align256(v_off + v_bytes);
        const std::size_t beta_off = align256(g_off + g_bytes);
        const std::size_t state_in_off = align256(beta_off + beta_bytes);
        const std::size_t out_off = align256(state_in_off + state_bytes);
        const std::size_t state_out_off = align256(out_off + out_bytes);
        const std::size_t total_bytes = state_out_off + state_bytes;

        ScopedVmm memory(handle_, device_ordinal_, total_bytes);

        using MemcpyHtoD =
            CUresult(*)(CUdeviceptr, const void*, std::size_t);
        using MemcpyDtoH =
            CUresult(*)(void*, CUdeviceptr, std::size_t);
        using ModuleLoadDataEx =
            CUresult(*)(CUmodule*, const void*, unsigned int, int*, void**);
        using ModuleUnload = CUresult(*)(CUmodule);
        using ModuleGetFunction =
            CUresult(*)(CUfunction*, CUmodule, const char*);
        using LaunchKernel =
            CUresult(*)(CUfunction,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, CUstream, void**, void**);
        using CtxSynchronize = CUresult(*)();

        const auto memcpy_htod =
            sym<MemcpyHtoD>(handle_, "cuMemcpyHtoD_v2");
        const auto memcpy_dtoh =
            sym<MemcpyDtoH>(handle_, "cuMemcpyDtoH_v2");
        const auto module_load_ex =
            sym<ModuleLoadDataEx>(handle_, "cuModuleLoadDataEx");
        const auto module_unload =
            sym<ModuleUnload>(handle_, "cuModuleUnload");
        const auto module_get_function =
            sym<ModuleGetFunction>(handle_, "cuModuleGetFunction");
        const auto launch =
            sym<LaunchKernel>(handle_, "cuLaunchKernel");
        const auto sync =
            sym<CtxSynchronize>(handle_, "cuCtxSynchronize");

        const CUdeviceptr base = memory.ptr();
        const CUdeviceptr q_ptr = base + q_off;
        const CUdeviceptr k_ptr = base + k_off;
        const CUdeviceptr v_ptr = base + v_off;
        const CUdeviceptr g_ptr = base + g_off;
        const CUdeviceptr beta_ptr = base + beta_off;
        const CUdeviceptr state_in_ptr = base + state_in_off;
        const CUdeviceptr out_ptr = base + out_off;
        const CUdeviceptr state_out_ptr = base + state_out_off;

        check(handle_, memcpy_htod(q_ptr, q.data(), q_bytes),
              "cuMemcpyHtoD(GDN q)");
        check(handle_, memcpy_htod(k_ptr, k.data(), k_bytes),
              "cuMemcpyHtoD(GDN k)");
        check(handle_, memcpy_htod(v_ptr, v.data(), v_bytes),
              "cuMemcpyHtoD(GDN v)");
        check(handle_, memcpy_htod(g_ptr, g.data(), g_bytes),
              "cuMemcpyHtoD(GDN g)");
        check(handle_, memcpy_htod(beta_ptr, beta.data(), beta_bytes),
              "cuMemcpyHtoD(GDN beta)");
        check(handle_, memcpy_htod(
            state_in_ptr, state_in.data(), state_bytes),
            "cuMemcpyHtoD(GDN state)");

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
            reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(jit_info.size())),
            jit_error.data(),
            reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(jit_error.size())),
            reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(1)),
        };

        CUmodule module{};
        const auto rc = module_load_ex(
            &module,
            kGdnAr128Ptx,
            static_cast<unsigned int>(
                sizeof(jit_options) / sizeof(jit_options[0])),
            jit_options,
            jit_values);
        if (rc != CUDA_SUCCESS) {
            std::string detail =
                cuda_error(handle_, rc, "cuModuleLoadDataEx(GDN AR)");
            if (jit_error[0] != '\0') {
                detail +=
                    std::string("\nPTX JIT error log:\n") +
                    jit_error.data();
            }
            if (jit_info[0] != '\0') {
                detail +=
                    std::string("\nPTX JIT info log:\n") +
                    jit_info.data();
            }
            throw std::runtime_error(detail);
        }

        try {
            CUfunction fn{};
            check(handle_, module_get_function(
                &fn, module, "q38_gdn_ar_128"),
                "cuModuleGetFunction(q38_gdn_ar_128)");

            CUdeviceptr arg_q = q_ptr;
            CUdeviceptr arg_k = k_ptr;
            CUdeviceptr arg_v = v_ptr;
            CUdeviceptr arg_g = g_ptr;
            CUdeviceptr arg_beta = beta_ptr;
            CUdeviceptr arg_state_in = state_in_ptr;
            CUdeviceptr arg_out = out_ptr;
            CUdeviceptr arg_state_out = state_out_ptr;
            std::uint32_t arg_qk_heads = qk_heads;
            std::uint32_t arg_value_heads = value_heads;
            float arg_scale =
                1.0f / std::sqrt(static_cast<float>(state_dim));
            float arg_log2e = 1.4426950408889634f;

            void* params[] = {
                &arg_q,
                &arg_k,
                &arg_v,
                &arg_g,
                &arg_beta,
                &arg_state_in,
                &arg_out,
                &arg_state_out,
                &arg_qk_heads,
                &arg_value_heads,
                &arg_scale,
                &arg_log2e,
            };

            const unsigned int grid_y =
                (state_dim + 3u) / 4u;

            check(handle_, launch(
                fn,
                value_heads, grid_y, 1,
                128, 1, 1,
                0, nullptr, params, nullptr),
                "cuLaunchKernel(q38_gdn_ar_128)");
            check(handle_, sync(), "cuCtxSynchronize(GDN AR correctness)");

            check(handle_, memcpy_dtoh(
                out_gpu.data(), out_ptr, out_bytes),
                "cuMemcpyDtoH(GDN output)");
            check(handle_, memcpy_dtoh(
                state_gpu.data(), state_out_ptr, state_bytes),
                "cuMemcpyDtoH(GDN state)");

            GdnArSmokeStats local{};
            bool bad = false;

            for (std::size_t i = 0; i < out_gpu.size(); ++i) {
                const double got =
                    static_cast<double>(out_gpu[i]);
                const double ref =
                    static_cast<double>(out_ref[i]);
                const double abs_err = std::abs(got - ref);
                const double rel_err =
                    abs_err / std::max(1.0e-6, std::abs(ref));
                local.output_max_abs =
                    std::max(local.output_max_abs, abs_err);
                local.output_max_rel =
                    std::max(local.output_max_rel, rel_err);
                if (abs_err > 5.0e-4 &&
                    rel_err > 5.0e-4) {
                    bad = true;
                }
            }

            for (std::size_t i = 0; i < state_gpu.size(); ++i) {
                const double got =
                    static_cast<double>(state_gpu[i]);
                const double ref =
                    static_cast<double>(state_ref[i]);
                const double abs_err = std::abs(got - ref);
                const double rel_err =
                    abs_err / std::max(1.0e-6, std::abs(ref));
                local.state_max_abs =
                    std::max(local.state_max_abs, abs_err);
                local.state_max_rel =
                    std::max(local.state_max_rel, rel_err);
                if (abs_err > 2.0e-4 &&
                    rel_err > 2.0e-4) {
                    bad = true;
                }
            }

            if (bad) {
                std::ostringstream oss;
                oss << "GDN AR mismatch: output_abs="
                    << local.output_max_abs
                    << " output_rel=" << local.output_max_rel
                    << " state_abs=" << local.state_max_abs
                    << " state_rel=" << local.state_max_rel;
                throw std::runtime_error(oss.str());
            }

            constexpr int kIters = 100;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) {
                check(handle_, launch(
                    fn,
                    value_heads, grid_y, 1,
                    128, 1, 1,
                    0, nullptr, params, nullptr),
                    "cuLaunchKernel(q38_gdn_ar_128 benchmark)");
            }
            check(handle_, sync(),
                  "cuCtxSynchronize(GDN AR benchmark)");
            const auto t1 = std::chrono::steady_clock::now();

            local.kernel_ms =
                std::chrono::duration<double, std::milli>(
                    t1 - t0).count() /
                static_cast<double>(kIters);
            local.state_bandwidth_gbps =
                static_cast<double>(state_bytes * 2) /
                (local.kernel_ms / 1000.0) / 1.0e9;

            if (stats) *stats = local;
        } catch (...) {
            module_unload(module);
            throw;
        }

        check(handle_, module_unload(module),
              "cuModuleUnload(GDN AR)");
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool NvidiaDriver::run_qwen35_layer0_projection_pack(
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
    Layer0ProjectionPackStats* stats,
    std::string* error) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error(
                "q38 layer0 projection pack requires sm_86; detected sm_" +
                std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (!norm_weight || !qkv_matrix || !gate_matrix ||
            !beta_matrix || !alpha_matrix) {
            throw std::invalid_argument("layer0 projection pack received null tensor");
        }
        if (cols == 0 || qkv_rows == 0 || gate_rows == 0 ||
            small_rows == 0 || (cols % kQ4KValuesPerBlock) != 0) {
            throw std::invalid_argument("invalid layer0 projection pack dimensions");
        }

        const std::size_t q5_blocks_per_row =
            cols / kQ4KValuesPerBlock;
        const std::size_t qkv_blocks =
            static_cast<std::size_t>(qkv_rows) * q5_blocks_per_row;
        const std::size_t gate_blocks =
            static_cast<std::size_t>(gate_rows) * q5_blocks_per_row;

        const std::size_t qkv_meta_bytes = qkv_blocks * 20;
        const std::size_t qkv_qh_bytes = qkv_blocks * 32;
        const std::size_t qkv_qs_bytes = qkv_blocks * 128;
        if (qkv_qh_offset < qkv_meta_bytes ||
            qkv_qs_offset < qkv_qh_offset + qkv_qh_bytes) {
            throw std::invalid_argument("invalid qkv SM86 Q5_K offsets");
        }
        const std::size_t qkv_bytes =
            qkv_qs_offset + qkv_qs_bytes;

        const std::size_t gate_meta_bytes = gate_blocks * 20;
        const std::size_t gate_qh_bytes = gate_blocks * 32;
        const std::size_t gate_qs_bytes = gate_blocks * 128;
        if (gate_qh_offset < gate_meta_bytes ||
            gate_qs_offset < gate_qh_offset + gate_qh_bytes) {
            throw std::invalid_argument("invalid gate SM86 Q5_K offsets");
        }
        const std::size_t gate_bytes =
            gate_qs_offset + gate_qs_bytes;

        const std::size_t q4_blocks_per_row =
            cols / kQ4KValuesPerBlock;
        const std::size_t small_matrix_bytes =
            static_cast<std::size_t>(small_rows) *
            q4_blocks_per_row * kQ4KBytesPerBlock;

        const std::size_t vec_bytes =
            static_cast<std::size_t>(cols) * sizeof(float);
        const std::size_t qkv_out_bytes =
            static_cast<std::size_t>(qkv_rows) * sizeof(float);
        const std::size_t gate_out_bytes =
            static_cast<std::size_t>(gate_rows) * sizeof(float);
        const std::size_t small_out_bytes =
            static_cast<std::size_t>(small_rows) * sizeof(float);

        auto align256 = [](std::size_t v) {
            return (v + 255u) & ~std::size_t(255u);
        };

        const std::size_t qkv_off = 0;
        const std::size_t gate_off = align256(qkv_off + qkv_bytes);
        const std::size_t beta_off = align256(gate_off + gate_bytes);
        const std::size_t alpha_off = align256(beta_off + small_matrix_bytes);
        const std::size_t x_off = align256(alpha_off + small_matrix_bytes);
        const std::size_t norm_w_off = align256(x_off + vec_bytes);
        const std::size_t normed_off = align256(norm_w_off + vec_bytes);
        const std::size_t sumsq_off = align256(normed_off + vec_bytes);
        const std::size_t qkv_out_off = align256(sumsq_off + sizeof(float));
        const std::size_t gate_out_off = align256(qkv_out_off + qkv_out_bytes);
        const std::size_t beta_out_off = align256(gate_out_off + gate_out_bytes);
        const std::size_t alpha_out_off = align256(beta_out_off + small_out_bytes);
        const std::size_t total_bytes =
            alpha_out_off + small_out_bytes;

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

        std::vector<float> x(cols), normed_cpu(cols);
        for (std::uint32_t i = 0; i < cols; ++i) {
            const float fi = static_cast<float>(i);
            x[i] =
                std::sin(fi * 0.017f) * 0.65f +
                std::cos(fi * 0.011f) * 0.35f;
        }

        double sumsq_cpu = 0.0;
        for (float v : x) {
            sumsq_cpu += static_cast<double>(v) * static_cast<double>(v);
        }
        const double inv_rms =
            1.0 / std::sqrt(
                sumsq_cpu / static_cast<double>(cols) +
                static_cast<double>(rms_eps));
        for (std::uint32_t i = 0; i < cols; ++i) {
            normed_cpu[i] = static_cast<float>(
                static_cast<double>(x[i]) * inv_rms *
                static_cast<double>(norm_weight[i]));
        }

        const CUdeviceptr base = memory.ptr();
        const CUdeviceptr qkv_ptr = base + qkv_off;
        const CUdeviceptr gate_ptr = base + gate_off;
        const CUdeviceptr beta_ptr = base + beta_off;
        const CUdeviceptr alpha_ptr = base + alpha_off;
        const CUdeviceptr x_ptr = base + x_off;
        const CUdeviceptr norm_w_ptr = base + norm_w_off;
        const CUdeviceptr normed_ptr = base + normed_off;
        const CUdeviceptr sumsq_ptr = base + sumsq_off;
        const CUdeviceptr qkv_out_ptr = base + qkv_out_off;
        const CUdeviceptr gate_out_ptr = base + gate_out_off;
        const CUdeviceptr beta_out_ptr = base + beta_out_off;
        const CUdeviceptr alpha_out_ptr = base + alpha_out_off;

        check(handle_, memcpy_htod(qkv_ptr, qkv_matrix, qkv_bytes),
              "cuMemcpyHtoD(layer0 qkv)");
        check(handle_, memcpy_htod(gate_ptr, gate_matrix, gate_bytes),
              "cuMemcpyHtoD(layer0 gate)");
        check(handle_, memcpy_htod(beta_ptr, beta_matrix, small_matrix_bytes),
              "cuMemcpyHtoD(layer0 beta)");
        check(handle_, memcpy_htod(alpha_ptr, alpha_matrix, small_matrix_bytes),
              "cuMemcpyHtoD(layer0 alpha)");
        check(handle_, memcpy_htod(x_ptr, x.data(), vec_bytes),
              "cuMemcpyHtoD(layer0 projection input)");
        check(handle_, memcpy_htod(norm_w_ptr, norm_weight, vec_bytes),
              "cuMemcpyHtoD(layer0 attn_norm)");

        constexpr int CU_JIT_INFO_LOG_BUFFER = 3;
        constexpr int CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4;
        constexpr int CU_JIT_ERROR_LOG_BUFFER = 5;
        constexpr int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;
        constexpr int CU_JIT_LOG_VERBOSE = 12;

        auto load_module = [&](const char* ptx, const char* label) {
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
                reinterpret_cast<void*>(
                    static_cast<std::uintptr_t>(jit_info.size())),
                jit_error.data(),
                reinterpret_cast<void*>(
                    static_cast<std::uintptr_t>(jit_error.size())),
                reinterpret_cast<void*>(
                    static_cast<std::uintptr_t>(1)),
            };

            CUmodule module{};
            const auto rc = module_load_ex(
                &module,
                ptx,
                static_cast<unsigned int>(
                    sizeof(jit_options) / sizeof(jit_options[0])),
                jit_options,
                jit_values);

            if (rc != CUDA_SUCCESS) {
                std::string detail = cuda_error(handle_, rc, label);
                if (jit_error[0] != '\0') {
                    detail +=
                        std::string("\nPTX JIT error log:\n") +
                        jit_error.data();
                }
                if (jit_info[0] != '\0') {
                    detail +=
                        std::string("\nPTX JIT info log:\n") +
                        jit_info.data();
                }
                throw std::runtime_error(detail);
            }
            return module;
        };

        CUmodule norm_module =
            load_module(kRmsNormPtx, "cuModuleLoadDataEx(layer0 pack RMSNorm)");
        CUmodule q5_module{};
        CUmodule q4_module{};

        try {
            q5_module = load_module(
                kQ5KSm86SoAGemvVecPtx,
                "cuModuleLoadDataEx(layer0 pack Q5)");
            q4_module = load_module(
                kQ4KGemvPtx,
                "cuModuleLoadDataEx(layer0 pack Q4)");

            CUfunction sum_fn{}, norm_fn{}, q5_fn{}, q4_fn{};
            check(handle_, module_get_function(
                &sum_fn, norm_module, "q38_sumsq"),
                "cuModuleGetFunction(layer0 pack sumsq)");
            check(handle_, module_get_function(
                &norm_fn, norm_module, "q38_rmsnorm_apply"),
                "cuModuleGetFunction(layer0 pack rmsnorm)");
            check(handle_, module_get_function(
                &q5_fn, q5_module, "q38_q5k_sm86_soa_gemv_vec"),
                "cuModuleGetFunction(layer0 pack q5)");
            check(handle_, module_get_function(
                &q4_fn, q4_module, "q38_q4k_gemv_f32"),
                "cuModuleGetFunction(layer0 pack q4)");

            constexpr unsigned int norm_block = 256;
            const unsigned int norm_grid =
                (cols + norm_block - 1) / norm_block;

            CUdeviceptr sum_x = x_ptr;
            CUdeviceptr sum_out = sumsq_ptr;
            std::uint32_t count = cols;
            void* sum_params[] = {
                &sum_x, &sum_out, &count
            };

            CUdeviceptr norm_x = x_ptr;
            CUdeviceptr norm_w = norm_w_ptr;
            CUdeviceptr norm_y = normed_ptr;
            CUdeviceptr norm_sumsq = sumsq_ptr;
            float kernel_eps = rms_eps;
            void* norm_params[] = {
                &norm_x, &norm_w, &norm_y,
                &norm_sumsq, &count, &kernel_eps
            };

            // Keep kernel arguments as stable local variables because the
            // Driver API launch parameter array stores pointers to them.
            CUdeviceptr qkv_meta = qkv_ptr;
            CUdeviceptr qkv_qh = qkv_ptr + qkv_qh_offset;
            CUdeviceptr qkv_qs = qkv_ptr + qkv_qs_offset;
            CUdeviceptr qkv_x = normed_ptr;
            CUdeviceptr qkv_y = qkv_out_ptr;
            std::uint32_t qkv_cols_arg = cols;
            std::uint32_t qkv_rows_arg = qkv_rows;
            void* qkv_params[] = {
                &qkv_meta, &qkv_qh, &qkv_qs, &qkv_x, &qkv_y,
                &qkv_cols_arg, &qkv_rows_arg
            };

            CUdeviceptr gate_meta = gate_ptr;
            CUdeviceptr gate_qh = gate_ptr + gate_qh_offset;
            CUdeviceptr gate_qs = gate_ptr + gate_qs_offset;
            CUdeviceptr gate_x = normed_ptr;
            CUdeviceptr gate_y = gate_out_ptr;
            std::uint32_t gate_cols_arg = cols;
            std::uint32_t gate_rows_arg = gate_rows;
            void* gate_params[] = {
                &gate_meta, &gate_qh, &gate_qs, &gate_x, &gate_y,
                &gate_cols_arg, &gate_rows_arg
            };

            CUdeviceptr beta_w = beta_ptr;
            CUdeviceptr beta_x = normed_ptr;
            CUdeviceptr beta_y = beta_out_ptr;
            std::uint32_t beta_cols = cols;
            std::uint32_t beta_rows = small_rows;
            void* beta_params[] = {
                &beta_w, &beta_x, &beta_y,
                &beta_cols, &beta_rows
            };

            CUdeviceptr alpha_w = alpha_ptr;
            CUdeviceptr alpha_x = normed_ptr;
            CUdeviceptr alpha_y = alpha_out_ptr;
            std::uint32_t alpha_cols = cols;
            std::uint32_t alpha_rows = small_rows;
            void* alpha_params[] = {
                &alpha_w, &alpha_x, &alpha_y,
                &alpha_cols, &alpha_rows
            };

            const unsigned int qkv_grid =
                (qkv_rows + 3u) / 4u;
            const unsigned int gate_grid =
                (gate_rows + 3u) / 4u;

            auto launch_norm = [&]() {
                check(handle_, memset_d32(sumsq_ptr, 0, 1),
                      "cuMemsetD32(layer0 pack sumsq)");
                check(handle_, launch(
                    sum_fn, norm_grid, 1, 1,
                    norm_block, 1, 1,
                    0, nullptr, sum_params, nullptr),
                    "cuLaunchKernel(layer0 pack sumsq)");
                check(handle_, launch(
                    norm_fn, norm_grid, 1, 1,
                    norm_block, 1, 1,
                    0, nullptr, norm_params, nullptr),
                    "cuLaunchKernel(layer0 pack rmsnorm)");
            };

            auto launch_qkv = [&]() {
                check(handle_, launch(
                    q5_fn, qkv_grid, 1, 1,
                    128, 1, 1,
                    0, nullptr, qkv_params, nullptr),
                    "cuLaunchKernel(layer0 pack qkv)");
            };

            auto launch_gate = [&]() {
                check(handle_, launch(
                    q5_fn, gate_grid, 1, 1,
                    128, 1, 1,
                    0, nullptr, gate_params, nullptr),
                    "cuLaunchKernel(layer0 pack gate)");
            };

            auto launch_beta = [&]() {
                check(handle_, launch(
                    q4_fn, small_rows, 1, 1,
                    32, 1, 1,
                    0, nullptr, beta_params, nullptr),
                    "cuLaunchKernel(layer0 pack beta)");
            };

            auto launch_alpha = [&]() {
                check(handle_, launch(
                    q4_fn, small_rows, 1, 1,
                    32, 1, 1,
                    0, nullptr, alpha_params, nullptr),
                    "cuLaunchKernel(layer0 pack alpha)");
            };

            launch_norm();
            launch_qkv();
            launch_gate();
            launch_beta();
            launch_alpha();
            check(handle_, sync(),
                  "cuCtxSynchronize(layer0 projection pack correctness)");

            std::vector<float> qkv_out(qkv_rows);
            std::vector<float> gate_out(gate_rows);
            std::vector<float> beta_out(small_rows);
            std::vector<float> alpha_out(small_rows);

            check(handle_, memcpy_dtoh(
                qkv_out.data(), qkv_out_ptr, qkv_out_bytes),
                "cuMemcpyDtoH(layer0 qkv)");
            check(handle_, memcpy_dtoh(
                gate_out.data(), gate_out_ptr, gate_out_bytes),
                "cuMemcpyDtoH(layer0 gate pack)");
            check(handle_, memcpy_dtoh(
                beta_out.data(), beta_out_ptr, small_out_bytes),
                "cuMemcpyDtoH(layer0 beta)");
            check(handle_, memcpy_dtoh(
                alpha_out.data(), alpha_out_ptr, small_out_bytes),
                "cuMemcpyDtoH(layer0 alpha)");

            auto validate_q5 = [&](const std::byte* matrix,
                                   std::size_t qh_offset,
                                   std::size_t qs_offset,
                                   std::uint32_t rows,
                                   const std::vector<float>& got,
                                   ProjectionErrorStats& out_stats) {
                const std::size_t checked =
                    std::min<std::size_t>(rows, 4);
                std::array<float, kQ4KValuesPerBlock> deq{};
                std::array<std::byte, kQ5KSm86BytesPerBlock> block{};
                double abs_max = 0.0;
                double rel_max = 0.0;
                bool bad = false;

                for (std::size_t row = 0; row < checked; ++row) {
                    double ref = 0.0;
                    for (std::size_t ib = 0;
                         ib < q5_blocks_per_row; ++ib) {
                        const std::size_t block_index =
                            row * q5_blocks_per_row + ib;
                        std::memcpy(
                            block.data() + 0,
                            matrix + block_index * 20,
                            20);
                        std::memcpy(
                            block.data() + 20,
                            matrix + qh_offset + block_index * 32,
                            32);
                        std::memcpy(
                            block.data() + 52,
                            matrix + qs_offset + block_index * 128,
                            128);
                        dequantize_q5_k_sm86_block_cpu(
                            block.data(), deq);
                        const std::size_t base_x =
                            ib * kQ4KValuesPerBlock;
                        for (std::size_t j = 0;
                             j < kQ4KValuesPerBlock; ++j) {
                            ref +=
                                static_cast<double>(deq[j]) *
                                static_cast<double>(
                                    normed_cpu[base_x + j]);
                        }
                    }
                    const double value =
                        static_cast<double>(got[row]);
                    const double abs_err =
                        std::abs(value - ref);
                    const double rel_err =
                        abs_err /
                        std::max(1.0e-5, std::abs(ref));
                    abs_max = std::max(abs_max, abs_err);
                    rel_max = std::max(rel_max, rel_err);
                    if (abs_err > 3.0e-3 &&
                        rel_err > 3.0e-3) {
                        bad = true;
                    }
                }

                out_stats.max_abs = abs_max;
                out_stats.max_rel = rel_max;
                if (bad) {
                    std::ostringstream oss;
                    oss << "layer0 Q5 projection mismatch: max_abs="
                        << abs_max << " max_rel=" << rel_max;
                    throw std::runtime_error(oss.str());
                }
            };

            auto validate_q4 = [&](const std::byte* matrix,
                                   const std::vector<float>& got,
                                   ProjectionErrorStats& out_stats) {
                const std::size_t checked =
                    std::min<std::size_t>(small_rows, 4);
                std::array<float, kQ4KValuesPerBlock> deq{};
                double abs_max = 0.0;
                double rel_max = 0.0;
                bool bad = false;

                for (std::size_t row = 0; row < checked; ++row) {
                    double ref = 0.0;
                    const auto* row_ptr =
                        matrix +
                        row * q4_blocks_per_row *
                            kQ4KBytesPerBlock;

                    for (std::size_t ib = 0;
                         ib < q4_blocks_per_row; ++ib) {
                        dequantize_q4_k_block_cpu(
                            row_ptr +
                                ib * kQ4KBytesPerBlock,
                            deq);
                        const std::size_t base_x =
                            ib * kQ4KValuesPerBlock;
                        for (std::size_t j = 0;
                             j < kQ4KValuesPerBlock; ++j) {
                            ref +=
                                static_cast<double>(deq[j]) *
                                static_cast<double>(
                                    normed_cpu[base_x + j]);
                        }
                    }

                    const double value =
                        static_cast<double>(got[row]);
                    const double abs_err =
                        std::abs(value - ref);
                    const double rel_err =
                        abs_err /
                        std::max(1.0e-5, std::abs(ref));

                    abs_max = std::max(abs_max, abs_err);
                    rel_max = std::max(rel_max, rel_err);
                    if (abs_err > 2.0e-3 &&
                        rel_err > 2.0e-3) {
                        bad = true;
                    }
                }

                out_stats.max_abs = abs_max;
                out_stats.max_rel = rel_max;
                if (bad) {
                    std::ostringstream oss;
                    oss << "layer0 Q4 projection mismatch: max_abs="
                        << abs_max << " max_rel=" << rel_max;
                    throw std::runtime_error(oss.str());
                }
            };

            Layer0ProjectionPackStats local_stats{};
            validate_q5(
                qkv_matrix, qkv_qh_offset, qkv_qs_offset,
                qkv_rows, qkv_out, local_stats.qkv);
            validate_q5(
                gate_matrix, gate_qh_offset, gate_qs_offset,
                gate_rows, gate_out, local_stats.gate);
            validate_q4(
                beta_matrix, beta_out, local_stats.beta);
            validate_q4(
                alpha_matrix, alpha_out, local_stats.alpha);

            auto bench = [&](int iters, auto&& fn, const char* label) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < iters; ++i) fn();
                check(handle_, sync(), label);
                const auto t1 = std::chrono::steady_clock::now();
                return std::chrono::duration<double, std::milli>(
                    t1 - t0).count() /
                    static_cast<double>(iters);
            };

            local_stats.rmsnorm_ms =
                bench(100, launch_norm,
                      "cuCtxSynchronize(layer0 pack RMSNorm benchmark)");
            local_stats.qkv_ms =
                bench(50, launch_qkv,
                      "cuCtxSynchronize(layer0 pack qkv benchmark)");
            local_stats.gate_ms =
                bench(50, launch_gate,
                      "cuCtxSynchronize(layer0 pack gate benchmark)");
            local_stats.beta_ms =
                bench(1000, launch_beta,
                      "cuCtxSynchronize(layer0 pack beta benchmark)");
            local_stats.alpha_ms =
                bench(1000, launch_alpha,
                      "cuCtxSynchronize(layer0 pack alpha benchmark)");

            auto launch_chain = [&]() {
                launch_norm();
                launch_qkv();
                launch_gate();
                launch_beta();
                launch_alpha();
            };
            local_stats.chain_ms =
                bench(50, launch_chain,
                      "cuCtxSynchronize(layer0 projection pack benchmark)");

            if (stats) *stats = local_stats;
        } catch (...) {
            if (q4_module) module_unload(q4_module);
            if (q5_module) module_unload(q5_module);
            module_unload(norm_module);
            throw;
        }

        check(handle_, module_unload(q4_module),
              "cuModuleUnload(layer0 pack Q4)");
        check(handle_, module_unload(q5_module),
              "cuModuleUnload(layer0 pack Q5)");
        check(handle_, module_unload(norm_module),
              "cuModuleUnload(layer0 pack RMSNorm)");
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool NvidiaDriver::run_qwen35_layer0_gate_smoke(
    const float* norm_weight,
    const std::byte* repacked_matrix,
    std::size_t qh_offset,
    std::size_t qs_offset,
    std::uint32_t cols,
    std::uint32_t rows,
    float rms_eps,
    std::string* error,
    double* max_abs_error,
    double* max_rel_error,
    double* rmsnorm_ms,
    double* projection_ms,
    double* chain_ms,
    double* projection_original_equiv_gbps) {
    try {
        if (!available()) throw std::runtime_error("driver is not initialized");
        if (!is_sm86()) {
            throw std::runtime_error("q38 layer0 gate smoke requires sm_86; detected sm_" +
                                     std::to_string(sm_major_) + std::to_string(sm_minor_));
        }
        if (!norm_weight || !repacked_matrix) {
            throw std::invalid_argument("layer0 gate smoke received null model tensor");
        }
        if (cols == 0 || rows == 0 || (cols % kQ4KValuesPerBlock) != 0) {
            throw std::invalid_argument("layer0 gate smoke requires cols divisible by 256");
        }

        const std::size_t blocks_per_row = cols / kQ4KValuesPerBlock;
        const std::size_t total_blocks = static_cast<std::size_t>(rows) * blocks_per_row;
        const std::size_t meta_bytes = total_blocks * 20;
        const std::size_t qh_bytes = total_blocks * 32;
        const std::size_t qs_bytes = total_blocks * 128;
        if (qh_offset < meta_bytes || qs_offset < qh_offset + qh_bytes) {
            throw std::invalid_argument("invalid persistent SM86 Q5_K plane offsets");
        }

        const std::size_t repacked_bytes = qs_offset + qs_bytes;
        const std::size_t original_bytes = total_blocks * kQ5KBytesPerBlock;
        const std::size_t x_bytes = static_cast<std::size_t>(cols) * sizeof(float);
        const std::size_t y_bytes = static_cast<std::size_t>(rows) * sizeof(float);

        auto align256 = [](std::size_t v) { return (v + 255u) & ~std::size_t(255u); };
        const std::size_t model_off = 0;
        const std::size_t x_off = align256(repacked_bytes);
        const std::size_t norm_w_off = align256(x_off + x_bytes);
        const std::size_t normed_off = align256(norm_w_off + x_bytes);
        const std::size_t sumsq_off = align256(normed_off + x_bytes);
        const std::size_t out_off = align256(sumsq_off + sizeof(float));
        const std::size_t total_bytes = out_off + y_bytes;

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

        std::vector<float> x(cols), normed_cpu(cols), out(rows);
        for (std::uint32_t i = 0; i < cols; ++i) {
            const float fi = static_cast<float>(i);
            x[i] = std::sin(fi * 0.017f) * 0.65f + std::cos(fi * 0.011f) * 0.35f;
        }

        double sumsq_cpu = 0.0;
        for (float v : x) sumsq_cpu += static_cast<double>(v) * static_cast<double>(v);
        const double inv_rms =
            1.0 / std::sqrt(sumsq_cpu / static_cast<double>(cols) + static_cast<double>(rms_eps));
        for (std::uint32_t i = 0; i < cols; ++i) {
            normed_cpu[i] = static_cast<float>(
                static_cast<double>(x[i]) * inv_rms * static_cast<double>(norm_weight[i]));
        }

        const CUdeviceptr model_ptr = memory.ptr() + model_off;
        const CUdeviceptr meta_ptr = model_ptr;
        const CUdeviceptr qh_ptr = model_ptr + qh_offset;
        const CUdeviceptr qs_ptr = model_ptr + qs_offset;
        const CUdeviceptr x_ptr = memory.ptr() + x_off;
        const CUdeviceptr norm_w_ptr = memory.ptr() + norm_w_off;
        const CUdeviceptr normed_ptr = memory.ptr() + normed_off;
        const CUdeviceptr sumsq_ptr = memory.ptr() + sumsq_off;
        const CUdeviceptr out_ptr = memory.ptr() + out_off;

        check(handle_, memcpy_htod(model_ptr, repacked_matrix, repacked_bytes),
              "cuMemcpyHtoD(layer0 gate weight)");
        check(handle_, memcpy_htod(x_ptr, x.data(), x_bytes), "cuMemcpyHtoD(layer0 input)");
        check(handle_, memcpy_htod(norm_w_ptr, norm_weight, x_bytes), "cuMemcpyHtoD(attn_norm weight)");

        constexpr int CU_JIT_INFO_LOG_BUFFER = 3;
        constexpr int CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4;
        constexpr int CU_JIT_ERROR_LOG_BUFFER = 5;
        constexpr int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;
        constexpr int CU_JIT_LOG_VERBOSE = 12;

        auto load_module = [&](const char* ptx, const char* label) {
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
            const auto rc = module_load_ex(
                &module, ptx,
                static_cast<unsigned int>(sizeof(jit_options) / sizeof(jit_options[0])),
                jit_options, jit_values);
            if (rc != CUDA_SUCCESS) {
                std::string detail = cuda_error(handle_, rc, label);
                if (jit_error[0] != '\0') detail += std::string("\nPTX JIT error log:\n") + jit_error.data();
                if (jit_info[0] != '\0') detail += std::string("\nPTX JIT info log:\n") + jit_info.data();
                throw std::runtime_error(detail);
            }
            return module;
        };

        CUmodule norm_module = load_module(kRmsNormPtx, "cuModuleLoadDataEx(layer0 RMSNorm)");
        CUmodule proj_module{};
        try {
            proj_module = load_module(kQ5KSm86SoAGemvVecPtx, "cuModuleLoadDataEx(layer0 gate projection)");

            CUfunction sum_fn{}, apply_fn{}, proj_fn{};
            check(handle_, module_get_function(&sum_fn, norm_module, "q38_sumsq"),
                  "cuModuleGetFunction(q38_sumsq)");
            check(handle_, module_get_function(&apply_fn, norm_module, "q38_rmsnorm_apply"),
                  "cuModuleGetFunction(q38_rmsnorm_apply)");
            check(handle_, module_get_function(&proj_fn, proj_module, "q38_q5k_sm86_soa_gemv_vec"),
                  "cuModuleGetFunction(q38_q5k_sm86_soa_gemv_vec)");

            constexpr unsigned int norm_block = 256;
            const unsigned int norm_grid = (cols + norm_block - 1) / norm_block;

            CUdeviceptr sx = x_ptr;
            CUdeviceptr ss = sumsq_ptr;
            std::uint32_t count = cols;
            void* sum_params[] = {&sx, &ss, &count};

            CUdeviceptr ax = x_ptr;
            CUdeviceptr aw = norm_w_ptr;
            CUdeviceptr ay = normed_ptr;
            CUdeviceptr as = sumsq_ptr;
            float kernel_eps = rms_eps;
            void* apply_params[] = {&ax, &aw, &ay, &as, &count, &kernel_eps};

            CUdeviceptr arg_meta = meta_ptr;
            CUdeviceptr arg_qh = qh_ptr;
            CUdeviceptr arg_qs = qs_ptr;
            CUdeviceptr arg_x = normed_ptr;
            CUdeviceptr arg_y = out_ptr;
            std::uint32_t arg_cols = cols;
            std::uint32_t arg_rows = rows;
            void* proj_params[] = {
                &arg_meta, &arg_qh, &arg_qs, &arg_x, &arg_y, &arg_cols, &arg_rows
            };

            auto launch_norm = [&]() {
                check(handle_, memset_d32(sumsq_ptr, 0, 1), "cuMemsetD32(layer0 sumsq)");
                check(handle_, launch(sum_fn, norm_grid, 1, 1, norm_block, 1, 1, 0, nullptr,
                                      sum_params, nullptr),
                      "cuLaunchKernel(layer0 sumsq)");
                check(handle_, launch(apply_fn, norm_grid, 1, 1, norm_block, 1, 1, 0, nullptr,
                                      apply_params, nullptr),
                      "cuLaunchKernel(layer0 rmsnorm apply)");
            };

            const unsigned int proj_grid_rows4 = (rows + 3u) / 4u;
            auto launch_proj = [&]() {
                check(handle_, launch(proj_fn, proj_grid_rows4, 1, 1, 128, 1, 1, 0, nullptr,
                                      proj_params, nullptr),
                      "cuLaunchKernel(layer0 attn_gate vec)");
            };

            // End-to-end correctness.
            launch_norm();
            launch_proj();
            check(handle_, sync(), "cuCtxSynchronize(layer0 norm->gate correctness)");
            check(handle_, memcpy_dtoh(out.data(), out_ptr, y_bytes), "cuMemcpyDtoH(layer0 gate)");

            const std::size_t checked_rows = std::min<std::size_t>(rows, 8);
            std::array<float, kQ4KValuesPerBlock> deq{};
            std::array<std::byte, kQ5KSm86BytesPerBlock> block_buf{};
            double abs_max = 0.0;
            double rel_max = 0.0;
            bool tolerance_violation = false;
            constexpr double kAbsTol = 3.0e-3;
            constexpr double kRelTol = 3.0e-3;

            for (std::size_t row = 0; row < checked_rows; ++row) {
                double ref = 0.0;
                for (std::size_t ib = 0; ib < blocks_per_row; ++ib) {
                    const std::size_t block_index = row * blocks_per_row + ib;
                    std::memcpy(block_buf.data() + 0,
                                repacked_matrix + block_index * 20, 20);
                    std::memcpy(block_buf.data() + 20,
                                repacked_matrix + qh_offset + block_index * 32, 32);
                    std::memcpy(block_buf.data() + 52,
                                repacked_matrix + qs_offset + block_index * 128, 128);
                    dequantize_q5_k_sm86_block_cpu(block_buf.data(), deq);
                    const std::size_t base = ib * kQ4KValuesPerBlock;
                    for (std::size_t j = 0; j < kQ4KValuesPerBlock; ++j) {
                        ref += static_cast<double>(deq[j]) *
                               static_cast<double>(normed_cpu[base + j]);
                    }
                }
                const double got = static_cast<double>(out[row]);
                const double abs_err = std::abs(got - ref);
                const double rel_err = abs_err / std::max(1.0e-5, std::abs(ref));
                abs_max = std::max(abs_max, abs_err);
                rel_max = std::max(rel_max, rel_err);
                if (abs_err > kAbsTol && rel_err > kRelTol) {
                    tolerance_violation = true;
                }
            }

            if (max_abs_error) *max_abs_error = abs_max;
            if (max_rel_error) *max_rel_error = rel_max;
            if (tolerance_violation) {
                std::ostringstream oss;
                oss << "Qwen3.8 layer0 norm->gate mismatch: max_abs=" << abs_max
                    << " max_rel=" << rel_max;
                throw std::runtime_error(oss.str());
            }

            constexpr int kRmsIters = 100;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kRmsIters; ++i) launch_norm();
            check(handle_, sync(), "cuCtxSynchronize(layer0 RMSNorm benchmark)");
            auto t1 = std::chrono::steady_clock::now();
            const double rms_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count() /
                static_cast<double>(kRmsIters);

            constexpr int kProjIters = 50;
            t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kProjIters; ++i) launch_proj();
            check(handle_, sync(), "cuCtxSynchronize(layer0 gate benchmark)");
            t1 = std::chrono::steady_clock::now();
            const double proj_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count() /
                static_cast<double>(kProjIters);

            constexpr int kChainIters = 50;
            t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kChainIters; ++i) {
                launch_norm();
                launch_proj();
            }
            check(handle_, sync(), "cuCtxSynchronize(layer0 chain benchmark)");
            t1 = std::chrono::steady_clock::now();
            const double full_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count() /
                static_cast<double>(kChainIters);

            if (rmsnorm_ms) *rmsnorm_ms = rms_ms;
            if (projection_ms) *projection_ms = proj_ms;
            if (chain_ms) *chain_ms = full_ms;
            if (projection_original_equiv_gbps) {
                *projection_original_equiv_gbps =
                    static_cast<double>(original_bytes) / (proj_ms / 1000.0) / 1.0e9;
            }
        } catch (...) {
            if (proj_module) module_unload(proj_module);
            module_unload(norm_module);
            throw;
        }

        check(handle_, module_unload(proj_module), "cuModuleUnload(layer0 gate projection)");
        check(handle_, module_unload(norm_module), "cuModuleUnload(layer0 RMSNorm)");
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

} // namespace q38
