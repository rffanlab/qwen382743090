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

            auto make_q5_args = [&](CUdeviceptr tensor_ptr,
                                     std::size_t qh_offset,
                                     std::size_t qs_offset,
                                     CUdeviceptr out_ptr,
                                     std::uint32_t rows) {
                struct Args {
                    CUdeviceptr meta;
                    CUdeviceptr qh;
                    CUdeviceptr qs;
                    CUdeviceptr x;
                    CUdeviceptr y;
                    std::uint32_t cols;
                    std::uint32_t rows;
                    void* params[7];
                };
                Args a{};
                a.meta = tensor_ptr;
                a.qh = tensor_ptr + qh_offset;
                a.qs = tensor_ptr + qs_offset;
                a.x = normed_ptr;
                a.y = out_ptr;
                a.cols = cols;
                a.rows = rows;
                a.params[0] = &a.meta;
                a.params[1] = &a.qh;
                a.params[2] = &a.qs;
                a.params[3] = &a.x;
                a.params[4] = &a.y;
                a.params[5] = &a.cols;
                a.params[6] = &a.rows;
                return a;
            };

            // Do not use make_q5_args() across statements: params contain
            // pointers to the object's own members, so instantiate in-place.
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
