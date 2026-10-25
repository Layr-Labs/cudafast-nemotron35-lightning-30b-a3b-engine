// Transparent generic-compression storage. DLPack legacy ABI, CUDA uint8.
// No allocator replacement, codec, model hook, or numerical operation.
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <string>

struct DLDevice { int device_type; int device_id; };
struct DLDataType { uint8_t code; uint8_t bits; uint16_t lanes; };
struct DLTensor {
    void* data; DLDevice device; int ndim; DLDataType dtype;
    int64_t* shape; int64_t* strides; uint64_t byte_offset;
};
struct DLManagedTensor {
    DLTensor dl_tensor; void* manager_ctx;
    void (*deleter)(DLManagedTensor*);
};
struct Allocation {
    DLManagedTensor managed = {};
    CUdeviceptr address = 0;
    CUmemGenericAllocationHandle handle = 0;
    size_t bytes = 0;
    int64_t shape = 0;
    bool mapped = false;
};
static thread_local std::string last_error;
static bool ok(CUresult rc, const char* call) {
    if (rc == CUDA_SUCCESS) return true;
    const char* name = nullptr;
    cuGetErrorName(rc, &name);
    last_error = std::string(call) + ": " + (name ? name : "unknown");
    return false;
}
static void destroy(DLManagedTensor* managed) {
    auto* a = static_cast<Allocation*>(managed->manager_ctx);
    if (a->mapped) cuMemUnmap(a->address, a->bytes);
    if (a->address) cuMemAddressFree(a->address, a->bytes);
    if (a->handle) cuMemRelease(a->handle);
    delete a;
}
extern "C" const char* cmp_last_error() { return last_error.c_str(); }
extern "C" int cmp_supported() {
    CUdevice device;
    int vmm = 0, compression = 0;
    if (!ok(cuCtxGetDevice(&device), "cuCtxGetDevice")) return 0;
    if (!ok(cuDeviceGetAttribute(&vmm,
            CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device),
            "VMM support query")) return 0;
    if (!ok(cuDeviceGetAttribute(&compression,
            CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device),
            "compression support query")) return 0;
    if (!vmm || !compression) {
        last_error = "device does not support VMM generic compression";
        return 0;
    }
    return 1;
}
extern "C" DLManagedTensor* cmp_allocate(size_t bytes, int compressed,
                                         int* actual_type, size_t* allocated) {
    auto* a = new Allocation;
    a->managed.manager_ctx = a;
    a->managed.deleter = destroy;
    auto fail = [&]() -> DLManagedTensor* { destroy(&a->managed); return nullptr; };
    CUdevice device;
    if (!ok(cuCtxGetDevice(&device), "cuCtxGetDevice")) return fail();
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.allocFlags.compressionType = compressed ? CU_MEM_ALLOCATION_COMP_GENERIC : CU_MEM_ALLOCATION_COMP_NONE;
    size_t granularity = 0;
    if (!ok(cuMemGetAllocationGranularity(&granularity, &prop,
        CU_MEM_ALLOC_GRANULARITY_MINIMUM), "cuMemGetAllocationGranularity")) return fail();
    // Equal 2 MiB alignment for the ordinary and compressed comparison arms.
    const size_t alignment = granularity > 2097152 ? granularity : 2097152;
    a->bytes = ((bytes + alignment - 1) / alignment) * alignment;
    if (!ok(cuMemCreate(&a->handle, a->bytes, &prop, 0), "cuMemCreate")) return fail();
    CUmemAllocationProp actual = {};
    if (!ok(cuMemGetAllocationPropertiesFromHandle(&actual, a->handle),
            "cuMemGetAllocationPropertiesFromHandle")) return fail();
    *actual_type = actual.allocFlags.compressionType;
    *allocated = a->bytes;
    if (*actual_type != compressed) {
        last_error = "allocation backing differs from requested compression type";
        return fail();
    }
    if (!ok(cuMemAddressReserve(&a->address, a->bytes, alignment, 0, 0),
            "cuMemAddressReserve")) return fail();
    if (!ok(cuMemMap(a->address, a->bytes, 0, a->handle, 0), "cuMemMap")) return fail();
    a->mapped = true;
    CUmemAccessDesc access = {};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (!ok(cuMemSetAccess(a->address, a->bytes, &access, 1), "cuMemSetAccess")) return fail();
    a->shape = static_cast<int64_t>(bytes);
    a->managed.dl_tensor = {reinterpret_cast<void*>(a->address), {2, device}, 1,
                            {1, 8, 1}, &a->shape, nullptr, 0};
    return &a->managed;
}
