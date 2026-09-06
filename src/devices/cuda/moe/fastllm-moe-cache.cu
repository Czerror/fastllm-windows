//
// Graph-capturable FP8 and compact NVFP4 expert cache.
//
// Host records retain quantized weights and their format-specific scales.
// A device LRU cache stores complete expert records in that same compact
// representation. Misses are pulled directly from mapped pinned host memory
// by a CUDA kernel; CPU cores do not participate in the decode hot path.
//

#include "fastllm-cuda.cuh"
#include "fastllm-cuda-expert-cache.cuh"
#include "fastllm-cuda-record-copy.cuh"
#include "fastllm.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cuda_bf16.h>

namespace {

constexpr int kMaxTopK = 16;
constexpr size_t kMinDeviceMemoryReserveBytes = 256ULL << 20;
constexpr size_t kMaxDeviceMemoryReserveBytes = 2ULL << 30;
constexpr size_t kDeviceMemoryReserveDivisor = 16;

bool PackedCacheRows(const fastllm::Data &data) {
    return data.dims.size() == 2 && data.strides.size() == 2 &&
           data.strides[1] == 1 &&
           (data.dims[0] == 1 || data.strides[0] == data.dims[1]);
}

bool SupportedCacheInput(const fastllm::Data &input) {
    return PackedCacheRows(input) && input.dims[0] > 0 &&
           input.dims[0] <= FASTLLM_CUDA_MOE_CACHE_MAX_BATCH &&
           input.dataDevice == fastllm::DataDevice::CUDA &&
           input.cudaData != nullptr &&
           (input.dataType == fastllm::DataType::FLOAT32 ||
            input.dataType == fastllm::DataType::FLOAT16 ||
            input.dataType == fastllm::DataType::BFLOAT16);
}

size_t AlignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct OffloadLayout {
    fastllm::DataType weightType = fastllm::DataType::NVFP4_BLOCK_16_E4M3;
    int experts = 0;
    int hidden = 0;
    int inter = 0;
    int gateBlockK = 0;
    int gateBlockM = 0;
    int downBlockK = 0;
    int downBlockM = 0;
    int gateScaleCols = 0;
    int downScaleCols = 0;
    size_t gateBytes = 0;
    size_t downBytes = 0;
    size_t downOffset = 0;
    size_t scalesOffset = 0;
    size_t gateScaleBytes = 0;
    size_t downScaleBytes = 0;
    size_t recordStride = 0;
};

const char *FormatName(fastllm::DataType type) {
    if (type == fastllm::DataType::FP8_E4M3) return "FP8 E4M3";
    if (type == fastllm::DataType::FP8_E4M3_BLOCK_128) return "FP8 block128";
    return "NVFP4";
}

int Fp8PointerTableCount(fastllm::DataType type) {
    return type == fastllm::DataType::FP8_E4M3 ? 4 :
        type == fastllm::DataType::FP8_E4M3_BLOCK_128 ? 2 : 0;
}

struct DeviceCache {
    bool attempted = false;
    bool ready = false;
    int device = -1;
    int slots = 0;
    fastllm::cuda::RecordCopyLaunch copyLaunch{0, 0};
    int ensureThreads = 0;
    uint8_t *records = nullptr;
    int32_t *keyToSlot = nullptr;
    int32_t *slotKeys = nullptr;
    unsigned long long *lastUsed = nullptr;
    unsigned long long *step = nullptr;
    int32_t *routeSlots = nullptr;
    int32_t *missExperts = nullptr;
    int32_t *missSlots = nullptr;
    int32_t *missCount = nullptr;
    unsigned long long *hitCount = nullptr;
    unsigned long long *totalMissCount = nullptr;
    // Contiguous slot pointer tables: gate/down weights, then external scales
    // for native E4M3. Packed block128 needs only the two weight tables.
    void **fp8Pointers = nullptr;
    void **numaPointers = nullptr;
};

struct OffloadGroup {
    OffloadLayout layout;
    size_t totalRecords = 0;
    uint8_t *hostRecords = nullptr;
    uint8_t *deviceHostRecords = nullptr;
    std::vector<const fastllm::Data *> tableKeys;
    // When NUMA storage is shared, hostRecords contains only original scales.
    // The weight shards are borrowed from the model's prefill backend.
    std::vector<void *> numaPointers;
    int numaCount = 0;
    int numaPlanarParts = 0;
    size_t hostScaleStride = 0;
    std::unordered_map<int, std::unique_ptr<DeviceCache> > deviceCaches;
    std::mutex mutex;
};

struct CachedTable {
    OffloadGroup *group;
    int layer;
};

size_t NumaScaleStride(const OffloadLayout &layout) {
    if (layout.weightType != fastllm::DataType::NVFP4_BLOCK_16_E4M3 ||
        layout.gateBlockK != 1 || layout.downBlockK != 1 ||
        layout.gateBlockM != 16 || layout.downBlockM != 16 ||
        layout.hidden % 16 != 0 || layout.inter % 16 != 0 ||
        layout.recordStride > UINT32_MAX / kMaxTopK) return 0;
    const size_t weightBytes = size_t(layout.hidden) * layout.inter * 3 / 2;
    const size_t bytes = layout.gateBytes + layout.downBytes - weightBytes + 3 * sizeof(float);
    return AlignUp(bytes, 16);
}

bool BindNumaWeights(OffloadGroup &group, const std::vector<fastllm::Data *> &weights) {
    const OffloadLayout &layout = group.layout;
    const int nodes = weights.front()->numasData.size();
    if (nodes <= 0 || (layout.inter * 2) % nodes != 0 ||
        layout.hidden % nodes != 0) return false;
    const fastllm::DataType storageTypes[]{weights[0]->dataType, weights[1]->dataType};
    int planarParts = 0;
    for (int part = 0; part < 2; ++part) {
        if (storageTypes[part] == fastllm::DataType::NVFP4_BLOCK_16_PLANAR) {
            const int rows = part == 0 ? 2 * layout.inter : layout.hidden;
            if ((rows / nodes) % fastllm::NVFP4_PLANAR_TILE_ROWS != 0) return false;
            planarParts |= 1 << part;
        } else if (storageTypes[part] != fastllm::DataType::NVFP4_BLOCK_16) {
            return false;
        }
    }
    std::vector<void *> pointers;
    pointers.reserve(weights.size() * nodes);
    for (size_t index = 0; index < weights.size(); ++index) {
        const fastllm::Data &weight = *weights[index];
        const int rows = index % 2 == 0 ? 2 * layout.inter : layout.hidden;
        const int columns = index % 2 == 0 ? layout.hidden : layout.inter;
        if (weight.dataType != storageTypes[index % 2] ||
            weight.dims.size() != 2 || weight.dims[0] != rows || weight.dims[1] != columns ||
            weight.blockK != 1 || weight.blockM != 16 ||
            weight.dataDevice != fastllm::DataDevice::CPU ||
            !weight.isPinned || weight.cpuData != nullptr ||
            weight.numasData.size() != size_t(nodes)) return false;
        for (uint8_t *source : weight.numasData) {
            void *mapped = nullptr;
            if (source == nullptr || (reinterpret_cast<uintptr_t>(source) & 15) != 0 ||
                cudaHostGetDevicePointer(&mapped, source, 0) != cudaSuccess ||
                mapped == nullptr) {
                cudaGetLastError();
                return false;
            }
            pointers.push_back(source);
        }
    }
    group.numaCount = nodes;
    group.numaPlanarParts = planarParts;
    group.numaPointers = std::move(pointers);
    return true;
}

std::mutex &RegistryMutex() {
    static auto *mutex = new std::mutex();
    return *mutex;
}

std::vector<std::unique_ptr<OffloadGroup> > &Groups() {
    // CUDA may already be unloading when ordinary static destructors run.
    // Process-lifetime ownership mirrors the other CUDA MoE registries.
    static auto *groups =
        new std::vector<std::unique_ptr<OffloadGroup> >();
    return *groups;
}

std::unordered_map<const fastllm::Data *, CachedTable> &TableRegistry() {
    static auto *registry =
        new std::unordered_map<const fastllm::Data *, CachedTable>();
    return *registry;
}

bool ValidateWeightPair(fastllm::Data *gate, fastllm::Data *down,
                        const OffloadLayout *expected,
                        OffloadLayout &observed) {
    if (gate == nullptr || down == nullptr ||
        (gate->dataType != fastllm::DataType::NVFP4_BLOCK_16_E4M3 &&
         gate->dataType != fastllm::DataType::FP8_E4M3 &&
         gate->dataType != fastllm::DataType::FP8_E4M3_BLOCK_128) ||
        down->dataType != gate->dataType ||
        gate->dataDevice != fastllm::DataDevice::CPU ||
        down->dataDevice != fastllm::DataDevice::CPU ||
        gate->cpuData == nullptr || down->cpuData == nullptr ||
        gate->dims.size() != 2 || down->dims.size() != 2 ||
        gate->dims[0] <= 0 || gate->dims[1] <= 0 ||
        down->dims[0] <= 0 || down->dims[1] <= 0 ||
        down->dims[1] > INT_MAX / 2 ||
        gate->dims[0] != down->dims[1] * 2 ||
        gate->dims[1] != down->dims[0]) {
        return false;
    }

    observed.weightType = gate->dataType;
    const bool block128 = gate->dataType == fastllm::DataType::FP8_E4M3_BLOCK_128;
    if (!block128 && (gate->blockK <= 0 || gate->blockM <= 0 ||
                     down->blockK <= 0 || down->blockM <= 0)) return false;
    if (gate->dataType == fastllm::DataType::NVFP4_BLOCK_16_E4M3 &&
        (gate->scales.size() != 2 || down->scales.size() != 1)) return false;
    observed.hidden = gate->dims[1];
    observed.inter = down->dims[1];
    observed.gateBlockK = block128 ? 1 : gate->blockK;
    observed.gateBlockM = block128 ? 128 : gate->blockM;
    observed.downBlockK = block128 ? 1 : down->blockK;
    observed.downBlockM = block128 ? 128 : down->blockM;
    // Existing indexed FP8 kernels load four weights at a time. Keep their
    // alignment and scale-block assumptions explicit; unsupported layouts
    // retain the configured backend rather than issuing misaligned reads.
    if (observed.weightType != fastllm::DataType::NVFP4_BLOCK_16_E4M3 &&
        ((observed.hidden & 3) || (observed.inter & 3) ||
         (observed.gateBlockM & 3) || (observed.downBlockM & 3))) return false;
    // The existing packed block128 indexed kernels expect complete blocks.
    if (block128 && ((observed.hidden & 127) || (observed.inter & 127))) return false;
    observed.gateScaleCols =
        (observed.hidden + observed.gateBlockM - 1) /
        observed.gateBlockM;
    observed.downScaleCols =
        (observed.inter + observed.downBlockM - 1) /
        observed.downBlockM;
    observed.gateBytes = gate->GetBytes();
    observed.downBytes = down->GetBytes();
    observed.downOffset = AlignUp(observed.gateBytes, 16);
    observed.scalesOffset =
        AlignUp(observed.downOffset + observed.downBytes, 16);
    if (gate->dataType == fastllm::DataType::NVFP4_BLOCK_16_E4M3) {
        observed.gateScaleBytes = 2 * sizeof(float);
        observed.downScaleBytes = sizeof(float);
    } else if (gate->dataType == fastllm::DataType::FP8_E4M3) {
        const size_t gateScaleCount =
            ((size_t(gate->dims[0]) - 1) / gate->blockK + 1) * observed.gateScaleCols;
        const size_t downScaleCount =
            ((size_t(down->dims[0]) - 1) / down->blockK + 1) * observed.downScaleCols;
        if (gate->scales.size() != gateScaleCount || down->scales.size() != downScaleCount)
            return false;
        observed.gateScaleBytes = gateScaleCount * sizeof(float);
        observed.downScaleBytes = downScaleCount * sizeof(float);
    }
    // All formats retain 128-byte alignment at every host record boundary.
    observed.recordStride = AlignUp(observed.scalesOffset +
        observed.gateScaleBytes + observed.downScaleBytes, 128);
    if (observed.gateBytes == 0 || observed.downBytes == 0) {
        return false;
    }
    if (expected == nullptr) {
        return true;
    }
    return expected->weightType == observed.weightType &&
           expected->hidden == observed.hidden &&
           expected->inter == observed.inter &&
           expected->gateBlockK == observed.gateBlockK &&
           expected->gateBlockM == observed.gateBlockM &&
           expected->downBlockK == observed.downBlockK &&
           expected->downBlockM == observed.downBlockM &&
           expected->gateBytes == observed.gateBytes &&
           expected->downBytes == observed.downBytes &&
           expected->downOffset == observed.downOffset &&
           expected->scalesOffset == observed.scalesOffset &&
           expected->gateScaleBytes == observed.gateScaleBytes &&
           expected->downScaleBytes == observed.downScaleBytes &&
           expected->recordStride == observed.recordStride;
}

size_t RequestedSlots(size_t recordStride, size_t totalRecords) {
    const uint64_t bytes = fastllm::GetMoeCudaCacheBytes();
    if (bytes == 0 || recordStride == 0) {
        return 0;
    }
    const uint64_t slots = bytes / recordStride;
    return slots >= totalRecords
        ? totalRecords : static_cast<size_t>(slots);
}

size_t DeviceMemoryReserveBytes(size_t totalBytes) {
    // Keep enough room for activations, CUDA Graph pools, and allocator
    // bookkeeping without imposing a fixed multi-GiB penalty on small GPUs.
    return std::min(
        kMaxDeviceMemoryReserveBytes,
        std::max(kMinDeviceMemoryReserveBytes,
                 totalBytes / kDeviceMemoryReserveDivisor));
}

void ReleaseDeviceCache(DeviceCache &cache) {
    if (cache.device >= 0) {
        cudaSetDevice(cache.device);
    }
    cudaFree(cache.records);
    cudaFree(cache.keyToSlot);
    cudaFree(cache.slotKeys);
    cudaFree(cache.lastUsed);
    cudaFree(cache.step);
    cudaFree(cache.routeSlots);
    cudaFree(cache.missExperts);
    cudaFree(cache.missSlots);
    cudaFree(cache.missCount);
    cudaFree(cache.hitCount);
    cudaFree(cache.totalMissCount);
    cudaFree(cache.fp8Pointers);
    cudaFree(cache.numaPointers);
    cache = DeviceCache();
}

void PrintDeviceCacheStats(const DeviceCache &cache, fastllm::DataType weightType) {
    if (!cache.ready || cache.hitCount == nullptr ||
        cache.totalMissCount == nullptr) {
        return;
    }
    cudaSetDevice(cache.device);
    unsigned long long hits = 0;
    unsigned long long misses = 0;
    const cudaError_t hitState = cudaMemcpy(
        &hits, cache.hitCount, sizeof(hits), cudaMemcpyDeviceToHost);
    const cudaError_t missState = cudaMemcpy(
        &misses, cache.totalMissCount, sizeof(misses),
        cudaMemcpyDeviceToHost);
    if (hitState != cudaSuccess || missState != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    const unsigned long long routes = hits + misses;
    const double hitRate = routes == 0 ? 0.0 :
        100.0 * static_cast<double>(hits) /
            static_cast<double>(routes);
    std::fprintf(
        stderr,
        "[Fastllm] %s GPU expert cache stats cuda:%d: "
        "%llu hits, %llu misses, %.3f%% hit rate, %d slots.\n",
        FormatName(weightType), cache.device, hits, misses, hitRate, cache.slots);
}

bool AllocateOne(void **pointer, size_t bytes) {
    *pointer = nullptr;
    return bytes > 0 && cudaMalloc(pointer, bytes) == cudaSuccess &&
           *pointer != nullptr;
}

bool PrepareFp8SlotPointers(DeviceCache &cache, const OffloadLayout &layout) {
    const int tables = Fp8PointerTableCount(layout.weightType);
    if (tables == 0) return true;
    const size_t slots = cache.slots;
    std::vector<void *> pointers(slots * tables);
    for (size_t slot = 0; slot < slots; ++slot) {
        uint8_t *record = cache.records + size_t(slot) * layout.recordStride;
        pointers[slot] = record;
        pointers[slots + slot] = record + layout.downOffset;
        if (tables == 4) {
            pointers[2 * slots + slot] = record + layout.scalesOffset;
            pointers[3 * slots + slot] = record + layout.scalesOffset + layout.gateScaleBytes;
        }
    }
    const size_t bytes = pointers.size() * sizeof(void *);
    return AllocateOne(reinterpret_cast<void **>(&cache.fp8Pointers), bytes) &&
        cudaMemcpy(cache.fp8Pointers, pointers.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

DeviceCache *GetDeviceCache(OffloadGroup &group) {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(group.mutex);
    std::unique_ptr<DeviceCache> &entry = group.deviceCaches[device];
    if (!entry) {
        entry.reset(new DeviceCache());
        entry->device = device;
    }
    DeviceCache &cache = *entry;
    if (cache.ready || cache.attempted) {
        return cache.ready ? &cache : nullptr;
    }
    cache.attempted = true;

    size_t slots = RequestedSlots(
        group.layout.recordStride, group.totalRecords);
    if (slots < kMaxTopK) {
        std::fprintf(stderr,
            "[Fastllm] CUDA expert cache needs at least %d slots; "
            "requested %zu.\n", kMaxTopK, slots);
        return nullptr;
    }

    size_t freeBytes = 0, totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) {
        return nullptr;
    }
    const size_t reserveBytes = std::min(
        freeBytes, DeviceMemoryReserveBytes(totalBytes));
    const size_t metadataBytes =
        group.totalRecords * sizeof(int32_t) +
        group.numaPointers.size() * sizeof(void *) +
        slots * (sizeof(int32_t) + sizeof(unsigned long long) +
            Fp8PointerTableCount(group.layout.weightType) * sizeof(void *)) + 4096;
    size_t usableBytes = freeBytes > reserveBytes
        ? freeBytes - reserveBytes : 0;
    usableBytes = usableBytes > metadataBytes
        ? usableBytes - metadataBytes : 0;
    slots = std::min(slots, usableBytes / group.layout.recordStride);
    if (slots < kMaxTopK) {
        std::fprintf(stderr,
            "[Fastllm] CUDA expert cache has insufficient free GPU "
            "memory after reserving %.3f GiB for runtime allocations.\n",
            static_cast<double>(reserveBytes) /
                (1024.0 * 1024.0 * 1024.0));
        return nullptr;
    }
    cache.slots = static_cast<int>(slots);

    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return nullptr;
    }
    cache.ensureThreads = fastllm::cuda::ExpertCacheThreads(
        cache.slots, properties.maxThreadsPerBlock);
    cache.copyLaunch = fastllm::cuda::RecordCopyConfiguration(
        group.layout.recordStride, kMaxTopK, properties);

    bool initialized =
        AllocateOne(reinterpret_cast<void **>(&cache.records),
                    slots * group.layout.recordStride) &&
        AllocateOne(reinterpret_cast<void **>(&cache.keyToSlot),
                    group.totalRecords * sizeof(int32_t)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.slotKeys),
                    slots * sizeof(int32_t)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.lastUsed),
                    slots * sizeof(unsigned long long)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.step),
                    sizeof(unsigned long long)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.routeSlots),
                    kMaxTopK * sizeof(int32_t)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.missExperts),
                    kMaxTopK * sizeof(int32_t)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.missSlots),
                    kMaxTopK * sizeof(int32_t)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.missCount),
                    sizeof(int32_t)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.hitCount),
                    sizeof(unsigned long long)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.totalMissCount),
                    sizeof(unsigned long long)) &&
        cudaMemsetAsync(cache.keyToSlot, 0xff,
            group.totalRecords * sizeof(int32_t), cudaStreamPerThread) == cudaSuccess &&
        cudaMemsetAsync(cache.slotKeys, 0xff,
            slots * sizeof(int32_t), cudaStreamPerThread) == cudaSuccess &&
        cudaMemsetAsync(cache.lastUsed, 0,
            slots * sizeof(unsigned long long), cudaStreamPerThread) == cudaSuccess &&
        cudaMemsetAsync(cache.step, 0,
            sizeof(unsigned long long), cudaStreamPerThread) == cudaSuccess &&
        cudaMemsetAsync(cache.hitCount, 0,
            sizeof(unsigned long long), cudaStreamPerThread) == cudaSuccess &&
        cudaMemsetAsync(cache.totalMissCount, 0,
            sizeof(unsigned long long), cudaStreamPerThread) == cudaSuccess &&
        PrepareFp8SlotPointers(cache, group.layout);
    if (initialized && !group.numaPointers.empty()) {
        std::vector<void *> pointers(group.numaPointers.size());
        for (size_t i = 0; initialized && i < pointers.size(); ++i) {
            initialized = cudaHostGetDevicePointer(
                &pointers[i], group.numaPointers[i], 0) == cudaSuccess;
        }
        const size_t bytes = pointers.size() * sizeof(void *);
        initialized = initialized &&
            AllocateOne(reinterpret_cast<void **>(&cache.numaPointers), bytes) &&
            cudaMemcpy(cache.numaPointers, pointers.data(), bytes,
                       cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (!initialized) {
        std::fprintf(stderr,
            "[Fastllm] CUDA expert cache allocation or initialization failed: "
            "%s.\n", cudaGetErrorString(cudaGetLastError()));
        ReleaseDeviceCache(cache);
        cache.attempted = true;
        cache.device = device;
        return nullptr;
    }

    cache.ready = true;

    std::fprintf(stderr,
        "[Fastllm] %s GPU expert cache: %d slots, %.3f GiB, "
        "%zu records in mapped host storage; refill grid %d x %d.\n",
        FormatName(group.layout.weightType), cache.slots,
        static_cast<double>(slots * group.layout.recordStride) /
            (1024.0 * 1024.0 * 1024.0),
        group.totalRecords, cache.copyLaunch.blocks, cache.copyLaunch.threads);
    return &cache;
}

OffloadGroup *FindGroup(fastllm::Data **weights, int weightsBatch,
                        int *tableId = nullptr) {
    if (weights == nullptr || weightsBatch < 4 || weights[2] == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(RegistryMutex());
    auto it = TableRegistry().find(weights[2]);
    if (it == TableRegistry().end()) {
        return nullptr;
    }
    OffloadGroup *group = it->second.group;
    if (weightsBatch != (group->layout.experts + 1) * 2) {
        return nullptr;
    }
    if (tableId != nullptr) {
        *tableId = it->second.layer;
    }
    return group;
}

template <typename T>
struct ActivationTraits;

template <>
struct ActivationTraits<float> {
    __device__ static float ToFloat(float value) { return value; }
    __device__ static float FromFloat(float value) { return value; }
    __device__ static float E4M3ToFloat(uint8_t value) {
        return __half2float(__ushort_as_half(
            ((value & 0x80) << 8) | ((value & 0x7f) << 7))) *
            exp2f(8.0f);
    }
};

template <>
struct ActivationTraits<half> : ActivationTraits<float> {
    __device__ static float ToFloat(half value) { return __half2float(value); }
    __device__ static half FromFloat(float value) { return __float2half(value); }
};

template <>
struct ActivationTraits<__nv_bfloat16> {
    __device__ static float ToFloat(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }
    __device__ static __nv_bfloat16 FromFloat(float value) {
        return __float2bfloat16_rn(value);
    }
    __device__ static float E4M3ToFloat(uint8_t value) {
        const uint16_t bits =
            ((value & 0x80) << 8) | ((value & 0x7f) << 4);
        return __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16 *>(&bits)) *
            exp2f(120.0f);
    }
};

__device__ __forceinline__ float NVFP4PseudoToFloat(uint8_t value) {
    const uint32_t bits =
        (static_cast<uint32_t>(value & 0x8) << 28) |
        (static_cast<uint32_t>(value & 0x7) << 22);
    return __uint_as_float(bits);
}

__device__ __forceinline__ float NVFP4MagicScale() {
    return __uint_as_float(253u << 23);
}

template <typename T>
__device__ __forceinline__ float CompactE4M3Scale(
        const uint8_t *scaleData, int scaleRow, int scaleCols,
        int column, int blockM, float globalScale) {
    const uint8_t scaleByte = scaleData[
        static_cast<size_t>(scaleRow) * scaleCols + column / blockM];
    return ActivationTraits<T>::E4M3ToFloat(scaleByte) * globalScale;
}

template <typename T>
__device__ __forceinline__ void AccumulateCompactE4M3(
        const T *activation, int offset, const uint8_t *weight, int row,
        int rows, int columns, int blockK, int blockM, int scaleCols,
        float globalScale, float &sum) {
    const int remaining = min(4, columns - offset);
    if (remaining <= 0) {
        return;
    }
    const int packedPerRow = (columns + 1) >> 1;
    const uint8_t *rowData = weight +
        static_cast<size_t>(row) * packedPerRow;
    const uint8_t *scaleData = weight +
        static_cast<size_t>(rows) * packedPerRow;
    const int scaleRow = row / blockK;
    if (blockK == 1 && blockM == 16 && (columns & 127) == 0 &&
        remaining == 4) {
        // Adjacent lanes consume the low/high halves of the same eight-code
        // word. Let the even lane issue one aligned 32-bit load and broadcast
        // it to its partner. Four neighboring lanes similarly share one
        // block-16 scale. The per-thread accumulation and the later two-warp
        // reduction stay unchanged, preserving the established decode
        // numerics while reducing weight/scale load instructions.
        const int lane = threadIdx.x & 31;
        uint32_t packedWord = 0;
        if ((lane & 1) == 0) {
            packedWord = *reinterpret_cast<const uint32_t *>(
                rowData + (offset >> 1));
        }
        packedWord = __shfl_sync(
            0xffffffffu, packedWord, lane & ~1);
        int scaleByte = 0;
        if ((lane & 3) == 0) {
            scaleByte = scaleData[
                static_cast<size_t>(scaleRow) * scaleCols +
                offset / blockM];
        }
        scaleByte = __shfl_sync(
            0xffffffffu, scaleByte, lane & ~3);
        const uint16_t packedHalf = (lane & 1)
            ? static_cast<uint16_t>(packedWord >> 16)
            : static_cast<uint16_t>(packedWord);
        const uint8_t packed01 = static_cast<uint8_t>(packedHalf);
        const uint8_t packed23 = static_cast<uint8_t>(packedHalf >> 8);
        const float blockSum =
            ActivationTraits<T>::ToFloat(activation[offset]) *
                NVFP4PseudoToFloat(packed01 & 0xf) +
            ActivationTraits<T>::ToFloat(activation[offset + 1]) *
                NVFP4PseudoToFloat(packed01 >> 4) +
            ActivationTraits<T>::ToFloat(activation[offset + 2]) *
                NVFP4PseudoToFloat(packed23 & 0xf) +
            ActivationTraits<T>::ToFloat(activation[offset + 3]) *
                NVFP4PseudoToFloat(packed23 >> 4);
        sum += (blockSum * NVFP4MagicScale()) *
            (ActivationTraits<T>::E4M3ToFloat(
                 static_cast<uint8_t>(scaleByte)) * globalScale);
        return;
    }
    if (remaining == 4 && offset / blockM == (offset + 3) / blockM) {
        const uint8_t packed01 = rowData[offset >> 1];
        const uint8_t packed23 = rowData[(offset + 2) >> 1];
        const float blockSum =
            ActivationTraits<T>::ToFloat(activation[offset]) *
                NVFP4PseudoToFloat(packed01 & 0xf) +
            ActivationTraits<T>::ToFloat(activation[offset + 1]) *
                NVFP4PseudoToFloat(packed01 >> 4) +
            ActivationTraits<T>::ToFloat(activation[offset + 2]) *
                NVFP4PseudoToFloat(packed23 & 0xf) +
            ActivationTraits<T>::ToFloat(activation[offset + 3]) *
                NVFP4PseudoToFloat(packed23 >> 4);
        sum += (blockSum * NVFP4MagicScale()) *
            CompactE4M3Scale<T>(scaleData, scaleRow, scaleCols, offset,
                                blockM, globalScale);
        return;
    }
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        if (item < remaining) {
            const int column = offset + item;
            const uint8_t packed = rowData[column >> 1];
            const uint8_t fp4 = (column & 1) ? packed >> 4 : packed & 0xf;
            const float product =
                ActivationTraits<T>::ToFloat(activation[column]) *
                NVFP4PseudoToFloat(fp4);
            sum += (product * NVFP4MagicScale()) *
                CompactE4M3Scale<T>(scaleData, scaleRow, scaleCols,
                                    column, blockM, globalScale);
        }
    }
}

template <typename T>
__device__ __forceinline__ void AccumulateCompactE4M3WordPair(
        const T *activation, int word, const uint8_t *weight, int row,
        int rows, int columns, int scaleCols, float globalScale,
        float &lowSum, float &highSum) {
    const int packedPerRow = columns >> 1;
    const uint8_t *rowData = weight +
        static_cast<size_t>(row) * packedPerRow;
    const uint8_t *scaleData = weight +
        static_cast<size_t>(rows) * packedPerRow;
    const uint32_t packed = reinterpret_cast<const uint32_t *>(
        rowData)[word];
    const float scale = ActivationTraits<T>::E4M3ToFloat(
        scaleData[static_cast<size_t>(row) * scaleCols + (word >> 1)]) *
        globalScale;
    const int offset = word << 3;
    const uint8_t packed01 = static_cast<uint8_t>(packed);
    const uint8_t packed23 = static_cast<uint8_t>(packed >> 8);
    const uint8_t packed45 = static_cast<uint8_t>(packed >> 16);
    const uint8_t packed67 = static_cast<uint8_t>(packed >> 24);
    const float low =
        ActivationTraits<T>::ToFloat(activation[offset]) *
            NVFP4PseudoToFloat(packed01 & 0xf) +
        ActivationTraits<T>::ToFloat(activation[offset + 1]) *
            NVFP4PseudoToFloat(packed01 >> 4) +
        ActivationTraits<T>::ToFloat(activation[offset + 2]) *
            NVFP4PseudoToFloat(packed23 & 0xf) +
        ActivationTraits<T>::ToFloat(activation[offset + 3]) *
            NVFP4PseudoToFloat(packed23 >> 4);
    const float high =
        ActivationTraits<T>::ToFloat(activation[offset + 4]) *
            NVFP4PseudoToFloat(packed45 & 0xf) +
        ActivationTraits<T>::ToFloat(activation[offset + 5]) *
            NVFP4PseudoToFloat(packed45 >> 4) +
        ActivationTraits<T>::ToFloat(activation[offset + 6]) *
            NVFP4PseudoToFloat(packed67 & 0xf) +
        ActivationTraits<T>::ToFloat(activation[offset + 7]) *
            NVFP4PseudoToFloat(packed67 >> 4);
    lowSum += (low * NVFP4MagicScale()) * scale;
    highSum += (high * NVFP4MagicScale()) * scale;
}

__device__ __forceinline__ float ReduceVirtual64(
        float low, float high) {
    // One physical lane represents two adjacent lanes from the established
    // 64-thread kernel. Half-warp shuffles reproduce its 16/8/4/2 stages;
    // adding low+high is the final offset-1 stage. Lanes 0 and 16 therefore
    // hold the two original warp partials.
    for (int offset = 8; offset > 0; offset >>= 1) {
        low += __shfl_down_sync(0xffffffffu, low, offset, 16);
        high += __shfl_down_sync(0xffffffffu, high, offset, 16);
    }
    const float half = low + high;
    return half + __shfl_sync(0xffffffffu, half, 16);
}

template <typename T, int RowPairsPerBlock>
__global__ void FastllmNVFP4OffloadGateExactWideKernel(
        const T *input, const int32_t *routeSlots,
        const uint8_t *records, T *output, int topk, int hidden, int inter,
        int scaleCols, size_t recordStride, size_t scalesOffset) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * RowPairsPerBlock + warp;
    const int route = blockIdx.y;
    if (warp >= RowPairsPerBlock || row >= inter || route >= topk) {
        return;
    }
    const int slot = routeSlots[route];
    if (slot < 0) {
        return;
    }
    const uint8_t *record = records +
        static_cast<size_t>(slot) * recordStride;
    const float *globalScales = reinterpret_cast<const float *>(
        record + scalesOffset);
    float gateLow = 0.0f, gateHigh = 0.0f;
    float upLow = 0.0f, upHigh = 0.0f;
    const int words = hidden >> 3;
    for (int word = lane; word < words; word += 32) {
        AccumulateCompactE4M3WordPair(
            input, word, record, row, inter * 2, hidden, scaleCols,
            globalScales[0], gateLow, gateHigh);
        AccumulateCompactE4M3WordPair(
            input, word, record, row + inter, inter * 2, hidden,
            scaleCols, globalScales[1], upLow, upHigh);
    }
    const float gate = ReduceVirtual64(gateLow, gateHigh);
    const float up = ReduceVirtual64(upLow, upHigh);
    if (lane == 0) {
        const float gateValue = ActivationTraits<T>::ToFloat(
            ActivationTraits<T>::FromFloat(gate));
        const float upValue = ActivationTraits<T>::ToFloat(
            ActivationTraits<T>::FromFloat(up));
        output[static_cast<size_t>(route) * inter + row] =
            ActivationTraits<T>::FromFloat(
                (gateValue / (1.0f + expf(-gateValue))) * upValue);
    }
}

template <typename T, int Threads>
__global__ void FastllmNVFP4OffloadGateKernel(
        const T *input, const int32_t *routeSlots,
        const uint8_t *records, T *output, int topk, int hidden, int inter,
        int blockK, int blockM, int scaleCols, size_t recordStride,
        size_t scalesOffset) {
    static_assert(Threads == 64, "gate reduction expects two warps");
    __shared__ float gateWarp[2];
    __shared__ float upWarp[2];
    const int row = blockIdx.x;
    const int route = blockIdx.y;
    if (route >= topk || routeSlots[route] < 0) {
        return;
    }
    const int local = threadIdx.x;
    const uint8_t *record = records +
        static_cast<size_t>(routeSlots[route]) * recordStride;
    const float *globalScales = reinterpret_cast<const float *>(
        record + scalesOffset);
    float gate = 0.0f;
    float up = 0.0f;
    for (int column = local * 4; column < hidden;
         column += Threads * 4) {
        AccumulateCompactE4M3(
            input, column, record, row, inter * 2, hidden,
            blockK, blockM, scaleCols, globalScales[0], gate);
        AccumulateCompactE4M3(
            input, column, record, row + inter, inter * 2, hidden,
            blockK, blockM, scaleCols, globalScales[1], up);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffffu, gate, offset);
        up += __shfl_down_sync(0xffffffffu, up, offset);
    }
    if ((local & 31) == 0) {
        gateWarp[local >> 5] = gate;
        upWarp[local >> 5] = up;
    }
    __syncthreads();
    if (local == 0) {
        const float gateValue = ActivationTraits<T>::ToFloat(
            ActivationTraits<T>::FromFloat(gateWarp[0] + gateWarp[1]));
        const float upValue = ActivationTraits<T>::ToFloat(
            ActivationTraits<T>::FromFloat(upWarp[0] + upWarp[1]));
        output[static_cast<size_t>(route) * inter + row] =
            ActivationTraits<T>::FromFloat(
                (gateValue / (1.0f + expf(-gateValue))) * upValue);
    }
}

template <typename T, int GroupThreads, int MaxTopK>
__global__ void FastllmNVFP4OffloadDownKernel(
        const T *input, const int32_t *routeSlots,
        const uint8_t *records, T *output, const float *scores,
        int topk, int inter, int hidden, int blockK, int blockM,
        int scaleCols, size_t recordStride, size_t downOffset,
        size_t scalesOffset) {
    __shared__ float warpPartials[MaxTopK * 2];
    __shared__ float expertOutputs[MaxTopK];
    const int route = threadIdx.x / GroupThreads;
    const int local = threadIdx.x % GroupThreads;
    const int row = blockIdx.x;
    const int slot = routeSlots[route];
    const size_t safeSlot = slot >= 0 ? static_cast<size_t>(slot) : 0;
    const uint8_t *record = records + safeSlot * recordStride;
    const uint8_t *weight = record + downOffset;
    const float *globalScales = reinterpret_cast<const float *>(
        record + scalesOffset);
    const T *expertInput = input + static_cast<size_t>(route) * inter;
    float value = 0.0f;
    if (route < topk && slot >= 0) {
        for (int column = local * 4; column < inter;
             column += GroupThreads * 4) {
            AccumulateCompactE4M3(
                expertInput, column, weight, row, hidden, inter,
                blockK, blockM, scaleCols, globalScales[2], value);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if ((local & 31) == 0) {
        warpPartials[route * 2 + (local >> 5)] = value;
    }
    __syncthreads();
    if (local == 0) {
        const float rounded = ActivationTraits<T>::ToFloat(
            ActivationTraits<T>::FromFloat(
                warpPartials[route * 2] + warpPartials[route * 2 + 1]));
        expertOutputs[route] = rounded * scores[route];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int item = 0; item < topk; ++item) {
            sum += expertOutputs[item];
        }
        output[row] = ActivationTraits<T>::FromFloat(sum);
    }
}

template <typename T, int MaxTopK>
__global__ void FastllmNVFP4OffloadDownExactWideKernel(
        const T *input, const int32_t *routeSlots,
        const uint8_t *records, T *output, const float *scores,
        int topk, int inter, int hidden, int scaleCols,
        size_t recordStride, size_t downOffset, size_t scalesOffset) {
    __shared__ float expertOutputs[MaxTopK];
    const int route = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x;
    float low = 0.0f, high = 0.0f;
    if (route < topk) {
        const int slot = routeSlots[route];
        if (slot >= 0) {
            const uint8_t *record = records +
                static_cast<size_t>(slot) * recordStride;
            const uint8_t *weight = record + downOffset;
            const T *expertInput = input +
                static_cast<size_t>(route) * inter;
            const int words = inter >> 3;
            const float globalScale = reinterpret_cast<const float *>(
                record + scalesOffset)[2];
            for (int word = lane; word < words; word += 32) {
                AccumulateCompactE4M3WordPair(
                    expertInput, word, weight, row, hidden, inter,
                    scaleCols, globalScale, low, high);
            }
        }
        const float value = ReduceVirtual64(low, high);
        if (lane == 0) {
            const float rounded = ActivationTraits<T>::ToFloat(
                ActivationTraits<T>::FromFloat(value));
            expertOutputs[route] = rounded * scores[route];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int item = 0; item < topk; ++item) {
            sum += expertOutputs[item];
        }
        output[row] = ActivationTraits<T>::FromFloat(sum);
    }
}

template <typename T>
bool LaunchNVFP4Cache(const fastllm::Data &input, fastllm::Data &gateOutput,
                       fastllm::Data &output, const OffloadLayout &layout,
                       const DeviceCache &cache, const float *scores, int topk) {
    const bool useExactWideDecode =
        layout.gateBlockK == 1 && layout.gateBlockM == 16 &&
        layout.downBlockK == 1 && layout.downBlockM == 16 &&
        (layout.hidden & 127) == 0 && (layout.inter & 127) == 0;
    if (useExactWideDecode) {
        constexpr int rowPairsPerBlock = 4;
        const dim3 gateGrid(
            (layout.inter + rowPairsPerBlock - 1) / rowPairsPerBlock,
            topk);
        FastllmNVFP4OffloadGateExactWideKernel<T, rowPairsPerBlock>
            <<<gateGrid, rowPairsPerBlock * 32, 0,
               cudaStreamPerThread>>>(
            reinterpret_cast<const T *>(input.cudaData), cache.routeSlots,
            cache.records, reinterpret_cast<T *>(gateOutput.cudaData), topk,
            layout.hidden, layout.inter, layout.gateScaleCols,
            layout.recordStride, layout.scalesOffset);
        FastllmNVFP4OffloadDownExactWideKernel<T, kMaxTopK>
            <<<layout.hidden, topk * 32, 0, cudaStreamPerThread>>>(
            reinterpret_cast<const T *>(gateOutput.cudaData),
            cache.routeSlots, cache.records,
            reinterpret_cast<T *>(output.cudaData), scores, topk,
            layout.inter, layout.hidden, layout.downScaleCols,
            layout.recordStride, layout.downOffset,
            layout.scalesOffset);
    } else {
        const dim3 gateGrid(layout.inter, topk);
        FastllmNVFP4OffloadGateKernel<T, 64>
            <<<gateGrid, 64, 0, cudaStreamPerThread>>>(
            reinterpret_cast<const T *>(input.cudaData), cache.routeSlots,
            cache.records, reinterpret_cast<T *>(gateOutput.cudaData), topk,
            layout.hidden, layout.inter, layout.gateBlockK,
            layout.gateBlockM, layout.gateScaleCols, layout.recordStride,
            layout.scalesOffset);
        FastllmNVFP4OffloadDownKernel<T, 64, kMaxTopK>
            <<<layout.hidden, topk * 64, 0, cudaStreamPerThread>>>(
            reinterpret_cast<const T *>(gateOutput.cudaData),
            cache.routeSlots, cache.records,
            reinterpret_cast<T *>(output.cudaData), scores, topk,
            layout.inter, layout.hidden, layout.downBlockK,
            layout.downBlockM, layout.downScaleCols,
            layout.recordStride, layout.downOffset,
            layout.scalesOffset);
    }
    return cudaGetLastError() == cudaSuccess;
}

// Gather NUMA weight bytes and append original E4M3/global scales. Planar
// 32-row tiles keep FP32 scales out of the weight loads crossing PCIe; legacy
// inline shards remain supported when a matrix cannot form complete tiles.
template<class Unit, int PlanarParts>
__device__ Unit LoadNumaWeightWord(
        void *const *pointers, int nodes, int expert, int part,
        int rows, int columns, uint32_t offset) {
    const int rowBytes = columns / 2, rowsPerNode = rows / nodes;
    int row = offset / rowBytes;
    const int columnByte = offset % rowBytes;
    if (part == 0) {
        row = row < rows / 2 ? row * 2 : (row - rows / 2) * 2 + 1;
    }
    const int node = row / rowsPerNode;
    const uint8_t *source = static_cast<const uint8_t *>(
        pointers[(expert * 2 + part) * nodes + node]);
    if (PlanarParts & (1 << part)) {
        const int localRow = row % rowsPerNode;
        source += fastllm::NVFP4PlanarWeightOffset(localRow, columns / 16) + columnByte;
    } else {
        source += size_t(row % rowsPerNode) * (columns / 16) * 12 +
                  (columnByte / 8) * 12 + columnByte % 8;
    }
    return *reinterpret_cast<const Unit *>(source);
}

template<class Unit, int PlanarParts>
__global__ void CopyNumaNVFP4Records(
        OffloadLayout layout, void *const *pointers, int nodes,
        const uint8_t *scales, size_t scaleStride, uint8_t *records,
        const int32_t *sourceIds, const int32_t *destinationIds,
        const int32_t *count) {
    const uint32_t units = layout.recordStride / sizeof(Unit);
    const uint32_t gateWeightBytes = layout.inter * layout.hidden;
    const uint32_t downWeightBytes = gateWeightBytes / 2;
    const uint32_t gateScales = layout.gateBytes - gateWeightBytes;
    const uint32_t downScales = layout.downBytes - downWeightBytes;
    const uint32_t total = *count * units;
    for (uint32_t flat = blockIdx.x * blockDim.x + threadIdx.x;
         flat < total; flat += gridDim.x * blockDim.x) {
        const uint32_t request = flat / units;
        const uint32_t offset = (flat % units) * sizeof(Unit);
        const int expert = sourceIds[request];
        const uint8_t *sourceScales = scales + size_t(expert) * scaleStride;
        Unit value{};
        if (offset < gateWeightBytes) {
            value = LoadNumaWeightWord<Unit, PlanarParts>(pointers, nodes, expert, 0,
                layout.inter * 2, layout.hidden, offset);
        } else if (offset < layout.gateBytes) {
            value = *reinterpret_cast<const Unit *>(
                sourceScales + offset - gateWeightBytes);
        } else if (offset >= layout.downOffset &&
                   offset < layout.downOffset + downWeightBytes) {
            value = LoadNumaWeightWord<Unit, PlanarParts>(pointers, nodes, expert, 1,
                layout.hidden, layout.inter, offset - layout.downOffset);
        } else if (offset >= layout.downOffset + downWeightBytes &&
                   offset < layout.downOffset + layout.downBytes) {
            value = *reinterpret_cast<const Unit *>(sourceScales + gateScales +
                offset - layout.downOffset - downWeightBytes);
        } else if (offset >= layout.scalesOffset &&
                   offset < layout.scalesOffset + 3 * sizeof(float)) {
            value = *reinterpret_cast<const Unit *>(sourceScales + gateScales +
                downScales + offset - layout.scalesOffset);
        }
        *reinterpret_cast<Unit *>(records +
            size_t(destinationIds[request]) * layout.recordStride + offset) = value;
    }
}

} // namespace

bool FastllmCudaMoeCacheRequested() {
    return fastllm::GetMoeCudaCacheBytes() > 0;
}

bool FastllmCudaPrepareMoeCache(
        const FastllmCudaMoeCacheLayer *layers, int layerCount,
        const std::function<void()> &registerNumaWeights) {
    if (!FastllmCudaMoeCacheRequested()) {
        return false;
    }
    if (layers == nullptr || layerCount <= 0 ||
        layers[0].weights == nullptr || layers[0].weightsBatch < 4 ||
        (layers[0].weightsBatch & 1)) {
        return false;
    }
    const int experts = layers[0].weightsBatch / 2 - 1;
    if (experts <= 0) {
        return false;
    }

    std::lock_guard<std::mutex> registryGuard(RegistryMutex());
    if (layers[0].weights[2] != nullptr &&
        TableRegistry().count(layers[0].weights[2]) != 0) {
        return true;
    }

    OffloadLayout layout;
    layout.experts = experts;
    bool first = true;
    for (int layer = 0; layer < layerCount; ++layer) {
        if (layers[layer].weights == nullptr ||
            layers[layer].weightsBatch != (experts + 1) * 2) {
            return false;
        }
        for (int expert = 0; expert < experts; ++expert) {
            const int position = (expert + 1) * 2;
            OffloadLayout observed;
            if (!ValidateWeightPair(
                    layers[layer].weights[position],
                    layers[layer].weights[position + 1],
                    first ? nullptr : &layout, observed)) {
                const fastllm::Data *gate =
                    layers[layer].weights[position];
                const fastllm::Data *down =
                    layers[layer].weights[position + 1];
                std::fprintf(stderr,
                    "[Fastllm] CUDA expert cache rejected layer %d expert %d: "
                    "requires matching host-resident NVFP4 or FP8 weights "
                    "and valid scales (gate dtype=%d, down dtype=%d).\n",
                    layer, expert,
                    gate == nullptr ? -1 : static_cast<int>(gate->dataType),
                    down == nullptr ? -1 : static_cast<int>(down->dataType));
                return false;
            }
            if (first) {
                observed.experts = experts;
                layout = observed;
                first = false;
            }
        }
    }

    std::unique_ptr<OffloadGroup> group(new OffloadGroup());
    group->layout = layout;
    if (static_cast<size_t>(layerCount) >
            static_cast<size_t>(INT_MAX) /
                static_cast<size_t>(experts)) {
        return false;
    }
    group->totalRecords = static_cast<size_t>(layerCount) * experts;
    if (group->totalRecords > SIZE_MAX / layout.recordStride) {
        return false;
    }
    const size_t requestedSlots = RequestedSlots(
        layout.recordStride, group->totalRecords);
    if (requestedSlots < kMaxTopK) {
        std::fprintf(
            stderr,
            "[Fastllm] CUDA expert cache needs at least %d slots; "
            "the configured budget holds %zu.\n",
            kMaxTopK, requestedSlots);
        return false;
    }
    // Capture scales directly from compact checkpoint tensors, before NUMA
    // registration releases them. Never allocate a duplicate weight snapshot.
    group->hostScaleStride = registerNumaWeights ? NumaScaleStride(layout) : 0;
    const bool shareNuma = group->hostScaleStride > 0;
    const size_t hostStride = shareNuma ? group->hostScaleStride : layout.recordStride;
    const size_t hostBytes = group->totalRecords * hostStride;
    void *host = nullptr;
    cudaError_t state = cudaHostAlloc(
        &host, hostBytes, cudaHostAllocMapped | cudaHostAllocPortable);
    if (state != cudaSuccess || host == nullptr) {
        std::fprintf(stderr,
            "[Fastllm] CUDA expert cache could not allocate %.3f GiB of mapped "
            "host storage: %s. The normal MoE backend remains active.\n",
            static_cast<double>(hostBytes) /
                (1024.0 * 1024.0 * 1024.0),
            cudaGetErrorString(state));
        cudaGetLastError();
        return false;
    }
    // Also releases the unpublished snapshot if registration fails or throws.
    std::unique_ptr<void, decltype(&cudaFreeHost)> hostOwner(host, cudaFreeHost);
    group->hostRecords = static_cast<uint8_t *>(host);
    void *deviceHost = nullptr;
    state = cudaHostGetDevicePointer(&deviceHost, host, 0);
    if (state != cudaSuccess || deviceHost == nullptr) {
        std::fprintf(stderr,
            "[Fastllm] CUDA expert cache could not map host storage: %s.\n",
            cudaGetErrorString(state));
        cudaGetLastError();
        return false;
    }
    group->deviceHostRecords = reinterpret_cast<uint8_t *>(deviceHost);
    group->tableKeys.reserve(layerCount);
    std::vector<fastllm::Data *> expertWeights;
    if (shareNuma) expertWeights.reserve(group->totalRecords * 2);

    for (int layer = 0; layer < layerCount; ++layer) {
        fastllm::Data *tableKey = layers[layer].weights[2];
        group->tableKeys.push_back(tableKey);
        for (int expert = 0; expert < experts; ++expert) {
            const int position = (expert + 1) * 2;
            fastllm::Data *gate = layers[layer].weights[position];
            fastllm::Data *down = layers[layer].weights[position + 1];
            uint8_t *record = group->hostRecords +
                (static_cast<size_t>(layer) * experts + expert) *
                hostStride;
            if (shareNuma) {
                expertWeights.push_back(gate);
                expertWeights.push_back(down);
                const size_t gateScales = layout.gateBytes - size_t(layout.inter) * layout.hidden;
                const size_t downScales = layout.downBytes - size_t(layout.inter) * layout.hidden / 2;
                std::memcpy(record, fastllm::GetNVFP4ScaleData(*gate), gateScales);
                std::memcpy(record + gateScales, fastllm::GetNVFP4ScaleData(*down), downScales);
                std::memcpy(record + gateScales + downScales, gate->scales.data(), 2 * sizeof(float));
                std::memcpy(record + gateScales + downScales + 2 * sizeof(float),
                            down->scales.data(), sizeof(float));
                std::memset(record + gateScales + downScales + 3 * sizeof(float), 0,
                            hostStride - gateScales - downScales - 3 * sizeof(float));
            } else {
                std::memcpy(record, gate->cpuData, layout.gateBytes);
                std::memcpy(record + layout.downOffset,
                            down->cpuData, layout.downBytes);
                if (layout.gateScaleBytes) {
                    std::memcpy(record + layout.scalesOffset,
                            gate->scales.data(), layout.gateScaleBytes);
                    std::memcpy(record + layout.scalesOffset + layout.gateScaleBytes,
                            down->scales.data(), layout.downScaleBytes);
                }
            }
        }
    }

    if (shareNuma) {
        registerNumaWeights();
        if (!BindNumaWeights(*group, expertWeights)) {
            std::fprintf(stderr, "[Fastllm] CUDA expert cache requires matching pinned NUMA "
                         "shards after registration; using the configured MoE backend.\n");
            return false;
        }
        std::fprintf(stderr,
            "[Fastllm] NVFP4 expert cache shares %d NUMA shards per weight; "
            "original scales use %.3f GiB, avoiding a %.3f GiB host weight snapshot.\n",
            group->numaCount, double(hostBytes) / (1ULL << 30),
            double(group->totalRecords * (layout.recordStride - hostStride)) / (1ULL << 30));
    }
    OffloadGroup *rawGroup = group.get();
    for (int layer = 0; layer < layerCount; ++layer) {
        TableRegistry()[group->tableKeys[layer]] = {rawGroup, layer};
    }
    Groups().push_back(std::move(group));
    hostOwner.release();
    std::fprintf(stderr,
        "[Fastllm] Prepared %d x %d %s experts in %.3f "
        "GiB of mapped host storage (record %zu bytes).\n",
        layerCount, experts, FormatName(layout.weightType),
        static_cast<double>(hostBytes) /
            (1024.0 * 1024.0 * 1024.0),
        layout.recordStride);
    return true;
}

bool FastllmCudaCanRunMoeCache(
        fastllm::Data **weights, int weightsBatch) {
    if (!FastllmCudaMoeCacheRequested()) {
        return false;
    }
    OffloadGroup *group = FindGroup(weights, weightsBatch);
    return group != nullptr && GetDeviceCache(*group) != nullptr;
}

bool FastllmCudaCanRunMoeCacheSmallBatch(
        const fastllm::Data &input, const fastllm::Data &index,
        const fastllm::Data &score, fastllm::Data **weights,
        int weightsBatch, fastllm::MoeGateType gateType) {
    const bool supportedInput = FastllmCudaMoeCacheRequested() &&
           gateType == fastllm::MoeGateSwiglu && SupportedCacheInput(input) &&
           index.dims.size() == 2 && index.dims[0] == input.dims[0] &&
           index.dims[1] > 0 && index.dims[1] <= kMaxTopK &&
           index.dataDevice == fastllm::DataDevice::CUDA &&
           index.dataType == fastllm::DataType::INT32 &&
           index.cudaData != nullptr && score.dims == index.dims &&
           score.dataDevice == fastllm::DataDevice::CUDA &&
           score.dataType == fastllm::DataType::FLOAT32 &&
           score.cudaData != nullptr &&
           PackedCacheRows(index) && PackedCacheRows(score);
    if (!supportedInput) {
        return false;
    }
    OffloadGroup *group = FindGroup(weights, weightsBatch);
    return group != nullptr && input.dims[1] == group->layout.hidden &&
           GetDeviceCache(*group) != nullptr;
}

void FastllmCudaReleaseMoeCache(
        fastllm::Data **weights, int weightsBatch) {
    if (weights == nullptr || weightsBatch < 4 || weights[2] == nullptr) {
        return;
    }

    std::unique_ptr<OffloadGroup> released;
    {
        std::lock_guard<std::mutex> registryGuard(RegistryMutex());
        auto table = TableRegistry().find(weights[2]);
        if (table == TableRegistry().end()) {
            return;
        }
        OffloadGroup *group = table->second.group;
        if (weightsBatch != (group->layout.experts + 1) * 2) {
            return;
        }
        for (const fastllm::Data *key : group->tableKeys) {
            auto entry = TableRegistry().find(key);
            if (entry != TableRegistry().end() && entry->second.group == group) {
                TableRegistry().erase(entry);
            }
        }
        auto entry = std::find_if(
            Groups().begin(), Groups().end(),
            [group](const std::unique_ptr<OffloadGroup> &candidate) {
                return candidate.get() == group;
            });
        if (entry != Groups().end()) {
            released = std::move(*entry);
            Groups().erase(entry);
        }
    }
    if (!released) {
        return;
    }

    int originalDevice = -1;
    cudaGetDevice(&originalDevice);
    for (auto &cache : released->deviceCaches) {
        PrintDeviceCacheStats(*cache.second, released->layout.weightType);
        ReleaseDeviceCache(*cache.second);
    }
    released->deviceCaches.clear();
    if (released->hostRecords != nullptr) {
        cudaFreeHost(released->hostRecords);
        released->hostRecords = nullptr;
        released->deviceHostRecords = nullptr;
    }
    if (originalDevice >= 0) {
        cudaSetDevice(originalDevice);
    }
}

bool FastllmCudaMergeMOECache(
        const fastllm::Data &input, fastllm::Data &gateOutput,
        fastllm::Data &output, fastllm::Data **weights, int weightsBatch,
        const int32_t *indices, const float *scores, int topk) {
    if (!SupportedCacheInput(input) || indices == nullptr || scores == nullptr ||
        topk <= 0 || topk > kMaxTopK) {
        return false;
    }
    int tableId = -1;
    OffloadGroup *group = FindGroup(weights, weightsBatch, &tableId);
    if (group == nullptr || input.dims.back() != group->layout.hidden) {
        return false;
    }
    DeviceCache *cache = GetDeviceCache(*group);
    if (cache == nullptr) {
        return false;
    }
    const OffloadLayout &layout = group->layout;
    gateOutput.dataType = input.dataType;
    gateOutput.dataDevice = input.dataDevice;
    gateOutput.dataDeviceIds = input.dataDeviceIds;
    const int rows = input.dims[0];
    gateOutput.Resize({rows * topk, layout.inter});
    output.dataType = input.dataType;
    output.dataDevice = input.dataDevice;
    output.dataDeviceIds = input.dataDeviceIds;
    output.Resize({rows, layout.hidden});
    gateOutput.Allocate(false);
    output.Allocate(false);
    if (gateOutput.cudaData == nullptr || output.cudaData == nullptr) {
        return false;
    }

    const fastllm::cuda::ExpertCacheView metadata{
        cache->keyToSlot, cache->slotKeys, cache->lastUsed, cache->step,
        cache->hitCount, cache->totalMissCount, cache->slots};
    const uint8_t *sourceTable = group->deviceHostRecords +
        static_cast<size_t>(tableId) * layout.experts *
            (group->numaCount > 0 ? group->hostScaleStride : layout.recordStride);
    auto computeRow = [&](const fastllm::Data &input, fastllm::Data &gateOutput,
                          fastllm::Data &output, const int32_t *indices,
                          const float *scores) {
        if (!fastllm::cuda::EnsureExpertCache<kMaxTopK>(
                metadata, indices, tableId * layout.experts, layout.experts, topk,
                cache->routeSlots, cache->missExperts, cache->missSlots, cache->missCount,
                cache->ensureThreads, cudaStreamPerThread)) return false;
        if (group->numaCount > 0) {
            void *const *pointers = cache->numaPointers +
                size_t(tableId) * layout.experts * 2 * group->numaCount;
#define FASTLLM_NUMA_COPY(Unit, Parts) \
            CopyNumaNVFP4Records<Unit, Parts><<<cache->copyLaunch.blocks, cache->copyLaunch.threads, \
                0, cudaStreamPerThread>>>(layout, pointers, group->numaCount, sourceTable, \
                group->hostScaleStride, cache->records, cache->missExperts, cache->missSlots, cache->missCount)
            if (group->numaPlanarParts == 3 && layout.hidden % 32 == 0 && layout.inter % 32 == 0) {
                FASTLLM_NUMA_COPY(uint4, 3);
            } else if (group->numaPlanarParts == 3) {
                FASTLLM_NUMA_COPY(uint32_t, 3);
            } else if (group->numaPlanarParts == 1) {
                FASTLLM_NUMA_COPY(uint32_t, 1);
            } else if (group->numaPlanarParts == 2) {
                FASTLLM_NUMA_COPY(uint32_t, 2);
            } else {
                FASTLLM_NUMA_COPY(uint32_t, 0);
            }
#undef FASTLLM_NUMA_COPY
        } else if (!fastllm::cuda::CopyRecords(
                {sourceTable, cache->records, layout.recordStride,
                 layout.recordStride, layout.recordStride},
                cache->missExperts, cache->missSlots, cache->missCount, topk,
                cache->copyLaunch, cudaStreamPerThread)) return false;
        if (layout.weightType != fastllm::DataType::NVFP4_BLOCK_16_E4M3) {
            const size_t slots = cache->slots;
            const bool externalScales = layout.weightType == fastllm::DataType::FP8_E4M3;
            const FastllmCudaMoeFP8CacheView view{
                layout.weightType,
                reinterpret_cast<uint8_t **>(cache->fp8Pointers),
                reinterpret_cast<uint8_t **>(cache->fp8Pointers + slots),
                externalScales ? reinterpret_cast<float **>(cache->fp8Pointers + 2 * slots) : nullptr,
                externalScales ? reinterpret_cast<float **>(cache->fp8Pointers + 3 * slots) : nullptr,
                layout.gateBlockM, layout.gateBlockK,
                layout.downBlockM, layout.downBlockK};
            return FastllmCudaMoeFP8CacheCompute(
                input, gateOutput, output, view, cache->routeSlots, scores, topk);
        }
        switch (input.dataType) {
            case fastllm::DataType::FLOAT32:
                return LaunchNVFP4Cache<float>(input, gateOutput, output, layout, *cache, scores, topk);
            case fastllm::DataType::FLOAT16:
                return LaunchNVFP4Cache<half>(input, gateOutput, output, layout, *cache, scores, topk);
            case fastllm::DataType::BFLOAT16:
                return LaunchNVFP4Cache<__nv_bfloat16>(input, gateOutput, output, layout, *cache, scores, topk);
            default:
                return false;
        }
    };
    if (rows == 1) {
        return computeRow(input, gateOutput, output, indices, scores);
    }

    // Each row completes lookup, refill and compute on the same stream
    // before the next row reuses route metadata or evicts an expert. This
    // also preserves the single-token kernels' reduction order in a graph.
    fastllm::Data inputRow, gateRow, outputRow;
    for (int row = 0; row < rows; ++row) {
        inputRow.FakeFrom(input, size_t(row) * layout.hidden * input.unitSize);
        inputRow.Resize({1, layout.hidden});
        inputRow.dataDeviceIds = input.dataDeviceIds;
        gateRow.FakeFrom(gateOutput, size_t(row) * topk * layout.inter * input.unitSize);
        gateRow.Resize({topk, layout.inter});
        gateRow.dataDeviceIds = input.dataDeviceIds;
        outputRow.FakeFrom(output, size_t(row) * layout.hidden * input.unitSize);
        outputRow.Resize({1, layout.hidden});
        outputRow.dataDeviceIds = input.dataDeviceIds;
        if (!computeRow(inputRow, gateRow, outputRow,
                        indices + row * topk, scores + row * topk)) {
            return false;
        }
    }
    return true;
}
