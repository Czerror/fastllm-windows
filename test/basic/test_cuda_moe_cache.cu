#include "fastllm.h"
#include "fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

static void Check(cudaError_t state) {
    if (state != cudaSuccess) throw std::runtime_error(cudaGetErrorString(state));
}
static void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
static void AllocateGpu(fastllm::Data &data) {
    data.dataDevice = fastllm::DataDevice::CUDA;
    data.dataDeviceIds = {0};
    data.Allocate(false);
}

static size_t ExpertRecordBytes(const fastllm::Data &gate, const fastllm::Data &down) {
    const size_t downOffset = (gate.GetBytes() + 15) / 16 * 16;
    const size_t scalesOffset = (downOffset + down.GetBytes() + 15) / 16 * 16;
    const size_t scaleBytes = gate.dataType == fastllm::DataType::FP8_E4M3_BLOCK_128 ? 0 :
        (gate.scales.size() + down.scales.size()) * sizeof(float);
    return (scalesOffset + scaleBytes + 127) / 128 * 128;
}

template<class T>
static void Run(fastllm::DataType dtype, int hidden, int inter,
                fastllm::DataType weightType = fastllm::DataType::NVFP4_BLOCK_16_E4M3) {
    constexpr int experts = 32, topk = 10;
    std::vector<std::unique_ptr<fastllm::Data>> owned;
    std::vector<fastllm::Data *> weights(2 * (experts + 1), nullptr);
    for (int e = 0; e < experts; ++e) for (int part = 0; part < 2; ++part) {
        auto weight = std::make_unique<fastllm::Data>(weightType);
        int rows = part == 0 ? 2 * inter : hidden;
        int cols = part == 0 ? hidden : inter;
        weight->blockK = weightType == fastllm::DataType::FP8_E4M3 ? 128 : 1;
        weight->blockM = weightType == fastllm::DataType::NVFP4_BLOCK_16_E4M3 ? 16 : 128;
        weight->Resize({rows, cols}); weight->Allocate(false);
        // Constant decoded weights give an independent closed-form oracle.
        // Distinct down scales expose wrong table/slot selection.
        const float scale = part == 0 ? 1 : float(e + 1) / 32;
        if (weightType == fastllm::DataType::NVFP4_BLOCK_16_E4M3) {
            size_t packed = size_t(rows) * ((cols + 1) / 2);
            std::memset(weight->cpuData, 0x22, packed);
            std::memset(weight->cpuData + packed, 0x38, weight->GetBytes() - packed);
            weight->scales = part == 0 ? std::vector<float>{1, 1}
                                      : std::vector<float>{scale};
        } else if (weightType == fastllm::DataType::FP8_E4M3) {
            std::memset(weight->cpuData, 0x38, weight->GetBytes());
            weight->scales.assign(size_t((rows + 127) / 128) * ((cols + 127) / 128), scale);
        } else {
            const int scaleCols = (cols + 127) / 128;
            const size_t pitch = cols + scaleCols * sizeof(float);
            for (int row = 0; row < rows; ++row) {
                uint8_t *p = weight->cpuData + row * pitch;
                for (int col = 0; col < scaleCols; ++col) {
                    uint8_t *block = p + col * (128 + sizeof(float));
                    const int count = std::min(128, cols - col * 128);
                    std::memset(block, 0x38, count);
                    std::memcpy(block + count, &scale, sizeof(float));
                }
            }
        }
        weights[2 * (e + 1) + part] = weight.get();
        owned.push_back(std::move(weight));
    }
    FastllmCudaMoeCacheLayer layer{weights.data(), int(weights.size())};
    fastllm::SetMoeCudaCacheBytes(0);
    Require(!FastllmCudaPrepareMoeCache(&layer, 1), "disabled cache prepared");
    fastllm::SetMoeCudaCacheBytes(1);
    Require(!FastllmCudaPrepareMoeCache(&layer, 1), "insufficient budget accepted");
    const size_t stride = ExpertRecordBytes(*weights[2], *weights[3]);
    fastllm::SetMoeCudaCacheBytes(16 * stride);
    if (weightType == fastllm::DataType::FP8_E4M3) {
        const float saved = weights[2]->scales.back();
        weights[2]->scales.pop_back();
        Require(!FastllmCudaPrepareMoeCache(&layer, 1), "missing FP8 scale accepted");
        weights[2]->scales.push_back(saved);
        weights[2]->blockM = 127;
        Require(!FastllmCudaPrepareMoeCache(&layer, 1), "unaligned FP8 scale block accepted");
        weights[2]->blockM = 128;
    }
    Require(FastllmCudaPrepareMoeCache(&layer, 1), "valid table rejected");

    fastllm::Data input(dtype, {1, hidden}), index(fastllm::DataType::INT32, {1, topk});
    fastllm::Data score(fastllm::DataType::FLOAT32, {1, topk}), gate, output;
    AllocateGpu(input); AllocateGpu(index); AllocateGpu(score);
    std::vector<T> activation(hidden, T(1.0f / 128));
    std::vector<float> scores(topk, 1.0f / topk);
    Check(cudaMemcpy(input.cudaData, activation.data(), hidden * sizeof(T), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(score.cudaData, scores.data(), topk * sizeof(float), cudaMemcpyHostToDevice));
    auto supported = [&] {
        return FastllmCudaCanRunMoeCacheBatch1(input, index, score, weights.data(),
                                              weights.size(), fastllm::MoeGateSwiglu);
    };
    Require(supported(), "valid decode rejected");
    input.dims[1] = hidden + 1;
    Require(!supported(), "mismatched hidden width accepted");
    input.dims[1] = hidden;

    auto launch = [&] {
        Require(FastllmCudaMergeMOECacheBatch1(input, gate, output, weights.data(),
                    weights.size(), static_cast<int32_t *>(index.cudaData),
                    static_cast<float *>(score.cudaData), topk), "decode failed");
    };
    cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
    for (int pass = 0; pass < 5; ++pass) {
        std::vector<int32_t> ids(topk);
        for (int k = 0; k < topk; ++k) ids[k] = (pass * 9 + k) % experts;
        Check(cudaMemcpy(index.cudaData, ids.data(), topk * sizeof(int32_t), cudaMemcpyHostToDevice));
        if (pass == 0) {
            launch(); Check(cudaStreamSynchronize(cudaStreamPerThread));
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            launch();
            Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        }
        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        std::vector<T> actual(hidden);
        Check(cudaMemcpy(actual.data(), output.cudaData, hidden * sizeof(T), cudaMemcpyDeviceToHost));
        float g = float(T(hidden / 128.0f));
        float activated = float(T((g / (1 + std::exp(-g))) * g));
        float expected = 0;
        for (int id : ids) expected += float(T(inter * activated * ((id + 1) / 32.0f))) / topk;
        expected = float(T(expected));
        float tolerance = dtype == fastllm::DataType::FLOAT32 ? 1e-5f :
                          dtype == fastllm::DataType::FLOAT16 ? 0.002f : 0.012f;
        for (T v : actual) Require(std::fabs(float(v) - expected) <= tolerance * std::max(1.0f, std::fabs(expected)),
                                  "expert output mismatch");
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    FastllmCudaReleaseMoeCache(weights.data(), weights.size());
    Require(!supported(), "released table still registered");
    FastllmCudaReleaseMoeCache(weights.data(), weights.size());
    fastllm::SetMoeCudaCacheBytes(0);
    std::printf("PASS adapter weight=%d dtype=%d hidden=%d inter=%d, graph/eviction/release/validation\n",
                int(weightType), int(dtype), hidden, inter);
}

template<class T>
static void CompareFP8(fastllm::DataType dtype, fastllm::DataType weightType,
                       std::vector<std::unique_ptr<fastllm::Data>> &keepAlive) {
    constexpr int hidden = 256, inter = 128, experts = 24, topk = 7, tables = 2;
    std::mt19937 rng(42);
    std::vector<std::unique_ptr<fastllm::Data>> owned;
    std::vector<fastllm::Data *> weights[tables];
    FastllmCudaMoeCacheLayer layers[tables];
    for (int table = 0; table < tables; ++table) {
        weights[table].resize(2 * (experts + 1), nullptr);
        for (int e = 0; e < experts; ++e) for (int part = 0; part < 2; ++part) {
            auto w = std::make_unique<fastllm::Data>(weightType);
            const int rows = part == 0 ? inter * 2 : hidden;
            const int cols = part == 0 ? hidden : inter;
            w->blockM = w->blockK = 128;
            w->Resize({rows, cols}); w->Allocate(false);
            if (weightType == fastllm::DataType::FP8_E4M3) {
                for (size_t i = 0; i < w->GetBytes(); ++i)
                    w->cpuData[i] = uint8_t(0x20 + rng() % 32) | ((rng() & 1) ? 0x80 : 0);
                w->scales.resize(size_t((rows + 127) / 128) * ((cols + 127) / 128));
                for (float &scale : w->scales) scale = float(1 + rng() % 7) / 16;
            } else {
                size_t offset = 0;
                for (int row = 0; row < rows; ++row) for (int col = 0; col < cols; col += 128) {
                    for (int i = 0; i < 128; ++i)
                        w->cpuData[offset++] = uint8_t(0x20 + rng() % 32) | ((rng() & 1) ? 0x80 : 0);
                    float scale = float(1 + rng() % 7) / 16;
                    std::memcpy(w->cpuData + offset, &scale, sizeof(float)); offset += sizeof(float);
                }
                Require(offset == w->GetBytes(), "packed FP8 test layout mismatch");
            }
            weights[table][2 * (e + 1) + part] = w.get();
            owned.push_back(std::move(w));
        }
        layers[table] = {weights[table].data(), int(weights[table].size())};
    }
    const size_t stride = ExpertRecordBytes(*weights[0][2], *weights[0][3]);
    fastllm::SetMoeCudaCacheBytes(16 * stride);
    Require(FastllmCudaPrepareMoeCache(layers, tables), "FP8 reference cache prepare failed");
    // The cache owns a compact host snapshot; the original tables can now
    // serve as an independent all-resident GPU backend reference.
    for (auto &w : owned) w->ToDevice(fastllm::DataDevice::CUDA);
    fastllm::Data input(dtype, {1, hidden}), ids(fastllm::DataType::INT32, {1, topk});
    fastllm::Data scores(fastllm::DataType::FLOAT32, {1, topk});
    fastllm::Data gate[tables], output[tables], refGate[tables], refOutput[tables];
    AllocateGpu(input); AllocateGpu(ids); AllocateGpu(scores);
    std::vector<T> activation(hidden);
    std::vector<float> score(topk);
    cudaGraph_t graph[tables]{}; cudaGraphExec_t exec[tables]{};
    for (int pass = 0; pass < 50; ++pass) {
        const int table = pass % tables;
        std::vector<int32_t> indices(topk);
        for (auto &v : activation) v = T((int(rng() % 31) - 15) / 64.0f);
        for (int k = 0; k < topk; ++k) {
            indices[k] = pass < 4 ? k : rng() % experts;
            score[k] = float(1 + rng() % 8) / 32;
        }
        Check(cudaMemcpy(input.cudaData, activation.data(), hidden * sizeof(T), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(ids.cudaData, indices.data(), topk * sizeof(int32_t), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(scores.cudaData, score.data(), topk * sizeof(float), cudaMemcpyHostToDevice));
        auto launch = [&] {
            Require(FastllmCudaMergeMOECacheBatch1(input, gate[table], output[table],
                weights[table].data(), weights[table].size(), static_cast<int32_t *>(ids.cudaData),
                static_cast<float *>(scores.cudaData), topk), "FP8 cached compute failed");
        };
        if (!exec[table]) {
            launch(); Check(cudaStreamSynchronize(cudaStreamPerThread));
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            launch(); Check(cudaStreamEndCapture(cudaStreamPerThread, &graph[table]));
            Check(cudaGraphInstantiate(&exec[table], graph[table], nullptr, nullptr, 0));
        }
        if (pass % 3 == 0) launch();
        else Check(cudaGraphLaunch(exec[table], cudaStreamPerThread));
        bool ok;
        auto *index = static_cast<int32_t *>(ids.cudaData);
        auto *s = static_cast<float *>(scores.cudaData);
        if (weightType == fastllm::DataType::FP8_E4M3) {
            if constexpr (std::is_same_v<T, half>)
                ok = FastllmCudaHalfMergeMOEFP8E4M3Batch1Indexed(input, refGate[table], refOutput[table],
                    weights[table].data(), weights[table].size(), index, s, topk, hidden, inter, false);
            else
                ok = FastllmCudaBFloat16MergeMOEFP8E4M3Batch1Indexed(input, refGate[table], refOutput[table],
                    weights[table].data(), weights[table].size(), index, s, topk, hidden, inter, false);
        } else {
            if constexpr (std::is_same_v<T, half>)
                ok = FastllmCudaHalfMergeMOEFP8E4M3Block128Batch1Indexed(input, refGate[table], refOutput[table],
                    weights[table].data(), weights[table].size(), index, s, topk, hidden, inter);
            else
                ok = FastllmCudaBFloat16MergeMOEFP8E4M3Block128Batch1Indexed(input, refGate[table], refOutput[table],
                    weights[table].data(), weights[table].size(), index, s, topk, hidden, inter);
        }
        Require(ok, "resident FP8 reference failed");
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        for (bool intermediate : {false, true}) {
            auto &actual = intermediate ? gate[table] : output[table];
            auto &expected = intermediate ? refGate[table] : refOutput[table];
            std::vector<uint8_t> a(actual.GetBytes()), b(expected.GetBytes());
            Check(cudaMemcpy(a.data(), actual.cudaData, a.size(), cudaMemcpyDeviceToHost));
            Check(cudaMemcpy(b.data(), expected.cudaData, b.size(), cudaMemcpyDeviceToHost));
            Require(a == b, "cached FP8 differs bitwise from resident GPU reference");
        }
    }
    for (int t = 0; t < tables; ++t) {
        Check(cudaGraphExecDestroy(exec[t])); Check(cudaGraphDestroy(graph[t]));
        FastllmCudaReleaseMoeCache(weights[t].data(), weights[t].size());
    }
    // Existing resident-table registries key by Data address. Keep those
    // reference weights alive until all cases finish so addresses cannot alias.
    for (auto &w : owned) keepAlive.push_back(std::move(w));
    fastllm::SetMoeCudaCacheBytes(0);
    std::printf("PASS FP8 resident-reference weight=%d dtype=%d: 50 steps, two tables, duplicates, eviction, eager/graph, bitwise gate/output\n",
                int(weightType), int(dtype));
}

int main() {
    try {
        for (bool wide : {false, true}) {
            int hidden = wide ? 128 : 19, inter = wide ? 128 : 23;
            Run<float>(fastllm::DataType::FLOAT32, hidden, inter);
            Run<half>(fastllm::DataType::FLOAT16, hidden, inter);
            Run<__nv_bfloat16>(fastllm::DataType::BFLOAT16, hidden, inter);
        }
        for (auto weightType : {fastllm::DataType::FP8_E4M3,
                                fastllm::DataType::FP8_E4M3_BLOCK_128}) {
            const int widerHidden = weightType == fastllm::DataType::FP8_E4M3_BLOCK_128 ? 256 : 260;
            for (int hidden : {128, widerHidden}) {
                Run<float>(fastllm::DataType::FLOAT32, hidden, 128, weightType);
                Run<half>(fastllm::DataType::FLOAT16, hidden, 128, weightType);
                Run<__nv_bfloat16>(fastllm::DataType::BFLOAT16, hidden, 128, weightType);
            }
        }
        std::vector<std::unique_ptr<fastllm::Data>> references;
        for (auto weightType : {fastllm::DataType::FP8_E4M3, fastllm::DataType::FP8_E4M3_BLOCK_128}) {
            CompareFP8<half>(fastllm::DataType::FLOAT16, weightType, references);
            CompareFP8<__nv_bfloat16>(fastllm::DataType::BFLOAT16, weightType, references);
        }
        std::puts("ALL_PASS"); return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
}
