#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

namespace fastllm {

// Timing and admission policy only: independent of weight encoding, memory
// placement and kernels. A backend reports measured CPU/compute/refill costs
// and executes the returned GPU count (resident experts first).
class MoeDecodeScheduler {
public:
    static constexpr int maxExperts = 16;
    struct Estimate {
        double us = 0;
        bool initialized = false;
        void Observe(double sample) {
            // Discard cold-start inflation promptly, but smooth ordinary
            // variation and limit the impact of isolated scheduling stalls.
            if (!initialized || sample < us * 0.5) us = sample;
            else us = us * 0.9 + std::min(sample, us * 3) * 0.1;
            initialized = true;
        }
    };
    std::array<Estimate, maxExperts + 1> cpu{}, compute{};
    Estimate refill, dispatch, ensure;
    uint64_t calls = 0;

    int SelectGpuCount(int topk, int hits, int layers = 1) const {
        // Populate each layer once, then calibrate splits across layer calls:
        // timings are shared, so repeating every split at every layer would
        // needlessly penalize the first tokens. Sparse prime-interval probes
        // avoid refreshing only a subset of layers (e.g. 1024 vs 48).
        layers = std::max(1, layers);
        if (calls < uint64_t(layers)) return topk;
        if (calls < uint64_t(layers + 2 * (topk + 1)))
            return topk - (calls - layers) % (topk + 1);
        if (calls % 1021 == 0) return topk - (calls / 1021) % (topk + 1);
        int selected = 0;
        double best = cpu[topk].us;
        for (int g = std::max(1, hits); g <= topk; ++g) {
            const double gpuUs = ensure.us + compute[g].us + (g - hits) * refill.us;
            const double expected = dispatch.us + std::max(cpu[topk - g].us, gpuUs);
            if (expected < best) { best = expected; selected = g; }
        }
        return selected;
    }
};
} // namespace fastllm
