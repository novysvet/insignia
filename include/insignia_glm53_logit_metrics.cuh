#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace insignia::glm53 {

// Full-vocabulary metrics for two FP32 logit vectors. Divergences and entropy
// use natural logarithms; JS includes the conventional one-half factor.
struct alignas(64) LogitMetrics {
    double left_max;
    double right_max;
    double left_logsumexp;
    double right_logsumexp;
    double left_mean;
    double right_mean;
    double left_centered_rms;
    double right_centered_rms;
    double mse;
    double centered_mse;
    double raw_cosine;
    double centered_cosine;
    double kl_left_right;
    double kl_right_left;
    double js;
    double left_entropy;
    double right_entropy;
    double left_top1_probability;
    double right_top1_probability;
    int32_t left_argmax;
    int32_t right_argmax;
};

static_assert(alignof(LogitMetrics) == 64);
static_assert(sizeof(LogitMetrics) == 192);

struct alignas(32) LogitRowStats {
    double maximum;
    double logsumexp;
    double top1_probability;
    int32_t argmax;
};

static_assert(sizeof(LogitRowStats) == 32);

// Workspace is tiny (at most ~33 KiB for the fixed launch geometry), reusable,
// and may be shared only after the preceding stream use has completed.
std::size_t logit_metrics_workspace_bytes(int count) noexcept;

// All pointers are device pointers. Every input logit must be finite; this
// focused prototype deliberately does not spend another pass validating the
// engine's finite-logit invariant. The result remains on device so callers
// download only one cache-line-aligned scalar record, never the logits.
cudaError_t logit_metrics_async(const float *left, const float *right, int count,
                                void *workspace, LogitMetrics *result,
                                cudaStream_t stream = nullptr) noexcept;

// Batched stable max/logsumexp/Top-1 for row-major logits (for DFlash's seven
// rows). This deliberately does not depend on the pairwise-metric workspace.
std::size_t logit_row_stats_workspace_bytes(int rows, int count) noexcept;
cudaError_t logit_row_stats_async(const float *logits, int rows, int count,
                                  void *workspace, LogitRowStats *result,
                                  cudaStream_t stream = nullptr) noexcept;

}  // namespace insignia::glm53
