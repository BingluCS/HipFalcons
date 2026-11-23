#ifndef HIP_FALCONS_ND_RECONSTRUCT_HPP
#define HIP_FALCONS_ND_RECONSTRUCT_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include <hip/hip_runtime.h>

namespace hip_falcons {

// Metadata from a serial, non-tuning, float32, LINEAR
// InterpolationDecomposition and LinearQuantizer. The quantizer radius must
// be 32768, as checked by the caller when unpacking the stream. Dimensions
// are in Falcons order.
struct NDReconstructParams {
    std::vector<std::size_t> dimensions;
    std::size_t block_size;
    std::size_t anchor_stride;
    int direction;
    double error_bound;
    double eb_alpha;
    double eb_beta;
};

struct NDReconstructTimings {
    double plan_ms = 0;
    double upload_ms = 0;
    double kernel_ms = 0;
    double download_ms = 0;
    double other_ms = 0;
    // Extends the structure; callers using timings must rebuild with this header.
    double to_device_ms = 0;
};

// The following two synchronous entry points use host-memory pointers; output
// has product(dimensions) elements and must not overlap either input. Codes are
// signed int16 values in CPU traversal order. Unpredictable values contain
// anchors first in CPU order, followed by escape values in code order.
// A null input pointer is permitted only for an empty input. Supported ranks
// are 2 and 3, with direction 0..1 and 0..5 respectively; every extent must
// be positive and their product at most UINT32_MAX. Anchor stride is zero or
// a power of two; a stride at least as large as the largest dimension is
// disabled, as in Falcons. The GPU entry point requires block_size to equal
// the largest dimension, as required by the stream parser; the host reference
// also accepts other block sizes when the reference planner supports them.
// The error bound and each derived quantization step
// must be finite and positive; alpha and beta must be finite, with beta
// positive when alpha >= 1. The planner also rejects a traversal whose
// topology does not meet its unique-output and dependency guarantees.
// Both functions validate metadata, code count and unpredictable count
// before writing output.
// The host reference does not require an available GPU. The GPU function
// executes reconstruction on the device and returns only after download.
//
// Returns hipErrorInvalidValue for invalid or unsupported input, a HIP error
// for device failures, and hipErrorOutOfMemory for host allocation failure.
// On a device error output may have been modified. Compile without fast-math,
// using round-to-nearest, gradual-underflow IEEE arithmetic on host/device.
// Level bounds use host std::pow; host libm and device denormal differences
// may affect bitwise identity.
//
// Times are wall-clock milliseconds. plan_ms covers compact planning; upload_ms and
// download_ms include their transfer completion waits; kernel_ms includes
// launches and their completion wait. other_ms is the remaining wrapper time,
// including escape-map preparation, allocation, stream creation and cleanup.
// to_device_ms runs from GPU wrapper entry through successful completion of the
// last kernel synchronization, before download or cleanup; it is zero if that
// point is not reached.
hipError_t reconstruct_nd_linear_f32_host(
    const NDReconstructParams& params,
    const std::int16_t* codes, std::size_t code_count,
    const float* unpredictable, std::size_t unpredictable_count,
    float* output) noexcept;

hipError_t reconstruct_nd_linear_f32_gpu(
    const NDReconstructParams& params,
    const std::int16_t* codes, std::size_t code_count,
    const float* unpredictable, std::size_t unpredictable_count,
    float* output, NDReconstructTimings* timings = nullptr) noexcept;

// Uses the same metadata and host-input contract as the GPU entry point above,
// but writes product(dimensions) floats directly to caller-owned device memory.
// The output must not overlap either input. The caller must provide sufficient
// device storage and keep the inputs, output and stream valid until return.
// The output and stream must belong to the current HIP device; stream may be
// nullptr for the default stream. Neither is freed or destroyed here. On
// success the stream is synchronized before return. On a HIP error, output may
// be partially written and cleanup attempts to finish any submitted work.
hipError_t reconstruct_nd_linear_f32_device_output(
    const NDReconstructParams& params,
    const std::int16_t* codes, std::size_t code_count,
    const float* unpredictable, std::size_t unpredictable_count,
    float* device_output, hipStream_t stream) noexcept;

}  // namespace hip_falcons

#endif  // HIP_FALCONS_ND_RECONSTRUCT_HPP
