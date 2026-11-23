#ifndef HIP_FALCONS_RECONSTRUCT_HPP
#define HIP_FALCONS_RECONSTRUCT_HPP

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>
#include "stream_params.hpp"

namespace hip_falcons {

enum class ReconstructStatus : int {
    success = 0,
    unpredictable_count_mismatch = 1,
    invalid_escape_map = 2,
};

struct ReconstructResult {
    ReconstructStatus status;
    std::size_t expected_unpredictable_count;
};

// Both entry points reconstruct serial, non-tuning, 1D float32 LINEAR streams
// with radius 32768. The caller parses the container and entropy-decodes the
// signed int16 codes. Codes are in traversal order; unpredictable values are
// all anchors first, followed by the values for -32768 codes in code order.
// An anchor stride >= num_elements is treated as zero, as in Falcons. With no
// anchors, code zero reconstructs element zero.
//
// Supported: num_elements in [1, UINT32_MAX], zero or power-of-two anchor
// stride, finite positive error_bound, finite alpha and beta (beta positive if
// alpha >= 1), and finite positive derived quantization steps. The code count
// must equal num_elements minus the effective anchor count. Reject OMP, cubic,
// other dimensions/types, and streams made by the legacy tuning traversal.
// Compile without fast-math and use round-to-nearest, gradual-underflow IEEE
// arithmetic on host and device. Level bounds use host std::pow; host libm or
// device denormal differences can affect bitwise identity.
//
// Enqueue the level-parallel GPU path; block_size must equal num_elements.
// All pointers are device pointers, must not overlap, and must remain valid
// until the stream completes. Output holds num_elements floats. An input may
// be null only when its count is zero; output is always required.
//
// escape_code_positions must contain exactly the zero-based code indices of
// every -32768 code, in strictly increasing order. The CPU entropy stage must
// build and validate this map and verify that effective anchor count plus
// escape_count equals unpredictable_count. The function checks the counts but
// does not validate device map contents; malformed maps produce unspecified
// output. Returns hipErrorInvalidValue for invalid host-side metadata/counts,
// or a HIP launch error. The call is asynchronous; also check stream
// synchronization for execution errors.
hipError_t reconstruct_1d_linear_f32_parallel_async(
    const Reconstruct1DParams& params,
    const std::int16_t* codes, std::size_t code_count,
    const float* unpredictable, std::size_t unpredictable_count,
    const std::uint32_t* escape_code_positions, std::size_t escape_count,
    float* output,
    hipStream_t stream = nullptr) noexcept;

// Synchronous host check of the same level plan and per-point functions as
// the GPU path. All pointers, including the escape map, are host pointers;
// the pointer/count, block-size, and non-overlap requirements are the same.
// The host check also validates the complete escape map before reconstruction.
// On hipSuccess, inspect result->status. An invalid map or unpredictable-count
// mismatch leaves output untouched. On hipErrorInvalidValue, result is
// unspecified. This function does not require a HIP device.
hipError_t reconstruct_1d_linear_f32_host(
    const Reconstruct1DParams& params,
    const std::int16_t* codes, std::size_t code_count,
    const float* unpredictable, std::size_t unpredictable_count,
    const std::uint32_t* escape_code_positions, std::size_t escape_count,
    float* output, ReconstructResult* result) noexcept;

}  // namespace hip_falcons

#endif  // HIP_FALCONS_RECONSTRUCT_HPP
