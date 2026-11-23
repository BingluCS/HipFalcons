#include "h1.hpp"

#include <limits>

namespace hip_falcons {
namespace entropy_trial {
namespace {

constexpr unsigned kThreads = 64;

__global__ void h1_kernel(const void* packed, std::int16_t* output) {
    __shared__ unsigned char ok[h1_detail::kStreams];
    const unsigned lane = threadIdx.x;
    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * h1_detail::kStreams;
    const bool active = lane < h1_detail::kStreams;
    if (active) ok[lane] = h1_detail::decode_one(packed, output, first + lane);
    __syncthreads();
    if (lane == 0 && ok[0] && ok[1] && ok[2] && ok[3])
        h1_detail::prepare_delta_carries(packed, output, first);
    __syncthreads();
    if (active && ok[0] && ok[1] && ok[2] && ok[3])
        h1_detail::apply_delta_carry(packed, output, first + lane);
}

}  // namespace

int launch_h1(const void* packed_device, std::int16_t* codes_device,
              std::size_t jobs, hipStream_t stream) noexcept {
    if (!jobs) return static_cast<int>(hipSuccess);
    if (!packed_device || !codes_device || jobs % h1_detail::kStreams)
        return static_cast<int>(hipErrorInvalidValue);
    const std::size_t blocks = jobs / h1_detail::kStreams;
    if (blocks > std::numeric_limits<unsigned>::max())
        return static_cast<int>(hipErrorInvalidValue);
    hipLaunchKernelGGL(h1_kernel, dim3(static_cast<unsigned>(blocks)), dim3(kThreads), 0, stream,
                       packed_device, codes_device);
    return static_cast<int>(hipGetLastError());
}

}  // namespace entropy_trial
}  // namespace hip_falcons
