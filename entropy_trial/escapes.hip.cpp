#include "escapes.hpp"

#include <rocprim/device/device_select.hpp>
#include <rocprim/iterator/counting_iterator.hpp>

namespace hip_falcons::entropy_trial {
namespace {
struct IsEscape {
    const std::int16_t* codes;
    __device__ bool operator()(std::uint32_t index) const { return codes[index] == INT16_MIN; }
};
}  // namespace

int collect_escapes(void* scratch, std::size_t& scratch_bytes,
                    const std::int16_t* codes, std::size_t code_count,
                    std::uint32_t* positions, std::uint32_t* count, hipStream_t stream) {
    return static_cast<int>(rocprim::select(scratch, scratch_bytes,
        rocprim::counting_iterator<std::uint32_t>(0), positions, count, code_count,
        IsEscape{codes}, stream));
}
}  // namespace hip_falcons::entropy_trial
