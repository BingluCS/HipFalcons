#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace hip_falcons::entropy_trial {

// With a null scratch argument, query the required scratch size.
int collect_escapes(void* scratch, std::size_t& scratch_bytes,
                    const std::int16_t* codes, std::size_t code_count,
                    std::uint32_t* positions, std::uint32_t* count, hipStream_t stream);

}  // namespace hip_falcons::entropy_trial
