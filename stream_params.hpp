#pragma once

#include <cstddef>
#include <cstdint>

namespace hip_falcons {
struct Reconstruct1DParams {
    std::size_t num_elements;
    std::uint32_t block_size;
    std::size_t anchor_stride;
    double error_bound;
    double eb_alpha;
    double eb_beta;
    int radius;
};
}
