#include "reconstruct.hpp"

#include <cmath>
#include <limits>

namespace hip_falcons {
namespace {

constexpr int kRadius = 32768;
constexpr int kMaxLevel = 32;
constexpr unsigned int kThreads = 256;

struct Plan {
    std::size_t num_elements;
    std::size_t anchor_stride;
    std::size_t anchor_count;
    int top_level;
    // Entry zero is the original quantizer step, used for the first code.
    double steps[kMaxLevel + 1];
};

struct Level {
    std::size_t stride;
    std::size_t n;
    std::size_t odd_count;
    std::size_t code_offset;
    std::size_t anchor_count;
    double step;
};

__host__ __device__ float recover(float prediction, std::int16_t code, double step,
                                  const float* unpredictable, std::size_t& unpred_index) {
    if (code == -kRadius) {
        return unpredictable[unpred_index++];
    }
    return static_cast<float>(fma(static_cast<double>(code), step,
                                  static_cast<double>(prediction)));
}

__host__ __device__ float linear_prediction(float left, float right) {
    // Keep the float add and multiply separate, as in interp_linear<float>.
    volatile float sum = left + right;
    return sum * 0.5f;
}

__host__ __device__ float edge_prediction(float previous, float last) {
    // Falcons rounds 1.5 * last to float before the fused float operation.
    volatile float scaled_last = 1.5f * last;
    return fmaf(-0.5f, previous, scaled_last);
}

__host__ __device__ bool recover_mapped(
    float prediction, std::int16_t code, double step, std::size_t code_index,
    std::size_t anchor_count, const float* unpredictable,
    const std::uint32_t* escape_positions, std::size_t escape_count, float& value) {
    std::size_t unpred_index = 0;
    if (code == -kRadius) {
        std::size_t lo = 0, hi = escape_count;
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (escape_positions[mid] < code_index) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo == escape_count || escape_positions[lo] != code_index) {
            return false;
        }
        unpred_index = anchor_count + lo;
    }
    value = recover(prediction, code, step, unpredictable, unpred_index);
    return true;
}

__host__ __device__ bool initialize_point(
    std::size_t index, std::size_t anchor_stride, std::size_t anchor_count,
    double seed_step, const std::int16_t* codes, const float* unpredictable,
    const std::uint32_t* escape_positions, std::size_t escape_count, float* output) {
    if (anchor_count != 0) {
        if (index < anchor_count) {
            output[index * anchor_stride] = unpredictable[index];
        }
    } else if (index == 0) {
        return recover_mapped(0.0f, codes[0], seed_step, 0, 0, unpredictable,
                              escape_positions, escape_count, output[0]);
    }
    return true;
}

__host__ __device__ bool reconstruct_level_point(
    const Level& level, std::size_t index, const std::int16_t* codes,
    const float* unpredictable, const std::uint32_t* escape_positions,
    std::size_t escape_count, float* output) {
    if (index >= level.odd_count) {
        return true;
    }
    const std::size_t pos = (2 * index + 1) * level.stride;
    float prediction;
    if (index + 1 < level.odd_count) {
        prediction = linear_prediction(output[pos - level.stride], output[pos + level.stride]);
    } else {
        const std::size_t even_count = level.n - level.odd_count;
        const std::size_t last_even = 2 * (even_count - 1) * level.stride;
        prediction = level.n < 3
            ? output[last_even]
            : edge_prediction(output[2 * (even_count - 2) * level.stride], output[last_even]);
    }
    const std::size_t code_index = level.code_offset + index;
    return recover_mapped(prediction, codes[code_index], level.step, code_index,
                          level.anchor_count, unpredictable, escape_positions,
                          escape_count, output[pos]);
}

__global__ void initialize_kernel(
    std::size_t anchor_stride, std::size_t anchor_count, double seed_step,
    const std::int16_t* codes, const float* unpredictable,
    const std::uint32_t* escape_positions, std::size_t escape_count, float* output) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    initialize_point(index, anchor_stride, anchor_count, seed_step, codes, unpredictable,
                     escape_positions, escape_count, output);
}

__global__ void reconstruct_level_kernel(
    Level level, const std::int16_t* codes, const float* unpredictable,
    const std::uint32_t* escape_positions, std::size_t escape_count, float* output) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    reconstruct_level_point(level, index, codes, unpredictable, escape_positions,
                            escape_count, output);
}

Level make_level(const Plan& plan, int level, std::size_t code_offset) {
    const std::size_t stride = std::size_t{1} << (level - 1);
    const std::size_t n = (plan.num_elements - 1) / stride + 1;
    return {stride, n, n / 2, code_offset, plan.anchor_count, plan.steps[level]};
}

unsigned int grid_size(std::size_t count) {
    return static_cast<unsigned int>(1 + (count - 1) / kThreads);
}

hipError_t prepare_plan(
    const Reconstruct1DParams& params,
    const std::int16_t* codes, std::size_t code_count,
    const float* unpredictable, std::size_t unpredictable_count,
    const std::uint32_t* escape_positions, std::size_t escape_count,
    float* output, Plan& plan) noexcept {
    if (params.num_elements == 0 ||
        params.num_elements > std::numeric_limits<std::uint32_t>::max() ||
        params.block_size != params.num_elements ||
        (params.anchor_stride != 0 &&
         (params.anchor_stride & (params.anchor_stride - 1)) != 0) ||
        params.radius != kRadius || !std::isfinite(params.error_bound) ||
        params.error_bound <= 0 || !std::isfinite(params.eb_alpha) ||
        !std::isfinite(params.eb_beta) ||
        (params.eb_alpha >= 1 && params.eb_beta <= 0) ||
        (code_count != 0 && codes == nullptr) ||
        (unpredictable_count != 0 && unpredictable == nullptr) ||
        (escape_count != 0 && escape_positions == nullptr) ||
        escape_count > code_count || output == nullptr) {
        return hipErrorInvalidValue;
    }

    plan.num_elements = params.num_elements;
    plan.anchor_stride = params.anchor_stride >= params.num_elements ? 0 : params.anchor_stride;
    for (std::size_t value = params.num_elements - 1; value != 0; value >>= 1) {
        ++plan.top_level;
    }
    if (plan.anchor_stride != 0) {
        plan.anchor_count = (params.num_elements - 1) / plan.anchor_stride + 1;
        int anchor_level = 1;
        for (std::size_t value = plan.anchor_stride; value > 1; value >>= 1) {
            ++anchor_level;
        }
        if (anchor_level <= plan.top_level) {
            plan.top_level = anchor_level;
        }
        --plan.top_level;
    }
    if (code_count != params.num_elements - plan.anchor_count) {
        return hipErrorInvalidValue;
    }

    plan.steps[0] = 2 * params.error_bound;
    if (!std::isfinite(plan.steps[0]) || plan.steps[0] <= 0) {
        return hipErrorInvalidValue;
    }
    for (int level = 1; level <= plan.top_level; ++level) {
        double level_eb = params.error_bound;
        if (params.eb_alpha < 0) {
            if (level >= 3) {
                level_eb = params.error_bound * 0.5;
            }
        } else if (params.eb_alpha >= 1) {
            double ratio = std::pow(params.eb_alpha, level - 1);
            if (ratio > params.eb_beta) {
                ratio = params.eb_beta;
            }
            level_eb = params.error_bound / ratio;
        }
        plan.steps[level] = 2 * level_eb;
        if (!std::isfinite(plan.steps[level]) || plan.steps[level] <= 0) {
            return hipErrorInvalidValue;
        }
    }
    return hipSuccess;
}

}  // namespace

hipError_t reconstruct_1d_linear_f32_parallel_async(
    const Reconstruct1DParams& params,
    const std::int16_t* codes, std::size_t code_count,
    const float* unpredictable, std::size_t unpredictable_count,
    const std::uint32_t* escape_positions, std::size_t escape_count,
    float* output, hipStream_t stream) noexcept {
    Plan plan{};
    hipError_t status = prepare_plan(params, codes, code_count, unpredictable,
                                     unpredictable_count, escape_positions, escape_count,
                                     output, plan);
    if (status != hipSuccess) {
        return status;
    }
    if (plan.anchor_count + escape_count != unpredictable_count) {
        return hipErrorInvalidValue;
    }

    const std::size_t init_count = plan.anchor_count != 0 ? plan.anchor_count : 1;
    hipLaunchKernelGGL(initialize_kernel, dim3(grid_size(init_count)), dim3(kThreads), 0, stream,
                       plan.anchor_stride, plan.anchor_count, plan.steps[0], codes,
                       unpredictable, escape_positions, escape_count, output);
    status = hipGetLastError();
    if (status != hipSuccess) {
        return status;
    }

    std::size_t code_offset = plan.anchor_count != 0 ? 0 : 1;
    for (int level = plan.top_level; level > 0; --level) {
        const Level current = make_level(plan, level, code_offset);
        if (current.odd_count != 0) {
            hipLaunchKernelGGL(reconstruct_level_kernel, dim3(grid_size(current.odd_count)),
                               dim3(kThreads), 0, stream, current, codes, unpredictable,
                               escape_positions, escape_count, output);
            status = hipGetLastError();
            if (status != hipSuccess) {
                return status;
            }
        }
        code_offset += current.odd_count;
    }
    return hipSuccess;
}

hipError_t reconstruct_1d_linear_f32_host(
    const Reconstruct1DParams& params,
    const std::int16_t* codes, std::size_t code_count,
    const float* unpredictable, std::size_t unpredictable_count,
    const std::uint32_t* escape_positions, std::size_t escape_count,
    float* output, ReconstructResult* result) noexcept {
    if (result == nullptr) {
        return hipErrorInvalidValue;
    }
    Plan plan{};
    const hipError_t status = prepare_plan(params, codes, code_count, unpredictable,
                                           unpredictable_count, escape_positions, escape_count,
                                           output, plan);
    if (status != hipSuccess) {
        return status;
    }
    result->expected_unpredictable_count = plan.anchor_count + escape_count;
    std::size_t rank = 0;
    for (std::size_t i = 0; i < code_count; ++i) {
        if (codes[i] == -kRadius) {
            if (rank == escape_count || escape_positions[rank] != i) {
                result->status = ReconstructStatus::invalid_escape_map;
                return hipSuccess;
            }
            ++rank;
        }
    }
    if (rank != escape_count) {
        result->status = ReconstructStatus::invalid_escape_map;
        return hipSuccess;
    }
    if (result->expected_unpredictable_count != unpredictable_count) {
        result->status = ReconstructStatus::unpredictable_count_mismatch;
        return hipSuccess;
    }

    const std::size_t init_count = plan.anchor_count != 0 ? plan.anchor_count : 1;
    for (std::size_t i = 0; i < init_count; ++i) {
        if (!initialize_point(i, plan.anchor_stride, plan.anchor_count, plan.steps[0],
                              codes, unpredictable, escape_positions, escape_count, output)) {
            result->status = ReconstructStatus::invalid_escape_map;
            return hipSuccess;
        }
    }
    std::size_t code_offset = plan.anchor_count != 0 ? 0 : 1;
    for (int level = plan.top_level; level > 0; --level) {
        const Level current = make_level(plan, level, code_offset);
        for (std::size_t i = 0; i < current.odd_count; ++i) {
            if (!reconstruct_level_point(current, i, codes, unpredictable,
                                         escape_positions, escape_count, output)) {
                result->status = ReconstructStatus::invalid_escape_map;
                return hipSuccess;
            }
        }
        code_offset += current.odd_count;
    }
    result->status = ReconstructStatus::success;
    return hipSuccess;
}

}  // namespace hip_falcons
