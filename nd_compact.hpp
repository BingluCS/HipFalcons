#ifndef HIP_FALCONS_ND_COMPACT_HPP
#define HIP_FALCONS_ND_COMPACT_HPP

#include "nd_plan.hpp"

#include <cassert>

#if defined(__HIPCC__) || defined(__CUDACC__) || defined(__CUDA__)
#define HIP_FALCONS_COMPACT_HD __host__ __device__
#else
#define HIP_FALCONS_COMPACT_HD
#endif

namespace hip_falcons {

// Extents and strides are in outer-to-inner traversal order, not necessarily
// dimension order. A descriptor can be passed to a kernel by value.
enum class CompactPredictorMode : std::uint8_t {
    zero,
    midpoint,
    copy,
    extrapolate_two,
    last_extrapolate_three,
    last_extrapolate_bracket,
    cubic_odd,
    cubic_even,
};

struct CompactGroup {
    std::uint64_t count = 0;
    int level = 0;
    int pass = 0;
    double step = 0;
    std::uint64_t extent[3] = {1, 1, 1};
    std::uint64_t out_base = 0;
    std::uint64_t out_stride[3] = {0, 0, 0};
    std::uint64_t code_base = 0;
    std::uint64_t code_stride[3] = {0, 0, 0};
    std::uint64_t delta = 0;
    std::uint8_t axis_slot = 0;
    CompactPredictorMode mode = CompactPredictorMode::zero;
};

struct CompactAnchors {
    std::uint64_t count = 0;
    std::uint64_t extent[3] = {1, 1, 1};
    std::uint64_t out_stride[3] = {0, 0, 0};
};

struct CompactPlan {
    CompactAnchors anchors;
    std::vector<CompactGroup> groups;
    std::size_t code_count = 0;
};

// For LINEAR groups only. lane must be less than group.count. code_index is
// the serial CPU index, which need not be consecutive in reordered groups.
HIP_FALCONS_COMPACT_HD inline NDOperation compact_operation(
    const CompactGroup& group, std::size_t lane) noexcept {
    // Every extent and valid lane fits uint32, including a UINT32_MAX-point plan.
    std::uint32_t index[3];
    std::uint32_t remaining = static_cast<std::uint32_t>(lane);
    for (int d = 2; d >= 0; --d) {
        const auto extent = static_cast<std::uint32_t>(group.extent[d]);
        index[d] = remaining % extent;
        remaining /= extent;
    }
    std::uint64_t out = group.out_base;
    std::uint64_t code = group.code_base;
    for (int d = 0; d < 3; ++d) {
        out += index[d] * group.out_stride[d];
        code += index[d] * group.code_stride[d];
    }

    std::uint64_t left = 0, right = 0;
    NDPredictor kind = NDPredictor::zero;
    const bool last = index[group.axis_slot] + 1 == group.extent[group.axis_slot];
    switch (group.mode) {
    case CompactPredictorMode::zero:
        break;
    case CompactPredictorMode::copy:
        kind = NDPredictor::copy;
        left = out - group.delta;
        break;
    case CompactPredictorMode::extrapolate_two:
        kind = NDPredictor::extrapolate;
        left = out - 2 * group.delta;
        right = out - group.delta;
        break;
    case CompactPredictorMode::last_extrapolate_three:
        if (last) {
            kind = NDPredictor::extrapolate;
            left = out - 3 * group.delta;
            right = out - group.delta;
            break;
        }
        // All other points in the line are midpoints.
        // fall through
    case CompactPredictorMode::midpoint:
        kind = NDPredictor::midpoint;
        left = out - group.delta;
        right = out + group.delta;
        break;
    case CompactPredictorMode::last_extrapolate_bracket:
        kind = last ? NDPredictor::extrapolate : NDPredictor::midpoint;
        left = out - group.delta;
        right = out + group.delta;
        break;
    case CompactPredictorMode::cubic_odd:
    case CompactPredictorMode::cubic_even:
        assert(false && "compact_operation is only defined for LINEAR groups");
        break;
    }
    return NDOperation{static_cast<std::uint32_t>(out),
                       static_cast<std::uint32_t>(left),
                       static_cast<std::uint32_t>(right),
                       static_cast<std::uint32_t>(code), kind};
}

// lane must be less than anchors.count; the result addresses the lane-th
// unpredictable value in the CPU anchor order.
HIP_FALCONS_COMPACT_HD inline std::uint32_t compact_anchor_index(
    const CompactAnchors& anchors, std::size_t lane) noexcept {
    std::uint32_t remaining = static_cast<std::uint32_t>(lane);
    std::uint64_t out = 0;
    for (int d = 2; d >= 0; --d) {
        const auto extent = static_cast<std::uint32_t>(anchors.extent[d]);
        const std::uint32_t index = remaining % extent;
        remaining /= extent;
        out += index * anchors.out_stride[d];
    }
    return static_cast<std::uint32_t>(out);
}

namespace nd_compact_detail {

using Triple = std::array<std::uint64_t, 3>;

inline double checked_step(double error_bound) {
    const double step = 2 * error_bound;
    if (!std::isfinite(step) || step <= 0)
        throw std::invalid_argument("Invalid Falcons quantization step");
    return step;
}

class Builder {
public:
    explicit Builder(const NDPlanParams& params) : params_(params) {
        const std::size_t rank = params.dimensions.size();
        if (rank != 2 && rank != 3)
            throw std::invalid_argument("Falcons ND planner requires 2D or 3D");
        if (params.cubic && rank != 2)
            throw std::invalid_argument("Compact CUBIC planner requires 2D");
        rank_ = static_cast<int>(rank);
        if (params.block_size == 0 ||
            params.block_size > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("Invalid Falcons block size");
        if (params.anchor_stride != 0 &&
            (params.anchor_stride & (params.anchor_stride - 1)) != 0)
            throw std::invalid_argument("Falcons anchor stride must be a power of two");
        if (params.direction < 0 || params.direction >= (rank_ == 2 ? 2 : 6))
            throw std::invalid_argument("Invalid Falcons interpolation direction");
        if (!std::isfinite(params.error_bound) || params.error_bound <= 0 ||
            !std::isfinite(params.eb_alpha) || !std::isfinite(params.eb_beta) ||
            (params.eb_alpha >= 1 && params.eb_beta <= 0))
            throw std::invalid_argument("Invalid Falcons error-bound metadata");

        const std::uint64_t max_count = std::numeric_limits<std::uint32_t>::max();
        std::uint64_t max_dim = 1;
        for (int d = 0; d < rank_; ++d) {
            const std::size_t extent = params.dimensions[static_cast<std::size_t>(d)];
            if (extent == 0 || extent > max_count / total_)
                throw std::invalid_argument("Falcons dimensions exceed uint32 indexing");
            dims_[d] = extent;
            total_ *= extent;
            max_dim = std::max(max_dim, static_cast<std::uint64_t>(extent));
        }
        if (params.block_size != max_dim)
            throw std::invalid_argument("Compact Falcons planner requires block size equal to max dimension");
        pitch_[1] = dims_[2];
        pitch_[0] = dims_[1] * pitch_[1];

        for (int i = 0; i < params.direction; ++i)
            std::next_permutation(axes_.begin(), axes_.begin() + rank_);
        for (std::uint64_t n = max_dim - 1; n != 0; n >>= 1) ++top_level_;
        anchor_stride_ = params.anchor_stride < max_dim ? params.anchor_stride : 0;
        if (anchor_stride_ != 0) {
            int anchor_level = 1;
            for (std::uint64_t n = anchor_stride_; n > 1; n >>= 1) ++anchor_level;
            top_level_ = std::min(top_level_, anchor_level) - 1;
        }
        seed_step_ = checked_step(params.error_bound);
    }

    CompactPlan build() {
        if (anchor_stride_ != 0) {
            plan_.anchors.count = 1;
            for (int d = 0; d < 3; ++d) {
                plan_.anchors.extent[d] = (dims_[d] - 1) / anchor_stride_ + 1;
                plan_.anchors.out_stride[d] = anchor_stride_ * pitch_[d];
                plan_.anchors.count *= plan_.anchors.extent[d];
            }
        } else {
            CompactGroup seed;
            seed.count = 1;
            seed.pass = -1;
            seed.step = seed_step_;
            plan_.groups.push_back(seed);
            plan_.code_count = 1;
        }

        for (int level = top_level_; level > 0; --level) {
            const std::uint64_t stride = std::uint64_t{1} << (level - 1);
            const double step = level_step(level);
            level_groups(stride, level, step);
        }
        if (plan_.code_count != total_ - plan_.anchors.count)
            throw std::invalid_argument("Falcons traversal does not cover the output exactly once");
        return std::move(plan_);
    }

private:
    double level_step(int level) const {
        double eb = params_.error_bound;
        if (params_.eb_alpha < 0) {
            if (level >= 3) eb *= 0.5;
        } else if (params_.eb_alpha >= 1) {
            double ratio = std::pow(params_.eb_alpha, level - 1);
            if (ratio > params_.eb_beta) ratio = params_.eb_beta;
            eb /= ratio;
        }
        return checked_step(eb);
    }

    // code_extent describes the serial loop, which may include tails removed
    // from this group's rectangular launch.
    CompactGroup rectangle(const Triple& extents, const Triple& code_extents,
                           const Triple& spacing, const std::array<int, 3>& order,
                           int axis, std::uint64_t axis_start, std::uint64_t stride,
                           std::uint64_t code_base, int level, int pass, double step,
                           CompactPredictorMode mode) const {
        CompactGroup group;
        group.count = 1;
        group.level = level;
        group.pass = pass;
        group.step = step;
        group.out_base = axis_start * pitch_[axis];
        group.code_base = code_base;
        group.delta = stride * pitch_[axis];
        group.mode = mode;
        std::uint64_t code_stride = 1;
        for (int slot = 2; slot >= 0; --slot) {
            const int d = order[slot];
            group.extent[slot] = extents[d];
            group.out_stride[slot] = spacing[d] * pitch_[d];
            group.code_stride[slot] = code_stride;
            group.count *= extents[d];
            code_stride *= code_extents[d];
            if (d == axis) group.axis_slot = static_cast<std::uint8_t>(slot);
        }
        return group;
    }

    void level_groups(std::uint64_t stride, int level, double step) {
        if (params_.cubic) {
            cubic_level_groups(stride, level, step);
            return;
        }
        Triple spacing{{1, 1, 1}};
        const std::uint64_t twice = 2 * stride;
        for (int p = 1; p < rank_; ++p) spacing[axes_[p]] = twice;

        for (int pass = 0; pass < rank_; ++pass) {
            const int axis = axes_[pass];
            if (pass != 0) spacing[axes_[pass - 1]] = stride;
            const std::uint64_t n = (dims_[axis] - 1) / stride + 1;
            if (n <= 1) continue;

            const bool specialized = rank_ == 2 || params_.direction == 0 || params_.direction == 5;
            const bool fastest = specialized && axis == rank_ - 1;
            const bool gathered = specialized && !fastest;
            const bool skip = gathered && level == 1 && pass == rank_ - 1;
            const std::uint64_t odd_count = n / 2;
            const bool tail = !fastest && (n % 2 == 0) && n >= 4 && !skip;

            Triple extents{{1, 1, 1}};
            Triple out_spacing = spacing;
            out_spacing[axis] = twice;
            for (int d = 0; d < 3; ++d)
                extents[d] = (dims_[d] - 1) / out_spacing[d] + 1;
            extents[axis] = odd_count - (tail ? 1 : 0);

            std::array<int, 3> order{{0, 1, 2}};
            if (gathered && rank_ == 3 && axis == 0) order = {{1, 0, 2}};
            const std::uint64_t base = plan_.code_count;
            Triple code_extents = extents;
            if (tail && gathered) code_extents[axis] = odd_count;
            CompactPredictorMode mode = CompactPredictorMode::midpoint;
            if (n == 2) mode = CompactPredictorMode::copy;
            else if (fastest) mode = n % 2 == 0
                ? CompactPredictorMode::last_extrapolate_three
                : CompactPredictorMode::last_extrapolate_bracket;
            else if (skip && n % 2 == 0) mode = CompactPredictorMode::last_extrapolate_three;

            CompactGroup main = rectangle(extents, code_extents, out_spacing, order,
                                          axis, stride, stride, base, level, pass, step, mode);
            plan_.groups.push_back(main);
            plan_.code_count += static_cast<std::size_t>(main.count);

            if (tail) {
                Triple tail_extents = extents;
                tail_extents[axis] = 1;
                const std::uint64_t tail_code_base = gathered
                    ? base + (odd_count - 1) * main.code_stride[main.axis_slot]
                    : base + main.count;
                const Triple& tail_code_extents = gathered ? code_extents : tail_extents;
                CompactGroup edge = rectangle(tail_extents, tail_code_extents, out_spacing,
                    order, axis, (n - 1) * stride, stride, tail_code_base,
                    level, pass, step, CompactPredictorMode::extrapolate_two);
                plan_.groups.push_back(edge);
                plan_.code_count += static_cast<std::size_t>(edge.count);
            }
        }
    }

    void cubic_level_groups(std::uint64_t stride, int level, double step) {
        Triple spacing{{1, 1, 1}};
        const std::uint64_t twice = 2 * stride;
        spacing[axes_[1]] = twice;
        for (int pass = 0; pass < 2; ++pass) {
            const int axis = axes_[pass];
            if (pass != 0) spacing[axes_[pass - 1]] = stride;
            const std::uint64_t n = (dims_[axis] - 1) / stride + 1;
            if (n <= 1) continue;

            Triple out_spacing = spacing;
            out_spacing[axis] = twice;
            Triple extents{{1, 1, 1}};
            for (int d = 0; d < 3; ++d)
                extents[d] = (dims_[d] - 1) / out_spacing[d] + 1;
            extents[axis] = n / 2;
            const std::array<int, 3> order{{0, 1, 2}};
            const auto mode = (n & 1) ? CompactPredictorMode::cubic_odd
                                      : CompactPredictorMode::cubic_even;
            CompactGroup group = rectangle(extents, extents, out_spacing, order,
                axis, stride, stride, plan_.code_count, level, pass, step, mode);
            plan_.groups.push_back(group);
            plan_.code_count += static_cast<std::size_t>(group.count);
        }
    }

    const NDPlanParams& params_;
    CompactPlan plan_;
    Triple dims_{{1, 1, 1}};
    Triple pitch_{{1, 1, 1}};
    std::array<int, 3> axes_{{0, 1, 2}};
    std::uint64_t total_ = 1;
    std::uint64_t anchor_stride_ = 0;
    int rank_ = 0;
    int top_level_ = 0;
    double seed_step_ = 0;
};

}  // namespace nd_compact_detail

// The compact planner has the same serial float32 LINEAR topology and
// quantization-step contract as make_nd_plan. It also supports serial 2D
// CUBIC when params.cubic is true. Both require block_size equal to
// max(dimensions). Each group is a flat, trivially-copyable launch
// descriptor; execute groups in order with a barrier between them. Memory and
// construction time are O(rank * log(max dimension)), independent of the
// number of points. No per-point topology is allocated or transferred.
inline CompactPlan make_compact_plan(const NDPlanParams& params) {
    return nd_compact_detail::Builder(params).build();
}

}  // namespace hip_falcons

#undef HIP_FALCONS_COMPACT_HD

#endif  // HIP_FALCONS_ND_COMPACT_HPP
