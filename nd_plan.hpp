#ifndef HIP_FALCONS_ND_PLAN_HPP
#define HIP_FALCONS_ND_PLAN_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hip_falcons {

// The operands are flattened row-major device indices. For extrapolate,
// left and right are respectively the first and second arguments to
// interp_linear1; right is not necessarily to the right of out. Copy uses
// left; zero uses neither operand.
enum class NDPredictor : std::uint8_t {
    midpoint = 0,
    extrapolate = 1,
    copy = 2,
    zero = 3,
};

struct NDOperation {
    std::uint32_t out;
    std::uint32_t left;
    std::uint32_t right;
    std::uint32_t code_index;
    NDPredictor kind;
};

struct NDGroup {
    std::size_t begin;
    std::size_t count;
    int level;
    int pass;
    double step;
};

struct NDPlan {
    // These address the first unpredictable values, in CPU anchor order.
    std::vector<std::uint32_t> anchors;
    std::vector<NDOperation> operations;
    std::vector<NDGroup> groups;
    std::size_t code_count = 0;
};

struct NDPlanParams {
    std::vector<std::size_t> dimensions;
    std::size_t block_size = 0;
    std::size_t anchor_stride = 0;
    int direction = 0;
    double error_bound = 0;
    double eb_alpha = 0;
    double eb_beta = 0;
    bool cubic = false;
};

namespace nd_plan_detail {

using Point = std::array<std::size_t, 3>;

template <class Function>
inline void each(std::size_t first, std::size_t last, std::size_t step,
                 Function&& function) {
    if (first > last) return;
    for (std::size_t value = first;;) {
        function(value);
        if (last - value < step) break;
        value += step;
    }
}

template <class Function>
inline void each_point(const Point& first, const Point& last, const Point& step,
                       Function&& function) {
    each(first[0], last[0], step[0], [&](std::size_t i) {
        each(first[1], last[1], step[1], [&](std::size_t j) {
            each(first[2], last[2], step[2], [&](std::size_t k) {
                function(Point{{i, j, k}});
            });
        });
    });
}

inline std::size_t checked_add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::overflow_error("Falcons interpolation coordinate overflow");
    return a + b;
}

inline std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("Falcons interpolation stride overflow");
    return a * b;
}

class Builder {
public:
    explicit Builder(const NDPlanParams& params) : params_(params) {
        if (params.cubic)
            throw std::invalid_argument("Expanded Falcons planner supports LINEAR only");
        const std::size_t rank = params.dimensions.size();
        if (rank != 2 && rank != 3)
            throw std::invalid_argument("Falcons ND planner requires 2D or 3D");
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

        dims_ = {{1, 1, 1}};
        total_ = 1;
        std::size_t max_dim = 1;
        const auto max_index_count = std::numeric_limits<std::uint32_t>::max();
        for (int d = 0; d < rank_; ++d) {
            const std::size_t extent = params.dimensions[static_cast<std::size_t>(d)];
            if (extent == 0 || extent > max_index_count / total_)
                throw std::invalid_argument("Falcons dimensions exceed uint32 indexing");
            dims_[d] = extent;
            total_ *= extent;
            max_dim = std::max(max_dim, extent);
        }
        pitch_[2] = 1;
        pitch_[1] = dims_[2];
        pitch_[0] = dims_[1] * pitch_[1];
        initialized_.resize(total_, 0);

        axes_ = {{0, 1, 2}};
        for (int i = 0; i < params.direction; ++i)
            std::next_permutation(axes_.begin(), axes_.begin() + rank_);

        for (std::size_t n = max_dim - 1; n != 0; n >>= 1) ++top_level_;
        anchor_stride_ = params.anchor_stride < max_dim ? params.anchor_stride : 0;
        if (anchor_stride_ != 0) {
            int anchor_level = 1;
            for (std::size_t n = anchor_stride_; n > 1; n >>= 1) ++anchor_level;
            top_level_ = std::min(top_level_, anchor_level) - 1;
        }
        seed_step_ = checked_step(params.error_bound);
    }

    NDPlan build() {
        if (anchor_stride_ != 0) {
            Point last{{dims_[0] - 1, dims_[1] - 1, dims_[2] - 1}};
            const Point step{{anchor_stride_, anchor_stride_, anchor_stride_}};
            each_point(Point{{0, 0, 0}}, last, step, [&](const Point& point) {
                const auto index = flatten(point);
                plan_.anchors.push_back(index);
                initialized_[index] = 1;
            });
        } else {
            initialized_[0] = 1;
            plan_.operations.push_back(NDOperation{0, 0, 0, 0, NDPredictor::zero});
            plan_.groups.push_back(NDGroup{0, 1, 0, -1, seed_step_});
            plan_.code_count = 1;
        }

        for (int level = top_level_; level > 0; --level) {
            const std::size_t stride = std::size_t{1} << (level - 1);
            const std::size_t block_extent = checked_mul(params_.block_size, stride);
            const Point block_step{{block_extent, block_extent, block_extent}};
            const Point last{{dims_[0] - 1, dims_[1] - 1, dims_[2] - 1}};
            const double step = level_step(level);
            each_point(Point{{0, 0, 0}}, last, block_step, [&](const Point& begin) {
                Point end{};
                for (int d = 0; d < 3; ++d)
                    end[d] = begin[d] + std::min(block_extent, last[d] - begin[d]);
                block(begin, end, stride, level, step);
            });
        }

        if (plan_.code_count != total_ - plan_.anchors.size() ||
            plan_.operations.size() != plan_.code_count)
            throw std::invalid_argument("Falcons traversal does not cover the output exactly once");
        return std::move(plan_);
    }

private:
    static double checked_step(double eb) {
        const double step = 2 * eb;
        if (!std::isfinite(step) || step <= 0)
            throw std::invalid_argument("Invalid Falcons quantization step");
        return step;
    }

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

    std::uint32_t flatten(const Point& p) const {
        return static_cast<std::uint32_t>(p[0] * pitch_[0] + p[1] * pitch_[1] + p[2]);
    }

    // State: 0 unavailable, 1 from an earlier group, 2 main pass, 3 tail.
    // Moving tails to the next group preserves the serial code index.
    void emit(std::size_t out, std::size_t left, std::size_t right,
              NDPredictor kind, bool tail) {
        if (out >= total_ || initialized_[out] != 0)
            throw std::invalid_argument("Falcons traversal writes an invalid or repeated point");
        const auto check_source = [&](std::size_t source) {
            if (source >= total_ || initialized_[source] == 0 ||
                (!tail && initialized_[source] != 1) ||
                (tail && initialized_[source] == 3))
                throw std::invalid_argument("Falcons traversal has an unavailable dependency");
        };
        if (kind != NDPredictor::zero) check_source(left);
        if (kind == NDPredictor::midpoint || kind == NDPredictor::extrapolate)
            check_source(right);
        if (plan_.code_count >= std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("Falcons code index exceeds uint32");
        const NDOperation op{static_cast<std::uint32_t>(out),
                             static_cast<std::uint32_t>(left),
                             static_cast<std::uint32_t>(right),
                             static_cast<std::uint32_t>(plan_.code_count++), kind};
        initialized_[out] = tail ? 3 : 2;
        if (tail) tails_.push_back(op);
        else plan_.operations.push_back(op);
    }

    void midpoint(const Point& point, int axis, std::size_t stride) {
        const std::size_t out = flatten(point);
        const std::size_t delta = stride * pitch_[axis];
        emit(out, out - delta, out + delta, NDPredictor::midpoint, false);
    }

    void edge(const Point& point, int axis, std::size_t stride,
              std::size_t n, bool skip_overwrite, bool generic) {
        const std::size_t out = flatten(point);
        const std::size_t delta = stride * pitch_[axis];
        if (n < 3) {
            emit(out, out - delta, 0, NDPredictor::copy, false);
        } else {
            const std::size_t back = generic || !skip_overwrite ? 2 : 3;
            emit(out, out - back * delta, out - delta,
                 NDPredictor::extrapolate, back == 2);
        }
    }

    void finish_pass(std::size_t begin, int level, int pass, double step) {
        const std::size_t main_count = plan_.operations.size() - begin;
        if (main_count != 0)
            plan_.groups.push_back(NDGroup{begin, main_count, level, pass, step});
        const std::size_t tail_begin = plan_.operations.size();
        if (!tails_.empty()) {
            plan_.operations.insert(plan_.operations.end(), tails_.begin(), tails_.end());
            plan_.groups.push_back(NDGroup{tail_begin, tails_.size(), level, pass, step});
            tails_.clear();
        }
        for (std::size_t i = begin; i < plan_.operations.size(); ++i)
            initialized_[plan_.operations[i].out] = 1;
    }

    // The SIMD contiguous-axis line has an extrapolated final odd sample,
    // even for odd n. Its other odd samples are midpoints.
    void line(Point point, int axis, std::size_t stride, std::size_t n) {
        const std::size_t odd_count = n / 2;
        const std::size_t even_count = n - odd_count;
        const std::size_t start = point[axis];
        for (std::size_t i = 0; i + 1 < odd_count; ++i) {
            point[axis] = start + (2 * i + 1) * stride;
            midpoint(point, axis, stride);
        }
        point[axis] = start + (2 * odd_count - 1) * stride;
        const std::size_t out = flatten(point);
        const std::size_t delta = stride * pitch_[axis];
        if (n < 3) {
            emit(out, out - delta, 0, NDPredictor::copy, false);
        } else {
            point[axis] = start + 2 * (even_count - 2) * stride;
            const auto left = flatten(point);
            point[axis] = start + 2 * (even_count - 1) * stride;
            emit(out, left, flatten(point), NDPredictor::extrapolate, false);
        }
    }

    void fastest(const Point& begin, const Point& end, const Point& steps,
                 int axis, std::size_t stride, std::size_t n) {
        Point first = begin, last = end, inc = steps;
        last[axis] = first[axis];
        inc[axis] = 1;
        each_point(first, last, inc, [&](const Point& point) {
            line(point, axis, stride, n);
        });
    }

    // Optimized non-contiguous-axis passes process a vector of fastest-axis
    // lanes at a time. The valid lanes, including a partial vector, are in
    // increasing order. Z is row-major within each Y row, unlike generic
    // row-major traversal over the whole 3D pass.
    void gathered(const Point& begin, const Point& end, const Point& steps,
                  int axis, std::size_t stride, std::size_t n, bool skip) {
        auto interior = [&](Point point) {
            each(begin[axis] + stride, begin[axis] + (n - 2) * stride,
                 2 * stride, [&](std::size_t pos) {
                point[axis] = pos;
                each(begin[rank_ - 1], end[rank_ - 1], steps[rank_ - 1],
                     [&](std::size_t k) {
                    point[rank_ - 1] = k;
                    midpoint(point, axis, stride);
                });
            });
        };
        auto boundary = [&](Point point) {
            if ((n & 1) != 0) return;
            point[axis] = begin[axis] + (n - 1) * stride;
            each(begin[rank_ - 1], end[rank_ - 1], steps[rank_ - 1],
                 [&](std::size_t k) {
                point[rank_ - 1] = k;
                edge(point, axis, stride, n, skip, false);
            });
        };

        if (rank_ == 2) {
            Point point = begin;
            interior(point);
            boundary(point);
        } else if (axis == 0) {
            each(begin[1], end[1], steps[1], [&](std::size_t j) {
                Point point = begin;
                point[1] = j;
                interior(point);
                boundary(point);
            });
        } else {
            each(begin[0], end[0], steps[0], [&](std::size_t i) {
                Point point = begin;
                point[0] = i;
                interior(point);
                boundary(point);
            });
        }
    }

    // Directions 1..4 in 3D use the current serial fastest-dimension-first
    // fallback, whose edge reads the preceding odd sample (out - 2*stride).
    void generic(const Point& begin, const Point& end, const Point& steps,
                 int axis, std::size_t stride, std::size_t n) {
        Point first = begin, last = end, inc = steps;
        first[axis] = begin[axis] + stride;
        last[axis] = begin[axis] + (n - 2) * stride;
        inc[axis] = 2 * stride;
        each_point(first, last, inc, [&](const Point& point) {
            midpoint(point, axis, stride);
        });
        if ((n & 1) == 0) {
            first[axis] = last[axis] = begin[axis] + (n - 1) * stride;
            each_point(first, last, inc, [&](const Point& point) {
                edge(point, axis, stride, n, false, true);
            });
        }
    }

    void block(const Point& block_begin, const Point& end, std::size_t stride,
               int level, double step) {
        Point begin = block_begin;
        Point steps{{1, 1, 1}};
        const std::size_t twice = checked_mul(stride, 2);
        for (int p = 1; p < rank_; ++p) {
            const int axis = axes_[p];
            begin[axis] = block_begin[axis] ? checked_add(block_begin[axis], twice) : 0;
            steps[axis] = twice;
        }

        for (int pass = 0; pass < rank_; ++pass) {
            const int axis = axes_[pass];
            if (pass != 0) {
                const int previous = axes_[pass - 1];
                begin[axis] = block_begin[axis];
                begin[previous] = block_begin[previous]
                    ? checked_add(block_begin[previous], stride) : 0;
                steps[previous] = stride;
            }
            bool empty = false;
            for (int d = 0; d < rank_; ++d) empty = empty || begin[d] > end[d];
            if (empty) continue;
            const std::size_t n = (end[axis] - begin[axis]) / stride + 1;
            if (n <= 1) continue;

            const std::size_t group_begin = plan_.operations.size();
            const bool specialized = rank_ == 2 || params_.direction == 0 || params_.direction == 5;
            if (!specialized) {
                generic(begin, end, steps, axis, stride, n);
            } else if (axis == rank_ - 1) {
                fastest(begin, end, steps, axis, stride, n);
            } else {
                const bool skip = level == 1 && pass == rank_ - 1;
                gathered(begin, end, steps, axis, stride, n, skip);
            }
            finish_pass(group_begin, level, pass, step);
        }
    }

    NDPlanParams params_;
    NDPlan plan_;
    Point dims_{{1, 1, 1}};
    Point pitch_{{1, 1, 1}};
    std::array<int, 3> axes_{{0, 1, 2}};
    std::vector<std::uint8_t> initialized_;
    std::vector<NDOperation> tails_;
    std::size_t total_ = 0;
    std::size_t anchor_stride_ = 0;
    int rank_ = 0;
    int top_level_ = 0;
    double seed_step_ = 0;
};

}  // namespace nd_plan_detail

// Plan serial, non-tuning float32 LINEAR streams with radius 32768. Dimensions
// must be positive and their product must fit uint32; block_size must be
// positive and fit uint32; anchor_stride must be zero or a power of two.
// Unsupported or malformed traversals throw rather than emitting a partial
// plan. Groups execute in order
// with a barrier between groups; operations in a group are independent.
// code_index always addresses the original serial code stream, not the
// potentially reordered operations vector. With no anchors, the first group
// is the zero-predicted seed using the original quantizer step. Otherwise the
// anchor indices address the first unpredictable values in CPU order. Escapes
// follow those anchors in increasing code_index order.
//
// Midpoint is float(left + right) * 0.5f; extrapolate is
// fmaf(-0.5f, left, float(1.5f * right)). Predictable reconstruction is
// float(fma(double(code), group.step, double(prediction))). The caller must
// use IEEE round-to-nearest arithmetic without fast-math for bitwise matching.
// Only the topology and double quantization steps are computed here; no data
// values or float predictions are computed on the host.
inline NDPlan make_nd_plan(const NDPlanParams& params) {
    return nd_plan_detail::Builder(params).build();
}

}  // namespace hip_falcons

#endif  // HIP_FALCONS_ND_PLAN_HPP
