#include "nd_reconstruct.hpp"
#include "nd_plan.hpp"
#include "nd_compact.hpp"
#include "dataloader/packed.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace hip_falcons {
namespace {

constexpr std::int16_t kEscape = std::numeric_limits<std::int16_t>::min();
constexpr unsigned int kThreads = 256;
constexpr std::size_t kMaxBlocks = 65535;
constexpr std::size_t kPackedSmallPlane = 4096;
constexpr std::size_t kPackedPrefixGroups = 8;
constexpr std::size_t kPackedBlocksPerPlane = 64;
using Clock = std::chrono::steady_clock;

static_assert(std::is_trivially_copyable<CompactAnchors>::value,
              "CompactAnchors must be usable as a kernel argument");
static_assert(std::is_trivially_copyable<CompactGroup>::value,
              "CompactGroup must be usable as a kernel argument");
static_assert(std::is_trivially_copyable<PackedHeader>::value &&
              std::is_standard_layout<PackedHeader>::value,
              "PackedHeader must be transferable as bytes");

double milliseconds(Clock::time_point start) noexcept {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

class TimingCollector {
public:
    explicit TimingCollector(NDReconstructTimings* result) noexcept
        : result_(result), start_(Clock::now()) {
        if (result_ != nullptr) {
            *result_ = {};
        }
    }

    ~TimingCollector() {
        if (result_ != nullptr) {
            times.other_ms = std::max(0.0, milliseconds(start_) - times.plan_ms -
                times.upload_ms - times.kernel_ms - times.download_ms);
            *result_ = times;
        }
    }

    void mark_to_device() noexcept {
        times.to_device_ms = milliseconds(start_);
    }

    NDReconstructTimings times{};

private:
    NDReconstructTimings* result_;
    Clock::time_point start_;
};

class TimedSection {
public:
    explicit TimedSection(double& elapsed) noexcept : elapsed_(elapsed), start_(Clock::now()) {}
    ~TimedSection() { elapsed_ += milliseconds(start_); }

private:
    double& elapsed_;
    Clock::time_point start_;
};

template<class Plan>
struct Prepared {
    Plan plan;
    std::vector<std::uint32_t> escapes;
    std::size_t num_elements = 0;
};

void build_plan(const NDPlanParams& params, NDPlan& plan) {
    plan = make_nd_plan(params);
}

void build_plan(const NDPlanParams& params, CompactPlan& plan) {
    plan = make_compact_plan(params);
}

std::size_t anchor_count(const NDPlan& plan) noexcept {
    return plan.anchors.size();
}

std::size_t anchor_count(const CompactPlan& plan) noexcept {
    return static_cast<std::size_t>(plan.anchors.count);
}

bool valid_plan(const NDPlan& plan, std::size_t num_elements,
                std::size_t code_count) noexcept {
    if (plan.code_count != code_count || plan.operations.size() != code_count ||
        plan.anchors.size() != num_elements - code_count) {
        return false;
    }
    for (const std::uint32_t anchor : plan.anchors) {
        if (anchor >= num_elements) {
            return false;
        }
    }
    for (const NDOperation& op : plan.operations) {
        if (op.out >= num_elements || op.code_index >= code_count) {
            return false;
        }
        switch (op.kind) {
        case NDPredictor::midpoint:
        case NDPredictor::extrapolate:
            if (op.left >= num_elements || op.right >= num_elements) {
                return false;
            }
            break;
        case NDPredictor::copy:
            if (op.left >= num_elements) {
                return false;
            }
            break;
        case NDPredictor::zero:
            break;
        default:
            return false;
        }
    }
    std::size_t next = 0;
    for (const NDGroup& group : plan.groups) {
        if (group.begin != next || group.count > code_count - next ||
            !std::isfinite(group.step) || group.step <= 0) {
            return false;
        }
        next += group.count;
    }
    return next == code_count;
}

bool valid_plan(const CompactPlan& plan, std::size_t num_elements,
                std::size_t code_count) noexcept {
    if (plan.code_count != code_count || plan.anchors.count != num_elements - code_count) {
        return false;
    }
    std::size_t count = 0;
    for (const CompactGroup& group : plan.groups) {
        if (group.count > code_count - count ||
            !std::isfinite(group.step) || group.step <= 0) {
            return false;
        }
        count += static_cast<std::size_t>(group.count);
    }
    return count == code_count;
}

template<class Plan>
hipError_t prepare(const NDReconstructParams& params,
                   const std::int16_t* codes, std::size_t code_count,
                   const float* unpredictable, std::size_t unpredictable_count,
                   float* output, Prepared<Plan>& prepared, double& plan_ms) {
    if (params.dimensions.size() < 2 || params.dimensions.size() > 3 ||
        (code_count != 0 && codes == nullptr) ||
        (unpredictable_count != 0 && unpredictable == nullptr) || output == nullptr) {
        return hipErrorInvalidValue;
    }
    std::size_t num_elements = 1;
    for (const std::size_t dim : params.dimensions) {
        if (dim == 0 || dim > std::numeric_limits<std::uint32_t>::max() / num_elements) {
            return hipErrorInvalidValue;
        }
        num_elements *= dim;
    }
    if (code_count > num_elements || unpredictable_count > num_elements) {
        return hipErrorInvalidValue;
    }

    NDPlanParams plan_params{};
    plan_params.dimensions = params.dimensions;
    plan_params.block_size = params.block_size;
    plan_params.anchor_stride = params.anchor_stride;
    plan_params.direction = params.direction;
    plan_params.error_bound = params.error_bound;
    plan_params.eb_alpha = params.eb_alpha;
    plan_params.eb_beta = params.eb_beta;
    {
        TimedSection timer(plan_ms);
        build_plan(plan_params, prepared.plan);
    }
    if (!valid_plan(prepared.plan, num_elements, code_count)) {
        return hipErrorInvalidValue;
    }

    const std::size_t anchors = anchor_count(prepared.plan);
    if (unpredictable_count < anchors ||
        unpredictable_count - anchors > code_count) {
        return hipErrorInvalidValue;
    }
    const std::size_t expected_escapes = unpredictable_count - anchors;
    prepared.escapes.reserve(expected_escapes);
    for (std::size_t i = 0; i < code_count; ++i) {
        if (codes[i] == kEscape) {
            if (prepared.escapes.size() == expected_escapes) {
                return hipErrorInvalidValue;
            }
            prepared.escapes.push_back(static_cast<std::uint32_t>(i));
        }
    }
    if (prepared.escapes.size() != expected_escapes) {
        return hipErrorInvalidValue;
    }
    prepared.num_elements = num_elements;
    return hipSuccess;
}

__host__ __device__ std::size_t escape_rank(const std::uint32_t* positions,
                                            std::size_t count, std::uint32_t code_index) {
    std::size_t lo = 0;
    std::size_t hi = count;
    while (lo < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (positions[mid] < code_index) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__host__ __device__ void apply_operation(
    const NDOperation& op, double step, const std::int16_t* codes,
    const float* unpredictable, std::size_t anchor_count,
    const std::uint32_t* escapes, std::size_t escape_count, float* output) {
    const std::int16_t code = codes[op.code_index];
    if (code == kEscape) {
        output[op.out] = unpredictable[anchor_count + escape_rank(escapes, escape_count,
                                                                    op.code_index)];
        return;
    }

    float prediction = 0.0f;
    switch (op.kind) {
    case NDPredictor::midpoint: {
        // Falcons rounds the float addition before multiplying by one half.
        volatile float sum = output[op.left] + output[op.right];
        prediction = sum * 0.5f;
        break;
    }
    case NDPredictor::extrapolate: {
        volatile float scaled_last = 1.5f * output[op.right];
        prediction = fmaf(-0.5f, output[op.left], scaled_last);
        break;
    }
    case NDPredictor::copy:
        prediction = output[op.left];
        break;
    case NDPredictor::zero:
        break;
    }
    output[op.out] = static_cast<float>(fma(static_cast<double>(code), step,
                                            static_cast<double>(prediction)));
}

__global__ void initialize_anchors(CompactAnchors anchors,
                                   const float* unpredictable, float* output) {
    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t increment = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = first; i < anchors.count; i += increment) {
        output[compact_anchor_index(anchors, i)] = unpredictable[i];
    }
}

__global__ void apply_group(CompactGroup group, const std::int16_t* codes,
                            const float* unpredictable, std::size_t anchor_count,
                            const std::uint32_t* escapes, std::size_t escape_count,
                            float* output) {
    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t increment = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = first; i < group.count; i += increment) {
        apply_operation(compact_operation(group, i), group.step, codes, unpredictable, anchor_count,
                        escapes, escape_count, output);
    }
}

__device__ const std::uint8_t* packed_plane(const std::uint8_t* bytes,
                                             const std::int64_t* offsets,
                                             std::size_t plane) {
    return bytes + offsets[plane];
}

__device__ void apply_packed_lane(const std::uint8_t* base, const PackedHeader& header,
                                  const CompactGroup& group, std::size_t lane,
                                  float* output) {
    const auto* codes = reinterpret_cast<const std::int16_t*>(base + header.code_off);
    const auto* unpredictable = reinterpret_cast<const float*>(base + header.unpred_off);
    const auto* escapes = reinterpret_cast<const std::uint32_t*>(base + header.escape_off);
    apply_operation(compact_operation(group, lane), group.step, codes, unpredictable,
                    header.anchors.count, escapes, header.escape_count, output);
}

// A block owns a plane while it executes these dependent groups.
__global__ void packed_prefix(const std::uint8_t* bytes, const std::int64_t* offsets,
                              float* output, std::size_t planes, std::size_t elements,
                              std::size_t groups, bool initialize) {
    for (std::size_t plane = blockIdx.x; plane < planes; plane += gridDim.x) {
        const auto* base = packed_plane(bytes, offsets, plane);
        const auto& header = *reinterpret_cast<const PackedHeader*>(base);
        float* plane_output = output + plane * elements;
        if (initialize) {
            const auto* unpredictable = reinterpret_cast<const float*>(base + header.unpred_off);
            for (std::size_t i = threadIdx.x; i < header.anchors.count; i += blockDim.x) {
                plane_output[compact_anchor_index(header.anchors, i)] = unpredictable[i];
            }
            __syncthreads();
        }
        const auto* descriptors = reinterpret_cast<const CompactGroup*>(base + header.group_off);
        const std::size_t count = header.group_count < groups ? header.group_count : groups;
        for (std::size_t g = 0; g < count; ++g) {
            const CompactGroup& group = descriptors[g];
            for (std::size_t i = threadIdx.x; i < group.count; i += blockDim.x) {
                apply_packed_lane(base, header, group, i, plane_output);
            }
            __syncthreads();
        }
    }
}

__global__ void packed_anchors(const std::uint8_t* bytes, const std::int64_t* offsets,
                               float* output, std::size_t planes, std::size_t elements,
                               std::size_t blocks_per_plane) {
    const std::size_t slots = planes * blocks_per_plane;
    for (std::size_t slot = blockIdx.x; slot < slots; slot += gridDim.x) {
        const std::size_t plane = slot / blocks_per_plane;
        const std::size_t tile = slot % blocks_per_plane;
        const auto* base = packed_plane(bytes, offsets, plane);
        const auto& header = *reinterpret_cast<const PackedHeader*>(base);
        const auto* unpredictable = reinterpret_cast<const float*>(base + header.unpred_off);
        float* plane_output = output + plane * elements;
        const std::size_t stride = blocks_per_plane * blockDim.x;
        for (std::size_t i = tile * blockDim.x + threadIdx.x;
             i < header.anchors.count; i += stride) {
            plane_output[compact_anchor_index(header.anchors, i)] = unpredictable[i];
        }
    }
}

__global__ void packed_group(const std::uint8_t* bytes, const std::int64_t* offsets,
                             float* output, std::size_t planes, std::size_t elements,
                             std::size_t blocks_per_plane, std::size_t group_index) {
    const std::size_t slots = planes * blocks_per_plane;
    for (std::size_t slot = blockIdx.x; slot < slots; slot += gridDim.x) {
        const std::size_t plane = slot / blocks_per_plane;
        const std::size_t tile = slot % blocks_per_plane;
        const auto* base = packed_plane(bytes, offsets, plane);
        const auto& header = *reinterpret_cast<const PackedHeader*>(base);
        if (group_index >= header.group_count) continue;
        const auto* descriptors = reinterpret_cast<const CompactGroup*>(base + header.group_off);
        const CompactGroup& group = descriptors[group_index];
        float* plane_output = output + plane * elements;
        const std::size_t stride = blocks_per_plane * blockDim.x;
        for (std::size_t i = tile * blockDim.x + threadIdx.x;
             i < group.count; i += stride) {
            apply_packed_lane(base, header, group, i, plane_output);
        }
    }
}

unsigned int blocks_for(std::size_t count) noexcept {
    return static_cast<unsigned int>(std::min(kMaxBlocks, 1 + (count - 1) / kThreads));
}

hipError_t launch_packed(const void* bytes, const std::int64_t* offsets, float* output,
                         std::size_t planes, std::size_t elements, std::size_t max_groups,
                         hipStream_t stream) noexcept {
    if (planes == 0) return hipSuccess;
    if (bytes == nullptr || offsets == nullptr || output == nullptr || elements == 0 ||
        elements > std::numeric_limits<std::uint32_t>::max() ||
        planes > std::numeric_limits<std::size_t>::max() / sizeof(float) / elements) {
        return hipErrorInvalidValue;
    }
    const auto* packed = static_cast<const std::uint8_t*>(bytes);
    const auto plane_blocks = static_cast<unsigned int>(std::min(kMaxBlocks, planes));
    if (elements <= kPackedSmallPlane) {
        hipLaunchKernelGGL(packed_prefix, dim3(plane_blocks), dim3(kThreads), 0, stream,
                           packed, offsets, output, planes, elements, max_groups, true);
        return hipGetLastError();
    }

    const std::size_t blocks_per_plane = std::min(kPackedBlocksPerPlane,
                                                   1 + (elements - 1) / kThreads);
    const auto blocks = static_cast<unsigned int>(std::min(kMaxBlocks,
                                                            planes * blocks_per_plane));
    hipLaunchKernelGGL(packed_anchors, dim3(blocks), dim3(kThreads), 0, stream,
                       packed, offsets, output, planes, elements, blocks_per_plane);
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) return status;

    const std::size_t prefix_groups = std::min(max_groups, kPackedPrefixGroups);
    if (prefix_groups != 0) {
        hipLaunchKernelGGL(packed_prefix, dim3(plane_blocks), dim3(kThreads), 0, stream,
                           packed, offsets, output, planes, elements, prefix_groups, false);
        status = hipGetLastError();
        if (status != hipSuccess) return status;
    }
    for (std::size_t group = prefix_groups; group < max_groups; ++group) {
        hipLaunchKernelGGL(packed_group, dim3(blocks), dim3(kThreads), 0, stream,
                           packed, offsets, output, planes, elements, blocks_per_plane, group);
        status = hipGetLastError();
        if (status != hipSuccess) return status;
    }
    return hipSuccess;
}

class DeviceResources {
public:
    hipStream_t stream = nullptr;
    std::int16_t* codes = nullptr;
    float* unpredictable = nullptr;
    std::uint32_t* escapes = nullptr;
    float* output = nullptr;

    DeviceResources() = default;
    DeviceResources(const DeviceResources&) = delete;
    DeviceResources& operator=(const DeviceResources&) = delete;
    ~DeviceResources() { (void)close(); }

    hipError_t create_stream() noexcept {
        const hipError_t status = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
        if (status == hipSuccess) {
            has_stream_ = true;
            owns_stream_ = true;
        }
        return status;
    }

    void borrow_stream(hipStream_t borrowed) noexcept {
        stream = borrowed;
        has_stream_ = true;
        owns_stream_ = false;
    }

    template<class T>
    hipError_t allocate(T*& ptr, std::size_t count) noexcept {
        if (count == 0) {
            return hipSuccess;
        }
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            return hipErrorInvalidValue;
        }
        return hipMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T));
    }

    hipError_t allocate_output(std::size_t count) noexcept {
        const hipError_t status = allocate(owned_output_, count);
        if (status == hipSuccess) {
            output = owned_output_;
        }
        return status;
    }

    hipError_t close() noexcept {
        hipError_t result = hipSuccess;
        auto record = [&](hipError_t status) {
            if (result == hipSuccess && status != hipSuccess) {
                result = status;
            }
        };
        if (has_stream_) {
            record(hipStreamSynchronize(stream));
            if (owns_stream_) {
                record(hipStreamDestroy(stream));
            }
            stream = nullptr;
            has_stream_ = false;
            owns_stream_ = false;
        }
        release(owned_output_, record);
        output = nullptr;
        release(escapes, record);
        release(unpredictable, record);
        release(codes, record);
        return result;
    }

private:
    bool has_stream_ = false;
    bool owns_stream_ = false;
    float* owned_output_ = nullptr;

    template<class T, class Record>
    static void release(T*& ptr, Record& record) noexcept {
        if (ptr != nullptr) {
            record(hipFree(ptr));
            ptr = nullptr;
        }
    }
};

template<class T>
hipError_t upload(T* destination, const T* source, std::size_t count,
                  hipStream_t stream) noexcept {
    return count == 0 ? hipSuccess : hipMemcpyAsync(destination, source, count * sizeof(T),
                                                   hipMemcpyHostToDevice, stream);
}

hipError_t launch_plan(const Prepared<CompactPlan>& prepared,
                       const DeviceResources& device) noexcept {
    const CompactPlan& plan = prepared.plan;
    const std::size_t anchors = anchor_count(plan);
    if (anchors != 0) {
        hipLaunchKernelGGL(initialize_anchors, dim3(blocks_for(anchors)),
                           dim3(kThreads), 0, device.stream, plan.anchors,
                           device.unpredictable, device.output);
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            return status;
        }
    }
    // Launches in this stream serialize groups with dependent tails.
    for (const CompactGroup& group : plan.groups) {
        if (group.count == 0) {
            continue;
        }
        hipLaunchKernelGGL(apply_group, dim3(blocks_for(group.count)), dim3(kThreads),
                           0, device.stream, group, device.codes, device.unpredictable, anchors,
                           device.escapes, prepared.escapes.size(), device.output);
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            return status;
        }
    }
    return hipSuccess;
}

}  // namespace

hipError_t reconstruct_nd_linear_f32_host(
    const NDReconstructParams& params,
    const std::int16_t* codes, std::size_t code_count,
    const float* unpredictable, std::size_t unpredictable_count,
    float* output) noexcept {
    try {
        Prepared<NDPlan> prepared;
        double plan_ms = 0;
        const hipError_t status = prepare(params, codes, code_count, unpredictable,
                                          unpredictable_count, output, prepared, plan_ms);
        if (status != hipSuccess) {
            return status;
        }
        const NDPlan& plan = prepared.plan;
        for (std::size_t i = 0; i < plan.anchors.size(); ++i) {
            output[plan.anchors[i]] = unpredictable[i];
        }
        for (const NDGroup& group : plan.groups) {
            for (std::size_t i = 0; i < group.count; ++i) {
                apply_operation(plan.operations[group.begin + i], group.step, codes,
                                unpredictable, plan.anchors.size(), prepared.escapes.data(),
                                prepared.escapes.size(), output);
            }
        }
        return hipSuccess;
    } catch (const std::bad_alloc&) {
        return hipErrorOutOfMemory;
    } catch (const std::invalid_argument&) {
        return hipErrorInvalidValue;
    } catch (const std::overflow_error&) {
        return hipErrorInvalidValue;
    } catch (const std::length_error&) {
        return hipErrorInvalidValue;
    } catch (...) {
        return hipErrorUnknown;
    }
}

namespace {

hipError_t reconstruct_nd_gpu_impl(
    const NDReconstructParams& params,
    const std::int16_t* codes, std::size_t code_count,
    const float* unpredictable, std::size_t unpredictable_count,
    float* output, hipStream_t stream, bool device_output,
    TimingCollector& clock) noexcept {
    try {
        Prepared<CompactPlan> prepared;
        hipError_t status = prepare(params, codes, code_count, unpredictable,
                                    unpredictable_count, output, prepared, clock.times.plan_ms);
        if (status != hipSuccess) {
            return status;
        }
        DeviceResources device;
        if (device_output) {
            device.borrow_stream(stream);
            device.output = output;
        } else {
            status = device.create_stream();
            if (status != hipSuccess) return status;
        }
        status = device.allocate(device.codes, code_count);
        if (status != hipSuccess) return status;
        status = device.allocate(device.unpredictable, unpredictable_count);
        if (status != hipSuccess) return status;
        status = device.allocate(device.escapes, prepared.escapes.size());
        if (status != hipSuccess) return status;
        if (!device_output) {
            status = device.allocate_output(prepared.num_elements);
            if (status != hipSuccess) return status;
        }

        {
            TimedSection timer(clock.times.upload_ms);
            status = upload(device.codes, codes, code_count, device.stream);
            if (status != hipSuccess) return status;
            status = upload(device.unpredictable, unpredictable, unpredictable_count, device.stream);
            if (status != hipSuccess) return status;
            status = upload(device.escapes, prepared.escapes.data(), prepared.escapes.size(),
                            device.stream);
            if (status != hipSuccess) return status;
            status = hipStreamSynchronize(device.stream);
            if (status != hipSuccess) return status;
        }
        {
            TimedSection timer(clock.times.kernel_ms);
            status = launch_plan(prepared, device);
            if (status != hipSuccess) return status;
            status = hipStreamSynchronize(device.stream);
            if (status != hipSuccess) return status;
            clock.mark_to_device();
        }
        if (!device_output) {
            TimedSection timer(clock.times.download_ms);
            status = hipMemcpyAsync(output, device.output, prepared.num_elements * sizeof(float),
                                    hipMemcpyDeviceToHost, device.stream);
            if (status != hipSuccess) return status;
            status = hipStreamSynchronize(device.stream);
            if (status != hipSuccess) return status;
        }
        return device.close();
    } catch (const std::bad_alloc&) {
        return hipErrorOutOfMemory;
    } catch (const std::invalid_argument&) {
        return hipErrorInvalidValue;
    } catch (const std::overflow_error&) {
        return hipErrorInvalidValue;
    } catch (const std::length_error&) {
        return hipErrorInvalidValue;
    } catch (...) {
        return hipErrorUnknown;
    }
}

}  // namespace

hipError_t reconstruct_nd_linear_f32_gpu(
    const NDReconstructParams& params,
    const std::int16_t* codes, std::size_t code_count,
    const float* unpredictable, std::size_t unpredictable_count,
    float* output, NDReconstructTimings* timings) noexcept {
    TimingCollector clock(timings);
    return reconstruct_nd_gpu_impl(params, codes, code_count, unpredictable,
                                   unpredictable_count, output, nullptr, false, clock);
}

hipError_t reconstruct_nd_linear_f32_device_output(
    const NDReconstructParams& params,
    const std::int16_t* codes, std::size_t code_count,
    const float* unpredictable, std::size_t unpredictable_count,
    float* device_output, hipStream_t stream) noexcept {
    TimingCollector clock(nullptr);
    return reconstruct_nd_gpu_impl(params, codes, code_count, unpredictable,
                                   unpredictable_count, device_output, stream, true, clock);
}

}  // namespace hip_falcons

extern "C" int hf_launch_packed(
    const void* device_bytes, const std::int64_t* device_offsets, float* device_out,
    std::size_t planes, std::size_t elements, std::size_t max_groups, void* stream) {
    return static_cast<int>(hip_falcons::launch_packed(
        device_bytes, device_offsets, device_out, planes, elements, max_groups,
        reinterpret_cast<hipStream_t>(stream)));
}
