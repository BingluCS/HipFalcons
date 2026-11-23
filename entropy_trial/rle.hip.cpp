#include "rle.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <limits>

namespace hip_falcons {
namespace entropy_trial {
namespace {

constexpr unsigned kThreads = 256;

// At most 17 bits plus a seven-bit byte offset fit in this three-byte window.
__device__ std::uint32_t gamma_at(const unsigned char* data, std::uint32_t bytes,
                                  std::uint32_t bit_offset, unsigned width) {
    if (!width) return 1;
    const std::uint32_t byte = bit_offset >> 3;
    const unsigned bit = bit_offset & 7;
    std::uint32_t word = static_cast<std::uint32_t>(data[byte]) << 16;
    if (byte + 1 < bytes) word |= static_cast<std::uint32_t>(data[byte + 1]) << 8;
    if (byte + 2 < bytes) word |= data[byte + 2];
    const std::uint32_t leading = std::uint32_t{1} << width;
    const std::uint32_t low = (word >> (24 - bit - width)) & (leading - 1);
    return leading | low;
}

__global__ void rle_kernel(const unsigned char* packed, std::int16_t* codes) {
    const auto& job = reinterpret_cast<const rle_detail::Job*>(packed)[blockIdx.x];
    const unsigned lane = threadIdx.x;
    if (!job.count) return;

    const std::int32_t mode = job.mode_bits < 32768
        ? static_cast<std::int32_t>(job.mode_bits)
        : static_cast<std::int32_t>(job.mode_bits) - 65536;
    for (std::uint32_t i = lane; i < job.count; i += kThreads)
        codes[job.output_offset + i] = static_cast<std::int16_t>(mode);
    if (!job.value_count) return;
    __syncthreads();

    const auto* rclass = packed + job.run_class_offset;
    const auto* vclass = packed + job.value_class_offset;
    const auto* rman = packed + job.run_mantissa_offset;
    const auto* vman = packed + job.value_mantissa_offset;
    __shared__ std::uint32_t run_scan[kThreads];
    __shared__ std::uint32_t value_scan[kThreads];
    __shared__ std::uint32_t advance_scan[kThreads];
    const std::uint32_t per_thread = (job.value_count + kThreads - 1) / kThreads;
    const std::uint32_t first = lane * per_thread;
    const std::uint32_t last = first < job.value_count
        ? (first + per_thread < job.value_count ? first + per_thread : job.value_count) : first;
    std::uint32_t run_width = 0, value_width = 0;
    for (std::uint32_t i = first; i < last; ++i) {
        run_width += job.run_repeat == rle_detail::kClassArray ? rclass[i] : job.run_repeat;
        value_width += job.value_repeat == rle_detail::kClassArray ? vclass[i] : job.value_repeat;
    }
    run_scan[lane] = run_width;
    value_scan[lane] = value_width;
    __syncthreads();

    // Scan contiguous segment widths to find each thread's two bit offsets.
    for (unsigned stride = 1; stride < kThreads; stride <<= 1) {
        const std::uint32_t r = lane >= stride ? run_scan[lane - stride] : 0;
        const std::uint32_t v = lane >= stride ? value_scan[lane - stride] : 0;
        __syncthreads();
        run_scan[lane] += r;
        value_scan[lane] += v;
        __syncthreads();
    }
    const std::uint32_t run_start = run_scan[lane] - run_width;
    std::uint32_t value_bit = value_scan[lane] - value_width;
    std::uint32_t run_bit = run_start, advance = 0;
    for (std::uint32_t i = first; i < last; ++i) {
        const unsigned width = job.run_repeat == rle_detail::kClassArray ? rclass[i] : job.run_repeat;
        advance += gamma_at(rman, job.run_mantissa_bytes, run_bit, width);
        run_bit += width;
    }
    advance_scan[lane] = advance;
    __syncthreads();

    // Runs advance by L+1 up to each non-mode value. The optional last
    // run has no value to scatter and is already covered by the mode fill.
    for (unsigned stride = 1; stride < kThreads; stride <<= 1) {
        const std::uint32_t a = lane >= stride ? advance_scan[lane - stride] : 0;
        __syncthreads();
        advance_scan[lane] += a;
        __syncthreads();
    }
    std::uint32_t produced = advance_scan[lane] - advance;
    run_bit = run_start;
    for (std::uint32_t i = first; i < last; ++i) {
        const unsigned rw = job.run_repeat == rle_detail::kClassArray ? rclass[i] : job.run_repeat;
        const unsigned vw = job.value_repeat == rle_detail::kClassArray ? vclass[i] : job.value_repeat;
        produced += gamma_at(rman, job.run_mantissa_bytes, run_bit, rw);
        const std::uint32_t zigzag = gamma_at(vman, job.value_mantissa_bytes, value_bit, vw) - 1;
        run_bit += rw;
        value_bit += vw;
        if (produced <= job.count) {
            const std::int32_t difference = (zigzag & 1)
                ? -static_cast<std::int32_t>((zigzag >> 1) + 1)
                : static_cast<std::int32_t>(zigzag >> 1);
            codes[job.output_offset + produced - 1] =
                static_cast<std::int16_t>(static_cast<std::int32_t>(job.mode_bits) + difference);
        }
    }
}

}  // namespace

int launch_rle(const void* packed_device, std::int16_t* codes_device,
               std::size_t jobs, hipStream_t stream) {
    if (!jobs) return static_cast<int>(hipSuccess);
    if (!packed_device || !codes_device || jobs > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return static_cast<int>(hipErrorInvalidValue);
    hipLaunchKernelGGL(rle_kernel, dim3(static_cast<unsigned>(jobs)), dim3(kThreads), 0, stream,
                       static_cast<const unsigned char*>(packed_device), codes_device);
    return static_cast<int>(hipGetLastError());
}

}  // namespace entropy_trial
}  // namespace hip_falcons
