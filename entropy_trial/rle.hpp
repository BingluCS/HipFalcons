#ifndef HIP_FALCONS_ENTROPY_TRIAL_RLE_HPP
#define HIP_FALCONS_ENTROPY_TRIAL_RLE_HPP

#if __has_include(<hip/hip_runtime_api.h>)
#include <hip/hip_runtime_api.h>
#else
struct ihipStream_t;
using hipStream_t = ihipStream_t*;
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "Falcons/encoder/zfse/fse.h"

namespace hip_falcons {
namespace entropy_trial {
namespace rle_detail {

constexpr std::size_t kMaxCodes = 32768;
constexpr std::uint32_t kClassArray = 256;

// Offsets are bytes from the start of the packed device allocation, except
// output_offset, which counts int16_t elements in the concatenated output.
struct alignas(8) Job {
    std::uint64_t output_offset = 0;
    std::uint64_t run_class_offset = 0;
    std::uint64_t value_class_offset = 0;
    std::uint64_t run_mantissa_offset = 0;
    std::uint64_t value_mantissa_offset = 0;
    std::uint32_t count = 0;
    std::uint32_t run_count = 0;
    std::uint32_t value_count = 0;
    std::uint32_t run_mantissa_bytes = 0;
    std::uint32_t value_mantissa_bytes = 0;
    std::uint32_t run_repeat = kClassArray;
    std::uint32_t value_repeat = kClassArray;
    std::uint32_t mode_bits = 0;
};
static_assert(std::is_trivially_copyable<Job>::value, "packed descriptor must be trivially copyable");
static_assert(sizeof(Job) % 8 == 0, "packed descriptors must remain aligned");

struct Cursor {
    const unsigned char* p;
    std::size_t remaining;

    const unsigned char* take(std::size_t n) noexcept {
        if (n > remaining) return nullptr;
        const unsigned char* result = p;
        p += n;
        remaining -= n;
        return result;
    }

    bool u8(std::uint8_t& value) noexcept {
        const auto* src = take(1);
        if (!src) return false;
        value = src[0];
        return true;
    }

    bool u16(std::uint16_t& value) noexcept {
        const auto* src = take(2);
        if (!src) return false;
        value = static_cast<std::uint16_t>(src[0] | (std::uint16_t{src[1]} << 8));
        return true;
    }

    bool u32(std::uint32_t& value) noexcept {
        const auto* src = take(4);
        if (!src) return false;
        value = std::uint32_t{src[0]} | (std::uint32_t{src[1]} << 8) |
                (std::uint32_t{src[2]} << 16) | (std::uint32_t{src[3]} << 24);
        return true;
    }

    bool size(std::size_t& value) noexcept {
        const auto* src = take(8);
        if (!src) return false;
        std::uint64_t word = 0;
        for (unsigned i = 0; i < 8; ++i) word |= std::uint64_t{src[i]} << (8 * i);
        if (word > std::numeric_limits<std::size_t>::max()) return false;
        value = static_cast<std::size_t>(word);
        return true;
    }
};

struct Classes {
    const unsigned char* data = nullptr;
    std::vector<unsigned char> decoded;
    std::uint32_t repeated = kClassArray;
    std::size_t bits = 0;
};

inline bool read_classes(const unsigned char* src, std::size_t bytes, std::size_t count,
                         unsigned max_width, Classes& result) {
    if (count == 0) return bytes == 0;
    Cursor input{src, bytes};
    std::uint8_t flag;
    if (!input.u8(flag)) return false;
    if (flag == 0) {
        result.data = input.take(count);
        if (!result.data) return false;
    } else if (flag == 2) {
        std::uint8_t repeated;
        if (!input.u8(repeated) || repeated > max_width) return false;
        result.repeated = repeated;
        result.bits = count * repeated;
    } else if (flag == 1) {
        std::uint32_t compressed;
        if (count < 4 || !input.u32(compressed) || compressed <= 1 || compressed >= count)
            return false;
        const auto* block = input.take(compressed);
        if (!block) return false;
        result.decoded.resize(count);
        const std::size_t produced = FSE_decompress(result.decoded.data(), count, block, compressed);
        if (FSE_isError(produced) || produced != count) return false;
        result.data = result.decoded.data();
    } else {
        return false;
    }
    if (input.remaining != 0) return false;
    if (result.repeated == kClassArray) {
        for (std::size_t i = 0; i < count; ++i) {
            if (result.data[i] > max_width) return false;
            result.bits += result.data[i];
        }
    }
    return true;
}

inline bool valid_mantissa(const unsigned char* data, std::size_t bytes,
                           std::size_t bits) noexcept {
    if (bytes != (bits + 7) / 8) return false;
    const unsigned partial = static_cast<unsigned>(bits & 7);
    return partial == 0 || (data[bytes - 1] & ((1u << (8 - partial)) - 1)) == 0;
}

inline bool little_endian() noexcept {
    const std::uint16_t one = 1;
    return *reinterpret_cast<const unsigned char*>(&one) == 1;
}

inline std::int32_t signed_mode(std::uint32_t bits) noexcept {
    return bits < 32768 ? static_cast<std::int32_t>(bits) : static_cast<std::int32_t>(bits) - 65536;
}

template <class T>
inline void reserve_growth(std::vector<T>& buffer, std::size_t required) {
    if (required <= buffer.capacity()) return;
    const std::size_t limit = buffer.max_size();
    if (required > limit) throw std::length_error("RLE batch capacity overflow");
    const std::size_t capacity = buffer.capacity();
    const std::size_t increment = capacity / 2 + 1;
    const std::size_t grown = increment > limit - capacity ? limit : capacity + increment;
    buffer.reserve(std::max(required, grown));
}

inline std::uint32_t read_gamma(const unsigned char* data, std::size_t& position,
                                unsigned width) noexcept {
    std::uint32_t low = 0;
    for (unsigned i = 0; i < width; ++i, ++position)
        low = (low << 1) | ((data[position >> 3] >> (7 - (position & 7))) & 1u);
    return (std::uint32_t{1} << width) | low;
}

}  // namespace rle_detail

// The caller removes Zstd and interpolation metadata before append(). Class
// Supports little-endian method 0 without delta and at most 32768 codes per
// job. Class decoding and framing checks remain on the host; semantic
// validation of mantissas is left to decode_rle_host or trusted Falcons input.
class RleBatch {
public:
    // payload begins at Falcons' method byte, before the two 16-bit fields.
    // Successful appends preserve input order; failure leaves the batch unchanged.
    bool append(const unsigned char* payload, std::size_t bytes, std::size_t n) noexcept {
        try {
            return append_impl(payload, bytes, n);
        } catch (...) {
            return false;
        }
    }

    // Device copies must be aligned to at least alignof(rle_detail::Job).
    std::vector<unsigned char> pack() const {
        using rle_detail::Job;
        const std::size_t max = std::numeric_limits<std::size_t>::max();
        if (jobs_.size() > max / sizeof(Job)) throw std::length_error("RLE descriptor size overflow");
        const std::size_t header_bytes = jobs_.size() * sizeof(Job);
        if (data_.size() > max - header_bytes) throw std::length_error("RLE packed size overflow");
        std::vector<unsigned char> packed(header_bytes + data_.size());
        for (std::size_t i = 0; i < jobs_.size(); ++i) {
            Job job = jobs_[i];
            if (job.run_repeat == rle_detail::kClassArray) job.run_class_offset += header_bytes;
            if (job.value_repeat == rle_detail::kClassArray) job.value_class_offset += header_bytes;
            job.run_mantissa_offset += header_bytes;
            job.value_mantissa_offset += header_bytes;
            std::memcpy(packed.data() + i * sizeof(Job), &job, sizeof(job));
        }
        if (!data_.empty()) std::memcpy(packed.data() + header_bytes, data_.data(), data_.size());
        return packed;
    }

    std::size_t jobs() const noexcept { return jobs_.size(); }
    std::size_t code_count = 0;  // Total in append order; treat as read-only.

private:
    bool append_impl(const unsigned char* payload, std::size_t bytes, std::size_t n) {
        using namespace rle_detail;
        if (!payload || !little_endian() || n > kMaxCodes ||
            n > std::numeric_limits<std::size_t>::max() - code_count)
            return false;

        Cursor input{payload, bytes};
        std::uint8_t method;
        std::uint16_t lo, mode;
        std::size_t nr, nv, rbytes, vbytes, rmbytes, vmbytes;
        if (!input.u8(method) || method != 0 || !input.u16(lo) || !input.u16(mode) ||
            !input.size(nr) || !input.size(nv) || !input.size(rbytes) || !input.size(vbytes) ||
            !input.size(rmbytes) || !input.size(vmbytes) || nr > n || nv > nr || nr - nv > 1 ||
            (n != 0 && nr == 0) || rmbytes > (15 * nr + 7) / 8 || vmbytes > (17 * nv + 7) / 8)
            return false;
        (void)lo;

        const auto* rclass = input.take(rbytes);
        const auto* vclass = input.take(vbytes);
        const auto* rman = input.take(rmbytes);
        const auto* vman = input.take(vmbytes);
        if (!rclass || !vclass || !rman || !vman) return false;
        Classes runs, values;
        if (!read_classes(rclass, rbytes, nr, 15, runs) ||
            !read_classes(vclass, vbytes, nv, 17, values) ||
            !valid_mantissa(rman, rmbytes, runs.bits) ||
            !valid_mantissa(vman, vmbytes, values.bits))
            return false;

        Job job;
        job.output_offset = code_count;
        job.count = static_cast<std::uint32_t>(n);
        job.run_count = static_cast<std::uint32_t>(nr);
        job.value_count = static_cast<std::uint32_t>(nv);
        job.run_mantissa_bytes = static_cast<std::uint32_t>(rmbytes);
        job.value_mantissa_bytes = static_cast<std::uint32_t>(vmbytes);
        job.run_repeat = runs.repeated;
        job.value_repeat = values.repeated;
        job.mode_bits = mode;

        const std::size_t run_class_bytes = runs.repeated == kClassArray ? nr : 0;
        const std::size_t value_class_bytes = values.repeated == kClassArray ? nv : 0;
        const std::size_t addition_bytes = run_class_bytes + value_class_bytes + rmbytes + vmbytes;
        const std::size_t max = std::numeric_limits<std::size_t>::max();
        if (addition_bytes > max - data_.size() || jobs_.size() == max) return false;
        std::size_t next = data_.size();
        job.run_class_offset = next; next += run_class_bytes;
        job.value_class_offset = next; next += value_class_bytes;
        job.run_mantissa_offset = next; next += rmbytes;
        job.value_mantissa_offset = next;

        // Reserve both containers before committing either logical size.
        reserve_growth(jobs_, jobs_.size() + 1);
        reserve_growth(data_, data_.size() + addition_bytes);
        auto add = [&](const unsigned char* src, std::size_t size) {
            if (size) data_.insert(data_.end(), src, src + size);
        };
        add(runs.data, run_class_bytes);
        add(values.data, value_class_bytes);
        add(rman, rmbytes);
        add(vman, vmbytes);
        jobs_.push_back(job);
        code_count += n;
        return true;
    }

    std::vector<rle_detail::Job> jobs_;
    std::vector<unsigned char> data_;
};

// The packed pointer and jobs must refer to an unmodified RleBatch::pack() result.
// This verifier also rejects semantically invalid run/value sequences; append
// checks their structure without unpacking mantissas on the CPU.
inline bool decode_rle_host(const void* packed_host, std::int16_t* codes_host,
                            std::size_t jobs) noexcept {
    using namespace rle_detail;
    if (!jobs) return true;
    if (!packed_host) return false;
    const auto* packed = static_cast<const unsigned char*>(packed_host);
    for (std::size_t j = 0; j < jobs; ++j) {
        Job job;
        std::memcpy(&job, packed + j * sizeof(Job), sizeof(job));
        if (job.count && !codes_host) return false;
        auto* output = job.count ? codes_host + job.output_offset : nullptr;
        const auto mode = static_cast<std::int16_t>(signed_mode(job.mode_bits));
        if (job.count) std::fill_n(output, job.count, mode);
        const auto* rclass = packed + job.run_class_offset;
        const auto* vclass = packed + job.value_class_offset;
        const auto* rman = packed + job.run_mantissa_offset;
        const auto* vman = packed + job.value_mantissa_offset;
        std::size_t rpos = 0, vpos = 0, produced = 0;
        for (std::size_t i = 0; i < job.value_count; ++i) {
            const unsigned rw = job.run_repeat == kClassArray ? rclass[i] : job.run_repeat;
            const unsigned vw = job.value_repeat == kClassArray ? vclass[i] : job.value_repeat;
            const std::uint32_t advance = read_gamma(rman, rpos, rw);
            if (advance > job.count - produced) return false;
            produced += advance;
            const std::uint32_t zigzag = read_gamma(vman, vpos, vw) - 1;
            const std::int32_t difference = (zigzag & 1)
                ? -static_cast<std::int32_t>((zigzag >> 1) + 1)
                : static_cast<std::int32_t>(zigzag >> 1);
            const std::int32_t value = static_cast<std::int32_t>(job.mode_bits) + difference;
            if (value < -32768 || value > 32767 || value == mode) return false;
            output[produced - 1] = static_cast<std::int16_t>(value);
        }
        if (job.run_count != job.value_count) {
            const std::size_t i = job.value_count;
            const unsigned rw = job.run_repeat == kClassArray ? rclass[i] : job.run_repeat;
            const std::size_t final_run = read_gamma(rman, rpos, rw) - 1;
            if (final_run == 0 || final_run != job.count - produced) return false;
            produced += final_run;
        }
        if (produced != job.count || (rpos + 7) / 8 != job.run_mantissa_bytes ||
            (vpos + 7) / 8 != job.value_mantissa_bytes)
            return false;
    }
    return true;
}

// Enqueues one block per job on stream; returns a HIP launch status without
// synchronizing. The caller owns and retains both device allocations.
int launch_rle(const void* packed_device, std::int16_t* codes_device,
               std::size_t jobs, hipStream_t stream);

}  // namespace entropy_trial
}  // namespace hip_falcons

#endif  // HIP_FALCONS_ENTROPY_TRIAL_RLE_HPP
