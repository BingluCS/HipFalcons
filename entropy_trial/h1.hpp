#ifndef HIP_FALCONS_ENTROPY_TRIAL_H1_HPP
#define HIP_FALCONS_ENTROPY_TRIAL_H1_HPP

#include "../dataloader/h1fast.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#if __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#else
// Allow the host verifier to be built on machines without the HIP SDK.
struct ihipStream_t;
using hipStream_t = ihipStream_t*;
#endif

#if defined(__HIPCC__)
#define HF_H1_HD __host__ __device__
#else
#define HF_H1_HD
#endif

namespace hip_falcons {
namespace entropy_trial {
namespace h1_detail {

constexpr std::size_t kStreams = 4;
constexpr std::uint32_t kTableSize = 1u << 13;
constexpr std::uint32_t kDelta = 1u;

// All offsets are from the beginning of the packed buffer. The first jobs()
// entries in that buffer are these descriptors, in groups of four.
struct Job {
    std::uint64_t table_offset;
    std::uint64_t bits_offset;
    std::uint64_t bits_size;
    std::uint64_t escapes_offset;
    std::uint64_t escapes_count;
    std::uint64_t output_offset;
    std::uint64_t output_count;
    std::uint32_t flags;
    std::uint32_t reserved;
};
static_assert(sizeof(Job) == 64, "Unexpected H1 job layout");

HF_H1_HD inline std::int16_t signed_bits(std::uint16_t value) {
    return static_cast<std::int16_t>(value <= 32767 ? static_cast<int>(value)
                                                     : static_cast<int>(value) - 65536);
}

HF_H1_HD inline unsigned context(int code) {
    const unsigned negative = code < 0;
    const unsigned magnitude = static_cast<unsigned>(negative ? -code : code);
    const unsigned bucket = (magnitude > 1) + (magnitude > 2) + (magnitude > 3) +
                            (magnitude > 4) + (magnitude > 7) + (magnitude > 11);
    return (magnitude != 0) * (1u + 2u * bucket + negative);
}

// FSE reads bit fields backwards, but each field itself is little-endian.
// pos is the number of data bits remaining below the final end marker.
HF_H1_HD inline bool read_bits(const unsigned char* data, std::size_t& pos,
                               unsigned count, unsigned& result) {
    if (count > 13 || pos < count) return false;
    if (!count) {
        result = 0;
        return true;
    }
    pos -= count;
    const std::size_t byte = pos >> 3;
    const unsigned shift = static_cast<unsigned>(pos & 7);
    unsigned value = static_cast<unsigned>(data[byte]) >> shift;
    if (count + shift > 8) value |= static_cast<unsigned>(data[byte + 1]) << (8 - shift);
    if (count + shift > 16) value |= static_cast<unsigned>(data[byte + 2]) << (16 - shift);
    result = value & ((1u << count) - 1u);
    return true;
}

HF_H1_HD inline bool decode_one(const void* packed, std::int16_t* output, std::size_t index) {
    const auto* base = static_cast<const unsigned char*>(packed);
    const Job job = reinterpret_cast<const Job*>(base)[index];
    const auto* table = reinterpret_cast<const std::uint32_t*>(base + job.table_offset);
    const auto* bits = base + job.bits_offset;
    const auto* escapes = reinterpret_cast<const std::int16_t*>(base + job.escapes_offset);
    if (!job.bits_size || !job.output_count || job.bits_size > (static_cast<std::size_t>(-1) / 8))
        return false;
    const unsigned last = bits[job.bits_size - 1];
    if (!last) return false;
    unsigned marker = 0;
    for (unsigned value = last; value > 1; value >>= 1) ++marker;
    std::size_t pos = (static_cast<std::size_t>(job.bits_size) - 1) * 8 + marker;
    unsigned state;
    if (!read_bits(bits, pos, 13, state)) return false;

    unsigned ctx = 0;
    std::size_t escape_index = 0;
    std::uint16_t partial_sum = 0;
    for (std::size_t i = 0; i < job.output_count; ++i) {
        if (state >= kTableSize) return false;
        const std::uint32_t entry = table[static_cast<std::size_t>(ctx) * kTableSize + state];
        const unsigned symbol = (entry >> 16) & 255u;
        int code;
        if (symbol == 255) {
            if (escape_index >= job.escapes_count) return false;
            code = escapes[escape_index++];
        } else {
            code = static_cast<int>(symbol >> 1) ^ -static_cast<int>(symbol & 1u);
        }
        ctx = context(code);
        if (job.flags & kDelta) {
            partial_sum = static_cast<std::uint16_t>(partial_sum + static_cast<std::uint16_t>(code));
            output[job.output_offset + i] = signed_bits(partial_sum);
        } else {
            output[job.output_offset + i] = static_cast<std::int16_t>(code);
        }

        // FSE_initCState2 initializes the final symbol without transition bits.
        if (i + 1 < job.output_count) {
            unsigned low;
            if (!read_bits(bits, pos, entry >> 24, low)) return false;
            state = (entry & 0xffffu) + low;
        }
    }
    return pos == 0 && escape_index == job.escapes_count;
}

// The four output chunks have already been locally prefix-summed. Update only
// their last values first, so the subsequent per-stream pass can read carries
// without racing another stream's writes.
HF_H1_HD inline void prepare_delta_carries(const void* packed, std::int16_t* output,
                                            std::size_t first) {
    const auto* jobs = static_cast<const Job*>(packed);
    if (!(jobs[first].flags & kDelta)) return;
    for (std::size_t k = 1; k < kStreams; ++k) {
        const Job& previous = jobs[first + k - 1];
        const Job& current = jobs[first + k];
        const std::size_t prev_end = previous.output_offset + previous.output_count - 1;
        const std::size_t end = current.output_offset + current.output_count - 1;
        output[end] = signed_bits(static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(output[prev_end]) + static_cast<std::uint16_t>(output[end])));
    }
}

HF_H1_HD inline void apply_delta_carry(const void* packed, std::int16_t* output,
                                       std::size_t index) {
    const auto* jobs = static_cast<const Job*>(packed);
    const Job& current = jobs[index];
    if (!(current.flags & kDelta) || index % kStreams == 0) return;
    const Job& previous = jobs[index - 1];
    const std::uint16_t carry = static_cast<std::uint16_t>(
        output[previous.output_offset + previous.output_count - 1]);
    for (std::size_t i = 0; i + 1 < current.output_count; ++i) {
        const std::size_t offset = current.output_offset + i;
        output[offset] = signed_bits(static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(output[offset]) + carry));
    }
}

struct Segment {
    std::vector<std::uint32_t> table;
    std::array<std::vector<unsigned char>, kStreams> bits;
    std::vector<std::int16_t> escapes;
    std::array<std::size_t, kStreams> escape_counts{};
    std::array<std::size_t, kStreams> output_counts{};
    std::size_t output_offset = 0;
    bool delta = false;
};

inline std::size_t append_region(std::vector<unsigned char>& dst, const void* src,
                                 std::size_t bytes, std::size_t alignment) {
    const std::size_t max = std::numeric_limits<std::size_t>::max();
    if (dst.size() > max - (alignment - 1)) throw std::length_error("H1 packed size overflow");
    const std::size_t offset = (dst.size() + alignment - 1) & ~(alignment - 1);
    if (bytes > max - offset) throw std::length_error("H1 packed size overflow");
    dst.resize(offset + bytes);
    if (bytes) std::memcpy(dst.data() + offset, src, bytes);
    return offset;
}

}  // namespace h1_detail

// append() takes the complete entropy payload starting with the method byte,
// after Zstd and interpolation metadata have been removed. It owns all data
// on success and does not change this batch on false, including allocation
// failures. Only 64-bit little-endian, two's-complement, single-segment H1
// streams with four nonempty chunks and n >= 16 are supported. Both escape
// formats and the method's delta flag are supported. append() checks the
// framing and model but does not fully decode or validate the bitstreams.
struct H1Batch {
    std::size_t code_count = 0;  // Total codes, in append order; treat as read-only.

    std::size_t jobs() const noexcept { return segments_.size() * h1_detail::kStreams; }

    bool append(const unsigned char* payload, std::size_t bytes, std::size_t n) noexcept {
        try {
            return append_impl(payload, bytes, n);
        } catch (...) {
            return false;
        }
    }

    // Descriptors and table/escape offsets are naturally aligned. The base of
    // the returned allocation, and any device copy, must be at least 8-byte aligned.
    std::vector<unsigned char> pack() const {
        using namespace h1_detail;
        if (jobs() > std::numeric_limits<std::size_t>::max() / sizeof(Job))
            throw std::length_error("H1 job count overflow");
        std::vector<Job> descriptors(jobs());
        std::vector<unsigned char> result(jobs() * sizeof(Job));
        std::size_t index = 0;
        for (const Segment& segment : segments_) {
            const std::size_t table = append_region(result, segment.table.data(),
                segment.table.size() * sizeof(std::uint32_t), alignof(std::uint32_t));
            const std::size_t escapes = append_region(result, segment.escapes.data(),
                segment.escapes.size() * sizeof(std::int16_t), alignof(std::int16_t));
            std::size_t escape_index = 0;
            std::size_t output_index = segment.output_offset;
            for (std::size_t k = 0; k < kStreams; ++k) {
                const std::size_t bits = append_region(result, segment.bits[k].data(),
                                                       segment.bits[k].size(), 1);
                descriptors[index++] = Job{table, bits, segment.bits[k].size(),
                    escapes + escape_index * sizeof(std::int16_t), segment.escape_counts[k],
                    output_index, segment.output_counts[k], segment.delta ? kDelta : 0u, 0};
                escape_index += segment.escape_counts[k];
                output_index += segment.output_counts[k];
            }
        }
        if (!descriptors.empty())
            std::memcpy(result.data(), descriptors.data(), descriptors.size() * sizeof(Job));
        return result;
    }

private:
    std::vector<h1_detail::Segment> segments_;

    bool append_impl(const unsigned char* payload, std::size_t bytes, std::size_t n) {
        using namespace h1fast_detail;
        const std::size_t max = std::numeric_limits<std::size_t>::max();
        const std::uint16_t endian = 1;
        const std::int16_t minus_one = -1;
        std::uint16_t signed_representation;
        std::memcpy(&signed_representation, &minus_one, sizeof(signed_representation));
        if (sizeof(std::size_t) != 8 || *reinterpret_cast<const unsigned char*>(&endian) != 1 ||
            signed_representation != 0xffffu || !payload || bytes < 5 || n < 16 ||
            n > max / sizeof(std::int16_t) || code_count > max / sizeof(std::int16_t) - n ||
            segments_.size() >= max / h1_detail::kStreams)
            return false;

        Cursor cursor{payload, bytes};
        std::uint8_t method;
        std::uint16_t lo, mode;
        std::size_t segments, segment_n, streams;
        if (!cursor.read(method) || (method & 0x7f) != 1 || !cursor.read(lo) || !cursor.read(mode) ||
            !read_size(cursor, segments) || segments != 1 || !read_size(cursor, segment_n) ||
            segment_n != n || !read_size(cursor, streams) || streams != h1_detail::kStreams)
            return false;

        h1_detail::Segment segment;
        segment.output_offset = code_count;
        segment.delta = (method & 0x80) != 0;
        std::array<std::size_t, h1_detail::kStreams> lengths{};
        for (std::size_t k = 0; k < h1_detail::kStreams; ++k)
            if (!read_size(cursor, lengths[k]) || !lengths[k] || lengths[k] > max / 8) return false;
        const std::size_t chunk = (n + 3) / 4;
        std::size_t total_escapes = 0;
        for (std::size_t k = 0; k < h1_detail::kStreams; ++k) {
            const std::size_t offset = k * chunk;
            const std::size_t count = offset < n ? std::min(chunk, n - offset) : 0;
            if (!count || !read_size(cursor, segment.escape_counts[k]) ||
                segment.escape_counts[k] > count || segment.escape_counts[k] > n - total_escapes)
                return false;
            segment.output_counts[k] = count;
            total_escapes += segment.escape_counts[k];
        }
        const unsigned char* model = cursor.take(kContexts * kSymbols * sizeof(short));
        if (!model) return false;
        for (std::size_t k = 0; k < h1_detail::kStreams; ++k) {
            const unsigned char* src = cursor.take(lengths[k]);
            if (!src || !src[lengths[k] - 1]) return false;
            unsigned marker = 0;
            for (unsigned last = src[lengths[k] - 1]; last > 1; last >>= 1) ++marker;
            if ((lengths[k] - 1) * 8 + marker < kLog) return false;
            segment.bits[k].assign(src, src + lengths[k]);
        }
        segment.escapes.resize(total_escapes);
        if (!read_escapes(cursor, total_escapes, segment.escapes)) return false;

        std::vector<FSE_DTable> storage(kContexts * FSE_DTABLE_SIZE_U32(kLog));
        FSE_DTable* dt[kContexts];
        if (!build_tables(model, storage.data(), dt)) return false;
        segment.table.resize(static_cast<std::size_t>(kContexts) * kTableSize);
        for (unsigned c = 0; c < kContexts; ++c) {
            const auto* entries = reinterpret_cast<const FSE_decode_t*>(dt[c] + 1);
            for (unsigned s = 0; s < kTableSize; ++s) {
                const FSE_decode_t entry = entries[s];
                segment.table[static_cast<std::size_t>(c) * kTableSize + s] =
                    static_cast<std::uint32_t>(entry.newState) |
                    (static_cast<std::uint32_t>(entry.symbol) << 16) |
                    (static_cast<std::uint32_t>(entry.nbBits) << 24);
            }
        }
        segments_.push_back(std::move(segment));
        code_count += n;
        return true;
    }
};

// The buffer must be an unmodified H1Batch::pack() result, jobs must equal
// that batch's jobs(), and dst must hold code_count int16_t values. This host
// verifier reports invalid bitstreams; dst may be partly written on failure.
inline bool decode_h1_host(const void* packed, std::int16_t* dst, std::size_t jobs) {
    if (jobs == 0) return true;
    if (!packed || !dst || jobs % h1_detail::kStreams) return false;
    for (std::size_t first = 0; first < jobs; first += h1_detail::kStreams) {
        for (std::size_t k = 0; k < h1_detail::kStreams; ++k)
            if (!h1_detail::decode_one(packed, dst, first + k)) return false;
        h1_detail::prepare_delta_carries(packed, dst, first);
        for (std::size_t k = 0; k < h1_detail::kStreams; ++k)
            h1_detail::apply_delta_carry(packed, dst, first + k);
    }
    return true;
}

// Enqueues one thread per bitstream on stream and returns a HIP status as int.
// packed_device is an unmodified device copy of pack(), jobs must be the
// corresponding jobs(), and codes_device must have code_count elements. The
// buffers must not overlap and must remain valid until the stream completes.
// Only immediate HIP launch errors are returned; use the host verifier for
// untrusted bitstreams. Neither buffer is freed or the stream synchronized.
int launch_h1(const void* packed_device, std::int16_t* codes_device,
              std::size_t jobs, hipStream_t stream) noexcept;

}  // namespace entropy_trial
}  // namespace hip_falcons

#undef HF_H1_HD

#endif  // HIP_FALCONS_ENTROPY_TRIAL_H1_HPP
