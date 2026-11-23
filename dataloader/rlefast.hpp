#ifndef HIP_FALCONS_DATALOADER_RLEFAST_HPP
#define HIP_FALCONS_DATALOADER_RLEFAST_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "Falcons/encoder/zfse/fse.h"

namespace hip_falcons {
namespace rlefast_detail {

// Restrict the fast path to one FSE block per class stream. Larger streams
// remain the responsibility of the canonical decoder.
constexpr std::size_t kMaxCodes = 32768;
constexpr std::size_t kPad = 64 / sizeof(std::int16_t);
constexpr std::size_t kChunk = 128;

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

    template <class T>
    bool read(T& value) noexcept {
        const unsigned char* src = take(sizeof(T));
        if (!src) return false;
        std::memcpy(&value, src, sizeof(value));
        return true;
    }
};

inline bool read_size(Cursor& cursor, std::size_t& value) noexcept {
    std::uint64_t encoded;
    if (!cursor.read(encoded) || encoded > std::numeric_limits<std::size_t>::max())
        return false;
    value = static_cast<std::size_t>(encoded);
    return true;
}

struct Classes {
    const unsigned char* data = nullptr;
    unsigned char repeated = 0;
    std::vector<unsigned char> decoded;

    unsigned operator[](std::size_t i) const noexcept {
        return data ? data[i] : repeated;
    }
};

inline bool read_classes(const unsigned char* bytes, std::size_t length,
                         std::size_t count, Classes& classes) {
    if (count == 0) return length == 0;
    Cursor input{bytes, length};
    std::uint8_t flag;
    if (!input.read(flag)) return false;
    if (flag == 0) {
        classes.data = input.take(count);
        if (!classes.data) return false;
    } else if (flag == 2) {
        if (!input.read(classes.repeated)) return false;
    } else if (flag == 1) {
        std::uint32_t compressed;
        if (count < 4 || !input.read(compressed) || compressed <= 1 || compressed >= count)
            return false;
        const unsigned char* src = input.take(compressed);
        if (!src) return false;
        classes.decoded.resize(count);
        const std::size_t result = FSE_decompress(classes.decoded.data(), count, src, compressed);
        if (FSE_isError(result) || result != count) return false;
        classes.data = classes.decoded.data();
    } else {
        return false;
    }
    return input.remaining == 0;
}

// Mantissas are MSB-first; zero-width classes are the common dense-run case.
struct Bits {
    const unsigned char* data;
    std::size_t size;
    std::size_t position = 0;

    bool read(unsigned width, std::uint32_t& value) noexcept {
        if (width == 0) {
            value = 0;
            return true;
        }
        if (position > size * 8 || width > size * 8 - position) return false;
        const std::size_t byte = position >> 3;
        const unsigned bit = static_cast<unsigned>(position & 7);
        std::uint32_t word = static_cast<std::uint32_t>(data[byte]) << 16;
        if (size - byte > 1) word |= static_cast<std::uint32_t>(data[byte + 1]) << 8;
        if (size - byte > 2) word |= data[byte + 2];
        value = (word >> (24 - bit - width)) & ((std::uint32_t{1} << width) - 1);
        position += width;
        return true;
    }

    bool finished() const noexcept {
        if ((position + 7) / 8 != size) return false;
        const unsigned partial = static_cast<unsigned>(position & 7);
        return partial == 0 || (data[size - 1] & ((1u << (8 - partial)) - 1)) == 0;
    }
};

#ifdef __AVX2__
inline __m256i prefix8(__m256i x) noexcept {
    x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
    x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
    const __m256i upper = _mm256_permute2x128_si256(x, x, 0x08);
    return _mm256_add_epi32(x, _mm256_shuffle_epi32(upper, _MM_SHUFFLE(3, 3, 3, 3)));
}

inline __m256i widths8(const Classes& classes, std::size_t offset) noexcept {
    if (!classes.data) return _mm256_set1_epi32(classes.repeated);
    std::uint64_t packed;
    std::memcpy(&packed, classes.data + offset, sizeof(packed));
    return _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(packed)));
}

// Use a single 16-byte window for nearby starts; retain bounded fallbacks.
template <unsigned MaxWidth>
inline bool gamma8(Bits& bits, __m256i widths, __m256i& gamma) noexcept {
    const __m256i ones = _mm256_set1_epi32(1);
    if (_mm256_testz_si256(widths, widths)) {
        gamma = ones;
        return true;
    }
    if constexpr (MaxWidth == 15) {
        const __m256i upper = _mm256_srli_epi32(widths, 1);
        if (_mm256_testz_si256(upper, upper)) {
            const __m256i sums = prefix8(widths);
            const std::size_t total = static_cast<unsigned>(_mm256_extract_epi32(sums, 7));
            if (bits.position > bits.size * 8 || total > bits.size * 8 - bits.position)
                return false;
            const std::size_t first = bits.position >> 3;
            std::uint32_t word = static_cast<std::uint32_t>(bits.data[first]) << 8;
            if (bits.size - first > 1) word |= bits.data[first + 1];
            const __m256i starts = _mm256_add_epi32(_mm256_sub_epi32(sums, widths),
                _mm256_set1_epi32(static_cast<int>(bits.position & 7)));
            const __m256i shift = _mm256_sub_epi32(_mm256_set1_epi32(15), starts);
            const __m256i low = _mm256_and_si256(widths,
                _mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(word)), shift));
            gamma = _mm256_add_epi32(ones, _mm256_add_epi32(widths, low));
            bits.position += total;
            return true;
        }
    }
    if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(widths, _mm256_set1_epi32(MaxWidth))))
        return false;
    const __m256i sums = prefix8(widths);
    const std::size_t total = static_cast<unsigned>(_mm256_extract_epi32(sums, 7));
    if (bits.position > bits.size * 8 || total > bits.size * 8 - bits.position) return false;
    const unsigned last_width = static_cast<unsigned>(_mm256_extract_epi32(widths, 7));
    const std::size_t last_byte = (bits.position + total - last_width) >> 3;
    const std::size_t first_byte = bits.position >> 3;
    const bool use_window = bits.size - first_byte >= 16 && last_byte - first_byte <= 12;
    if (!use_window && (bits.size < 4 || last_byte > bits.size - 4)) {
        alignas(32) std::uint32_t scalar_widths[8], scalar_gamma[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(scalar_widths), widths);
        for (std::size_t i = 0; i < 8; ++i) {
            std::uint32_t low;
            if (!bits.read(scalar_widths[i], low)) return false;
            scalar_gamma[i] = (std::uint32_t{1} << scalar_widths[i]) | low;
        }
        gamma = _mm256_load_si256(reinterpret_cast<const __m256i*>(scalar_gamma));
        return true;
    }

    const __m256i starts = _mm256_add_epi32(_mm256_sub_epi32(sums, widths),
                                             _mm256_set1_epi32(static_cast<int>(bits.position)));
    const __m256i offsets = _mm256_srli_epi32(starts, 3);
    __m256i big;
    if (use_window) {
        const __m128i window = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bits.data + first_byte));
        const __m256i local = _mm256_sub_epi32(offsets, _mm256_set1_epi32(static_cast<int>(first_byte)));
        const __m256i mask = _mm256_add_epi32(
            _mm256_mullo_epi32(local, _mm256_set1_epi32(0x01010101)),
            _mm256_set1_epi32(0x00010203));
        big = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(window), mask);
    } else {
        alignas(32) std::uint32_t byte_offsets[8], words[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(byte_offsets), offsets);
        for (std::size_t i = 0; i < 8; ++i)
            std::memcpy(&words[i], bits.data + byte_offsets[i], sizeof(words[i]));
        const __m256i swap = _mm256_setr_epi8(
            3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
            3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
        big = _mm256_shuffle_epi8(_mm256_load_si256(reinterpret_cast<const __m256i*>(words)), swap);
    }
    const __m256i shift = _mm256_sub_epi32(_mm256_set1_epi32(32),
        _mm256_add_epi32(widths, _mm256_and_si256(starts, _mm256_set1_epi32(7))));
    const __m256i leading = _mm256_sllv_epi32(ones, widths);
    const __m256i low = _mm256_and_si256(_mm256_srlv_epi32(big, shift),
                                          _mm256_sub_epi32(leading, ones));
    gamma = _mm256_or_si256(leading, low);
    bits.position += total;
    return true;
}
#endif

inline bool decode(const unsigned char* header, std::size_t length, std::size_t n,
                   std::unique_ptr<std::int16_t[]>& result) {
    const std::uint16_t endian = 1;
    const std::int16_t minus_one = -1;
    std::uint16_t minus_one_bits;
    std::memcpy(&minus_one_bits, &minus_one, sizeof(minus_one_bits));
    if (!header || n > kMaxCodes || sizeof(std::size_t) != 8 ||
        *reinterpret_cast<const unsigned char*>(&endian) != 1 || minus_one_bits != 0xffffu)
        return false;

    Cursor input{header, length};
    std::uint8_t method;
    std::uint16_t lo, mode_bits;
    if (!input.read(method) || !input.read(lo) || !input.read(mode_bits) ||
        (method & 0x7f) != 0)
        return false;
    (void)lo;

    std::size_t nr, nv, run_bytes, value_bytes, run_mantissa_bytes, value_mantissa_bytes;
    if (!read_size(input, nr) || !read_size(input, nv) ||
        !read_size(input, run_bytes) || !read_size(input, value_bytes) ||
        !read_size(input, run_mantissa_bytes) || !read_size(input, value_mantissa_bytes) ||
        nr > n || nv > nr || nr - nv > 1 || (n != 0 && nr == 0) ||
        run_mantissa_bytes > (15 * nr + 7) / 8 ||
        value_mantissa_bytes > (17 * nv + 7) / 8)
        return false;

    const unsigned char* run_classes = input.take(run_bytes);
    const unsigned char* value_classes = input.take(value_bytes);
    const unsigned char* run_mantissa = input.take(run_mantissa_bytes);
    const unsigned char* value_mantissa = input.take(value_mantissa_bytes);
    if (!run_classes || !value_classes || !run_mantissa || !value_mantissa) return false;

    Classes runs, values;
    if (!read_classes(run_classes, run_bytes, nr, runs) ||
        !read_classes(value_classes, value_bytes, nv, values))
        return false;
    Bits run_bits{run_mantissa, run_mantissa_bytes};
    Bits value_bits{value_mantissa, value_mantissa_bytes};
    std::unique_ptr<std::int16_t[]> output(new std::int16_t[n + kPad]);
    std::int16_t mode;
    std::memcpy(&mode, &mode_bits, sizeof(mode));
    std::fill_n(output.get(), n, mode);
    std::fill_n(output.get() + n, kPad, std::int16_t{});

    // A run L and its following value advance by L+1. Process the two gamma
    // streams independently so the dense case avoids a fill call per run.
    std::array<std::uint32_t, kChunk> positions{};
    std::size_t produced = 0;
    std::size_t base = 0;
#ifdef __AVX2__
    for (; nv - base >= 8; base += 8) {
        const __m256i run_widths = widths8(runs, base);
        const bool consecutive = _mm256_testz_si256(run_widths, run_widths) != 0;
        const std::size_t start = produced;
        alignas(32) std::uint32_t indices[8];
        if (consecutive) {
            if (n - produced < 8) return false;
            produced += 8;
        } else {
            __m256i advances;
            if (!gamma8<15>(run_bits, run_widths, advances)) return false;
            const __m256i offsets = prefix8(advances);
            const std::size_t total = static_cast<unsigned>(_mm256_extract_epi32(offsets, 7));
            if (total > n - produced) return false;
            produced += total;
            _mm256_store_si256(reinterpret_cast<__m256i*>(indices),
                _mm256_add_epi32(offsets, _mm256_set1_epi32(static_cast<int>(start) - 1)));
        }

        __m256i gamma;
        if (!gamma8<17>(value_bits, widths8(values, base), gamma)) return false;
        const __m256i one = _mm256_set1_epi32(1);
        const __m256i zigzag = _mm256_sub_epi32(gamma, one);
        const __m256i sign = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_and_si256(zigzag, one));
        const __m256i decoded = _mm256_add_epi32(
            _mm256_xor_si256(_mm256_srli_epi32(zigzag, 1), sign), _mm256_set1_epi32(mode_bits));
        const __m256i bad = _mm256_or_si256(
            _mm256_or_si256(_mm256_cmpgt_epi32(decoded, _mm256_set1_epi32(32767)),
                            _mm256_cmpgt_epi32(_mm256_set1_epi32(-32768), decoded)),
            _mm256_cmpeq_epi32(decoded, _mm256_set1_epi32(mode)));
        if (_mm256_movemask_epi8(bad)) return false;
        const __m128i packed = _mm_packs_epi32(_mm256_castsi256_si128(decoded),
                                                _mm256_extracti128_si256(decoded, 1));
        if (consecutive) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output.get() + start), packed);
        } else {
            alignas(16) std::int16_t decoded_values[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(decoded_values), packed);
            for (std::size_t i = 0; i < 8; ++i) output[indices[i]] = decoded_values[i];
        }
    }
#endif
    for (; base < nv; base += kChunk) {
        const std::size_t count = std::min(kChunk, nv - base);
        for (std::size_t i = 0; i < count; ++i) {
            const unsigned width = runs[base + i];
            if (width > 15) return false;
            std::uint32_t low;
            if (!run_bits.read(width, low)) return false;
            const std::size_t advance = (std::size_t{1} << width) | low;
            if (advance > n - produced) return false;
            produced += advance;
            positions[i] = static_cast<std::uint32_t>(produced - 1);
        }
        for (std::size_t i = 0; i < count; ++i) {
            const unsigned width = values[base + i];
            if (width > 17) return false;
            std::uint32_t low;
            if (!value_bits.read(width, low)) return false;
            const std::uint32_t zigzag = ((std::uint32_t{1} << width) | low) - 1;
            const std::int32_t difference = (zigzag & 1)
                ? -static_cast<std::int32_t>((zigzag >> 1) + 1)
                : static_cast<std::int32_t>(zigzag >> 1);
            const std::int32_t value = static_cast<std::int32_t>(mode_bits) + difference;
            if (value < std::numeric_limits<std::int16_t>::min() ||
                value > std::numeric_limits<std::int16_t>::max() || value == mode)
                return false;
            output[positions[i]] = static_cast<std::int16_t>(value);
        }
    }
    if (nr != nv) {
        const unsigned width = runs[nv];
        if (width > 15) return false;
        std::uint32_t low;
        if (!run_bits.read(width, low)) return false;
        const std::size_t final_run = ((std::size_t{1} << width) | low) - 1;
        if (final_run == 0 || final_run != n - produced) return false;
        produced += final_run;
    }
    if (produced != n || !run_bits.finished() || !value_bits.finished()) return false;

    if (method & 0x80) {
        std::uint16_t previous = 0;
        for (std::size_t i = 0; i < n; ++i) {
            std::uint16_t bits;
            std::memcpy(&bits, output.get() + i, sizeof(bits));
            previous = static_cast<std::uint16_t>(previous + bits);
            std::memcpy(output.get() + i, &previous, sizeof(previous));
        }
    }
    result = std::move(output);
    return true;
}

}  // namespace rlefast_detail

// Decode one small method-0 entropy stream, including the optional delta flag.
// entropy_header points to the five-byte encoder header and len bounds the
// remaining input. Returns false without changing out for unsupported or
// malformed streams; callers can then use RleFseEncoder's canonical decoder.
// Requires the same vendored FSE implementation and a 64-bit little-endian,
// two's-complement host. The result has 64 bytes of zero padding.
inline bool try_decode_rle(const unsigned char* entropy_header, std::size_t len,
                           std::size_t n, std::unique_ptr<std::int16_t[]>& out) noexcept {
    try {
        return rlefast_detail::decode(entropy_header, len, n, out);
    } catch (...) {
        return false;
    }
}

}  // namespace hip_falcons

#endif  // HIP_FALCONS_DATALOADER_RLEFAST_HPP
