#ifndef HIP_FALCONS_DATALOADER_H1FAST_HPP
#define HIP_FALCONS_DATALOADER_H1FAST_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#ifndef FSE_STATIC_LINKING_ONLY
#define FSE_STATIC_LINKING_ONLY
#endif
#include "Falcons/encoder/zfse/fse.h"

extern "C" size_t FSE_buildDTable(FSE_DTable*, const short*, unsigned, unsigned);

namespace hip_falcons {
namespace h1fast_detail {

// These are the on-disk constants of RleFseEncoder's order-1 method.
constexpr unsigned kContexts = 15;
constexpr unsigned kSymbols = 256;
constexpr unsigned kLog = 13;
constexpr unsigned kTableSize = 1u << kLog;
constexpr std::size_t kStreams = 4;
constexpr std::size_t kFseBlock = 1u << 16;
static_assert(sizeof(short) == 2 && sizeof(FSE_DTable) == 4 &&
              sizeof(FSE_decode_t) == sizeof(FSE_DTable),
              "Falcons FSE table ABI is unsupported");

struct Cursor {
    const unsigned char* p;
    std::size_t left;

    const unsigned char* take(std::size_t count) noexcept {
        if (count > left) return nullptr;
        const unsigned char* result = p;
        p += count;
        left -= count;
        return result;
    }

    template<class T>
    bool read(T& value) noexcept {
        const unsigned char* src = take(sizeof(T));
        if (!src) return false;
        std::memcpy(&value, src, sizeof(T));
        return true;
    }
};

inline bool read_size(Cursor& cursor, std::size_t& value) noexcept {
    std::uint64_t v;
    if (!cursor.read(v) || v > std::numeric_limits<std::size_t>::max()) return false;
    value = static_cast<std::size_t>(v);
    return true;
}

constexpr unsigned context(int code) noexcept {
    const unsigned negative = code < 0;
    const unsigned magnitude = static_cast<unsigned>(negative ? -code : code);
    const unsigned bucket = (magnitude > 1) + (magnitude > 2) + (magnitude > 3) +
                            (magnitude > 4) + (magnitude > 7) + (magnitude > 11);
    return (magnitude != 0) * (1u + 2u * bucket + negative);
}

constexpr int unzigzag(unsigned symbol) noexcept {
    return static_cast<int>(symbol >> 1) ^ -static_cast<int>(symbol & 1u);
}

constexpr std::array<std::uint8_t, 255> make_symbol_contexts() noexcept {
    std::array<std::uint8_t, 255> contexts{};
    for (unsigned s = 0; s < contexts.size(); ++s)
        contexts[s] = static_cast<std::uint8_t>(context(unzigzag(s)));
    return contexts;
}

constexpr auto kSymbolContexts = make_symbol_contexts();

inline bool build_tables(const unsigned char* model, FSE_DTable* storage,
                         FSE_DTable* (&dt)[kContexts]) {
    short norm[kSymbols];
    for (unsigned c = 0; c < kContexts; ++c) {
        unsigned total = 0;
        for (unsigned s = 0; s < kSymbols; ++s) {
            std::memcpy(&norm[s], model + 2 * (c * kSymbols + s), sizeof(short));
            if (norm[s] < -1 || norm[s] > static_cast<int>(kTableSize)) return false;
            total += norm[s] < 0 ? 1u : static_cast<unsigned>(norm[s]);
        }
        dt[c] = storage + static_cast<std::size_t>(c) * FSE_DTABLE_SIZE_U32(kLog);
        if (total != kTableSize || FSE_isError(FSE_buildDTable(dt[c], norm, 255, kLog)))
            return false;
    }
    return true;
}

inline bool decode_escape_classes(const unsigned char* data, std::size_t size,
                                  std::vector<unsigned char>& classes) {
    Cursor cursor{data, size};
    std::size_t done = 0;
    while (done < classes.size()) {
        const std::size_t count = std::min(kFseBlock, classes.size() - done);
        std::uint8_t mode;
        if (!cursor.read(mode)) return false;
        if (mode == 0) {
            const unsigned char* src = cursor.take(count);
            if (!src) return false;
            std::memcpy(classes.data() + done, src, count);
        } else if (mode == 2) {
            std::uint8_t value;
            if (!cursor.read(value)) return false;
            std::memset(classes.data() + done, value, count);
        } else if (mode == 1) {
            std::uint32_t bytes;
            if (count < 4 || !cursor.read(bytes) || bytes <= 1 || bytes >= count) return false;
            const unsigned char* src = cursor.take(bytes);
            if (!src) return false;
            const std::size_t decoded = FSE_decompress(classes.data() + done, count, src, bytes);
            if (FSE_isError(decoded) || decoded != count) return false;
        } else {
            return false;
        }
        done += count;
    }
    return cursor.left == 0;
}

inline bool read_escapes(Cursor& cursor, std::size_t count, std::vector<std::int16_t>& values) {
    std::uint8_t mode;
    if (!cursor.read(mode)) return false;
    if (mode == 0) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(std::int16_t)) return false;
        const unsigned char* src = cursor.take(count * sizeof(std::int16_t));
        if (!src) return false;
        if (count) std::memcpy(values.data(), src, count * sizeof(std::int16_t));
        return true;
    }
    if (mode != 1 || count == 0) return false;
    std::size_t class_size, mantissa_size;
    if (!read_size(cursor, class_size) || !read_size(cursor, mantissa_size)) return false;
    const unsigned char* class_data = cursor.take(class_size);
    const unsigned char* mantissa = cursor.take(mantissa_size);
    if (!class_data || !mantissa) return false;
    std::vector<unsigned char> classes(count);
    if (!decode_escape_classes(class_data, class_size, classes)) return false;
    std::size_t pos = 0;
    std::uint64_t bits = 0;
    unsigned available = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const unsigned width = classes[i];
        if (width < 8 || width > 16) return false;
        while (available < width) {
            if (pos == mantissa_size) return false;
            bits = (bits << 8) | mantissa[pos++];
            available += 8;
        }
        available -= width;
        const std::uint32_t value = (1u << width) | ((bits >> available) & ((1u << width) - 1u));
        const std::uint32_t zigzag = value - 1;
        if (zigzag > 65535) return false;
        const int code = static_cast<int>(zigzag >> 1) ^ -static_cast<int>(zigzag & 1);
        values[i] = static_cast<std::int16_t>(code);
    }
    return pos == mantissa_size;
}

inline bool decode(const unsigned char* header, std::size_t len, std::size_t n,
                   std::unique_ptr<std::int16_t[]>& result) {
    constexpr std::size_t pad = 64 / sizeof(std::int16_t);
    const std::uint16_t endian = 1;
    const std::int16_t minus_one = -1;
    std::uint16_t signed_bits;
    std::memcpy(&signed_bits, &minus_one, sizeof(signed_bits));
    if (*reinterpret_cast<const unsigned char*>(&endian) != 1 || signed_bits != 0xffffu ||
        sizeof(std::size_t) != 8)
        return false;
    if (!header || len < 5 || n < 16 || n > std::numeric_limits<std::size_t>::max() - 3 ||
        n > std::numeric_limits<std::size_t>::max() / sizeof(std::int16_t) - pad) return false;
    Cursor cursor{header, len};
    std::uint8_t method;
    std::uint16_t lo, mode;
    if (!cursor.read(method) || !cursor.read(lo) || !cursor.read(mode) || (method & 0x7f) != 1)
        return false;
    (void)lo;
    (void)mode;
    std::size_t segments, segment_n, streams;
    if (!read_size(cursor, segments) || segments != 1 || !read_size(cursor, segment_n) ||
        segment_n != n || !read_size(cursor, streams) || streams != kStreams) return false;

    std::size_t byte_lengths[kStreams], escape_counts[kStreams];
    for (std::size_t k = 0; k < kStreams; ++k)
        if (!read_size(cursor, byte_lengths[k]) || byte_lengths[k] == 0) return false;
    const std::size_t chunk = (n + 3) / 4;
    std::size_t total_escapes = 0;
    for (std::size_t k = 0; k < kStreams; ++k) {
        if (!read_size(cursor, escape_counts[k])) return false;
        const std::size_t offset = k * chunk;
        const std::size_t count = offset < n ? std::min(chunk, n - offset) : 0;
        if (!count || escape_counts[k] > count || escape_counts[k] > n - total_escapes) return false;
        total_escapes += escape_counts[k];
    }
    const unsigned char* model = cursor.take(kContexts * kSymbols * sizeof(short));
    if (!model) return false;
    const unsigned char* bitstreams[kStreams];
    for (std::size_t k = 0; k < kStreams; ++k) {
        bitstreams[k] = cursor.take(byte_lengths[k]);
        if (!bitstreams[k]) return false;
    }
    std::vector<std::int16_t> escapes(total_escapes);
    if (!read_escapes(cursor, total_escapes, escapes)) return false;

    std::unique_ptr<FSE_DTable[]> tables(
        new (std::nothrow) FSE_DTable[kContexts * FSE_DTABLE_SIZE_U32(kLog)]);
    FSE_DTable* dt[kContexts];
    if (!tables || !build_tables(model, tables.get(), dt)) return false;
    std::unique_ptr<std::int16_t[]> output(new (std::nothrow) std::int16_t[n + pad]);
    if (!output) return false;
    std::fill_n(output.get() + n, pad, std::int16_t{});
    BIT_DStream_t bd[kStreams]{};
    FSE_DState_t ds[kStreams]{};
    std::size_t ovi[kStreams], escape_end[kStreams], lengths[kStreams];
    std::size_t escape_offset = 0;
    for (std::size_t k = 0; k < kStreams; ++k) {
        if (FSE_isError(BIT_initDStream(&bd[k], bitstreams[k], byte_lengths[k])) ||
            bd[k].bitsConsumed + kLog > 8 * sizeof(bd[k].bitContainer)) return false;
        FSE_initDState(&ds[k], &bd[k], dt[0]);
        ovi[k] = escape_offset;
        escape_offset += escape_counts[k];
        escape_end[k] = escape_offset;
        const std::size_t offset = k * chunk;
        lengths[k] = std::min(chunk, n - offset);
    }

    const void* next_table[256];
    for (int s = 0; s < 255; ++s)
        next_table[s] = dt[kSymbolContexts[s]] + 1;
    std::int16_t* out = output.get();
    const std::int16_t* ovf = escapes.data();
    // Leave the final symbol to the checked tail; it has no transition bits.
    const std::size_t stop = (lengths[3] - 1) & ~std::size_t(3);
    std::size_t jj = 0;
    while (jj < stop) {
        std::size_t groups = std::min<std::size_t>(16, (stop - jj) / 4);
        for (int k = 0; k < 4; ++k)
            groups = std::min(groups, static_cast<std::size_t>(bd[k].ptr - bd[k].start) / 8);
        if (!groups) break;
        const std::size_t end = jj + 4 * groups;
        for (; jj < end; jj += 4) {
            // Each reload can move at most eight bytes; the window above
            // keeps every eight-byte load within its original bitstream.
            for (int k = 0; k < 4; ++k) {
                bd[k].ptr -= bd[k].bitsConsumed >> 3;
                bd[k].bitsConsumed &= 7;
                bd[k].bitContainer = MEM_readLEST(bd[k].ptr);
            }
            for (int j = 0; j < 4; ++j) {
#pragma GCC unroll 4
                for (int k = 0; k < 4; ++k) {
                    const unsigned s = FSE_decodeSymbol(&ds[k], &bd[k]);
                    int code;
                    if (s == 255) {
                        if (ovi[k] == escape_end[k]) return false;
                        code = ovf[ovi[k]++];
                        ds[k].table = dt[context(code)] + 1;
                    } else {
                        code = unzigzag(s);
                        ds[k].table = next_table[s];
                    }
                    out[k * chunk + jj + j] = static_cast<std::int16_t>(code);
                }
            }
        }
    }
    for (std::size_t k = 0; k < kStreams; ++k) {
        for (std::size_t j = jj; j < lengths[k]; ++j) {
            const auto entry = static_cast<const FSE_decode_t*>(ds[k].table)[ds[k].state];
            unsigned s = entry.symbol;
            if (j + 1 < lengths[k]) {
                if (bd[k].bitsConsumed + entry.nbBits > 8 * sizeof(bd[k].bitContainer)) {
                    if (BIT_reloadDStream(&bd[k]) == BIT_DStream_overflow ||
                        bd[k].bitsConsumed + entry.nbBits > 8 * sizeof(bd[k].bitContainer))
                        return false;
                }
                s = FSE_decodeSymbol(&ds[k], &bd[k]);
            }
            int code;
            if (s == 255) {
                if (ovi[k] == escape_end[k]) return false;
                code = ovf[ovi[k]++];
                ds[k].table = dt[context(code)] + 1;
            } else {
                code = unzigzag(s);
                ds[k].table = next_table[s];
            }
            out[k * chunk + j] = static_cast<std::int16_t>(code);
        }
        if (BIT_reloadDStream(&bd[k]) == BIT_DStream_overflow ||
            !BIT_endOfDStream(&bd[k]) || ovi[k] != escape_end[k]) return false;
    }
    if (method & 0x80) {
        std::uint16_t previous = 0;
        for (std::size_t j = 0; j < n; ++j) {
            std::uint16_t bits;
            std::memcpy(&bits, output.get() + j, sizeof(bits));
            previous = static_cast<std::uint16_t>(previous + bits);
            std::memcpy(output.get() + j, &previous, sizeof(previous));
        }
    }
    result = std::move(output);
    return true;
}

}  // namespace h1fast_detail

// entropy_header points to RleFseEncoder::save's method byte; len bounds all
// following data. The caller passes the entropy code count n. Returns false
// without changing out for malformed or unsupported streams, so the caller
// can use the canonical decoder. This optional path supports 64-bit little-
// endian, two's-complement hosts and one segment of at least 16 codes with four
// nonempty streams, including both escape encodings and delta. Link against
// the same vendored FSE implementation used by RleFseEncoder.
inline bool try_decode_h1(const unsigned char* entropy_header, std::size_t len, std::size_t n,
                          std::unique_ptr<std::int16_t[]>& out) noexcept {
    try {
        return h1fast_detail::decode(entropy_header, len, n, out);
    } catch (...) {
        return false;
    }
}

}  // namespace hip_falcons

#endif  // HIP_FALCONS_DATALOADER_H1FAST_HPP
