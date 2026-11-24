// Host half of the Falcons 1.1.0 interpolation decoder.
#pragma once
#include "zstd_frame.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Falcons/encoder/RleFseEncoder.hpp"
#include "Falcons/lossless/Lossless_zstd.hpp"
#include "Falcons/utils/BufferBounds.hpp"
#include "Falcons/utils/Config.hpp"
#include "fgpu_recon.h"

namespace fgpu {

template <class T>
struct HostStreams {
    // The pointers remain valid until this object is destroyed or moved from.
    std::vector<int16_t *> codes;
    std::vector<T> anchors, unpred;
    int numStreams = 0;
    PlanParams params;
    Plan plan;

    HostStreams() = default;
    HostStreams(const HostStreams &) = delete;
    HostStreams &operator=(const HostStreams &) = delete;
    HostStreams(HostStreams &&) noexcept = default;
    HostStreams &operator=(HostStreams &&) noexcept = default;

    // Kept separately from codes so existing ReconInput callers can use raw pointers.
    std::vector<std::unique_ptr<int16_t[]>> ownedCodes;
};

namespace current_detail {

class Reader {
public:
    Reader(const unsigned char *p, size_t n) : p_(p), n_(n) {}

    template <class V> V read() {
        V value;
        std::memcpy(&value, take(sizeof(V)), sizeof(V));
        return value;
    }
    const unsigned char *take(size_t n) {
        if (n > n_) throw std::invalid_argument("truncated Falcons 1.1.0 stream");
        const auto *p = p_;
        p_ += n;
        n_ -= n;
        return p;
    }
    size_t remaining() const { return n_; }
    const unsigned char *position() const { return p_; }

private:
    const unsigned char *p_;
    size_t n_;
};

inline size_t checked_size(uint64_t n, size_t limit, const char *what) {
    if (n > limit) throw std::invalid_argument(what);
    return static_cast<size_t>(n);
}

// The method-0/1 decoder in the bundled encoder uses unbounded internal reads.
// Decode its FSE blocks and bit streams here while checking their boundaries.
inline std::vector<uint8_t> fse_classes(Reader &in, size_t count) {
    std::vector<uint8_t> out(count);
    size_t at = 0;
    while (at < count) {
        const size_t n = std::min<size_t>(65536, count - at);
        const uint8_t flag = in.read<uint8_t>();
        if (flag == 0) {
            std::memcpy(out.data() + at, in.take(n), n);
        } else if (flag == 2) {
            std::memset(out.data() + at, in.read<uint8_t>(), n);
        } else if (flag == 1) {
            const uint32_t len = in.read<uint32_t>();
            if (len <= 1 || len >= n) throw std::invalid_argument("invalid FSE block length");
            const auto *src = in.take(len);
            const size_t got = FSE_decompress(out.data() + at, n, src, len);
            if (FSE_isError(got) || got != n) throw std::invalid_argument("invalid FSE block");
        } else {
            throw std::invalid_argument("invalid FSE block mode");
        }
        at += n;
    }
    if (in.remaining()) throw std::invalid_argument("extra FSE class bytes");
    return out;
}

class ForwardBits {
public:
    ForwardBits(const unsigned char *p, size_t n) : p_(p), n_(n) {}
    uint64_t read(unsigned n) {
        uint64_t value = 0;
        while (n) {
            if (!left_) {
                if (pos_ == n_) throw std::invalid_argument("truncated entropy mantissa");
                byte_ = p_[pos_++];
                left_ = 8;
            }
            const unsigned take = std::min(n, left_);
            left_ -= take;
            value = (value << take) | ((byte_ >> left_) & ((1u << take) - 1));
            n -= take;
        }
        return value;
    }
    void finish() const {
        if (pos_ != n_ || (left_ && (byte_ & ((1u << left_) - 1))))
            throw std::invalid_argument("invalid entropy mantissa length");
    }
private:
    const unsigned char *p_;
    size_t n_, pos_ = 0;
    unsigned byte_ = 0, left_ = 0;
};

inline int16_t signed_value(uint64_t base, uint64_t bits) {
    const uint64_t zz = (base | bits) - 1;
    if (zz > 65535) throw std::invalid_argument("entropy value outside int16 range");
    const int32_t value = static_cast<int32_t>(zz >> 1) ^ -static_cast<int32_t>(zz & 1);
    return static_cast<int16_t>(value);
}

inline void decode_rle(Reader &in, size_t count, int16_t *out) {
    const size_t nr = checked_size(in.read<uint64_t>(), count, "invalid RLE run count");
    const size_t nv = checked_size(in.read<uint64_t>(), count, "invalid RLE value count");
    const size_t sr = checked_size(in.read<uint64_t>(), in.remaining(), "invalid RLE block size");
    const size_t sv = checked_size(in.read<uint64_t>(), in.remaining(), "invalid RLE block size");
    const size_t rb = checked_size(in.read<uint64_t>(), in.remaining(), "invalid RLE bit size");
    const size_t vb = checked_size(in.read<uint64_t>(), in.remaining(), "invalid RLE bit size");
    Reader rcl(in.take(sr), sr), vcl(in.take(sv), sv);
    auto runs = fse_classes(rcl, nr);
    auto vals = fse_classes(vcl, nv);
    const auto *run_bits = in.take(rb);
    const auto *val_bits = in.take(vb);
    ForwardBits rbits(run_bits, rb), vbits(val_bits, vb);
    size_t produced = 0, vi = 0;
    for (size_t i = 0; i < nr; ++i) {
        const unsigned k = runs[i];
        if (k > 63) throw std::invalid_argument("invalid RLE run class");
        const uint64_t length = ((uint64_t(1) << k) | rbits.read(k)) - 1;
        if (length > count - produced) throw std::invalid_argument("RLE run exceeds quantization count");
        std::fill_n(out + produced, static_cast<size_t>(length), int16_t(0));
        produced += static_cast<size_t>(length);
        if (produced == count) {
            if (i + 1 != nr) throw std::invalid_argument("extra RLE runs");
            break;
        }
        if (vi == nv || vals[vi] > 16) throw std::invalid_argument("invalid RLE value class");
        const unsigned v = vals[vi++];
        out[produced++] = signed_value(uint64_t(1) << v, vbits.read(v));
    }
    if (produced != count || vi != nv) throw std::invalid_argument("invalid RLE quantization count");
    rbits.finish();
    vbits.finish();
}

class ReverseBits {
public:
    ReverseBits(const unsigned char *p, size_t n) : p_(p), pos_(n) {
        if (!n || !p_[n - 1]) throw std::invalid_argument("invalid tANS end marker");
        unsigned last = p_[--pos_];
        for (unsigned v = last; v > 1; v >>= 1) ++available_;
        bits_ = last & ((1u << available_) - 1);
    }
    unsigned read(unsigned count) {
        while (available_ < count && pos_) {
            bits_ = (bits_ << 8) | p_[--pos_];
            available_ += 8;
        }
        if (available_ < count) throw std::invalid_argument("truncated tANS bit stream");
        available_ -= count;
        return static_cast<unsigned>((bits_ >> available_) & ((1u << count) - 1));
    }
    bool finished() const { return pos_ == 0 && available_ == 0; }
private:
    const unsigned char *p_;
    size_t pos_;
    uint64_t bits_ = 0;
    unsigned available_ = 0;
};

inline int context(int code) {
    const bool negative = code < 0;
    const int magnitude = negative ? -code : code;
    if (!magnitude) return 0;
    const int bucket = (magnitude > 1) + (magnitude > 2) + (magnitude > 3) +
                       (magnitude > 4) + (magnitude > 7) + (magnitude > 11);
    return 1 + bucket * 2 + negative;
}

inline std::vector<int16_t> read_overflow(Reader &in, size_t count) {
    const uint8_t mode = in.read<uint8_t>();
    std::vector<int16_t> out(count);
    if (mode == 0) {
        const auto *src = in.take(count * sizeof(int16_t));
        if (count) std::memcpy(out.data(), src, count * sizeof(int16_t));
    } else if (mode == 1 && count) {
        const size_t nc = checked_size(in.read<uint64_t>(), in.remaining(), "invalid tANS overflow size");
        const size_t nb = checked_size(in.read<uint64_t>(), in.remaining(), "invalid tANS overflow size");
        Reader cls(in.take(nc), nc);
        auto classes = fse_classes(cls, count);
        ForwardBits bits(in.take(nb), nb);
        for (size_t i = 0; i < count; ++i) {
            const unsigned k = classes[i];
            if (k < 8 || k > 16) throw std::invalid_argument("invalid tANS overflow class");
            out[i] = signed_value(uint64_t(1) << k, bits.read(k));
        }
        bits.finish();
    } else {
        throw std::invalid_argument("invalid tANS overflow mode");
    }
    for (int16_t v : out)
        if (v >= -127 && v <= 127) throw std::invalid_argument("invalid tANS overflow value");
    return out;
}

inline void decode_tans(Reader &in, size_t count, int16_t *out) {
    constexpr size_t contexts = 15, symbols = 256, table_log = 13;
    const size_t expected_segments = std::min<size_t>(16, std::max<size_t>(1, count / (size_t(8) << 20)));
    const size_t segments = checked_size(in.read<uint64_t>(), 16, "invalid tANS segment count");
    if (!count || segments != expected_segments) throw std::invalid_argument("invalid tANS segment count");
    const size_t segment_size = count / segments + (count % segments != 0);
    const size_t table_words = FSE_DTABLE_SIZE_U32(table_log);
    std::vector<FSE_DTable> tables(contexts * table_words);
    for (size_t g = 0; g < segments; ++g) {
        const size_t n = std::min(segment_size, count - g * segment_size);
        if (in.read<uint64_t>() != n || in.read<uint64_t>() != 4)
            throw std::invalid_argument("unsupported tANS segment layout");
        std::array<size_t, 4> lengths{}, counts{};
        for (auto &v : lengths) v = checked_size(in.read<uint64_t>(), in.remaining(), "invalid tANS length");
        const size_t chunk = n / 4 + (n % 4 != 0);
        size_t total_overflow = 0;
        for (size_t k = 0; k < 4; ++k) {
            const size_t chain_n = std::min(chunk, n > k * chunk ? n - k * chunk : 0);
            counts[k] = checked_size(in.read<uint64_t>(), chain_n, "invalid tANS overflow count");
            total_overflow += counts[k];
            if ((chain_n != 0) != (lengths[k] != 0)) throw std::invalid_argument("invalid tANS chain length");
        }
        std::array<int16_t, symbols> norm{};
        for (size_t c = 0; c < contexts; ++c) {
            unsigned total = 0;
            for (auto &v : norm) {
                v = in.read<int16_t>();
                if (v < -1) throw std::invalid_argument("invalid tANS model");
                total += static_cast<unsigned>(v < 0 ? -v : v);
            }
            if (total != (1u << table_log) || FSE_isError(FSE_buildDTable(
                    tables.data() + c * table_words, norm.data(), symbols - 1, table_log)))
                throw std::invalid_argument("invalid tANS model");
        }
        std::array<const unsigned char *, 4> chain{};
        for (size_t k = 0; k < 4; ++k) chain[k] = in.take(lengths[k]);
        auto overflow = read_overflow(in, total_overflow);
        std::array<const FSE_decode_t *, symbols - 1> next_table{};
        for (unsigned s = 0; s < symbols - 1; ++s) {
            const int code = static_cast<int>(s >> 1) ^ -static_cast<int>(s & 1);
            next_table[s] = reinterpret_cast<const FSE_decode_t *>(
                tables.data() + context(code) * table_words + 1);
        }
        std::array<std::optional<ReverseBits>, 4> bits;
        std::array<unsigned, 4> state{};
        std::array<size_t, 4> chain_counts{}, oi{}, ov_end{};
        std::array<const FSE_decode_t *, 4> table{};
        size_t ov_base = 0;
        for (size_t k = 0; k < 4; ++k) {
            chain_counts[k] = std::min(chunk, n > k * chunk ? n - k * chunk : 0);
            oi[k] = ov_base;
            ov_base += counts[k];
            ov_end[k] = ov_base;
            if (!chain_counts[k]) continue;
            bits[k].emplace(chain[k], lengths[k]);
            state[k] = bits[k]->read(table_log);
            table[k] = reinterpret_cast<const FSE_decode_t *>(tables.data() + 1);
        }
        // The chains are independent, so interleave them without relaxing the
        // per-chain state, overflow, and bitstream end checks.
        for (size_t i = 0; i < chunk; ++i) {
            for (size_t k = 0; k < 4; ++k) {
                if (i >= chain_counts[k]) continue;
                const FSE_decode_t entry = table[k][state[k]];
                if (entry.nbBits > table_log) throw std::invalid_argument("invalid tANS transition");
                if (i + 1 < chain_counts[k]) {
                    state[k] = entry.newState + bits[k]->read(entry.nbBits);
                    if (state[k] >= (1u << table_log)) throw std::invalid_argument("invalid tANS state");
                }
                if (entry.symbol == 255) {
                    if (oi[k] == ov_end[k]) throw std::invalid_argument("tANS overflow count mismatch");
                    const int16_t code = overflow[oi[k]++];
                    out[g * segment_size + k * chunk + i] = code;
                    table[k] = reinterpret_cast<const FSE_decode_t *>(
                        tables.data() + context(code) * table_words + 1);
                } else {
                    out[g * segment_size + k * chunk + i] = static_cast<int16_t>(
                        static_cast<int>(entry.symbol >> 1) ^ -static_cast<int>(entry.symbol & 1));
                    table[k] = next_table[entry.symbol];
                }
            }
        }
        for (size_t k = 0; k < 4; ++k) {
            if (chain_counts[k] && (oi[k] != ov_end[k] || !bits[k]->finished()))
                throw std::invalid_argument("invalid tANS chain ending");
        }
    }
}

inline std::unique_ptr<int16_t[]> decode_codes(Reader &in, size_t count) {
    const auto *start = in.position();
    const size_t size = in.remaining();
    const uint8_t flags = in.read<uint8_t>();
    const unsigned method = flags & 0x7f;
    if (method > 2 || in.read<uint16_t>() != 0 || in.read<uint16_t>() != 0)
        throw std::invalid_argument("unsupported Falcons entropy header");
    if (method == 2) {
        const auto *p = start;
        size_t remaining = size;
        Falcons::RleFseEncoder<int16_t> encoder;
        encoder.load(p, remaining);
        std::unique_ptr<int16_t[]> result(encoder.decode(p, count));
        encoder.postprocess_decode();
        if (p != start + size) throw std::invalid_argument("extra or missing entropy bytes");
        in.take(in.remaining());
        return result;
    }
    constexpr size_t pad = 64 / sizeof(int16_t);
    if (count > std::numeric_limits<size_t>::max() / sizeof(int16_t) - pad)
        throw std::length_error("entropy decoded size overflow");
    std::unique_ptr<int16_t[]> result(new int16_t[count + pad]);
    std::fill_n(result.get() + count, pad, int16_t(0));
    if (method == 0) decode_rle(in, count, result.get());
    else decode_tans(in, count, result.get());
    if (in.remaining()) throw std::invalid_argument("extra entropy bytes");
    if (flags & 0x80) {
        uint16_t previous = 0;
        for (size_t i = 0; i < count; ++i) {
            uint16_t bits;
            std::memcpy(&bits, result.get() + i, sizeof(bits));
            previous = static_cast<uint16_t>(previous + bits);
            std::memcpy(result.get() + i, &previous, sizeof(previous));
        }
    }
    return result;
}

struct Header {
    std::array<size_t, 4> dims{1, 1, 1, 1};
    uint32_t blocksize = 0;
    int interp = 0, direction = 0;
    size_t anchor = 0;
    double alpha = 0, beta = 0, eb = 0;
    int radius = 0;
};

inline Header read_header(Reader &in, int n, bool omp) {
    Header h;
    for (int i = 0; i < n; ++i) h.dims[i] = in.read<size_t>();
    h.blocksize = in.read<uint32_t>();
    h.interp = in.read<int>();
    h.direction = in.read<int>();
    h.anchor = in.read<size_t>();
    h.alpha = in.read<double>();
    h.beta = in.read<double>();
    if (in.read<uint8_t>() != (omp ? 3 : 2)) throw std::invalid_argument("quantizer/layout mismatch");
    h.eb = in.read<double>();
    h.radius = in.read<int>();
    return h;
}

inline std::vector<unsigned char> decompress_block(const unsigned char *p, size_t n, size_t max_size) {
    Reader r(p, n);
    const size_t expanded = r.read<size_t>();
    if (expanded > max_size) throw std::length_error("Falcons lossless block exceeds data-derived bound");
    if (std::getenv("FALCONS_NOZSTD")) {
        if (r.remaining() != expanded) throw std::invalid_argument("invalid Falcons raw block size");
        const auto *src = r.take(expanded);
        return std::vector<unsigned char>(src, src + expanded);
    }
    std::vector<unsigned char> out(expanded);
    decode_zstd_frame(r.position(), r.remaining(), out.data(), out.size());
    return out;
}

} // namespace current_detail

// cmp starts immediately after Config. 1.1.0 does not record the producer's
// OpenMP scheduling or SIMD width: this plan requires contiguous static
// partitions and 256-bit logical chunks for the 1D OMP path.
template <class T>
HostStreams<T> decode_current(const Falcons::Config &conf, const unsigned char *cmp,
                              size_t cmpSize, bool avx = true) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "GPU interpolation supports float and double only");
    static_assert(sizeof(size_t) == 8 && sizeof(int) == 4,
                  "Falcons 1.1.0 stream bridge requires the 64-bit stream ABI");
    using namespace current_detail;
    if (conf.falconsMagicNumber != Falcons_MAGIC_NUMBER ||
        conf.falconsDataVer != versionInt("1.1.0"))
        throw std::invalid_argument("only current Falcons 1.1.0 streams are supported");
    if (conf.cmprAlgo != Falcons::ALGO_INTERP) throw std::invalid_argument("not an interpolation stream");
    const int n = static_cast<int>(conf.N);
    if (!cmp || !cmpSize || n < 1 || n > 4 || conf.dims.size() != static_cast<size_t>(n))
        throw std::invalid_argument("invalid Falcons input or dimensions");
    size_t num = 1, max_dim = 1;
    for (size_t d : conf.dims) {
        if (!d || d > std::numeric_limits<uint32_t>::max() ||
            d > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(T) / num)
            throw std::invalid_argument("unsupported Falcons dimensions");
        num *= d;
        max_dim = std::max(max_dim, d);
    }
    if (num != conf.num) throw std::invalid_argument("Falcons dimensions do not match element count");
    const bool serial = !conf.openmp;
    Reader container(cmp, cmpSize);
    int streams = 1;
    std::vector<size_t> offsets(1, 0);
    if (!serial) {
        streams = container.read<int>();
        size_t max_threads = n == 1 ? conf.dims[0] : num;
        for (int i = 1; i < n; ++i) max_threads = std::min(max_threads, conf.dims[i]);
        if (streams <= 0 || static_cast<size_t>(streams) > max_threads ||
            static_cast<size_t>(streams) > container.remaining() / sizeof(size_t))
            throw std::invalid_argument("invalid OpenMP stream count");
        offsets.resize(streams);
        for (auto &offset : offsets) offset = container.read<size_t>();
        if (offsets[0] != 0) throw std::invalid_argument("invalid OpenMP block offset");
        for (int t = 0; t < streams; ++t)
            if (offsets[t] >= container.remaining() || (t && offsets[t] <= offsets[t - 1]))
                throw std::invalid_argument("invalid OpenMP block offset");
    }
    const auto *payload = container.position();
    const size_t payload_size = container.remaining();
    const size_t max_block = Falcons::buffer_bounds::scratch_bound<T>(num, num);
    auto last_block = decompress_block(payload + offsets.back(), payload_size - offsets.back(), max_block);
    Reader last_reader(last_block.data(), last_block.size());
    Header h = read_header(last_reader, n, !serial);
    for (int i = 0; i < n; ++i)
        if (h.dims[i] != conf.dims[i]) throw std::invalid_argument("interpolation dimensions mismatch");
    static constexpr int directions[] = {0, 1, 2, 6, 24};
    if (h.blocksize != max_dim || h.interp < 0 || h.interp > 1 || h.direction < 0 ||
        h.direction >= directions[n] || (h.anchor && (h.anchor & (h.anchor - 1))) ||
        (h.anchor && h.anchor >= max_dim) || h.radius != 32768 ||
        !std::isfinite(h.eb) || h.eb <= 0 || !std::isfinite(2 * h.eb) ||
        !std::isfinite(h.alpha) || !std::isfinite(h.beta) || (h.alpha >= 1 && h.beta <= 0))
        throw std::invalid_argument("unsupported interpolation header or quantizer");

    HostStreams<T> result;
    result.numStreams = streams;
    auto &p = result.params;
    p.N = n; p.dims = h.dims; p.sizeT = sizeof(T); p.threads = streams;
    p.interp = h.interp; p.direction = h.direction; p.anchorStride = h.anchor;
    p.eb = h.eb; p.alpha = h.alpha; p.beta = h.beta;
    p.serial = serial; p.fusedAll = true; p.avx = avx;
    result.plan = build_plan(p);
    if (result.plan.num != num || result.plan.streamLen.size() != static_cast<size_t>(streams) ||
        result.plan.anchorLen.size() != static_cast<size_t>(streams))
        throw std::runtime_error("GPU interpolation plan shape mismatch");
    for (const auto &phase : result.plan.phases)
        if (phase.workN > static_cast<uint64_t>(std::numeric_limits<int>::max()))
            throw std::length_error("GPU interpolation phase exceeds launch limit");

    result.codes.reserve(streams);
    result.ownedCodes.reserve(streams);
    struct DecodedStream {
        std::vector<unsigned char> block;
        std::unique_ptr<T[]> savedValues;
        std::unique_ptr<int16_t[]> codes;
        const unsigned char *values = nullptr;
        size_t count = 0, anchors = 0, escapes = 0;
        std::exception_ptr error;
    };

    auto decode_stream = [&](int t, DecodedStream &decoded, bool save_values) {
        const size_t expected = checked_size(result.plan.streamLen[t], num, "invalid plan stream length");
        const size_t expected_anchors = checked_size(result.plan.anchorLen[t], num, "invalid plan anchor length");
        if (expected_anchors > num - expected) throw std::runtime_error("invalid plan stream coverage");
        const size_t block_limit = Falcons::buffer_bounds::scratch_bound<T>(expected + expected_anchors, expected);
        if (t + 1 == streams) {
            if (last_block.size() > block_limit) throw std::length_error("Falcons block exceeds stream bound");
        } else {
            decoded.block = decompress_block(payload + offsets[t], offsets[t + 1] - offsets[t], block_limit);
        }
        Reader current(decoded.block.data(), decoded.block.size());
        Reader &r = t + 1 == streams ? last_reader : current;
        const int un_count = r.read<int>();
        if (un_count < 0 || static_cast<size_t>(un_count) > num ||
            static_cast<size_t>(un_count) > r.remaining() / sizeof(T))
            throw std::invalid_argument("invalid unpredictable count");
        const size_t value_bytes = static_cast<size_t>(un_count) * sizeof(T);
        decoded.values = r.take(value_bytes);
        if (serial || streams == 1) {
            decoded.count = checked_size(r.read<uint64_t>(), num, "invalid quantization count");
        } else {
            const int q = r.read<int>();
            if (q < 0 || static_cast<size_t>(q) > num) throw std::invalid_argument("invalid quantization count");
            decoded.count = static_cast<size_t>(q);
        }
        if (result.plan.streamLen[t] != decoded.count)
            throw std::runtime_error("GPU plan disagrees with stream " + std::to_string(t) +
                                     ": expected " + std::to_string(result.plan.streamLen[t]) +
                                     ", got " + std::to_string(decoded.count));
        decoded.anchors = checked_size(result.plan.anchorLen[t], static_cast<size_t>(un_count),
                                       "stream has fewer unpredictable values than anchors");
        decoded.codes = decode_codes(r, decoded.count);
        decoded.escapes = static_cast<size_t>(std::count(decoded.codes.get(), decoded.codes.get() + decoded.count,
                                                         std::numeric_limits<int16_t>::min()));
        if (decoded.escapes != static_cast<size_t>(un_count) - decoded.anchors)
            throw std::invalid_argument("stream escape count does not match unpredictable values");
        if (save_values && t + 1 != streams) {
            if (value_bytes) {
                decoded.savedValues.reset(new T[static_cast<size_t>(un_count)]);
                std::memcpy(decoded.savedValues.get(), decoded.values, value_bytes);
                decoded.values = reinterpret_cast<const unsigned char *>(decoded.savedValues.get());
            } else {
                decoded.values = nullptr;
            }
            std::vector<unsigned char>().swap(decoded.block);
        }
    };

    size_t total_codes = 0, total_anchors = 0;
    auto merge_stream = [&](DecodedStream &decoded) {
        const size_t a0 = result.anchors.size(), u0 = result.unpred.size();
        result.anchors.resize(a0 + decoded.anchors);
        result.unpred.resize(u0 + decoded.escapes);
        if (decoded.anchors)
            std::memcpy(result.anchors.data() + a0, decoded.values, decoded.anchors * sizeof(T));
        if (decoded.escapes)
            std::memcpy(result.unpred.data() + u0, decoded.values + decoded.anchors * sizeof(T),
                        decoded.escapes * sizeof(T));
        result.codes.push_back(decoded.codes.get());
        result.ownedCodes.push_back(std::move(decoded.codes));
        if (decoded.count > num - total_codes || decoded.anchors > num - total_anchors)
            throw std::invalid_argument("stream totals exceed element count");
        total_codes += decoded.count;
        total_anchors += decoded.anchors;
    };

#ifdef _OPENMP
    constexpr size_t min_work_per_worker = size_t(1) << 16;
    int workers = 1;
    if (streams > 1 && num / min_work_per_worker >= 2 && !omp_in_parallel()) {
        const int max_workers = omp_get_max_threads();
        if (max_workers > 1)
            workers = static_cast<int>(std::min({static_cast<size_t>(streams),
                                                 static_cast<size_t>(max_workers),
                                                 num / min_work_per_worker}));
    }
    if (workers > 1) {
        std::vector<DecodedStream> decoded(static_cast<size_t>(streams));
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers)
        for (int t = 0; t < streams; ++t) {
            try {
                decode_stream(t, decoded[static_cast<size_t>(t)], true);
            } catch (...) {
                decoded[static_cast<size_t>(t)].error = std::current_exception();
            }
        }
        for (auto &stream : decoded) {
            if (stream.error) std::rethrow_exception(stream.error);
            merge_stream(stream);
            stream.savedValues.reset();
        }
    } else
#endif
    {
        for (int t = 0; t < streams; ++t) {
            DecodedStream decoded;
            decode_stream(t, decoded, false);
            merge_stream(decoded);
        }
    }
    if (total_anchors > num || total_codes != num - total_anchors)
        throw std::runtime_error("GPU interpolation plan does not cover the data");
    return result;
}

} // namespace fgpu
