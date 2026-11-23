#pragma once

#include "Falcons/api/falcons.hpp"
#include "stream_params.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#include <zstd.h>

namespace hip_falcons {
using Falcons::uchar;
struct UnpackTiming {
    double zstd = 0, entropy = 0;
    unsigned method = 0;
};
inline double unpack_elapsed(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

struct Inputs {
    Reconstruct1DParams params{};
    std::vector<size_t> dims;
    int direction = 0;
    std::unique_ptr<int16_t[]> codes;
    size_t code_count = 0;
    std::vector<float> unpredictable;
};

using FastEntropy = bool (*)(const uchar*, size_t, size_t, std::unique_ptr<int16_t[]>&);

// Batch workers retain their decompression workspace, but do not retain an
// unusually large plane indefinitely. Legacy unpack uses Lossless_zstd itself.
class BatchUnpackWorkspace {
public:
    uchar* decompress(const uchar* src, size_t src_length, size_t& dst_length,
                      std::unique_ptr<uchar[]>& temporary) {
        if (src == nullptr || src_length < sizeof(size_t))
            throw std::invalid_argument("Truncated Falcons lossless payload");
        Falcons::read(dst_length, src);
        const size_t payload_size = src_length - sizeof(dst_length);
        const bool raw = std::getenv("FALCONS_NOZSTD") != nullptr;
        if (raw) {
            if (dst_length > payload_size)
                throw std::invalid_argument("Truncated Falcons raw payload");
        } else {
            const auto frame_size = ZSTD_getFrameContentSize(src, payload_size);
            if (frame_size == ZSTD_CONTENTSIZE_ERROR ||
                (frame_size != ZSTD_CONTENTSIZE_UNKNOWN && frame_size != dst_length))
                throw std::invalid_argument("Invalid Falcons Zstd payload size");
        }

        const size_t required = dst_length == 0 ? 1 : dst_length;
        uchar* dst;
        if (required > kMaxRetainedBytes) {
            temporary.reset(new uchar[required]);
            dst = temporary.get();
        } else {
            if (required > capacity_) {
                const size_t grown = std::max(required, capacity_ + capacity_ / 2);
                const size_t next = std::min(kMaxRetainedBytes, grown);
                buffer_.reset(new uchar[next]);
                capacity_ = next;
            }
            dst = buffer_.get();
        }
        if (raw) {
            if (dst_length != 0) std::memcpy(dst, src, dst_length);
            return dst;
        }
        if (!context_) {
            context_.reset(ZSTD_createDCtx());
            if (!context_) throw std::bad_alloc();
        }
        const size_t result = ZSTD_decompressDCtx(context_.get(), dst, dst_length,
                                                   src, payload_size);
        if (ZSTD_isError(result) || result != dst_length)
            throw std::invalid_argument("Invalid or truncated Falcons Zstd payload");
        return dst;
    }

private:
    static constexpr size_t kMaxRetainedBytes = size_t{64} * 1024 * 1024;
    std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> context_{nullptr, &ZSTD_freeDCtx};
    std::unique_ptr<uchar[]> buffer_;
    size_t capacity_ = 0;
};

inline Inputs unpack_impl(const char* stream, size_t length, UnpackTiming& t,
                          FastEntropy fast, BatchUnpackWorkspace* workspace) {
    Falcons::Config conf;
    Falcons_read_config(conf, stream, length);
    if (conf.dataType != Falcons_FLOAT || conf.N < 1 || conf.N > 3 || conf.cmprAlgo != Falcons::ALGO_INTERP || conf.openmp)
        throw std::invalid_argument("requires a serial 1D/2D/3D float32 Falcons interpolation stream");
    const size_t hdr = static_cast<unsigned char>(stream[0]);
    uchar* buffer = nullptr;
    size_t bytes = 0;
    std::unique_ptr<uchar, decltype(&std::free)> owner(nullptr, &std::free);
    std::unique_ptr<uchar[]> temporary;
    const auto zstart = std::chrono::steady_clock::now();
    const auto* payload = reinterpret_cast<const uchar*>(stream) + hdr;
    if (workspace) {
        buffer = workspace->decompress(payload, length - hdr, bytes, temporary);
    } else {
        Falcons::Lossless_zstd().decompress(payload, length - hdr, buffer, bytes);
        owner.reset(buffer);
    }
    t.zstd = unpack_elapsed(zstart);
    const uchar* p = buffer;
    auto read = [&](auto& x) {
        if (bytes < sizeof(x)) throw std::runtime_error("truncated interpolation metadata");
        Falcons::read(x, p, bytes);
    };

    // Serial InterpolationDecomposition::save followed by LinearQuantizer::save.
    Inputs in;
    auto& m = in.params;
    int algo;
    in.dims.resize(conf.N);
    for (auto& dim : in.dims) read(dim);
    m.num_elements = conf.num;
    read(m.block_size); read(algo); read(in.direction);
    read(m.anchor_stride); read(m.eb_alpha); read(m.eb_beta);
    if (in.dims != conf.dims || !conf.num || conf.num > UINT32_MAX ||
        m.block_size != *std::max_element(in.dims.begin(), in.dims.end()) || algo != Falcons::INTERP_ALGO_LINEAR)
        throw std::invalid_argument("requires LINEAR interpolation with matching serialized dimensions and block size");
    uchar uid;
    int unpred;
    read(uid); read(m.error_bound); read(m.radius); read(unpred);
    if (uid != 2 || m.radius != 32768 || unpred < 0 || size_t(unpred) > conf.num ||
        size_t(unpred) > bytes / sizeof(float))
        throw std::invalid_argument("unsupported or invalid float32 quantizer metadata");
    in.unpredictable.resize(unpred);
    Falcons::read(in.unpredictable.data(), in.unpredictable.size(), p, bytes);
    uint64_t count;
    read(count);
    if (count > conf.num || bytes == 0) throw std::runtime_error("invalid entropy payload");
    t.method = *p & 0x7f;
    const auto estart = std::chrono::steady_clock::now();
    in.code_count = count;
    if (!(fast && fast(p, bytes, in.code_count, in.codes))) {
        Falcons::RleFseEncoder<int16_t> entropy;
        entropy.load(p, bytes);
        in.codes.reset(entropy.decode(p, in.code_count));
    }
    t.entropy = unpack_elapsed(estart);
    return in;
}

inline Inputs unpack(const char* stream, size_t length, UnpackTiming& t, FastEntropy fast = nullptr) {
    return unpack_impl(stream, length, t, fast, nullptr);
}

inline Inputs unpack_batch(const char* stream, size_t length, UnpackTiming& t,
                           FastEntropy fast, BatchUnpackWorkspace& workspace) {
    return unpack_impl(stream, length, t, fast, &workspace);
}

} // namespace hip_falcons
