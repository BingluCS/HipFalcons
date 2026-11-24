#pragma once

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
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(__ARM_FEATURE_SVE2)
#include <arm_sve.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "Falcons/encoder/RleFseEncoder.hpp"
#include "Falcons/lossless/Lossless_zstd.hpp"
#include "Falcons/utils/BufferBounds.hpp"
#include "Falcons/utils/Config.hpp"
#include "Falcons/utils/Statistic.hpp"
#include "fgpu_recon.h"

namespace fgpu {

// Resolve a current-format interpolation configuration and its device plan parameters.
// A device-only caller can provide deviceData instead of hostData.
template <class T>
Falcons::Config prepare_current_compress(const Falcons::Config &config, const T *hostData,
                                         int streams, PlanParams &pp,
                                         const void *deviceData = nullptr) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "current GPU interpolation supports float and double only");
    using namespace Falcons;
    Config conf(config);
    const int n = static_cast<int>(conf.N);
    if (streams <= 0 || n < 1 || n > 4 || conf.dims.size() != static_cast<size_t>(n) ||
        (!hostData && !deviceData))
        throw std::invalid_argument("invalid GPU compression input");
    if (conf.cmprAlgo != ALGO_INTERP)
        throw std::invalid_argument("GPU compression requires ALGO_INTERP");
    if (conf.quantbinCnt != 65536)
        throw std::invalid_argument("GPU compression requires 65536 quantization bins");
    if (conf.interpAlgo > INTERP_ALGO_CUBIC || conf.interpDirection >= factorial(n))
        throw std::invalid_argument("invalid interpolation algorithm or direction");
    if (!std::isfinite(conf.interpAlpha) || !std::isfinite(conf.interpBeta) ||
        (conf.interpAlpha >= 1 && conf.interpBeta < 1))
        throw std::invalid_argument("invalid interpolation error-bound parameters");

    size_t count = 1;
    for (size_t dim : conf.dims) {
        if (dim == 0 || dim > static_cast<size_t>((std::numeric_limits<int>::max)()))
            throw std::invalid_argument("invalid GPU interpolation dimension");
        count = buffer_bounds::mul(count, dim);
    }
    if (count != conf.num) throw std::invalid_argument("dimensions do not match element count");
    buffer_bounds::mul(count, sizeof(T));

    conf.openmp = streams > 1;
    if (conf.errorBoundMode > EB_ABS_OR_REL)
        throw std::invalid_argument("invalid error-bound mode");
    if (!deviceData || conf.errorBoundMode == EB_ABS || conf.errorBoundMode == EB_L2NORM) {
        calAbsErrorBound(conf, hostData);
    } else {
        // calAbsErrorBound treats a zero supplied range as a request to scan its pointer.
        const T range = static_cast<T>(device_range(deviceData, count, sizeof(T)));
        const T zero = 0;
        const size_t savedCount = conf.num;
        conf.num = 1;
        calAbsErrorBound(conf, &zero, range);
        conf.num = savedCount;
    }
    if (!(conf.absErrorBound > 0) || !std::isfinite(conf.absErrorBound) ||
        !std::isfinite(2 * conf.absErrorBound) ||
        !std::isfinite(0.5 / conf.absErrorBound))
        throw std::invalid_argument("GPU interpolation requires a finite positive error bound");

    if (conf.interpAnchorStride < 0) {
        constexpr std::array<int, 4> defaults{4096, 128, 32, 16};
        conf.interpAnchorStride = defaults[n - 1];
    }
    const auto anchor = static_cast<size_t>(conf.interpAnchorStride);
    if (anchor && (anchor & (anchor - 1)))
        throw std::invalid_argument("interpolation anchor stride must be zero or a power of two");

    size_t threads = static_cast<size_t>(streams);
    if (n == 1) threads = (std::min)(threads, conf.dims[0]);
    for (int i = 1; i < n; ++i) threads = (std::min)(threads, conf.dims[i]);
#if defined(__ARM_FEATURE_SVE2)
    // The upstream plan partitions 1D lanes in 32-byte groups; the current CPU
    // OMP decoder partitions SVE lanes using the runtime vector length.
    if (conf.openmp && n == 1 && threads > 1 && svcntb() != 32)
        throw std::runtime_error("1D OpenMP streams require a 256-bit SVE vector length");
#endif
    conf.falconsMagicNumber = Falcons_MAGIC_NUMBER;
    conf.falconsDataVer = versionInt(Falcons_DATA_VER);
    conf.dataType = std::is_same_v<T, float> ? Falcons_FLOAT : Falcons_DOUBLE;

    pp = PlanParams{};
    pp.N = n;
    std::copy(conf.dims.begin(), conf.dims.end(), pp.dims.begin());
    pp.sizeT = sizeof(T);
    pp.threads = static_cast<int>(threads);
    pp.interp = conf.interpAlgo;
    pp.direction = conf.interpDirection;
    pp.anchorStride = anchor;
    pp.eb = conf.absErrorBound;
    pp.alpha = conf.interpAlpha;
    pp.beta = conf.interpBeta;
    pp.serial = !conf.openmp;
    pp.fusedAll = true;
    pp.avx = true;
    return conf;
}

// Decomposition and quantizer metadata, stored at the start of the last block.
inline std::vector<uint8_t> current_interp_header(const Falcons::Config &conf, int streams) {
    using namespace Falcons;
    if (streams <= 0 || conf.N < 1 || conf.N > 4 ||
        conf.dims.size() != static_cast<size_t>(conf.N) || conf.interpAnchorStride < 0)
        throw std::invalid_argument("invalid interpolation header configuration");
    size_t maxDim = 1;
    bool useAnchor = false;
    for (size_t dim : conf.dims) {
        maxDim = (std::max)(maxDim, dim);
        useAnchor = useAnchor || dim > static_cast<size_t>(conf.interpAnchorStride);
    }
    if (maxDim > (std::numeric_limits<uint32_t>::max)())
        throw std::length_error("interpolation block size exceeds stream format");

    std::vector<uint8_t> header(128);
    uchar *p = header.data();
    for (size_t dim : conf.dims) write(dim, p);
    write(static_cast<uint32_t>(maxDim), p);
    write(static_cast<int>(conf.interpAlgo), p);
    write(static_cast<int>(conf.interpDirection), p);
    write(useAnchor ? static_cast<size_t>(conf.interpAnchorStride) : size_t(0), p);
    write(conf.interpAlpha, p);
    write(conf.interpBeta, p);
    write(static_cast<uchar>(streams == 1 ? 0b10 : 0b11), p);
    write(conf.absErrorBound, p);
    write(conf.quantbinCnt / 2, p);
    header.resize(static_cast<size_t>(p - header.data()));
    return header;
}

// Blocks already include the Falcons lossless wrapper. The requested stream count
// selects serial vs OpenMP even when the effective OpenMP team is one thread.
inline std::vector<char> join_current_blocks(const Falcons::Config &conf, const PlanParams &pp,
                                             const std::vector<std::vector<uint8_t>> &blocks,
                                             int streams) {
    using namespace Falcons;
    if (streams <= 0 || pp.threads <= 0 || blocks.size() != static_cast<size_t>(pp.threads) ||
        (streams == 1 && pp.threads != 1) || conf.openmp != (streams > 1))
        throw std::invalid_argument("invalid GPU compression blocks");
    const size_t confSize = conf.size_est();
    size_t total = confSize;
    if (conf.openmp)
        total = buffer_bounds::add(total, buffer_bounds::add(sizeof(int),
            buffer_bounds::mul(blocks.size(), sizeof(size_t))));
    for (const auto &block : blocks) {
        if (block.empty()) throw std::invalid_argument("empty GPU compression block");
        total = buffer_bounds::add(total, block.size());
    }

    std::vector<char> out(total);
    uchar *p = reinterpret_cast<uchar *>(out.data());
    const size_t saved = conf.save(p);
    if (saved != confSize || static_cast<size_t>(p - reinterpret_cast<uchar *>(out.data())) != confSize)
        throw std::runtime_error("unexpected Falcons configuration size");
    if (conf.openmp) {
        write(pp.threads, p);
        size_t offset = 0;
        for (const auto &block : blocks) {
            write(offset, p);
            offset = buffer_bounds::add(offset, block.size());
        }
    }
    for (const auto &block : blocks) {
        std::memcpy(p, block.data(), block.size());
        p += block.size();
    }
    return out;
}

template <class T>
std::vector<char> compress_current_hybrid(const Falcons::Config &config, const T *hostData, int streams) {
    using namespace Falcons;
    PlanParams pp;
    Config conf = prepare_current_compress<T>(config, hostData, streams, pp);
    if (std::getenv("FALCONS_NOZSTD"))
        throw std::invalid_argument("FALCONS_NOZSTD cannot produce a conventional Falcons stream");
    Plan plan = build_plan(pp);
    const size_t k = static_cast<size_t>(pp.threads);
    if (plan.num != conf.num || plan.streamLen.size() != k || plan.anchorLen.size() != k)
        throw std::runtime_error("invalid GPU interpolation plan");
    for (const auto &phase : plan.phases) {
        if (!(phase.ebx2 > 0) || !std::isfinite(phase.ebx2) ||
            !(phase.ebx2r > 0) || !std::isfinite(phase.ebx2r) ||
            !std::isfinite(phase.ebT))
            throw std::invalid_argument("interpolation level has an invalid error bound");
    }

    std::vector<size_t> base(k + 1, 0), anchors(k + 1, 0);
    for (size_t t = 0; t < k; ++t) {
        if (plan.streamLen[t] > static_cast<uint64_t>((std::numeric_limits<int>::max)()) ||
            plan.anchorLen[t] > static_cast<uint64_t>((std::numeric_limits<int>::max)()))
            throw std::length_error("GPU stream exceeds the CPU cursor range");
        base[t + 1] = buffer_bounds::add(base[t], static_cast<size_t>(plan.streamLen[t]));
        anchors[t + 1] = buffer_bounds::add(anchors[t], static_cast<size_t>(plan.anchorLen[t]));
    }
    if (buffer_bounds::add(base[k], anchors[k]) != conf.num)
        throw std::runtime_error("GPU interpolation plan does not cover the input");

    CompressInput input;
    input.plan = &plan;
    input.sizeT = sizeof(T);
    CompressOutput output;
    compress_gpu(input, hostData, output);
    if (!output.ok)
        throw std::runtime_error("GPU compression failed, possibly because its escape capacity was exceeded");
    if (output.codes.size() != base[k] || output.anchors.size() != buffer_bounds::mul(anchors[k], sizeof(T)) ||
        output.escVal.size() != buffer_bounds::mul(output.escPos.size(), sizeof(T)))
        throw std::runtime_error("inconsistent GPU compression output sizes");

    if (static_cast<size_t>(std::count(output.codes.begin(), output.codes.end(), FG_ESCAPE)) != output.escPos.size())
        throw std::runtime_error("invalid GPU escape count");
    for (size_t e = 0; e < output.escPos.size(); ++e) {
        const auto i = output.escPos[e];
        if (i >= output.codes.size() || output.codes[i] != FG_ESCAPE || (e && i <= output.escPos[e - 1]))
            throw std::runtime_error("GPU escape positions do not match quantization codes");
    }

    std::vector<size_t> escapes(k + 1, 0);
    for (size_t t = 0; t < k; ++t) {
        escapes[t + 1] = static_cast<size_t>(std::lower_bound(output.escPos.begin(), output.escPos.end(),
                                                              base[t + 1]) - output.escPos.begin());
        const size_t unpred = buffer_bounds::add(anchors[t + 1] - anchors[t], escapes[t + 1] - escapes[t]);
        if (unpred > static_cast<size_t>((std::numeric_limits<int>::max)()))
            throw std::length_error("GPU outlier stream exceeds the CPU cursor range");
    }
    const auto header = current_interp_header(conf, streams);
    std::vector<std::vector<uint8_t>> blocks(k);
    std::vector<std::exception_ptr> errors(k);
#ifdef _OPENMP
    const int workers = (std::min)(pp.threads, omp_get_max_threads());
#pragma omp parallel for schedule(dynamic) num_threads(workers) if(workers > 1)
#endif
    for (int tid = 0; tid < pp.threads; ++tid) {
        try {
            const size_t t = static_cast<size_t>(tid);
            const size_t n = base[t + 1] - base[t];
            const size_t na = anchors[t + 1] - anchors[t];
            const size_t ne = escapes[t + 1] - escapes[t];
            const size_t unpred = na + ne;
            const size_t capacity_scratch = buffer_bounds::scratch_bound<T>(unpred, n);
            // The encoder initializes the prefix passed to Zstd; the unused capacity need not be touched.
            std::unique_ptr<uchar[]> scratch(new uchar[capacity_scratch]);
            uchar *p = scratch.get();
            if (t + 1 == k) {
                std::memcpy(p, header.data(), header.size());
                p += header.size();
            }
            write(static_cast<int>(unpred), p);
            if (na) {
                const size_t bytes = buffer_bounds::mul(na, sizeof(T));
                std::memcpy(p, output.anchors.data() + anchors[t] * sizeof(T), bytes);
                p += bytes;
            }
            if (ne) {
                const size_t bytes = buffer_bounds::mul(ne, sizeof(T));
                std::memcpy(p, output.escVal.data() + escapes[t] * sizeof(T), bytes);
                p += bytes;
            }
            if (!conf.openmp) write(static_cast<uint64_t>(n), p);
            else if (k == 1) write(n, p);
            else write(static_cast<int>(n), p);

            int16_t empty = 0;
            int16_t *codes = n ? output.codes.data() + base[t] : &empty;
            RleFseEncoder<int16_t> encoder;
            encoder.preprocess_encode(codes, n, conf.quantbinCnt);
            encoder.save(p);
            encoder.encode(codes, n, p);
            encoder.postprocess_encode();
            const size_t sourceSize = static_cast<size_t>(p - scratch.get());
            if (sourceSize > capacity_scratch) throw std::length_error("entropy output exceeds scratch capacity");
            const size_t bound = ZSTD_compressBound(sourceSize);
            if (ZSTD_isError(bound)) throw std::length_error("Zstd capacity overflow");
            const size_t capacity = buffer_bounds::add(sizeof(size_t), (std::max)(bound, sourceSize));
            blocks[t].resize(capacity);
            Lossless_zstd zstd;
            const size_t written = zstd.compress(scratch.get(), sourceSize, blocks[t].data(), capacity);
            if (written > capacity) throw std::length_error("invalid Zstd output size");
            blocks[t].resize(written);
        } catch (...) {
            errors[static_cast<size_t>(tid)] = std::current_exception();
        }
    }
    for (const auto &error : errors) if (error) std::rethrow_exception(error);
    return join_current_blocks(conf, pp, blocks, streams);
}

}  // namespace fgpu
