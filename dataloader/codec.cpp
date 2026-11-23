#include "stream.hpp"
#ifdef HF_CPU_ONLY
#include "packed.hpp"
#include "h1fast.hpp"
#ifdef __AVX2__
#include "rlefast.hpp"
#endif
#else
#include "nd_reconstruct.hpp"
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {
thread_local std::string last_error;

template <class Function>
int checked(Function run) noexcept {
    try {
        run();
        return 0;
    } catch (const std::exception& e) {
        last_error = e.what();
    } catch (...) {
        last_error = "unknown native codec error";
    }
    return -1;
}
}

extern "C" const char* hf_last_error() { return last_error.c_str(); }

#ifdef HF_CPU_ONLY
namespace {

std::size_t checked_add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::length_error("packed plane size overflow");
    return a + b;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    const std::size_t remainder = value % alignment;
    return remainder == 0 ? value : checked_add(value, alignment - remainder);
}

std::size_t reserve_array(std::size_t& total, std::size_t count,
                          std::size_t width, std::size_t alignment) {
    const std::size_t offset = align_up(total, alignment);
    if (count > std::numeric_limits<std::size_t>::max() / width)
        throw std::length_error("packed plane size overflow");
    total = checked_add(offset, count * width);
    return offset;
}

struct PackedPlane {
    hip_falcons::Inputs input;
    hip_falcons::CompactPlan plan;
    hip_falcons::PackedHeader header{};
    std::size_t bytes = 0;
};

bool decode_worker_codes(const Falcons::uchar* header, std::size_t bytes, std::size_t n,
                         std::unique_ptr<std::int16_t[]>& out) {
    if ((*header & 0x7f) == 1) return hip_falcons::try_decode_h1(header, bytes, n, out);
#ifdef __AVX2__
    if ((*header & 0x7f) == 0) return hip_falcons::try_decode_rle(header, bytes, n, out);
#endif
    return false;
}

PackedPlane prepare_plane(const void* src, std::size_t bytes,
                          std::size_t h, std::size_t w, hip_falcons::FastEntropy fast = nullptr,
                          hip_falcons::BatchUnpackWorkspace* workspace = nullptr) {
    using namespace hip_falcons;
    if (!src || bytes == 0)
        throw std::invalid_argument("empty packed plane input");
    PackedPlane plane;
    UnpackTiming timing;
    if (workspace)
        plane.input = unpack_batch(static_cast<const char*>(src), bytes, timing, fast, *workspace);
    else
        plane.input = unpack(static_cast<const char*>(src), bytes, timing, fast);
    const auto& in = plane.input;
    if (in.dims.size() != 2 || in.dims[0] != h || in.dims[1] != w)
        throw std::invalid_argument("Falcons stream dimensions differ from HDF5 plane");
    const auto& p = in.params;
    plane.plan = make_compact_plan({in.dims, p.block_size, p.anchor_stride, in.direction,
                                    p.error_bound, p.eb_alpha, p.eb_beta});
    const auto& plan = plane.plan;
    if (in.code_count != plan.code_count || in.unpredictable.size() < plan.anchors.count ||
        (in.code_count != 0 && !in.codes))
        throw std::invalid_argument("inconsistent plane code or anchor count");
    const std::size_t escapes = in.unpredictable.size() - static_cast<std::size_t>(plan.anchors.count);
    if (escapes > in.code_count)
        throw std::invalid_argument("inconsistent plane escape count");

    auto& header = plane.header;
    header.anchors = plan.anchors;
    header.group_count = plan.groups.size();
    header.escape_count = escapes;
    plane.bytes = sizeof(header);
    header.group_off = reserve_array(plane.bytes, plan.groups.size(), sizeof(CompactGroup), alignof(CompactGroup));
    header.unpred_off = reserve_array(plane.bytes, in.unpredictable.size(), sizeof(float), alignof(float));
    header.escape_off = reserve_array(plane.bytes, escapes, sizeof(std::uint32_t), alignof(std::uint32_t));
    header.code_off = reserve_array(plane.bytes, in.code_count, sizeof(std::int16_t), alignof(std::int16_t));
    return plane;
}

// Scan even when the metadata claims no escapes; the map is optional when
// sizing an undersized batch, but the code stream must still be validated.
void copy_codes_and_escapes(const PackedPlane& plane, unsigned char* code_out,
                            unsigned char* escape_out) {
    const auto* codes = plane.input.codes.get();
    const std::size_t count = plane.input.code_count;
    const std::size_t expected = static_cast<std::size_t>(plane.header.escape_count);
    constexpr auto escape = std::numeric_limits<std::int16_t>::min();
    std::size_t found = 0;
    const auto record = [&](std::size_t index) {
        if (found == expected) throw std::invalid_argument("unexpected plane escape");
        const auto position = static_cast<std::uint32_t>(index);
        if (escape_out != nullptr)
            std::memcpy(escape_out + found * sizeof(position), &position, sizeof(position));
        ++found;
    };

    std::size_t i = 0;
#if defined(__AVX2__)
    const __m256i needle = _mm256_set1_epi16(escape);
    for (; count - i >= 16; i += 16) {
        const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + i));
        if (code_out != nullptr)
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(code_out + i * sizeof(*codes)), values);
        unsigned mask = static_cast<unsigned>(_mm256_movemask_epi8(_mm256_cmpeq_epi16(values, needle))) & 0x55555555u;
        if (mask != 0 && expected == 0) throw std::invalid_argument("unexpected plane escape");
        while (mask != 0) {
            record(i + static_cast<unsigned>(__builtin_ctz(mask)) / 2);
            mask &= mask - 1;
        }
    }
    if (code_out != nullptr && i < count)
        std::memcpy(code_out + i * sizeof(*codes), codes + i, (count - i) * sizeof(*codes));
#else
    if (code_out != nullptr && count != 0)
        std::memcpy(code_out, codes, count * sizeof(*codes));
#endif
    for (; i < count; ++i) {
        if (codes[i] == escape) record(i);
    }
    if (found != expected) throw std::invalid_argument("missing plane escape");
}

void emit_plane(const PackedPlane& plane, unsigned char* output) {
    const auto& in = plane.input;
    const auto& plan = plane.plan;
    const auto& header = plane.header;
    if (output != nullptr) {
        std::memcpy(output, &header, sizeof(header));
        if (!plan.groups.empty())
            std::memcpy(output + header.group_off, plan.groups.data(), plan.groups.size() * sizeof(hip_falcons::CompactGroup));
        if (!in.unpredictable.empty())
            std::memcpy(output + header.unpred_off, in.unpredictable.data(), in.unpredictable.size() * sizeof(float));
    }
    copy_codes_and_escapes(plane, output ? output + header.code_off : nullptr,
                           output ? output + header.escape_off : nullptr);
}

}  // namespace

// The returned buffer remains compatible with the legacy per-plane caller.
extern "C" int hf_pack_plane(const void* src, std::size_t bytes, std::size_t h, std::size_t w,
                             void** data, std::size_t* packed_bytes, std::size_t* groups) {
    return checked([&] {
        if (!data || !packed_bytes || !groups)
            throw std::invalid_argument("empty packed plane argument");
        *data = nullptr;
        *packed_bytes = 0;
        *groups = 0;
        const auto plane = prepare_plane(src, bytes, h, w);
        std::unique_ptr<unsigned char[]> buffer(new unsigned char[plane.bytes]);
        emit_plane(plane, buffer.get());
        *packed_bytes = plane.bytes;
        *groups = plane.plan.groups.size();
        *data = buffer.release();
    });
}

extern "C" int hf_pack_batch(const void* const* inputs, const std::size_t* lengths,
                             std::size_t planes, std::size_t h, std::size_t w,
                             void* slab, std::size_t capacity, std::int64_t* offsets,
                             std::size_t* used, std::size_t* max_groups) {
    int result = 0;
    const int status = checked([&] {
        if (!used || !max_groups) throw std::invalid_argument("missing packed batch result");
        *used = 0;
        *max_groups = 0;
        if (planes == 0) return;
        if (!inputs || !lengths || !offsets || (!slab && capacity != 0) ||
            h == 0 || w == 0 || h > std::numeric_limits<std::uint32_t>::max() / w ||
            (slab && reinterpret_cast<std::uintptr_t>(slab) % alignof(hip_falcons::PackedHeader) != 0))
            throw std::invalid_argument("invalid packed batch argument");

        std::size_t total = 0, largest_group_count = 0;
        auto* output = static_cast<unsigned char*>(slab);
        thread_local hip_falcons::BatchUnpackWorkspace workspace;
        for (std::size_t i = 0; i < planes; ++i) {
            const auto plane = prepare_plane(inputs[i], lengths[i], h, w, decode_worker_codes, &workspace);
            const std::size_t start = align_up(total, alignof(hip_falcons::PackedHeader));
            const std::size_t end = checked_add(start, plane.bytes);
            if (end > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
                throw std::length_error("packed batch exceeds int64 indexing");
            emit_plane(plane, output && end <= capacity ? output + start : nullptr);
            offsets[i] = static_cast<std::int64_t>(start);
            largest_group_count = std::max(largest_group_count, plane.plan.groups.size());
            total = end;
        }
        *used = total;
        *max_groups = largest_group_count;
        if (total > capacity) result = 1;
    });
    return status == 0 ? result : status;
}

extern "C" void hf_free_plane(void* data) { delete[] static_cast<unsigned char*>(data); }
#endif

extern "C" int hf_cpu_f32(const void* src, size_t bytes, float* output, size_t n) {
    return checked([&] {
        if (!src || !bytes || !output || !n) throw std::invalid_argument("empty chunk or output");
        Falcons::Config conf;
        const auto* stream = static_cast<const char*>(src);
        Falcons_read_config(conf, stream, bytes);
        if (conf.num != n || conf.dataType != Falcons_FLOAT) throw std::invalid_argument("unexpected float32 chunk size or dtype");
        Falcons_decompress<float>(conf, stream, bytes, output, n);
    });
}

#ifndef HF_CPU_ONLY
extern "C" int hf_gpu_f32(const void* src, size_t bytes, float* output, size_t n, void* torch_stream) {
    return checked([&] {
        if (!src || !bytes || !output || !n) throw std::invalid_argument("empty chunk or output");
        hip_falcons::UnpackTiming timing;
        auto in = hip_falcons::unpack(static_cast<const char*>(src), bytes, timing);
        const auto& m = in.params;
        if (m.num_elements != n || in.dims.size() != 2)
            throw std::invalid_argument("device loader expects a 2D float32 plane of the requested size");
        const hip_falcons::NDReconstructParams params{
            in.dims, m.block_size, m.anchor_stride, in.direction, m.error_bound, m.eb_alpha, m.eb_beta
        };
        const auto err = hip_falcons::reconstruct_nd_linear_f32_device_output(
            params, in.codes.get(), in.code_count, in.unpredictable.data(), in.unpredictable.size(),
            output, reinterpret_cast<hipStream_t>(torch_stream));
        if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
    });
}
#endif

// Fixture/packing helper. A chunk is an ordinary serial CPU Falcons stream.
extern "C" int hf_compress_2d_f32(const float* input, size_t h, size_t w,
                                  double abs_error, void** out, size_t* bytes) {
    return checked([&] {
        if (!input || !out || !bytes || h < 2 || w < 2 ||
            w > std::numeric_limits<uint32_t>::max() / h || !std::isfinite(abs_error) || abs_error <= 0)
            throw std::invalid_argument("invalid plane or error bound");
        *out = nullptr;
        *bytes = 0;
        Falcons::Config conf(h, w);
        conf.cmprAlgo = Falcons::ALGO_INTERP;
        conf.interpAlgo = Falcons::INTERP_ALGO_LINEAR;
        conf.errorBoundMode = Falcons::EB_ABS;
        conf.absErrorBound = abs_error;
        conf.dataType = Falcons_FLOAT;
        conf.openmp = false;
        std::vector<float> work(input, input + h*w);
        std::size_t length = 0;
        std::unique_ptr<char[]> stream(Falcons_compress<float>(conf, work.data(), length));
        Falcons::Config stored;
        Falcons_read_config(stored, stream.get(), length);
        if (stored.cmprAlgo != Falcons::ALGO_INTERP)
            throw std::runtime_error("Falcons selected lossless fallback for this plane; DCU mode requires interpolation");
        *out = stream.release();
        *bytes = length;
    });
}

extern "C" void hf_release(void* data) { delete[] static_cast<char*>(data); }
