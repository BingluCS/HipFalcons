#pragma once

#include <cstddef>
#include <stdexcept>
#include <zstd.h>

namespace fgpu {
inline void decode_zstd_frame(const void* src, std::size_t bytes, void* dst, std::size_t expected) {
    const auto frame = ZSTD_findFrameCompressedSize(src, bytes);
    if (ZSTD_isError(frame) || frame != bytes) throw std::invalid_argument("invalid Falcons Zstd frame");
    const auto size = ZSTD_getFrameContentSize(src, bytes);
    if (size == ZSTD_CONTENTSIZE_ERROR || (size != ZSTD_CONTENTSIZE_UNKNOWN && size != expected))
        throw std::invalid_argument("invalid Falcons Zstd content size");
    const auto got = ZSTD_decompress(dst, expected, src, bytes);
    if (ZSTD_isError(got) || got != expected) throw std::invalid_argument("invalid Falcons Zstd block");
}
} // namespace fgpu
