// Standalone 1.1.0 lossless-only adapter for the existing device Zstd decoder.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <zstd.h>

namespace fgpu {

struct LosslessInput {
    // Payload immediately after the Falcons Config, not an OMP stream table.
    const unsigned char *cmp = nullptr;
    size_t cmpSize = 0;
    size_t expectedBytes = 0;
    // FALCONS_NOZSTD is an out-of-band writer setting; it is not stored in Config.
    bool rawNoZstd = false;
};

struct LosslessTiming {
    double parse = 0, upload = 0, decode = 0, download = 0;
    size_t rawBlocks = 0, rleBlocks = 0, compressedBlocks = 0;
};

// Uses the existing device Zstd kernel, which is not hardened against hostile
// compressed data. Only pass trusted, integrity-checked streams. A false return
// means that a well-framed input uses a layout this kernel cannot represent.
// On success, out may receive host bytes and/or device receives a HIP allocation
// released with fgpu::free_device. Neither output is changed on a false return.
bool decode_lossless_current(const LosslessInput &in, void *out,
                             LosslessTiming *timing = nullptr, void **device = nullptr);

namespace lossless_detail {

struct Block {
    size_t src, dst, length, stored;
    unsigned type;
};

struct Layout {
    size_t decoded = 0;
    std::vector<Block> blocks;
};

// The device decoder assumes each non-final block expands to 128 KiB.
// Bound every metadata read before constructing any device work descriptors.
inline bool parse(const LosslessInput &in, Layout &layout) {
    static_assert(sizeof(size_t) == 8, "Falcons lossless payload requires the 64-bit stream ABI");
    if (!in.cmp || in.cmpSize < sizeof(size_t))
        throw std::invalid_argument("truncated Falcons lossless payload");
    size_t decoded;
    std::memcpy(&decoded, in.cmp, sizeof(decoded));
    if (decoded != in.expectedBytes || decoded > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) - 512)
        throw std::invalid_argument("Falcons lossless output size mismatch");
    Layout result;
    result.decoded = decoded;
    const size_t prefix = sizeof(size_t);
    if (in.rawNoZstd) {
        if (in.cmpSize - prefix != decoded) throw std::invalid_argument("invalid Falcons raw payload size");
        if (decoded) result.blocks.push_back({prefix, 0, decoded, decoded, 0});
        layout = std::move(result);
        return true;
    }
    const unsigned char *frame = in.cmp + prefix;
    const size_t frame_size = in.cmpSize - prefix;
    const size_t found = ZSTD_findFrameCompressedSize(frame, frame_size);
    if (ZSTD_isError(found) || found != frame_size)
        throw std::invalid_argument("invalid or truncated Falcons Zstd frame");
    const unsigned long long content = ZSTD_getFrameContentSize(frame, frame_size);
    if (content == ZSTD_CONTENTSIZE_ERROR) throw std::invalid_argument("invalid Falcons Zstd header");
    if (content == ZSTD_CONTENTSIZE_UNKNOWN) return false;
    if (content != decoded) throw std::invalid_argument("Falcons Zstd content size mismatch");
    if (frame_size < 5 || frame[0] != 0x28 || frame[1] != 0xb5 ||
        frame[2] != 0x2f || frame[3] != 0xfd)
        throw std::invalid_argument("invalid Falcons Zstd magic");
    const unsigned desc = frame[4];
    if ((desc & 0x18) || (desc & 3) || (desc & 4)) return false; // reserved, dictionary, or checksum
    const bool single = (desc & 0x20) != 0;
    const unsigned fcs_flag = desc >> 6;
    const size_t fcs_size = fcs_flag ? (size_t(1) << fcs_flag) : (single ? 1 : 0);
    size_t at = 5 + (single ? 0 : 1) + fcs_size;
    if (at > frame_size) throw std::invalid_argument("truncated Falcons Zstd header");
    size_t produced = 0;
    bool last = false;
    while (!last) {
        if (frame_size - at < 3) throw std::invalid_argument("truncated Falcons Zstd block header");
        const uint32_t bh = uint32_t(frame[at]) | (uint32_t(frame[at + 1]) << 8) |
                            (uint32_t(frame[at + 2]) << 16);
        at += 3;
        last = (bh & 1) != 0;
        const unsigned type = (bh >> 1) & 3;
        const size_t stored_size = bh >> 3;
        if (type == 3 || stored_size > 128 * 1024)
            throw std::invalid_argument("invalid Falcons Zstd block type or size");
        const size_t remaining = decoded - produced;
        if (!last && remaining <= 128 * 1024) return false;
        const size_t output_size = last ? remaining : 128 * 1024;
        if (output_size > 128 * 1024 || (type != 2 && stored_size != output_size)) return false;
        const size_t input_size = type == 1 ? 1 : stored_size;
        if (input_size > frame_size - at) throw std::invalid_argument("truncated Falcons Zstd block");
        if (type == 2 && !input_size) throw std::invalid_argument("empty compressed Zstd block");
        result.blocks.push_back({prefix + at, produced, output_size, input_size, type});
        produced += output_size;
        at += input_size;
    }
    if (produced != decoded || at != frame_size) throw std::invalid_argument("invalid Falcons Zstd frame size");
    layout = std::move(result);
    return true;
}

} // namespace lossless_detail
} // namespace fgpu
