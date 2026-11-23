#ifndef HIP_FALCONS_DATALOADER_PACKED_HPP
#define HIP_FALCONS_DATALOADER_PACKED_HPP

#include "../nd_compact.hpp"

#include <cstddef>
#include <cstdint>

namespace hip_falcons {

// Each offset is in bytes from the beginning of this plane's PackedHeader.
// The corresponding arrays are CompactGroup, int16_t, float, and uint32_t;
// their addresses must satisfy the alignment of their respective element type.
// Escape positions are sorted code indices; unpredictable values start with
// the anchors and then contain one value per escape in code-index order.
struct PackedHeader {
    CompactAnchors anchors;
    std::uint64_t group_off;
    std::uint64_t code_off;
    std::uint64_t unpred_off;
    std::uint64_t escape_off;
    std::uint64_t escape_count;
    std::uint64_t group_count;
};

}  // namespace hip_falcons

// device_offsets contains one nonnegative byte offset into device_bytes per
// plane. Each plane start must be aligned for PackedHeader; its descriptor
// groups must be in dependency order and group_count must not exceed
// max_groups. All planes contain elements floats in the output. The buffers
// and stream are borrowed, and valid contents, sizes, and device placement are
// the caller's responsibility. This function only enqueues work; it neither
// transfers data nor synchronizes the stream.
extern "C" int hf_launch_packed(
    const void* device_bytes, const std::int64_t* device_offsets,
    float* device_out, std::size_t planes, std::size_t elements,
    std::size_t max_groups, void* stream);

#endif  // HIP_FALCONS_DATALOADER_PACKED_HPP
