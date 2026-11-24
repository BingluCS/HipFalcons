// Include after the closing namespace in fgpu_recon.hip.cpp. The adapter reuses
// that translation unit's private device Zstd implementation.
#include "fgpu_lossless.hpp"

namespace fgpu {
namespace {

__global__ void lossless_fill_kernel(uint8_t *dst, const uint8_t *src, const CopyChunk *work) {
    const CopyChunk c = work[blockIdx.x];
    if (threadIdx.x < c.len) dst[c.dst + threadIdx.x] = src[c.src];
}

} // namespace

bool decode_lossless_current(const LosslessInput &in, void *out, LosslessTiming *timing, void **device) {
    if (!out && !device) throw std::invalid_argument("no Falcons lossless output requested");
    const auto t0 = Clock::now();
    lossless_detail::Layout layout;
    if (!lossless_detail::parse(in, layout)) return false;
    if (in.cmpSize > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) - FG_BLOCK ||
        layout.decoded > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) - (128 * 1024 + FG_BLOCK))
        throw std::length_error("Falcons lossless allocation overflow");

    std::vector<CopyChunk> raw, fill;
    std::vector<ZBlk> compressed;
    LosslessTiming local;
    for (const auto &b : layout.blocks) {
        if (b.type == 0) {
            add_copy(raw, b.src, b.dst, b.length);
            ++local.rawBlocks;
        } else if (b.type == 1) {
            // The existing copy descriptors are contiguous; an RLE chunk always
            // reads the same source byte instead of advancing its source offset.
            const size_t first = fill.size();
            add_copy(fill, b.src, b.dst, b.length);
            for (size_t i = first; i < fill.size(); ++i) fill[i].src = b.src;
            ++local.rleBlocks;
        } else {
            compressed.push_back({b.src, b.dst, b.dst + b.length,
                                  static_cast<uint32_t>(b.stored), 2});
            ++local.compressedBlocks;
        }
    }
    if (raw.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        fill.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        compressed.size() > std::numeric_limits<uint32_t>::max())
        throw std::length_error("Falcons lossless work table too large");
    const auto t1 = Clock::now();
    Dev<uint8_t> src(in.cmpSize + FG_BLOCK), dst(layout.decoded + 128 * 1024 + FG_BLOCK);
    CK(hipMemcpy(src.p, in.cmp, in.cmpSize, hipMemcpyHostToDevice));
    CK(hipMemset(src.p + in.cmpSize, 0, FG_BLOCK));
    const auto t2 = Clock::now();
    std::unique_ptr<Dev<CopyChunk>> raw_work, fill_work;
    if (!raw.empty()) {
        raw_work.reset(new Dev<CopyChunk>(raw.size()));
        CK(hipMemcpy(raw_work->p, raw.data(), raw.size() * sizeof(CopyChunk), hipMemcpyHostToDevice));
        copy_bytes_kernel<<<static_cast<int>(raw.size()), FG_BLOCK>>>(dst.p, src.p, raw_work->p);
        CK(hipGetLastError());
    }
    if (!fill.empty()) {
        fill_work.reset(new Dev<CopyChunk>(fill.size()));
        CK(hipMemcpy(fill_work->p, fill.data(), fill.size() * sizeof(CopyChunk), hipMemcpyHostToDevice));
        lossless_fill_kernel<<<static_cast<int>(fill.size()), FG_BLOCK>>>(dst.p, src.p, fill_work->p);
        CK(hipGetLastError());
    }
    if (!compressed.empty()) {
        const ZStream stream{0, static_cast<uint32_t>(compressed.size())};
        Dev<ZStream> streams(1);
        Dev<ZBlk> blocks(compressed.size());
        Dev<uint8_t> scratch(ZSCRATCH);
        Dev<uint32_t> bad(1);
        CK(hipMemset(bad.p, 0, sizeof(uint32_t)));
        CK(hipMemcpy(streams.p, &stream, sizeof(stream), hipMemcpyHostToDevice));
        CK(hipMemcpy(blocks.p, compressed.data(), compressed.size() * sizeof(ZBlk), hipMemcpyHostToDevice));
        // Zstd's history, repeat offsets, and entropy tables cross block boundaries.
        zstd_kernel<<<1, 1>>>(dst.p, src.p, streams.p, blocks.p, scratch.p, bad.p);
        CK(hipGetLastError());
        uint32_t bad_result = 0;
        CK(hipMemcpy(&bad_result, bad.p, sizeof(bad_result), hipMemcpyDeviceToHost));
        if (bad_result) return false;
    }
    CK(hipDeviceSynchronize());
    const auto t3 = Clock::now();
    if (out && layout.decoded) CK(hipMemcpy(out, dst.p, layout.decoded, hipMemcpyDeviceToHost));
    if (device) *device = dst.release();
    const auto t4 = Clock::now();
    local.parse = ms(t0, t1); local.upload = ms(t1, t2);
    local.decode = ms(t2, t3); local.download = ms(t3, t4);
    if (timing) *timing = local;
    return true;
}

} // namespace fgpu
