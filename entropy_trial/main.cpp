#include "h1.hpp"
#include "rle.hpp"
#include "escapes.hpp"
#include "../dataloader/packed.hpp"
#include "../stream.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" int hf_pack_batch(const void* const*, const std::size_t*, std::size_t,
                             std::size_t, std::size_t, void*, std::size_t, std::int64_t*,
                             std::size_t*, std::size_t*);
extern "C" int hf_cpu_f32(const void*, std::size_t, float*, std::size_t);

namespace {
using namespace hip_falcons;
using namespace hip_falcons::entropy_trial;
using Clock = std::chrono::steady_clock;
using Bytes = std::vector<unsigned char>;

void check(int status, const char* operation) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 hipGetErrorString(static_cast<hipError_t>(status)));
}

Bytes read_file(const char* path) {
    std::ifstream stream(path, std::ios::ate | std::ios::binary);
    if (!stream || stream.tellg() <= 0) throw std::runtime_error(std::string("cannot read ") + path);
    Bytes data(static_cast<std::size_t>(stream.tellg()));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char*>(data.data()), data.size()))
        throw std::runtime_error(std::string("incomplete file: ") + path);
    return data;
}

double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

template<class T>
class Device {
public:
    explicit Device(std::size_t n) : count(n) {
        if (count) check(hipMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)), "hipMalloc");
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { if (data) (void)hipFree(data); }
    void upload(const T* host) const {
        if (count) check(hipMemcpy(data, host, count * sizeof(T), hipMemcpyHostToDevice), "upload");
    }
    void download(T* host) const {
        if (count) check(hipMemcpy(host, data, count * sizeof(T), hipMemcpyDeviceToHost), "download");
    }
    T* data = nullptr;
    std::size_t count;
};

template<class Batch>
struct Capture {
    static inline thread_local Batch* active = nullptr;
    static bool unpack(const unsigned char* p, std::size_t bytes, std::size_t n,
                       std::unique_ptr<std::int16_t[]>&) {
        if (!active->append(p, bytes, n))
            throw std::runtime_error("unsupported entropy layout in prototype (or malformed stream)");
        return true;  // The device path does not materialize the codes on the host.
    }
};

template<class Batch>
std::vector<Inputs> prepare(const std::vector<Bytes>& streams, Batch& batch) {
    Capture<Batch>::active = &batch;
    struct Reset { ~Reset() { Capture<Batch>::active = nullptr; } } reset;
    thread_local BatchUnpackWorkspace workspace;
    std::vector<Inputs> planes;
    for (const auto& stream : streams) {
        UnpackTiming t;
        planes.push_back(unpack_batch(reinterpret_cast<const char*>(stream.data()), stream.size(),
                                      t, Capture<Batch>::unpack, workspace));
    }
    return planes;
}

// Use the packed interpolation layout, with codes generated directly on device.
struct Reconstruction {
    Bytes slab;
    std::vector<std::int64_t> plane_offsets;
    std::vector<std::size_t> code_offsets;
    std::vector<std::size_t> escape_offsets;
    std::vector<std::size_t> escape_counts;
    std::size_t groups = 0, elements = 0, code_count = 0;

    bool append(const Inputs& input) {
        const auto& p = input.params;
        if (input.dims.size() != 2 || (elements && elements != p.num_elements)) return false;
        const auto plan = make_compact_plan({input.dims, p.block_size, p.anchor_stride, input.direction,
                                             p.error_bound, p.eb_alpha, p.eb_beta});
        if (plan.code_count != input.code_count || input.unpredictable.size() < plan.anchors.count)
            return false;
        elements = p.num_elements;
        groups = std::max(groups, plan.groups.size());
        const std::size_t start = (slab.size() + 7) & ~std::size_t{7};
        PackedHeader header{};
        header.anchors = plan.anchors;
        header.group_off = sizeof(header);
        header.group_count = plan.groups.size();
        header.unpred_off = header.group_off + header.group_count * sizeof(CompactGroup);
        header.escape_off = header.unpred_off + input.unpredictable.size() * sizeof(float);
        header.escape_count = input.unpredictable.size() - plan.anchors.count;
        slab.resize(start + header.escape_off + header.escape_count * sizeof(std::uint32_t));
        auto* out = slab.data() + start;
        std::memcpy(out, &header, sizeof(header));
        if (!plan.groups.empty())
            std::memcpy(out + header.group_off, plan.groups.data(), plan.groups.size() * sizeof(CompactGroup));
        if (!input.unpredictable.empty())
            std::memcpy(out + header.unpred_off, input.unpredictable.data(), input.unpredictable.size() * sizeof(float));
        plane_offsets.push_back(static_cast<std::int64_t>(start));
        code_offsets.push_back(code_count * sizeof(std::int16_t));
        escape_offsets.push_back(start + header.escape_off);
        escape_counts.push_back(header.escape_count);
        code_count += input.code_count;
        return true;
    }

    void finish() {
        for (std::size_t i = 0; i < plane_offsets.size(); ++i) {
            auto* base = slab.data() + plane_offsets[i];
            const std::uint64_t offset = slab.size() + code_offsets[i] - plane_offsets[i];
            std::memcpy(base + offsetof(PackedHeader, code_off), &offset, sizeof(offset));
        }
    }
};

Reconstruction prepare_reconstruction(const std::vector<Inputs>& planes) {
    Reconstruction r;
    for (const auto& plane : planes)
        if (!r.append(plane)) return {};
    r.finish();
    return r;
}

template<class Batch, class HostDecode, class Launch>
void run(const std::vector<Bytes>& streams, const std::vector<std::int16_t>& oracle,
         bool use_device, HostDecode decode, Launch launch) {
    Batch batch;
    auto start = Clock::now();
    auto planes = prepare(streams, batch);
    const auto packed = batch.pack();
    const double prepare_ms = elapsed(start);
    if (batch.code_count != oracle.size() || batch.jobs() == 0)
        throw std::runtime_error("unexpected code count");
    std::vector<std::int16_t> output(oracle.size());
    if (!decode(packed.data(), output.data(), batch.jobs()) || output != oracle)
        throw std::runtime_error("host decoder differs from CPU Falcons");
    std::cout << "host parity OK: planes=" << streams.size() << " jobs=" << batch.jobs()
              << " codes=" << batch.code_count << " uploaded_entropy_bytes=" << packed.size()
              << " dense_code_bytes=" << oracle.size() * sizeof(std::int16_t)
              << " prepare_ms=" << prepare_ms << '\n';
    if (!use_device) return;

    const auto reconstruction = prepare_reconstruction(planes);
    const bool reconstruct = !reconstruction.slab.empty();
    if (!reconstruct)
        std::cout << "interpolation check skipped (requires equal planar sizes)\n";
    if (reconstruct) {
        std::size_t start_code = 0;
        for (std::size_t i = 0; i < planes.size(); ++i) {
            const auto lo = oracle.begin() + start_code;
            start_code += planes[i].code_count;
            if (static_cast<std::size_t>(std::count(lo, oracle.begin() + start_code, INT16_MIN)) !=
                reconstruction.escape_counts[i])
                throw std::runtime_error("quantizer escape metadata differs from decoded codes");
        }
    }
    Device<unsigned char> device_input(packed.size());
    Device<std::int16_t> standalone_codes(reconstruct ? 0 : oracle.size());
    Device<unsigned char> recon(reconstruct ? reconstruction.slab.size() + oracle.size() * sizeof(std::int16_t) : 0);
    Device<std::int64_t> offsets(reconstruct ? streams.size() : 0);
    Device<float> field(reconstruct ? reconstruction.elements * streams.size() : 0);
    if (reconstruct) {
        check(hipMemcpy(recon.data, reconstruction.slab.data(), reconstruction.slab.size(),
                        hipMemcpyHostToDevice), "upload reconstruction metadata");
        offsets.upload(reconstruction.plane_offsets.data());
    }
    auto* device_codes = reconstruct ? reinterpret_cast<std::int16_t*>(recon.data + reconstruction.slab.size())
                                    : standalone_codes.data;
    const auto device_decode = [&] {
        check(launch(device_input.data, device_codes, batch.jobs(), nullptr), "launch entropy");
    };

    std::size_t scratch_bytes = 0;
    const bool escapes = reconstruct && std::any_of(reconstruction.escape_counts.begin(),
        reconstruction.escape_counts.end(), [](auto count) { return count != 0; });
    Device<std::uint32_t> escape_counts(escapes ? planes.size() : 0);
    if (escapes) {
        check(collect_escapes(nullptr, scratch_bytes, device_codes, reconstruction.elements,
                              nullptr, escape_counts.data, nullptr), "query escape scratch");
    }
    Device<unsigned char> scratch(scratch_bytes);
    const auto collect = [&] {
        if (!escapes) return;
        for (std::size_t i = 0; i < planes.size(); ++i) {
            if (!reconstruction.escape_counts[i]) continue;
            const auto* codes = device_codes + reconstruction.code_offsets[i] / sizeof(std::int16_t);
            auto* dst = reinterpret_cast<std::uint32_t*>(recon.data + reconstruction.escape_offsets[i]);
            auto capacity = scratch.count;
            check(collect_escapes(scratch.data, capacity, codes, planes[i].code_count, dst,
                                  escape_counts.data + i, nullptr), "collect escape positions");
        }
    };
    device_input.upload(packed.data());
    device_decode();
    check(hipStreamSynchronize(nullptr), "verify entropy sync");
    check(hipMemcpy(output.data(), device_codes, output.size() * sizeof(std::int16_t),
                    hipMemcpyDeviceToHost), "download codes");
    if (output != oracle) throw std::runtime_error("HIP entropy output differs from CPU Falcons");
    collect();
    if (escapes) {
        std::vector<std::uint32_t> counts(planes.size());
        escape_counts.download(counts.data());
        for (std::size_t i = 0; i < planes.size(); ++i)
            if (reconstruction.escape_counts[i] && counts[i] != reconstruction.escape_counts[i])
                throw std::runtime_error("HIP escape position count differs from CPU Falcons");
    }

    const auto device_interp = [&](const Device<unsigned char>& data, std::size_t groups) {
        check(hf_launch_packed(data.data, offsets.data, field.data, planes.size(),
                              reconstruction.elements, groups, nullptr), "launch interpolation");
        check(hipStreamSynchronize(nullptr), "interpolation sync");
    };
    const auto candidate = [&](int trial) {
        const auto t0 = Clock::now();
        Batch current;
        const auto input = prepare(streams, current);
        const auto payload = current.pack();
        const auto r = reconstruct ? prepare_reconstruction(input) : Reconstruction{};
        const auto t1 = Clock::now();
        if (payload.size() != device_input.count || r.slab.size() != reconstruction.slab.size())
            throw std::runtime_error("trial buffer size changed");
        device_input.upload(payload.data());
        if (reconstruct) {
            check(hipMemcpy(recon.data, r.slab.data(), r.slab.size(), hipMemcpyHostToDevice), "upload metadata");
            offsets.upload(r.plane_offsets.data());
        }
        const auto t2 = Clock::now();
        device_decode();
        check(hipStreamSynchronize(nullptr), "entropy sync");
        const auto t3 = Clock::now();
        if (reconstruct) {
            collect();
            device_interp(recon, r.groups);
        }
        if (trial)
            std::cout << "route=hip-entropy trial=" << trial
                      << " end_to_device_ms=" << elapsed(t0)
                      << " cpu_prepare_ms=" << std::chrono::duration<double, std::milli>(t1-t0).count()
                      << " h2d_ms=" << std::chrono::duration<double, std::milli>(t2-t1).count()
                      << " dcu_entropy_ms=" << std::chrono::duration<double, std::milli>(t3-t2).count()
                      << " escape_and_interp_ms=" << (reconstruct ? elapsed(t3) : 0.) << '\n';
    };
    if (!reconstruct) {
        for (int trial = 1; trial <= 3; ++trial) candidate(trial);
        return;
    }
    if (!std::all_of(planes.begin(), planes.end(), [&](const auto& plane) { return plane.dims == planes[0].dims; }))
        throw std::runtime_error("paired baseline requires homogeneous plane shapes");

    std::vector<const void*> source;
    std::vector<std::size_t> lengths;
    for (const auto& stream : streams) {
        source.push_back(stream.data());
        lengths.push_back(stream.size());
    }
    Bytes cpu_buffer(reconstruction.slab.size() + oracle.size() * sizeof(std::int16_t) + 8 * planes.size());
    std::vector<std::int64_t> cpu_offsets(planes.size());
    std::size_t used = 0, groups = 0;
    const auto cpu_pack = [&] {
        if (hf_pack_batch(source.data(), lengths.data(), planes.size(), planes[0].dims[0], planes[0].dims[1],
                          cpu_buffer.data(), cpu_buffer.size(), cpu_offsets.data(), &used, &groups) != 0)
            throw std::runtime_error("current CPU packer failed");
    };
    cpu_pack();
    Device<unsigned char> cpu_device(used);
    const auto baseline = [&](int trial) {
        const auto t0 = Clock::now();
        cpu_pack();
        const auto t1 = Clock::now();
        if (used != cpu_device.count) throw std::runtime_error("baseline buffer size changed");
        cpu_device.upload(cpu_buffer.data());
        offsets.upload(cpu_offsets.data());
        const auto t2 = Clock::now();
        device_interp(cpu_device, groups);
        if (trial)
            std::cout << "route=cpu-entropy trial=" << trial
                      << " end_to_device_ms=" << elapsed(t0)
                      << " cpu_prepare_ms=" << std::chrono::duration<double, std::milli>(t1-t0).count()
                      << " h2d_ms=" << std::chrono::duration<double, std::milli>(t2-t1).count()
                      << " interp_ms=" << elapsed(t2) << '\n';
    };
    baseline(0);
    candidate(0);
    // The paired comparison excludes file I/O, buffer setup, and tensor assembly.
    for (int trial = 1; trial <= 3; ++trial) {
        if (trial % 2) { baseline(trial); candidate(trial); }
        else { candidate(trial); baseline(trial); }
    }
    {
        std::vector<float> result(field.count), expected(reconstruction.elements);
        field.download(result.data());
        for (std::size_t i = 0; i < streams.size(); ++i) {
            if (hf_cpu_f32(streams[i].data(), streams[i].size(), expected.data(), expected.size()))
                throw std::runtime_error("CPU reference reconstruction failed");
            if (std::memcmp(expected.data(), result.data() + i * reconstruction.elements,
                            expected.size() * sizeof(float)) != 0)
                throw std::runtime_error("GPU reconstructed field differs from CPU Falcons");
        }
        std::cout << "device float32 parity OK\n";
    }
}

} // namespace

int main(int argc, char** argv) try {
    const bool use_device = argc > 1 && std::string(argv[1]) == "--device";
    const bool host_check = argc > 1 && std::string(argv[1]) == "--host-check";
    if (argc < 3 || (!host_check && !use_device))
        throw std::invalid_argument("usage: hipfalcons_entropy_trial --host-check|--device file.falcons [...]");
    if (use_device) {
        int devices = 0;
        check(hipGetDeviceCount(&devices), "HIP device enumeration");
        if (!devices) throw std::runtime_error("no available HIP devices");
    }
    std::vector<Bytes> streams;
    std::vector<std::int16_t> reference;
    unsigned method = std::numeric_limits<unsigned>::max();
    for (int i = 2; i < argc; ++i) {
        auto& payload = streams.emplace_back(read_file(argv[i]));
        UnpackTiming t;
        const auto plane = unpack(reinterpret_cast<const char*>(payload.data()), payload.size(), t);
        if (i == 2) method = t.method;
        if (t.method != method) throw std::invalid_argument("supply a batch with a single entropy method");
        if (plane.code_count) reference.insert(reference.end(), plane.codes.get(), plane.codes.get() + plane.code_count);
    }
    if (method == 1) run<H1Batch>(streams, reference, use_device, decode_h1_host, launch_h1);
    else if (method == 0) run<RleBatch>(streams, reference, use_device, decode_rle_host, launch_rle);
    else throw std::invalid_argument("only entropy methods 0 and 1 are supported");
    return 0;
} catch (const std::exception& e) {
    std::cerr << "hipfalcons_entropy_trial: " << e.what() << '\n';
    return 1;
}
