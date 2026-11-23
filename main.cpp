#include "stream.hpp"
#include "reconstruct.hpp"
#include "nd_reconstruct.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Falcons::uchar;
using hip_falcons::Reconstruct1DParams;
using hip_falcons::ReconstructResult;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Timing : hip_falcons::UnpackTiming {
    Clock::time_point started;
    double unpack = 0, plan = 0, upload = 0, kernel = 0, download = 0, total = 0;
    double device_ready = 0;
};

void hip_check(hipError_t err, const char* operation) {
    if (err != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(err));
}

void require_device() {
    int count = 0;
    auto rc = hipGetDeviceCount(&count);
    if (rc != hipSuccess || !count)
        throw std::runtime_error("no accessible HIP device; check rocminfo, /dev/kfd and job allocation");
    int device = 0;
    hipDeviceProp_t props{};
    hip_check(hipGetDevice(&device), "hipGetDevice");
    hip_check(hipGetDeviceProperties(&props, device), "hipGetDeviceProperties");
    std::cout << "HIP device " << device << ": " << props.name << " (" << props.gcnArchName << ")\n";
}

template<class T>
struct Device {
    T* ptr = nullptr;
    explicit Device(size_t n) {
        if (n) hip_check(hipMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)), "hipMalloc");
    }
    Device(const Device&) = delete;
    ~Device() { if (ptr) (void)hipFree(ptr); }
    void upload(const T* src, size_t n) {
        if (n) hip_check(hipMemcpy(ptr, src, n * sizeof(T), hipMemcpyHostToDevice), "upload");
    }
    void download(T* dst, size_t n) {
        if (n) hip_check(hipMemcpy(dst, ptr, n * sizeof(T), hipMemcpyDeviceToHost), "download");
    }
};

using hip_falcons::Inputs;
using hip_falcons::unpack;

std::vector<float> reconstruct_1d(const Inputs& in, bool host_check, Timing& times) {
    std::vector<float> out(in.params.num_elements);
    auto start = Clock::now();
    std::vector<uint32_t> escapes;
    for (size_t i = 0; i < in.code_count; ++i)
        if (in.codes[i] == INT16_MIN) escapes.push_back(static_cast<uint32_t>(i));
    const size_t stride = in.params.anchor_stride;
    const size_t anchors = stride && stride < out.size() ? 1 + (out.size() - 1) / stride : 0;
    if (anchors + escapes.size() != in.unpredictable.size())
        throw std::invalid_argument("unpredictable count does not match anchors and escapes");
    times.plan = elapsed(start);
    if (host_check) {
        ReconstructResult result{};
        hip_check(hip_falcons::reconstruct_1d_linear_f32_host(in.params, in.codes.get(), in.code_count,
            in.unpredictable.data(), in.unpredictable.size(), escapes.data(), escapes.size(),
            out.data(), &result), "host traversal");
        if (result.status != hip_falcons::ReconstructStatus::success)
            throw std::runtime_error("host traversal count mismatch");
    } else {
        Device<int16_t> codes(in.code_count);
        Device<float> unpred(in.unpredictable.size()), output(out.size());
        Device<uint32_t> index(escapes.size());
        start = Clock::now();
        codes.upload(in.codes.get(), in.code_count);
        unpred.upload(in.unpredictable.data(), in.unpredictable.size());
        index.upload(escapes.data(), escapes.size());
        times.upload = elapsed(start);
        start = Clock::now();
        hip_check(hip_falcons::reconstruct_1d_linear_f32_parallel_async(in.params, codes.ptr, in.code_count,
            unpred.ptr, in.unpredictable.size(), index.ptr, escapes.size(), output.ptr), "HIP reconstruction");
        hip_check(hipStreamSynchronize(nullptr), "finish 1D reconstruction");
        times.kernel = elapsed(start);
        times.device_ready = elapsed(times.started);
        start = Clock::now();
        output.download(out.data(), out.size());
        times.download = elapsed(start);
    }
    return out;
}

std::vector<float> reconstruct_nd(const Inputs& in, bool host, Timing& times) {
    const auto& m = in.params;
    hip_falcons::NDReconstructParams params{
        in.dims, m.block_size, m.anchor_stride, in.direction, m.error_bound, m.eb_alpha, m.eb_beta
    };
    std::vector<float> out(m.num_elements);
    if (host) {
        hip_check(hip_falcons::reconstruct_nd_linear_f32_host(params, in.codes.get(), in.code_count,
            in.unpredictable.data(), in.unpredictable.size(), out.data()), "host ND reconstruction");
    } else {
        hip_falcons::NDReconstructTimings t;
        const double prefix = elapsed(times.started);
        hip_check(hip_falcons::reconstruct_nd_linear_f32_gpu(params, in.codes.get(), in.code_count,
            in.unpredictable.data(), in.unpredictable.size(), out.data(), &t), "HIP ND reconstruction");
        times.plan = t.plan_ms;
        times.upload = t.upload_ms;
        times.kernel = t.kernel_ms;
        times.download = t.download_ms;
        times.device_ready = prefix + t.to_device_ms;
    }
    return out;
}

std::vector<float> decode(const char* stream, size_t len, bool host, Timing& t) {
    auto start = Clock::now();
    t.started = start;
    std::vector<float> out;
    {
        auto in = unpack(stream, len, t);
        t.unpack = elapsed(start);
        out = in.dims.size() == 1 ? reconstruct_1d(in, host, t) : reconstruct_nd(in, host, t);
    }
    t.total = elapsed(start);
    return out;
}

struct Case {
    std::vector<size_t> dims;
    int anchor = -1;
    double alpha = 1.25;
    bool relative = false;
    int direction = 0;
};

void run_case(const Case& c, bool host, bool benchmark) {
    Falcons::Config conf;
    conf.setDims(c.dims.begin(), c.dims.end());
    conf.cmprAlgo = Falcons::ALGO_INTERP;
    conf.interpAlgo = Falcons::INTERP_ALGO_LINEAR;
    conf.interpDirection = c.direction;
    conf.interpAnchorStride = c.anchor;
    conf.interpAlpha = c.alpha;
    conf.errorBoundMode = c.relative ? Falcons::EB_REL : Falcons::EB_ABS;
    conf.absErrorBound = .001;
    conf.relErrorBound = 1e-6;
    conf.openmp = false;
    std::vector<float> original(conf.num);
    const size_t nx = c.dims.back(), ny = c.dims.size() >= 2 ? c.dims[c.dims.size()-2] : 1;
    for (size_t i = 0; i < conf.num; ++i) {
        const size_t x = i % nx, y = i / nx % ny, z = i / nx / ny;
        original[i] = conf.num != 2048 && i % 947 == 0 ? 1200.f :
            static_cast<float>(3. * std::sin(.017*x + .013*y) + .0004*x + std::sin(.023*y + .031*z));
    }
    auto working = original;
    size_t bytes = 0;
    std::unique_ptr<char[]> compressed(Falcons_compress(conf, working.data(), bytes));
    std::string shape;
    for (auto dim : c.dims) shape += (shape.empty() ? "" : "x") + std::to_string(dim);
    std::cout << "shape=" << shape << " direction=" << c.direction << " anchor=" << c.anchor
              << " backend=" << (host ? "host" : "DCU") << std::endl;
    const auto range = std::minmax_element(original.begin(), original.end());
    const double bound = c.relative ? conf.relErrorBound * (double(*range.second) - *range.first) : conf.absErrorBound;
    if (benchmark) {
        Timing warmup;
        (void)decode(compressed.get(), bytes, host, warmup);
        Falcons::Config tmp;
        std::unique_ptr<float[]> warmref(Falcons_decompress<float>(tmp, compressed.get(), bytes));
    }
    for (int rep = 0; rep < (benchmark ? 3 : 1); ++rep) {
        Timing t;
        Falcons::Config decoded;
        auto cpu_start = Clock::now();
        std::unique_ptr<float[]> ref(Falcons_decompress<float>(decoded, compressed.get(), bytes));
        double cpu = elapsed(cpu_start);
        auto out = decode(compressed.get(), bytes, host, t);
        size_t mismatch = 0, first = out.size();
        double maxerr = 0;
        for (size_t i = 0; i < out.size(); ++i) {
            if (std::memcmp(&out[i], &ref[i], sizeof(float))) { ++mismatch; first = std::min(first, i); }
            if (!std::isfinite(out[i])) throw std::runtime_error("non-finite reconstruction");
            maxerr = std::max(maxerr, std::abs(double(out[i]) - original[i]));
        }
        std::cout << std::setprecision(10) << "  mismatches=" << mismatch << " max_abs_error=" << maxerr
                  << " bound=" << bound;
        if (mismatch) std::cout << " first=" << first << " got=" << out[first] << " cpu=" << ref[first];
        if (benchmark) {
            std::cout << " rep=" << rep+1 << " bytes=" << out.size()*sizeof(float) << " compressed=" << bytes
                      << " cpu_ms=" << cpu << " total_ms=" << t.total << " unpack_ms=" << t.unpack
                      << " plan_ms=" << t.plan << " h2d_ms=" << t.upload << " kernel_ms=" << t.kernel
                      << " d2h_ms=" << t.download << " to_device_ms=" << t.device_ready
                      << " zstd_ms=" << t.zstd << " entropy_ms=" << t.entropy << " method=" << t.method;
        }
        std::cout << std::endl;
        if (mismatch || maxerr > bound) throw std::runtime_error("cross-decode failed");
    }
}

void self_test(bool host, bool benchmark, bool large) {
    if (!host) require_device();
    if (benchmark) {
        if (large) {
            for (const auto& c : {Case{{4096,4096}}, Case{{256,256,256}}}) run_case(c, host, true);
            return;
        }
        for (const auto& c : {Case{{1048576}}, Case{{1024,1024}}, Case{{128,128,64}}})
            run_case(c, host, true);
        return;
    }
    for (const auto& c : {Case{{2048}, 0, 1.}, Case{{4095}, 0}, Case{{4096}, 0, -1.},
                          Case{{4097}}, Case{{8194}, 64, 1.5, true},
                          Case{{65,129}, 0}, Case{{70,130}, 16, 1.5, true, 1},
                          Case{{2,3073}, 8}, Case{{17,22,35}, 0}, Case{{18,25,38}, 8, 1.25, true, 5},
                          Case{{3,2,769}, 8}, Case{{17,24,33}, 8, 1.25, false, 1},
                          Case{{18,23,34}, 0, -1., false, 2}, Case{{19,22,35}, 16, 1.5, true, 3},
                          Case{{20,21,36}, 0, 1., false, 4}})
        run_case(c, host, false);
}
} // namespace

int main(int argc, char** argv) try {
    if (argc == 2 && (std::string(argv[1]) == "--self-test" || std::string(argv[1]) == "--host-check" ||
                      std::string(argv[1]) == "--bench" || std::string(argv[1]) == "--bench-large")) {
        const std::string mode = argv[1];
        self_test(mode == "--host-check", mode == "--bench" || mode == "--bench-large", mode == "--bench-large");
        return 0;
    }
    if (argc != 3) {
        std::cerr << "usage: hipfalcons input.falcons output.f32 | --self-test | --host-check | --bench | --bench-large\n";
        return 2;
    }
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0) throw std::runtime_error("cannot open input or empty file");
    std::vector<char> bytes(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    if (!file.read(bytes.data(), bytes.size())) throw std::runtime_error("read failed");
    require_device();
    Timing times;
    auto out = decode(bytes.data(), bytes.size(), false, times);
    std::ofstream dest(argv[2], std::ios::binary);
    dest.write(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(float));
    dest.close();
    if (!dest) throw std::runtime_error("output write failed");
    return 0;
} catch (const std::exception& err) {
    std::cerr << "hipfalcons: " << err.what() << '\n';
    return 1;
}
