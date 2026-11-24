#include "current_compress.hpp"
#include "current_stream.hpp"
#include "fgpu_lossless.hpp"
#include "fgpu_recon.h"

#include "Falcons/api/falcons.hpp"
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;

struct Options {
    std::string mode = "hybrid";
    std::string dtype = "f32";
    std::vector<std::size_t> dims;
    int streams = 1;
    int interp = Falcons::INTERP_ALGO_CUBIC;
    double bound = 0.;
    bool relative = false;
    bool bench = false;
};

std::size_t positive(std::string_view s) {
    std::size_t value = 0;
    const auto end = s.data() + s.size();
    const auto result = std::from_chars(s.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end || value == 0)
        throw std::invalid_argument("expected a positive integer");
    return value;
}

Options options(int argc, char** argv, int first) {
    Options out;
    for (int i = first; i < argc; ++i) {
        const std::string a(argv[i]);
        const auto next = [&]() -> std::string {
            if (++i == argc) throw std::invalid_argument("missing value for " + a);
            return argv[i];
        };
        if (a == "--bench") out.bench = true;
        else if (a == "--mode") out.mode = next();
        else if (a == "--dtype") out.dtype = next();
        else if (a == "--streams") {
            const auto n = positive(next());
            if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("too many streams");
            out.streams = static_cast<int>(n);
        } else if (a == "--shape") {
            while (i + 1 < argc && argv[i + 1][0] != '-')
                out.dims.push_back(positive(argv[++i]));
        } else if (a == "--linear") out.interp = Falcons::INTERP_ALGO_LINEAR;
        else if (a == "--cubic") out.interp = Falcons::INTERP_ALGO_CUBIC;
        else if (a == "--abs" || a == "--rel") {
            const auto s = next();
            std::size_t pos = 0;
            out.bound = std::stod(s, &pos);
            if (pos != s.size() || !(out.bound > 0.) || !std::isfinite(out.bound))
                throw std::invalid_argument("error bound must be finite and positive");
            out.relative = a == "--rel";
        } else throw std::invalid_argument("unknown option " + a);
    }
    if (out.mode != "hybrid" && out.mode != "pure") throw std::invalid_argument("expected hybrid or pure mode");
    if (out.dtype != "f32" && out.dtype != "f64") throw std::invalid_argument("expected f32 or f64");
    return out;
}

std::vector<char> read_file(const char* path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() <= 0) throw std::runtime_error(std::string("cannot read ") + path);
    std::vector<char> bytes(static_cast<std::size_t>(stream.tellg()));
    stream.seekg(0);
    if (!stream.read(bytes.data(), bytes.size())) throw std::runtime_error(std::string("short read from ") + path);
    return bytes;
}

void write_file(const char* path, const void* bytes, std::size_t size) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream.write(static_cast<const char*>(bytes), size)) throw std::runtime_error(std::string("cannot write ") + path);
    stream.close();
    if (!stream) throw std::runtime_error(std::string("cannot close ") + path);
}

void require_device() {
    int n = 0;
    const auto status = hipGetDeviceCount(&n);
    if (status != hipSuccess || n == 0) throw std::runtime_error("no accessible HIP device");
}

template<class T>
constexpr int data_type = std::is_same_v<T, float> ? Falcons_FLOAT : Falcons_DOUBLE;

struct Sample { double total = 0, recon = 0; };

// The codec calls are synchronous at the host-output boundary. Disk I/O is outside the timer.
template<class F>
Sample measure(bool bench, F&& run) {
    if (bench) run();
    const int n = bench ? 3 : 1;
    std::array<Sample, 3> samples{};
    for (int i = 0; i < n; ++i) {
        const auto start = Clock::now();
        samples[i].recon = run();
        samples[i].total = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    if (!bench) return samples[0];
    std::sort(samples.begin(), samples.end(), [](Sample a, Sample b) { return a.total < b.total; });
    return samples[1];
}

template<class T>
bool reconstruct(const std::vector<char>& cmp, const Falcons::Config& cfg, T* out, bool pure, double* recon_ms = nullptr) {
    if (cfg.cmprAlgo != Falcons::ALGO_INTERP || cfg.N < 1 || cfg.N > 4 ||
        cfg.dataType != data_type<T>)
        throw std::invalid_argument("HIP codec expects float32/float64 interpolation in 1D-4D");
    const auto header = static_cast<unsigned char>(cmp[0]);
    const auto* payload = reinterpret_cast<const unsigned char*>(cmp.data()) + header;
    const auto size = cmp.size() - header;
    if (pure) {
        if (size < 12) throw std::invalid_argument("truncated pure decoder payload");
        const auto v100 = versionInt("1.0.0");
        if (cfg.falconsDataVer != v100 && cfg.falconsDataVer != versionInt("1.0.1") &&
            cfg.falconsDataVer != versionInt("1.1.0"))
            return false;
        fgpu::StreamConfig sc{static_cast<std::size_t>(cfg.N), cfg.falconsDataVer == v100};
        fgpu::PureInput args;
        args.conf = &sc;
        args.sizeT = sizeof(T);
        args.cmp = payload;
        args.cmpSize = size;
        fgpu::PureTiming tm;
        const bool ok = fgpu::decode_pure(args, out, recon_ms ? &tm : nullptr);
        if (recon_ms) *recon_ms = tm.recon;
        return ok;
    }

    auto streams = fgpu::decode_current<T>(cfg, payload, size, true);
    fgpu::ReconInput args;
    args.plan = &streams.plan;
    args.sizeT = sizeof(T);
    args.streams.assign(streams.codes.begin(), streams.codes.end());
    args.anchors = streams.anchors.data();
    args.unpred = streams.unpred.data();
    args.numUnpred = streams.unpred.size();
    args.escapeCountValidated = true;
    fgpu::Timing tm;
    fgpu::reconstruct(args, out, recon_ms ? &tm : nullptr);
    if (recon_ms) *recon_ms = tm.recon;
    return true;
}

template<class T>
Falcons::Config encode_config(const Options& opts) {
    if (opts.dims.empty() || opts.dims.size() > 4 || opts.bound <= 0.)
        throw std::invalid_argument("encoding requires 1D-4D --shape and --abs or --rel");
    std::size_t n = 1;
    for (std::size_t dim : opts.dims) {
        if (dim < 2 || dim > std::numeric_limits<std::size_t>::max() / sizeof(T) / n)
            throw std::invalid_argument("invalid dimensions or field size overflow");
        n *= dim;
    }
    Falcons::Config cfg;
    cfg.setDims(opts.dims.begin(), opts.dims.end());
    cfg.cmprAlgo = Falcons::ALGO_INTERP;
    cfg.dataType = data_type<T>;
    cfg.openmp = opts.streams > 1;
    cfg.interpAlgo = opts.interp;
    cfg.errorBoundMode = opts.relative ? Falcons::EB_REL : Falcons::EB_ABS;
    if (opts.relative) cfg.relErrorBound = opts.bound;
    else cfg.absErrorBound = opts.bound;
    return cfg;
}

template<class T>
std::vector<char> compress(const Falcons::Config& cfg, const T* data, const Options& opts) {
    if (opts.mode == "hybrid") return fgpu::compress_current_hybrid<T>(cfg, data, opts.streams);
    fgpu::PlanParams pp;
    const auto conf = fgpu::prepare_current_compress<T>(cfg, data, opts.streams, pp);
    const auto plan = fgpu::build_plan(pp);
    fgpu::PureCompressInput in;
    in.plan = &plan;
    in.sizeT = sizeof(T);
    in.serial = !conf.openmp;
    in.header = fgpu::current_interp_header(conf, opts.streams);
    fgpu::PureCompressOutput out;
    fgpu::compress_pure(in, data, out);
    if (!out.ok) throw std::runtime_error("pure HIP compressor declined this layout");
    return fgpu::join_current_blocks(conf, pp, out.blocks, opts.streams);
}

template<class T>
void encode_file(const std::vector<char>& raw, const char* out, const Options& opts) {
    const auto cfg = encode_config<T>(opts);
    if (raw.size() != cfg.num * sizeof(T)) throw std::invalid_argument("input size does not match --shape and --dtype");
    std::vector<T> data(cfg.num);
    std::memcpy(data.data(), raw.data(), raw.size());
    std::vector<char> cmp;
    const auto t = measure(opts.bench, [&] {
        cmp = compress<T>(cfg, data.data(), opts);
        if (cmp.empty()) throw std::runtime_error("DCU compressor declined this input");
        return 0.;
    });
    write_file(out, cmp.data(), cmp.size());
    std::cout << "hip-" << opts.mode << " encode: " << raw.size() << " -> " << cmp.size() << " bytes, " << t.total << " ms\n";
}

template<class T>
void decode_file(const std::vector<char>& raw, const char* out, const Options& opts, const Falcons::Config& cfg) {
    std::vector<T> data(cfg.num);
    const auto t = measure(opts.bench, [&] {
        double recon = 0;
        if (!reconstruct(raw, cfg, data.data(), opts.mode == "pure", opts.bench ? &recon : nullptr))
            throw std::runtime_error("pure HIP decode declined this entropy or Zstd layout; try --mode hybrid");
        return recon;
    });
    write_file(out, data.data(), data.size() * sizeof(T));
    std::cout << "hip-" << opts.mode << " decode: " << data.size() << " values, " << t.total << " ms to host";
    if (opts.bench) std::cout << ", " << t.recon << " ms reconstruction";
    std::cout << '\n';
}

std::size_t element_bytes(int dtype) {
    switch (dtype) {
        case Falcons_INT8: case Falcons_UINT8: return 1;
        case Falcons_INT16: case Falcons_UINT16: return 2;
        case Falcons_INT32: case Falcons_UINT32: case Falcons_FLOAT: return 4;
        case Falcons_INT64: case Falcons_UINT64: case Falcons_DOUBLE: return 8;
        default: throw std::invalid_argument("unknown Falcons data type");
    }
}

void decode_lossless(const std::vector<char>& raw, const char* out, const Falcons::Config& cfg, const Options& opts) {
    if (cfg.falconsDataVer != versionInt("1.1.0"))
        throw std::invalid_argument("expected a Falcons 1.1.0 lossless-only stream");
    const auto width = element_bytes(cfg.dataType);
    if (cfg.num > std::numeric_limits<std::size_t>::max() / width)
        throw std::length_error("output size overflow");
    const auto header = static_cast<unsigned char>(raw[0]);
    fgpu::LosslessInput in{reinterpret_cast<const unsigned char*>(raw.data()) + header,
                           raw.size() - header, cfg.num * width};
    std::vector<unsigned char> data(in.expectedBytes);
    const auto t = measure(opts.bench, [&] {
        if (opts.mode == "pure") {
            if (!fgpu::decode_lossless_current(in, data.data()))
                throw std::runtime_error("HIP lossless-only decoder declined this Zstd layout");
        } else {
            // Host-output hybrid: there is no interpolation to offload in a lossless-only stream.
            fgpu::current_detail::Reader r(in.cmp, in.cmpSize);
            if (r.read<size_t>() != data.size()) throw std::runtime_error("lossless size mismatch");
            fgpu::decode_zstd_frame(r.position(), r.remaining(), data.data(), data.size());
        }
        return 0.;
    });
    write_file(out, data.data(), data.size());
    std::cout << (opts.mode == "pure" ? "hip-zstd" : "cpu-zstd") << " lossless decode: " << data.size()
              << " bytes, " << t.total << " ms to host\n";
}

template<class T>
void verify(std::vector<char> bytes, const std::vector<T>& first, double bound, bool pure) {
    Falcons::Config cfg;
    Falcons_read_config(cfg, bytes.data(), bytes.size());
    std::vector<T> cpu(first.size()), gpu(first.size());
    T* p = cpu.data();
    Falcons_decompress<T>(cfg, bytes.data(), bytes.size(), p, first.size());
    if (!reconstruct(bytes, cfg, gpu.data(), pure)) throw std::runtime_error("HIP decoder declined");
    if (std::memcmp(cpu.data(), gpu.data(), cpu.size() * sizeof(T)) != 0) {
        std::size_t i = 0;
        while (i < cpu.size() && std::memcmp(&cpu[i], &gpu[i], sizeof(T)) == 0) ++i;
        throw std::runtime_error("CPU/DCU reconstruction differs at index " + std::to_string(i));
    }
    for (std::size_t i = 0; i < first.size(); ++i)
        if (!(std::abs(static_cast<double>(first[i]) - static_cast<double>(gpu[i])) <= bound))
            throw std::runtime_error("absolute bound exceeded at index " + std::to_string(i));
}

template<class T>
void verify_case(const Options& opts, bool encode_on_dcu) {
    auto cfg = encode_config<T>(opts);
    std::vector<T> data(cfg.num);
    for (std::size_t i = 0; i < data.size(); ++i) {
        const double x = double(i % opts.dims.back()) / opts.dims.back();
        const double y = static_cast<double>(i / opts.dims.back()) / cfg.num;
        const double noise = (static_cast<int>((i * 2654435761u) % 1009u) - 504) * 1.e-6;
        data[i] = static_cast<T>(2.0 + 0.5 * std::sin(6.0 * x) + 0.2 * std::cos(13.0 * y) + noise);
    }
    if (encode_on_dcu) {
        auto cmp = compress<T>(cfg, data.data(), opts);
        if (cmp.empty()) throw std::runtime_error("DCU compressor declined test field");
        verify(std::move(cmp), data, cfg.absErrorBound, opts.mode == "pure");
    } else {
        std::vector<T> work = data;
        std::size_t size = 0;
        std::unique_ptr<char[]> cmp(Falcons_compress<T>(cfg, work.data(), size));
        verify(std::vector<char>(cmp.get(), cmp.get() + size), data, cfg.absErrorBound, opts.mode == "pure");
    }
}

bool self_test(const std::string& filter, bool pure) {
    const std::vector<std::vector<std::size_t>> shapes{{4097}, {129, 257}, {17, 49, 65}, {7, 9, 17, 33}};
    bool ok = true;
    std::size_t ran = 0;
    for (const auto& shape : shapes)
        for (int streams : {1, 4})
            for (int interp : {Falcons::INTERP_ALGO_LINEAR, Falcons::INTERP_ALGO_CUBIC})
                for (bool f64 : {false, true})
                    for (bool encode : {false, true}) {
                        Options opts;
                        opts.mode = pure ? "pure" : "hybrid";
                        opts.dims = shape;
                        opts.streams = streams;
                        opts.interp = interp;
                        opts.bound = 1e-3;
                        const auto name = std::to_string(shape.size()) + "d/" + (f64 ? "f64" : "f32") + "/" +
                            (streams == 1 ? "serial" : "omp") + "/" +
                            (interp == Falcons::INTERP_ALGO_LINEAR ? "linear" : "cubic") + "/" +
                            (encode ? "hip-encode" : "cpu-encode");
                        if (name.find(filter) == std::string::npos) continue;
                        ++ran;
                        try {
#ifdef _OPENMP
                            omp_set_dynamic(0);
                            omp_set_num_threads(streams);
#endif
                            if (f64) verify_case<double>(opts, encode);
                            else verify_case<float>(opts, encode);
                            std::cout << "PASS " << name << std::endl;
                        } catch (const std::exception& e) {
                            ok = false;
                            std::cerr << "FAIL " << name << ": " << e.what() << std::endl;
                        }
                    }
    if (!ran) throw std::invalid_argument("test filter matched no cases");
    return ok;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc >= 2 && argc <= 3 && (std::string_view(argv[1]) == "--self-test" ||
                                  std::string_view(argv[1]) == "--self-test-pure")) {
        require_device();
        return self_test(argc == 3 ? argv[2] : "", std::string_view(argv[1]) == "--self-test-pure") ? 0 : 1;
    }
    if (argc < 4 || (std::string_view(argv[1]) != "encode" && std::string_view(argv[1]) != "decode")) {
        std::cerr << "usage: hipfalcons_codec decode input.falcons output.raw [--mode hybrid|pure] [--bench]\n"
                     "       hipfalcons_codec encode input.raw output.falcons --dtype f32|f64"
                     " --shape H W --abs E|--rel E [--streams N] [--linear|--cubic] [--mode hybrid|pure] [--bench]\n"
                     "       hipfalcons_codec --self-test|--self-test-pure [filter]\n";
        return 2;
    }
    const auto opts = options(argc, argv, 4);
    if (opts.bench) std::cout << "benchmark: 1 warmup, median of 3, excluding file I/O\n";
    const auto bytes = read_file(argv[2]);
    if (std::string_view(argv[1]) == "encode") {
        require_device();
        if (opts.dtype == "f32") encode_file<float>(bytes, argv[3], opts);
        else encode_file<double>(bytes, argv[3], opts);
    } else {
        Falcons::Config cfg;
        Falcons_read_config(cfg, bytes.data(), bytes.size());
        if (cfg.cmprAlgo != Falcons::ALGO_LOSSLESS || opts.mode == "pure") require_device();
        if (cfg.cmprAlgo == Falcons::ALGO_LOSSLESS) decode_lossless(bytes, argv[3], cfg, opts);
        else if (cfg.dataType == Falcons_FLOAT) decode_file<float>(bytes, argv[3], opts, cfg);
        else if (cfg.dataType == Falcons_DOUBLE) decode_file<double>(bytes, argv[3], opts, cfg);
        else throw std::invalid_argument("non-floating-point codes are not supported by this HIP codec");
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "hipfalcons_codec: " << e.what() << '\n';
    return 1;
}
