// HIP port by OpenAI, 2026, of Falcons b564b45. See notices/ for upstream licenses.
// Build with floating-point contraction disabled (-ffp-contract=off); explicit fma calls remain.
#include "fgpu_recon.h"

#include <hip/hip_runtime.h>
#include <zstd.h>
#define FSE_STATIC_LINKING_ONLY
#include "Falcons/encoder/zfse/fse.h"
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <chrono>
#include <map>
#include <mutex>
#include <unordered_map>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <algorithm>

namespace fgpu {

#define CK(x)                                                                          \
    do {                                                                               \
        hipError_t e_ = (x);                                                          \
        if (e_ != hipSuccess) throw std::runtime_error(std::string(#x " : ") + hipGetErrorString(e_)); \
    } while (0)

template <class T> struct Bits;
template <> struct Bits<float> { using U = uint32_t; };
template <> struct Bits<double> { using U = uint64_t; };

template <class T> __device__ __forceinline__ T lin(T a, T b) { return (a + b) * T(0.5); }
template <class T> __device__ __forceinline__ T lin1(T a, T b) { return fma(-T(0.5), a, T(1.5) * b); }
template <class T> __device__ __forceinline__ T quad1(T a, T b, T c) {
    T t = fma(T(6), b, -c);
    t = fma(T(3), a, t);
    return t * T(0.125);
}
template <class T> __device__ __forceinline__ T quad2(T a, T b, T c) {
    T t = fma(T(6), b, -a);
    t = fma(T(3), c, t);
    return t * T(0.125);
}
template <class T> __device__ __forceinline__ T cubic(T a, T b, T c, T d) {
    T t = (b + c) * T(9);
    t = t - a;
    t = t - d;
    return t * T(0.0625);
}

__device__ __forceinline__ void resolve(const FgNest &n, uint32_t off, uint64_t &d, uint64_t &p) {
    uint32_t r = off;
    const uint32_t i3 = r % n.cnt[3];
    r /= n.cnt[3];
    const uint32_t i2 = r % n.cnt[2];
    r /= n.cnt[2];
    const uint32_t i1 = r % n.cnt[1];
    const uint32_t i0 = r / n.cnt[1];
    d = n.data0 + i0 * n.dstr[0] + i1 * n.dstr[1] + i2 * n.dstr[2] + i3 * n.dstr[3];
    p = n.pos0 + i0 * n.pstr[0] + i1 * n.pstr[1] + i2 * n.pstr[2] + i3 * n.pstr[3];
}

template <class T>
__global__ void recon_kernel(T *__restrict__ data, const int16_t *__restrict__ stream, const FgNest *__restrict__ nests,
                             const FgWork *__restrict__ work, const uint64_t *__restrict__ escPos,
                             const typename Bits<T>::U *__restrict__ escVal, uint64_t numEsc,
                             const typename Bits<T>::U *__restrict__ anchors, double ebx2) {
    using U = typename Bits<T>::U;
    const FgWork w = work[blockIdx.x];
    const uint32_t off = w.start + threadIdx.x;
    const FgNest &n = nests[w.nest];
    if (off >= n.points) return;
    uint64_t d, p;
    resolve(n, off, d, p);

    if (n.op == FG_ANCHOR) {
        reinterpret_cast<U *>(data)[d] = anchors[p];
        return;
    }

    const int16_t q = stream[p];
    if (q == FG_ESCAPE) {
        uint64_t lo = 0, hi = numEsc;
        while (lo < hi) {
            uint64_t mid = (lo + hi) >> 1;
            if (escPos[mid] < p) lo = mid + 1;
            else hi = mid;
        }
        reinterpret_cast<U *>(data)[d] = escVal[lo];
        return;
    }

    const T *x = data + d;
    T pred;
    switch (n.op) {
        case FG_EQUAL: pred = x[n.opnd[0]]; break;
        case FG_LINEAR: pred = lin(x[n.opnd[0]], x[n.opnd[1]]); break;
        case FG_LINEAR1: pred = lin1(x[n.opnd[0]], x[n.opnd[1]]); break;
        case FG_QUAD1: pred = quad1(x[n.opnd[0]], x[n.opnd[1]], x[n.opnd[2]]); break;
        case FG_QUAD2: pred = quad2(x[n.opnd[0]], x[n.opnd[1]], x[n.opnd[2]]); break;
        case FG_CUBIC: pred = cubic(x[n.opnd[0]], x[n.opnd[1]], x[n.opnd[2]], x[n.opnd[3]]); break;
        default: pred = T(0); break;
    }
    const double qd = q;
    data[d] = n.fused ? static_cast<T>(fma(qd, ebx2, static_cast<double>(pred)))
                        : static_cast<T>(static_cast<double>(pred) + qd * ebx2);
}

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

// Freed device blocks are kept for reuse: hipMalloc costs milliseconds at these sizes.
class Pool {
   public:
    void *get(size_t bytes) {
        std::lock_guard<std::mutex> lk(m_);
        auto it = free_.lower_bound(bytes);
        if (it != free_.end() && it->first <= 2 * bytes + 4096) {
            void *p = it->second;
            size_[p] = it->first;
            free_.erase(it);
            return p;
        }
        void *p;
        hipError_t e = hipMalloc(&p, bytes);
        if (e == hipErrorOutOfMemory) {   // make room and retry
            trimLocked();
            e = hipMalloc(&p, bytes);
        }
        CK(e);
        size_[p] = bytes;
        return p;
    }
    void put(void *p) {
        std::lock_guard<std::mutex> lk(m_);
        free_.emplace(size_[p], p);
    }
    void forget(void *p) {
        std::lock_guard<std::mutex> lk(m_);
        size_.erase(p);
    }
    void trim() {
        std::lock_guard<std::mutex> lk(m_);
        trimLocked();
    }

   private:
    void trimLocked() {
        for (auto &kv : free_) {
            size_.erase(kv.second);
            hipFree(kv.second);
        }
        free_.clear();
    }
    std::mutex m_;
    std::multimap<size_t, void *> free_;
    std::unordered_map<void *, size_t> size_;
};

Pool &pool() {
    static Pool *p = new Pool;
    return *p;
}

template <class D> struct Dev {
    D *p = nullptr;
    explicit Dev(size_t n) { p = static_cast<D *>(pool().get(std::max<size_t>(n, 1) * sizeof(D))); }
    ~Dev() {
        if (p) pool().put(p);
    }
    void reset() {
        if (p) pool().put(p);
        p = nullptr;
    }
    D *release() {
        pool().forget(p);
        D *r = p;
        p = nullptr;
        return r;
    }
    Dev(const Dev &) = delete;
    Dev &operator=(const Dev &) = delete;
};

struct IsEsc {
    __host__ __device__ bool operator()(int16_t q) const { return q == FG_ESCAPE; }
};

struct WidenClass {
    __host__ __device__ uint64_t operator()(uint8_t x) const { return uint64_t(x); }
};

// Escape values are already concatenated in code-stream order. Only their positions need a scan.
void locate_escapes(const Plan &P, const int16_t *stream, uint64_t numUnpred,
                    Dev<uint64_t> &escPos, bool countValidated) {
    if (countValidated && !numUnpred) return;
    uint64_t nstream = 0;
    for (auto l : P.streamLen) nstream += l;
    auto codes = thrust::device_pointer_cast(const_cast<int16_t *>(stream));
    if (!countValidated && static_cast<uint64_t>(thrust::count_if(codes, codes + nstream, IsEsc())) != numUnpred)
        throw std::runtime_error("escape count does not match the outlier count");
    if (numUnpred) {
        auto first = thrust::counting_iterator<uint64_t>(0);
        auto outPos = thrust::device_pointer_cast(escPos.p);
        thrust::copy_if(first, first + nstream, codes, outPos, IsEsc());
    }
}

template <class T>
void run_phases(const Plan &P, T *data, const int16_t *stream, const FgNest *nests, const FgWork *work, const uint64_t *escPos,
                const typename Bits<T>::U *escVal, uint64_t numEsc, const typename Bits<T>::U *anchors) {
    const bool timed = getenv("FGPU_TIMES") != nullptr;
    for (const auto &ph : P.phases) {
        auto a = Clock::now();
        recon_kernel<T><<<static_cast<int>(ph.workN), FG_BLOCK>>>(data, stream, nests, work + ph.work0, escPos, escVal, numEsc,
                                                                  anchors, ph.ebx2);
        CK(hipGetLastError());
        if (timed) {
            CK(hipDeviceSynchronize());
            const double t = ms(a, Clock::now());
            printf("  phase ops %.2f M, %u nests, %.3f ms, %.0f M/s\n", ph.points / 1e6, ph.nestN, t, ph.points / 1e3 / t);
        }
    }
    CK(hipDeviceSynchronize());
}

template <class T>
void run(const ReconInput &in, void *out, Timing *tm, void **device) {
    using U = typename Bits<T>::U;
    const Plan &P = *in.plan;
    uint64_t nstream = 0;
    for (auto l : P.streamLen) nstream += l;
    uint64_t nanchor = 0;
    for (auto l : P.anchorLen) nanchor += l;

    auto t0 = Clock::now();
    Dev<T> data(P.num);
    Dev<int16_t> stream(nstream);
    Dev<FgNest> nests(P.nests.size());
    Dev<FgWork> work(P.work.size());
    Dev<U> anchors(nanchor), unpred(in.numUnpred);
    uint64_t b = 0;
    for (size_t i = 0; i < in.streams.size(); ++i) {
        CK(hipMemcpy(stream.p + b, in.streams[i], P.streamLen[i] * sizeof(int16_t), hipMemcpyHostToDevice));
        b += P.streamLen[i];
    }
    CK(hipMemcpy(nests.p, P.nests.data(), P.nests.size() * sizeof(FgNest), hipMemcpyHostToDevice));
    CK(hipMemcpy(work.p, P.work.data(), P.work.size() * sizeof(FgWork), hipMemcpyHostToDevice));
    if (nanchor) CK(hipMemcpy(anchors.p, in.anchors, nanchor * sizeof(U), hipMemcpyHostToDevice));
    if (in.numUnpred) CK(hipMemcpy(unpred.p, in.unpred, in.numUnpred * sizeof(U), hipMemcpyHostToDevice));
    auto t1 = Clock::now();

    Dev<uint64_t> escPos(in.numUnpred);
    locate_escapes(P, stream.p, in.numUnpred, escPos, in.escapeCountValidated);
    auto t2 = Clock::now();

    run_phases<T>(P, data.p, stream.p, nests.p, work.p, escPos.p, unpred.p, in.numUnpred, anchors.p);
    auto t3 = Clock::now();
    if (out) CK(hipMemcpy(out, data.p, P.num * sizeof(T), hipMemcpyDeviceToHost));
    auto t4 = Clock::now();
    if (device) *device = data.release();
    if (tm) {
        tm->upload = ms(t0, t1);
        tm->escape = ms(t1, t2);
        tm->recon = ms(t2, t3);
        tm->download = ms(t3, t4);
    }
}
}  // namespace

void reconstruct(const ReconInput &in, void *out, Timing *t, void **device) {
    if (in.sizeT == 4) run<float>(in, out, t, device);
    else run<double>(in, out, t, device);
}

void *upload(const void *src, size_t bytes) {
    void *p = nullptr;
    CK(hipMalloc(&p, bytes));
    CK(hipMemcpy(p, src, bytes, hipMemcpyHostToDevice));
    return p;
}

void free_device(void *p) { hipFree(p); }

void trim_pool() { pool().trim(); }

void download(void *dst, const void *device, size_t bytes) { CK(hipMemcpy(dst, device, bytes, hipMemcpyDeviceToHost)); }

#include "fgpu_pure.inl"
#include "fgpu_compress.inl"

}  // namespace fgpu

#include "fgpu_lossless.inl"
