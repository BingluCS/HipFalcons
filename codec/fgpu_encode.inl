// HIP port by OpenAI, 2026, of Falcons b564b45. See notices/ for upstream licenses.
// Pure-GPU entropy encoding: RLE + FSE and order-1 tANS. The result is a valid Falcons stream, with
// the zstd layer written as raw blocks. Included by fgpu_compress.inl.

#include "fgpu_fse_enc.inl"

struct StreamStat {
    uint64_t z, zd, cnt, cntd;
};

__global__ void stat_kernel(const int16_t *__restrict__ codes, const uint64_t *__restrict__ base, const uint64_t *__restrict__ len, StreamStat *__restrict__ st) {
    const uint32_t t = blockIdx.x;
    const int16_t *b = codes + base[t];
    const uint64_t n = len[t];
    uint64_t NW = 256, WL = 32;
    if (n < NW * WL) {
        WL = n / NW;
        if (!WL) WL = 1;
    }
    uint64_t step = n > NW * WL ? (n - WL) / NW : WL;
    if (!step) step = 1;
    StreamStat s{};
    for (uint64_t w = 0; w < NW; ++w) {
        const uint64_t off = w * step;
        if (off + WL > n) break;
        int16_t prev = b[off];
        for (uint64_t i = 0; i < WL; ++i) {
            const int16_t x = b[off + i];
            s.z += x == 0;
            s.zd += i > 0 && x == prev;
            prev = x;
        }
        s.cnt += WL;
        s.cntd += WL - 1;
    }
    st[t] = s;
}

__global__ void delta_kernel(const int16_t *__restrict__ in, int16_t *__restrict__ out, uint64_t n) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = i ? static_cast<int16_t>(in[i] - in[i - 1]) : in[0];
}

struct NonZero {
    __host__ __device__ bool operator()(int16_t q) const { return q != 0; }
};

__device__ __forceinline__ uint32_t hb64(uint64_t v) { return 63 - __clzll(v); }

// ORs k bits of v at MSB-first bit offset b into a byte stream stored in little-endian words.
__device__ __forceinline__ void put_msb(uint32_t *words, uint64_t b, uint64_t v, uint32_t k) {
    while (k) {
        const uint32_t bit = static_cast<uint32_t>(b & 7);
        const uint32_t n = k < 8 - bit ? k : 8 - bit;
        const uint32_t part = static_cast<uint32_t>((v >> (k - n)) & ((1u << n) - 1)) << (8 - bit - n);
        const uint64_t byte = b >> 3;
        if (part) atomicOr(&words[byte >> 2], part << (8 * (byte & 3)));
        b += n;
        k -= n;
    }
}

__global__ void run_class_kernel(const uint64_t *__restrict__ pos, uint8_t *__restrict__ cl, uint64_t *__restrict__ val, uint64_t nv, uint64_t nr, uint64_t n) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nr) return;
    const uint64_t cur = i < nv ? pos[i] : n, prev = i ? pos[i - 1] + 1 : 0;
    const uint64_t v = cur - prev + 1;   // L + 1
    val[i] = v;
    cl[i] = static_cast<uint8_t>(hb64(v));
}

__global__ void value_class_kernel(const uint64_t *__restrict__ pos, const int16_t *__restrict__ codes, uint8_t *__restrict__ cl, uint64_t *__restrict__ val, uint64_t nv) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nv) return;
    const int32_t fct = codes[pos[i]];
    const uint32_t zz = (static_cast<uint32_t>(fct) << 1) ^ static_cast<uint32_t>(fct >> 31);
    const uint64_t vv = static_cast<uint64_t>(zz) + 1;
    val[i] = vv;
    cl[i] = static_cast<uint8_t>(hb64(vv));
}

__global__ void pack_kernel(const uint8_t *__restrict__ cl, const uint64_t *__restrict__ val, const uint64_t *__restrict__ bits, uint32_t *words, uint64_t count) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t k = cl[i];
    put_msb(words, bits[i], val[i] & ((1ull << k) - 1), k);
}

struct EncBlk {
    uint64_t src;
    uint32_t len, result;
};

constexpr uint32_t FO_CAP = FSE_FBLK + 1024;

__global__ void fse_enc_kernel(const uint8_t *__restrict__ cls, EncBlk *__restrict__ blks, uint8_t *__restrict__ out) {
    alignas(8) __shared__ uint8_t scratch[FE_SCRATCH];
    const uint32_t id = blockIdx.x;
    const EncBlk b = blks[id];
    blks[id].result = fe_compress(out + uint64_t(id) * FO_CAP, FO_CAP, cls + b.src, b.len, scratch);
}

// ---- order-1 tANS ----

constexpr uint64_t O1_SEG_TARGET = uint64_t(8) << 20;
constexpr uint32_t CT_U16 = O1_CELLS, CT_BYTES = CT_U16 * 2 + 256 * sizeof(SymTT) + O1_CELLS;

struct EncSeg {
    uint64_t code;     // first code in the working array
    uint64_t N;
    uint32_t K, pad;
};

__device__ __forceinline__ uint32_t o1_sym(int16_t code) {
    const int32_t c = code;
    const uint32_t zz = (static_cast<uint32_t>(c) << 1) ^ static_cast<uint32_t>(c >> 31);
    return zz < O1_NSYM - 1 ? zz : O1_NSYM - 1;
}

// Each tile counts an independent subset. Chain starts and sampled windows match the CPU model.
__global__ void o1_hist_kernel(const int16_t *__restrict__ codes, const EncSeg *__restrict__ segs, uint32_t *__restrict__ cnt, const uint8_t *__restrict__ ctx) {
    const uint32_t g = blockIdx.x;
    const EncSeg s = segs[g];
    uint32_t *c = cnt + uint64_t(g) * O1_NCTX * O1_NSYM;
    __shared__ uint32_t hist[O1_NCTX * O1_NSYM];
    const bool sampled = s.N >= (1u << 20);
    for (uint32_t i = threadIdx.x; i < O1_NCTX * O1_NSYM; i += blockDim.x)
        hist[i] = sampled && blockIdx.y == 0 ? 1 : 0;
    __syncthreads();
    const int16_t *bins = codes + s.code;
    const uint64_t cs = (s.N + s.K - 1) / s.K;
    const uint64_t step = s.N / 2048 > 512 ? s.N / 2048 : 512;
    const uint64_t limit = sampled ? ((s.N - 1) / step + 1) * 512 : s.N;
    for (uint64_t j = blockIdx.y * blockDim.x + threadIdx.x; j < limit; j += gridDim.y * blockDim.x) {
        const uint64_t i = sampled ? (j / 512) * step + j % 512 : j;
        if (i >= s.N) continue;
        const uint16_t prev = i && (sampled || i % cs != 0) ? static_cast<uint16_t>(bins[i - 1]) : 0;
        atomicAdd(&hist[uint32_t(ctx[prev]) * O1_NSYM + o1_sym(bins[i])], 1u);
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < O1_NCTX * O1_NSYM; i += blockDim.x)
        if (hist[i]) atomicAdd(c + i, hist[i]);
}

__global__ void o1_ctable_kernel(const uint32_t *__restrict__ cnt, uint16_t *__restrict__ nf, uint8_t *__restrict__ ct) {
    const uint32_t id = blockIdx.x;   // segment * O1_NCTX + context
    const uint32_t *c = cnt + uint64_t(id) * O1_NSYM;
    uint16_t *f = nf + uint64_t(id) * O1_NSYM;
    uint64_t total = 0;
    int nonzero = 0;
    for (int s = 0; s < O1_NSYM; ++s) {
        total += c[s];
        nonzero += c[s] != 0;
    }
    int16_t norm[O1_NSYM];
    bool done = false;
    if (total == 0) {
        for (int s = 0; s < O1_NSYM; ++s) norm[s] = 0;
        norm[0] = O1_CELLS;
        done = true;
    } else if (nonzero > 1) {
        done = fe_normalize(norm, O1_PB, c, total, O1_NSYM - 1);
    }
    if (!done) {
        int sum = 0, maxf = 0, argmax = 0;
        for (int s = 0; s < O1_NSYM; ++s) {
            norm[s] = 0;
            if (c[s]) {
                int v = static_cast<int>(uint64_t(c[s]) * O1_CELLS / total);
                if (v == 0) v = 1;
                norm[s] = static_cast<int16_t>(v);
                sum += v;
                if (v > maxf) {
                    maxf = v;
                    argmax = s;
                }
            }
        }
        int e = int(O1_CELLS) - sum;
        if (e > 0) {
            norm[argmax] += static_cast<int16_t>(e);
        } else if (e < 0) {
            e = -e;
            int take = maxf - 1 < e ? maxf - 1 : e;
            norm[argmax] -= static_cast<int16_t>(take);
            e -= take;
            while (e > 0) {
                int mi = -1, mv = 1;
                for (int s = 0; s < O1_NSYM; ++s)
                    if (norm[s] > mv) {
                        mv = norm[s];
                        mi = s;
                    }
                if (mi < 0) break;
                take = norm[mi] - 1;
                if (take > e) take = e;
                norm[mi] -= static_cast<int16_t>(take);
                e -= take;
            }
        }
    }
    for (int s = 0; s < O1_NSYM; ++s) f[s] = static_cast<uint16_t>(norm[s]);
    uint8_t *t = ct + uint64_t(id) * CT_BYTES;
    fe_build_ctable(reinterpret_cast<uint16_t *>(t), reinterpret_cast<SymTT *>(t + CT_U16 * 2), t + CT_U16 * 2 + 256 * sizeof(SymTT), norm, O1_NSYM - 1, O1_PB);
}

struct EncChain {
    uint64_t lo, hi;      // code range of the chain in the working array
    uint64_t out;         // bitstream offset
    uint64_t outCap;      // bytes reserved for this chain
    uint64_t ct;          // first context table
    uint64_t ovf;         // overflow scratch offset
};

struct ChainResult {
    uint64_t blen, novf;
};

__global__ void o1_enc_kernel(const int16_t *__restrict__ codes, const EncChain *__restrict__ chains, const uint8_t *__restrict__ ct, const uint8_t *__restrict__ ctx, uint8_t *__restrict__ out, int16_t *__restrict__ ovf, ChainResult *__restrict__ res) {
    const uint32_t id = blockIdx.x;
    const EncChain c = chains[id];
    ChainResult r{0, 0};
    if (c.hi > c.lo) {
        const uint8_t *tables = ct + c.ct;
        auto st = [&](uint32_t x) { return reinterpret_cast<const uint16_t *>(tables + uint64_t(x) * CT_BYTES); };
        auto tt = [&](uint32_t x) { return reinterpret_cast<const SymTT *>(tables + uint64_t(x) * CT_BYTES + CT_U16 * 2); };
        FwdBits b;
        b.init(out + c.out, c.outCap);
        uint64_t idx = c.hi - 1;
        uint32_t cx = ctx[static_cast<uint16_t>(idx > c.lo ? codes[idx - 1] : 0)];
        uint64_t state = fe_init2(st(cx), tt(cx), o1_sym(codes[idx]));
        uint32_t since = 0;
        while (idx-- > c.lo) {
            cx = ctx[static_cast<uint16_t>(idx > c.lo ? codes[idx - 1] : 0)];
            fe_encode(b, state, st(cx), tt(cx), o1_sym(codes[idx]));
            if (++since == 4) {
                b.flush();
                since = 0;
            }
        }
        b.flush();
        b.add(state, O1_PB);
        b.flush();
        r.blen = b.close();
        int16_t *o = ovf + c.ovf;
        for (uint64_t i = c.lo; i < c.hi; ++i)
            if (o1_sym(codes[i]) == O1_NSYM - 1) o[r.novf++] = codes[i];
    }
    res[id] = r;
}

// ---- host side ----

struct EncodedStream {
    std::vector<uint8_t> bytes;
};

inline void put_u64(std::vector<uint8_t> &v, uint64_t x) {
    const auto *p = reinterpret_cast<const uint8_t *>(&x);
    v.insert(v.end(), p, p + 8);
}

// Same framing as RleFseEncoder::fse_enc, with the per-block FSE already done.
inline void append_fse(std::vector<uint8_t> &out, const std::vector<EncBlk> &blks, size_t first, size_t count, const uint8_t *fo, const uint8_t *cls, uint64_t clsDev) {
    (void)clsDev;
    for (size_t i = first; i < first + count; ++i) {
        const EncBlk &b = blks[i];
        const uint8_t *c = fo + i * FO_CAP;
        if (b.result == 0 || b.result >= b.len) {
            out.push_back(0);
            out.insert(out.end(), cls + b.src, cls + b.src + b.len);
        } else if (b.result == 1) {
            out.push_back(2);
            out.push_back(cls[b.src]);
        } else {
            out.push_back(1);
            const auto *p = reinterpret_cast<const uint8_t *>(&b.result);
            out.insert(out.end(), p, p + 4);
            out.insert(out.end(), c, c + b.result);
        }
    }
}

// encode_o1_ovf, run on the host: overflow codes are rare and already gathered.
inline void append_ovf(std::vector<uint8_t> &out, const int16_t *ovf, uint64_t n) {
    if (n == 0) {
        out.push_back(0);
        return;
    }
    std::vector<uint8_t> vcl(n), vman(2 * n + 16), fse;
    fse.reserve(n + n / FSE_FBLK * 8 + 64);
    uint64_t acc = 0, vbp = 0;
    int avail = 0;
    for (uint64_t i = 0; i < n; ++i) {
        const int32_t fct = ovf[i];
        const uint32_t zz = (static_cast<uint32_t>(fct) << 1) ^ static_cast<uint32_t>(fct >> 31);
        const uint64_t vv = static_cast<uint64_t>(zz) + 1;
        const int vk = 63 - __builtin_clzll(vv);
        vcl[i] = static_cast<uint8_t>(vk);
        acc = (acc << vk) | (vv & ((1ull << vk) - 1));
        avail += vk;
        while (avail >= 8) {
            avail -= 8;
            vman[vbp++] = static_cast<uint8_t>(acc >> avail);
        }
    }
    if (avail > 0) vman[vbp++] = static_cast<uint8_t>(acc << (8 - avail));
    std::vector<uint8_t> tmp(FSE_FBLK + 1024);
    for (uint64_t o = 0; o < n; o += FSE_FBLK) {
        const size_t len = std::min<uint64_t>(n - o, FSE_FBLK);
        const size_t r = FSE_compress(tmp.data(), tmp.size(), vcl.data() + o, len);
        if (FSE_isError(r) || r == 0 || r >= len) {
            fse.push_back(0);
            fse.insert(fse.end(), vcl.begin() + o, vcl.begin() + o + len);
        } else if (r == 1) {
            fse.push_back(2);
            fse.push_back(vcl[o]);
        } else {
            fse.push_back(1);
            const uint32_t rr = static_cast<uint32_t>(r);
            const auto *p = reinterpret_cast<const uint8_t *>(&rr);
            fse.insert(fse.end(), p, p + 4);
            fse.insert(fse.end(), tmp.begin(), tmp.begin() + r);
        }
    }
    if (fse.size() + vbp + 16 >= n * 2) {
        out.push_back(0);
        const auto *p = reinterpret_cast<const uint8_t *>(ovf);
        out.insert(out.end(), p, p + n * 2);
    } else {
        out.push_back(1);
        put_u64(out, fse.size());
        put_u64(out, vbp);
        out.insert(out.end(), fse.begin(), fse.end());
        out.insert(out.end(), vman.begin(), vman.begin() + vbp);
    }
}

// Stores `n` bytes as raw zstd blocks, after the size prefix Lossless_zstd writes.
inline void wrap_zstd_raw(std::vector<uint8_t> &out, const uint8_t *src, uint64_t n) {
    constexpr uint64_t BLOCK = 128 * 1024;
    size_t p = out.size();
    out.resize(p + 8 + 4 + 1 + 8 + 3 * (n / BLOCK + 1) + n);
    uint8_t *q = out.data() + p;
    std::memcpy(q, &n, 8);
    q += 8;
    const uint32_t magic = 0xFD2FB528u;
    std::memcpy(q, &magic, 4);
    q += 4;
    *q++ = 0xE0;   // single segment, 8-byte content size
    std::memcpy(q, &n, 8);
    q += 8;
    uint64_t o = 0;
    do {
        const uint32_t len = static_cast<uint32_t>(std::min(n - o, BLOCK));
        const bool last = o + len == n;
        const uint32_t bh = (len << 3) | (last ? 1 : 0);
        q[0] = bh & 0xFF;
        q[1] = (bh >> 8) & 0xFF;
        q[2] = (bh >> 16) & 0xFF;
        q += 3;
        std::memcpy(q, src + o, len);
        q += len;
        o += len;
    } while (o < n);
    out.resize(q - out.data());
}
