// HIP byte-split FSE decoder by OpenAI, 2026, for the Falcons 1.1.0 format.
// See notices/ for the upstream licenses and attribution.

struct SplitJob {
    uint64_t src = 0, dst = 0, local = 0;
    uint32_t stream = 0, count = 0, plane = 0, mode = 0, value = 0;
    uint32_t tableLog = 0, maxSymbol = 0;
    uint32_t chainOffset[4]{}, chainLength[4]{};
    int16_t norm[256]{};
};

struct SplitBits {
    const uint8_t *data;
    uint32_t pos, available;
    uint64_t bits;
    bool ok;

    __device__ void init(const uint8_t *src, uint32_t size) {
        data = src;
        pos = size ? size - 1 : 0;
        const uint32_t last = size ? src[pos] : 0;
        ok = last != 0;
        available = last ? 31 - __clz(last) : 0;
        bits = last & ((1u << available) - 1);
    }
    __device__ uint32_t read(uint32_t n) {
        while (available < n && pos) {
            bits = (bits << 8) | data[--pos];
            available += 8;
        }
        if (available < n) {
            ok = false;
            return 0;
        }
        available -= n;
        return static_cast<uint32_t>((bits >> available) & ((1u << n) - 1));
    }
};

__global__ void split_decode_kernel(const uint8_t *__restrict__ raw, const SplitJob *__restrict__ jobs,
                                    uint8_t *__restrict__ bytes, uint32_t *__restrict__ bad) {
    __shared__ uint32_t table[1u << 11];
    const uint32_t id = blockIdx.x;
    const SplitJob &job = jobs[id];
    if (job.mode == 0 || job.mode == 2) {
        for (uint32_t i = threadIdx.x; i < job.count; i += blockDim.x)
            bytes[job.dst + i] = job.mode == 0 ? raw[job.src + i] : static_cast<uint8_t>(job.value);
        return;
    }

    if (threadIdx.x == 0) fse_build_table(table, job.norm, job.maxSymbol, job.tableLog);
    __syncthreads();
    const uint32_t lane = threadIdx.x;
    if (lane >= 4) return;
    SplitBits bits;
    bits.init(raw + job.src + job.chainOffset[lane], job.chainLength[lane]);
    uint32_t state = bits.read(job.tableLog);
    const uint32_t quarter = job.count / 4, remainder = job.count % 4;
    const uint32_t count = quarter + (lane < remainder);
    const uint32_t first = lane * quarter + (lane < remainder ? lane : remainder);
    for (uint32_t i = 0; i < count && bits.ok; ++i) {
        if (state >= (1u << job.tableLog)) {
            bits.ok = false;
            break;
        }
        const uint32_t cell = table[state];
        bytes[job.dst + first + i] = static_cast<uint8_t>(cell >> 16);
        state = (cell & 0xffffu) + bits.read(cell >> 24);
    }
    if (!bits.ok || state != 0 || bits.pos != 0 || bits.available != 0) atomicOr(bad + id, 1u);
}

__global__ void split_join_kernel(const SplitJob *__restrict__ jobs, const uint8_t *__restrict__ bytes,
                                  uint64_t planeSize, int16_t *__restrict__ codes) {
    const SplitJob &job = jobs[blockIdx.x];
    if (job.plane != 0) return;
    for (uint32_t i = threadIdx.x; i < job.count; i += blockDim.x) {
        const uint64_t p = job.dst + i;
        const uint16_t zz = uint16_t(bytes[p]) | (uint16_t(bytes[planeSize + p]) << 8);
        const uint16_t value = static_cast<uint16_t>((zz >> 1) ^ (0u - (zz & 1u)));
        codes[p] = static_cast<int16_t>(value);
    }
}
