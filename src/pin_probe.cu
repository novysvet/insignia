// Probe: how much pinned host memory will cudaHostAlloc grant on this WSL2?
// Usage: pinprobe [start_mib] [step_mib]  (default: scan 1024..13312 in 256 steps)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int main(int argc, char **argv) {
    const size_t lo = argc > 1 ? size_t(std::atol(argv[1])) : 1024;
    const size_t hi = argc > 2 ? size_t(std::atol(argv[2])) : 13312;
    const size_t step = argc > 3 ? size_t(std::atol(argv[3])) : 256;
    size_t best = 0;
    for (size_t mib = lo; mib <= hi; mib += step) {
        void *p = nullptr;
        const cudaError_t st = cudaHostAlloc(&p, mib << 20, cudaHostAllocDefault);
        if (st != cudaSuccess) {
            cudaGetLastError();
            std::printf("%6zu MiB  FAIL\n", mib);
            continue;
        }
        // Touch every page so the pin is real, then free.
        volatile unsigned char *b = (volatile unsigned char *)p;
        for (size_t off = 0; off < (mib << 20); off += 4096) b[off] = 1;
        std::printf("%6zu MiB  OK\n", mib);
        std::fflush(stdout);
        best = mib;
        cudaFreeHost(p);
    }
    std::printf("max single-block pinned: %zu MiB\n", best);
    return 0;
}
