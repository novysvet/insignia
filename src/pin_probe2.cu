// Probe: does the locked-memory ceiling apply per-allocation or globally?
// Allocates cumulative 1-GiB pinned blocks until failure.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

int main() {
    std::vector<void *> blocks;
    size_t total = 0;
    for (;;) {
        void *p = nullptr;
        const cudaError_t st = cudaHostAlloc(&p, size_t(1) << 30, cudaHostAllocDefault);
        if (st != cudaSuccess) { cudaGetLastError(); break; }
        volatile unsigned char *b = (volatile unsigned char *)p;
        for (size_t off = 0; off < (size_t(1) << 30); off += 4096) b[off] = 1;
        blocks.push_back(p);
        total += 1;
        std::printf("%zu GiB cumulative OK\n", total);
        std::fflush(stdout);
    }
    std::printf("global pinned ceiling: %zu GiB (in 1-GiB blocks)\n", total);
    for (void *p : blocks) cudaFreeHost(p);
    return 0;
}
