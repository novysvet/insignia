// 30-line probe: kernel reads from cudaHostRegister'd (VirtualAlloc) and cudaHostAlloc'd
// memory via plain loads and __ldcs. Decides the v1 zero-copy decode design.
#include <cuda_runtime.h>
#include <cstdio>
#include <windows.h>

__global__ void read_plain(const float4 *p, float *out) {
    const float4 v = p[threadIdx.x];
    out[threadIdx.x] = v.x + v.y + v.z + v.w;
}
__global__ void read_ldcs(const uint4 *__restrict__ p, float *out) {
    const uint4 v = __ldcs(p + threadIdx.x);
    out[threadIdx.x] = float(v.x ^ v.y ^ v.z ^ v.w);
}

int main() {
    const size_t N = 256 * 16;   // 4KB of float4/uint4
    float *out; cudaMalloc(&out, N * 4);
    // 1) VirtualAlloc + cudaHostRegister
    void *reg = VirtualAlloc(nullptr, N * 16, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    printf("VirtualAlloc %p\n", reg);
    for (int i = 0; i < int(N * 4); i++) static_cast<float *>(reg)[i] = 1.f;
    cudaError_t e1 = cudaHostRegister(reg, N * 16, cudaHostRegisterDefault);
    printf("cudaHostRegister: %s\n", cudaGetErrorString(e1));
    // 2) cudaHostAlloc
    float *alloc_p = nullptr;
    cudaError_t e2 = cudaHostAlloc((void **)&alloc_p, N * 16, cudaHostAllocDefault);
    printf("cudaHostAlloc: %s -> %p\n", cudaGetErrorString(e2), alloc_p);
    for (int i = 0; i < int(N * 4); i++) alloc_p[i] = 2.f;

    auto probe = [&](const char *name, const void *p) {
        read_plain<<<1, 256>>>(static_cast<const float4 *>(p), out);
        cudaError_t e = cudaDeviceSynchronize();
        printf("%s plain: %s\n", name, cudaGetErrorString(e));
        if (e == cudaSuccess) {
            read_ldcs<<<1, 256>>>(static_cast<const uint4 *>(p), out);
            e = cudaDeviceSynchronize();
            printf("%s __ldcs: %s\n", name, cudaGetErrorString(e));
        }
    };
    probe("registered", reg);
    probe("hostAlloc", alloc_p);
    return 0;
}
