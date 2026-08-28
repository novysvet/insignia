// cuda13_probe.cu -- CUDA 13.x feature probe for sm_89 (RTX 4070 SUPER / WSL2 Arch).
// Tests: (1) capture of {memcpyAsync(pinned->dev)+kernel} loop + cudaGraphExecKernelNodeSetParams;
//        (2) conditional nodes (cudaGraphAddNode, IF on host-set device flag, CUDA 12.4+);
//        (3) device-side cudaGraphLaunch from a __global__ (CUDA 12.0+, needs -rdc=true);
//        (4) cudaEventRecord + cudaStreamWaitEvent captured INSIDE the graph, replay semantics;
//        (5) 400 trivial kernels/token: stream launches vs graph replay, 100 iterations.
// Build: /opt/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_89 -rdc=true cuda13_probe.cu -o cuda13_probe -lcudadevrt
// Exit code: 0 if every test RAN (PASS or FAIL printed); 1 only if no device.
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

#define CK(call) chk((call), #call, __LINE__)
static bool chk(cudaError_t e, const char* what, int line) {
    if (e != cudaSuccess) { printf("    CUDA error, line %d [%s]: %s\n", line, what, cudaGetErrorString(e)); return false; }
    return true;
}
static bool eq4(const int* a, const int* b) { return a[0]==b[0] && a[1]==b[1] && a[2]==b[2] && a[3]==b[3]; }
static void pr4(const int* a) { printf("[%d %d %d %d]", a[0], a[1], a[2], a[3]); }

__global__ void k_combine(const int* in, int* out, int v) { if (!threadIdx.x) *out = *in + v; }
__global__ void k_write  (int* out, int v)                { if (!threadIdx.x) *out = v; }
__global__ void k_mul2   (const int* in, int* out)        { if (!threadIdx.x) *out = *in * 2; }
__global__ void k_child  (int* out)                       { if (!threadIdx.x) out[1] = 88; }
__global__ void k_noop   ()                               {}

__global__ void k_parent(cudaGraphExec_t ge, int* out) {
    if (!threadIdx.x) { out[0] = 77; cudaGraphLaunch(ge, cudaStreamTailLaunch); }
}

static bool test1_exec_update() {
    printf("\n[1] capture loop {memcpyAsync(pinned->dev) + kernel}, then cudaGraphExecKernelNodeSetParams\n");
    cudaStream_t s = nullptr; int *h_in = nullptr, *d_in = nullptr, *d_out = nullptr;
    bool ok = CK(cudaStreamCreate(&s));
    ok &= CK(cudaHostAlloc((void**)&h_in, 4*sizeof(int), cudaHostAllocDefault));
    ok &= CK(cudaMalloc((void**)&d_in, 4*sizeof(int)));
    ok &= CK(cudaMalloc((void**)&d_out, 4*sizeof(int)));
    ok &= CK(cudaMemset(d_out, 0, 4*sizeof(int)));
    for (int i = 0; i < 4; i++) h_in[i] = (i+1)*10;
    if (!ok) return false;
    ok &= CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < 4; i++) {
        cudaMemcpyAsync(d_in + i, h_in + i, sizeof(int), cudaMemcpyHostToDevice, s);
        k_combine<<<1,1,0,s>>>(d_in + i, d_out + i, 0xAA);
    }
    cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
    ok &= CK(cudaStreamEndCapture(s, &g));
    ok &= CK(cudaGraphInstantiate(&ge, g, 0));
    ok &= CK(cudaGraphLaunch(ge, s));
    ok &= CK(cudaStreamSynchronize(s));
    int h_out[4] = {-1,-1,-1,-1}, exp[4];
    ok &= CK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 4; i++) exp[i] = (i+1)*10 + 0xAA;
    bool passA = ok && eq4(h_out, exp);
    printf("    replay #1 (v=0xAA, h_in=10..40): out="); pr4(h_out); printf(" exp="); pr4(exp);
    printf(" -> %s\n", passA ? "PASS" : "FAIL");
    for (int i = 0; i < 4; i++) h_in[i] = (i+1)*100;
    cudaGraphNode_t nodes[16]; size_t nn = 16;
    ok &= CK(cudaGraphGetNodes(g, nodes, &nn));
    int updated = 0;
    for (size_t i = 0; i < nn; i++) {
        cudaGraphNodeType t; cudaGraphNodeGetType(nodes[i], &t);
        if (t != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams kp;
        if (CK(cudaGraphKernelNodeGetParams(nodes[i], &kp))) {
            int nv = 0x42;
            void* args[3] = { kp.kernelParams[0], kp.kernelParams[1], &nv };
            kp.kernelParams = args;
            if (CK(cudaGraphExecKernelNodeSetParams(ge, nodes[i], &kp))) updated++;
        }
    }
    printf("    graph nodes=%zu, kernel nodes exec-updated=%d (v 0xAA -> 0x42)\n", nn, updated);
    ok &= CK(cudaGraphLaunch(ge, s));
    ok &= CK(cudaStreamSynchronize(s));
    ok &= CK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 4; i++) exp[i] = (i+1)*100 + 0x42;
    bool passB = ok && updated == 4 && eq4(h_out, exp);
    printf("    replay #2 (v=0x42, h_in=100..400): out="); pr4(h_out); printf(" exp="); pr4(exp);
    printf(" -> %s\n", passB ? "PASS" : "FAIL");
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    cudaFree(d_in); cudaFree(d_out); cudaFreeHost(h_in); cudaStreamDestroy(s);
    return passA && passB;
}

static bool test2_conditional() {
    printf("\n[2] conditional node (CUDA 13 API: handle + cudaGraphCondTypeIf)\n");
    int *d_body = nullptr;
    bool ok = CK(cudaMalloc((void**)&d_body, sizeof(int)));
    cudaGraphExec_t ge = nullptr;
    cudaStream_t s = nullptr;
    bool pass = false;
    for (int trial = 0; trial < 2; ++trial) {
        cudaGraph_t g = nullptr;
        ok &= CK(cudaGraphCreate(&g, 0));
        cudaGraphConditionalHandle handle = 0;
        ok &= CK(cudaGraphConditionalHandleCreate(&handle, g, trial ? 0u : 1u, 0));
        cudaGraphNodeParams np = {};
        np.type = cudaGraphNodeTypeConditional;
        np.conditional.handle = handle;
        np.conditional.type = cudaGraphCondTypeIf;
        np.conditional.size = 1;
        cudaGraph_t body_graph = nullptr;
        np.conditional.phGraph_out = &body_graph;
        cudaGraphNode_t cond = nullptr;
        if (!CK(cudaGraphAddNode(&cond, g, nullptr, nullptr, 0, &np))) {
            printf("    -> FAIL: conditional node rejected (trial %d)\n", trial);
            cudaGraphDestroy(g);
            goto done;
        }
        // Body kernels are added to the child graph the conditional created.
        int v = 123; void* args[2] = { &d_body, &v };
        cudaKernelNodeParams kp = {};
        kp.func = (void*)k_write; kp.gridDim = dim3(1,1,1); kp.blockDim = dim3(1,1,1); kp.kernelParams = args;
        cudaGraphNode_t body = nullptr;
        ok &= CK(cudaGraphAddKernelNode(&body, body_graph, nullptr, 0, &kp));
        if (ge) cudaGraphExecDestroy(ge);
        ge = nullptr;
        ok &= CK(cudaGraphInstantiate(&ge, g, 0));
        if (!s) ok &= CK(cudaStreamCreate(&s));
        ok &= CK(cudaMemset(d_body, 0, sizeof(int)));
        ok &= CK(cudaGraphLaunch(ge, s));
        ok &= CK(cudaStreamSynchronize(s));
        int h = -1;
        ok &= CK(cudaMemcpy(&h, d_body, sizeof(int), cudaMemcpyDeviceToHost));
        bool ran = (h == (trial ? 0 : 123));
        printf("    default=%d -> body output=%d (expect %3d) %s\n",
               trial ? 0 : 1, h, trial ? 0 : 123, ran ? "ok" : "FAIL");
        pass = ok && ran;
        cudaGraphDestroy(g);
    }
    printf("    -> %s (conditional nodes on sm_89, CUDA 13.3)\n", pass ? "PASS" : "FAIL");
done:
    if (ge) cudaGraphExecDestroy(ge);
    if (s) cudaStreamDestroy(s);
    cudaFree(d_body);
    (void)ok;
    return pass;
}

static bool test3_device_launch() {
    printf("\n[3] device-side graph launch: __global__ kernel calls cudaGraphLaunch (tail stream)\n");
    cudaStream_t s = nullptr; int* d_out = nullptr;
    bool ok = CK(cudaStreamCreate(&s));
    ok &= CK(cudaMalloc((void**)&d_out, 2*sizeof(int)));
    ok &= CK(cudaMemset(d_out, 0, 2*sizeof(int)));
    ok &= CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    k_child<<<1,1,0,s>>>(d_out);
    cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
    ok &= CK(cudaStreamEndCapture(s, &g));
    ok &= CK(cudaGraphInstantiateWithFlags(&ge, g, cudaGraphInstantiateFlagDeviceLaunch));
    ok &= CK(cudaGraphUpload(ge, s));
    if (!ok) {
        printf("    -> FAIL: device-launch instantiate/upload rejected\n");
    } else {
        k_parent<<<1,1,0,s>>>(ge, d_out);
        ok &= CK(cudaGetLastError());
        ok &= CK(cudaStreamSynchronize(s));
        int h[2] = {-1,-1};
        ok &= CK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
        bool pass = ok && h[0] == 77 && h[1] == 88;
        printf("    parent wrote h[0]=%d (exp 77), device-launched graph wrote h[1]=%d (exp 88) -> %s\n",
               h[0], h[1], pass ? "PASS" : "FAIL");
        ok = pass;
    }
    if (ge) cudaGraphExecDestroy(ge);
    if (g)  cudaGraphDestroy(g);
    cudaFree(d_out); cudaStreamDestroy(s);
    return ok;
}

static bool test4_capture_events() {
    printf("\n[4] event record + stream wait captured INSIDE the graph (cross-stream fork/join)\n");
    cudaStream_t s1 = nullptr, s2 = nullptr; cudaEvent_t e1 = nullptr, e2 = nullptr;
    bool ok = CK(cudaStreamCreate(&s1));
    ok &= CK(cudaStreamCreate(&s2));
    ok &= CK(cudaEventCreateWithFlags(&e1, cudaEventDisableTiming));
    ok &= CK(cudaEventCreateWithFlags(&e2, cudaEventDisableTiming));
    int *d_a = nullptr, *d_b = nullptr;
    ok &= CK(cudaMalloc((void**)&d_a, sizeof(int)));
    ok &= CK(cudaMalloc((void**)&d_b, sizeof(int)));
    ok &= CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal));
    k_write<<<1,1,0,s1>>>(d_a, 7);
    ok &= CK(cudaEventRecord(e1, s1));
    ok &= CK(cudaStreamWaitEvent(s2, e1, 0));
    k_mul2<<<1,1,0,s2>>>(d_a, d_b);
    ok &= CK(cudaEventRecord(e2, s2));
    ok &= CK(cudaStreamWaitEvent(s1, e2, 0));
    cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
    ok &= CK(cudaStreamEndCapture(s1, &g));
    cudaGraphNode_t nodes[16]; size_t nn = 16;
    int nRec = 0, nWait = 0;
    ok &= CK(cudaGraphGetNodes(g, nodes, &nn));
    for (size_t i = 0; i < nn; i++) {
        cudaGraphNodeType t; cudaGraphNodeGetType(nodes[i], &t);
        nRec  += (t == cudaGraphNodeTypeEventRecord);
        nWait += (t == cudaGraphNodeTypeWaitEvent);
    }
    printf("    %zu nodes captured: %d eventRecord + %d eventWait (need >=1 each)\n", nn, nRec, nWait);
    ok &= CK(cudaGraphInstantiate(&ge, g, 0));
    bool pass = (nRec >= 1) && (nWait >= 1);
    for (int r = 0; ok && r < 2; r++) {
        ok &= CK(cudaMemset(d_a, 0, sizeof(int)));
        ok &= CK(cudaMemset(d_b, 0, sizeof(int)));
        ok &= CK(cudaGraphLaunch(ge, s1));
        ok &= CK(cudaStreamSynchronize(s1));
        int a = -1, b = -1;
        ok &= CK(cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost));
        ok &= CK(cudaMemcpy(&b, d_b, sizeof(int), cudaMemcpyDeviceToHost));
        bool p = (a == 7 && b == 14);
        pass &= p;
        printf("    replay %d: d_a=%d d_b=%d (expect 7, 14) %s\n", r, a, b, p ? "ok" : "FAIL");
    }
    printf("    -> %s\n", (ok && pass) ? "PASS" : "FAIL");
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    cudaEventDestroy(e1); cudaEventDestroy(e2);
    cudaStreamDestroy(s1); cudaStreamDestroy(s2);
    cudaFree(d_a); cudaFree(d_b);
    return ok && pass;
}

static double bench(bool graph, cudaStream_t s, cudaGraphExec_t ge, int K, int iters) {
    using clk = std::chrono::steady_clock;
    for (int w = 0; w < 3; w++) {
        if (graph) cudaGraphLaunch(ge, s);
        else for (int i = 0; i < K; i++) k_noop<<<1,1,0,s>>>();
        cudaStreamSynchronize(s);
    }
    auto t0 = clk::now();
    for (int it = 0; it < iters; it++) {
        if (graph) cudaGraphLaunch(ge, s);
        else for (int i = 0; i < K; i++) k_noop<<<1,1,0,s>>>();
        cudaStreamSynchronize(s);
    }
    auto t1 = clk::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

static bool test5_timing() {
    const int K = 400, ITERS = 100;
    printf("\n[5] %d trivial kernels per token, %d tokens: stream launches vs graph replay\n", K, ITERS);
    cudaStream_t s = nullptr;
    bool ok = CK(cudaStreamCreate(&s));
    ok &= CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < K; i++) k_noop<<<1,1,0,s>>>();
    cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
    ok &= CK(cudaStreamEndCapture(s, &g));
    ok &= CK(cudaGraphInstantiate(&ge, g, 0));
    ok &= CK(cudaGraphUpload(ge, s));
    ok &= CK(cudaStreamSynchronize(s));
    double usN = ok ? bench(false, s, ge, K, ITERS) : -1.0;
    double usG = ok ? bench(true,  s, ge, K, ITERS) : -1.0;
    bool pass = ok && usN > 0 && usG > 0 && usG <= usN;
    printf("    stream launches : %8.1f us/token\n", usN);
    printf("    graph replay    : %8.1f us/token  (%d kernels/replay)\n", usG, K);
    printf("    -> %s (speedup %.2fx)\n", pass ? "PASS" : "FAIL", (usN > 0 && usG > 0) ? usN / usG : 0.0);
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(s);
    return pass;
}

int main() {
    int ndev = 0;
    cudaError_t e = cudaGetDeviceCount(&ndev);
    if (e != cudaSuccess || ndev < 1) {
        printf("no CUDA device available (%s, count=%d)\n", cudaGetErrorString(e), ndev);
        return 1;
    }
    int rt = 0, dr = 0;
    cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&dr);
    cudaDeviceProp p {};
    cudaGetDeviceProperties(&p, 0);
    printf("CUDA runtime %d.%d, driver %d.%d\n", rt/1000, (rt%1000)/10, dr/1000, (dr%1000)/10);
    printf("Device 0: %s, CC %d.%d, %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);

    bool r[5] = { test1_exec_update(), test2_conditional(), test3_device_launch(),
                  test4_capture_events(), test5_timing() };
    static const char* names[5] = {
        "capture + memcpy(pinned) + exec-update", "conditional nodes (IF, host-set)",
        "device-side graph launch", "event record/wait inside capture",
        "graph replay timing (400 kernels/token)" };
    printf("\n==================== summary ====================\n");
    int npass = 0;
    for (int i = 0; i < 5; i++) { printf("  [%s] test %d: %s\n", r[i] ? "PASS" : "FAIL", i+1, names[i]); npass += r[i]; }
    printf("  %d/5 passed. All tests ran; exit code 0.\n", npass);
    return 0;
}
