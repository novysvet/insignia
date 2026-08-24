#include "insignia_layout.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#define CUDA_OK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){std::fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); return 2;} } while(0)

int main() {
    constexpr int rows = 257;
    constexpr int cols = 4096;
    constexpr int blocks = cols / 32;
    std::vector<insignia::MxFp4Block> w(rows * blocks);
    std::vector<float> x(cols), expected(rows), actual(rows);

    for (int i=0;i<cols;i++) x[i] = float((i * 17) % 31 - 15) / 16.0f;
    for (int r=0;r<rows;r++) {
        double sum=0.0;
        for (int b=0;b<blocks;b++) {
            auto &q=w[r*blocks+b];
            q.scale=uint8_t(124 + ((r+b)%7));
            for(int j=0;j<16;j++) q.q[j]=uint8_t(((r+b+j*3)&15) | (((r*5+b+j*7)&15)<<4));
            for(int lane=0;lane<32;lane++) sum += double(insignia::mxfp4_value(q,lane))*x[b*32+lane];
        }
        expected[r]=float(sum);
    }

    insignia::MxFp4Block *dw=nullptr; float *dx=nullptr,*dy=nullptr;
    CUDA_OK(cudaMalloc(&dw,w.size()*sizeof(w[0]))); CUDA_OK(cudaMalloc(&dx,x.size()*sizeof(x[0]))); CUDA_OK(cudaMalloc(&dy,actual.size()*sizeof(actual[0])));
    CUDA_OK(cudaMemcpy(dw,w.data(),w.size()*sizeof(w[0]),cudaMemcpyHostToDevice)); CUDA_OK(cudaMemcpy(dx,x.data(),x.size()*sizeof(x[0]),cudaMemcpyHostToDevice));
    insignia::mxfp4_gemv(dw,dx,dy,rows,cols); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize()); CUDA_OK(cudaMemcpy(actual.data(),dy,actual.size()*sizeof(actual[0]),cudaMemcpyDeviceToHost));

    cudaEvent_t begin{}, end{}; CUDA_OK(cudaEventCreate(&begin)); CUDA_OK(cudaEventCreate(&end));
    constexpr int iterations=1000; CUDA_OK(cudaEventRecord(begin));
    for(int i=iterations;i!=0;--i) insignia::mxfp4_gemv(dw,dx,dy,rows,cols);
    CUDA_OK(cudaEventRecord(end)); CUDA_OK(cudaEventSynchronize(end)); float elapsed_ms=0.0f; CUDA_OK(cudaEventElapsedTime(&elapsed_ms,begin,end));
    cudaEventDestroy(begin); cudaEventDestroy(end);

    float max_abs=0.0f,max_rel=0.0f;
    for(int r=0;r<rows;r++){float ae=std::fabs(actual[r]-expected[r]); float re=ae/(std::fabs(expected[r])+1e-6f); if(ae>max_abs)max_abs=ae;if(re>max_rel)max_rel=re;}
    cudaFree(dw);cudaFree(dx);cudaFree(dy);
    const double ms=elapsed_ms/iterations; const double gib_s=(double(w.size()*sizeof(w[0])+x.size()*sizeof(x[0]))/1073741824.0)/(ms/1000.0);
    std::printf("MXFP4 reference GEMV: rows=%d cols=%d packed=%zu bytes max_abs=%g max_rel=%g %.3f ms %.1f GiB/s\n",rows,cols,w.size()*sizeof(w[0]),max_abs,max_rel,ms,gib_s);
    return max_rel < 2e-4f ? 0 : 1;
}
