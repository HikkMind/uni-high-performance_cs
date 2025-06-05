#include <iostream>
#include <cuda_runtime.h>

__global__ void helloFromThreads() {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from thread %d\n", threadId);
}

int main() {
    helloFromThreads << <2, 4 >> > ();
    cudaDeviceSynchronize();

    return 0;
}