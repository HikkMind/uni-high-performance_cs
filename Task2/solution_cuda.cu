#include <iostream>
#include <cuda_runtime.h>
#include <vector>

// Упрощённое параллельное суммирование с редукцией в блоке
__global__ void sum_reduction(int* input, int* result, int N) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? input[i] : 0;
    __syncthreads();

    // редукция в блоке
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // записываем сумму блока в выходной массив
    if (tid == 0) result[blockIdx.x] = sdata[0];
}

void compute_sum(int N) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    std::vector<int> h_input(N, 1); // массив из единиц
    int* d_input, * d_partial, * h_partial;

    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_partial, numBlocks * sizeof(int));
    h_partial = new int[numBlocks];

    cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // вызов ядра
    sum_reduction << <numBlocks, blockSize, blockSize * sizeof(int) >> > (d_input, d_partial, N);
    cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    int total_sum = 0;
    for (int i = 0; i < numBlocks; ++i) {
        total_sum += h_partial[i];
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "N = " << N << ", sum = " << total_sum
        << ", time = " << milliseconds << " ms" << std::endl;

    cudaFree(d_input);
    cudaFree(d_partial);
    delete[] h_partial;
}

int main() {
    compute_sum(10);
    compute_sum(1000);
    compute_sum(10000000);
    return 0;
}
