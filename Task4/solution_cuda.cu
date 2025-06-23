#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16 

__global__ void matMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int N_values[] = { 10, 100, 10000 };
    for (int test = 0; test < 3; ++test) {
        int N = N_values[test];
        size_t bytes = N * N * sizeof(float);

        float* h_A = new float[N * N];
        float* h_B = new float[N * N];
        float* h_C = new float[N * N];

        for (int i = 0; i < N * N; ++i) {
            h_A[i] = clock() % 10;
            h_B[i] = clock() % 10;
        }

        float* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        matMulKernel << <gridSize, blockSize >> > (d_A, d_B, d_C, N);
        cudaEventRecord(stop);

        cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout << "matrix size: " << N << "x" << N << "\n";
        std::cout << "time : " << milliseconds << " ms\n";

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
    }

    return 0;
}
