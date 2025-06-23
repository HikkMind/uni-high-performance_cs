#include <iostream>
#include <cmath>
#include <chrono>

__global__ void compute_derivative(double* A, double* B, int nx, int ny, double dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 

    if (i >= 1 && i < nx - 1 && j >= 0 && j < ny) {
        int idx = i * ny + j;
        int idx_left = (i - 1) * ny + j;
        int idx_right = (i + 1) * ny + j;
        B[idx] = (A[idx_right] - A[idx_left]) / (2.0 * dx);
    }
}

int main(int argc, char** argv) {
    int nx = 1000;
    int ny = 1000;
    double dx = 0.01;

    if (argc > 3) {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
        dx = atof(argv[3]);
    }

    int N = nx * ny;

    double* h_A = new double[N];
    double* h_B = new double[N];

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            h_A[i * ny + j] = sin(i * dx) * cos(j * dx); // пример f(x,y)
        }
    }

    double* d_A, * d_B;
    cudaMalloc(&d_A, N * sizeof(double));
    cudaMalloc(&d_B, N * sizeof(double));

    cudaMemcpy(d_A, h_A, N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
        (ny + blockSize.y - 1) / blockSize.y);

    auto start = std::chrono::high_resolution_clock::now();

    compute_derivative << <gridSize, blockSize >> > (d_A, d_B, nx, ny, dx);
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop - start;

    std::cout << "CUDA: nx = " << nx << ", ny = " << ny << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " sec" << std::endl;

    cudaMemcpy(h_B, d_B, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    delete[] h_A;
    delete[] h_B;

    return 0;
}
