#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

void matrixMultiply(const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    std::vector<std::vector<double>>& C,
                    int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main() {
    std::vector<int> sizes = {10, 100, 1000};

    for (int N : sizes) {
        std::vector<std::vector<double>> A(N, std::vector<double>(N, 1.0));
        std::vector<std::vector<double>> B(N, std::vector<double>(N, 1.0));
        std::vector<std::vector<double>> C(N, std::vector<double>(N, 0.0));

        auto start = std::chrono::high_resolution_clock::now();

        matrixMultiply(A, B, C, N);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_ms = end - start;

        std::cout << "matrix size: " << N << "x" << N << std::endl;
        std::cout << "time: " << duration_ms.count() << " ms" << std::endl;
    }

    return 0;
}
