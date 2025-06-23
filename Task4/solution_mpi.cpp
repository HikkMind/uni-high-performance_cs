#include <mpi.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <chrono>

void matrixMultiply(const std::vector<std::vector<double>>& A_part,
    const std::vector<std::vector<double>>& B,
    std::vector<std::vector<double>>& C_part,
    int rows_per_proc, int N) {
    for (int i = 0; i < rows_per_proc; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A_part[i][k] * B[k][j];
            }
            C_part[i][j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> sizes = { 10, 100, 1000 };

    for (int N : sizes) {
        int rows_per_proc = N / size;
        int remainder = N % size;

        if (rank == size - 1) {
            rows_per_proc += remainder;
        }

        std::vector<std::vector<double>> A_part(rows_per_proc, std::vector<double>(N));
        std::vector<std::vector<double>> B(N, std::vector<double>(N));
        std::vector<std::vector<double>> C_part(rows_per_proc, std::vector<double>(N, 0.0));

        if (rank == 0) {
            std::vector<std::vector<double>> A(N, std::vector<double>(N));

            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    A[i][j] = (clock() + i + j) % 10;
                    B[i][j] = (clock() + i - j + 1) % 10;
                }
            }

            for (int i = 1; i < size; ++i) {
                int rows = N / size + (i == size - 1 ? remainder : 0);
                for (int r = 0; r < rows; ++r) {
                    MPI_Send(A[i * (N / size) + r].data(), N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                }
            }

            for (int i = 0; i < rows_per_proc; ++i) {
                A_part[i] = A[i];
            }
        }
        else {
            for (int i = 0; i < rows_per_proc; ++i) {
                MPI_Recv(A_part[i].data(), N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        for (int i = 0; i < N; ++i) {
            MPI_Bcast(B[i].data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();

        matrixMultiply(A_part, B, C_part, rows_per_proc, N);

        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_ms = end - start;

        if (rank == 0) {
            std::cout << "matrix size: " << N << "x" << N << std::endl;
            std::cout << "time : " << duration_ms.count() << " ms" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
