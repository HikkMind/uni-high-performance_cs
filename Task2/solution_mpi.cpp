#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <ctime>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> data;
    int n = 10000000;


    data.resize(n);
    for (int i = 0; i < n; ++i) data[i] = clock() % 10;
    int chunk_size = n / size;
    std::vector<int> sub_data(chunk_size);

    MPI_Scatter(data.data(), chunk_size, MPI_INT, sub_data.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    int local_sum = std::accumulate(sub_data.begin(), sub_data.end(), 0);

    int total_sum = 0;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Total sum = " << total_sum << "\n";
        std::cout << "Elapsed time: " << elapsed.count() << " sec\n";
    }

    MPI_Finalize();
    return 0;
}