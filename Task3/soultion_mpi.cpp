#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

using namespace std;

double f(double x, double y) {
    return sin(x) * cos(y);
}

double df_dx(double x, double y, double dx) {
    return (f(x + dx, y) - f(x, y)) / dx;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nx = 1000;
    int ny = 1000; 
    double dx = 0.01;
    if (argc > 3) {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
        dx = atof(argv[3]);
    }

    int rows_per_proc = nx / size;
    int extra = nx % size;

    int start_row = rank * rows_per_proc + min(rank, extra);
    int local_rows = rows_per_proc + (rank < extra ? 1 : 0);

    vector<vector<double>> A(local_rows, vector<double>(ny));
    vector<vector<double>> B(local_rows, vector<double>(ny));

    auto t0 = chrono::high_resolution_clock::now();

    for (int i = 0; i < local_rows; ++i) {
        int global_i = start_row + i;
        for (int j = 0; j < ny; ++j) {
            double x = global_i * dx;
            double y = j * dx;
            A[i][j] = f(x, y);
            B[i][j] = df_dx(x, y, dx);
        }
    }
    auto t1 = chrono::high_resolution_clock::now();
    double local_time = chrono::duration<double>(t1 - t0).count();

    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        cout << "x_grid = " << nx << ", y_grid = " << ny << endl;
        cout << "time : " << max_time << " sec" << endl;
    }

    MPI_Finalize();
    return 0;
}
