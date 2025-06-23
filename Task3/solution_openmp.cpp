#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

// Пример функции f(x, y) = sin(x) * cos(y)
double f(double x, double y) {
    return sin(x) * cos(y);
}

int main(int argc, char* argv[]) {
    int nx = 100;
    int ny = 100;
    double dx = 0.01;

    if (argc > 3) {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
        dx = atof(argv[3]);
    }

    int rank = 0; 
    int size = 1;

    vector<vector<double>> A(nx, vector<double>(ny, 0.0));
    vector<vector<double>> B(nx, vector<double>(ny, 0.0));

    double start = omp_get_wtime();
#pragma omp parallel for collapse(2)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            A[i][j] = f(i * dx, j * dx);
        }
    }
#pragma omp parallel for collapse(2)
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 0; j < ny; ++j) {
            B[i][j] = (A[i + 1][j] - A[i - 1][j]) / (2.0 * dx);
        }
    }

    double end = omp_get_wtime();

    if (rank == 0) {
        cout << "x_grid = " << nx << ", y_grid = " << ny << endl;
        cout << "time : " << (end - start) << " sec" << endl;
    }

    return 0;
}
