#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>

void compute_sum(int n) {
    std::vector<int> arr(n);
    for (int i = 0; i < n; ++i) arr[i] = clock() % 10;
    int sum = 0;

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += arr[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "n = " << n << ", sum = " << sum
        << ", time = " << elapsed.count() << " seconds" << std::endl;
}

int main() {
    std::cout << "threads count : " << omp_get_max_threads() << std::endl;

    compute_sum(10);
    compute_sum(1000);
    compute_sum(10000000);

    return 0;
}
