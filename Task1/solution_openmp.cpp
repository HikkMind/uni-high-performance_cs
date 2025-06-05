#include <omp.h>
#include <stdio.h>

int main() {
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        printf("Hello from OpenMP thread %d\n", id);
    }
    return 0;
}