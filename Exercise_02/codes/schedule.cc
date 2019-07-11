#include <iostream>
#include <omp.h>

void print_usage(int* a, int N, int nthreads);

int main(int argc, char const* argv[])
{
    const int N = 110;
    int a[N];
    int thread_id = 0;
    int nthreads = 1;

    for (int i = 0; i < N; ++i) {
        a[i] = thread_id;
    }

    print_usage(a, N, nthreads);
    std::cout << "\n";

#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
#pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            a[i] = omp_get_thread_num();
        }
//Only one Thread will print the result
#pragma omp single
        {
            std::cout << "This is the Schedule(static)"
                      << "\n";
            std::cout << "\n";
            print_usage(a, N, nthreads);
            std::cout << "\n";
        }
//static, with chunk size 1
#pragma omp for schedule(static, 1)
        for (int i = 0; i < N; ++i) {
            a[i] = omp_get_thread_num();
        }
//Only one Thread will print the result
#pragma omp single
        {
            std::cout << "This is the Schedule(static,1)"
                      << "\n";
            std::cout << "\n";
            print_usage(a, N, nthreads);
            std::cout << "\n";
        }
//static, with chunk size 10
#pragma omp for schedule(static, 10)
        for (int i = 0; i < N; ++i) {
            a[i] = omp_get_thread_num();
        }
//Only one Thread will print the result
#pragma omp single
        {
            std::cout << "This is the Schedule(static,10)"
                      << "\n";
            std::cout << "\n";
            print_usage(a, N, nthreads);
            std::cout << "\n";
        }
//dynamic
#pragma omp for schedule(dynamic)
        for (int i = 0; i < N; ++i) {
            a[i] = omp_get_thread_num();
        }
//Only one Thread will print the result
#pragma omp single
        {
            std::cout << "This is the Schedule(dynamic)"
                      << "\n";
            std::cout << "\n";
            print_usage(a, N, nthreads);
            std::cout << "\n";
        }
//dynamic, with chunk size 1
#pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < N; ++i) {
            a[i] = omp_get_thread_num();
        }
//Only one Thread will print the result
#pragma omp single
        {
            std::cout << "This is the Schedule(dynamic,1)"
                      << "\n";
            std::cout << "\n";
            print_usage(a, N, nthreads);
            std::cout << "\n";
        }
//dynamic, with chunk size 10
#pragma omp for schedule(dynamic, 10)
        for (int i = 0; i < N; ++i) {
            a[i] = omp_get_thread_num();
        }
//Only one Thread will print the result
#pragma omp single
        {
            std::cout << "This is the Schedule(dynamic,10)"
                      << "\n";
            std::cout << "\n";
            print_usage(a, N, nthreads);
            std::cout << "\n";
        }
    }

    return 0;
}

void print_usage(int* a, int N, int nthreads)
{
    int tid, i;
    for (tid = 0; tid < nthreads; ++tid) {

        std::cout << tid << ": ";

        for (i = 0; i < N; ++i) {

            if (a[i] == tid)
                std::cout << "*";
            else
                std::cout << " ";
        }
        std::cout << std::endl;
    }
}
