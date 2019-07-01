#include <iostream>
#include <omp.h>
#include <chrono>
#include <vector>

//ATOMIC

double function(double x);

int main() {

        double a = 0.0;
        double b = 1.0;
        int n = 2147483647;
        double integral = 0.0;
        double h = (b - a) / n;

auto start = std::chrono::high_resolution_clock::now();
double tstart = omp_get_wtime();

#pragma omp parallel
{
double local = 0.0;
#pragma omp for
        for (int i = 0; i <= (int) n-1; ++i) {
                double x_i = h*(i+0.5);
                local += function(x_i);
                }
#pragma omp atomic
integral += local;
}

double duration = omp_get_wtime() - tstart;
auto finish = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed_vector = finish - start;

        double final= 4*h*integral;
        std::cout << "Number of threads: " << omp_get_max_threads() << "\n";
        std::cout << "Integral is equal to: " << final << "\n";
        std::cout << "Time for loop: " << duration << " s\n";
        std::cout << "Chrono (std::vector) time: " << elapsed_vector.count() << " s\n";
              return 0;
    }
    double function(double x) {
      return 1.0/( 1.0 + x * x);
    }
