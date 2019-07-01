#include <iostream>
#include <chrono>
#include <vector>

// SERIAL

double function(double x);

int main() {

        double a = 0.0;
        double b = 1.0;
        int n = 2147483647;
        double integral = 0.0;
        double h = (b - a) / n;

auto start = std::chrono::high_resolution_clock::now();

for (int i = 0; i <= (int) n-1; ++i) {
      double x_i = h*(i+0.5);
      integral += function(x_i);
}

auto finish = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed_vector = finish - start;

double final= 4*h*integral;

        std::cout << "Integral is equal to: " << final << "\n";
        std::cout << "Chrono (std::vector) time: " << elapsed_vector.count() 
<<  " s\n";
              return 0;
    }
    double function(double x) {
      return 1.0/( 1.0 + x * x);
    }
