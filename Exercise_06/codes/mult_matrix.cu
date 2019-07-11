#include <iostream>

__global__ void matrix_multiply(const int* d_a, const int* d_b, int* d_c, const int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int temp = 0;
    if ((row < n) && (col < n)) {
        for (int k = 0; k < n; k++) {
            temp += d_a[row * n + k] * d_b[k * n + col];
        }
        d_c[row * n + col] = temp;
    }
};

int main(int argc, char** argv)
{

    const int n = atoi(argv[1]);

    size_t bytes = n * n * sizeof(int);

    int* h_a = new int[n * n];
    int* h_b = new int[n * n];
    int* h_c = new int[n * n];
    ;

    int *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    //Matrix initialization

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            h_a[row * n + col] = row;
        }
    }

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            h_b[row * n + col] = row;
        }
    }
    //Host ---> Device ( h_a -> d_a )
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 32;
    const int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(GRID_SIZE, GRID_SIZE, 1);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Matrix Multiplication - CUDA KERNEL
    matrix_multiply<<<grid, threads> > >(d_a, d_b, d_c, n);

    //Device --> Host (d_c -> h_c)
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    std::cout << "\nResult of matrix multiplication on GPU: \n";

    //Print only if n < 20
    if (n < 20 && h_c != NULL) {

        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                std::cout << "\t" << h_c[row * n + col];
            }
            std::cout << "\n";
        }
    }

    std::cout << "\n";
    //Completed GPU computation
    std::cout << "The Matrix multiplication of size n = " << n << " was completed \n\n";

    // Free Host memory
    delete[] h_a;
    h_a = NULL;
    delete[] h_b;
    h_b = NULL;
    delete[] h_c;
    h_c = NULL;

    // Free Device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
