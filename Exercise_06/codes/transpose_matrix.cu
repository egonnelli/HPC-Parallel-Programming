#include <iostream>

__global__ void matrixTranspose(int* d_in, int* d_out, const int n)
{
    //Row index
    int row = ((blockIdx.y * blockDim.y) + threadIdx.y);
    //Column index
    int col = ((blockIdx.x * blockDim.x) + threadIdx.x);
    //Boundary protection
    if ((row < n) && (col < n)) {
        d_out[col * n + row] = d_in[row * n + col];
    }
};

int main(int argc, char** argv)
{
    std::cout << "\n";
    std::cout << "This Program Transposes a Matrix of size N = 10"
              << "\n";

    //Size of matrix n x n | n = 10
    const int N = 10;

    size_t bytes = N * N * sizeof(int);

    int* h_in = new int[N * N];
    int* h_out = new int[N * N];
    int *d_in, *d_out;

    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // Initialize matrix | each line has the same value
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            h_in[row * N + col] = row;
        }
    }

    std::cout << "\n";
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            std::cout << " " << h_in[row * N + col] << " ";
        }
        std::cout << "\n";
    }

    // Host --> Device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    const int BLOCK_SIZE = 16;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 blocks(GRID_SIZE, GRID_SIZE, 1);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Transposing matrix on GPU
    matrixTranspose<<<blocks, threads> > >(d_in, d_out, N);

    // Device --> Host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Print the Transpose
    std::cout << "\n";
    std::cout << "The Transpose matrix is: "
              << "\n";
    std::cout << "\n";

    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            std::cout << " " << h_out[row * N + col] << " ";
        }
        std::cout << "\n";
    }

    // Free memory
    delete[] h_in;
    h_in = NULL;
    delete[] h_out;
    h_out = NULL;
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
