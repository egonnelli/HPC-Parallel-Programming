
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <mpi.h>

int main(int argc, char** argv)
{

    int rank, size;
    const int N = atoi(argv[1]);
    FILE* f = fopen("binary_matrix_i.txt", "wb");

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rest = N % size;
    int local_N = (N / size) + (((int)rank < rest) ? 1 : 0);
    int* A = (int*)calloc(local_N * N, sizeof(int));
    int start = (rank * local_N) + (((int)rank < rest) ? 0 : rest);

    for (int i = 0; i < local_N; i++) {
        A[start + (i * N) + i] = 1;
    }

    if (rank != 0) {

        MPI_Send(A, N * local_N, MPI_INT, 0, 101, MPI_COMM_WORLD);
    }

    if (rank == 0) {

        int* received = (int*)malloc(N * local_N * sizeof(int));

        MPI_Request req;

        // For loop calling the ranks
        for (int i = 1; i < size; i++) {

            MPI_Irecv(received, N * local_N, MPI_INT, i, 101, MPI_COMM_WORLD, &req);

            if (N < 10) {
                for (int row = 0; row < local_N; row++) {
                    for (int col = 0; col < N; col++) {
                        std::cout << " " << A[col + row * N];
                    }
                    std::cout << std::endl;
                }
            }
            else if (N >= 10) {
                for (int row = 0; row < local_N; row++) {
                    for (int col = 0; col < N; col++) {
                        fprintf(f, " %d", A[col + row * N]);
                    }
                    fprintf(f, "\n");
                }
            }

            MPI_Wait(&req, MPI_STATUS_IGNORE);
            // swap
            int* tmp = NULL;
            tmp = received;
            received = A;
            A = tmp;
            tmp = NULL;

            if (i == rest && rest != 0) {
                local_N--;
            }

            // End of for loop calling the ranks
        }

        if (N < 10) {
            for (int row = 0; row < local_N; row++) {
                for (int col = 0; col < N; col++) {
                    std::cout << " " << A[col + row * N];
                }
                std::cout << std::endl;
            }
        }
        else if (N >= 10) {
            for (int row = 0; row < local_N; row++) {
                for (int col = 0; col < N; col++) {
                    fprintf(f, " %d", A[col + row * N]);
                }
                fprintf(f, "\n");
            }

            std::cout << "\n"
                      << " Matrix size (N >= 10)  DETECTED !!!"
                      << "\n";
            std::cout << " Saving the result in a Binary file --> binary_matrix_i.txt"
                      << "\n";
            std::cout << "\n";
        }

        free(received);
    }
    free(A);

    MPI_Finalize();

    fclose(f);

    return 0;
}
