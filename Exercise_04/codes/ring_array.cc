/* The Program sends an array in a ring-like (circle) fashion */

#include <iostream>
#include <mpi.h>

int main(int argc, char** argv)
{

    const int sizeArray = 2048;
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Request request;

    if (size < 2) {
        std::cout << "Number of processes must be at least 2 "
                  << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // size of array
    // dynamic allocation employing the 'new'

    int* array = new int[sizeArray];
    int* from_Right = new int[sizeArray];
    int* sum = new int[sizeArray];

    /* each rank will have an array filled with
      the own rank value */

    for (int i = 0; i < sizeArray; i++) {
        array[i] = rank;
        sum[i] = 0;
        from_Right[i] = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int j = 0; j < size; j++) {

        // MPI_ISEND(buf, count, type, dest, tag, comm, request, ierr)

        MPI_Isend(array, sizeArray, MPI_INT, (rank - 1 + size) % size, 101,
            MPI_COMM_WORLD, &request);

        //MPI_RECV(buf, count, type, dest, tag, comm, status, ierr)

        MPI_Recv(from_Right, sizeArray, MPI_INT, (rank + 1 + size) % size, 101,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int j = 0; j < sizeArray; j++) {
            sum[j] += from_Right[j];
        }

        MPI_Wait(&request, MPI_STATUS_IGNORE);

        array = from_Right;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //print the first element of the array to confirm the summation

    std::cout << "Rank [" << rank << "]"
              << " ----> "
              << " The sum is: [" << sum[0] << "]"
              << "\n";

    MPI_Finalize();

    delete[] array;
    delete[] from_Right;
    delete[] sum;

    return 0;
}
