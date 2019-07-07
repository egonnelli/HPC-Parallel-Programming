/* This Program send an integer number in a ring-like (circle) fashion */

#include<iostream>
#include<mpi.h>

    int main(int argc, char** argv){

    int size, rank;
    int message = -999;
    
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Request request;

    if (size < 2)
    {
        std::cout << "Number of processes must be at least 2 " << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* each rank will have an array filled with
       the own rank value */

    for(int i=0;i<size;i++)
    {
          MPI_Isend(&message, 1, MPI_INT,(rank - 1 + size) % size,
           101, MPI_COMM_WORLD, &request);

          MPI_Recv(&message, 1, MPI_INT, (rank + 1 + size) % size,
           101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

          MPI_Barrier(MPI_COMM_WORLD);

    std::cout << message << ": " << rank << " ----> " <<
     (rank -  1 + size) % size  << "\n";

          MPI_Finalize();

        return 0;
      }
