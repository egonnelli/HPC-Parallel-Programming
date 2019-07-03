#include <iostream>
#include <mpi.h>

double function(const double x);

int main(int argc, char ** argv)
{

   const int N = 2147483647;
   const double a = 0.0;
   const double b = 1.0;
   const double h = (b-a)/N; // implementation of the midpoint method

   int rank, size;
MPI_Status status;
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);


   const unsigned int local_N = N/size;
   double local_pi = 0.0;
   double start_t = 0.0;
   double end_t = 0.0;

//Reference points for computation:
   const double start_computation = rank * local_N;
   const double end_computation = start_computation  + local_N;

start_t = MPI_Wtime();

for (int i = start_computation ; i < end_computation ; ++i)
      {
        double x_i = (i + 0.5)* h;
        local_pi += h*function(x_i);
        //The result will be PI/4, that's why we multiply by 4
      }
double total_pi = 4*local_pi;
  //variable for final result:
   double global_pi = 0.0;
//Computing the final result:
MPI_Reduce(&total_pi, &global_pi, 1, MPI_DOUBLE, MPI_SUM, (size-1) , MPI_COMM_WORLD);
end_t = MPI_Wtime() - start_t;

//The last rank will send the message to rank 0:
if ( rank == (size - 1 ))
    {
      MPI_Send(&global_pi, 1, MPI_DOUBLE, 0  , 101, MPI_COMM_WORLD); //101 as TAG


//Rank 0 will recieve the message and print the result
   }
if ( rank == 0 ){
     MPI_Recv(&global_pi, 1, MPI_DOUBLE, (size -1) , 101, MPI_COMM_WORLD, &status);
     std::cout << "This is the Process: " << rank << "\n";
     std::cout << "Execution time = " << end_t << " s\n";
     std::cout << "Pi Approximation = " << global_pi << "\n";
    }
  MPI_Finalize();
  return 0;
}

double function(const double x)
        {return 1.0/(1.0+(x*x));}                         
