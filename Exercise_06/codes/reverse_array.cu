#include<iostream>

      __global__ void reverse( int* d_out, int* d_in, const int N )
      {
       int tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
       if ( tid < N ){d_out[ tid ] = N - d_in[ tid ] - 1;}
      };
             
int main( int argc, char** argv) 
{
      std::cout << "This program prints the reverse array of size N = 10" << "\n";

      const int N = 10;
      int* h_in  = new int [ N ];
      int* h_out = new int [ N ];

      //Host and Device memory
      const int bytes = N * sizeof(int);

      // Pointer for Device memory + Allocation
      int *d_in, *d_out;
      cudaMalloc( (void **) &d_in, bytes );
      cudaMalloc( (void **) &d_out, bytes );

      // Threadblock size
      const int NUM_THREADS = 256;  
      // Grid size
      const int NUM_BLOCKS = ( N + NUM_THREADS - 1 ) / NUM_THREADS;
   
      //Host initialization
      for ( int i = 0; i < N ; ++i )
          {
            h_in[i] = i;
          }
    
      //Print the array - normal order
      std::cout << "\n";
      std::cout << "Array on the normal order: " << "\n";      
      for ( int i = 0; i < N ; ++i )
      {
      std::cout << " " << h_in[i];                         
      }
      
      std::cout << "\n";
      std::cout << "\n";
 
      //Host --> Device
      //h_in to d_in
      cudaMemcpy( d_in, h_in, bytes, cudaMemcpyHostToDevice );
 
      // Reversing the array on GPU
      //CUDA kernel - Reverse Array
      reverse<<< NUM_BLOCKS, NUM_THREADS >>>( d_out, d_in, N );
      //Copy: Device to Host ( d_out -> h_out )
      cudaMemcpy( h_out, d_out, bytes, cudaMemcpyDeviceToHost );

          std::cout << "\n";
          std::cout << "Array on the reverse order: " << "\n";
          for ( int i = 0; i < N ; ++i )
          {
            std::cout << " " << h_out[i];                         
          }

      //Free Device and Host memory
      cudaFree( d_in );
      cudaFree( d_out );
      delete[] h_in;
      delete[] h_out;
     
   return 0;
}
