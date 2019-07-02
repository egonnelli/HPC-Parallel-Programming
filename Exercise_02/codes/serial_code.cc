#include <iostream >

void print_usage( int * a, int N, int nthreads );

int main(int argc, char const *argv[])
{   
    const int N = 110;
    int a[N];
    int thread_id = 0;
    int nthreads = 1;

for(int i = 0; i < N; ++i){
    a[i] = thread_id;
}
    print_usage(a, N, nthreads);
    
    return 0;
}

void print_usage( int * a, int N, int nthreads ) {
    
    int tid, i;
    
    for( tid = 0; tid < nthreads; ++tid ) {
        std::cout << tid << ": ";
            for( i = 0; i < N; ++i ) {
                if( a[ i ] == tid) std::cout << "*";
                    else std::cout << " ";
                                    }
         std::cout << std::endl;
}
}
