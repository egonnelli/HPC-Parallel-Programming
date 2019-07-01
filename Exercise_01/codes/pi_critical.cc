\documentclass[USenglish]{article}

\usepackage[utf8]{inputenc}
\usepackage[toc,page]{appendix}
\usepackage{datetime}
\usepackage{relsize}
\usepackage{isodate}
\usepackage{geometry}
\usepackage{graphicx} % Required for including images
\usepackage[font=small,labelfont=bf]{caption} % Required for specifying captions to tables and figures
%\usepackage{minted}

\usepackage{listings}
\usepackage{xcolor}
\lstset { %
   language=C++,
    backgroundcolor=\color{black!5}, % set backgroundcolor
    basicstyle=\footnotesize,% basic font setting
}

\lstset{basicstyle=\ttfamily,
  showstringspaces=false,
  commentstyle=\color{red},
  keywordstyle=\color{blue},
 backgroundcolor=\color{black!5}
}


\geometry{a4paper,total={170mm,257mm}, left=20mm, top=20mm,}

\title{ HPC - Exercise 1}
\author{Eduardo Gonnelli}
\date{01 July 2019}

\begin{document}
\maketitle

\section{Introduction}

This report aims to provide the $\pi$ implementation employing the midpoint numerical method. The numerical method was written in C++ programming language and executed in two versions: in serial and in parallel. The execution time for both codes was measured, considering the same problem size, and the scalability and speedup graphs were generated. The parallel part of the problem took into account the variation of the number of threads, from 1 up to 20, and three synchronization pragma OpemMP derivatives: Critical, Atomic and Reduction. All the codes were executed on Ulysses Cluster.

 \subsection{OpenMP} 

The OpenMP is an Application Programming Interface (API) that supports multi-platform shared-memory parallel programming in C/C++ and Fortran. The OpenMP API defines a portable, scalable model with a simple and flexible interface for developing parallel applications on platforms from the desktop to the supercomputer.

As long as different threads write to a different memory location, for example, different elements of the same vector, there is no reason to worry. Problems arise if they simultaneously write to the same address in memory. Then, threads may step on each other and generate incorrect results. This is a bug in the code and is called a "data race."

Synchronizing, or coordinating the actions of, threads is sometimes necessary in
order to ensure the proper ordering of their accesses to shared data and to prevent data corruption. A thread is not allowed to enter a critical region, as long as another thread executes it. As a result, a thread cannot perform the update, while another thread is inside the critical region. For this reason, the mechanisms have been proposed to support the synchronization needs of a variety of applications. For our exercise, the Atomic and Critical Construct and Reduction Clause were implemented.

\subsubsection{Critical}

The critical construct provides a means to ensure that multiple threads do not
attempt to update the same shared data simultaneously. The associated code is
referred to as a critical region, or a critical section. The Figure \ref{fig:critical} refers to the critical section of code.

\begin{center}
\fbox{\includegraphics[scale=0.5]{critical.PNG}}
\captionof{figure}{Syntax of the critical construct in C/C++ – The structured
block is executed by all threads, but only one at a time executes the block. Optionally,
the construct can have a name.}
\label{fig:critical}
\end{center}
 
\subsubsection{Atomic}

The atomic construct, which also enables multiple threads to update shared data
without interference, can be an efficient alternative to the critical region. it is applied only to the (single) assignment statement that immediately follows it. The Figure \ref{fig:atomic} refers to the atomic section of code.

\begin{center}
\fbox{\includegraphics[scale=0.5]{atomic.PNG}}
\captionof{figure}{Syntax of the atomic construct in C/C++ – The statement is
executed by all threads, but only one thread at a time executes the statement.}
\label{fig:atomic}
\end{center}

\subsubsection{Reduction}
 
The reduction operator(s) and variable(s) are specified in the reduction clause.
By definition, the result variable, like \textit{sum} in this case, is shared in the enclosing
OpenMP region. The command \textit{\#pragma omp parallel for reduction(+:sum)} is commonly used to parallelize the loop. The Figure \ref{fig:reduction} refers to the reduction section of code.  

\begin{center}
\fbox{\includegraphics[scale=0.5]{reduction.PNG}}
\captionof{figure}{Reduction Example of the reduction clause – This clause gets the OpenMP
compiler to generate code that performs the summation in parallel}
\label{fig:reduction}
\end{center}

\subsubsection{ Execution time measurement}

The execution time of the parallel part of the program was measured employing the function name \textit{omp\_get\_wtime()}. This function provides the absolute wall-clock time in seconds. It must to be highlighted that the \textit{omp\_get\_wtime()} returns the number of wall-clock or "elapsed" seconds. In other words, it returns an absolute value. A meaningful timing value is therefore obtained by taking the difference between two calls. Due to this fact, the \textit{omp\_get\_wtime()} was called before and after the specific portion of the program in which execution time must be measured. The Figure \ref{fig:wtime} shows the piece of code that contains the application of \textit{omp\_get\_wtime()}.

\begin{center}
\fbox{\includegraphics[scale=0.5]{wtime.PNG}}
\captionof{figure}{An example how to use function \textit{omp\_get\_wtime()}}
\label{fig:wtime}
\end{center} 

\subsection{Midpoint Rule} 

The rectangular rule (also called the midpoint rule) is a numerical method in which is possible to estimate an integral value with finite sums of rectangles. Using more rectangles can increase the accuracy of the approximation. As can be seen on Figure \ref{fig:midpoint}, the midpoint rule uses rectangles whose heights are the values of \textit{f} at the midpoints of their bases. The interval integration is defined as $a \le x \le b$ and divided up into \textit{n} equal subintervals of length $h = (b - a)/n$.

\begin{center}
\fbox{\includegraphics[]{midpoint1.PNG}}
\captionof{figure}{The midpoint rule uses rectangles whose height is the value of f(xi) at the midpoints of their bases.}
\label{fig:midpoint}
\end{center}

For $\pi$ approximation, the integral of the function $F(x) = 1 / (1+x^2)$ was numerically obtained applying the midpoint procedure.

\section{Results}

The speedup was calculated based on the measured execution time. A simple approach was employed for the calculation, i.e., the same program was executed on a single processor (Nº threads equal 1), and on a parallel machine with p processors, and to compare the runtimes were compared. The speedup can be defined as: 
\begin{equation}
    \label{simple_equation}
    \ S_p = T_1 / T_p \
\end{equation}
 where $S_p$ is the speedup, $T_1$ is the execution time on a single processor and $T_p$ is the time on \textit{p} processors. Furthermore, to measure how far the the results are from the ideal speedup, the efficiency $E_p = S_p / p$. Clearly, $0 < E_p \le 1$.


 The Figure \ref{fig:result1} shows the results for critical, reduction and atomic OpenMP.


The implementation of the serial code for $\pi$ approximation achieved the execution time of 42.10 $\pm$ 0.02 s. The code was executed on the node cn04-33 and the integral approximation of $\pi$ was equal to: 3.14159.
The execution time was measured employing the Chrono library.

\begin{center}
\fbox{\includegraphics[scale=0.5]{ex01-OpenMP-Pi.eps}}
\captionof{figure}{Time measurements for different numbers of threads.}
\label{fig:result1}
\end{center}



\begin{thebibliography}{9}

\bibitem{openmpwebsite} 
The OpenMP: Architecture Review Boards (ARB)
\\\texttt{https://www.openmp.org/}

\bibitem{usingopenmp1} 
Barbara Chapman, Gabriele Jost, and Ruud van der Pas. 
\textit{Using OpenMP-Portable Shared Memory Parallel Programming}. 
The MIT Press, Cambridge, 2008.
 
\bibitem{usingopenmp2} 
Ruud van der Pas, Eric Stotzer, and Christian Terboven. 
\textit{Using OpenMP-The Next Step: Affinity, Accelerators, Tasking, and SIMD}. 
The MIT Press, Cambridge, 2017.

\bibitem{calculus} 
George B. Thomas, Jr. 
\textit{Thomas' CALCULUS}. 
Pearson, Boston, 2016
 
\end{thebibliography}



\pagebreak

\begin{appendices}

This section contains the codes developed for this work. 

\section{Job Script}

 
\begin{lstlisting}[language=bash,caption={Job Script}]
#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=02:30:00
#PBS -q regular

cd /home/egonnell/igirotto/ex01

#load the modules

module load openmpi/1.8.3/intel/14.0

echo "REDUCTION"

for j in {1..5..1};
do for i in {1..20..1};
do export OMP_NUM_THREADS=$i; ./reduction_pi_chromo.o;
sleep 3;
done
done

echo "CRITICAL"

for j in {1..5..1};
do for i in {1..20..1};
do export OMP_NUM_THREADS=$i; ./critical_pi_chromo.o;
sleep 3;
done
done

echo "ATOMIC"

for j in {1..5..1};
do for i in {1..20..1};
do export OMP_NUM_THREADS=$i; ./atomic_pi_chromo.o;
sleep 3;
done
done
\end{lstlisting}

\pagebreak


\section{C++ codes}

\begin{lstlisting}[language=c++,caption={Serial Code}]

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


\end{lstlisting}
 
\begin{lstlisting}[language=c++,caption={Critical construct}]
#include <iostream>
#include <omp.h>
#include <chrono>
#include <vector>

// CRITICAL

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
        for (int i = 0; i <= (int) n-1; ++i){
                double x_i = h*(i+0.5);
                local += function(x_i);
                }
#pragma omp critical
integral += local;
}

double duration = omp_get_wtime() - tstart;
auto finish = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed_vector = finish - start;

        double final= 4*h*integral;
        std::cout << "Number of threads: " << omp_get_max_threads() 
<< "\n";
        std::cout << "Integral is equal to: " << final << "\n";
        std::cout << "Time for loop: " << duration << " s\n";
        std::cout << "Chrono (std::vector) time: " << elapsed_vector.count() 
<< " s\n";
              return 0;
    }
    double function(double x) {
      return 1.0/( 1.0 + x * x);
    }

\end{lstlisting}


\end{lstlisting}

\end{appendices}
\end{document}