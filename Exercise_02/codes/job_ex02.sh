#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=02:30:00
#PBS -q regular
cd /home/egonnell/igirotto/ex02
#load the modules
module load gnu/4.9.2
#call 10 threads , run the program and salve on txt file
export OMP_NUM_THREADS=10; ./schedule.o > 110N.txt
