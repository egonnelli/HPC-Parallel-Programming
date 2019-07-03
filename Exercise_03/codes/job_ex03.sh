#!/bin/bash
#PBS -l nodes=2:ppn=20
#PBS -l walltime=02:30:00
#PBS -q regular
cd /home/egonnell/igirotto/ex03
#load the modules
module load openmpi/1.8.3/intel/14.0
for j in {1..3..1};
do for i in 1 2 4 8 16 20 24 28 32 40;
do mpirun -np $i ./reduce_ex03.o >> time.txt;
sleep 3;
done
done
