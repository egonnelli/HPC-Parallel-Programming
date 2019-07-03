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
do export OMP_NUM_THREADS=$i; ./reduction_pi_chrono.o;
sleep 3;
done
done

echo "CRITICAL"

for j in {1..5..1};
do for i in {1..20..1};
do export OMP_NUM_THREADS=$i; ./critical_pi_chrono.o;
sleep 3;
done
done

echo "ATOMIC"

for j in {1..5..1};
do for i in {1..20..1};
do export OMP_NUM_THREADS=$i; ./atomic_pi_chrono.o;
sleep 3;
done
done
