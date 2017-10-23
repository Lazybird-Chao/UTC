#!/bin/bash
# a sample job submission script to submit an MPI job to the broadwl partition on Midway2

# set the job name to hello-mpi
#SBATCH --job-name=utc-mc

# send output to hello-mpi.out
#SBATCH --output=test-mpi.out

# receive an email when job starts, ends, and fails
#SBATCH --mail-type=BEGIN,END,DAIL

# this job requests 32 cores. Cores can be selected from various nodes.
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=32
#SBATCH --exclusive

# there are a few partitions on Midway2 and it is important to specify which
# partition you want to run your job on. Not having the following line, the 
# sandby partition on Midway1 will be selected as the default partition
#SBATCH --partition=ser-par-10g-4


# load the openmpi module
module load gnu-4.9.3-compilers
module load openmpi-2.0.1
module load boost-1.58.0_gnu-4.9.3_mpich-3.0.4

# Run the process with mpirun. Notice -n is not required. mpirun will
# automatically figure out how many processes to run from the slurm options
#mpirun /home/liu.chao/utc-workingon/benchapps/mm/mpi/mm -v -s 1024 -p 1024


MPI_RUN='mpirun --use-hwthread-cpus --map-by socket --mca mpi_cuda_support 0'
# set exe file and args
EXE_FILE='/home/liu.chao/utc-workingon/benchapps/mc/utc-cpu/mc'
EXE_ARGS="-v  -p 32 -t 24"

# set iterations and run
iter=30
i=1
while [ $i -le $iter ]
do
	${MPI_RUN} ${EXE_FILE} ${EXE_ARGS}
	echo "round " ${i} " finish ..."
	((i++))
done

# collect time record info
if [ -f time_record.txt ]
then
	echo "-----------------------------------------" >> total_time_record.txt
	echo ${MPI_RUN} -N 1  ${EXE_FILE} ${EXE_ARGS} >> total_time_record.txt
	cat time_record.txt >> total_time_record.txt
	echo "-----------------------------------------" >> total_time_record.txt
	rm  -f time_record.txt
else
	echo "time_record.txt not found"
fi
