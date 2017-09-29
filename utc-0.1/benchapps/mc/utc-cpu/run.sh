#!/bin/bash

# mpi command
MPI_RUN='mpirun --mca mpi_cuda_support 0' #socket:overload-allowed  -bind-to none  -report-bindings
MPI_PROCS=1   #set to the number of socket for better performance with mpirun
#MPI_OPTION="-host node2,node3,node4,node5 -N ${MPI_PROCS}"
MPI_OPTION="-N ${MPI_PROCS}  -hostfile myhosts1"


# set exe file and args
EXE_FILE='./mc'
EXE_ARGS="1  3840000000"

# set iterations and run
iter=10
i=1
while [ $i -le $iter ]
do
	${MPI_RUN} ${MPI_OPTION} ${EXE_FILE} ${EXE_ARGS}
	echo "round " ${i} " finished ..."
	((i++))
	sleep 1
done

# collect time record info
if [ -f time_record.txt ]
then
	echo "-----------------------------------------" >> total_time_record.txt
	echo ${EXE_FILE} ${EXE_ARGS} >> total_time_record
	cat time_record.txt >> total_time_record.txt
	echo "-----------------------------------------" >> total_time_record.txt
	rm  -f time_record.txt
else
	echo "time_record.txt not found"
fi
