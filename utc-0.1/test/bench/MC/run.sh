#!/bin/bash

# mpi command
MPI_RUN='mpirun -bind-to none -report-bindings'
MPI_PROCS=1
#MPI_OPTION="-host node2,node3,node4,node5 -N ${MPI_PROCS}"
MPI_OPTION="-n ${MPI_PROCS}"

# set exe file and args
EXE_FILE='./bin/mc_taskcreate.out'
EXE_ARGS="24 100000000"

# set iterations and run
iter=10
i=1
while [ $i -le 10 ]
do
	${MPI_RUN} ${MPI_OPTION} ${EXE_FILE} ${EXE_ARGS}
	((i++))
done

# collect time record info
if [ -f time_record.txt ]
then
    echo "-----------------------------------------" >> total_time_record.txt
    cat time_record.txt >> total_time_record.txt
    echo "-----------------------------------------" >> total_time_record.txt
    rm  -f time_record.txt
else
    echo "time_record.txt not found"
fi
