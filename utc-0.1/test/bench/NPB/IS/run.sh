#!/bin/bash

# mpi command
MPI_RUN='mpirun -bind-to none -report-bindings'
MPI_PROCS=2
MPI_OPTION="-n ${MPI_PROCS}"

# set exe file and args
EXE_FILE='./bin/is.A.2.utc'
EXE_ARGS="-t 4 -p 2"

# set iterations and run
iter=10
i=1
while [ $i -le 10 ]
do
	${MPI_RUN} ${MPI_OPTION} ${EXE_FILE} ${EXE_ARGS}
	((i++))
done

# collect time record info
cat time_record.txt >> total_time_record.txt
rm  -f time_record.txt

