#!/bin/bash

# mpi command
#MPI_RUN='mpirun -report-bindings -bind-to none' #socket:overload-allowed'
#MPI_PROCS=1
#MPI_OPTION="-host node2,node3,node4,node5 -N ${MPI_PROCS}"
#MPI_OPTION="-N ${MPI_PROCS}  -hostfile myhosts -host node2,node3,node4,node5"


# set exe file and args
EXE_FILE='./yuvconvert'
EXE_ARGS="-t 8 -i ../../image-input/8k.ppm"

# set iterations and run
iter=15
i=1
while [ $i -le $iter ]
do
	${EXE_FILE} ${EXE_ARGS}
	echo "round " ${i} " finished ..."
	((i++))
	sleep 1
done

# collect time record info
if [ -f time_record.txt ]
then
	echo "-----------------------------------------" >> total_time_record.txt
	echo ${EXE_FILE} ${EXE_ARGS} >> total_time_record.txt
	cat time_record.txt >> total_time_record.txt
	echo "-----------------------------------------" >> total_time_record.txt
	rm  -f time_record.txt
else
	echo "time_record.txt not found"
fi


