OSH_SRC     = shmem_kmeans.c   \
              parallel-kmeans-NWU/mpi_io.c\
              parallel-kmeans-NWU/file_io.c
              
OSH_OBJ = shmem_kmeans.o mpi_io.o file_io.o

default: shmem_a.out

shmem_kmeans.o: shmem_kmeans.c
	mpicc -O2 -DNDEBUG -loshmem -c shmem_kmeans.c

mpi_io.o: parallel-kmeans-NWU/mpi_io.c
	mpicc -O2 -DNDEBUG -c parallel-kmeans-NWU/mpi_io.c
	
file_io.o: parallel-kmeans-NWU/file_io.c
	mpicc -O2 -DNDEBUG -c parallel-kmeans-NWU/file_io.c
	
shmem_a.out:$(OSH_OBJ)
	mpicc -o shmem_a.out -O2 -DNDEBUG -loshmem $(OSH_OBJ)
	