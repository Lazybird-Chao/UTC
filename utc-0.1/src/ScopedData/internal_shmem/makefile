INCLUDE = -I../../../include/ScopedData/internal_shmem
LIB = ./libutc-scopedshmem.a
OBJS:= 	internal_win.o \
    	mpi_win_lock.o \
    	scoped_shmem.o \
    	dlmalloc.o \



CC	= mpicc
C++	= mpicxx

CCFLAGS  = -g -O2 -std=c++11


internal_win_test: internal_win_test.o $(OBJS)
	$(C++) -o $@ $^

internal_win_test.o: internal_win_test.cc
	$(C++)	-o $@ -c $< $(CCFLAGS) $(INCLUDE)