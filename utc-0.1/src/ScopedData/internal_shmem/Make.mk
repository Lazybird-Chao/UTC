#####################
default:libutc-scopedshmem.a

#include ../../for-dis.comm

#####################
vpath %.h  ../../../include/ScopedData/internal_shmem

INCLUDE = -I../../../include/ScopedData/internal_shmem

CC	= mpicc
C++	= mpicxx

CCFLAGS  = -g -O2 -std=c++11


#####################
OBJS:= 	dlmalloc.o \
		internal_win.o \
    	mpi_win_lock.o \
    	scoped_shmem.o \
    	
		
libutc-scopedshmem.a: $(OBJS)
	ar -r -uv libutc-scopedshmem.a $(OBJS)
	cp libutc-scopedshmem.a $(PROJECT_LIBDIR)
	cp $(OBJS) $(PROJECT_LIBDIR)
	@echo "make scopedshmem library succefful !!!"

internal_win.o: internal_win.cc internal_win.h dlmalloc.h
	$(C++)	-o $@ -c $< $(CCFLAGS) $(INCLUDE)
mpi_win_lock.o: mpi_win_lock.cc mpi_win_lock.h
	$(C++)	-o $@ -c $< $(CCFLAGS) $(INCLUDE)
scoped_shmem.o: scoped_shmem.cc scoped_shmem.h dlmalloc.h internal_win.h mpi_win_lock.h
	$(C++)	-o $@ -c $< $(CCFLAGS) $(INCLUDE)
dlmalloc.o: dlmalloc.c dlmalloc.h
	$(C++)	-o $@ -c $< -g -O2  $(INCLUDE)
	
clean:
	rm -rf *.o *.a
	