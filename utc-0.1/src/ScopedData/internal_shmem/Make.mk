#####################
default:libutc-scopedshmem.a

include ../../../for-dis.comm

#####################
vpath %.h  $(PROJECT_INCLUDEDIR)
vpath %.inc $(PROJECT_INCLUDEDIR)
vpath %.cc $(PROJECT_SRCDIR)
vpath %.h  $(PROJECT_INCLUDEDIR)/ScopedData
vpath %.inc $(PROJECT_INCLUDEDIR)/ScopedData
vpath %.cc $(PROJECT_SRCDIR)/ScopedData
vpath %.h  $(PROJECT_INCLUDEDIR)/ScopedData/internal_shmem
vpath %.inc  $(PROJECT_INCLUDEDIR)/ScopedData/internal_shmem
vpath %.cc  $(PROJECT_SRCDIR)/ScopedData/internal_shmem

ifeq ($(ENABLE_GPU), 1)
    vpath %.h  $(PROJECT_INCLUDEDIR)/gpu
    vpath %.inc $(PROJECT_INCLUDEDIR)/gpu
    vpath %.cc $(PROJECT_SRCDIR)/gpu
endif

#####################
OBJS:= 	dlmalloc.o \
		internal_win.o \
    	mpi_win_lock.o \
    	scoped_shmem.o \
    	
		
libutc-scopedshmem.a: $(OBJS)
	ar -r -uv libutc-scopedshmem.a $(OBJS)
	mv libutc-scopedshmem.a $(PROJECT_LIBDIR)
	cp $(OBJS) $(PROJECT_LIBDIR)
	@echo "make scopedshmem library succefful !!!"

internal_win.o: internal_win.cc internal_win.h dlmalloc.h
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
mpi_win_lock.o: mpi_win_lock.cc mpi_win_lock.h
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
scoped_shmem.o: scoped_shmem.cc scoped_shmem.h dlmalloc.h internal_win.h mpi_win_lock.h
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
dlmalloc.o: dlmalloc.c dlmalloc.h
	$(C++)	-o $@ -c $< -g -O2  $(INCLUDE)
	
clean:
	rm -rf *.o *.a
	