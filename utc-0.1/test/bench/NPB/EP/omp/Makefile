SHELL=/bin/sh
BENCHMARK=ep
BENCHMARKU=EP

include ../../config/make.def

PROGRAM  = $(BINDIR)/$(BENCHMARK).$(CLASS).omp

include ../../sys/make.common

CCOMPILE += -fopenmp
CLINKFLAGS += -fopenmp

OBJS = ep.o \
			${COMMON}/print_results_omp.o \
       		${COMMON}/c_timers_omp.o \
       		${COMMON}/${RAND}.o \
	   		${COMMON}/wtime.o


${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

ep.o: ep.c npbparams.h

clean:
	- rm -f *.o *~ 
	- rm -f npbparams.h core
	- rm ${COMMON}/print_results_omp.o ${COMMON}/c_timers_omp.o ${COMMON}/${RAND}.o ${COMMON}/wtime.o
	- if [ -d rii_files ]; then rm -r rii_files; fi
