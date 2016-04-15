SHELL=/bin/sh
BENCHMARK=is


include ../../config/make.def

PROGRAM  = $(BINDIR)/$(BENCHMARK).$(CLASS).omp

include ../../sys/make.common

CCOMPILE += -fopenmp
CLINKFLAGS += -fopenmp

OBJS = is.o \
			${COMMON}/c_print_results_omp.o \
       		${COMMON}/c_timers_omp.o \
       		${COMMON}/c_wtime.o


${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

is.o: is.c npbparams.h

clean:
	- rm -f *.o *~ 
	- rm -f npbparams.h core
	- rm ${COMMON}/c_print_results_omp.o \
       		${COMMON}/c_timers_omp.o \
       		${COMMON}/c_wtime.o
	- if [ -d rii_files ]; then rm -r rii_files; fi
