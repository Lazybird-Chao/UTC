SHELL=/bin/sh
BENCHMARK=ft


include ../../config/make.def

PROGRAM  = $(BINDIR)/$(BENCHMARK).$(CLASS).omp

include ../../sys/make.common

CCOMPILE += -fopenmp
CLINKFLAGS += -fopenmp

OBJS = ft.o \
			${COMMON}/${RAND}.o \
			${COMMON}/print_results_omp.o \
       		${COMMON}/c_timers_omp.o \
       		${COMMON}/wtime.o


${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

fto: ft.c global.h npbparams.h

clean:
	- rm -f *.o *~ 
	- rm -f ft npbparams.h core
	- rm ${COMMON}/${RAND}.o \
			${COMMON}/print_results_omp.o \
       		${COMMON}/c_timers_omp.o \
       		${COMMON}/wtime.o
	- if [ -d rii_files ]; then rm -r rii_files; fi
