SHELL=/bin/sh
BENCHMARK=is


include ../../config/make.def

PROGRAM  = $(BINDIR)/$(BENCHMARK).$(CLASS).seq

include ../../sys/make.common


OBJS = is.o \
			${COMMON}/c_print_results.o \
			${COMMON}/c_timers.o \
			${COMMON}/c_wtime.o


${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

is.o: is.c npbparams.h

clean:
	- rm -f *.o *~ 
	- rm -f npbparams.h core
	- rm ${COMMON}/c_print_results.o \
			${COMMON}/c_timers.o \
			${COMMON}/c_wtime.o

