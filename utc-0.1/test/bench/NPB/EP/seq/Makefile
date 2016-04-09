SHELL=/bin/sh
BENCHMARK=ep


include ../../config/make.def

PROGRAM  = $(BINDIR)/$(BENCHMARK).$(CLASS).seq

include ../../sys/make.common


OBJS = ep.o \
			${COMMON}/print_results.o \
			${COMMON}/${RAND}.o \
       		${COMMON}/c_timers.o \
			${COMMON}/wtime.o


${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

ep.o: ep.c npbparams.h

clean:
	- rm -f *.o *~ 
	- rm -f npbparams.h core
	- rm ${COMMON}/print_results.o \
			${COMMON}/${RAND}.o \
       		${COMMON}/c_timers.o \
			${COMMON}/wtime.o

