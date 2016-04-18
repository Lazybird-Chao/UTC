SHELL=/bin/sh
BENCHMARK=ft


include ../../config/make.def

PROGRAM  = $(BINDIR)/$(BENCHMARK).$(CLASS).seq

include ../../sys/make.common


OBJS=appft.o auxfnct.o fft3d.o mainft.o verify.o \
			${COMMON}/${RAND}.o \
			${COMMON}/print_results.o \
			${COMMON}/c_timers.o \
			${COMMON}/wtime.o


${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

appft.o:	appft.c  global.h npbparams.h
auxfnct.o:	auxfnct.c  global.h npbparams.h
fft3d.o:	fft3d.c  global.h npbparams.h
mainft.o:	mainft.c  global.h npbparams.h
verify.o:	verify.c  global.h npbparams.h

clean:
	- rm -f *.o *~ 
	- rm -f ft npbparams.h core
	- rm ${COMMON}/${RAND}.o \
			${COMMON}/print_results.o \
			${COMMON}/c_timers.o \
			${COMMON}/wtime.o

