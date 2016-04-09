BENCHMARK = ep
BINDIR = ./bin
PROGRAM_SOURCE:= utc_ep.cc


PROGRAM = $(BINDIR)/$(BENCHMARK).$(CLASS).out
default: $(PROGRAM)

config:
	cd ../sys; $(MAKE) all -f makefile_for_utc
	../sys/setparams_for_utc $(BENCHMARK) $(CLASS)

include ../../../../for-lab.comm

COMMON = ../common
INCLUDE += -I$(COMMON)

libutc:
	cd $(PROJECT_HOMEDIR);$(MAKE) -f for-lab.mk

### helper function for test program
HELPER_OBJ = helper_getopt.o 
$(HELPER_OBJ): $(PROJECT_HOMEDIR)/test/bench/helper_getopt.c
	$(G++) -o $(HELPER_OBJ) -c $<
	
${COMMON}/print_results_utc.o: ${COMMON}/print_results_utc.cc
	cd ${COMMON};  $(G++) -c -g -Wall -O3 -mcmodel=medium print_results_utc.cc
${COMMON}/randdp.o: ${COMMON}/randdp.c 
	cd ${COMMON}; $(GCC) -c -g -Wall -O3 -mcmodel=medium randdp.c
	
PROGRAM_OBJ:= $(addsuffix .o, $(basename $(PROGRAM_SOURCE))) \
			$(HELPER_OBJ) \
			${COMMON}/print_results_utc.o \
			${COMMON}/randdp.o 
	
$(PROGRAM): libutc config $(PROGRAM_OBJ)
	$(G++)	$(CCFLAG) $(LINKFLAG) -o $(PROGRAM) $(PROGRAM_OBJ) $(LIB)
	@echo "make target successful !!!" 
	
utc_ep.o: utc_ep.cc npbparams.h
	$(G++)	-c $< $(CCFLAG) $(INCLUDE)
	

	
###################################	
clean:
	- rm *.o *~	
	- rm npbparams.h ${COMMON}/print_results_utc.o ${COMMON}/randdp.o
	