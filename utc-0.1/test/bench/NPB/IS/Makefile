BENCHMARK = is
BINDIR = ./bin
PROGRAM_SOURCE:= utc_is.cc


ifndef NPROCS
NPROCS = 1
endif
PROGRAM = $(BINDIR)/$(BENCHMARK).$(CLASS).$(NPROCS).utc


default: $(PROGRAM)

config:
	cd ../sys; $(MAKE) all -f makefile_for_utc
	../sys/setparams_for_utc $(BENCHMARK) $(NPROCS) $(CLASS)

include ../../../../for-vm.comm

COMMON = ../common
INCLUDE += -I$(COMMON)

libutc:
	cd $(PROJECT_HOMEDIR);$(MAKE) -f for-vm.mk

### helper function for test program
HELPER_OBJ = helper_getopt.o 
$(HELPER_OBJ): $(PROJECT_HOMEDIR)/test/bench/helper_getopt.c
	$(GCC) -o $(HELPER_OBJ) -c -g $<
	
${COMMON}/c_print_results_utc.o: ${COMMON}/c_print_results_utc.cc
	cd ${COMMON};  $(G++) -c -g -Wall -O3 -mcmodel=medium c_print_results_utc.cc
	
PROGRAM_OBJ:= $(addsuffix .o, $(basename $(PROGRAM_SOURCE))) \
			$(HELPER_OBJ) \
			${COMMON}/c_print_results_utc.o 
	
$(PROGRAM): libutc config $(PROGRAM_OBJ)
	$(G++)	$(CCFLAG) $(LINKFLAG) -o $(PROGRAM) $(PROGRAM_OBJ) $(LIB)
	@echo "make target successful !!!" 
	
utc_is.o: utc_is.cc npbparams.h
	$(G++)	-c $< $(CCFLAG) $(INCLUDE)
	

	
###################################	
clean:
	- rm -f *.o *~	
	- rm -f npbparams.h ${COMMON}/c_print_results_utc.o
	