BENCHMARK = utc_latency
BINDIR = ./bin
PROGRAM_SOURCE:= utc_latency.cc


PROGRAM = $(BINDIR)/$(BENCHMARK).out
default: $(PROGRAM)


include ../../../for-nodes.comm


libutc:
	cd $(PROJECT_HOMEDIR);$(MAKE) -f for-nodes.mk

### helper function for test program
#HELPER_OBJ = helper_getopt.o 
#$(HELPER_OBJ): $(PROJECT_HOMEDIR)/test/bench/helper_getopt.c
#	$(G++) -o $(HELPER_OBJ) -c $<

	
PROGRAM_OBJ:= $(addsuffix .o, $(basename $(PROGRAM_SOURCE))) \
			
	
$(PROGRAM): libutc $(PROGRAM_OBJ)
	$(G++)	$(CCFLAG) $(LINKFLAG) -o $(PROGRAM) $(PROGRAM_OBJ) $(LIB)
	@echo "make target successful !!!" 
	
utc_latency.o: utc_latency.cc
	$(G++)	-c $< $(CCFLAG) $(INCLUDE)
	

	
###################################	
clean:
	- rm *.o *~	
	
	