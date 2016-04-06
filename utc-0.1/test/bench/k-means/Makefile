#################################
BENCHMARK = utc_kmeans
BINDIR = .
PROGRAM_SOURCE:= utc_kmeans_stask_v2.cc


#################################
PROGRAM = $(BINDIR)/$(BENCHMARK).out
default: $(PROGRAM)

include ../../../comm.mk

### helper function for test program
HELPER_OBJ = helper_getopt.o 
$(HELPER_OBJ): $(PROJECT_HOMEDIR)/test/bench/helper_getopt.c
	$(G++) -o $(HELPER_OBJ) -c $<
	
PROGRAM_OBJ:= $(addsuffix .o, $(basename $(PROGRAM_SOURCE))) \
				$(HELPER_OBJ)

$(PROGRAM): $(PROGRAM_OBJ) $(LIB_UTC)
	$(G++)	$(CCFLAG) $(LINKFLAG) -o $(PROGRAM) $(PROGRAM_OBJ) $(LIB)
	@echo "make target successful !!!" 

utc_kmeans_stask_v2.o:utc_kmeans_stask_v2.cc
	$(G++)	-c $< $(CCFLAG) $(INCLUDE)

				

###################################	
clean:
	- rm -f *.o *~
