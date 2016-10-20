#################################
BENCHMARK = utc_sh2D_heat
BINDIR = .
PROGRAM_SOURCE:= $(BENCHMARK).cc
COMM_FILE = ../../../for-nodes.comm
LIBMK_FILE = for-nodes.mk


#################################
PROGRAM = $(BINDIR)/$(BENCHMARK).utc
default: $(PROGRAM)

include $(COMM_FILE)

libutc:
	cd $(PROJECT_HOMEDIR);$(MAKE) -f $(LIBMK_FILE)

LIB += -lm

### helper function for test program
HELPER_OBJ = helper_getopt.o 
$(HELPER_OBJ): $(PROJECT_HOMEDIR)/test/bench/helper_getopt.c
	$(G++) -o $(HELPER_OBJ) -c $<
	
PROGRAM_OBJ:= $(addsuffix .o, $(basename $(PROGRAM_SOURCE))) \
				$(HELPER_OBJ)

$(PROGRAM): $(PROGRAM_OBJ) libutc
	$(G++)	$(CCFLAG) $(LINKFLAG) -o $(PROGRAM) $(PROGRAM_OBJ) $(LIB)
	@echo "make target successful !!!" 

$(BENCHMARK).o:$(BENCHMARK).cc
	$(G++)	-c $< $(CCFLAG) $(INCLUDE)

				

###################################	
clean:
	- rm -f *.o *~
