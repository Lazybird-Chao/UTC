#################################
BENCHMARK = utc_sh2D_heat
BINDIR = .
PROGRAM_SOURCE:= utc_sh2D_heat.cc
COMM_FILE = ../../../for-vm.comm
LIBMK_FILE = for-vm.mk


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

utc_sh2D_heat.o:utc_sh2D_heat.cc
	$(G++)	-c $< $(CCFLAG) $(INCLUDE)

				

###################################	
clean:
	- rm -f *.o *~