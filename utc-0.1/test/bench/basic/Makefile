#########################
BENCHMARK = BasicGPUTaskTest
BINDIR = .
PROGRAM_SOURCE:= BasicGPUTaskTest.cc
COMM_FILE = ../../../for-lab.comm
LIBMK_FILE = for-lab.mk


#################################
PROGRAM = $(BINDIR)/$(BENCHMARK).utc
default: $(PROGRAM)

include $(COMM_FILE)

libutc:
	cd $(PROJECT_HOMEDIR);$(MAKE) -f $(LIBMK_FILE)
	
libutcgpu:
	cd $(PROJECT_SRCDIR)/gpu;$(MAKE) -f Make.mk

### helper function for test program
#HELPER_OBJ = helper_getopt.o 
#$(HELPER_OBJ): $(PROJECT_HOMEDIR)/test/bench/helper_getopt.c
#	$(G++) -o $(HELPER_OBJ) -c $<
	
PROGRAM_OBJ:= $(addsuffix .o, $(basename $(PROGRAM_SOURCE))) \
				#$(HELPER_OBJ)

$(PROGRAM): $(PROGRAM_OBJ) libutc libutcgpu
	$(G++)	$(CCFLAG) $(LINKFLAG) -o $(PROGRAM) $(PROGRAM_OBJ) $(LIB)
	@echo "make target successful !!!" 

BasicGPUTaskTest.o:BasicGPUTaskTest.cc
	$(G++)	-c $< $(CCFLAG) $(INCLUDE)

				

###################################	
clean:
	- rm -f *.o *~