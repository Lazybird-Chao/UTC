#########################
BENCHMARK = utc_2Dheat_gpu
BINDIR = .
PROGRAM_SOURCE:= utc_2Dheat_gpu.cc \
				 utc_2Dheat_gpu_kernel.cc
COMM_FILE = ../../../../for-lab.comm
LIBMK_FILE = for-lab.mk


#################################
PROGRAM = $(BINDIR)/$(BENCHMARK).utc
default: $(PROGRAM)

include $(COMM_FILE)

LIBUTC = $(PROJECT_LIBDIR)/libutc
LIBUTC_GPU = $(PROJECT_LIBDIR)/libutcgpu

$(PROJECT_LIBDIR)/libutc:
	cd $(PROJECT_HOMEDIR);$(MAKE) -f $(LIBMK_FILE)
	
$(PROJECT_LIBDIR)/libutcgpu:
	cd $(PROJECT_SRCDIR)/gpu;$(MAKE) -f Make.mk

### helper function for test program
HELPER_OBJ = helper_getopt.o 
$(HELPER_OBJ): $(PROJECT_HOMEDIR)/test/bench/helper_getopt.c
	$(G++) -o $(HELPER_OBJ) -c $<
	
PROGRAM_OBJ:= $(addsuffix .o, $(basename $(PROGRAM_SOURCE))) \
				#$(HELPER_OBJ)

$(PROGRAM): $(PROGRAM_OBJ) $(LIBUTC) $(LIBUTC_GPU)
	$(G++)	$(CCFLAG) $(LINKFLAG) -o $(PROGRAM) $(PROGRAM_OBJ) $(LIB)
	@echo "make target successful !!!" 

utc_2Dheat_gpu.o:utc_2Dheat_gpu.cc
	$(G++)	-c $< $(CCFLAG) $(INCLUDE)
utc_2Dheat_gpu_kernel.o:utc_2Dheat_gpu_kernel.cc
	$(NVCC) -x cu -ccbin g++-4.8 $(NVCCFLAG) $(INCLUDE) -c $<

				

###################################	
clean:
	- rm -f *.o *~
