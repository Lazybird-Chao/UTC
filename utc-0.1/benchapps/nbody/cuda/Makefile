obj = nbody_main.o
bin = nbody_main
kernelObj = body_system.o body_kernel.o
default: $(bin)

C++ = g++
CCFLAGS = -O2 -g -std=c++11
NWARNING = -w 

#CUDA_DIR = /usr/local/cuda
CUDA_DIR = /shared/apps/cuda7.5
CUDA_INCLUDE = -I$(CUDA_DIR)/include
CUDA_LIB := -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(CUDA_DIR)/lib64 -lcudart

GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35

NVCC = nvcc
NVCCFLAGS = -g -G $(GENCODE_SM35) --device-c
NVLINKFLAGS = $(GENCODE_SM35) --device-link


COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<

	
nbody_main.o: nbody_main.cc nbody.h bodysystem.h
	$(C++) -o $@   $(CCFLAGS) $(NWARNING)  $(CUDA_INCLUDE) -c $<

body_system.o: body_system.cu body_system.h
	$(NVCC) -o $@  $(CCFLAGS) $(NVCCFLAGS) $(NWARNING)  $(CUDA_INCLUDE) -c $<

body_kernel.o: body_kernel.cu body_system.h
	$(NVCC) -o $@  $(CCFLAGS) $(NVCCFLAGS) $(NWARNING)  $(CUDA_INCLUDE) -c $<	

tmplink.o: body_system.o body_kernel.o
	$(NVCC) -o $@ $(NVLINKFLAGS) $^ 
	
bfs_main: $(obj) $(kernelObj) $(helperObj) tmplink.o
	$(C++) -o $@ $^  $(CUDA_LIB)
	

	
.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(kernelObj) $(helperObj) tmplink.o