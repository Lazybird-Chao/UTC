obj = rotate_main.o image.o
bin = rotate_main
kernelObj = rotate_kernel.o
default: $(bin)



C++ = g++
CCFLAGS = -O2  #-std=c++11

# change CUDA_DIR if nessary
CUDA_DIR = /usr/local/cuda
CUDA_INCLUDE = -I$(CUDA_DIR)/include
CUDA_LIB := -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(CUDA_DIR)/lib64 -lcudart

GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35

NVCC = nvcc
NVCCFLAGS =  $(GENCODE_SM20) --device-c

DEBUG = 0
ifeq ($(DEBUG), 1)
	NVCCFLAGS += -g -G
	CCFLAGS += -g
endif


COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<

rotate_main.o: rotate_main.cu image.h rotate_kernel.h
	$(NVCC) -o $@ -ccbin g++-4.8 $(CCFLAGS) $(NVCCFLAGS) $(CUDA_INCLUDE) -c $<

rotate_kernel.o: rotate_kernel.cc rotate_kernel.h image.h
	$(NVCC) -o $@ -x cu -ccbin g++-4.8 $(CCFLAGS) $(NVCCFLAGS) $(CUDA_INCLUDE) -c $<
	
tmplink.o: rotate_main.o rotate_kernel.o
	$(NVCC) -o $@ $(GENCODE_SM20) --device-link $^
	
image.o: image.cc image.h
	$(C++) -o $@  $(CCFLAGS) -c $<
	
rotate_main:$(obj) $(kernelObj) $(helperObj) tmplink.o
	$(C++) -o $@ $^ $(CUDA_LIB)

.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(kernelObj) $(helperObj) tmplink.o

	
