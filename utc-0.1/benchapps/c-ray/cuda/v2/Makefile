obj = c-ray_main_v2.o
bin = c-ray_main_v2
kernelObj = c-ray_kernel_v2.o
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
NVCCFLAGS = -g -G $(GENCODE_SM20) --device-c



COMMONDIR = ../../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<
	

# the need of "-ccbin g++-4.8" depends on the CUDA SDK version and g++ version
c-ray_main_v2.o: c-ray_main_v2.cu c-ray_kernel_v2.h
	$(NVCC) -o $@ -ccbin g++-4.8 $(CCFLAGS) $(NVCCFLAGS) $(CUDA_INCLUDE) -c $<

c-ray_kernel_v2.o: c-ray_kernel_v2.cc c-ray_kernel_v2.h c-ray_kernel_device_v2.h
	$(NVCC) -o $@ -x cu -ccbin g++-4.8 $(CCFLAGS) $(NVCCFLAGS) $(CUDA_INCLUDE) -c $<
	
tmplink.o: c-ray_main_v2.o c-ray_kernel_v2.o
	$(NVCC) -o $@ $(GENCODE_SM20) --device-link c-ray_main_v2.o c-ray_kernel_v2.o
	
c-ray_main_v2: $(obj) $(kernelObj) $(helperObj) tmplink.o
	$(C++) -o $@ c-ray_main_v2.o c-ray_kernel_v2.o tmplink.o $(helperObj) $(CUDA_LIB)



.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(kernelObj) $(helperObj) tmplink.o

