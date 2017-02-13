obj = c-ray_main.o
bin = c-ray_main
kernelObj = c-ray_kernel.o
default: $(bin)


C++ = g++
CCFLAGS = -O2 -g -G #-std=c++11

# change CUDA_DIR if nessary
CUDA_DIR = /usr/local/cuda
CUDA_INCLUDE = -I$(CUDA_DIR)/include
CUDA_LIB := -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(CUDA_DIR)/lib64 -lcudart

GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35

NVCC = nvcc
NVCCFLAGS = -O2 -g -G $(GENCODE_SM20)



COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<
	

# the need of "-ccbin g++-4.8" depends on the CUDA SDK version and g++ version
c-ray_main.o: c-ray_main.cu c-ray_kernel.h
	$(NVCC) -o $@ -ccbin g++-4.8 $(CCFLAGS) $(CUDA_INCLUDE) -c $<

c-ray_kernel.o: c-ray_kernel.cc c-ray_kernel.h c-ray_kernel_device.h
	$(NVCC) -o $@ -x cu -ccbin g++-4.8 $(NVCCFLAGS) $(CUDA_INCLUDE) -c $<
	
c-ray_main: $(obj) $(kernelObj) $(helperObj)
	$(C++) -o $@ c-ray_main.o c-ray_kernel.o $(helperObj) $(CUDA_LIB)



.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(kernelObj) $(helperObj)

