obj = mm_main.o
bin = mm_main
kernelObj = mm_kernel.o
default: $(bin)

C++ = g++
CCFLAGS = -O2 -std=c++11

# change the CUDA_DIR if nessary
#CUDA_DIR = /usr/local/cuda
CUDA_DIR = /shared/apps/cuda7.5
CUDA_INCLUDE = -I$(CUDA_DIR)/include
CUDA_LIB := -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(CUDA_DIR)/lib64 -lcudart

GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35

NVCC = nvcc
NVCCFLAGS = -O2 $(GENCODE_SM35)



COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<

# the need of "-ccbin g++-4.8" depends on the CUDA SDK version and g++ version	
$(obj):mm_main.cu
	$(NVCC) -o $@  $(CCFLAGS) $(CUDA_INCLUDE) -c $<
#$(C++) -o $@ $(CCFLAGS) $(CUDA_INCLUDE) -c $<

$(kernelObj):mm_kernel.cc
	$(NVCC) -x cu  $(NVCCFLAGS) $(CUDA_INCLUDE) -c $<


$(bin): $(obj) $(kernelObj) $(helperObj)
	$(C++) -o $@ $(obj) $(kernelObj) $(helperObj) $(CUDA_LIB)

.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(kernelObj) $(helperObj)
	
	